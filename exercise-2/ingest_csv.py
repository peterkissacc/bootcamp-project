# exercise-2/ingest_csv.py
# ---------------------------------------------------------
# CSV -> LangChain Documents -> Azure OpenAI Embeddings
# -> Chroma (persistent), with batching + retry + checkpoint
#
# UPDATED: 
# - REMOVED auto-reset logic to allow resuming interrupted jobs.
# - Processes only missing records based on ingest_state.json.
# ---------------------------------------------------------

import os
import sys
import json
import time
import math
import random
from pathlib import Path
from typing import List, Iterable, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv, find_dotenv

from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

try:
    from openai import (
        RateLimitError, APIError, APIConnectionError,
        PermissionDeniedError, AuthenticationError,
    )
except Exception:
    class RateLimitError(Exception): ...
    class APIError(Exception): ...
    class APIConnectionError(Exception): ...
    class PermissionDeniedError(Exception): ...
    class AuthenticationError(Exception): ...

# --- SETTINGS ---
DEBUG_LIMIT: Optional[int] = None 

print("DEBUG START")
print("CWD =", os.getcwd())

# -----------------------------
# .env loading
# -----------------------------
def _clean(s: str | None) -> str | None:
    if s is None: return None
    s = s.strip().strip('"').strip("'")
    if s.lower().startswith("http") and s.endswith("/"):
        s = s[:-1]
    return s

def load_bootcamp_env() -> dict:
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        env_path = Path(__file__).resolve().parents[1] / ".env"
    
    load_dotenv(dotenv_path=env_path, override=False)
    endpoint_raw = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_ENDPOINT")
    
    return {
        "AZURE_OPENAI_API_KEY": _clean(os.getenv("AZURE_OPENAI_API_KEY")),
        "AZURE_OPENAI_API_VERSION": _clean(os.getenv("AZURE_OPENAI_API_VERSION")),
        "AZURE_OPENAI_ENDPOINT": _clean(endpoint_raw),
        "AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS")),
        "BATCH_SIZE": int(os.getenv("INGEST_BATCH_SIZE", "100")),
        "BASE_SLEEP": int(os.getenv("INGEST_BASE_SLEEP_SECONDS", "1")),
        "MAX_RETRIES": int(os.getenv("INGEST_MAX_RETRIES", "6")),
    }

# -----------------------------
# MAPPING LOGIC
# -----------------------------
REQUIRED_COLS = [
    "track_id", "track_name", "track_genre", "artists", "album_name", 
    "popularity", "duration_ms", "explicit", 
    "danceability", "energy", "key", "loudness", "mode", 
    "speechiness", "acousticness", "instrumentalness", 
    "liveness", "valence", "tempo", "time_signature"
]

def build_metadata(row: pd.Series) -> Dict[str, Any]:
    artists_list = [a.strip() for a in str(row["artists"]).split(",") if a.strip()]
    artists_str = ", ".join(artists_list)

    def _to_int(x):
        try: return int(float(str(x).strip()))
        except: return 0
    
    def _to_float(x):
        try: return float(str(x).strip())
        except: return 0.0

    explicit_flag = str(row["explicit"]).strip().lower() in ["true", "1", "yes"]

    return {
        "id": str(row["track_id"]),
        "name": str(row["track_name"]),
        "album": str(row["album_name"]),
        "artists": artists_str,
        "genre": str(row["track_genre"]),
        "popularity": _to_int(row["popularity"]),
        "duration_ms": _to_int(row["duration_ms"]),
        "explicit": explicit_flag,
        "danceability": _to_float(row["danceability"]),
        "energy": _to_float(row["energy"]),
        "key": _to_int(row["key"]),
        "loudness": _to_float(row["loudness"]),
        "mode": _to_int(row["mode"]),
        "speechiness": _to_float(row["speechiness"]),
        "acousticness": _to_float(row["acousticness"]),
        "instrumentalness": _to_float(row["instrumentalness"]),
        "liveness": _to_float(row["liveness"]),
        "valence": _to_float(row["valence"]),
        "tempo": _to_float(row["tempo"]),
        "time_signature": _to_int(row["time_signature"]),
    }

def row_to_document(row: pd.Series, row_index: int) -> Document:
    metadata = build_metadata(row)
    unique_id = f"{row_index}_{metadata['id']}"
    metadata["unique_id"] = unique_id
    
    content = (
        f"Track: {row['track_name']}\nArtists: {row['artists']}\nAlbum: {row['album_name']}\n"
        f"Genre: {row['track_genre']}\nPopularity: {row['popularity']}\nTempo: {row['tempo']} BPM"
    )
    return Document(page_content=content, metadata=metadata)

# -----------------------------
# Checkpoint utils
# -----------------------------
def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists(): return {"processed_ids": []}
    try:
        with path.open("r", encoding="utf-8") as f: 
            data = json.load(f)
            return data if isinstance(data, dict) else {"processed_ids": []}
    except: return {"processed_ids": []}

def save_checkpoint(path: Path, processed_ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"processed_ids": processed_ids}, f, indent=2)

def chunked(seq, size):
    for i in range(0, len(seq), size): yield seq[i:i+size]

# -----------------------------
# MAIN
# -----------------------------
def main():
    cfg = load_bootcamp_env()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    csv_path = project_root / "data" / "dataset.csv"
    chroma_dir = project_root / "db" / "chroma"
    ckpt_path = project_root / "db" / "ingest_state.json"

    # --- NO RESET ---
    # We keep the old DB and checkpoint file to resume work.
    print(f"\n[INFO] Target Chroma DB: {chroma_dir}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"[ERROR] CSV not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    if DEBUG_LIMIT: df = df.head(DEBUG_LIMIT)

    # Load existing progress
    checkpoint = load_checkpoint(ckpt_path)
    processed_ids = checkpoint.get("processed_ids", [])
    processed_set = set(processed_ids)
    print(f"[INFO] Checkpoint loaded. Already processed: {len(processed_set)} tracks.")

    # Prepare documents and filter out the ones already in DB
    print("[INFO] Filtering remaining documents...")
    all_docs = [row_to_document(row, i) for i, (_, row) in enumerate(df.iterrows())]
    docs_to_process = [d for d in all_docs if d.metadata["unique_id"] not in processed_set]
    
    total_remaining = len(docs_to_process)
    if total_remaining == 0:
        print("[SUCCESS] All documents in this batch are already indexed.")
        return

    print(f"[INFO] Ready to index {total_remaining} new documents.")

    # --- EMBEDDINGS & DB ---
    embeddings = AzureOpenAIEmbeddings(
        model=cfg["AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS"],
        api_key=cfg["AZURE_OPENAI_API_KEY"],
        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=cfg["AZURE_OPENAI_API_VERSION"],
    )

    vectorstore = Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
    )

    # --- INGEST LOOP ---
    BATCH = cfg["BATCH_SIZE"]
    done = 0
    
    for batch in chunked(docs_to_process, BATCH):
        batch_ids = [d.metadata["unique_id"] for d in batch]
        try:
            vectorstore.add_documents(batch, ids=batch_ids)
            done += len(batch)
            processed_ids.extend(batch_ids)
            
            # Save progress every batch
            save_checkpoint(ckpt_path, processed_ids)
            print(f"[OK] Ingested {done}/{total_remaining} new tracks (Total in DB: {len(processed_ids)})")
            
        except Exception as e:
            print(f"[ERROR] Batch failed: {e}. Progress saved up to last successful batch.")
            break
            
    print("[SUCCESS] Ingestion process finished or paused.")

if __name__ == "__main__":
    main()