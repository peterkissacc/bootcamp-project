
# src/ingest/ingest_csv.py
# ---------------------------------------------------------
# CSV -> LangChain Documents -> Azure OpenAI Embeddings
# -> Chroma (persistent), with batching + retry + checkpoint
# Compatible with Bootcamp .env variable names.
# Uses UNIQUE IDs per document to avoid DuplicateIDError.
# Works across langchain-chroma / chromadb versions (safe persist).
# ---------------------------------------------------------

import os
import sys
import json
import time
import math
import random
from pathlib import Path
from typing import List, Iterable, Dict, Any

import pandas as pd
from dotenv import load_dotenv, find_dotenv

from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

# Try to import OpenAI 1.x errors; fall back if not available
try:
    from openai import (
        RateLimitError,
        APIError,
        APIConnectionError,
        PermissionDeniedError,
        AuthenticationError,
    )
except Exception:
    class RateLimitError(Exception): ...
    class APIError(Exception): ...
    class APIConnectionError(Exception): ...
    class PermissionDeniedError(Exception): ...
    class AuthenticationError(Exception): ...

# --- Early debug (verify working dir & CSV presence) ---
print("DEBUG START")
print("CWD =", os.getcwd())
print("File exists?", os.path.exists("data/tracks.csv"))


# -----------------------------
# .env loading & normalization
# -----------------------------
def _clean(s: str | None) -> str | None:
    """Strip surrounding quotes; drop trailing slash on endpoints."""
    if s is None:
        return None
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if s.lower().startswith("http") and s.endswith("/"):
        s = s[:-1]
    return s


def load_bootcamp_env() -> dict:
    """
    Load Bootcamp-style .env variable names and fail-fast if anything critical is missing.
    Supports both AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_ENDPOINT.
    """
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    # Accept both styles for endpoint to avoid naming drift in materials
    endpoint_raw = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_ENDPOINT")

    cfg = {
        "AZURE_OPENAI_API_KEY": _clean(os.getenv("AZURE_OPENAI_API_KEY")),
        "AZURE_OPENAI_API_INSTANCE_NAME": _clean(os.getenv("AZURE_OPENAI_API_INSTANCE_NAME")),
        "AZURE_OPENAI_API_VERSION": _clean(os.getenv("AZURE_OPENAI_API_VERSION")),
        "AZURE_OPENAI_ENDPOINT": _clean(endpoint_raw),
        "AZURE_OPENAI_DEPLOYMENT_NAME": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")),  # chat (later)
        "AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS")),
        "CHROMA_DIR": _clean(os.getenv("CHROMA_DIR", "./db/chroma")),
        # Optional tuning knobs via env:
        "BATCH_SIZE": int(os.getenv("INGEST_BATCH_SIZE", "100")),
        "BASE_SLEEP": int(os.getenv("INGEST_BASE_SLEEP_SECONDS", "65")),
        "JITTER": int(os.getenv("INGEST_JITTER_SECONDS", "6")),
        "MAX_RETRIES": int(os.getenv("INGEST_MAX_RETRIES", "6")),
    }

    # Build endpoint from instance name if needed
    if not cfg["AZURE_OPENAI_ENDPOINT"] and cfg["AZURE_OPENAI_API_INSTANCE_NAME"]:
        cfg["AZURE_OPENAI_ENDPOINT"] = f"https://{cfg['AZURE_OPENAI_API_INSTANCE_NAME']}.openai.azure.com"

    required = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS",
        "CHROMA_DIR",
    ]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        print("[ERROR] Missing required environment variables:")
        for k in missing:
            print(f"  - {k}")
        print("\n[Hint] Keep Bootcamp names in .env, for example:")
        print("  AZURE_OPENAI_API_KEY=...")
        print("  AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com")
        print("  AZURE_OPENAI_API_VERSION=2024-06-01")
        print("  AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS=text-embedding-3-large")
        print("  CHROMA_DIR=./db/chroma\n")
        sys.exit(1)

    print(f"[DEBUG] .env loaded from: {env_path}")
    print(f"[DEBUG] Endpoint: {cfg['AZURE_OPENAI_ENDPOINT']}")
    print(f"[DEBUG] Embeddings deployment: {cfg['AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS']}")
    print(f"[DEBUG] Chroma dir: {cfg['CHROMA_DIR']}")
    print(f"[DEBUG] Batch={cfg['BATCH_SIZE']} BaseSleep={cfg['BASE_SLEEP']}s Jitter={cfg['JITTER']}s MaxRetries={cfg['MAX_RETRIES']}")
    return cfg


# -----------------------------
# CSV -> Documents  (with UNIQUE IDs)
# -----------------------------
REQUIRED_COLS = ["id", "name", "genre", "artists", "album", "popularity", "duration_ms", "explicit"]

def build_page_content(row: pd.Series) -> str:
    """English content to embed. Short & consistent helps semantic search."""
    return (
        f"Track: {row['name']}\n"
        f"Artists: {row['artists']}\n"   # raw CSV string (comma-separated if multiple)
        f"Album: {row['album']}\n"
        f"Genre: {row['genre']}\n"
        f"Popularity: {row['popularity']} /100\n"
        f"Duration (ms): {row['duration_ms']}\n"
        f"Explicit: {row['explicit']}\n"
        f"Spotify ID: {row['id']}"
    )


def build_metadata(row: pd.Series) -> Dict[str, Any]:
    """
    Metadata is NOT embedded — used for filtering and exact grading.
    Chroma expects simple value types: str | int | float | bool | None.
    """
    # Normalize artists into list, then store as STRING (not list!) for Chroma
    artists_list = [a.strip() for a in str(row["artists"]).split(",") if a.strip()]
    artists_str = ", ".join(artists_list)

    # Robust numeric conversions
    def _to_int(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None

    explicit_flag = str(row["explicit"]).strip().lower() in ["true", "1", "yes"]

    return {
        "id": str(row["id"]),              # original Spotify track ID (may be duplicated upstream)
        "name": str(row["name"]),
        "album": str(row["album"]),
        "artists": artists_str,            # string; valid Chroma metadata type
        "genre": str(row["genre"]),
        "popularity": _to_int(row["popularity"]),
        "duration_ms": _to_int(row["duration_ms"]),
        "explicit": explicit_flag,
    }


def row_to_document(row: pd.Series, row_index: int) -> Document:
    """
    Create a Document with a UNIQUE ID stored in metadata['unique_id'].
    This avoids DuplicateIDError even if Spotify 'id' appears multiple times.
    """
    metadata = build_metadata(row)
    unique_id = f"{row_index}_{metadata['id']}"  # guaranteed unique per row
    metadata["unique_id"] = unique_id
    return Document(page_content=build_page_content(row), metadata=metadata)


# -----------------------------
# Checkpointing (resume support)
# -----------------------------
def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load checkpoint JSON with the set of processed UNIQUE IDs."""
    if not path.exists():
        return {"processed_ids": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"processed_ids": []}


def save_checkpoint(path: Path, processed_ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = {"processed_ids": processed_ids}
    with path.open("w", encoding="utf-8") as f:
        json.dump(tmp, f, ensure_ascii=False, indent=2)


def chunked(seq: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# -----------------------------
# Safe persist helper (works across versions)
# -----------------------------
def safe_persist(vectorstore: Chroma) -> None:
    """
    Persist to disk if current langchain_chroma/chromadb exposes a persist method.
    Newer versions persist automatically with persist_directory, so this may no-op.
    """
    try:
        if hasattr(vectorstore, "persist") and callable(getattr(vectorstore, "persist")):
            vectorstore.persist()
            return
        client = getattr(vectorstore, "_client", None)
        if client and hasattr(client, "persist") and callable(getattr(client, "persist")):
            client.persist()
            return
    except Exception:
        pass


# -----------------------------
# Main ingest with batching + UNIQUE IDs
# -----------------------------
def main():
    cfg = load_bootcamp_env()

    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "tracks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"[ERROR] CSV not found at: {csv_path}")

    print(f"[INFO] Loading CSV from: {csv_path.as_posix()}")
    df = pd.read_csv(csv_path)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[ERROR] CSV missing required columns: {missing_cols}")

    # Build all documents WITH unique_id in metadata
    print("[INFO] Converting rows into Documents...")
    documents: List[Document] = [
        row_to_document(row, idx) for idx, (_, row) in enumerate(df.iterrows())
    ]
    print(f"[INFO] Total documents created: {len(documents)}")

    # Initialize embeddings (Bootcamp style: use 'model' with Azure endpoint + version)
    print("[INFO] Initializing Azure OpenAI Embeddings...")
    embeddings = AzureOpenAIEmbeddings(
        model=cfg["AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS"],  # Azure: deployment name
        api_key=cfg["AZURE_OPENAI_API_KEY"],
        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=cfg["AZURE_OPENAI_API_VERSION"],
    )

    # Create (or load) the persistent Chroma store
    chroma_dir = cfg["CHROMA_DIR"]
    vectorstore = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )

    # Checkpoint file — stores processed UNIQUE IDs
    ckpt_path = project_root / "db" / "ingest_state.json"
    ckpt = load_checkpoint(ckpt_path)
    processed_ids: List[str] = ckpt.get("processed_ids", [])
    processed_set = set(processed_ids)

    # Filter out already-processed docs (by unique_id)
    docs_to_process: List[Document] = [
        d for d in documents if d.metadata.get("unique_id") not in processed_set
    ]
    total_remaining = len(docs_to_process)
    if total_remaining == 0:
        print("[INFO] Nothing to do. All documents are already processed.")
        return

    print(f"[INFO] Remaining docs to index: {total_remaining}")

    BATCH = cfg["BATCH_SIZE"]
    BASE_SLEEP = cfg["BASE_SLEEP"]
    JITTER = cfg["JITTER"]
    MAX_RETRIES = cfg["MAX_RETRIES"]

    done = 0
    total_batches = math.ceil(total_remaining / BATCH)

    for b_idx, batch_docs in enumerate(chunked(docs_to_process, BATCH), start=1):
        # UNIQUE IDs for this batch
        filtered_docs = [d for d in batch_docs if d.metadata["unique_id"] not in processed_set]
        filtered_ids = [d.metadata["unique_id"] for d in filtered_docs]

        if not filtered_docs:
            print(f"[SKIP] Batch {b_idx}/{total_batches}: all items already processed.")
            continue

        attempt = 0
        while True:
            try:
                # Add to Chroma; pass UNIQUE IDs explicitly to avoid duplicates
                vectorstore.add_documents(filtered_docs, ids=filtered_ids)
                done += len(filtered_docs)
                processed_ids.extend(filtered_ids)
                processed_set.update(filtered_ids)

                # Persist (may no-op), then checkpoint every batch
                safe_persist(vectorstore)
                save_checkpoint(ckpt_path, processed_ids)

                print(f"[OK] Batch {b_idx}/{total_batches} inserted ({len(filtered_docs)} docs). Total done={done}.")
                # Friendly throttle to avoid bursts even without 429
                time.sleep(0.8)
                break

            except RateLimitError:
                attempt += 1
                if attempt > MAX_RETRIES:
                    print(f"[FATAL] Batch {b_idx}: giving up after {MAX_RETRIES} retries due to 429.")
                    raise
                wait_s = BASE_SLEEP + random.randint(0, JITTER)
                print(f"[WARN] 429 RateLimit on batch {b_idx} (attempt {attempt}/{MAX_RETRIES}). Sleeping {wait_s}s ...")
                time.sleep(wait_s)

            except PermissionDeniedError as e:
                print(f"[FATAL] Permission denied (likely network/firewall): {e}")
                raise

            except AuthenticationError as e:
                print(f"[FATAL] Authentication error (key/endpoint/version): {e}")
                raise

            except APIConnectionError as e:
                attempt += 1
                if attempt > MAX_RETRIES:
                    print(f"[FATAL] Batch {b_idx}: connection failed after {MAX_RETRIES} retries.")
                    raise
                wait_s = 10 + random.randint(0, 5)
                print(f"[WARN] Connection issue on batch {b_idx}. Sleeping {wait_s}s then retry ...")
                time.sleep(wait_s)

            except APIError as e:
                # Generic APIError — retry for transient signals
                msg = str(e).lower()
                if ("429" in msg or "rate" in msg or "temporar" in msg or "5" in msg):
                    attempt += 1
                    if attempt > MAX_RETRIES:
                        print(f"[FATAL] Batch {b_idx}: APIError after {MAX_RETRIES} retries.")
                        raise
                    wait_s = BASE_SLEEP + random.randint(0, JITTER)
                    print(f"[WARN] APIError on batch {b_idx}. Sleeping {wait_s}s then retry ...")
                    time.sleep(wait_s)
                else:
                    print(f"[FATAL] Non-retriable APIError: {e}")
                    raise

            except Exception as e:
                print(f"[FATAL] Unexpected error on batch {b_idx}: {e}")
                raise

    print(f"[SUCCESS] Incremental Chroma index created/updated at: {chroma_dir}")
    print(f"[SUCCESS] Processed total documents: {len(processed_ids)}")
    print(f"[SUCCESS] Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    # Run from project root:
    #   python src/ingest/ingest_csv.py
    main()
