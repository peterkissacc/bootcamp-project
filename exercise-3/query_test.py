# exercise-3/query_test.py
# --------------------------------------------
# Simple smoke test for Chroma retriever
# - Location: /exercise-3/query_test.py
# - Target DB: /db/chroma (Root level)
# --------------------------------------------

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

def _clean(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().strip('"').strip("'")
    if s.lower().startswith("http") and s.endswith("/"):
        s = s[:-1]
    return s

def load_env():
    # 1. Calculate paths based on this script being in "exercise-3/"
    script_dir = Path(__file__).resolve().parent  # .../BOOTCAMP-PROJECT/exercise-3
    project_root = script_dir.parent              # .../BOOTCAMP-PROJECT
    
    # 2. Find .env in project root
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        env_path = project_root / ".env"
    
    load_dotenv(dotenv_path=env_path, override=False)

    # 3. Define default DB path (Root level)
    default_chroma_path = project_root / "db" / "chroma"

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_ENDPOINT")
    cfg = {
        "AZURE_OPENAI_API_KEY": _clean(os.getenv("AZURE_OPENAI_API_KEY")),
        "AZURE_OPENAI_API_VERSION": _clean(os.getenv("AZURE_OPENAI_API_VERSION")),
        "AZURE_OPENAI_ENDPOINT": _clean(endpoint),
        "AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS")),
        # Use absolute path to the root 'db/chroma'
        "CHROMA_DIR": _clean(os.getenv("CHROMA_DIR", str(default_chroma_path))),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env var(s): {missing}")
    return cfg

def main():
    cfg = load_env()

    # Recreate embeddings
    embeddings = AzureOpenAIEmbeddings(
        model=cfg["AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS"],
        api_key=cfg["AZURE_OPENAI_API_KEY"],
        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=cfg["AZURE_OPENAI_API_VERSION"],
    )

    # Load Chroma
    print(f"[INFO] Loading Chroma index from: {cfg['CHROMA_DIR']}")
    
    if not os.path.exists(cfg["CHROMA_DIR"]):
        print(f"[ERROR] Database not found at {cfg['CHROMA_DIR']}! Did you run the ingest script?")
        return

    vectorstore = Chroma(
        persist_directory=cfg["CHROMA_DIR"],
        embedding_function=embeddings,
    )

    print("[INFO] Chroma index loaded successfully.")

    # --- 1) Simple similarity search
    # We use a generic query because we don't know exactly which songs are in the first N rows
    query = "sad acoustic songs" 
    print(f"\n[TEST 1] Similarity Search Query: '{query}'")
    
    docs = vectorstore.similarity_search(query, k=5)
    
    if not docs:
        print("[WARN] No results returned.")
    else:
        for i, d in enumerate(docs, start=1):
            m = d.metadata
            print(f"  [{i}] {m.get('name')} | Artist: {m.get('artists')} | Genre: {m.get('genre')}")

    # --- 2) Metadata filter test
    # Ensure this genre actually exists in your ingested subset!
    target_genre = "acoustic" 
    print(f"\n[TEST 2] Metadata Filter: genre='{target_genre}'")
    
    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 5,
        "where": {"genre": target_genre}
    })
    
    try:
        docs_filtered = retriever.invoke(query)
    except AttributeError:
        docs_filtered = retriever.get_relevant_documents(query)

    if not docs_filtered:
        print(f"[WARN] No results for genre '{target_genre}'. (This is normal if you ingested a small subset that lacks this genre).")
    else:
        for i, d in enumerate(docs_filtered, start=1):
            m = d.metadata
            print(f"  [{i}] {m.get('name')} | Genre: {m.get('genre')} | Popularity: {m.get('popularity')}")

    print("\n[DONE] Smoke test completed.")

if __name__ == "__main__":
    main()