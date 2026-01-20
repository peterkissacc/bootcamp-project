
# src/smoke_test/query_test.py
# --------------------------------------------
# Simple smoke test for Chroma retriever
# - loads persistent Chroma index
# - runs a semantic query
# - demonstrates metadata filtering (genre='acoustic')
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
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_ENDPOINT")
    cfg = {
        "AZURE_OPENAI_API_KEY": _clean(os.getenv("AZURE_OPENAI_API_KEY")),
        "AZURE_OPENAI_API_VERSION": _clean(os.getenv("AZURE_OPENAI_API_VERSION")),
        "AZURE_OPENAI_ENDPOINT": _clean(endpoint),
        "AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS")),
        "CHROMA_DIR": _clean(os.getenv("CHROMA_DIR", "./db/chroma")),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env var(s): {missing}")
    return cfg

def main():
    cfg = load_env()

    # Recreate the same embeddings used during ingest
    embeddings = AzureOpenAIEmbeddings(
        model=cfg["AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS"],   # Azure: deployment name
        api_key=cfg["AZURE_OPENAI_API_KEY"],
        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=cfg["AZURE_OPENAI_API_VERSION"],
    )

    # Load existing Chroma index (must match ingest CHROMA_DIR)
    vectorstore = Chroma(
        persist_directory=cfg["CHROMA_DIR"],
        embedding_function=embeddings,
    )

    print("\n[INFO] Chroma index loaded.")
    print(f"[INFO] Persist directory: {cfg['CHROMA_DIR']}")

    # --- 1) Simple similarity search (no filter)
    query = "Find an acoustic track by Frank Ocean"
    docs = vectorstore.similarity_search(query, k=5)
    print(f"\n[TEST 1] Query: {query}")
    if not docs:
        print("[WARN] No results returned.")
    else:
        for i, d in enumerate(docs, start=1):
            m = d.metadata
            print(f"  [{i}] name='{m.get('name')}', artists='{m.get('artists')}', album='{m.get('album')}', genre='{m.get('genre')}'")

    # --- 2) Similarity search with metadata filter (genre='acoustic')
    print("\n[TEST 2] Same query with metadata filter: genre='acoustic'")
    # langchain_chroma exposes a ._collection.query in some versions, but prefer retriever API:
    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 5,
        "where": {"genre": "acoustic"}     # Chroma metadata filter
    })
    docs_filtered = retriever.get_relevant_documents(query)
    if not docs_filtered:
        print("[WARN] No results with filter (genre='acoustic'). Try removing the filter or change the query.")
    else:
        for i, d in enumerate(docs_filtered, start=1):
            m = d.metadata
            print(f"  [{i}] name='{m.get('name')}', artists='{m.get('artists')}', album='{m.get('album')}', genre='{m.get('genre')}'")

    # --- 3) Another quick query example (album-oriented)
    another_query = "Which album contains the track 'Cayendo (Side A - Acoustic)'?"
    print(f"\n[TEST 3] Query: {another_query}")
    docs2 = vectorstore.similarity_search(another_query, k=3)
    if not docs2:
        print("[WARN] No results.")
    else:
        for i, d in enumerate(docs2, start=1):
            m = d.metadata
            print(f"  [{i}] name='{m.get('name')}', album='{m.get('album')}', artists='{m.get('artists')}', genre='{m.get('genre')}'")

    print("\n[DONE] Smoke test completed.")

if __name__ == "__main__":
    main()
