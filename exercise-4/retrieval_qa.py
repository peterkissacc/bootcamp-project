# src/qa/retrieval_qa.py
# ------------------------------------------------------------
# Basic RetrievalQA over Chroma with Azure OpenAI Chat (RAG)
# - Loads persistent Chroma index from ROOT/db/chroma
# - Retriever fetches documents based on semantic similarity
# - Context builder includes AUDIO FEATURES (tempo, energy, etc.)
# - LLM answers grounded in provided context
# ------------------------------------------------------------

import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv, find_dotenv

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# ---------------------------
# Helpers
# ---------------------------
def _clean(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().strip('"').strip("'")
    if s.lower().startswith("http") and s.endswith("/"):
        s = s[:-1]
    return s


def load_env() -> Dict[str, str]:
    """Load env vars and calculate absolute path to the database."""
    # 1. Locate .env (2 levels up from src/qa/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]  # .../BOOTCAMP-PROJECT
    
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        env_path = project_root / ".env"
    
    load_dotenv(dotenv_path=env_path, override=False)

    # 2. Define DB path relative to project root
    default_chroma_path = project_root / "db" / "chroma"

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_ENDPOINT")
    cfg = {
        "AZURE_OPENAI_API_KEY": _clean(os.getenv("AZURE_OPENAI_API_KEY")),
        "AZURE_OPENAI_API_VERSION": _clean(os.getenv("AZURE_OPENAI_API_VERSION")),
        "AZURE_OPENAI_ENDPOINT": _clean(endpoint),
        "AZURE_OPENAI_DEPLOYMENT_NAME": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")),
        "AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS")),
        # Use absolute path ensures we find the DB regardless of where script is run
        "CHROMA_DIR": _clean(os.getenv("CHROMA_DIR", str(default_chroma_path))),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env var(s): {missing}")
    return cfg


def format_docs(docs: List[Any]) -> str:
    """
    Join retrieved docs into a context block. 
    UPDATED: Now explicitly extracts audio features from metadata to ensure the LLM sees them.
    """
    chunks = []
    for d in docs:
        m = d.metadata or {}
        
        # Basic Info
        name = m.get("name", "Unknown")
        artists = m.get("artists", "Unknown")
        album = m.get("album", "Unknown")
        genre = m.get("genre", "Unknown")
        sid = m.get("id", "")
        
        # Audio Features (Extract these so the LLM can reason about 'mood', 'speed', etc.)
        # We handle cases where they might be missing (defaults to N/A)
        features = []
        if "tempo" in m: features.append(f"Tempo: {m['tempo']} BPM")
        if "energy" in m: features.append(f"Energy: {m['energy']}")
        if "valence" in m: features.append(f"Valence (Mood): {m['valence']}")
        if "danceability" in m: features.append(f"Danceability: {m['danceability']}")
        if "popularity" in m: features.append(f"Popularity: {m['popularity']}")
        if "explicit" in m: features.append(f"Explicit: {m['explicit']}")
        
        features_str = " | ".join(features)
        
        # The 'snippet' (page_content) usually contains the full text from ingest,
        # but adding a structured header helps the LLM focus.
        snippet = d.page_content or ""

        chunks.append(
            f"[TRACK] {name}\n"
            f" - Artists: {artists}\n"
            f" - Album: {album}\n"
            f" - Genre: {genre}\n"
            f" - Audio Features: {features_str}\n"
            f" - SpotifyID: {sid}\n"
            f"--- Content dump ---\n{snippet}\n"
        )
    return "\n".join(chunks)


def format_sources(docs: List[Any]) -> str:
    """Prepare a short sources list for display."""
    seen = set()
    lines = []
    for d in docs:
        m = d.metadata or {}
        key = (m.get("name"), m.get("album"), m.get("artists"))
        if key in seen:
            continue
        seen.add(key)
        # Show Genre and Popularity in sources to help user verify relevance
        lines.append(f"- \"{m.get('name')}\" by {m.get('artists')} [{m.get('genre')}, Pop: {m.get('popularity')}]")
    
    if not lines:
        return "- No sources found."
    return "\n".join(lines)


# ---------------------------
# Build chain
# ---------------------------
def build_chain(cfg: Dict[str, str], k: int = 5, where: Dict[str, Any] | None = None):
    """
    Build a Retrieval-Augmented Generation chain:
      question -> retriever(context) -> prompt -> Azure Chat -> answer
    """
    # 1) Embeddings
    embeddings = AzureOpenAIEmbeddings(
        model=cfg["AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS"],
        api_key=cfg["AZURE_OPENAI_API_KEY"],
        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=cfg["AZURE_OPENAI_API_VERSION"],
    )

    # 2) Vector store
    if not os.path.exists(cfg["CHROMA_DIR"]):
        print(f"[WARN] Database path does not exist: {cfg['CHROMA_DIR']}")
    
    vectorstore = Chroma(
        persist_directory=cfg["CHROMA_DIR"],
        embedding_function=embeddings,
    )

    # 3) Retriever
    search_kwargs = {"k": k}
    if where:
        search_kwargs["where"] = where

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # 4) Chat model
    llm = AzureChatOpenAI(
        model=cfg["AZURE_OPENAI_DEPLOYMENT_NAME"],
        api_key=cfg["AZURE_OPENAI_API_KEY"],
        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
        api_version=cfg["AZURE_OPENAI_API_VERSION"],
        temperature=0.0,
    )

    # 5) Grounded prompt
    # Updated system prompt to mention audio features
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a music expert assistant. You answer questions based on a catalog of tracks.\n"
         "The context includes metadata like 'Valence' (musical positiveness, 0.0-1.0), 'Energy', 'Danceability', and 'Tempo'.\n"
         " - High Valence = Happy/Cheerful\n"
         " - Low Valence = Sad/Depressing\n"
         " - High Energy = Intense/Fast\n"
         "Use ONLY the provided context. If the answer is not present, reply exactly: \"I don't know.\""),
        ("human",
         "Question:\n{question}\n\n"
         "Context chunks:\n{context}\n\n"
         "Answer in 1-3 concise sentences. Recommend tracks if they match the user's description.")
    ])

    # 6) Pipeline
    retrieve_and_format = RunnableParallel(
        context=(retriever | format_docs),
        question=RunnablePassthrough()
    )

    chain = retrieve_and_format | prompt | llm | StrOutputParser()

    return chain, retriever, vectorstore


# ---------------------------
# CLI runner
# ---------------------------
def main():
    try:
        cfg = load_env()
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return

    # You can change k here if you need more candidates
    chain, retriever, _ = build_chain(cfg, k=5, where=None)

    print("\n[READY] RetrievalQA is running. Ask about songs, genres, or moods.")
    print("Example: 'Suggest a high energy techno track' or 'Find me a sad acoustic song'.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[BYE]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("[BYE]")
            break

        # Run RAG chain
        print("Thinking...")
        try:
            answer = chain.invoke(question)
        except Exception as e:
            print(f"[ERROR] Chain invocation failed: {e}")
            continue

        # Fetch docs again just for displaying sources (cheap local lookup)
        try:
            docs = retriever.invoke(question)
        except AttributeError:
            docs = retriever.get_relevant_documents(question)

        print("\n=== Answer ===")
        print(answer)
        print("\n--- Sources (Top 5 Matches) ---")
        print(format_sources(docs))
        print("-------------------------------\n")


if __name__ == "__main__":
    main()