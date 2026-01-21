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
    if s is None: return None
    s = s.strip().strip('"').strip("'")
    if s.lower().startswith("http") and s.endswith("/"):
        s = s[:-1]
    return s


def load_env() -> Dict[str, str]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1] 
    
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        env_path = project_root / ".env"
    
    load_dotenv(dotenv_path=env_path, override=False)

    default_chroma_path = project_root / "db" / "chroma"

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_ENDPOINT")
    cfg = {
        "AZURE_OPENAI_API_KEY": _clean(os.getenv("AZURE_OPENAI_API_KEY")),
        "AZURE_OPENAI_API_VERSION": _clean(os.getenv("AZURE_OPENAI_API_VERSION")),
        "AZURE_OPENAI_ENDPOINT": _clean(endpoint),
        "AZURE_OPENAI_DEPLOYMENT_NAME": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")),
        "AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS")),
        "CHROMA_DIR": _clean(os.getenv("CHROMA_DIR", str(default_chroma_path))),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env var(s): {missing}")
    return cfg


def format_docs(docs: List[Any]) -> str:
    chunks = []
    for d in docs:
        m = d.metadata or {}
        name = m.get("name", "Unknown")
        artists = m.get("artists", "Unknown")
        album = m.get("album", "Unknown")
        genre = m.get("genre", "Unknown")
        sid = m.get("id", "")
        
        features = []
        if "tempo" in m: features.append(f"Tempo: {m['tempo']} BPM")
        if "energy" in m: features.append(f"Energy: {m['energy']}")
        if "valence" in m: features.append(f"Valence (Mood): {m['valence']}")
        if "danceability" in m: features.append(f"Danceability: {m['danceability']}")
        if "popularity" in m: features.append(f"Popularity: {m['popularity']}")
        
        features_str = " | ".join(features)
        snippet = d.page_content or ""

        chunks.append(
            f"[TRACK] {name}\n"
            f" - Artists: {artists}\n"
            f" - Genre: {genre}\n"
            f" - Audio Features: {features_str}\n"
            f"--- Content dump ---\n{snippet}\n"
        )
    return "\n".join(chunks)


def format_sources(docs: List[Any]) -> str:
    seen = set()
    lines = []
    for d in docs:
        m = d.metadata or {}
        key = (m.get("name"), m.get("artists"))
        if key in seen: continue
        seen.add(key)
        lines.append(f"- \"{m.get('name')}\" by {m.get('artists')} [{m.get('genre')}, Pop: {m.get('popularity')}]")
    return "\n".join(lines) if lines else "- No sources found."


# ---------------------------
# Build chain
# ---------------------------
def build_chain(cfg: Dict[str, str], k: int = 10):
    embeddings = AzureOpenAIEmbeddings(
        model=cfg["AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS"],
        api_key=cfg["AZURE_OPENAI_API_KEY"],
        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=cfg["AZURE_OPENAI_API_VERSION"],
    )

    vectorstore = Chroma(
        persist_directory=cfg["CHROMA_DIR"],
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = AzureChatOpenAI(
        model=cfg["AZURE_OPENAI_DEPLOYMENT_NAME"],
        api_key=cfg["AZURE_OPENAI_API_KEY"],
        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
        api_version=cfg["AZURE_OPENAI_API_VERSION"],
        temperature=0.0,
    )

    # FRISSÍTETT PROMPT: Rugalmasabb "Popularity" kezelés
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a music expert assistant. You recommend tracks from the provided context.\n"
         "IMPORTANT:\n"
         "1. Use ONLY the provided context.\n"
         "2. If the user asks for 'popular' tracks, look at the 'Popularity' score (0-100). "
         "Even if scores are low (e.g., 30-40), recommend the highest ones available in the context as the 'most popular' options.\n"
         "3. If no tracks match the genre or mood at all, then say \"I don't know.\""),
        ("human",
         "Question:\n{question}\n\n"
         "Context chunks:\n{context}\n\n"
         "Answer in 1-3 concise sentences. Mention specific track names.")
    ])

    retrieve_and_format = RunnableParallel(
        context=(retriever | format_docs),
        question=RunnablePassthrough()
    )

    chain = retrieve_and_format | prompt | llm | StrOutputParser()
    return chain, retriever


# ---------------------------
# CLI runner
# ---------------------------
def main():
    cfg = load_env()
    # Növeltük k értékét 10-re a szélesebb merítésért
    chain, retriever = build_chain(cfg, k=10)

    print("\n[READY] RetrievalQA is running.")
    while True:
        try:
            question = input("Q> ").strip()
            if not question or question.lower() in {"exit", "quit"}: break

            print("Thinking...")
            answer = chain.invoke(question)
            docs = retriever.invoke(question)

            print(f"\n=== Answer ===\n{answer}")
            print(f"\n--- Sources (Top Matches) ---\n{format_sources(docs)}\n")
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()