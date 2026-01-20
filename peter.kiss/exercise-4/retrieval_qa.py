
# src/qa/retrieval_qa.py
# ------------------------------------------------------------
# Basic RetrievalQA over Chroma with Azure OpenAI Chat (RAG)
# - loads persistent Chroma built during ingest
# - builds a retriever (k=5 by default) with optional metadata filters
# - uses a grounded prompt: "Use only the provided context; otherwise say 'I don't know.'"
# - prints the final answer + sources (track/album/artist)
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
    """Load Bootcamp-style env vars with both endpoint name variants supported."""
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        # project root:  .../Spotify-quiz/.env
        env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_ENDPOINT")
    cfg = {
        "AZURE_OPENAI_API_KEY": _clean(os.getenv("AZURE_OPENAI_API_KEY")),
        "AZURE_OPENAI_API_VERSION": _clean(os.getenv("AZURE_OPENAI_API_VERSION")),
        "AZURE_OPENAI_ENDPOINT": _clean(endpoint),
        # Chat deployment (bootcamp szerint: gpt-4o-mini)
        "AZURE_OPENAI_DEPLOYMENT_NAME": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")),
        # Embedding deployment (ingestnél használtad)
        "AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS")),
        "CHROMA_DIR": _clean(os.getenv("CHROMA_DIR", "./db/chroma")),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env var(s): {missing}")
    return cfg


def format_docs(docs: List[Any]) -> str:
    """Join retrieved docs into a compact context block (English app)."""
    chunks = []
    for d in docs:
        m = d.metadata or {}
        name = m.get("name", "")
        artists = m.get("artists", "")
        album = m.get("album", "")
        genre = m.get("genre", "")
        sid = m.get("id", "")
        snippet = d.page_content or ""

        # Keep it compact to avoid token bloat; the model only needs essentials.
        chunks.append(
            f"[TRACK] {name} | Artists: {artists} | Album: {album} | Genre: {genre} | SpotifyID: {sid}\n"
            f"{snippet}\n"
        )
    return "\n---\n".join(chunks)


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
        lines.append(f"- Track: {m.get('name')} | Album: {m.get('album')} | Artists: {m.get('artists')}")
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
    # 1) Embeddings (same deployment as ingest)
    embeddings = AzureOpenAIEmbeddings(
        model=cfg["AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS"],  # Azure: deployment name
        api_key=cfg["AZURE_OPENAI_API_KEY"],
        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=cfg["AZURE_OPENAI_API_VERSION"],
    )

    # 2) Vector store (persistent Chroma)
    vectorstore = Chroma(
        persist_directory=cfg["CHROMA_DIR"],
        embedding_function=embeddings,
    )

    # 3) Retriever (k & metadata filter)
    search_kwargs = {"k": k}
    if where:
        # For Chroma retriever, use 'where' to filter metadata server-side.
        search_kwargs["where"] = where

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # 4) Chat model for generation (grounded by context)
    llm = AzureChatOpenAI(
        model=cfg["AZURE_OPENAI_DEPLOYMENT_NAME"],         # Azure: chat deployment name (e.g., gpt-4o-mini)
        api_key=cfg["AZURE_OPENAI_API_KEY"],
        azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
        api_version=cfg["AZURE_OPENAI_API_VERSION"],
        temperature=0.0,  # deterministic answers for QA
    )

    # 5) Grounded prompt (English app, strict about context)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about tracks in a music catalog.\n"
         "Use ONLY the provided context. If the answer is not present in the context, reply exactly: \"I don't know.\" "
         "Be concise, precise, and include only the necessary details."),
        ("human",
         "Question:\n{question}\n\n"
         "Context chunks (may be partial):\n{context}\n\n"
         "Answer in 1-3 concise sentences. If the answer isn't in the context, say \"I don't know.\"")
    ])

    # 6) Build a small pipeline:
    #    - left branch: retrieve docs -> format as context
    #    - right branch: passthrough the user question
    retrieve_and_format = RunnableParallel(
        context=(retriever | format_docs),
        question=RunnablePassthrough()
    )

    chain = retrieve_and_format | prompt | llm | StrOutputParser()

    # Expose vectorstore & retriever for debugging / evaluation (optional)
    return chain, retriever, vectorstore


# ---------------------------
# CLI runner
# ---------------------------
def main():
    cfg = load_env()

    # Example: you can set a default metadata filter here if desired
    # (e.g., restrict to acoustic tracks)
    default_filter = None
    # default_filter = {"genre": "acoustic"}

    chain, retriever, _ = build_chain(cfg, k=5, where=default_filter)

    print("\n[READY] RetrievalQA is running. Type your question (English).")
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
        answer = chain.invoke(question)

        # Also show the top-k docs (for sources)
        # NOTE: invoke() already triggered retrieval internally; to show sources,
        # we call the retriever again (cheap: vector search only).
        try:
            docs = retriever.invoke(question)
        except AttributeError:
            # fallback for older LC (rare)
            docs = retriever.get_relevant_documents(question)

        print("\n=== Answer ===")
        print(answer)
        print("\n--- Sources ---")
        print(format_sources(docs))
        print("--------------\n")


if __name__ == "__main__":
    main()
