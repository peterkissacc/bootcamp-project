
# exercise-6/langgraph_router.py
# ------------------------------------------------------------
# LangGraph Router / Orchestrator
# - Intent classification (playlist | info | debug | fallback)
# - Playlist branch: NL→JSON filter (uses Exercise-5 schema) → Chroma → post-filters → selection
# - Info branch: Grounded Retrieval QA ("Use only the provided context; else 'I don't know'")
# - Debug branch: collection count & CHROMA_DIR
# - Fallback branch: plain semantic search
# - Resolves CHROMA_DIR to absolute path (relative to .env)
# - Dynamically imports nl2filters from exercise-5 (hyphenated folder)
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import json
import random
import importlib.util
from pathlib import Path
from typing import TypedDict, Literal, Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# -----------------------
# ENV & config
# -----------------------
def _clean(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().strip('"').strip("'")
    if s.lower().startswith("http") and s.endswith("/"):
        s = s[:-1]
    return s

def load_cfg() -> Dict[str, str]:
    """Load .env (repo root) and resolve CHROMA_DIR to absolute path."""
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        # router is under: .../bootcamp-project/exercise-6/langgraph_router.py
        env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_ENDPOINT")
    cfg = {
        "API_KEY": _clean(os.getenv("AZURE_OPENAI_API_KEY")),
        "API_VERSION": _clean(os.getenv("AZURE_OPENAI_API_VERSION")),
        "ENDPOINT": _clean(endpoint),  # <-- fixed: was __clean
        "CHAT_DEPLOYMENT": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")),              # e.g., gpt-4o-mini
        "EMB_DEPLOYMENT": _clean(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS")),   # e.g., text-embedding-3-large
        "CHROMA_DIR": _clean(os.getenv("CHROMA_DIR", "./db/chroma")),
        "ENV_PATH": str(env_path),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env var(s): {missing}")

    # Resolve CHROMA_DIR relative to the .env directory (repo root)
    base_dir = Path(env_path).parent
    chroma_dir_path = Path(cfg["CHROMA_DIR"])
    if not chroma_dir_path.is_absolute():
        cfg["CHROMA_DIR"] = str((base_dir / chroma_dir_path).resolve())

    return cfg


# -----------------------
# Dynamic import of exercise-5/nl2filters.py
# -----------------------
def import_nl2filters() -> Any:
    """
    Dynamically import nl2filters.py from exercise-5 (hyphenated folder).
    Returns the imported module object.
    """
    base = Path(__file__).resolve().parents[1]
    nl_path = base / "exercise-5" / "nl2filters.py"
    if not nl_path.exists():
        raise FileNotFoundError(f"Could not find nl2filters.py at: {nl_path}")
    spec = importlib.util.spec_from_file_location("nl2filters", nl_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# -----------------------
# Clients and utilities
# -----------------------
def get_clients(cfg: Dict[str, str]):
    llm = AzureChatOpenAI(
        model=cfg["CHAT_DEPLOYMENT"],
        api_key=cfg["API_KEY"],
        azure_endpoint=cfg["ENDPOINT"],
        api_version=cfg["API_VERSION"],
        temperature=0.0,
    )
    embeddings = AzureOpenAIEmbeddings(
        model=cfg["EMB_DEPLOYMENT"],
        api_key=cfg["API_KEY"],
        azure_endpoint=cfg["ENDPOINT"],
        openai_api_version=cfg["API_VERSION"],
    )
    vectorstore = Chroma(
        persist_directory=cfg["CHROMA_DIR"],
        embedding_function=embeddings,
    )
    return llm, embeddings, vectorstore

def collection_count(vectorstore: Chroma) -> int:
    try:
        return int(vectorstore._collection.count())
    except Exception:
        return -1

def normalize_where(where: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma >=0.5 expects one top-level operator; wrap multiple terms in {$and: [...]}."""
    if not where:
        return {}
    terms: List[Dict[str, Any]] = [{k: v} for k, v in where.items()]
    return terms[0] if len(terms) == 1 else {"$and": terms}

def to_snippets(docs) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata or {}
        lines.append(
            f"[{i}] name='{m.get('name')}', artists='{m.get('artists')}', album='{m.get('album')}', "
            f"genre='{m.get('genre')}', duration_ms='{m.get('duration_ms')}', explicit='{m.get('explicit')}', "
            f"popularity='{m.get('popularity')}'"
        )
    return "\n".join(lines)

def parse_requested_count(request: str) -> int | None:
    """
    Extracts a requested count from the natural-language query if present.
    Examples: "give me 7", "show me 3", "list 10", "return 4".
    """
    text = request.lower()
    m = re.search(r"\b(?:give|show|list|return|provide|send|display)\s+(?:me\s+)?(\d{1,2})\b", text)
    if m:
        try:
            n = int(m.group(1))
            return n if 1 <= n <= 50 else None
        except Exception:
            return None
    return None


# -----------------------
# Playlist helpers (build where + post filters, search, select)
# -----------------------
def build_where_and_post_filters_from_filter(filt: Any) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Build server-side 'where' (no $contains) + client-side post-filters (contains/not_contains).
    Compatible with nl2filters.NLFilter.
    """
    where: Dict[str, Any] = {}
    # Server-side equality/range:
    if getattr(filt, "genre", None):
        where["genre"] = filt.genre
    if getattr(filt, "explicit", None) is not None:
        where["explicit"] = bool(filt.explicit)
    if getattr(filt, "duration_ms", None):
        frag = filt.duration_ms.to_chroma_filter()
        if frag:
            where["duration_ms"] = frag
    if getattr(filt, "popularity", None):
        frag = filt.popularity.to_chroma_filter()
        if frag:
            where["popularity"] = frag

    # Client-side contains / not_contains:
    post: Dict[str, str] = {}
    if getattr(filt, "artists_contains", None):
        post["artists_contains"] = filt.artists_contains.lower()
    if getattr(filt, "name_contains", None):
        post["name_contains"] = filt.name_contains.lower()
    if getattr(filt, "artists_not_contains", None):
        post["artists_not_contains"] = filt.artists_not_contains.lower()
    if getattr(filt, "name_not_contains", None):
        post["name_not_contains"] = filt.name_not_contains.lower()

    return normalize_where(where), post

def apply_post_filters(docs: List[Any], post: Dict[str, str]) -> List[Any]:
    """Client-side contains/not_contains filters on artists/name (case-insensitive)."""
    if not post:
        return docs
    a_contains = post.get("artists_contains")
    n_contains = post.get("name_contains")
    a_not = post.get("artists_not_contains")
    n_not = post.get("name_not_contains")

    out: List[Any] = []
    for d in docs:
        m = d.metadata or {}
        artists = str(m.get("artists", "")).lower()
        name = str(m.get("name", "")).lower()

        ok = True
        if a_contains and a_contains not in artists:
            ok = False
        if n_contains and n_contains not in name:
            ok = False
        if ok and a_not and a_not in artists:
            ok = False
        if ok and n_not and n_not in name:
            ok = False

        if ok:
            out.append(d)

    return out

def search_with_filter(vectorstore: Chroma, query: str, where: Dict[str, Any], post: Dict[str, str], k: int = 5) -> List[Any]:
    """
    Robust search:
    - If no where: plain similarity.
    - Else: retriever(where), fallback to similarity(filter/where).
    - Then apply client-side post-filters.
    - If empty, retry with larger k; then plain similarity + post-filter.
    """
    if not where:
        docs = vectorstore.similarity_search(query, k=k)
        return apply_post_filters(docs, post)

    docs: List[Any] = []
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k, "where": where})
        docs = retriever.invoke(query)
    except Exception:
        pass

    if not docs:
        try:
            docs = vectorstore.similarity_search(query, k=k, filter=where)
        except TypeError:
            docs = vectorstore.similarity_search(query, k=k, where=where)

    docs = apply_post_filters(docs, post)
    if docs:
        return docs

    k2 = max(k * 5, 25)
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k2, "where": where})
        docs = retriever.invoke(query)
    except Exception:
        docs = []
    if not docs:
        try:
            docs = vectorstore.similarity_search(query, k=k2, filter=where)
        except TypeError:
            docs = vectorstore.similarity_search(query, k=k2, where=where)

    docs = apply_post_filters(docs, post)
    if docs:
        return docs

    docs = vectorstore.similarity_search(query, k=k2)
    docs = apply_post_filters(docs, post)
    return docs

def select_docs_for_playlist(docs: List[Any], filt: Any, request: str) -> List[Any]:
    """
    Selection policy (same idea as in Exercise-5):
    - If 'sort_by' is set: respect sorting and take top N (N = filt.limit or parsed_count or 5).
    - Else: RANDOM selection among matches (N = filt.limit or parsed_count or random 3..5).
    """
    if not docs:
        return docs

    def _get(f, attr, default=None):
        return getattr(f, attr, default) if hasattr(f, attr) else default

    parsed = parse_requested_count(request)
    limit = int(_get(filt, "limit", 0) or 0) or (parsed or 0)

    sort_by = _get(filt, "sort_by", None)
    if sort_by == "popularity_desc":
        docs = sorted(docs, key=lambda d: (d.metadata or {}).get("popularity") or -1, reverse=True)
        n = limit if limit > 0 else 5
        return docs[: min(n, len(docs))]
    if sort_by == "popularity_asc":
        docs = sorted(docs, key=lambda d: (d.metadata or {}).get("popularity") or 1_000_000)
        n = limit if limit > 0 else 5
        return docs[: min(n, len(docs))]

    if limit <= 0:
        limit = random.randint(3, 5)
    limit = min(limit, len(docs))
    if limit >= len(docs):
        random.shuffle(docs)
        return docs
    return random.sample(docs, limit)


# -----------------------
# Router state
# -----------------------
class RouterState(TypedDict, total=False):
    query: str
    intent: Literal["playlist", "info", "debug", "fallback"]
    results: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]
    debug: Dict[str, Any]


# -----------------------
# Intent classifier node
# -----------------------
def build_intent_classifier(cfg: Dict[str, str]):
    system_text = (
        "Classify the user's request into exactly one of these intents: "
        "playlist | info | debug | fallback.\n"
        "- 'playlist' if the user asks to find/suggest tracks/playlists using constraints (duration, popularity, genre, explicit, contains/not_contains, etc.).\n"
        "- 'info' if the user asks a factual question about tracks/artists/albums (who, which album, etc.).\n"
        "- 'debug' if the user asks about diagnostics (counts, where the DB is, collection size).\n"
        "- 'fallback' for anything else.\n"
        "Return ONLY one token: playlist | info | debug | fallback."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "{query}")
    ])
    llm = AzureChatOpenAI(
        model=cfg["CHAT_DEPLOYMENT"],
        api_key=cfg["API_KEY"],
        azure_endpoint=cfg["ENDPOINT"],
        api_version=cfg["API_VERSION"],
        temperature=0.0,
    )
    chain = prompt | llm
    def node(state: RouterState) -> RouterState:
        out = chain.invoke({"query": state["query"]})
        text = out.content.strip().lower()
        intent = "fallback"
        if "playlist" in text:
            intent = "playlist"
        elif "info" in text:
            intent = "info"
        elif "debug" in text:
            intent = "debug"
        return {"intent": intent}
    return node


# -----------------------
# Playlist node
# -----------------------
def build_playlist_node(cfg: Dict[str, str], vectorstore: Chroma):
    # LLM for explanations
    explainer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You briefly explain why each retrieved track matches the user's request. "
         "Use ONLY metadata: name, artists, album, genre, duration_ms, explicit, popularity. "
         "Do not invent facts. Keep 1–2 sentences."),
        ("human", "User request:\n{request}\n\nTop results (metadata only):\n{snippets}\n\nExplain briefly:")
    ])
    llm = AzureChatOpenAI(
        model=cfg["CHAT_DEPLOYMENT"],
        api_key=cfg["API_KEY"],
        azure_endpoint=cfg["ENDPOINT"],
        api_version=cfg["API_VERSION"],
        temperature=0.0,
    )

    nl2filters = import_nl2filters()
    # Escape braces in SYSTEM_PROMPT for LC templates
    system_text = nl2filters.SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")
    filter_prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "{user_prompt}")
    ])

    def node(state: RouterState) -> RouterState:
        # 1) NL → JSON filter
        raw_json = (filter_prompt | llm).invoke({"user_prompt": nl2filters.build_user_prompt(state["query"])}).content.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.strip("`")
            if raw_json.lower().startswith("json"):
                raw_json = raw_json[4:].strip()
        try:
            json.loads(raw_json)
        except Exception:
            raw_json = "{}"

        filt = nl2filters.interpret_nl_to_filter(raw_json)

        # 2) where + post, search, select
        where, post = build_where_and_post_filters_from_filter(filt)
        docs = search_with_filter(vectorstore, state["query"], where, post, k=5)
        if not docs:
            return {"results": [], "answer": "No results for the current filters. Try relaxing constraints."}

        docs = select_docs_for_playlist(docs, filt, state["query"])

        # 3) explanation + results
        snippets = to_snippets(docs)
        explanation = (explainer_prompt | llm).invoke({"request": state["query"], "snippets": snippets}).content
        results = [
            {
                "name": d.metadata.get("name"),
                "artists": d.metadata.get("artists"),
                "album": d.metadata.get("album"),
                "genre": d.metadata.get("genre"),
                "explicit": d.metadata.get("explicit"),
                "duration_ms": d.metadata.get("duration_ms"),
                "popularity": d.metadata.get("popularity"),
            } for d in docs
        ]
        return {"results": results, "answer": explanation}

    return node


# -----------------------
# Info (RetrievalQA) node
# -----------------------
def build_info_node(cfg: Dict[str, str], vectorstore: Chroma):
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You answer questions about tracks/artists/albums using ONLY the provided context. "
         "If the answer is not present in the context, reply exactly: \"I don't know.\""),
        ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer in 1–3 concise sentences.")
    ])
    llm = AzureChatOpenAI(
        model=cfg["CHAT_DEPLOYMENT"],
        api_key=cfg["API_KEY"],
        azure_endpoint=cfg["ENDPOINT"],
        api_version=cfg["API_VERSION"],
        temperature=0.0,
    )
    def node(state: RouterState) -> RouterState:
        docs = vectorstore.similarity_search(state["query"], k=6)
        if not docs:
            return {"answer": "I don't know.", "sources": []}
        context = "\n---\n".join([d.page_content for d in docs])
        answer = (qa_prompt | llm).invoke({"question": state["query"], "context": context}).content
        sources = [
            {
                "name": d.metadata.get("name"),
                "album": d.metadata.get("album"),
                "artists": d.metadata.get("artists")
            } for d in docs
        ]
        return {"answer": answer, "sources": sources}
    return node


# -----------------------
# Debug node
# -----------------------
def build_debug_node(cfg: Dict[str, str], vectorstore: Chroma):
    def node(state: RouterState) -> RouterState:
        return {"debug": {"collection_count": collection_count(vectorstore),
                          "chroma_dir": cfg["CHROMA_DIR"]}}
    return node


# -----------------------
# Fallback node (semantic search only)
# -----------------------
def build_fallback_node(cfg: Dict[str, str], vectorstore: Chroma):
    def node(state: RouterState) -> RouterState:
        docs = vectorstore.similarity_search(state["query"], k=5)
        if not docs:
            return {"results": [], "answer": "No results."}
        results = [
            {
                "name": d.metadata.get("name"),
                "artists": d.metadata.get("artists"),
                "album": d.metadata.get("album"),
                "genre": d.metadata.get("genre"),
                "explicit": d.metadata.get("explicit"),
                "duration_ms": d.metadata.get("duration_ms"),
                "popularity": d.metadata.get("popularity"),
            } for d in docs
        ]
        return {"results": results, "answer": "Top semantic matches (no filters)."}
    return node


# -----------------------
# Build graph
# -----------------------
def build_graph():
    cfg = load_cfg()
    llm, embeddings, vectorstore = get_clients(cfg)

    graph = StateGraph(RouterState)

    # Nodes
    graph.add_node("classify", build_intent_classifier(cfg))
    graph.add_node("playlist", build_playlist_node(cfg, vectorstore))
    graph.add_node("info", build_info_node(cfg, vectorstore))
    graph.add_node("debug", build_debug_node(cfg, vectorstore))
    graph.add_node("fallback", build_fallback_node(cfg, vectorstore))

    # Edges
    graph.add_edge(START, "classify")

    def route(state: RouterState) -> str:
        intent = state.get("intent", "fallback")
        if intent in {"playlist", "info", "debug"}:
            return intent
        return "fallback"

    graph.add_conditional_edges("classify", route, {
        "playlist": "playlist",
        "info": "info",
        "debug": "debug",
        "fallback": "fallback",
    })

    # End every branch
    for n in ["playlist", "info", "debug", "fallback"]:
        graph.add_edge(n, END)

    return graph.compile()


# -----------------------
# CLI
# -----------------------
def main():
    app = build_graph()
    cfg = load_cfg()
    _, _, vectorstore = get_clients(cfg)
    print(f"[DEBUG] CHROMA_DIR = {cfg['CHROMA_DIR']}")
    print(f"[DEBUG] Collection count = {collection_count(vectorstore)}\n")

    print("[READY] LangGraph Router. Type your request (English). Type 'exit' to quit.\n")
    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[BYE]")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("[BYE]")
            break

        state: RouterState = {"query": q}
        out: RouterState = app.invoke(state)

        # Pretty-print result
        if out.get("debug"):
            print("\n[DEBUG INFO]")
            print(json.dumps(out["debug"], indent=2))
        if out.get("answer"):
            print("\n[ANSWER]")
            print(out["answer"])
        if out.get("results"):
            print("\n[RESULTS]")
            for i, r in enumerate(out["results"], start=1):
                print(f"  [{i}] {r.get('name')} | {r.get('artists')} | {r.get('album')} | "
                      f"genre={r.get('genre')} | explicit={r.get('explicit')} | "
                      f"duration_ms={r.get('duration_ms')} | popularity={r.get('popularity')}")
        if out.get("sources"):
            print("\n[SOURCES]")
            for s in out["sources"]:
                print(f"  - {s.get('name')} | {s.get('album')} | {s.get('artists')}")
        print("\n" + "-"*70 + "\n")

if __name__ == "__main__":
    main()
