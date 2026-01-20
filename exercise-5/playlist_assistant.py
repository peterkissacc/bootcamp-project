# exercise-5/playlist_assistant.py
import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from nl2filters import NLFilter, SYSTEM_PROMPT, interpret_nl_to_filter

def load_cfg() -> Dict[str, str]:
    env_path = find_dotenv(usecwd=True)
    if not env_path: env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    project_root = Path(__file__).resolve().parents[1]
    return {
        "API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
        "ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "CHAT_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "EMB_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS"),
        "CHROMA_DIR": str(project_root / "db" / "chroma"),
    }

def build_filters(filt: NLFilter) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    where = {}
    post = {}
    
    # Genre Handling (string or list)
    if filt.genre:
        if isinstance(filt.genre, dict) and "$in" in filt.genre:
            post["genre"] = [g.lower() for g in filt.genre["$in"]]
        else:
            post["genre"] = filt.genre.lower()

    # Musical Key & Mode (Numeric match)
    if filt.key is not None: where["key"] = int(filt.key)
    if filt.mode is not None: where["mode"] = int(filt.mode)
    if filt.explicit is not None: where["explicit"] = bool(filt.explicit)
    
    numeric_fields = ["popularity", "energy", "valence", "tempo", "duration_ms", "acousticness", "instrumentalness"]
    for f in numeric_fields:
        op = getattr(filt, f, None)
        if op:
            frag = op.to_chroma_filter()
            if frag: where[f] = frag
        
    final_where = {}
    if where:
        terms = [{k: v} for k, v in where.items()]
        final_where = terms[0] if len(terms) == 1 else {"$and": terms}
    return final_where, post

def apply_post_filters(docs: List[Any], post: Dict[str, Any]) -> List[Any]:
    if not post: return docs
    out = []
    for d in docs:
        m = d.metadata or {}
        ok = True
        for key, val in post.items():
            meta_val = str(m.get(key, "")).lower()
            if isinstance(val, list):
                if not any(v in meta_val for v in val): ok = False
            else:
                if val not in meta_val: ok = False
        if ok: out.append(d)
    return out

def smart_search(vectorstore, query, where, post, limit):
    # Attempt 1: Strict
    print(f"[DEBUG] Attempt 1: Strict filters (Key/Mode/Ranges)...")
    docs = vectorstore.similarity_search(query, k=400, filter=where if where else None)
    docs = apply_post_filters(docs, post)
    if len(docs) >= limit: return docs

    # Attempt 2: Relax technical, keep Genre
    print(f"[DEBUG] Attempt 2: Relaxing filters, keeping Genre...")
    more_docs = vectorstore.similarity_search(query, k=400)
    more_docs = apply_post_filters(more_docs, post)
    
    existing_ids = {d.metadata.get('unique_id') for d in docs}
    for d in more_docs:
        if d.metadata.get('unique_id') not in existing_ids:
            docs.append(d)
    
    if len(docs) >= limit: return docs

    # Attempt 3: Pure Semantic Fallback
    print(f"[DEBUG] Attempt 3: Pure semantic search fallback...")
    fallback = vectorstore.similarity_search(query, k=limit)
    for d in fallback:
        if d.metadata.get('unique_id') not in existing_ids:
            docs.append(d)
    return docs

def main():
    cfg = load_cfg()
    embeddings = AzureOpenAIEmbeddings(model=cfg["EMB_DEPLOYMENT"], api_key=cfg["API_KEY"],
                                       azure_endpoint=cfg["ENDPOINT"], openai_api_version=cfg["API_VERSION"])
    vectorstore = Chroma(persist_directory=cfg["CHROMA_DIR"], embedding_function=embeddings)
    llm = AzureChatOpenAI(model=cfg["CHAT_DEPLOYMENT"], api_key=cfg["API_KEY"],
                          azure_endpoint=cfg["ENDPOINT"], api_version=cfg["API_VERSION"], temperature=0.0)

    print(f"\n[READY] Database: {vectorstore._collection.count()} tracks.")

    while True:
        req = input("\nQ> ").strip()
        if req.lower() in ["exit", "quit"]: break
        if not req: continue

        raw_json = (ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{r}")]) 
                    | llm | StrOutputParser()).invoke({"r": req})
        
        print(f"\n[DEBUG] Raw JSON: {raw_json}")
        filt = interpret_nl_to_filter(raw_json)
        where, post = build_filters(filt)
        limit = filt.limit if filt.limit and filt.limit > 0 else 5
        
        docs = smart_search(vectorstore, req, where, post, limit)
        
        # Sorting
        if filt.sort_by == "popularity_desc":
            docs = sorted(docs, key=lambda d: d.metadata.get("popularity", 0), reverse=True)
        elif filt.sort_by == "popularity_asc":
            docs = sorted(docs, key=lambda d: d.metadata.get("popularity", 100))

        selected = docs[:min(limit, len(docs))]

        # Detailed AI Explanation
        meta_summary = "\n".join([f"Track: {m.get('name')}, Key: {m.get('key')}, Mode: {m.get('mode')}, Genre: {m.get('genre')}" for m in [d.metadata for d in selected]])
        explanation = llm.invoke(f"User request: {req}\nExplain why these {len(selected)} match: {meta_summary}").content

        print(f"\n=== RECOMMENDED ({len(selected)} track(s)) ===")
        for i, d in enumerate(selected, start=1):
            m = d.metadata
            print(f"[{i}] {m.get('name')} - {m.get('artists')} (Genre: {m.get('genre')})")
            print(f"    Key: {m.get('key')} (0=C, 2=D, etc.) | Mode: {m.get('mode')} (0=moll, 1=dúr)")
            print(f"    Energy: {m.get('energy')} | Tempo: {m.get('tempo')} | Pop: {m.get('popularity')}")
            print("-" * 50)
        
        print(f"\n[AI EXPLANATION]\n{explanation}")

if __name__ == "__main__":
    main()