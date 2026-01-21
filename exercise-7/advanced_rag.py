import os
import json
import re
import importlib.util
from pathlib import Path
from typing import TypedDict, Literal, Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

# --- 1. CONFIGURATION ---
def get_cfg():
    load_dotenv(find_dotenv(), override=True)
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_ENDPOINT")
    return {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "endpoint": endpoint.strip().rstrip("/") if endpoint else "",
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "emb_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS"),
        "chroma_dir": str(Path(__file__).resolve().parents[1] / "db" / "chroma")
    }

def get_llm(temp=0):
    cfg = get_cfg()
    return AzureChatOpenAI(
        azure_deployment=cfg["deployment"],
        api_version=cfg["api_version"],
        azure_endpoint=cfg["endpoint"],
        api_key=cfg["api_key"],
        temperature=temp
    )

def get_embeddings():
    cfg = get_cfg()
    return AzureOpenAIEmbeddings(
        azure_deployment=cfg["emb_deployment"],
        api_version=cfg["api_version"],
        azure_endpoint=cfg["endpoint"],
        api_key=cfg["api_key"]
    )

def get_nl2filters():
    path = Path(__file__).resolve().parents[1] / "exercise-5" / "nl2filters.py"
    spec = importlib.util.spec_from_file_location("nl2filters", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# --- 2. ADVANCED RAG CORE WITH DURATION SORTING ---
def advanced_retrieval_logic(query: str) -> Tuple[str, List[Dict]]:
    llm = get_llm()
    cfg = get_cfg()
    vectorstore = Chroma(persist_directory=cfg["chroma_dir"], embedding_function=get_embeddings())

    # --- PHASE 1: MULTI-QUERY ---
    print(f"\n--- [ADVANCED RAG: MULTI-QUERY] ---")
    mq_resp = (ChatPromptTemplate.from_messages([
        ("system", "Generate 2 variations of the user question. Focus on technical attributes if mentioned."),
        ("human", "{q}")
    ]) | llm | StrOutputParser()).invoke({"q": query})
    queries = [q.strip() for q in mq_resp.split("\n") if q.strip()] + [query]
    for i, q in enumerate(queries): print(f"  > Var {i+1}: {q}")

    # --- PHASE 2: RETRIEVAL ---
    all_docs = []
    for q in queries:
        all_docs.extend(vectorstore.similarity_search(q, k=10)) # Higher K for technical queries
    
    unique_docs = []
    seen = set()
    for d in all_docs:
        uid = d.metadata.get("id") or hash(d.page_content)
        if uid not in seen:
            unique_docs.append(d); seen.add(uid)
    
    # --- PHASE 3: RE-RANKING & DYNAMIC SORTING ---
    print(f"\n--- [ADVANCED RAG: RE-RANKING] ---")
    final_docs = []
    is_from_db = False

    if unique_docs:
        doc_context = "\n".join([f"ID:{i} | {d.page_content} | Metadata: {d.metadata}" for i, d in enumerate(unique_docs)])
        rr_resp = (ChatPromptTemplate.from_messages([
            ("system", "Return a JSON list of the Top 5 IDs. Even if the text doesn't explicitly mention 'length', select the tracks that match the Artist/Title so we can check their metadata."),
            ("human", "Query: {q}\nDocs:\n{docs}")
        ]) | llm | StrOutputParser()).invoke({"q": query, "docs": doc_context})
        
        print(f"  > [DEBUG RAW JSON RE-RANK]: {rr_resp}")
        
        try:
            json_match = re.search(r"\[.*\]", rr_resp, re.DOTALL)
            top_ids = json.loads(json_match.group(0)) if json_match else []
            final_docs = [unique_docs[i] for i in top_ids if i < len(unique_docs)]
            
            # --- SUPERLATIVE SORTING (Including Duration) ---
            q_lower = query.lower()
            if "longest" in q_lower or "duration" in q_lower or "length" in q_lower:
                final_docs.sort(key=lambda x: x.metadata.get('duration_ms', 0), reverse=True)
                print("  > [DEBUG] Sorted by Duration (Descending)")
            elif "shortest" in q_lower:
                final_docs.sort(key=lambda x: x.metadata.get('duration_ms', 0), reverse=False)
                print("  > [DEBUG] Sorted by Duration (Ascending)")
            elif "popular" in q_lower:
                reverse_sort = "least" not in q_lower
                final_docs.sort(key=lambda x: x.metadata.get('popularity', 0), reverse=reverse_sort)
            elif "fastest" in q_lower or "tempo" in q_lower:
                final_docs.sort(key=lambda x: x.metadata.get('tempo', 0), reverse=True)
            
            if final_docs: is_from_db = True
        except:
            final_docs = unique_docs[:5]
            is_from_db = True

    # --- PHASE 4: SYNTHESIS ---
    print(f"\n--- [ADVANCED RAG: SYNTHESIS] ---")
    source_label = "[SOURCE: CHROMA DB]" if is_from_db else "[SOURCE: LLM KNOWLEDGE - NOT IN DB]"
    context_str = "\n".join([str(d.metadata) for d in final_docs]) if final_docs else "NO_CONTEXT"
    
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", f"Prefix: {source_label}\n"
                   "Identify the winner based on the metadata values provided. Convert duration_ms to minutes and seconds for the user."),
        ("human", "Query: {q}\nContext: {c}")
    ])
    
    explanation = (gen_prompt | llm | StrOutputParser()).invoke({"q": query, "c": context_str})
    return explanation, [d.metadata for d in final_docs]

# --- 3. NODES ---
class RouterState(TypedDict, total=False):
    query: str
    intent: str
    results: List[Dict[str, Any]]
    answer: str

def classify_node(state: RouterState) -> RouterState:
    print(f"\n>>> [NODE: CLASSIFY] Identifying intent...")
    intent = (ChatPromptTemplate.from_messages([
        ("system", "Classify as 'playlist' or 'info'."),
        ("human", "{q}")
    ]) | get_llm() | StrOutputParser()).invoke({"q": state["query"]}).lower().strip()
    return {"intent": "playlist" if "playlist" in intent else "info"}

def playlist_node(state: RouterState) -> RouterState:
    print(f"\n>>> [NODE: PLAYLIST] Metadata search...")
    llm = get_llm()
    query_text = state["query"].lower()
    has_all = re.search(r'\ball\b', query_text)
    num_match = re.findall(r'\b\d+\b', query_text)
    target_limit = 100 if has_all else (int(num_match[0]) if num_match else 5)

    vectorstore = Chroma(persist_directory=get_cfg()["chroma_dir"], embedding_function=get_embeddings())
    docs = vectorstore.similarity_search(state["query"], k=max(target_limit * 5, 50))
    
    final_results = []; seen = set()
    for d in docs:
        key = f"{d.metadata.get('name')}-{d.metadata.get('artists')}".lower()
        if key not in seen:
            final_results.append(d.metadata); seen.add(key)
        if len(final_results) >= target_limit: break

    ans = (ChatPromptTemplate.from_messages([
        ("system", "Explain selection based on metadata."),
        ("human", "Req: {q}\nMeta: {m}")
    ]) | llm | StrOutputParser()).invoke({"q": state["query"], "m": str(final_results)})
    return {"results": final_results, "answer": ans}

def info_node(state: RouterState) -> RouterState:
    print(f"\n>>> [NODE: INFO] Advanced RAG execution...")
    ans, meta = advanced_retrieval_logic(state["query"])
    return {"answer": ans, "results": meta}

# --- 4. GRAPH & MAIN ---
def build_graph():
    builder = StateGraph(RouterState)
    builder.add_node("classify", classify_node); builder.add_node("playlist", playlist_node); builder.add_node("info", info_node)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", lambda s: s["intent"])
    for n in ["playlist", "info"]: builder.add_edge(n, END)
    return builder.compile()

if __name__ == "__main__":
    app = build_graph()
    print("\n[SYSTEM] Master Router Active. (Enhanced with Duration Support)")
    while True:
        try:
            q = input("\nQ> ").strip()
            if q.lower() in ["exit", "quit"]: break
            if not q: continue
            out = app.invoke({"query": q})
            print(f"\n{'='*20} RESPONSE {'='*20}\n{out.get('answer')}")
            
            res = out.get('results', [])
            print(f"\n--- FULL DATABASE ATTRIBUTES (Debug Mode: {len(res)} items) ---")
            for i, r in enumerate(res, 1):
                print(f"{i}. {r.get('name')} by {r.get('artists')}")
                for key, value in r.items():
                    if key not in ['name', 'artists']:
                        print(f"   | {key}: {value}")
            print("="*60)
        except Exception as e: print(f"\n[ERROR]: {e}")