import os
import json
import re
import sqlite3
import operator
from pathlib import Path
from typing import TypedDict, Dict, Any, List, Annotated

from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- 1. CONFIGURATION ---
def get_cfg():
    load_dotenv(find_dotenv(), override=True)
    return {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT").strip().rstrip("/"),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "emb_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDINGS"),
        "chroma_dir": str(Path(__file__).resolve().parents[1] / "db" / "chroma"),
        "memory_db": "memory_ex8_stable.db"
    }

def get_llm(temp=0):
    cfg = get_cfg()
    return AzureChatOpenAI(azure_deployment=cfg["deployment"], api_version=cfg["api_version"], 
                          azure_endpoint=cfg["endpoint"], api_key=cfg["api_key"], temperature=temp)

def get_embeddings():
    cfg = get_cfg()
    return AzureOpenAIEmbeddings(azure_deployment=cfg["emb_deployment"], api_version=cfg["api_version"], 
                                azure_endpoint=cfg["endpoint"], api_key=cfg["api_key"])

# --- 2. STATE ---
class RouterState(TypedDict):
    query: str
    processed_data: Dict[str, int]
    processed_query: str
    intent: str
    results: List[Dict[str, Any]]
    answer: str
    messages: Annotated[List[BaseMessage], operator.add]

# --- 3. NODES ---

def initial_router_node(state: RouterState) -> Dict:
    print(f"\n--- [STEP 1: ROUTER] ---")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify user intent.\n"
                   "- 'playlist': modifying list, adding/removing songs, creating lists.\n"
                   "- 'info': asking facts, who wrote, details, longest, fastest, what is."),
        ("human", "{q}")
    ])
    
    q = state["query"].lower()
    if any(x in q for x in ["create", "list", "add", "remove", "more"]):
        val = "playlist"
    elif any(x in q for x in ["what", "who", "longest", "fastest", "details"]):
        val = "info"
    else:
        intent = (prompt | llm | StrOutputParser()).invoke({"q": state["query"]}).lower()
        val = "playlist" if "playlist" in intent else "info"
    
    print(f"    [DEBUG] Routing to: {val.upper()}")
    return {"intent": val}

def state_manager_node(state: RouterState) -> Dict:
    print(f"\n--- [STEP 2A: STATE MANAGER] ---")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Playlist Math Engine. Output ONLY JSON.\n"
                   "1. Analyze history for current song counts.\n"
                   "2. Apply user changes.\n"
                   "3. Return a JSON object where keys are Artist Names and values are Integers.\n"
                   "Example: {{\"Metallica\": 2, \"The Prodigy\": 3}}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{q}")
    ])
    try:
        plan = (prompt | llm | JsonOutputParser()).invoke({"q": state["query"], "history": state["messages"]})
    except:
        plan = {"error": 1}
    print(f"    [DEBUG] State: {plan}")
    return {"processed_data": plan}

def query_optimizer_node(state: RouterState) -> Dict:
    print(f"\n--- [STEP 2B: QUERY OPTIMIZER] ---")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite query for vector search. Keep artist names."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{q}")
    ])
    opt = (prompt | llm | StrOutputParser()).invoke({"q": state["query"], "history": state["messages"]})
    print(f"    [DEBUG] Search Query: {opt}")
    return {"processed_query": opt}

def playlist_exec_node(state: RouterState) -> Dict:
    print(f"\n--- [STEP 3: PLAYLIST FETCH] ---")
    vectorstore = Chroma(persist_directory=get_cfg()["chroma_dir"], embedding_function=get_embeddings())
    targets = state.get("processed_data", {})
    if not targets or "error" in targets: targets = {state["query"]: 5}

    final_results = []
    global_seen = set()

    for artist, target_count in targets.items():
        print(f"    [ACTION] Searching for {target_count} x '{artist}'...")
        docs = vectorstore.similarity_search(artist, k=max(target_count * 5, 20))
        found = 0
        for d in docs:
            meta_artist = d.metadata.get("artists", "").lower()
            if artist.lower() in meta_artist or artist.lower() in d.metadata.get("name", "").lower():
                key = f"{d.metadata.get('name')}-{meta_artist}"
                if key not in global_seen:
                    d.metadata["source"] = "DB"
                    final_results.append(d.metadata)
                    global_seen.add(key)
                    found += 1
            if found >= target_count: break
        
        if found < target_count:
            for i in range(target_count - found):
                final_results.append({"name": f"Suggested {artist} Track {i+1}", "artists": artist, "source": "LLM"})

    ans_prompt = ChatPromptTemplate.from_messages([
        ("system", "Confirm list update."),
        ("human", "List: {r}")
    ])
    answer = (ans_prompt | get_llm() | StrOutputParser()).invoke({"r": str(final_results)})
    return {"results": final_results, "answer": answer, "messages": [AIMessage(content=answer)]}

def info_exec_node(state: RouterState) -> Dict:
    print(f"\n--- [STEP 3: INFO RAG] ---")
    llm = get_llm()
    vectorstore = Chroma(persist_directory=get_cfg()["chroma_dir"], embedding_function=get_embeddings())
    
    docs = vectorstore.similarity_search(state["processed_query"], k=10)
    
    # FIX: Convert Document objects to Dictionaries immediately
    results_list = []
    for d in docs:
        meta = d.metadata
        meta["source"] = "DB" # Tag it here
        results_list.append(meta)
    
    # Sort
    q_lower = state["query"].lower()
    if "longest" in q_lower:
        results_list.sort(key=lambda x: x.get('duration_ms', 0), reverse=True)
    elif "fastest" in q_lower:
        results_list.sort(key=lambda x: x.get('tempo', 0), reverse=True)
    
    top_results = results_list[:5]

    ans_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question. Cite source as [DB] or [LLM]."),
        ("human", "Query: {q}\nContext: {c}")
    ])
    answer = (ans_prompt | llm | StrOutputParser()).invoke({
        "q": state["query"], 
        "c": str(top_results)
    })
    
    # FIX: Return the list of DICTIONARIES, not Documents
    return {"answer": answer, "results": top_results, "messages": [AIMessage(content=answer)]}

# --- 4. GRAPH ---
def build_graph(checkpointer):
    builder = StateGraph(RouterState)
    builder.add_node("router", initial_router_node)
    builder.add_node("state_manager", state_manager_node)
    builder.add_node("query_optimizer", query_optimizer_node)
    builder.add_node("playlist_exec", playlist_exec_node)
    builder.add_node("info_exec", info_exec_node)
    
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", lambda s: s["intent"], 
                                  {"playlist": "state_manager", "info": "query_optimizer"})
    builder.add_edge("state_manager", "playlist_exec")
    builder.add_edge("query_optimizer", "info_exec")
    builder.add_edge("playlist_exec", END)
    builder.add_edge("info_exec", END)
    return builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    conn = sqlite3.connect(get_cfg()["memory_db"], check_same_thread=False)
    memory = SqliteSaver(conn)
    app = build_graph(memory)
    config = {"configurable": {"thread_id": "STABLE_SESSION_V3"}}

    print("\n[SYSTEM] Exercise 8: Stable Version")
    while True:
        try:
            q = input("\nQ> ").strip()
            if q.lower() in ["exit", "quit"]: break
            output = app.invoke({"query": q, "messages": [HumanMessage(content=q)]}, config=config)
            
            print(f"\n{'='*20} FINAL SOLUTION {'='*20}")
            print(f"AI: {output.get('answer')}")
            print(f"\n--- DETAILED TRACK LIST ---")
            
            # This loop is now safe because we ensured 'results' contains dicts
            for i, r in enumerate(output.get('results', []), 1):
                print(f"{i}. [{r.get('source', 'LLM')}] {r.get('name')} - {r.get('artists')}")
            print("="*60)
        except Exception as e: 
            import traceback
            traceback.print_exc()
            print(f"[ERROR]: {e}")