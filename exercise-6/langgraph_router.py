import os
import json
import importlib.util
from pathlib import Path
from typing import TypedDict, Literal, Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

# --- Central Configuration and Clients ---
def get_cfg():
    """Loads environment variables and returns a configuration dictionary."""
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

def get_llm():
    """Initializes and returns the Azure OpenAI Chat model."""
    cfg = get_cfg()
    return AzureChatOpenAI(
        azure_deployment=cfg["deployment"],
        api_version=cfg["api_version"],
        azure_endpoint=cfg["endpoint"],
        api_key=cfg["api_key"],
        temperature=0
    )

def get_embeddings():
    """Initializes and returns the Azure OpenAI Embeddings model."""
    cfg = get_cfg()
    return AzureOpenAIEmbeddings(
        azure_deployment=cfg["emb_deployment"],
        api_version=cfg["api_version"],
        azure_endpoint=cfg["endpoint"],
        api_key=cfg["api_key"]
    )

# --- Dynamic Module Loader ---
def get_nl2filters():
    """Dynamically imports the nl2filters module from exercise-5."""
    path = Path(__file__).resolve().parents[1] / "exercise-5" / "nl2filters.py"
    spec = importlib.util.spec_from_file_location("nl2filters", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# --- State and Nodes ---
class RouterState(TypedDict, total=False):
    query: str
    intent: str
    results: List[Dict[str, Any]]
    answer: str

def classify_node(state: RouterState) -> RouterState:
    """Classifies the user query into one of the predefined intents."""
    print(f"\n>>> [NODE: CLASSIFY] Analyzing intent...")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify intent: playlist (request for music), info (facts about songs/artists), debug (system status), or fallback (unclear). Return ONLY one word."),
        ("human", "{query}")
    ])
    intent = (prompt | llm | StrOutputParser()).invoke({"query": state["query"]}).lower().strip()
    print(f"    DECISION: Routing to -> '{intent}'")
    return {"intent": intent}

def playlist_node(state: RouterState) -> RouterState:
    """Handles music requests by converting NL to JSON filters and searching ChromaDB."""
    print(f"\n>>> [NODE: PLAYLIST] Generating music list...")
    nl2filters = get_nl2filters()
    llm = get_llm()
    
    # NL to JSON Conversion
    prompt_text = nl2filters.build_user_prompt(state["query"])
    raw_json = llm.invoke(prompt_text).content
    print(f"    DEBUG [JSON Filter]: {raw_json}")
    
    # Execute Search
    filt = nl2filters.interpret_nl_to_filter(raw_json)
    vectorstore = Chroma(persist_directory=get_cfg()["chroma_dir"], embedding_function=get_embeddings())
    
    # Search for candidates
    docs = vectorstore.similarity_search(state["query"], k=6)
    res = [{"name": d.metadata.get("name"), "artists": d.metadata.get("artists")} for d in docs]
    
    return {"results": res, "answer": f"I found {len(res)} tracks matching your request."}

def info_node(state: RouterState) -> RouterState:
    """Handles factual queries about the music database."""
    print(f"\n>>> [NODE: INFO] Executing RAG search for facts...")
    return {"answer": "This module would perform a RAG search based on your Exercise-4 logic."}

def debug_node(state: RouterState) -> RouterState:
    """Returns diagnostic information about the system."""
    print(f"\n>>> [NODE: DEBUG] System diagnostics...")
    return {"answer": f"Database Path: {get_cfg()['chroma_dir']}"}

def fallback_node(state: RouterState) -> RouterState:
    """Handles ambiguous or unsupported queries."""
    print(f"\n>>> [NODE: FALLBACK] Unclear intent.")
    return {"answer": "I am not sure how to help with that. Please ask about music styles or song facts."}

# --- Graph Construction ---
def build_graph():
    """Builds and compiles the LangGraph state machine."""
    builder = StateGraph(RouterState)
    builder.add_node("classify", classify_node)
    builder.add_node("playlist", playlist_node)
    builder.add_node("info", info_node)
    builder.add_node("debug", debug_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", lambda s: s["intent"])
    
    for n in ["playlist", "info", "debug", "fallback"]:
        builder.add_edge(n, END)
        
    return builder.compile()

if __name__ == "__main__":
    app = build_graph()
    print("\n[SYSTEM] LangGraph Router is active.")
    
    while True:
        q = input("\nQ> ").strip()
        if not q or q.lower() in ["exit", "quit"]: break
        try:
            out = app.invoke({"query": q})
            print(f"\nANSWER: {out.get('answer')}")
            if out.get('results'):
                print("TRACKS FOUND:")
                for r in out['results']: 
                    print(f"  - {r['name']} by {r['artists']}")
        except Exception as e:
            print(f"\n[ERROR]: {e}")