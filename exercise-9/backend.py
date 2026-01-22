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
from langchain_core.output_parsers import StrOutputParser
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
        "memory_db": "memory_ex9_heavyduty_final.db"
    }

def get_llm(temp=0):
    cfg = get_cfg()
    return AzureChatOpenAI(azure_deployment=cfg["deployment"], api_version=cfg["api_version"], 
                          azure_endpoint=cfg["endpoint"], api_key=cfg["api_key"], temperature=temp)

def get_embeddings():
    cfg = get_cfg()
    return AzureOpenAIEmbeddings(azure_deployment=cfg["emb_deployment"], api_version=cfg["api_version"], 
                                azure_endpoint=cfg["endpoint"], api_key=cfg["api_key"])

# --- 2. STATE DEFINITION ---
class AgentState(TypedDict):
    query: str
    plan: Dict[str, Any]           # Structured plan from Orchestrator
    results: List[Dict[str, Any]]  # Final results for the UI
    answer: str                    # Text explanation
    messages: Annotated[List[BaseMessage], operator.add] # Chat History
    logs: Annotated[List[str], operator.add] # Debug trail

# --- 3. NODES ---

def orchestrator_node(state: AgentState) -> Dict:
    """
    THE BRAIN: 
    1. Analyzes User Intent (Playlist vs Info).
    2. Performs Cumulative Math (for playlists).
    3. Categorizes Search Terms (Artist vs Vibe).
    """
    log_entry = "🧠 [ORCHESTRATOR] Analyzing request..."
    llm = get_llm()
    
    # FIX: Double curly braces {{ }} for literal JSON examples
    system_prompt = """You are a Music Assistant / Orchestrator.
    
    YOUR GOAL:
    Create a precise JSON execution plan based on the user's request and chat history.
    
    INSTRUCTIONS:
    1. **Intent Detection**: 
       - If the user wants to create/edit a list, intent is "playlist".
       - If the user asks a factual question (fastest, longest, who is), intent is "info".
    
    2. **Playlist Math (Cumulative)**:
       - Review the history. If the user says "Add 2 Prodigy", and they previously had "2 Metallica", 
         the plan must include BOTH.
       - Output the TOTAL target count for each item.
    
    3. **Categorization**:
       - Tag every search term as "artist" (specific name) or "vibe" (genre/mood/description).
       - Examples: 
         - "Metallica" -> "artist"
         - "Chill songs" -> "vibe"
         - "Fast rock" -> "vibe"
         - "Michael Jackson" -> "artist"
    
    4. **Sorting (For Info)**:
       - If the user asks for "fastest", set "sort": "fastest".
       - If "longest", set "sort": "longest".
    
    OUTPUT JSON FORMAT:
    {{
      "intent": "playlist" | "info",
      "actions": [
        {{
          "term": "Search Term",
          "count": "Integer (Total)",
          "category": "artist" | "vibe",
          "sort": "fastest" | "longest" | null
        }}
      ]
    }}
    
    EXAMPLE (Playlist):
    User: "2 Metallica and 3 chill songs"
    Output: {{ "intent": "playlist", "actions": [ {{ "term": "Metallica", "count": 2, "category": "artist" }}, {{ "term": "Relaxing Chill Music", "count": 3, "category": "vibe" }} ] }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{q}")
    ])
    
    raw = (prompt | llm | StrOutputParser()).invoke({"q": state["query"], "history": state["messages"]})
    
    try:
        # Robust JSON Parser: Handles markdown blocks or raw text
        match = re.search(r"\{.*\}", raw.replace("```json", "").replace("```", ""), re.DOTALL)
        if match:
            plan = json.loads(match.group(0))
            
            # Extract and Log Decision Explicitly
            intent_decision = plan.get("intent", "UNKNOWN").upper()
            log_detail = f"   👉 Decision: {intent_decision} MODULE\n   ✅ Plan Created: {json.dumps(plan)}"
            
        else:
            raise ValueError("No JSON found in LLM response")
    except Exception as e:
        # Emergency Fallback: Treat as a generic info search
        plan = {"intent": "info", "actions": [{"term": state["query"], "category": "vibe", "count": 5}]}
        log_detail = f"   ❌ Planning Failed: {e}. Using fallback to INFO."
        
    return {"plan": plan, "logs": [log_entry, log_detail]}

def executor_node(state: AgentState) -> Dict:
    """
    THE TOOL:
    Executes the 'Waterfall Search Strategy'.
    Stage 1: Exact Metadata Match (High Precision)
    Stage 2: Strict Vector Match (Medium Precision)
    Stage 3: Semantic Vibe Match (Low Precision, High Recall)
    """
    plan = state["plan"]
    intent = plan.get("intent", "info")
    actions = plan.get("actions", [])
    
    logs = [f"💾 [EXECUTOR] Running {len(actions)} actions..."]
    vectorstore = Chroma(persist_directory=get_cfg()["chroma_dir"], embedding_function=get_embeddings())
    
    all_results = []
    global_seen = set()
    
    for action in actions:
        term = action.get("term", "Music")
        category = action.get("category", "vibe")
        sort_mode = action.get("sort", None)
        
        # Info intent needs more candidates for sorting; Playlist needs specific count
        target_count = action.get("count", 5) if intent == "playlist" else 50
        
        logs.append(f"   🔍 Processing: '{term}' [{category.upper()}] Target: {target_count}")
        
        # --- WATERFALL STAGE 1: EXACT METADATA FILTER ---
        # (Only applicable if category is 'artist')
        stage1_docs = []
        if category == "artist":
            try:
                # Direct lookup: Extremely fast and accurate for exact names
                meta_response = vectorstore.get(where={"artists": term})
                if meta_response and meta_response['ids']:
                    for i in range(len(meta_response['ids'])):
                        d = meta_response['metadatas'][i]
                        d['match_type'] = "Stage 1 (Exact Metadata)"
                        stage1_docs.append(d)
                logs.append(f"      -> Stage 1 (Metadata): Found {len(stage1_docs)} exact matches.")
            except:
                pass # Fallback if metadata lookup fails
        
        # --- WATERFALL STAGE 2: VECTOR SEARCH + STRICT FILTER ---
        # (Used if Stage 1 didn't fill the quota, or for generic terms)
        stage2_docs = []
        # We assume stage 1 docs are effectively 'Document' objects for sorting logic later
        # But here we need to run vector search to find things *like* the term
        
        # Fetch deep candidates (x5 target) to ensure we have enough to filter
        candidate_docs = vectorstore.similarity_search(term, k=max(target_count * 5, 50))
        
        for d in candidate_docs:
            artist_name = d.metadata.get("artists", "").lower()
            term_lower = term.lower()
            
            # Check strict containment (e.g. "Metallica" in "Metallica")
            if term_lower in artist_name:
                d.metadata['match_type'] = "Stage 2 (Strict Vector)"
                stage2_docs.append(d.metadata)
        
        # --- WATERFALL STAGE 3: SEMANTIC VIBE MATCH ---
        # (Only used if category is 'vibe' OR if strict matches failed)
        stage3_docs = []
        if category == "vibe":
            for d in candidate_docs:
                # Apply Semantic Logic:
                # If searching for "Slow", exclude "Fast" songs via tempo check
                bpm = d.metadata.get("tempo", 120)
                if "slow" in term.lower() and bpm > 110: continue
                if "fast" in term.lower() and bpm < 130: continue
                
                d.metadata['match_type'] = "Stage 3 (Semantic Vibe)"
                stage3_docs.append(d.metadata)

        # --- SELECTION LOGIC ---
        selected_metas = []
        
        # Priority 1: Exact Metadata (Stage 1)
        selected_metas.extend(stage1_docs)
        
        # Priority 2: Strict Vector (Stage 2) - Deduping against Stage 1
        for m in stage2_docs:
            if len(selected_metas) >= target_count and intent == "playlist": break
            
            # Simple dedup check
            is_dup = any(x['name'] == m['name'] and x['artists'] == m['artists'] for x in selected_metas)
            if not is_dup:
                selected_metas.append(m)

        # Priority 3: Vibe Match (Stage 3) - Only if we still need songs AND category is 'vibe'
        # (Or if strict failed completely for an artist search, though rare)
        if len(selected_metas) < target_count and category == "vibe":
            remaining = target_count - len(selected_metas)
            logs.append(f"      -> Filling {remaining} slots with Semantic/Vibe matches.")
            count_added = 0
            for m in stage3_docs:
                if count_added >= remaining: break
                is_dup = any(x['name'] == m['name'] and x['artists'] == m['artists'] for x in selected_metas)
                if not is_dup:
                    selected_metas.append(m)
                    count_added += 1

        # --- SORTING (For Info Intent) ---
        if intent == "info" and sort_mode:
            logs.append(f"      -> Sorting results by: {sort_mode}")
            if "fast" in sort_mode or "tempo" in sort_mode:
                selected_metas.sort(key=lambda x: x.get('tempo', 0), reverse=True)
                # Decorate for UI
                for m in selected_metas:
                    m['debug_info'] = f"BPM: {round(m.get('tempo', 0))}"
            elif "long" in sort_mode:
                selected_metas.sort(key=lambda x: x.get('duration_ms', 0), reverse=True)
                for m in selected_metas:
                    mins = int(m.get('duration_ms', 0) / 60000)
                    secs = int((m.get('duration_ms', 0) % 60000) / 1000)
                    m['debug_info'] = f"Duration: {mins}:{secs:02d}"

        # --- FINAL COMMIT ---
        limit = target_count if intent == "playlist" else 5
        count_committed = 0
        
        for m in selected_metas[:limit]:
            key = f"{m.get('name')}-{m.get('artists')}"
            if key not in global_seen:
                m["source"] = "DB"
                all_results.append(m)
                global_seen.add(key)
                count_committed += 1
        
        logs.append(f"      -> Final Committed: {count_committed}/{target_count}")

    # --- LLM FALLBACK (Ghost Entries) ---
    # Only for playlists: If we promised X songs but found < X, generate placeholders
    if intent == "playlist":
        for action in actions:
            term = action.get("term")
            req_count = action.get("count", 1)
            
            # Count how many we actually provided for this term
            # (Heuristic: checks if artist name contains term OR match_type exists (vibe))
            found_actual = 0
            for r in all_results:
                if term.lower() in r.get("artists", "").lower():
                    found_actual += 1
                elif r.get("match_type") == "Stage 3 (Semantic Vibe)":
                     # This logic assumes sequential processing order roughly holds, 
                     # but for simplicity we assume Vibe matches count towards the Vibe target
                     found_actual += 1
            
            # Simple total check is safer to prevent over-generation
            # We compare total results vs total requested
            pass 
        
        # Total padding logic
        total_requested = sum(a.get("count", 1) for a in actions)
        if len(all_results) < total_requested:
            missing = total_requested - len(all_results)
            logs.append(f"      -> ⚠️ List short by {missing}. Generating LLM suggestions.")
            for i in range(missing):
                all_results.append({"name": f"Suggested Track {i+1}", "artists": "Unknown", "source": "LLM"})

    return {"results": all_results, "logs": logs}

def synthesizer_node(state: AgentState) -> Dict:
    """
    THE VOICE:
    Generates the final response based on the executed results.
    """
    logs = ["🗣️ [SYNTHESIZER] Generating response..."]
    llm = get_llm()
    results = state.get("results", [])
    plan = state["plan"]
    
    if not results and plan.get("intent") == "info":
        sys_msg = "The database search yielded NO results. Answer from your general knowledge. Label the source as [LLM]."
        context = "DB Results: None."
    else:
        # --- NEW FEATURE: Explanations and Detail ---
        sys_msg = """You are a Music Assistant.
        1. Explain the solution based on the provided DB Results.
        2. Label sources as [DB] if present.
        3. IF CREATING A PLAYLIST:
           - List every track found.
           - Show key details provided in the DB results (e.g., Artist, BPM, Genre, Duration).
           - CRITICAL: Add a short (4-5 words) explanation for EACH song on why it fits the request (e.g., 'Matches the chill vibe' or 'High tempo for running')."""
        
        context = json.dumps(results, indent=2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("human", "User Query: {q}\n\nExecution Results:\n{c}")
    ])
    
    response = (prompt | llm | StrOutputParser()).invoke({"q": state["query"], "c": context})
    return {"answer": response, "messages": [AIMessage(content=response)], "logs": logs}

# --- 4. GRAPH ---
def build_graph(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("executor", executor_node)
    builder.add_node("synthesizer", synthesizer_node)
    
    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", "executor")
    builder.add_edge("executor", "synthesizer")
    builder.add_edge("synthesizer", END)
    
    return builder.compile(checkpointer=checkpointer)