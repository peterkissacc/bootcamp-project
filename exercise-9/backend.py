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
    3. Categorizes Search Terms (Artist vs Song vs Vibe vs Genre).
    4. Translates Activities to Vibes.
    """
    log_entry = "🧠 [ORCHESTRATOR] Analyzing request..."
    llm = get_llm()
    
    # UPDATED: REWROTE PROMPT TO ENFORCE 'DELTA ONLY' LOGIC & GENRE PRESERVATION
    system_prompt = """You are a Music Assistant / Orchestrator.
    
    YOUR GOAL:
    Create a precise JSON execution plan based on the user's request and chat history.
    
    INSTRUCTIONS:
    1. **Intent Detection**: 
       - If the user wants to create/edit a list, intent is "playlist".
       - If the user asks a factual question (fastest, longest, who is), intent is "info".
    
    2. **Playlist Edits (Additions) -- CRITICAL RULE**:
       - If the user says "add 1 more", "add 2 more", etc.:
         a) **DO NOT** re-list the songs/artists already in the playlist.
         b) **DO NOT** guess specific new artists (like "Led Zeppelin") unless the user named them.
         c) **REUSE** the *Broadest* search term from history (e.g., "Rock", "Fast songs").
         d) Set "count" to the number of **NEW** items requested (e.g., 1 or 2).
       - Example: User "Add 1 more" -> Action: Term="Rock", Category="genre", Count=1.
    
    3. **Term Translation (Activity -> Vibe) & Genre Preservation**:
       - If the user mentions an **ACTIVITY** (e.g., "Running", "Workout", "Sleeping", "Studying"), do NOT search for that word literally.
       - **TRANSLATE** it into a musical VIBE or GENRE description.
       - **CRITICAL**: If a specific **GENRE** is also mentioned (e.g., "Rock", "Pop", "Metal"), you **MUST INCLUDE IT** in the translation.
       - Bad Example: "Rock music for workout" -> term: "High Tempo Energetic" (MISSING GENRE).
       - Good Example: "Rock music for workout" -> term: "High Tempo Rock", category: "vibe".
    
    4. **Categorization**:
       - Tag every search term as "artist", "song", "genre", or "vibe".
       - **Artist**: Specific name of a person or band (e.g., "Metallica").
       - **Song**: Specific title of a track.
       - **Genre**: A specific musical style (e.g., "Techno", "Rock").
       - **Vibe**: Adjectives, moods, or descriptions (e.g., "Chill", "Fast").
    
    5. **Sorting (For Info)**:
       - If the user asks for "fastest", set "sort": "fastest".
       - If "longest", set "sort": "longest".
    
    OUTPUT JSON FORMAT:
    {{
      "intent": "playlist" | "info",
      "actions": [
        {{
          "term": "Search Term",
          "count": "Integer (Items to Add)",
          "category": "artist" | "song" | "genre" | "vibe",
          "sort": "fastest" | "longest" | null
        }}
      ]
    }}
    
    EXAMPLE 1 (New List):
    User: "2 Metallica and 3 chill songs"
    Output: {{ "intent": "playlist", "actions": [ {{ "term": "Metallica", "count": 2, "category": "artist" }}, {{ "term": "Relaxing Chill Music", "count": 3, "category": "vibe" }} ] }}

    EXAMPLE 2 (Add More - Contextual):
    History: [User created a Rock playlist]
    User: "Add 1 more"
    Output: {{ "intent": "playlist", "actions": [ {{ "term": "Rock", "count": 1, "category": "genre" }} ] }}
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
        # Emergency Fallback: Intelligent heuristic
        # If the LLM fails, we perform SMART RECOVERY based on history
        q_lower = state["query"].lower()
        
        # 1. Intent Guess
        fallback_intent = "info"
        if any(w in q_lower for w in ["playlist", "list", "create", "add", "make"]):
            fallback_intent = "playlist"
            
        # 2. Count Guess
        fallback_count = 5 
        count_match = re.search(r"\b\d+\b", state["query"])
        if count_match:
            fallback_count = int(count_match.group(0))
        elif "add" in q_lower:
            fallback_count = 2 # Default for "add more"
            
        # 3. Context Recovery (FIX for "Add 2 more")
        fallback_term = state["query"]
        fallback_category = "vibe"
        
        # If query is very short/generic (e.g. "add 2 more"), look back in history
        is_generic_request = len(state["query"].split()) < 5 and ("add" in q_lower or "more" in q_lower)
        
        if is_generic_request:
            # Iterate backwards through history to find the last Human Message that wasn't this one
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage) and msg.content != state["query"]:
                    fallback_term = msg.content # Use the previous user message as the search term
                    log_entry += f"\n   🔄 Context Recovery: Interpreted '{state['query']}' as '{fallback_term}'"
                    break

        plan = {
            "intent": fallback_intent, 
            "actions": [{
                "term": fallback_term, 
                "category": fallback_category, 
                "count": fallback_count
            }]
        }
        log_detail = f"   ❌ Planning Failed: {e}. Using intelligent fallback: Searching '{fallback_term}' (Count: {fallback_count})."
        
    return {"plan": plan, "logs": [log_entry, log_detail]}

def executor_node(state: AgentState) -> Dict:
    """
    THE TOOL:
    Executes the 'Waterfall Search Strategy'.
    Stage 1: Exact Metadata Match (High Precision) - Now supports Artist, Genre AND Song (Name)
    Stage 2: Strict Vector Match (Medium Precision)
    Stage 3: Semantic Vibe Match (Low Precision, High Recall)
    """
    plan = state["plan"]
    intent = plan.get("intent", "info")
    actions = plan.get("actions", [])
    
    logs = [f"💾 [EXECUTOR] Running {len(actions)} actions..."]
    vectorstore = Chroma(persist_directory=get_cfg()["chroma_dir"], embedding_function=get_embeddings())
    
    # --- 1. CONTEXT PRESERVATION (Accumulation Logic) ---
    # UPDATED: Start with existing results if intent is playlist, so we don't lose previous songs.
    if intent == "playlist":
        previous_results = state.get("results", []) or []
        if previous_results:
            logs.append(f"   🔄 Context: Keeping {len(previous_results)} existing tracks in the list.")
    else:
        previous_results = []

    new_results_to_add = [] 
    global_seen = set()
    
    # Mark existing items as 'seen' to prevent duplicates
    for r in previous_results:
        key = f"{r.get('name', r.get('title'))}-{r.get('artists')}"
        global_seen.add(key)
    
    # --- 2. EXECUTION LOOP ---
    for action in actions:
        term = action.get("term", "Music")
        category = action.get("category", "vibe")
        sort_mode = action.get("sort", None)
        
        target_count = action.get("count", 5) if intent == "playlist" else 50
        
        # --- FIXED: SMART LIMIT CALCULATION ---
        # Determine if 'target_count' is a Total Goal (e.g., 7) or just New Items (e.g., 2) based on history.
        limit = target_count 
        
        if intent == "playlist":
            current_total = len(previous_results)
            # If requested total is significantly larger than current, assume it's a Total Goal.
            if target_count > current_total:
                limit = target_count - current_total
            # If requested total is small (<= current), assume it's a Delta (e.g., "Add 2").
            else:
                limit = target_count
        else:
            limit = 50 # Info intent gets plenty
            
        logs.append(f"   🔍 Processing: '{term}' [{category.upper()}] Adding: {limit}")
        
        # --- WATERFALL STAGE 1: EXACT METADATA FILTER ---
        # (Applicable if category is 'artist', 'genre' OR 'song')
        stage1_docs = []
        
        if category == "artist":
            try:
                meta_response = vectorstore.get(where={"artists": term})
                if meta_response and meta_response['ids']:
                    for i in range(len(meta_response['ids'])):
                        d = meta_response['metadatas'][i]
                        d['match_type'] = "Stage 1 (Exact Metadata - Artist)"
                        stage1_docs.append(d)
                logs.append(f"      -> Stage 1 (Metadata): Found {len(stage1_docs)} exact matches.")
            except: pass 

        elif category == "genre":
            try:
                meta_response = vectorstore.get(where={"genre": term})
                if meta_response and meta_response['ids']:
                    for i in range(len(meta_response['ids'])):
                        d = meta_response['metadatas'][i]
                        d['match_type'] = "Stage 1 (Exact Metadata - Genre)"
                        stage1_docs.append(d)
                logs.append(f"      -> Stage 1 (Metadata): Found {len(stage1_docs)} exact matches.")
            except: pass

        elif category == "song": 
            # UPDATED: Added logic for Song Title Lookup (Trying 'name' then 'title')
            try:
                # First try "name"
                meta_response = vectorstore.get(where={"name": term})
                if meta_response and meta_response['ids']:
                    for i in range(len(meta_response['ids'])):
                        d = meta_response['metadatas'][i]
                        d['match_type'] = "Stage 1 (Exact Metadata - Song Name)"
                        stage1_docs.append(d)
                
                # If nothing found, try "title" (fallback for schema diffs)
                if not stage1_docs:
                    meta_response = vectorstore.get(where={"title": term})
                    if meta_response and meta_response['ids']:
                        for i in range(len(meta_response['ids'])):
                            d = meta_response['metadatas'][i]
                            d['match_type'] = "Stage 1 (Exact Metadata - Song Title)"
                            stage1_docs.append(d)

                logs.append(f"      -> Stage 1 (Metadata): Found {len(stage1_docs)} exact matches.")
            except: pass

        # --- WATERFALL STAGE 2: VECTOR SEARCH + STRICT FILTER ---
        stage2_docs = []
        
        # Fetch deep candidates (x5 target) to ensure we have enough to filter
        candidate_docs = vectorstore.similarity_search(term, k=max(limit * 5, 50))
        
        # UPDATED: Debug Log for DB Schema
        if candidate_docs:
             keys = list(candidate_docs[0].metadata.keys())
             logs.append(f"      ℹ️ [DB SCHEMA DEBUG] Available fields: {keys}")

        for d in candidate_docs:
            term_lower = term.lower()
            field_to_check = ""
            
            if category == "artist":
                field_to_check = d.metadata.get("artists", "").lower()
            elif category == "genre":
                field_to_check = d.metadata.get("genre", "").lower()
            elif category == "song":
                # Check both common keys for song titles
                name_val = d.metadata.get("name", "")
                title_val = d.metadata.get("title", "")
                field_to_check = (name_val + " " + title_val).lower()

            # Check strict containment 
            if field_to_check and term_lower in field_to_check:
                d.metadata['match_type'] = f"Stage 2 (Strict Vector - {category.capitalize()})"
                stage2_docs.append(d.metadata)
        
        # --- WATERFALL STAGE 3: SEMANTIC VIBE MATCH ---
        stage3_docs = []
        
        # We allow Stage 3 for Vibe, OR if we need backfill for Others
        if category == "vibe" or (len(stage1_docs) + len(stage2_docs) < limit):
            for d in candidate_docs:
                bpm = d.metadata.get("tempo", 120)
                if "slow" in term.lower() and bpm > 110: continue
                if "fast" in term.lower() and bpm < 130: continue
                
                # If we are here for a Genre/Artist search that failed strict matching, 
                # we tag it as Semantic fallback.
                d.metadata['match_type'] = "Stage 3 (Semantic Vibe/Fallback)"
                stage3_docs.append(d.metadata)

        # --- SELECTION LOGIC (UPDATED FOR DUPLICATE SKIPPING) ---
        selected_metas = []
        
        # Priority 1: Stage 1 Docs (Metadata Exact)
        for m in stage1_docs:
            if len(selected_metas) >= limit and intent == "playlist": break
            
            # CRITICAL FIX: Check History Before Adding from Stage 1
            t_name = m.get('name', m.get('title', 'Unknown'))
            t_artist = m.get('artists', 'Unknown')
            g_key = f"{t_name}-{t_artist}"
            
            if g_key in global_seen: continue # Skip if already in list
            
            is_dup = any(x.get('name', x.get('title')) == m.get('name', m.get('title')) and x.get('artists') == m.get('artists') for x in selected_metas)
            if not is_dup: selected_metas.append(m)
        
        # Priority 2: Stage 2 Docs (Strict Vector)
        for m in stage2_docs: 
            if len(selected_metas) >= limit and intent == "playlist": break
            
            # CRITICAL FIX: Check History Before Adding from Stage 2
            t_name = m.get('name', m.get('title', 'Unknown'))
            t_artist = m.get('artists', 'Unknown')
            g_key = f"{t_name}-{t_artist}"
            
            if g_key in global_seen: continue # Skip if already in list

            is_dup = any(x.get('name', x.get('title')) == m.get('name', m.get('title')) and x.get('artists') == m.get('artists') for x in selected_metas)
            if not is_dup: selected_metas.append(m)

        # Priority 3: Stage 3 Docs (Semantic Fallback)
        if len(selected_metas) < limit: 
            remaining = limit - len(selected_metas)
            
            if category == "vibe" or len(selected_metas) == 0: 
                logs.append(f"      -> Filling {remaining} slots with Semantic/Vibe matches.")
                count_added = 0
                for m in stage3_docs:
                    if count_added >= remaining: break
                    
                    # CRITICAL FIX: Check History Before Adding from Stage 3
                    t_name = m.get('name', m.get('title', 'Unknown'))
                    t_artist = m.get('artists', 'Unknown')
                    g_key = f"{t_name}-{t_artist}"
                    if g_key in global_seen: continue

                    is_dup = any(x.get('name', x.get('title')) == m.get('name', m.get('title')) and x.get('artists') == m.get('artists') for x in selected_metas)
                    if not is_dup:
                        selected_metas.append(m)
                        count_added += 1

        # --- SORTING (For Info Intent) ---
        if intent == "info" and sort_mode:
            logs.append(f"      -> Sorting results by: {sort_mode}")
            if "fast" in sort_mode or "tempo" in sort_mode:
                selected_metas.sort(key=lambda x: x.get('tempo', 0), reverse=True)
                for m in selected_metas: m['debug_info'] = f"BPM: {round(m.get('tempo', 0))}"
            elif "long" in sort_mode:
                selected_metas.sort(key=lambda x: x.get('duration_ms', 0), reverse=True)
                for m in selected_metas:
                    mins = int(m.get('duration_ms', 0) / 60000)
                    secs = int((m.get('duration_ms', 0) % 60000) / 1000)
                    m['debug_info'] = f"Duration: {mins}:{secs:02d}"

        # --- FINAL COMMIT FOR THIS ACTION ---
        count_committed_for_this_action = 0
        
        for m in selected_metas[:limit]:
            # Generate Unique Key
            track_name = m.get('name', m.get('title', 'Unknown'))
            key = f"{track_name}-{m.get('artists')}"
            
            # Add ONLY if not in Global Seen (Previous + Current)
            if key not in global_seen:
                m["source"] = "DB"
                new_results_to_add.append(m)
                global_seen.add(key)
                count_committed_for_this_action += 1
        
        logs.append(f"      -> Added: {count_committed_for_this_action}/{limit}")

    # --- 3. MERGE RESULTS ---
    # Combine Old + New
    final_results = previous_results + new_results_to_add
    
    # --- LLM FALLBACK (Ghost Entries) ---
    if intent == "playlist":
        # FIXED: Correctly calculate missing items based on Grand Total Goal vs Actual.
        total_requested_in_actions = sum(a.get("count", 0) for a in actions)
        
        # Calculate the Grand Total Goal
        # If the requested count was a Total (e.g. 7), goal is 7.
        # If requested count was a Delta (e.g. 2), goal is Previous + 2.
        if total_requested_in_actions > len(previous_results):
             grand_total_goal = total_requested_in_actions
        else:
             grand_total_goal = len(previous_results) + total_requested_in_actions

        current_total = len(final_results)
        
        if current_total < grand_total_goal:
            missing = grand_total_goal - current_total
            if missing > 0:
                logs.append(f"      -> ⚠️ List short by {missing}. Generating LLM suggestions.")
                for i in range(missing):
                    final_results.append({"name": f"Suggested Track {i+1}", "artists": "Unknown", "source": "LLM"})

    return {"results": final_results, "logs": logs}

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
        sys_msg = """You are a Music Assistant.
        1. Explain the solution based on the provided DB Results.
        2. Label sources as [DB] if present.
        3. IF CREATING A PLAYLIST:
            - List every track found.
            - Show key details provided in the DB results (e.g., Artist, BPM, Genre, Duration).
            - Include Album Name, Energy, and Popularity.
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