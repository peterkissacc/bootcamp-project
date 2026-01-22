import gradio as gr
import sqlite3
import uuid
from langchain_core.messages import HumanMessage

# Import from backend
from backend import build_graph, get_cfg, SqliteSaver

# --- SETUP ---
THREAD_ID = str(uuid.uuid4())
conn = sqlite3.connect(get_cfg()["memory_db"], check_same_thread=False)
memory = SqliteSaver(conn)
app = build_graph(memory)
config = {"configurable": {"thread_id": THREAD_ID}}

print(f"[SYSTEM] UI Loaded. Session ID: {THREAD_ID}")

# --- CHAT LOGIC ---
def chat_logic(message, history):
    if not message:
        return ""
    
    try:
        # Run Graph
        response = app.invoke(
            {"query": message, "messages": [HumanMessage(content=message)]}, 
            config=config
        )
        
        ai_text = response.get("answer", "No answer generated.")
        
        # --- 1. DEBUG LOGS DISPLAY ---
        logs = response.get("logs", [])
        debug_section = ""
        if logs:
            debug_section += "\n\n<details open><summary>🛠️ <b>System Logic (Debug Trace)</b></summary>\n\n"
            debug_section += "```text\n"
            for log in logs:
                debug_section += f"{log}\n"
            debug_section += "```\n</details>"

        # --- 2. VERIFIED SOURCES DISPLAY ---
        results = response.get("results", [])
        source_section = ""
        if results:
            source_section += "\n\n<details open><summary>🎵 <b>Verified Sources (Deep Scan)</b></summary>\n\n"
            source_section += "| Source | Artist | Track | Details |\n"
            source_section += "| :--- | :--- | :--- | :--- |\n"
            
            for r in results:
                src_label = r.get("source", "LLM")
                icon = "💾" if src_label == "DB" else "🤖"
                name = r.get("name", "Unknown")
                artist = r.get("artists", "Unknown")
                # Show extra debug info (BPM/Duration) if available
                extra = r.get("debug_info", "-")
                
                source_section += f"| {icon} {src_label} | {artist} | {name} | {extra} |\n"
            
            source_section += "</details>"
        
        # Combine All
        final_output = ai_text + debug_section + source_section
        return final_output

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"⚠️ **Critical Error:** {str(e)}"

# --- UI ---
ui = gr.ChatInterface(
    fn=chat_logic,
    title="🎵 AI Music Curator (Debug Mode)",
    description="Full transparency demo. See exactly how the AI thinks, filters, and sorts.",
    examples=[
        "Create a list with 2 Metallica and 1 Prodigy song",
        "Add 2 more Prodigy songs",
        "What is the fastest Michael Jackson song?",
    ]
)

if __name__ == "__main__":
    ui.launch(inbrowser=True)