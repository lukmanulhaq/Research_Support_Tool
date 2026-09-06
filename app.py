"""
LitSynth AI — Main Streamlit Application
Fully self-contained, deployable to Streamlit Community Cloud.

Architecture:
  - 5 Agents: Orchestrator, Router, Retriever, Synthesizer, Critic
  - 3 Patterns: Orchestrator-Worker, ReAct/Tool-Use, Reflection/Self-Critique
  - RAG: FAISS + all-MiniLM-L6-v2 + 22-paper JSON corpus
  - A2A: Structured JSON messages via st.session_state
  - UI:  Custom Streamlit component (frontend/index.html)
"""
import json
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────
# Page Config — Must be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LitSynth AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Hide Streamlit chrome for full-screen custom UI
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    iframe {
        border: none !important;
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Secrets Validation
# ─────────────────────────────────────────────────────────────
if "GROQ_API_KEY" not in st.secrets or "OPENROUTER_API_KEY" not in st.secrets:
    st.error(
        "⚠️ API keys missing. Please configure `.streamlit/secrets.toml`:\n"
        "```toml\nGROQ_API_KEY = \"gsk_...\"\nOPENROUTER_API_KEY = \"sk-or-...\"\n```"
    )
    st.stop()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# ─────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────
if "faiss_store" not in st.session_state:
    from rag import faiss_store
    with st.spinner("🔧 Building FAISS index from 22-paper corpus..."):
        st.session_state["faiss_store"] = faiss_store.build_index()

if "a2a_trace" not in st.session_state:
    st.session_state["a2a_trace"] = []
if "scratchpad" not in st.session_state:
    st.session_state["scratchpad"] = []
if "last_query_ts" not in st.session_state:
    st.session_state["last_query_ts"] = 0
if "last_response" not in st.session_state:
    st.session_state["last_response"] = None

# ─────────────────────────────────────────────────────────────
# Declare Custom Component
# ─────────────────────────────────────────────────────────────
litsynth_ui = components.declare_component("litsynth_ui", path="frontend")

# ─────────────────────────────────────────────────────────────
# Render Component — pass last response if available
# ─────────────────────────────────────────────────────────────
component_value = litsynth_ui(
    response=st.session_state["last_response"],
    key="litsynth_main",
)

# ─────────────────────────────────────────────────────────────
# Pipeline Execution — triggered by component sending a query
# ─────────────────────────────────────────────────────────────
if component_value and isinstance(component_value, dict) and component_value.get("query"):
    query = component_value["query"]
    ts = component_value.get("ts", 0)

    if ts != st.session_state["last_query_ts"]:
        st.session_state["last_query_ts"] = ts

        # Reset trace and scratchpad for new query
        scratchpad = []
        a2a_trace = []

        from agents import orchestrator, router, retriever, synthesizer, critic

        try:
            # ── Step 1: Orchestrator ──────────────────────────
            task_plan = orchestrator.run(
                query=query,
                groq_api_key=GROQ_API_KEY,
                scratchpad=scratchpad,
                a2a_trace=a2a_trace,
            )

            # ── Step 2: Router ────────────────────────────────
            routing_result = router.run(
                query=query,
                task_plan=task_plan,
                groq_api_key=GROQ_API_KEY,
                scratchpad=scratchpad,
                a2a_trace=a2a_trace,
            )

            intent = routing_result.get("intent", "RAG_REQUIRED")

            # ── Step 3: Retriever (if RAG needed) ────────────
            chunks = []
            if intent == "RAG_REQUIRED":
                chunks = retriever.run(
                    query=query,
                    task_plan=task_plan,
                    routing_result=routing_result,
                    faiss_store_state=st.session_state["faiss_store"],
                    scratchpad=scratchpad,
                    a2a_trace=a2a_trace,
                    k=5,
                )
            else:
                scratchpad.append({
                    "agent": "Retriever",
                    "thought": "Skipped — Router classified as CONVERSATIONAL.",
                    "color": "cyan"
                })

            # ── Step 4: Synthesizer ───────────────────────────
            synthesis_text = synthesizer.run(
                query=query,
                chunks=chunks,
                task_plan=task_plan,
                openrouter_api_key=OPENROUTER_API_KEY,
                scratchpad=scratchpad,
                a2a_trace=a2a_trace,
            )

            # ── Step 5: Critic ────────────────────────────────
            critique = critic.run(
                query=query,
                synthesis=synthesis_text,
                chunks=chunks,
                openrouter_api_key=OPENROUTER_API_KEY,
                scratchpad=scratchpad,
                a2a_trace=a2a_trace,
            )

            # Use revised synthesis if critic score was low
            final_text = critique.get("revised_synthesis") or synthesis_text
            critic_score = critique.get("score", 7.5)
            verdict = critique.get("verdict", "APPROVED")

            # Build sources list for the UI
            sources = [
                {
                    "id": c["id"],
                    "title": c["title"],
                    "year": c["year"],
                    "score": round(c["score"], 4),
                }
                for c in chunks
            ]

            # Persist state
            st.session_state["a2a_trace"] = a2a_trace
            st.session_state["scratchpad"] = scratchpad

            # Build rich response for the frontend component
            st.session_state["last_response"] = {
                "response": final_text,
                "critic_score": critic_score,
                "verdict": verdict,
                "hallucinations": critique.get("hallucinations_detected", []),
                "quality_notes": critique.get("quality_notes", ""),
                "sources": sources,
                "agent_trace": a2a_trace,
                "scratchpad": scratchpad,
                "intent": intent,
                "task_plan": task_plan,
            }

        except Exception as e:
            import traceback
            err_msg = f"Pipeline error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            st.session_state["last_response"] = {
                "response": err_msg,
                "critic_score": 0,
                "verdict": "ERROR",
                "hallucinations": [],
                "quality_notes": "An error occurred during pipeline execution.",
                "sources": [],
                "agent_trace": a2a_trace,
                "scratchpad": scratchpad,
                "intent": "ERROR",
                "task_plan": {},
            }

        # Rerun so the component receives the new response
        st.rerun()
