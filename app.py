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
import time
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
from langchain_openai import ChatOpenAI

# ─────────────────────────────────────────────────────────────
# Telemetry Monkey Patch & Helpers
# ─────────────────────────────────────────────────────────────
if not hasattr(ChatOpenAI, "_patched_for_telemetry"):
    original_invoke = ChatOpenAI.invoke
    def patched_invoke(self, *args, **kwargs):
        fallback_models = [
            "z-ai/glm-5.2:free",
            "google/gemma-4-31b-it:free",
            "minimax/minimax-m2.7:free",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        ]
        
        last_exception = None
        for attempt in range(len(fallback_models) + 1):
            try:
                current_model = self
                if attempt > 0:
                    current_model = ChatOpenAI(
                        model=fallback_models[attempt - 1],
                        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=getattr(self, "temperature", 0.3),
                        max_tokens=getattr(self, "max_tokens", 2048)
                    )
                
                res = original_invoke(current_model, *args, **kwargs)
                
                if "telemetry" not in st.session_state:
                    st.session_state["telemetry"] = {"tokens": 0, "cost": 0.0}
                
                tokens = 0
                if hasattr(res, "usage_metadata") and res.usage_metadata:
                    tokens = res.usage_metadata.get("total_tokens", 0)
                elif hasattr(res, "response_metadata") and "token_usage" in res.response_metadata:
                    tokens = res.response_metadata["token_usage"].get("total_tokens", 0)
                else:
                    tokens = len(str(res.content)) // 4
                    
                st.session_state["telemetry"]["tokens"] += tokens
                st.session_state["telemetry"]["cost"] += (tokens / 1000.0) * 0.0005
                return res
                
            except Exception as e:
                last_exception = e
                err_msg = str(e).lower()
                if any(x in err_msg for x in ["502", "503", "429", "404", "overloaded", "rate limit", "timeout", "service unavailable", "bad gateway", "not found", "unavailable for free"]):
                    if attempt < len(fallback_models):
                        import time
                        time.sleep(2)
                        continue
                raise last_exception
    
    ChatOpenAI.invoke = patched_invoke
    ChatOpenAI._patched_for_telemetry = True

def update_scratchpad_timestamps(sp):
    for entry in sp:
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().strftime("%I:%M:%S %p")
    return sp

def get_telemetry():
    latency = time.time() - st.session_state.get("start_time", time.time())
    tel = st.session_state.get("telemetry", {"tokens": 0, "cost": 0.0})
    return {
        "latency": round(latency, 1),
        "tokens": tel["tokens"],
        "cost": round(tel["cost"], 4)
    }

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
litsynth_ui = components.declare_component("litsynth_v5", path="frontend")

# ─────────────────────────────────────────────────────────────
# Render Component — pass last response if available
# ─────────────────────────────────────────────────────────────
component_value = litsynth_ui(
    response=st.session_state["last_response"],
    corpus=st.session_state.get("faiss_store", {}).get("corpus", []),
    key="litsynth_v5",
)

# ─────────────────────────────────────────────────────────────
# Pipeline Execution — State Machine
# ─────────────────────────────────────────────────────────────
if "pipeline_step" not in st.session_state:
    st.session_state["pipeline_step"] = 0

if component_value and isinstance(component_value, dict) and component_value.get("query"):
    query = component_value["query"]
    ts = component_value.get("ts", 0)

    # NEW QUERY TRIGGERS PIPELINE START
    if ts != st.session_state.get("last_query_ts", 0):
        st.session_state["last_query_ts"] = ts
        st.session_state["query"] = query
        st.session_state["pipeline_step"] = 1
        st.session_state["scratchpad"] = []
        st.session_state["a2a_trace"] = []
        st.session_state["task_plan"] = {}
        st.session_state["routing_result"] = {}
        st.session_state["chunks"] = []
        st.session_state["synthesis_text"] = ""
        st.session_state["intent"] = ""
        
        # Initial running state
        st.session_state["start_time"] = time.time()
        st.session_state["telemetry"] = {"tokens": 0, "cost": 0.0}
        st.session_state["last_response"] = {
            "status": "running",
            "step": 0,
            "scratchpad": [],
            "telemetry": get_telemetry(),
        }
        st.rerun()

# Execute current step if pipeline is active
step = st.session_state.get("pipeline_step", 0)

if step > 0:
    from agents import orchestrator, router, retriever, synthesizer, critic
    query = st.session_state["query"]
    scratchpad = st.session_state["scratchpad"]
    a2a_trace = st.session_state["a2a_trace"]

    try:
        if step == 1:
            # Step 1: Orchestrator
            task_plan = orchestrator.run(
                query=query,
                openrouter_api_key=OPENROUTER_API_KEY,
                scratchpad=scratchpad,
                a2a_trace=a2a_trace,
            )
            st.session_state["task_plan"] = task_plan
            st.session_state["pipeline_step"] = 2
            st.session_state["last_response"] = {"status": "running", "step": 1, "scratchpad": update_scratchpad_timestamps(scratchpad), "telemetry": get_telemetry()}
            st.rerun()

        elif step == 2:
            # Step 2: Router
            routing_result = router.run(
                query=query,
                task_plan=st.session_state["task_plan"],
                openrouter_api_key=OPENROUTER_API_KEY,
                scratchpad=scratchpad,
                a2a_trace=a2a_trace,
            )
            st.session_state["routing_result"] = routing_result
            st.session_state["intent"] = routing_result.get("intent", "RAG_REQUIRED")
            st.session_state["pipeline_step"] = 3
            st.session_state["last_response"] = {"status": "running", "step": 2, "scratchpad": update_scratchpad_timestamps(scratchpad), "telemetry": get_telemetry()}
            st.rerun()

        elif step == 3:
            # Step 3: Retriever
            intent = st.session_state["intent"]
            if intent == "RAG_REQUIRED":
                chunks = retriever.run(
                    query=query,
                    task_plan=st.session_state["task_plan"],
                    routing_result=st.session_state["routing_result"],
                    faiss_store_state=st.session_state["faiss_store"],
                    scratchpad=scratchpad,
                    a2a_trace=a2a_trace,
                    k=5,
                )
                st.session_state["chunks"] = chunks
            else:
                scratchpad.append({
                    "agent": "Retriever",
                    "thought": "Skipped — Router classified as CONVERSATIONAL.",
                    "color": "cyan"
                })
                st.session_state["chunks"] = []
                
            st.session_state["pipeline_step"] = 4
            st.session_state["last_response"] = {"status": "running", "step": 3, "scratchpad": update_scratchpad_timestamps(scratchpad), "telemetry": get_telemetry()}
            st.rerun()

        elif step == 4:
            # Step 4: Synthesizer
            synthesis_text = synthesizer.run(
                query=query,
                chunks=st.session_state["chunks"],
                task_plan=st.session_state["task_plan"],
                openrouter_api_key=OPENROUTER_API_KEY,
                scratchpad=scratchpad,
                a2a_trace=a2a_trace,
            )
            st.session_state["synthesis_text"] = synthesis_text
            st.session_state["pipeline_step"] = 5
            st.session_state["last_response"] = {"status": "running", "step": 4, "scratchpad": update_scratchpad_timestamps(scratchpad), "telemetry": get_telemetry()}
            st.rerun()

        elif step == 5:
            # Step 5: Critic
            chunks = st.session_state["chunks"]
            synthesis_text = st.session_state["synthesis_text"]
            
            critique = critic.run(
                query=query,
                synthesis=synthesis_text,
                chunks=chunks,
                openrouter_api_key=OPENROUTER_API_KEY,
                scratchpad=scratchpad,
                a2a_trace=a2a_trace,
            )

            # Finalizing Response
            final_text = critique.get("revised_synthesis") or synthesis_text
            critic_score = critique.get("score", 7.5)
            verdict = critique.get("verdict", "APPROVED")

            sources = [
                {
                    "id": c["id"],
                    "title": c["title"],
                    "year": c["year"],
                    "score": round(c["score"], 4),
                }
                for c in chunks
            ]

            st.session_state["pipeline_step"] = 0  # Execution complete
            st.session_state["last_response"] = {
                "status": "complete",
                "response": final_text,
                "critic_score": critic_score,
                "verdict": verdict,
                "hallucinations": critique.get("hallucinations_detected", []),
                "quality_notes": critique.get("quality_notes", ""),
                "sources": sources,
                "agent_trace": a2a_trace,
                "scratchpad": update_scratchpad_timestamps(scratchpad),
                "telemetry": get_telemetry(),
                "intent": st.session_state["intent"],
                "task_plan": st.session_state["task_plan"],
            }
            st.rerun()

    except Exception as e:
        import traceback
        error_str = str(e).lower()
        
        if "429" in error_str or "rate limit" in error_str:
            err_msg = "### ⚠️ API Rate Limit Reached\n\nOne of the free OpenRouter models used in the pipeline has reached its rate limit. The provider returned a `429 Too Many Requests` error.\n\n**Remedy:**\n- Wait a few moments and try your request again.\n- Free models are shared globally, so temporary congestion is common."
            quality_notes = "Rate limited upstream."
        else:
            err_msg = f"Pipeline error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            quality_notes = "An error occurred during pipeline execution."

        st.session_state["pipeline_step"] = 0
        st.session_state["last_response"] = {
            "status": "complete",
            "response": err_msg,
            "critic_score": 0,
            "verdict": "ERROR",
            "hallucinations": [],
            "quality_notes": quality_notes,
            "sources": [],
            "agent_trace": a2a_trace,
            "scratchpad": update_scratchpad_timestamps(scratchpad),
            "telemetry": get_telemetry(),
            "intent": "ERROR",
            "task_plan": {},
        }
        st.rerun()
