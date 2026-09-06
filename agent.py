import streamlit as st
from typing import Annotated, TypedDict, Literal
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from tools.rag_tool import extract_academic_evidence, initialize_rag

# Initialize the RAG corpus
initialize_rag()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    research_context: str
    needs_reflection: bool


# Deliberate Model Selection[cite: 1]
# 1. Groq for low-latency routing decisions[cite: 1]
router_model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0
)

# 2. OpenRouter (Claude 3.5 Sonnet) for deep reasoning and synthesis[cite: 1]
synthesis_model = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",
    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2
)


def router_node(state: AgentState) -> dict:
    """Agentic Pattern 1: Router. Decides if RAG is needed."""
    sys_prompt = SystemMessage(
        content="You are a routing agent. If the user asks about literature, research methodologies, or datasets, respond with 'RAG'. Otherwise, respond with 'CHAT'.")
    messages = [sys_prompt] + state["messages"]
    response = router_model.invoke(messages)
    return {"needs_reflection": "RAG" in response.content.strip().upper()}


def rag_extraction_node(state: AgentState) -> dict:
    """Agentic Pattern 2: Tool-Use. Extracts evidence from the RAG tool."""
    last_user_msg = state["messages"][-1].content
    evidence = extract_academic_evidence.invoke(last_user_msg)
    return {"research_context": evidence}


def synthesis_reflection_node(state: AgentState) -> dict:
    """Agentic Pattern 3: Reflection & Self-Critique. Checks for hallucinations before output."""
    context = state.get("research_context", "No context provided.")
    user_query = state["messages"][-1].content

    sys_prompt = SystemMessage(content=f"""
    You are an academic synthesis agent specializing in computer vision, edge computing architectures (like NVIDIA Jetson Nano), and TensorRT optimization. 
    1. Analyze this extracted literature context: {context}
    2. Answer the user query: {user_query}
    3. Self-Critique: Ensure you do not hallucinate benchmarks or metrics not present in the provided context.
    """)

    response = synthesis_model.invoke([sys_prompt])
    return {"messages": [response]}


def standard_chat_node(state: AgentState) -> dict:
    """Handles general routing without using OpenRouter credits."""
    response = router_model.invoke(state["messages"])
    return {"messages": [response]}


def route_flow(state: AgentState) -> Literal["rag_extraction_node", "standard_chat_node"]:
    return "rag_extraction_node" if state.get("needs_reflection") else "standard_chat_node"


# Orchestrate Agent-to-Agent Communication[cite: 1]
builder = StateGraph(AgentState)

builder.add_node("router_node", router_node)
builder.add_node("rag_extraction_node", rag_extraction_node)
builder.add_node("synthesis_reflection_node", synthesis_reflection_node)
builder.add_node("standard_chat_node", standard_chat_node)

builder.add_edge(START, "router_node")
builder.add_conditional_edges("router_node", route_flow)
builder.add_edge("rag_extraction_node", "synthesis_reflection_node")
builder.add_edge("synthesis_reflection_node", END)
builder.add_edge("standard_chat_node", END)

graph = builder.compile()
