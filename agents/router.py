import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are the Router agent in the LitSynth AI pipeline.

Classify the research query into one of two categories:
- RAG_REQUIRED: The query asks about specific papers, methodologies, benchmarks, datasets, or literature content that requires searching the academic corpus.
- CONVERSATIONAL: The query is a general question, greeting, or meta-question that can be answered from parametric knowledge alone.

Respond ONLY with valid JSON:
{
  "intent": "RAG_REQUIRED" or "CONVERSATIONAL",
  "confidence": 0.0 to 1.0,
  "reasoning": "<1 sentence explaining the classification>"
}"""


def run(query: str, task_plan: dict, openrouter_api_key: str, scratchpad: list, a2a_trace: list) -> dict:
    """
    Run the Router agent.

    Returns:
        dict with keys: intent, confidence, reasoning
    """
    scratchpad.append({
        "agent": "Router",
        "thought": "Analyzing query intent. Determining if academic corpus retrieval is needed.",
        "color": "blue"
    })

    model = ChatOpenAI(
        model="liquid/lfm-2.5-2.6b:free",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=128,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}"),
    ]

    response = model.invoke(messages)
    raw = response.content.strip()

    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception:
        # Default to RAG if parsing fails — safer for academic queries
        result = {
            "intent": "RAG_REQUIRED",
            "confidence": 0.85,
            "reasoning": "Unable to parse response; defaulting to RAG for safety."
        }

    intent = result.get("intent", "RAG_REQUIRED")
    confidence = result.get("confidence", 0.85)

    scratchpad.append({
        "agent": "Router",
        "thought": f"Intent classified as [{intent}] with confidence {confidence:.0%}. {result.get('reasoning', '')}",
        "color": "blue"
    })

    # A2A Message: router → retriever or synthesizer
    next_agent = "retriever" if intent == "RAG_REQUIRED" else "synthesizer"
    a2a_trace.append({
        "from": "router",
        "to": next_agent,
        "type": "routing_decision",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "payload": {
            "intent": intent,
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
            "query": query,
            "task_plan": task_plan,
        }
    })

    return result
