import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are the Orchestrator agent in a multi-agent academic research pipeline named LitSynth AI.

Your job is to:
1. Analyze the user's research query.
2. Decompose it into a structured JSON task plan with clear sub-tasks.
3. Determine which domain areas are covered.

Respond ONLY with valid JSON in this exact format:
{
  "task_plan": {
    "primary_objective": "<1-sentence objective>",
    "sub_tasks": [
      "<sub-task 1>",
      "<sub-task 2>",
      "<sub-task 3>"
    ],
    "domain_areas": ["<area1>", "<area2>"],
    "requires_retrieval": true,
    "complexity": "high|medium|low"
  }
}"""


def run(query: str, openrouter_api_key: str, scratchpad: list, a2a_trace: list) -> dict:
    """
    Run the Orchestrator agent.

    Args:
        query: User's research question.
        groq_api_key: Groq API key from st.secrets.
        scratchpad: Shared scratchpad list (appended in place).
        a2a_trace: Shared A2A trace list (appended in place).

    Returns:
        task_plan dict.
    """
    scratchpad.append({
        "agent": "Orchestrator",
        "thought": f"Received query: '{query[:80]}...'. Decomposing into structured task plan.",
        "color": "emerald"
    })

    model = ChatOpenAI(
        model="minimax/minimax-m3:free",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=512,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Research Query: {query}"),
    ]

    response = model.invoke(messages)
    raw = response.content.strip()

    # Robust JSON extraction
    try:
        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        task_plan = json.loads(raw.strip())
    except Exception:
        task_plan = {
            "task_plan": {
                "primary_objective": query[:120],
                "sub_tasks": ["Classify intent", "Retrieve relevant literature", "Synthesize findings", "Critique output"],
                "domain_areas": ["Agentic AI", "RAG Systems"],
                "requires_retrieval": True,
                "complexity": "high"
            }
        }

    plan = task_plan.get("task_plan", task_plan)

    scratchpad.append({
        "agent": "Orchestrator",
        "thought": f"Task plan created. Sub-tasks: {len(plan.get('sub_tasks', []))}. Delegating to Router.",
        "color": "emerald"
    })

    # A2A Message: orchestrator → router
    a2a_trace.append({
        "from": "orchestrator",
        "to": "router",
        "type": "task_delegation",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "payload": {
            "query": query,
            "task_plan": plan,
            "instruction": "Classify the intent of this research query and determine if RAG retrieval is required."
        }
    })

    return plan
