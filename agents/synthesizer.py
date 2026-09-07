import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are the Synthesis Agent in the LitSynth AI academic research pipeline.

Your job is to produce a structured, high-quality systematic literature review based on the retrieved evidence.

STRICT RULES:
1. Only use facts, benchmarks, and metrics that appear in the provided evidence chunks.
2. Cite every claim using inline paper IDs like [P1], [P4], [P21].
3. Do NOT hallucinate statistics, author names, or publication years.
4. Structure your response with these exact sections:
   - ## Executive Summary
   - ## Methodology Comparison Matrix (use a markdown table)
   - ## Key Findings & Benchmarks
   - ## Research Gaps & Future Directions
   - ## Grounded Citations

Keep the response academic, precise, and citation-rich. Minimum 400 words."""


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as a numbered evidence block."""
    parts = []
    for chunk in chunks:
        authors_str = ", ".join(chunk["authors"])
        parts.append(
            f"[{chunk['id']}] {chunk['title']} ({chunk['year']}, {chunk['venue']})\n"
            f"Authors: {authors_str}\n"
            f"Content: {chunk['chunk_text']}\n"
            f"Cosine Similarity Score: {chunk['score']:.4f}"
        )
    return "\n\n---\n\n".join(parts)


def run(
    query: str,
    chunks: list[dict],
    task_plan: dict,
    openrouter_api_key: str,
    scratchpad: list,
    a2a_trace: list,
    model_name: str = "nvidia/nemotron-3-super-120b-a12b:free",
) -> str:
    """
    Run the Synthesizer agent using a frontier model via OpenRouter.

    Returns:
        Synthesized markdown string with inline citations.
    """
    scratchpad.append({
        "agent": "Synthesizer",
        "thought": (
            f"Received {len(chunks)} evidence chunks. "
            f"Model: {model_name}. Generating academic synthesis with citation grounding."
        ),
        "color": "purple"
    })

    context = _build_context(chunks)
    objective = task_plan.get("primary_objective", query) if task_plan else query
    sub_tasks_str = ""
    if task_plan and task_plan.get("sub_tasks"):
        sub_tasks_str = "\n".join(f"  - {t}" for t in task_plan["sub_tasks"])

    user_prompt = f"""Research Objective: {objective}

Sub-tasks to address:
{sub_tasks_str if sub_tasks_str else "  - Provide a comprehensive synthesis."}

Retrieved Evidence Chunks ({len(chunks)} papers):
{context}

Generate a structured systematic literature review addressing the research objective.
Cite all evidence with [PX] inline citations."""

    model = ChatOpenAI(
        model=model_name,
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
        max_tokens=2048,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = model.invoke(messages)
    synthesis = response.content.strip()

    scratchpad.append({
        "agent": "Synthesizer",
        "thought": f"Synthesis complete. {len(synthesis.split())} words generated. Forwarding to Critic for reflection.",
        "color": "purple"
    })

    # A2A Message: synthesizer → critic
    a2a_trace.append({
        "from": "synthesizer",
        "to": "critic",
        "type": "synthesis_draft",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "payload": {
            "query": query,
            "model_used": model_name,
            "word_count": len(synthesis.split()),
            "paper_ids_cited": [c["id"] for c in chunks],
            "synthesis_preview": synthesis[:300] + "...",
        }
    })

    return synthesis
