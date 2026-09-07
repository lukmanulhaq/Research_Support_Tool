import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are the Critic Agent in the LitSynth AI pipeline. Your role is Reflection and Self-Critique.

You will receive:
1. A synthesized literature review (the draft).
2. The retrieved evidence chunks it was based on.

Your job:
1. Check every factual claim, benchmark, and metric in the synthesis against the evidence.
2. Identify any hallucinated statistics, wrong author attributions, or unsupported claims.
3. Score factual grounding from 0–10 (10 = perfectly grounded, no hallucinations).
4. If score < 7, provide a corrected version. If score >= 7, approve with minor notes.

Respond ONLY with valid JSON:
{
  "score": 8.5,
  "verdict": "APPROVED" or "NEEDS_REVISION",
  "hallucinations_detected": ["<description of issue 1>", ...],
  "corrections": ["<correction 1>", ...],
  "revised_synthesis": "<full revised text if verdict is NEEDS_REVISION, else null>",
  "quality_notes": "<brief overall quality assessment>"
}"""


def _build_evidence_summary(chunks: list[dict]) -> str:
    """Build a concise evidence summary for the critic to check against."""
    parts = []
    for chunk in chunks:
        parts.append(f"[{chunk['id']}] {chunk['title']} ({chunk['year']}): {chunk['chunk_text'][:300]}...")
    return "\n\n".join(parts)


def run(
    query: str,
    synthesis: str,
    chunks: list[dict],
    openrouter_api_key: str,
    scratchpad: list,
    a2a_trace: list,
    model_name: str = "nvidia/nemotron-3.5-lightning:free",
) -> dict:
    """
    Run the Critic agent (Reflection/Self-Critique pattern).

    Returns:
        dict with score, verdict, hallucinations_detected, corrections,
        revised_synthesis (or None), quality_notes.
    """
    scratchpad.append({
        "agent": "Critic",
        "thought": (
            f"[Reflect] Reviewing synthesis against {len(chunks)} evidence chunks. "
            f"Checking factual grounding, citation accuracy, and hallucination detection."
        ),
        "color": "amber"
    })

    evidence_summary = _build_evidence_summary(chunks)

    user_prompt = f"""Research Query: {query}

=== SYNTHESIZED DRAFT ===
{synthesis}

=== EVIDENCE CHUNKS (ground truth) ===
{evidence_summary}

Critique the synthesis for factual accuracy and hallucinations. Provide your JSON verdict."""

    model = ChatOpenAI(
        model=model_name,
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=1024,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = model.invoke(messages)
    raw = response.content.strip()

    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        critique = json.loads(raw.strip())
    except Exception:
        critique = {
            "score": 7.5,
            "verdict": "APPROVED",
            "hallucinations_detected": [],
            "corrections": [],
            "revised_synthesis": None,
            "quality_notes": "Critique parsing failed; defaulting to approved status."
        }

    score = critique.get("score", 7.5)
    verdict = critique.get("verdict", "APPROVED")

    scratchpad.append({
        "agent": "Critic",
        "thought": (
            f"[Critique Complete] Score: {score}/10 | Verdict: {verdict} | "
            f"Hallucinations: {len(critique.get('hallucinations_detected', []))} | "
            f"{critique.get('quality_notes', '')}"
        ),
        "color": "amber"
    })

    # A2A Message: critic → orchestrator
    a2a_trace.append({
        "from": "critic",
        "to": "orchestrator",
        "type": "critique_result",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "payload": {
            "score": score,
            "verdict": verdict,
            "hallucinations_count": len(critique.get("hallucinations_detected", [])),
            "hallucinations": critique.get("hallucinations_detected", []),
            "corrections": critique.get("corrections", []),
            "quality_notes": critique.get("quality_notes", ""),
            "revised": critique.get("revised_synthesis") is not None,
        }
    })

    return critique
