from datetime import datetime
from rag import faiss_store


def run(
    query: str,
    task_plan: dict,
    routing_result: dict,
    faiss_store_state: dict,
    scratchpad: list,
    a2a_trace: list,
    k: int = 5,
) -> list[dict]:
    """
    Run the Retriever agent using the ReAct tool-use pattern.

    ReAct Loop:
        Iteration 1:
            Thought: "Search for papers matching the primary query."
            Action:  faiss_search(query, k=5)
            Observation: [top-k chunks with scores]

        Iteration 2 (optional, if top score < 0.4):
            Thought: "Low confidence results. Expanding search with sub-task keywords."
            Action:  faiss_search(expanded_query, k=3)
            Observation: [additional chunks]

    Returns:
        List of retrieved chunk dicts with id, title, score, chunk_text.
    """
    # --- Thought 1 ---
    scratchpad.append({
        "agent": "Retriever",
        "thought": f"[Thought] Initiating FAISS vector search for: '{query[:60]}...'",
        "color": "cyan"
    })

    # --- Action 1 ---
    scratchpad.append({
        "agent": "Retriever",
        "thought": f"[Action] faiss_search(query='{query[:50]}...', k={k})",
        "color": "cyan"
    })

    results = faiss_store.search(query, faiss_store_state, k=k)

    # --- Observation 1 ---
    top_score = results[0]["score"] if results else 0.0
    scratchpad.append({
        "agent": "Retriever",
        "thought": (
            f"[Observation] Retrieved {len(results)} chunks. "
            f"Top cosine similarity: {top_score:.4f}. "
            f"Papers: {', '.join([r['id'] for r in results])}"
        ),
        "color": "cyan"
    })

    # --- ReAct Iteration 2 (expand if low confidence) ---
    if top_score < 0.40 and task_plan:
        sub_tasks = task_plan.get("sub_tasks", [])
        if sub_tasks:
            expanded_query = f"{query} {' '.join(sub_tasks[:2])}"
            scratchpad.append({
                "agent": "Retriever",
                "thought": f"[Thought] Low confidence ({top_score:.2f}). Expanding with sub-task keywords.",
                "color": "cyan"
            })
            scratchpad.append({
                "agent": "Retriever",
                "thought": f"[Action] faiss_search(expanded_query, k=3)",
                "color": "cyan"
            })
            extra = faiss_store.search(expanded_query, faiss_store_state, k=3)
            scratchpad.append({
                "agent": "Retriever",
                "thought": f"[Observation] Additional chunks: {', '.join([r['id'] for r in extra])}",
                "color": "cyan"
            })
            # Merge, deduplicate by id
            seen_ids = {r["id"] for r in results}
            for r in extra:
                if r["id"] not in seen_ids:
                    results.append(r)
                    seen_ids.add(r["id"])

    scratchpad.append({
        "agent": "Retriever",
        "thought": f"[Final] Sending {len(results)} grounded evidence chunks to Synthesizer.",
        "color": "cyan"
    })

    # A2A Message: retriever → synthesizer
    a2a_trace.append({
        "from": "retriever",
        "to": "synthesizer",
        "type": "retrieved_evidence",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "payload": {
            "query": query,
            "chunks_retrieved": len(results),
            "top_cosine_score": round(top_score, 4),
            "paper_ids": [r["id"] for r in results],
            "paper_titles": [r["title"] for r in results],
        }
    })

    return results
