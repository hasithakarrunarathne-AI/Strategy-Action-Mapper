# src/pipeline_score.py (improved)
from __future__ import annotations
from typing import List, Dict, Any
from collections import defaultdict

from src.embeddings import EmbedStore
from src.retrieve import top_actions_for_strategy, rerank_with_linked_to


def label(score: float) -> str:
    if score >= 0.80:
        return "Strong"
    if score >= 0.55:
        return "Partial"
    return "Weak"


def build_alignment_table(
    store: EmbedStore,
    strategies: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    retrieve_k: int = 15,
    top_k: int = 3,
    linked_bonus: float = 0.20,
) -> List[Dict[str, Any]]:

    # Build index: SOx.y -> [action_ids]
    linked_index = defaultdict(list)
    for a in actions:
        for so in a.get("linked_to", []) or []:
            linked_index[so].append(a["id"])

    rows: List[Dict[str, Any]] = []

    for s in strategies:
        sid = s["id"]

        # 1) Retrieve candidates by embeddings
        raw = top_actions_for_strategy(store, s["text"], k=retrieve_k)
        ranked = rerank_with_linked_to(raw, sid, bonus=linked_bonus)

        # 2) If explicit linked actions exist, prefer them
        linked_ids = set(linked_index.get(sid, []))
        linked_only = [m for m in ranked if m["id"] in linked_ids]

        final_ranked = linked_only if linked_only else ranked
        top = final_ranked[:top_k]

        best = top[0] if top else None
        best_boosted = best["boosted_similarity"] if best else 0.0

        # Decide label AFTER best_boosted exists
        if not linked_ids:
            final_label = "Not Covered (Year-1)"
        else:
            final_label = label(best_boosted)

        is_explicitly_linked = False
        if best:
            linked = best["meta"].get("linked_to") or []
            is_explicitly_linked = sid in linked

        rows.append({
            "strategy_id": sid,
            "strategy_text": s["text"],
            "best_similarity": best["similarity"] if best else 0.0,
            "best_boosted_similarity": best_boosted,
            "label": final_label,
            "best_action_id": best["id"] if best else None,
            "best_action_linked_to_strategy": is_explicitly_linked,
            "top_actions": [
                {
                    "action_id": m["id"],
                    "similarity": m["similarity"],
                    "boosted_similarity": m["boosted_similarity"],
                    "linked_to": m["meta"].get("linked_to"),
                }
                for m in top
            ],
            "mapping_mode": "linked_to" if linked_only else "embedding_fallback"
        })


    return rows
