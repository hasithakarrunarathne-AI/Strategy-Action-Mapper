from __future__ import annotations

from typing import List, Dict, Any
import json

from src.embeddings import EmbedStore
from src.ontology import expand_with_ontology


def distance_to_similarity(distance: float) -> float:
    """
    Convert cosine distance (lower is better) to a simple similarity.
    With normalized embeddings, cosine distance is usually ~[0, 1].
    """
    return max(0.0, 1.0 - float(distance))


def _maybe_json_load(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def top_actions_for_strategy(store: EmbedStore, strategy_text: str, k: int = 5) -> List[Dict[str, Any]]:
    # res = store.query(
    #     query_text=strategy_text,
    #     n_results=k,
    #     where={"type": "action"},
    # )
    expanded = expand_with_ontology(strategy_text)
    res = store.query(
    query_text=expanded,
    n_results=k,
    where={"type": "action"},
    )

    out = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i] or {}
        # linked_to was stored as JSON string; convert back if possible
        if "linked_to" in meta:
            meta["linked_to"] = _maybe_json_load(meta["linked_to"])

        out.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": meta,
            "distance": float(res["distances"][0][i]),
            "similarity": distance_to_similarity(res["distances"][0][i]),
        })
    return out


def top_strategies_for_action(store: EmbedStore, action_text: str, k: int = 5) -> List[Dict[str, Any]]:
    # res = store.query(
    #     query_text=action_text,
    #     n_results=k,
    #     where={"type": "strategy"},
    # )
    expanded = expand_with_ontology(action_text)
    res = store.query(
    query_text=expanded,
    n_results=k,
    where={"type": "strategy"},
    )

    out = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i] or {}
        out.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": meta,
            "distance": float(res["distances"][0][i]),
            "similarity": distance_to_similarity(res["distances"][0][i]),
        })
    return out

def rerank_with_linked_to(matches, strategy_id: str, bonus: float = 0.20):
    """
    Boost matches that are explicitly linked to the given strategy_id.
    Adds 'boosted_similarity' and sorts descending by it.
    """
    reranked = []
    for m in matches:
        linked = m["meta"].get("linked_to") or []
        boosted = m["similarity"] + (bonus if strategy_id in linked else 0.0)
        boosted = min(1.0, boosted)  # keep nice scale
        m2 = dict(m)
        m2["boosted_similarity"] = boosted
        reranked.append(m2)

    reranked.sort(key=lambda x: x["boosted_similarity"], reverse=True)
    return reranked

