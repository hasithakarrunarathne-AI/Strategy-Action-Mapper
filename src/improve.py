# src/improve.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from src.embeddings import EmbedStore
from src.retrieve import top_actions_for_strategy, rerank_with_linked_to
from src.llm_ollama import ollama_generate


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Ollama sometimes returns extra text.
    We parse the first {...} block.
    """
    t = text.strip()
    i = t.find("{")
    j = t.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(t[i : j + 1])


def _topk_actions_context(ranked: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    out = []
    for m in ranked[:k]:
        out.append({
            "action_id": m["id"],
            "action_text": m["text"],
            "similarity": float(m["similarity"]),
            "boosted_similarity": float(m.get("boosted_similarity", m["similarity"])),
            "linked_to": m.get("meta", {}).get("linked_to", []),
        })
    return out


def _best_scores(ranked: List[Dict[str, Any]]) -> Tuple[float, float, str]:
    """
    returns (best_raw_sim, best_boosted_sim, best_action_id)
    """
    if not ranked:
        return 0.0, 0.0, ""
    best = ranked[0]
    raw = float(best["similarity"])
    boosted = float(best.get("boosted_similarity", raw))
    return raw, boosted, str(best["id"])


def build_improvement_prompt(
    strategy_id: str,
    strategy_text: str,
    actions_ctx: List[Dict[str, Any]],
    label: str,
    before_boosted: float
) -> str:
    actions_block = "\n".join(
        [f"- {a['action_id']} (boosted={a['boosted_similarity']:.3f}): {a['action_text']}"
         for a in actions_ctx]
    )

    return f"""
You improve a strategy so it aligns with the provided action plan items.

HARD RULES:
- Use ONLY the action items listed under CONTEXT_ACTIONS. Do not invent actions.
- Output MUST be valid JSON only. No markdown, no commentary.
- Proposed strategy must stay realistic and measurable.
- Include 1–3 KPIs with: name, metric, target, timeframe.
- In "citations", include action_ids you used.

INPUT:
strategy_id: {strategy_id}
current_label: {label}
current_boosted_similarity: {before_boosted:.3f}
strategy_text: {strategy_text}

CONTEXT_ACTIONS:
{actions_block}

OUTPUT JSON SCHEMA:
{{
  "strategy_id": "{strategy_id}",
  "issue": "low_alignment|not_covered|missing_kpi|too_vague|other",
  "top_actions_used": ["A1.2"],
  "suggestions": [
    {{
      "type": "rewrite_strategy",
      "proposed_text": "...",
      "rationale": "...",
      "kpis": [
        {{"name":"...", "metric":"...", "target":"...", "timeframe":"..."}}
      ],
      "expected_alignment_gain": "high|medium|low",
      "citations": ["A1.2"]
    }}
  ]
}}
""".strip()


def build_retry_prompt(prev_prompt: str, reason: str, before: float, after: float) -> str:
    return (
        prev_prompt
        + "\n\n"
        + f"RETRY_CONDITION:\nThe previous suggestion failed evaluation.\n"
          f"Reason: {reason}\nBefore boosted={before:.3f}, After boosted={after:.3f}\n"
          f"Revise proposed_text to better match the actions (still only using CONTEXT_ACTIONS), "
          f"and keep KPI measurable. Output JSON only."
    )


def generate_strategy_improvement(
    store: EmbedStore,
    strategy_id: str,
    strategy_text: str,
    label: str = "Weak",
    *,
    retrieve_k: int = 15,
    context_k: int = 5,
    linked_bonus: float = 0.20,
    model: str = "qwen2.5:7b",
    temperature: float = 0.2,
    min_gain: float = 0.05,
    target_boosted: float = 0.55,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Returns JSON payload + evaluation block:
    - before/after boosted similarity (Chroma distance->similarity + linked_to boost)
    """

    # ---- BEFORE retrieval (same logic as your alignment pipeline) ----
    raw = top_actions_for_strategy(store, strategy_text, k=retrieve_k)
    ranked = rerank_with_linked_to(raw, strategy_id, bonus=linked_bonus)
    before_raw, before_boosted, before_best_id = _best_scores(ranked)

    actions_ctx = _topk_actions_context(ranked, k=context_k)
    prompt = build_improvement_prompt(strategy_id, strategy_text, actions_ctx, label, before_boosted)

    last_error = ""
    last_after_boosted = 0.0

    for attempt in range(max_retries + 1):
        out = ollama_generate(prompt, model=model, temperature=temperature)

        try:
            payload = _extract_json(out)

            # minimal validation
            if payload.get("strategy_id") != strategy_id:
                raise ValueError("strategy_id mismatch in JSON")
            sug = (payload.get("suggestions") or [None])[0]
            if not sug or not isinstance(sug, dict):
                raise ValueError("missing suggestions[0]")
            proposed = sug.get("proposed_text", "")
            if not isinstance(proposed, str) or len(proposed) < 20:
                raise ValueError("proposed_text too short")

            # ---- AFTER retrieval: re-query using proposed strategy text ----
            raw2 = top_actions_for_strategy(store, proposed, k=retrieve_k)
            ranked2 = rerank_with_linked_to(raw2, strategy_id, bonus=linked_bonus)
            after_raw, after_boosted, after_best_id = _best_scores(ranked2)

            passed = (after_boosted >= target_boosted) or ((after_boosted - before_boosted) >= min_gain)

            payload["evaluation"] = {
                "before_best_action_id": before_best_id,
                "after_best_action_id": after_best_id,
                "before_raw_similarity": round(before_raw, 4),
                "before_boosted_similarity": round(before_boosted, 4),
                "after_raw_similarity": round(after_raw, 4),
                "after_boosted_similarity": round(after_boosted, 4),
                "delta_boosted": round(after_boosted - before_boosted, 4),
                "passed": bool(passed),
            }

            if passed:
                return payload

            # not good enough -> retry with feedback
            last_after_boosted = after_boosted
            last_error = "No sufficient alignment gain"
            prompt = build_retry_prompt(prompt, last_error, before_boosted, after_boosted)

        except Exception as e:
            last_error = str(e)
            prompt = build_retry_prompt(prompt, f"Invalid JSON/validation: {last_error}", before_boosted, last_after_boosted)

    # fallback payload
    return {
        "strategy_id": strategy_id,
        "issue": "other",
        "top_actions_used": [],
        "suggestions": [{
            "type": "rewrite_strategy",
            "proposed_text": strategy_text,
            "rationale": f"LLM improvement failed: {last_error}",
            "kpis": [{"name": "Define KPI", "metric": "TBD", "target": "TBD", "timeframe": "TBD"}],
            "expected_alignment_gain": "low",
            "citations": []
        }],
        "evaluation": {"passed": False, "error": last_error}
    }

import re

def _simple_keywords(text: str, max_terms: int = 10) -> List[str]:
    """Tiny keyword extractor (no extra libs)."""
    stop = {
        "the","a","an","and","or","to","of","in","for","on","with","by","as","is","are","be",
        "this","that","these","those","it","from","at","into","will","should","can"
    }
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", (text or "").lower())
    out = []
    for w in words:
        if w in stop:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= max_terms:
            break
    return out


def build_agent_revision_prompt(
    base_prompt: str,
    *,
    iteration: int,
    target_boosted: float,
    before_boosted: float,
    after_boosted: float,
    best_action_id: str,
    best_action_text: str,
) -> str:
    missing_terms = ", ".join(_simple_keywords(best_action_text, max_terms=10))
    return (
        base_prompt
        + "\n\n"
        + f"AGENT_LOOP_ITERATION: {iteration}\n"
          f"GOAL: Make after_boosted_similarity >= {target_boosted:.2f}.\n"
          f"CURRENT: before_boosted={before_boosted:.3f}, after_boosted={after_boosted:.3f}\n"
          f"BEST_ACTION_TO_ALIGN: {best_action_id}\n"
          f"KEY_TERMS_FROM_BEST_ACTION (use 2-4 naturally): {missing_terms}\n"
          f"REVISION_RULES:\n"
          f"- Use ONLY CONTEXT_ACTIONS.\n"
          f"- Make strategy wording closer to BEST_ACTION_TO_ALIGN.\n"
          f"- KPIs must measure the action outcome (metric+target+timeframe).\n"
          f"- Output JSON only.\n"
    )


def generate_strategy_improvement_agentic(
    store: EmbedStore,
    strategy_id: str,
    strategy_text: str,
    label: str = "Weak",
    *,
    retrieve_k: int = 15,
    context_k: int = 5,
    linked_bonus: float = 0.20,
    model: str = "qwen2.5:7b",
    temperature: float = 0.2,
    min_gain: float = 0.05,
    target_boosted: float = 0.60,
    max_iters: int = 5,   # <-- real agentic loop
) -> Dict[str, Any]:
    """
    True agentic loop:
      propose -> re-score -> targeted revise -> repeat (up to max_iters)
    Returns best attempt + iteration trace.
    """

    # ---- BEFORE retrieval ----
    raw = top_actions_for_strategy(store, strategy_text, k=retrieve_k)
    ranked = rerank_with_linked_to(raw, strategy_id, bonus=linked_bonus)
    if not ranked:
        return {
            "strategy_id": strategy_id,
            "issue": "other",
            "top_actions_used": [],
            "suggestions": [],
            "evaluation": {"passed": False, "error": "No actions retrieved", "iterations": []},
        }

    before_raw, before_boosted, before_best_id = _best_scores(ranked)

    # fixed context for the whole loop (stable grounding)
    actions_ctx = _topk_actions_context(ranked, k=context_k)
    base_prompt = build_improvement_prompt(strategy_id, strategy_text, actions_ctx, label, before_boosted)
    prompt = base_prompt

    best_payload: Optional[Dict[str, Any]] = None
    best_after = -1.0
    trace: List[Dict[str, Any]] = []

    current_text = strategy_text

    for it in range(1, max_iters + 1):
        out = ollama_generate(prompt, model=model, temperature=temperature)

        try:
            payload = _extract_json(out)

            # minimal validation (same as your retry version)
            if payload.get("strategy_id") != strategy_id:
                raise ValueError("strategy_id mismatch in JSON")
            sug = (payload.get("suggestions") or [None])[0]
            if not sug or not isinstance(sug, dict):
                raise ValueError("missing suggestions[0]")
            proposed = sug.get("proposed_text", "")
            if not isinstance(proposed, str) or len(proposed) < 20:
                raise ValueError("proposed_text too short")

            # ---- AFTER retrieval: re-query using proposed strategy text ----
            raw2 = top_actions_for_strategy(store, proposed, k=retrieve_k)
            ranked2 = rerank_with_linked_to(raw2, strategy_id, bonus=linked_bonus)
            after_raw, after_boosted, after_best_id = _best_scores(ranked2)

            passed = (after_boosted >= target_boosted) or ((after_boosted - before_boosted) >= min_gain)

            trace.append({
                "iter": it,
                "after_best_action_id": after_best_id,
                "after_boosted": round(after_boosted, 4),
                "delta": round(after_boosted - before_boosted, 4),
                "passed": bool(passed),
            })

            # keep best attempt
            if after_boosted > best_after:
                best_after = after_boosted
                best_payload = payload

            # attach evaluation each round (use latest best)
            payload["evaluation"] = {
                "before_best_action_id": before_best_id,
                "after_best_action_id": after_best_id,
                "before_raw_similarity": round(before_raw, 4),
                "before_boosted_similarity": round(before_boosted, 4),
                "after_raw_similarity": round(after_raw, 4),
                "after_boosted_similarity": round(after_boosted, 4),
                "delta_boosted": round(after_boosted - before_boosted, 4),
                "passed": bool(passed),
                "iterations": trace,
            }

            if passed:
                return payload

            # targeted revision prompt for next iteration
            best_action = ranked2[0] if ranked2 else ranked[0]
            prompt = build_agent_revision_prompt(
                base_prompt,
                iteration=it,
                target_boosted=target_boosted,
                before_boosted=before_boosted,
                after_boosted=after_boosted,
                best_action_id=str(best_action.get("id", "")),
                best_action_text=str(best_action.get("text", "")),
            )
            current_text = proposed

        except Exception as e:
            trace.append({"iter": it, "error": str(e), "passed": False})
            prompt = base_prompt + "\n\nIMPORTANT: Previous output invalid. Output JSON only."

    # return best attempt + full trace
    if best_payload is None:
        return {
            "strategy_id": strategy_id,
            "issue": "other",
            "top_actions_used": [],
            "suggestions": [],
            "evaluation": {"passed": False, "error": "No valid attempt", "iterations": trace},
        }

    best_payload["evaluation"] = {
        "before_best_action_id": before_best_id,
        "before_raw_similarity": round(before_raw, 4),
        "before_boosted_similarity": round(before_boosted, 4),
        "best_after_boosted_similarity": round(best_after, 4),
        "passed": False,
        "iterations": trace,
    }
    return best_payload

