# src/evaluation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, List, Any, Tuple, Optional
from pathlib import Path
import json
import pandas as pd


@dataclass
class PRF:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _safe_set(x) -> Set[str]:
    if not x:
        return set()
    if isinstance(x, (list, tuple, set)):
        return {str(i).strip() for i in x if str(i).strip()}
    return {str(x).strip()}


# -------------------------------
# 1) Ground truth from ActionPlan
# -------------------------------
def build_ground_truth(actions: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Ground truth: for each strategy_id SOx.y, which action_ids are explicitly linked_to it.
    Uses your parsed actions where each action has linked_to list. (ingest.py) 
    """
    gt: Dict[str, Set[str]] = {}
    for a in actions:
        aid = str(a.get("id", "")).strip()
        for sid in _safe_set(a.get("linked_to", [])):
            gt.setdefault(sid, set()).add(aid)
    return gt


# ------------------------------------
# 2) Predictions from alignment results
# ------------------------------------
def predictions_from_alignment_table_rows(
    rows: List[Dict[str, Any]],
    *,
    threshold: float = 0.55,
    use_boosted: bool = True,
    top_k: int = 3
) -> Dict[str, Set[str]]:
    """
    Build predicted alignments as a set of action_ids per strategy_id.
    We treat any retrieved action with score >= threshold as a predicted positive.

    rows are the dict rows produced by build_alignment_table() in pipeline_score.py.
    """
    pred: Dict[str, Set[str]] = {}

    score_key = "boosted_similarity" if use_boosted else "similarity"

    for r in rows:
        sid = str(r.get("strategy_id", "")).strip()
        actions = r.get("top_actions", [])[:top_k] if r.get("top_actions") else []
        chosen = set()
        for m in actions:
            aid = str(m.get("action_id", "")).strip()
            sc = float(m.get(score_key, 0.0) or 0.0)
            if aid and sc >= threshold:
                chosen.add(aid)
        pred[sid] = chosen
    return pred


def predictions_from_alignment_csv(
    csv_path: str = "outputs/alignment_table.csv",
    *,
    threshold: float = 0.55,
    use_boosted: bool = True,
    top_k: int = 3
) -> Dict[str, Set[str]]:
    """
    If you prefer evaluating using your saved CSV instead of raw rows,
    this reads top1/top2/top3 ids + boosted scores.
    """
    df = pd.read_csv(csv_path)
    pred: Dict[str, Set[str]] = {}

    # Decide which columns to use
    if use_boosted:
        score_cols = ["top1_boosted", "top2_boosted", "top3_boosted"]
    else:
        score_cols = ["top1_sim", "top2_sim", "top3_sim"]

    id_cols = ["top1_action_id", "top2_action_id", "top3_action_id"]

    for _, row in df.iterrows():
        sid = str(row.get("strategy_id", "")).strip()
        chosen = set()
        for aid_col, sc_col in zip(id_cols[:top_k], score_cols[:top_k]):
            aid = str(row.get(aid_col, "")).strip()
            try:
                sc = float(row.get(sc_col, 0.0) or 0.0)
            except Exception:
                sc = 0.0
            if aid and sc >= threshold:
                chosen.add(aid)
        pred[sid] = chosen

    return pred


# -----------------------------
# 3) Precision/Recall/F1 (micro)
# -----------------------------
def micro_prf(
    gt: Dict[str, Set[str]],
    pred: Dict[str, Set[str]],
    *,
    only_strategies_with_gt: bool = True
) -> PRF:
    """
    Micro-average over all strategy-action pairs, but implemented as set comparisons per strategy.
    - only_strategies_with_gt=True: evaluate only strategies that actually have linked actions in GT.
      This avoids unfairly penalizing Year-1 not-covered strategies.
    """
    tp = fp = fn = 0

    all_sids = set(pred.keys()) | set(gt.keys())
    for sid in all_sids:
        gt_set = gt.get(sid, set())
        if only_strategies_with_gt and not gt_set:
            continue
        pr_set = pred.get(sid, set())

        tp += len(gt_set & pr_set)
        fp += len(pr_set - gt_set)
        fn += len(gt_set - pr_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return PRF(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


# -----------------------------
# 4) IR-style metrics: Recall@K
# -----------------------------
def recall_at_k_from_rows(
    rows: List[Dict[str, Any]],
    gt: Dict[str, Set[str]],
    *,
    k: int = 1,
    use_boosted: bool = True
) -> float:
    """
    Recall@K (per-strategy success rate):
      For strategies that have ground-truth linked actions,
      success = at least one GT action appears in top-K retrieved actions.
    """
    hits = 0
    total = 0

    for r in rows:
        sid = str(r.get("strategy_id", "")).strip()
        gt_set = gt.get(sid, set())
        if not gt_set:
            continue
        total += 1

        top = (r.get("top_actions", []) or [])[:k]
        retrieved = {str(m.get("action_id", "")).strip() for m in top if m.get("action_id")}
        if retrieved & gt_set:
            hits += 1

    return hits / total if total else 0.0


def precision_at_k_from_rows(
    rows: List[Dict[str, Any]],
    gt: Dict[str, Set[str]],
    *,
    k: int = 3
) -> float:
    """
    Precision@K (micro-ish):
      Across strategies with GT, count how many retrieved in top-K are correct / total retrieved.
    """
    correct = 0
    retrieved_total = 0

    for r in rows:
        sid = str(r.get("strategy_id", "")).strip()
        gt_set = gt.get(sid, set())
        if not gt_set:
            continue

        top = (r.get("top_actions", []) or [])[:k]
        retrieved = [str(m.get("action_id", "")).strip() for m in top if m.get("action_id")]
        retrieved_total += len(retrieved)
        correct += sum(1 for aid in retrieved if aid in gt_set)

    return correct / retrieved_total if retrieved_total else 0.0


# ---------------------------------------
# 5) Expert validation of LLM recommendations
# ---------------------------------------
def export_expert_review_sheet(
    alignment_csv: str = "outputs/alignment_table.csv",
    improvements_json: str = "outputs/improvements_latest.json",
    out_csv: str = "outputs/expert_review.csv",
    *,
    include_labels: Tuple[str, ...] = ("Weak", "Not Covered (Year-1)"),
    max_items: int = 20
) -> str:
    """
    Creates a CSV for manual/expert review.
    Reviewer fills: relevance/feasibility/kpi_quality (1-5) + comments.
    """
    df = pd.read_csv(alignment_csv)

    # load improvements if available
    imp_path = Path(improvements_json)
    improvements = {}
    if imp_path.exists():
        improvements = json.loads(imp_path.read_text(encoding="utf-8") or "{}")

    # filter candidates
    cand = df[df["label"].isin(include_labels)].copy()

    # prioritize weakest first
    if "best_boosted_similarity" in cand.columns:
        cand = cand.sort_values("best_boosted_similarity", ascending=True)

    cand = cand.head(max_items)

    rows = []
    for _, r in cand.iterrows():
        sid = str(r.get("strategy_id", ""))
        head = str(r.get("strategy_headline", ""))

        payload = improvements.get(sid, {})
        sug0 = (payload.get("suggestions") or [{}])[0]
        proposed = str(sug0.get("proposed_text", ""))
        kpis = sug0.get("kpis") or []
        citations = sug0.get("citations") or []

        rows.append({
            "strategy_id": sid,
            "strategy_headline": head,
            "current_label": str(r.get("label", "")),
            "best_action_id": str(r.get("best_action_id", "")),
            "best_boosted_similarity": r.get("best_boosted_similarity", ""),
            "llm_proposed_strategy": proposed,
            "llm_kpis_json": json.dumps(kpis, ensure_ascii=False),
            "llm_citations": ", ".join(citations) if citations else "",

            # ---- reviewer fills these ----
            "expert_relevance_1to5": "",
            "expert_feasibility_1to5": "",
            "expert_kpi_quality_1to5": "",
            "expert_comments": "",
        })

    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv


def summarize_expert_reviews(
    review_csv: str = "outputs/expert_review.csv"
) -> Dict[str, Any]:
    """
    Reads filled expert_review.csv and summarizes average scores + counts.
    """
    df = pd.read_csv(review_csv)

    def to_num(col):
        return pd.to_numeric(df[col], errors="coerce")

    rel = to_num("expert_relevance_1to5")
    feas = to_num("expert_feasibility_1to5")
    kpiq = to_num("expert_kpi_quality_1to5")

    summary = {
        "n_rows": int(len(df)),
        "n_scored_relevance": int(rel.notna().sum()),
        "avg_relevance": float(rel.mean()) if rel.notna().any() else None,
        "avg_feasibility": float(feas.mean()) if feas.notna().any() else None,
        "avg_kpi_quality": float(kpiq.mean()) if kpiq.notna().any() else None,
    }
    return summary
