# dev_test_evaluation.py
from pathlib import Path
from docx import Document
import json

from src.ingest import extract_strategic_objectives, extract_actions
from src.embeddings import EmbedStore
from src.pipeline_score import build_alignment_table

from src.evaluation import (
    build_ground_truth,
    predictions_from_alignment_table_rows,
    micro_prf,
    recall_at_k_from_rows,
    precision_at_k_from_rows,
    export_expert_review_sheet,
    summarize_expert_reviews
)

def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def main():
    strategic_text = read_docx(Path("data") / "STRATEGIC-PLAN-2024.docx")
    action_text    = read_docx(Path("data") / "ACTION-PLAN-2024.docx")

    strategies = extract_strategic_objectives(strategic_text)
    actions    = extract_actions(action_text)

    store = EmbedStore(persist_dir="chroma", collection_name="isps")
    store.upsert_items(strategies)
    store.upsert_items(actions)

    # Build alignment rows (same structure you already use)
    rows = build_alignment_table(
        store,
        strategies,
        actions,
        retrieve_k=15,
        top_k=3,
        linked_bonus=0.20
    )

    # ---- 1) Ground truth mapping comparison ----
    gt = build_ground_truth(actions)

    # Predictions: any top-3 action with boosted >= threshold is "predicted aligned"
    pred = predictions_from_alignment_table_rows(
        rows,
        threshold=0.55,
        use_boosted=True,
        top_k=3
    )

    prf = micro_prf(gt, pred, only_strategies_with_gt=True)
    print("\n=== Ground Truth Comparison (micro, only strategies with GT links) ===")
    print(f"TP={prf.tp} FP={prf.fp} FN={prf.fn}")
    print(f"Precision={prf.precision:.3f} Recall={prf.recall:.3f} F1={prf.f1:.3f}")

    # ---- 2) Precision/Recall IR metrics ----
    r1 = recall_at_k_from_rows(rows, gt, k=1)
    r3 = recall_at_k_from_rows(rows, gt, k=3)
    p3 = precision_at_k_from_rows(rows, gt, k=3)

    print("\n=== IR-style metrics ===")
    print(f"Recall@1 = {r1:.3f}")
    print(f"Recall@3 = {r3:.3f}")
    print(f"Precision@3 = {p3:.3f}")

    # ---- 3) Expert validation sheet for LLM recommendations ----
    out_csv = export_expert_review_sheet(
        alignment_csv="outputs/alignment_table.csv",          # your existing CSV
        improvements_json="outputs/improvements_latest.json", # created by Streamlit app
        out_csv="outputs/expert_review.csv",
        include_labels=("Weak", "Not Covered (Year-1)"),
        max_items=20
    )
    print(f"\nExpert review sheet exported: {out_csv}")
    print("Fill the 1–5 columns, then run summarize_expert_reviews() (optional).")

    # Optional: summarize if already filled
    try:
        summary = summarize_expert_reviews(out_csv)
        print("\n=== Expert Review Summary (if filled) ===")
        print(json.dumps(summary, indent=2))
    except Exception:
        pass

if __name__ == "__main__":
    main()
