# dev_alignment_table.py
from pathlib import Path
from docx import Document
import shutil
import os
import pandas as pd

from src.ingest import extract_strategic_objectives, extract_actions
from src.embeddings import EmbedStore
from src.pipeline_score import build_alignment_table
from collections import defaultdict

def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def main():
    # Reset DB each run while testing
    if Path("chroma").exists():
        shutil.rmtree("chroma")

    strategic_text = read_docx(Path("data") / "STRATEGIC-PLAN-2024.docx")
    action_text = read_docx(Path("data") / "ACTION-PLAN-2024.docx")

    strategies = extract_strategic_objectives(strategic_text)
    actions = extract_actions(action_text)

    print("Strategies:", len(strategies))
    print("Actions:", len(actions))

    store = EmbedStore(persist_dir="chroma", collection_name="isps")
    store.upsert_items(strategies)
    store.upsert_items(actions)

    idx = defaultdict(int)
    for a in actions:
        for so in a.get("linked_to", []) or []:
            idx[so] += 1

    has_linked = sum(1 for s in strategies if idx.get(s["id"], 0) > 0)
    print(f"\nStrategies with at least one linked action: {has_linked}/{len(strategies)}")

    table = build_alignment_table(
        store,
        strategies,
        actions,
        retrieve_k=15,
        top_k=3,
        linked_bonus=0.20
    )

    # Summary counts
    strong = sum(1 for r in table if r["label"] == "Strong")
    partial = sum(1 for r in table if r["label"] == "Partial")
    weak = sum(1 for r in table if r["label"] == "Weak")
    not_covered = sum(1 for r in table if r["label"] == "Not Covered (Year-1)")

    linked_best = sum(1 for r in table if r["best_action_linked_to_strategy"])
    total = len(table)

    print(f"\nTotal strategies: {total}")
    print(f"Strong: {strong} | Partial: {partial} | Weak: {weak} | Not Covered (Year-1): {not_covered}")
    print(f"Best action explicitly linked: {linked_best}/{total} ({(linked_best/total)*100:.1f}%)\n")

    strategy_head = {s["id"]: s["text"].replace("\n", " ").strip() for s in strategies}
    action_head = {a["id"]: a["text"].split("\n")[0].strip() for a in actions}  # first line as action title

    # Flatten for CSV
    flat_rows = []
    for r in table:
        top1 = r["top_actions"][0] if len(r["top_actions"]) > 0 else {}
        top2 = r["top_actions"][1] if len(r["top_actions"]) > 1 else {}
        top3 = r["top_actions"][2] if len(r["top_actions"]) > 2 else {}

        flat_rows.append({
            "strategy_id": r["strategy_id"],
            "strategy_headline": strategy_head.get(r["strategy_id"], ""),

            "label": r["label"],
            "mapping_mode": r.get("mapping_mode", ""),

            "best_action_id": r.get("best_action_id", ""),
            "best_action_headline": action_head.get(r.get("best_action_id", ""), ""),
            "best_similarity": r.get("best_similarity"),
            "best_boosted_similarity": r.get("best_boosted_similarity"),
            "best_action_linked_to_strategy": r.get("best_action_linked_to_strategy"),

            "top1_action_id": top1.get("action_id", "") if top1 else "",
            "top1_action_headline": action_head.get(top1.get("action_id", "") if top1 else "", ""),
            "top1_sim": top1.get("similarity") if top1 else None,
            "top1_boosted": top1.get("boosted_similarity") if top1 else None,

            "top2_action_id": top2.get("action_id", "") if top2 else "",
            "top2_action_headline": action_head.get(top2.get("action_id", "") if top2 else "", ""),
            "top2_sim": top2.get("similarity") if top2 else None,
            "top2_boosted": top2.get("boosted_similarity") if top2 else None,

            "top3_action_id": top3.get("action_id", "") if top3 else "",
            "top3_action_headline": action_head.get(top3.get("action_id", "") if top3 else "", ""),
            "top3_sim": top3.get("similarity") if top3 else None,
            "top3_boosted": top3.get("boosted_similarity") if top3 else None,
    })


    df = pd.DataFrame(flat_rows)
    os.makedirs("outputs", exist_ok=True)
    out_path = Path("outputs") / "alignment_table.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print first 8 rows quick view
    print("\nFirst 8 rows:")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
