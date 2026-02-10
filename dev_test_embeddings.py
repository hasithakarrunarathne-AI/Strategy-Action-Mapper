# dev_test_embeddings.py
from pathlib import Path
from docx import Document

from src.ingest import extract_strategic_objectives, extract_actions
from src.embeddings import EmbedStore
from src.retrieve import top_actions_for_strategy, rerank_with_linked_to


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def main():
    strategic_text = read_docx(Path("data") / "STRATEGIC-PLAN-2024.docx")
    action_text    = read_docx(Path("data") / "ACTION-PLAN-2024.docx")

    strategies = extract_strategic_objectives(strategic_text)
    actions    = extract_actions(action_text)

    print("Strategies:", len(strategies))
    print("Actions:", len(actions))

    store = EmbedStore(persist_dir="chroma", collection_name="isps")

    # Upsert
    store.upsert_items(strategies)
    store.upsert_items(actions)

    # Test retrieval for the first strategy
    if not strategies:
        print("No strategies found.")
        return

    s0 = strategies[0]
    print("\nStrategy:", s0["id"], "-", s0["text"])

    matches = top_actions_for_strategy(store, s0["text"], k=15)
    matches = rerank_with_linked_to(matches, s0["id"], bonus=0.20)

    print("\nTop 5 matching actions (after boost):")
    for m in matches[:5]:
        print(f"- {m['id']} | sim={m['similarity']:.3f} | boosted={m['boosted_similarity']:.3f} | linked_to={m['meta'].get('linked_to')}")


if __name__ == "__main__":
    main()
