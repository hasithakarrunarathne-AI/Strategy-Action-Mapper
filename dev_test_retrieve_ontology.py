from src.embeddings import EmbedStore
from src.retrieve import top_actions_for_strategy
from src.ontology import expand_with_ontology

store = EmbedStore(persist_dir="chroma", collection_name="isps")

strategy = "Increase annual research output to 200+ publications by 2028"

# A) WITHOUT ontology (manual)
res_no = store.query(query_text=strategy, n_results=5, where={"type": "action"})

print("\n=== WITHOUT ONTOLOGY ===")
for m in res_no:
    print(m["id"], round(m["similarity"], 4), m["text"][:90])

# B) WITH ontology (manual)
expanded = expand_with_ontology(strategy)
res_yes = store.query(query_text=expanded, n_results=5, where={"type": "action"})

print("\n=== WITH ONTOLOGY ===")
for m in res_yes:
    print(m["id"], round(m["similarity"], 4), m["text"][:90])
