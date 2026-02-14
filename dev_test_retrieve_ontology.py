from src.embeddings import EmbedStore
from src.retrieve import top_actions_for_strategy
from src.ontology import expand_with_ontology

store = EmbedStore(persist_dir="chroma", collection_name="isps")

strategy = "Increase annual research output to 200+ publications by 2028"

# A) WITHOUT ontology (manual)
res_no = store.query(query_text=strategy, n_results=5, where={"type": "action"})

print("=== WITHOUT ONTOLOGY ===")
res_no = store.query(query_text=strategy, n_results=5, where={"type": "action"})

ids = res_no["ids"][0]
docs = res_no["documents"][0]
dists = res_no["distances"][0]

for i in range(len(ids)):
    sim = max(0.0, 1.0 - float(dists[i]))  # distance_to_similarity logic
    print(ids[i], round(sim, 4), docs[i][:90])


print("=== WITH ONTOLOGY ===")
expanded = expand_with_ontology(strategy)
res_yes = store.query(query_text=expanded, n_results=5, where={"type": "action"})

ids = res_yes["ids"][0]
docs = res_yes["documents"][0]
dists = res_yes["distances"][0]

for i in range(len(ids)):
    sim = max(0.0, 1.0 - float(dists[i]))
    print(ids[i], round(sim, 4), docs[i][:90])

