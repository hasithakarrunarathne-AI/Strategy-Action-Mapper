import pandas as pd
import streamlit as st
from pathlib import Path
from src.embeddings import EmbedStore
from src.improve import generate_strategy_improvement
from src.improve import generate_strategy_improvement_agentic
from src.ontology import expand_with_ontology

import json
from datetime import datetime
from pathlib import Path
from src.report import generate_markdown_report

Path("outputs").mkdir(exist_ok=True)

st.set_page_config(page_title="ISPS Alignment Dashboard", layout="wide")

st.title("ISPS: Strategy–Action Alignment Dashboard")
st.caption("Embeddings + Chroma + Linked-to boosting + Year-1 coverage labeling")

CSV_PATH = Path("outputs") / "alignment_table.csv"

# ---------- Load Data ----------
@st.cache_data
def load_alignment_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure consistent types
    for c in ["best_similarity", "best_boosted_similarity", "top1_sim", "top1_boosted",
              "top2_sim", "top2_boosted", "top3_sim", "top3_boosted"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Fill missing ids as empty string for nicer UI
    for c in ["top1_action_id", "top2_action_id", "top3_action_id", "best_action_id", "mapping_mode"]:
        if c in df.columns:
            df[c] = df[c].fillna("")
    if "label" in df.columns:
        df["label"] = df["label"].fillna("")
    return df

if not CSV_PATH.exists():
    st.error(f"Cannot find {CSV_PATH}. Run dev_test_alignment_table.py first to generate the CSV.")
    st.stop()

df = load_alignment_table(CSV_PATH)

@st.cache_resource
def get_store():
    return EmbedStore(persist_dir="chroma", collection_name="isps")

store = get_store()

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")

labels = sorted(df["label"].dropna().unique().tolist())
selected_labels = st.sidebar.multiselect("Label", labels, default=labels)

modes = sorted(df["mapping_mode"].dropna().unique().tolist()) if "mapping_mode" in df.columns else []
selected_modes = st.sidebar.multiselect("Mapping Mode", modes, default=modes) if modes else []

min_score = st.sidebar.slider(
    "Minimum boosted similarity",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01
)

debug_onto = st.sidebar.checkbox("Debug ontology expansion", value=False)

#Executive report
if st.sidebar.button("Generate Executive Report"):
    out_path = generate_markdown_report(
        alignment_csv="outputs/alignment_table.csv",
        improvements_json="outputs/improvements_latest.json",
    )
    st.sidebar.success(f"Report generated: {out_path}")
    st.sidebar.download_button(
        "Download report (Markdown)",
        data=open(out_path, "r", encoding="utf-8").read(),
        file_name=out_path.split("/")[-1],
        mime="text/markdown"
    )

filtered = df[df["label"].isin(selected_labels)]
if selected_modes:
    filtered = filtered[filtered["mapping_mode"].isin(selected_modes)]
filtered = filtered[filtered["best_boosted_similarity"] >= min_score]

# ---------- Summary Metrics ----------
col1, col2, col3, col4, col5 = st.columns(5)

total = len(df)
col1.metric("Total Strategies", total)

def count_label(x): 
    return int((df["label"] == x).sum())

col2.metric("Strong", count_label("Strong"))
col3.metric("Partial", count_label("Partial"))
col4.metric("Weak", count_label("Weak"))
col5.metric("Not Covered (Year-1)", count_label("Not Covered (Year-1)"))

# ---------- Main Table ----------

def highlight_label(row):
    if row["label"] == "Strong":
        return ["background-color: #1b5e20"] * len(row)
    if row["label"] == "Partial":
        return ["background-color: #f9a825"] * len(row)
    if row["label"] == "Weak":
        return ["background-color: #b71c1c"] * len(row)   # red
    if row["label"] == "Not Covered (Year-1)":
        return ["background-color: #37474f"] * len(row)
    return [""] * len(row)



st.subheader("Alignment Table")

display_cols = [
    "strategy_id", "label", "mapping_mode",
    "best_action_id", "best_boosted_similarity",
    "top1_action_id", "top1_boosted",
    "top2_action_id", "top2_boosted",
    "top3_action_id", "top3_boosted",
]
display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[display_cols]
    .sort_values(["label", "best_boosted_similarity"], ascending=[True, False])
    .style.apply(highlight_label, axis=1),
    use_container_width=True,
    height=420
)

# ---------- Strategy Drill-down ----------
st.subheader("Strategy Details")

strategy_ids = filtered["strategy_id"].tolist()
selected = st.selectbox("Select a strategy to inspect", strategy_ids)

row = df[df["strategy_id"] == selected].iloc[0]
st.markdown("#### Strategy headline")
st.info(row.get("strategy_headline", ""))

tab_details, tab_graph = st.tabs(["Details", "Knowledge Graph"])

with tab_details:
    left, right = st.columns([2, 1])

with left:
    st.markdown(f"### {row['strategy_id']} — {row['label']}")
    #st.write(row.get("strategy_headline", ""))

    if debug_onto:
        expanded_query = expand_with_ontology(row.get("strategy_headline", ""))
        st.write("Expanded query used for retrieval:")
        st.code(expanded_query)

    if row["label"] == "Partial":
        st.warning(
            "Partial alignment: at least one action exists, but similarity is moderate or coverage is limited."
        )

    st.markdown("#### Top Matching Actions")
    actions_table = pd.DataFrame([
        {"Rank": 1, "Action": row.get("top1_action_id", ""), "Headline": row.get("top1_action_headline", ""),  "Boosted": row.get("top1_boosted", None)},
        {"Rank": 2, "Action": row.get("top2_action_id", ""), "Headline": row.get("top2_action_headline", ""),  "Boosted": row.get("top2_boosted", None)},
        {"Rank": 3, "Action": row.get("top3_action_id", ""), "Headline": row.get("top3_action_headline", ""),  "Boosted": row.get("top3_boosted", None)},
    ])
    actions_table = actions_table[actions_table["Action"] != ""]
    st.dataframe(actions_table, use_container_width=True, hide_index=True)

with right:
    st.markdown("### Actions")
    st.metric("Best Action", row.get("best_action_id", ""))
    st.metric("Best Boosted Similarity", f"{row.get('best_boosted_similarity', 0.0):.3f}")
    st.metric("Mapping Mode", row.get("mapping_mode", ""))

with left:
    st.markdown("---")
    st.markdown("### LLM Improvements (Ollama)")
    if row["label"] in ["Weak", "Not Covered (Year-1)"]:
        st.info("This strategy needs improvement suggestions.")
        if st.button("Generate Improvement Suggestions"):
            with st.spinner("Generating with Ollama + RAG..."):
                # payload = generate_strategy_improvement(
                #     store=store,
                #     strategy_id=row["strategy_id"],
                #     strategy_text=row.get("strategy_headline", "") or row.get("strategy_text", ""),
                #     label=row.get("label", "Weak"),
                #     model="qwen2.5:7b",
                #     temperature=0.2
                # )
                payload = generate_strategy_improvement_agentic(
                    store=store,
                    strategy_id=row["strategy_id"],
                    strategy_text=row.get("strategy_headline", ""),
                    label=row.get("label", "Weak"),
                    max_iters=5,
                    target_boosted=0.60,
                )

            st.json(payload)
            #for knowledge graph
            st.session_state.setdefault("improvements", {})
            st.session_state["improvements"][row["strategy_id"]] = payload
            
            iters = payload.get("evaluation", {}).get("iterations", [])
            if iters:
                st.write("Agent loop iterations:")
                st.table(iters)
            # ev = payload.get("evaluation", {})
            # if ev:
            #     st.metric("Before boosted", ev.get("before_boosted_similarity", 0.0))
            #     st.metric("After boosted", ev.get("after_boosted_similarity", 0.0))
            #     st.metric("Delta", ev.get("delta_boosted", 0.0))
            
            # Save per-strategy latest
            with open("outputs/improvements_latest.json", "w", encoding="utf-8") as f:
                json.dump(st.session_state.get("improvements", {}), f, ensure_ascii=False, indent=2)

            # Append history 
            with open("outputs/improvements_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": datetime.utcnow().isoformat(),
                    "strategy_id": row["strategy_id"],
                    "payload": payload
                }, ensure_ascii=False) + "\n")

    else:
        st.success("No improvement needed for this strategy (Strong/Partial).")

with tab_graph:
    sid = row["strategy_id"]
    s_head = row.get("strategy_headline", "") or row.get("strategy_text", "")

    # Collect top1-3 actions + their boosted scores
    actions = []
    for i in [1, 2, 3]:
        aid = row.get(f"top{i}_action_id", "") or ""
        atext = row.get(f"top{i}_action_headline", "") or ""
        aboost = row.get(f"top{i}_boosted", None)
        if aid:
            actions.append((i, aid, atext, aboost))

    # LLM payload (for KPIs + citations)
    payload = st.session_state.get("improvements", {}).get(sid)
    kpis = []
    cited_actions = set()

    if payload:
        sug0 = (payload.get("suggestions") or [{}])[0]
        kpis = sug0.get("kpis") or []
        cited_actions = set(sug0.get("citations") or [])  # e.g., ["A3.2"]

    # If LLM didn't provide citations, fall back to best_action_id
    if kpis and not cited_actions:
        cited_actions = {row.get("best_action_id", "")}

    # Build DOT (Graphviz)
    def esc(x: str) -> str:
        return (x or "").replace('"', "'")

    s_node = f'{sid}\\n{esc(s_head)[:70]}'
    dot = [
        'digraph G {',
        'rankdir=LR;',
        'splines=true;',
        'node [shape=box, style="rounded"];'
    ]

    # Strategy node
    dot.append(f'"{s_node}" [shape=box, style="rounded,filled", fillcolor="#1f77b4"];')

    # Action nodes + Strategy->Action edges with similarity labels
    action_nodes = {}  # aid -> node label
    for (rank, aid, atext, aboost) in actions:
        a_node = f'{aid}\\n{esc(atext)[:70]}'
        action_nodes[aid] = a_node

        dot.append(f'"{a_node}" [shape=box, style="rounded,filled", fillcolor="#ff7f0e"];')

        # Edge label: boosted similarity for that rank if available
        if aboost is not None and str(aboost) != "nan":
            try:
                lbl = f"boosted={float(aboost):.3f}"
            except:
                lbl = "boosted=?"
            dot.append(f'"{s_node}" -> "{a_node}" [label="{lbl}"];')
        else:
            dot.append(f'"{s_node}" -> "{a_node}";')

    # KPI nodes (LLM) + Action->KPI edges (only from cited actions)
    if kpis:
        for idx, k in enumerate(kpis, start=1):
            kname = esc(k.get("name", f"KPI {idx}"))
            metric = esc(k.get("metric", ""))
            target = esc(k.get("target", ""))
            timeframe = esc(k.get("timeframe", ""))

            k_node = (
                f'KPI (LLM): {kname}'
                f'\\nMetric: {metric}'
                f'\\nTarget: {target}'
                f'\\nTime: {timeframe}'
            )

            dot.append(f'"{k_node}" [shape=note, style="filled", fillcolor="#2ca02c"];')

            # Connect KPI only to cited action(s) that exist in top1-3.
            connected = False
            for aid in cited_actions:
                if aid in action_nodes:
                    dot.append(f'"{action_nodes[aid]}" -> "{k_node}";')
                    connected = True

            # If cited action not in top1-3, attach KPIs to best action if present
            if not connected and row.get("best_action_id", "") in action_nodes:
                dot.append(f'"{action_nodes[row.get("best_action_id","")]}" -> "{k_node}";')

    dot.append('}')
    st.graphviz_chart("\n".join(dot), use_container_width=True)

    if not payload:
        st.info("No LLM KPI data yet. Generate improvement suggestions to see KPI nodes.")


