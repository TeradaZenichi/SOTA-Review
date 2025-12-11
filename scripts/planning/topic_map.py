"""
Generate a 2D topic map ("bolinhas") for the planning corpus using embeddings,
KMeans clusters, and UMAP projection. Saves CSV with coordinates and clusters,
and PNG/PDF scatter plot.

Inputs: data/planning/merged_prioritized.csv (or merged_dedup.csv fallback)
Outputs: results/planning/topic_map.csv, topic_map.png, topic_map.pdf

Dependencies (install before running):
  pip install sentence-transformers umap-learn scikit-learn matplotlib
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import umap

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "planning"
RESULTS_DIR = ROOT / "results" / "planning"
PLOT_CONFIG = ROOT / ".plot" / "plotconfig.py"
DEFAULT_INPUTS = [DATA_DIR / "merged_prioritized.csv", DATA_DIR / "merged_dedup.csv"]

MODEL_NAME = "all-MiniLM-L6-v2"
N_CLUSTERS = 12
RANDOM_STATE = 42
TOP_TERMS_PER_CLUSTER = 3


def load_input() -> pd.DataFrame:
    for path in DEFAULT_INPUTS:
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            print(f"Loaded {len(df)} records from {path}")
            return df
    raise FileNotFoundError("No input file found (expected merged_prioritized.csv or merged_dedup.csv)")


def main() -> None:
    # Font/plot config
    if PLOT_CONFIG.exists():
        sys.path.insert(0, str(PLOT_CONFIG.parent))
        import plotconfig  # noqa: F401
    else:
        print("plotconfig.py not found; using matplotlib defaults")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_input()
    df["text"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.strip()
    df = df[df["text"] != ""].copy()
    print(f"Using {len(df)} records with non-empty text")

    # Embeddings
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True)

    # Cluster
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    clusters = km.fit_predict(embeddings)

    # UMAP projection
    reducer = umap.UMAP(random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1, metric="cosine")
    coords = reducer.fit_transform(embeddings)

    df_out = df.copy()
    df_out["cluster"] = clusters
    df_out["x"] = coords[:, 0]
    df_out["y"] = coords[:, 1]

    # Top terms per cluster (CountVectorizer on text)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(df_out["text"].fillna(""))
    vocab = vectorizer.get_feature_names_out()
    cluster_terms = {}
    cluster_centroids = {}
    cluster_counts = {}
    for cid in sorted(df_out["cluster"].unique()):
        mask = (df_out["cluster"] == cid).to_numpy()
        cluster_counts[cid] = int(mask.sum())
        cluster_centroids[cid] = (df_out.loc[mask, "x"].mean(), df_out.loc[mask, "y"].mean())
        sub = X[mask]
        term_freq = sub.sum(axis=0).A1
        top_idx = term_freq.argsort()[::-1][:TOP_TERMS_PER_CLUSTER]
        top_terms = [vocab[i] for i in top_idx if term_freq[i] > 0]
        cluster_terms[cid] = ", ".join(top_terms)

    out_csv = RESULTS_DIR / "topic_map.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with coordinates and clusters")

    # Cluster summary CSV
    summary_rows = []
    for cid in sorted(cluster_terms.keys()):
        summary_rows.append({
            "cluster": cid,
            "count": cluster_counts.get(cid, 0),
            "top_terms": cluster_terms[cid],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "topic_map_cluster_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv} with counts and top terms per cluster")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        df_out["x"],
        df_out["y"],
        c=df_out["cluster"],
        cmap="tab20",
        s=12,
        alpha=0.8,
        edgecolor="none",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Planning corpus: topic map (UMAP + KMeans)")
    ax.grid(False)

    # Annotate clusters with top terms
    for cid, (cx, cy) in cluster_centroids.items():
        label = f"{cid}: {cluster_terms.get(cid, '')}"
        ax.text(cx, cy, label, fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

    # Legend with cluster colors and top terms for quick reference
    handles = []
    for cid in sorted(cluster_centroids.keys()):
        color = scatter.cmap(scatter.norm(cid))
        handles.append(mpatches.Patch(color=color, label=f"{cid}: {cluster_terms.get(cid, '')}"))
    ax.legend(handles=handles, title="Cluster: top terms", fontsize=7, title_fontsize=8,
              loc="upper right", bbox_to_anchor=(1.35, 1))

    # Caption to clarify interpretation
    fig.text(0.5, 0.02, "Positions come from UMAP on title+abstract embeddings; axes have no direct meaning. Colors/legend map clusters to top terms.",
             ha="center", fontsize=8)

    fig.tight_layout()
    out_png = RESULTS_DIR / "topic_map.png"
    out_pdf = RESULTS_DIR / "topic_map.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved plot to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
