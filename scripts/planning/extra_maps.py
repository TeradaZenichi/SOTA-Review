"""
Additional maps for the planning corpus:
- t-SNE map: alternative 2D projection of text embeddings (title + abstract) with KMeans clusters.
- Cluster share over time: stacked area showing the proportion of each cluster by publication year.
- Problem x Method heatmap: co-occurrence counts of predefined problem/method terms in the text.

Inputs: data/planning/merged_prioritized.csv (preferred) or merged_dedup.csv.
Outputs (results/planning/):
- tsne_map.csv, tsne_map.png, tsne_map.pdf
- cluster_share_by_year.csv, cluster_share_by_year.png, cluster_share_by_year.pdf
- problem_method_heatmap.csv, problem_method_heatmap.png, problem_method_heatmap.pdf

Dependencies: sentence-transformers, scikit-learn, matplotlib
Optional: .plot/plotconfig.py for font/figure defaults (Gulliver).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "planning"
RESULTS_DIR = ROOT / "results" / "planning"
PLOT_CONFIG = ROOT / ".plot" / "plotconfig.py"
DEFAULT_INPUTS = [DATA_DIR / "merged_prioritized.csv", DATA_DIR / "merged_dedup.csv"]

MODEL_NAME = "all-MiniLM-L6-v2"
N_CLUSTERS = 12
RANDOM_STATE = 42
TOP_TERMS_PER_CLUSTER = 3

# Simple keyword sets reused from abstract_stats
PROBLEM_TERMS: Dict[str, List[str]] = {
    "expansion": ["expansion planning", "capacity expansion", "generation expansion"],
    "siting": ["siting", "siting and sizing", "sizing"],
    "uc": ["unit commitment", "generation scheduling", "economic dispatch"],
    "opf": ["optimal power flow", "opf"],
}

METHOD_TERMS: Dict[str, List[str]] = {
    "robust": ["robust optimization"],
    "stochastic": ["stochastic programming", "chance-constrained", "scenario"],
    "milp": ["milp", "mixed-integer linear", "mixed integer linear"],
    "metaheuristic": [
        "genetic algorithm",
        "particle swarm",
        "pso",
        "differential evolution",
        "simulated annealing",
        "metaheuristic",
    ],
    "ml": ["machine learning", "neural", "svm", "random forest"],
    "rl": ["reinforcement learning", "deep reinforcement", "q-learning"],
    "mpc": ["model predictive control", "mpc"],
}

YEAR_MIN = 1900
YEAR_MAX = 2100


def load_input() -> pd.DataFrame:
    for path in DEFAULT_INPUTS:
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            print(f"Loaded {len(df)} records from {path}")
            return df
    raise FileNotFoundError("No input file found (expected merged_prioritized.csv or merged_dedup.csv)")


def to_int_year(val) -> int:
    try:
        y = int(float(str(val).strip()))
        return y if YEAR_MIN <= y <= YEAR_MAX else 0
    except Exception:
        return 0


def contains_any(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def tsne_map(df: pd.DataFrame, embeddings: np.ndarray, clusters: np.ndarray, cluster_terms: Dict[int, str]) -> None:
    reducer = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, metric="cosine", init="pca")
    coords = reducer.fit_transform(embeddings)

    df_out = df.copy()
    df_out["cluster"] = clusters
    df_out["x"] = coords[:, 0]
    df_out["y"] = coords[:, 1]

    out_csv = RESULTS_DIR / "tsne_map.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with t-SNE coordinates")

    # Plot with cluster annotations for interpretability
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(df_out["x"], df_out["y"], c=df_out["cluster"], cmap="tab20", s=12, alpha=0.8, edgecolor="none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Planning corpus: t-SNE map (embeddings + KMeans)")
    ax.grid(False)

    # Annotate clusters at their centroids with top terms
    centroids = df_out.groupby("cluster")[ ["x", "y"] ].mean()
    for cid, row in centroids.iterrows():
        label = f"{cid}: {cluster_terms.get(cid, '')}"
        ax.text(row["x"], row["y"], label, fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

    # Colorbar to map colors to cluster IDs
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cluster ID")

    fig.text(0.5, 0.02, "Position comes from t-SNE on title+abstract embeddings; axes have no direct meaning.",
             ha="center", fontsize=8)
    fig.tight_layout()
    out_png = RESULTS_DIR / "tsne_map.png"
    out_pdf = RESULTS_DIR / "tsne_map.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved plot to {out_png} and {out_pdf}")


def cluster_share_over_time(df: pd.DataFrame, clusters: np.ndarray, cluster_terms: Dict[int, str]) -> None:
    df_tmp = df.copy()
    df_tmp["cluster"] = clusters
    df_tmp["year_int"] = df_tmp["year"].apply(to_int_year)
    df_tmp = df_tmp[df_tmp["year_int"] > 0]

    counts = df_tmp.groupby(["year_int", "cluster"]).size().reset_index(name="count")
    total_by_year = counts.groupby("year_int")["count"].transform("sum")
    counts["share"] = counts["count"] / total_by_year

    pivot = counts.pivot(index="year_int", columns="cluster", values="share").fillna(0)
    pivot = pivot.sort_index()
    out_csv = RESULTS_DIR / "cluster_share_by_year.csv"
    pivot.to_csv(out_csv)
    print(f"Saved {out_csv} with cluster share per year")

    # Plot stacked area
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(pivot.shape[1])]
    labels = [f"{cid}: {cluster_terms.get(cid, '')}" for cid in pivot.columns]
    ax.stackplot(pivot.index, pivot.T.values, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cluster share")
    ax.set_title("Planning corpus: cluster share over time")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    fig.text(0.5, 0.02, "Share = cluster count / total papers that year. Legend shows cluster ID and its top terms.",
             ha="center", fontsize=8)
    fig.tight_layout()
    out_png = RESULTS_DIR / "cluster_share_by_year.png"
    out_pdf = RESULTS_DIR / "cluster_share_by_year.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved plot to {out_png} and {out_pdf}")


def problem_method_heatmap(df: pd.DataFrame) -> None:
    df_tmp = df.copy()
    df_tmp["text"] = (df_tmp["title"].fillna("") + " " + df_tmp["abstract"].fillna("")).str.lower()

    for p, terms in PROBLEM_TERMS.items():
        df_tmp[f"problem_{p}"] = df_tmp["text"].apply(lambda x: contains_any(x, terms))
    for m, terms in METHOD_TERMS.items():
        df_tmp[f"method_{m}"] = df_tmp["text"].apply(lambda x: contains_any(x, terms))

    rows = []
    for p in PROBLEM_TERMS:
        for m in METHOD_TERMS:
            mask = df_tmp[f"problem_{p}"] & df_tmp[f"method_{m}"]
            rows.append({"problem": p, "method": m, "count": int(mask.sum())})
    matrix_df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "problem_method_heatmap.csv"
    matrix_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with problem-method co-occurrence counts")

    # Plot heatmap
    pivot = matrix_df.pivot(index="problem", columns="method", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(pivot.values, cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Problem × Method co-occurrence")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")
    fig.text(0.5, 0.02, "Counts of papers whose title+abstract mention the problem AND the method (keyword search).",
             ha="center", fontsize=8)
    fig.tight_layout()
    out_png = RESULTS_DIR / "problem_method_heatmap.png"
    out_pdf = RESULTS_DIR / "problem_method_heatmap.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved plot to {out_png} and {out_pdf}")


def main() -> None:
    # Optional plot config
    if PLOT_CONFIG.exists():
        sys.path.insert(0, str(PLOT_CONFIG.parent))
        import plotconfig  # noqa: F401
    else:
        print("plotconfig.py not found; using matplotlib defaults")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_input()
    df = df.fillna("")
    df["text"] = (df["title"] + " " + df["abstract"]).str.strip()
    df = df[df["text"] != ""].copy()
    print(f"Using {len(df)} records with non-empty text")

    # Embeddings
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True)

    # Clusters
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    clusters = km.fit_predict(embeddings)

    # Top terms per cluster (for reference, not plotted)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(df["text"])
    vocab = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for cid in sorted(set(clusters)):
        mask = clusters == cid
        term_freq = X[mask].sum(axis=0).A1
        top_idx = term_freq.argsort()[::-1][:TOP_TERMS_PER_CLUSTER]
        top_terms = [vocab[i] for i in top_idx if term_freq[i] > 0]
        cluster_terms[cid] = ", ".join(top_terms)
    summary_rows = [{"cluster": cid, "top_terms": cluster_terms[cid]} for cid in sorted(cluster_terms)]
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "tsne_cluster_terms.csv", index=False)

    # Generate maps
    tsne_map(df, embeddings, clusters, cluster_terms)
    cluster_share_over_time(df, clusters, cluster_terms)
    problem_method_heatmap(df)


if __name__ == "__main__":
    main()
