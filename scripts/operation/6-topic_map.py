"""6-topic_map.py

Cria mapa UMAP+KMeans para operação EV+BESS e salva no results/operation.
"""

from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import sys

# Importar configuração de plot
plotconfig_path = Path(__file__).resolve().parents[1] / ".plot" / "plotconfig.py"
if plotconfig_path.exists():
    sys.path.insert(0, str(plotconfig_path.parent))
    import plotconfig
else:
    plotconfig = type("cfg", (), {"fig_width_inch": 8.5, "fig_height_inch": 6})()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "operation"
RESULTS_DIR = ROOT / "results" / "operation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = [
    DATA_DIR / "merged_prioritized_operation.csv",
    DATA_DIR / "merged_raw_operation_dedup.csv",
    DATA_DIR / "merged_raw_operation.csv",
]


def main():
    for path in INPUTS:
        if path.exists():
            in_path = path
            break
    else:
        print("Nenhum arquivo de entrada encontrado em data/operation.")
        return

    df = pd.read_csv(in_path, dtype=str).fillna("")
    texts = (df.get("Title", "") + " " + df.get("Abstract", "")).tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)

    kmeans = KMeans(n_clusters=8, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(embeddings)

    df["cluster"] = clusters
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df.to_csv(RESULTS_DIR / "topic_map.csv", index=False)

    vectorizer = CountVectorizer(stop_words="english", max_features=100)
    vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    summary = []
    for c in range(kmeans.n_clusters):
        mask = df["cluster"] == c
        if mask.sum() == 0:
            continue
        texts_c = [texts[i] for i, flag in enumerate(mask) if flag]
        cluster_X = vectorizer.transform(texts_c)
        word_freq = cluster_X.sum(axis=0).A1
        top_indices = word_freq.argsort()[-3:][::-1]
        top_words = [terms[i] for i in top_indices]
        summary.append({"cluster": c, "count": mask.sum(), "top_terms": ", ".join(top_words)})

    summary_df = pd.DataFrame(summary)
    label_map = {
        int(row["cluster"]): f"Cluster {int(row['cluster'])}: {row['top_terms']}"
        for _, row in summary_df.iterrows()
    }
    summary_df["label"] = summary_df["cluster"].map(label_map)
    summary_df.to_csv(RESULTS_DIR / "topic_map_cluster_summary.csv", index=False)

    plt.figure(figsize=(plotconfig.fig_width_inch, plotconfig.fig_height_inch))
    scatter = plt.scatter(df["x"], df["y"], c=df["cluster"], cmap="tab10", alpha=0.7)
    for cluster_id, label in label_map.items():
        subset = df[df["cluster"] == cluster_id]
        if subset.empty:
            continue
        centroid = subset[["x", "y"]].mean()
        plt.text(
            centroid["x"],
            centroid["y"],
            label.replace("Cluster ", ""),
            fontsize=7,
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.6, "pad": 0.2},
        )
    plt.colorbar(scatter, label="Cluster")
    plt.title("Operation Topic Map (UMAP + KMeans)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "topic_map.png")
    plt.savefig(RESULTS_DIR / "topic_map.pdf")
    plt.close()

    print(f"Mapa de tópicos gerado com {len(df)} pontos. Arquivos em {RESULTS_DIR}")


if __name__ == "__main__":
    main()
