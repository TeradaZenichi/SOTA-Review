"""6-topic_map.py

Cria o mapa de tópicos 2D (UMAP+KMeans) e salva clusters, coordenadas e resumos em results/planning/.

Input: merged_prioritized.csv (ou fallback)
Output: results/planning/topic_map.csv, topic_map_cluster_summary.csv, topic_map.png, topic_map.pdf
"""

from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import sys
import os

# Importar configuração de plot
plotconfig_path = Path(__file__).resolve().parents[2] / ".plot" / "plotconfig.py"
if plotconfig_path.exists():
    sys.path.insert(0, str(plotconfig_path.parent))
    import plotconfig

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "planning"
RESULTS_DIR = ROOT / "results" / "planning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Escolher arquivo de entrada
    for candidate in ["merged_prioritized.csv", "merged_raw_planning_dedup.csv", "merged_raw_planning.csv"]:
        in_path = DATA_DIR / candidate
        if in_path.exists():
            break
    else:
        print("Nenhum arquivo de entrada encontrado.")
        return

    df = pd.read_csv(in_path, dtype=str).fillna("")
    texts = df["Title"] + " " + df["Abstract"]
    texts = texts.tolist()

    # Embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)

    # Clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(embeddings)

    df["cluster"] = clusters
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    # Salvar CSV
    df.to_csv(RESULTS_DIR / "topic_map.csv", index=False)

    # Resumo de clusters
    vectorizer = CountVectorizer(stop_words='english', max_features=100)
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    summary = []
    for c in range(10):
        mask = df["cluster"] == c
        if mask.sum() == 0:
            continue
        cluster_texts = [texts[i] for i in range(len(texts)) if mask.iloc[i]]
        cluster_X = vectorizer.transform(cluster_texts)
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

    # Plot
    plt.figure(figsize=(plotconfig.fig_width_inch, plotconfig.fig_height_inch))
    scatter = plt.scatter(df["x"], df["y"], c=df["cluster"], cmap='tab10', alpha=0.7)
    for cluster_id, label in label_map.items():
        subset = df[df["cluster"] == cluster_id]
        if subset.empty:
            continue
        centroid = subset[["x", "y"]].mean()
        display_label = label.replace("Cluster ", "")
        plt.text(
            centroid["x"],
            centroid["y"],
            display_label,
            fontsize=7,
            ha='center',
            va='center',
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.6, "pad": 0.2},
        )
    plt.colorbar(scatter, label='Cluster')
    plt.title('Topic Map (UMAP + KMeans)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig(RESULTS_DIR / "topic_map.png")
    plt.savefig(RESULTS_DIR / "topic_map.pdf")
    plt.close()

    print(f"Mapa de tópicos gerado com {len(df)} pontos. Arquivos salvos em {RESULTS_DIR}")

if __name__ == "__main__":
    main()
