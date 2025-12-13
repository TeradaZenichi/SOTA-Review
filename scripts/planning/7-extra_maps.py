"""7-extra_maps.py

Gera mapas complementares: t-SNE, evolução temporal dos clusters, heatmap problema×método.

Input: merged_prioritized.csv (ou fallback)
Output: results/planning/tsne_map.csv/png/pdf, cluster_share_by_year.csv/png/pdf, problem_method_heatmap.csv/png/pdf
"""

from pathlib import Path
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import sys

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
    # Ler o topic_map.csv gerado pelo script 6
    topic_map_path = RESULTS_DIR / "topic_map.csv"
    if not topic_map_path.exists():
        print("Arquivo topic_map.csv não encontrado. Execute 6-topic_map.py primeiro.")
        return

    df = pd.read_csv(topic_map_path, dtype=str).fillna("")
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    summary_path = RESULTS_DIR / "topic_map_cluster_summary.csv"
    label_map: dict[int, str] = {}
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path, dtype=str)
        for _, row in summary_df.iterrows():
            try:
                cid = int(row["cluster"])
            except (ValueError, TypeError):
                continue
            label = row.get("label") or row.get("top_terms") or f"Cluster {cid}"
            label_map[cid] = label

    # t-SNE
    if "x" in df.columns and "y" in df.columns:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_coords = tsne.fit_transform(df[["x", "y"]].values)  # Usando coords UMAP como proxy
        df["tsne_x"] = tsne_coords[:, 0]
        df["tsne_y"] = tsne_coords[:, 1]

        plt.figure(figsize=(plotconfig.fig_width_inch, plotconfig.fig_height_inch))
        scatter = plt.scatter(df["tsne_x"], df["tsne_y"], c=df["cluster"], cmap='tab10', alpha=0.7)
        for cluster_id, label in label_map.items():
            subset = df[df["cluster"] == cluster_id]
            if subset.empty:
                continue
            centroid = subset[["tsne_x", "tsne_y"]].mean()
            plt.text(
                centroid["tsne_x"],
                centroid["tsne_y"],
                label.replace("Cluster ", ""),
                fontsize=7,
                ha='center',
                va='center',
                bbox={"boxstyle": "round", "fc": "white", "alpha": 0.6, "pad": 0.2},
            )
        plt.colorbar(scatter, label='Cluster')
        plt.title('t-SNE Map')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig(RESULTS_DIR / "tsne_map.png")
        plt.savefig(RESULTS_DIR / "tsne_map.pdf")
        plt.close()

        df.to_csv(RESULTS_DIR / "tsne_map.csv", index=False)

    # Evolução temporal dos clusters
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    cluster_year = df.groupby(["Year", "cluster"]).size().unstack(fill_value=0)
    cluster_year_pct = cluster_year.div(cluster_year.sum(axis=1), axis=0) * 100
    def _friendly_column(col):
        try:
            cid = int(col)
        except (ValueError, TypeError):
            return col
        return label_map.get(cid, f"Cluster {cid}")
    cluster_year_pct = cluster_year_pct.rename(columns=_friendly_column)

    cluster_year_pct.plot(kind='area', stacked=True, figsize=(plotconfig.fig_width_inch, plotconfig.fig_height_inch), colormap='tab10')
    plt.title('Cluster Share by Year')
    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cluster_share_by_year.png")
    plt.savefig(RESULTS_DIR / "cluster_share_by_year.pdf")
    plt.close()

    cluster_year.to_csv(RESULTS_DIR / "cluster_share_by_year.csv")

    # Heatmap problema x método (se colunas existirem)
    if "problems" in df.columns and "methods" in df.columns:
        # Simples contagem
        heatmap_data = {}
        for _, row in df.iterrows():
            probs = row["problems"].strip("[]").replace("'", "").split(", ") if row["problems"] else []
            meths = row["methods"].strip("[]").replace("'", "").split(", ") if row["methods"] else []
            for p in probs:
                for m in meths:
                    key = (p.strip(), m.strip())
                    heatmap_data[key] = heatmap_data.get(key, 0) + 1

        heatmap_df = pd.DataFrame([(p, m, c) for (p, m), c in heatmap_data.items()], columns=["problem", "method", "count"])
        pivot = heatmap_df.pivot(index="problem", columns="method", values="count").fillna(0)

        plt.figure(figsize=(plotconfig.fig_width_inch, plotconfig.fig_height_inch))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title('Problem vs Method Heatmap')
        plt.savefig(RESULTS_DIR / "problem_method_heatmap.png")
        plt.savefig(RESULTS_DIR / "problem_method_heatmap.pdf")
        plt.close()

        pivot.to_csv(RESULTS_DIR / "problem_method_heatmap.csv")

    print(f"Mapas complementares gerados. Arquivos salvos em {RESULTS_DIR}")

if __name__ == "__main__":
    main()
