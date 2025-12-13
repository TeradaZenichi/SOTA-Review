"""8-pub_trend.py

Conta publicações por ano e gera gráficos de tendência em results/planning/.

Input: merged_prioritized.csv (ou fallback)
Output: results/planning/year_counts.csv, year_counts.png, year_counts.pdf
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
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
    # Escolher arquivo de entrada
    for candidate in ["merged_prioritized.csv", "merged_raw_planning_dedup.csv", "merged_raw_planning.csv"]:
        in_path = DATA_DIR / candidate
        if in_path.exists():
            break
    else:
        print("Nenhum arquivo de entrada encontrado.")
        return

    df = pd.read_csv(in_path, dtype=str).fillna("")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    year_counts = df["Year"].value_counts().sort_index()

    # Salvar CSV
    year_counts.to_csv(RESULTS_DIR / "year_counts.csv", header=["count"])

    # Plot
    plt.figure(figsize=(plotconfig.fig_width_inch, plotconfig.fig_height_inch))
    year_counts.plot(kind='bar', color='skyblue')
    plt.title('Publications per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Publications')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "year_counts.png")
    plt.savefig(RESULTS_DIR / "year_counts.pdf")
    plt.close()

    print(f"Tendência de publicações gerada. Arquivos salvos em {RESULTS_DIR}")

if __name__ == "__main__":
    main()
