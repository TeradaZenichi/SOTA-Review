"""8-pub_trend.py

Gera contagem de publicações por ano para corpus de operação.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys

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
        print("Nenhum arquivo de entrada encontrado para gerar tendências.")
        return

    df = pd.read_csv(in_path, dtype=str).fillna("")
    df["Year"] = pd.to_numeric(df.get("Year", ""), errors="coerce")
    year_counts = df["Year"].value_counts().sort_index()
    year_counts.index = year_counts.index.astype(int)

    year_counts.to_csv(RESULTS_DIR / "year_counts.csv", header=["count"])

    plt.figure(figsize=(plotconfig.fig_width_inch, plotconfig.fig_height_inch))
    year_counts.plot(kind="bar", color="skyblue")
    plt.title("Publicações por Ano (Operação)")
    plt.xlabel("Year")
    plt.ylabel("Number of Publications")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "year_counts.png")
    plt.savefig(RESULTS_DIR / "year_counts.pdf")
    plt.close()

    print(f"Tendência anual gerada. Arquivos em {RESULTS_DIR}")


if __name__ == "__main__":
    main()
