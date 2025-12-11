"""
Count publications per year for the planning corpus and produce CSV + plot.

Inputs: data/planning/merged_prioritized.csv (preferred) or merged_dedup.csv.
Outputs in results/planning/:
- year_counts.csv
- year_counts.png

Uses .plot/plotconfig.py for Gulliver font and figure sizing.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "planning"
RESULTS_DIR = ROOT / "results" / "planning"
PLOT_CONFIG = ROOT / ".plot" / "plotconfig.py"

DEFAULT_INPUTS = [DATA_DIR / "merged_prioritized.csv", DATA_DIR / "merged_dedup.csv"]
YEAR_MIN = 1900
YEAR_MAX = 2100


def to_int_year(val) -> int:
    try:
        y = int(float(str(val).strip()))
        return y if YEAR_MIN <= y <= YEAR_MAX else 0
    except Exception:
        return 0


def load_input() -> pd.DataFrame:
    for path in DEFAULT_INPUTS:
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            print(f"Loaded {len(df)} records from {path}")
            return df
    raise FileNotFoundError("No input file found (expected merged_prioritized.csv or merged_dedup.csv)")


def main() -> None:
    # Configure plotting (font Gulliver if available)
    if PLOT_CONFIG.exists():
        sys.path.insert(0, str(PLOT_CONFIG.parent))
        import plotconfig  # noqa: F401
    else:
        print("plotconfig.py not found; using matplotlib defaults")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_input()
    df["year_int"] = df["year"].apply(to_int_year)
    counts = (
        df[df["year_int"] > 0]
        .groupby("year_int")
        .size()
        .reset_index(name="count")
        .sort_values("year_int")
    )

    counts.to_csv(RESULTS_DIR / "year_counts.csv", index=False)
    print("Saved year_counts.csv with", len(counts), "rows")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(counts["year_int"], counts["count"], color="#4a90e2", edgecolor="#1f4e79")
    ax.set_xlabel("Year")
    ax.set_ylabel("Publications")
    ax.set_title("Planning corpus: publications per year")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Tight layout and save (PNG + PDF)
    fig.tight_layout()
    out_png = RESULTS_DIR / "year_counts.png"
    out_pdf = RESULTS_DIR / "year_counts.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved plot to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
