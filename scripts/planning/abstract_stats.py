"""
Quick abstract-level stats for planning corpus.

Inputs: data/planning/merged_prioritized.csv (default) or merged_dedup.csv.
Outputs (written to results/planning/):
- problem_counts.csv
- method_counts.csv
- problem_method_matrix.csv
- term_trend_year.csv (per-term per-year counts)
Also prints a short console summary.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "planning"
RESULTS_DIR = ROOT / "results" / "planning"
DEFAULT_INPUTS = [DATA_DIR / "merged_prioritized.csv", DATA_DIR / "merged_dedup.csv"]

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

YEAR_MIN = 2015
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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_input()
    df["year_int"] = df["year"].apply(to_int_year)
    df["text"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.lower()

    # Detect tags
    for p, terms in PROBLEM_TERMS.items():
        df[f"problem_{p}"] = df["text"].apply(lambda x: contains_any(x, terms))
    for m, terms in METHOD_TERMS.items():
        df[f"method_{m}"] = df["text"].apply(lambda x: contains_any(x, terms))

    # Counts per problem/method
    problem_counts = {p: int(df[f"problem_{p}"].sum()) for p in PROBLEM_TERMS}
    method_counts = {m: int(df[f"method_{m}"].sum()) for m in METHOD_TERMS}

    # Problem x Method matrix
    rows = []
    for p in PROBLEM_TERMS:
        for m in METHOD_TERMS:
            mask = df[f"problem_{p}"] & df[f"method_{m}"]
            rows.append({"problem": p, "method": m, "count": int(mask.sum())})
    matrix_df = pd.DataFrame(rows)

    # Trend per year for selected terms (problem+method)
    term_trend_rows = []
    for term_label in list(PROBLEM_TERMS.keys()) + list(METHOD_TERMS.keys()):
        colname = f"problem_{term_label}" if term_label in PROBLEM_TERMS else f"method_{term_label}"
        tmp = df[df[colname]]
        trend = tmp.groupby("year_int").size().reset_index(name="count")
        trend = trend[trend["year_int"] > 0]
        for _, r in trend.iterrows():
            term_trend_rows.append({"term": term_label, "year": int(r.year_int), "count": int(r["count"])})
    term_trend_df = pd.DataFrame(term_trend_rows)

    # Save outputs
    pd.DataFrame([problem_counts]).to_csv(RESULTS_DIR / "problem_counts.csv", index=False)
    pd.DataFrame([method_counts]).to_csv(RESULTS_DIR / "method_counts.csv", index=False)
    matrix_df.to_csv(RESULTS_DIR / "problem_method_matrix.csv", index=False)
    term_trend_df.to_csv(RESULTS_DIR / "term_trend_year.csv", index=False)

    # Console summary
    print("Problem counts:", problem_counts)
    print("Method counts:", method_counts)
    print("Saved stats to", RESULTS_DIR)


if __name__ == "__main__":
    main()
