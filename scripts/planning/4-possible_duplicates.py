"""Gera um CSV com possíveis duplicatas por título normalizado e ano.

Input:  data/planning/merged_raw_planning_dedup.csv
Output: results/planning/possible_duplicates.csv

Uso:
    python scripts/planning/4-possible_duplicates.py
"""
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "planning" / "merged_raw_planning_dedup.csv"
OUT_PATH = ROOT / "results" / "planning" / "possible_duplicates.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def normalize_title(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9]", "", t)
    return t

def main():
    df = pd.read_csv(IN_PATH, dtype=str).fillna("")
    df["title_norm"] = df["Title"].apply(normalize_title)
    df["year_norm"] = df["Year"].astype(str)
    grouped = df.groupby(["title_norm", "year_norm"])
    dups = grouped.filter(lambda g: len(g) > 1)
    if dups.empty:
        print("Nenhum possível duplicado encontrado.")
        return
    dups = dups.sort_values(["title_norm", "year_norm"])
    dups.to_csv(OUT_PATH, index=False)
    print(f"Salvo possíveis duplicatas em {OUT_PATH} ({len(dups)} linhas)")

if __name__ == "__main__":
    main()
