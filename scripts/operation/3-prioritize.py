"""3-prioritize.py

Gera score de prioridade para os registros deduplicados (foco em operação recente) e bucketiza.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "operation"
INPUT = DATA_DIR / "merged_raw_operation_dedup.csv"
OUTPUT = DATA_DIR / "merged_prioritized_operation.csv"


def score_record(row):
    year = pd.to_numeric(row.get("Year", ""), errors="coerce")
    score = 0.0
    if not np.isnan(year):
        score += min(max(year - 2015, 0), 10) / 10
    text = f"{row.get('Title', '')} {row.get('Abstract', '')}".lower()
    if "real-time" in text or "control" in text or "dispatch" in text:
        score += 0.5
    if "reinforcement learning" in text or "model predictive" in text:
        score += 0.3
    return min(score, 1.0)


def bucket(score):
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def main():
    if not INPUT.exists():
        print(f"Arquivo {INPUT} não encontrado. Rode o script de dedup primeiro.")
        return

    df = pd.read_csv(INPUT, dtype=str).fillna("")
    df["priority_score"] = df.apply(score_record, axis=1)
    df["priority_bucket"] = df["priority_score"].apply(bucket)
    df.to_csv(OUTPUT, index=False)
    print(f"Pontuação e buckets escritos em {OUTPUT}")


if __name__ == "__main__":
    main()
