import math
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IN_PATH = BASE_DIR / "merged_dedup.csv"
OUT_PATH = BASE_DIR / "merged_prioritized.csv"

SOURCE_WEIGHT = {
    "scopus": 1,
    "web-of-science": 1,
    "ieee": 1,
    "sciencedirect": 1,
}


def parse_year(val) -> int:
    try:
        y = int(float(str(val).strip()))
        return y if 1900 <= y <= 2100 else 0
    except Exception:
        return 0


def score_row(row: pd.Series) -> int:
    score = 0
    tags = set(filter(None, str(row.get("keyword_tags", "")).lower().split(",")))
    if "planning" in tags:
        score += 3
    if "uc_ed" in tags:
        score += 3
    if "grid" in tags:
        score += 2
    if "ev" in tags:
        score += 2

    year = parse_year(row.get("year", ""))
    if year:
        # reward recency (0–8 points from 2018 onward)
        score += max(0, min(8, year - 2017))

    source = str(row.get("source_db", "")).lower()
    score += SOURCE_WEIGHT.get(source, 0)

    return score


def bucket(score: int) -> str:
    if score >= 10:
        return "high"
    if score >= 6:
        return "medium"
    return "low"


def main() -> None:
    if not IN_PATH.exists():
        print(f"Input not found: {IN_PATH}")
        return

    df = pd.read_csv(IN_PATH, dtype=str)
    if df.empty:
        print("Input is empty.")
        return

    # Filter out records with year < 2015 (or invalid year)
    df["year_int"] = df["year"].apply(parse_year)
    before = len(df)
    df = df[df["year_int"] >= 2015].copy()
    after = len(df)
    print(f"Filtered by year >= 2015: kept {after} of {before} (removed {before - after})")

    df["priority_score"] = df.apply(score_row, axis=1)
    df["priority_bucket"] = df["priority_score"].apply(bucket)

    df_sorted = df.sort_values(
        by=["priority_score", "year_int"],
        ascending=[False, False],
        kind="mergesort",
    )

    df_sorted = df_sorted.drop(columns=["year_int"])
    df_sorted.to_csv(OUT_PATH, index=False)

    total = len(df_sorted)
    counts = df_sorted["priority_bucket"].value_counts().to_dict()
    print(f"Saved prioritized file to {OUT_PATH} with {total} records.")
    print("Buckets:", counts)


if __name__ == "__main__":
    main()
