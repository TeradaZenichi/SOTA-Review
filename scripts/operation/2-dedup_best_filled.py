"""2-dedup_best_filled.py

Remove duplicatas do arquivo merged_raw_operation.csv, mantendo o registro com mais campos preenchidos.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "operation"
INPUT = DATA_DIR / "merged_raw_operation.csv"
OUTPUT = DATA_DIR / "merged_raw_operation_dedup.csv"


def richness_score(row):
    return row.astype(bool).sum()


def main():
    if not INPUT.exists():
        print(f"Arquivo {INPUT} não encontrado. Rode o script de merge primeiro.")
        return

    df = pd.read_csv(INPUT, dtype=str).fillna("")
    df["DOI_norm"] = df.get("DOI", "").str.lower().str.strip()
    df["Title_norm"] = df.get("Title", "").str.lower().str.strip()
    df["Year_norm"] = df.get("Year", "").str.extract(r"(\d{4})").fillna("")

    df["dedup_key"] = df.apply(
        lambda r: r["DOI_norm"] or f"{r['Title_norm']}|{r['Year_norm']}", axis=1
    )

    best = (
        df.sort_values("dedup_key")
        .groupby("dedup_key", as_index=False)
        .apply(lambda group: group.loc[group.apply(richness_score, axis=1).idxmax()])
        .reset_index(drop=True)
    )

    best.drop(columns=["DOI_norm", "Title_norm", "Year_norm"], inplace=True)
    best.to_csv(OUTPUT, index=False)
    print(f"Deduplicado final com {len(best)} registros. Arquivo em {OUTPUT}")


if __name__ == "__main__":
    main()
