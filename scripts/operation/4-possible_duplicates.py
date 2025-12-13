"""4-possible_duplicates.py

Gera relatório de possíveis duplicatas para revisão manual.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "operation"
INPUT = DATA_DIR / "merged_raw_operation_dedup.csv"
OUTPUT = ROOT / "results" / "operation" / "possible_duplicates.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    if not INPUT.exists():
        print(f"Arquivo {INPUT} não encontrado. Rode o script de dedup primeiro.")
        return

    df = pd.read_csv(INPUT, dtype=str).fillna("")
    df["title_norm"] = df.get("Title", "").str.lower().str.strip()
    df["year"] = df.get("Year", "").str.extract(r"(\d{4})").fillna("")
    df["key"] = df["title_norm"] + "|" + df["year"]

    candidates = (
        df[df.duplicated("key", keep=False)]
        .sort_values(["key", "priority_score"], ascending=[True, False])
    )

    if candidates.empty:
        print("Nenhuma duplicata detectada além daquelas removidas.")
        return

    candidates.to_csv(OUTPUT, index=False)
    print(f"Possíveis duplicatas salvas em {OUTPUT} ({len(candidates)} linhas).")


if __name__ == "__main__":
    main()
