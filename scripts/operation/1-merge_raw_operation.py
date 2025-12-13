"""1-merge_raw_operation.py

Agrupa todos os CSVs da pasta data/operation em um único arquivo.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "operation"
OUT_PATH = DATA_DIR / "merged_raw_operation.csv"


def main():
    csv_paths = sorted(DATA_DIR.rglob("*.csv"))
    csv_paths = [p for p in csv_paths if p.name not in ("merged_raw_operation.csv",)]
    if not csv_paths:
        print("Nenhum CSV de entrada encontrado em data/operation")
        return

    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
            df["source_file"] = path.name
            frames.append(df)
        except Exception as exc:
            print(f"Falha ao ler {path}: {exc}")

    if not frames:
        print("Não foi possível ler nenhum CSV válido.")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Merge concluído com {len(merged)} registros. Arquivo em {OUT_PATH}")


if __name__ == "__main__":
    main()
