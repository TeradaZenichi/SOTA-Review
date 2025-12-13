"""5-abstract_stats.py

Taggea cada artigo com problemas/métodos (por palavras-chave em título+abstract). Salva contagens e matrizes em results/planning/.

Input: merged_prioritized.csv (ou fallback para merged_raw_planning_dedup.csv ou merged_raw_planning.csv)
Output: results/planning/problem_counts.csv, method_counts.csv, problem_method_matrix.csv, term_trend_year.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "planning"
RESULTS_DIR = ROOT / "results" / "planning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Palavras-chave para problemas
PROBLEM_TERMS = {
    "planning": ["planning", "allocation", "siting", "sizing", "capacity", "expansion"],
    "uc_ed": ["unit commitment", "uc", "generation scheduling", "economic dispatch"],
    "grid": ["power system", "power grid", "electric grid", "distribution network", "transmission"],
    "ev": ["electric vehicle", "ev", "plug-in", "charging station", "evcs"],
}

# Palavras-chave para métodos
METHOD_TERMS = {
    "stochastic": ["stochastic", "uncertainty", "probabilistic", "monte carlo"],
    "robust": ["robust", "worst-case", "min-max"],
    "meta-heuristics": ["genetic algorithm", "ga", "particle swarm", "pso", "ant colony", "aco", "simulated annealing", "sa", "tabu search", "ts"],
    "ml": ["machine learning", "ml", "neural network", "deep learning", "reinforcement learning", "rl", "svm", "random forest", "decision tree"],
}

def find_terms(text, term_dict):
    found = []
    for category, terms in term_dict.items():
        if any(term in text.lower() for term in terms):
            found.append(category)
    return found

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
    df["problems"] = df.apply(lambda r: find_terms(f"{r['Title']} {r['Abstract']}", PROBLEM_TERMS), axis=1)
    df["methods"] = df.apply(lambda r: find_terms(f"{r['Title']} {r['Abstract']}", METHOD_TERMS), axis=1)

    # Contagens de problemas
    problem_counts = {}
    for probs in df["problems"]:
        for p in probs:
            problem_counts[p] = problem_counts.get(p, 0) + 1
    pd.DataFrame(list(problem_counts.items()), columns=["problem", "count"]).to_csv(RESULTS_DIR / "problem_counts.csv", index=False)

    # Contagens de métodos
    method_counts = {}
    for meths in df["methods"]:
        for m in meths:
            method_counts[m] = method_counts.get(m, 0) + 1
    pd.DataFrame(list(method_counts.items()), columns=["method", "count"]).to_csv(RESULTS_DIR / "method_counts.csv", index=False)

    # Matriz problema x método
    matrix = {}
    for _, row in df.iterrows():
        for p in row["problems"]:
            for m in row["methods"]:
                key = (p, m)
                matrix[key] = matrix.get(key, 0) + 1
    matrix_df = pd.DataFrame([(p, m, c) for (p, m), c in matrix.items()], columns=["problem", "method", "count"])
    matrix_df.to_csv(RESULTS_DIR / "problem_method_matrix.csv", index=False)

    # Tendência por ano
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    trend = {}
    for _, row in df.iterrows():
        year = row["Year"]
        if pd.isna(year):
            continue
        year = int(year)
        for p in row["problems"]:
            key = (p, year)
            trend[key] = trend.get(key, 0) + 1
        for m in row["methods"]:
            key = (m, year)
            trend[key] = trend.get(key, 0) + 1
    trend_df = pd.DataFrame([(term, year, count) for (term, year), count in trend.items()], columns=["term", "year", "count"])
    trend_df.to_csv(RESULTS_DIR / "term_trend_year.csv", index=False)

    print(f"Processado {len(df)} registros. Arquivos salvos em {RESULTS_DIR}")

if __name__ == "__main__":
    main()
