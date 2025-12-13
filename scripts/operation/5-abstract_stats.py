"""5-abstract_stats.py

Taggea cada artigo operação com termos de problema/método e salva painéis em results/operation/.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "operation"
RESULTS_DIR = ROOT / "results" / "operation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INPUTS = [
    DATA_DIR / "merged_prioritized_operation.csv",
    DATA_DIR / "merged_raw_operation_dedup.csv",
    DATA_DIR / "merged_raw_operation.csv",
]

PROBLEM_TERMS = {
    "operation": ["operation", "operational", "dispatch", "control", "real-time", "online"],
    "coordination": ["coordinate", "coordinated", "joint", "integrated", "coupled"],
    "resiliency": ["resilience", "stability", "reliability"],
}

METHOD_TERMS = {
    "mpc": ["model predictive", "mpc", "receding horizon", "rolling horizon"],
    "rule-based": ["rule-based", "heuristic", "threshold", "priority-based", "if-then"],
    "supervised": ["supervised learning", "machine learning", "neural network", "random forest", "svm", "lstm", "regression"],
    "rl": ["reinforcement learning", "reinforcement", "drl", "ddpg", "ppo", "sac", "actor-critic"],
}


def find_terms(text, terms_map):
    found = []
    text = text.lower()
    for label, candidates in terms_map.items():
        if any(term in text for term in candidates):
            found.append(label)
    return found


def main():
    for path in INPUTS:
        if path.exists():
            in_path = path
            break
    else:
        print("Nenhum arquivo de entrada encontrado em data/operation.")
        return

    df = pd.read_csv(in_path, dtype=str).fillna("")
    df["combined_text"] = (df.get("Title", "") + " " + df.get("Abstract", "")).fillna("")
    df["problems"] = df["combined_text"].apply(lambda text: find_terms(text, PROBLEM_TERMS))
    df["methods"] = df["combined_text"].apply(lambda text: find_terms(text, METHOD_TERMS))

    problem_counts = df["problems"].explode().value_counts().rename_axis("problem").reset_index(name="count")
    method_counts = df["methods"].explode().value_counts().rename_axis("method").reset_index(name="count")

    matrix = (
        df.explode("problems").explode("methods")["problems", "methods"].dropna()
    )
    matrix = matrix.groupby(["problems", "methods"]).size().reset_index(name="count")

    pd.DataFrame(problem_counts).to_csv(RESULTS_DIR / "problem_counts.csv", index=False)
    pd.DataFrame(method_counts).to_csv(RESULTS_DIR / "method_counts.csv", index=False)
    matrix.to_csv(RESULTS_DIR / "problem_method_matrix.csv", index=False)

    df["Year"] = pd.to_numeric(df.get("Year", ""), errors="coerce")
    trend = (
        df.explode("problems")["problems", "Year"].dropna()
        .groupby(["problems", "Year"]).size()
        .reset_index(name="count")
    )
    trend.to_csv(RESULTS_DIR / "term_trend_year.csv", index=False)

    print(f"Abstract stats concluídas com {len(df)} registros. Resultados em {RESULTS_DIR}")


if __name__ == "__main__":
    main()
