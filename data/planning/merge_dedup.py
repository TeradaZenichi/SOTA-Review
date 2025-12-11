import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "merged_dedup.csv"

# Simple keyword groups for quick tagging
KEYWORDS_INCLUDE = {
    "planning": [
        "planning",
        "expansion",
        "siting",
        "sizing",
        "capacity",
        "sitio",
    ],
    "uc_ed": ["unit commitment", "uc", "generation scheduling", "economic dispatch"],
    "grid": ["power system", "power grid", "electric grid", "distribution network", "transmission"],
    "ev": ["electric vehicle", "ev", "plug-in"],
}

# Columns we will try to normalize
CANDIDATE_TITLE_COLS = [
    "title",
    "document title",
    "dc:title",
    "primarytitle",
    "article title",
    "ti",
]
CANDIDATE_ABS_COLS = ["abstract", "description", "abstracttext", "ab"]
CANDIDATE_DOI_COLS = ["doi", "dc:identifier", "articledoi", "eissn"]
CANDIDATE_YEAR_COLS = ["year", "publication year", "py", "pubyear"]


def normalize_text(val: Any) -> str:
    if pd.isna(val):
        return ""
    txt = str(val)
    return re.sub(r"\s+", " ", txt).strip()


def pick_first(cols: List[str], row: pd.Series) -> str:
    for c in cols:
        val = row.get(c, None)
        if isinstance(val, pd.Series):
            # handle duplicate column names by taking the first non-null
            val = val.dropna().iloc[0] if not val.dropna().empty else None
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            if pd.notna(val):
                return normalize_text(val)
    return ""


def make_key(doi: str, title: str, year: str) -> str:
    if doi:
        return doi.lower().strip()
    base = re.sub(r"[^a-z0-9]", "", title.lower())
    return f"title_{base}_{year}"


def tag_keywords(title: str, abstract: str) -> Tuple[str, List[str]]:
    text = f"{title} {abstract}".lower()
    hits = []
    for group, kws in KEYWORDS_INCLUDE.items():
        for kw in kws:
            if kw in text:
                hits.append(group)
                break
    status = "maybe" if hits else "needs_screen"
    return status, hits


def load_csv(path: Path, source: str) -> pd.DataFrame:
    # Try comma, then tab
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, engine="python")
            break
        except Exception:
            df = None
    if df is None:
        return pd.DataFrame()
    df["source_db"] = source
    df["original_file"] = path.name
    return df


def load_bibtex(path: Path, source: str) -> pd.DataFrame:
    try:
        import bibtexparser
    except ImportError:
        print(f"bibtexparser not installed; skipping {path.name}")
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        bib_db = bibtexparser.load(f)
    records = []
    for entry in bib_db.entries:
        records.append({
            "title": entry.get("title", ""),
            "abstract": entry.get("abstract", ""),
            "doi": entry.get("doi", ""),
            "year": entry.get("year", ""),
            "source_db": source,
            "original_file": path.name,
        })
    return pd.DataFrame(records)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Lowercase copy for easier selection
    lower_df = df.rename(columns=str.lower)
    lower_df["title_norm"] = lower_df.apply(lambda r: pick_first(CANDIDATE_TITLE_COLS, r), axis=1)
    lower_df["abstract_norm"] = lower_df.apply(lambda r: pick_first(CANDIDATE_ABS_COLS, r), axis=1)
    lower_df["doi_norm"] = lower_df.apply(lambda r: pick_first(CANDIDATE_DOI_COLS, r), axis=1)
    lower_df["year_norm"] = lower_df.apply(lambda r: pick_first(CANDIDATE_YEAR_COLS, r), axis=1)

    def build_row(r: pd.Series) -> Dict[str, Any]:
        title = normalize_text(r.get("title_norm", ""))
        abstract = normalize_text(r.get("abstract_norm", ""))
        doi = normalize_text(r.get("doi_norm", ""))
        year = normalize_text(r.get("year_norm", ""))
        status, hits = tag_keywords(title, abstract)
        return {
            "title": title,
            "abstract": abstract,
            "doi": doi,
            "year": year,
            "status": status,
            "keyword_tags": ",".join(hits),
            "source_db": r.get("source_db", ""),
            "original_file": r.get("original_file", ""),
        }

    norm_rows = [build_row(r) for _, r in lower_df.iterrows()]
    norm_df = pd.DataFrame(norm_rows)
    norm_df["dedup_key"] = norm_df.apply(lambda r: make_key(r["doi"], r["title"], r["year"]), axis=1)
    return norm_df


def main() -> None:
    sources = []
    # CSV files
    for path in BASE_DIR.glob("**/*.csv"):
        source = path.parent.name  # e.g., scopus, ieee
        sources.append(load_csv(path, source))
    # BibTeX files
    for path in BASE_DIR.glob("**/*.bib"):
        source = path.parent.name
        sources.append(load_bibtex(path, source))

    if not sources:
        print("No sources found.")
        return

    raw_df = pd.concat(sources, ignore_index=True, sort=False)
    raw_len = len(raw_df)
    print(f"Loaded raw records: {raw_len}")

    norm_df = normalize_frame(raw_df)
    if norm_df.empty:
        print("No normalized records.")
        return

    # Drop empties
    norm_df = norm_df[norm_df["title"] != ""].copy()

    # Dedup
    dedup_df = norm_df.sort_values(["dedup_key", "year"], ascending=[True, False])
    dedup_df = dedup_df.drop_duplicates(subset=["dedup_key"], keep="first")
    dedup_len = len(dedup_df)
    print(f"After dedup: {dedup_len} (removed {raw_len - dedup_len})")

    # Add placeholders for manual screening
    dedup_df["reason_exclusion"] = ""

    # Reorder columns
    cols = [
        "status",
        "reason_exclusion",
        "title",
        "abstract",
        "doi",
        "year",
        "keyword_tags",
        "source_db",
        "original_file",
        "dedup_key",
    ]
    dedup_df = dedup_df[cols]

    dedup_df.to_csv(OUT_PATH, index=False)
    print(f"Saved deduplicated file to {OUT_PATH} with {len(dedup_df)} records.")


if __name__ == "__main__":
    main()
