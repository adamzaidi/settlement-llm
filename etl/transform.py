# What this does
# 1) Read the extracted CSV.
# 2) Clean up columns (types, missing values).
# 3) Build coarse + fine outcome labels from simple text rules (heuristics).
# 4) Save a clean CSV to data/processed/ for modeling + charts.
import logging
logger = logging.getLogger("pipeline")

import os
import numpy as np
import pandas as pd

RAW_PATH = "data/extracted/raw_data.csv"
OUT_PATH = "data/processed/processed_data.csv"

# Columns we want in front (we keep extras after these)
PRESERVE_ORDER = [
    "case_id","date_created","date_modified","opinion_year","page_count",
    "per_curiam","type","author_id","joined_by_count","case_name",
    "citation","court","jurisdiction_state","text_char_count","text_word_count",
    "label_heuristic","plain_text_present","plain_text_url","download_url",
    "cluster_url","court_url"
]

def _fill_unknown(series: pd.Series) -> pd.Series:
    """Replace NaN/blank strings with 'Unknown' (but keep valid values)."""
    s = series.copy()
    if s.dtype == object:
        s = s.replace(r"^\s*$", np.nan, regex=True)
    return s.fillna("Unknown")

def transform_data(raw_path: str = RAW_PATH, out_path: str = OUT_PATH) -> pd.DataFrame:
    """
    Clean, type-cast, and label the dataset. Writes processed CSV.
    """
    # STEP 1: Load raw
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.lower()

    # STEP 2: Ensure expected columns exist
    for col in PRESERVE_ORDER:
        if col not in df.columns:
            df[col] = np.nan

    # STEP 3: Fill basic string fields (only true-missing)
    str_cols = [
        "case_name","court","citation","jurisdiction_state",
        "label_heuristic","type","download_url","cluster_url","court_url","plain_text_url"
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = _fill_unknown(df[col])

    # STEP 4: Parse dates + numeric types
    for col in ["date_created","date_modified"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["opinion_year","page_count","author_id","joined_by_count","text_char_count","text_word_count","plain_text_present"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # STEP 5: Booleans from flags
    if "citation" in df.columns:
        df["has_citation"] = (df["citation"].str.lower() != "unknown").astype(int)
    if "court" in df.columns:
        df["has_court"] = (df["court"].str.lower() != "unknown").astype(int)
    if "per_curiam" in df.columns:
        df["is_per_curiam"] = df["per_curiam"].astype(bool).astype(int)

    # STEP 6: Build outcome labels
    # 6a) Coarse: Loss(0) / Win(1) / Settlement(2)
    # 6b) Fine: Loss(0) / Win(1) / Mixed(2) / Partial(3) / Settlement(4) / Other(5)
    t = (df.get("label_heuristic", pd.Series(dtype=str)).astype(str).str.lower().str.strip())

    POS = r"\b(affirmed|granted|upheld|sustain(?:ed)?)\b"
    NEG = r"\b(reversed|denied|vacated|dismissed)\b"
    MIX = r"\b(affirmed.*reversed|reversed.*affirmed|mixed)\b"
    PAR1 = r"\bpartial(?:ly)?\s+(affirmed|reversed|granted)\b"
    PAR2 = r"\b(affirmed|reversed|granted)\s+in\s+part\b"
    SETL = r"\bsettlement\b"

    # Booleans for each pattern (regex=True: we want regex behavior)
    has_pos  = t.str.contains(POS,  regex=True, na=False)
    has_neg  = t.str.contains(NEG,  regex=True, na=False)
    has_mix  = t.str.contains(MIX,  regex=True, na=False)
    has_par  = t.str.contains(PAR1, regex=True, na=False) | t.str.contains(PAR2, regex=True, na=False)
    has_setl = t.str.contains(SETL, regex=True, na=False)

    # Coarse (0/1/2)
    coarse = pd.Series(np.nan, index=df.index)
    coarse[has_setl] = 2
    coarse[has_pos & ~has_setl] = 1
    coarse[has_neg & ~has_setl] = 0
    coarse = coarse.fillna(1)  # default to Win if unknown
    df["outcome_code"] = coarse.astype(int)

    # Fine (0..5)
    fine = pd.Series(np.nan, index=df.index)
    fine[has_setl] = 4             # Settlement
    fine[has_mix & ~has_setl] = 2  # Mixed
    fine[has_par & ~has_setl] = 3  # Partial
    fine[has_pos & ~has_setl & ~has_mix & ~has_par] = 1  # Win
    fine[has_neg & ~has_setl & ~has_mix & ~has_par] = 0  # Loss
    fine = fine.fillna(5)          # Other
    df["outcome_code_fine"] = fine.astype(int)

    # Optional human-readable fine label (nice for EDA)
    label_names = {0: "Loss", 1: "Win", 2: "Mixed", 3: "Partial", 4: "Settlement", 5: "Other"}
    df["outcome_label_fine"] = df["outcome_code_fine"].map(label_names)

    # STEP 7: Order columns and write out
    ordered = [c for c in PRESERVE_ORDER if c in df.columns]
    tail = [c for c in df.columns if c not in ordered]
    df = df[ordered + tail]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    # STEP 8: Small report (printed + in logs)
    nonempty_words = (df.get("text_word_count", pd.Series(dtype=float)) > 0).mean() if "text_word_count" in df.columns else 0
    print(
        f"Processed {len(df)} cases → {out_path}\n"
        f"  • non-empty court: {(df['has_court'].mean()*100 if 'has_court' in df.columns else 0):.1f}%\n"
        f"  • non-empty citation: {(df['has_citation'].mean()*100 if 'has_citation' in df.columns else 0):.1f}%\n"
        f"  • text present: {(df.get('plain_text_present', pd.Series(dtype=float)).mean()*100 if 'plain_text_present' in df.columns else 0):.1f}%\n"
        f"  • >0 text words: {nonempty_words*100:.1f}%"
    )

    # Log distributions (helps debug labels)
    try:
        logger.info("Outcome_code distribution after transform:\n%s", df["outcome_code"].value_counts())
        logger.info("Fine labels:\n%s", df["outcome_label_fine"].value_counts())
        logger.info("Fine codes:\n%s", df["outcome_code_fine"].value_counts())
    except Exception:
        pass

    return df


if __name__ == "__main__":
    transform_data()