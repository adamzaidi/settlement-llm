# utils/validators.py
"""
Lightweight post-load validation utilities.

I use these checks to quickly confirm that my processed data
has the expected shape, labels, and quality. They are fast and
side-effect free, so they only log results for review.
"""

from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger("pipeline")

# Columns expected to always exist in the processed output
REQUIRED_COLS = [
    "case_id", "court", "jurisdiction_state", "citation",
    "outcome_code", "outcome_code_fine", "outcome_label_fine"
]

# Allowed values for outcome codes
ALLOWED_COARSE = {0, 1, 2}          # Loss/Win/Settlement
ALLOWED_FINE   = {0, 1, 2, 3, 4, 5} # Loss/Win/Mixed/Partial/Settlement/Other


def _pct_nonempty(series: pd.Series) -> float:
    """
    Calculate the share of values that are not empty, 'unknown', or 'nan'.

    Args:
        series: A pandas Series to check.

    Returns:
        Percentage of non-empty entries as a float (0–100).
    """
    s = series.astype(str).str.strip().str.lower()
    good = (s != "") & (s != "unknown") & (s != "nan")
    return float(good.mean()) * 100.0


def validate_processed(df: pd.DataFrame) -> None:
    """
    Run lightweight validation on the processed DataFrame.

    This function checks schema, row counts, non-empty shares, and
    outcome label domains. It never mutates the data—only logs results.

    Args:
        df: The processed dataframe returned by `load_data()`.

    Returns:
        None. All results are written to the logger.
    """
    # 1. Schema check
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.warning("Processed DF missing columns: %s", missing)
    else:
        logger.info("Processed DF schema: OK (%d columns).", df.shape[1])

    # 2. Row count
    logger.info("Processed DF rows: %d", len(df))

    # 3. Coverage of key fields
    for col in ["court", "jurisdiction_state", "citation"]:
        if col in df.columns:
            logger.info("Non-empty '%s': %.1f%%", col, _pct_nonempty(df[col]))

    # 4. Outcome code domains
    issues = False

    if "outcome_code" in df.columns:
        bad_coarse = set(pd.unique(df["outcome_code"].dropna())) - ALLOWED_COARSE
        if bad_coarse:
            issues = True
            logger.warning("Invalid values in outcome_code: %s", sorted(bad_coarse))
        else:
            logger.info("Outcome_code domain OK (allowed %s).", sorted(ALLOWED_COARSE))

    if "outcome_code_fine" in df.columns:
        bad_fine = set(pd.unique(df["outcome_code_fine"].dropna())) - ALLOWED_FINE
        if bad_fine:
            issues = True
            logger.warning("Invalid values in outcome_code_fine: %s", sorted(bad_fine))
        else:
            logger.info("Outcome_code_fine domain OK (allowed %s).", sorted(ALLOWED_FINE))

    # 5. Quick distribution peek
    try:
        if "outcome_label_fine" in df.columns:
            dist = df["outcome_label_fine"].value_counts(dropna=False).head(6)
            logger.info("Outcome_label_fine distribution (top):\n%s", dist.to_string())
    except Exception:
        pass

    # 6. Duplicate check for IDs
    if len(df) > 0 and "case_id" in df.columns:
        dup = df["case_id"].duplicated().sum()
        if dup:
            issues = True
            logger.warning("Duplicate case_id rows: %d", dup)
        else:
            logger.info("case_id uniqueness: OK")

    # 7. Final pass/fail note
    if not issues:
        logger.info("Light validation: PASS")
    else:
        logger.info("Light validation: PASS with warnings (see above)")