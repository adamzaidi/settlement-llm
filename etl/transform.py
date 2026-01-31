import logging
logger = logging.getLogger("pipeline.transform")

import os
import re
import time
import random
import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from etl.enrich_helpers import (
    infer_court_from_url,
    infer_state_from_court,
    limited_online_court_lookup,
    limited_online_cluster_citation,
    limited_online_cluster_texts,
    pick_citation_from_header,
    detect_per_curiam,
    outcome_from_text,
    normalize_court_name,
    infer_court_from_header,
    header_from_text_body,
)

# Paths and constants
RAW_PATH = "data/extracted/raw_data.csv"
OUT_PATH = "data/processed/processed_data.csv"

# Columns I want to preserve and order first
PRESERVE_ORDER = [
    "case_id","date_created","date_modified","opinion_year","page_count",
    "per_curiam","type","author_id","joined_by_count","case_name",
    "citation","court","jurisdiction_state","text_char_count","text_word_count",
    "label_heuristic","plain_text_present","plain_text_url","download_url",
    "cluster_url","court_url"
]

MAX_HEADER_BYTES = 40_000
MAX_BODY_BYTES   = 600_000  # pull enough body to include disposition tail
REQUEST_TIMEOUT = 15

# NEW: outcome labeling fetch controls
MAX_TEXT_FETCHES = int(os.getenv("MAX_TEXT_FETCHES", "350"))        # start modest; raise later
TEXT_SLICE_CHARS = int(os.getenv("TEXT_SLICE_CHARS", "12000"))      # enough for dispo language
SLEEP_BETWEEN_FETCH = float(os.getenv("TEXT_FETCH_SLEEP", "0.12"))  # gentle pacing

# NEW: prioritize labeling from existing text in the CSV (fast_mode=False)
TEXT_COL_CANDIDATES = [
    "text",                # recommended column name (we'll add in extract)
    "plain_text",          # common alt
    "opinion_text",
    "body_text",
    "text_snippet",        # if you decide to store a slice instead of full text
]

# Shared requests session with auth/headers
_CL_API_KEY = os.getenv("COURTLISTENER_API_KEY") or ""
_CL_EMAIL   = os.getenv("COURTLISTENER_EMAIL") or ""
_HDRS = {
    "User-Agent": f"inst414-transform ({_CL_EMAIL})" if _CL_EMAIL else "inst414-transform",
}
if _CL_API_KEY:
    _HDRS["Authorization"] = f"Token {_CL_API_KEY}"

_session: requests.Session | None = None
def _sess() -> requests.Session:
    """Return a shared requests.Session with CourtListener headers."""
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(_HDRS)
        _session = s
    return _session

# Utility functions
def _fill_unknown(series: pd.Series) -> pd.Series:
    """Replace blanks/NaNs in a string column with 'Unknown'."""
    s = series.copy()
    if s.dtype == object:
        s = s.replace(r"^\s*$", np.nan, regex=True)
    return s.fillna("Unknown")

def _as_str(v) -> str:
    """Convert any value to a safe string. Empty string if None/NaN."""
    try:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        return str(v)
    except Exception:
        return ""

def _normalize_text_url(u: str) -> str:
    """
    CourtListener often returns relative URLs for plain_text (e.g. '/download/...').
    Convert to absolute so we can fetch.
    """
    if not u:
        return ""
    s = str(u).strip()
    if not s or s.lower() in {"unknown", "nan", "none"}:
        return ""
    if s.startswith("/"):
        return "https://www.courtlistener.com" + s
    return s

def _read_header_text(url: str) -> str:
    """Fetch a header (small slice) if URL looks like text/html."""
    if not url or not url.startswith(("http://", "https://")):
        return ""
    try:
        r = _sess().get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if not (ctype.startswith("text/") or "html" in ctype):
            return ""
        return (r.text or "")[:MAX_HEADER_BYTES]
    except Exception:
        return ""

def _read_text_body(url: str) -> str:
    """Fetch a larger slice of the opinion text for outcome detection."""
    if not url or not url.startswith(("http://", "https://")):
        return ""
    try:
        r = _sess().get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if not (ctype.startswith("text/") or "html" in ctype):
            return ""
        return (r.text or "")[:MAX_BODY_BYTES]
    except Exception:
        return ""

def _sanitize_text_fields(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Sanitize text-like columns to remove embedded newlines/tabs.
    This prevents misalignment when viewing CSVs in Excel/Sheets.
    """
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                     .str.replace(r"[\r\n\t]+", " ", regex=True)
                     .str.replace(r"\s{2,}", " ", regex=True)
                     .str.strip()
            )
    return df

def _get_existing_text_for_row(df: pd.DataFrame, i: int) -> str:
    """
    Return already-present text content for a row if available.
    Tries a few candidate columns. Returns '' if none found.
    """
    for c in TEXT_COL_CANDIDATES:
        if c in df.columns:
            v = df.at[i, c]
            if v is None:
                continue
            s = str(v)
            if s and s.lower() not in {"nan", "none", "unknown"}:
                return s
    return ""

# -----------------------------------------------------------------------------
# Rule-based outcome labeling (Option A)
# Keep fine codes within 0..5 to match your pipeline validation.
# -----------------------------------------------------------------------------

_FINE_RULES = [
    ("vacated",  r"\bvacat(?:e|ed|ing|es)\b"),
    ("reversed", r"\brevers(?:e|ed|ing|es)\b"),
    ("remanded", r"\bremand(?:ed|ing|s)?\b"),
    ("affirmed", r"\baffirm(?:e|ed|ing|s)?\b"),
    ("dismissed",r"\bdismiss(?:e|ed|ing|es)\b"),
]

_COARSE_MAP = {
    "affirmed": 1,
    "dismissed": 1,
    "reversed": 2,
    "vacated": 2,
    "remanded": 2,
}

_FINE_CODE_MAP = {
    "other": 0,
    "affirmed": 1,
    "reversed": 2,
    "vacated": 3,
    "remanded": 4,
    "dismissed": 5,
}

def _extract_evidence_snippet(text: str, match_span: tuple[int, int], window: int = 180) -> str:
    """Grab a small snippet around a match span for auditability."""
    if not text or not match_span:
        return ""
    start, end = match_span
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    snippet = text[lo:hi].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet

def _label_outcome_from_text(text: str) -> tuple[int, str, int, str]:
    """
    Returns:
      (outcome_code, outcome_label_fine, outcome_code_fine, outcome_evidence)
    """
    if not text:
        return 0, "other", 0, ""

    t = re.sub(r"\s+", " ", text.lower())

    # Prefer a "disposition" zone if present (helps precision a bit)
    zone = t
    m = re.search(r"\b(opinion|decision|judgment|order|conclusion|disposition)\b", t)
    if m:
        zone = t[m.start():]

    for fine, pat in _FINE_RULES:
        m2 = re.search(pat, zone)
        if m2:
            coarse = _COARSE_MAP.get(fine, 0)
            fine_code = _FINE_CODE_MAP.get(fine, 0)
            evidence = _extract_evidence_snippet(zone, (m2.start(), m2.end()))
            return coarse, fine, fine_code, evidence

    return 0, "other", 0, ""

def _fetch_plain_text_slice(url: str) -> str:
    """
    Fetch a small slice from CourtListener plain_text_url.
    Keeps things fast (and less likely to hit rate limits).
    """
    if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return ""
    r = _sess().get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    if not (ctype.startswith("text/") or "html" in ctype):
        return ""
    txt = (r.text or "").strip()
    if len(txt) > TEXT_SLICE_CHARS:
        txt = txt[:TEXT_SLICE_CHARS]
    return txt

# -----------------------------------------------------------------------------
# Main transformation
# -----------------------------------------------------------------------------
def transform_data(raw_path: str = RAW_PATH, out_path: str = OUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.lower()

    # Ensure all expected columns exist
    for col in PRESERVE_ORDER:
        if col not in df.columns:
            df[col] = np.nan

    # Normalize string columns (IMPORTANT: do NOT _fill_unknown plain_text_url)
    str_cols = [
        "case_name","court","citation","jurisdiction_state",
        "label_heuristic","type","download_url","cluster_url","court_url"
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = _fill_unknown(df[col])

    # Keep plain_text_url raw-ish (blank is fine); clean common junk tokens
    if "plain_text_url" in df.columns:
        df["plain_text_url"] = df["plain_text_url"].fillna("").astype(str)
        df["plain_text_url"] = df["plain_text_url"].replace(
            {"Unknown": "", "None": "", "nan": "", "NaN": ""}
        )
    else:
        df["plain_text_url"] = ""

    # Normalize datetimes and numerics
    for col in ["date_created","date_modified"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["opinion_year","page_count","author_id","joined_by_count",
                "text_char_count","text_word_count","plain_text_present"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Make sure flags exist (extract usually sets these)
    if "has_court" not in df.columns:
        df["has_court"] = (df.get("court", "Unknown") != "Unknown").astype(int)
    if "has_citation" not in df.columns:
        df["has_citation"] = (df.get("citation", "Unknown") != "Unknown").astype(int)

    # Fresh outcome columns
    df["outcome_code"] = 0
    df["outcome_code_fine"] = 0
    df["outcome_label_fine"] = "other"
    df["outcome_evidence"] = ""

    # ---------------------------------------------------------------------
    # Outcome labeling: 1) from existing text columns, 2) fallback to URL fetch
    # ---------------------------------------------------------------------

    # (1) Label from existing text columns if present
    present_text_cols = [c for c in TEXT_COL_CANDIDATES if c in df.columns]
    labeled_existing = 0

    if present_text_cols:
        for i in df.index.tolist():
            try:
                txt = _get_existing_text_for_row(df, i)
                if not txt:
                    continue

                # counts
                df.at[i, "text_char_count"] = int(len(txt))
                df.at[i, "text_word_count"] = int(len(re.findall(r"\w+", txt)))

                # label
                coarse, fine, fine_code, evidence = _label_outcome_from_text(txt)
                df.at[i, "outcome_code"] = int(coarse)
                df.at[i, "outcome_label_fine"] = fine
                df.at[i, "outcome_code_fine"] = int(fine_code)
                df.at[i, "outcome_evidence"] = evidence

                labeled_existing += 1
            except Exception:
                continue

    logger.info(
        "Outcome labeling: labeled %d rows from existing text columns (%s).",
        labeled_existing,
        ", ".join(present_text_cols) if present_text_cols else "none found"
    )

    # (2) Fallback: only fetch for rows that still have no words AND have a URL
    df["plain_text_url_norm"] = df["plain_text_url"].map(_normalize_text_url)

    if "text_word_count" in df.columns:
        twc = pd.to_numeric(df["text_word_count"], errors="coerce").fillna(0)
    else:
        twc = pd.Series([0] * len(df))

    needs_fetch_mask = (twc <= 0) & df["plain_text_url_norm"].astype(str).str.startswith(("http://","https://"), na=False)
    fetchable_idx = df[needs_fetch_mask].index.tolist()

    logger.info(
        "Outcome labeling: fetchable plain_text_url rows=%d (total=%d).",
        len(fetchable_idx), len(df)
    )

    if len(fetchable_idx) > MAX_TEXT_FETCHES:
        random.seed(7)
        fetchable_idx = random.sample(fetchable_idx, MAX_TEXT_FETCHES)

    fetched = 0
    for i in fetchable_idx:
        url = _as_str(df.at[i, "plain_text_url_norm"])
        if not url.startswith(("http://","https://")):
            continue

        try:
            time.sleep(SLEEP_BETWEEN_FETCH + random.uniform(0, 0.08))
            txt = _fetch_plain_text_slice(url)
            if not txt:
                continue

            df.at[i, "text_char_count"] = int(len(txt))
            df.at[i, "text_word_count"] = int(len(re.findall(r"\w+", txt)))

            coarse, fine, fine_code, evidence = _label_outcome_from_text(txt)
            df.at[i, "outcome_code"] = int(coarse)
            df.at[i, "outcome_label_fine"] = fine
            df.at[i, "outcome_code_fine"] = int(fine_code)
            df.at[i, "outcome_evidence"] = evidence

            fetched += 1
        except Exception:
            continue

    logger.info("Outcome labeling: fetched text for %d rows (cap=%d).", fetched, MAX_TEXT_FETCHES)

    # Ensure numeric types are clean
    df["outcome_code"] = pd.to_numeric(df["outcome_code"], errors="coerce").fillna(0).astype(int)
    df["outcome_code_fine"] = pd.to_numeric(df["outcome_code_fine"], errors="coerce").fillna(0).astype(int)
    df["outcome_label_fine"] = df["outcome_label_fine"].fillna("other")

    # Save final
    ordered = [c for c in PRESERVE_ORDER if c in df.columns]
    tail = [c for c in df.columns if c not in ordered]
    df = df[ordered + tail]

    # Sanitize text fields before saving
    textish_cols = [
        "case_name","citation","court","jurisdiction_state","label_heuristic",
        "type","download_url","cluster_url","court_url","plain_text_url",
        "outcome_label_fine","outcome_evidence"
    ]
    df = _sanitize_text_fields(df, textish_cols)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    # Prints a quick validation summary
    nonempty_words = (df.get("text_word_count", pd.Series(dtype=float)).fillna(0) > 0).mean() if "text_word_count" in df.columns else 0
    print(
        f"Processed {len(df)} cases → {out_path}\n"
        f"  • non-empty court: {(df['has_court'].mean()*100 if 'has_court' in df.columns else 0):.1f}%\n"
        f"  • non-empty citation: {(df['has_citation'].mean()*100 if 'has_citation' in df.columns else 0):.1f}%\n"
        f"  • text present: {(df.get('plain_text_present', pd.Series(dtype=float)).fillna(0).mean()*100 if 'plain_text_present' in df.columns else 0):.1f}%\n"
        f"  • >0 text words: {nonempty_words*100:.1f}%"
    )

    # Log distributions for grader visibility
    try:
        logger.info("Outcome_code distribution after transform:\n%s", df["outcome_code"].value_counts(dropna=False))
        logger.info("Fine labels:\n%s", df["outcome_label_fine"].value_counts(dropna=False))
        logger.info("Fine codes:\n%s", df["outcome_code_fine"].value_counts(dropna=False))
    except Exception:
        pass

    return df

if __name__ == "__main__":
    transform_data()