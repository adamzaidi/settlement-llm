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

# Paths and constants
RAW_PATH = "data/extracted/raw_data.csv"
OUT_PATH = "data/processed/processed_data.csv"

# CHANGE: review queue output
REVIEW_PATH = "data/processed/review_queue.csv"

# CHANGE: include opinion_id / opinion_api_url in preserve order so outputs are stable + review queue always has them
PRESERVE_ORDER = [
    "opinion_id", "opinion_api_url",  # CHANGE

    "case_id","date_created","date_modified","opinion_year","page_count",
    "per_curiam","type","author_id","joined_by_count","case_name",
    "citation","court","jurisdiction_state","text_char_count","text_word_count",
    "label_heuristic","plain_text_present","plain_text_url","download_url",
    "cluster_url","court_url",

    # keep head/tail snippets if present
    "text_snippet_head","text_snippet_tail","text_snippet",
]

MAX_HEADER_BYTES = 40_000
MAX_BODY_BYTES   = 600_000
REQUEST_TIMEOUT = 15

MAX_TEXT_FETCHES = int(os.getenv("MAX_TEXT_FETCHES", "350"))
TEXT_SLICE_CHARS = int(os.getenv("TEXT_SLICE_CHARS", "12000"))
SLEEP_BETWEEN_FETCH = float(os.getenv("TEXT_FETCH_SLEEP", "0.12"))

# CHANGE: tail-first candidates (we want tail for disposition)
TEXT_COL_CANDIDATES = [
    "text_snippet",       # in extract, this is already tail-preferred
    "text_snippet_tail",
    "text_snippet_head",
    "text",
    "plain_text",
    "opinion_text",
    "body_text",
]

_CL_API_KEY = os.getenv("COURTLISTENER_API_KEY") or ""
_CL_EMAIL   = os.getenv("COURTLISTENER_EMAIL") or ""
_HDRS = {"User-Agent": f"inst414-transform ({_CL_EMAIL})" if _CL_EMAIL else "inst414-transform"}
if _CL_API_KEY:
    _HDRS["Authorization"] = f"Token {_CL_API_KEY}"

_session: requests.Session | None = None
def _sess() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(_HDRS)
        _session = s
    return _session

def _fill_unknown(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object:
        s = s.replace(r"^\s*$", np.nan, regex=True)
    return s.fillna("Unknown")

def _as_str(v) -> str:
    try:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        return str(v)
    except Exception:
        return ""

def _normalize_text_url(u: str) -> str:
    if not u:
        return ""
    s = str(u).strip()
    if not s or s.lower() in {"unknown", "nan", "none"}:
        return ""
    if s.startswith("/"):
        return "https://www.courtlistener.com" + s
    return s

def _sanitize_text_fields(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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
# Outcome labeling (rule-based) + confidence + review flags
# -----------------------------------------------------------------------------

_FINE_RULES = [
    ("vacated",   r"\bvacat(?:e|ed|ing|es)\b"),
    ("reversed",  r"\brevers(?:e|ed|ing|es)\b"),
    ("remanded",  r"\bremand(?:ed|ing|s)?\b"),
    ("affirmed",  r"\baffirm(?:e|ed|ing|s)?\b"),
    ("dismissed", r"\bdismiss(?:e|ed|ing|es)\b"),
]

_COARSE_MAP = {
    "affirmed": 1,
    "dismissed": 1,
    "reversed": 2,
    "vacated": 2,
    "remanded": 2,
    "mixed": 2,
}

_FINE_CODE_MAP = {
    "other": 0,
    "affirmed": 1,
    "reversed": 2,
    "vacated": 3,
    "remanded": 4,
    "dismissed": 5,
    "mixed": 0,
}

_STRONG_PHRASES = [
    r"\bjudgment\s+is\s+affirmed\b",
    r"\bjudgment\s+is\s+reversed\b",
    r"\border\s+is\s+affirmed\b",
    r"\border\s+is\s+reversed\b",
    r"\bappeal\s+is\s+dismissed\b",
    r"\bwe\s+affirm\b",
    r"\bwe\s+reverse\b",
    r"\bwe\s+vacate\b",
    r"\bwe\s+remand\b",
    r"\bis\s+hereby\s+affirmed\b",
    r"\bis\s+hereby\s+reversed\b",
    r"\breversed\s+and\s+remanded\b",
    r"\baffirmed\s+in\s+part\s+and\s+reversed\s+in\s+part\b",
]

_DISPO_ZONE_HINT = r"\b(opinion|decision|judgment|order|conclusion|disposition)\b"

def _extract_evidence_snippet(text: str, match_span: tuple[int, int], window: int = 180) -> str:
    if not text or not match_span:
        return ""
    start, end = match_span
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    snippet = text[lo:hi].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet

def _label_outcome_from_text_with_meta(text: str) -> tuple[int, str, int, str, float, int, int, float]:
    """
    Returns:
      (coarse_code, fine_label, fine_code, evidence_snippet,
       evidence_pos_ratio, disposition_zone_found, strong_phrase_found, confidence)
    """
    if not text:
        return 0, "other", 0, "", 0.0, 0, 0, 0.0

    t_raw = text.lower()
    t = re.sub(r"\s+", " ", t_raw)

    zone = t
    zone_offset = 0
    m_zone = re.search(_DISPO_ZONE_HINT, t)
    disposition_zone_found = 1 if m_zone else 0
    if m_zone:
        zone_offset = m_zone.start()
        zone = t[zone_offset:]

    has_affirm = re.search(r"\baffirm(?:e|ed|ing|s)?\b", zone) is not None
    has_change = re.search(r"\b(revers(?:e|ed|ing|es)|vacat(?:e|ed|ing|es)|remand(?:ed|ing|s)?)\b", zone) is not None
    if has_affirm and has_change:
        m_any = re.search(
            r"\baffirm(?:e|ed|ing|s)?\b|\brevers(?:e|ed|ing|es)\b|\bvacat(?:e|ed|ing|es)\b|\bremand(?:ed|ing|s)?\b",
            zone
        )
        evidence = _extract_evidence_snippet(zone, (m_any.start(), m_any.end())) if m_any else ""
        evidence_pos_ratio = (zone_offset + (m_any.start() if m_any else 0)) / max(1, len(t))
        strong_found = 1 if any(re.search(p, zone) for p in _STRONG_PHRASES) else 0

        confidence = 0.55
        confidence += 0.15 if disposition_zone_found else 0.0
        confidence += 0.15 if evidence_pos_ratio > 0.60 else 0.0
        confidence += 0.10 if strong_found else 0.0
        confidence = max(0.0, min(1.0, confidence))

        return _COARSE_MAP["mixed"], "mixed", _FINE_CODE_MAP["mixed"], evidence, evidence_pos_ratio, disposition_zone_found, strong_found, confidence

    for fine, pat in _FINE_RULES:
        m = re.search(pat, zone)
        if m:
            coarse = _COARSE_MAP.get(fine, 0)
            fine_code = _FINE_CODE_MAP.get(fine, 0)
            evidence = _extract_evidence_snippet(zone, (m.start(), m.end()))
            evidence_pos_ratio = (zone_offset + m.start()) / max(1, len(t))
            strong_found = 1 if any(re.search(p, zone) for p in _STRONG_PHRASES) else 0

            confidence = 0.50
            confidence += 0.20 if disposition_zone_found else 0.0
            confidence += 0.20 if evidence_pos_ratio > 0.60 else 0.0
            confidence += 0.10 if strong_found else 0.0
            if (not strong_found) and evidence_pos_ratio < 0.35:
                confidence -= 0.15
            confidence = max(0.0, min(1.0, confidence))

            return coarse, fine, fine_code, evidence, evidence_pos_ratio, disposition_zone_found, strong_found, confidence

    return 0, "other", 0, "", 0.0, disposition_zone_found, 0, 0.25


# CHANGE: make fallback fetch tail-biased (dispositions tend to be near the end)
def _fetch_plain_text_slice(url: str) -> str:
    if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return ""
    r = _sess().get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    if not (ctype.startswith("text/") or "html" in ctype):
        return ""
    txt = (r.text or "").strip()
    if not txt:
        return ""

    # CHANGE: take tail slice (better chance of capturing disposition)
    if len(txt) > TEXT_SLICE_CHARS:
        txt = txt[-TEXT_SLICE_CHARS:]

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

    # Normalize string columns
    str_cols = [
        "case_name","court","citation","jurisdiction_state",
        "label_heuristic","type","download_url","cluster_url","court_url",
        "opinion_api_url",  # CHANGE: keep this stable too
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = _fill_unknown(df[col])

    # Keep plain_text_url raw-ish (blank is fine)
    if "plain_text_url" in df.columns:
        df["plain_text_url"] = df["plain_text_url"].fillna("").astype(str)
        df["plain_text_url"] = df["plain_text_url"].replace({"Unknown": "", "None": "", "nan": "", "NaN": ""})
    else:
        df["plain_text_url"] = ""

    # Normalize datetimes and numerics
    for col in ["date_created","date_modified"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    for col in ["opinion_year","page_count","author_id","joined_by_count",
                "text_char_count","text_word_count","plain_text_present","opinion_id","case_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flags
    if "has_court" not in df.columns:
        df["has_court"] = (df.get("court", "Unknown") != "Unknown").astype(int)
    if "has_citation" not in df.columns:
        df["has_citation"] = (df.get("citation", "Unknown") != "Unknown").astype(int)

    # Fresh outcome columns
    df["outcome_code"] = 0
    df["outcome_code_fine"] = 0
    df["outcome_label_fine"] = "other"
    df["outcome_evidence"] = ""

    # New triage fields
    df["outcome_confidence"] = 0.0
    df["needs_review"] = 1
    df["disposition_zone_found"] = 0
    df["evidence_contains_strong_phrase"] = 0
    df["evidence_match_position"] = 0.0

    # (1) Label from existing text columns if present
    present_text_cols = [c for c in TEXT_COL_CANDIDATES if c in df.columns]
    labeled_existing = 0

    if present_text_cols:
        for i in df.index.tolist():
            try:
                txt = _get_existing_text_for_row(df, i)
                if not txt:
                    continue

                df.at[i, "text_char_count"] = int(len(txt))
                df.at[i, "text_word_count"] = int(len(re.findall(r"\w+", txt)))

                coarse, fine, fine_code, evidence, pos, dz, strong, conf = _label_outcome_from_text_with_meta(txt)
                df.at[i, "outcome_code"] = int(coarse)
                df.at[i, "outcome_label_fine"] = fine
                df.at[i, "outcome_code_fine"] = int(fine_code)
                df.at[i, "outcome_evidence"] = evidence

                df.at[i, "outcome_confidence"] = float(conf)
                df.at[i, "disposition_zone_found"] = int(dz)
                df.at[i, "evidence_contains_strong_phrase"] = int(strong)
                df.at[i, "evidence_match_position"] = float(pos)

                needs = 0
                if fine in {"other", "mixed"}:
                    needs = 1
                if conf < 0.60:
                    needs = 1
                df.at[i, "needs_review"] = int(needs)

                labeled_existing += 1
            except Exception:
                continue

    logger.info(
        "Outcome labeling: labeled %d rows from existing text columns (%s).",
        labeled_existing,
        ", ".join(present_text_cols) if present_text_cols else "none found"
    )

    # (2) Fallback fetch only if still no words and URL exists
    df["plain_text_url_norm"] = df["plain_text_url"].map(_normalize_text_url)
    twc = pd.to_numeric(df.get("text_word_count", 0), errors="coerce").fillna(0)

    needs_fetch_mask = (twc <= 0) & df["plain_text_url_norm"].astype(str).str.startswith(("http://","https://"), na=False)
    fetchable_idx = df[needs_fetch_mask].index.tolist()

    logger.info("Outcome labeling: fetchable plain_text_url rows=%d (total=%d).", len(fetchable_idx), len(df))

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

            coarse, fine, fine_code, evidence, pos, dz, strong, conf = _label_outcome_from_text_with_meta(txt)
            df.at[i, "outcome_code"] = int(coarse)
            df.at[i, "outcome_label_fine"] = fine
            df.at[i, "outcome_code_fine"] = int(fine_code)
            df.at[i, "outcome_evidence"] = evidence

            df.at[i, "outcome_confidence"] = float(conf)
            df.at[i, "disposition_zone_found"] = int(dz)
            df.at[i, "evidence_contains_strong_phrase"] = int(strong)
            df.at[i, "evidence_match_position"] = float(pos)

            needs = 0
            if fine in {"other", "mixed"}:
                needs = 1
            if conf < 0.60:
                needs = 1
            df.at[i, "needs_review"] = int(needs)

            fetched += 1
        except Exception:
            continue

    logger.info("Outcome labeling: fetched text for %d rows (cap=%d).", fetched, MAX_TEXT_FETCHES)

    # Clean types
    df["outcome_code"] = pd.to_numeric(df["outcome_code"], errors="coerce").fillna(0).astype(int)
    df["outcome_code_fine"] = pd.to_numeric(df["outcome_code_fine"], errors="coerce").fillna(0).astype(int)
    df["outcome_label_fine"] = df["outcome_label_fine"].fillna("other")
    df["outcome_confidence"] = pd.to_numeric(df["outcome_confidence"], errors="coerce").fillna(0.0).astype(float)
    df["needs_review"] = pd.to_numeric(df["needs_review"], errors="coerce").fillna(1).astype(int)

    # Save final processed
    ordered = [c for c in PRESERVE_ORDER if c in df.columns]
    tail = [c for c in df.columns if c not in ordered]
    df = df[ordered + tail]

    # Sanitize text fields
    textish_cols = [
        "case_name","citation","court","jurisdiction_state","label_heuristic",
        "type","download_url","cluster_url","court_url","plain_text_url",
        "outcome_label_fine","outcome_evidence",
        "text_snippet","text_snippet_head","text_snippet_tail",
        "opinion_api_url",  # CHANGE
    ]
    df = _sanitize_text_fields(df, textish_cols)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    # Export review queue
    review_cols = [
        "opinion_id","case_id","case_name","court","jurisdiction_state","opinion_year",
        "citation","outcome_code","outcome_label_fine","outcome_code_fine",
        "outcome_confidence","needs_review","disposition_zone_found",
        "evidence_contains_strong_phrase","evidence_match_position",
        "outcome_evidence","plain_text_url","opinion_api_url","cluster_url","court_url",
    ]
    review_cols = [c for c in review_cols if c in df.columns]
    review_df = df[(df["needs_review"] == 1)].copy()
    review_df = review_df.sort_values(["outcome_confidence", "opinion_year"], ascending=[True, False])
    review_df[review_cols].to_csv(REVIEW_PATH, index=False)

    # Summary print
    nonempty_words = (df.get("text_word_count", pd.Series(dtype=float)).fillna(0) > 0).mean() if "text_word_count" in df.columns else 0
    print(
        f"Processed {len(df)} cases → {out_path}\n"
        f"  • non-empty court: {(df['has_court'].mean()*100 if 'has_court' in df.columns else 0):.1f}%\n"
        f"  • non-empty citation: {(df['has_citation'].mean()*100 if 'has_citation' in df.columns else 0):.1f}%\n"
        f"  • plain_text_present: {(df.get('plain_text_present', pd.Series(dtype=float)).fillna(0).mean()*100 if 'plain_text_present' in df.columns else 0):.1f}%\n"
        f"  • has_any_text (>0 words): {nonempty_words*100:.1f}%\n"
        f"  • review_queue rows: {len(review_df)} → {REVIEW_PATH}"
    )

    # Log distributions
    try:
        logger.info("Outcome_code distribution:\n%s", df["outcome_code"].value_counts(dropna=False))
        logger.info("Fine labels:\n%s", df["outcome_label_fine"].value_counts(dropna=False))
        logger.info("Confidence (p50/p90): %.2f / %.2f",
                    float(df["outcome_confidence"].quantile(0.50)),
                    float(df["outcome_confidence"].quantile(0.90)))
        logger.info("Needs review: %.1f%%", float((df["needs_review"] == 1).mean()*100))
    except Exception:
        pass

    return df

if __name__ == "__main__":
    transform_data()