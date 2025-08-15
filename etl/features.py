# Turns a single CourtListener "opinion" JSON object into a flat set of features for later use (case metadata, simple text stats, etc.).

import logging
logger = logging.getLogger('pipeline')

import re
import time
from typing import Dict, Any, Optional
from urllib.parse import urlparse, urlunparse


# Small helpers
def _normalize_url(u: str) -> str:
    """
    CourtListener links sometimes miss a trailing slash. This normalizes the URL
    so our tiny cache / equality checks work consistently.
    """
    if not u:
        return u
    p = urlparse(u)
    path = p.path if p.path.endswith("/") else p.path + "/"
    return urlunparse((p.scheme, p.netloc, path, "", "", ""))


def _safe_get_json(session, url: Optional[str]) -> Dict[str, Any]:
    """
    Make a GET request and parse JSON. If url is empty, return {}.
    NOTE: this will raise for non-200s (caller can try/except if desired).
    """
    if not url:
        return {}
    url = _normalize_url(url)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _pick_citation(cites) -> str:
    """
    The API returns citations in different shapes:
      - list[dict or str or int]
      - dict
      - scalar (str/int/float)
    (I'm basically trying to pull a single human-readable citation string.)
    """
    if cites is None:
        return ""
    if isinstance(cites, list) and cites:
        # first prioritiy is a dict with type == "official", otherwise first dict, otherwise first scalar
        dict_items = [c for c in cites if isinstance(c, dict)]
        if dict_items:
            official = next((c for c in dict_items if str(c.get("type", "")).lower() == "official"), None)
            return str((official or dict_items[0]).get("cite", "")) or ""
        scalar = next((c for c in cites if isinstance(c, (str, int, float))), None)
        return "" if scalar is None else str(scalar)
    if isinstance(cites, dict):
        return str(cites.get("cite", "")) or ""
    if isinstance(cites, (str, int, float)):
        return str(cites)
    return ""


# Simple signal that a line probably contains a court name
COURT_HEADING_RE = re.compile(
    r'\b(COURT|SUPREME|APPEALS|CIRCUIT|DISTRICT|JUDICIAL|COUNTY|STATE|COMMONWEALTH|CHANCERY)\b',
    re.I
)


def _extract_court_from_text(text: str) -> str:
    """
    If the court link is missing, this looks in the first ~60 lines of the opinion for a header that looks like a court name.
    """
    if not text:
        return ""
    header = text.splitlines()[:60]
    candidates = []
    for line in header:
        l = line.strip()
        if len(l) < 8:
            continue
        if COURT_HEADING_RE.search(l):
            candidates.append(l)
    if not candidates:
        return ""
    best = max(candidates, key=len)

    # tidy up things: stuff like "AT NASHVILLE" at the end
    best = re.sub(r'\bAT\s+[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)*\b$', '', best, flags=re.I).strip()
    # and "IN THE " at the beginning
    best = re.sub(r'^\s*IN THE\s+', '', best, flags=re.I).strip()
    return best


def _infer_state_from_court_name(court_name: str) -> str:
    """
    Tiny function to guess the state/jurisdiction from a court string
    ex:
    "Supreme Court of California" -> "California".
    """
    if not court_name:
        return ""
    m = re.search(r'\bof\s+([A-Z][A-Za-z ]+)$', court_name)
    if m:
        cand = m.group(1).strip()
        # drop a trailing "at City" if present (already handled above, but this catches some cases for some reason)
        cand = re.sub(r'\bat\s+.+$', '', cand, flags=re.I).strip()
        return cand
    return ""


def _label_from_text(text: str) -> str:
    """
    Temp code to guess outcome from the opinion text.
    """
    if not text:
        return "unknown"
    t = text.lower()
    if "affirmed" in t and "reversed" in t:
        return "mixed"
    if "affirmed" in t:
        return "affirmed"
    if "reversed" in t:
        return "reversed"
    if "remanded" in t:
        return "remanded"
    if "dismissed" in t:
        return "dismissed"
    return "other"


def _word_count(text: str) -> int:
    # Token counter
    if not text:
        return 0
    return len(re.findall(r'\w+', text))


# Main feature builder
def extract_features_from_opinion(op: Dict[str, Any], session, court_cache=None) -> Dict[str, Any]:
    """
    Take one opinion JSON blob (from /opinions/) and returns a flat dict of features.

    What this collects:
    - basic metadata (ids, dates, page_count, author, etc.)
    - case name and a single citation string (via the cluster)
    - court name (via link if possible; otherwise guess from top-of-text)
    - a very rough outcome signal from text (affirmed/reversed/etc.)
    - tiny text stats (char count, word count)

    Notes:
    - We use a small (dict) for court lookups so repeated courts are faster.
    - If `plain_text` is a URL, it fetches it to compute word counts.
    - This function is basically a substitute for heavy NLP.
    """
    if court_cache is None:
        court_cache = {}

    features: Dict[str, Any] = {}

    # Basic fields straight from the opinion
    features["case_id"]         = op.get("id")
    features["date_created"]    = op.get("date_created", "")
    features["date_modified"]   = op.get("date_modified", "")
    features["page_count"]      = op.get("page_count", None)
    features["per_curiam"]      = bool(op.get("per_curiam", False))
    features["type"]            = op.get("type", "")
    features["author_id"]       = op.get("author_id", None)
    features["joined_by_count"] = len(op.get("joined_by", []) or [])
    features["plain_text_url"]  = op.get("plain_text") or ""

    # Derives year from date_created (for plots)
    m = re.match(r'^(\d{4})-', features["date_created"] or "")
    features["opinion_year"] = int(m.group(1)) if m else None

    # Checls the cluster link for case name/citations/court
    cluster_url = op.get("cluster") or ""
    cluster: Dict[str, Any] = {}
    if cluster_url:
        try:
            cluster = _safe_get_json(session, cluster_url)
        except Exception:
            cluster = {}

    features["case_name"] = (
        cluster.get("caseName")
        or cluster.get("case_name")
        or cluster.get("caption")
        or ""
    )

    citations_list = cluster.get("citations")
    if citations_list is None:
        citations_list = op.get("citations", [])
    features["citation"] = _pick_citation(citations_list)

    # -------- court: try API first, fall back to header text -------- #
    court_name = ""
    court_url = cluster.get("court") or op.get("court") or ""
    if court_url:
        court_url = _normalize_url(court_url)
        court_json = court_cache.get(court_url)
        if court_json is None:
            try:
                court_json = _safe_get_json(session, court_url)
            except Exception:
                court_json = {}
            court_cache[court_url] = court_json

        court_name = (
            court_json.get("full_name")
            or court_json.get("name_abbreviation")
            or court_json.get("name")
            or ""
        )

    # Extracting opinion text because we may need the opinion text 
    # (for word counts / heuristic labels / court fallback)
    text = ""
    need_text = True  # flip to False for max speed with no text fetch
    if need_text and features["plain_text_url"]:
        try:
            tr = session.get(features["plain_text_url"], timeout=30)
            if tr.ok:
                text = tr.text or ""
        except Exception:
            text = ""

    # Parses court from header lines (last resort)
    if not court_name:
        parsed = _extract_court_from_text(text)
        if parsed:
            court_name = parsed

    features["court_name"]         = court_name
    features["jurisdiction_state"] = _infer_state_from_court_name(court_name)

    # Extra label features
    features["text_char_count"] = len(text) if text else 0
    features["text_word_count"] = _word_count(text)
    features["label_heuristic"] = _label_from_text(text)

    # I got an email from using the API too much, I had to pump the brakes a bit
    time.sleep(0.02)

    return features