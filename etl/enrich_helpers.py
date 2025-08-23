# --------------------------------------------------------
# Lightweight enrichment helpers used by transform.py
# --------------------------------------------------------

from __future__ import annotations

import os
import re
import shelve
import logging
from typing import Dict, Tuple, Optional, Any, List
from urllib.parse import urlparse

import requests
import html as _html
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

CL_API_KEY = os.getenv("COURTLISTENER_API_KEY") or ""
CL_EMAIL   = os.getenv("COURTLISTENER_EMAIL") or ""

# CONFIG 
HEADER_LINES = 150
HTTP_CACHE = "cache/enrich_http_cache.db"
REQUEST_TIMEOUT = 20
USER_AGENT = "inst414-enrich"
DEFAULT_UA = f"{USER_AGENT} ({CL_EMAIL})" if CL_EMAIL else USER_AGENT

# Hard caps so I never hammer the API
MAX_ONLINE_COURT_LOOKUPS   = 50
MAX_ONLINE_CLUSTER_LOOKUPS = 1000  

# Regex patterns for reporters, outcomes, etc.
REPORTER_REGEX = re.compile(
    r"\b(\d{1,4}\s+(F\.?Supp\.?3d|F\.?4th|F\.?3d|F\.?2d|U\.S\.|S\.Ct\.|L\.Ed\.?2d|"
    r"N\.E\.?3d|N\.W\.?2d|So\.?3d|P\.?3d|P\.?2d|Cal\.?App\.?5th|Cal\.?App\.?4th|"
    r"A\.?3d|A\.?2d|N\.Y\.S\.?3d|N\.Y\.2d|Mass\.|Ill\.|Tex\.|Ohio St\.?3d)\s+\d{1,6})\b"
)
PER_CURIAM_RE = re.compile(r"\bper\s+curiam\b", re.I)

# Keywords I use for simple outcome inference
OUTCOME_PATTERNS = [
    (re.compile(r"\baffirm(ed|ance|ing)\b", re.I), "affirmed"),
    (re.compile(r"\brevers(e|ed|al|ing)\b", re.I), "reversed"),
    (re.compile(r"\bremand(ed|ing|s)\b", re.I), "remanded"),
    (re.compile(r"\bdismiss(ed|al|ing)\b", re.I), "dismissed"),
    (re.compile(r"\bvacat(e|ed|ing|ur)\b", re.I), "vacated"),
]

# Header detection
COURT_LINE_RE = re.compile(
    r"^\s*(SUPREME\s+COURT|COURT\s+OF\s+APPEALS|DISTRICT\s+COURT|BANKRUPTCY\s+COURT|"
    r"U\.S\.\s+COURT\s+OF\s+APPEALS|UNITED\s+STATES\s+COURT\s+OF\s+APPEALS|"
    r"UNITED\s+STATES\s+DISTRICT\s+COURT|U\.S\.\s+DISTRICT\s+COURT|SUPERIOR\s+COURT|"
    r"COMMONWEALTH\s+COURT|TAX\s+COURT)"
    r".{0,120}$",
    re.I
)

# Many mappings omitted (CIRCUIT_HOST_MAP, STATE_DOMAIN_HINTS, etc.)

# HTTP session & cache
_session: Optional[requests.Session] = None
_court_lookups_done = 0
_cluster_lookups_done = 0

def _get_sess() -> requests.Session:
    """
    Return a shared requests.Session with retry + headers configured.

    I include retry logic to be resilient against transient HTTP failures,
    and I attach CourtListener credentials if present.
    """
    global _session
    if _session is None:
        s = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers["User-Agent"] = DEFAULT_UA
        if CL_API_KEY:
            s.headers["Authorization"] = f"Token {CL_API_KEY}"
        _session = s
    return _session


def _http_get_json(url: str, logger: logging.Logger) -> dict:
    """
    Fetch JSON with simple shelve-based caching.

    I cache each response by URL so repeated lookups don't re-hit the network.
    """
    os.makedirs(os.path.dirname(HTTP_CACHE), exist_ok=True)
    with shelve.open(HTTP_CACHE) as db:
        key = f"GET::{url}"
        if key in db:
            return db[key]
        try:
            r = _get_sess().get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code >= 400:
                logger.debug("HTTP %s on %s", r.status_code, url)
            r.raise_for_status()
            data = r.json()
            db[key] = data
            return data
        except Exception as e:
            logger.debug("HTTP JSON fail %s: %s", url, e)
            return {}

# small helpers
def _strip_html(s: str) -> str:
    """
    Strip HTML tags + entities down to text.
    Used when plain_text isn't present in opinion responses.
    """
    if not s:
        return ""
    s = _html.unescape(s)
    s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return s.strip()


def _cluster_id_from_url(cluster_url: str) -> str:
    """
    Extract numeric cluster ID from a CourtListener cluster URL.
    Returns an empty string if not matched.
    """
    m = re.search(r"/clusters/(\d+)", cluster_url or "")
    return m.group(1) if m else ""

def normalize_court_name(name: str) -> str:
    """Normalize whitespace and abbreviate common 'United States' forms."""
    if not name:
        return ""
    s = re.sub(r"\s+", " ", str(name)).strip(" ,")
    s = re.sub(r"(?i)^United States Court of Appeals", "U.S. Court of Appeals", s)
    s = re.sub(r"(?i)^United States District Court", "U.S. District Court", s)
    s = re.sub(r"(?i)^United States Bankruptcy Court", "U.S. Bankruptcy Court", s)
    return s


def infer_court_from_header(text: str) -> str:
    """
    Scan the first HEADER_LINES of a document for a court line.
    Returns a normalized name if found.
    """
    if not text:
        return ""
    for raw in text.splitlines()[:HEADER_LINES]:
        line = raw.strip()
        if not line:
            continue
        if COURT_LINE_RE.search(line):
            line = re.sub(r"\s{2,}", " ", line).strip(" ,")
            return normalize_court_name(line)
    return ""


def header_from_text_body(body: str) -> str:
    """Take the first HEADER_LINES lines from a full text body as 'header'."""
    if not body:
        return ""
    return "\n".join(body.splitlines()[:HEADER_LINES])


def infer_state_from_court(court_name: str) -> str:
    """
    Infer jurisdiction (state or 'Federal') from a court name.
    I fall back to regex matching on state words.
    """
    if not court_name:
        return ""
    if re.search(r"U\.S\.", court_name, re.I):
        return "Federal"
    m = re.search(STATE_WORDS, court_name, re.I)
    if m:
        return re.sub(r"\s+", " ", m.group(0)).title()
    return ""


def infer_court_from_url(download_url: str) -> Tuple[str, str]:
    """
    Infer court name and state from the host in a download_url.
    I use overrides, circuit maps, and state domain hints in priority order.
    """
    if not download_url:
        return "", ""
    try:
        host = urlparse(download_url).netloc.lower()
    except Exception:
        return "", ""

    # Priority 1: explicit overrides
    if host in COURT_HOST_OVERRIDES:
        name, state = COURT_HOST_OVERRIDES[host]
        return normalize_court_name(name), state

    # Priority 2: circuit courts
    for circ_host, circ_name in CIRCUIT_HOST_MAP.items():
        if host == circ_host or host.endswith("." + circ_host):
            return circ_name, "Federal"

    # Priority 3: state domain hints
    for dom, state in STATE_DOMAIN_HINTS.items():
        if host == dom or host.endswith("." + dom):
            if not state:
                return "", ""
            if state == "Federal":
                return "", "Federal"
            return (f"{state} Appellate Court"), state

    # Special-case fallback
    if host.endswith("supremecourt.ohio.gov"):
        return "Ohio Court of Appeals", "Ohio"

    return "", ""


def pick_citation_from_header(text: str) -> str:
    """Return the first reporter-style citation I detect in header text."""
    if not text:
        return ""
    m = REPORTER_REGEX.search(text)
    return m.group(1) if m else ""


def detect_per_curiam(text: str) -> int:
    """Return 1 if 'per curiam' appears in text, else 0."""
    return int(bool(text and PER_CURIAM_RE.search(text)))


def outcome_from_text(text: str) -> Tuple[str, int]:
    """
    Classify the opinion's outcome based on keyword hits in the last ~200 lines.
    I return (label, fine_code) where fine_code follows:
      0=Loss, 1=Win, 2=Mixed, 3=Partial, 4=Settlement, 5=Other
    """
    if not text:
        return ("Other", 5)
    tail = "\n".join(text.splitlines()[-200:])
    hits = {name for (rx, name) in OUTCOME_PATTERNS if rx.search(tail)}
    if "affirmed" in hits and "reversed" in hits:
        return ("Mixed", 2)
    if "affirmed" in hits:
        return ("Win", 1)
    if "reversed" in hits:
        return ("Loss", 0)
    if "remanded" in hits:
        return ("Partial", 3)
    if "dismissed" in hits or "vacated" in hits:
        return ("Other", 5)
    return ("Other", 5)

def limited_online_court_lookup(court_url: str, logger: logging.Logger) -> str:
    """
    Fetch a court's full name from CourtListener /courts/ endpoint.
    I cap requests globally and cache responses to avoid re-fetching.
    """
    global _court_lookups_done
    if not court_url or _court_lookups_done >= MAX_ONLINE_COURT_LOOKUPS:
        return ""
    data = _http_get_json(court_url, logger)
    if data:
        _court_lookups_done += 1
    return (data.get("full_name") or data.get("name_abbreviation") or data.get("name") or "").strip()


def _best_citation_from_citations_list(citations: Any) -> str:
    """
    From a CourtListener citations array, pick the best reporter-style cite.
    I prefer 'official' or 'federal/state' types if available.
    """
    if not isinstance(citations, list):
        return ""
    preferred = ("official", "official_parallel", "federal", "state")
    for c in citations:
        try:
            typ = str((c or {}).get("type", "")).lower()
            cite = str((c or {}).get("cite", "")).strip()
        except Exception:
            continue
        if typ in preferred and REPORTER_REGEX.search(cite):
            return cite
    # fallback passes
    for c in citations:
        try:
            cite = str((c or {}).get("cite", "")).strip()
        except Exception:
            continue
        if REPORTER_REGEX.search(cite):
            return cite
    for c in citations:
        try:
            cite = str((c or {}).get("cite", "")).strip()
        except Exception:
            continue
        if cite:
            return cite
    return ""


def limited_online_cluster_citation(cluster_url: str, logger: logging.Logger) -> str:
    """
    Fetch a cluster's citations from CourtListener and pick the best one.
    I cap lookups globally to respect API limits.
    """
    global _cluster_lookups_done
    if not cluster_url or _cluster_lookups_done >= MAX_ONLINE_CLUSTER_LOOKUPS:
        return ""
    url = cluster_url
    if "?" not in url:
        url = f"{url}?fields=citations"
    data = _http_get_json(url, logger)
    if not data:
        return ""
    _cluster_lookups_done += 1
    cite = _best_citation_from_citations_list(data.get("citations"))
    return cite or ""


def _collect_from_opinions_list(base_url: str, logger: logging.Logger) -> List[str]:
    """
    Paginate through /opinions/ for a cluster and collect plain_text/html bodies.
    I limit to 5 pages defensively.
    """
    global _cluster_lookups_done
    texts: List[str] = []
    url = base_url
    for _ in range(5):
        data = _http_get_json(url, logger)
        if not data:
            break
        _cluster_lookups_done += 1
        for op in data.get("results") or []:
            txt = (op.get("plain_text") or "").strip()
            if not txt:
                txt = _strip_html(op.get("html") or "")
            if txt:
                texts.append(txt)
        nxt = data.get("next") or ""
        if not nxt:
            break
        url = nxt
    return texts


def _cluster_texts_via_cluster(cluster_url: str, logger: logging.Logger) -> List[str]:
    """
    Fetch opinion_texts directly from the cluster resource.
    This is a second path I try if the opinions list gave me nothing.
    """
    global _cluster_lookups_done
    cid = _cluster_id_from_url(cluster_url)
    if not cid:
        return []
    url = f"https://www.courtlistener.com/api/rest/v4/clusters/{cid}/?fields=opinion_texts"
    data = _http_get_json(url, logger) or {}
    if data:
        _cluster_lookups_done += 1
    out: List[str] = []
    for t in (data.get("opinion_texts") or []):
        try:
            txt = str((t or {}).get("text", "")).strip()
        except Exception:
            txt = ""
        if txt:
            out.append(txt)
    return out


def limited_online_cluster_texts(cluster_url: str, logger: logging.Logger) -> List[str]:
    """
    Fetch opinion texts for a given cluster using BOTH paths:
      1) /opinions/?cluster=… (plain_text or stripped html)
      2) /clusters/{id}/ (opinion_texts array)
    I cap requests globally to avoid hammering CourtListener.
    """
    global _cluster_lookups_done
    if not cluster_url or _cluster_lookups_done >= MAX_ONLINE_CLUSTER_LOOKUPS:
        return []

    cid = _cluster_id_from_url(cluster_url)
    texts: List[str] = []

    # Path 1: opinions list
    if cid:
        base1 = f"https://www.courtlistener.com/api/rest/v4/opinions/?cluster={cid}&page_size=100&fields=plain_text,html"
        texts.extend(_collect_from_opinions_list(base1, logger))

    # Path 2: cluster opinion_texts
    if not texts:
        texts.extend(_cluster_texts_via_cluster(cluster_url, logger))

    return texts