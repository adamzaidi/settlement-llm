# -----------------------------------------------------------------------------
# What this file does
# 1) Call the CourtListener API to fetch opinions about a topic.
# 2) Pull useful fields (case name, court, citation, links).
# 3) Optionally grab a plain-text URL (or inline plain_text) for later processing.
# 4) Write a CSV to data/extracted/raw_data.csv (+ periodic partial CSVs).
#
# IMPORTANT COURTLISTENER COMPLIANCE NOTES
# - Do keyword searching ONLY via /search/ using the "q" parameter.
# - Do NOT pass invalid params like "search" or "id__in" to /opinions/.
# - This script enforces a hard max_cases row limit AND an optional hard page cap.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import re
import time
import json
import random
import logging
import shelve
from typing import Any, Dict, Optional, List, Tuple
from urllib.parse import urlparse, urlunparse

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# CONFIG
DEFAULT_FAST_MODE = False  # fast by default: fewer text ops

# --- CourtListener endpoints ---
SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
OPINION_DETAIL_URL = "https://www.courtlistener.com/api/rest/v4/opinions/{id}/"

CHECKPOINT_EVERY = 200
CACHE_FILE = "data/extracted/http_cache.db"  # shelve DB (single file)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Snippet controls
TEXT_SNIPPET_CHARS = int(os.getenv("TEXT_SNIPPET_CHARS", "12000"))
SNIPPET_FETCH_SLEEP = float(os.getenv("SNIPPET_FETCH_SLEEP", "0.10"))
SNIPPET_FETCH_MAX = int(os.getenv("SNIPPET_FETCH_MAX", "450"))

# CHANGE: prefer tail snippet for outcome labeling; keep head too for context/EDA
TEXT_SNIPPET_HEAD_CHARS = int(os.getenv("TEXT_SNIPPET_HEAD_CHARS", str(TEXT_SNIPPET_CHARS)))
TEXT_SNIPPET_TAIL_CHARS = int(os.getenv("TEXT_SNIPPET_TAIL_CHARS", str(TEXT_SNIPPET_CHARS)))

# Diagnostics / guardrails
SKIP_HOPS = False
INCLUDE_DOCKET_HOP = True
SLOW_OP_THRESHOLD_SEC = 5.0

MAX_SEARCH_PAGES = int(os.getenv("MAX_SEARCH_PAGES", "0")) or None

API_KEY = os.getenv("COURTLISTENER_API_KEY")
EMAIL = os.getenv("COURTLISTENER_EMAIL")

# CHANGE: persist snippet-text caching across runs (separate shelve so you can wipe independently)
SNIPPET_CACHE_FILE = os.getenv("SNIPPET_CACHE_FILE", "data/extracted/snippet_cache.db")

# CHANGE: add head/tail snippet columns; keep text_snippet for backward compatibility
OUTPUT_COLUMNS = [
    "opinion_id", "opinion_api_url",
    "case_id", "date_created", "date_modified", "opinion_year", "page_count", "per_curiam", "type",
    "author_id", "joined_by_count", "case_name", "citation", "court", "jurisdiction_state",
    "text_char_count", "text_word_count", "label_heuristic",
    "plain_text_present", "plain_text_url",
    "text_snippet_head", "text_snippet_tail", "text_snippet",
    "download_url", "cluster_url", "court_url", "has_citation", "has_court", "is_per_curiam",
    "outcome_code"
]

# -----------------------------------------------------------------------------
# SMALL HELPERS
# -----------------------------------------------------------------------------
def _normalize_url(u: Any) -> str:
    """Normalize any URL-ish value into a safe absolute URL string."""
    if u is None:
        return ""
    s = str(u).strip()
    if not s:
        return ""
    if s.isdigit():
        return s
    if s.startswith("/api/"):
        s = "https://www.courtlistener.com" + s
    if s.startswith(("http://", "https://")):
        p = urlparse(s)
        path = p.path if p.path.endswith("/") else p.path + "/"
        return urlunparse((p.scheme, p.netloc, path, "", "", ""))
    return s


def _normalize_api_url(val: Any, resource: str) -> str:
    """Normalize a CourtListener API reference into a full URL."""
    base_root = "https://www.courtlistener.com"
    if val is None:
        return ""
    raw = str(val).strip()
    if not raw:
        return ""
    if raw.isdigit():
        return f"{base_root}/api/rest/v4/{resource}/{raw}/"
    if raw.startswith("/api/"):
        return f"{base_root}{raw if raw.endswith('/') else raw + '/'}"
    if raw.startswith(("http://", "https://")):
        p = urlparse(raw)
        path = p.path if p.path.endswith("/") else p.path + "/"
        return urlunparse((p.scheme, p.netloc, path, "", "", ""))
    token = raw.strip("/").split("/")[-1]
    token = token if token else raw
    return f"{base_root}/api/rest/v4/{resource}/{token}/"


def _split_plain_text(raw_plain: Any) -> Tuple[str, str]:
    """
    CourtListener 'plain_text' can be either:
      1) a URL string (absolute or relative '/download/...')
      2) the actual opinion text (inline)
    Returns: (plain_text_url, plain_text_inline)
    """
    if not raw_plain or not isinstance(raw_plain, str):
        return ("", "")

    rp = raw_plain.strip()
    if not rp:
        return ("", "")

    if rp.startswith(("http://", "https://")):
        return (rp, "")
    if rp.startswith("/"):
        return ("https://www.courtlistener.com" + rp, "")

    return ("", rp)


def _pick_citation(cites: Any) -> str:
    if cites is None:
        return ""

    if isinstance(cites, list):
        items = cites
    elif isinstance(cites, dict):
        items = [cites]
    elif isinstance(cites, (str, int, float)):
        return str(cites).strip()
    else:
        return ""

    dict_items = [c for c in items if isinstance(c, dict)]
    if not dict_items:
        scalar = next((c for c in items if isinstance(c, (str, int, float))), None)
        return "" if scalar is None else str(scalar).strip()

    def _score(c: dict) -> tuple:
        has_cite = int(bool(c.get("cite")))
        has_parts = int(bool(c.get("volume") and c.get("reporter") and c.get("page")))
        t = c.get("type")
        try:
            t_val = int(t)
        except Exception:
            t_val = 999
        return (-has_cite, -has_parts, t_val)

    dict_items.sort(key=_score)
    best = dict_items[0]

    if best.get("cite"):
        return str(best.get("cite")).strip()

    vol = str(best.get("volume", "")).strip()
    rep = str(best.get("reporter", "")).strip()
    page = str(best.get("page", "")).strip()

    return " ".join([x for x in (vol, rep, page) if x]).strip()


def _infer_state_from_court_name(court_name: str) -> str:
    if not court_name:
        return ""
    if re.search(r"U\.S\.", court_name, re.I):
        return "Federal"
    m = re.search(
        r"(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|"
        r"Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|"
        r"Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|"
        r"New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|"
        r"Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
        r"West\s+Virginia|Wisconsin|Wyoming)",
        court_name, re.I
    )
    if m:
        return re.sub(r"\s+", " ", m.group(0)).title()
    return ""


def _retry_get_json(
    session: requests.Session,
    url: Optional[str],
    logger: logging.Logger,
    attempts: int = 3,
    backoff: float = 0.6
) -> Dict[str, Any]:
    if not url:
        return {}
    url = _normalize_url(url)
    for i in range(attempts):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 429 and i < attempts - 1:
                wait = 10.0
                try:
                    retry_after = r.headers.get("Retry-After")
                    if retry_after:
                        wait = float(retry_after)
                    else:
                        body = r.json()
                        m = re.search(r"(\d+)\s*seconds", str(body.get("detail", "")), re.I)
                        if m:
                            wait = float(m.group(1))
                except Exception:
                    pass
                logger.warning("429 on %s; backoff try %d/%d (%.1fs)", url, i + 1, attempts, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = backoff * (2 ** i)
            logger.debug("GET failed (%s) on %s; retrying in %.1fs", e, url, wait)
            time.sleep(wait)
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("GET JSON final failure on %s: %s", url, e)
        return {}


def _page_get_with_backoff(
    session: requests.Session,
    url: str,
    params: dict | None,
    logger: logging.Logger,
    max_retries: int = 6
) -> requests.Response:
    attempt = 0
    while True:
        try:
            logger.info("Fetching page: %s params=%s", url, params)
            t0 = time.time()
            r = session.get(url, params=params, timeout=45)
            dt = time.time() - t0

            if r.status_code == 429:
                wait = 10.0
                try:
                    retry_after = r.headers.get("Retry-After")
                    if retry_after:
                        wait = float(retry_after)
                except Exception:
                    pass
                wait = min(wait + (2 ** attempt) * 0.6 + random.uniform(0, 1.0), 120.0)
                attempt += 1
                if attempt > max_retries:
                    raise Exception("Too many 429s on page fetch; giving up.")
                logger.warning("429 on page; sleeping %.1fs then retry", wait)
                time.sleep(wait)
                continue

            if r.status_code >= 500:
                wait = min((2 ** attempt) * 1.25 + random.uniform(0, 1.0), 45.0)
                attempt += 1
                if attempt > max_retries:
                    logger.error("Server error %s after retries. Body: %s", r.status_code, r.text[:300])
                    raise Exception(f"Server error {r.status_code} after retries")
                logger.warning("Server %s. Backoff %.1fs then retry.", r.status_code, wait)
                time.sleep(wait)
                continue

            r.raise_for_status()
            logger.info("OK (%.1f KB) in %.2fs", len(r.content) / 1024.0, dt)
            return r

        except (requests.ReadTimeout, requests.ConnectTimeout, requests.ConnectionError) as e:
            wait = min((2 ** attempt) * 1.15 + random.uniform(0, 1.0), 60.0)
            attempt += 1
            if attempt > max_retries:
                logger.error("Network timeout/connection error after retries: %s", e)
                raise
            logger.warning("Timeout/network issue: %s. Retry in %.1fs…", e, wait)
            time.sleep(wait)


def _strip_html(s: str) -> str:
    """Very lightweight HTML -> text converter."""
    if not s:
        return ""
    s = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_for_csv(s: str) -> str:
    # CHANGE: keep CSV/Sheets sane (your sample showed multi-line snippet)
    if not s:
        return ""
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


# CHANGE: central helper to compute (head, tail, default_text_snippet)
def _build_head_tail_snippets(full_text: str) -> tuple[str, str, str]:
    """
    Returns:
      (text_snippet_head, text_snippet_tail, text_snippet_default)

    Default is tail, because disposition/status language is typically in the last pages/paragraphs.
    """
    if not full_text:
        return "", "", ""
    t = _sanitize_for_csv(full_text)
    head = t[:TEXT_SNIPPET_HEAD_CHARS] if TEXT_SNIPPET_HEAD_CHARS > 0 else ""
    tail = t[-TEXT_SNIPPET_TAIL_CHARS:] if TEXT_SNIPPET_TAIL_CHARS > 0 else ""
    default = tail or head  # prefer tail; fallback to head if tail empty
    return head, tail, default


def _fetch_text_snippet(session: requests.Session, url: str, logger: logging.Logger, snippet_cache) -> str:
    """
    Fetch a snippet of text from a plain_text_url.
    Cached so repeated runs do not re-hit CourtListener.

    NOTE: we still fetch the full TEXT_SNIPPET_CHARS slice here (like before),
    then later split into head/tail for storage.
    """
    if not url or not url.startswith(("http://", "https://")):
        return ""

    cache_key = f"SNIP::{url}"
    try:
        if cache_key in snippet_cache:
            return snippet_cache[cache_key]
    except Exception:
        pass

    try:
        r = session.get(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if not (ctype.startswith("text/") or "html" in ctype):
            return ""
        txt = (r.text or "").strip()
        if "html" in ctype:
            txt = _strip_html(txt)
        if len(txt) > TEXT_SNIPPET_CHARS:
            txt = txt[:TEXT_SNIPPET_CHARS]
        txt = _sanitize_for_csv(txt)

        try:
            snippet_cache[cache_key] = txt
        except Exception:
            pass

        return txt
    except Exception as e:
        logger.debug("Snippet fetch failed for %s: %s", url, e)
        return ""


def _extract_opinion_id_from_search_hit(hit: Dict[str, Any]) -> Optional[int]:
    """
    /search/?type=o returns cluster-like hits.
    Opinion ids are typically in hit["opinions"].
    """
    opinions = hit.get("opinions")
    if isinstance(opinions, list) and opinions:
        for item in opinions:
            if isinstance(item, int) and item > 0:
                return item
            if isinstance(item, str):
                s = item.strip()
                if s.isdigit():
                    return int(s)
                m = re.search(r"/opinions?/(\d+)/", s)
                if m:
                    return int(m.group(1))

    for k in ("absolute_url", "resource_uri", "cluster", "opinion"):
        s = str(hit.get(k) or "")
        m = re.search(r"/opinions?/(\d+)/", s)
        if m:
            return int(m.group(1))

    oid = hit.get("id")
    if isinstance(oid, int) and oid > 0:
        return oid
    if isinstance(oid, str) and oid.strip().isdigit():
        return int(oid.strip())

    return None


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def extract_data(
    query: str = "corporation",
    max_cases: int = 1000,
    debug_print_one: bool = False,
    *,
    fast_mode: Optional[bool] = None,
    page_size: Optional[int] = None,
    logger_name: str = "pipeline.extract",
) -> pd.DataFrame:
    logger = logging.getLogger(logger_name)

    if not API_KEY or not EMAIL:
        raise RuntimeError("Set COURTLISTENER_API_KEY and COURTLISTENER_EMAIL in your environment or .env")

    # Strict max_cases
    try:
        max_cases = int(max_cases)
    except Exception:
        max_cases = 1000
    max_cases = max(1, max_cases)

    FAST_MODE = DEFAULT_FAST_MODE if fast_mode is None else bool(fast_mode)

    headers = {"User-Agent": f"inst414-final-project ({EMAIL})", "Authorization": f"Token {API_KEY}"}
    session = requests.Session()
    session.headers.update(headers)

    retry = Retry(
        total=6,
        connect=4,
        read=4,
        backoff_factor=0.7,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    eff_page_size = page_size if page_size is not None else (100 if FAST_MODE else 50)
    params = {"q": query, "type": "o", "page_size": eff_page_size}
    logger.info(
        "Searching opinions from CourtListener /search/… (page_size=%s, max_cases=%s, max_pages=%s)",
        eff_page_size, max_cases, MAX_SEARCH_PAGES if MAX_SEARCH_PAGES is not None else "None"
    )

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    http_cache = shelve.open(CACHE_FILE)

    # open snippet cache once per run
    os.makedirs(os.path.dirname(SNIPPET_CACHE_FILE), exist_ok=True)
    snippet_cache = shelve.open(SNIPPET_CACHE_FILE)

    def _cached_get_json(url: str) -> dict:
        key = f"GET::{url}"
        if key in http_cache:
            logger.debug("Cache hit: %s", url)
            return http_cache[key]
        data = _retry_get_json(session, url, logger)
        http_cache[key] = data
        return data

    court_cache: Dict[str, Dict[str, Any]] = {}
    cluster_cache: Dict[str, Dict[str, Any]] = {}
    docket_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    # dedupe at the extractor level to avoid refetching the same opinion
    seen_opinion_ids: set[int] = set()

    url = SEARCH_URL
    first = True
    printed_debug = False
    page_idx = 0
    snippet_fetches = 0

    try:
        while url and len(rows) < max_cases:
            if MAX_SEARCH_PAGES is not None and page_idx >= MAX_SEARCH_PAGES:
                logger.info("Reached MAX_SEARCH_PAGES=%d; stopping pagination.", MAX_SEARCH_PAGES)
                break

            r = _page_get_with_backoff(session, url, params if first else None, logger)
            payload = r.json()
            first = False
            page_idx += 1

            results = payload.get("results", []) or []
            logger.info(
                "Search page %d: got %d results | rows=%d/%d | next=%s",
                page_idx, len(results), len(rows), max_cases, "yes" if payload.get("next") else "no"
            )

            rows_before_page = len(rows)
            skipped_no_id = 0
            skipped_seen = 0
            skipped_no_op = 0

            for hit in results:
                if len(rows) >= max_cases:
                    break

                op_id = _extract_opinion_id_from_search_hit(hit)
                if not op_id:
                    skipped_no_id += 1
                    if (page_idx <= 3) or (random.random() < 0.02):
                        logger.warning(
                            "Skipping search hit: could not extract opinion id. keys=%s",
                            sorted(list(hit.keys()))
                        )
                    continue

                if op_id in seen_opinion_ids:
                    skipped_seen += 1
                    continue
                seen_opinion_ids.add(op_id)

                t_op = time.time()

                op_url = OPINION_DETAIL_URL.format(id=op_id)
                op = _cached_get_json(op_url)
                if not op:
                    skipped_no_op += 1
                    continue

                if debug_print_one and not printed_debug:
                    print("\n=== DEBUG: first opinion payload ===")
                    print("keys:", sorted(list(op.keys())))
                    print(json.dumps(op, indent=2)[:3000])
                    print("=== END DEBUG ===\n")
                    printed_debug = True
                    return pd.DataFrame()

                # NOTE: CourtListener opinion object's "id" is the opinion id.
                case_id = op.get("id")  # "case_id" here is opinion-level id in your current dataset

                date_created = op.get("date_created", "")
                date_modified = op.get("date_modified", "")
                page_count = op.get("page_count", None)
                per_curiam = bool(op.get("per_curiam", False))
                optype = op.get("type", "")
                author_id = op.get("author_id", op.get("author"))
                joined_by_cnt = len(op.get("joined_by", []) or [])
                download_url = op.get("download_url", "")

                cluster_raw = op.get("cluster") or ""
                cluster_url = _normalize_api_url(cluster_raw, "clusters") if cluster_raw else ""
                opinion_cites = op.get("citations", None)

                m = re.match(r"^(\d{4})-", date_created or "")
                opinion_year = int(m.group(1)) if m else None

                # --- TEXT: robust, fast-first strategy ---
                raw_plain = op.get("plain_text") or ""
                plain_text_url, plain_text_inline = _split_plain_text(raw_plain)

                plain_text_present = int(bool(plain_text_url or plain_text_inline))

                # CHANGE: store head/tail; default snippet is tail
                text_snippet_head = ""
                text_snippet_tail = ""
                text_snippet = ""  # default used by transform

                if not FAST_MODE:
                    full_text_for_snippets = ""

                    if plain_text_inline:
                        full_text_for_snippets = plain_text_inline
                    elif plain_text_url and snippet_fetches < SNIPPET_FETCH_MAX:
                        time.sleep(SNIPPET_FETCH_SLEEP + random.uniform(0, 0.08))
                        full_text_for_snippets = _fetch_text_snippet(session, plain_text_url, logger, snippet_cache)
                        if full_text_for_snippets:
                            snippet_fetches += 1
                    else:
                        raw_html = (op.get("html_with_citations") or op.get("html") or "")
                        if isinstance(raw_html, str) and raw_html.strip():
                            full_text_for_snippets = _strip_html(raw_html)

                    # Build head/tail + default snippet (tail) from whatever text we got
                    text_snippet_head, text_snippet_tail, text_snippet = _build_head_tail_snippets(full_text_for_snippets)

                # Cluster enrichment
                cluster: Dict[str, Any] = {}
                if cluster_url:
                    ckey = _normalize_url(cluster_url)
                    cluster = cluster_cache.get(ckey) or _cached_get_json(ckey)
                    cluster_cache[ckey] = cluster

                case_name = (
                    cluster.get("caseName")
                    or cluster.get("case_name")
                    or cluster.get("caption")
                    or op.get("case_name", "")
                    or ""
                )

                cluster_citations = cluster.get("citation", None)
                if cluster_citations is None:
                    cluster_citations = cluster.get("citations", None)
                all_citations = cluster_citations if cluster_citations is not None else opinion_cites

                citation = _pick_citation(all_citations) or "Unknown"
                has_citation = int(citation != "Unknown")

                # Court resolution
                court_name = ""
                court_url = ""

                if not SKIP_HOPS:
                    if not court_name:
                        clu_court = (
                            cluster.get("court") or cluster.get("court_id")
                            or op.get("court") or op.get("court_id")
                            or ""
                        )
                        if isinstance(clu_court, (str, int)) and str(clu_court).strip():
                            court_api = _normalize_api_url(clu_court, "courts")
                            ccu = _normalize_url(court_api)
                            if ccu:
                                cj = court_cache.get(ccu) or _cached_get_json(ccu)
                                court_cache[ccu] = cj
                                court_name = cj.get("full_name") or cj.get("name_abbreviation") or cj.get("name") or ""
                                court_url = ccu
                        elif isinstance(clu_court, dict):
                            court_name = (
                                clu_court.get("full_name")
                                or clu_court.get("name_abbreviation")
                                or clu_court.get("name")
                                or ""
                            )
                            if clu_court.get("resource_uri"):
                                court_url = _normalize_api_url(clu_court.get("resource_uri", ""), "courts")

                    if INCLUDE_DOCKET_HOP and not court_name:
                        docket_val = (
                            cluster.get("docket") or cluster.get("docket_id")
                            or op.get("docket") or op.get("docket_id")
                            or ""
                        )
                        if isinstance(docket_val, (str, int)) and str(docket_val).strip():
                            dcu = _normalize_api_url(docket_val, "dockets")
                            docket = docket_cache.get(dcu) or _cached_get_json(dcu)
                            docket_cache[dcu] = docket
                            d_court = docket.get("court")
                            if isinstance(d_court, (str, int)) and str(d_court).strip():
                                dcu2 = _normalize_api_url(d_court, "courts")
                                cj = court_cache.get(dcu2) or _cached_get_json(dcu2)
                                court_cache[dcu2] = cj
                                court_name = cj.get("full_name") or cj.get("name_abbreviation") or cj.get("name") or ""
                                court_url = dcu2
                            elif isinstance(d_court, dict):
                                court_name = (
                                    d_court.get("full_name")
                                    or d_court.get("name_abbreviation")
                                    or d_court.get("name")
                                    or ""
                                )
                                if d_court.get("resource_uri"):
                                    court_url = _normalize_api_url(d_court.get("resource_uri", ""), "courts")

                court_name = re.sub(r"\s+", " ", (court_name or "")).strip()
                jurisdiction_state = _infer_state_from_court_name(court_name) or "Unknown"
                has_court = int(bool(court_name and court_name != "Unknown"))

                # Heuristic labeling (uses default text_snippet which is tail)
                if FAST_MODE:
                    text_char_count = 0
                    text_word_count = 0
                    label_heuristic = "other"
                else:
                    text_char_count = len(text_snippet) if text_snippet else 0
                    text_word_count = len(re.findall(r"\w+", text_snippet)) if text_snippet else 0
                    t = (text_snippet or "").lower()
                    if "affirmed" in t and "reversed" in t:
                        label_heuristic = "mixed"
                    elif "affirmed" in t:
                        label_heuristic = "affirmed"
                    elif "reversed" in t:
                        label_heuristic = "reversed"
                    elif "remanded" in t:
                        label_heuristic = "remanded"
                    elif "dismissed" in t:
                        label_heuristic = "dismissed"
                    else:
                        label_heuristic = "other"

                rows.append({
                    "opinion_id": op_id,
                    "opinion_api_url": op_url,
                    "case_id": case_id,
                    "date_created": date_created,
                    "date_modified": date_modified,
                    "opinion_year": opinion_year,
                    "page_count": page_count,
                    "per_curiam": per_curiam,
                    "type": optype,
                    "author_id": author_id,
                    "joined_by_count": joined_by_cnt,
                    "case_name": _sanitize_for_csv(case_name),
                    "citation": citation if citation else "Unknown",
                    "court": court_name if court_name else "Unknown",
                    "jurisdiction_state": jurisdiction_state,
                    "text_char_count": text_char_count,
                    "text_word_count": text_word_count,
                    "label_heuristic": label_heuristic,
                    "plain_text_present": plain_text_present,
                    "plain_text_url": plain_text_url,

                    # CHANGE: new head/tail fields + default snippet (tail)
                    "text_snippet_head": text_snippet_head,
                    "text_snippet_tail": text_snippet_tail,
                    "text_snippet": text_snippet,

                    "download_url": download_url or "",
                    "cluster_url": cluster_url or "",
                    "court_url": court_url or "",
                    "has_citation": has_citation,
                    "has_court": has_court,
                    "is_per_curiam": int(per_curiam),
                    "outcome_code": 0,
                })

                op_dt = time.time() - t_op
                if op_dt > SLOW_OP_THRESHOLD_SEC:
                    logger.debug("Slow opinion id=%s took %.2fs", case_id, op_dt)

                if len(rows) % CHECKPOINT_EVERY == 0:
                    tmp = pd.DataFrame(rows)
                    os.makedirs("data/extracted", exist_ok=True)
                    tmp.to_csv("data/extracted/raw_data_partial.csv", index=False, columns=OUTPUT_COLUMNS)
                    logger.info("Checkpoint wrote %d rows -> data/extracted/raw_data_partial.csv", len(rows))

            if len(rows) == rows_before_page:
                logger.warning(
                    "No rows added on page %d (hits=%d, skipped_no_id=%d, skipped_seen=%d, skipped_no_op=%d).",
                    page_idx, len(results), skipped_no_id, skipped_seen, skipped_no_op
                )

            time.sleep(0.15)
            url = payload.get("next")
            params = None

        df = pd.DataFrame(rows)

        # final dedupe safety
        if "opinion_id" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["opinion_id"], keep="first")
            after = len(df)
            if after != before:
                logger.warning("Final dedupe on opinion_id: %d -> %d rows", before, after)

        os.makedirs("data/extracted", exist_ok=True)
        outpath = "data/extracted/raw_data.csv"
        df.to_csv(outpath, index=False, columns=OUTPUT_COLUMNS)
        logger.info("Extracted %d cases to %s", len(df), outpath)
        logger.info("Snippet fetches: %d (cap=%d)", snippet_fetches, SNIPPET_FETCH_MAX)
        return df

    finally:
        try:
            http_cache.close()
        except Exception:
            pass
        try:
            snippet_cache.close()
        except Exception:
            pass


if __name__ == "__main__":
    extract_data(query="corporation", max_cases=200, fast_mode=False, page_size=50)