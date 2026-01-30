# -----------------------------------------------------------------------------
# What this file does
# 1) Call the CourtListener API to fetch opinions about a topic.
# 2) Pull useful fields (case name, court, citation, links).
# 3) Optionally grab a plain-text URL for later processing.
# 4) Write a CSV to data/extracted/raw_data.csv (+ periodic partial CSVs).
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import re
import time
import json
import random
import logging
import shelve
from typing import Any, Dict, Optional, List
from urllib.parse import urlparse, urlunparse

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# CONFIG
DEFAULT_FAST_MODE = True                 # fast by default: fewer text ops
API_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"
CHECKPOINT_EVERY = 200                   # write partial CSV every N rows
CACHE_FILE = "data/extracted/http_cache.db"  # shelve DB (single file)

# Diagnostics / guardrails
SKIP_HOPS = False               # True -> skip court lookups entirely
INCLUDE_DOCKET_HOP = True      # False -> skip docket lookups (faster)
SLOW_OP_THRESHOLD_SEC = 5.0     # log an opinion if processing takes longer

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("COURTLISTENER_API_KEY")
EMAIL   = os.getenv("COURTLISTENER_EMAIL")

OUTPUT_COLUMNS = [
    "case_id","date_created","date_modified","opinion_year","page_count","per_curiam","type",
    "author_id","joined_by_count","case_name","citation","court","jurisdiction_state",
    "text_char_count","text_word_count","label_heuristic","plain_text_present","plain_text_url",
    "download_url","cluster_url","court_url","has_citation","has_court","is_per_curiam",
    "outcome_code"
]

# SMALL HELPERS
def _normalize_url(u: Any) -> str:
    """
    Normalize any URL-ish value into a safe, absolute URL string.

    If I get an integer-like value or a relative API path, I return a usable
    CourtListener URL. This makes downstream fetches robust.
    """
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
    """
    Normalize a CourtListener API reference into a full URL.

    Works whether I start with:
      - an integer id,
      - a relative API path,
      - or a full URL.
    """
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


def _pick_citation(cites) -> str:
    """
    Choose a citation string from a CourtListener citations field.

    I prefer 'official' citations when available, otherwise I take the first
    usable citation in the list or dict.
    """
    if cites is None:
        return ""
    if isinstance(cites, list) and cites:
        dict_items = [c for c in cites if isinstance(c, dict)]
        if dict_items:
            official = next((c for c in dict_items if str(c.get("type","")).lower()=="official"), None)
            return str((official or dict_items[0]).get("cite","")) or ""
        scalar = next((c for c in cites if isinstance(c, (str,int,float))), None)
        return "" if scalar is None else str(scalar)
    if isinstance(cites, dict):
        return str(cites.get("cite","")) or ""
    if isinstance(cites, (str,int,float)):
        return str(cites)
    return ""


def _infer_state_from_court_name(court_name: str) -> str:
    """
    Infer a state name from a court string.
    Returns 'Federal' if the court mentions U.S.
    """
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


def _retry_get_json(session: requests.Session, url: Optional[str], logger: logging.Logger, attempts: int = 3, backoff: float = 0.6) -> Dict[str, Any]:
    """
    Fetch JSON with retries + exponential backoff.

    I deal with 429 Retry-After headers when present, and log slow calls.
    """
    if not url:
        return {}
    url = _normalize_url(url)
    for i in range(attempts):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 429 and i < attempts - 1:
                # If throttled, I wait per Retry-After header if given
                wait = 10.0
                try:
                    retry_after = r.headers.get("Retry-After")
                    if retry_after:
                        wait = float(retry_after)
                    else:
                        body = r.json()
                        m = re.search(r"(\d+)\s*seconds", str(body.get("detail","")), re.I)
                        if m:
                            wait = float(m.group(1))
                except Exception:
                    pass
                logger.warning("429 on %s; backoff try %d/%d (%.1fs)", url, i+1, attempts, wait)
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


def _page_get_with_backoff(session: requests.Session, url: str, params: dict | None, logger: logging.Logger, max_retries: int = 6) -> requests.Response:
    """
    Fetch a paginated CourtListener API response with retries/backoff.

    I log 429s and 5xx responses with exponential backoff, and stop after too many.
    """
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
                wait = min(wait + (2 ** attempt) * 0.6 + random.uniform(0,1.0), 120.0)
                attempt += 1
                if attempt > max_retries:
                    raise Exception("Too many 429s on page fetch; giving up.")
                logger.warning("429 on page; sleeping %.1fs then retry", wait)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = min((2 ** attempt) * 1.25 + random.uniform(0,1.0), 45.0)
                attempt += 1
                if attempt > max_retries:
                    logger.error("Server error %s after retries. Body: %s", r.status_code, r.text[:300])
                    raise Exception(f"Server error {r.status_code} after retries")
                logger.warning("Server %s. Backoff %.1fs then retry.", r.status_code, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            logger.info("OK (%.1f KB) in %.2fs", len(r.content)/1024.0, dt)
            return r
        except (requests.ReadTimeout, requests.ConnectTimeout, requests.ConnectionError) as e:
            wait = min((2 ** attempt) * 1.15 + random.uniform(0,1.0), 60.0)
            attempt += 1
            if attempt > max_retries:
                logger.error("Network timeout/connection error after retries: %s", e)
                raise
            logger.warning("Timeout/network issue: %s. Retry in %.1fs…", e, wait)
            time.sleep(wait)

# ------------------------------- MAIN ----------------------------------------
def extract_data(
    query: str = "corporation",
    max_cases: int = 1000,
    debug_print_one: bool = False,
    *,
    fast_mode: Optional[bool] = None,
    page_size: Optional[int] = None,
    logger_name: str = "pipeline.extract",
) -> pd.DataFrame:
    """
    Pull opinions from CourtListener matching a keyword search.

    I stream results page by page, follow `next` links, and resolve clusters
    to enrich case_name, court, and citations. Writes checkpoint CSVs every
    CHECKPOINT_EVERY rows and a final CSV to data/extracted/raw_data.csv.

    Args:
        query: keyword to search CourtListener opinions.
        max_cases: maximum number of cases to pull.
        debug_print_one: print the first opinion JSON for inspection.
        fast_mode: if True, skip most text parsing (faster).
        page_size: override API page_size (default depends on fast_mode).
        logger_name: logger name to use.
    """
    logger = logging.getLogger(logger_name)

    if not API_KEY or not EMAIL:
        raise RuntimeError("Set COURTLISTENER_API_KEY and COURTLISTENER_EMAIL in your environment or .env")

    FAST_MODE = DEFAULT_FAST_MODE if fast_mode is None else bool(fast_mode)

    # Session with retries
    headers = {"User-Agent": f"inst414-final-project ({EMAIL})", "Authorization": f"Token {API_KEY}"}
    session = requests.Session()
    session.headers.update(headers)
    retry = Retry(total=6, connect=4, read=4, backoff_factor=0.7,
                  status_forcelist=[429,502,503,504], allowed_methods=["GET"], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    eff_page_size = page_size if page_size is not None else (100 if FAST_MODE else 50)
    params = {"search": query, "page_size": eff_page_size}
    logger.info("Extracting cases from CourtListener API… (page_size=%s)", eff_page_size)

    # Ensure cache dir exists
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    http_cache = shelve.open(CACHE_FILE)

    def _cached_get_json(url: str) -> dict:
        """Use the shelve cache for GET JSON calls."""
        key = f"GET::{url}"
        if key in http_cache:
            logger.debug("Cache hit: %s", url)
            return http_cache[key]
        logger.debug("GET JSON %s", url)
        t0 = time.time()
        data = _retry_get_json(session, url, logger)
        http_cache[key] = data
        logger.debug("DONE %s in %.2fs (%d bytes)", url, time.time()-t0, len(json.dumps(data)) if data else 0)
        return data

    court_cache: Dict[str, Dict[str, Any]] = {}
    cluster_cache: Dict[str, Dict[str, Any]] = {}
    docket_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    url = API_URL
    first = True
    printed_debug = False
    page_idx = 0

    try:
        while url and len(rows) < max_cases:
            r = _page_get_with_backoff(session, url, params if first else None, logger)
            payload = r.json()
            first = False
            page_idx += 1

            results = payload.get("results", []) or []
            logger.info("Page %d: got %d results | next=%s", page_idx, len(results), "yes" if payload.get("next") else "no")

            for op in results:
                t_op = time.time()

                if debug_print_one and not printed_debug:
                    print("\n=== DEBUG: first opinion payload ===")
                    print("keys:", sorted(list(op.keys())))
                    print(json.dumps(op, indent=2)[:3000])  # truncate so it doesn't spam
                    print("=== END DEBUG ===\n")
                    printed_debug = True
                    return pd.DataFrame()  # stop early; avoids writing files

                if len(rows) >= max_cases:
                    break

                # ---- pull fields safely ----
                case_id        = op.get("id")
                date_created   = op.get("date_created", "")
                date_modified  = op.get("date_modified", "")
                page_count     = op.get("page_count", None)
                per_curiam     = bool(op.get("per_curiam", False))
                optype         = op.get("type", "")
                author_id      = op.get("author_id", op.get("author"))
                joined_by_cnt  = len(op.get("joined_by", []) or [])
                download_url   = op.get("download_url", "")

                # cluster may be id/int, relative, or full URL
                cluster_raw    = op.get("cluster") or ""
                cluster_url    = _normalize_api_url(cluster_raw, "clusters") if cluster_raw else ""

                opinion_cites  = op.get("citations", None)

                m = re.match(r"^(\d{4})-", date_created or "")
                opinion_year = int(m.group(1)) if m else None

                raw_plain = op.get("plain_text") or ""
                plain_text_present = int(bool(raw_plain))
                plain_text_url = raw_plain if (isinstance(raw_plain, str) and raw_plain.startswith(("http://","https://"))) else ""
                text = "" if FAST_MODE else (raw_plain if isinstance(raw_plain, str) and not plain_text_url else "")

                # Cluster enrichment
                cluster: Dict[str, Any] = {}
                if cluster_url:
                    ckey = _normalize_url(cluster_url)
                    cluster = cluster_cache.get(ckey) or _cached_get_json(ckey)
                    cluster_cache[ckey] = cluster

                case_name = (cluster.get("caseName") or cluster.get("case_name") or cluster.get("caption") or
                             op.get("case_name","") or "")

                # CourtListener v4 commonly uses "citation" (singular) on clusters.
                # Some older/alt payloads may use "citations", so we support both.
                cluster_citations = cluster.get("citation", None)
                if cluster_citations is None:
                    cluster_citations = cluster.get("citations", None)

                all_citations = cluster_citations if cluster_citations is not None else opinion_cites
                
                citation = _pick_citation(all_citations) or "Unknown"
                has_citation = int(citation != "Unknown")

                # Court resolution
                court_name = ""
                court_url  = ""

                if not SKIP_HOPS:
                    # A) cluster -> court (fastest)
                    if not court_name:
                        clu_court = (cluster.get("court") or cluster.get("court_id") or op.get("court") or op.get("court_id") or "")
                        if isinstance(clu_court, (str, int)) and str(clu_court).strip():
                            court_api = _normalize_api_url(clu_court, "courts")
                            ccu = _normalize_url(court_api)
                            if ccu:
                                cj = court_cache.get(ccu) or _cached_get_json(ccu)
                                court_cache[ccu] = cj
                                court_name = cj.get("full_name") or cj.get("name_abbreviation") or cj.get("name") or ""
                                court_url = ccu
                        elif isinstance(clu_court, dict):
                            court_name = (clu_court.get("full_name") or
                                          clu_court.get("name_abbreviation") or
                                          clu_court.get("name") or "")
                            court_url = _normalize_api_url(clu_court.get("resource_uri",""), "courts") if clu_court.get("resource_uri") else court_url

                    # B) (optional) docket hop
                    if INCLUDE_DOCKET_HOP and not court_name:
                        docket_val = (cluster.get("docket") or cluster.get("docket_id") or op.get("docket") or op.get("docket_id") or "")
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
                                court_name = (d_court.get("full_name") or
                                              d_court.get("name_abbreviation") or
                                              d_court.get("name") or "")
                                court_url = _normalize_api_url(d_court.get("resource_uri",""), "courts") if d_court.get("resource_uri") else court_url

                court_name = re.sub(r"\s+", " ", (court_name or "")).strip()
                jurisdiction_state = _infer_state_from_court_name(court_name) or "Unknown"
                has_court = int(bool(court_name and court_name != "Unknown"))

                # Heuristic labeling if not in FAST_MODE
                if FAST_MODE:
                    text_char_count = 0
                    text_word_count = 0
                    label_heuristic = "other"
                else:
                    text_char_count = len(text) if text else 0
                    text_word_count = len(re.findall(r"\w+", text)) if text else 0
                    t = (text or "").lower()
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
                    "case_id": case_id,
                    "date_created": date_created,
                    "date_modified": date_modified,
                    "opinion_year": opinion_year,
                    "page_count": page_count,
                    "per_curiam": per_curiam,
                    "type": optype,
                    "author_id": author_id,
                    "joined_by_count": joined_by_cnt,
                    "case_name": case_name,
                    "citation": citation if citation else "Unknown",
                    "court": court_name if court_name else "Unknown",
                    "jurisdiction_state": jurisdiction_state,
                    "text_char_count": text_char_count,
                    "text_word_count": text_word_count,
                    "label_heuristic": label_heuristic,
                    "plain_text_present": plain_text_present,
                    "plain_text_url": plain_text_url,
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

            time.sleep(0.15)
            url = payload.get("next")
            params = None

        # Final write
        df = pd.DataFrame(rows)
        os.makedirs("data/extracted", exist_ok=True)
        outpath = "data/extracted/raw_data.csv"
        df.to_csv(outpath, index=False, columns=OUTPUT_COLUMNS)
        logger.info("Extracted %d cases to %s", len(df), outpath)
        return df

    finally:
        try:
            http_cache.close()
        except Exception:
            pass

# Manual test
if __name__ == "__main__":
    extract_data(query="corporation", max_cases=200, fast_mode=True, page_size=100)