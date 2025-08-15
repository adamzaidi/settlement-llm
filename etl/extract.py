# What this does
# 1) Calls the CourtListener API to fetch opinions about a topic.
# 2) Pulls useful fields (case name, court, citation, links).
# 3) (Optional) grabs text URL for later processing.
# 4) Writes a CSV to data/extracted/raw_data.csv.
import logging
logger = logging.getLogger("pipeline")

import os
import re
import time
import json
import random
import shelve
from typing import Any, Dict, Optional, List, Tuple
from urllib.parse import urlparse, urlunparse

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FAST_MODE = True # True = faster: skips heavy text downloads
API_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"
CHECKPOINT_EVERY = 200 # writes to a partial CSV every X rows
CACHE_PATH = "data/extracted/_http_cache"  # small on-disk cache for GETs

# for .env: (COURTLISTENER_API_KEY, COURTLISTENER_EMAIL)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("COURTLISTENER_API_KEY")
EMAIL   = os.getenv("COURTLISTENER_EMAIL")

# Output CSV column order (keeps extra columns at the end if they exist)
OUTPUT_COLUMNS = [
    "case_id","date_created","date_modified","opinion_year","page_count","per_curiam","type",
    "author_id","joined_by_count","case_name","citation","court","jurisdiction_state",
    "text_char_count","text_word_count","label_heuristic","plain_text_present","plain_text_url",
    "download_url","cluster_url","court_url","has_citation","has_court","is_per_curiam",
    "outcome_code"
]

# Helpers (URLs, retries, small lookups)
def _normalize_url(u: str) -> str:
    if not u:
        return u
    p = urlparse(u)
    path = p.path if p.path.endswith("/") else p.path + "/"
    return urlunparse((p.scheme, p.netloc, path, "", "", ""))

def _retry_get_json(session: requests.Session, url: Optional[str], attempts: int = 3, backoff: float = 0.6) -> Dict[str, Any]:
    """
    GET JSON with a tiny retry loop (helps with rate limits or network hiccups).
    """
    if not url:
        return {}
    url = _normalize_url(url)
    for i in range(attempts):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 429 and i < attempts - 1:
                time.sleep(backoff * (2 ** i))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(backoff * (2 ** i))
    # last try, then give up
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def _page_get_with_backoff(session: requests.Session, url: str, params: dict | None, max_retries: int = 6) -> requests.Response:
    """
    GET one API page, handles rate-limits (429) and server errors with a backoff.
    """
    attempt = 0
    while True:
        try:
            r = session.get(url, params=params, timeout=45)
            if r.status_code == 429:
                # look for "Retry-After" or just wait a bit
                wait = 10 + (2 ** attempt) * 0.6 + random.uniform(0, 1.0)
                attempt += 1
                if attempt > max_retries:
                    raise Exception("Too many 429s, giving up.")
                print(f"429 rate limit. Sleeping {wait:.1f}s…")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = min((2 ** attempt) * 1.25 + random.uniform(0, 1.0), 45.0)
                attempt += 1
                if attempt > max_retries:
                    print(r.text)
                    raise Exception(f"Server error {r.status_code} after retries")
                print(f"Server {r.status_code}. Backoff {wait:.1f}s…")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except (requests.ReadTimeout, requests.ConnectTimeout, requests.ConnectionError) as e:
            wait = min((2 ** attempt) * 1.15 + random.uniform(0, 1.0), 60.0)
            attempt += 1
            if attempt > max_retries:
                raise
            print(f"Timeout/network issue: {e}. Retry in {wait:.1f}s…")
            time.sleep(wait)

def _pick_citation(cites) -> str:
    """
    Pulls a single citation string from possible shapes: list/dict/scalar.
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
    Guesses a state/federal tag from a court name string.
    """
    if not court_name:
        return ""
    if re.search(r"U\.S\.", court_name, re.I):
        return "Federal"
    m = re.search(r"(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming)", court_name, re.I)
    if m:
        return re.sub(r"\s+", " ", m.group(0)).title()
    return ""

# Main functino: downloads pages, collects rows, writes the CSV

def extract_data(query: str = "corporation", max_cases: int = 1000, debug_print_one: bool = False) -> pd.DataFrame:
    """
    Pulls opinions that match a keyword search.
    - query: keyword to search for
    - max_cases: how many cases to collect total
    - debug_print_one: print one full JSON for inspection
    """
    if not API_KEY or not EMAIL:
        raise RuntimeError("Set COURTLISTENER_API_KEY and COURTLISTENER_EMAIL in your environment or .env") # I figured you might need this

    # 1) Session with basic retry adapter + headers
    headers = {"User-Agent": f"inst414-final-project ({EMAIL})", "Authorization": f"Token {API_KEY}"}
    session = requests.Session()
    session.headers.update(headers)

    retry = Retry(total=6, connect=4, read=4, backoff_factor=0.7,
                  status_forcelist=[429,502,503,504], allowed_methods=["GET"], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # 2) Prepare params + caches
    page_size = min(max_cases, 100 if FAST_MODE else 50)
    params = {"search": query, "page_size": page_size}

    court_cache: Dict[str, Dict[str, Any]] = {}
    cluster_cache: Dict[str, Dict[str, Any]] = {}
    docket_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    http_cache = shelve.open(CACHE_PATH)

    def _cached_get_json(url: str) -> dict:
        key = f"GET::{url}"
        if key in http_cache:
            return http_cache[key]
        data = _retry_get_json(session, url)
        http_cache[key] = data
        return data

    url = API_URL
    first = True
    printed_debug = False

    print(f"Extracting cases from CourtListener API with query: {query}")

    try:
        # 3) Loop over pages until it hits max_cases or if there arent any pages left
        while url and len(rows) < max_cases:
            r = _page_get_with_backoff(session, url, params if first else None)
            payload = r.json()
            first = False

            # 4) Each item = one opinion
            for op in payload.get("results", []):
                if debug_print_one and not printed_debug:
                    print(json.dumps(op, indent=2))
                    printed_debug = True
                if len(rows) >= max_cases:
                    break

                # 4a) Pulls fields
                case_id        = op.get("id")
                date_created   = op.get("date_created", "")
                date_modified  = op.get("date_modified", "")
                page_count     = op.get("page_count", None)
                per_curiam     = bool(op.get("per_curiam", False))
                optype         = op.get("type", "")
                author_id      = op.get("author_id", None)
                joined_by_cnt  = len(op.get("joined_by", []) or [])
                download_url   = op.get("download_url", "")
                cluster_url    = op.get("cluster") or ""
                opinion_cites  = op.get("citations", None)

                m = re.match(r"^(\d{4})-", date_created or "")
                opinion_year = int(m.group(1)) if m else None

                raw_plain = op.get("plain_text") or ""
                plain_text_present = int(bool(raw_plain))
                plain_text_url = raw_plain if (isinstance(raw_plain, str) and raw_plain.startswith(("http://","https://"))) else ""
                text = "" if FAST_MODE else (raw_plain if isinstance(raw_plain, str) and not plain_text_url else "")

                # 4b) Cluster (cached)
                cluster: Dict[str, Any] = {}
                if cluster_url:
                    cluster = cluster_cache.get(cluster_url) or _cached_get_json(cluster_url)
                    cluster_cache[cluster_url] = cluster

                case_name = (cluster.get("caseName") or cluster.get("case_name") or cluster.get("caption") or
                             op.get("case_name","") or "")

                # 4c) Citations
                cluster_citations = cluster.get("citations", None)
                all_citations = cluster_citations if cluster_citations is not None else opinion_cites
                citation = _pick_citation(all_citations)
                has_citation = int(bool(citation and citation != "Unknown"))

                # 4d) Court (tries docket->court or cluster->court)
                court_name = ""
                court_url  = ""
                if not court_name:
                    docket_url = (cluster.get("docket") or op.get("docket") or "")
                    if docket_url and isinstance(docket_url, str):
                        dcu = _normalize_url(docket_url)
                        docket = docket_cache.get(dcu) or _cached_get_json(dcu)
                        docket_cache[dcu] = docket
                        d_court = docket.get("court")
                        if isinstance(d_court, str) and d_court.startswith(("http://","https://")):
                            dcu2 = _normalize_url(d_court)
                            cj = court_cache.get(dcu2) or _cached_get_json(dcu2)
                            court_cache[dcu2] = cj
                            court_name = cj.get("full_name") or cj.get("name_abbreviation") or cj.get("name") or ""
                            court_url = dcu2
                        elif isinstance(d_court, dict):
                            court_name = d_court.get("full_name") or d_court.get("name_abbreviation") or d_court.get("name") or ""
                            court_url = d_court.get("resource_uri","") or court_url

                if not court_name:
                    clu_court = cluster.get("court") or ""
                    if clu_court:
                        ccu = _normalize_url(clu_court)
                        cj = court_cache.get(ccu) or _cached_get_json(ccu)
                        court_cache[ccu] = cj
                        court_name = cj.get("full_name") or cj.get("name_abbreviation") or cj.get("name") or ""
                        court_url = ccu

                # 4e) Small cleanups to court names + adds basic flags if an Unknown gets through
                court_name = re.sub(r"\s+", " ", (court_name or "")).strip()
                jurisdiction_state = _infer_state_from_court_name(court_name) or "Unknown"
                has_court = int(bool(court_name and court_name != "Unknown"))

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

                # 4f) Collect one row
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
                    "outcome_code": 0,  # temp value, transform.py replaces this value with the correct one
                })

                # checkpoint CSV every N rows (helps to see if code is working)
                if len(rows) % CHECKPOINT_EVERY == 0:
                    tmp = pd.DataFrame(rows)
                    os.makedirs("data/extracted", exist_ok=True)
                    tmp.to_csv("data/extracted/raw_data_partial.csv", index=False, columns=OUTPUT_COLUMNS)
                    print(f"Checkpoint wrote {len(rows)} rows -> data/extracted/raw_data_partial.csv")

            time.sleep(0.15)
            url = payload.get("next")

        # 5) Final CSV
        df = pd.DataFrame(rows)
        os.makedirs("data/extracted", exist_ok=True)
        outpath = "data/extracted/raw_data.csv"
        df.to_csv(outpath, index=False, columns=OUTPUT_COLUMNS)
        print(f"Extracted {len(df)} cases to {outpath}")
        return df

    finally:
        try:
            http_cache.close()
        except Exception:
            pass


if __name__ == "__main__":
    extract_data(query="corporation", max_cases=1000, debug_print_one=False)