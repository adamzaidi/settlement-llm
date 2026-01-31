import os
import random
import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()


def audit_citations(
    raw_csv_path: str = "data/extracted/raw_data.csv",
    sample_size: int = 250,
    seed: int = 7,
    debug_examples: int = 3,
) -> pd.DataFrame:
    """
    Standalone sanity check:
    - Loads extracted raw_data.csv
    - Randomly samples cluster_url values
    - Fetches each cluster JSON directly from CourtListener
    - Summarizes whether citations exist upstream (not your ETL)

    Returns a dataframe of per-cluster audit results.
    """
    api_key = os.getenv("COURTLISTENER_API_KEY")
    email = os.getenv("COURTLISTENER_EMAIL")
    assert api_key and email, (
        f"Missing COURTLISTENER_API_KEY / COURTLISTENER_EMAIL "
        f"(API_KEY present: {bool(api_key)}, EMAIL present: {bool(email)})"
    )

    df = pd.read_csv(raw_csv_path)
    if "cluster_url" not in df.columns:
        raise ValueError(f"Expected 'cluster_url' column in {raw_csv_path}")

    clusters = [
        c for c in df["cluster_url"].dropna().astype(str).unique()
        if c.startswith("http")
    ]
    if not clusters:
        raise ValueError("No usable cluster_url values found in raw_data.csv")

    random.seed(seed)
    sample = random.sample(clusters, min(sample_size, len(clusters)))

    headers = {
        "User-Agent": f"inst414-citation-audit ({email})",
        "Authorization": f"Token {api_key}",
    }

    rows = []
    debug_dumped = 0

    with requests.Session() as s:
        s.headers.update(headers)
        for url in sample:
            try:
                r = s.get(url, timeout=30)
                r.raise_for_status()
                j = r.json()

                citations_list = j.get("citations") or []
                citation_singular = j.get("citation")
                citation_count = j.get("citation_count")

                # Optional: print a few examples of the actual citation object shape
                if citations_list and debug_dumped < debug_examples:
                    first = citations_list[0]
                    print("\n--- DEBUG citation object example ---")
                    print("cluster_url:", url)
                    if isinstance(first, dict):
                        print("first citation keys:", sorted(list(first.keys())))
                        print("first citation object:", first)
                    else:
                        print("first citation is not a dict:", type(first), first)
                    debug_dumped += 1

                rows.append({
                    "cluster_url": url,
                    "citation": citation_singular,
                    "citations_len": len(citations_list),
                    "citation_count": citation_count,
                })
            except Exception as e:
                rows.append({
                    "cluster_url": url,
                    "error": str(e)[:200],
                })

    out = pd.DataFrame(rows)

    # Normalize numeric fields so summary/filtering behaves correctly
    if "citations_len" in out.columns:
        out["citations_len"] = pd.to_numeric(out["citations_len"], errors="coerce").fillna(0).astype(int)
    if "citation_count" in out.columns:
        out["citation_count"] = pd.to_numeric(out["citation_count"], errors="coerce").fillna(0).astype(int)

    return out


def print_summary(out: pd.DataFrame) -> None:
    ok = out[out.get("error").isna()] if "error" in out.columns else out

    print("\n=== Citation Audit Summary ===")
    print("Rows audited:", len(out))
    print("Fetch OK:", len(ok))
    if len(ok) == 0:
        print("No successful fetches. Check API key/email, network, or rate limits.")
        return

    # treat literal "None" strings as empty too
    citation_series = ok["citation"] if "citation" in ok.columns else pd.Series([], dtype=object)
    has_citation_field = (
        citation_series.notna()
        & (citation_series.astype(str).str.strip() != "")
        & (citation_series.astype(str).str.strip().str.lower() != "none")
    )

    citations_len = ok["citations_len"] if "citations_len" in ok.columns else pd.Series([0] * len(ok))
    citation_count = ok["citation_count"] if "citation_count" in ok.columns else pd.Series([0] * len(ok))

    has_citations_list = citations_len.fillna(0).astype(int) > 0
    has_citation_count = citation_count.fillna(0).astype(int) > 0

    print("% citation (singular) present:", round(has_citation_field.mean() * 100, 2))
    print("% citations list non-empty:", round(has_citations_list.mean() * 100, 2))
    print("% citation_count > 0:", round(has_citation_count.mean() * 100, 2))

    print("\nTop 10 rows where citation_count > 0:")
    if "citation_count" in ok.columns:
        hits = ok[ok["citation_count"] > 0].sort_values("citation_count", ascending=False).head(10)
        if hits.empty:
            print("(none)")
        else:
            cols = [c for c in ["citation", "citations_len", "citation_count", "cluster_url"] if c in hits.columns]
            print(hits[cols].to_string(index=False))

    print("\nTop 10 rows where citations list is non-empty:")
    if "citations_len" in ok.columns:
        hits2 = ok[ok["citations_len"] > 0].sort_values("citations_len", ascending=False).head(10)
        if hits2.empty:
            print("(none)")
        else:
            cols2 = [c for c in ["citation", "citations_len", "citation_count", "cluster_url"] if c in hits2.columns]
            print(hits2[cols2].to_string(index=False))


if __name__ == "__main__":
    out = audit_citations(sample_size=250, debug_examples=3)
    print_summary(out)

    # Optional: write results for your report / debugging
    os.makedirs("data/outputs", exist_ok=True)
    out.to_csv("data/outputs/citation_audit.csv", index=False)
    print("\nWrote: data/outputs/citation_audit.csv")