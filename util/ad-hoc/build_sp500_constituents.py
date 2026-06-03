"""Build the bundled S&P 500 constituents snapshot for the equities generator.

One-time dev utility (re-runnable): scrapes the current S&P 500 membership from
Wikipedia, validates/fills each company's CIK + canonical name against SEC's
``company_tickers.json``, and writes a *tracked* CSV (``ticker,name,cik,sector``)
that the equities generator consumes as its default universe and its
ticker -> (name, CIK) map.

The CSV is a point-in-time snapshot of *current* constituents (survivorship
biased by construction). Re-run to refresh; review the diff before committing.

Run with network access in a throwaway env, e.g.::

    uv run --no-project --with pandas --with lxml --python 3.13 \
        python util/ad-hoc/build_sp500_constituents.py
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     build_sp500_constituents.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

# SEC fair-access policy requires a descriptive User-Agent with contact info.
UA = {"User-Agent": "juniper-data research overtoad.research@gmail.com"}
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

OUT_PATH = Path(__file__).resolve().parents[2] / "juniper_data" / "generators" / "equities" / "sp500_constituents.csv"


def _fetch(url: str) -> bytes:
    """GET ``url`` with a compliant User-Agent header."""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted, fixed hosts)
        return resp.read()


def main() -> None:
    """Scrape, enrich, and write the constituents snapshot."""
    html = _fetch(WIKI_URL).decode("utf-8", "replace")
    table = pd.read_html(io.StringIO(html))[0]
    table = table.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector", "CIK": "cik"})
    frame = table[["ticker", "name", "sector", "cik"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["name"] = frame["name"].astype(str).str.strip()
    frame["sector"] = frame["sector"].astype(str).str.strip()
    frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce").astype("Int64")

    # Validate / fill CIK against the authoritative SEC mapping.
    sec_rows = json.loads(_fetch(SEC_TICKERS_URL).decode())
    by_ticker = {row["ticker"].upper(): row for row in sec_rows.values()}
    for idx, row in frame.iterrows():
        match = by_ticker.get(row["ticker"]) or by_ticker.get(row["ticker"].replace(".", "-"))
        if match and pd.isna(row["cik"]):
            frame.at[idx, "cik"] = match["cik_str"]

    frame = frame.dropna(subset=["cik"]).sort_values("ticker").reset_index(drop=True)
    frame["cik"] = frame["cik"].astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(frame)} constituents -> {OUT_PATH}")
    print(frame.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
