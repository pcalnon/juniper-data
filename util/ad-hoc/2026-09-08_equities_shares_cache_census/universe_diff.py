"""Bundled default universe (503 tickers / 500 CIKs) vs the cached payload set; names the members with no cached payload.

Project: juniper-data
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-08
Status: ad-hoc — investigation (round-37 defect-register handoff validation, lane A3)
Retire when: the SEC shares cache gains a versioned key / TTL and the restatement + outlier
    findings are closed in the defect register (or the instruments are superseded by a test).
Related: juniper-ml/notes/JUNIPER_2026-08-14_JUNIPER-ECOSYSTEM_DEFECT-REGISTER.md;
    juniper-ml/prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-08_defect-register-round-38-*.md

READ-ONLY on the cache (``~/.cache/juniper_data/equities`` or ``JUNIPER_DATA_EQUITIES_CACHE_DIR``);
the network is blocked (``_sec_get`` is patched to raise) so a missing payload is reported, never
fetched. Output goes to the directory given as the first argument. Run with the JuniperData
interpreter from this directory (the scripts import ``a3lib`` from beside themselves):

    cd util/ad-hoc/2026-09-08_equities_shares_cache_census
    /opt/miniforge3/envs/JuniperData/bin/python universe_diff.py /tmp/shares-census-out

Original notes
--------------
Lane A3 — bundled universe vs on-disk shares cache (claim 2).

Loads sp500_constituents.csv exactly the way EquitiesGenerator._load_constituents
does (ticker upper, cik int if digits else None), then diffs the CIK set against
shares/*.json filenames.  Read-only.
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/pcalnon/Development/python/Juniper/juniper-data")
CSV = REPO / "juniper_data/generators/equities/sp500_constituents.csv"
CACHE = Path(os.environ.get("JUNIPER_DATA_EQUITIES_CACHE_DIR", str(Path.home() / ".cache" / "juniper_data" / "equities")))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rows = {}
with open(CSV, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        ticker = row["ticker"].strip().upper()
        cik_raw = (row.get("cik") or "").strip()
        rows[ticker] = {"name": row.get("name", ticker).strip(), "cik": int(cik_raw) if cik_raw.isdigit() else None}

tickers = sorted(rows)
ciks_by_ticker = {t: r["cik"] for t, r in rows.items()}
none_cik = [t for t, c in ciks_by_ticker.items() if c is None]
cik_to_tickers = defaultdict(list)
for t, c in ciks_by_ticker.items():
    if c is not None:
        cik_to_tickers[c].append(t)
universe_ciks = set(cik_to_tickers)
dups = {c: ts for c, ts in cik_to_tickers.items() if len(ts) > 1}

cached_ciks = {int(p.stem) for p in (CACHE / "shares").glob("*.json")}

missing = sorted(universe_ciks - cached_ciks)
extra = sorted(cached_ciks - universe_ciks)
missing_tickers = sorted(t for c in missing for t in cik_to_tickers[c])

print(f"tickers in CSV: {len(tickers)}")
print(f"tickers with cik=None: {len(none_cik)} {none_cik}")
print(f"distinct CIKs in universe: {len(universe_ciks)}")
print(f"CIKs shared by >1 ticker: {len(dups)} {dict(sorted(dups.items()))}")
print(f"cached shares payloads: {len(cached_ciks)}")
print(f"universe CIKs WITHOUT cached payload: {len(missing)}")
print("missing tickers:", missing_tickers)
print("missing (cik, tickers, name):")
for c in missing:
    print("   ", c, cik_to_tickers[c], rows[cik_to_tickers[c][0]]["name"])
print(f"cached CIKs NOT in universe: {len(extra)} {extra}")
print(f"universe CIKs WITH cached payload: {len(universe_ciks & cached_ciks)}")

DOC = sorted("EL TSN RL META XYZ ABNB TTD STZ DASH TKO UHS HRL MKC LEN ERIE".split())
print("doc's 15:", DOC)
print("doc - actual:", sorted(set(DOC) - set(missing_tickers)))
print("actual - doc:", sorted(set(missing_tickers) - set(DOC)))
with open(OUT / "universe_missing.txt", "w") as fh:
    for c in missing:
        fh.write(f"{c}\t{','.join(cik_to_tickers[c])}\t{rows[cik_to_tickers[c][0]]['name']}\n")
with open(OUT / "universe_ciks.txt", "w") as fh:
    for c in sorted(universe_ciks):
        fh.write(f"{c}\t{','.join(cik_to_tickers[c])}\n")
