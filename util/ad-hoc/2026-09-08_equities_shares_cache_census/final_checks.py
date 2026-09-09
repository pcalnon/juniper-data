"""Re-derives the first-filing coverage under alternative definitions (end-date, business-day) for the sensitivity note.

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
    /opt/miniforge3/envs/JuniperData/bin/python final_checks.py /tmp/shares-census-out

Original notes
--------------
Lane A3 — closing checks: business-day vs calendar-day sensitivity of the
first-filing fraction; ticker names for cited CIKs.  Read-only.
"""

import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1])
tick = {}
for line in (OUT / "universe_ciks.txt").read_text().splitlines():
    c, t = line.split("\t")
    tick[int(c)] = t
ff = pd.read_csv(OUT / "first_filing_per_cik.csv")
START = pd.Timestamp("2000-01-01")
for asof in (pd.Timestamp("2026-09-07"),):
    nb_total = len(pd.bdate_range(START, asof)) - 1
    for col in ("raw_min_filed", "gen_min_filed"):
        d = pd.to_datetime(ff[col])
        cal = ((d - START).dt.days / (asof - START).days).mean() * 100
        bd = d.apply(lambda x: (len(pd.bdate_range(START, x)) - 1) / nb_total).mean() * 100
        print(f"as-of {asof.date()} {col}: calendar-day mean {cal:.3f}%  business-day mean {bd:.3f}%  (delta {bd-cal:+.3f} pp)")
print("\nCIK 1267238 =", tick.get(1267238))
for c in (1324424, 1564708, 1413447, 24741, 916365, 1652044, 2023554, 1633978, 1324404):
    print(c, tick.get(c))
print("\nsix placeholder CIKs:", {c: tick.get(c) for c in (24545, 1561550, 1690820, 1754301, 2041610, 1067983)})
