"""Cached payloads whose deduped series is all-zero or placeholder-valued (silent market_cap 0.0 on a warm cache).

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
    /opt/miniforge3/envs/JuniperData/bin/python placeholder_scan.py /tmp/shares-census-out

Original notes
--------------
Lane A3 — scan every cached CIK's GENERATOR-FAITHFUL surviving series
(real _fetch_shares, cache hit, network blocked) for implausible share counts
(< 1e6 for an S&P 500 constituent), and detail Expedia / Datadog / CIK 2041610.
Read-only.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import a3lib  # noqa: E402
import juniper_data.generators.equities.generator as g  # noqa: E402


def _blocked(*a, **k):
    raise RuntimeError("NETWORK BLOCKED by lane A3")


g._sec_get = _blocked
OUT = Path(sys.argv[1])
tick = {}
for line in (OUT / "universe_ciks.txt").read_text().splitlines():
    c, t = line.split("\t")
    tick[int(c)] = t

rows = []
for path in sorted((a3lib.CACHE / "shares").glob("*.json")):
    cik = int(path.stem)
    real = g.EquitiesGenerator._fetch_shares(cik, True)
    s = real["shares"]
    known = real.dropna(subset=["filed"])
    rows.append({"cik": cik, "ticker": tick.get(cik), "n": len(s), "min": float(s.min()), "max": float(s.max()), "median": float(s.median()), "n_zero": int((s == 0).sum()), "n_lt_1e6": int((s < 1e6).sum()), "first_filed": known["filed"].min().date(), "last_end": s.index.max().date()})
df = pd.DataFrame(rows)
df.to_csv(OUT / "placeholder_scan.csv", index=False)
sus = df[df["n_lt_1e6"] > 0].sort_values("min")
print("CIKs whose GENERATOR-SURVIVING series contains a value < 1e6 shares:", len(sus))
print(sus.to_string(index=False))
print("\nCIKs whose surviving series contains an exact 0:", int((df["n_zero"] > 0).sum()), df[df["n_zero"] > 0][["cik", "ticker", "n", "n_zero"]].to_string(index=False))

for cik in (1324424, 1561550, 2041610, 1067983):
    data = a3lib.load_payload(cik)
    pts = pd.DataFrame(a3lib.raw_points(data))[["end", "filed", "val", "form", "fy", "fp"]].sort_values(["end", "filed"])
    print(f"\n===== CIK {cik} {tick.get(cik)} {data.get('entityName')}  tag={data.get('taxonomy')}/{data.get('tag')}  raw points={len(pts)}")
    print(pts.to_string(index=False) if len(pts) <= 40 else pts.head(14).to_string(index=False))
    real = g.EquitiesGenerator._fetch_shares(cik, True)
    print("generator surviving series:")
    print(real[["shares", "filed"]].to_string() if len(real) <= 12 else real[["shares", "filed"]].head(12).to_string())
