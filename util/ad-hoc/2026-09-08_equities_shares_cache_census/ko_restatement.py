"""Replays the generator's latest-filed-per-period-end dedup on KO (CIK 21344) against a first-publication alternative; the three overstated episodes.

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
    /opt/miniforge3/envs/JuniperData/bin/python ko_restatement.py /tmp/shares-census-out

Original notes
--------------
Lane A3 — claim 4: KO (CIK 21344) restatement exposure.
Generator-faithful series (real _fetch_shares + replica of _condition_one
706-716) vs (A1) per-end EARLIEST-filed dedup and (A2) true point-in-time.
Trading days from KO's cached OHLCV; business-day calendar as the alternative.
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
CIK = 21344
data = a3lib.load_payload(CIK)
pts = a3lib.raw_points(data)
print("KO cached payload: taxonomy/tag =", data.get("taxonomy"), data.get("tag"), " units:", {k: len(v) for k, v in data["units"].items()})
print("raw points:", len(pts), " distinct end:", len({p["end"] for p in pts}), " distinct (end,val):", len({(p["end"], p["val"]) for p in pts}), " distinct filed:", len({p["filed"] for p in pts}))
pdf = pd.DataFrame(pts)[["end", "filed", "val", "form", "fy", "fp", "accn"]].sort_values(["end", "filed"])
pdf.to_csv(OUT / "ko_raw_points.csv", index=False)
dup_ends = pdf[pdf.duplicated("end", keep=False)]
print("\nends reported by >1 filing (the restatement candidates):")
print(dup_ends.to_string(index=False))

# generator-faithful shares frame
real = g.EquitiesGenerator._fetch_shares(CIK, True)
print("\n_fetch_shares -> rows:", len(real), " quality:", real["shares_quality"].iloc[0], " origin:", real["shares_origin"].iloc[0])

# trade calendar: KO cached OHLCV
ohlcv_path = a3lib.CACHE / "ohlcv" / "KO_2000-01-01_2026-06-03.csv"
ohlcv = pd.read_csv(ohlcv_path, index_col=0)
ohlcv.index = pd.to_datetime(ohlcv.index)
trade_index = ohlcv.index.sort_values()
print("KO OHLCV:", ohlcv_path.name, " rows:", len(trade_index), trade_index.min().date(), "->", trade_index.max().date())

gen_total, gen_asof = a3lib.align_on_filed(real, trade_index)
alt1 = a3lib.replica_fetch_shares(data, dedup=a3lib.dedup_earliest_filed)
alt1_total, _ = a3lib.align_on_filed(alt1, trade_index)
alt2_total = a3lib.point_in_time_series(data, trade_index)

bdays = pd.bdate_range(trade_index.min(), trade_index.max())
gen_b, _ = a3lib.align_on_filed(real, bdays)
alt2_b = a3lib.point_in_time_series(data, bdays)


def episodes(gen: pd.Series, alt: pd.Series, label: str):
    diff = (gen != alt) & gen.notna() & alt.notna()
    print(f"\n=== generator vs {label}: differing rows = {int(diff.sum())} of {int((gen.notna() & alt.notna()).sum())} rows with a value")
    runs = []
    in_run = False
    for d, flag in diff.items():
        if flag and not in_run:
            start = d
            in_run = True
        elif not flag and in_run:
            runs.append((start, prev))
            in_run = False
        prev = d
    if in_run:
        runs.append((start, prev))
    total_days = 0
    for s, e in runs:
        seg = diff.loc[s:e]
        n = int(seg.sum())
        total_days += n
        gv = gen.loc[s]
        av = alt.loc[s]
        print(f"   {s.date()}..{e.date()}  {n:3d} rows  ships {gv:,.0f}  correct {av:,.0f}  ratio-1 = {(gv/av-1)*100:+.5f}%  ({'OVER' if gv > av else 'UNDER'}stated)")
    print(f"   TOTAL differing rows: {total_days} in {len(runs)} episode(s)")
    return runs, total_days


episodes(gen_total, alt1_total, "A1 per-end EARLIEST-filed (trading days, KO OHLCV)")
episodes(gen_total, alt2_total, "A2 true point-in-time (trading days, KO OHLCV)")
episodes(gen_b, alt2_b, "A2 true point-in-time (BUSINESS-day calendar, no holidays)")

# the specific figures the document quotes
for v in (4485161506, 4456717996):
    hits = pdf[pdf["val"] == v]
    print(f"\nfacts with val={v:,}:")
    print(hits.to_string(index=False))
print("\narithmetic: 4485161506/4456717996 - 1 =", f"{(4485161506/4456717996-1)*100:.5f}%")
print("rows 2013-02-26..2013-04-26 generator vs A2:")
seg = pd.DataFrame({"gen": gen_total, "alt2": alt2_total, "gen_asof": gen_asof}).loc["2013-02-25":"2013-04-28"]
print(seg.iloc[[0, 1, 2, -3, -2, -1]].to_string())
pd.DataFrame({"gen": gen_total, "alt1": alt1_total, "alt2": alt2_total, "gen_asof": gen_asof}).to_csv(OUT / "ko_series.csv")
