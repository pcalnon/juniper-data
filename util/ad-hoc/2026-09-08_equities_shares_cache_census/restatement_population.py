"""The 8-K restatement effect across every cached CIK (corrections included).

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
    /opt/miniforge3/envs/JuniperData/bin/python restatement_population.py /tmp/shares-census-out

Original notes
--------------
Lane A3 — population-level size of the §0.7 latest-filed-dedup defect.
For every cached CIK: generator-faithful series (real _fetch_shares on the
cache hit + replica of _condition_one 706-716) vs A2 true point-in-time
(all facts kept; at date t the effective figure is the latest-END fact with
filed <= t, an older period never displacing a newer one; same outlier band).
Calendar: BUSINESS days (Mon-Fri, no holidays) from each CIK's first raw
filed date to 2026-06-02 (the OHLCV cache horizon), because only 520 of the
tickers' OHLCV windows are cached and none for today's default window.
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
HORIZON = pd.Timestamp("2026-06-02")

rows, episodes = [], []
for path in sorted((a3lib.CACHE / "shares").glob("*.json")):
    cik = int(path.stem)
    data = a3lib.load_payload(cik)
    pts = [p for p in a3lib.raw_points(data) if p.get("val") is not None and p.get("end") and p.get("filed")]
    start = pd.Timestamp(min(p["filed"] for p in pts))
    cal = pd.bdate_range(start, HORIZON)
    real = g.EquitiesGenerator._fetch_shares(cik, True)
    gen, _ = a3lib.align_on_filed(real, cal)
    alt = a3lib.point_in_time_series(data, cal)
    both = gen.notna() & alt.notna()
    diff = (gen != alt) & both
    nan_only_gen = gen.isna() & alt.notna()  # rows the generator leaves NaN but a value was public
    n_diff = int(diff.sum())
    rel = ((gen / alt) - 1.0)[diff]
    rows.append({"cik": cik, "ticker": tick.get(cik), "n_raw_pts": len(pts), "n_bdays": len(cal), "n_rows_differ": n_diff, "n_rows_gen_nan_but_known": int(nan_only_gen.sum()), "n_over": int((rel > 0).sum()), "n_under": int((rel < 0).sum()), "max_abs_rel_pct": float(rel.abs().max() * 100) if n_diff else 0.0, "rows_abs_rel_gt_1pct": int((rel.abs() > 0.01).sum())})
    # episodes
    in_run = False
    for d, flag in diff.items():
        if flag and not in_run:
            s = d; in_run = True
        elif not flag and in_run:
            episodes.append((cik, tick.get(cik), s.date(), prev.date(), int(diff.loc[s:prev].sum()), float((gen.loc[s] / alt.loc[s] - 1) * 100)))
            in_run = False
        prev = d
    if in_run:
        episodes.append((cik, tick.get(cik), s.date(), prev.date(), int(diff.loc[s:prev].sum()), float((gen.loc[s] / alt.loc[s] - 1) * 100)))

df = pd.DataFrame(rows)
df.to_csv(OUT / "restatement_population.csv", index=False)
ep = pd.DataFrame(episodes, columns=["cik", "ticker", "start", "end", "bdays", "rel_pct_at_start"])
ep.to_csv(OUT / "restatement_episodes.csv", index=False)
aff = df[df["n_rows_differ"] > 0]
print("CIKs:", len(df), " CIKs with >=1 business-day row differing (generator vs point-in-time):", len(aff))
print("total differing rows:", int(df["n_rows_differ"].sum()), " episodes:", len(ep))
print("rows overstated:", int(df["n_over"].sum()), " understated:", int(df["n_under"].sum()))
print("CIKs with any row off by >1%:", int((df["rows_abs_rel_gt_1pct"] > 0).sum()), " rows off by >1%:", int(df["rows_abs_rel_gt_1pct"].sum()))
print("CIKs with rows the generator leaves NaN although a figure was public:", int((df["n_rows_gen_nan_but_known"] > 0).sum()), " rows:", int(df["n_rows_gen_nan_but_known"].sum()))
print("\nKO anchor (business-day calendar):")
print(df[df["cik"] == 21344].to_string(index=False))
print("\nTop 15 by max |rel| %:")
print(aff.sort_values("max_abs_rel_pct", ascending=False).head(15).to_string(index=False))
print("\nTop 15 by differing rows:")
print(aff.sort_values("n_rows_differ", ascending=False).head(15).to_string(index=False))
print("\nExpedia episodes:")
print(ep[ep["cik"] == 1324424].to_string(index=False))
