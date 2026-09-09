"""Same, restricted to same-value re-statements (the handoff's mechanism only), business-day calendar.

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
    /opt/miniforge3/envs/JuniperData/bin/python restatement_population_v2.py /tmp/shares-census-out

Original notes
--------------
Lane A3 — population-level §0.7 measurement, v2: separates the document's
exact mechanism (SAME-value restatement moving a figure's publication date)
from corrections (DIFFERENT-value restatement), and flags default-14 reach.

References per CIK (business-day calendar, first raw filed -> 2026-06-02):
  gen : generator-faithful (real _fetch_shares on cache hit + _condition_one 706-716 replica)
  A2  : true point-in-time over ALL facts (corrections apply when filed)
  A3  : generator's dedup, but each end's FINAL figure dated at its FIRST
        publication of that same value  -> isolates the doc's mechanism only
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
DEFAULT14 = {"A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP"}
HORIZON = pd.Timestamp("2026-06-02")


def dedup_final_value_first_pub(data: dict):
    """A3: per end, val = latest-filed val (as the generator), filed = EARLIEST
    filed among facts of that end carrying that same val."""
    best, filed_on = a3lib.dedup_latest_filed(data)
    first_pub = {}
    for p in a3lib.raw_points(data):
        if p.get("val") is None or not p.get("end") or not p.get("filed"):
            continue
        if float(p["val"]) == best.get(p["end"]):
            first_pub[p["end"]] = min(first_pub.get(p["end"], p["filed"]), p["filed"])
    return best, {e: first_pub.get(e, f) for e, f in filed_on.items()}


rows = []
for path in sorted((a3lib.CACHE / "shares").glob("*.json")):
    cik = int(path.stem)
    data = a3lib.load_payload(cik)
    pts = [p for p in a3lib.raw_points(data) if p.get("val") is not None and p.get("end") and p.get("filed")]
    by_end = {}
    for p in pts:
        by_end.setdefault(p["end"], []).append((p["filed"], float(p["val"])))
    same_val = sum(1 for e, l in by_end.items() if len({f for f, _ in l}) > 1 and len({v for _, v in l}) == 1)
    diff_val = sum(1 for e, l in by_end.items() if len({v for _, v in l}) > 1)
    start = pd.Timestamp(min(p["filed"] for p in pts))
    cal = pd.bdate_range(start, HORIZON)
    real = g.EquitiesGenerator._fetch_shares(cik, True)
    gen, _ = a3lib.align_on_filed(real, cal)
    a2 = a3lib.point_in_time_series(data, cal)
    a3 = a3lib.replica_fetch_shares(data, dedup=dedup_final_value_first_pub)
    a3s, _ = a3lib.align_on_filed(a3, cal)
    d2 = (gen != a2) & gen.notna() & a2.notna()
    d3 = (gen != a3s) & gen.notna() & a3s.notna()
    r3 = ((gen / a3s) - 1.0)[d3]
    rows.append(
        {
            "cik": cik,
            "ticker": tick.get(cik),
            "in_default14": bool(set(tick.get(cik, "").split(",")) & DEFAULT14),
            "n_pts": len(pts),
            "ends_restated_same_val": same_val,
            "ends_restated_diff_val": diff_val,
            "rows_diff_A2": int(d2.sum()),
            "rows_diff_A3": int(d3.sum()),
            "A3_over": int((r3 > 0).sum()),
            "A3_under": int((r3 < 0).sum()),
            "A3_max_abs_pct": float(r3.abs().max() * 100) if len(r3) else 0.0,
            "A3_rows_gt_1pct": int((r3.abs() > 0.01).sum()),
            "gen_nan_but_A3_known": int((gen.isna() & a3s.notna()).sum()),
        }
    )
df = pd.DataFrame(rows)
df.to_csv(OUT / "restatement_population_v2.csv", index=False)
print("CIKs:", len(df))
print("CIKs with >=1 end reported by >1 filing with the SAME value:", int((df["ends_restated_same_val"] > 0).sum()), " ends:", int(df["ends_restated_same_val"].sum()))
print("CIKs with >=1 end reported with DIFFERENT values (corrections):", int((df["ends_restated_diff_val"] > 0).sum()), " ends:", int(df["ends_restated_diff_val"].sum()))
a3aff = df[df["rows_diff_A3"] > 0]
print("\n[A3 = doc's mechanism only] CIKs with >=1 differing business-day row:", len(a3aff), " rows:", int(df["rows_diff_A3"].sum()), " over:", int(df["A3_over"].sum()), " under:", int(df["A3_under"].sum()))
print("   CIKs with a row off by >1%:", int((df["A3_rows_gt_1pct"] > 0).sum()), " rows >1%:", int(df["A3_rows_gt_1pct"].sum()), " max |rel|:", f"{df['A3_max_abs_pct'].max():.2f}%")
print("   CIKs with rows NaN in generator though the final figure was already public:", int((df["gen_nan_but_A3_known"] > 0).sum()), " rows:", int(df["gen_nan_but_A3_known"].sum()))
print("[A2 = all facts, corrections included] CIKs differing:", int((df["rows_diff_A2"] > 0).sum()), " rows:", int(df["rows_diff_A2"].sum()))
print("\nDefault-14 prefix members among cached CIKs:", df[df["in_default14"]]["ticker"].tolist())
print("Default-14 members affected under A3:", a3aff[a3aff["in_default14"]][["ticker", "rows_diff_A3", "A3_max_abs_pct"]].to_string(index=False))
print("\nKO anchor:", df[df["cik"] == 21344][["ticker", "rows_diff_A2", "rows_diff_A3", "A3_over", "A3_under", "A3_max_abs_pct"]].to_string(index=False))
print("\nTop 12 under A3 by max |rel| %:")
print(a3aff.sort_values("A3_max_abs_pct", ascending=False).head(12)[["cik", "ticker", "in_default14", "n_pts", "ends_restated_same_val", "rows_diff_A3", "A3_over", "A3_under", "A3_max_abs_pct", "A3_rows_gt_1pct", "gen_nan_but_A3_known"]].to_string(index=False))
print("\nTop 12 under A3 by rows:")
print(a3aff.sort_values("rows_diff_A3", ascending=False).head(12)[["cik", "ticker", "in_default14", "n_pts", "ends_restated_same_val", "rows_diff_A3", "A3_over", "A3_under", "A3_max_abs_pct", "A3_rows_gt_1pct"]].to_string(index=False))
# outlier-loss CIKs in default 14
oc = pd.read_csv(OUT / "outlier_census.csv")
lost = oc[oc["lost"] > 0]["cik"].tolist()
print("\nOutlier-filter losers in default-14:", [tick.get(c) for c in lost if set(tick.get(c, "").split(",")) & DEFAULT14])
