"""Per CIK, the first filing date and the share of a default 2000-01-01..today window that precedes it (raw min-filed and generator-faithful definitions).

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
    /opt/miniforge3/envs/JuniperData/bin/python first_filing.py /tmp/shares-census-out

Original notes
--------------
Lane A3 — claim 3: first filing per CIK and the fraction of the default
window that precedes it.  Several definitions, several as-of dates.
Read-only.  Uses the REAL generator._fetch_shares (cache hit, network blocked)
for the generator-faithful column and the replica for the alternatives.
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
START = pd.Timestamp("2000-01-01")
AS_OF = [pd.Timestamp(x) for x in ("2026-09-07", "2026-09-08", "2026-01-01", "2025-01-01")]

rows = []
for path in sorted((a3lib.CACHE / "shares").glob("*.json")):
    cik = int(path.stem)
    data = a3lib.load_payload(cik)
    pts = [p for p in a3lib.raw_points(data) if p.get("val") is not None and p.get("end")]
    raw_min_filed = min((p["filed"] for p in pts if p.get("filed")), default=None)
    raw_min_end = min(p["end"] for p in pts)
    n_unfiled = sum(1 for p in pts if not p.get("filed"))
    # generator-faithful: real function on the cache hit
    real = g.EquitiesGenerator._fetch_shares(cik, True)
    assert real is not None, cik
    known = real.dropna(subset=["filed"])
    gen_min_filed = known["filed"].min()
    gen_min_end = real.index.min()
    # replica cross-check
    rep = a3lib.replica_fetch_shares(data)
    assert rep is not None and len(rep) == len(real) and (rep["shares"].values == real["shares"].values).all(), cik
    rows.append(
        {
            "cik": cik,
            "n_raw": len(pts),
            "n_unfiled": n_unfiled,
            "raw_min_filed": raw_min_filed,
            "raw_min_end": raw_min_end,
            "gen_min_filed": gen_min_filed.strftime("%Y-%m-%d"),
            "gen_min_end": gen_min_end.strftime("%Y-%m-%d"),
            "quality": real["shares_quality"].iloc[0],
            "origin": real["shares_origin"].iloc[0],
        }
    )

df = pd.DataFrame(rows)
df.to_csv(OUT / "first_filing_per_cik.csv", index=False)
print("CIKs:", len(df), " unfiled points total:", int(df["n_unfiled"].sum()), " CIKs with any unfiled:", int((df["n_unfiled"] > 0).sum()))
print("quality/origin on cache hit:", df["quality"].unique(), df["origin"].unique())
print()
for col in ("gen_min_filed", "raw_min_filed", "gen_min_end", "raw_min_end"):
    d = pd.to_datetime(df[col])
    srt = d.sort_values().reset_index(drop=True)
    print(f"== {col}: earliest {srt.iloc[0].date()}  median(sorted[242]) {srt.iloc[242].date()}  pandas-median {d.median().date()}  latest {srt.iloc[-1].date()}")
    for asof in AS_OF:
        window = (asof - START).days
        frac = (d - START).dt.days / window
        mean485 = frac.mean()
        univ = (frac.sum() + 15 * 1.0) / 500
        print(f"   as-of {asof.date()} window={window}d  mean(485)={mean485*100:.3f}%  median={frac.median()*100:.3f}%  universe-wide(15@100%)/500={univ*100:.3f}%")
print()
print("gen_min_filed != raw_min_filed (dedup moved the first publication):", int((df["gen_min_filed"] != df["raw_min_filed"]).sum()))
print(df[df["gen_min_filed"] != df["raw_min_filed"]][["cik", "raw_min_filed", "gen_min_filed"]].to_string())
