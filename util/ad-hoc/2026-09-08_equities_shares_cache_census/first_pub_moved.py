"""CIKs whose FIRST available count is deferred by the latest-filed dedup (NaN rows where the figure was already public).

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
    /opt/miniforge3/envs/JuniperData/bin/python first_pub_moved.py /tmp/shares-census-out

Original notes
--------------
Lane A3 — follow-up: for the CIKs whose earliest `filed` differs between the
raw payload and the generator's surviving frame, attribute the move to the
outlier filter (earliest point dropped) or to the latest-filed dedup at
generator.py:918-923 (earliest point's end restated later).  Also detail CIK
1561550 for the causal-filter nuance.  Read-only.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import a3lib  # noqa: E402

OUT = Path(sys.argv[1])
rows = []
for path in sorted((a3lib.CACHE / "shares").glob("*.json")):
    cik = int(path.stem)
    data = a3lib.load_payload(cik)
    pts = [p for p in a3lib.raw_points(data) if p.get("val") is not None and p.get("end") and p.get("filed")]
    raw_min = min(p["filed"] for p in pts)
    best, filed_on = a3lib.dedup_latest_filed(data)
    dedup_min = min(filed_on.values())  # after dedup, BEFORE filter
    series = a3lib.whole_history_filter(a3lib.to_series(best))
    surv_ends = {d.strftime("%Y-%m-%d") for d in series.index}
    gen_min = min(f for e, f in filed_on.items() if e in surv_ends)
    if gen_min != raw_min:
        cause = []
        if dedup_min != raw_min:
            cause.append("DEDUP(:918-923)")
        if gen_min != dedup_min:
            cause.append("OUTLIER_FILTER(:932-934)")
        first_raw = min(pts, key=lambda p: (p["filed"], p["end"]))
        rows.append({"cik": cik, "entity": data.get("entityName"), "raw_min_filed": raw_min, "after_dedup": dedup_min, "after_filter": gen_min, "deferral_days": (pd.Timestamp(gen_min) - pd.Timestamp(raw_min)).days, "cause": "+".join(cause), "first_raw_end": first_raw["end"], "first_raw_val": first_raw["val"], "first_raw_form": first_raw.get("form")})
df = pd.DataFrame(rows).sort_values("deferral_days", ascending=False)
df.to_csv(OUT / "first_pub_moved.csv", index=False)
print("CIKs whose first available filed date moved:", len(df))
print(df["cause"].value_counts().to_dict())
print(df.to_string(index=False))
print("\nDEDUP-only cases (the 8-K restatement mechanism acting on the FIRST point):")
d = df[df["cause"] == "DEDUP(:918-923)"]
print(d[["cik", "entity", "raw_min_filed", "after_dedup", "deferral_days"]].to_string(index=False))
print("sum of deferral days (dedup-only):", int(d["deferral_days"].sum()), " max:", int(d["deferral_days"].max()) if len(d) else None)

# CIK 1561550 detail: causal filters change it, whole-history does not
data = a3lib.load_payload(1561550)
best, _ = a3lib.dedup_latest_filed(data)
s = a3lib.to_series(best)
print("\nCIK 1561550", data.get("entityName"), "n points", len(s), "whole-history median", s.median())
print(s.head(8).to_string())
