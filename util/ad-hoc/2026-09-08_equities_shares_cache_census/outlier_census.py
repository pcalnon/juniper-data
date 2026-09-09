"""The whole-history-median outlier filter (_SHARES_OUTLIER_FACTOR): CIKs losing >=1 point, and the causal-median alternatives.

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
    /opt/miniforge3/envs/JuniperData/bin/python outlier_census.py /tmp/shares-census-out

Original notes
--------------
Lane A3 — claim 5: _SHARES_OUTLIER_FACTOR whole-history median filter.
Counts CIKs losing >=1 point under the generator's own code path (real
_fetch_shares on a cache hit, network blocked) and under causal variants.
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
F = g._SHARES_OUTLIER_FACTOR


def causal_prior_survivors(s: pd.Series) -> pd.Series:
    """C1: first point kept; each later point tested against the median of the
    points KEPT so far (expanding window of survivors)."""
    kept_idx, kept_vals = [], []
    for ts, v in s.items():
        if not kept_vals:
            kept_idx.append(ts); kept_vals.append(v); continue
        med = float(pd.Series(kept_vals).median())
        if med > 0 and not (med / F <= v <= med * F):
            continue
        kept_idx.append(ts); kept_vals.append(v)
    return pd.Series(kept_vals, index=kept_idx)


def causal_prior_raw(s: pd.Series) -> pd.Series:
    """C2: each point tested against the median of ALL prior raw points
    (survivors or not); first point kept."""
    vals = list(s.values)
    keep = []
    for i, v in enumerate(vals):
        if i == 0:
            keep.append(True); continue
        med = float(pd.Series(vals[:i]).median())
        keep.append(not (med > 0 and not (med / F <= v <= med * F)))
    return s[keep]


def causal_inclusive_raw(s: pd.Series) -> pd.Series:
    """C3: each point tested against the median of all raw points up to and
    INCLUDING itself (expanding window, inclusive)."""
    vals = list(s.values)
    keep = []
    for i, v in enumerate(vals):
        med = float(pd.Series(vals[: i + 1]).median())
        keep.append(not (med > 0 and not (med / F <= v <= med * F)))
    return s[keep]


rows = []
for path in sorted((a3lib.CACHE / "shares").glob("*.json")):
    cik = int(path.stem)
    data = a3lib.load_payload(cik)
    best, _ = a3lib.dedup_latest_filed(data)
    pre = a3lib.to_series(best)
    real = g.EquitiesGenerator._fetch_shares(cik, True)
    post_real = real["shares"]
    post_rep = a3lib.whole_history_filter(pre)
    assert len(post_real) == len(post_rep) and (post_real.index == post_rep.index).all(), cik
    c1, c2, c3 = causal_prior_survivors(pre), causal_prior_raw(pre), causal_inclusive_raw(pre)
    rows.append(
        {
            "cik": cik,
            "n_pre": len(pre),
            "n_post": len(post_real),
            "lost": len(pre) - len(post_real),
            "dropped_dates": ";".join(d.strftime("%Y-%m-%d") for d in pre.index.difference(post_real.index)),
            "dropped_vals": ";".join(f"{v:.0f}" for v in pre[pre.index.difference(post_real.index)].values),
            "median": float(pre.median()),
            "c1_differs": set(c1.index) != set(post_real.index),
            "c2_differs": set(c2.index) != set(post_real.index),
            "c3_differs": set(c3.index) != set(post_real.index),
            "c1_n": len(c1),
            "c2_n": len(c2),
            "c3_n": len(c3),
        }
    )
df = pd.DataFrame(rows)
df.to_csv(OUT / "outlier_census.csv", index=False)
print("CIKs:", len(df))
print("CIKs losing >=1 point under the generator's whole-history filter:", int((df["lost"] > 0).sum()), " total points lost:", int(df["lost"].sum()))
print("distribution of points lost:", df["lost"].value_counts().sort_index().to_dict())
print("C1 (expanding median of SURVIVORS so far) keeps a different set than whole-history for:", int(df["c1_differs"].sum()), "CIKs")
print("C2 (expanding median of all PRIOR raw points)  differs for:", int(df["c2_differs"].sum()), "CIKs")
print("C3 (expanding median INCLUSIVE of current point) differs for:", int(df["c3_differs"].sum()), "CIKs")
print("\nCIKs that lose points (cik, n_pre, n_post, dropped_dates, dropped_vals, median):")
print(df[df["lost"] > 0][["cik", "n_pre", "n_post", "dropped_dates", "dropped_vals", "median"]].to_string(index=False))
print("\nCIKs where C1 differs:", sorted(df[df["c1_differs"]]["cik"].tolist()))
print("CIKs where C2 differs:", sorted(df[df["c2_differs"]]["cik"].tolist()))
print("CIKs where C3 differs:", sorted(df[df["c3_differs"]]["cik"].tolist()))
