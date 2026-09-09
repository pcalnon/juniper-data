"""Traces which guards and rescue rungs run on a warm cache hit vs cold, in a SCRATCH cache directory (never the real one).

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
    /opt/miniforge3/envs/JuniperData/bin/python warm_cache_trace.py /tmp/shares-census-out

Original notes
--------------
Lane A3 — claim 8: what runs on a WARM cache.  Uses a SCRATCH cache dir
(JUNIPER_DATA_EQUITIES_CACHE_DIR is read at import time into _CACHE_DIR), never
the real one.  Network is blocked; instrumented _sec_get and
_fetch_shares_from_facts record whether they are reached.
"""

import json
import os
import sys
from pathlib import Path

SCRATCH_CACHE = Path(sys.argv[1]) / "scratch_cache"
(SCRATCH_CACHE / "shares").mkdir(parents=True, exist_ok=True)
os.environ["JUNIPER_DATA_EQUITIES_CACHE_DIR"] = str(SCRATCH_CACHE)

import juniper_data.generators.equities.generator as g  # noqa: E402

assert g._CACHE_DIR == SCRATCH_CACHE, g._CACHE_DIR
calls = {"sec_get": 0, "facts": 0}


def _sec_get(*a, **k):
    calls["sec_get"] += 1
    raise RuntimeError("NETWORK BLOCKED by lane A3")


def _facts(cik):
    calls["facts"] += 1
    raise RuntimeError("LADDER REACHED (blocked) by lane A3")


g._sec_get = _sec_get
g.EquitiesGenerator._fetch_shares_from_facts = staticmethod(_facts)

# Case A: warm cache with a populated companyconcept payload (copy KO's real one)
real_ko = json.loads((Path.home() / ".cache/juniper_data/equities/shares/0000021344.json").read_text())
(SCRATCH_CACHE / "shares" / "0000021344.json").write_text(json.dumps(real_ko))
calls.update(sec_get=0, facts=0)
res = g.EquitiesGenerator._fetch_shares(21344, True)
print(f"A populated warm cache: rows={len(res)} quality={res['shares_quality'].iloc[0]} origin={res['shares_origin'].iloc[0]} sec_get calls={calls['sec_get']} ladder calls={calls['facts']}")

# Case B: warm cache holding the truthy-EMPTY payload
(SCRATCH_CACHE / "shares" / "0000000001.json").write_text(json.dumps({"units": {"shares": {}}}))
calls.update(sec_get=0, facts=0)
res = g.EquitiesGenerator._fetch_shares(1, True)
print(f"B EMPTY warm cache {{'units': {{'shares': {{}}}}}}: returns {res!r} sec_get calls={calls['sec_get']} ladder calls={calls['facts']}  -> guard at :910 ran, no refetch, no ladder")

# Case C: warm cache holding a LADDER-shaped payload (what _fetch_shares_from_facts returns: {'units': ...} only)
ladder_payload = {"units": {"shares": [{"end": "2020-12-31", "val": 1000000.0, "filed": "2021-02-01"}, {"end": "2021-03-31", "val": 1010000.0, "filed": "2021-05-01"}]}}
(SCRATCH_CACHE / "shares" / "0000000002.json").write_text(json.dumps(ladder_payload))
calls.update(sec_get=0, facts=0)
res = g.EquitiesGenerator._fetch_shares(2, True)
print(f"C ladder-shaped warm cache: rows={len(res)} quality={res['shares_quality'].iloc[0]!r} origin={res['shares_origin'].iloc[0]!r} sec_get calls={calls['sec_get']} ladder calls={calls['facts']}")

# Case D: cold cache -> concept loop -> (blocked) ; shows the path that WOULD run
calls.update(sec_get=0, facts=0)
try:
    g.EquitiesGenerator._fetch_shares(3, True)
except RuntimeError as exc:
    print(f"D cold cache: raised {exc!s} after sec_get calls={calls['sec_get']} ladder calls={calls['facts']}")

# Case E: use_cache=False with a populated cache present -> ignores cache, goes to network
calls.update(sec_get=0, facts=0)
try:
    g.EquitiesGenerator._fetch_shares(21344, False)
except RuntimeError as exc:
    print(f"E use_cache=False, populated cache present: raised {exc!s} after sec_get calls={calls['sec_get']}")

# Case F: corrupt JSON on disk -> data None -> network
(SCRATCH_CACHE / "shares" / "0000000004.json").write_text("{not json")
calls.update(sec_get=0, facts=0)
try:
    g.EquitiesGenerator._fetch_shares(4, True)
except RuntimeError as exc:
    print(f"F corrupt cache file: raised {exc!s} after sec_get calls={calls['sec_get']}")
print("scratch cache used:", SCRATCH_CACHE)
