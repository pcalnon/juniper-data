# SEC shares-cache census instruments (2026-09-08)

The measurement instruments behind the round-37 handoff validation's SEC-shares-cache findings
(juniper-ml, lane A3 of the 2026-09-08 consensus run). Graduated here from a session scratch
directory because an instrument that lives only in scratch is lost with the session, and every
number below was re-derived by it.

All scripts are **read-only on the cache** and **block the network**; each takes the output
directory as its first argument and imports `a3lib.py` from beside itself. Run with the
JuniperData interpreter from this directory.

| Script | Measures | Headline (2026-09-08, cache of 2026-06-03, main 03b7548f) |
|---|---|---|
| `census_cache.py` | payload count, concepts, empties, all-zero, mtimes | 485 payloads (473 dei + 12 us-gaap), 0 empty, one batch 2026-06-03T01:18-01:58Z |
| `universe_diff.py` | bundled universe vs cached set | 503 tickers / 500 CIKs; 15 members with no payload (EL TSN RL META XYZ ABNB TTD STZ DASH TKO UHS HRL MKC LEN ERIE) |
| `first_filing.py`, `final_checks.py` | share of the default window before the first filing | earliest 2009-04-15, median 2009-12-18; mean 43.1% (raw min-filed) / 43.2% (generator-faithful) as of 2026-09-07; 44.85% universe-wide |
| `ko_restatement.py` | latest-filed dedup vs first publication, KO | 3 episodes, 103 trading days, all OVERSTATED: +0.638% (40 d), +0.450% (44 d), +0.104% (19 d) |
| `restatement_population_v2.py` | the same effect, every CIK | 162/485 CIKs with a re-stated end; 155 with >=1 differing row; 17,569 rows; ADM (default prefix) up to +11.55% |
| `first_pub_moved.py` | first available count deferred by the dedup | 9 CIKs; 8 CIKs / 1,225 business-day rows NaN where the figure was public (EXPE 521) |
| `outlier_census.py` | whole-history-median outlier filter | 61/485 CIKs lose >=1 point (92 points); causal alternatives keep a different set for 15 or 16 |
| `placeholder_scan.py` | all-zero / placeholder series | 6 payloads (TAP, DDOG, CVNA, FOX/FOXA, PSKY, BRK.B-scale) -> silent market_cap 0.0, not NaN |
| `warm_cache_trace.py` | which guards run warm vs cold | warm hit skips the concept loop, its empty-units guard and the rescue ladder; the post-load guard still runs and returns None |

Definitions and instrument caveats are in each script's docstring; the handoff that consumed these
numbers records what the evidence cannot support (no live-endpoint comparison, business-day vs
trading-day calendars for tickers other than KO).
