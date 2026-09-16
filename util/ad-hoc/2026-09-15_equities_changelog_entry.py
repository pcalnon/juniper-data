#!/usr/bin/env python3
"""Add the head-typo regression entry to juniper-data's CHANGELOG [Unreleased] / Fixed.

Project: juniper-ml
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-15
Status: ad-hoc -- one-off (applied to the juniper-data head-typo worktree)
Retire when: RETAINED -- ad-hoc scripts are kept as provenance of record (owner policy 2026-08-25)
Related: juniper-data#395; register row APD-DATA-047
"""

from __future__ import annotations

from pathlib import Path

WORK = Path("/home/pcalnon/Development/python/Juniper/worktrees/juniper-data--fix--equities-head-typo-regression--20260915-2130--f3797634")

ANCHOR = "## [Unreleased]\n\n### Fixed\n\n"

ENTRY = """- **The causal scale-typo filter shipped in #395 included each point in the median that judged
  it, so a cover-page typo in a series' opening filings survived and was delivered.**
  `pandas.Series.expanding().median()` at position *i* includes position *i*, so an outlier
  dominates its own basis and no `min_periods` value repairs it -- the problem is membership, not
  sample size. Measured against the shipped module and the real SEC cache, AIZ's typo at position
  1 and EOG's at position 0 survived at `min_periods` 1, 2 and 3 alike:

  | ticker | delivered at 4.0.0 | correct | factor |
  |--------|--------------------|---------|--------|
  | AIZ    | 116,799,796,000    | 117,926,517 | 990x |
  | EOG    | 251,931,774,000    | 587,723,622 | 428x |

  The pre-#395 filter caught both, but only by consulting filings that had not happened yet --
  the look-ahead #395 was written to remove. So this is not a revert; it is two changes that keep
  causality and close the hole, and **neither is redundant -- the two regressions prove it one
  each**:

  1. **Every point is judged against what was accepted strictly BEFORE it**
     (`generators/equities/generator.py`). The basis is the median of the values already
     *accepted*, not of everything already *seen*, so a rejected typo cannot go on to poison its
     neighbours -- `expanding().median().shift(1)` is the tempting one-liner and gets that wrong,
     deleting the genuine third filing after a typo in the second. `min_periods` is gone: with
     nothing filed before it, position 0 has no relative basis at all, and inventing one means
     looking forward. This alone returns AIZ (position 1) to 117,926,517.
  2. **`_SHARES_ABSOLUTE_CEILING` tightened from `1e13` to `1e11`.** Nothing relative can reach a
     typo in a series' *first* filing, which is EOG's case, and `1e13` was chosen for headroom
     rather than against the data: over the 486-payload cache, 18 observations across 24 series
     sit between `1e11` and `1e13`. `1e11` sits between the largest genuine count in the bundled
     universe (AAPL, 1.70e10) and the smallest demonstrated typo in it (AIZ, 1.168e11) -- 5.9x of
     headroom below, every observed scale error above. Re-measure with
     `util/ad-hoc/2026-09-15_remeasure_shares_cache_figures.py`.

  **A known false positive is pinned rather than papered over.** Two genuine share classes in one
  series (Berkshire's Class A at 941,481 and Class B at 1,071,666,977) are 1,138x apart, which no
  scale filter can distinguish from a typo of the same magnitude; the later class is now rejected.
  It survived before only because `min_periods=3` switched the filter off for a series' opening
  points -- the same hole that delivered AIZ. The remedy is a class-aware `dei` lookup, recorded
  as deferred work on the register row; until it lands, one filtered class is the better trade
  than a scale typo in every series' first filings.

  **`equities` and `equities_seq` go to generator version `5.0.0`.** `generator_version` is hashed
  into `dataset_id`, so without the bump the corrected values would be served under the ID that
  carried the wrong ones. Every artifact minted at `4.0.0` for a symbol with such a typo carries
  it.

  The staleness annotation also stops naming the issuer as the cause: Ford, Nike, Hershey and
  Regeneron are all flagged over a window ending today and all four file a share count on every
  10-Q -- what stopped is this cache's extraction of the `dei` concept, not the company. The
  annotation is still correct; it now says *available* rather than *filed*. juniper-data#395.

"""


path = WORK / "CHANGELOG.md"
text = path.read_text()
assert text.count(ANCHOR) == 1, "Unreleased/Fixed anchor is not unique"
path.write_text(text.replace(ANCHOR, ANCHOR + ENTRY, 1))
print("ok  CHANGELOG entry added")
