#!/usr/bin/env python3
"""Rewrite the CHANGELOG's item 1 for the corrected (non-absorbing) basis.

Project: juniper-ml
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-15
Status: ad-hoc -- one-off (applied to the juniper-data head-typo worktree)
Retire when: RETAINED -- ad-hoc scripts are kept as provenance of record (owner policy 2026-08-25)
Related: juniper-data#404; round-39 round-2 validation lane B1
"""

from __future__ import annotations

import sys
from pathlib import Path

WORK = Path("/home/pcalnon/Development/python/Juniper/worktrees/juniper-data--fix--equities-head-typo-regression--20260915-2130--f3797634")
CHANGELOG = WORK / "CHANGELOG.md"

OLD = """  1. **Every point is judged against what was accepted strictly BEFORE it**
     (`generators/equities/generator.py`). The basis is the median of the values already
     *accepted*, not of everything already *seen*, so a rejected typo cannot go on to poison its
     neighbours -- `expanding().median().shift(1)` is the tempting one-liner and gets that wrong,
     deleting the genuine third filing after a typo in the second. `min_periods` is gone: with
     nothing filed before it, position 0 has no relative basis at all, and inventing one means
     looking forward. This alone returns AIZ (position 1) to 117,926,517.
"""

NEW = """  1. **Every point is judged against the LOWER MEDIAN of what was filed strictly BEFORE it**
     (`generators/equities/generator.py`) -- `prior[(n-1)//2]` of the sorted priors. Both halves of
     that were chosen against a failure the other half causes, and both were found by running the
     code rather than reading it.

     - **Prior, not prior-*accepted*.** Excluding rejected values looks strictly safer and makes
       the filter **absorbing**: if the first value a series offers is a typo the ceiling cannot
       reach, the accepted set is that typo alone, every genuine value is more than a hundredfold
       away from it, and nothing is ever accepted again. The whole real series is deleted and the
       typo is what ships, with no way back, because only an acceptance could widen the basis.
       Both ingredients are in the bundled cache -- a position-0 typo is real (EOG), and four
       series carry sub-ceiling ~1000x typos (PNR 9.84e10, PKG 8.99e10, REG 8.19e10, MAA 7.50e10).
       Only their coincidence is absent, and the shares cache has a 7-day TTL. Pinned by
       `test_a_sub_ceiling_typo_in_the_first_filing_does_not_delete_the_series`, which loses
       **0 of 20** genuine counts under the accepted-only basis and 19 of 20 survive under this one.
     - **Lower median, not the interpolating one.** `statistics.median` of two disagreeing values
       returns their mean, a magnitude neither is near: after a genuine 1.0e8 and a 5.0e10 typo the
       basis for the third point becomes 2.55e10 and the genuine third filing is deleted as a
       hundredfold-low outlier. The lower median is always a number some filing actually reported.

     `min_periods` is gone: with nothing filed before it, position 0 has no relative basis at all,
     and inventing one means looking forward. This alone returns AIZ (position 1) to 117,926,517.

     All three candidate bases deliver **identical multisets across the 483 in-bounds series** of
     the real cache, because the ceiling removes the poisoners before the relative test runs --
     which is exactly why the wrong one looked correct. The separation is visible only on
     constructed shapes; `util/ad-hoc/2026-09-15_compare_outlier_basis_designs.py` is that
     comparison and `util/ad-hoc/2026-09-15_verify_lower_median_over_cache.py` is the whole-cache
     equivalence check.
"""

text = CHANGELOG.read_text()
if NEW in text:
    print("  --  already applied")
elif text.count(OLD) != 1:
    sys.exit(f"FAIL: found {text.count(OLD)} occurrences of the item-1 paragraph")
else:
    CHANGELOG.write_text(text.replace(OLD, NEW, 1))
    print("  ok  CHANGELOG item 1 rewritten for the non-absorbing basis")
