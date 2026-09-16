#!/usr/bin/env python3
"""Remove the absorbing state from juniper-data#404's scale-typo filter, before it merges.

Project: juniper-ml
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-15
Status: ad-hoc -- one-off (applied to the juniper-data head-typo worktree)
Retire when: RETAINED -- ad-hoc scripts are kept as provenance of record (owner policy 2026-08-25)
Related: juniper-data#404; register row APD-DATA-050;
         round-39 round-2 validation lane B1, which found the absorbing state

#404 judged each point against the median of the values already ACCEPTED. That closes the
poisoning hole -- a rejected typo cannot drag the basis for its neighbours -- and opens a worse
one: if the FIRST value a series offers is a typo the absolute ceiling does not reach, then
``kept_values == [typo]``, every genuine value afterwards is more than 100x away from it, and
nothing is ever accepted again. The whole genuine series is deleted and the typo is what ships.
There is no recovery path, because only an acceptance can widen the basis.

Both ingredients are in the bundled cache: a position-0 scale typo occurs for real (EOG), and
sub-ceiling ~1000x typos occur in four series (PNR 9.84e10, PKG 8.99e10, REG 8.19e10, MAA 7.50e10,
all below the 1e11 ceiling). Only their coincidence is absent today, and the shares cache has a
7-day TTL.

The fix is the LOWER MEDIAN of prior SEEN values -- ``seen[(len(seen) - 1) // 2]`` over the sorted
priors:

* **Seen, not accepted**, so a rejected value still counts towards the sample and the basis
  re-converges on the genuine population within a couple of observations. That is what kills the
  absorbing state.
* **Lower median, not the interpolating one**, so the basis is always a number some filing actually
  reported. An ordinary median over two disagreeing values lands halfway between them, at a
  magnitude neither is near -- which is exactly how the plain shifted median deletes a genuine
  third filing after a typo in the second. The lower median cannot do that.

Measured (``util/ad-hoc/2026-09-15_compare_outlier_basis_designs.py`` and
``2026-09-15_verify_lower_median_over_cache.py``): identical delivered multisets on all 483
in-bounds series of the real cache -- zero observations gained, zero lost, none emptied -- while on
the adversarial shapes it keeps 19 of 20 genuine values where #404 kept 0, and 4 of 4 where the
plain shifted median keeps 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORK = Path("/home/pcalnon/Development/python/Juniper/worktrees/juniper-data--fix--equities-head-typo-regression--20260915-2130--f3797634")


def sub(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text and old not in text:
        print(f"  --  {label} (already applied)")
        return
    n = text.count(old)
    if n != 1:
        sys.exit(f"FAIL [{label}]: found {n} occurrences in {path.name} of:\n{old[:200]}")
    path.write_text(text.replace(old, new, 1))
    print(f"  ok  {label}")


GEN = WORK / "juniper_data/generators/equities/generator.py"

# ------------------------------------------------------------------ the loop
sub(
    GEN,
    "        kept_values: list[float] = []\n"
    "        keep: list[bool] = []\n"
    '        for value in observations["shares"]:\n'
    "            basis = statistics.median(kept_values) if kept_values else 0.0\n"
    "            accepted = not kept_values or basis <= 0 or (basis / _SHARES_OUTLIER_FACTOR) <= value <= (basis * _SHARES_OUTLIER_FACTOR)\n"
    "            keep.append(bool(accepted))\n"
    "            if accepted:\n"
    "                bisect.insort(kept_values, float(value))\n",
    "        prior_values: list[float] = []\n"
    "        keep: list[bool] = []\n"
    '        for value in observations["shares"]:\n'
    "            if prior_values:\n"
    "                # LOWER median: the element at (n-1)//2 of the sorted priors, which is always a\n"
    "                # number some filing actually reported.\n"
    "                basis = prior_values[(len(prior_values) - 1) // 2]\n"
    "                accepted = basis <= 0 or (basis / _SHARES_OUTLIER_FACTOR) <= value <= (basis * _SHARES_OUTLIER_FACTOR)\n"
    "            else:\n"
    "                accepted = True\n"
    "            keep.append(bool(accepted))\n"
    "            bisect.insort(prior_values, float(value))\n",
    "the loop: lower median over prior SEEN values",
)

# ------------------------------------------------------------------ the comment
sub(
    GEN,
    "        # The basis is the median of the values ALREADY ACCEPTED, not of everything already\n"
    "        # seen, so a rejected typo cannot go on to poison the judgement of its neighbours.\n"
    "        # ``expanding().median().shift(1)`` would be the one-liner, and it is wrong for the\n"
    "        # same family of reason as the unshifted median: with a genuine 1.0e8 followed by a\n"
    "        # 5.0e10 typo, its basis for the THIRD point is median(1e8, 5e10) = 2.55e10, so the\n"
    "        # genuine third filing is deleted as a hundredfold-low outlier. On the bundled cache\n"
    "        # the two agree on every one of the 486 series -- the ceiling removes the poisoners\n"
    "        # first -- which is exactly why the one-liner would have looked fine and stayed wrong.\n"
    "        # The series are short (median 66 observations, longest 86), so the loop is free.\n",
    "        # The basis is the LOWER MEDIAN of the prior values -- ``prior[(n-1)//2]`` of the sorted\n"
    "        # priors -- and both halves of that were chosen against a failure the other half causes.\n"
    "        #\n"
    "        # PRIOR, not prior-ACCEPTED. Excluding rejected values looks strictly safer and is not:\n"
    "        # it makes the filter ABSORBING. If the first value a series offers is a typo the\n"
    "        # ceiling cannot reach, the accepted set is that typo alone, every genuine value is more\n"
    "        # than a hundredfold away from it, and nothing is ever accepted again -- the entire real\n"
    "        # series is deleted and the typo is what ships. Only an acceptance could widen the\n"
    "        # basis, so there is no way back. Counting rejected values towards the SAMPLE (never\n"
    "        # towards the output) lets the basis re-converge within a couple of observations. Both\n"
    "        # ingredients for that trap are in the bundled cache: a position-0 typo is real (EOG),\n"
    "        # and four series carry sub-ceiling ~1000x typos (PNR 9.84e10, PKG 8.99e10, REG 8.19e10,\n"
    "        # MAA 7.50e10); only their coincidence is absent, and the cache TTL is 7 days.\n"
    "        #\n"
    "        # LOWER median, not the interpolating one. ``statistics.median`` of two disagreeing\n"
    "        # values returns their mean, a magnitude neither is near: with a genuine 1.0e8 followed\n"
    "        # by a 5.0e10 typo, the basis for the THIRD point becomes 2.55e10 and the genuine third\n"
    "        # filing is deleted as a hundredfold-low outlier. The lower median is always a number\n"
    "        # some filing actually reported, so it cannot land in between.\n"
    "        #\n"
    "        # On the bundled cache all three candidates deliver identical multisets across the 483\n"
    "        # in-bounds series -- the ceiling removes the poisoners before the relative test runs --\n"
    "        # which is exactly why the wrong one would have looked fine. The separation is only\n"
    "        # visible on the shapes in ``util/ad-hoc/2026-09-15_compare_outlier_basis_designs.py``.\n"
    "        # The series are short (median 66 observations, longest 86), so the loop is free.\n",
    "the comment: why prior-seen and why the lower median",
)

sub(
    GEN,
    "        observations = observations[pd.Series(keep, index=observations.index)]\n",
    "        observations = observations[pd.Series(keep, index=observations.index)]\n",
    "keep mask unchanged",
)

# ------------------------------------------------------------------ statistics import is now unused
text = GEN.read_text()
if "statistics." not in text.replace("import statistics", ""):
    sub(GEN, "import os\nimport statistics\nimport time\n", "import os\nimport time\n", "drop the now-unused statistics import")

print("\nabsorbing state removed")
