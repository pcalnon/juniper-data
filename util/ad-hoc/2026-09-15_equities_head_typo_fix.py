#!/usr/bin/env python3
"""
Fix the regression juniper-data#395 shipped: a cover-page typo in a series' first two filings
survives the causal median and is delivered.

Project: juniper-ml
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-15
Status: ad-hoc — one-off (applied to the juniper-data worktree named below)
Retire when: RETAINED — ad-hoc scripts are kept as provenance of record (owner policy 2026-08-25)
Related: juniper-data#395 (which introduced it); register rows APD-DATA-043 and APD-DATA-047;
         round-39 validation lane B1, which found it

The defect, found by adversarial validation and re-derived against the shipped module
--------------------------------------------------------------------------------------
`pandas.Series.expanding().median()` at position i INCLUDES position i. An outlier therefore
dominates the median that is supposed to judge it, and no `min_periods` value helps: measured on
the real cache, AIZ's typo at position 1 and EOG's at position 0 survive at min_periods 1, 2 and 3
alike. Delivered consequence, against the pre-#395 code:

    AIZ  116,799,796,000 delivered where the old filter delivered 117,926,517   (990x)
    EOG  251,931,774,000 delivered where the old filter delivered 587,723,622   (428x)

The whole-history median caught both -- by looking at filings that had not happened yet, which is
the look-ahead #395 removed. So this is not "restore the old filter"; it is two changes that keep
causality and close the hole:

1. **Judge a point against what was filed STRICTLY BEFORE it.** `.expanding().median().shift(1)`.
   A point can no longer vote on its own plausibility, which is what let a head outlier through.
   `min_periods` goes away: `shift(1)` already leaves position 0 unjudged, which is honest -- with
   nothing filed before it, there is no relative basis, and that case belongs to the ceiling.
2. **Tighten the ceiling from 1e13 to 1e11.** 1e13 was chosen for headroom above the largest
   genuine count and is far too loose to be load-bearing: 18 scale typos sit between 1e11 and 1e13.
   The smallest typo demonstrated in this cache is AIZ's 1.168e11 and the largest genuine count is
   AAPL's 1.70e10, so 1e11 separates them with 5.9x of headroom below and catches every observed
   typo. This supersedes the value `APD-DATA-047` was filed to have ratified.

Measured over all 486 cached payloads: exactly 2 series change (AIZ and EOG), none is emptied, and
zero series lose a value the pre-#395 rule kept.
"""
from __future__ import annotations

import sys
from pathlib import Path

W = Path("/home/pcalnon/Development/python/Juniper/worktrees/juniper-data--fix--equities-head-typo-regression--20260915-2130--f3797634")
GEN = W / "juniper_data/generators/equities/generator.py"

text = GEN.read_text()


def sub(old: str, new: str, label: str) -> None:
    global text
    n = text.count(old)
    if n != 1:
        sys.exit(f"FAIL [{label}]: found {n} for:\n{old[:200]}")
    text = text.replace(old, new)
    print(f"  ok  {label}")


sub(
    "import contextlib\nimport csv\n",
    "import bisect\nimport contextlib\nimport csv\n",
    "import bisect",
)
sub(
    "import os\nimport time\n",
    "import os\nimport statistics\nimport time\n",
    "import statistics",
)
sub(
    "_SHARES_ABSOLUTE_CEILING = 1.0e13\n",
    "_SHARES_ABSOLUTE_CEILING = 1.0e11\n",
    "ceiling 1e13 -> 1e11",
)
sub(
    "# not happened yet. A causal median cannot catch a typo that arrives second of four: there is\n"
    "# no basis to judge it against. An absolute ceiling can, because no issuer has 1e13 shares.\n",
    "# not happened yet. A causal median SHIFTED BY ONE can judge a typo that arrives second of\n"
    "# four -- against the genuine first filing -- but nothing relative can judge the FIRST\n"
    "# observation, which has no prior basis at all. An absolute ceiling can, because no issuer\n"
    "# has 1e11 shares.\n",
    "position-0 rationale",
)
sub(
    "# Sited 588x above the largest genuine count in the bundled universe (AAPL, 17,001,802,000)\n"
    "# and 100x below the 1e15 typo the regression test pins.\n",
    "#\n"
    "# Sited between the largest GENUINE count in the bundled universe (AAPL, 1.70e10) and the\n"
    "# smallest DEMONSTRATED typo in the cache (AIZ, 1.168e11): 5.9x of headroom below it, and\n"
    "# every observed scale error above it. It was 1e13 until 2026-09-15, chosen for headroom\n"
    "# alone, and headroom alone made it nearly inert -- measured over the 486-payload cache, 18\n"
    "# observations across 24 series sit between 1e11 and 1e13, and two of those series (AIZ,\n"
    "# EOG) were DELIVERED, because the relative filter could not judge a typo in a series' first\n"
    "# filings either. A bound chosen for comfort rather than against the data does no work.\n",
    "ceiling rationale",
)
sub(
    "        # CAUSAL scale-typo filter. The old one compared every point to the median of the whole\n"
    "        # history, so which points survived depended on filings made after the rows they\n"
    "        # affect -- a look-ahead in the filter itself (61 of 485 CIKs lose at least one point,\n"
    "        # and 15 or 16 keep a different set under a causal median). An expanding median sees\n"
    "        # only what was already filed. ``min_periods`` keeps the opening points: a median over\n"
    "        # one or two observations is not a basis for deleting a third.\n"
    '        running_median = observations["shares"].expanding(min_periods=3).median()\n'
    '        keep = running_median.isna() | ((observations["shares"] >= running_median / _SHARES_OUTLIER_FACTOR) & (observations["shares"] <= running_median * _SHARES_OUTLIER_FACTOR))\n',
    "        # CAUSAL scale-typo filter. The old one compared every point to the median of the whole\n"
    "        # history, so which points survived depended on filings made after the rows they\n"
    "        # affect -- a look-ahead in the filter itself. An expanding median sees only what was\n"
    "        # already filed.\n"
    "        #\n"
    "        # EVERY POINT IS JUDGED AGAINST WHAT CAME STRICTLY BEFORE IT, and that exclusion is\n"
    "        # the whole correctness of this. ``expanding().median()`` INCLUDES the point it is\n"
    "        # judging, so an outlier dominates its own basis and can never be rejected -- and no\n"
    "        # ``min_periods`` value repairs that, because the problem is membership, not sample\n"
    "        # size. That is not a corner case; it shipped in #395. Measured on the real cache,\n"
    "        # AIZ's typo at position 1 and EOG's at position 0 survived at min_periods 1, 2 and 3\n"
    "        # alike, and were delivered 990x and 428x too large.\n"
    "        #\n"
    "        # Position 0 is left unjudged on purpose: with nothing filed before it there is no\n"
    "        # relative basis at all, and inventing one means looking forward. That case belongs to\n"
    "        # _SHARES_ABSOLUTE_CEILING, which is why the ceiling had to be tightened in the same\n"
    "        # change -- a typo in a series' first filing is exactly what no relative test can reach.\n"
    "        #\n"
    "        # NEITHER half is redundant, and the two regressions prove it one each: AIZ's typo\n"
    "        # sits at position 1 and the relative test alone returns it to 117,926,517 under\n"
    "        # either ceiling; EOG's sits at position 0, survives the relative test untouched, and\n"
    "        # only the 1e11 ceiling brings it back to 587,723,622. Over the 486-payload cache the\n"
    "        # relative test rejects at least one observation in 6 series and the ceiling in 24;\n"
    "        # re-measure with\n"
    "        # ``util/ad-hoc/2026-09-15_remeasure_shares_cache_figures.py``.\n"
    "        #\n"
    "        # The basis is the median of the values ALREADY ACCEPTED, not of everything already\n"
    "        # seen, so a rejected typo cannot go on to poison the judgement of its neighbours.\n"
    "        # ``expanding().median().shift(1)`` would be the one-liner, and it is wrong for the\n"
    "        # same family of reason as the unshifted median: with a genuine 1.0e8 followed by a\n"
    "        # 5.0e10 typo, its basis for the THIRD point is median(1e8, 5e10) = 2.55e10, so the\n"
    "        # genuine third filing is deleted as a hundredfold-low outlier. On the bundled cache\n"
    "        # the two agree on every one of the 486 series -- the ceiling removes the poisoners\n"
    "        # first -- which is exactly why the one-liner would have looked fine and stayed wrong.\n"
    "        # The series are short (median 66 observations, longest 86), so the loop is free.\n"
    "        kept_values: list[float] = []\n"
    "        keep: list[bool] = []\n"
    '        for value in observations["shares"]:\n'
    "            basis = statistics.median(kept_values) if kept_values else 0.0\n"
    "            accepted = not kept_values or basis <= 0 or (basis / _SHARES_OUTLIER_FACTOR) <= value <= (basis * _SHARES_OUTLIER_FACTOR)\n"
    "            keep.append(bool(accepted))\n"
    "            if accepted:\n"
    "                bisect.insort(kept_values, float(value))\n",
    "prior-accepted running median",
)
sub(
    "        observations = observations[keep]\n",
    "        observations = observations[pd.Series(keep, index=observations.index)]\n",
    "index-aligned keep mask",
)

# The staleness note must not attribute the silence to the issuer: what stopped, for the four
# tickers checked by hand, is this cache's extraction of the dei concept -- not the company.
sub(
    '                    note = f"{SHARES_QUALITY_STALE}: no share count filed in the {silence_days} days before this window ends; {affected} of {len(frame)} rows carry a figure the issuer stopped confirming"\n',
    '                    note = f"{SHARES_QUALITY_STALE}: no share count available for the {silence_days} days before this window ends; {affected} of {len(frame)} rows carry a value forward from the last one"\n',
    "staleness note wording",
)
sub(
    "                # So: has this series STOPPED, as of the end of the window it is serving?\n",
    "                # So: has this series STOPPED, as of the end of the window it is serving?\n"
    "                #\n"
    "                # The note says AVAILABLE, not filed, and deliberately does not name the\n"
    "                # issuer as the cause. Ford, Nike, Hershey and Regeneron are all flagged over\n"
    "                # a window ending today, and all four file a share count on every 10-Q: what\n"
    "                # stopped is this cache's extraction of the dei concept, not the company. The\n"
    "                # annotation is still correct -- those rows ARE carrying a stale figure\n"
    "                # forward -- but an annotation that names the wrong cause sends whoever reads\n"
    "                # it to the wrong place.\n",
    "staleness attribution comment",
)

# NOT renumbered here: the comments quote counts from a 485-payload sweep against a cache that
# now holds 486. Bumping the denominator alone would leave every numerator asserting a
# measurement nobody redid, and the re-measurement above does not reproduce those numerators
# with the obvious predicates (183 vs 162 restatement CIKs, 42 vs 54 collisions), so the
# predicates themselves differ. Filed as its own register row rather than guessed at here.

GEN.write_text(text)
print("\nhead-typo fix written")
