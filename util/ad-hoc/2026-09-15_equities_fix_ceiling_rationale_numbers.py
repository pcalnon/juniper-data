#!/usr/bin/env python3
"""Correct two wrong numbers in juniper-data#404's ceiling rationale, before it merges.

Project: juniper-ml
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-15
Status: ad-hoc -- one-off (applied to the juniper-data head-typo worktree)
Retire when: RETAINED -- ad-hoc scripts are kept as provenance of record (owner policy 2026-08-25)
Related: juniper-data#404; register row APD-DATA-047; round-39 round-2 validation lane A

Lane A refuted two figures I wrote into the comment, the CHANGELOG and the register, and
re-measuring confirms it:

1. **"the largest GENUINE count in the bundled universe (AAPL, 1.70e10)" is false.** AAPL's
   1.70e10 is the largest in the DEFAULT 14-SYMBOL PREFIX, not in the universe. Citigroup
   (CIK 831001) delivers 2.92e10 and NVIDIA (1045810) 2.45e10, both genuine -- both survive the
   filter, verified by running the shipped module. Headroom below the ceiling is therefore
   **3.4x**, not 5.9x.
2. **"18 observations across 24 series sit between 1e11 and 1e13" conflates two measurements.**
   Measured: `(1e11, 1e13]` holds **18 observations across 9 series**; `> 1e11` holds **39
   observations across 24 series**. The 18 and the 24 came from different bands.

And a third thing, which lane A raised as an unverified side observation and which turns out to
be the strongest available argument for the pair of instruments:

3. **The four largest values that PASS the 1e11 ceiling are themselves typos** -- Pentair
   98,419,314,000 (592x its median), Packaging Corp 89,932,185,000 (949x), Regency Centers
   81,867,549,000 (483x), Mid-America 75,009,068,000 (659x). None is caught by the ceiling; all
   four are caught by the RELATIVE filter, and all four deliver correct maxima. The ceiling
   cannot be tightened to reach them without crossing Citigroup's genuine 2.92e10, so the two
   instruments are not redundant and the ceiling is not the one to tighten further. That bears
   directly on the APD-DATA-047 owner decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORK = Path("/home/pcalnon/Development/python/Juniper/worktrees/juniper-data--fix--equities-head-typo-regression--20260915-2130--f3797634")


def sub(path: Path, old: str, new: str, label: str) -> None:
    """Apply one replacement. Re-runnable: a hunk already applied is reported, not re-failed.

    This script was written after its own first hunk had already landed -- the CHANGELOG anchor
    was wrong on the first pass and the generator hunk had succeeded before the failure -- so
    idempotence here is not a nicety, it is what let the remaining hunks be applied at all.
    """
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

sub(
    GEN,
    "# Sited between the largest GENUINE count in the bundled universe (AAPL, 1.70e10) and the\n"
    "# smallest DEMONSTRATED typo in the cache (AIZ, 1.168e11): 5.9x of headroom below it, and\n"
    "# every observed scale error above it. It was 1e13 until 2026-09-15, chosen for headroom\n"
    "# alone, and headroom alone made it nearly inert -- measured over the 486-payload cache, 18\n"
    "# observations across 24 series sit between 1e11 and 1e13, and two of those series (AIZ,\n"
    "# EOG) were DELIVERED, because the relative filter could not judge a typo in a series' first\n"
    "# filings either. A bound chosen for comfort rather than against the data does no work.\n",
    "# Sited between the largest GENUINE count in the cache (Citigroup, 2.92e10; NVIDIA is second\n"
    "# at 2.45e10) and the smallest DEMONSTRATED typo in it (AIZ, 1.168e11): 3.4x of headroom\n"
    "# below. It was 1e13 until 2026-09-15, chosen for headroom alone, and headroom alone made it\n"
    "# nearly inert -- measured over the 486-payload cache, 18 observations across 9 series sit in\n"
    "# (1e11, 1e13], and two of those series (AIZ, EOG) were DELIVERED, because the relative filter\n"
    "# could not judge a typo in a series' opening filings either.\n"
    "#\n"
    "# DO NOT TIGHTEN IT FURTHER TO CHASE THE REST. The four largest values that pass this ceiling\n"
    "# are themselves typos -- Pentair 9.84e10 (592x its own median), Packaging Corp 8.99e10\n"
    "# (949x), Regency Centers 8.19e10 (483x), Mid-America 7.50e10 (659x) -- and every one of them\n"
    "# is caught by the relative filter, which delivers all four correctly. Reaching them with an\n"
    "# absolute bound would mean dropping below Citigroup's genuine 2.92e10 and deleting real\n"
    "# mega-cap history. That is the division of labour: the ceiling exists for the ONE case\n"
    "# nothing relative can reach, a typo in a series' first filing.\n"
    "#\n"
    "# An earlier draft of this comment said 1e11 sits above \"the largest genuine count in the\n"
    "# bundled universe (AAPL, 1.70e10)\" with 5.9x of headroom. AAPL is the largest in the DEFAULT\n"
    "# 14-SYMBOL PREFIX, not in the universe, and the two are not the same population.\n",
    "generator: ceiling rationale numbers",
)

CHANGELOG = WORK / "CHANGELOG.md"

sub(
    CHANGELOG,
    "     typo in a series' *first* filing, which is EOG's case, and `1e13` was chosen for headroom\n"
    "     rather than against the data: over the 486-payload cache, 18 observations across 24 series\n"
    "     sit between `1e11` and `1e13`. `1e11` sits between the largest genuine count in the bundled\n"
    "     universe (AAPL, 1.70e10) and the smallest demonstrated typo in it (AIZ, 1.168e11) -- 5.9x of\n"
    "     headroom below, every observed scale error above. Re-measure with\n"
    "     `util/ad-hoc/2026-09-15_remeasure_shares_cache_figures.py`.\n",
    "     typo in a series' *first* filing, which is EOG's case, and `1e13` was chosen for headroom\n"
    "     rather than against the data: over the 486-payload cache, 18 observations across 9 series\n"
    "     sit in `(1e11, 1e13]` (39 observations across 24 series sit above `1e11` in total). `1e11`\n"
    "     sits between the largest genuine count in the cache (Citigroup, 2.92e10; NVIDIA second at\n"
    "     2.45e10) and the smallest demonstrated typo in it (AIZ, 1.168e11) -- 3.4x of headroom below.\n"
    "\n"
    "     **The ceiling is deliberately not tightened further.** The four largest values that pass it\n"
    "     are themselves typos -- Pentair 9.84e10 (592x its own median), Packaging Corp 8.99e10\n"
    "     (949x), Regency Centers 8.19e10 (483x), Mid-America 7.50e10 (659x) -- and the relative\n"
    "     filter catches every one, delivering all four correctly. Reaching them absolutely would mean\n"
    "     dropping below Citigroup's genuine 2.92e10 and deleting real mega-cap history. The ceiling\n"
    "     exists for the one case nothing relative can reach: a typo in a series' first filing.\n"
    "     Re-measure with `util/ad-hoc/2026-09-15_remeasure_shares_cache_figures.py`.\n",
    "CHANGELOG: ceiling rationale numbers",
)

TEST = WORK / "juniper_data/tests/unit/test_equities_generator.py"

sub(
    TEST,
    "        A series whose first filing is simply the largest genuine count it ever reports must keep\n"
    "        it. AAPL's real maximum (1.70e10) sits 5.9x below the ceiling, so the bound has room for\n"
    "        the largest count in the bundled universe and still catches the smallest demonstrated\n"
    "        typo in the cache (AIZ, 1.168e11).\n",
    "        A series whose first filing is simply the largest genuine count it ever reports must keep\n"
    "        it. The largest genuine count in the cache is Citigroup's 2.92e10, which sits 3.4x below\n"
    "        the ceiling; the smallest demonstrated typo is AIZ's 1.168e11, above it. This fixture\n"
    "        uses AAPL's 1.70e10 -- the largest in the DEFAULT 14-SYMBOL PREFIX -- because that is the\n"
    "        population the rest of this module's fixtures are drawn from.\n",
    "test: correct the headroom claim in the docstring",
)

print("\nceiling rationale corrected in three files")
