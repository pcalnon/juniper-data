#!/usr/bin/env python3
"""Negative control for the entity-tag equivalence script: its sweeps must catch known mutants.

Project:     Juniper
Sub-Project: juniper-data
Application: ad-hoc verification
Author:      Paul Calnon
Version:     1.1.0
License:     MIT
Created:     2026-09-24
Status:      ad-hoc -- one-off verification
Retire when: the equivalence script it checks is retired
Related:     juniper-data#428; round-3 validation, lane A2, F4

WHY THIS EXISTS
---------------
``util/ad-hoc/2026-09-23_verify_entity_tag_list_regex_equivalence.py`` reports 0 mismatches
between the linear and the backtracking entity-tag list grammars. A check that reports 0 for
EVERY pattern has no power, so this feeds the same sweeps a set of language-changing mutants
of the shipped pattern and counts what each sweep finds.

Round-3 validation found the weak spot: its mutant "at most 7 list elements after the first"
drew 0 exhaustive mismatches (seven characters cannot form eight elements) and 1 in 300,000
random inputs. The equivalence script then gained a structured sweep over list-element counts
up to 12. This requires:

* the shipped pattern: 0 mismatches in every sweep;
* every mutant: caught by at least one sweep;
* the four long-list mutants -- at most 7 and at most 10 elements after the first, and a
  whitespace-only element refused from the 9th and from the 12th element on: caught by the
  STRUCTURED sweep, the one built to reach them.

The whitespace-only pair was added when round-1 validation of juniper-data#438 (lane A, F4)
showed the first of them, its "W8", scoring 0 in all three sweeps. The second is the same
mutant one position short of the structured sweep's reach, so it pins that reach.

Run from the repo root::

    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-24_verify_equivalence_sweeps_catch_long_list_mutants.py

Exit 0 when all three hold, 1 otherwise. It takes a minute or two: every mutant is matched
against the three sweeps' 3.5 million inputs.
"""

from __future__ import annotations

import importlib.util
import itertools
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EQUIVALENCE = REPO / "util/ad-hoc/2026-09-23_verify_entity_tag_list_regex_equivalence.py"

_TAG = r'(?:W/)?"[^"]*"'
MUTANTS = {
    "no OWS after the first tag": r"[ \t]*(?:" + _TAG + r")?(?:,[ \t]*(?:" + _TAG + r"[ \t]*)?)*",
    "no empty element after a comma": r"[ \t]*(?:" + _TAG + r"[ \t]*)?(?:,[ \t]*" + _TAG + r"[ \t]*)*",
    "HTAB is not OWS": r"[ ]*(?:" + _TAG + r"[ ]*)?(?:,[ ]*(?:" + _TAG + r"[ ]*)?)*",
    "at most 5 list elements after the first": r"[ \t]*(?:" + _TAG + r"[ \t]*)?(?:,[ \t]*(?:" + _TAG + r"[ \t]*)?){0,5}",
    "at most 7 list elements after the first": r"[ \t]*(?:" + _TAG + r"[ \t]*)?(?:,[ \t]*(?:" + _TAG + r"[ \t]*)?){0,7}",
    "at most 10 list elements after the first": r"[ \t]*(?:" + _TAG + r"[ \t]*)?(?:,[ \t]*(?:" + _TAG + r"[ \t]*)?){0,10}",
    # Lane A's W8: after the 8th element, an element is empty or carries a tag, never OWS alone.
    "whitespace-only element refused from the 9th": r"[ \t]*(?:" + _TAG + r"[ \t]*)?(?:,[ \t]*(?:" + _TAG + r"[ \t]*)?){0,7}(?:,(?:[ \t]*" + _TAG + r"[ \t]*)?)*",
    "whitespace-only element refused from the 12th": r"[ \t]*(?:" + _TAG + r"[ \t]*)?(?:,[ \t]*(?:" + _TAG + r"[ \t]*)?){0,10}(?:,(?:[ \t]*" + _TAG + r"[ \t]*)?)*",
}
LONG_LIST = ("at most 7 list elements after the first", "at most 10 list elements after the first", "whitespace-only element refused from the 9th", "whitespace-only element refused from the 12th")


def _load_equivalence_script():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("equivalence_under_test", EQUIVALENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    equivalence = _load_equivalence_script()
    exhaustive = ["".join(chars) for length in range(equivalence.EXHAUSTIVE_MAX_LEN + 1) for chars in itertools.product(equivalence.ALPHABET, repeat=length)]
    rng = random.Random(20260923)  # the equivalence script's own seed, so this is its random sweep
    rand = ["".join(rng.choice(equivalence.TOKENS) for _ in range(rng.randint(0, equivalence.RANDOM_MAX_TOKENS))) for _ in range(equivalence.RANDOM_CASES)]
    structured = equivalence._structured_inputs()
    sweeps = {"exhaustive": exhaustive, "random": rand, "structured": structured}
    print(f"sweeps: exhaustive={len(exhaustive):,} random={len(rand):,} structured={len(structured):,}")

    def mismatches(pattern: re.Pattern[str]) -> dict[str, int]:
        return {name: sum((equivalence.OLD.fullmatch(text) is None) != (pattern.fullmatch(text) is None) for text in inputs) for name, inputs in sweeps.items()}

    failures: list[str] = []
    shipped = mismatches(equivalence.NEW)
    print(f"{'the shipped pattern':42s} {shipped}")
    if any(shipped.values()):
        failures.append(f"the shipped pattern disagrees with the old one: {shipped}")
    for label, pattern in MUTANTS.items():
        found = mismatches(re.compile(pattern))
        print(f"{label:42s} {found}")
        if not any(found.values()):
            failures.append(f"no sweep catches the mutant '{label}'")
        if label in LONG_LIST and not found["structured"]:
            failures.append(f"the structured sweep misses the long-list mutant '{label}'")
    print()
    if failures:
        print("FAIL")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"PASS: the shipped pattern agrees in every sweep, and each of {len(MUTANTS)} mutants is caught -- the long-list ones by the structured sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
