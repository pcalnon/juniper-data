#!/usr/bin/env python3
"""Prove the linear ``_ENTITY_TAG_LIST`` accepts exactly the language the backtracking one did.

Project:     Juniper
Sub-Project: juniper-data
Application: ad-hoc verification
Author:      Paul Calnon
Version:     1.1.0
License:     MIT

WHY THIS EXISTS
---------------
Round-2 validation of juniper-data#428 found the entity-tag list grammar in
``juniper_data/api/http_cache.py`` backtracking exponentially: an element with no tag
was ``[ \\t]*`` followed by ``[ \\t]*``, so each ``", "`` could be split two ways and
``", " * 22 + "x"`` took about 1.8 s, doubling per element. It runs on the event loop
for a GET and under the store's ``_version_lock`` for a PATCH.

The fix moves the trailing whitespace INSIDE the optional tag group, so an element is
``OWS`` or ``OWS TAG OWS`` -- the same two shapes as before, with one way to match each.
That is an argument, and this script is the check behind it: the old and new patterns
must agree on every input of an exhaustive small-alphabet sweep, a token-built random sweep,
and a STRUCTURED sweep over list-element counts from 0 to 12.

The structured sweep was added when round-3 validation (lane A2, F4) showed the other two
barely reach long lists: a mutant that differed only from the eighth list element on produced
one mismatch in 300,000 random inputs, and seven characters is too short for eight elements.
For every count it takes every sequence of elements over three shapes -- empty, a strong tag,
and a weak tag wrapped in OWS -- then puts a malformed element at every position of an
all-tag and an all-empty list, and joins tag lists with every separator spelling. Long
well-formed lists and long lists broken at any one position are therefore reached by
construction, not by chance. Its negative control,
``util/ad-hoc/2026-09-24_verify_equivalence_sweeps_catch_long_list_mutants.py``, feeds these
sweeps language-changing mutants, the long-list ones among them.

The new pattern is imported from the module, not copied, so the check covers what
ships; the old one is the literal from the PR head it replaced (3ecb106).

Run from the repo root::

    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-23_verify_entity_tag_list_regex_equivalence.py

Exit 0 when there are zero mismatches; exit 1 otherwise. Timings are printed, never
asserted -- they are evidence of the shape of the cost, not a gate.
"""

from __future__ import annotations

import itertools
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from juniper_data.api.http_cache import _ENTITY_TAG_LIST as NEW  # noqa: E402

OLD = re.compile(r'[ \t]*(?:(?:W/)?"[^"]*")?[ \t]*(?:,[ \t]*(?:(?:W/)?"[^"]*")?[ \t]*)*')
EXPECTED_NEW = r'[ \t]*(?:(?:W/)?"[^"]*"[ \t]*)?(?:,[ \t]*(?:(?:W/)?"[^"]*"[ \t]*)?)*'

# Every character the grammar distinguishes, plus one it does not ("a"). "*" is here
# because the callers special-case it before the list grammar runs.
ALPHABET = [" ", "\t", ",", '"', "W", "/", "a", "*"]
EXHAUSTIVE_MAX_LEN = 7

# Grammar-shaped pieces, so random strings are often WELL-FORMED long lists -- the side
# of the language a character-level sweep barely reaches.
TOKENS = ['"a"', '"a,b"', 'W/"a"', 'W/""', '""', ",", ", ", " ,", "\t", " ", "W/", '"', "W", "x", "*"]
RANDOM_CASES = 300_000
RANDOM_MAX_TOKENS = 10

# The structured sweep (F4). Elements, not characters: every sequence of STRUCTURED_MAX_ELEMENTS
# or fewer over ELEMENT_SHAPES, then one BROKEN_ELEMENTS entry at each position, then each
# SEPARATORS spelling between tags.
STRUCTURED_MAX_ELEMENTS = 12
ELEMENT_SHAPES = ["", '"a"', ' W/"a,b"\t']
BROKEN_ELEMENTS = ["x", '"', "W/", 'W/ "a"', '"a""b"', '"a" x', "*"]
SEPARATORS = [",", ", ", " ,", " , ", ",\t", "\t,\t"]


def _agree(text: str) -> bool:
    return (OLD.fullmatch(text) is None) == (NEW.fullmatch(text) is None)


def _structured_inputs() -> list[str]:
    """Every input of the structured sweep, one list-element count at a time."""
    inputs: list[str] = []
    for count in range(STRUCTURED_MAX_ELEMENTS + 1):
        inputs.extend(",".join(shapes) for shapes in itertools.product(ELEMENT_SHAPES, repeat=count))
        for fill in ('"a"', ""):
            for position in range(count):
                for broken in BROKEN_ELEMENTS:
                    elements = [fill] * count
                    elements[position] = broken
                    inputs.append(",".join(elements))
        inputs.extend(separator.join(['"a"'] * count) for separator in SEPARATORS)
    return inputs


def main() -> int:
    if NEW.pattern != EXPECTED_NEW:
        print(f"FAIL: the module's pattern is not the one this script verifies:\n  {NEW.pattern!r}")
        return 1
    checked = mismatches = accepted = 0
    examples: list[str] = []
    for length in range(EXHAUSTIVE_MAX_LEN + 1):
        for chars in itertools.product(ALPHABET, repeat=length):
            text = "".join(chars)
            checked += 1
            accepted += NEW.fullmatch(text) is not None
            if not _agree(text):
                mismatches += 1
                examples.append(text)
    exhaustive = checked
    rng = random.Random(20260923)
    for _ in range(RANDOM_CASES):
        text = "".join(rng.choice(TOKENS) for _ in range(rng.randint(0, RANDOM_MAX_TOKENS)))
        checked += 1
        accepted += NEW.fullmatch(text) is not None
        if not _agree(text):
            mismatches += 1
            examples.append(text)
    structured = structured_accepted = 0
    for text in _structured_inputs():
        checked += 1
        structured += 1
        hit = NEW.fullmatch(text) is not None
        accepted += hit
        structured_accepted += hit
        if not _agree(text):
            mismatches += 1
            examples.append(text)
    print(f"exhaustive: every string over {len(ALPHABET)} symbols up to length {EXHAUSTIVE_MAX_LEN} ({exhaustive:,} inputs)")
    print(f"random:     {RANDOM_CASES:,} token-built inputs of up to {RANDOM_MAX_TOKENS} tokens")
    print(f"structured: {structured:,} lists of 0 to {STRUCTURED_MAX_ELEMENTS} elements ({structured_accepted:,} accepted, {structured - structured_accepted:,} refused)")
    print(f"checked={checked:,} accepted-by-both={accepted:,} mismatches={mismatches}")
    for text in examples[:5]:
        print(f"  MISMATCH {text!r}")

    print("\ncost of a FAILING match (printed, not asserted):")
    for k in (14, 16, 18, 20):
        text = ", " * k + "x"
        started = time.perf_counter()
        OLD.fullmatch(text)
        print(f"  old  ', ' * {k} + 'x'  {1000 * (time.perf_counter() - started):10.1f} ms")
    for text, label in ((", " * 4000 + "x", "', ' * 4000 + 'x'"), (",   " * 2000 + "x", "',   ' * 2000 + 'x'"), (" " * 8192 + "x", "' ' * 8192 + 'x'")):
        started = time.perf_counter()
        NEW.fullmatch(text)
        print(f"  new  {label:24s} {1000 * (time.perf_counter() - started):10.3f} ms")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
