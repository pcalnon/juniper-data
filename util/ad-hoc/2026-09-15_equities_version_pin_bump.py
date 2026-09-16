#!/usr/bin/env python3
"""Bump the equities pair's generator VERSION pins from 4.0.0 to 5.0.0 and record why.

Project: juniper-ml
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-15
Status: ad-hoc -- one-off (applied to the juniper-data head-typo worktree)
Retire when: RETAINED -- ad-hoc scripts are kept as provenance of record (owner policy 2026-08-25)
Related: juniper-data#395 (which shipped the regression); register row APD-DATA-047

``generator_version`` is hashed into ``dataset_id``, so a value-changing fix that does not bump it
serves the OLD numbers for the same request. 4.0.0 delivered AIZ 990x and EOG 428x too large; the
corrected values must not resolve to the ID that carried the wrong ones.
"""

from __future__ import annotations

from pathlib import Path

WORK = Path("/home/pcalnon/Development/python/Juniper/worktrees/juniper-data--fix--equities-head-typo-regression--20260915-2130--f3797634")

target = WORK / "juniper_data/tests/unit/test_equities_generator.py"
text = target.read_text()
OLD = (
    "        # 4.0.0 since the 2026-09-09 owner rulings. Two of them are breaking on their own --\n"
    "        # the default matrix lost a column (``adj_close``, APD-DATA-041) and ``cost_basis`` is\n"
    "        # absent before the purchase date instead of constant (APD-DATA-042) -- and three more\n"
    "        # change values without changing shape: the as-of publication history, the absolute\n"
    "        # floor, and the causal median. Any one of those is a reason the same params must not\n"
    "        # resolve to the same dataset ID as before.\n"
    '        assert VERSION == "4.0.0"\n'
)
NEW = (
    "        # 4.0.0 at the 2026-09-09 owner rulings. Two of them are breaking on their own --\n"
    "        # the default matrix lost a column (``adj_close``, APD-DATA-041) and ``cost_basis`` is\n"
    "        # absent before the purchase date instead of constant (APD-DATA-042) -- and three more\n"
    "        # change values without changing shape: the as-of publication history, the absolute\n"
    "        # floor, and the causal median. Any one of those is a reason the same params must not\n"
    "        # resolve to the same dataset ID as before.\n"
    "        #\n"
    "        # 5.0.0 on 2026-09-15, because 4.0.0 shipped a REGRESSION and the corrected values\n"
    "        # must not be served under the ID that addressed the wrong ones: the causal median\n"
    "        # included the point it was judging, so a scale typo in a series' opening filings\n"
    "        # survived and was delivered (AIZ 990x, EOG 428x too large). Every artifact minted\n"
    "        # at 4.0.0 for a symbol with such a typo carries it.\n"
    '        assert VERSION == "5.0.0"\n'
)
assert OLD in text, "version-pin comment anchor not found"
target.write_text(text.replace(OLD, NEW, 1))
print("ok  equities VERSION pin -> 5.0.0")

target = WORK / "juniper_data/tests/unit/test_val_emission_guards.py"
text = target.read_text()
assert '{"equities": "4.0.0", "equities_seq": "4.0.0"}' in text, "fleet-guard expectation not found"
text = text.replace('{"equities": "4.0.0", "equities_seq": "4.0.0"}', '{"equities": "5.0.0", "equities_seq": "5.0.0"}', 1)
text = text.replace(
    "only the equities pair is deliberately past 3.0.0 (owner rulings 2026-09-09)",
    "only the equities pair is deliberately past 3.0.0 (owner rulings 2026-09-09; 5.0.0 since the 2026-09-15 head-typo regression fix)",
    1,
)
target.write_text(text)
print("ok  fleet-wide version guard -> 5.0.0")
