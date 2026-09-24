#!/usr/bin/env python3
"""Count what the conditional-request non-vacuity harness covers: which tests its arms name, and how.

Project:     Juniper
Sub-Project: juniper-data
Application: ad-hoc verification
Author:      Paul Calnon
Version:     1.0.0
License:     MIT
Created:     2026-09-24
Status:      ad-hoc -- one-off verification
Retire when: the harness it counts is retired
Related:     juniper-data#428; round-3 validation, lane A2, F6

WHY THIS EXISTS
---------------
``util/ad-hoc/2026-09-22_verify_conditional_request_tests_are_not_vacuous.py`` said every
behaviour the PR claimed was reverted, and round-3 validation counted seven of the test file's
56 tests that no arm named. Its docstring now states its coverage in numbers. This produces
those numbers from the harness's own ``MUTATIONS`` and pytest's own collection, so they can be
re-derived instead of trusted:

* every collected test of ``test_conditional_requests.py`` that no arm names;
* the tests an arm names only as a CONTROL (shown to stay green, never shown to fail);
* the count named as MUST-FAIL by some arm;
* any node id an arm names that pytest does not collect -- a typo there makes the arm vacuous.

Run from the repo root::

    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-24_count_conditional_request_harness_coverage.py

Exit 0 when every node id the arms name in that file is collected; 1 when one is not. The
counts are reported, never gated.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "util/ad-hoc/2026-09-22_verify_conditional_request_tests_are_not_vacuous.py"


class _Collector:
    """A pytest plugin that keeps the node ids of the collected items."""

    def __init__(self) -> None:
        self.node_ids: list[str] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.node_ids = [item.nodeid for item in session.items]


def main() -> int:
    spec = importlib.util.spec_from_file_location("harness_under_count", HARNESS)
    harness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = harness  # its @dataclass resolves the module through sys.modules
    spec.loader.exec_module(harness)

    collector = _Collector()
    pytest.main(["--collect-only", "-qq", "-p", "no:cacheprovider", "-o", "addopts=", str(REPO / harness.T)], plugins=[collector])
    collected = sorted(collector.node_ids)
    must_fail = {node for mutation in harness.MUTATIONS for node in mutation.must_fail}
    named = must_fail | {node for mutation in harness.MUTATIONS for node in mutation.must_still_pass}
    in_file = {node for node in named if node.startswith(harness.T)}
    missing = sorted(node for node in in_file if node not in set(collected))
    unnamed = [node for node in collected if node not in named]
    controls_only = [node for node in collected if node in named and node not in must_fail]

    print(f"arms: {len(harness.MUTATIONS)}; tests collected in {harness.T}: {len(collected)}")
    print(f"named by some arm: {len(set(collected) & named)}; as must-fail: {len(set(collected) & must_fail)}; only as controls: {len(controls_only)}; by no arm: {len(unnamed)}")
    for label, nodes in (("named by NO arm", unnamed), ("named only as a CONTROL", controls_only)):
        print(f"\n{label}:")
        for node in nodes:
            print(f"  {node.split('::', 1)[1]}")
    outside = sorted(node for node in named if not node.startswith(harness.T))
    print(f"\nnamed outside {harness.T}: {outside}")
    if missing:
        print("\nFAIL: an arm names a test pytest does not collect, so that arm is vacuous:")
        for node in missing:
            print(f"  {node}")
        return 1
    print("\nOK: every test an arm names in the file is collected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
