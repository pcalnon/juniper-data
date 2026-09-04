#!/usr/bin/env python3
"""Mutation matrix for the APD-DATA-018 csv_import byte cap.

Project:     Juniper
Sub-Project: juniper-data
Application: APD-DATA-018 verification
Author:      Paul Calnon
Version:     0.1.0
License:     MIT
Status:      ad-hoc (evidence for the fix PR)

A green suite proves the code passes its tests; it does not prove the tests
would notice the defect coming back. Each mutation below reintroduces one
specific failure the fix exists to prevent, and asserts the suite goes red --
naming which arms caught it.

**One row expects SURVIVAL.** Renaming a private helper must break nothing: if
it does, a test is pinned to an implementation detail rather than to behaviour.
A matrix with no expected-survival row cannot distinguish "the tests are
sensitive" from "the tests are brittle" (round-27 lesson, defect register).

Two traps this runner is built around:

* **Stale bytecode.** A restore that lands in the same second as the mutation
  can leave a VALID cached ``.pyc``, so the "restored" run silently executes
  mutated code. ``PYTHONDONTWRITEBYTECODE=1`` removes the cache from the
  picture entirely rather than trying to invalidate it.
* **A collection ERROR carries no ``FAILED`` line.** Matching only ``^FAILED``
  reports a mutation that broke *import* as a clean SURVIVAL -- a vacuous pass
  inside the very tool meant to detect them. This matches ``ERROR`` too and
  falls back to the process exit status.

Run:
    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-04_apd_data_018_mutation_check.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "juniper_data/generators/csv_import/generator.py"
META = REPO / "juniper_data/core/meta.py"
TEST_TARGET = "juniper_data/tests/unit/test_csv_import_generator.py"
# ``-p juniper_data.api.app`` is not a plugin, it is an import-order fix.
#
# juniper-data carries a PRE-EXISTING circular import (confirmed on main, not
# introduced by this change): importing ``juniper_data.generators.csv_import``
# first runs its ``__init__`` -> ``generator`` -> ``juniper_data.api.settings``
# -> ``api/__init__`` -> ``app`` -> ``routes.generators`` -> back into the
# half-built csv_import package, which raises on ``VERSION``. So this test file
# cannot be run ALONE; it only passes inside a suite where something imported
# the api package first. pytest imports ``-p`` modules before collection, which
# makes that ordering explicit instead of accidental -- and keeps each mutation
# run at ~0.3 s instead of the ~48 s a whole-directory run costs.
PYTEST = [sys.executable, "-m", "pytest", TEST_TARGET, "-p", "juniper_data.api.app"]


@dataclass
class Mutation:
    name: str
    path: Path
    old: str
    new: str
    expect_fail: bool = True
    occurrences: int = 0
    why: str = ""
    caught_by: list[str] = field(default_factory=list)


MUTATIONS = [
    Mutation(
        name="M1-refusal-removed",
        path=GENERATOR,
        old="        if not allow_truncation:\n            raise InputTooLargeError(source=params.file_path, bytes_total=bytes_total, cap_bytes=cap_bytes)",
        new="        if False:\n            raise InputTooLargeError(source=params.file_path, bytes_total=bytes_total, cap_bytes=cap_bytes)",
        why="Silently truncates without an opt-in -- the exact behaviour the owner's decision forbids.",
    ),
    Mutation(
        name="M2-record-boundary-trim-removed",
        path=GENERATOR,
        old="        if drop_trailing_partial and data and (any(value is None for value in data[-1].values()) or (source_text is not None and CsvImportGenerator._has_unclosed_quote(source_text))):\n            data.pop()",
        new="        if False and data and (any(value is None for value in data[-1].values()) or (source_text is not None and CsvImportGenerator._has_unclosed_quote(source_text))):\n            data.pop()",
        why="Lets a half-row through when the cap lands inside a quoted field.",
    ),
    Mutation(
        name="M3-deployment-opt-in-ignored",
        path=GENERATOR,
        old="        allow = bool(params.allow_truncation or settings.csv_import_allow_truncation)",
        new="        allow = bool(params.allow_truncation)",
        why="Drops the env-var / .env opt-in surface the owner required for CLI callers.",
    ),
    Mutation(
        name="M4-pop-returns-empty-dict",
        path=META,
        old="    return arrays.pop(TRUNCATION_META_KEY, None) or None",
        new="    return arrays.pop(TRUNCATION_META_KEY, None) or {}",
        why="Makes 'complete' and 'reported nothing' indistinguishable on DatasetMeta.truncation.",
    ),
    Mutation(
        name="M5-corrupt-source-tolerated",
        path=GENERATOR,
        old="                if tolerate_truncated and index == len(lines) - 1:\n                    break\n                raise",
        new="                if tolerate_truncated:\n                    break\n                raise",
        why="A corrupt line mid-file would import as a short dataset -- silent partial data, reintroduced via error handling.",
    ),
    Mutation(
        name="M6-helper-renamed (EXPECTED SURVIVAL)",
        path=GENERATOR,
        old="_read_capped_text",
        new="_read_bounded_text",
        expect_fail=False,
        occurrences=2,
        why="A private rename must break nothing. If it fails, a test is pinned to a name instead of a behaviour.",
    ),
]


def run_suite() -> tuple[bool, list[str]]:
    """Return (went_red, failing test names)."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(PYTEST, cwd=REPO, capture_output=True, text=True, env=env, check=False)
    out = proc.stdout + proc.stderr
    names = [line.split("::")[-1].split()[0] for line in out.splitlines() if line.startswith("FAILED") or line.startswith("ERROR")]
    went_red = bool(names) or proc.returncode != 0
    return went_red, names


def main() -> int:
    baseline_red, _ = run_suite()
    if baseline_red:
        print("BASELINE IS RED -- fix that before trusting any mutation result.")
        return 1
    print("baseline: GREEN\n")

    verdicts: list[tuple[str, bool]] = []
    for mut in MUTATIONS:
        original = mut.path.read_text(encoding="utf-8")
        occurrences = original.count(mut.old)
        # A rename mutation is expected to hit every occurrence (definition plus
        # call sites); a behavioural mutation must hit exactly one, or it is not
        # the surgical change it claims to be.
        expected_occurrences = mut.occurrences if mut.occurrences else 1
        if occurrences != expected_occurrences:
            print(f"{mut.name}: SKIPPED -- anchor matched {occurrences} times, expected {expected_occurrences}")
            verdicts.append((mut.name, False))
            continue
        try:
            mut.path.write_text(original.replace(mut.old, mut.new), encoding="utf-8")
            went_red, names = run_suite()
        finally:
            mut.path.write_text(original, encoding="utf-8")

        ok = went_red == mut.expect_fail
        expected = "RED" if mut.expect_fail else "GREEN"
        actual = "RED" if went_red else "GREEN"
        print(f"{'PASS' if ok else 'PROBLEM'}  {mut.name}: expected {expected}, got {actual}")
        if names:
            print(f"         caught by: {', '.join(sorted(set(names))[:6])}")
        print(f"         {mut.why}")
        verdicts.append((mut.name, ok))

    post_red, _ = run_suite()
    print(f"\nrestored baseline: {'RED -- RESTORE FAILED' if post_red else 'GREEN'}")

    failures = [name for name, ok in verdicts if not ok]
    print(f"\n{len(verdicts) - len(failures)}/{len(verdicts)} mutations behaved as specified")
    return 1 if failures or post_red else 0


if __name__ == "__main__":
    raise SystemExit(main())
