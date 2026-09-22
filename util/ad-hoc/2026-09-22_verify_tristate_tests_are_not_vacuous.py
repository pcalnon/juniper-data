#!/usr/bin/env python3
"""Negative control for APD-DATA-052: prove the new tri-state tests can FAIL.

Project:     Juniper
Sub-Project: juniper-data
Application: ad-hoc verification
Author:      Paul Calnon
Version:     1.0.0
License:     MIT

WHY THIS EXISTS
---------------
A test that passes against both the fixed and the broken implementation pins
nothing (memory ``reference_vacuous_pass_check_class``). The tri-state's two
halves fail in *opposite* directions, so one mutation cannot screen both:

* **M1 -- revert the three resolution sites to the old logical ``or``.**
  ``None or settings.X`` still equals ``settings.X``, so the "omitted defers"
  tests stay green; only the "explicit ``false`` refuses" tests can see it.
  This is the mutation that restores the behaviour the owner overruled.

* **M2 -- revert the two schemas to a plain ``bool`` defaulting to ``False``.**
  The sites keep their ``is None`` check, which a ``False`` default never
  satisfies, so every silent caller becomes a refusal. Only the "omitted
  defers" tests can see it.

A mutation that leaves every test green is reported as a FAILURE of this
script: it means the site is unpinned.

Run from the repo root::

    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-22_verify_tristate_tests_are_not_vacuous.py

Exit 0 when every mutation is caught by the expected tests and the unmutated
baseline is green; exit 1 otherwise.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = "/opt/miniforge3/envs/JuniperData/bin/python"

CSV_GEN = REPO / "juniper_data/generators/csv_import/generator.py"
EQ_GEN = REPO / "juniper_data/generators/equities/generator.py"
CSV_PARAMS = REPO / "juniper_data/generators/csv_import/params.py"
EQ_PARAMS = REPO / "juniper_data/generators/equities/params.py"

# The tests that must go RED under each mutation, as pytest node ids.
OPT_OUT_TESTS = [
    "juniper_data/tests/unit/test_csv_import_generator.py::TestInputByteCap::test_request_can_opt_out_of_deployment_allow_truncation",
    "juniper_data/tests/unit/test_equities_generator.py::TestUniverseSymbolCap::test_explicit_false_refuses_against_a_deployment_opt_in",
    "juniper_data/tests/unit/test_equities_seq_deployment_policy.py::TestIncompleteDataPolicy::test_explicit_false_refuses_against_a_deployment_opt_in",
]
DEFERS_TESTS = [
    "juniper_data/tests/unit/test_csv_import_generator.py::TestInputByteCap::test_omitted_allow_truncation_still_defers_to_the_deployment",
    "juniper_data/tests/unit/test_equities_generator.py::TestUniverseSymbolCap::test_omitted_allow_truncation_still_defers_to_the_deployment",
    "juniper_data/tests/unit/test_equities_seq_deployment_policy.py::TestIncompleteDataPolicy::test_omitted_gate_still_defers_to_a_deployment_opt_in",
]


@dataclass
class Mutation:
    """One revert of the shipped change, plus the tests that must catch it."""

    name: str
    why: str
    # (path, exact text to find, replacement) -- every edit must match exactly once.
    edits: list[tuple[Path, str, str]]
    must_fail: list[str]
    must_still_pass: list[str] = field(default_factory=list)


MUTATIONS = [
    Mutation(
        name="M1: three resolution sites back to the logical OR",
        why="restores the behaviour the owner overruled -- an explicit false cannot refuse",
        edits=[
            (
                CSV_GEN,
                "        allow = settings.csv_import_allow_truncation if params.allow_truncation is None else params.allow_truncation",
                "        allow = bool(params.allow_truncation or settings.csv_import_allow_truncation)",
            ),
            (
                EQ_GEN,
                "        allow = settings.equities_allow_truncation if params.allow_truncation is None else params.allow_truncation",
                "        allow = bool(params.allow_truncation or settings.equities_allow_truncation)",
            ),
            (
                EQ_GEN,
                "        allowed = settings.equities_allow_truncation if params.allow_truncation is None else params.allow_truncation",
                "        allowed = bool(params.allow_truncation or settings.equities_allow_truncation)",
            ),
        ],
        must_fail=OPT_OUT_TESTS,
        must_still_pass=DEFERS_TESTS,
    ),
    Mutation(
        name="M2: both schemas back to a plain bool defaulting to False",
        why="turns every silent caller into a refusal -- the half an OR-only mutation cannot reach",
        edits=[
            (CSV_PARAMS, "    allow_truncation: bool | None = Field(\n        default=None,", "    allow_truncation: bool = Field(\n        default=False,"),
            (EQ_PARAMS, "    allow_truncation: bool | None = Field(\n        default=None,", "    allow_truncation: bool = Field(\n        default=False,"),
        ],
        must_fail=DEFERS_TESTS,
        must_still_pass=OPT_OUT_TESTS,
    ),
]


def run_tests(node_ids: list[str]) -> dict[str, bool]:
    """Return {node_id: passed}. Each id runs alone so one red does not mask another.

    ``PYTHONDONTWRITEBYTECODE`` because this script rewrites source files in
    place, sometimes twice within one filesystem timestamp tick: a cached
    ``.pyc`` whose mtime still matches would serve the PREVIOUS variant and the
    mutation would score as "not caught" for a reason that has nothing to do
    with the test (memory ``reference_mutation_check_stale_pyc_and_piped_exit``).
    """
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    results: dict[str, bool] = {}
    for node_id in node_ids:
        proc = subprocess.run(
            [PYTHON, "-m", "pytest", node_id, "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=REPO,
            capture_output=True,
            text=True,
            env=env,
        )
        results[node_id] = proc.returncode == 0
    return results


def apply(edits: list[tuple[Path, str, str]]) -> dict[Path, str]:
    """Apply every edit, returning the original text of each touched file."""
    originals: dict[Path, str] = {}
    for path, find, replace in edits:
        if path not in originals:
            originals[path] = path.read_text(encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        count = text.count(find)
        if count != 1:
            for restore_path, restore_text in originals.items():
                restore_path.write_text(restore_text, encoding="utf-8")
            raise SystemExit(f"FATAL: pattern matched {count} times (expected 1) in {path.relative_to(REPO)}:\n  {find.strip()}")
        path.write_text(text.replace(find, replace), encoding="utf-8")
    return originals


def restore(originals: dict[Path, str]) -> bool:
    """Put every touched file back, then PROVE it. Returns True when the tree is clean.

    Verifying the restore is not paranoia. This script rewrites tracked source in
    place, so a restore that silently half-applies leaves mutated code in the
    working tree, and the next commit ships it. The sibling instrument
    ``2026-09-04_apd_data_018_mutation_check.py`` re-runs its baseline for exactly
    this reason; this one did not until adversarial validation pointed it out.
    """
    ok = True
    for path, text in originals.items():
        path.write_text(text, encoding="utf-8")
    for path, text in originals.items():
        if path.read_text(encoding="utf-8") != text:
            print(f"  RESTORE FAILED: {path} does not match its pre-mutation content")
            ok = False
    return ok


def main() -> int:
    failures: list[str] = []

    print("baseline (unmutated tree) -- every tri-state test must PASS")
    baseline = run_tests(OPT_OUT_TESTS + DEFERS_TESTS)
    for node_id, passed in baseline.items():
        print(f"  {'OK  ' if passed else 'RED '} {node_id.split('::')[-1]}")
        if not passed:
            failures.append(f"baseline: {node_id} is RED before any mutation")

    for mutation in MUTATIONS:
        print(f"\n{mutation.name}")
        print(f"  ({mutation.why})")
        # apply() INSIDE the try: it edits file-by-file, so a failure partway
        # through leaves earlier files mutated. Its own rollback covers the
        # pattern-mismatch path, but not an IOError on the third of three writes.
        originals: dict[Path, str] = {}
        try:
            originals = apply(mutation.edits)
            caught = run_tests(mutation.must_fail)
            survived = run_tests(mutation.must_still_pass)
        finally:
            if not restore(originals):
                failures.append(f"{mutation.name}: RESTORE FAILED -- the working tree still holds mutated source")

        for node_id, passed in caught.items():
            verdict = "CAUGHT " if not passed else "VACUOUS"
            print(f"  {verdict} {node_id.split('::')[-1]}")
            if passed:
                failures.append(f"{mutation.name}: {node_id} stayed GREEN -- it pins nothing")
        for node_id, passed in survived.items():
            verdict = "OK     " if passed else "OVERBRD"
            print(f"  {verdict} {node_id.split('::')[-1]} (expected unaffected)")
            if not passed:
                failures.append(f"{mutation.name}: {node_id} also went red -- the mutation is not isolating the half it claims")

    print()
    if failures:
        print("FAIL")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("PASS: every mutation is caught by exactly the tests that should see it, and by no others.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
