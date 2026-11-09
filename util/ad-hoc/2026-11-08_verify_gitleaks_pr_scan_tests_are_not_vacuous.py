#!/usr/bin/env python3
"""Negative control for the pull-request Gitleaks scan: prove the new test_gitleaks_config.py cases can FAIL.

Project: juniper-data
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-11-08
Status: ad-hoc — investigation (negative control for the juniper-data defect in §4 of juniper-ml's notes/JUNIPER_2026-11-06_JUNIPER-ECOSYSTEM_CURSOR-FLOOD-4-DISPOSITION.md)
Retire when: the pull-request scan steps in ci.yml are replaced or removed.
Related: .github/workflows/ci.yml ("Install Gitleaks (pull request)", "Run Gitleaks over the pull request"),
         juniper_data/tests/unit/test_gitleaks_config.py, util/ad-hoc/2026-11-08_reproduce_gitleaks_pr_commit_window.py

WHY THIS EXISTS
---------------
A test that passes against both the fixed and the broken workflow pins nothing. Each mutation below puts back one
property of the defect, or removes one guard of the fix, by an exact-match edit of ci.yml, runs the tests that must
go red one at a time, and the tests that must stay green, then restores ci.yml and proves the restore byte for byte:

    page cap            the range ends at the 30th commit, as the action's headRef does
    action's flags      --no-merges --first-parent back on the log options
    first-parent        --first-parent alone back: the walk skips what a merge brought in
    no remerge-diff     --remerge-diff dropped: a merge commit's own changes go unscanned again
    first-parent diff   a merge's first-parent patch in place of --remerge-diff: what it merged is scanned again
    off-by-one          base^..head, which reaches the base commit
    leak passes         a leak (gitleaks exit 2) passes the step
    no SHA checks       the SHA checks removed: an unresolvable range reaches the engine
    action on PRs       the action scans pull requests again
    no digest check     the tarball is unpacked without the digest check
    version drift       the pull-request install drifts to another engine version
    octopus passes      the octopus-merge refusal no longer stops the step: such a merge's own changes pass unscanned
    no git filter       the engine reads git's `remerge ...` notes again: a merge's finding can lose its commit
    filter status       the filter exits with sed's status, not git's: gitleaks misreads a repository with no remote

The real-engine cases need a gitleaks 8.24.3 binary: set GITLEAKS_BIN. GITLEAKS_PARITY_REQUIRED=1 is forced, so a
missing binary turns the baseline red instead of skipping (a skip would score as a pass).

Run from the repo root::

    GITLEAKS_BIN=<gitleaks 8.24.3> /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-11-08_verify_gitleaks_pr_scan_tests_are_not_vacuous.py --basetemp <dir>

Exit 0 when the baseline is green, every mutation is caught by every test listed for it, no listed survivor goes
red, and every restore is exact; exit 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import subprocess  # nosec B404 - runs pytest on fixed node ids
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = "/opt/miniforge3/envs/JuniperData/bin/python"
CI = REPO / ".github" / "workflows" / "ci.yml"
T = "juniper_data/tests/unit/test_gitleaks_config.py::"

PAST_PAGE = T + "test_real_engine_scan_step_finds_a_token_past_the_actions_first_page"
MERGE = T + "test_real_engine_scan_step_finds_a_token_a_merge_brought_in"
MERGE_OWN = [T + f"test_real_engine_scan_step_finds_a_token_only_a_merge_commit_holds[{w}]" for w in ("added-during-the-merge", "conflict-resolution")]
CLEAN = T + "test_real_engine_scan_step_passes_a_clean_range"
CLEAN_MERGE = T + "test_real_engine_scan_step_passes_a_clean_merge_of_the_base_branch"
ARGV = T + "test_the_scan_step_hands_the_engine_the_whole_range"
EVENT_RANGE = T + "test_the_pull_request_scan_reads_its_range_from_the_event"
LEAKS = T + "test_the_scan_step_fails_the_way_the_action_does[leaks]"
ENGINE_ERROR = T + "test_the_scan_step_fails_the_way_the_action_does[engine-error]"
REFUSE = [T + f"test_the_scan_step_refuses_a_range_it_cannot_resolve[{w}]" for w in ("base-empty", "base-not-hex", "head-not-in-checkout")]
ACTION_SKIPS_PR = T + "test_the_action_no_longer_scans_a_pull_request"
DIGEST = T + "test_a_pull_request_installs_the_same_engine_checked_by_digest"
WARN = T + "test_the_scan_step_warns_when_the_range_and_the_pull_request_disagree"
OCTOPUS = T + "test_the_scan_step_refuses_an_octopus_merge"
NOTES = T + "test_the_scan_step_gives_the_engine_a_git_without_remerge_notes"
MERGE_KEEPS = [T + f"test_real_engine_scan_step_keeps_a_merge_findings_commit_and_path[{w}]" for w in ("conflicted-path", "parent-subject")]

RANGE = '          range="${BASE_SHA}..${HEAD_SHA}"'
LOG_OPTS = '"--log-opts=--remerge-diff ${range}"'


@dataclass
class Mutation:
    """One revert, plus the tests that must catch it and the tests it must leave green."""

    name: str
    edits: list[tuple[str, str]]  # (exact text in ci.yml, replacement); each must match exactly once
    must_fail: list[str]
    must_still_pass: list[str] = field(default_factory=list)


MUTATIONS = [
    Mutation("page cap: the range ends at the 30th commit", [(RANGE, '          range="${BASE_SHA}..$(git rev-list --reverse "${BASE_SHA}..${HEAD_SHA}" | head -n 30 | tail -n 1)"')], [PAST_PAGE, CLEAN], [MERGE, ARGV, *MERGE_OWN, CLEAN_MERGE]),
    Mutation("action's flags: --no-merges --first-parent back on the log options", [(LOG_OPTS, '"--log-opts=--no-merges --first-parent --remerge-diff ${range}"')], [EVENT_RANGE, MERGE, ARGV, *MERGE_OWN], [PAST_PAGE, CLEAN, CLEAN_MERGE]),
    Mutation("first-parent: --first-parent alone back on the log options", [(LOG_OPTS, '"--log-opts=--first-parent --remerge-diff ${range}"')], [EVENT_RANGE, MERGE, ARGV], [PAST_PAGE, CLEAN, CLEAN_MERGE, *MERGE_OWN]),
    Mutation("no remerge-diff: a merge commit's own changes go unscanned", [(LOG_OPTS, '"--log-opts=${range}"')], [ARGV, *MERGE_OWN], [EVENT_RANGE, MERGE, PAST_PAGE, CLEAN, CLEAN_MERGE]),
    Mutation("first-parent diff: a merge's first-parent patch in place of --remerge-diff", [(LOG_OPTS, '"--log-opts=--diff-merges=first-parent ${range}"')], [ARGV, CLEAN_MERGE, MERGE], [PAST_PAGE, CLEAN, *MERGE_OWN]),
    Mutation("off-by-one: base^..head", [(RANGE, '          range="${BASE_SHA}^..${HEAD_SHA}"')], [PAST_PAGE, CLEAN, ARGV], []),
    Mutation("leak passes: a leak passes the step", [('reports/security/gitleaks.sarif"\n              exit 1\n', 'reports/security/gitleaks.sarif"\n              exit 0\n')], [LEAKS, PAST_PAGE, MERGE], [ENGINE_ERROR, CLEAN]),
    Mutation("no SHA checks: the SHA checks removed", [('            if ! [[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then\n', "            if false; then\n")], REFUSE, [ARGV, WARN]),
    Mutation("action on PRs: the action scans pull requests again", [("        if: github.event_name != 'repository_dispatch' && github.event_name != 'pull_request'\n", "        if: github.event_name != 'repository_dispatch'\n")], [ACTION_SKIPS_PR], [EVENT_RANGE]),
    Mutation("no digest check: the tarball is unpacked without it", [('          echo "${GITLEAKS_SHA256}  ${RUNNER_TEMP}/${asset}" | sha256sum --check --strict\n', "          true\n")], [DIGEST], [EVENT_RANGE]),
    Mutation("version drift: the pull-request install moves to another engine version", [('          GITLEAKS_VERSION: "8.24.3"\n          GITLEAKS_SHA256:', '          GITLEAKS_VERSION: "8.24.2"\n          GITLEAKS_SHA256:')], [DIGEST], [EVENT_RANGE]),
    Mutation("octopus passes: the octopus-merge refusal no longer stops the step", [('replace it with two-parent merges"\n            exit 1\n', 'replace it with two-parent merges"\n')], [OCTOPUS], [ARGV, MERGE, *MERGE_OWN, NOTES]),
    Mutation("no git filter: the engine reads git's remerge notes again", [('REAL_GIT="${real_git}" PATH="${filter}:${PATH}" gitleaks detect', "gitleaks detect")], [NOTES, *MERGE_KEEPS], [ARGV, OCTOPUS, *MERGE_OWN, CLEAN_MERGE]),
    Mutation("filter status: the filter exits with sed's status, not git's", [("          #!/usr/bin/env bash\n          set -o pipefail\n", "          #!/usr/bin/env bash\n")], [NOTES], [ARGV, OCTOPUS, *MERGE_KEEPS]),
]


def run_tests(node_ids: list[str], basetemp: str | None) -> dict[str, bool]:
    """Return {node_id: passed}; each id runs alone so one red cannot mask another."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "GITLEAKS_PARITY_REQUIRED": "1"}
    env.pop("GITLEAKS_CONFIG", None)
    env.pop("GITLEAKS_CONFIG_TOML", None)
    results: dict[str, bool] = {}
    for node_id in node_ids:
        argv = [PYTHON, "-m", "pytest", node_id, "-q", "--no-header", "-p", "no:cacheprovider"]
        if basetemp:
            argv.append(f"--basetemp={basetemp}")
        proc = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, env=env, check=False)  # nosec B603
        results[node_id] = proc.returncode == 0
    return results


def apply(original: str, edits: list[tuple[str, str]]) -> str:
    text = original
    for find, replace in edits:
        count = text.count(find)
        if count != 1:
            raise SystemExit(f"FATAL: a mutation pattern matched {count} times (expected 1) in ci.yml:\n  {find.strip()}")
        text = text.replace(find, replace)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basetemp", help="pytest --basetemp for every run (default: pytest's own, under TMPDIR)")
    args = ap.parse_args()
    if not os.environ.get("GITLEAKS_BIN"):
        print("GITLEAKS_BIN is unset: the real-engine cases cannot run, so the baseline would be red")
        return 1
    original = CI.read_text(encoding="utf-8")
    failures: list[str] = []
    every = sorted({n for m in MUTATIONS for n in m.must_fail + m.must_still_pass})

    print("baseline (unmutated ci.yml): every listed test must PASS")
    for node_id, passed in run_tests(every, args.basetemp).items():
        print(f"  {'OK  ' if passed else 'RED '} {node_id.removeprefix(T)}")
        if not passed:
            failures.append(f"baseline: {node_id} is RED before any mutation")

    for mutation in MUTATIONS:
        print(f"\n{mutation.name}")
        mutated = apply(original, mutation.edits)
        try:
            CI.write_text(mutated, encoding="utf-8")
            caught = run_tests(mutation.must_fail, args.basetemp)
            survived = run_tests(mutation.must_still_pass, args.basetemp)
        finally:
            CI.write_text(original, encoding="utf-8")
            if CI.read_text(encoding="utf-8") != original:
                failures.append(f"{mutation.name}: RESTORE FAILED -- ci.yml still holds the mutation")
        for node_id, passed in caught.items():
            print(f"  {'VACUOUS' if passed else 'CAUGHT '} {node_id.removeprefix(T)}")
            if passed:
                failures.append(f"{mutation.name}: {node_id} stayed GREEN -- it pins nothing here")
        for node_id, passed in survived.items():
            print(f"  {'OK     ' if passed else 'OVERBRD'} {node_id.removeprefix(T)} (expected unaffected)")
            if not passed:
                failures.append(f"{mutation.name}: {node_id} went red too -- the mutation is not the property it claims")

    print()
    if failures:
        print("FAIL")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("PASS: every mutation is caught by every test listed for it, no listed survivor went red, and ci.yml was restored exactly each time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
