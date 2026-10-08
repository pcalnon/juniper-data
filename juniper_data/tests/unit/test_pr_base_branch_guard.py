"""Execute the PR base-branch guard's own shell.

``.github/workflows/pr-base-branch-guard.yml`` is the only check that runs on a
pull request whose base is a feature branch: rulesets are scoped to the default
branch, so a stacked PR otherwise merges with no required checks. The shell has
three fail-safe edges that a reimplementation can get backwards:

- a ``merge_group`` event has no PR base and must exit 0, or every queued merge fails
- a missing default branch fails open, or one empty payload fails every PR
- the ``stacked-pr`` hatch is the exact string ``true`` GitHub emits for
  ``contains(...)``; ``True``, a trailing space, or any other value still fails

A base that merely starts with the default branch (``main2``, ``Main``) is not a match.
The job name is the required-status context and must stay ``Guard PR base branch``.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs the workflow's own extracted shell with fixed argv
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "pr-base-branch-guard.yml"
REQUIRED_TYPES = ["opened", "reopened", "edited", "synchronize", "labeled", "unlabeled"]


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _run_script() -> str:
    steps = _workflow()["jobs"]["guard-base-branch"]["steps"]
    assert len(steps) == 1
    assert "run" in steps[0]
    return str(steps[0]["run"])


def _guard(tmp_path: Path, **env_vars: str | None) -> subprocess.CompletedProcess[str]:
    """Run the guard. A value of None leaves that variable unset."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - bash fallback only; no secrets
        "LANG": "C",
    }
    for key, value in env_vars.items():
        if value is not None:
            env[key] = value
    return subprocess.run(  # nosec B603 B607 - fixed bash argv; script is the workflow's own step
        ["bash", "-c", _run_script()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class TestGuardDecisions:
    def test_merge_group_exits_before_the_base_comparison(self, tmp_path: Path) -> None:
        proc = _guard(
            tmp_path,
            EVENT_NAME="merge_group",
            BASE_REF="",
            DEFAULT_BRANCH="main",
            HAS_BYPASS="false",
        )
        assert proc.returncode == 0, proc.stdout
        assert "merge_group event" in proc.stdout
        assert "::error" not in proc.stdout

    def test_merge_group_ignores_a_feature_base(self, tmp_path: Path) -> None:
        proc = _guard(
            tmp_path,
            EVENT_NAME="merge_group",
            BASE_REF="feature/stacked",
            DEFAULT_BRANCH="main",
            HAS_BYPASS="false",
        )
        assert proc.returncode == 0, proc.stdout
        assert "not the default branch" not in proc.stdout

    @pytest.mark.parametrize("default_branch", ["", None])
    def test_a_missing_default_branch_fails_open(self, tmp_path: Path, default_branch: str | None) -> None:
        proc = _guard(
            tmp_path,
            EVENT_NAME="pull_request",
            BASE_REF="feature/stacked",
            DEFAULT_BRANCH=default_branch,
            HAS_BYPASS="false",
        )
        assert proc.returncode == 0, proc.stdout
        assert "Could not resolve the repository default branch" in proc.stdout
        assert "::error" not in proc.stdout

    def test_the_default_branch_passes(self, tmp_path: Path) -> None:
        proc = _guard(
            tmp_path,
            EVENT_NAME="pull_request",
            BASE_REF="main",
            DEFAULT_BRANCH="main",
            HAS_BYPASS="false",
        )
        assert proc.returncode == 0, proc.stdout
        assert "targets the default branch (main)" in proc.stdout

    def test_a_feature_base_fails_and_names_both_branches(self, tmp_path: Path) -> None:
        proc = _guard(
            tmp_path,
            EVENT_NAME="pull_request",
            BASE_REF="feature/stacked",
            DEFAULT_BRANCH="main",
            HAS_BYPASS="false",
        )
        assert proc.returncode == 1
        assert "PR base is 'feature/stacked', not the default branch 'main'" in proc.stdout
        assert "::error title=Base-branch guard::" in proc.stdout

    def test_an_empty_base_is_not_the_default_branch(self, tmp_path: Path) -> None:
        proc = _guard(
            tmp_path,
            EVENT_NAME="pull_request",
            BASE_REF="",
            DEFAULT_BRANCH="main",
            HAS_BYPASS="false",
        )
        assert proc.returncode == 1
        assert "PR base is '', not the default branch 'main'" in proc.stdout

    @pytest.mark.parametrize("base_ref", ["main2", "Main", "refs/heads/main", "main "])
    def test_a_lookalike_base_does_not_match(self, tmp_path: Path, base_ref: str) -> None:
        proc = _guard(
            tmp_path,
            EVENT_NAME="pull_request",
            BASE_REF=base_ref,
            DEFAULT_BRANCH="main",
            HAS_BYPASS="false",
        )
        assert proc.returncode == 1
        assert f"PR base is '{base_ref}', not the default branch 'main'" in proc.stdout

    def test_the_stacked_pr_label_is_the_exact_string_true(self, tmp_path: Path) -> None:
        proc = _guard(
            tmp_path,
            EVENT_NAME="pull_request",
            BASE_REF="feature/stacked",
            DEFAULT_BRANCH="main",
            HAS_BYPASS="true",
        )
        assert proc.returncode == 0, proc.stdout
        assert "Allowed via the 'stacked-pr' label" in proc.stdout
        assert "::error" not in proc.stdout

    @pytest.mark.parametrize("bypass", ["True", "TRUE", "true ", " yes", "1", "false", None])
    def test_anything_other_than_true_does_not_bypass(self, tmp_path: Path, bypass: str | None) -> None:
        proc = _guard(
            tmp_path,
            EVENT_NAME="pull_request",
            BASE_REF="feature/stacked",
            DEFAULT_BRANCH="main",
            HAS_BYPASS=bypass,
        )
        assert proc.returncode == 1
        assert "::error title=Base-branch guard::" in proc.stdout

    def test_a_matching_base_passes_before_the_label_is_consulted(self, tmp_path: Path) -> None:
        proc = _guard(
            tmp_path,
            EVENT_NAME="pull_request",
            BASE_REF="main",
            DEFAULT_BRANCH="main",
            HAS_BYPASS="not-a-bool",
        )
        assert proc.returncode == 0, proc.stdout
        assert "targets the default branch (main)" in proc.stdout


class TestGuardWorkflowWiring:
    def test_labeled_and_unlabeled_retrigger_the_hatch(self) -> None:
        triggers = _workflow()["on"]
        assert triggers["pull_request"]["types"] == REQUIRED_TYPES
        assert "merge_group" in triggers

    def test_the_job_name_is_the_required_context(self) -> None:
        job = _workflow()["jobs"]["guard-base-branch"]
        assert job["name"] == "Guard PR base branch"

    def test_there_is_no_concurrency_group(self) -> None:
        """cancel-in-progress would post a non-success conclusion on a required context."""
        assert "concurrency" not in _workflow()

    def test_permissions_do_not_grant_a_write(self) -> None:
        assert _workflow()["permissions"] == {"contents": "read"}

    def test_the_hatch_looks_for_the_stacked_pr_label(self) -> None:
        step = _workflow()["jobs"]["guard-base-branch"]["steps"][0]
        assert step["env"]["HAS_BYPASS"] == "${{ contains(github.event.pull_request.labels.*.name, 'stacked-pr') }}"
