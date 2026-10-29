"""Rehearse the ``Slow Tests`` shell in ``ci.yml`` against the exit statuses pytest can return.

``Slow Tests`` is one of the 22 contexts ruleset ``juniper-data-rules`` (id ``14748749``) requires,
and the job runs only on scheduled and manually dispatched runs. Its pytest call used to end
``|| echo "No slow tests found"``, so the job reported success whatever pytest returned: a failing
slow test, a collection error, and a run that selected no test all read as green (juniper-ml's
flood-3 record, ``notes/JUNIPER_2026-10-08_JUNIPER-ECOSYSTEM_CURSOR-FLOOD-3-DISPOSITION.md`` §4).

pytest's exit status is now the step's. Exit 5 (no test selected) fails the step too, with an
annotation that names it: a lane that runs nothing must not pass.

The test runs the workflow's own ``run:`` block under ``bash -e``, which is what GitHub runs a
``run:`` step with on Linux and macOS when the step names no ``shell:``. A stand-in ``python`` on
``PATH`` records its arguments and exits with the status under test, so a copy of the shell cannot
drift from the step.
"""

from __future__ import annotations

import os
import shlex
import subprocess  # nosec B404 - runs the workflow's own extracted shell (fixed argv)
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ci.yml"
_JOB_ID = "slow-tests"
_STEP_NAME = "Run Slow Tests"
_PYTEST_ARGV = ["-m", "pytest", "-m", "slow", "--verbose", "--timeout=600"]
_NO_TESTS_ERROR = "::error title=Slow Tests::"


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _job() -> dict[str, Any]:
    job = _workflow()["jobs"][_JOB_ID]
    # The job name is the required-status context; renaming it strands the requirement.
    assert job["name"] == "Slow Tests"
    return job


def _step() -> dict[str, Any]:
    return next(s for s in _job()["steps"] if s.get("name") == _STEP_NAME)


def _run(tmp_path: Path, status: int) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the step's shell with a ``python`` that exits ``status``; return the result and its argv."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "python-argv.txt"
    stub = bindir / "python"
    stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {shlex.quote(str(argv_log))}\nexit {status}\n', encoding="utf-8")
    stub.chmod(0o755)
    env = {
        "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        "LANG": "C",
    }
    proc = subprocess.run(  # nosec B603 B607 - workflow shell, fixed argv, the stub is written above
        ["bash", "-e", "-c", str(_step()["run"])],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    argv = argv_log.read_text(encoding="utf-8").splitlines() if argv_log.exists() else []
    return proc, argv


def test_the_step_runs_under_the_default_shell() -> None:
    """``bash -e`` is the default only when the step names no ``shell:``; a named one changes the flags."""
    assert "shell" not in _step()


def test_the_lane_runs_only_on_schedule_and_dispatch() -> None:
    """A pull request reports this required context as skipped, which counts as passing."""
    assert _job()["if"] == "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    triggers = _workflow()["on"]
    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers


def test_a_passing_lane_passes(tmp_path: Path) -> None:
    proc, argv = _run(tmp_path, 0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert argv == _PYTEST_ARGV
    assert _NO_TESTS_ERROR not in proc.stdout


@pytest.mark.parametrize("status", [1, 2, 3, 4])
def test_a_failing_or_broken_run_fails_the_job(tmp_path: Path, status: int) -> None:
    """1 = a test failed, 2 = interrupted (a collection error), 3 = internal error, 4 = usage error."""
    proc, argv = _run(tmp_path, status)
    assert argv == _PYTEST_ARGV
    assert proc.returncode == status, proc.stdout + proc.stderr
    assert "No slow tests found" not in proc.stdout


def test_a_lane_that_selects_no_test_fails_and_says_why(tmp_path: Path) -> None:
    """pytest exits 5 when the marker expression deselects every collected test."""
    proc, argv = _run(tmp_path, 5)
    assert argv == _PYTEST_ARGV
    assert proc.returncode == 5, proc.stdout + proc.stderr
    assert _NO_TESTS_ERROR in proc.stdout
    assert "exit 5" in proc.stdout
