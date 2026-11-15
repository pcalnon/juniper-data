"""Pin the marker selector of the ``Scheduled Tests`` workflow, ``scheduled-tests.yml``.

The workflow runs daily at 06:00 UTC and on manual dispatch. Its selector is ``slow or integration or performance``.
``performance`` is the marker of the pytest-benchmark benchmarks in ``juniper_data/tests/performance/``, which no
other lane selects, so this lane is the only place they run. They run as a smoke run: ``pyproject.toml``'s addopts
pass ``--benchmark-disable``, so each benchmarked function runs once and is not timed. The lane catches a benchmark
that breaks, not one that slows down, and it gates no pull request, because the workflow has no pull_request trigger.

The test runs the step's own ``run:`` block under ``bash -e``, which is what GitHub runs a ``run:`` step with on Linux
and macOS when the step names no ``shell:``. A stand-in ``python`` on ``PATH`` records its arguments, so a copy of the
command cannot drift from the step.
"""

from __future__ import annotations

import os
import shlex
import subprocess  # nosec B404 - runs the workflow's own extracted shell (fixed argv)
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[3]
_WORKFLOW = _REPO / ".github" / "workflows" / "scheduled-tests.yml"
_PYPROJECT = _REPO / "pyproject.toml"
_JOB_ID = "slow-or-integration-tests"
_STEP_NAME = "Run slow / integration tests"
_SELECTOR = "slow or integration or performance"
_PYTEST_ARGV = ["-m", "pytest", "-m", _SELECTOR, "--tb=short", "--maxfail=10", "--junitxml=reports/junit/junit-slow.xml"]
_CALL_END = "--end-of-call--"


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _job() -> dict[str, Any]:
    return _workflow()["jobs"][_JOB_ID]


def _step() -> dict[str, Any]:
    return next(s for s in _job()["steps"] if s.get("name") == _STEP_NAME)


def _python_calls(tmp_path: Path) -> list[list[str]]:
    """Run the step's shell with a ``python`` that records each call's arguments; return one list per call."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "python-argv.txt"
    stub = bindir / "python"
    stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" {shlex.quote(_CALL_END)} >> {shlex.quote(str(argv_log))}\nexit 0\n', encoding="utf-8")
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
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls: list[list[str]] = [[]]
    for line in argv_log.read_text(encoding="utf-8").splitlines() if argv_log.exists() else []:
        if line == _CALL_END:
            calls.append([])
        else:
            calls[-1].append(line)
    return calls[:-1]


def test_the_step_runs_under_the_default_shell() -> None:
    """``bash -e`` is the default only when the step names no ``shell:``; a named one changes the flags."""
    assert "shell" not in _step()


def test_the_step_selects_slow_integration_and_performance(tmp_path: Path) -> None:
    """One pytest call, over ``testpaths``, whose selector adds ``performance`` to ``slow or integration``."""
    assert _python_calls(tmp_path) == [_PYTEST_ARGV]


def test_the_benchmarks_run_untimed(tmp_path: Path) -> None:
    """``--benchmark-disable`` comes from addopts; nothing in the workflow turns timing back on."""
    addopts = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--benchmark-disable" in addopts
    assert not [arg for call in _python_calls(tmp_path) for arg in call if arg.startswith("--benchmark")]
    for scope in (_workflow(), _job(), _step()):
        assert "PYTEST_ADDOPTS" not in (scope.get("env") or {})


def test_the_lane_gates_no_pull_request() -> None:
    """The workflow runs only on its schedule and on manual dispatch, so no pull request waits on it."""
    assert set(_workflow()["on"]) == {"schedule", "workflow_dispatch"}
