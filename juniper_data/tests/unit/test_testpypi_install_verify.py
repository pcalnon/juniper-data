"""Rehearse the TestPyPI install check in ``publish.yml``.

Production publish (``jobs.pypi``) waits on ``verify-testpypi-install``, the job the install check has run in since
the one-artifact template's sweep (juniper-cascor#777; ruling 2 of 2026-11-16): it needs ``testpypi`` and holds
``contents: read`` alone, so nothing it runs can mint a credential. Its last step installs the release just uploaded
to TestPyPI, pinned to the tag's version by pip's ``juniper-data==<version>``, reads the installed version from its
metadata, and finds the package with ``importlib.util.find_spec``. It does not compare that version with the tag; the
pin does. Two ways that step can go green without proving the upload:

* a second index (``--extra-index-url`` or a production PyPI fallback) can
  resolve a different ``juniper-data`` than the one just published, and with
  ``--no-deps`` that fallback fetches nothing else — it only widens which
  copy of this package pip will accept;
* ``pip`` or the package check exiting non-zero has to fail the step. A
  ``|| true``, or a ``continue-on-error`` on the step, lets ``pypi`` publish
  after a verify that did not succeed. The install is tried five times, 10 s
  apart, for TestPyPI's index lag, and the fifth failure fails the step with
  an error that names the lag.

The check reads installed metadata and locates the package without running it. ``juniper_data`` imports
``python-dotenv`` at import time, and ``--no-deps`` does not install it, so ``import juniper_data`` would fail a good
publish.

The step reads the release tag from its own ``env`` (``RELEASE_TAG``, which GitHub fills from
``${{ github.event.release.tag_name }}``), so no expression is pasted into the script, and GitHub runs a ``run:`` step
that names no ``shell:`` under ``bash -e``. This test sets that variable and the runner's ``RUNNER_TEMP`` and then
executes the workflow's own ``run:`` block. Stand-ins for ``sleep``, ``pip`` and ``python`` record their arguments, so
neither the 30 s TestPyPI lag wait nor a retry's 10 s is actually slept.
"""

from __future__ import annotations

import os
import re
import subprocess  # nosec B404 - runs the workflow's own extracted shell (fixed argv)
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "publish.yml"
_INSTALL_JOB = "verify-testpypi-install"
_STEP_NAME = "Verify TestPyPI install"
_TAG_EXPR = "${{ github.event.release.tag_name }}"
_TEST_INDEX = "https://test.pypi.org/simple/"
_TAG = re.compile(r"[A-Za-z0-9._-]+")
_LAG_ERROR = "is not installable from TestPyPI after 5 tries (index lag?): Re-run failed jobs; a real failure repeats"


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _step() -> dict[str, Any]:
    return next(s for s in _workflow()["jobs"][_INSTALL_JOB]["steps"] if s.get("name") == _STEP_NAME)


def _script() -> str:
    step = _step()
    script = step["run"]
    assert isinstance(script, str)
    # The tag reaches the script through the step's env, never as an expression pasted into its text.
    assert step.get("env") == {"RELEASE_TAG": _TAG_EXPR}
    assert "${{" not in script
    return script


def _code(script: str) -> str:
    """The step with comment lines removed, so a warning in a comment is not the command."""
    return "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))


def _run(tmp_path: Path, tag: str, *, pip_rc: int = 0, python_rc: int = 0) -> tuple[subprocess.CompletedProcess[str], list[str], list[str] | None, list[str] | None]:
    """Run the step's shell. ``sleep`` records its argument and returns immediately; ``python`` records where it ran."""
    assert _TAG.fullmatch(tag), tag
    script = _script()

    bindir = tmp_path / "bin"
    bindir.mkdir()
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    trace = tmp_path / "trace.txt"
    pip_argv = tmp_path / "pip-argv.txt"
    python_argv = tmp_path / "python-argv.txt"
    (bindir / "sleep").write_text('#!/bin/sh\nprintf \'sleep %s\\n\' "$*" >> "$TRACE"\nexit 0\n', encoding="utf-8")
    (bindir / "pip").write_text('#!/bin/sh\nprintf \'pip\\n\' >> "$TRACE"\nprintf \'%s\\n\' "$@" > "$PIP_ARGV"\nexit "${PIP_RC:-0}"\n', encoding="utf-8")
    (bindir / "python").write_text('#!/bin/sh\nprintf \'python\\n\' >> "$TRACE"\npwd > "$PYTHON_ARGV"\nprintf \'%s\\n\' "$@" >> "$PYTHON_ARGV"\nexit "${PYTHON_RC:-0}"\n', encoding="utf-8")
    for name in ("sleep", "pip", "python"):
        (bindir / name).chmod(0o755)

    env = {
        "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        "LANG": "C",
        "RELEASE_TAG": tag,
        "RUNNER_TEMP": str(runner_temp),
        "TRACE": str(trace),
        "PIP_ARGV": str(pip_argv),
        "PYTHON_ARGV": str(python_argv),
        "PIP_RC": str(pip_rc),
        "PYTHON_RC": str(python_rc),
    }
    proc = subprocess.run(  # nosec B603 B607 - workflow shell, fixed argv, stubs are written above
        ["bash", "-e", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        encoding="utf-8",
        check=False,
        timeout=15,
    )
    trace_lines = trace.read_text(encoding="utf-8").splitlines() if trace.exists() else []
    pip_args = pip_argv.read_text(encoding="utf-8").splitlines() if pip_argv.exists() else None
    python_args = python_argv.read_text(encoding="utf-8").splitlines() if python_argv.exists() else None
    if python_args is not None:
        # The check runs from the runner's temporary directory, outside any checkout: its first line is where.
        assert python_args[0] == str(runner_temp), python_args[0]
        python_args = python_args[1:]
    return proc, trace_lines, pip_args, python_args


def test_production_publish_waits_on_a_verify_step_that_can_fail() -> None:
    """A swallowed verify still lets ``pypi`` publish: it ``needs`` this job."""
    doc = _workflow()
    needs = doc["jobs"]["pypi"]["needs"]
    if not isinstance(needs, list):
        needs = [needs]
    assert "testpypi" in needs
    assert _INSTALL_JOB in needs
    assert doc["jobs"]["testpypi"]["environment"] == "testpypi"
    assert doc["jobs"]["testpypi"]["name"] == "Publish to TestPyPI"
    job = doc["jobs"][_INSTALL_JOB]
    assert job["name"] == "Verify the TestPyPI install"
    assert job["needs"] == "testpypi"
    assert job["permissions"] == {"contents": "read"}
    assert "environment" not in job
    assert "continue-on-error" not in job

    step = _step()
    assert "shell" not in step, "a named shell replaces GitHub's bash -e {0}, which this rehearsal reproduces (shell: bash adds pipefail; a custom template such as 'bash {0}' would drop -e and let a failing pip pass)"
    assert "continue-on-error" not in step
    assert set(step) == {"name", "env", "run"}

    code = _code(_script())
    assert "--no-deps" in code
    assert "--extra-index-url" not in code
    assert "https://pypi.org" not in code
    assert "|| true" not in code
    assert _TEST_INDEX in code
    assert "importlib.metadata" in code
    assert "import juniper_data" not in code
    assert "u.find_spec('juniper_data')" in code


@pytest.mark.parametrize(
    ("tag", "version"),
    [
        ("v0.17.1", "0.17.1"),
        # No release run reaches this step with a tag that lacks the `v`: since 2026-11-15 publish.yml's build job runs
        # only for startsWith(tag, 'v'), and every later job needs it. The case stays because it pins the step's own
        # `${RELEASE_TAG#v}`, which strips at most one `v` and needs none, should the guard or the tag convention change.
        ("0.17.1", "0.17.1"),
        ("v1.2.3-rc.1", "1.2.3-rc.1"),
    ],
)
def test_installs_the_stripped_version_from_testpypi_only(tmp_path: Path, tag: str, version: str) -> None:
    """One leading ``v`` comes off. The suffix stays. Nothing is added to the index list."""
    proc, trace, pip_args, python_args = _run(tmp_path, tag)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert trace == ["sleep 30", "pip", "python"]
    assert pip_args == ["install", "--no-deps", "--index-url", _TEST_INDEX, f"juniper-data=={version}"]
    assert python_args is not None
    assert python_args[0] == "-c"
    check = python_args[1]
    assert "importlib.metadata" in check
    assert "m.version('juniper-data')" in check
    assert "u.find_spec('juniper_data')" in check
    assert "import juniper_data" not in check


def test_a_failed_install_fails_before_the_version_check(tmp_path: Path) -> None:
    """An install that never succeeds is tried five times, 10 s apart, then fails naming index lag; the check never runs."""
    proc, trace, pip_args, python_args = _run(tmp_path, "v0.17.1", pip_rc=1)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert trace == ["sleep 30", "pip", "sleep 10", "pip", "sleep 10", "pip", "sleep 10", "pip", "sleep 10", "pip"]
    assert pip_args is not None and pip_args[0] == "install"
    assert python_args is None
    assert f"::error::juniper-data 0.17.1 {_LAG_ERROR}" in proc.stdout


def test_a_failed_version_check_fails_the_step(tmp_path: Path) -> None:
    """A non-zero exit from the installed-package check fails the step, after pip has run."""
    proc, trace, _pip_args, python_args = _run(tmp_path, "v0.17.1", python_rc=1)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert trace == ["sleep 30", "pip", "python"]
    assert python_args is not None
