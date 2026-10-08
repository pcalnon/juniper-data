#!/usr/bin/env python3
"""Execute the main-verify screen-verdict shells.

``test_main_verify_catchup_base.py`` rehearses how BASE is chosen and pins the
step names. It never runs the two shells those names belong to. A finding
(exit 1) must stay screened, so the next catch-up advances past it. An
invocation error (exit >= 2), or a missing code, must not. The clean assert
must still fail the job on a finding. Those are different thresholds in two
steps; renaming a step does not keep them apart.

Project: juniper-data
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs the workflow's own extracted shell (fixed argv)
import tempfile
import unittest
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW_NAME = "main-verify.yml"
VERDICT_STEP = "Assert screens reached a verdict"
CLEAN_STEP = "Assert screens clean"


def _child_env(**overrides: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),  # nosec B108 - git/bash fallback only
        "LANG": "C",
    }
    env.update(overrides)
    return env


def _repo_root() -> Path:
    cur = Path(__file__).resolve().parent
    for _ in range(8):
        if (cur / ".github" / "workflows").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise AssertionError(f"could not locate repo root from {Path(__file__)}")


def _step_script(name: str) -> str:
    wf = _repo_root() / ".github" / "workflows" / WORKFLOW_NAME
    if not wf.is_file():
        raise AssertionError(f"{WORKFLOW_NAME} missing at {wf}")
    doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
    steps = doc.get("jobs", {}).get("symbol-screen", {}).get("steps", [])
    step = next((s for s in steps if s.get("name") == name), None)
    if step is None or "run" not in step:
        raise AssertionError(f"{name!r} run step missing from {WORKFLOW_NAME}")
    return step["run"]


def _run(script: str, **env_vars: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "step.sh"
        path.write_text(script, encoding="utf-8")
        env = _child_env(**env_vars)
        return subprocess.run(  # nosec B603 B607 - extracted workflow shell, fixed argv
            ["bash", str(path)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=15,
        )


class ScreenVerdictRehearsalTest(unittest.TestCase):
    """Drive the real assert shells. Codes are the screens' own exit statuses."""

    verdict: str
    clean: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.verdict = _step_script(VERDICT_STEP)
        cls.clean = _step_script(CLEAN_STEP)

    def _assert_status(self, script: str, code: int, needle: str, **env_vars: str) -> None:
        proc = _run(script, **env_vars)
        combined = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, code, msg=combined)
        self.assertIn(needle, combined)

    def test_a_finding_is_screened_and_is_not_clean(self) -> None:
        """Exit 1 is a verdict. The job still goes red, from the clean assert."""
        for src, drc in (("1", "0"), ("0", "1"), ("1", "1")):
            with self.subTest(src=src, drc=drc):
                self._assert_status(self.verdict, 0, "IS screened", SRC=src, DRC=drc)
                self._assert_status(self.clean, 1, "compositional-loss finding", SRC=src, DRC=drc, HEAD_SHA="abc123")

    def test_an_invocation_error_is_not_coverage(self) -> None:
        """Exit >= 2 on either screen means the window was not screened."""
        for src, drc in (("2", "0"), ("0", "2"), ("3", "1"), ("0", "9")):
            with self.subTest(src=src, drc=drc):
                self._assert_status(self.verdict, 2, "NOT screened", SRC=src, DRC=drc)

    def test_absent_and_empty_codes_are_invocation_errors(self) -> None:
        """A missing output is 99, never a successful screen. Empty is missing."""
        self._assert_status(self.verdict, 2, "NOT screened")
        self._assert_status(self.verdict, 2, "NOT screened", SRC="", DRC="")
        self._assert_status(self.verdict, 2, "NOT screened", SRC="0", DRC="")
        self._assert_status(self.verdict, 2, "NOT screened", SRC="", DRC="0")
        clean = _run(self.clean, HEAD_SHA="abc123")
        self.assertEqual(clean.returncode, 1, msg=clean.stdout + clean.stderr)
        self.assertIn("compositional-loss finding", clean.stdout + clean.stderr)

    def test_both_screens_clean_exits_zero(self) -> None:
        self._assert_status(self.verdict, 0, "IS screened", SRC="0", DRC="0")
        self._assert_status(self.clean, 0, "screens clean", SRC="0", DRC="0", HEAD_SHA="abc123def456")
