"""Pin the Bandit SARIF upload in the Security Scans job.

The required ``Bandit`` check is created by code scanning from this upload, not
by the bandit process itself. Until juniper-data#502 the step ran only when
``reports/security/bandit.sarif`` already existed, and ``continue-on-error`` was
true. A bandit that wrote no file, or an upload that failed, left Security
Scans green and the pull request waiting on a ``Bandit`` context that never
posted. On a pull request the upload now always runs, and a failed upload fails
the job. Every other event keeps the old tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ci.yml"
_SARIF = "reports/security/bandit.sarif"
_IF = "always() && (github.event_name == 'pull_request' || hashFiles('reports/security/bandit.sarif') != '')"
_CONTINUE = "${{ github.event_name != 'pull_request' }}"


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _job() -> dict[str, Any]:
    job = _workflow()["jobs"]["security"]
    assert job["name"] == "Security Scans"
    return job


def _step(name: str) -> dict[str, Any]:
    return next(step for step in _job()["steps"] if step.get("name") == name)


def test_the_scan_writes_the_file_the_upload_reads() -> None:
    scan = _step("Run Bandit (SAST)")
    upload = _step("Upload Bandit SARIF to GitHub Security")
    assert f"-f sarif -o {_SARIF}" in scan["run"]
    assert upload["with"]["sarif_file"] == _SARIF


@pytest.mark.parametrize(
    ("event", "sarif_present", "runs", "failure_fails_the_job"),
    [
        ("pull_request", False, True, True),
        ("pull_request", True, True, True),
        ("push", False, False, False),
        ("push", True, True, False),
        ("schedule", False, False, False),
        ("workflow_dispatch", True, True, False),
        ("repository_dispatch", False, False, False),
    ],
)
def test_a_pull_request_uploads_even_when_the_sarif_is_missing(event: str, sarif_present: bool, runs: bool, failure_fails_the_job: bool) -> None:
    """A missing file used to skip the step; a failed upload used to leave the job green."""
    step = _step("Upload Bandit SARIF to GitHub Security")
    assert step["if"] == _IF
    assert step["continue-on-error"] is not True
    assert step["continue-on-error"] == _CONTINUE
    assert (event == "pull_request" or sarif_present) is runs
    assert (event == "pull_request") is failure_fails_the_job
