"""Rehearse the Quality Gate shell in ``ci.yml`` against the job results it branches on.

``required-checks`` is the merge aggregator. It is not a reimplementation of the
other jobs: it reads each ``needs.<job>.result`` and decides whether the workflow
is green. Two predicates are in the same step, and they are not the same:

* pre-commit, unit tests, the build, the docs lint, and the lockfile check pass
  only when the result is exactly ``success``. A skip or a cancellation is a
  failed gate — those jobs are not allowed to vanish.
* dependency docs, security, integration tests, and the Docker build fail the
  gate only on exactly ``failure``. ``skipped`` is the documented pass for the
  lanes that do not run on every event, and ``cancelled`` takes that same path.

GitHub substitutes ``${{ needs.<job>.result }}`` before bash runs, so this test
does the same substitution and then executes the workflow's own ``run:`` block.
A copy of the shell would go stale the moment the step was edited.

The job also has to *run* when a needed job failed. Without ``if: always()``
GitHub skips the aggregator, and a skipped required check is not a red one.
"""

from __future__ import annotations

import re
import subprocess  # nosec B404 - runs the workflow's own extracted shell (fixed argv, no shell interpolation)
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ci.yml"
_STEP_NAME = "Check Quality Gate Status"
_PLACEHOLDER = re.compile(r"\$\{\{\s*needs\.([A-Za-z0-9_-]+)\.result\s*\}\}")

# Exact ``::error::`` text the shell prints. A swapped predicate that still
# exits 1 on the wrong job would otherwise look green.
_REQUIRED: dict[str, str] = {
    "pre-commit": "Pre-commit checks failed",
    "unit-tests": "Unit tests failed",
    "build": "Build failed",
    "docs": "Documentation link validation failed",
    "lockfile-check": "Lockfile freshness check failed — requirements.lock is stale",
}
_OPTIONAL: dict[str, str] = {
    "dependency-docs": "Dependency documentation generation failed",
    "security": "Security scans failed",
    "integration-tests": "Integration tests failed",
    "docker-build": "Docker build and smoke test failed",
}


def _job() -> dict:
    doc = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    job = doc["jobs"]["required-checks"]
    assert job["name"] == "Quality Gate"
    return job


def _script() -> str:
    step = next(s for s in _job()["steps"] if s.get("name") == _STEP_NAME)
    script = step["run"]
    assert isinstance(script, str) and "Quality Gate PASSED" in script
    return script


def _passing(**overrides: str) -> dict[str, str]:
    results = dict.fromkeys(_REQUIRED, "success")
    results.update(dict.fromkeys(_OPTIONAL, "skipped"))
    results.update(overrides)
    return results


def _run(results: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script = _script()
    found = set(_PLACEHOLDER.findall(script))
    assert found == set(results), f"shell reads {sorted(found)}; fixture has {sorted(results)}"

    def repl(match: re.Match[str]) -> str:
        return results[match.group(1)]

    rendered = _PLACEHOLDER.sub(repl, script)
    assert "${{" not in rendered
    return subprocess.run(  # nosec B603 B607 - workflow shell, fixed argv, values are test literals
        ["bash", "-c", rendered],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_the_gate_runs_after_a_failed_need_and_reads_every_need() -> None:
    """A missing ``if: always()`` skips this job when a need fails, which is a silent green."""
    job = _job()
    assert job["if"] == "always()"
    read = set(_PLACEHOLDER.findall(_script()))
    assert read == set(job["needs"])
    assert read == set(_REQUIRED) | set(_OPTIONAL)


def test_passes_when_required_jobs_succeed_and_conditional_jobs_are_skipped() -> None:
    proc = _run(_passing())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Quality Gate PASSED" in proc.stdout


def test_passes_when_every_job_succeeded() -> None:
    proc = _run(_passing(**dict.fromkeys(_OPTIONAL, "success")))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Quality Gate PASSED" in proc.stdout


@pytest.mark.parametrize("job", list(_REQUIRED))
@pytest.mark.parametrize("result", ["failure", "skipped", "cancelled", "", "Success"])
def test_a_required_job_passes_only_on_exact_success(job: str, result: str) -> None:
    proc = _run(_passing(**{job: result}))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert _REQUIRED[job] in proc.stdout
    assert "Quality Gate PASSED" not in proc.stdout


@pytest.mark.parametrize("job", list(_OPTIONAL))
def test_a_conditional_job_fails_the_gate_on_failure(job: str) -> None:
    proc = _run(_passing(**{job: "failure"}))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert _OPTIONAL[job] in proc.stdout
    assert "Quality Gate PASSED" not in proc.stdout


@pytest.mark.parametrize("job", list(_OPTIONAL))
@pytest.mark.parametrize("result", ["skipped", "cancelled", "success", "Failure"])
def test_a_conditional_job_stays_green_unless_the_result_is_failure(job: str, result: str) -> None:
    proc = _run(_passing(**{job: result}))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Quality Gate PASSED" in proc.stdout
    assert _OPTIONAL[job] not in proc.stdout


def test_the_first_failing_required_job_is_the_one_named() -> None:
    """pre-commit is checked before unit-tests, so a double failure names only the first."""
    proc = _run(_passing(**{"pre-commit": "failure", "unit-tests": "skipped"}))
    assert proc.returncode == 1
    assert "Pre-commit checks failed" in proc.stdout
    assert "Unit tests failed" not in proc.stdout
    assert "Quality Gate PASSED" not in proc.stdout
