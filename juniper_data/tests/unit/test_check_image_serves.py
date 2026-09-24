"""Pin ``util/check_image_serves.py`` and its wiring into ``publish-image.yml``.

The script asserts what the image checks before it could not: that the image SERVES (liveness on
:8100, started as deployed) and that every version it reports is the one it was built as. The
ecosystem shipped the stale-version shape with every publish-path check green:
juniper-cascor-worker 0.5.0 / 0.6.0 imported fine while its ``__version__`` read ``0.4.0``, and
juniper-cascor 0.11.0 stamps ``meta.version: "0.6.0"`` on every enveloped response. This repo's
own source-checkout fallback literal read ``0.14.0`` at 0.15.0; the installed metadata was right,
which is why the check compares against metadata, not a literal.

These tests need no Docker. ``evaluate`` is pure, so each failure mode is driven with synthetic
observations. The workflow tests pin where the check runs -- on the PR arm, and against each pushed
digest BEFORE that digest is exported -- and that a release compares against the TAG. The real
execution happens in CI: the PR arm runs the script against the image it just built.
"""

from __future__ import annotations

import importlib.util
import subprocess  # nosec B404 - only referenced to patch subprocess.run
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "publish-image.yml"
SCRIPT = REPO / "util" / "check_image_serves.py"
SCRIPT_REL = "util/check_image_serves.py"
PUBLISH_IF = "github.event_name == 'release' || inputs.push"
BUILD_ONLY_IF = "github.event_name != 'release' && !inputs.push"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_image_serves", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def _step(job: str, name_prefix: str) -> dict[str, Any]:
    matches = [s for s in _workflow()["jobs"][job]["steps"] if str(s.get("name", "")).startswith(name_prefix)]
    assert len(matches) == 1, f"expected exactly one step in job {job!r} named {name_prefix!r}..., found {len(matches)}"
    return matches[0]


HEALTHY = {"status": "ok", "version": "0.17.0", "service": "juniper-data"}


def _evaluate(**overrides: Any) -> list[str]:
    kwargs: dict[str, Any] = {
        "expect": "0.17.0",
        "metadata": "0.17.0",
        "module": "juniper_data",
        "module_version": "0.17.0",
        "module_error": None,
        "health_status": 200,
        "health_body": dict(HEALTHY),
        "health_version_required": True,
        "enveloped": {},
    }
    kwargs.update(overrides)
    return list(_load().evaluate(**kwargs))


# ─────────────────────────────────────────────────────────────────────────────────────────────
# The pure verdict: one passing shape, and every failure mode it must catch
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestEvaluate:
    def test_a_healthy_image_at_the_expected_version_passes(self) -> None:
        assert _evaluate() == []

    def test_a_stale_package_version_fails(self) -> None:
        failures = _evaluate(module_version="0.16.0")
        assert any("__version__ 0.16.0 != metadata 0.17.0" in f for f in failures), failures

    def test_an_absent_version_is_a_failure_not_a_pass(self) -> None:
        """The class-2 sweep scored ABSENT as PASS; that is how a stale image slipped through."""
        failures = _evaluate(module_version=None)
        assert any("has no __version__" in f for f in failures), failures

    def test_a_module_that_does_not_import_fails(self) -> None:
        failures = _evaluate(module_version=None, module_error="ModuleNotFoundError: No module named 'juniper_data'")
        assert any("does not import" in f for f in failures), failures

    def test_an_image_that_is_not_the_tag_fails(self) -> None:
        """The published 0.15.0 image, checked as if it were 0.16.0."""
        failures = _evaluate(expect="0.16.0", metadata="0.15.0", module_version="0.15.0", health_body={"status": "ok", "version": "0.15.0"})
        assert failures == ["installed metadata version 0.15.0 != expected 0.16.0"], failures

    def test_missing_metadata_fails(self) -> None:
        failures = _evaluate(metadata=None, module_error="no distribution 'juniper-data'")
        assert any("no installed distribution metadata" in f for f in failures), failures

    def test_no_liveness_fails(self) -> None:
        failures = _evaluate(health_status=None, health_body=None)
        assert any("liveness answered None" in f for f in failures), failures

    def test_a_disagreeing_health_version_fails(self) -> None:
        failures = _evaluate(health_body={"status": "ok", "version": "0.16.0"})
        assert any("reports version 0.16.0 != metadata 0.17.0" in f for f in failures), failures

    def test_an_absent_health_version_fails_when_required(self) -> None:
        failures = _evaluate(health_body={"status": "ok"})
        assert any("carries no version field" in f for f in failures), failures

    def test_a_stale_envelope_fails(self) -> None:
        failures = _evaluate(enveloped={"/v1/example": (200, {"meta": {"version": "0.6.0"}})})
        assert any("/v1/example meta.version 0.6.0 != metadata 0.17.0" in f for f in failures), failures


class TestUsage:
    @pytest.mark.parametrize("value", ["", "0.17", "v0.17.0", "latest", "0.17.0 "])
    def test_a_malformed_expected_version_is_a_usage_error(self, value: str) -> None:
        assert _load().main(["--image", "x", "--dist", "juniper-data", "--port", "8100", "--expect-version", value]) == 2

    def test_no_docker_is_an_environment_error_not_a_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(*args: Any, **kwargs: Any) -> Any:
            raise FileNotFoundError("docker")

        monkeypatch.setattr(subprocess, "run", missing)
        assert _load().main(["--image", "x", "--dist", "juniper-data", "--port", "8100", "--expect-version", "0.17.0"]) == 2


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Wiring: the check runs on the PR arm AND against each pushed digest, before export
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestPublishWorkflowRunsTheServeCheck:
    def test_paths_filter_covers_the_script(self) -> None:
        assert SCRIPT_REL in _workflow()["on"]["pull_request"]["paths"]

    @pytest.mark.parametrize("name", ["Serve and version check (build-only runs)", "Verify pushed image serves and reports its version"])
    def test_both_steps_check_the_package_version_on_the_service_port(self, name: str) -> None:
        run = _step("build", name)["run"]
        assert "--dist juniper-data" in run and "--module juniper_data" in run and "--port 8100" in run

    def test_the_pr_arm_runs_it_against_the_image_it_built(self) -> None:
        step = _step("build", "Serve and version check (build-only runs)")
        assert step["if"] == BUILD_ONLY_IF
        assert SCRIPT_REL in step["run"]
        assert "data-smoke:${{ matrix.arch }}" in step["run"]
        assert step["env"]["APP_VERSION"] == "${{ steps.prov.outputs.app_version }}"

    def test_the_publish_path_runs_it_on_each_pushed_digest_before_export(self) -> None:
        names = [str(s.get("name", "")) for s in _workflow()["jobs"]["build"]["steps"]]
        verify = next(i for i, n in enumerate(names) if n.startswith("Verify pushed image serves and reports its version"))
        export = next(i for i, n in enumerate(names) if n.startswith("Export digest"))
        assert verify < export, "a failing arch must never export its digest to the merge job"
        step = _step("build", "Verify pushed image serves and reports its version")
        assert step["if"] == PUBLISH_IF, "the publish-path check must run under the same condition as the push itself"
        assert SCRIPT_REL in step["run"]
        assert "@${digest}" in step["run"], "the check must address the image by the digest just pushed"

    def test_a_release_is_checked_against_its_tag(self) -> None:
        step = _step("build", "Verify pushed image serves and reports its version")
        assert step["env"]["RELEASE_TAG"] == "${{ github.event.release.tag_name }}"
        run = step["run"]
        assert 'expect="${RELEASE_TAG#v}"' in run, "the release guard is 'v', so the tag's version is the tag minus 'v'"
        assert '"${expect}" != "${APP_VERSION}"' in run, "a tag that disagrees with pyproject.toml must fail before the image is judged"
        assert '--expect-version "${expect}"' in run
