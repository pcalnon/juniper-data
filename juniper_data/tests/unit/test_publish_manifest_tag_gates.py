"""Rehearse the tag gates in the Publish manifest job, not the digest-identity shell.

``publish-image.yml``'s merge job is the only place a tag is written. The
identity shell — an index digest resolves to the linux image it carries — is
pinned on its own. Three neighboring gates in the same job are not, and each
one used to be able to publish the wrong thing while the job stayed green:

* metadata-action produced no tags, and ``imagetools create`` with no ``-t``
  failed with docker's error instead of naming the event
* the manifest list existed but did not contain both linux/amd64 and
  linux/arm64 (a windows entry must not count as the missing arch)
* a release whose semver rules matched nothing still applied ``:latest``, and
  metadata-action only warns when a tag does not parse

The tests execute those ``run:`` blocks under ``bash -e``, which is how GitHub
runs a ``run:`` step that names no ``shell:`` (``bash -e {0}``, no pipefail;
naming ``shell: bash`` would add ``-o pipefail``). ``docker`` is a stub;
``jq`` is the real binary. Scratch paths the workflow hardcodes under ``/tmp``
are relocated into the test's temporary directory. This repo does not write
fixtures under ``/tmp``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404 - runs the workflow's own extracted shell with fixed argv
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
_WORKFLOW = REPO / ".github" / "workflows" / "publish-image.yml"
# The step's shell sees only the stub directory and these, so bash is resolved from them too.
# A hard-coded /usr/bin/bash does not exist on macOS, where bash is /bin/bash.
_SYSTEM_PATH = os.pathsep.join(("/usr/bin", "/bin"))
_IMAGE = "ghcr.io/pcalnon/juniper-data"
_AMD_INDEX = "sha256:111"
_ARM_INDEX = "sha256:222"

_EXPRESSIONS = {
    "${{ github.event_name }}": "workflow_dispatch",
    "${{ github.event.release.tag_name }}": "v9.9.9",
}


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _job() -> dict[str, Any]:
    job = _workflow()["jobs"]["merge"]
    assert job["name"] == "Publish manifest"
    return job


def _step(name: str) -> dict[str, Any]:
    matches = [step for step in _job()["steps"] if step.get("name") == name]
    assert len(matches) == 1, f"expected one {name!r} step, found {len(matches)}"
    return matches[0]


def _script(name: str) -> str:
    step = _step(name)
    # A step that names no shell runs as ``bash -e {0}``, which _run() reproduces. A named
    # shell would change that (``shell: bash`` adds pipefail), so the rehearsal would not match.
    assert "shell" not in step
    assert "continue-on-error" not in step
    script = step.get("run")
    assert isinstance(script, str) and script.strip()
    return script


def _write_digest_files(root: Path, digests: dict[str, str]) -> None:
    directory = root / "digests"
    directory.mkdir(parents=True, exist_ok=True)
    for arch, digest in digests.items():
        (directory / arch).write_text(digest + "\n", encoding="utf-8")


def _docker_stub(root: Path, tag_manifest: dict[str, Any] | None) -> Path:
    """A ``docker`` that prints one fixture for a tag inspect and records every argv."""
    bindir = root / "bin"
    bindir.mkdir()
    fixdir = root / "fixtures"
    fixdir.mkdir()
    if tag_manifest is not None:
        (fixdir / "TAG").write_text(json.dumps(tag_manifest), encoding="utf-8")
    log = root / "docker.log"
    stub = bindir / "docker"
    stub.write_text(
        f"""#!/bin/sh
printf '%s\\n' "$*" >> {json.dumps(str(log))}
if [ "$1" = "buildx" ] && [ "$2" = "imagetools" ] && [ "$3" = "create" ]; then
  exit 0
fi
if [ "$1" = "buildx" ] && [ "$2" = "imagetools" ] && [ "$3" = "inspect" ]; then
  fixture={json.dumps(str(fixdir))}/TAG
  if [ ! -f "$fixture" ]; then
    printf 'no tag fixture\\n' >&2
    exit 1
  fi
  cat "$fixture"
  exit 0
fi
printf 'unexpected docker %s\\n' "$*" >&2
exit 1
""",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return log


def _relocate(script: str, root: Path) -> str:
    rendered = script
    for source, value in _EXPRESSIONS.items():
        rendered = rendered.replace(source, value)
    rendered = rendered.replace("/tmp/manifest.json", str(root / "manifest.json"))
    rendered = rendered.replace("/tmp/digests", str(root / "digests"))
    assert "${{" not in rendered
    # The relocated scratch paths sit under ``root``. Anything else under /tmp is a fixture leak.
    assert "/tmp/" not in rendered.replace(str(root), "")
    return rendered


def _run(
    tmp_path: Path,
    script: str,
    *,
    tags: list[str],
    digests: dict[str, str] | None = None,
    tag_manifest: dict[str, Any] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bash = shutil.which("bash", path=_SYSTEM_PATH)
    if bash is None:
        # Not a skip: the step runs under bash on the runner, and a skip would leave a required lane green.
        pytest.fail(f"no bash on {_SYSTEM_PATH}; the rehearsed step runs under bash")
    if digests is not None:
        _write_digest_files(tmp_path, digests)
    log = _docker_stub(tmp_path, tag_manifest)
    env = {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{_SYSTEM_PATH}",
        "HOME": str(tmp_path),
        "LANG": "C",
        "LC_ALL": "C",
        "REGISTRY": "ghcr.io",
        "IMAGE_NAME": "pcalnon/juniper-data",
        "DOCKER_METADATA_OUTPUT_JSON": json.dumps({"tags": tags}),
    }
    proc = subprocess.run(  # nosec B603 - workflow shell, fixed argv, fixtures written above
        [bash, "-e", "-c", _relocate(script, tmp_path)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return proc, lines


class TestArchitectures:
    """linux/amd64 and linux/arm64, both present. A non-linux manifest does not count."""

    def test_unsorted_linux_arches_and_a_windows_manifest_pass(self, tmp_path: Path) -> None:
        tag_manifest = {
            "manifests": [
                {"platform": {"architecture": "arm64", "os": "linux"}},
                {"platform": {"architecture": "amd64", "os": "windows"}},
                {"platform": {"architecture": "amd64", "os": "linux"}},
            ]
        }
        proc, _docker = _run(
            tmp_path,
            _script("Verify both architectures are present"),
            tags=[f"{_IMAGE}:1.2.3"],
            tag_manifest=tag_manifest,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "linux/amd64 + linux/arm64" in proc.stdout

    def test_a_single_architecture_fails(self, tmp_path: Path) -> None:
        tag_manifest = {"manifests": [{"platform": {"architecture": "amd64", "os": "linux"}}]}
        proc, _docker = _run(
            tmp_path,
            _script("Verify both architectures are present"),
            tags=[f"{_IMAGE}:1.2.3"],
            tag_manifest=tag_manifest,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "expected 'amd64,arm64'" in proc.stdout
        assert "got 'amd64'" in proc.stdout

    def test_an_empty_tag_list_fails_before_inspect(self, tmp_path: Path) -> None:
        proc, docker = _run(tmp_path, _script("Verify both architectures are present"), tags=[])
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "no tag to verify" in proc.stdout
        assert docker == []


class TestReleaseVersionTag:
    """``latest`` still fires when the semver rules match nothing. That must not publish as success."""

    def test_the_step_runs_on_a_release_only(self) -> None:
        step = _step("Verify the release produced a version tag")
        assert step["if"] == "github.event_name == 'release'"
        assert "${{ github.event.release.tag_name }}" in str(step["run"])

    def test_an_x_y_z_tag_passes(self, tmp_path: Path) -> None:
        tags = [f"{_IMAGE}:1.2.3", f"{_IMAGE}:1.2", f"{_IMAGE}:latest"]
        proc, _docker = _run(tmp_path, _script("Verify the release produced a version tag"), tags=tags)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "version tag present" in proc.stdout

    @pytest.mark.parametrize(
        "tag",
        [
            f"{_IMAGE}:latest",
            f"{_IMAGE}:1.2",
            f"{_IMAGE}:v1.2.3",
            f"{_IMAGE}:1.2.3-rc.1",
        ],
    )
    def test_anything_other_than_x_y_z_fails(self, tmp_path: Path, tag: str) -> None:
        proc, _docker = _run(tmp_path, _script("Verify the release produced a version tag"), tags=[tag])
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "produced no X.Y.Z" in proc.stdout
        assert "v9.9.9" in proc.stdout


class TestManifestCreate:
    """No tags means nothing was published. ``imagetools create`` with no ``-t`` is a different error."""

    def test_no_tags_exits_before_create(self, tmp_path: Path) -> None:
        raw = _script("Create manifest list")
        assert "${{ github.event_name }}" in raw
        proc, docker = _run(
            tmp_path,
            raw,
            tags=[],
            digests={"amd64": _AMD_INDEX, "arm64": _ARM_INDEX},
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "produced no tags" in proc.stdout
        assert "workflow_dispatch" in proc.stdout
        assert not any("imagetools create" in line for line in docker)

    def test_create_receives_every_tag_and_every_pushed_digest(self, tmp_path: Path) -> None:
        proc, docker = _run(
            tmp_path,
            _script("Create manifest list"),
            tags=[f"{_IMAGE}:1.2.3", f"{_IMAGE}:1.2"],
            digests={"amd64": _AMD_INDEX, "arm64": _ARM_INDEX},
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert len(docker) == 1
        command = docker[0]
        assert "imagetools create" in command
        assert f"-t {_IMAGE}:1.2.3" in command
        assert f"-t {_IMAGE}:1.2" in command
        assert f"{_IMAGE}@{_AMD_INDEX}" in command
        assert f"{_IMAGE}@{_ARM_INDEX}" in command


def test_the_merge_job_publishes_only_a_versioned_release_or_an_explicit_push() -> None:
    job = _job()
    assert job["needs"] == ["build"]
    assert job["if"] == "(github.event_name == 'release' && startsWith(github.event.release.tag_name, 'v')) || inputs.push"
