"""Rehearse the publish-manifest checks in ``publish-image.yml``.

The merge job is the only place a tag is written. Four fail-closed edges in its
own shell are what keep a tag from pointing at something other than the images
the build jobs verified:

* an empty metadata tag set is refused before ``imagetools create``
* the published list must contain linux/amd64 and linux/arm64, and nothing else
  counts toward that pair
* a release must produce an ``X.Y.Z`` tag; ``latest`` alone is not one
* the list's linux digests must be exactly the image manifests the pushed
  digests resolve to

That last comparison is the one that used to fail every publish. With
provenance attestations on (the build-push-action default) the digest a build
job pushes is a per-arch OCI index, and ``imagetools create`` flattens it, so
the tag's list cites the image manifests inside those indexes. Comparing the
list to the pushed index digests directly does not match (worker dispatch run
34292278641, 2026-09-08). A pushed digest that is already an image manifest has
no ``.manifests`` and stands for itself. An index that carries more than one
linux image, or a linux image for the other arch, is not "the image this build
verified".

The tests execute the workflow's own ``run:`` blocks under ``bash -eo pipefail``,
which is how GitHub runs a ``run:`` step that names no ``shell:``. ``docker`` is
a stub that prints fixture manifests; ``jq`` is the real binary, so the filters
are the workflow's. Scratch paths the workflow hardcodes under ``/tmp`` are
relocated into the test's temporary directory. This repo does not write fixtures
under ``/tmp``.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - runs the workflow's own extracted shell with fixed argv
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
_WORKFLOW = REPO / ".github" / "workflows" / "publish-image.yml"
_IMAGE = "ghcr.io/pcalnon/juniper-data"

# Chosen so filename order (amd64, then arm64) is the opposite of digest sort
# order (aaa, then fff). Dropping either sort then fails the identity check.
_AMD_IMAGE = "sha256:fff"
_ARM_IMAGE = "sha256:aaa"
_AMD_INDEX = "sha256:111"
_ARM_INDEX = "sha256:222"
_WINDOWS = "sha256:eee"
_ATTESTATION = "sha256:ddd"

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
    # ``bash -eo pipefail`` is the default only when the step names no shell.
    assert "shell" not in step
    assert "continue-on-error" not in step
    script = step.get("run")
    assert isinstance(script, str) and script.strip()
    return script


def _linux(digest: str, arch: str) -> dict[str, Any]:
    return {"digest": digest, "platform": {"architecture": arch, "os": "linux"}}


def _published(*digests: tuple[str, str], windows: bool = True) -> dict[str, Any]:
    manifests = [_linux(digest, arch) for digest, arch in digests]
    if windows:
        manifests.append({"digest": _WINDOWS, "platform": {"architecture": "amd64", "os": "windows"}})
    return {"manifests": manifests}


def _index(arch: str, image: str, *, attest: bool = False) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    if attest:
        manifests.append({"digest": _ATTESTATION, "platform": {"architecture": "unknown", "os": "unknown"}})
    manifests.append(_linux(image, arch))
    return {"manifests": manifests}


def _write_digest_files(root: Path, digests: dict[str, str]) -> None:
    directory = root / "digests"
    directory.mkdir(parents=True, exist_ok=True)
    for arch, digest in digests.items():
        (directory / arch).write_text(digest + "\n", encoding="utf-8")


def _docker_stub(root: Path, inspect_by_digest: dict[str, Any], tag_manifest: dict[str, Any] | None) -> Path:
    """A ``docker`` that prints fixture JSON for ``imagetools inspect`` and records every argv."""
    bindir = root / "bin"
    bindir.mkdir()
    fixdir = root / "fixtures"
    fixdir.mkdir()
    for digest, body in inspect_by_digest.items():
        (fixdir / digest).write_text(json.dumps(body), encoding="utf-8")
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
if [ "$1" = "run" ]; then
  exit 0
fi
if [ "$1" = "buildx" ] && [ "$2" = "imagetools" ] && [ "$3" = "inspect" ]; then
  ref="$4"
  case "$ref" in
    *@*)
      digest="${{ref##*@}}"
      fixture={json.dumps(str(fixdir))}/"$digest"
      ;;
    *)
      fixture={json.dumps(str(fixdir))}/TAG
      ;;
  esac
  if [ ! -f "$fixture" ]; then
    printf 'no fixture for %s\\n' "$ref" >&2
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
    manifest: dict[str, Any] | None = None,
    inspect_by_digest: dict[str, Any] | None = None,
    tag_manifest: dict[str, Any] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    if digests is not None:
        _write_digest_files(tmp_path, digests)
    if manifest is not None:
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    log = _docker_stub(tmp_path, inspect_by_digest or {}, tag_manifest)
    env = {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}/usr/bin:/bin",
        "HOME": str(tmp_path),
        "LANG": "C",
        "LC_ALL": "C",
        "REGISTRY": "ghcr.io",
        "IMAGE_NAME": "pcalnon/juniper-data",
        "DOCKER_METADATA_OUTPUT_JSON": json.dumps({"tags": tags}),
    }
    proc = subprocess.run(  # nosec B603 - workflow shell, fixed argv, fixtures written above
        ["/usr/bin/bash", "-e", "-o", "pipefail", "-c", _relocate(script, tmp_path)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return proc, lines


def _index_world() -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    """Pushed indexes that flatten to image digests, plus the tag's manifest list."""
    digests = {"amd64": _AMD_INDEX, "arm64": _ARM_INDEX}
    inspect = {
        _AMD_INDEX: _index("amd64", _AMD_IMAGE, attest=True),
        _ARM_INDEX: _index("arm64", _ARM_IMAGE),
    }
    # fff before aaa: the array order is not the sorted order.
    manifest = _published((_AMD_IMAGE, "amd64"), (_ARM_IMAGE, "arm64"))
    return digests, inspect, manifest


class TestDigestIdentity:
    """The tag's linux digests are the image manifests inside the pushed indexes."""

    def test_an_index_matches_the_image_it_carries_not_itself(self, tmp_path: Path) -> None:
        digests, inspect, manifest = _index_world()
        # Vacuity: the pushed digests and the listed image digests are different sets.
        assert {_AMD_INDEX, _ARM_INDEX}.isdisjoint({_AMD_IMAGE, _ARM_IMAGE})
        proc, docker = _run(
            tmp_path,
            _script("Verify published image is CPU-only"),
            tags=[f"{_IMAGE}:1.2.3"],
            digests=digests,
            manifest=manifest,
            inspect_by_digest=inspect,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "references exactly the verified images" in proc.stdout
        assert any("imagetools inspect" in line and _AMD_INDEX in line for line in docker)
        assert any(line.startswith("run ") and "EXPECT_TORCH=absent" in line for line in docker)

    def test_a_bare_image_manifest_stands_for_itself(self, tmp_path: Path) -> None:
        """Provenance off: inspect returns no ``.manifests``, so the pushed digest is the image."""
        bare = {"schemaVersion": 2}
        assert "manifests" not in bare
        proc, docker = _run(
            tmp_path,
            _script("Verify published image is CPU-only"),
            tags=[f"{_IMAGE}:1.2.3"],
            digests={"amd64": _AMD_IMAGE, "arm64": _ARM_IMAGE},
            manifest=_published((_AMD_IMAGE, "amd64"), (_ARM_IMAGE, "arm64")),
            inspect_by_digest={_AMD_IMAGE: bare, _ARM_IMAGE: bare},
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert any("imagetools inspect" in line and _AMD_IMAGE in line for line in docker)

    def test_two_linux_images_in_one_index_fail(self, tmp_path: Path) -> None:
        digests, inspect, manifest = _index_world()
        inspect[_AMD_INDEX] = {"manifests": [_linux(_AMD_IMAGE, "amd64"), _linux("sha256:ccc", "amd64")]}
        proc, docker = _run(
            tmp_path,
            _script("Verify published image is CPU-only"),
            tags=[f"{_IMAGE}:1.2.3"],
            digests=digests,
            manifest=manifest,
            inspect_by_digest=inspect,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "expected exactly one" in proc.stdout
        assert "amd64" in proc.stdout
        assert not any(line.startswith("run ") for line in docker)

    def test_an_index_for_the_wrong_architecture_fails(self, tmp_path: Path) -> None:
        digests, inspect, manifest = _index_world()
        inspect[_AMD_INDEX] = _index("arm64", _AMD_IMAGE)
        proc, _docker = _run(
            tmp_path,
            _script("Verify published image is CPU-only"),
            tags=[f"{_IMAGE}:1.2.3"],
            digests=digests,
            manifest=manifest,
            inspect_by_digest=inspect,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "expected exactly one" in proc.stdout
        assert "arm64" in proc.stdout

    @pytest.mark.parametrize(
        "listed",
        [
            [(_AMD_IMAGE, "amd64"), (_ARM_IMAGE, "arm64"), ("sha256:abc", "amd64")],
            [(_ARM_IMAGE, "arm64")],
        ],
        ids=["extra", "missing"],
    )
    def test_a_list_that_is_not_the_resolved_set_fails(self, tmp_path: Path, listed: list[tuple[str, str]]) -> None:
        digests, inspect, _manifest = _index_world()
        proc, _docker = _run(
            tmp_path,
            _script("Verify published image is CPU-only"),
            tags=[f"{_IMAGE}:1.2.3"],
            digests=digests,
            manifest=_published(*listed),
            inspect_by_digest=inspect,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "manifest list references" in proc.stdout
        assert _AMD_IMAGE in proc.stdout and _ARM_IMAGE in proc.stdout


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
