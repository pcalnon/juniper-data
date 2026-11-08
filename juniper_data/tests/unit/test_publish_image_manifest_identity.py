"""Rehearse the digest-identity half of ``Publish manifest``.

``imagetools create`` flattens each per-arch push. With provenance attestations on (the
build-push-action default) the pushed digest is an OCI index — the linux image plus an
attestation — and the published list names the image manifests, not those index digests.
Comparing the two directly fails every publish (worker dispatch run 34292278641, 2026-09-08).
The step resolves each pushed digest to the one linux image it carries, and a digest that
already is an image manifest stands for itself.

The rehearsal runs the workflow's own shell, with ``/tmp/digests`` and ``/tmp/manifest.json``
rewritten to the test's directory so nothing is written under ``/tmp``. A stub ``docker``
returns fixture manifests. The CPU-only ``docker run`` that follows the identity check is not
executed.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess  # nosec B404 - runs the workflow's own extracted shell (fixed argv)
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "publish-image.yml"
_JOB_ID = "merge"
_STEP_NAME = "Verify published image is CPU-only"
_IDENTITY_END = 'echo "--- CPU-only'
_FALLBACK = 'verified="${verified} ${images:-${pushed}}"'
_DIRECT = 'verified="${verified} ${pushed}"'
_AMD_IMAGE = "sha256:111amd64image"
_ARM_IMAGE = "sha256:222arm64image"
_AMD_INDEX = "sha256:333indexamd64"
_ARM_INDEX = "sha256:444indexarm64"
_OTHER = "sha256:555otherimage"


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _merge_job() -> dict[str, Any]:
    job = _workflow()["jobs"][_JOB_ID]
    assert job["name"] == "Publish manifest"
    return job


def _step() -> dict[str, Any]:
    return next(s for s in _merge_job()["steps"] if s.get("name") == _STEP_NAME)


def _linux(digest: str, arch: str) -> dict[str, Any]:
    return {"digest": digest, "platform": {"os": "linux", "architecture": arch}}


def _attestation(digest: str) -> dict[str, Any]:
    return {"digest": digest, "platform": {"os": "unknown", "architecture": "unknown"}}


def _index(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"manifests": list(entries)}


def _image_manifest() -> dict[str, Any]:
    """An image manifest has no ``manifests`` array, so the shell keeps the pushed digest."""
    return {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json"}


def _identity_shell(digests: Path, manifest: Path) -> str:
    run = str(_step()["run"])
    identity, marker, _rest = run.partition(_IDENTITY_END)
    assert marker, "the identity check must stay ahead of the CPU-only docker run"
    assert "/tmp/digests" in identity
    assert "/tmp/manifest.json" in identity
    script = identity.replace("/tmp/digests", str(digests)).replace("/tmp/manifest.json", str(manifest))
    # Drop the paths just substituted before scanning: tmp_path itself may live under /tmp on a CI runner.
    scrubbed = script.replace(str(digests), "").replace(str(manifest), "")
    assert "/tmp/" not in scrubbed, "the rehearsal must not keep a /tmp path the substitution missed"
    return script


def _run(
    tmp_path: Path,
    *,
    digest_files: dict[str, str],
    inspect: dict[str, dict[str, Any]],
    published: list[dict[str, Any]],
    shell: str | None = None,
) -> subprocess.CompletedProcess[str]:
    digests = tmp_path / "digests"
    digests.mkdir()
    for name, digest in digest_files.items():
        (digests / name).write_text(digest + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifests": published}), encoding="utf-8")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    for digest, body in inspect.items():
        (fixtures / digest).write_text(json.dumps(body), encoding="utf-8")

    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "docker"
    stub.write_text(
        f'#!/bin/sh\nref=\nfor a in "$@"; do\n  case "$a" in *@*) ref="$a" ;; esac\ndone\ndigest=${{ref##*@}}\nfile={shlex.quote(str(fixtures))}/"$digest"\nif [ ! -f "$file" ]; then echo "docker stub: no fixture for $digest" >&2; exit 1; fi\ncat "$file"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    script = shell if shell is not None else _identity_shell(digests, manifest)
    env = {
        "PATH": f"{bindir}{os.pathsep}/usr/bin:/bin",
        "HOME": str(tmp_path),
        "LANG": "C",
        "REGISTRY": "ghcr.io/example",
        "IMAGE_NAME": "juniper-data",
        "DOCKER_METADATA_OUTPUT_JSON": json.dumps({"tags": ["ghcr.io/example/juniper-data:1.2.3"]}),
    }
    return subprocess.run(  # nosec B603 - workflow shell, fixed argv, stub docker written above
        ["bash", "-e", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )


def test_the_step_runs_under_the_default_shell_and_fails_the_job() -> None:
    """A step that names no shell runs as ``bash -e {0}`` (no pipefail), which the rehearsal reproduces.

    A named ``shell:`` would change that (``shell: bash`` adds pipefail), so the rehearsal would stop
    matching the runner. ``continue-on-error`` would let a mismatch leave the job green.
    """
    step = _step()
    assert "shell" not in step
    assert "continue-on-error" not in step
    export = next(s for s in _workflow()["jobs"]["build"]["steps"] if s.get("name") == "Export digest")
    # The digest file is named after the arch. basename of that file is what the identity check compares.
    assert "/tmp/digests/${{ matrix.arch }}" in str(export["run"])
    assert "for f in /tmp/digests/*" in str(step["run"])
    assert 'arch_file="$(basename "${f}")"' in str(step["run"])


def test_a_provenance_index_resolves_to_its_linux_image(tmp_path: Path) -> None:
    """Attestations sit beside the image in the index and must not count as a second linux digest."""
    proc = _run(
        tmp_path,
        digest_files={"amd64": _AMD_INDEX, "arm64": _ARM_INDEX},
        inspect={
            _AMD_INDEX: _index(_linux(_AMD_IMAGE, "amd64"), _attestation("sha256:aaaamd64attest")),
            _ARM_INDEX: _index(_linux(_ARM_IMAGE, "arm64"), _attestation("sha256:bbbarm64attest")),
        },
        published=[_linux(_AMD_IMAGE, "amd64"), _linux(_ARM_IMAGE, "arm64"), _attestation("sha256:cccpublishedattest")],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "references exactly the verified images" in proc.stdout
    assert "::error::" not in proc.stdout


def test_comparing_pushed_digests_directly_rejects_that_same_publish(tmp_path: Path) -> None:
    """Dropping the resolution step is the 2026-09-08 failure: the list does not name the index digests."""
    digests = tmp_path / "digests"
    manifest = tmp_path / "manifest.json"
    script = _identity_shell(digests, manifest)
    mutated = script.replace(_FALLBACK, _DIRECT, 1)
    assert mutated != script, "the resolution fallback is no longer the line this mutation replaces"
    proc = _run(
        tmp_path,
        digest_files={"amd64": _AMD_INDEX, "arm64": _ARM_INDEX},
        inspect={
            _AMD_INDEX: _index(_linux(_AMD_IMAGE, "amd64")),
            _ARM_INDEX: _index(_linux(_ARM_IMAGE, "arm64")),
        },
        published=[_linux(_AMD_IMAGE, "amd64"), _linux(_ARM_IMAGE, "arm64")],
        shell=mutated,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "manifest list references" in proc.stdout
    assert _AMD_INDEX in proc.stdout
    assert _AMD_IMAGE in proc.stdout


def test_an_image_manifest_digest_stands_for_itself(tmp_path: Path) -> None:
    """Provenance off: the pushed digest has no ``manifests`` array and is the verified image."""
    proc = _run(
        tmp_path,
        digest_files={"amd64": _AMD_IMAGE, "arm64": _ARM_IMAGE},
        inspect={_AMD_IMAGE: _image_manifest(), _ARM_IMAGE: _image_manifest()},
        published=[_linux(_ARM_IMAGE, "arm64"), _linux(_AMD_IMAGE, "amd64")],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "references exactly the verified images" in proc.stdout


def test_a_list_that_names_a_different_image_fails(tmp_path: Path) -> None:
    proc = _run(
        tmp_path,
        digest_files={"amd64": _AMD_INDEX, "arm64": _ARM_INDEX},
        inspect={
            _AMD_INDEX: _index(_linux(_AMD_IMAGE, "amd64")),
            _ARM_INDEX: _index(_linux(_ARM_IMAGE, "arm64")),
        },
        published=[_linux(_AMD_IMAGE, "amd64"), _linux(_OTHER, "arm64")],
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "manifest list references" in proc.stdout
    assert _OTHER in proc.stdout
    assert _ARM_IMAGE in proc.stdout


def test_an_index_of_the_wrong_arch_fails(tmp_path: Path) -> None:
    """The digest file is named for the arch the census ran on. Another arch's image is not that census."""
    proc = _run(
        tmp_path,
        digest_files={"amd64": _AMD_INDEX},
        inspect={_AMD_INDEX: _index(_linux(_ARM_IMAGE, "arm64"))},
        published=[_linux(_ARM_IMAGE, "arm64")],
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "expected exactly one, for amd64" in proc.stdout
    assert _AMD_INDEX in proc.stdout


def test_an_index_with_two_linux_images_fails(tmp_path: Path) -> None:
    proc = _run(
        tmp_path,
        digest_files={"amd64": _AMD_INDEX},
        inspect={_AMD_INDEX: _index(_linux(_AMD_IMAGE, "amd64"), _linux(_ARM_IMAGE, "arm64"))},
        published=[_linux(_AMD_IMAGE, "amd64"), _linux(_ARM_IMAGE, "arm64")],
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "expected exactly one, for amd64" in proc.stdout
    assert "amd64 arm64" in proc.stdout


def test_an_index_with_no_linux_image_does_not_verify_the_published_list(tmp_path: Path) -> None:
    """No linux entry means the digest stands for itself, which is not the image the list publishes."""
    proc = _run(
        tmp_path,
        digest_files={"amd64": _AMD_INDEX, "arm64": _ARM_INDEX},
        inspect={
            _AMD_INDEX: _index(_attestation("sha256:aaaamd64attest")),
            _ARM_INDEX: _index(_attestation("sha256:bbbarm64attest")),
        },
        published=[_linux(_AMD_IMAGE, "amd64"), _linux(_ARM_IMAGE, "arm64")],
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "manifest list references" in proc.stdout
    assert _AMD_INDEX in proc.stdout
    assert _AMD_IMAGE in proc.stdout
