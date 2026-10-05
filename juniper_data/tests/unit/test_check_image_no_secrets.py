"""Pin ``util/check_image_no_secrets.py`` and its wiring into ``publish-image.yml``.

The script is the check that does not depend on ``.dockerignore`` or a COPY allowlist being
right: it walks the trees an image actually ships and refuses a credential-shaped path. It
landed with the image credential exclusions (#408) and is piped into the image on the smoke
arm and against every pushed digest. It had no tests. The workflow comment records the
negative controls that were run by hand — a planted ``secrets/`` directory, a planted
``.env``, and a planted ``.pem`` each exit 1; no scan root, or a walk of zero files, exits 2
rather than reporting a pass.

These tests need no Docker. ``scan_roots`` is pointed at a temporary tree, and the pure
filename predicate is driven directly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "publish-image.yml"
SCRIPT = REPO / "util" / "check_image_no_secrets.py"
SCRIPT_REL = "util/check_image_no_secrets.py"
PUBLISH_IF = "github.event_name == 'release' || inputs.push"
BUILD_ONLY_IF = "github.event_name != 'release' && !inputs.push"

# One name per glob in BAD_FILE_GLOBS. Dropping or widening a glob fails this list.
CREDENTIAL_NAMES = (
    ".env",
    ".env.local",
    ".env.production",
    "server.key",
    "server.pem",
    "client.p12",
    "client.pfx",
    "store.jks",
    "store.keystore",
    "id_rsa",
    "id_rsa.pub",
    "id_ecdsa",
    "id_ed25519",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "vault.kdbx",
)

ALLOWED_NAMES = (
    ".env.example",
    ".env.sample",
    "server.pem.template",
    "id_rsa.dist",
    "credentials.example",
)

BENIGN_NAMES = ("readme.txt", "generator.py", "notes.env", "mycredentials", "credentials_old")


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_image_no_secrets", SCRIPT)
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


def _run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], roots: list[Path]) -> tuple[int, str]:
    module = _load()
    monkeypatch.setattr(module, "scan_roots", lambda: list(roots))
    code = module.main()
    return code, capsys.readouterr().out


def _root_with(tmp_path: Path, name: str, body: str = "x") -> Path:
    root = tmp_path / "root"
    root.mkdir()
    (root / name).write_text(body)
    return root


class TestIsBadFile:
    @pytest.mark.parametrize("name", CREDENTIAL_NAMES)
    def test_each_credential_glob_matches(self, name: str) -> None:
        assert _load().is_bad_file(name) is True

    @pytest.mark.parametrize("name", ALLOWED_NAMES)
    def test_a_template_suffix_is_not_a_credential(self, name: str) -> None:
        assert _load().is_bad_file(name) is False

    @pytest.mark.parametrize("name", BENIGN_NAMES)
    def test_a_benign_name_is_not_a_credential(self, name: str) -> None:
        assert _load().is_bad_file(name) is False

    def test_the_allowlist_is_a_suffix_not_a_substring(self) -> None:
        """``.env.example.key`` ends in ``.key``. Containing ``.example`` does not permit it."""
        assert _load().is_bad_file(".env.example.key") is True


class TestScanVerdict:
    def test_no_scan_root_is_a_failure_not_a_pass(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(monkeypatch, capsys, [])
        assert code == 2
        assert "no scan root found" in out
        assert "no credential-shaped file" not in out

    def test_zero_files_is_a_failure_not_a_pass(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 2
        assert "ZERO files" in out
        assert "no credential-shaped file" not in out

    def test_a_clean_tree_passes_and_does_not_echo_file_contents(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        root = _root_with(tmp_path, "generator.py", "TOKEN-SHOULD-NOT-LEAK")
        (root / "pkg").mkdir()
        (root / "pkg" / "mod.py").write_text("also-secret-looking")
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 0
        assert "scanned 2 files across 1 root(s):" in out
        assert "no credential-shaped file" in out
        assert "TOKEN-SHOULD-NOT-LEAK" not in out
        assert str(root) in out

    def test_a_planted_env_file_exits_1_without_printing_its_contents(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        root = _root_with(tmp_path, ".env", "AWS_SECRET_ACCESS_KEY=supersecretvalue")
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 1
        assert str(root / ".env") in out
        assert "supersecretvalue" not in out
        assert "1 credential-shaped path" in out

    def test_a_planted_pem_exits_1(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        root = _root_with(tmp_path, "server.pem", "-----BEGIN")
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 1
        assert str(root / "server.pem") in out

    @pytest.mark.parametrize("dirname", sorted({"secrets", ".git", ".ssh", ".aws", ".gnupg", "private"}))
    def test_each_forbidden_directory_is_reported_even_when_its_files_look_benign(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, dirname: str) -> None:
        root = tmp_path / "root"
        (root / dirname).mkdir(parents=True)
        (root / dirname / "readme.txt").write_text("not a matching filename")
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 1
        assert f"{root / dirname}/  (directory)" in out
        assert "readme.txt" not in out.split("::error::", 1)[1]

    def test_findings_from_every_root_are_reported_in_sorted_order(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        (first / "z.pem").write_text("a")
        (second / "a.key").write_text("b")
        code, out = _run(monkeypatch, capsys, [first, second])
        assert code == 1
        assert "scanned 2 files across 2 root(s):" in out
        reported = [line.strip() for line in out.split("::error::", 1)[1].splitlines() if line.startswith("    ")]
        assert reported == sorted(reported)
        assert str(first / "z.pem") in reported
        assert str(second / "a.key") in reported

    @pytest.mark.parametrize("prune", ("__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules"))
    def test_a_credential_inside_a_pruned_directory_is_not_a_finding(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, prune: str) -> None:
        root = _root_with(tmp_path, "generator.py", "ok")
        hidden = root / prune
        hidden.mkdir()
        (hidden / ".env").write_text("supersecretvalue")
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 0, out
        assert "supersecretvalue" not in out
        assert ".env" not in out

    def test_pruning_a_cache_does_not_hide_a_sibling_credential(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        root = _root_with(tmp_path, ".env", "supersecretvalue")
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "mod.pyc").write_bytes(b"\x00")
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 1
        assert str(root / ".env") in out
        assert "supersecretvalue" not in out

    def test_a_template_tree_passes(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        for name in ALLOWED_NAMES:
            (root / name).write_text("template")
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 0, out

    def test_a_symlinked_env_file_is_judged_by_its_name(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        target = tmp_path / "real.txt"
        target.write_text("supersecretvalue")
        root = tmp_path / "root"
        root.mkdir()
        (root / ".env").symlink_to(target)
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 1
        assert str(root / ".env") in out
        assert "supersecretvalue" not in out

    def test_a_symlinked_secrets_directory_is_flagged_without_being_followed(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "only-inside-link.pem").write_text("supersecretvalue")
        root = _root_with(tmp_path, "generator.py", "ok")
        (root / "secrets").symlink_to(outside, target_is_directory=True)
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 1
        assert f"{root / 'secrets'}/  (directory)" in out
        assert "only-inside-link.pem" not in out
        assert "supersecretvalue" not in out


class TestScanRoots:
    def _patch_app(self, monkeypatch: pytest.MonkeyPatch, module: Any, app: Path) -> None:
        real_path = module.Path

        def path_factory(*args: Any, **kwargs: Any) -> Path:
            if args == ("/app",):
                return real_path(app)
            return real_path(*args, **kwargs)

        monkeypatch.setattr(module, "Path", path_factory)

    def test_app_comes_first_and_only_juniper_trees_join_it(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        module = _load()
        app = tmp_path / "app"
        app.mkdir()
        site = tmp_path / "purelib"
        site.mkdir()
        kept = ["candidate_unit", "cascade_correlation", "juniper_data", "juniper_data-0.16.0.dist-info"]
        for name in kept + ["numpy", "pip"]:
            (site / name).mkdir()
        (site / "juniper_extra.py").write_text("not a package directory")
        self._patch_app(monkeypatch, module, app)
        monkeypatch.setattr(module.sysconfig, "get_paths", lambda: {"purelib": str(site)})

        roots = module.scan_roots()

        assert [path.name for path in roots] == ["app", *sorted(kept)]

    def test_a_missing_purelib_still_scans_app(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        module = _load()
        app = tmp_path / "app"
        app.mkdir()
        self._patch_app(monkeypatch, module, app)
        monkeypatch.setattr(module.sysconfig, "get_paths", lambda: {})
        assert module.scan_roots() == [app]

    def test_a_purelib_that_is_not_a_directory_is_skipped(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        module = _load()
        app = tmp_path / "app"
        app.mkdir()
        purelib = tmp_path / "not-a-dir"
        purelib.write_text("file")
        self._patch_app(monkeypatch, module, app)
        monkeypatch.setattr(module.sysconfig, "get_paths", lambda: {"purelib": str(purelib)})
        assert module.scan_roots() == [app]

    def test_no_app_and_no_package_is_an_empty_scan(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        module = _load()
        self._patch_app(monkeypatch, module, tmp_path / "missing-app")
        monkeypatch.setattr(module.sysconfig, "get_paths", lambda: {"purelib": None})
        assert module.scan_roots() == []


class TestPublishWorkflowRunsTheCredentialScan:
    def test_paths_filter_covers_the_script(self) -> None:
        assert SCRIPT_REL in _workflow()["on"]["pull_request"]["paths"]

    def test_the_smoke_arm_pipes_the_script_into_the_image_it_built(self) -> None:
        step = _step("build", "Smoke test (build-only runs)")
        assert step["if"] == BUILD_ONLY_IF
        assert f"python - < {SCRIPT_REL}" in step["run"]
        assert "data-smoke:${{ matrix.arch }}" in step["run"]

    def test_the_publish_path_scans_each_pushed_digest_before_export(self) -> None:
        names = [str(step.get("name", "")) for step in _workflow()["jobs"]["build"]["steps"]]
        scan = next(i for i, name in enumerate(names) if name.startswith("Verify pushed image is CPU-only"))
        export = next(i for i, name in enumerate(names) if name.startswith("Export digest"))
        assert scan < export, "a credential hit must fail the arch before its digest is exported"
        step = _step("build", "Verify pushed image is CPU-only")
        assert step["if"] == PUBLISH_IF
        assert f"python - < {SCRIPT_REL}" in step["run"]
        assert "@${digest}" in step["run"], "the scan must address the image by the digest just pushed"
