"""Pin ``util/check_image_no_secrets.py``.

The publish workflow pipes this into the image. It is the check that looks at the
built filesystem, because a directory COPY and a root-anchored ``.dockerignore``
both miss a nested credential tree, and a scan of ``/app`` alone is vacuous on the
images whose code lives in site-packages. An empty scan must not report success.

These tests need no Docker. ``is_bad_file`` and ``scan_roots`` are pure enough to
drive directly; ``main`` walks a temporary tree through the real ``os.walk``.

Consolidated from juniper-data #450 and #449 (Cursor fleet): #450's suite, plus #449's
symlink, per-directory, per-prune-directory and do-not-echo-contents cases.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "util" / "check_image_no_secrets.py"
WORKFLOW = REPO / ".github" / "workflows" / "publish-image.yml"
SCRIPT_REL = "util/check_image_no_secrets.py"
PUBLISH_IF = "github.event_name == 'release' || inputs.push"
BUILD_ONLY_IF = "github.event_name != 'release' && !inputs.push"

_spec = importlib.util.spec_from_file_location("check_image_no_secrets", SCRIPT)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _bind_app(monkeypatch: pytest.MonkeyPatch, app: Path) -> None:
    real = _mod.Path

    def fake(value: str, *args: Any, **kwargs: Any) -> Path:
        if value == "/app":
            return app
        return real(value, *args, **kwargs)

    monkeypatch.setattr(_mod, "Path", fake)


def _run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], roots: list[Path]) -> tuple[int, str]:
    monkeypatch.setattr(_mod, "scan_roots", lambda: roots)
    code = _mod.main()
    return code, capsys.readouterr().out


def _reported(out: str) -> list[str]:
    marker = "credential-shaped path(s) in the image:"
    assert marker in out, out
    tail = out.split(marker, 1)[1]
    return [line.strip() for line in tail.splitlines() if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Filename contract: every glob is a refusal, and a template suffix beats the glob
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestIsBadFile:
    @pytest.mark.parametrize(
        "name",
        [
            ".env",
            ".env.local",
            ".env.production",
            "server.key",
            "cert.pem",
            "store.p12",
            "store.pfx",
            "trust.jks",
            "app.keystore",
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
            # "example" in the middle is not a template. The allow-list is a suffix.
            ".env.example.bak",
        ],
    )
    def test_credential_shaped_names_are_refused(self, name: str) -> None:
        assert _mod.is_bad_file(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            ".env.example",
            ".env.sample",
            ".env.template",
            ".env.dist",
            "server.key.example",
            "credentials.template",
            "id_rsa.dist",
            "cert.pem.sample",
            "README.md",
            "notes.env",
            # ``credentials`` is a whole-name glob, not a substring.
            "mycredentials",
            "credentials_old",
        ],
    )
    def test_templates_and_ordinary_names_are_kept(self, name: str) -> None:
        assert _mod.is_bad_file(name) is False


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Roots: /app plus installed Juniper trees. A scan of /app alone is the vacuous pass.
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestScanRoots:
    def test_juniper_trees_are_roots_and_unrelated_packages_are_not(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        app = tmp_path / "app"
        app.mkdir()
        site = tmp_path / "site"
        site.mkdir()
        for name in (
            "candidate_unit",
            "cascade_correlation",
            "juniper_ml",
            "juniper_data-0.16.0.dist-info",
            "numpy",
            "JuniperData",
        ):
            (site / name).mkdir()
        (site / "juniper_extra").write_text("not a package", encoding="utf-8")

        _bind_app(monkeypatch, app)
        monkeypatch.setattr(_mod.sysconfig, "get_paths", lambda: {"purelib": str(site)})

        roots = _mod.scan_roots()
        assert roots[0] == app
        assert [p.name for p in roots[1:]] == [
            "candidate_unit",
            "cascade_correlation",
            "juniper_data-0.16.0.dist-info",
            "juniper_ml",
        ]

    def test_a_missing_app_and_a_missing_purelib_scan_nothing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _bind_app(monkeypatch, tmp_path / "absent")
        monkeypatch.setattr(_mod.sysconfig, "get_paths", lambda: {})
        assert _mod.scan_roots() == []

    def test_an_empty_or_non_directory_purelib_adds_no_site_root(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        app = tmp_path / "app"
        app.mkdir()
        purelib = tmp_path / "purelib-is-a-file"
        purelib.write_text("x", encoding="utf-8")
        _bind_app(monkeypatch, app)
        monkeypatch.setattr(_mod.sysconfig, "get_paths", lambda: {"purelib": str(purelib)})
        assert _mod.scan_roots() == [app]

        monkeypatch.setattr(_mod.sysconfig, "get_paths", lambda: {"purelib": ""})
        assert _mod.scan_roots() == [app]

    def test_a_file_at_app_is_not_a_root(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        app = tmp_path / "app"
        app.write_text("x", encoding="utf-8")
        _bind_app(monkeypatch, app)
        monkeypatch.setattr(_mod.sysconfig, "get_paths", lambda: {})
        assert _mod.scan_roots() == []


# ─────────────────────────────────────────────────────────────────────────────────────────────
# The verdict. Exit 2 is an invalid scan, 1 is a credential, 0 is a real clean tree.
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestMain:
    def test_no_root_is_a_failure_not_a_pass(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        code, out = _run(monkeypatch, capsys, [])
        assert code == 2
        assert "no scan root found" in out
        assert "inspected NOTHING" in out
        assert "no credential-shaped file" not in out

    def test_a_root_that_walks_zero_files_is_a_failure(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        (tmp_path / "empty-child").mkdir()
        code, out = _run(monkeypatch, capsys, [tmp_path])
        assert code == 2
        assert "ZERO files" in out
        assert "proved nothing" in out
        assert "no credential-shaped file" not in out

    def test_an_empty_secrets_directory_cannot_pass(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        """Zero files short-circuits before the finding list; the exit is still a refusal."""
        (tmp_path / "secrets").mkdir()
        code, out = _run(monkeypatch, capsys, [tmp_path])
        assert code != 0
        assert "no credential-shaped file" not in out

    def test_a_secrets_directory_with_any_file_is_reported(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        secrets = tmp_path / "pkg" / "secrets"
        secrets.mkdir(parents=True)
        (secrets / "README.md").write_text("benign", encoding="utf-8")
        code, out = _run(monkeypatch, capsys, [tmp_path])
        assert code == 1
        assert _reported(out) == [f"{secrets}/  (directory)"]
        assert "no credential-shaped file" not in out

    def test_a_credential_file_and_its_template_neighbour(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        env = tmp_path / "nested" / ".env"
        env.parent.mkdir()
        env.write_text("SECRET=1", encoding="utf-8")
        (tmp_path / ".env.example").write_text("SECRET=", encoding="utf-8")
        code, out = _run(monkeypatch, capsys, [tmp_path])
        assert code == 1
        assert _reported(out) == [str(env)]
        assert ".env.example" not in out.split("credential-shaped", 1)[1]

    @pytest.mark.parametrize("name", [".env", "cert.pem", "id_rsa"])
    def test_each_planted_credential_exits_1_without_echoing_its_contents(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, name: str) -> None:
        planted = tmp_path / "juniper_data" / name
        planted.parent.mkdir()
        planted.write_text("AWS_SECRET_ACCESS_KEY=supersecretvalue", encoding="utf-8")
        code, out = _run(monkeypatch, capsys, [tmp_path])
        assert code == 1
        assert str(planted) in out
        assert "supersecretvalue" not in out

    @pytest.mark.parametrize("dirname", sorted({"secrets", ".git", ".ssh", ".aws", ".gnupg", "private"}))
    def test_each_forbidden_directory_is_reported_even_when_its_files_look_benign(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, dirname: str) -> None:
        (tmp_path / dirname).mkdir()
        (tmp_path / dirname / "readme.txt").write_text("not a matching filename", encoding="utf-8")
        code, out = _run(monkeypatch, capsys, [tmp_path])
        assert code == 1
        assert _reported(out) == [f"{tmp_path / dirname}/  (directory)"]

    def test_a_symlinked_env_file_is_judged_by_its_name(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        target = tmp_path / "real.txt"
        target.write_text("supersecretvalue", encoding="utf-8")
        root = tmp_path / "root"
        root.mkdir()
        (root / ".env").symlink_to(target)
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 1
        assert _reported(out) == [str(root / ".env")]
        assert "supersecretvalue" not in out

    def test_a_symlinked_secrets_directory_is_flagged_without_being_followed(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "only-inside-link.pem").write_text("supersecretvalue", encoding="utf-8")
        root = tmp_path / "root"
        root.mkdir()
        (root / "generator.py").write_text("ok", encoding="utf-8")
        (root / "secrets").symlink_to(outside, target_is_directory=True)
        code, out = _run(monkeypatch, capsys, [root])
        assert code == 1
        assert _reported(out) == [f"{root / 'secrets'}/  (directory)"]
        assert "only-inside-link.pem" not in out

    def test_findings_from_every_root_are_sorted(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        first = tmp_path / "site"
        second = tmp_path / "app"
        (first / "b.pem").parent.mkdir()
        (first / "b.pem").write_text("x", encoding="utf-8")
        (second).mkdir()
        (second / "a.key").write_text("x", encoding="utf-8")
        (second / ".ssh").mkdir()
        code, out = _run(monkeypatch, capsys, [first, second])
        assert code == 1
        assert "scanned 2 files across 2 root(s):" in out
        assert _reported(out) == sorted(
            [
                str(first / "b.pem"),
                str(second / "a.key"),
                f"{second / '.ssh'}/  (directory)",
            ]
        )

    def test_a_clean_tree_passes_and_counts_the_files_it_walked(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        (tmp_path / "juniper_data").mkdir()
        (tmp_path / "juniper_data" / "generator.py").write_text("x", encoding="utf-8")
        (tmp_path / ".env.example").write_text("x", encoding="utf-8")
        code, out = _run(monkeypatch, capsys, [tmp_path])
        assert code == 0
        assert "scanned 2 files across 1 root(s):" in out
        assert str(tmp_path) in out
        assert "no credential-shaped file in any shipped tree" in out

    @pytest.mark.parametrize("prune", ["__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules"])
    def test_a_cache_directory_is_not_walked_and_its_sibling_still_is(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, prune: str) -> None:
        cache = tmp_path / prune
        cache.mkdir()
        (cache / ".env").write_text("SECRET=1", encoding="utf-8")
        (cache / "secrets").mkdir()
        (tmp_path / "ok.py").write_text("x", encoding="utf-8")
        code, out = _run(monkeypatch, capsys, [tmp_path])
        assert code == 0
        assert "scanned 1 files across 1 root(s):" in out
        assert ".env" not in out
        assert "secrets" not in out

    def test_pruning_a_cache_does_not_hide_a_sibling_credential(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("supersecretvalue", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.pyc").write_bytes(b"\x00")
        code, out = _run(monkeypatch, capsys, [tmp_path])
        assert code == 1
        assert _reported(out) == [str(tmp_path / ".env")]
        assert "supersecretvalue" not in out


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Wiring: the check runs on the PR arm and against each pushed digest, before export
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))
    return data


def _step(job: str, name: str) -> dict[str, Any]:
    matches = [s for s in _workflow()["jobs"][job]["steps"] if s.get("name") == name]
    assert len(matches) == 1, name
    return matches[0]


class TestPublishWorkflowRunsTheSecretCheck:
    def test_paths_filter_covers_the_script(self) -> None:
        assert SCRIPT_REL in _workflow()["on"]["pull_request"]["paths"]

    def test_the_pr_arm_pipes_the_script_into_the_image_it_built(self) -> None:
        step = _step("build", "Smoke test (build-only runs)")
        assert step["if"] == BUILD_ONLY_IF
        assert f"python - < {SCRIPT_REL}" in step["run"]
        assert "data-smoke:${{ matrix.arch }}" in step["run"]

    def test_the_publish_path_pipes_it_at_the_pushed_digest_before_export(self) -> None:
        names = [str(s.get("name", "")) for s in _workflow()["jobs"]["build"]["steps"]]
        verify = names.index("Verify pushed image is CPU-only (publish runs)")
        export = names.index("Export digest")
        assert verify < export
        step = _step("build", "Verify pushed image is CPU-only (publish runs)")
        assert step["if"] == PUBLISH_IF
        assert f"python - < {SCRIPT_REL}" in step["run"]
        assert "@${digest}" in step["run"]
