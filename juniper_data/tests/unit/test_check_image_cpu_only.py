"""Pin ``util/check_image_cpu_only.py``.

A version string of ``+cpu`` is not a CPU-only image: the 2026-09-07 worker image
kept ``nvidia-*``, ``cuda-*`` and ``triton`` beside a CPU wheel installed last.
This image never ships torch, so ``EXPECT_TORCH=absent`` must also fail when torch
is merely importable. A malformed expectation is a usage error and must not scan.

These tests need no Docker and no torch. Distribution metadata and the torch
module are fakes.

Consolidated from juniper-data #450 and #449 (Cursor fleet): #450's suite, plus #449's
check that the manifest job re-checks the tag reference it publishes.
"""

from __future__ import annotations

import builtins
import importlib.util
import platform
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "util" / "check_image_cpu_only.py"
WORKFLOW = REPO / ".github" / "workflows" / "publish-image.yml"
SCRIPT_REL = "util/check_image_cpu_only.py"
PUBLISH_IF = "github.event_name == 'release' || inputs.push"
BUILD_ONLY_IF = "github.event_name != 'release' && !inputs.push"

_spec = importlib.util.spec_from_file_location("check_image_cpu_only", SCRIPT)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class _Dist:
    def __init__(self, name: str | None) -> None:
        self.metadata = {"Name": name}


def _torch(version: str, cuda: str | None, *, has_cuda_attr: bool = True) -> types.SimpleNamespace:
    module = types.SimpleNamespace()
    module.__version__ = version
    module.version = types.SimpleNamespace(cuda=cuda) if has_cuda_attr else types.SimpleNamespace()
    return module


def _install_torch(monkeypatch: pytest.MonkeyPatch, module: types.SimpleNamespace | None) -> None:
    if module is None:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        real_import = builtins.__import__

        def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        return
    monkeypatch.setitem(sys.modules, "torch", module)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Names: PEP 503 folding, and the three CUDA families (nvidia-*, cuda-*, triton)
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestDistributionNames:
    @pytest.mark.parametrize(
        ("raw", "normalised"),
        [
            ("nvidia_cublas", "nvidia-cublas"),
            ("NVIDIA.cuBLAS", "nvidia-cublas"),
            ("cuda__toolkit", "cuda-toolkit"),
            ("triton", "triton"),
        ],
    )
    def test_pep503_normalisation_folds_separators(self, raw: str, normalised: str) -> None:
        assert _mod._normalise(raw) == normalised

    def test_installed_distributions_drop_blank_names_and_fold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _mod.importlib.metadata,
            "distributions",
            lambda: [_Dist("nvidia_cublas"), _Dist("NVIDIA.cuBLAS"), _Dist(""), _Dist(None), _Dist("FastAPI")],
        )
        assert _mod.installed_distributions() == {"nvidia-cublas", "fastapi"}

    def test_forbidden_names_are_the_cuda_stack_only(self) -> None:
        names = {
            "nvidia-cublas",
            "cuda-toolkit",
            "cuda-bindings",
            "triton",
            "torch",
            "numpy",
            "fastapi",
        }
        assert _mod.forbidden_distributions(names) == [
            "cuda-bindings",
            "cuda-toolkit",
            "nvidia-cublas",
            "triton",
        ]


# ─────────────────────────────────────────────────────────────────────────────────────────────
# The contract. Empty means pass. Offenders are reported even when the wheel says +cpu.
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestCheck:
    def test_absent_with_nothing_installed_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_mod, "torch_importable", lambda: False)
        assert _mod.check("absent", {"numpy", "fastapi"}) == []

    def test_absent_refuses_an_importable_torch_that_metadata_missed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_mod, "torch_importable", lambda: True)
        assert _mod.check("absent", set()) == ["torch is installed, but EXPECT_TORCH=absent"]

    def test_absent_refuses_torch_metadata_without_importing_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_mod, "torch_importable", lambda: False)
        _install_torch(monkeypatch, None)
        assert _mod.check("absent", {"torch"}) == ["torch is installed, but EXPECT_TORCH=absent"]

    def test_absent_reports_the_cuda_stack_and_torch_together(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_mod, "torch_importable", lambda: False)
        assert _mod.check("absent", {"triton", "nvidia-cublas", "torch"}) == [
            "CUDA/NVIDIA distributions are installed: nvidia-cublas, triton",
            "torch is installed, but EXPECT_TORCH=absent",
        ]

    def test_a_matching_cpu_wheel_with_no_cuda_attribute_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_torch(monkeypatch, _torch("2.12.0+cpu", None, has_cuda_attr=False))
        assert _mod.check("2.12.0+cpu", {"torch"}) == []

    def test_leftover_cuda_wheels_fail_a_cpu_version_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The vacuous fix: install the CPU wheel last and leave the CUDA stack behind."""
        _install_torch(monkeypatch, _torch("2.12.0+cpu", None))
        assert _mod.check("2.12.0+cpu", {"torch", "nvidia-cudnn-cu13", "triton"}) == [
            "CUDA/NVIDIA distributions are installed: nvidia-cudnn-cu13, triton",
        ]

    def test_a_cuda_build_reports_the_version_and_the_cuda_marker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_torch(monkeypatch, _torch("2.12.1+cu130", "13.0"))
        assert _mod.check("2.12.0+cpu", {"torch"}) == [
            "torch.__version__ is '2.12.1+cu130', expected '2.12.0+cpu'",
            "torch.version.cuda is '13.0'; a CPU build reports None",
        ]

    def test_a_missing_torch_stops_at_the_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_torch(monkeypatch, None)
        problems = _mod.check("2.12.0+cpu", {"cuda-toolkit"})
        assert problems == [
            "CUDA/NVIDIA distributions are installed: cuda-toolkit",
            "torch failed to import: no torch",
        ]


# ─────────────────────────────────────────────────────────────────────────────────────────────
# main(): usage errors scan nothing; a real census is printed on both pass and fail
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestMain:
    def _names(self, monkeypatch: pytest.MonkeyPatch, names: set[str], importable: bool) -> None:
        monkeypatch.setattr(_mod, "installed_distributions", lambda: set(names))
        monkeypatch.setattr(_mod, "torch_importable", lambda: importable)

    @pytest.mark.parametrize(
        "value",
        ["", "2.12.0", "2.12.0+cu130", "2.12+cpu", "2.12.0+CPU", "2.12.0+cpu-extra", "latest"],
    )
    def test_a_malformed_expectation_is_a_usage_error_and_does_not_scan(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str) -> None:
        def boom() -> set[str]:
            raise AssertionError("a usage error must not scan distributions")

        monkeypatch.setattr(_mod, "installed_distributions", boom)
        monkeypatch.setenv("EXPECT_TORCH", value)
        assert _mod.main() == 2
        err = capsys.readouterr().err
        assert f"got {value!r}" in err
        assert "CPU-only contract holds" not in err

    def test_an_unset_expectation_is_the_same_usage_error(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.delenv("EXPECT_TORCH", raising=False)
        assert _mod.main() == 2
        assert "got ''" in capsys.readouterr().err

    def test_whitespace_around_absent_is_stripped(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        self._names(monkeypatch, {"numpy", "fastapi"}, importable=False)
        monkeypatch.setenv("EXPECT_TORCH", "  absent\n")
        assert _mod.main() == 0
        out, err = capsys.readouterr()
        assert err == ""
        assert "torch=absent" in out
        assert "distributions=2" in out
        assert "cuda_stack=0" in out
        assert "expect=absent" in out
        assert f"machine={platform.machine()}" in out
        assert f"python={platform.python_version()}" in out
        assert "CPU-only contract holds" in out

    def test_a_padded_cpu_pin_is_checked_against_the_stripped_value(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        self._names(monkeypatch, {"torch"}, importable=True)
        _install_torch(monkeypatch, _torch("2.12.0+cpu", None))
        monkeypatch.setenv("EXPECT_TORCH", "  2.12.0+cpu  ")
        assert _mod.main() == 0
        out = capsys.readouterr().out
        assert "torch=2.12.0+cpu (cuda=None)" in out
        assert "expect=2.12.0+cpu" in out
        assert "CPU-only contract holds" in out

    def test_absent_fails_closed_when_torch_is_installed(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        self._names(monkeypatch, {"torch", "triton"}, importable=False)
        monkeypatch.setenv("EXPECT_TORCH", "absent")
        assert _mod.main() == 1
        out, err = capsys.readouterr()
        assert "distributions=2" in out
        assert "cuda_stack=1" in out
        assert "::error::CUDA/NVIDIA distributions are installed: triton" in out
        assert "::error::torch is installed, but EXPECT_TORCH=absent" in out
        assert "CPU-only contract holds" not in out
        assert "CPU-only contract VIOLATED" in err

    def test_a_cpu_pin_with_no_torch_is_a_violation(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        self._names(monkeypatch, {"numpy"}, importable=False)
        _install_torch(monkeypatch, None)
        monkeypatch.setenv("EXPECT_TORCH", "2.12.0+cpu")
        assert _mod.main() == 1
        out, err = capsys.readouterr()
        assert "torch=absent" in out
        assert "::error::torch failed to import: no torch" in out
        assert "CPU-only contract VIOLATED" in err


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))
    return data


def _step(job: str, name: str) -> dict[str, Any]:
    matches = [s for s in _workflow()["jobs"][job]["steps"] if s.get("name") == name]
    assert len(matches) == 1, name
    return matches[0]


class TestPublishWorkflowRunsTheCpuCheck:
    def test_paths_filter_covers_the_script(self) -> None:
        assert SCRIPT_REL in _workflow()["on"]["pull_request"]["paths"]

    def test_provenance_declares_this_image_has_no_torch(self) -> None:
        step = _step("build", "Resolve build provenance")
        assert "expect_torch=absent" in step["run"]

    @pytest.mark.parametrize(
        ("name", "condition"),
        [
            ("Smoke test (build-only runs)", BUILD_ONLY_IF),
            ("Verify pushed image is CPU-only (publish runs)", PUBLISH_IF),
        ],
    )
    def test_both_build_arms_pass_the_provenance_expectation(self, name: str, condition: str) -> None:
        step = _step("build", name)
        assert step["if"] == condition
        assert "EXPECT_TORCH='${{ steps.prov.outputs.expect_torch }}'" in step["run"]
        assert f"python - < {SCRIPT_REL}" in step["run"]

    def test_the_manifest_job_rechecks_the_published_image_as_absent(self) -> None:
        step = _step("merge", "Verify published image is CPU-only")
        assert "EXPECT_TORCH=absent" in step["run"]
        assert f"python - < {SCRIPT_REL}" in step["run"]
        assert "EXPECT_TORCH='${{ steps.prov.outputs.expect_torch }}'" not in step["run"]
        # The tag a consumer pulls, not a digest or a local build, is what gets re-checked.
        assert '"${ref}" python - < ' + SCRIPT_REL in step["run"]
