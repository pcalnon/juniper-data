"""Pin ``util/check_image_cpu_only.py`` and its wiring into ``publish-image.yml``.

The script is what makes "CPU-only" mean the image, not the version string. The 2026-09-07
worker image shipped ``torch 2.12.1+cu130`` plus the NVIDIA stack because a lockfile install
re-resolved torch from PyPI. A version assertion alone misses the vacuous fix: installing the
CPU wheel last replaces ``torch`` and leaves every orphaned ``nvidia-*`` / ``triton`` wheel
behind, so ``__version__`` reads ``+cpu`` while the image is still the CUDA stack.

juniper-data's image never ships torch, so the workflow passes ``EXPECT_TORCH=absent``. The
script had no tests. These drive ``check`` and ``main`` with a fake distribution census and a
fake ``torch`` module. They need no Docker and do not import a real torch.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "publish-image.yml"
SCRIPT = REPO / "util" / "check_image_cpu_only.py"
SCRIPT_REL = "util/check_image_cpu_only.py"
PUBLISH_IF = "github.event_name == 'release' || inputs.push"
BUILD_ONLY_IF = "github.event_name != 'release' && !inputs.push"
EXPECT = "2.12.0+cpu"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_image_cpu_only", SCRIPT)
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


def _install_torch(monkeypatch: pytest.MonkeyPatch, version: str = EXPECT, cuda: Any = None) -> None:
    fake = types.ModuleType("torch")
    fake.__version__ = version  # type: ignore[attr-defined]
    fake.version = types.SimpleNamespace(cuda=cuda)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)


def _block_torch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import torch`` raises ImportError when the module cache holds None."""
    monkeypatch.setitem(sys.modules, "torch", None)


class _Dist:
    def __init__(self, name: Any) -> None:
        self.metadata = {"Name": name}


class TestCensus:
    def test_pep503_normalisation_equates_separators_and_case(self) -> None:
        assert _load()._normalise("NVIDIA_Cublas.Cu13") == "nvidia-cublas-cu13"

    def test_installed_names_are_normalised_and_blank_names_are_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load()
        monkeypatch.setattr(
            module.importlib.metadata,
            "distributions",
            lambda: [_Dist("nvidia_cublas"), _Dist(""), _Dist(None), _Dist("Triton"), _Dist("numpy")],
        )
        assert module.installed_distributions() == {"nvidia-cublas", "triton", "numpy"}

    def test_the_three_cuda_families_are_the_census(self) -> None:
        offenders = _load().forbidden_distributions({"triton", "cuda-toolkit", "nvidia-cublas", "numpy", "pydantic"})
        assert offenders == ["cuda-toolkit", "nvidia-cublas", "triton"]

    def test_an_underscore_nvidia_name_is_still_the_cuda_stack(self) -> None:
        module = _load()
        assert module.forbidden_distributions({module._normalise("nvidia_cudnn_cu13")}) == ["nvidia-cudnn-cu13"]


class TestCheck:
    def test_absent_with_no_torch_and_no_cuda_stack_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load()
        monkeypatch.setattr(module, "torch_importable", lambda: False)
        assert module.check("absent", {"numpy", "pydantic"}) == []

    def test_absent_rejects_a_torch_distribution_that_does_not_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load()
        monkeypatch.setattr(module, "torch_importable", lambda: False)
        problems = module.check("absent", {"torch"})
        assert problems == ["torch is installed, but EXPECT_TORCH=absent"]

    def test_absent_rejects_an_importable_torch_missing_from_the_census(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load()
        monkeypatch.setattr(module, "torch_importable", lambda: True)
        problems = module.check("absent", set())
        assert problems == ["torch is installed, but EXPECT_TORCH=absent"]

    def test_absent_reports_the_cuda_stack_and_torch_together(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load()
        monkeypatch.setattr(module, "torch_importable", lambda: False)
        problems = module.check("absent", {"torch", "nvidia-cublas", "triton"})
        assert problems[0] == "CUDA/NVIDIA distributions are installed: nvidia-cublas, triton"
        assert problems[1] == "torch is installed, but EXPECT_TORCH=absent"

    def test_a_matching_cpu_build_with_cuda_none_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load()
        _install_torch(monkeypatch)
        assert module.check(EXPECT, {"torch", "numpy"}) == []

    def test_a_missing_cuda_attribute_is_treated_as_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load()
        fake = types.ModuleType("torch")
        fake.__version__ = EXPECT  # type: ignore[attr-defined]
        fake.version = types.SimpleNamespace()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert module.check(EXPECT, {"torch"}) == []

    def test_the_cu130_version_fails_the_cpu_pin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load()
        _install_torch(monkeypatch, version="2.12.1+cu130", cuda="13.0")
        problems = module.check(EXPECT, {"torch"})
        assert any("2.12.1+cu130" in problem and EXPECT in problem for problem in problems), problems
        assert any("torch.version.cuda is '13.0'" in problem for problem in problems), problems

    def test_orphaned_nvidia_wheels_fail_even_when_the_version_string_says_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The vacuous fix: CPU torch installed last, CUDA wheels left behind."""
        module = _load()
        _install_torch(monkeypatch, cuda=None)
        problems = module.check(EXPECT, {"torch", "nvidia-cublas", "nvidia-cudnn-cu13", "triton"})
        assert problems == ["CUDA/NVIDIA distributions are installed: nvidia-cublas, nvidia-cudnn-cu13, triton"]

    def test_a_torch_that_does_not_import_fails_the_version_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load()
        _block_torch_import(monkeypatch)
        problems = module.check(EXPECT, set())
        assert len(problems) == 1
        assert problems[0].startswith("torch failed to import:")


class TestMain:
    def _main(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], expect: str | None, names: set[str], *, importable: bool) -> tuple[int, str, str]:
        module = _load()
        if expect is None:
            monkeypatch.delenv("EXPECT_TORCH", raising=False)
        else:
            monkeypatch.setenv("EXPECT_TORCH", expect)
        monkeypatch.setattr(module, "installed_distributions", lambda: set(names))
        monkeypatch.setattr(module, "torch_importable", lambda: importable)
        code = module.main()
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    @pytest.mark.parametrize("value", ["", "2.12.0", "2.12.0+cu130", "2.12.0+CPU", "latest", "2.12+cpu"])
    def test_a_malformed_expect_torch_is_a_usage_error(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str) -> None:
        code, out, err = self._main(monkeypatch, capsys, value, set(), importable=False)
        assert code == 2
        assert "EXPECT_TORCH" in err
        assert repr(value) in err
        assert "CPU-only contract holds" not in out

    def test_an_unset_expect_torch_is_a_usage_error(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, err = self._main(monkeypatch, capsys, None, set(), importable=False)
        assert code == 2
        assert "got ''" in err
        assert "CPU-only contract holds" not in out

    def test_surrounding_whitespace_does_not_invalidate_a_real_contract(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, err = self._main(monkeypatch, capsys, "  absent\n", {"numpy"}, importable=False)
        assert code == 0, err
        assert "expect=absent" in out
        assert "CPU-only contract holds" in out

    def test_absent_passes_and_does_not_need_torch(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, err = self._main(monkeypatch, capsys, "absent", {"numpy"}, importable=False)
        assert code == 0, err
        assert "torch=absent" in out
        assert "cuda_stack=0" in out
        assert "CPU-only contract holds" in out
        assert err == ""

    def test_absent_with_torch_installed_is_a_violation(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, err = self._main(monkeypatch, capsys, "absent", {"torch"}, importable=False)
        assert code == 1
        assert "::error::torch is installed, but EXPECT_TORCH=absent" in out
        assert "CPU-only contract VIOLATED" in err
        assert "CPU-only contract holds" not in out

    def test_a_matching_cpu_build_passes_and_reports_the_version(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        _install_torch(monkeypatch)
        code, out, err = self._main(monkeypatch, capsys, EXPECT, {"torch", "numpy"}, importable=True)
        assert code == 0, err
        assert f"torch={EXPECT}" in out
        assert "cuda=None" in out
        assert "CPU-only contract holds" in out

    def test_orphaned_nvidia_wheels_violate_through_main(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        _install_torch(monkeypatch)
        code, out, err = self._main(monkeypatch, capsys, f"  {EXPECT}  ", {"torch", "nvidia-cublas"}, importable=True)
        assert code == 1
        assert "::error::CUDA/NVIDIA distributions are installed: nvidia-cublas" in out
        assert "CPU-only contract VIOLATED" in err
        assert "CPU-only contract holds" not in out


class TestPublishWorkflowRunsTheCpuCheck:
    def test_paths_filter_covers_the_script(self) -> None:
        assert SCRIPT_REL in _workflow()["on"]["pull_request"]["paths"]

    def test_provenance_selects_the_absent_contract(self) -> None:
        step = _step("build", "Resolve build provenance")
        assert 'echo "expect_torch=absent"' in step["run"]

    def test_the_smoke_arm_passes_that_contract_into_the_image(self) -> None:
        step = _step("build", "Smoke test (build-only runs)")
        assert step["if"] == BUILD_ONLY_IF
        assert "EXPECT_TORCH='${{ steps.prov.outputs.expect_torch }}'" in step["run"]
        assert f"python - < {SCRIPT_REL}" in step["run"]

    def test_the_publish_path_checks_each_pushed_digest_before_export(self) -> None:
        names = [str(step.get("name", "")) for step in _workflow()["jobs"]["build"]["steps"]]
        check = next(i for i, name in enumerate(names) if name.startswith("Verify pushed image is CPU-only"))
        export = next(i for i, name in enumerate(names) if name.startswith("Export digest"))
        assert check < export, "a CUDA image must fail the arch before its digest is exported"
        step = _step("build", "Verify pushed image is CPU-only")
        assert step["if"] == PUBLISH_IF
        assert "EXPECT_TORCH='${{ steps.prov.outputs.expect_torch }}'" in step["run"]
        assert f"python - < {SCRIPT_REL}" in step["run"]
        assert "@${digest}" in step["run"]

    def test_the_tag_a_consumer_pulls_is_checked_again_as_absent(self) -> None:
        step = _step("merge", "Verify published image is CPU-only")
        assert "EXPECT_TORCH=absent" in step["run"]
        assert f"python - < {SCRIPT_REL}" in step["run"]
        assert '"${ref}"' in step["run"]
