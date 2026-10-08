"""Drive ``util/check_image_serves.py`` the way CI does: through ``main``, with Docker scripted.

``test_check_image_serves.py`` owns the pure ``evaluate`` verdict and the workflow wiring. It never
starts the container, so it cannot see a driver that overrides the image entrypoint, parses the
wrong probe line, treats a dead container as alive, or skips ``docker rm``. These tests script
``_docker`` (and, for the two process-boundary cases, ``subprocess.run``). No Docker daemon.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess  # nosec B404 - only referenced to patch subprocess.run
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.unit]

SCRIPT = Path(__file__).resolve().parents[3] / "util" / "check_image_serves.py"
IMAGE = "ghcr.io/pcalnon/juniper-data@sha256:abc"
VERSION = "0.17.0"
CID = "cid-deadbeef"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_image_serves_driver", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe_json(version: str = VERSION, *, module_version: str | None = VERSION, module_error: str | None = None) -> str:
    return json.dumps({"metadata": version, "module_version": module_version, "module_error": module_error})


def _health_body(version: str | None = VERSION, *, extra: dict[str, Any] | None = None) -> str:
    body: dict[str, Any] = {"status": "ok"}
    if version is not None:
        body["version"] = version
    if extra:
        body.update(extra)
    return "200\n" + json.dumps(body)


def _argv(*extra: str, version: str = VERSION, module: str | None = "juniper_data") -> list[str]:
    args = ["--image", IMAGE, "--dist", "juniper-data", "--port", "8100", "--expect-version", version]
    if module is not None:
        args.extend(["--module", module])
    args.extend(extra)
    return args


class _Clock:
    """Advances only when the driver sleeps, so a poll loop cannot spin the test."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds
        if self.t > 1000:
            raise AssertionError("the liveness poll did not honor its deadline")


class _Docker:
    def __init__(self, handler: Any) -> None:
        self.calls: list[tuple[list[str], int]] = []
        self._handler = handler

    def __call__(self, args: list[str], timeout: int = 60) -> tuple[int, str]:
        recorded = (list(args), timeout)
        self.calls.append(recorded)
        return self._handler(list(args), timeout)

    def named(self, command: str) -> list[tuple[list[str], int]]:
        return [call for call in self.calls if call[0][0] == command]


def _install(monkeypatch: pytest.MonkeyPatch, handler: Any) -> tuple[Any, _Docker, _Clock]:
    module = _load()
    docker = _Docker(handler)
    clock = _Clock()
    monkeypatch.setattr(module, "_docker", docker)
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    return module, docker, clock


def _serving(
    probe: tuple[int, str] = (0, ""),
    *,
    health: str = "",
    inspect: str = "true",
    start: tuple[int, str] = (0, f"Status: pulled\n{CID}"),
    rm: tuple[int, str] = (0, ""),
    execs: list[tuple[int, str]] | None = None,
) -> Any:
    """A container that stays up. ``execs`` is consumed in order; otherwise every GET is ``health``."""
    queue = list(execs) if execs is not None else None

    def handler(args: list[str], timeout: int) -> tuple[int, str]:
        if args[0] == "run" and "--entrypoint" in args:
            return probe
        if args[:2] == ["run", "-d"]:
            return start
        if args[0] == "inspect":
            return 0, inspect
        if args[0] == "exec":
            if queue is not None:
                return queue.pop(0)
            return 0, health
        if args[0] == "rm":
            return rm
        if args[0] == "logs":
            return 0, "booted"
        raise AssertionError(f"unexpected docker argv: {args}")

    return handler


def _started(docker: _Docker) -> list[tuple[list[str], int]]:
    return [call for call in docker.calls if call[0][:2] == ["run", "-d"]]


def _removed(docker: _Docker) -> list[str]:
    return [call[0][2] for call in docker.named("rm")]


# ─────────────────────────────────────────────────────────────────────────────────────────────
# The version probe decides whether the image is started at all
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestVersionProbeDoesNotStartABadImage:
    @pytest.mark.parametrize(
        ("code", "out"),
        [
            (1, _probe_json()),
            (0, ""),
            (0, "not-json"),
            (0, _probe_json() + "\ntrailing junk"),
            (124, "docker run timed out after 180s"),
        ],
    )
    def test_an_untrustworthy_probe_starts_nothing(self, monkeypatch: pytest.MonkeyPatch, code: int, out: str) -> None:
        module, docker, _clock = _install(monkeypatch, _serving(probe=(code, out)))
        assert module.main(_argv()) == 2
        assert _started(docker) == []
        assert docker.named("rm") == []

    def test_a_banner_before_the_json_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module, docker, clock = _install(monkeypatch, _serving(probe=(0, "WARNING: old manifest\n" + _probe_json()), health=_health_body()))
        assert module.main(_argv()) == 0
        assert _started(docker) != []
        assert clock.sleeps == []

    def test_stderr_is_appended_after_stdout_so_a_warning_is_the_last_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load()

        class _Proc:
            returncode = 0
            stdout = _probe_json() + "\n"
            stderr = "warning from the daemon\n"

        monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: _Proc())
        code, out = module._docker(["run", "--rm", IMAGE])
        assert code == 0
        assert out.splitlines()[-1] == "warning from the daemon"

    def test_a_docker_timeout_is_exit_124_and_names_the_subcommand(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _load()

        def _expire(*args: Any, **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd=["docker"], timeout=kwargs.get("timeout", 60))

        monkeypatch.setattr(module.subprocess, "run", _expire)
        code, out = module._docker(["inspect", CID], timeout=20)
        assert code == 124
        assert out == "docker inspect timed out after 20s"


# ─────────────────────────────────────────────────────────────────────────────────────────────
# The serve container is the image's own entrypoint; the id is the last line; it is always removed
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestServeContainer:
    def test_run_detached_keeps_the_image_entrypoint_and_the_id_is_the_last_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module, docker, _clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), health=_health_body()))
        assert module.main(_argv()) == 0
        probe, probe_timeout = docker.calls[0]
        assert probe == ["run", "--rm", "--entrypoint", "python", IMAGE, "-c", module._VERSION_PROBE, "juniper-data", "juniper_data"]
        assert probe_timeout == 180
        serve, serve_timeout = _started(docker)[0]
        assert serve == ["run", "-d", IMAGE]
        assert serve_timeout == 120
        assert "--entrypoint" not in serve
        inspect, inspect_timeout = docker.named("inspect")[0]
        assert inspect == ["inspect", "-f", "{{.State.Running}}", CID]
        assert inspect_timeout == 20
        health, health_timeout = docker.named("exec")[0]
        assert health[-2:] == ["8100", "/v1/health"]
        assert health_timeout == 20
        removed = docker.named("rm")
        assert [call[0] for call in removed] == [["rm", "-f", CID]]
        assert removed[0][1] == 60
        assert docker.named("logs") == []

    def test_an_image_that_does_not_start_is_exit_1_and_is_not_removed(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        module, docker, _clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), start=(1, "Error: pull access denied")))
        assert module.main(_argv()) == 1
        assert "did not start" in capsys.readouterr().err
        assert docker.named("rm") == []
        assert docker.named("exec") == []

    @pytest.mark.parametrize("state", ["false", "True", "", "yes", "docker inspect timed out after 20s"])
    def test_inspect_other_than_true_fails_closed_and_still_removes(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], state: str) -> None:
        module, docker, _clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), inspect=state, health=_health_body()))
        assert module.main(_argv()) == 1
        assert "exited before liveness" in capsys.readouterr().err
        assert docker.named("exec") == []
        assert _removed(docker) == [CID]
        assert docker.named("logs") != []

    def test_a_stale_version_fails_and_still_removes(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        probe = _probe_json(module_version="0.16.0")
        module, docker, _clock = _install(monkeypatch, _serving(probe=(0, probe), health=_health_body()))
        assert module.main(_argv()) == 1
        assert "__version__ 0.16.0 != metadata 0.17.0" in capsys.readouterr().err
        assert _removed(docker) == [CID]

    def test_an_import_error_is_named_and_the_container_is_removed(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        probe = _probe_json(module_version=None, module_error="ModuleNotFoundError: No module named 'juniper_data'")
        module, docker, _clock = _install(monkeypatch, _serving(probe=(0, probe), health=_health_body()))
        assert module.main(_argv()) == 1
        err = capsys.readouterr().err
        assert "does not import" in err
        assert "ModuleNotFoundError" in err
        assert _removed(docker) == [CID]

    def test_a_failed_rm_does_not_hide_a_pass(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        module, docker, _clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), health=_health_body(), rm=(1, "device or resource busy")))
        assert module.main(_argv()) == 0
        assert f"reports {VERSION}" in capsys.readouterr().out
        assert _removed(docker) == [CID]

    def test_liveness_that_never_arrives_still_removes_the_container(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        module, docker, clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), health="nope"))
        assert module.main(_argv("--timeout", "3")) == 1
        assert "liveness answered None" in capsys.readouterr().err
        assert clock.sleeps == [3]
        assert _removed(docker) == [CID]
        assert docker.named("logs") != []

    @pytest.mark.parametrize("version", ["0.17.0+local", "0.17.0.dev1", "0.17.0-rc.1"])
    def test_a_prerelease_or_local_version_is_not_a_usage_error(self, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
        module, docker, _clock = _install(monkeypatch, _serving(probe=(0, _probe_json(version, module_version=version)), health=_health_body(version)))
        assert module.main(_argv(version=version)) == 0
        assert _started(docker) != []

    def test_an_omitted_module_does_not_require_a_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probe = _probe_json(module_version=None)
        module, docker, _clock = _install(monkeypatch, _serving(probe=(0, probe), health=_health_body()))
        assert module.main(_argv(module=None)) == 0
        probe_argv = docker.calls[0][0]
        assert probe_argv[-1] == ""


# ─────────────────────────────────────────────────────────────────────────────────────────────
# What an in-container GET counts as, and which version field is required
# ─────────────────────────────────────────────────────────────────────────────────────────────
class TestLivenessProbe:
    def test_an_unreadable_status_line_is_not_liveness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module, docker, clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), execs=[(0, "nope"), (0, _health_body())]))
        assert module.main(_argv("--timeout", "30")) == 0
        assert len(docker.named("exec")) == 2
        assert clock.sleeps == [3]

    def test_a_nonzero_exec_is_not_an_http_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed exec that prints ``200`` and a stale version must not end the poll."""
        stale = (1, _health_body("0.16.0"))
        module, docker, _clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), execs=[stale, (0, _health_body())]))
        assert module.main(_argv("--timeout", "30")) == 0
        assert len(docker.named("exec")) == 2

    def test_an_empty_probe_is_not_liveness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module, docker, _clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), execs=[(0, ""), (0, _health_body())]))
        assert module.main(_argv("--timeout", "30")) == 0
        assert len(docker.named("exec")) == 2

    def test_a_non_json_200_body_is_not_a_version(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        module, docker, clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), health="200\nOK"))
        assert module.main(_argv()) == 1
        assert "carries no version field" in capsys.readouterr().err
        assert len(docker.named("exec")) == 1
        assert clock.sleeps == []

    def test_the_default_health_version_is_required(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        module, _docker, _clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), health=_health_body(version=None)))
        assert module.main(_argv()) == 1
        assert "carries no version field" in capsys.readouterr().err

    def test_optional_health_version_accepts_an_absent_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module, _docker, _clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), health=_health_body(version=None)))
        assert module.main(_argv("--health-version", "optional")) == 0

    def test_optional_health_version_rejects_a_present_mismatch(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        module, _docker, _clock = _install(monkeypatch, _serving(probe=(0, _probe_json()), health=_health_body("0.16.0")))
        assert module.main(_argv("--health-version", "optional")) == 1
        assert "reports version 0.16.0 != metadata 0.17.0" in capsys.readouterr().err

    def test_every_enveloped_path_is_probed_in_cli_order_and_reported(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        def handler(args: list[str], timeout: int) -> tuple[int, str]:
            if args[0] == "run" and "--entrypoint" in args:
                return 0, _probe_json()
            if args[:2] == ["run", "-d"]:
                return 0, CID
            if args[0] == "inspect":
                return 0, "true"
            if args[0] == "exec":
                path = args[-1]
                if path == "/v1/health":
                    return 0, _health_body()
                if path == "/v1/z":
                    return 0, "200\n" + json.dumps({"meta": {"version": "0.1.0"}})
                if path == "/v1/a":
                    return 0, "404\n" + json.dumps({"detail": "missing"})
                raise AssertionError(path)
            if args[0] == "rm":
                return 0, ""
            if args[0] == "logs":
                return 0, ""
            raise AssertionError(args)

        module, docker, _clock = _install(monkeypatch, handler)
        assert module.main(_argv("--enveloped-path", "/v1/z", "--enveloped-path", "/v1/a")) == 1
        paths = [call[0][-1] for call in docker.named("exec")]
        assert paths == ["/v1/health", "/v1/z", "/v1/a"]
        err = capsys.readouterr().err
        assert "/v1/a answered 404, not 200" in err
        assert "/v1/z meta.version 0.1.0 != metadata 0.17.0" in err
        assert _removed(docker) == [CID]
