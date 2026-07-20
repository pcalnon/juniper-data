"""Unit tests for the SEC-F01 boot-time auth-posture self-check (HO-2 class).

The API lifespan calls juniper-service-core's ``enforce_auth_posture(
settings.api_keys, require_auth=settings.require_auth,
service_name="juniper-data")`` beside the startup banner — before serving — so
an empty/blank ``JUNIPER_DATA_API_KEYS`` secret (which silently disables
``APIKeyAuth`` and serves the API open behind a healthy health check) is LOUD
at boot, and — with ``JUNIPER_DATA_REQUIRE_AUTH=true`` (default false) — a
boot FAILURE instead (the fail-closed posture for deployments where secrets
are provisioned).

The wiring test monkeypatches the module attribute with a recorder that raises
a sentinel, proving the lifespan invokes the check with the resolved keys
without running the rest of startup. The behavioural tests exercise the helper
directly.
"""

import asyncio

import pytest

from juniper_data.api import app as app_module
from juniper_data.api.app import create_app, lifespan
from juniper_data.api.settings import Settings


class _Sentinel(Exception):
    """Raised by the recorder to stop the lifespan right after the posture check."""


@pytest.mark.unit
class TestAuthPostureLifespanWiring:
    """The check is actually invoked at application startup (before serving)."""

    @pytest.mark.parametrize("require_auth", [False, True])
    def test_lifespan_invokes_posture_check_with_resolved_keys(self, monkeypatch, tmp_path, require_auth) -> None:
        calls: list[tuple[list[str], bool, str]] = []

        def _recorder(api_keys, *, require_auth, service_name, logger=None, **_kwargs):
            calls.append((list(api_keys or []), require_auth, service_name))
            raise _Sentinel

        monkeypatch.setattr(app_module, "enforce_auth_posture", _recorder)
        app = create_app(settings=Settings(storage_path=str(tmp_path), api_keys=["k1", "k2"], require_auth=require_auth))

        async def _enter() -> None:
            async with lifespan(app):
                pass  # pragma: no cover — the recorder raises before yield

        with pytest.raises(_Sentinel):
            asyncio.run(_enter())
        assert calls == [(["k1", "k2"], require_auth, "juniper-data")]

    def test_require_auth_defaults_to_false(self) -> None:
        # Default keeps today's loud-WARNING posture; deployments opt in to
        # fail-closed explicitly (the composed stack sets the env flag).
        assert Settings.model_fields["require_auth"].default is False

    def test_env_flag_flips_posture(self, monkeypatch) -> None:
        monkeypatch.setenv("JUNIPER_DATA_REQUIRE_AUTH", "true")
        assert Settings().require_auth is True

    def test_required_with_no_keys_refuses_startup(self, monkeypatch, tmp_path) -> None:
        # The fail-closed posture end-to-end: the REAL lifespan raises
        # AuthPostureError before serving, so uvicorn startup fails instead
        # of coming up open.
        from juniper_service_core import AuthPostureError

        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)
        monkeypatch.delenv("JUNIPER_DATA_API_KEYS", raising=False)
        app = create_app(settings=Settings(storage_path=str(tmp_path), api_keys=None, require_auth=True))

        async def _enter() -> None:
            async with lifespan(app):
                pass  # pragma: no cover — the posture check raises before yield

        with pytest.raises(AuthPostureError):
            asyncio.run(_enter())

    def test_create_app_itself_does_not_invoke_the_check(self, monkeypatch, tmp_path) -> None:
        # Construction must stay check-free — the posture fires at startup
        # (lifespan), not at factory time.
        monkeypatch.setattr(app_module, "enforce_auth_posture", _recorder_that_raises)
        app = create_app(settings=Settings(storage_path=str(tmp_path), api_keys=None))
        assert app.state.settings.api_keys is None


def _recorder_that_raises(*_args, **_kwargs):
    raise _Sentinel


@pytest.mark.unit
class TestAuthPostureBehaviour:
    """The helper's three outcomes, exercised directly (hermetic)."""

    def test_no_keys_and_not_required_warns_open(self, monkeypatch, caplog) -> None:
        from juniper_service_core import enforce_auth_posture

        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)
        with caplog.at_level("WARNING"):
            enforce_auth_posture(None, require_auth=False, service_name="juniper-data")
        assert any("running OPEN" in rec.getMessage() and "juniper-data" in rec.getMessage() for rec in caplog.records)

    def test_blank_key_counts_as_unset(self, monkeypatch, caplog) -> None:
        # Exactly what an empty secret file resolves to (the HO-2 class).
        from juniper_service_core import auth_is_configured, enforce_auth_posture

        assert not auth_is_configured([""])
        assert not auth_is_configured(["   "])
        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)
        with caplog.at_level("WARNING"):
            enforce_auth_posture(["   "], require_auth=False, service_name="juniper-data")
        assert any("running OPEN" in rec.getMessage() for rec in caplog.records)

    def test_real_key_passes_quietly(self, monkeypatch, caplog) -> None:
        from juniper_service_core import enforce_auth_posture

        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)
        with caplog.at_level("INFO"):
            enforce_auth_posture(["a-real-data-key"], require_auth=True, service_name="juniper-data")
        assert not any(rec.levelname in ("WARNING", "CRITICAL") for rec in caplog.records)

    def test_required_with_no_key_raises(self, monkeypatch) -> None:
        # The fail-closed posture the follow-up flag will enable.
        from juniper_service_core import AuthPostureError, enforce_auth_posture

        monkeypatch.delenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", raising=False)
        with pytest.raises(AuthPostureError):
            enforce_auth_posture([], require_auth=True, service_name="juniper-data")

    def test_escape_hatch_bypasses_the_check(self, monkeypatch) -> None:
        from juniper_service_core import enforce_auth_posture

        monkeypatch.setenv("JUNIPER_SKIP_AUTH_POSTURE_CHECK", "1")
        enforce_auth_posture([], require_auth=True, service_name="juniper-data")
