"""Regression: a developer's local .env must not pollute the test session.

Background. ``juniper_data/api/settings.py`` configures
pydantic-settings with ``env_file=".env"``, so every ``Settings()``
constructor call reads ``./.env`` (the gitignored, developer-local
copy of ``.env.example``). pydantic-settings layers .env *under*
``os.environ``, which means a per-test ``monkeypatch.delenv(...)``
removes the OS-level value but leaves the .env value in effect.

CI never reproduces this — runner checkouts have no ``.env`` — so the
failure mode is strictly local. The fix is an autouse session-scoped
fixture in ``juniper_data/tests/conftest.py``
(``_disable_settings_env_file_for_tests``) that sets
``Settings.model_config["env_file"] = None`` for the session. This
regression test pins that behavior so a future refactor that drops
or breaks the fixture fails loudly here, not via a mysterious
"default ``storage_path`` is ``/some/path/from/my-dotenv``" failure.

Sibling pin landed in juniper-cascor PR #309 and juniper-canopy
PR #325 (both 2026-05-26).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from juniper_data.api.settings import Settings

pytestmark = pytest.mark.unit


class TestSettingsEnvFileIsolation:
    """Pin the conftest autouse fixture that disables .env loading in tests."""

    def test_settings_env_file_is_none_during_test_session(self):
        """The session-scoped autouse fixture must set env_file to None.

        Direct read of ``Settings.model_config["env_file"]`` after pytest
        has loaded conftest.py. If this assertion fails, the
        ``_disable_settings_env_file_for_tests`` fixture has been
        dropped, renamed, or its scope/autouse semantics broken.
        """
        assert Settings.model_config.get("env_file") is None, "Settings.model_config['env_file'] should be None for the test session. Check juniper_data/tests/conftest.py::_disable_settings_env_file_for_tests."

    def test_local_dot_env_in_cwd_does_not_leak_into_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Behavioral check: a .env in CWD must not override class defaults.

        Synthesizes a temp directory with a polluting ``.env`` that
        sets ``JUNIPER_DATA_STORAGE_PATH`` and ``JUNIPER_DATA_HOST``,
        chdirs into it, clears the same env vars from ``os.environ``,
        and verifies that ``Settings()`` returns the class defaults
        rather than the values written to the file. This exercises the
        actual anti-regression contract — even if the fixture's
        mechanism changes (e.g. from ``env_file=None`` to a chdir+stub
        approach), as long as ``.env`` does not leak in, this test
        passes.
        """
        leaked_storage = "/etc/leaked/by/dot-env/datasets"
        leaked_host = "192.0.2.234"
        env_file = tmp_path / ".env"
        env_file.write_text(
            f"JUNIPER_DATA_STORAGE_PATH={leaked_storage}\nJUNIPER_DATA_HOST={leaked_host}\n",
            encoding="utf-8",
        )

        monkeypatch.chdir(tmp_path)
        for var in ("JUNIPER_DATA_STORAGE_PATH", "JUNIPER_DATA_HOST"):
            monkeypatch.delenv(var, raising=False)

        # Sanity: confirm the file we just wrote is visible from CWD.
        assert (Path.cwd() / ".env").exists(), "Test setup failed: .env not written to tmp_path"

        # Sanity: confirm the env vars are actually unset.
        assert "JUNIPER_DATA_STORAGE_PATH" not in os.environ
        assert "JUNIPER_DATA_HOST" not in os.environ

        settings = Settings()

        assert settings.storage_path != leaked_storage, f"Settings.storage_path leaked from the .env in CWD (got {settings.storage_path!r}). The autouse fixture in conftest.py is no longer preventing pydantic-settings from reading the developer's local .env."
        assert settings.host != leaked_host, f"Settings.host leaked from the .env in CWD (got {settings.host!r}). The autouse fixture in conftest.py is no longer preventing pydantic-settings from reading the developer's local .env."
