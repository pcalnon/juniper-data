"""Unit tests for Docker secrets support."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from juniper_data.core.secrets import get_secret


@pytest.mark.unit
class TestGetSecret:
    """Tests for the get_secret function."""

    def test_returns_env_var_when_no_file_var_set(self, tmp_path: Path) -> None:
        """Test reading from env var when no file var is set."""
        env = {"MY_SECRET": "env-value"}
        with patch.dict(os.environ, env, clear=True):
            result = get_secret("MY_SECRET")
            assert result == "env-value"

    def test_returns_file_contents_when_file_var_set(self, tmp_path: Path) -> None:
        """Test reading from file when file var is set."""
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("file-secret-value\n")
        env = {"MY_SECRET_FILE": str(secret_file)}
        with patch.dict(os.environ, env, clear=True):
            result = get_secret("MY_SECRET")
            assert result == "file-secret-value"

    def test_file_takes_precedence_over_env_var(self, tmp_path: Path) -> None:
        """Test file-based secret takes precedence over env var."""
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("from-file\n")
        env = {
            "MY_SECRET": "from-env",
            "MY_SECRET_FILE": str(secret_file),
        }
        with patch.dict(os.environ, env, clear=True):
            result = get_secret("MY_SECRET")
            assert result == "from-file"

    def test_returns_none_when_neither_set(self) -> None:
        """Test returns None when neither env var nor file var is set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_secret("NONEXISTENT_SECRET")
            assert result is None

    def test_default_file_env_var_appends_file_suffix(self, tmp_path: Path) -> None:
        """Test default file_env_var naming appends _FILE."""
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("auto-suffix-value\n")
        env = {"API_KEY_FILE": str(secret_file)}
        with patch.dict(os.environ, env, clear=True):
            result = get_secret("API_KEY")
            assert result == "auto-suffix-value"

    def test_custom_file_env_var(self, tmp_path: Path) -> None:
        """Test explicit custom file_env_var parameter."""
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("custom-path-value\n")
        env = {"CUSTOM_PATH": str(secret_file)}
        with patch.dict(os.environ, env, clear=True):
            result = get_secret("MY_SECRET", file_env_var="CUSTOM_PATH")
            assert result == "custom-path-value"

    def test_file_not_found_falls_back_to_env_var(self) -> None:
        """Test that a missing file falls back to env var."""
        env = {
            "MY_SECRET": "fallback-value",
            "MY_SECRET_FILE": "/nonexistent/path/secret.txt",
        }
        with patch.dict(os.environ, env, clear=True):
            result = get_secret("MY_SECRET")
            assert result == "fallback-value"

    def test_file_content_is_stripped(self, tmp_path: Path) -> None:
        """Test that whitespace and newlines are stripped from file contents."""
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("  secret-with-whitespace  \n\n")
        env = {"MY_SECRET_FILE": str(secret_file)}
        with patch.dict(os.environ, env, clear=True):
            result = get_secret("MY_SECRET")
            assert result == "secret-with-whitespace"
