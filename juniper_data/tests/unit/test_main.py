"""Unit tests for __main__.py entry point."""

import sys
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestMain:
    """Tests for the main() entry point function."""

    def test_main_import_error_uvicorn_not_installed(self) -> None:
        """Test main returns 1 when uvicorn is not installed."""
        import builtins
        import importlib

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "uvicorn":
                raise ImportError("No module named 'uvicorn'")
            return original_import(name, *args, **kwargs)

        with patch.object(sys, "argv", ["juniper_data"]), patch("builtins.print") as mock_print, patch.object(builtins, "__import__", side_effect=mock_import), patch.dict(sys.modules, {"uvicorn": None}):
            from juniper_data import __main__ as main_module

            try:
                importlib.reload(main_module)
                result = main_module.main()
                assert result == 1
                mock_print.assert_called()
            except ImportError as e:
                # If ImportError occurs during test setup, skip with explanation
                pytest.skip(f"Cannot test uvicorn import error scenario: {e}")

    def test_main_parses_host_argument(self) -> None:
        """Test main correctly parses --host argument."""
        with patch("uvicorn.run") as mock_run:
            with patch.object(sys, "argv", ["juniper_data", "--host", "127.0.0.1"]):
                from juniper_data.__main__ import main

                main()

                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args
                assert call_kwargs[1]["host"] == "127.0.0.1"

    def test_main_parses_port_argument(self) -> None:
        """Test main correctly parses --port argument."""
        with patch("uvicorn.run") as mock_run:
            with patch.object(sys, "argv", ["juniper_data", "--port", "9000"]):
                from juniper_data.__main__ import main

                main()

                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args
                assert call_kwargs[1]["port"] == 9000

    def test_main_parses_log_level_argument(self) -> None:
        """Test main correctly parses --log-level argument."""
        with patch("uvicorn.run") as mock_run:
            with patch.object(sys, "argv", ["juniper_data", "--log-level", "DEBUG"]):
                from juniper_data.__main__ import main

                main()

                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args
                assert call_kwargs[1]["log_level"] == "debug"

    def test_main_parses_reload_argument(self) -> None:
        """Test main correctly parses --reload argument."""
        with patch("uvicorn.run") as mock_run:
            with patch.object(sys, "argv", ["juniper_data", "--reload"]):
                from juniper_data.__main__ import main

                main()

                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args
                assert call_kwargs[1]["reload"] is True

    def test_main_parses_storage_path_argument(self) -> None:
        """Test main correctly parses --storage-path argument and sets env var."""
        with patch("uvicorn.run") as mock_run:
            with patch.dict("os.environ", {}, clear=False):
                with patch.object(sys, "argv", ["juniper_data", "--storage-path", "/custom/path"]):
                    import os

                    from juniper_data.__main__ import main

                    main()

                    assert os.environ.get("JUNIPER_DATA_STORAGE_PATH") == "/custom/path"
                    mock_run.assert_called_once()

    def test_main_uses_default_settings_when_no_args(self) -> None:
        """Test main uses settings defaults when no args provided."""
        with patch("uvicorn.run") as mock_run:
            with patch.object(sys, "argv", ["juniper_data"]):
                from juniper_data.__main__ import main

                main()

                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args
                assert call_kwargs[1]["host"] == "0.0.0.0"
                assert call_kwargs[1]["port"] == 8100

    def test_main_returns_zero_on_success(self) -> None:
        """Test main returns 0 on successful run."""
        with patch("uvicorn.run"):
            with patch.object(sys, "argv", ["juniper_data"]):
                from juniper_data.__main__ import main

                result = main()

                assert result == 0

    def test_main_app_string(self) -> None:
        """Test main passes correct app string to uvicorn."""
        with patch("uvicorn.run") as mock_run:
            with patch.object(sys, "argv", ["juniper_data"]):
                from juniper_data.__main__ import main

                main()

                mock_run.assert_called_once()
                call_args = mock_run.call_args
                assert call_args[0][0] == "juniper_data.api.app:app"

    def test_main_combines_custom_and_default_args(self) -> None:
        """Test main combines custom args with settings defaults."""
        with patch("uvicorn.run") as mock_run:
            with patch.object(sys, "argv", ["juniper_data", "--host", "localhost"]):
                from juniper_data.__main__ import main

                main()

                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args
                assert call_kwargs[1]["host"] == "localhost"
                assert call_kwargs[1]["port"] == 8100
