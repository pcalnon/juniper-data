#!/usr/bin/env python
"""Every generator subpackage must import on its own (juniper-data#316).

``juniper_data.generators.csv_import`` could not be imported standalone. The cycle:

    csv_import/__init__
      -> csv_import.generator          imports juniper_data.api.settings
      -> juniper_data.api/__init__     eagerly imported .app
      -> api.app                       imports .routes.datasets / .routes.generators
      -> api.routes.generators         imports juniper_data.generators.csv_import  <-- still
                                       initialising, so VERSION did not exist yet
      -> ImportError: cannot import name 'VERSION'

Importing a submodule initialises its parent package first, so an ordinary leaf import of
``juniper_data.api.settings`` dragged in the whole FastAPI app and every route.

It was masked in service use -- app startup imports the routes long before anything touches
``csv_import`` -- and surfaced only when the subpackage was imported FIRST, which is exactly
what a test, a script, or an external consumer does. It broke pytest collection of
``test_csv_import_generator.py`` in isolation.

EVERY TEST HERE RUNS IN A SUBPROCESS. A cycle is a property of a cold interpreter: once
``juniper_data`` is imported, ``sys.modules`` is populated and a same-process import succeeds
regardless. An in-process assertion here would pass with the defect fully present, which is
worse than no test at all.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

# Every registered generator, as the service knows them.
GENERATOR_SUBPACKAGES = [
    "ar_p",
    "arc_agi",
    "checkerboard",
    "circles",
    "csv_import",
    "delay_product",
    "equities",
    "equities_seq",
    "gaussian",
    "irregular_sine",
    "mackey_glass",
    "mnist",
    "moon",
    "multi_sine",
    "spiral",
    "xor",
]


def _import_in_cold_interpreter(statement: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", statement], capture_output=True, text=True, check=False)


class TestGeneratorSubpackagesImportStandalone:
    @pytest.mark.parametrize("name", GENERATOR_SUBPACKAGES)
    def test_subpackage_imports_first_in_a_cold_interpreter(self, name):
        """The regression. ``csv_import`` failed this before #316 was fixed."""
        result = _import_in_cold_interpreter(f"import juniper_data.generators.{name}")
        assert result.returncode == 0, f"juniper_data.generators.{name} cannot be imported on its own:\n{result.stderr[-1500:]}"

    def test_csv_import_public_names_resolve(self):
        """The specific failure was on a NAME (``VERSION``), not on the module object."""
        statement = "from juniper_data.generators.csv_import import CsvImportGenerator, CsvImportParams, VERSION, get_schema; print(VERSION)"
        result = _import_in_cold_interpreter(statement)
        assert result.returncode == 0, result.stderr[-1500:]
        assert result.stdout.strip(), "VERSION resolved but is empty"


class TestApiPackageStaysLazy:
    """The fix defers ``create_app``; these pin both halves of that bargain."""

    def test_importing_settings_does_not_pull_in_the_routes(self):
        """The property that actually breaks the cycle.

        If ``api/__init__`` goes back to importing ``.app`` eagerly this fails, and it fails
        for the right reason -- naming the module that should not have been loaded.
        """
        statement = "import sys; import juniper_data.api.settings; print('juniper_data.api.routes.generators' in sys.modules)"
        result = _import_in_cold_interpreter(statement)
        assert result.returncode == 0, result.stderr[-1500:]
        assert result.stdout.strip() == "False", "importing api.settings should not load the route modules"

    def test_create_app_is_still_importable_from_the_package(self):
        """Deferring must not remove it from the public surface."""
        statement = "from juniper_data.api import create_app; print(callable(create_app))"
        result = _import_in_cold_interpreter(statement)
        assert result.returncode == 0, result.stderr[-1500:]
        assert result.stdout.strip() == "True"

    def test_settings_are_still_eager(self):
        statement = "from juniper_data.api import Settings, get_settings; print(callable(get_settings))"
        result = _import_in_cold_interpreter(statement)
        assert result.returncode == 0, result.stderr[-1500:]
        assert result.stdout.strip() == "True"

    def test_unknown_attribute_still_raises_attribute_error(self):
        """The module ``__getattr__`` must not swallow genuine typos."""
        statement = "import juniper_data.api as a\ntry:\n    a.no_such_attribute\nexcept AttributeError:\n    print('AttributeError')"
        result = _import_in_cold_interpreter(statement)
        assert result.returncode == 0, result.stderr[-1500:]
        assert result.stdout.strip() == "AttributeError"
