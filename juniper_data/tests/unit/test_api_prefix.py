"""The API version prefix is spelled once and pinned (APD-DATA-020).

``/v1`` used to be an independent literal in eight places -- the three ``include_router``
calls, the two self-referential ``artifact_url`` f-strings, and the three health entries
of the auth / rate-limit exempt-path set. Nothing tied them together: a version bump, or
a typo in one of them, would have split the routers from the URLs the service hands out
and from the paths its own middleware exempts. ``API_VERSION`` / ``API_PREFIX`` in
``api/constants.py`` are now the single spelling every site derives from.

Three kinds of pin:

* the PUBLISHED VALUE -- ``API_PREFIX == "/v1"``. Changing it is a breaking change for
  every client (``juniper-data-client`` hard-codes ``/v1/...``), so it has to fail a test
  rather than slip through as a constant edit;
* the CALL SITES -- an AST walk over the non-test package fails on any string constant or
  f-string whose text *starts with* ``/v1``. The defect was implicit spelling at the call
  site, and a value assertion cannot see a literal creeping back (a guard equal to the
  default passes the mutation that deletes the guard). Docstrings and metric
  descriptions that merely *mention* ``/v1/...`` mid-sentence are deliberately not
  matched -- they are prose, not paths; and
* the DERIVED SURFACES -- every published OpenAPI path sits under ``API_PREFIX``, the
  exempt-path set carries the three health probes under it, and a freshly created
  dataset's ``artifact_url`` is built from it.

Mutation-checked before shipping: restoring ``prefix="/v1"`` on one router fails the
call-site pin; setting ``API_VERSION = "v2"`` fails the published-value pin (and, as it
should, every request-level test in the suite that spells the wire contract).
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from juniper_data.api import constants as api_constants
from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore

PUBLISHED_PREFIX = "/v1"
PACKAGE_ROOT = Path(api_constants.__file__).resolve().parents[1]


def _leading_literal(node: ast.AST) -> tuple[int, str] | None:
    """Return ``(lineno, text)`` for the leading literal of a string constant or f-string, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.lineno, node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return node.lineno, first.value
    return None


def _source_files() -> list[Path]:
    """Every ``.py`` file in the package outside ``tests/`` -- the surface the constant must own."""
    files = sorted(path for path in PACKAGE_ROOT.rglob("*.py") if "tests" not in path.relative_to(PACKAGE_ROOT).parts)
    assert files, "no source files found -- the call-site walk would be vacuous"
    return files


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """In-memory-store client for the one request-level pin."""
    app = create_app(settings=Settings(storage_path=str(tmp_path)))
    datasets.set_store(InMemoryDatasetStore())
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.unit
class TestApiPrefix:
    """``API_PREFIX`` is the published value, owns every call site, and feeds every derived surface."""

    def test_prefix_is_the_published_value(self) -> None:
        """Published-value pin: the wire contract is ``/v1`` and the prefix is derived from the version."""
        assert api_constants.API_VERSION == "v1"
        assert api_constants.API_PREFIX == PUBLISHED_PREFIX
        assert f"/{api_constants.API_VERSION}" == api_constants.API_PREFIX

    def test_no_prefix_literal_remains_at_any_call_site(self) -> None:
        """Call-site pin: no string constant or f-string in the package starts with ``/v1``."""
        offenders: list[str] = []
        for path in _source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                hit = _leading_literal(node)
                if hit is not None and hit[1].startswith(PUBLISHED_PREFIX):
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{hit[0]}: {hit[1]!r}")
        assert not offenders, "literal API prefixes have crept back -- derive them from API_PREFIX:\n  " + "\n  ".join(offenders)

    def test_every_published_path_sits_under_the_prefix(self, tmp_path: Path) -> None:
        """Derived-surface pin: the OpenAPI document publishes nothing outside ``API_PREFIX``."""
        app = create_app(settings=Settings(storage_path=str(tmp_path)))
        published = set(app.openapi()["paths"])
        assert published, "no published paths -- the pin would be vacuous"
        outside = sorted(path for path in published if not path.startswith(f"{api_constants.API_PREFIX}/"))
        assert not outside, f"paths published outside {api_constants.API_PREFIX}: {outside}"

    def test_exempt_health_probes_derive_from_the_prefix(self) -> None:
        """Derived-surface pin: the auth / rate-limit exempt set names the health probes under the prefix."""
        expected = {f"{api_constants.API_PREFIX}/health", f"{api_constants.API_PREFIX}/health/live", f"{api_constants.API_PREFIX}/health/ready"}
        assert expected <= api_constants.EXEMPT_PATHS, f"exempt set is missing {sorted(expected - api_constants.EXEMPT_PATHS)}"

    def test_artifact_url_derives_from_the_prefix(self, client: TestClient) -> None:
        """Derived-surface pin: the self-referential URL the service hands out is built from the prefix."""
        response = client.post(
            f"{api_constants.API_PREFIX}/datasets",
            json={"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": 1}, "persist": True},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["artifact_url"] == f"{api_constants.API_PREFIX}/datasets/{body['dataset_id']}/artifact"
