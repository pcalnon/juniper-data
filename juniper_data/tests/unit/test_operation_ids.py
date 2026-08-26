"""Every route declares an explicit, stable ``operation_id`` (APD-DATA-023).

When a route declares none, FastAPI derives its ``operationId`` from the handler name, the
full path and the method (``fastapi.utils.generate_unique_id`` --
``create_dataset_v1_datasets_post``). The id is what a code generator names the SDK method
after, so its *stability* is a public-API concern -- and with the default in force, renaming
a handler, moving a route between routers or changing the version prefix silently renames
every generated method. None of the 21 routes declared one.

Every decorator now carries ``operation_id="<name>"`` -- the handler name at introduction
(``download_artifact``, ``list_generators``, ``readiness_probe`` ...) -- which decouples the
published id from the Python name. That is a one-time change of every id from the generated
shape to the explicit one; no generated SDK exists in the ecosystem (``juniper-data-client``
is hand-written), so nothing consumed the old ids.

Three kinds of pin:

* the CALL SITES -- an AST walk over ``api/routes/`` fails on any ``@router.<verb>(...)``
  decorator whose ``operation_id`` keyword is missing, non-literal, empty or not an
  identifier. A route registered without one inherits the generated id silently; the
  syntactic pin is what sees the omission at the line it happens;
* the REGISTERED ROUTES -- every ``APIRoute`` on the three routers has ``operation_id`` set
  and serves exactly that id (``unique_id``), never the generated fallback. Read from
  ``APIRouter.routes``, not ``app.routes`` (FastAPI 0.137+ keeps included routers opaque
  there -- see ``test_route_order_guard``); and
* the PUBLISHED CONTRACT -- the OpenAPI document's ``operationId`` list has no duplicate and
  its set equals the frozen set below. Adding, removing or renaming an id is a conscious
  contract change that must edit this file in the same PR; renaming a handler is not one,
  and must not fail anything here.

Mutation-checked before shipping (``juniper-ml/util/ad-hoc/apd_data_023_mutation_check.py``):
deleting one ``operation_id=`` fails the call-site, registered-route, fallback and contract
pins; renaming a handler fails nothing (the decoupling this file exists for); changing one id
string fails only the contract pin; duplicating an id fails the uniqueness arm and, because the
displaced id vanishes from the set, the contract pin.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.utils import generate_unique_id

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets, generators, health
from juniper_data.api.settings import Settings

ROUTES_DIR = Path(datasets.__file__).resolve().parent
ROUTERS = (health.router, generators.router, datasets.router)
HTTP_VERBS = frozenset({"get", "post", "put", "patch", "delete", "head", "options", "trace"})

# The published contract. Each id is served by exactly one operation. The handler behind an
# id may be renamed freely; the id itself may not change without editing this set.
PUBLISHED_OPERATION_IDS = frozenset(
    {
        # health
        "health_check",
        "liveness_probe",
        "readiness_probe",
        # generators
        "list_generators",
        "get_generator_schema",
        # datasets
        "create_dataset",
        "list_datasets",
        "filter_datasets",
        "get_dataset_stats",
        "batch_delete_datasets",
        "batch_create_datasets",
        "batch_update_tags",
        "batch_export_datasets",
        "cleanup_expired_datasets",
        "list_dataset_versions",
        "get_latest_version",
        "get_dataset_metadata",
        "download_artifact",
        "preview_dataset",
        "delete_dataset",
        "update_dataset_tags",
    }
)


def _route_decorators(tree: ast.AST) -> Iterator[tuple[ast.Call, str]]:
    """Yield ``(decorator call, handler name)`` for every ``@router.<verb>(...)`` in ``tree``."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr in HTTP_VERBS and isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == "router":
                yield decorator, node.name


def _route_files() -> list[Path]:
    """Every route module -- the surface the call-site pin walks."""
    files = sorted(path for path in ROUTES_DIR.glob("*.py") if path.name != "__init__.py")
    assert files, "no route modules found -- the call-site walk would be vacuous"
    return files


def _api_routes() -> list[APIRoute]:
    """Every ``APIRoute`` registered on the three routers the app includes."""
    routes = [route for router in ROUTERS for route in router.routes if isinstance(route, APIRoute)]
    assert routes, "no APIRoute registered -- the registered-route pin would be vacuous"
    return routes


def _published_operation_ids(tmp_path: Path) -> list[str]:
    """The ``operationId`` of every operation in the served OpenAPI document, in document order."""
    app = create_app(settings=Settings(storage_path=str(tmp_path)))
    ids = [operation["operationId"] for path_item in app.openapi()["paths"].values() for operation in path_item.values()]
    assert ids, "no published operations -- the contract pin would be vacuous"
    return ids


@pytest.mark.unit
class TestOperationIds:
    """Every route's ``operation_id`` is explicit at the call site, served as declared, and pinned as a contract."""

    def test_every_decorator_declares_a_literal_operation_id(self) -> None:
        """Call-site pin: each ``@router.<verb>(...)`` names a literal, identifier-shaped ``operation_id``."""
        offenders: list[str] = []
        census = 0
        for path in _route_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for decorator, handler in _route_decorators(tree):
                census += 1
                keyword = next((keyword for keyword in decorator.keywords if keyword.arg == "operation_id"), None)
                value = keyword.value if keyword is not None else None
                if not (isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.isidentifier()):
                    offenders.append(f"{path.name}:{decorator.lineno}: {handler} declares no literal operation_id")
        assert census == len(PUBLISHED_OPERATION_IDS), f"decorator census {census} != pinned contract size {len(PUBLISHED_OPERATION_IDS)} -- a route was added or removed without editing PUBLISHED_OPERATION_IDS"
        assert not offenders, "routes without an explicit operation_id inherit FastAPI's generated one -- declare it:\n  " + "\n  ".join(offenders)

    def test_every_registered_route_serves_its_declared_id(self) -> None:
        """Registered-route pin: ``operation_id`` is set on every ``APIRoute`` and is the id it serves."""
        offenders: list[str] = []
        for route in _api_routes():
            if route.operation_id is None:
                offenders.append(f"{route.path}: no explicit operation_id (would serve {generate_unique_id(route)!r})")
            elif route.unique_id != route.operation_id:
                offenders.append(f"{route.path}: serves {route.unique_id!r}, declared {route.operation_id!r}")
        assert not offenders, "\n  ".join(["registered routes drifting from their declared id:", *offenders])

    def test_no_route_serves_the_generated_fallback(self) -> None:
        """Fallback pin: no route's served id is the one FastAPI would have generated for it."""
        fallbacks = sorted(f"{route.path}: {route.unique_id}" for route in _api_routes() if route.unique_id == generate_unique_id(route))
        assert not fallbacks, "routes serving FastAPI's generated operationId shape:\n  " + "\n  ".join(fallbacks)

    def test_published_ids_are_unique(self, tmp_path: Path) -> None:
        """Contract pin, uniqueness arm: no two operations share an ``operationId``."""
        ids = _published_operation_ids(tmp_path)
        duplicates = sorted({operation_id for operation_id in ids if ids.count(operation_id) > 1})
        assert not duplicates, f"duplicate operationIds in the published document: {duplicates}"

    def test_published_ids_are_the_pinned_contract(self, tmp_path: Path) -> None:
        """Contract pin: the served ``operationId`` set is exactly the frozen set -- renames are conscious."""
        published = set(_published_operation_ids(tmp_path))
        missing = sorted(PUBLISHED_OPERATION_IDS - published)
        unexpected = sorted(published - PUBLISHED_OPERATION_IDS)
        assert published == PUBLISHED_OPERATION_IDS, f"operationId contract changed -- missing={missing} unexpected={unexpected}; edit PUBLISHED_OPERATION_IDS deliberately if this is intended"
