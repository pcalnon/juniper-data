"""Route declaration order is load-bearing -- pin it (APD-DATA-015).

FastAPI dispatches a request to the FIRST route in an ``APIRouter``'s declaration order
whose method and path regex match. ``datasets.py`` declares the parametrised catch-all
``GET /{dataset_id}`` after its literal siblings ``/filter``, ``/stats``, ``/versions``
and ``/latest``, and nothing enforced that order: hoisting the catch-all above any of
them makes that literal route unreachable -- ``GET /v1/datasets/versions?name=x`` then
resolves to ``get_dataset_metadata`` with ``dataset_id="versions"`` and 404s with
``Dataset 'versions' not found`` -- while every test that exercises the catch-all
directly keeps passing.

The pins read ``APIRouter.routes`` -- the public, version-stable declaration table that
FastAPI's matcher iterates (0.137 stopped flattening included routers into ``app.routes``,
which now holds opaque ``_IncludedRouter`` nodes; the router's own list is what those
nodes walk). They never read request outcomes: a request-level probe passes as long as
*something* answers, which is exactly the vacuous shape this guard exists to rule out.

Four pins, so the guard cannot go vacuous:

* the GENERIC invariant walks each router in order and fails if any parametrised route
  would capture the path of a literal route declared after it for a shared HTTP method;
* the NAMED pin asserts the four at-risk datasets ``GET`` literals still exist and each
  precedes ``GET /{dataset_id}`` -- deleting a literal route would satisfy the generic
  invariant trivially, so existence is part of the claim;
* the FIRST-SEGMENT pin asserts every router owns one literal first path segment,
  disjoint from the others', so cross-router shadowing is impossible and the per-router
  invariant is complete for the app; and
* the MOUNT pin asserts the app actually publishes these routers under ``/v1``, so the
  tables under test are the tables that serve.

Mutation-checked before shipping: hoisting the catch-all above ``/versions`` fails the
generic and the named pin; deleting ``/versions`` fails only the named pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI
from starlette.routing import BaseRoute, compile_path

from juniper_data.api.app import create_app
from juniper_data.api.constants import API_PREFIX
from juniper_data.api.routes import datasets, generators, health
from juniper_data.api.settings import Settings

# The routers ``create_app`` includes, in inclusion order (``api/app.py``).
ROUTERS: dict[str, APIRouter] = {
    "health": health.router,
    "generators": generators.router,
    "datasets": datasets.router,
}

DATASETS_CATCH_ALL = "/datasets/{dataset_id}"

# Literal GET siblings whose path also satisfies the catch-all's regex. Each MUST be
# declared before it. The collection root ``/datasets`` is not listed: it has no trailing
# segment for ``{dataset_id}`` to capture.
DATASETS_AT_RISK_LITERALS = (
    "/datasets/filter",
    "/datasets/stats",
    "/datasets/versions",
    "/datasets/latest",
)


def _method_routes(router: APIRouter) -> list[tuple[int, str, frozenset[str], BaseRoute]]:
    """Return ``(index, path, methods, route)`` for every method-bearing route, in declaration order."""
    found: list[tuple[int, str, frozenset[str], BaseRoute]] = []
    for index, route in enumerate(router.routes):
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or path is None:
            continue
        found.append((index, path, frozenset(methods), route))
    assert found, "router declares no method routes -- the walk would be vacuous"
    return found


def _is_parametrised(path: str) -> bool:
    return "{" in path


def _first_segment(path: str) -> str:
    return path.strip("/").split("/", 1)[0]


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    """Build the app without starting it -- the route table is fixed at construction."""
    return create_app(settings=Settings(storage_path=str(tmp_path)))


@pytest.mark.unit
class TestRouteDeclarationOrder:
    """Declaration-order pins for every router the app mounts."""

    @pytest.mark.parametrize("router_name", sorted(ROUTERS))
    def test_no_parametrised_route_shadows_a_later_literal(self, router_name: str) -> None:
        """Generic invariant: an earlier parametrised route must not capture a later literal path."""
        routes = _method_routes(ROUTERS[router_name])
        shadowed: list[str] = []
        for i, literal_path, literal_methods, _ in routes:
            if _is_parametrised(literal_path):
                continue
            for j, param_path, param_methods, _ in routes:
                if j >= i or not _is_parametrised(param_path):
                    continue
                shared = literal_methods & param_methods
                # Same compiler FastAPI uses for the route's own ``path_regex`` (anchored ^...$).
                param_regex, _, _ = compile_path(param_path)
                if shared and param_regex.match(literal_path):
                    shadowed.append(f"{','.join(sorted(shared))} {literal_path} (declared #{i}) is shadowed by {param_path} (declared #{j})")
        assert not shadowed, f"{router_name} router: declaration order shadows literal routes:\n  " + "\n  ".join(shadowed)

    def test_datasets_literals_exist_and_precede_the_catch_all(self) -> None:
        """Named pin: the at-risk GET literals exist and each is declared before ``GET /{dataset_id}``."""
        get_index: dict[str, int] = {}
        for index, path, methods, _ in _method_routes(ROUTERS["datasets"]):
            if "GET" in methods and path not in get_index:
                get_index[path] = index

        assert DATASETS_CATCH_ALL in get_index, f"catch-all GET {DATASETS_CATCH_ALL} is missing from the datasets router"
        catch_all_index = get_index[DATASETS_CATCH_ALL]

        missing = [path for path in DATASETS_AT_RISK_LITERALS if path not in get_index]
        assert not missing, f"expected GET literal routes are missing: {missing}"

        misordered = [f"{path} (declared #{get_index[path]}) is after the catch-all (declared #{catch_all_index})" for path in DATASETS_AT_RISK_LITERALS if get_index[path] > catch_all_index]
        assert not misordered, "datasets literal routes declared after the catch-all:\n  " + "\n  ".join(misordered)

    def test_routers_own_disjoint_literal_first_segments(self) -> None:
        """First-segment pin: each router owns one literal first segment, disjoint from the others'."""
        owned: dict[str, set[str]] = {}
        for router_name, router in ROUTERS.items():
            segments = {_first_segment(path) for _, path, _, _ in _method_routes(router)}
            parametrised = sorted(segment for segment in segments if _is_parametrised(segment))
            assert not parametrised, f"{router_name} router has a parametrised first segment {parametrised}: it could shadow every router included after it"
            assert len(segments) == 1, f"{router_name} router spans several first segments {sorted(segments)}; the per-router order pin assumes one"
            owned[router_name] = segments

        all_segments = [segment for segments in owned.values() for segment in segments]
        assert len(all_segments) == len(set(all_segments)), f"routers share a first segment: {owned}"

    def test_app_mounts_the_pinned_routers_under_the_api_prefix(self, app: FastAPI) -> None:
        """Mount pin: the app publishes every pinned router path under ``/v1``."""
        published = set(app.openapi()["paths"])
        expected = {f"{API_PREFIX}{path}" for router in ROUTERS.values() for _, path, _, route in _method_routes(router) if getattr(route, "include_in_schema", True)}
        assert expected, "no router paths to check -- the pin would be vacuous"
        unpublished = sorted(expected - published)
        assert not unpublished, f"router paths the app does not publish under {API_PREFIX}: {unpublished}"
