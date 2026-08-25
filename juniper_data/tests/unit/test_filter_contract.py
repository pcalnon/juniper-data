"""``GET /v1/datasets/filter``'s query contract is the route -- and it is pinned (APD-DATA-021).

``juniper_data.core.models.DatasetListFilter`` was declared, exported nowhere and used
nowhere: the ``/filter`` route re-declared every field as an individual ``Query`` param,
so the model was a second, silently drifting spelling of the contract. It could not
simply be wired in -- measured before deciding
(``juniper-ml/util/ad-hoc/apd_data_021_openapi_probe.py``): ``Depends(DatasetListFilter)``
drops ``tags`` from the query parameters altogether (the route takes a comma-separated
``str``; the model declared ``list[str]``) and strips every parameter's description. Both
are wire-contract changes for ``juniper-data-client``. So the dead model is gone, the route
consumes the ``TAGS_MATCH_*`` constants it used to duplicate, and this module pins the
contract of record so a future "helpful" wiring cannot reshape it silently.

Four pins:

* the PARAMETER SET -- the exact query-parameter names of ``GET /v1/datasets/filter``;
* the TAGS SHAPE -- ``tags`` is a nullable *string* (comma-separated), never an array;
* the SHARED CONSTANTS, at the CALL SITE -- an AST read of ``filter_datasets`` asserts the
  ``tags_match`` ``Query(...)`` names ``TAGS_MATCH_DEFAULT`` / ``TAGS_MATCH_PATTERN`` rather
  than spelling literals. A value comparison against the constants cannot see an inline
  literal that happens to equal them (the vacuous-guard shape), so the published
  ``pattern`` / ``default`` are checked *as well as* the names; and
* ANTI-RESURRECTION -- no ``DatasetListFilter`` on ``juniper_data.core.models``. A filter
  model that is not what the route validates with is a second spelling, and this repo has
  already paid for one. If a model is ever wanted, wire it into the route in the same
  change and update these pins consciously.

Mutation-checked before shipping: spelling ``pattern="^(any|all)$"`` inline in the route
fails only the call-site pin; re-adding a bare ``DatasetListFilter`` class fails only the
anti-resurrection pin; the probe above is the evidence behind the parameter-set pin.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi import FastAPI

from juniper_data.api import constants as api_constants
from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets as datasets_routes
from juniper_data.api.settings import Settings
from juniper_data.core import constants as core_constants
from juniper_data.core import models

FILTER_PATH = f"{api_constants.API_PREFIX}/datasets/filter"

# The contract of record. Adding or removing a query parameter is a client-visible change
# and must be made here on purpose.
EXPECTED_FILTER_PARAMS = frozenset(
    {
        "generator",
        "tags",
        "tags_match",
        "created_after",
        "created_before",
        "min_samples",
        "max_samples",
        "include_expired",
        "dataset_name",
        "dataset_version",
        "limit",
        "offset",
        "cursor",
    }
)


def _filter_params(app: FastAPI) -> dict[str, dict]:
    params = {p["name"]: p for p in app.openapi()["paths"][FILTER_PATH]["get"]["parameters"]}
    assert params, "the /filter route publishes no parameters -- the pins would be vacuous"
    return params


def _tags_match_query_call() -> ast.Call:
    """Return the ``Query(...)`` call that is ``filter_datasets``'s ``tags_match`` default."""
    source = Path(datasets_routes.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=datasets_routes.__file__)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == "filter_datasets":
            args = node.args
            positional = args.posonlyargs + args.args
            defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
            for arg, default in zip(positional, defaults, strict=True):
                if arg.arg == "tags_match":
                    assert isinstance(default, ast.Call), "tags_match default is not a Query(...) call"
                    return default
            for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if arg.arg == "tags_match":
                    assert isinstance(default, ast.Call), "tags_match default is not a Query(...) call"
                    return default
    raise AssertionError("filter_datasets(tags_match=Query(...)) not found in api/routes/datasets.py")


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    return create_app(settings=Settings(storage_path=str(tmp_path)))


@pytest.mark.unit
class TestFilterContract:
    """The ``/filter`` route is the only spelling of its query contract."""

    def test_query_parameter_set_is_the_published_contract(self, app: FastAPI) -> None:
        """Parameter-set pin: exactly the expected names, no more, no fewer."""
        published = frozenset(_filter_params(app))
        assert published == EXPECTED_FILTER_PARAMS, f"missing={sorted(EXPECTED_FILTER_PARAMS - published)} extra={sorted(published - EXPECTED_FILTER_PARAMS)}"

    def test_tags_is_a_comma_separated_string_not_an_array(self, app: FastAPI) -> None:
        """Tags-shape pin: ``tags`` stays a nullable string; wiring a ``list[str]`` model would change it."""
        schema = _filter_params(app)["tags"]["schema"]
        variants = schema.get("anyOf", [schema])
        assert {"type": "string"} in variants, f"tags is not a string parameter: {schema}"
        assert not any(v.get("type") == "array" for v in variants), f"tags became an array parameter: {schema}"

    def test_tags_match_is_bound_to_the_shared_constants_at_the_call_site(self, app: FastAPI) -> None:
        """Shared-constants pin: the route names the constants (AST) and publishes their values (OpenAPI)."""
        call = _tags_match_query_call()
        bound = {kw.arg: kw.value for kw in call.keywords}
        for keyword, expected_name in (("default", "TAGS_MATCH_DEFAULT"), ("pattern", "TAGS_MATCH_PATTERN")):
            value = bound.get(keyword)
            assert isinstance(value, ast.Name) and value.id == expected_name, f"tags_match Query({keyword}=...) does not name {expected_name}: {ast.dump(value) if value is not None else None}"

        schema = _filter_params(app)["tags_match"]["schema"]
        assert schema.get("pattern") == core_constants.TAGS_MATCH_PATTERN
        assert schema.get("default") == core_constants.TAGS_MATCH_DEFAULT

    def test_no_second_spelling_of_the_filter_contract(self) -> None:
        """Anti-resurrection pin: the never-wired model must not quietly come back."""
        assert not hasattr(models, "DatasetListFilter"), "DatasetListFilter is back -- wire it into the /filter route in the same change, or leave it out"
