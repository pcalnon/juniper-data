"""Both binary routes declare one media type, spelled once and pinned (APD-DATA-025).

``GET /v1/datasets/{id}/artifact`` used to answer ``application/octet-stream`` while
``POST /v1/datasets/batch-export`` answered ``application/zip`` -- two inline literals, one
per route, with no rule tying them together. Both payloads are ZIP containers: an NPZ
artifact is numpy's zip of ``.npy`` members, and the batch export is a zip of NPZ files.
``application/octet-stream`` is the RFC 9110 §8.3 fallback a recipient may assume when no
``Content-Type`` is present at all, so ``application/zip`` is strictly more informative --
and it was already what the other binary route said. ``BINARY_MEDIA_TYPE`` in
``api/constants.py`` is now the one spelling; both routes derive from it, and the format each
route serves is named by its ``Content-Disposition`` filename (``<id>.npz`` /
``datasets.zip``), not by the media type.

Three kinds of pin:

* the PUBLISHED VALUE -- ``BINARY_MEDIA_TYPE == "application/zip"``. Changing it is a wire
  change for every client, so it must fail a test rather than slip through as a constant
  edit. (``juniper-data-client`` returns ``response.content`` without reading the header,
  which is why this change was safe to make -- and why nothing but this test would notice
  the next one);
* the CALL SITES -- an AST walk over ``api/routes/`` fails on any ``media_type=`` keyword
  whose value is not the ``BINARY_MEDIA_TYPE`` name. A value assertion cannot see an inline
  literal creeping back that happens to equal the constant; and
* the WIRE -- both routes answer with the published value, their ``Content-Disposition``
  filenames keep their extensions, and the bytes really are what the type claims: a ZIP the
  ``zipfile`` module recognises, with ``np.load`` reading the artifact and ``.npz`` members
  inside the export.

Mutation-checked before shipping (``juniper-ml/util/ad-hoc/apd_data_025_mutation_check.py``):
an inline ``"application/zip"`` literal on one route fails only the call-site pin; the
artifact route restored to ``"application/octet-stream"`` fails the call-site pin and the
artifact wire pin; ``BINARY_MEDIA_TYPE = "application/octet-stream"`` fails the
published-value pin and both wire pins.
"""

from __future__ import annotations

import ast
import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from juniper_data.api import constants as api_constants
from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore

PUBLISHED_MEDIA_TYPE = "application/zip"
ROUTES_DIR = Path(datasets.__file__).resolve().parent


def _spiral_request(seed: int) -> dict:
    """A small persisted spiral dataset; distinct seeds give distinct dataset ids."""
    return {"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": seed}, "persist": True}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """In-memory-store client for the wire pins."""
    app = create_app(settings=Settings(storage_path=str(tmp_path)))
    datasets.set_store(InMemoryDatasetStore())
    with TestClient(app) as test_client:
        yield test_client


def _create_dataset(client: TestClient, seed: int) -> str:
    response = client.post(f"{api_constants.API_PREFIX}/datasets", json=_spiral_request(seed))
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


def _media_type_keywords(tree: ast.AST) -> Iterator[tuple[int, ast.expr]]:
    """Yield ``(lineno, value)`` for every ``media_type=`` keyword in any call in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "media_type":
                    yield node.lineno, keyword.value


@pytest.mark.unit
class TestBinaryMediaTypes:
    """``BINARY_MEDIA_TYPE`` is the published value, owns every call site, and is true of the bytes."""

    def test_published_value(self) -> None:
        """Published-value pin: the wire contract for both binary routes is ``application/zip``."""
        assert api_constants.BINARY_MEDIA_TYPE == PUBLISHED_MEDIA_TYPE

    def test_every_media_type_call_site_names_the_constant(self) -> None:
        """Call-site pin: no ``media_type=`` in any route module is anything but the constant's name."""
        offenders: list[str] = []
        census = 0
        for path in sorted(ROUTES_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for lineno, value in _media_type_keywords(tree):
                census += 1
                if not (isinstance(value, ast.Name) and value.id == "BINARY_MEDIA_TYPE"):
                    offenders.append(f"{path.name}:{lineno}: media_type={ast.unparse(value)}")
        assert census >= 2, f"only {census} media_type call site(s) found -- the walk would be vacuous"
        assert not offenders, "media types spelled inline again -- derive them from BINARY_MEDIA_TYPE:\n  " + "\n  ".join(offenders)

    def test_artifact_download_is_a_zip_declared_as_one(self, client: TestClient) -> None:
        """Wire pin: the NPZ artifact answers ``application/zip``, keeps its ``.npz`` name, and is a zip."""
        dataset_id = _create_dataset(client, seed=1)
        response = client.get(f"{api_constants.API_PREFIX}/datasets/{dataset_id}/artifact")
        assert response.status_code == 200
        assert response.headers["content-type"] == PUBLISHED_MEDIA_TYPE
        assert response.headers["content-disposition"] == f"attachment; filename={dataset_id}.npz"
        assert zipfile.is_zipfile(io.BytesIO(response.content)), "the declared media type is not true of the bytes"
        with np.load(io.BytesIO(response.content)) as npz:
            assert "X_train" in npz.files

    def test_batch_export_is_a_zip_declared_as_one(self, client: TestClient) -> None:
        """Wire pin: the export answers ``application/zip``, keeps its ``.zip`` name, and holds ``.npz`` members."""
        dataset_ids = [_create_dataset(client, seed=1), _create_dataset(client, seed=2)]
        assert len(set(dataset_ids)) == 2, "the two seeds collapsed to one dataset -- the export pin would be vacuous"
        response = client.post(f"{api_constants.API_PREFIX}/datasets/batch-export", json={"dataset_ids": dataset_ids})
        assert response.status_code == 200
        assert response.headers["content-type"] == PUBLISHED_MEDIA_TYPE
        assert response.headers["content-disposition"] == "attachment; filename=datasets.zip"
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            members = archive.namelist()
        assert len(members) == 2 and all(member.endswith(".npz") for member in members), members
