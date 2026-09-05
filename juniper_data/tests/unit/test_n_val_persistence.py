"""n_val persistence and route-wiring leftover from juniper-data#358.

``test_meta_dispatch.py`` already pins compute_shape_meta counts and the Field
default. Those tests stay green if:

* local_fs / redis refuse a pre-n_val ``.meta.json`` (R-3 is the *load*, not
  the Field object);
* ``_SQL_DEFAULTS`` drops ``n_val``, so ``ADD COLUMN`` on a populated table
  has no ``NOT NULL DEFAULT 0``;
* the Postgres mapper hard-codes ``n_val=0`` (fixtures omit the field);
* ``create_dataset`` drops ``n_val=shape_meta["n_val"]`` and the model
  silently defaults to 0 even when ``X_val`` is present.

No generator emits a val partition yet, so the existing API suite cannot see
that last failure. This file covers those four surfaces and does not re-pin
the shape-count arms already in #358.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.core.meta import compute_shape_meta
from juniper_data.core.models import DatasetMeta
from juniper_data.storage.local_fs import LocalFSDatasetStore
from juniper_data.storage.memory import InMemoryDatasetStore
from juniper_data.storage.postgres_store import (
    _SQL_DEFAULTS,
    PostgresDatasetStore,
    build_schema_sql,
)
from juniper_data.storage.redis_store import RedisDatasetStore

pytestmark = [pytest.mark.unit]


def _onehot(labels: list[int], n_classes: int) -> np.ndarray:
    oh = np.zeros((len(labels), n_classes), dtype=np.float32)
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


def _legacy_meta_dict(*, dataset_id: str = "spiral-1.0.0-legacy00000001") -> dict:
    """A stored ``.meta.json`` written before ``n_val`` existed.

    The key is deliberately absent — not present-and-zero. That is the R-3
    load: every artifact predating the third partition.
    """
    payload = {
        "dataset_id": dataset_id,
        "generator": "spiral",
        "generator_version": "1.0.0",
        "params": {"seed": 1},
        "n_samples": 100,
        "n_features": 2,
        "n_classes": 2,
        "n_train": 80,
        "n_test": 20,
        "class_distribution": {"0": 50, "1": 50},
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    assert "n_val" not in payload
    return payload


def _meta_with_val(*, n_val: int = 40) -> DatasetMeta:
    """A three-partition meta. Existing postgres fixtures omit ``n_val`` (→ 0)."""
    return DatasetMeta(
        dataset_id="spiral-1.0.0-val000000000001",
        generator="spiral",
        generator_version="1.0.0",
        params={"seed": 7},
        n_samples=800 + n_val + 200,
        n_features=2,
        n_classes=2,
        n_train=800,
        n_val=n_val,
        n_test=200,
        class_distribution={"0": 520, "1": 520},
        created_at=datetime(2026, 9, 5, 0, 0, tzinfo=UTC),
    )


def _val_arrays() -> dict[str, np.ndarray]:
    """Train 6 / val 3 / test 2. Entire class 1 lives in val so a dropped
    ``n_val=`` kwarg is not disguised as a slightly-wrong count.
    """
    return {
        "X_train": np.zeros((6, 2), np.float32),
        "y_train": _onehot([0, 0, 0, 0, 0, 0], 2),
        "X_val": np.zeros((3, 2), np.float32),
        "y_val": _onehot([1, 1, 1], 2),
        "X_test": np.zeros((2, 2), np.float32),
        "y_test": _onehot([0, 0], 2),
    }


class _ValEmittingGenerator:
    """Stand-in generator: no production generator emits val yet."""

    @staticmethod
    def generate(params: object) -> dict[str, np.ndarray]:
        return _val_arrays()


def test_local_fs_loads_legacy_meta_json_without_n_val(tmp_path):
    """R-3 at the actual load site: ``DatasetMeta(**meta_dict)`` in get_meta.

    Field introspection in #358 stays green if get_meta started requiring the
    key. This writes a pre-n_val document and reads it through the store.
    """
    store = LocalFSDatasetStore(tmp_path)
    payload = _legacy_meta_dict()
    store._meta_path(payload["dataset_id"]).write_text(json.dumps(payload), encoding="utf-8")

    meta = store.get_meta(payload["dataset_id"])

    assert meta is not None
    assert meta.n_val == 0
    assert meta.n_train == 80
    assert meta.n_test == 20


def test_redis_decodes_legacy_meta_without_n_val():
    """Redis uses the same ``DatasetMeta(**parsed)`` constructor as local_fs."""
    store = RedisDatasetStore.__new__(RedisDatasetStore)
    encoded = json.dumps(_legacy_meta_dict()).encode("utf-8")

    meta = store._decode_meta(encoded)

    assert meta.n_val == 0
    assert meta.dataset_id == "spiral-1.0.0-legacy00000001"


def test_postgres_ddl_gives_n_val_a_not_null_default():
    """Populated-table ``ADD COLUMN ... NOT NULL`` needs a default or it fails.

    The generic schema test only checks ALTERs that already carry NOT NULL.
    Dropping ``n_val`` from ``_SQL_DEFAULTS`` makes the ALTER omit the
    constraint entirely, and that generic test stays green.
    """
    assert _SQL_DEFAULTS["n_val"] == "0"
    ddl = build_schema_sql()
    assert "    n_val INTEGER NOT NULL DEFAULT 0" in ddl
    assert "ADD COLUMN IF NOT EXISTS n_val INTEGER NOT NULL DEFAULT 0" in ddl


def test_postgres_older_row_missing_n_val_defaults_to_zero():
    """A table not yet migrated has no n_val key. The model's default must apply."""
    store = PostgresDatasetStore.__new__(PostgresDatasetStore)
    row = store._meta_to_row(_meta_with_val(n_val=40))
    row.pop("n_val")

    restored = store._row_to_meta(row)

    assert restored.n_val == 0


def test_postgres_older_row_null_n_val_defaults_to_zero():
    """An older table can hold NULL where the model now refuses one."""
    store = PostgresDatasetStore.__new__(PostgresDatasetStore)
    row = store._meta_to_row(_meta_with_val(n_val=40))
    row["n_val"] = None

    restored = store._row_to_meta(row)

    assert restored.n_val == 0


def test_postgres_round_trip_preserves_nonzero_n_val():
    """Existing fixtures omit n_val (→ 0). A mapper that hard-codes 0 stays green."""
    store = PostgresDatasetStore.__new__(PostgresDatasetStore)
    original = _meta_with_val(n_val=40)

    restored = store._row_to_meta(store._meta_to_row(original))

    assert restored.n_val == 40
    assert restored.n_samples == 1040
    assert restored == original


def test_regression_with_val_counts_n_val_and_skips_class_distribution():
    """#358's class-distribution arms are classification-only. Regression still
    counts the third partition and must not invent a one-hot distribution.
    """
    arrays = {
        "X_train": np.zeros((6, 4), np.float32),
        "X_val": np.zeros((3, 4), np.float32),
        "X_test": np.zeros((2, 4), np.float32),
        "y_reg_train": np.zeros((6, 1), np.float32),
        "y_reg_val": np.zeros((3, 1), np.float32),
        "y_reg_test": np.zeros((2, 1), np.float32),
    }

    meta = compute_shape_meta(arrays, "regression")

    assert meta["n_val"] == 3
    assert meta["n_samples"] == 11
    assert meta["n_classes"] is None
    assert meta["class_distribution"] is None


def test_create_dataset_wires_n_val_from_shape_meta(tmp_path):
    """Dropping ``n_val=shape_meta["n_val"]`` silently stores 0 on a val artifact.

    compute_shape_meta tests cannot see a dropped constructor kwarg. No
    shipped generator emits val, so the live spiral path cannot either.
    """
    storage = tmp_path / "storage"
    storage.mkdir()
    memory_store = InMemoryDatasetStore()
    app = create_app(settings=Settings(storage_path=str(storage)))
    datasets.set_store(memory_store)
    client = TestClient(app)

    entry = {**datasets.GENERATOR_REGISTRY["spiral"], "generator": _ValEmittingGenerator}
    with patch.dict(datasets.GENERATOR_REGISTRY, {"spiral": entry}):
        created = client.post(
            "/v1/datasets",
            json={
                "generator": "spiral",
                "params": {"n_spirals": 2, "n_points_per_spiral": 10, "seed": 424242},
                "persist": True,
            },
        )

    assert created.status_code == 201, created.text
    body = created.json()
    assert body["meta"]["n_val"] == 3
    assert body["meta"]["n_train"] == 6
    assert body["meta"]["n_test"] == 2
    assert body["meta"]["n_samples"] == 11
    # Omitting y_val from the distribution would drop class 1 entirely.
    assert body["meta"]["class_distribution"] == {"0": 8, "1": 3}

    stored = client.get(f"/v1/datasets/{body['dataset_id']}")
    assert stored.status_code == 200
    assert stored.json()["n_val"] == 3
    assert stored.json()["n_samples"] == 11
