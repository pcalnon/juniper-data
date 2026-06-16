"""End-to-end route tests for the synthetic regression generators (juniper-data#179 §A).

Closes the WS-1 §B acceptance check that a pure-``task_type="regression"``
generator traverses the ``/datasets`` route end to end: POST -> persisted meta
(regression dispatch: ``n_classes`` / ``class_distribution`` None, sequence meta
derived from the 3-D X) -> NPZ artifact download -> contract conformance,
including the juniper-data-client ``validate_npz_contract`` classifier.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     test_e2e_synthetic_regression.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore

pytestmark = [pytest.mark.integration]

# (generator, params); each yields W = n_steps - lookback - horizon + 1 = 184 windows.
SYNTHETIC_CASES = [
    ("multi_sine", {"n_steps": 200, "lookback": 16, "horizon": 1, "seed": 0}),
    ("mackey_glass", {"n_steps": 200, "lookback": 16, "horizon": 1, "discard": 50, "seed": 0}),
    ("ar_p", {"n_steps": 200, "lookback": 16, "horizon": 1, "burn_in": 20, "seed": 0}),
    ("irregular_sine", {"n_steps": 200, "lookback": 16, "horizon": 1, "jitter": 0.5, "seed": 0}),
]


@pytest.fixture
def client() -> TestClient:
    """E2E client backed by a fresh in-memory store."""
    app = create_app(settings=Settings(storage_path="/tmp/juniper_data_synth_e2e_test"))
    datasets.set_store(InMemoryDatasetStore())
    return TestClient(app)


@pytest.mark.parametrize("generator,params", SYNTHETIC_CASES)
def test_e2e_regression_sequence_meta_and_artifact(client: TestClient, generator: str, params: dict) -> None:
    """POST a synthetic generator, then verify regression meta + 3-D artifact."""
    lookback = params["lookback"]
    expected_w = params["n_steps"] - lookback - params["horizon"] + 1

    response = client.post("/v1/datasets", json={"generator": generator, "params": params, "persist": True})
    assert response.status_code == 201, response.text
    body = response.json()
    meta = body["meta"]

    # Regression dispatch: no one-hot/argmax assumption.
    assert meta["task_type"] == "regression"
    assert meta["n_classes"] is None
    assert meta["class_distribution"] is None
    # Sequence meta derived from the 3-D X + registry-declared time_unit.
    assert meta["sequence"] is True
    assert meta["lookback"] == lookback
    assert meta["time_unit"] == "steps"
    assert meta["n_features"] == 1
    assert meta["n_samples"] == expected_w

    artifact = client.get(f"/v1/datasets/{body['dataset_id']}/artifact")
    assert artifact.status_code == 200
    with np.load(io.BytesIO(artifact.content)) as data:
        assert data["X_full"].shape == (expected_w, lookback, 1)
        assert data["y_full"].shape == (expected_w, 1)
        assert data["dt_full"].shape == (expected_w, lookback)
        assert np.all(data["dt_full"][:, 0] == 0)
        assert np.all(data["observed_mask_full"] == 1)


@pytest.mark.parametrize("generator,params", SYNTHETIC_CASES)
def test_e2e_artifact_passes_client_contract_validator(client: TestClient, generator: str, params: dict) -> None:
    """The downloaded NPZ classifies as ``"sequence"`` under the client validator."""
    contract = pytest.importorskip("juniper_data_client.contract")

    response = client.post("/v1/datasets", json={"generator": generator, "params": params, "persist": True})
    assert response.status_code == 201, response.text
    artifact = client.get(f"/v1/datasets/{response.json()['dataset_id']}/artifact")
    assert artifact.status_code == 200
    with np.load(io.BytesIO(artifact.content)) as data:
        arrays = {key: data[key] for key in data.files}

    assert contract.validate_npz_contract(arrays) == "sequence"
