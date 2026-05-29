"""Regression tests for Phase 2D metrics and cardinality fixes.

Covers:

- BUG-JD-06: ``ReadinessResponse.timestamp`` is timezone-aware UTC.
- BUG-JD-07: ``record_dataset_generation`` is invoked from ``create_dataset``.
- BUG-JD-08: ``record_access`` is invoked from artifact + meta GET handlers.
- BUG-JD-09: ``PrometheusMiddleware`` labels use the route template, not the
  parameterized URL path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.constants import GENERATION_STATUS_ERROR, POST_CACHE_MISS
from juniper_data.api.models.health import ReadinessResponse
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore


@pytest.fixture(autouse=True)
def _reset_prometheus_registry():
    """Clear the default Prometheus registry between tests so middleware can re-register."""
    from prometheus_client import REGISTRY

    collectors = list(getattr(REGISTRY, "_collector_to_names", {}).keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass
    # Force the lazily-cached dataset metrics to rebuild against the cleared registry.
    import juniper_data.api.observability as obs_mod

    obs_mod._dataset_metrics = None
    yield


@pytest.fixture
def memory_store() -> InMemoryDatasetStore:
    return InMemoryDatasetStore()


@pytest.fixture
def client(memory_store: InMemoryDatasetStore) -> TestClient:
    # SEC-16: ``starlette.TestClient`` defaults to client ``("testclient", 50000)``
    # but ``"testclient"`` is no longer accepted in the allowlist (fail-loud
    # rejects non-IP literals — see ``MetricsAuthMiddleware`` and
    # ``Settings._validate_metrics_trusted_ips``). Override the spoofed client
    # to a real IP that's in the allowlist. See
    # ``test_phase1d_security.TestSEC16MetricsAppIntegration``.
    settings = Settings(
        storage_path="/tmp/juniper_test_phase2d",
        metrics_enabled=True,
        metrics_trusted_ips=["127.0.0.1", "::1"],
    )
    app = create_app(settings=settings)
    datasets.set_store(memory_store)
    return TestClient(app, client=("127.0.0.1", 12345))


# ---------------------------------------------------------------------------
# BUG-JD-06: ReadinessResponse timestamp is UTC-anchored.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadinessTimestampUtc:
    def test_default_factory_uses_utc_now(self) -> None:
        """ReadinessResponse.timestamp must equal datetime.now(UTC).timestamp()."""
        before = datetime.now(UTC).timestamp()
        resp = ReadinessResponse(status="ready", version="0.0.0", service="juniper-data")
        after = datetime.now(UTC).timestamp()
        # The default-factory result should fall in the [before, after] window
        # produced by the same UTC clock — confirms timezone-aware origin.
        assert before <= resp.timestamp <= after

    def test_default_factory_does_not_drift_from_naive_utc_window(self) -> None:
        """The factory uses UTC explicitly, so it must not drift from a naive UTC clock."""
        resp = ReadinessResponse(status="ready", version="0.0.0", service="juniper-data")
        utc_now = datetime.now(UTC).timestamp()
        # Tolerate a 5-second test-host clock skew window.
        delta = abs(resp.timestamp - utc_now)
        assert delta < 5.0


# ---------------------------------------------------------------------------
# BUG-JD-07: record_dataset_generation is wired into create_dataset.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecordDatasetGenerationWiring:
    def test_create_dataset_records_generation_metric(self, client: TestClient) -> None:
        """A successful POST /v1/datasets must call record_dataset_generation(success)."""
        request_body = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "noise": 0.1},
            "persist": True,
        }
        with patch("juniper_data.api.routes.datasets.record_dataset_generation") as mock_record:
            response = client.post("/v1/datasets", json=request_body)
            assert response.status_code == 201
            assert mock_record.called
            call_kwargs = mock_record.call_args.kwargs
            assert call_kwargs["generator"] == "spiral"
            assert call_kwargs["status"] == "success"
            assert isinstance(call_kwargs["duration"], float)
            assert call_kwargs["duration"] >= 0.0

    def test_create_dataset_records_post_metric_when_generation_fails(self, memory_store: InMemoryDatasetStore) -> None:
        """A generator failure must still count the POST as an error cache miss."""
        settings = Settings(
            storage_path="/tmp/juniper_test_phase2d_error",
            metrics_enabled=True,
            metrics_trusted_ips=["127.0.0.1", "::1"],
        )
        app = create_app(settings=settings)
        datasets.set_store(memory_store)
        request_body = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "noise": 0.1},
            "persist": True,
        }

        with (
            TestClient(app, client=("127.0.0.1", 12345), raise_server_exceptions=False) as error_client,
            patch("juniper_data.api.routes.datasets.record_dataset_generation") as mock_record_generation,
            patch("juniper_data.api.routes.datasets.record_dataset_post") as mock_record_post,
            patch("juniper_data.generators.spiral.generator.SpiralGenerator.generate", side_effect=RuntimeError("synthetic generator failure")),
        ):
            response = error_client.post("/v1/datasets", json=request_body)

        assert response.status_code == 500
        assert mock_record_generation.called
        generation_kwargs = mock_record_generation.call_args.kwargs
        assert generation_kwargs["generator"] == "spiral"
        assert generation_kwargs["status"] == GENERATION_STATUS_ERROR
        assert isinstance(generation_kwargs["duration"], float)
        assert generation_kwargs["duration"] >= 0.0
        mock_record_post.assert_called_once_with(
            generator="spiral",
            status=GENERATION_STATUS_ERROR,
            cache=POST_CACHE_MISS,
        )


# ---------------------------------------------------------------------------
# BUG-JD-08: record_access is wired into both metadata + artifact GETs.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecordAccessWiring:
    def _create_dataset(self, client: TestClient) -> str:
        request_body = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "noise": 0.1},
            "persist": True,
        }
        response = client.post("/v1/datasets", json=request_body)
        assert response.status_code == 201
        return response.json()["dataset_id"]

    def test_get_metadata_schedules_record_access(self, client: TestClient, memory_store: InMemoryDatasetStore) -> None:
        """GET /v1/datasets/{id} must schedule record_access via the event loop."""
        dataset_id = self._create_dataset(client)
        with patch.object(memory_store, "record_access") as mock_access:
            response = client.get(f"/v1/datasets/{dataset_id}")
            assert response.status_code == 200
            # call_soon hands off to the loop — TestClient drives it inline so the
            # callback runs before the response returns to the test.
            called = mock_access.called
            assert called
            args = mock_access.call_args.args
            assert args[0] == dataset_id

    def test_get_artifact_schedules_record_access(self, client: TestClient, memory_store: InMemoryDatasetStore) -> None:
        """GET /v1/datasets/{id}/artifact must schedule record_access via the event loop."""
        dataset_id = self._create_dataset(client)
        with patch.object(memory_store, "record_access") as mock_access:
            response = client.get(f"/v1/datasets/{dataset_id}/artifact")
            assert response.status_code == 200
            called = mock_access.called
            assert called
            args = mock_access.call_args.args
            assert args[0] == dataset_id


# ---------------------------------------------------------------------------
# BUG-JD-09: Prometheus labels use the route template, not the live URL.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrometheusRouteTemplate:
    def _create_dataset(self, client: TestClient) -> str:
        request_body = {
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "noise": 0.1},
            "persist": True,
        }
        response = client.post("/v1/datasets", json=request_body)
        assert response.status_code == 201
        return response.json()["dataset_id"]

    def test_metrics_endpoint_uses_route_template_for_dataset_path(self, client: TestClient) -> None:
        """The /metrics output must contain the route template (with ``{dataset_id}``)
        rather than a concrete dataset id, otherwise Prometheus cardinality is unbounded."""
        dataset_id = self._create_dataset(client)

        # Hit the parameterized route a couple of times so PrometheusMiddleware records labels.
        meta_response = client.get(f"/v1/datasets/{dataset_id}")
        artifact_response = client.get(f"/v1/datasets/{dataset_id}/artifact")
        assert meta_response.status_code == 200
        assert artifact_response.status_code == 200

        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        body = metrics_response.text

        # The route template must appear (literal ``{dataset_id}``).
        assert "{dataset_id}" in body
        # And the concrete dataset id must NOT appear in metric labels —
        # that would prove cardinality leakage from the parameter.
        assert dataset_id not in body
