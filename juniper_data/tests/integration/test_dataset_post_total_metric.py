"""Live integration test for the POST cache-hit observability counter.

METRICS-MON R4.5 / R3.1 follow-up.

R3.1 surfaced a gap: ``api/routes/datasets.create_dataset`` short-circuits
on a cached ``dataset_id`` and skips ``record_dataset_generation``. R3.1's
own test worked around the symptom by varying the ``noise`` parameter
between POSTs (different ``dataset_id`` → different cache key → real
generator runs both times). The actual observability fix lives here:
``juniper_data_dataset_post_total{generator, status, cache="hit"|"miss"}``
counts every incoming POST regardless of cache state.

This test pins the regression by driving the inverse of R3.1: two POSTs
with **identical** parameters (same ``dataset_id``, second hits the
cache). The new ``post_total`` counter must increment by 2 (one miss +
one hit); the existing ``generations_total`` counter must increment by
only 1 (only the first POST ran the generator). The ratio
``post_total - generations_total`` is the cache-hit rate that capacity
planning will key off.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore


@pytest.fixture(autouse=True)
def _reset_prometheus_registry():
    """Clear the default Prometheus registry between tests so middleware can re-register.

    Mirrors ``test_phase_2d_metrics._reset_prometheus_registry``. Without this,
    metrics accumulated by other test modules in the same process would inflate
    the baseline and break the increment-by-N assertions below.
    """
    from prometheus_client import REGISTRY

    collectors = list(getattr(REGISTRY, "_collector_to_names", {}).keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass
    import juniper_data.api.observability as obs_mod

    obs_mod._dataset_metrics = None
    yield


@pytest.fixture
def memory_store() -> InMemoryDatasetStore:
    return InMemoryDatasetStore()


@pytest.fixture
def client(memory_store: InMemoryDatasetStore, tmp_path) -> TestClient:
    storage = tmp_path / "juniper_data_r4_5"
    storage.mkdir()
    settings = Settings(
        storage_path=str(storage),
        metrics_enabled=True,
        metrics_trusted_ips=["testclient", "127.0.0.1", "::1"],
    )
    app = create_app(settings=settings)
    datasets.set_store(memory_store)
    return TestClient(app, raise_server_exceptions=False)


class _FailingParams(BaseModel):
    """Minimal params model for a generator that fails after validation."""


class _FailingGenerator:
    @staticmethod
    def generate(params: _FailingParams) -> dict:
        raise RuntimeError("synthetic generator failure")


@pytest.fixture
def client_allowing_server_errors(memory_store: InMemoryDatasetStore, tmp_path) -> TestClient:
    storage = tmp_path / "juniper_data_r4_5_errors"
    storage.mkdir()
    settings = Settings(
        storage_path=str(storage),
        metrics_enabled=True,
        metrics_trusted_ips=["testclient", "127.0.0.1", "::1"],
    )
    app = create_app(settings=settings)
    datasets.set_store(memory_store)
    return TestClient(app, raise_server_exceptions=False)


_POST_TOTAL_RE = re.compile(
    r"^juniper_data_dataset_post_total\{([^}]*)\}\s+([0-9.eE+\-]+)\s*$",
    re.MULTILINE,
)
_COUNTER_RE = re.compile(
    r"^juniper_data_dataset_generations_total\{([^}]*)\}\s+([0-9.eE+\-]+)\s*$",
    re.MULTILINE,
)
_HIST_COUNT_RE = re.compile(
    r"^juniper_data_dataset_generation_duration_seconds_count\{([^}]*)\}\s+([0-9.eE+\-]+)\s*$",
    re.MULTILINE,
)


def _parse_label_set(labels: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for kv in re.findall(r'(\w+)="([^"]*)"', labels):
        out[kv[0]] = kv[1]
    return out


def _scrape_post_total(client: TestClient, *, generator: str, status: str, cache: str) -> float:
    response = client.get("/metrics")
    assert response.status_code == 200, response.text
    body = response.text
    for label_str, value_str in _POST_TOTAL_RE.findall(body):
        labels = _parse_label_set(label_str)
        if labels.get("generator") == generator and labels.get("status") == status and labels.get("cache") == cache:
            return float(value_str)
    return 0.0


def _scrape_generations_total(client: TestClient, *, generator: str, status: str) -> float:
    response = client.get("/metrics")
    assert response.status_code == 200, response.text
    body = response.text
    for label_str, value_str in _COUNTER_RE.findall(body):
        labels = _parse_label_set(label_str)
        if labels.get("generator") == generator and labels.get("status") == status:
            return float(value_str)
    return 0.0


def _scrape_generation_duration_count(client: TestClient, *, generator: str) -> float:
    response = client.get("/metrics")
    assert response.status_code == 200, response.text
    body = response.text
    for label_str, value_str in _HIST_COUNT_RE.findall(body):
        labels = _parse_label_set(label_str)
        if labels.get("generator") == generator:
            return float(value_str)
    return 0.0


def _post_spiral(client: TestClient, *, noise: float) -> str:
    response = client.post(
        "/v1/datasets",
        json={
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "noise": noise},
            "persist": True,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


@pytest.mark.integration
class TestDatasetPostTotalMetric:
    """METRICS-MON R4.5: ``juniper_data_dataset_post_total`` counts every POST."""

    def test_cache_miss_then_hit_yields_post_total_2_generations_1(self, client: TestClient) -> None:
        """Two POSTs with identical params: first is a cache miss (generator
        runs), second is a cache hit (route short-circuits). The ``post_total``
        counter must reflect both; ``generations_total`` only the first.
        """
        first_id = _post_spiral(client, noise=0.1)
        miss_count = _scrape_post_total(client, generator="spiral", status="success", cache="miss")
        gen_count_after_miss = _scrape_generations_total(client, generator="spiral", status="success")
        assert miss_count == 1.0
        assert gen_count_after_miss == 1.0

        # Second POST with identical params → ``dataset_id`` matches first
        # → cache hit → route short-circuits → ``post_total{cache="hit"}``
        # increments but ``generations_total`` does not.
        second_id = _post_spiral(client, noise=0.1)
        assert second_id == first_id, "Identical params must produce identical dataset_id (cache key)"

        hit_count = _scrape_post_total(client, generator="spiral", status="success", cache="hit")
        gen_count_after_hit = _scrape_generations_total(client, generator="spiral", status="success")

        assert hit_count == 1.0, "POST cache-hit branch must bump post_total{cache='hit'}"
        assert gen_count_after_hit == 1.0, "POST cache-hit must NOT bump generations_total (regression guard for the R3.1-surfaced gap)"

        # Total POSTs (hit + miss) = 2; total generations = 1; cache-hit
        # rate = (post_total - generations_total) / post_total = 50%.
        assert miss_count + hit_count == 2.0

    def test_cache_miss_each_time_yields_post_total_equals_generations_total(self, client: TestClient) -> None:
        """Two POSTs with different params: both miss the cache, both run the
        generator. ``post_total{cache="miss"}`` and ``generations_total``
        increment in lockstep.
        """
        _post_spiral(client, noise=0.1)
        _post_spiral(client, noise=0.2)

        miss_count = _scrape_post_total(client, generator="spiral", status="success", cache="miss")
        gen_count = _scrape_generations_total(client, generator="spiral", status="success")
        hit_count = _scrape_post_total(client, generator="spiral", status="success", cache="hit")

        assert miss_count == 2.0
        assert gen_count == 2.0
        assert hit_count == 0.0, "No POST should have hit the cache when params differ"

    def test_generator_error_records_post_total_error_miss_without_duration_observation(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """A generator exception is still a POST cache miss, but must not add
        a success-duration histogram observation.
        """

        class FailingSpiralGenerator:
            @staticmethod
            def generate(params: object) -> None:  # noqa: ARG004
                raise RuntimeError("synthetic generator failure")

        monkeypatch.setitem(
            datasets.GENERATOR_REGISTRY,
            "spiral",
            {
                **datasets.GENERATOR_REGISTRY["spiral"],
                "generator": FailingSpiralGenerator,
            },
        )

        error_client = TestClient(client.app, raise_server_exceptions=False)
        response = error_client.post(
            "/v1/datasets",
            json={
                "generator": "spiral",
                "params": {"n_spirals": 2, "n_points_per_spiral": 50, "noise": 0.4},
                "persist": True,
            },
        )

        assert response.status_code == 500
        assert _scrape_post_total(error_client, generator="spiral", status="error", cache="miss") == 1.0
        assert _scrape_generations_total(error_client, generator="spiral", status="error") == 1.0
        assert _scrape_generation_duration_count(error_client, generator="spiral") == 0.0

    def test_metrics_body_exposes_help_and_type_for_post_total(self, client: TestClient) -> None:
        """Sanity guard: HELP + TYPE lines pin the metric name."""
        _post_spiral(client, noise=0.3)
        response = client.get("/metrics")
        assert response.status_code == 200
        body = response.text
        assert "# HELP juniper_data_dataset_post_total" in body
        assert "# TYPE juniper_data_dataset_post_total counter" in body

    def test_generator_error_records_post_total_error_miss(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generator exceptions must still count the failed POST as a cache miss.

        Without this, generator failure rate would appear in
        ``generations_total`` but disappear from the request-volume counter,
        skewing cache-hit-rate and error-rate dashboards.
        """
        failing_generator = "__failing_metrics__"
        patched_registry = {
            **datasets.GENERATOR_REGISTRY,
            failing_generator: {
                "generator": _FailingGenerator,
                "params_class": _FailingParams,
                "version": "test",
                "description": "Synthetic failing generator for metrics regression coverage.",
            },
        }
        monkeypatch.setattr(datasets, "GENERATOR_REGISTRY", patched_registry)

        response = client.post(
            "/v1/datasets",
            json={
                "generator": failing_generator,
                "params": {},
                "persist": True,
            },
        )

        assert response.status_code == 500
        assert _scrape_post_total(client, generator=failing_generator, status="error", cache="miss") == 1.0
        assert _scrape_generations_total(client, generator=failing_generator, status="error") == 1.0
