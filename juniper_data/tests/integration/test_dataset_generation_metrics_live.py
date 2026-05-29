"""Live integration test for the dataset-generation Prometheus pipeline.

METRICS-MON R3.1 / seed-08 / BUG-JD-07.

Existing unit coverage in
``juniper_data/tests/unit/test_phase_2d_metrics.py::TestRecordDatasetGenerationWiring``
verifies that ``create_dataset`` *calls* ``record_dataset_generation`` via a
``patch(...)`` mock. That asserts the wiring contract, but it does not catch:

* a regression where the helper writes to the wrong metric name or labels;
* a regression where the metric is registered against a private registry that
  the ``/metrics`` mount does not see;
* a regression where the histogram is observed but the route handler exits
  before the registry is committed (e.g. background-task lifecycle bug);
* a label drift like ``status="ok"`` vs ``status="success"`` that mocks would
  silently accept but Prometheus consumers (alerts, dashboards) would not.

This test goes end-to-end:

1. Start the FastAPI app with metrics enabled.
2. POST ``/v1/datasets`` (``generator="spiral"``) twice — the second post
   pins counter increment-by-one against a pre-call baseline so the test is
   robust to test-isolation bleed (the autouse fixture
   ``_reset_prometheus_registry`` rebuilds the registry per test, but a
   future contributor could remove it).
3. GET ``/metrics`` and parse the Prometheus exposition format.
4. Assert ``juniper_data_dataset_generations_total{generator="spiral",
   status="success"}`` increased by exactly **1** between the two scrapes
   that bracket the second POST.
5. Assert ``juniper_data_dataset_generation_duration_seconds_count{generator
   ="spiral"}`` is exactly **2** after the two POSTs (one observation per
   successful generation; histogram is suppressed for ``status="error"``).
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage.memory import InMemoryDatasetStore


@pytest.fixture(autouse=True)
def _reset_prometheus_registry():
    """Clear the default Prometheus registry between tests so middleware can re-register.

    Mirrors ``test_phase_2d_metrics._reset_prometheus_registry``. Without this,
    metrics accumulated by other test modules in the same process would inflate
    the baseline and break the increment-by-one assertion below.
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
    storage = tmp_path / "juniper_data_r3_1"
    storage.mkdir()
    # SEC-16: fail-loud validation now rejects non-IP literals in
    # ``metrics_trusted_ips``, so override the TestClient's spoofed client
    # to a real loopback IP that is in the allowlist.
    settings = Settings(
        storage_path=str(storage),
        metrics_enabled=True,
        metrics_trusted_ips=["127.0.0.1", "::1"],
    )
    app = create_app(settings=settings)
    datasets.set_store(memory_store)
    return TestClient(app, client=("127.0.0.1", 50001))


_COUNTER_RE = re.compile(
    r"^juniper_data_dataset_generations_total\{([^}]*)\}\s+([0-9.eE+\-]+)\s*$",
    re.MULTILINE,
)
_HIST_COUNT_RE = re.compile(
    r"^juniper_data_dataset_generation_duration_seconds_count\{([^}]*)\}\s+([0-9.eE+\-]+)\s*$",
    re.MULTILINE,
)


def _parse_label_set(labels: str) -> dict[str, str]:
    """Parse a Prometheus label set like ``generator="spiral",status="success"``."""
    out: dict[str, str] = {}
    for kv in re.findall(r'(\w+)="([^"]*)"', labels):
        out[kv[0]] = kv[1]
    return out


def _scrape_counter(client: TestClient, generator: str, status: str) -> float:
    """Return current value of the spiral/success counter sample, or 0.0 if absent."""
    response = client.get("/metrics")
    assert response.status_code == 200, response.text
    body = response.text
    for label_str, value_str in _COUNTER_RE.findall(body):
        labels = _parse_label_set(label_str)
        if labels.get("generator") == generator and labels.get("status") == status:
            return float(value_str)
    return 0.0


def _scrape_histogram_count(client: TestClient, generator: str) -> float:
    """Return ``_count`` (number of observations) for the duration histogram, or 0.0 if absent."""
    response = client.get("/metrics")
    assert response.status_code == 200, response.text
    body = response.text
    for label_str, value_str in _HIST_COUNT_RE.findall(body):
        labels = _parse_label_set(label_str)
        if labels.get("generator") == generator:
            return float(value_str)
    return 0.0


def _post_spiral(client: TestClient, *, noise: float) -> None:
    """POST a spiral dataset. ``noise`` varies between calls to defeat the
    content-addressed cache short-circuit in
    ``api/routes/datasets.py::create_dataset`` (which returns the cached meta
    without calling ``record_dataset_generation`` when ``dataset_id`` —
    derived from generator+version+params — already exists in the store).
    """
    response = client.post(
        "/v1/datasets",
        json={
            "generator": "spiral",
            "params": {"n_spirals": 2, "n_points_per_spiral": 50, "noise": noise},
            "persist": True,
        },
    )
    assert response.status_code == 201, response.text


@pytest.mark.integration
class TestDatasetGenerationMetricsLive:
    """METRICS-MON R3.1: ``record_dataset_generation`` → ``/metrics`` is end-to-end live."""

    def test_counter_and_histogram_observed_after_post(self, client: TestClient) -> None:
        # First POST establishes the metric exists with the expected labels and
        # gives us a non-zero baseline to increment from. (Counters are not
        # exposed in /metrics until they have been incremented at least once
        # under a given label set, so a pre-POST scrape would return 0.0 from
        # our absent-label fallback path either way.)
        _post_spiral(client, noise=0.1)
        baseline_counter = _scrape_counter(client, "spiral", "success")
        baseline_hist_count = _scrape_histogram_count(client, "spiral")
        assert baseline_counter == 1.0
        assert baseline_hist_count == 1.0

        # Second POST must increment counter by exactly 1 and add exactly 1
        # histogram observation. Anything else means the production wiring
        # is double-counting, mis-labeling, or skipping the metric entirely.
        # Use a different ``noise`` value so ``dataset_id`` differs and the
        # content-addressed cache doesn't short-circuit the generator path
        # (which would skip ``record_dataset_generation``).
        _post_spiral(client, noise=0.2)
        assert _scrape_counter(client, "spiral", "success") == baseline_counter + 1.0
        assert _scrape_histogram_count(client, "spiral") == baseline_hist_count + 1.0

    def test_metrics_body_exposes_help_and_type_lines(self, client: TestClient) -> None:
        """Sanity guard: the exposition format includes HELP+TYPE for both metrics.

        Pins that the metric registration produced the expected names. Catches
        a regression where the metric is renamed without a wire-compat shim
        (alerts and dashboards would silently break otherwise).
        """
        _post_spiral(client, noise=0.3)
        response = client.get("/metrics")
        assert response.status_code == 200
        body = response.text
        assert "# HELP juniper_data_dataset_generations_total" in body
        assert "# TYPE juniper_data_dataset_generations_total counter" in body
        assert "# HELP juniper_data_dataset_generation_duration_seconds" in body
        assert "# TYPE juniper_data_dataset_generation_duration_seconds histogram" in body
