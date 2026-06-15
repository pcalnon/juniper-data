"""Tests for enhanced health check endpoints with dependency status."""

import pytest
from fastapi.testclient import TestClient

from juniper_data import __version__
from juniper_data.api.app import create_app
from juniper_data.api.models.health import DependencyStatus, ReadinessResponse
from juniper_data.api.settings import Settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear cached settings between tests."""
    from juniper_data.api.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """PERF-JD-01: reset the readiness probe's dataset-count cache between
    tests so a file added in one test isn't masked by a count cached in
    an earlier test."""
    from juniper_data.api.routes.health import _reset_probe_cache

    _reset_probe_cache()
    yield
    _reset_probe_cache()


@pytest.fixture
def test_settings(tmp_path, monkeypatch) -> Settings:
    """Create test settings with a real storage directory."""
    storage = tmp_path / "datasets"
    storage.mkdir()
    monkeypatch.setenv("JUNIPER_DATA_STORAGE_PATH", str(storage))
    return Settings(storage_path=str(storage))


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    """Create a test client."""
    app = create_app(settings=test_settings)
    return TestClient(app)


@pytest.mark.unit
class TestDependencyStatusModel:
    """Test DependencyStatus Pydantic model."""

    def test_healthy_status(self):
        dep = DependencyStatus(name="Test", status="healthy", latency_ms=1.5, message="ok")
        assert dep.name == "Test"
        assert dep.status == "healthy"
        assert dep.latency_ms == 1.5

    def test_unhealthy_status(self):
        dep = DependencyStatus(name="Test", status="unhealthy", message="connection refused")
        assert dep.status == "unhealthy"
        assert dep.latency_ms is None

    def test_not_configured_status(self):
        dep = DependencyStatus(name="Optional", status="not_configured")
        assert dep.status == "not_configured"
        assert dep.message is None


@pytest.mark.unit
class TestReadinessResponseModel:
    """Test ReadinessResponse Pydantic model."""

    def test_ready_response(self):
        resp = ReadinessResponse(status="ready", version="0.4.2", service="juniper-data")
        assert resp.status == "ready"
        assert resp.version == "0.4.2"
        assert resp.service == "juniper-data"
        assert resp.timestamp > 0
        assert resp.dependencies == {}
        assert resp.details == {}

    def test_degraded_response_with_deps(self):
        dep = DependencyStatus(name="Storage", status="unhealthy", message="not found")
        resp = ReadinessResponse(
            status="degraded",
            version="0.4.2",
            service="juniper-data",
            dependencies={"storage": dep},
        )
        assert resp.status == "degraded"
        assert resp.dependencies["storage"].status == "unhealthy"


@pytest.mark.unit
class TestEnhancedReadinessEndpoint:
    """Test enhanced /v1/health/ready endpoint (R1.2 contract)."""

    def test_readiness_with_valid_storage(self, client, test_settings):
        """Ready when storage directory exists; 200 + X-Juniper-Readiness=ready."""
        response = client.get("/v1/health/ready")
        assert response.status_code == 200
        assert response.headers.get("X-Juniper-Readiness") == "ready"
        body = response.json()
        assert body["status"] == "ready"
        assert body["version"] == __version__
        assert body["service"] == "juniper-data"
        assert "timestamp" in body
        assert body["dependencies"]["storage"]["status"] == "healthy"
        assert body["dependencies"]["storage"]["name"] == "Dataset Storage"

    def test_readiness_with_datasets(self, client, test_settings, tmp_path):
        """Storage message includes dataset count."""
        storage = tmp_path / "datasets"
        (storage / "test1.npz").touch()
        (storage / "test2.npz").touch()
        response = client.get("/v1/health/ready")
        body = response.json()
        assert "2 datasets" in body["dependencies"]["storage"]["message"]

    def test_readiness_503_when_required_dep_unhealthy(self, monkeypatch):
        """R1.2 / seed-02: missing storage → 503 + status="not_ready" so LBs shed traffic."""
        monkeypatch.setenv("JUNIPER_DATA_STORAGE_PATH", "/nonexistent/path/datasets")
        settings = Settings(storage_path="/nonexistent/path/datasets")
        app = create_app(settings=settings)
        c = TestClient(app)
        response = c.get("/v1/health/ready")
        assert response.status_code == 503
        assert response.headers.get("X-Juniper-Readiness") == "not_ready"
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["dependencies"]["storage"]["status"] == "unhealthy"
        assert "not found" in body["dependencies"]["storage"]["message"]

    def test_readiness_probe_cache_reuses_count_within_ttl(self, client, test_settings, tmp_path):
        """PERF-JD-01: a second readiness probe within the cache window must
        not re-glob the storage directory. The endpoint should reuse the
        cached (is_dir, count) tuple even if files are added after the
        first probe — proving the cache holds — until ``_reset_probe_cache``
        or the TTL clears it. This pins the steady-state O(1) hot-path."""
        storage = tmp_path / "datasets"
        # First probe: 1 dataset
        (storage / "alpha.npz").touch()
        first = client.get("/v1/health/ready").json()
        assert "1 dataset" in first["dependencies"]["storage"]["message"]

        # Add a second file. Within TTL the cache should still report 1.
        (storage / "beta.npz").touch()
        second = client.get("/v1/health/ready").json()
        assert "1 dataset" in second["dependencies"]["storage"]["message"], "expected probe cache to return stale count within TTL"

        # Manual invalidation surfaces the fresh count.
        from juniper_data.api.routes.health import _reset_probe_cache

        _reset_probe_cache()
        third = client.get("/v1/health/ready").json()
        assert "2 datasets" in third["dependencies"]["storage"]["message"]

    def test_readiness_probe_cache_invalidates_on_path_change(self, monkeypatch, tmp_path):
        """PERF-JD-01: switching ``storage_path`` between requests must NOT
        return the cached count from the previous path. Without the
        path-equality check in ``_probe_storage`` the cache would alias
        tests that share a test_settings fixture but install different
        storage dirs at the request level."""
        from juniper_data.api.routes.health import _reset_probe_cache

        # First app: 1 file at path A
        path_a = tmp_path / "a"
        path_a.mkdir()
        (path_a / "1.npz").touch()
        _reset_probe_cache()
        settings_a = Settings(storage_path=str(path_a))
        client_a = TestClient(create_app(settings=settings_a))
        body_a = client_a.get("/v1/health/ready").json()
        assert "1 dataset" in body_a["dependencies"]["storage"]["message"]

        # Second app: 3 files at path B. Cache from path A must not bleed.
        path_b = tmp_path / "b"
        path_b.mkdir()
        for n in ("1.npz", "2.npz", "3.npz"):
            (path_b / n).touch()
        settings_b = Settings(storage_path=str(path_b))
        client_b = TestClient(create_app(settings=settings_b))
        body_b = client_b.get("/v1/health/ready").json()
        assert "3 datasets" in body_b["dependencies"]["storage"]["message"]


@pytest.mark.unit
class TestLivenessProbe:
    """Test /v1/health/live with R1.2 tick contract."""

    def test_liveness_200_when_tick_succeeds(self, client):
        """Healthy storage → 200 + tick/duration_ms in body."""
        response = client.get("/v1/health/live")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "alive"
        assert body["tick"] == "juniper-data"
        assert isinstance(body["duration_ms"], int)
        assert body["duration_ms"] >= 0

    def test_liveness_503_when_tick_raises(self, monkeypatch):
        """R1.2 / seed-03: storage gone → tick raises → 503."""
        monkeypatch.setenv("JUNIPER_DATA_STORAGE_PATH", "/nonexistent/path/datasets")
        settings = Settings(storage_path="/nonexistent/path/datasets")
        app = create_app(settings=settings)
        c = TestClient(app)
        response = c.get("/v1/health/live")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unresponsive"
        assert body["tick"] == "juniper-data"
        assert "storage path" in body["error"]
        assert isinstance(body["duration_ms"], int)

    def test_liveness_503_when_tick_exceeds_budget(self, client, monkeypatch):
        """R1.2 / seed-03: tick exceeding LIVENESS_TICK_BUDGET_MS → 503."""
        from juniper_data.api.routes import health as health_module

        def slow_tick(_settings):
            import time as _t

            _t.sleep((health_module.LIVENESS_TICK_BUDGET_MS + 50) / 1000)

        monkeypatch.setattr(health_module, "_liveness_tick", slow_tick)
        response = client.get("/v1/health/live")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unresponsive"
        assert body["tick"] == "juniper-data"
        assert "exceeded budget" in body["error"]
        assert body["duration_ms"] > health_module.LIVENESS_TICK_BUDGET_MS


@pytest.mark.unit
class TestBackwardCompatibleEndpoints:
    """Test that /v1/health remains unchanged for legacy integrations."""

    def test_health_check_unchanged(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__


@pytest.mark.unit
class TestBuildProvenance:
    """Build provenance on /v1/health + /v1/health/ready (stale-image detection).

    juniper-ml notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md — the image stamps
    ``JUNIPER_DATA_GIT_SHA`` / ``JUNIPER_DATA_BUILD_DATE`` env vars at build
    time (from build-args); the health endpoints surface them and the
    ``provenance`` accessor reads them so ``make doctor`` can detect when a
    running container has fallen behind its source.
    """

    def test_health_includes_provenance_null_outside_image(self, client, monkeypatch):
        """Outside a provenance-stamped image the fields are present but null."""
        monkeypatch.delenv("JUNIPER_DATA_GIT_SHA", raising=False)
        monkeypatch.delenv("JUNIPER_DATA_BUILD_DATE", raising=False)
        body = client.get("/v1/health").json()
        assert body["git_sha"] is None
        assert body["build_date"] is None

    def test_health_surfaces_baked_provenance(self, client, monkeypatch):
        """When the image baked the env vars, /v1/health reports them."""
        monkeypatch.setenv("JUNIPER_DATA_GIT_SHA", "abc1234")
        monkeypatch.setenv("JUNIPER_DATA_BUILD_DATE", "2026-06-14T00:00:00Z")
        body = client.get("/v1/health").json()
        assert body["git_sha"] == "abc1234"
        assert body["build_date"] == "2026-06-14T00:00:00Z"

    def test_readiness_surfaces_baked_provenance(self, client, monkeypatch):
        """The shared ReadinessResponse also carries git_sha/build_date."""
        monkeypatch.setenv("JUNIPER_DATA_GIT_SHA", "def5678")
        monkeypatch.setenv("JUNIPER_DATA_BUILD_DATE", "2026-06-14T01:02:03Z")
        body = client.get("/v1/health/ready").json()
        assert body["git_sha"] == "def5678"
        assert body["build_date"] == "2026-06-14T01:02:03Z"

    def test_readiness_provenance_null_outside_image(self, client, monkeypatch):
        """Readiness fields default to null with no provenance env present."""
        monkeypatch.delenv("JUNIPER_DATA_GIT_SHA", raising=False)
        monkeypatch.delenv("JUNIPER_DATA_BUILD_DATE", raising=False)
        body = client.get("/v1/health/ready").json()
        assert body["git_sha"] is None
        assert body["build_date"] is None

    def test_accessor_returns_none_when_unset(self, monkeypatch):
        from juniper_data import provenance

        monkeypatch.delenv("JUNIPER_DATA_GIT_SHA", raising=False)
        monkeypatch.delenv("JUNIPER_DATA_BUILD_DATE", raising=False)
        assert provenance.git_sha() is None
        assert provenance.build_date() is None

    def test_accessor_empty_string_is_none(self, monkeypatch):
        """A bare ``docker build`` leaves the env var empty-string → None."""
        from juniper_data import provenance

        monkeypatch.setenv("JUNIPER_DATA_GIT_SHA", "")
        monkeypatch.setenv("JUNIPER_DATA_BUILD_DATE", "")
        assert provenance.git_sha() is None
        assert provenance.build_date() is None

    def test_accessor_returns_value_when_set(self, monkeypatch):
        from juniper_data import provenance

        monkeypatch.setenv("JUNIPER_DATA_GIT_SHA", "deadbee")
        monkeypatch.setenv("JUNIPER_DATA_BUILD_DATE", "2026-06-14T12:00:00Z")
        assert provenance.git_sha() == "deadbee"
        assert provenance.build_date() == "2026-06-14T12:00:00Z"
