"""Integration tests for API security middleware."""

import pytest
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.storage import InMemoryDatasetStore


@pytest.fixture
def auth_enabled_client() -> TestClient:
    """Create a test client with API key authentication enabled."""
    settings = Settings(
        storage_path="./test_data",
        api_keys=["valid-key-1", "valid-key-2"],
        rate_limit_enabled=False,
    )
    app = create_app(settings)
    datasets.set_store(InMemoryDatasetStore())
    return TestClient(app)


@pytest.fixture
def rate_limited_client() -> TestClient:
    """Create a test client with rate limiting enabled."""
    settings = Settings(
        storage_path="./test_data",
        api_keys=None,
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=5,
    )
    app = create_app(settings)
    datasets.set_store(InMemoryDatasetStore())
    return TestClient(app)


@pytest.fixture
def fully_secured_client() -> TestClient:
    """Create a test client with both auth and rate limiting enabled."""
    settings = Settings(
        storage_path="./test_data",
        api_keys=["secure-key"],
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=10,
    )
    app = create_app(settings)
    datasets.set_store(InMemoryDatasetStore())
    return TestClient(app)


class TestAPIKeyAuthentication:
    """Integration tests for API key authentication."""

    def test_health_endpoint_exempt(self, auth_enabled_client: TestClient) -> None:
        """Health endpoints should be accessible without API key."""
        response = auth_enabled_client.get("/v1/health")
        assert response.status_code == 200

        response = auth_enabled_client.get("/v1/health/live")
        assert response.status_code == 200

        response = auth_enabled_client.get("/v1/health/ready")
        assert response.status_code == 200

    def test_protected_endpoint_requires_key(self, auth_enabled_client: TestClient) -> None:
        """Protected endpoints should require API key."""
        response = auth_enabled_client.get("/v1/generators")
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_invalid_key_rejected(self, auth_enabled_client: TestClient) -> None:
        """Invalid API keys should be rejected."""
        response = auth_enabled_client.get(
            "/v1/generators",
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_valid_key_accepted(self, auth_enabled_client: TestClient) -> None:
        """Valid API keys should be accepted."""
        response = auth_enabled_client.get(
            "/v1/generators",
            headers={"X-API-Key": "valid-key-1"},
        )
        assert response.status_code == 200

        response = auth_enabled_client.get(
            "/v1/generators",
            headers={"X-API-Key": "valid-key-2"},
        )
        assert response.status_code == 200

    def test_create_dataset_with_auth(self, auth_enabled_client: TestClient) -> None:
        """Dataset creation should work with valid API key."""
        response = auth_enabled_client.post(
            "/v1/datasets",
            json={"generator": "spiral", "params": {"seed": 42}},
            headers={"X-API-Key": "valid-key-1"},
        )
        assert response.status_code == 201
        assert "dataset_id" in response.json()


class TestRateLimiting:
    """Integration tests for rate limiting."""

    def test_health_endpoint_exempt(self, rate_limited_client: TestClient) -> None:
        """Health endpoints should be exempt from rate limiting."""
        for _ in range(20):
            response = rate_limited_client.get("/v1/health")
            assert response.status_code == 200

    def test_allows_requests_within_limit(self, rate_limited_client: TestClient) -> None:
        """Requests within limit should be allowed."""
        for i in range(5):
            response = rate_limited_client.get("/v1/generators")
            assert response.status_code == 200
            assert "X-RateLimit-Remaining" in response.headers
            assert int(response.headers["X-RateLimit-Remaining"]) == 4 - i

    def test_blocks_requests_over_limit(self, rate_limited_client: TestClient) -> None:
        """Requests over limit should be blocked with 429."""
        for _ in range(5):
            rate_limited_client.get("/v1/generators")

        response = rate_limited_client.get("/v1/generators")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
        assert "Retry-After" in response.headers

    def test_rate_limit_headers_present(self, rate_limited_client: TestClient) -> None:
        """Rate limit headers should be present in responses."""
        response = rate_limited_client.get("/v1/generators")
        assert response.status_code == 200

        assert "X-RateLimit-Limit" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "5"
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestCombinedSecurity:
    """Integration tests for combined auth and rate limiting."""

    def test_auth_checked_before_rate_limit(self, fully_secured_client: TestClient) -> None:
        """Authentication should be checked before rate limiting."""
        response = fully_secured_client.get("/v1/generators")
        assert response.status_code == 401

        response = fully_secured_client.get(
            "/v1/generators",
            headers={"X-API-Key": "invalid"},
        )
        assert response.status_code == 401

    def test_rate_limit_applied_after_auth(self, fully_secured_client: TestClient) -> None:
        """Rate limiting should be applied after successful auth."""
        for i in range(10):
            response = fully_secured_client.get(
                "/v1/generators",
                headers={"X-API-Key": "secure-key"},
            )
            assert response.status_code == 200

        response = fully_secured_client.get(
            "/v1/generators",
            headers={"X-API-Key": "secure-key"},
        )
        assert response.status_code == 429

    def test_full_workflow_with_security(self, fully_secured_client: TestClient) -> None:
        """Full dataset workflow should work with security enabled."""
        response = fully_secured_client.post(
            "/v1/datasets",
            json={"generator": "xor", "params": {"seed": 42}},
            headers={"X-API-Key": "secure-key"},
        )
        assert response.status_code == 201
        dataset_id = response.json()["dataset_id"]

        response = fully_secured_client.get(
            f"/v1/datasets/{dataset_id}",
            headers={"X-API-Key": "secure-key"},
        )
        assert response.status_code == 200
        assert response.json()["dataset_id"] == dataset_id
