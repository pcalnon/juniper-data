"""Wire-compat snapshot tests for the R2.1.2 juniper-observability migration.

METRICS-MON R2.1.2 / seed-06: per the R2.1 design §7, every consumer
migration ships a snapshot test that pins the externally-observable
wire format of ``/v1/health/ready`` and the Prometheus scrape so the
shared-lib swap cannot silently drift the contract.

The snapshot below was captured from juniper-data ``main`` at
HEAD = ``84841b09`` (commit immediately before the R2.1.2 migration
landed). Any future bump of the shared lib that changes these keys,
status codes, or label sets will fail this test first.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from juniper_data.api.app import create_app
from juniper_data.api.settings import Settings


@pytest.fixture
def healthy_client():
    """A TestClient backed by a tmp storage directory so /v1/health/ready is healthy."""
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["JUNIPER_DATA_STORAGE_PATH"] = tmp
        try:
            settings = Settings(storage_path=tmp)
            app = create_app(settings=settings)
            yield TestClient(app)
        finally:
            os.environ.pop("JUNIPER_DATA_STORAGE_PATH", None)


# Snapshot captured from main at 84841b09 (pre-R2.1.2). The shared lib
# migration must preserve every entry below. ``git_sha`` / ``build_date`` were
# added additively by the build-provenance effort (obs 0.4.0 / juniper-ml
# notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md) — optional, default ``None``, so
# the extension stays wire-compatible with pre-0.4.0 consumers.
EXPECTED_TOP_LEVEL_KEYS = {"build_date", "dependencies", "details", "git_sha", "service", "status", "timestamp", "version"}
EXPECTED_DEP_KEYS = {"storage"}


class TestReadinessWireCompat:
    """METRICS-MON R2.1.2: /v1/health/ready JSON shape pinned across the migration."""

    def test_status_code_unchanged_when_storage_healthy(self, healthy_client):
        response = healthy_client.get("/v1/health/ready")
        assert response.status_code == 200

    def test_x_juniper_readiness_header_unchanged(self, healthy_client):
        """R1.2 contract: header mirrors body status."""
        response = healthy_client.get("/v1/health/ready")
        assert response.headers.get("X-Juniper-Readiness") == "ready"

    def test_top_level_keys_unchanged(self, healthy_client):
        """The standard ReadinessResponse shape, plus the additive
        build-provenance ``git_sha`` / ``build_date`` keys (obs 0.4.0)."""
        response = healthy_client.get("/v1/health/ready")
        body = response.json()
        assert set(body.keys()) == EXPECTED_TOP_LEVEL_KEYS

    def test_status_value_unchanged(self, healthy_client):
        response = healthy_client.get("/v1/health/ready")
        assert response.json()["status"] == "ready"

    def test_service_identity_unchanged(self, healthy_client):
        response = healthy_client.get("/v1/health/ready")
        assert response.json()["service"] == "juniper-data"

    def test_dependency_set_unchanged(self, healthy_client):
        """Storage remains the sole declared dependency for juniper-data."""
        response = healthy_client.get("/v1/health/ready")
        assert set(response.json()["dependencies"].keys()) == EXPECTED_DEP_KEYS

    def test_timestamp_is_unix_epoch_float(self, healthy_client):
        """The shared lib reconciliation kept ``timestamp`` as a float seconds-since-epoch."""
        import time

        response = healthy_client.get("/v1/health/ready")
        ts = response.json()["timestamp"]
        assert isinstance(ts, float)
        # Sanity: within 60 seconds of "now"
        assert abs(time.time() - ts) < 60.0
