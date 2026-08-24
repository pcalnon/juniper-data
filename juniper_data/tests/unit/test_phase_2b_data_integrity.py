"""Regression tests for Phase 2B (Track 2) data-integrity bug fixes in juniper-data.

Covers:
- BUG-JD-01: ``batch_export`` streams ZIP output instead of buffering it all in memory.
- BUG-JD-02: ``LocalFSDatasetStore.delete`` is now an idempotent, TOCTOU-safe unlink.
- BUG-JD-03: ``LocalFSDatasetStore.update_meta`` uses an atomic temp+replace write.
- BUG-JD-04: ``generate_dataset_id`` mixes in a nonce when ``params['seed']`` is None,
  preventing stale cache hits for non-deterministic generation requests.
"""

import io
import json
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from juniper_data.core.dataset_id import generate_dataset_id
from juniper_data.core.models import DatasetMeta
from juniper_data.storage import LocalFSDatasetStore

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_store():
    """Fresh LocalFSDatasetStore rooted in a tempdir (auto-cleaned)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield LocalFSDatasetStore(Path(tmpdir))


def _make_meta(dataset_id: str = "phase2b-test-001") -> DatasetMeta:
    return DatasetMeta(
        dataset_id=dataset_id,
        generator="spiral",
        generator_version="1.0.0",
        params={"n_spirals": 2, "seed": 42},
        n_samples=10,
        n_features=2,
        n_classes=2,
        n_train=8,
        n_test=2,
        class_distribution={"0": 5, "1": 5},
        artifact_formats=["npz"],
        created_at=datetime(2026, 4, 24, tzinfo=UTC),
        checksum="deadbeef",
    )


def _make_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "X_train": rng.standard_normal((8, 2)).astype(np.float32),
        "y_train": np.eye(2, dtype=np.float32)[rng.integers(0, 2, 8)],
        "X_test": rng.standard_normal((2, 2)).astype(np.float32),
        "y_test": np.eye(2, dtype=np.float32)[rng.integers(0, 2, 2)],
    }


# ---------------------------------------------------------------------------
# BUG-JD-02: idempotent atomic delete
# ---------------------------------------------------------------------------
class TestBugJD02AtomicDelete:
    """``delete`` must not race between ``exists`` and ``unlink``."""

    def test_delete_returns_true_when_files_present(self, tmp_store: LocalFSDatasetStore) -> None:
        meta = _make_meta("del-present")
        tmp_store.save(meta.dataset_id, meta, _make_arrays())

        # Split the side-effecting ``delete`` call from the assertion so the
        # filesystem mutation is not embedded in an ``assert`` expression
        # (CodeQL py/side-effect-in-assert).
        result = tmp_store.delete(meta.dataset_id)
        assert result is True
        assert not tmp_store.exists(meta.dataset_id)

    def test_delete_returns_false_when_nothing_to_delete(self, tmp_store: LocalFSDatasetStore) -> None:
        """Previously ``exists()`` short-circuit returned False via explicit check.

        Post-fix, we rely on ``FileNotFoundError`` from ``unlink`` to discover
        absence, so the return value stays False but via a different path.
        """
        result = tmp_store.delete("never-existed")
        assert result is False

    def test_delete_is_idempotent_second_call_returns_false(self, tmp_store: LocalFSDatasetStore) -> None:
        meta = _make_meta("del-idem")
        tmp_store.save(meta.dataset_id, meta, _make_arrays())

        first_result = tmp_store.delete(meta.dataset_id)
        assert first_result is True
        # Second delete: files are already gone, but call must not raise.
        second_result = tmp_store.delete(meta.dataset_id)
        assert second_result is False

    def test_delete_tolerates_race_with_concurrent_unlink(self, tmp_store: LocalFSDatasetStore) -> None:
        """Simulate the TOCTOU: another process unlinks between our check and unlink.

        We monkey-patch ``Path.unlink`` to raise ``FileNotFoundError`` the first
        time it is invoked and succeed thereafter. Pre-fix code would have
        propagated this error; post-fix it is swallowed quietly.
        """
        meta = _make_meta("del-race")
        tmp_store.save(meta.dataset_id, meta, _make_arrays())
        real_unlink = Path.unlink
        call_counter = {"n": 0}

        def racy_unlink(self, *args, **kwargs):
            call_counter["n"] += 1
            if call_counter["n"] == 1:
                raise FileNotFoundError(self)
            return real_unlink(self, *args, **kwargs)

        with patch.object(Path, "unlink", racy_unlink):
            # Must not propagate the simulated race; second path unlinks normally
            # so ``deleted`` ends up True.
            result = tmp_store.delete(meta.dataset_id)

        assert result is True


# ---------------------------------------------------------------------------
# BUG-JD-03: atomic update_meta
# ---------------------------------------------------------------------------
class TestBugJD03AtomicUpdateMeta:
    """``update_meta`` must write via a temp file and atomically replace."""

    def test_update_meta_persists_new_content(self, tmp_store: LocalFSDatasetStore) -> None:
        meta = _make_meta("upd-ok")
        tmp_store.save(meta.dataset_id, meta, _make_arrays())

        updated = meta.model_copy(update={"checksum": "newchecksum"})
        assert tmp_store.update_meta(meta.dataset_id, updated) is True

        reloaded = tmp_store.get_meta(meta.dataset_id)
        assert reloaded is not None
        assert reloaded.checksum == "newchecksum"

    def test_update_meta_cleans_tmp_file_on_failure(self, tmp_store: LocalFSDatasetStore) -> None:
        """A write failure must not leave a ``.tmp`` sibling behind.

        The original code wrote directly to the final path, so a partial
        write left corrupt JSON. Post-fix we write to ``<name>.tmp`` and
        atomically replace; on error, the temp file must be removed.
        """
        meta = _make_meta("upd-crash")
        tmp_store.save(meta.dataset_id, meta, _make_arrays())
        updated = meta.model_copy(update={"checksum": "neverpersisted"})

        real_replace = Path.replace

        def failing_replace(self, target):
            raise OSError("simulated replace failure")

        with patch.object(Path, "replace", failing_replace), pytest.raises(OSError, match="simulated replace failure"):
            tmp_store.update_meta(meta.dataset_id, updated)

        # No lingering .tmp siblings for this dataset.
        tmp_leftovers = list(tmp_store.base_path.glob(f"{meta.dataset_id}.meta.json.tmp"))
        assert tmp_leftovers == [], f"temp files not cleaned: {tmp_leftovers}"

        # Original meta is still intact and readable (atomicity: either full
        # new content or unchanged old content — never partial).
        _ = real_replace  # silence flake8; reserved for symmetry
        reloaded = tmp_store.get_meta(meta.dataset_id)
        assert reloaded is not None
        assert reloaded.checksum == meta.checksum

    def test_update_meta_returns_false_for_unknown_id(self, tmp_store: LocalFSDatasetStore) -> None:
        meta = _make_meta("upd-missing")
        assert tmp_store.update_meta(meta.dataset_id, meta) is False


# ---------------------------------------------------------------------------
# BUG-JD-04: nonce for seedless dataset IDs
# ---------------------------------------------------------------------------
class TestBugJD04SeedlessNonce:
    """Seedless requests must not collide on a cached ID; seeded ones stay stable."""

    def test_seeded_requests_remain_deterministic(self) -> None:
        params = {"n_spirals": 2, "seed": 7}
        id1 = generate_dataset_id("spiral", "v1.0.0", params)
        id2 = generate_dataset_id("spiral", "v1.0.0", params)
        assert id1 == id2

    def test_seedless_requests_produce_distinct_ids(self) -> None:
        """Two requests with ``seed=None`` must yield different IDs."""
        params = {"n_spirals": 2, "seed": None}
        ids = {generate_dataset_id("spiral", "v1.0.0", params) for _ in range(5)}
        # Collisions over 5 samples at 32-bit nonce are astronomically unlikely.
        assert len(ids) == 5

    def test_missing_seed_key_also_gets_nonce(self) -> None:
        """``seed`` absent entirely is treated the same as ``seed=None``."""
        params = {"n_spirals": 2}
        ids = {generate_dataset_id("spiral", "v1.0.0", params) for _ in range(5)}
        assert len(ids) == 5

    def test_seeded_zero_is_deterministic_not_seedless(self) -> None:
        """``seed=0`` is a valid seed and must not trigger the nonce path."""
        params_a = {"seed": 0, "n_spirals": 2}
        params_b = {"seed": 0, "n_spirals": 2}
        assert generate_dataset_id("spiral", "v1.0.0", params_a) == generate_dataset_id("spiral", "v1.0.0", params_b)

    def test_id_format_preserved_with_nonce(self) -> None:
        """Nonce must not break the ``{generator}-{version}-{hash16}`` shape."""
        dataset_id = generate_dataset_id("spiral", "v1.0.0", {"seed": None})
        parts = dataset_id.split("-")
        assert parts[0] == "spiral"
        assert parts[1] == "v1.0.0"
        assert len(parts[2]) == 16
        int(parts[2], 16)  # valid hex


# ---------------------------------------------------------------------------
# BUG-JD-01: streaming batch_export
# ---------------------------------------------------------------------------
class TestBugJD01StreamingBatchExport:
    """``batch_export`` must yield a valid ZIP and not accumulate it in memory."""

    @pytest.fixture
    def app_client(self, tmp_store: LocalFSDatasetStore):
        """Build a FastAPI app wired to a per-test filesystem store.

        The app's lifespan handler reconstructs the store from Settings at
        startup, so we must override via ``set_store`` *after* entering the
        TestClient context, not before.
        """
        from juniper_data.api.app import create_app
        from juniper_data.api.routes.datasets import set_store

        app = create_app()
        with TestClient(app) as client:
            set_store(tmp_store)
            yield client, tmp_store

    def test_returns_404_when_no_datasets_match(self, app_client) -> None:
        client, _ = app_client
        resp = client.post(
            "/v1/datasets/batch-export",
            json={"dataset_ids": ["does-not-exist-1", "does-not-exist-2"]},
        )
        assert resp.status_code == 404

    def test_returns_valid_zip_with_expected_entries(self, app_client) -> None:
        client, store = app_client
        ids = []
        for i in range(3):
            meta = _make_meta(f"export-{i}")
            store.save(meta.dataset_id, meta, _make_arrays())
            ids.append(meta.dataset_id)

        resp = client.post(
            "/v1/datasets/batch-export",
            json={"dataset_ids": ids},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/zip")

        zbuf = io.BytesIO(resp.content)
        with zipfile.ZipFile(zbuf, mode="r") as zf:
            names = sorted(zf.namelist())
            assert names == sorted(f"{dsid}.npz" for dsid in ids)
            # Every entry must decompress cleanly to a non-empty payload.
            for name in names:
                payload = zf.read(name)
                assert len(payload) > 0

    def test_skips_missing_datasets_in_partial_request(self, app_client) -> None:
        client, store = app_client
        meta = _make_meta("export-partial")
        store.save(meta.dataset_id, meta, _make_arrays())

        resp = client.post(
            "/v1/datasets/batch-export",
            json={"dataset_ids": [meta.dataset_id, "missing-A", "missing-B"]},
        )
        assert resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(resp.content), mode="r") as zf:
            # APD-DATA-010: the present dataset is still the only ARTIFACT exported --
            # that is what this test is for and it is unchanged. The archive now also
            # carries a manifest naming the two ids it could not include; asserting the
            # bare namelist was pinning the silence this endpoint used to keep.
            assert [n for n in zf.namelist() if n.endswith(".npz")] == [f"{meta.dataset_id}.npz"]
            assert json.loads(zf.read("manifest.json"))["missing"] == {"missing-A": "not_found", "missing-B": "not_found"}

    def test_uses_streaming_store_compression(self, app_client) -> None:
        """ZIP entries must be ZIP_STORED, a prerequisite for true streaming
        (ZIP_DEFLATED would require seeking back to patch the local header)."""
        client, store = app_client
        meta = _make_meta("export-stored")
        store.save(meta.dataset_id, meta, _make_arrays())

        resp = client.post(
            "/v1/datasets/batch-export",
            json={"dataset_ids": [meta.dataset_id]},
        )
        assert resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(resp.content), mode="r") as zf:
            info = zf.infolist()[0]
            assert info.compress_type == zipfile.ZIP_STORED
