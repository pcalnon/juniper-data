"""APD-DATA-007 — the metadata read-modify-write must be atomic ACROSS PROCESSES.

``DatasetStore._version_lock`` is a ``threading.Lock``, so it orders the
read-modify-write only among threads of one interpreter. Both writers of the whole
``DatasetMeta`` document -- ``record_access`` and ``update_tags`` -- rewrite every
field, so two *processes* against one ``storage_path`` interleave freely and the
loser's change is overwritten in full. ``APD-DATA-006`` closed the in-process half of
this; the ``base.py`` docstring it added named the remaining per-process caveat
explicitly, and this is that caveat.

Measured on the unfixed code with the barrier below: twelve processes each adding one
distinct tag left **two** tags on disk, and one process additionally died with
``FileNotFoundError`` because the atomic-write temp path was derived from the final
path alone, so concurrent writers shared it.

Real subprocesses are used rather than ``multiprocessing``: no pickling, no re-import
of the test module under spawn/forkserver, and no fork-with-threads caveat on the
free-threaded build. The ``go`` file is a cross-process barrier -- without it,
subprocess start-up jitter serialises the workers by luck and the test passes even
against the unfixed code, which would make it a vacuous regression test.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.local_fs import LocalFSDatasetStore

DATASET_ID = "spiral-1.0.0-abcdef0123456789"
N_WORKERS = 8

_WORKER = """
import sys, time
from pathlib import Path
sys.path.insert(0, {repo!r})
from juniper_data.storage.local_fs import LocalFSDatasetStore

storage, go, action, arg = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
store = LocalFSDatasetStore(storage)
# Two-phase barrier. Announce readiness only AFTER the interpreter and the store are
# up, then spin (not sleep) until released, so every worker enters the
# read-modify-write inside the same microseconds. Signalling before start-up -- or
# releasing before all workers are ready -- lets process start-up jitter serialise
# them, and the race under test never happens.
(go.parent / ("ready." + arg)).write_text("1")
while not go.exists():
    pass
if action == "tag":
    store.update_tags({dataset_id!r}, [arg], [])
else:
    store.record_access({dataset_id!r})
"""


def _seed_store(storage: Path) -> LocalFSDatasetStore:
    store = LocalFSDatasetStore(storage)
    meta = DatasetMeta(
        dataset_id=DATASET_ID,
        generator="spiral",
        generator_version="1.0.0",
        params={},
        n_samples=4,
        n_features=2,
        n_train=2,
        n_test=2,
        tags=[],
        created_at=datetime.now(UTC),
    )
    arrays = {k: np.zeros((2, 2), dtype=np.float32) for k in ("X_train", "y_train", "X_test", "y_test")}
    store.save(DATASET_ID, meta, arrays)
    return store


def _race(tmp_path: Path, action: str, args: list[str]) -> list[subprocess.CompletedProcess[str]]:
    """Start one subprocess per arg, release them together, return their results."""
    repo_root = str(Path(__file__).resolve().parents[3])
    script = _WORKER.format(repo=repo_root, dataset_id=DATASET_ID)
    go = tmp_path / "go"
    procs = [subprocess.Popen([sys.executable, "-c", textwrap.dedent(script), str(tmp_path), str(go), action, a], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for a in args]
    deadline = time.monotonic() + 120
    while len(list(tmp_path.glob("ready.*"))) < len(args):
        if time.monotonic() > deadline:
            for p in procs:
                p.kill()
            raise AssertionError("workers never became ready")
        time.sleep(0.005)
    go.write_text("go")
    results = []
    for p in procs:
        out, err = p.communicate(timeout=120)
        results.append(subprocess.CompletedProcess(p.args, p.returncode, out, err))
    return results


@pytest.mark.unit
class TestMetaWriteCrossProcessLock:
    def test_concurrent_tag_updates_from_separate_processes_lose_nothing(self, tmp_path: Path) -> None:
        """Every process's tag must survive. A missing tag is a silently lost write."""
        store = _seed_store(tmp_path)
        tags = [f"t{i:02d}" for i in range(N_WORKERS)]

        results = _race(tmp_path, "tag", tags)

        failures = [r for r in results if r.returncode != 0]
        assert not failures, f"worker process crashed: {failures[0].stderr}"

        meta = store.get_meta(DATASET_ID)
        assert meta is not None
        lost = sorted(set(tags) - set(meta.tags))
        assert not lost, f"{len(lost)} of {N_WORKERS} tag writes were silently lost: {lost}"

    def test_concurrent_record_access_from_separate_processes_counts_exactly(self, tmp_path: Path) -> None:
        """The same guarantee for the other whole-document writer.

        APD-DATA-006's lesson was that a lock only one writer takes protects nothing.
        ``record_access`` fires on every metadata read and every artifact download, so
        if it skipped this lock it would go on clobbering tag edits from another
        process -- the original defect, merely relocated.
        """
        store = _seed_store(tmp_path)

        results = _race(tmp_path, "access", [str(i) for i in range(N_WORKERS)])

        failures = [r for r in results if r.returncode != 0]
        assert not failures, f"worker process crashed: {failures[0].stderr}"

        meta = store.get_meta(DATASET_ID)
        assert meta is not None
        assert meta.access_count == N_WORKERS, f"expected exactly {N_WORKERS} accesses, counted {meta.access_count}"

    def test_temp_write_path_is_unique_per_call(self, tmp_path: Path) -> None:
        """Two writers must never choose the same temp file.

        The deterministic temp name was a second, unrecorded cross-process defect found
        while reproducing this entry: concurrent writers shared one temp path, so one
        ``replace()``d it away while the other was still writing and the loser raised
        ``FileNotFoundError``.
        """
        store = LocalFSDatasetStore(tmp_path)
        final = tmp_path / f"{DATASET_ID}.meta.json"

        paths = {store._tmp_path(final) for _ in range(50)}

        assert len(paths) == 50, "temp paths collide within a single process"
        assert all(p != final for p in paths)
        assert all(str(p).startswith(str(final)) for p in paths)

    def test_lock_file_is_invisible_to_dataset_enumeration(self, tmp_path: Path) -> None:
        """The lock file must not read as a dataset.

        Datasets are enumerated with a ``*.meta.json`` glob; the lock suffix is appended
        after that, so the name cannot match. Asserted rather than assumed -- a lock file
        that listed as a dataset would be a self-inflicted correctness bug.
        """
        store = _seed_store(tmp_path)
        with store._meta_write_lock(DATASET_ID):
            pass

        assert store._lock_path(DATASET_ID).exists(), "lock file was not created"
        assert [m.dataset_id for m in store.list_all_metadata()] == [DATASET_ID]
