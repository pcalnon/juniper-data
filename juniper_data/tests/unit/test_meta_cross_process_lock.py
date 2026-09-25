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

import fcntl
import json
import os
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

    def test_a_create_in_another_process_checks_for_its_dataset_only_under_the_file_lock(self, tmp_path: Path) -> None:
        """A create checks that its dataset is absent, then commits; the check must hold the FILE lock too.

        Under ``_version_lock`` alone the check orders creates within one process only: another
        process's create could land between this one's check and its commit, and be overwritten
        after it had been answered 201. The test stands in for that other process: it holds the
        dataset's lock stripe, as a create mid-commit does, while the child process creates the
        same id. The child reports each existence check through a file, so a check made before
        the lock is an observed event, not an inference from timing.
        """
        storage = tmp_path / "storage"
        store = LocalFSDatasetStore(storage)
        signals = tmp_path / "signals"
        signals.mkdir()
        script = _CREATE_WORKER.format(repo=str(Path(__file__).resolve().parents[3]), dataset_id=DATASET_ID)
        lock_fd = os.open(store._lock_path(DATASET_ID), os.O_RDWR)
        child: subprocess.Popen[str] | None = None
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            child = subprocess.Popen([sys.executable, "-c", textwrap.dedent(script), str(storage), str(signals)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 120
            while not (signals / "ready").exists():
                assert child.poll() is None, f"the child exited early: {child.communicate()[1]}"
                assert time.monotonic() < deadline, "the child never became ready"
                time.sleep(0.005)
            # "ready" is written just before the create. Its staging takes milliseconds; a check
            # made before the file lock would be reported well inside this window.
            window = time.monotonic() + 1.5
            while time.monotonic() < window and not (signals / "checked").exists():
                time.sleep(0.01)
            checked_while_locked = (signals / "checked").exists()
            # The other process's create lands now, inside its lock, while the child waits.
            store.save(DATASET_ID, _meta(["from-the-other-process"]), _arrays(0.0))
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        assert child is not None
        out, err = child.communicate(timeout=120)
        assert child.returncode == 0, err
        assert not checked_while_locked, "the create checked for its dataset before it had the file lock"
        assert (signals / "checked").exists(), "control: the child did check, once it had the lock"
        stored = store.get_meta(DATASET_ID)
        assert stored is not None
        assert stored.tags == ["from-the-other-process"], "the late create must write nothing"
        assert json.loads(out) == ["from-the-other-process"], "the late create must answer with what is stored"


_CREATE_WORKER = """
import json, sys
from datetime import UTC, datetime
from pathlib import Path
sys.path.insert(0, {repo!r})
import numpy as np
from juniper_data.core.models import DatasetMeta
from juniper_data.storage.local_fs import LocalFSDatasetStore

storage, signals = Path(sys.argv[1]), Path(sys.argv[2])
store = LocalFSDatasetStore(storage)
real_get_meta = store.get_meta


def get_meta(dataset_id):
    (signals / "checked").write_text("1")
    return real_get_meta(dataset_id)


store.get_meta = get_meta
meta = DatasetMeta(dataset_id={dataset_id!r}, generator="spiral", generator_version="1.0.0", params={{}}, n_samples=4, n_features=2, n_train=2, n_test=2, tags=["from-the-child"], created_at=datetime.now(UTC))
arrays = {{name: np.ones((2, 2), dtype=np.float32) for name in ("X_train", "y_train", "X_test", "y_test")}}
(signals / "ready").write_text("1")
print(json.dumps(store.save_versioned({dataset_id!r}, meta, arrays).tags))
"""


def _meta(tags: list[str]) -> DatasetMeta:
    return DatasetMeta(
        dataset_id=DATASET_ID,
        generator="spiral",
        generator_version="1.0.0",
        params={},
        n_samples=4,
        n_features=2,
        n_train=2,
        n_test=2,
        tags=tags,
        created_at=datetime.now(UTC),
    )


def _arrays(fill: float) -> dict[str, np.ndarray]:
    return {name: np.full((2, 2), fill, dtype=np.float32) for name in ("X_train", "y_train", "X_test", "y_test")}
