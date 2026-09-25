"""ETags, conditional requests and the access-counter split (APD-DATA-017 / -029 / -032).

The owner rulings. 2026-09-11: an ``ETag`` derived from the stored SHA-256, and the access
counters moved OUT of the representation so the metadata body can carry a strong one;
``Content-Location`` on ``/latest`` naming the canonical ``/{dataset_id}``. Rejected: ETags
on artifacts only; a weak validator that churns on every read; a 307 from ``/latest``.
2026-09-23: the ARTIFACT's tag is WEAK, ``W/"<checksum>"`` -- that SHA-256 covers the arrays,
not the bytes served -- while the metadata tags stay strong.

The load-bearing test is ``test_metadata_etag_survives_recorded_accesses``. Before the
split, ``access_count`` / ``last_accessed_at`` sat in the body and changed on every read,
so any honest hash of the body changed with them -- a strong validator was impossible.
"""

import contextlib
import errno
import fcntl
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from juniper_data.api import http_cache
from juniper_data.api.app import create_app
from juniper_data.api.http_cache import MAX_PRECONDITION_FIELD_LENGTH, body_etag, combine_field_lines, if_match_fails, if_none_match_fails_write, if_none_match_hits, strong_etag, weak_etag, write_preconditions_hold
from juniper_data.api.routes import datasets
from juniper_data.api.settings import Settings
from juniper_data.core.models import DatasetMeta, PublicDatasetMeta
from juniper_data.storage import local_fs
from juniper_data.storage.base import DatasetStore, StorageContainmentError
from juniper_data.storage.local_fs import LocalFSDatasetStore
from juniper_data.storage.memory import InMemoryDatasetStore

if TYPE_CHECKING:
    from juniper_data.storage.base import StagedSave

COUNTERS = ("access_count", "last_accessed_at")


@pytest.fixture
def store() -> InMemoryDatasetStore:
    """In-memory store the app under test is wired to."""
    return InMemoryDatasetStore()


@pytest.fixture
def client(store: InMemoryDatasetStore, tmp_path) -> TestClient:
    """A test client over ``store``, with an existing storage directory for readiness."""
    storage = tmp_path / "juniper_data_storage"
    storage.mkdir()
    app = create_app(settings=Settings(storage_path=str(storage)))
    datasets.set_store(store)
    return TestClient(app)


def _create(client: TestClient, *, seed: int = 1, name: str | None = None) -> str:
    """Create a small spiral dataset and return its id."""
    body: dict = {"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": seed}, "persist": True}
    if name is not None:
        body["name"] = name
    response = client.post("/v1/datasets", json=body)
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


def _stored_meta(dataset_id: str, **overrides) -> DatasetMeta:
    """A hand-built stored metadata record for tests that need exact field values."""
    fields = {
        "dataset_id": dataset_id,
        "generator": "spiral",
        "generator_version": "3.0.0",
        "params": {"seed": 1},
        "n_samples": 4,
        "n_features": 2,
        "n_train": 2,
        "n_test": 2,
        "created_at": datetime(2026, 9, 22, 20, 0, tzinfo=UTC),
        "checksum": "ab" * 32,
    }
    fields.update(overrides)
    return DatasetMeta(**fields)


def _arrays() -> dict[str, np.ndarray]:
    x = np.arange(8, dtype=np.float32).reshape(4, 2)
    y = np.eye(2, dtype=np.float32)[[0, 1, 0, 1]]
    return {"X_train": x[:2], "y_train": y[:2], "X_test": x[2:], "y_test": y[2:]}


@pytest.fixture
def localfs(tmp_path) -> tuple[TestClient, LocalFSDatasetStore]:
    """A client over a LocalFS store: the store the service wires, and the one that validates ids."""
    storage = tmp_path / "juniper_data_storage"
    storage.mkdir()
    store = LocalFSDatasetStore(storage)
    app = create_app(settings=Settings(storage_path=str(storage)))
    datasets.set_store(store)
    return TestClient(app), store


class _ClosingProbe:
    """Wraps an artifact stream and records whether the route closed it."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        return next(self._inner)

    def close(self) -> None:
        self.closed = True
        getattr(self._inner, "close", lambda: None)()


def _probe_streams(store, monkeypatch: pytest.MonkeyPatch) -> list[_ClosingProbe]:
    """Wrap every artifact stream ``store`` opens; the returned list fills as the route opens them."""
    opened: list[_ClosingProbe] = []
    real_open = store.open_artifact_stream

    def probing_open(*args, **kwargs):
        stream = real_open(*args, **kwargs)
        if stream is None:
            return None
        opened.append(_ClosingProbe(stream))
        return opened[-1]

    monkeypatch.setattr(store, "open_artifact_stream", probing_open)
    return opened


def _file_lock_is_held(store: LocalFSDatasetStore, dataset_id: str) -> bool:
    """Whether LocalFS's cross-process lock on ``dataset_id`` is held right now.

    ``flock(2)`` locks belong to an open file DESCRIPTION, so a second ``open`` of the lock file
    is refused a non-blocking lock while the store holds one -- even in this process, which is
    what lets a single-process test see the lock another PROCESS would wait on.
    """
    return _flock_is_held(store._lock_path(dataset_id))


def _flock_is_held(path: Path) -> bool:
    """Whether anyone -- this process included -- holds an ``flock`` on the file at ``path`` right now."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _root_is_locked_exclusively(storage: Path) -> bool:
    """Whether anyone holds an EXCLUSIVE ``flock`` on the storage root: a shared probe is refused only then."""
    fd = os.open(storage, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


class _LockHolder:
    """Holds a ``threading.Lock`` in ANOTHER thread, as a writer mid-save would, until released -- or ``limit`` seconds at most.

    The bound keeps a regression from hanging the suite: a read that waits for the lock on the
    event loop completes only when the holder gives up, and the test sees how long that took.
    """

    def __init__(self, lock: threading.Lock, limit: float = 10.0) -> None:
        self._lock = lock
        self._limit = limit
        self._held = threading.Event()
        self._release = threading.Event()
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        with self._lock:
            self._held.set()
            self._release.wait(self._limit)

    def __enter__(self) -> "_LockHolder":
        self._thread.start()
        assert self._held.wait(_RACE_TIMEOUT_SECONDS), "the holder never took the lock"
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._release.set()
        self._thread.join(_RACE_TIMEOUT_SECONDS)


class _AnnouncingLock:
    """Stands in for ``DatasetStore._version_lock``, and announces each thread that has to WAIT for it.

    That announcement is what makes the interleaving tests deterministic in both directions. A
    writer that takes the lock reports itself the moment it reaches it, and a writer that does
    not take it simply finishes; each outcome is an event the test observes, never a timeout it
    infers from.
    """

    def __init__(self, on_wait: Callable[[], None]) -> None:
        self._lock = threading.Lock()
        self._on_wait = on_wait

    def __enter__(self) -> "_AnnouncingLock":
        if not self._lock.acquire(blocking=False):
            self._on_wait()
            self._lock.acquire()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


# A bound on each wait below, so a broken run fails instead of hanging. The interleavings are
# decided by events, not by this bound: a passing run never comes near it.
_RACE_TIMEOUT_SECONDS = 30


def _race_a_held_conditional_patch(client: TestClient, store: LocalFSDatasetStore, monkeypatch: pytest.MonkeyPatch, dataset_id: str, rival: Callable[[TestClient], httpx.Response]) -> tuple[str, httpx.Response, httpx.Response]:
    """Send ``rival`` while a conditional PATCH sits between its PASSED check and its write.

    The PATCH's ``If-Match`` is current when it is checked. The PATCH is then held inside
    ``update_tags``, right after the check -- the window the store's locks exist to close --
    and the rival request is sent. Returns ``(first, patch_response, rival_response)``, where
    ``first`` is what the rival did while the PATCH was held: ``"waited for the lock"`` or
    ``"finished"``.
    """
    # A PATCH, not a GET, supplies the ETag: a GET schedules ``record_access``, which would take
    # the lock under test from outside the interleaving.
    etag = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["before"]}).headers["etag"]
    events: queue.Queue[str] = queue.Queue()
    monkeypatch.setattr(store, "_version_lock", _AnnouncingLock(lambda: events.put("waited for the lock")))
    in_window, release = threading.Event(), threading.Event()
    real_update_tags = store.update_tags

    def held_after_the_check(target: str, add_tags: list[str], remove_tags: list[str], precondition: Callable[[DatasetMeta], bool] | None = None) -> DatasetMeta | None:
        if precondition is None:
            return real_update_tags(target, add_tags, remove_tags)

        def check_then_hold(current: DatasetMeta) -> bool:
            passed = precondition(current)
            in_window.set()
            release.wait(_RACE_TIMEOUT_SECONDS)
            return passed

        return real_update_tags(target, add_tags, remove_tags, check_then_hold)

    monkeypatch.setattr(store, "update_tags", held_after_the_check)
    responses: dict[str, httpx.Response] = {}
    failures: list[Exception] = []

    def run(name: str, request: Callable[[], httpx.Response]) -> None:
        try:
            responses[name] = request()
        except Exception as exc:  # re-raised below, in the test's own thread
            failures.append(exc)
        finally:
            if name == "rival":
                events.put("finished")

    patch = threading.Thread(target=run, args=("patch", lambda: client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["mine"]}, headers={"If-Match": etag})))
    patch.start()
    assert in_window.wait(_RACE_TIMEOUT_SECONDS), "the conditional PATCH never reached its write"
    rival_thread = threading.Thread(target=run, args=("rival", lambda: rival(client)))
    rival_thread.start()
    first = events.get(timeout=_RACE_TIMEOUT_SECONDS)
    release.set()
    patch.join(_RACE_TIMEOUT_SECONDS)
    rival_thread.join(_RACE_TIMEOUT_SECONDS)
    if failures:
        raise failures[0]
    return first, responses["patch"], responses["rival"]


def _checked_before_it_existed(store: LocalFSDatasetStore, monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """Arrange for the NEXT ``get_meta`` to find nothing, and return the function that arms it.

    That is what a create sees when it checks for its dataset before a rival's copy exists: the
    route's existence check passes, the create generates, and only the store's own check, under
    the locks, can find that the dataset has appeared since. It is armed from inside the racing
    request, so no earlier read is affected.
    """
    real_get_meta = store.get_meta
    armed: list[bool] = []

    def get_meta(dataset_id: str) -> DatasetMeta | None:
        if armed:
            armed.clear()
            return None
        return real_get_meta(dataset_id)

    monkeypatch.setattr(store, "get_meta", get_meta)
    return lambda: armed.append(True)


class _OrderProbingLock:
    """Stands in for ``DatasetStore._version_lock``, and records on each entry whether the file lock is already held.

    Every taker must take ``_version_lock`` first and the file lock second, or two threads taking
    them in opposite orders deadlock. Nothing else observes the order: a delete that took the file
    lock first still ran under both, so the tests of what a delete holds passed, and only an
    interleaving test noticed -- as a timeout, 36 seconds later.
    """

    def __init__(self, store: LocalFSDatasetStore, dataset_id: str) -> None:
        self._lock = threading.Lock()
        self._store = store
        self._dataset_id = dataset_id
        self.file_lock_held_on_entry: list[bool] = []

    def __enter__(self) -> "_OrderProbingLock":
        self.file_lock_held_on_entry.append(_file_lock_is_held(self._store, self._dataset_id))
        self._lock.acquire()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


def _entry_exists(path: str | os.PathLike[str], dir_fd: int | None) -> bool:
    try:
        os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


class _NoFreeInodes:
    """Stands in for ``os`` inside ``local_fs``: creating a file or a directory fails with ENOSPC, as on a volume out of inodes.

    Only ``local_fs``'s own ``os.open`` and ``os.mkdir`` calls see it: the lock directory, the
    stripes, and the storage root it opens. Opening what exists needs no inode, and neither does
    anything else a delete does. A name is looked up relative to ``dir_fd``, as the call does.
    """

    def __getattr__(self, name: str) -> object:
        return getattr(os, name)

    @staticmethod
    def open(path: str | os.PathLike[str], flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if flags & os.O_CREAT and not _entry_exists(path, dir_fd):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), os.fspath(path))
        return os.open(path, flags, mode, dir_fd=dir_fd)

    @staticmethod
    def mkdir(path: str | os.PathLike[str], mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        if not _entry_exists(path, dir_fd):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), os.fspath(path))
        os.mkdir(path, mode, dir_fd=dir_fd)


def _probe_create_phases(store: LocalFSDatasetStore, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[tuple[bool, bool]]]:
    """Record, at each half of a create's save, whether ``_version_lock`` and the dataset's file lock are held.

    ``stage`` writes the compressed artifact and must hold neither; ``commit`` checks nothing and
    renames, and must hold both. Returns the two lists, which fill as creates run.
    """
    phases: dict[str, list[tuple[bool, bool]]] = {"stage": [], "commit": []}
    if not hasattr(store, "stage_save"):
        # A tree from before staging: the whole save is what a create does under its locks, and
        # nothing is staged -- which is what ``stage`` staying empty then says.
        real_save = store.save

        def probing_save(dataset_id: str, meta: DatasetMeta, arrays: dict[str, np.ndarray]) -> None:
            phases["commit"].append((store._version_lock.locked(), _file_lock_is_held(store, dataset_id)))
            real_save(dataset_id, meta, arrays)

        monkeypatch.setattr(store, "save", probing_save)
        return phases
    real_stage = store.stage_save

    def probing_stage(dataset_id: str, arrays: dict[str, np.ndarray]) -> "StagedSave":
        phases["stage"].append((store._version_lock.locked(), _file_lock_is_held(store, dataset_id)))
        staged = real_stage(dataset_id, arrays)
        real_commit = staged.commit

        def probing_commit(meta: DatasetMeta) -> None:
            phases["commit"].append((store._version_lock.locked(), _file_lock_is_held(store, dataset_id)))
            real_commit(meta)

        staged.commit = probing_commit  # type: ignore[method-assign]
        return staged

    monkeypatch.setattr(store, "stage_save", probing_stage)
    return phases


def _lead_out_of_the_store(stored: Path, outside: Path) -> None:
    """Replace a stored file with an absolute symlink to a copy outside the storage root: valid content, in the wrong place."""
    outside.mkdir(exist_ok=True)
    elsewhere = outside / stored.name
    elsewhere.write_bytes(stored.read_bytes())
    stored.unlink()
    stored.symlink_to(elsewhere)


@contextlib.asynccontextmanager
async def _raw_header_client(store: InMemoryDatasetStore, tmp_path) -> AsyncIterator[tuple[httpx.AsyncClient, list[bytes]]]:
    """An async client that hands the app RAW header bytes, and a record of the precondition bytes it received.

    Starlette's TestClient re-encodes a non-ASCII header value as UTF-8 -- ``b"\\xa0"`` arrives
    as ``b"\\xc2\\xa0"`` -- so it cannot deliver an obs-text byte the way a real server passes it
    through. ``httpx.ASGITransport`` puts the bytes in the ASGI scope as sent, and the record
    lets each test prove that they arrived.
    """
    storage = tmp_path / "juniper_data_storage"
    storage.mkdir()
    app = create_app(settings=Settings(storage_path=str(storage)))
    datasets.set_store(store)
    received: list[bytes] = []

    async def recording_app(scope, receive, send) -> None:
        if scope["type"] == "http":
            received.extend(value for name, value in scope["headers"] if name in (b"if-match", b"if-none-match"))
        await app(scope, receive, send)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=recording_app), base_url="http://testserver") as client:
        yield client, received


async def _create_over(client: httpx.AsyncClient) -> str:
    """``_create`` for an async client."""
    response = await client.post("/v1/datasets", json={"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": 1}, "persist": True})
    assert response.status_code == 201, response.text
    return response.json()["dataset_id"]


@pytest.mark.unit
class TestIfNoneMatchParsing:
    """The comparison rules of RFC 9110 §13.1.2, and the safe direction on garbage."""

    ETAG = strong_etag("abc")

    def test_absent_or_empty_matches_nothing(self) -> None:
        assert not if_none_match_hits(None, self.ETAG)
        assert not if_none_match_hits("", self.ETAG)

    def test_star_matches_any_current_representation(self) -> None:
        assert if_none_match_hits("*", self.ETAG)
        assert if_none_match_hits("  *  ", self.ETAG)

    def test_list_form_and_weak_comparison(self) -> None:
        assert if_none_match_hits('"zzz", "abc"', self.ETAG)
        assert if_none_match_hits('W/"abc"', self.ETAG)
        assert not if_none_match_hits('"abd"', self.ETAG)

    def test_a_comma_inside_an_opaque_tag_is_not_a_list_separator(self) -> None:
        # Splitting on "," would cut '"x,abc"' into '"x' and 'abc"' -- neither of which
        # is a tag -- and a sloppier split could match the fragment. Scanning for quoted
        # tags keeps the tag whole.
        assert not if_none_match_hits('"x,abc"', self.ETAG)
        assert if_none_match_hits('"x,abc"', strong_etag("x,abc"))

    def test_unparseable_field_serves_the_full_body(self) -> None:
        assert not if_none_match_hits("abc", self.ETAG)
        # A tag embedded in garbage is not a list element; scanning without the grammar
        # check would find it.
        assert not if_none_match_hits('foo"abc"bar', self.ETAG)
        assert not if_none_match_hits('x, "abc"', self.ETAG)

    def test_empty_list_elements_are_allowed(self) -> None:
        assert if_none_match_hits('"zzz", , "abc"', self.ETAG)

    def test_if_match_uses_the_strong_comparison(self) -> None:
        assert not if_match_fails('"abc"', self.ETAG)
        assert not if_match_fails("*", self.ETAG)
        assert if_match_fails('W/"abc"', self.ETAG)
        assert if_match_fails('"abd"', self.ETAG)
        assert if_match_fails("garbage", self.ETAG), "a precondition the server cannot read must not pass"
        assert not if_match_fails(None, self.ETAG), "no header, no precondition"

    def test_list_valued_header_lines_are_combined(self) -> None:
        assert combine_field_lines(['"zzz"', '"abc"']) == '"zzz", "abc"'
        assert combine_field_lines(None) is None

    def test_a_field_over_the_length_cap_is_malformed(self) -> None:
        # Defence in depth beside the linear grammar: a longer field is never parsed. At the
        # cap it still is; one character over, each direction treats it as unreadable.
        at_cap = '"abc"' + " " * (MAX_PRECONDITION_FIELD_LENGTH - len('"abc"'))
        over_cap = at_cap + " "
        assert len(at_cap) == MAX_PRECONDITION_FIELD_LENGTH
        assert if_none_match_hits(at_cap, self.ETAG), "at the cap the field is still read"
        assert not if_match_fails(at_cap, self.ETAG), "at the cap the field is still read"
        assert not if_none_match_hits(over_cap, self.ETAG), "a read: an unreadable If-None-Match names nothing"
        assert if_match_fails(over_cap, self.ETAG), "an unreadable If-Match fails"
        assert if_none_match_fails_write(over_cap, self.ETAG), "a write fails closed"
        assert not if_none_match_hits("*" + " " * MAX_PRECONDITION_FIELD_LENGTH, self.ETAG), "the cap applies to * as well"
        # The cap counts the COMBINED field: two lines, each under it, that join to over it.
        lines = ['"abc"', " " * (MAX_PRECONDITION_FIELD_LENGTH - 4)]
        assert all(len(line) <= MAX_PRECONDITION_FIELD_LENGTH for line in lines)
        assert not if_none_match_hits(combine_field_lines(lines), self.ETAG)

    def test_a_write_fails_closed_on_an_unreadable_if_none_match(self) -> None:
        # A read treats an unreadable If-None-Match as naming nothing, because a wrong 304 is
        # the harm there; a write treats it as a failed precondition, because proceeding
        # under a condition the server could not read is the harm here.
        for garbage in ("abc", 'foo"abc"bar', 'x, "abc"', '"unterminated'):
            assert not if_none_match_hits(garbage, self.ETAG), garbage
            assert if_none_match_fails_write(garbage, self.ETAG), garbage
            assert not write_preconditions_hold(None, garbage, self.ETAG), garbage
        # A well-formed field keeps its meaning on a write.
        assert if_none_match_fails_write("*", self.ETAG)
        assert if_none_match_fails_write('"zzz", W/"abc"', self.ETAG), "weak comparison, as on a read"
        assert not if_none_match_fails_write('"zzz"', self.ETAG)
        assert not if_none_match_fails_write("", self.ETAG), "an empty field is an empty list and names nothing"
        assert not if_none_match_fails_write(None, self.ETAG), "no header, no precondition"


@pytest.mark.unit
class TestStarTakesOnlySpacesAndTabs:
    """``*`` may be wrapped in RFC 9110 OWS -- spaces and tabs -- and in nothing else.

    ``str.strip()`` also removes NBSP (0xA0) and NEL (0x85), which a server passes through as
    obs-text, so a ``*`` wrapped in either read as ``*``: a 304 where the full body was owed, an
    ``If-Match`` that held, and a PATCH that should have failed closed wrote. Each test sends RAW
    header bytes (``_raw_header_client``) and checks the app received them as sent.
    """

    WRAPPED = (b"\xa0*", b"*\xa0", b"\x85*", b"*\x85")

    @pytest.mark.asyncio
    async def test_on_a_read_a_star_wrapped_in_nbsp_or_nel_is_malformed(self, store: InMemoryDatasetStore, tmp_path) -> None:
        async with _raw_header_client(store, tmp_path) as (client, received):
            dataset_id = await _create_over(client)
            for raw in self.WRAPPED:
                full = await client.get(f"/v1/datasets/{dataset_id}", headers={"If-None-Match": raw})
                assert received[-1] == raw, "the app must receive the byte as sent, or this proves nothing"
                assert full.status_code == 200, raw
                assert full.json()["dataset_id"] == dataset_id, "a malformed If-None-Match names nothing: the full body"
                artifact = await client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": raw})
                assert artifact.status_code == 200, raw
                assert artifact.content
                assert (await client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": raw})).status_code == 412, raw

    @pytest.mark.asyncio
    async def test_on_the_patch_a_star_wrapped_in_nbsp_or_nel_fails_closed(self, store: InMemoryDatasetStore, tmp_path) -> None:
        async with _raw_header_client(store, tmp_path) as (client, received):
            dataset_id = await _create_over(client)
            for raw in self.WRAPPED:
                for field in ("If-Match", "If-None-Match"):
                    response = await client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["blind"]}, headers={field: raw})
                    assert received[-1] == raw, "the app must receive the byte as sent, or this proves nothing"
                    assert response.status_code == 412, (field, raw)
                    assert "blind" not in store.get_meta(dataset_id).tags, (field, raw)

    @pytest.mark.asyncio
    async def test_a_star_wrapped_in_spaces_and_tabs_is_still_a_star(self, store: InMemoryDatasetStore, tmp_path) -> None:
        # The control for the two tests above: the same raw path, with the whitespace RFC 9110
        # does allow, still reads as ``*``.
        async with _raw_header_client(store, tmp_path) as (client, received):
            dataset_id = await _create_over(client)
            for raw in (b" * ", b"\t*\t"):
                assert (await client.get(f"/v1/datasets/{dataset_id}", headers={"If-None-Match": raw})).status_code == 304, raw
                assert received[-1] == raw
                assert (await client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": raw})).status_code == 200, raw
                assert (await client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["x"]}, headers={"If-None-Match": raw})).status_code == 412, raw
            written = await client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["star"]}, headers={"If-Match": b" * "})
            assert written.status_code == 200
            assert "star" in written.json()["tags"]


# Run in a CHILD interpreter, so a parse that never returns cannot hang the suite: the parent
# kills it at the bound and fails. The module is loaded from the exact file this process
# imported, so the child exercises the code under test -- in the non-vacuity harness, the
# mutated scratch copy -- and never an installed copy.
_PARSE_HOSTILE_FIELDS = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("http_cache_under_test", sys.argv[1])
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
etag = cache.strong_etag("abc")
fields = json.loads(sys.argv[2])
print(json.dumps([[cache.if_none_match_hits(f, etag), cache.if_match_fails(f, etag), cache.write_preconditions_hold(None, f, etag)] for f in fields]))
"""


@pytest.mark.unit
class TestEntityTagListRunsInLinearTime:
    """The list grammar runs on the event loop (GET) and under ``_version_lock`` (PATCH): no backtracking.

    The first form of ``_ENTITY_TAG_LIST`` took about 1.8 s for ``", " * 22 + "x"`` and doubled
    per element, so each field below takes minutes or longer under it (``", " * 30 + "x"`` alone
    is about seven minutes on a dev box). The parse runs in a SUBPROCESS with a timeout, so a
    return of that form fails here in seconds instead of hanging CI. The bound is generous on
    purpose: this asserts "finishes", not a speed -- a wall-clock threshold flakes on a loaded
    runner.
    """

    HOSTILE = (", " * 30 + "x", ",\t\t" * 22 + "x", " , " * 26 + "x")
    BOUND_SECONDS = 30

    def test_hostile_fields_are_refused_well_inside_a_generous_bound(self) -> None:
        # Each field must REACH the grammar. One the length cap refused first would pass
        # against the backtracking form too, and pin nothing.
        assert all(len(field) <= MAX_PRECONDITION_FIELD_LENGTH for field in self.HOSTILE)
        # At the bound, subprocess.run kills the child and raises; ``child`` then stays None.
        child: subprocess.CompletedProcess[str] | None = None
        with contextlib.suppress(subprocess.TimeoutExpired):
            child = subprocess.run([sys.executable, "-c", _PARSE_HOSTILE_FIELDS, http_cache.__file__, json.dumps(self.HOSTILE)], capture_output=True, text=True, timeout=self.BOUND_SECONDS)
        assert child is not None, f"parsing {len(self.HOSTILE)} hostile precondition fields did not finish in {self.BOUND_SECONDS} s: the entity-tag list grammar backtracks"
        assert child.returncode == 0, child.stderr
        # All malformed: a read names nothing, If-Match fails, and a write fails closed.
        assert json.loads(child.stdout) == [[False, True, False]] * len(self.HOSTILE)


@pytest.mark.unit
class TestMetadataValidator:
    """``GET /v1/datasets/{dataset_id}``: a strong ETag over the exact body."""

    def test_etag_is_strong_and_is_the_hash_of_the_exact_body(self, client: TestClient) -> None:
        dataset_id = _create(client)
        response = client.get(f"/v1/datasets/{dataset_id}")
        assert response.status_code == 200
        etag = response.headers["etag"]
        assert not etag.startswith("W/"), "the ruling is a STRONG validator"
        assert etag == body_etag(response.content)
        assert response.headers["cache-control"] == "private, no-cache"

    def test_metadata_etag_survives_recorded_accesses(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        first = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        store.record_access(dataset_id)
        store.record_access(dataset_id)
        assert store.get_meta(dataset_id).access_count >= 2, "the counters really did move"
        assert client.get(f"/v1/datasets/{dataset_id}").headers["etag"] == first

    def test_body_carries_no_access_counter(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        store.record_access(dataset_id)
        body = client.get(f"/v1/datasets/{dataset_id}").json()
        for counter in COUNTERS:
            assert counter not in body
        assert store.get_meta(dataset_id).access_count >= 1, "stored, just not represented"

    def test_matching_if_none_match_answers_304_with_no_body(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        for header in (etag, "*", f'"elsewhere", {etag}', f"W/{etag}"):
            response = client.get(f"/v1/datasets/{dataset_id}", headers={"If-None-Match": header})
            assert response.status_code == 304, header
            assert response.content == b""
            assert response.headers["etag"] == etag
            assert response.headers["cache-control"] == "private, no-cache"

    def test_stale_if_none_match_gets_the_full_body(self, client: TestClient) -> None:
        dataset_id = _create(client)
        response = client.get(f"/v1/datasets/{dataset_id}", headers={"If-None-Match": '"not-this-one"'})
        assert response.status_code == 200
        assert response.json()["dataset_id"] == dataset_id

    def test_if_none_match_on_two_header_lines_is_one_list(self, client: TestClient) -> None:
        # RFC 9110 §5.3: the two lines are the same list. Typed ``str``, FastAPI would read
        # only the first line and serve a 200 the client did not need.
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        response = client.get(f"/v1/datasets/{dataset_id}", headers=[("If-None-Match", '"zzz"'), ("If-None-Match", etag)])
        assert response.status_code == 304

    def test_a_tag_edit_moves_the_etag_and_the_patch_carries_the_new_one(self, client: TestClient) -> None:
        dataset_id = _create(client)
        before = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        patched = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["edited"]})
        assert patched.status_code == 200
        assert "edited" in patched.json()["tags"]
        after = patched.headers["etag"]
        assert after != before
        assert after == body_etag(patched.content)
        assert client.get(f"/v1/datasets/{dataset_id}").headers["etag"] == after
        # A client holding the pre-edit copy must not be told it is current.
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-None-Match": before}).status_code == 200

    def test_a_304_is_recorded_as_an_access(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        count_before = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-None-Match": etag}).status_code == 304
        assert client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"] == count_before + 1

    def test_bytes_match_fastapi_rendering_of_the_public_model(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        """The route renders its own body; it must be the body FastAPI would have sent.

        The reference is a real ``response_model=PublicDatasetMeta`` route, so the test
        follows FastAPI's rendering rather than asserting which encoder FastAPI uses --
        that choice has moved across FastAPI versions. The fixture must DISCRIMINATE: a
        float in exponent form is where pydantic-core (``1e-7``) and ``json.dumps``
        (``1e-07``) disagree, so a route rendering through the other encoder fails here
        instead of silently changing the wire format and hashing bytes nobody sent.
        """
        meta = _stored_meta("exact-bytes", params={"noise": 1e-07, "seed": 1}, description="héllo")
        store.save("exact-bytes", meta, _arrays())
        reference = FastAPI()

        @reference.get("/m", response_model=PublicDatasetMeta)
        def _m() -> DatasetMeta:
            return store.get_meta("exact-bytes")

        expected = TestClient(reference).get("/m").content
        other_encoder = JSONResponse(content=TypeAdapter(PublicDatasetMeta).dump_python(store.get_meta("exact-bytes"), mode="json")).body
        assert expected != other_encoder, "the fixture must separate the two encoders, or this test proves nothing"
        assert "héllo".encode() in expected
        assert client.get("/v1/datasets/exact-bytes").content == expected


@pytest.mark.unit
class TestArtifactValidator:
    """``GET /v1/datasets/{dataset_id}/artifact``: the stored checksum as a WEAK ETag, ``W/"<checksum>"``."""

    def test_etag_is_the_stored_checksum(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact")
        assert response.status_code == 200
        checksum = store.get_meta(dataset_id).checksum
        assert checksum
        # WEAK (owner ruling 2026-09-23): the checksum covers the arrays, not the served bytes.
        assert response.headers["etag"] == weak_etag(checksum)
        assert response.headers["cache-control"] == "private, no-cache"
        assert response.headers["content-disposition"] == f"attachment; filename={dataset_id}.npz"

    def test_matching_if_none_match_answers_304_and_reads_no_artifact(self, client: TestClient, store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset_id = _create(client)
        full = client.get(f"/v1/datasets/{dataset_id}/artifact")
        opened: list[str] = []
        real_open, real_bytes = store.open_artifact_stream, store.get_artifact_bytes

        def counting_open(*args, **kwargs):
            opened.append("open")
            return real_open(*args, **kwargs)

        def counting_bytes(*args, **kwargs):
            opened.append("bytes")
            return real_bytes(*args, **kwargs)

        # Both artifact readers: a 304 that read the whole artifact through get_artifact_bytes
        # would be as wrong as one that opened the stream.
        monkeypatch.setattr(store, "open_artifact_stream", counting_open)
        monkeypatch.setattr(store, "get_artifact_bytes", counting_bytes)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": full.headers["etag"]})
        assert response.status_code == 304
        assert response.content == b""
        assert response.headers["etag"] == full.headers["etag"]
        assert opened == [], "a 304 must be decided before the artifact is opened"

    def test_stale_if_none_match_gets_the_same_full_body(self, client: TestClient) -> None:
        dataset_id = _create(client)
        full = client.get(f"/v1/datasets/{dataset_id}/artifact")
        again = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": '"stale"'})
        assert again.status_code == 200
        assert again.content == full.content

    def test_a_dataset_without_a_checksum_has_no_etag_but_star_still_matches(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        # No checksum, no validator: a tagged If-None-Match can name nothing and the full body
        # is served. ``*`` is different -- RFC 9110 §13.1.2 makes it match any CURRENT
        # representation, validator or not, so it answers 304.
        store.save("no-checksum", _stored_meta("no-checksum", checksum=None), _arrays())
        tagged = client.get("/v1/datasets/no-checksum/artifact", headers={"If-None-Match": '"anything"'})
        assert tagged.status_code == 200
        assert "etag" not in tagged.headers
        assert tagged.content
        assert client.get("/v1/datasets/no-checksum/artifact", headers={"If-None-Match": "*"}).status_code == 304

    def test_a_304_on_the_artifact_is_recorded_as_an_access(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}/artifact").headers["etag"]
        count_before = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": etag}).status_code == 304
        assert client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"] == count_before + 1

    def test_stale_if_match_is_412_before_any_artifact_is_served(self, client: TestClient) -> None:
        dataset_id = _create(client)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": '"stale"'})
        assert response.status_code == 412

    def test_if_match_cannot_name_a_weak_artifact_tag_but_star_matches(self, client: TestClient) -> None:
        # If-Match compares STRONGLY, so the artifact's weak tag never satisfies it -- not even
        # its own current value. That is the cost the weak ruling accepted; ``*`` still works.
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}/artifact").headers["etag"]
        assert etag.startswith("W/")
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": etag}).status_code == 412
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": "*"}).status_code == 200

    def test_unreadable_metadata_still_serves_the_artifact(self, client: TestClient, store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch) -> None:
        # Before validators existed this route never read the metadata, so a corrupt
        # metadata document still served the artifact. It must still, just without an ETag.
        dataset_id = _create(client)

        def broken_get_meta(_dataset_id: str) -> DatasetMeta:
            raise ValueError("truncated metadata document")

        monkeypatch.setattr(store, "get_meta", broken_get_meta)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact")
        assert response.status_code == 200
        assert response.content
        assert "etag" not in response.headers

    def test_an_artifact_304_carries_its_caching_fields(self, client: TestClient) -> None:
        # RFC 9110 §15.4.5: a 304 carries the Cache-Control (and validator) the 200 would have.
        # Dropping Cache-Control lets a cache that stored the 200 lose "private, no-cache".
        dataset_id = _create(client)
        full = client.get(f"/v1/datasets/{dataset_id}/artifact")
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": full.headers["etag"]})
        assert response.status_code == 304
        assert response.headers["cache-control"] == full.headers["cache-control"] == "private, no-cache"
        assert response.headers["etag"] == full.headers["etag"]

    def test_a_412_on_the_artifact_is_not_an_access(self, client: TestClient) -> None:
        # Nothing was read: a failed If-Match stops the request before the artifact is touched.
        dataset_id = _create(client)
        count_before = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": '"stale"'}).status_code == 412
        assert client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"] == count_before

    def test_the_unreadable_metadata_warning_names_the_type_and_carries_no_caller_text(self, client: TestClient, store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        # The id in the URL is caller-supplied, and an exception message can carry it -- this
        # one does, deliberately. A traceback at WARNING would put it in a log record at
        # request rate; ERR-08 keeps caller strings out of log records.
        caller_id = "caller-chosen-7f3a91"
        store.save(caller_id, _stored_meta(caller_id), _arrays())
        real_get_meta = store.get_meta
        failed: list[str] = []

        def unreadable_once(dataset_id: str) -> DatasetMeta | None:
            # Only the route's read fails; record_access, which runs after it, reads normally.
            if not failed:
                failed.append(dataset_id)
                raise RuntimeError(f"cannot decode the metadata document of {dataset_id!r}")
            return real_get_meta(dataset_id)

        monkeypatch.setattr(store, "get_meta", unreadable_once)
        with caplog.at_level(logging.WARNING, logger="juniper_data"):
            response = client.get(f"/v1/datasets/{caller_id}/artifact")
        assert response.status_code == 200, "the artifact is still served, without a validator"
        assert "etag" not in response.headers
        assert failed == [caller_id], "the route's metadata read must actually have failed"
        loud = [record for record in caplog.records if record.levelno >= logging.WARNING]
        # Formatting renders any traceback too, which is where the caller's text would travel.
        assert not any(caller_id in logging.Formatter("%(message)s").format(record) for record in loud)
        ours = [record for record in loud if record.name.startswith("juniper_data")]
        assert [record.name for record in ours] == ["juniper_data.api.routes.datasets"]
        (record,) = ours
        assert record.exc_info is None, "no traceback above DEBUG"
        assert "RuntimeError" in record.getMessage(), "the exception TYPE is what the operator gets"


@pytest.mark.unit
class TestPreconditionsRespectExistence:
    """RFC 9110 §13.2.1: a precondition is never answered for a target that would 404 -- and never ignored for one that would be served."""

    def test_a_deleted_artifact_is_404_even_when_if_none_match_matches(self, tmp_path) -> None:
        storage = tmp_path / "juniper_data_storage"
        storage.mkdir()
        store = LocalFSDatasetStore(storage)
        app = create_app(settings=Settings(storage_path=str(storage)))
        datasets.set_store(store)
        client = TestClient(app)
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}/artifact").headers["etag"]
        store._npz_path(dataset_id).unlink()  # metadata left behind, artifact gone
        assert client.get(f"/v1/datasets/{dataset_id}/artifact").status_code == 404
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": etag}).status_code == 404
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": "*"}).status_code == 404

    @staticmethod
    def _orphan(localfs: tuple[TestClient, LocalFSDatasetStore]) -> tuple[TestClient, LocalFSDatasetStore, str]:
        """A dataset whose metadata is gone and whose artifact is still on disk."""
        client, store = localfs
        dataset_id = _create(client)
        store._meta_path(dataset_id).unlink()
        assert not store.exists(dataset_id), "LocalFS exists() needs the metadata too"
        assert client.get(f"/v1/datasets/{dataset_id}/artifact").status_code == 200, "unconditionally, an orphan is served"
        return client, store, dataset_id

    def test_an_orphaned_artifact_is_412_on_a_failing_if_match_and_the_stream_is_closed(self, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # RFC 9110 §13.1.1: when If-Match is false the method MUST NOT be performed. The
        # orphan has no validator, so no listed tag can name it -- but it would be served.
        client, store, dataset_id = self._orphan(localfs)
        opened = _probe_streams(store, monkeypatch)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": '"not-current"'})
        assert response.status_code == 412
        assert [probe.closed for probe in opened] == [True], "a stream that will not be sent must be closed"

    def test_an_orphaned_artifact_still_satisfies_if_match_star(self, localfs: tuple[TestClient, LocalFSDatasetStore]) -> None:
        # ``*`` names any CURRENT representation, and the artifact is one.
        client, _store, dataset_id = self._orphan(localfs)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": "*"})
        assert response.status_code == 200
        assert response.content
        assert "etag" not in response.headers

    def test_an_orphaned_artifact_answers_if_none_match_star_with_a_304(self, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # What read_precondition_status answers for any present representation: ``*`` hits.
        # A listed tag cannot name a representation with no validator, so the body is served.
        client, store, dataset_id = self._orphan(localfs)
        opened = _probe_streams(store, monkeypatch)
        response = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": "*"})
        assert response.status_code == 304
        assert response.content == b""
        assert response.headers["cache-control"] == "private, no-cache"
        assert "etag" not in response.headers
        assert [probe.closed for probe in opened] == [True], "a stream that will not be sent must be closed"
        tagged = client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": '"anything"'})
        assert tagged.status_code == 200
        assert tagged.content

    def test_a_dataset_that_is_really_absent_is_404_under_every_precondition(self, localfs: tuple[TestClient, LocalFSDatasetStore]) -> None:
        # Neither metadata nor artifact: there is no representation for a precondition to judge.
        client, _store = localfs
        for headers in ({}, {"If-Match": "*"}, {"If-Match": '"x"'}, {"If-None-Match": "*"}, {"If-None-Match": '"x"'}):
            assert client.get("/v1/datasets/never-created-0000/artifact", headers=headers).status_code == 404, headers

    def test_an_invalid_id_on_the_artifact_route_is_the_normal_400_and_logs_no_warning(self, localfs: tuple[TestClient, LocalFSDatasetStore], caplog: pytest.LogCaptureFixture) -> None:
        # A malformed id is the CALLER's error, not unreadable metadata: it must not take the
        # "serve without a validator" fallback, whose warning would fire at request rate, and
        # it gets the same 400 the metadata route gives the same id.
        client, _store = localfs
        invalid = "CALLER$CONTROLLED"
        expected = client.get(f"/v1/datasets/{invalid}")
        assert expected.status_code == 400
        with caplog.at_level(logging.DEBUG, logger="juniper_data"):
            for headers in ({}, {"If-None-Match": "*"}, {"If-Match": '"x"'}):
                response = client.get(f"/v1/datasets/{invalid}/artifact", headers=headers)
                assert response.status_code == 400, headers
                assert response.json() == expected.json()
        loud = [record for record in caplog.records if record.levelno >= logging.WARNING]
        assert not any(invalid in logging.Formatter("%(message)s").format(record) for record in loud)
        ours = [record.getMessage() for record in loud if record.name.startswith("juniper_data")]
        assert ours == [], ours

    def test_a_metadata_file_that_leads_out_of_the_store_still_serves_the_artifact(self, localfs: tuple[TestClient, LocalFSDatasetStore], tmp_path) -> None:
        # A symlink in the storage directory that leads outside it is the STORE's fault. LocalFS
        # refuses to follow it -- correctly -- but it raised the error that means "the caller's
        # id is malformed", so this route re-raised it as a 400 that blamed the caller, where
        # 0.15.0 served the artifact. It is metadata the route cannot read: the artifact is
        # served without a validator, and a conditional request is judged as for an orphan.
        client, store = localfs
        dataset_id = _create(client)
        meta_path = store._meta_path(dataset_id)
        outside = tmp_path / "outside"
        outside.mkdir()
        elsewhere = outside / meta_path.name
        elsewhere.write_bytes(meta_path.read_bytes())  # valid metadata: only where it lives is wrong
        meta_path.unlink()
        meta_path.symlink_to(elsewhere)  # an absolute target, outside the storage root
        plain = client.get(f"/v1/datasets/{dataset_id}/artifact")
        assert plain.status_code == 200, "a storage fault must not be answered as the caller's 400"
        assert plain.content
        assert "etag" not in plain.headers
        for headers, expected in (({"If-None-Match": "*"}, 304), ({"If-None-Match": '"x"'}, 200), ({"If-Match": "*"}, 200), ({"If-Match": '"x"'}, 412)):
            assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers=headers).status_code == expected, headers
        # The store's own verdict: a storage fault, not a malformed id.
        from juniper_data.storage.base import InvalidDatasetIdError, StorageContainmentError

        with pytest.raises(StorageContainmentError) as refused:
            store.get_meta(dataset_id)
        assert not isinstance(refused.value, InvalidDatasetIdError)

    def test_serving_a_dataset_whose_metadata_leads_out_of_the_store_logs_nothing_that_carries_its_id(self, localfs: tuple[TestClient, LocalFSDatasetStore], tmp_path, caplog: pytest.LogCaptureFixture) -> None:
        # ``record_access`` reads the same metadata, so it failed the same way -- inside an
        # event-loop callback, where asyncio logged "Exception in callback" at ERROR with a
        # traceback ending in the store's message, which names the id. On every 200 and every
        # 304. ERR-08 keeps caller strings out of log records, and the route's own warning keeps
        # to the exception type.
        client, store = localfs
        dataset_id = _create(client)
        other = _create(client, seed=2)
        _lead_out_of_the_store(store._meta_path(dataset_id), tmp_path / "outside")
        with caplog.at_level(logging.DEBUG):
            assert client.get(f"/v1/datasets/{dataset_id}/artifact").status_code == 200
            assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-None-Match": "*"}).status_code == 304
            # Accesses are recorded on the recorder's thread, and reading any dataset's counters
            # waits for every access handed over before it.
            assert client.get(f"/v1/datasets/{other}/access").status_code == 200
        loud = [record for record in caplog.records if record.levelno >= logging.WARNING]
        assert [record.getMessage().split(";")[0] for record in loud if record.name.startswith("juniper_data")] == ["Artifact download: dataset metadata unreadable (StorageContainmentError)"] * 2, "control: the route's own warning, once per request, and no failed recording"
        assert [record.getMessage() for record in loud if record.name == "asyncio"] == [], "an event-loop callback failed"
        # Formatting renders any traceback too, which is where the id travelled.
        assert not any(dataset_id in logging.Formatter("%(message)s").format(record) for record in loud)

    @pytest.mark.parametrize("target", ["metadata", "filter", "artifact-npz"])
    def test_a_stored_file_that_leads_out_of_the_store_is_a_generic_500(self, target: str, localfs: tuple[TestClient, LocalFSDatasetStore], tmp_path, caplog: pytest.LogCaptureFixture) -> None:
        # A storage fault, from a request that was fine. It was answered 400 "Invalid request
        # parameters", logged at DEBUG only -- and ``/filter`` answered 400 with the store's
        # message, which names the dataset id, as its detail. On the artifact route an ``.npz``
        # that leads out of the root is refused, as it should be, but with the same 400.
        client, store = localfs
        dataset_id = _create(client)
        stored = store._npz_path(dataset_id) if target == "artifact-npz" else store._meta_path(dataset_id)
        _lead_out_of_the_store(stored, tmp_path / "outside")
        url = {"metadata": f"/v1/datasets/{dataset_id}", "filter": "/v1/datasets/filter", "artifact-npz": f"/v1/datasets/{dataset_id}/artifact"}[target]
        with caplog.at_level(logging.DEBUG, logger="juniper_data"):
            response = client.get(url)
        assert response.status_code == 500, response.text
        assert response.json() == {"detail": "Internal server error"}
        assert dataset_id not in response.text
        faults = [record for record in caplog.records if record.name.startswith("juniper_data") and record.levelno >= logging.ERROR]
        assert len(faults) == 1, [record.getMessage() for record in faults]
        assert "StorageContainmentError" in faults[0].getMessage(), "the operator gets the exception TYPE"
        assert faults[0].exc_info is None
        loud = [record for record in caplog.records if record.levelno >= logging.WARNING]
        assert not any(dataset_id in logging.Formatter("%(message)s").format(record) for record in loud)

    def test_a_malformed_filter_cursor_is_still_the_callers_400_naming_the_cursor(self, localfs: tuple[TestClient, LocalFSDatasetStore]) -> None:
        # /filter used to turn EVERY ValueError from the store into a 400 carrying its text, which
        # is how a malformed cursor got its own message. It now decodes the cursor before calling
        # the store: that error alone is the caller's, and it keeps its detail.
        from juniper_data.storage.base import decode_cursor

        client, _store = localfs
        response = client.get("/v1/datasets/filter", params={"cursor": "not-a-real-cursor"})
        assert response.status_code == 400
        with pytest.raises(ValueError) as refused:
            decode_cursor("not-a-real-cursor")
        assert response.json()["detail"] == str(refused.value), "the cursor's own error, not the handler's generic detail"


@pytest.mark.unit
class TestIfMatch:
    """If-Match: strong comparison, evaluated before If-None-Match (RFC 9110 §13.1.1, §13.2.2)."""

    def test_reads_honour_if_match(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": etag}).status_code == 200
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": "*"}).status_code == 200
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": '"stale"'}).status_code == 412
        # Strong comparison: a weak tag never satisfies If-Match, even with the same opaque value.
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": f"W/{etag}"}).status_code == 412

    def test_failed_if_match_wins_over_a_matching_if_none_match(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        response = client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": '"stale"', "If-None-Match": etag})
        assert response.status_code == 412

    def test_a_412_on_a_read_is_not_an_access(self, client: TestClient) -> None:
        dataset_id = _create(client)
        count_before = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": '"stale"'}).status_code == 412
        assert client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"] == count_before

    def test_an_empty_if_match_is_412_on_a_read(self, client: TestClient) -> None:
        # An empty field is a well-formed EMPTY list: it names no representation, so If-Match
        # fails. It is still a precondition -- a check that read an empty field as "no header"
        # would serve the body.
        dataset_id = _create(client)
        assert client.get(f"/v1/datasets/{dataset_id}", headers={"If-Match": ""}).status_code == 412
        assert client.get(f"/v1/datasets/{dataset_id}/artifact", headers={"If-Match": ""}).status_code == 412

    def test_an_empty_if_match_is_412_on_the_patch_and_writes_nothing(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        # The write's own gate: a route that decided "is this conditional?" by the field's
        # truthiness would skip the precondition and write.
        dataset_id = _create(client)
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["blind"]}, headers={"If-Match": ""})
        assert response.status_code == 412
        assert "blind" not in store.get_meta(dataset_id).tags


@pytest.mark.unit
class TestConditionalTagWrite:
    """PATCH .../tags as an optimistic-concurrency write (If-Match / If-None-Match -> 412)."""

    def test_the_patch_response_names_the_resource_its_etag_describes(self, client: TestClient) -> None:
        # The request target, .../tags, has no GET; Content-Location names the representation.
        dataset_id = _create(client)
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["loc"]})
        assert response.headers["content-location"] == f"/v1/datasets/{dataset_id}"
        assert response.headers["etag"] == client.get(f"/v1/datasets/{dataset_id}").headers["etag"]

    def test_current_if_match_applies_the_edit(self, client: TestClient) -> None:
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["cas"]}, headers={"If-Match": etag})
        assert response.status_code == 200
        assert "cas" in response.json()["tags"]

    def test_stale_if_match_is_412_and_writes_nothing(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        stale = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        assert client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["first"]}).status_code == 200
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["lost-update"]}, headers={"If-Match": stale})
        assert response.status_code == 412
        assert "lost-update" not in store.get_meta(dataset_id).tags

    def test_if_none_match_star_on_an_existing_dataset_is_412(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["x"]}, headers={"If-None-Match": "*"})
        assert response.status_code == 412
        assert "x" not in store.get_meta(dataset_id).tags

    def test_if_none_match_naming_the_current_tag_is_412_and_writes_nothing(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        # Not only ``*``: a tag naming the CURRENT representation fails the write too
        # (RFC 9110 §13.1.2), compared weakly as on a read -- so ``W/<etag>`` does as well.
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        for header in (etag, f'"elsewhere", {etag}', f"W/{etag}"):
            response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["named"]}, headers={"If-None-Match": header})
            assert response.status_code == 412, header
            assert "named" not in store.get_meta(dataset_id).tags, header

    def test_a_malformed_if_none_match_is_412_and_writes_nothing(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        # Fail CLOSED: a write does not proceed under a condition the server could not read.
        # A read serves the full body for the same field (``test_unparseable_field_serves_the_full_body``).
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        # Well-formed and naming nothing current, so only the length cap can refuse it.
        over_cap = ", ".join(['"elsewhere"'] * (MAX_PRECONDITION_FIELD_LENGTH // len('"elsewhere", ') + 1))
        assert len(over_cap) > MAX_PRECONDITION_FIELD_LENGTH
        for header in ("garbage", f"foo{etag}bar", '"unterminated', over_cap):
            response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["blind"]}, headers={"If-None-Match": header})
            assert response.status_code == 412, header[:40]
            assert "blind" not in store.get_meta(dataset_id).tags, header[:40]

    def test_a_well_formed_if_none_match_naming_another_tag_applies_the_edit(self, client: TestClient) -> None:
        # The control for the two tests above: a readable field that names nothing current,
        # and an empty one (an empty list), let the write through.
        dataset_id = _create(client)
        for tag, header in (("other", '"some-other-representation"'), ("empty", "")):
            response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": [tag]}, headers={"If-None-Match": header})
            assert response.status_code == 200, header
            assert tag in response.json()["tags"]

    def test_the_store_evaluates_the_precondition_against_current_metadata_and_writes_nothing_on_false(self, store: InMemoryDatasetStore) -> None:
        from juniper_data.storage.base import PreconditionFailedError

        store.save("guarded", _stored_meta("guarded", tags=["a"]), _arrays())
        seen: list[list[str]] = []

        def refuse(current: DatasetMeta) -> bool:
            seen.append(list(current.tags))
            return False

        with pytest.raises(PreconditionFailedError):
            store.update_tags("guarded", ["b"], [], refuse)
        assert seen == [["a"]], "the precondition must see the CURRENT metadata"
        assert store.get_meta("guarded").tags == ["a"], "a failed precondition must write nothing"


@pytest.mark.unit
class TestConditionalWriteIsAtomic:
    """The PATCH precondition is checked INSIDE the locks that guard the write, or it is a race.

    Every way to lose that passes every functional test -- a stale ``If-Match`` still gets its
    412 -- because the check still happens, just where another writer can slip in between it and
    the write. Each is pinned here: ``update_tags`` evaluating the precondition before it takes
    ``_version_lock``, or inside it but outside the cross-process lock; the route evaluating it
    itself and handing the store ``None``; and another writer that takes neither lock --
    ``PATCH /batch-tags`` and ``DELETE`` did, and a delete landing in the window turned the
    PATCH into a 200 for a dataset that was gone. So did a create: it checked that its dataset
    was absent before generating, and saved without the file lock.
    """

    def test_the_store_evaluates_the_precondition_under_its_version_lock(self, store: InMemoryDatasetStore) -> None:
        store.save("guarded", _stored_meta("guarded"), _arrays())
        assert not store._version_lock.locked(), "nothing else may hold the lock, or this proves nothing"
        held: list[bool] = []

        def check(_current: DatasetMeta) -> bool:
            held.append(store._version_lock.locked())
            return True

        store.update_tags("guarded", ["b"], [], check)
        assert held == [True], "the precondition must run while update_tags holds its lock"
        assert "b" in store.get_meta("guarded").tags

    def test_a_write_that_lands_after_the_route_and_before_the_store_is_412(self, client: TestClient, store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch) -> None:
        # Another writer wins the race at the latest point a route-level check could miss:
        # after the route has decided, before the store takes its lock. Only a precondition
        # the store evaluates under that lock sees the concurrent tag.
        dataset_id = _create(client)
        etag = client.get(f"/v1/datasets/{dataset_id}").headers["etag"]
        real_update_tags = store.update_tags

        def another_writer_first(target: str, add_tags: list[str], remove_tags: list[str], precondition=None):
            real_update_tags(target, ["concurrent"], [])
            return real_update_tags(target, add_tags, remove_tags, precondition)

        monkeypatch.setattr(store, "update_tags", another_writer_first)
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["mine"]}, headers={"If-Match": etag})
        tags = store.get_meta(dataset_id).tags
        assert "concurrent" in tags, "the other writer's edit must have landed, or no race was simulated"
        assert response.status_code == 412
        assert "mine" not in tags, "the stale write must not be applied over the concurrent one"

    def test_the_store_evaluates_the_precondition_under_its_cross_process_file_lock(self, tmp_path) -> None:
        # The half the version-lock test cannot see. ``_version_lock`` orders threads; on LocalFS
        # what orders PROCESSES is the flock ``_meta_write_lock`` takes. A precondition checked
        # inside the first but outside the second passed every other test in the suite, and let
        # a second process write between the check and the write.
        store = LocalFSDatasetStore(tmp_path / "storage")
        store.save("guarded", _stored_meta("guarded"), _arrays())
        held: list[bool] = []

        def check(_current: DatasetMeta) -> bool:
            held.append(_file_lock_is_held(store, "guarded"))
            return True

        store.update_tags("guarded", ["b"], [], check)
        assert held == [True], "the precondition must run while update_tags holds the cross-process lock"
        assert not _file_lock_is_held(store, "guarded"), "control: the probe must see the lock free once the write is done, or it proves nothing"

    def test_batch_tags_cannot_land_inside_a_conditional_patchs_window(self, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # batch-tags read the metadata and wrote it back in two unlocked hops, so its edit could
        # land between a conditional PATCH's passed check and its write. The PATCH then erased it,
        # and answered 200 although the If-Match it checked was stale by the time it wrote.
        client, store = localfs
        dataset_id = _create(client)
        first, patched, batch = _race_a_held_conditional_patch(client, store, monkeypatch, dataset_id, lambda c: c.patch("/v1/datasets/batch-tags", json={"dataset_ids": [dataset_id], "add_tags": ["batch"]}))
        assert first == "waited for the lock", "batch-tags finished inside the conditional PATCH's window"
        assert patched.status_code == 200
        assert batch.status_code == 200
        assert batch.json()["updated"] == [dataset_id]
        assert {"mine", "batch"} <= set(store.get_meta(dataset_id).tags), "both acknowledged edits must survive"

    def test_a_delete_cannot_land_inside_a_conditional_patchs_window(self, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # DELETE took no lock either. Inside the window it removed the dataset; the PATCH's write
        # then found nothing to update, and the PATCH answered 200 with an ETag anyway.
        client, store = localfs
        dataset_id = _create(client)
        first, patched, deleted = _race_a_held_conditional_patch(client, store, monkeypatch, dataset_id, lambda c: c.delete(f"/v1/datasets/{dataset_id}"))
        assert first == "waited for the lock", "DELETE finished inside the conditional PATCH's window"
        assert patched.status_code == 200, "the PATCH held the lock first, so its edit applied"
        assert "mine" in patched.json()["tags"]
        assert deleted.status_code == 204, "and then the delete ran"
        assert store.get_meta(dataset_id) is None

    @pytest.mark.parametrize("route", ["delete", "batch-delete", "cleanup-expired"])
    def test_every_route_that_deletes_holds_both_locks(self, route: str, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # The interleaving above proves the in-process half for DELETE; this proves the store's
        # own delete runs under BOTH locks, from each route that deletes -- the flock is what
        # orders a delete against a PATCH in another worker process.
        client, store = localfs
        store.save("doomed", _stored_meta("doomed", expires_at=datetime(2026, 9, 22, 21, 0, tzinfo=UTC)), _arrays())
        held: list[tuple[bool, bool]] = []
        real_delete = store.delete

        def probing_delete(dataset_id: str) -> bool:
            held.append((store._version_lock.locked(), _file_lock_is_held(store, dataset_id)))
            return real_delete(dataset_id)

        monkeypatch.setattr(store, "delete", probing_delete)
        send = {
            "delete": lambda: client.delete("/v1/datasets/doomed"),
            "batch-delete": lambda: client.post("/v1/datasets/batch-delete", json={"dataset_ids": ["doomed"]}),
            "cleanup-expired": lambda: client.post("/v1/datasets/cleanup-expired"),
        }
        assert send[route]().status_code in (200, 204)
        assert held == [(True, True)], f"{route} must delete under _version_lock and the cross-process lock"
        assert store.get_meta("doomed") is None

    def test_a_dataset_gone_by_the_write_is_404_and_carries_no_etag(self, client: TestClient, store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch) -> None:
        # Every delete in this service takes the locks now, but something outside them can still
        # remove a dataset between update_tags' read and its write: a file deleted by hand,
        # another host, another process on a store whose cross-process lock is the no-op.
        # update_meta reports that with False, which update_tags ignored -- answering 200 and a
        # new ETag for a dataset that was gone.
        dataset_id = _create(client)
        etag = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["before"]}).headers["etag"]
        real_update_meta = store.update_meta

        def removed_first(target: str, meta: DatasetMeta) -> bool:
            store.delete(target)  # the store's own delete, outside every lock, as an outside actor's would be
            return real_update_meta(target, meta)

        monkeypatch.setattr(store, "update_meta", removed_first)
        response = client.patch(f"/v1/datasets/{dataset_id}/tags", json={"add_tags": ["mine"]}, headers={"If-Match": etag})
        assert response.status_code == 404
        assert "etag" not in response.headers
        assert store.get_meta(dataset_id) is None

    def test_a_create_cannot_land_inside_a_conditional_patchs_window(self, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # A create checks that its dataset is absent BEFORE it generates, which can take seconds,
        # and saved without the file lock -- without any lock, unnamed. A create of the same id
        # whose check ran before the dataset existed could therefore save inside a conditional
        # PATCH's window, and the PATCH then wrote its stale copy over the create.
        client, store = localfs
        dataset_id = _create(client)
        arm = _checked_before_it_existed(store, monkeypatch)

        def create_again(c: TestClient) -> httpx.Response:
            arm()
            return c.post("/v1/datasets", json={"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": 1}, "persist": True, "tags": ["from-create"]})

        first, patched, created = _race_a_held_conditional_patch(client, store, monkeypatch, dataset_id, create_again)
        assert first == "waited for the lock", "the create saved inside the conditional PATCH's window"
        assert patched.status_code == 200
        assert created.status_code == 201
        assert created.json()["dataset_id"] == dataset_id, "the rival must create the SAME dataset, or nothing was raced"
        tags = store.get_meta(dataset_id).tags
        assert {"before", "mine"} <= set(tags), "the PATCH's acknowledged edit must survive"
        assert "from-create" not in tags, "a create must write nothing over a dataset that exists by then"

    @pytest.mark.parametrize("case", ["unnamed", "named", "different-content"])
    def test_a_late_create_of_an_existing_dataset_writes_nothing_and_answers_with_the_stored_one(self, case: str, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # Two creates of one id, both past the route's existence check: the later save replaced a
        # create already answered 201. Checking again under the locks makes create-if-absent
        # atomic, and the late create then describes the dataset that IS stored -- as a cache hit
        # does -- not the copy it did not write. For a NAMED dataset too, which also allocates a
        # version; and when the late create's content differs from what is stored, as an
        # unseeded generator's can under one id.
        client, store = localfs
        body = {"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": 1}, "persist": True}
        if case == "named":
            body["name"] = "late-named"
        early = client.post("/v1/datasets", json={**body, "tags": ["early"], "description": "early"})
        assert early.status_code == 201
        dataset_id = early.json()["dataset_id"]
        if case == "different-content":
            meta = store.get_meta(dataset_id)
            meta.checksum = "00" * 32
            assert store.update_meta(dataset_id, meta)
        stored = store._meta_path(dataset_id).read_bytes()
        _checked_before_it_existed(store, monkeypatch)()
        late = client.post("/v1/datasets", json={**body, "tags": ["late"], "description": "late", "ttl_seconds": 3600})
        assert late.status_code == 201
        assert late.json()["dataset_id"] == dataset_id, "the late create must name the SAME dataset, or nothing was raced"
        assert store._meta_path(dataset_id).read_bytes() == stored, "the late create must write nothing"
        described = late.json()["meta"]
        assert (described["tags"], described["description"], described["expires_at"]) == (["early"], "early", None), "the late create must describe what is stored"

    @pytest.mark.parametrize(("route", "name"), [("create", None), ("create", "named"), ("batch-create", None), ("batch-create", "named")], ids=["create-unnamed", "create-named", "batch-create-unnamed", "batch-create-named"])
    def test_every_route_that_creates_holds_both_locks(self, route: str, name: str | None, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # The flock is what orders a create against a PATCH in another worker process. A named
        # create took ``_version_lock`` only, to allocate its version; an unnamed one took nothing.
        client, store = localfs
        phases = _probe_create_phases(store, monkeypatch)
        item = {"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": 4}, "persist": True}
        if name is not None:
            item["name"] = name
        response = client.post("/v1/datasets", json=item) if route == "create" else client.post("/v1/datasets/batch-create", json={"datasets": [item]})
        assert response.status_code == 201, response.text
        assert phases["commit"] == [(True, True)], f"{route} must commit under _version_lock and the cross-process lock"

    def test_a_create_stages_its_artifact_before_it_takes_either_lock(self, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # Compressing and writing the artifact is the expensive part of a create -- seconds for a
        # large dataset -- and every other writer in the process, and in any other process on the
        # dataset's stripe, waits while the locks are held. It needs neither: only the check, the
        # version and the renames do.
        client, store = localfs
        phases = _probe_create_phases(store, monkeypatch)
        response = client.post("/v1/datasets", json={"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": 5}, "persist": True})
        assert response.status_code == 201, response.text
        assert phases["stage"] == [(False, False)], "the artifact must be written before either lock is taken"
        assert phases["commit"] == [(True, True)], "control: the commit holds both, and the probe can see them"

    @pytest.mark.parametrize("taker", ["record_access", "update_tags", "delete_under_lock", "save_versioned"])
    def test_every_lock_taker_enters_the_version_lock_before_the_file_lock(self, taker: str, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        # One order everywhere, or two threads taking the locks in opposite orders deadlock. The
        # stand-in records whether the file lock is already held as ``_version_lock`` is entered;
        # the probe inside the store's write shows the file lock IS held there, so a False on
        # entry is an observation, not a probe that cannot see the lock.
        store = LocalFSDatasetStore(tmp_path / "storage")
        if taker != "save_versioned":
            store.save("guarded", _stored_meta("guarded"), _arrays())
        order = _OrderProbingLock(store, "guarded")
        monkeypatch.setattr(store, "_version_lock", order)
        held_inside: list[bool] = []
        if taker == "save_versioned":
            # A create writes under the locks only in its commit; its staging holds neither.
            phases = _probe_create_phases(store, monkeypatch)
        else:
            write = {"record_access": "update_meta", "update_tags": "update_meta", "delete_under_lock": "delete"}[taker]
            real_write = getattr(store, write)

            def probing_write(*args: object) -> object:
                held_inside.append(_file_lock_is_held(store, "guarded"))
                return real_write(*args)

            monkeypatch.setattr(store, write, probing_write)
        call = {
            "record_access": lambda: store.record_access("guarded"),
            "update_tags": lambda: store.update_tags("guarded", ["b"], []),
            "delete_under_lock": lambda: store.delete_under_lock("guarded"),
            "save_versioned": lambda: store.save_versioned("guarded", _stored_meta("guarded"), _arrays()),
        }[taker]
        call()
        if taker == "save_versioned":
            held_inside = [file_lock for _version, file_lock in phases["commit"]]
        assert order.file_lock_held_on_entry == [False], f"{taker} must enter _version_lock once, BEFORE it takes the file lock"
        assert held_inside == [True], "control: the file lock is held inside, and the probe can see it"


@pytest.mark.unit
class TestLockStripes:
    """LocalFS's cross-process locks are a fixed set of stripes, created with the store.

    A lock file per dataset id was created on first use and never removed. Once every delete
    and every batch edit took the lock, each request naming an absent id left one behind, and a
    delete could not unlink anything until it had created one.
    """

    def test_requests_naming_absent_ids_leave_no_lock_files(self, localfs: tuple[TestClient, LocalFSDatasetStore]) -> None:
        client, store = localfs
        before = sorted(store.base_path.rglob("*"))
        absent = ["absent-0000", "absent-0001", "absent-0002"]
        assert client.delete(f"/v1/datasets/{absent[0]}").status_code == 404
        assert client.post("/v1/datasets/batch-delete", json={"dataset_ids": absent}).json()["not_found"] == absent
        assert client.patch("/v1/datasets/batch-tags", json={"dataset_ids": absent, "add_tags": ["x"]}).json()["not_found"] == absent
        assert client.patch(f"/v1/datasets/{absent[1]}/tags", json={"add_tags": ["x"]}).status_code == 404
        assert sorted(store.base_path.rglob("*")) == before, "a request naming an absent id must create nothing"
        assert sorted(path.name for path in (store.base_path / "locks").iterdir()) == [f"{stripe:x}.lock" for stripe in range(16)], "the stripes exist from the start"

    @pytest.mark.parametrize("route", ["delete", "batch-delete", "cleanup-expired"])
    def test_the_delete_paths_need_no_free_inode(self, route: str, localfs: tuple[TestClient, LocalFSDatasetStore], monkeypatch: pytest.MonkeyPatch) -> None:
        # On a volume out of inodes, the operations that free space were the ones that failed:
        # each had to create its dataset's lock file first, and answered 500 with nothing deleted.
        client, store = localfs
        store.save("doomed", _stored_meta("doomed", expires_at=datetime(2026, 9, 22, 21, 0, tzinfo=UTC)), _arrays())
        monkeypatch.setattr(local_fs, "os", _NoFreeInodes())
        with pytest.raises(OSError) as refused:
            local_fs.os.open(store.base_path / "one-more-file", os.O_CREAT | os.O_RDWR, 0o600)
        assert refused.value.errno == errno.ENOSPC, "control: the simulated volume must refuse a new file"
        send = {
            "delete": lambda: client.delete("/v1/datasets/doomed"),
            "batch-delete": lambda: client.post("/v1/datasets/batch-delete", json={"dataset_ids": ["doomed"]}),
            "cleanup-expired": lambda: client.post("/v1/datasets/cleanup-expired"),
        }
        response = send[route]()
        assert response.status_code in (200, 204), response.text
        assert store.get_meta("doomed") is None

    def test_a_symlink_planted_at_a_lock_file_is_refused_not_followed(self, tmp_path) -> None:
        # The lock file was opened with O_CREAT and without O_NOFOLLOW, so a symlink planted at it
        # -- which takes write access to the storage directory -- was followed, and DELETE, batch
        # delete and batch-tags each created the lock file outside the storage root.
        store = LocalFSDatasetStore(tmp_path / "storage")
        store.save("guarded", _stored_meta("guarded"), _arrays())
        planted = tmp_path / "outside" / "planted.lock"
        planted.parent.mkdir()
        lock_path = store._lock_path("guarded")
        # The directory is there once the store is. Creating it here keeps this test about following
        # the symlink, not about WHEN the stripes are created: harness arm M57 names it as a control.
        lock_path.parent.mkdir(exist_ok=True)
        lock_path.unlink(missing_ok=True)
        lock_path.symlink_to(planted)  # dangling, and outside the storage root
        with pytest.raises(OSError):
            store.delete_under_lock("guarded")
        assert not planted.exists(), "the store followed a symlink out of its storage root"
        assert store.get_meta("guarded") is not None, "nothing may be deleted without the lock"

    def test_a_storage_directory_that_cannot_hold_the_stripes_still_opens(self, tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        # Before the stripes existed, a store opened on any directory and failed only when it first
        # wrote. Creating them must not make a read-only volume, or one out of inodes when the
        # service first starts, fatal at startup: a stripe is then created when first locked.
        monkeypatch.setattr(local_fs, "os", _NoFreeInodes())
        with caplog.at_level(logging.WARNING, logger="juniper_data.storage.local_fs"):
            store = LocalFSDatasetStore(tmp_path / "storage")
        assert not (tmp_path / "storage" / "locks").exists()
        assert [record.levelno for record in caplog.records if record.name == "juniper_data.storage.local_fs"] == [logging.WARNING]
        monkeypatch.setattr(local_fs, "os", os)
        store.save("guarded", _stored_meta("guarded"), _arrays())
        assert store.update_tags("guarded", ["b"], []) is not None
        assert store._lock_path("guarded").is_file(), "the stripe is created when first locked, once the directory is writable"

    @pytest.mark.parametrize("route", ["delete", "batch-delete", "cleanup-expired"])
    def test_a_volume_out_of_inodes_since_before_the_upgrade_still_deletes(self, route: str, tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        # The upgrade path: an earlier version filled the volume -- with its per-id lock files,
        # among other things -- and this one starts on it, so it cannot create ``locks/`` either.
        # Every lock then needed the inode a delete is trying to free, and deletes answered 500.
        # The lock falls back to the storage root, which needs no inode.
        storage = tmp_path / "juniper_data_storage"
        LocalFSDatasetStore(storage).save("doomed", _stored_meta("doomed", expires_at=datetime(2026, 9, 22, 21, 0, tzinfo=UTC)), _arrays())
        shutil.rmtree(storage / "locks")
        monkeypatch.setattr(local_fs, "os", _NoFreeInodes())
        with caplog.at_level(logging.WARNING, logger="juniper_data.storage.local_fs"):
            store = LocalFSDatasetStore(storage)
        app = create_app(settings=Settings(storage_path=str(storage)))
        datasets.set_store(store)
        client = TestClient(app)
        send = {
            "delete": lambda: client.delete("/v1/datasets/doomed"),
            "batch-delete": lambda: client.post("/v1/datasets/batch-delete", json={"dataset_ids": ["doomed"]}),
            "cleanup-expired": lambda: client.post("/v1/datasets/cleanup-expired"),
        }
        response = send[route]()
        assert response.status_code in (200, 204), response.text
        assert store.get_meta("doomed") is None
        assert not (storage / "locks").exists(), "control: the simulated volume never let the lock directory be created"

    def test_the_lock_files_an_earlier_version_left_are_removed_when_the_store_opens(self, tmp_path) -> None:
        # They lock nothing now, and on a volume they filled they are the inodes a delete needs.
        # Only what matches exactly -- a valid dataset id, then ``.meta.json.lock``, directly in the
        # root, and not a directory -- is removed.
        storage = tmp_path / "storage"
        (storage / "nested").mkdir(parents=True)
        legacy = [storage / f"{dataset_id}.meta.json.lock" for dataset_id in ("spiral-3.0.0-0123456789abcdef", "absent-0001")]
        kept = [storage / "notes.meta.json.lock.bak", storage / "nested" / "absent-0002.meta.json.lock", storage / "absent-0003.meta.json"]
        for path in legacy + kept:
            path.write_bytes(b"")
        (storage / "absent-0004.meta.json.lock").mkdir()
        LocalFSDatasetStore(storage)
        assert [path.name for path in legacy if path.exists()] == [], "the lock files an earlier version left must go"
        assert all(path.exists() for path in kept) and (storage / "absent-0004.meta.json.lock").is_dir(), "only an exact match is removed"

    def test_a_symlink_planted_at_a_stripe_before_the_store_opens_is_not_followed(self, tmp_path) -> None:
        # Creating the stripes opens each with O_CREAT: following a symlink planted there would
        # create its target, outside the storage root. And one bad stripe used to stop the rest,
        # so every other stripe then needed an inode when first locked.
        storage = tmp_path / "storage"
        (storage / "locks").mkdir(parents=True)
        planted = tmp_path / "outside" / "planted.lock"
        planted.parent.mkdir()
        (storage / "locks" / "1.lock").symlink_to(planted)  # dangling, and outside the storage root
        LocalFSDatasetStore(storage)
        assert not planted.exists(), "creating the stripes followed a symlink out of the storage root"
        created = sorted(path.name for path in (storage / "locks").iterdir() if path.is_file() and not path.is_symlink())
        assert created == [f"{stripe:x}.lock" for stripe in range(16) if stripe != 1], "one bad stripe must not stop the others"

    def test_a_symlink_to_a_live_file_planted_at_a_stripe_is_refused(self, tmp_path) -> None:
        # Its target exists, so an open that followed it would succeed -- and lock a file outside
        # the root, which excludes nobody who locks the real stripe.
        store = LocalFSDatasetStore(tmp_path / "storage")
        store.save("guarded", _stored_meta("guarded"), _arrays())
        live = tmp_path / "outside" / "live.lock"
        live.parent.mkdir()
        live.write_bytes(b"")
        lock_path = store._lock_path("guarded")
        lock_path.unlink()
        lock_path.symlink_to(live)
        with pytest.raises(OSError) as refused:
            store.update_tags("guarded", ["b"], [])
        assert refused.value.errno == errno.ELOOP
        assert store.get_meta("guarded").tags == [], "nothing may be written without the lock"

    @pytest.mark.parametrize("kind", ["symlink-outside", "symlink-inside", "file"])
    def test_a_locks_entry_that_is_not_a_real_directory_fails_the_store_when_it_opens(self, kind: str, tmp_path) -> None:
        # A symlink there -- wherever it points -- or a file. The store used to open over a file
        # and answer every write with a 500, from a service whose health check was green.
        storage = tmp_path / "storage"
        storage.mkdir()
        if kind == "file":
            (storage / "locks").write_bytes(b"")
        else:
            target = tmp_path / "outside" if kind == "symlink-outside" else storage / "elsewhere"
            target.mkdir()
            (storage / "locks").symlink_to(target)
        with pytest.raises(StorageContainmentError):
            LocalFSDatasetStore(storage)
        if kind == "symlink-outside":
            assert list((tmp_path / "outside").iterdir()) == [], "nothing may be created through the symlink"

    def test_a_locks_directory_swapped_for_a_symlink_after_the_store_opens_is_not_followed(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The stripes are opened through the descriptor of the ``locks/`` the store opened, never
        # by path: a ``locks`` replaced by a symlink later -- to a directory that even holds a
        # stripe of the right name -- is not followed, and the lock stays where the others take it.
        storage = tmp_path / "storage"
        store = LocalFSDatasetStore(storage)
        store.save("guarded", _stored_meta("guarded"), _arrays())
        stripe = store._lock_path("guarded").name
        (storage / "locks").rename(storage / "moved")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / stripe).write_bytes(b"")
        (storage / "locks").symlink_to(outside)
        seen: list[tuple[bool, bool]] = []
        real_update_meta = store.update_meta

        def probing_update_meta(dataset_id: str, meta: DatasetMeta) -> bool:
            seen.append((_flock_is_held(outside / stripe), _flock_is_held(storage / "moved" / stripe)))
            return real_update_meta(dataset_id, meta)

        monkeypatch.setattr(store, "update_meta", probing_update_meta)
        assert store.update_tags("guarded", ["b"], []) is not None
        assert seen == [(False, True)], "the lock must stay on the stripe the store opened, not follow the symlink"

    def test_a_lock_on_the_storage_root_excludes_every_stripe_lock(self, tmp_path) -> None:
        # The lock of last resort, when a stripe cannot be made, is the storage root, locked
        # exclusively -- by a process on a full volume, say. Every stripe lock holds the root
        # shared, so that process excludes everyone else, and stripes still exclude only each other.
        store = LocalFSDatasetStore(tmp_path / "storage")
        store.save("guarded", _stored_meta("guarded"), _arrays())
        edited = threading.Event()

        def edit() -> None:
            if store.update_tags("guarded", ["b"], []) is not None:
                edited.set()

        root = os.open(tmp_path / "storage", os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(root, fcntl.LOCK_EX)
            editing = threading.Thread(target=edit)
            editing.start()
            assert not edited.wait(0.5), "a stripe lock was taken while another held the storage root"
        finally:
            fcntl.flock(root, fcntl.LOCK_UN)
            os.close(root)
        editing.join(_RACE_TIMEOUT_SECONDS)
        assert edited.is_set(), "control: the edit completes once the root is free"
        assert store.get_meta("guarded").tags == ["b"]

    def test_a_stripe_that_cannot_be_made_is_replaced_by_an_exclusive_lock_on_the_root(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        # ``locks/`` is there, but this dataset's stripe is not -- removed, or never made on a volume
        # already full -- and cannot be made now. The lock falls back to the storage root, taken
        # exclusively, instead of failing the write.
        storage = tmp_path / "storage"
        store = LocalFSDatasetStore(storage)
        store.save("guarded", _stored_meta("guarded"), _arrays())
        store._lock_path("guarded").unlink()
        monkeypatch.setattr(local_fs, "os", _NoFreeInodes())
        seen: list[bool] = []
        real_update_meta = store.update_meta

        def probing_update_meta(dataset_id: str, meta: DatasetMeta) -> bool:
            seen.append(_root_is_locked_exclusively(storage))
            return real_update_meta(dataset_id, meta)

        monkeypatch.setattr(store, "update_meta", probing_update_meta)
        assert store.update_tags("guarded", ["b"], []) is not None
        assert seen == [True], "the edit must run under an exclusive lock on the storage root"
        assert not store._lock_path("guarded").exists(), "control: the simulated volume never let the stripe be made"

    def test_a_locks_directory_removed_while_the_store_runs_is_made_again(self, tmp_path) -> None:
        # The store holds a descriptor to the ``locks/`` it opened. Once that directory is gone --
        # removed by hand, say -- nothing can be made in it, so the store opens whatever is at
        # ``locks`` now, making it if need be, and locks there.
        storage = tmp_path / "storage"
        store = LocalFSDatasetStore(storage)
        store.save("guarded", _stored_meta("guarded"), _arrays())
        shutil.rmtree(storage / "locks")
        assert store.update_tags("guarded", ["b"], []) is not None
        assert store._lock_path("guarded").is_file(), "the stripe is made again, in a new locks/"

    def test_a_storage_root_made_again_while_the_store_runs_is_opened_again(self, tmp_path) -> None:
        # The store holds a descriptor to the root it opened, and every other operation finds the
        # root by path. Once that directory is removed and made again at the same path -- by hand,
        # say -- the locks follow it: nothing can be made through the old descriptor, so every
        # write would fail until the service restarted.
        storage = tmp_path / "storage"
        store = LocalFSDatasetStore(storage)
        shutil.rmtree(storage)
        storage.mkdir()
        store.save("guarded", _stored_meta("guarded"), _arrays())
        assert store.update_tags("guarded", ["b"], []) is not None
        assert store.get_meta("guarded").tags == ["b"]
        assert store._lock_path("guarded").is_file(), "the stripe is made in the new root's locks/"


@pytest.mark.unit
class TestLatestContentLocation:
    """``GET /v1/datasets/latest``: the canonical URI and the canonical validator."""

    def test_latest_names_its_canonical_uri_and_shares_its_etag(self, client: TestClient) -> None:
        _create(client, seed=1, name="cl-demo")
        newest = _create(client, seed=2, name="cl-demo")
        latest = client.get("/v1/datasets/latest", params={"name": "cl-demo"})
        assert latest.status_code == 200
        assert latest.json()["dataset_id"] == newest
        assert latest.headers["content-location"] == f"/v1/datasets/{newest}"
        canonical = client.get(f"/v1/datasets/{newest}")
        assert latest.headers["etag"] == canonical.headers["etag"]
        assert latest.content == canonical.content

    def test_latest_304_keeps_content_location(self, client: TestClient) -> None:
        newest = _create(client, seed=3, name="cl-304")
        etag = client.get("/v1/datasets/latest", params={"name": "cl-304"}).headers["etag"]
        response = client.get("/v1/datasets/latest", params={"name": "cl-304"}, headers={"If-None-Match": etag})
        assert response.status_code == 304
        assert response.headers["content-location"] == f"/v1/datasets/{newest}"


@pytest.mark.unit
class TestAccessRecordingNeverBlocksTheLoop:
    """``record_access`` waits for the store's locks, so nothing may run it on the event loop.

    It did, through ``call_soon``: while a writer held ``_version_lock`` -- a create does, for its
    save -- one read's access blocked the loop, and every request waited with it, ``/v1/health``
    included. Measured on a live server: 6.6 s for one 80 MB create, 10.5 s for two (round-1
    validation of the #438 fix-forward, lane B M-1).
    """

    # Far above a read's own cost, far below ``_LockHolder``'s limit: a read that waited for the
    # lock takes the whole limit.
    _PROMPT_SECONDS = 2.0

    @pytest.mark.asyncio
    async def test_reads_and_health_answer_at_once_while_a_writer_holds_the_lock(self, store: InMemoryDatasetStore, tmp_path) -> None:
        # One event loop serves every request here, as in the service: httpx's ASGI transport runs
        # the app on the test's own loop. The writer is another thread, holding the lock.
        storage = tmp_path / "juniper_data_storage"
        storage.mkdir()
        app = create_app(settings=Settings(storage_path=str(storage)))
        datasets.set_store(store)
        timings: dict[str, float] = {}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            created = await client.post("/v1/datasets", json={"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": 1}, "persist": True})
            dataset_id = created.json()["dataset_id"]
            before = store.get_meta(dataset_id).access_count
            with _LockHolder(DatasetStore._version_lock):
                for label, url in (("metadata", f"/v1/datasets/{dataset_id}"), ("artifact", f"/v1/datasets/{dataset_id}/artifact"), ("health", "/v1/health")):
                    started = time.monotonic()
                    response = await client.get(url)
                    timings[label] = time.monotonic() - started
                    assert response.status_code == 200, (label, response.status_code)
            assert max(timings.values()) < self._PROMPT_SECONDS, f"a request waited for the writer's lock: {timings}"
            after = (await client.get(f"/v1/datasets/{dataset_id}/access")).json()["access_count"]
        assert after == before + 2, "control: both reads were recorded, once the writer let go"

    def test_the_access_counters_include_every_read_answered_before_them(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        # Accesses are recorded after their reads are answered, so ``/access`` waits for the ones
        # handed over before it: a client that reads and then asks for the count sees its read.
        dataset_id = _create(client)
        before = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        answered: dict[str, httpx.Response] = {}
        with _LockHolder(DatasetStore._version_lock):
            assert client.get(f"/v1/datasets/{dataset_id}").status_code == 200, "the read is answered while its access waits"
            asking = threading.Thread(target=lambda: answered.update(access=client.get(f"/v1/datasets/{dataset_id}/access")))
            asking.start()
            asking.join(0.5)
            assert asking.is_alive(), "the counters were served before the access they must include was recorded"
        asking.join(_RACE_TIMEOUT_SECONDS)
        assert answered["access"].json()["access_count"] == before + 1

    def test_a_download_whose_metadata_read_fails_for_a_moment_is_still_counted(self, client: TestClient, store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch) -> None:
        # Only a file the store REFUSES (one that leads out of the root) skips the recording: the
        # recording reads the same metadata and would only fail again. A transient failure -- a
        # descriptor limit -- skipped it too, and a real download went uncounted.
        dataset_id = _create(client)
        real_get_meta = store.get_meta
        before = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        failures = [OSError(errno.EMFILE, os.strerror(errno.EMFILE))]

        def get_meta_failing_once(target: str) -> DatasetMeta | None:
            if failures:
                raise failures.pop()
            return real_get_meta(target)

        monkeypatch.setattr(store, "get_meta", get_meta_failing_once)
        assert client.get(f"/v1/datasets/{dataset_id}/artifact").status_code == 200
        assert not failures, "control: the route's read did fail"
        assert client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"] == before + 1

    def test_an_access_that_cannot_be_recorded_is_logged_by_type_only(self, client: TestClient, store: InMemoryDatasetStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        # It fails on the recorder's thread, after the read was answered; the message can carry the
        # caller's id, and ERR-08 keeps caller strings out of log records.
        dataset_id = _create(client)

        def refuse(target: str) -> None:
            raise ValueError(f"cannot record {target}")

        monkeypatch.setattr(store, "record_access", refuse)
        with caplog.at_level(logging.DEBUG):
            assert client.get(f"/v1/datasets/{dataset_id}").status_code == 200
            assert client.get(f"/v1/datasets/{dataset_id}/access").status_code == 200
        failed = [record for record in caplog.records if "record an access" in record.getMessage()]
        assert [(record.levelno, record.getMessage()) for record in failed] == [(logging.WARNING, "Could not record an access (ValueError)")]
        assert not any(dataset_id in logging.Formatter("%(message)s").format(record) for record in caplog.records if record.levelno >= logging.WARNING)


@pytest.mark.unit
class TestAccessCountersMoved:
    """The counters are still maintained, and are read from their own sub-resource."""

    def test_access_endpoint_serves_the_counters_uncached(self, client: TestClient, store: InMemoryDatasetStore) -> None:
        dataset_id = _create(client)
        store.record_access(dataset_id)
        store.record_access(dataset_id)
        response = client.get(f"/v1/datasets/{dataset_id}/access")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        body = response.json()
        assert body["dataset_id"] == dataset_id
        assert body["access_count"] == store.get_meta(dataset_id).access_count
        assert body["last_accessed_at"] is not None

    def test_reading_the_counters_is_not_itself_an_access(self, client: TestClient) -> None:
        dataset_id = _create(client)
        first = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        second = client.get(f"/v1/datasets/{dataset_id}/access").json()["access_count"]
        assert second == first

    def test_access_endpoint_404s_for_an_unknown_dataset(self, client: TestClient) -> None:
        assert client.get("/v1/datasets/no-such-dataset/access").status_code == 404

    def test_no_representation_that_embeds_metadata_carries_a_counter(self, client: TestClient) -> None:
        created = client.post("/v1/datasets", json={"generator": "spiral", "params": {"n_spirals": 2, "n_points_per_spiral": 20, "seed": 9}, "name": "embed"})
        assert created.status_code == 201
        listed = client.get("/v1/datasets/filter").json()["datasets"]
        versions = client.get("/v1/datasets/versions", params={"name": "embed"}).json()["versions"]
        assert listed and versions, "an empty listing would make this loop pass vacuously"
        for representation in [created.json()["meta"], *listed, *versions]:
            for counter in COUNTERS:
                assert counter not in representation
