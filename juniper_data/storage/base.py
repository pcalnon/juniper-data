"""Abstract base class for dataset storage."""

import base64
import contextlib
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

# from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from juniper_data.core.constants import CHARSET_UTF8
from juniper_data.core.models import DatasetMeta
from juniper_data.storage.constants import ARTIFACT_STREAM_CHUNK_SIZE

# from typing import Dict, List, Optional


# JD-PERF-02: short-lived cache for ``list_all_metadata()`` results. Backs
# ``filter_datasets`` / ``get_stats`` / ``delete_expired`` / ``list_versions``
# / ``next_version_number`` — all of which previously called
# ``list_all_metadata()`` and paid the full disk-walk on every invocation.
# A 5 s TTL is short enough that interactive create-then-list flows feel
# fresh; a longer TTL would risk surprising callers who just saved a
# dataset. Subclasses can opt in to immediate freshness by calling
# ``_invalidate_metadata_cache()`` from their concrete ``save`` / ``delete``
# / ``update_meta`` implementations.
_METADATA_CACHE_TTL_SECONDS = 5.0

# APD-DATA-011: keyset pagination. The cursor encodes one row's position in
# ``filter_datasets``' total order -- ``(created_at DESC, dataset_id ASC)`` -- and a page
# is "everything strictly after that position". Unlike an offset it names a *place in the
# ordering* rather than a count of rows before it, so rows inserted or deleted ahead of
# the cursor cannot shift the next page.
_CURSOR_SEPARATOR = "|"


def encode_cursor(meta: DatasetMeta) -> str:
    """Encode a row's position in the total order as an opaque cursor.

    Opaque by intent: callers must treat it as a token to hand back, not as a structure
    to build. Encoding the sort key rather than an index is the whole point -- an index
    would drift for exactly the reasons keyset pagination exists to avoid.
    """
    raw = f"{meta.created_at.isoformat()}{_CURSOR_SEPARATOR}{meta.dataset_id}"
    return base64.urlsafe_b64encode(raw.encode(CHARSET_UTF8)).decode("ascii")


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    """Decode a cursor produced by :func:`encode_cursor`.

    Raises:
        ValueError: If the cursor is not a well-formed token. The caller is expected to
            translate this into an HTTP 400 -- the cursor is schema-valid as a string but
            semantically wrong, which is the 400/422 rule stated in ``create_dataset``
            (APD-DATA-014).
    """
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode(CHARSET_UTF8)
        encoded_at, _, dataset_id = raw.partition(_CURSOR_SEPARATOR)
        if not encoded_at or not dataset_id:
            raise ValueError("cursor is missing a component")
        return datetime.fromisoformat(encoded_at), dataset_id
    except ValueError:
        raise
    except Exception as exc:  # undecodable base64 / non-ascii / bad utf-8
        raise ValueError(f"Malformed pagination cursor: {cursor!r}") from exc


def _strictly_after(meta: DatasetMeta, cursor_created_at: datetime, cursor_dataset_id: str) -> bool:
    """Is ``meta`` strictly after the cursor position in ``(created_at DESC, id ASC)``?

    ``created_at`` descends, so "after" means OLDER; ``dataset_id`` ascends, so within one
    timestamp "after" means a GREATER id. Getting either comparison backwards silently
    returns the page the caller already has, or skips the rest of a tie group -- which is
    why both halves are pinned by their own tests.
    """
    if meta.created_at != cursor_created_at:
        return meta.created_at < cursor_created_at
    return meta.dataset_id > cursor_dataset_id


class PreconditionFailedError(Exception):
    """A conditional write's precondition evaluated false against the CURRENT metadata.

    Raised by :meth:`DatasetStore.update_tags` from inside its lock, so the check and the
    write cannot be interleaved by another writer; the route turns it into a 412
    (RFC 9110 §13.1, APD-DATA-017).
    """


class InvalidDatasetIdError(ValueError):
    """A ``dataset_id`` the store refuses to address: the CALLER's error, never a storage fault.

    Raised for the id itself -- LocalFS's ``_validate_dataset_id`` refuses it before any path
    is built -- and for nothing else. A well-formed id whose stored file resolves outside the
    storage root is the store's fault, and raises :class:`StorageContainmentError` instead.

    A ``ValueError``, so every existing handler keeps working unchanged -- the app's
    ``ValueError`` handler answers it with the 400 every route gives a malformed id. It is a
    class of its own so that a route which degrades on a storage failure can tell the two
    apart: ``download_artifact`` serves the artifact without a validator when the metadata
    cannot be read, and a malformed id must not take that path. Its message carries the
    caller's id, so a handler that logs it is bound by ERR-08; the app's handler logs it at
    DEBUG only.
    """


class StorageContainmentError(ValueError):
    """A stored file of a well-formed ``dataset_id`` resolves outside the storage root.

    LocalFS's second traversal layer (JD-SEC-01): the id passed validation, but the path built
    from it resolves -- through a symlink in the storage directory -- to somewhere outside it,
    and the store refuses to follow. That is a fault in the STORAGE, not the caller's error,
    so it is not an :class:`InvalidDatasetIdError`: ``download_artifact`` treats it as
    metadata it cannot read and serves the artifact without a validator, as that route did
    before validators existed.

    It is still a ``ValueError``, as this check's error was before ``InvalidDatasetIdError``
    existed, so ``batch_delete`` keeps classifying such an id as not found without failing the
    batch, and batch-create reports the item as failed and goes on with the rest. Everywhere
    else the app's ``ValueError`` handler answers it as the server fault it is: a generic
    ``500``, logged by type only. ``/filter``, ``/stats``, ``/versions``, ``/latest``,
    expired-dataset cleanup and every named create read every dataset's metadata, so one such
    file fails them for the whole store.

    LocalFS raises it too when what is at ``locks``, its lock directory, is not a real
    directory -- a symlink, wherever it points, or a file: the store then does not open, or,
    finding one while it runs, locks nothing through it and fails the write.

    It covers those two storage faults, nothing else: a metadata document that does not parse
    still raises the parser's ``ValueError``, which the app answers as the caller's ``400`` --
    a known issue.
    """


class StagedSave:
    """A save in two parts: the work that needs no lock, done first, and a commit made under the locks.

    :meth:`DatasetStore.stage_save` returns one. :meth:`commit` stores ``meta`` and the staged
    arrays; :meth:`discard` undoes the staging when nothing was committed, and does nothing after
    a commit, so ``try: ... finally: staged.discard()`` is always right.
    """

    def __init__(self, commit: Callable[[DatasetMeta], None], discard: Callable[[], None] | None = None) -> None:
        self._commit = commit
        self._discard = discard
        self._settled = False

    def commit(self, meta: DatasetMeta) -> None:
        """Finish the save. Called at most once, under the locks."""
        self._commit(meta)
        self._settled = True

    def discard(self) -> None:
        """Undo the staging unless :meth:`commit` completed. Idempotent."""
        if not self._settled:
            self._settled = True
            if self._discard is not None:
                self._discard()


class DatasetStore(ABC):
    """Abstract dataset storage interface.

    Provides a common interface for storing and retrieving datasets,
    supporting different backends (in-memory, local filesystem, cloud, etc.).
    """

    _version_lock = threading.Lock()

    def __init__(self) -> None:
        # JD-PERF-02: cache state initialised here so subclasses that don't
        # call ``super().__init__()`` lazily lose the cache (graceful
        # degrade) but everything else keeps working — the cache lookup
        # tolerates absent attrs via ``getattr(..., None)``.
        self._metadata_cache_lock = threading.Lock()
        self._metadata_cache: list[DatasetMeta] | None = None
        self._metadata_cache_at: float = 0.0

    def _list_all_metadata_cached(self) -> list[DatasetMeta]:
        """Return cached metadata if fresh, otherwise re-fetch.

        Stale-tolerant TTL cache: bounds the steady-state cost of
        ``filter_datasets`` / ``get_stats`` / etc. to one disk walk per
        ``_METADATA_CACHE_TTL_SECONDS`` window instead of O(n) per call.
        Subclasses that need immediate freshness on writes should call
        :meth:`_invalidate_metadata_cache` from their ``save`` /
        ``delete`` / ``update_meta`` overrides.

        Concurrent callers that race the cache miss both do the disk walk
        and the last writer wins — benign because both walks produce
        equivalent state.
        """
        lock = getattr(self, "_metadata_cache_lock", None)
        if lock is None:
            # Subclass skipped ``super().__init__``. Degrade to uncached
            # behaviour so we keep the old contract rather than crash.
            return self.list_all_metadata()

        now = time.monotonic()
        cached = self._metadata_cache
        if cached is not None and (now - self._metadata_cache_at) < _METADATA_CACHE_TTL_SECONDS:
            # Return a snapshot copy so a caller mutating the list (e.g.
            # ``filter_datasets`` appends to its local ``filtered`` list,
            # but a buggy caller could ``.remove(...)``) cannot corrupt
            # the cache.
            return list(cached)

        fresh = self.list_all_metadata()
        with lock:
            self._metadata_cache = list(fresh)
            self._metadata_cache_at = now
        return fresh

    def _invalidate_metadata_cache(self) -> None:
        """Drop the cached ``list_all_metadata`` result.

        Subclasses should call this from their concrete ``save`` /
        ``delete`` / ``update_meta`` implementations so a write is
        immediately visible to subsequent ``filter_datasets`` /
        ``get_stats`` calls instead of having to wait out the TTL.
        Safe to call when the cache is empty (no-op).
        """
        lock = getattr(self, "_metadata_cache_lock", None)
        if lock is None:
            return
        with lock:
            self._metadata_cache = None
            self._metadata_cache_at = 0.0

    @abstractmethod
    def save(
        self,
        dataset_id: str,
        meta: DatasetMeta,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Save dataset metadata and arrays.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Dataset metadata.
            arrays: Dictionary of numpy arrays (e.g., X_train, y_train, etc.).

        Raises:
            IOError: If the save operation fails.
        """
        pass

    @abstractmethod
    def get_meta(self, dataset_id: str) -> DatasetMeta | None:
        """Get dataset metadata.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            Dataset metadata if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_artifact_bytes(self, dataset_id: str) -> bytes | None:
        """Get dataset artifact as bytes (NPZ format).

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            NPZ file contents as bytes if found, None otherwise.
        """
        pass

    def open_artifact_stream(self, dataset_id: str, chunk_size: int = ARTIFACT_STREAM_CHUNK_SIZE) -> Iterator[bytes] | None:
        """Yield the artifact in chunks without materialising it whole.

        Defect-register ``APD-DATA-016``. ``download_artifact`` wrapped
        :meth:`get_artifact_bytes` in ``io.BytesIO`` and returned a
        ``StreamingResponse``, which bounds the **socket buffer**, not process
        memory: the entire artifact existed in RAM before the response object did,
        once per concurrent request. Calling that "streaming" invites the
        assumption that it is safe for arbitrarily large artifacts, and it was not.

        **Deliberately NOT abstract.** A backend that has no cheaper path than
        reading the whole blob inherits this default and is unchanged — the seven
        existing stores keep working without edits, and adding one does not become
        harder. Only a backend that can genuinely do better overrides it (see
        :class:`~juniper_data.storage.local_fs.LocalFSDatasetStore`, which reads the
        NPZ file in chunks). So the interface widens without a flag day, and the
        route gets real streaming exactly where a backend can supply it.

        Args:
            dataset_id: Unique identifier for the dataset.
            chunk_size: Bytes per yielded chunk. Overriders should honour it.

        Returns:
            An iterator over the artifact's bytes, or ``None`` when the dataset has
            no artifact — the same "absent" signal :meth:`get_artifact_bytes` uses,
            so callers keep one 404 branch rather than two.
        """
        payload = self.get_artifact_bytes(dataset_id)
        if payload is None:
            return None
        return iter((payload,))

    @abstractmethod
    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset exists, False otherwise.
        """
        pass

    @abstractmethod
    def delete(self, dataset_id: str) -> bool:
        """Delete dataset.

        Args:
            dataset_id: Unique identifier for the dataset.

        Returns:
            True if the dataset was deleted, False if it didn't exist.
        """
        pass

    @abstractmethod
    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List dataset IDs.

        Args:
            limit: Maximum number of dataset IDs to return.
            offset: Number of dataset IDs to skip.

        Returns:
            List of dataset IDs.
        """
        pass

    def update_meta(self, dataset_id: str, meta: DatasetMeta) -> bool:
        """Update dataset metadata.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Updated dataset metadata.

        Returns:
            True if the dataset was updated, False if it didn't exist.
        """
        raise NotImplementedError("update_meta not implemented for this storage backend")

    def list_all_metadata(self) -> list[DatasetMeta]:
        """List all dataset metadata (for filtering/stats).

        Returns:
            List of all DatasetMeta objects.
        """
        raise NotImplementedError("list_all_metadata not implemented for this storage backend")

    def record_access(self, dataset_id: str) -> None:
        """Record an access to a dataset (updates last_accessed_at and access_count).

        Args:
            dataset_id: Unique identifier for the dataset.

        CONC-12 / BUG-JD-11 (Phase 3D): the read-modify-write on
        ``access_count`` happens across three calls — ``get_meta``, the
        in-memory increment, and ``update_meta``. Two concurrent requests
        racing on the same dataset both used to read the same count, both
        increment locally, and both write back the *same* new value, so
        one access was silently lost from the counter on every collision.
        Hold the existing ``_version_lock`` across the whole sequence so
        the increment is atomic from the perspective of any other thread
        in the same process, and :meth:`_meta_write_lock` so that it is
        atomic against other processes too, on a store that overrides it
        (LocalFS, since APD-DATA-007). Elsewhere the count is exact within
        one process only (BUG-JD-05).

        It waits for both locks, so it must never run on the event loop:
        the routes hand it to the access recorder's own thread
        (``juniper_data.api.routes.datasets``), and a read never waits for it --
        only ``GET /{dataset_id}/access``, for the accesses handed over before it.
        """
        with self._version_lock, self._meta_write_lock(dataset_id):
            meta = self.get_meta(dataset_id)
            if meta is not None:
                meta.last_accessed_at = datetime.now(UTC)
                meta.access_count += 1
                self.update_meta(dataset_id, meta)

    def update_tags(
        self,
        dataset_id: str,
        add_tags: list[str],
        remove_tags: list[str],
        precondition: Callable[[DatasetMeta], bool] | None = None,
    ) -> DatasetMeta | None:
        """Atomically add and/or remove tags on a dataset's metadata.

        Args:
            dataset_id: Unique identifier for the dataset.
            add_tags: Tags to add.
            remove_tags: Tags to remove. Applied after ``add_tags``.
            precondition: Evaluated against the CURRENT metadata inside the lock, before any
                change; when it returns False nothing is written and
                :class:`PreconditionFailedError` is raised. This is what makes ``If-Match``
                on ``PATCH .../tags`` a real optimistic-concurrency check (APD-DATA-017): a
                check made in the route, outside this lock, could pass and then lose the race.

        Returns:
            The updated metadata, or ``None`` if the dataset does not exist -- including when
            ``update_meta`` reports it gone by the time of the write. The creates, edits and
            deletes this class performs all take the same two locks, so on LocalFS only
            something outside them can do that (a file removed by hand, another host); on a
            store whose cross-process lock is the no-op, another process can. Answering the
            edit as applied would hand the client a new ``ETag`` for a dataset that no longer
            exists. Not every such removal is seen: LocalFS's ``update_meta`` checks existence
            and then renames, so one landing between those two steps is still answered as
            applied, and leaves metadata without an artifact.

        Raises:
            PreconditionFailedError: ``precondition`` returned False.

        APD-DATA-006: this exists so the tag read-modify-write happens under
        the same ``_version_lock`` that :meth:`record_access` holds. Both
        methods rewrite the *whole* ``DatasetMeta`` document, so a lock taken
        by only one of them protects nothing: the route previously read the
        metadata, mutated ``tags``, and wrote it back across two separate
        ``asyncio.to_thread`` hops with no lock at all. ``record_access`` fires
        on every metadata read and every artifact download, so a plain ``GET``
        could interleave between those hops and write back its own pre-edit
        snapshot -- silently discarding a committed tag edit. Losing a write to
        a *safe* method is the failure nobody thinks to look for.

        As for :meth:`record_access`: ``_version_lock`` orders threads within one process, and
        :meth:`_meta_write_lock` orders processes on a store that overrides it (LocalFS). On
        any other store the edit is atomic within one process only (BUG-JD-05).
        """
        with self._version_lock, self._meta_write_lock(dataset_id):
            meta = self.get_meta(dataset_id)
            if meta is None:
                return None
            if precondition is not None and not precondition(meta):
                raise PreconditionFailedError(dataset_id)
            tags = set(meta.tags)
            tags.update(add_tags)
            tags -= set(remove_tags)
            meta.tags = sorted(tags)
            if not self.update_meta(dataset_id, meta):
                return None
            return meta

    def delete_under_lock(self, dataset_id: str) -> bool:
        """Delete a dataset under the two locks every metadata read-modify-write holds.

        Returns:
            True if the dataset was deleted, False if it did not exist -- :meth:`delete`'s
            contract, unchanged.

        :meth:`update_tags` evaluates a ``PATCH .../tags`` precondition and writes under
        ``_version_lock`` and :meth:`_meta_write_lock`, which is what lets ``If-Match`` promise
        that the check cannot pass and then lose the race. A delete that took neither landed
        inside that window: the PATCH's write found the dataset gone, and the PATCH still
        answered ``200`` with an ``ETag``. Every route that deletes -- ``DELETE /{dataset_id}``,
        batch delete and expired-dataset cleanup -- comes through here, taking the locks in
        ``update_tags``' order, so a delete is ordered before or after a tag edit, never inside
        it.

        The locks are taken HERE, around the store's own :meth:`delete`, never inside it:
        ``_version_lock`` is one non-reentrant lock shared by every store instance, and
        ``CachedDatasetStore.delete`` calls its primary's and its cache's ``delete``, so a store
        whose ``delete`` took it would deadlock under the cached one. The per-process caveat of
        :meth:`record_access` applies: only a store that overrides :meth:`_meta_write_lock`
        (LocalFS) orders processes.

        The cost: the process-global ``_version_lock`` is held for the store's whole
        ``delete``, as for every metadata write, so every other lock taker in the process waits
        for it -- on LocalFS a few unlinks; on a store whose ``delete`` does more
        (``CachedDatasetStore`` lists its cache to update a gauge), longer. The event loop never
        waits: :meth:`record_access` runs on the access recorder's own thread.
        """
        with self._version_lock, self._meta_write_lock(dataset_id):
            return self.delete(dataset_id)

    @contextlib.contextmanager
    def _meta_write_lock(self, dataset_id: str) -> Iterator[None]:
        """Serialise a metadata read-modify-write against *other processes*.

        APD-DATA-007. ``_version_lock`` is a ``threading.Lock``, so it orders the
        read-modify-write only among threads of one interpreter. Both writers of the
        whole ``DatasetMeta`` document -- :meth:`record_access` and :meth:`update_tags`
        -- rewrite every field, so two processes that interleave read/write lose one
        side's change entirely. Measured before this existed: twelve processes each
        adding one distinct tag left **two** tags on disk.

        The default is a no-op, which is correct for any store whose state does not
        outlive the process (``InMemoryDatasetStore``); a store backed by shared
        durable state overrides it. Every call site -- :meth:`record_access`,
        :meth:`update_tags`, :meth:`delete_under_lock` and :meth:`save_versioned` -- takes
        ``_version_lock`` first and this second, so the acquisition order is uniform and cannot
        deadlock; ``test_every_lock_taker_enters_the_version_lock_before_the_file_lock`` pins it.

        This closes lost updates between processes on ONE host. It is not a
        distributed lock: separate hosts sharing network storage still need
        coordination this class does not provide.
        """
        yield

    def is_expired(self, meta: DatasetMeta) -> bool:
        """Check if a dataset has expired based on its TTL.

        Args:
            meta: Dataset metadata.

        Returns:
            True if the dataset has expired, False otherwise.
        """
        if meta.expires_at is None:
            return False
        return datetime.now(UTC) > meta.expires_at

    def delete_expired(self) -> list[str]:
        """Delete all expired datasets.

        Returns:
            List of dataset IDs that were deleted.

        Known issue, predating the locks: expiry is decided outside them, on the metadata
        snapshot ``_list_all_metadata_cached`` returns, which can be up to
        ``_METADATA_CACHE_TTL_SECONDS`` old -- and another process's writes never refresh it.
        A dataset deleted and re-created without a TTL after that snapshot was taken is still
        deleted. Closing it means re-reading the metadata and re-deciding expiry under the
        locks.
        """
        deleted: list[str] = []
        deleted.extend(meta.dataset_id for meta in self._list_all_metadata_cached() if self.is_expired(meta) and self.delete_under_lock(meta.dataset_id))
        return deleted

    def list_versions(self, dataset_name: str) -> list[DatasetMeta]:
        """List all versions of a named dataset, sorted by version ascending.

        Args:
            dataset_name: The logical dataset name.

        Returns:
            List of DatasetMeta objects sorted by version number ascending.
        """
        all_meta = self._list_all_metadata_cached()
        versions = [m for m in all_meta if m.dataset_name == dataset_name]
        versions.sort(key=lambda m: m.dataset_version or 0)
        return versions

    def get_latest_version(self, dataset_name: str) -> DatasetMeta | None:
        """Get the latest version of a named dataset.

        Args:
            dataset_name: The logical dataset name.

        Returns:
            DatasetMeta for the latest version, or None if no versions exist.
        """
        versions = self.list_versions(dataset_name)
        return versions[-1] if versions else None

    def next_version_number(self, dataset_name: str) -> int:
        """Get the next version number for a named dataset.

        Note: This method is NOT concurrency-safe on its own. For atomic
        version allocation during save, use ``save_versioned()`` instead.

        Args:
            dataset_name: The logical dataset name.

        Returns:
            The next version number (1 if no versions exist).
        """
        versions = self.list_versions(dataset_name)
        if not versions:
            return 1
        return max(m.dataset_version or 0 for m in versions) + 1

    def save_versioned(
        self,
        dataset_id: str,
        meta: DatasetMeta,
        arrays: dict[str, np.ndarray],
    ) -> DatasetMeta:
        """Create a dataset unless it already exists, allocating its version number atomically.

        If ``meta.dataset_name`` is set and ``meta.dataset_version`` is None,
        the next version number is computed and assigned under the same locks so
        that concurrent callers in one process cannot receive the same version.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Dataset metadata. ``dataset_version`` is set in-place.
            arrays: Dictionary of numpy arrays.

        Returns:
            The metadata now stored under ``dataset_id``: ``meta`` when this call saved
            it, or the existing dataset's when it already existed -- nothing is written then.

        The existence check and the commit happen under ``_version_lock`` and
        :meth:`_meta_write_lock`, the locks every metadata write takes. The create route
        checks existence before it generates, which can take seconds, and so did two
        creates of one id: both passed the check, and the later save overwrote the earlier
        one -- or overwrote a conditional ``PATCH .../tags`` that had passed its check in
        between, which then wrote its stale copy back over the create. Checking again under
        the locks makes create-if-absent atomic: the second create returns the dataset that
        is there.

        What is written first, outside the locks, is whatever :meth:`stage_save` stages: on
        LocalFS the compressed artifact, the expensive part, so the locks are held only for the
        check, the version (a named create lists every version of its name), the metadata write
        and the renames. A store that stages nothing does its whole :meth:`save` under the
        locks, as every create did before staging existed.
        """
        staged = self.stage_save(dataset_id, arrays)
        try:
            with self._version_lock, self._meta_write_lock(dataset_id):
                existing = self.get_meta(dataset_id)
                if existing is not None:
                    return existing
                if meta.dataset_name is not None and meta.dataset_version is None:
                    meta.dataset_version = self.next_version_number(meta.dataset_name)
                staged.commit(meta)
                return meta
        finally:
            staged.discard()

    def stage_save(self, dataset_id: str, arrays: dict[str, np.ndarray]) -> StagedSave:
        """Do the part of saving ``arrays`` that needs no lock, and return its commit.

        :meth:`save_versioned` calls this before it takes any lock and commits under them, so
        whatever is staged here costs no other writer a wait. The default stages nothing: its
        commit is the store's whole :meth:`save`. LocalFS overrides it to write the compressed
        artifact first.

        Args:
            dataset_id: Unique identifier for the dataset.
            arrays: Dictionary of numpy arrays.

        Returns:
            The staged save. Its ``commit(meta)`` finishes the save; its ``discard()`` undoes
            the staging if nothing was committed.
        """
        return StagedSave(lambda meta: self.save(dataset_id, meta, arrays))

    def filter_datasets(
        self,
        generator: str | None = None,
        tags: list[str] | None = None,
        tags_match: str = "any",
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_samples: int | None = None,
        max_samples: int | None = None,
        include_expired: bool = False,
        dataset_name: str | None = None,
        dataset_version: int | None = None,
        limit: int = 100,
        offset: int = 0,
        cursor: str | None = None,
    ) -> tuple[list[DatasetMeta], int]:
        """Filter datasets by various criteria.

        Args:
            generator: Filter by generator name.
            tags: Filter by tags.
            tags_match: "any" (OR) or "all" (AND) for tag matching.
            created_after: Filter by creation date (after).
            created_before: Filter by creation date (before).
            min_samples: Minimum number of samples.
            max_samples: Maximum number of samples.
            include_expired: Include expired datasets.
            dataset_name: Filter by logical dataset name.
            dataset_version: Filter by dataset version number.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Tuple of (filtered metadata list, total count before pagination).
        """
        all_meta = self._list_all_metadata_cached()
        filtered = []

        for meta in all_meta:
            if not include_expired and self.is_expired(meta):
                continue
            if generator is not None and meta.generator != generator:
                continue
            if dataset_name is not None and meta.dataset_name != dataset_name:
                continue
            if dataset_version is not None and meta.dataset_version != dataset_version:
                continue
            if tags is not None:
                if tags_match == "all":
                    if any(t not in meta.tags for t in tags):
                        continue
                elif all(t not in meta.tags for t in tags):
                    continue
            if created_after is not None and meta.created_at < created_after:
                continue
            if created_before is not None and meta.created_at > created_before:
                continue
            if min_samples is not None and meta.n_samples < min_samples:
                continue
            if max_samples is not None and meta.n_samples > max_samples:
                continue
            filtered.append(meta)

        # APD-DATA-012: sort on a TOTAL order. ``created_at`` alone is not one --
        # ``list.sort`` is stable, so datasets sharing a timestamp came back in whatever
        # order the enumeration produced, and ``LocalFSDatasetStore`` enumerates with
        # ``Path.glob``, which specifies no ordering at all (measured: its order is not
        # sorted order even on ext4). Feeding the same six datasets in two enumeration
        # orders produced two different pages. Sorting by ``dataset_id`` first and then
        # stably by ``created_at`` descending breaks ties by id ASCENDING, which is
        # reproducible across calls, across processes, and across the two store
        # implementations -- whose enumeration orders otherwise disagree by construction
        # (glob order vs dict insertion order).
        filtered.sort(key=lambda m: m.dataset_id)
        filtered.sort(key=lambda m: m.created_at, reverse=True)
        total = len(filtered)

        # APD-DATA-011: keyset pagination. ``filtered[offset:offset+limit]`` re-slices a
        # collection that may have changed since the previous page, so an insert repeats
        # a row across pages and a delete skips one -- reproduced, an insert between two
        # fetches returned the same dataset on both. A cursor names the last row's
        # position in the total order above and asks for what strictly follows it, which
        # no insert or delete before that point can shift.
        if cursor is not None:
            cursor_created_at, cursor_dataset_id = decode_cursor(cursor)
            filtered = [m for m in filtered if _strictly_after(m, cursor_created_at, cursor_dataset_id)]
            return filtered[:limit], total

        return filtered[offset : offset + limit], total

    def batch_delete(self, dataset_ids: list[str]) -> tuple[list[str], list[str]]:
        """Delete multiple datasets.

        Args:
            dataset_ids: List of dataset IDs to delete.

        Returns:
            Tuple of (deleted IDs, not found IDs).
        """
        deleted = []
        not_found = []
        for dataset_id in dataset_ids:
            try:
                ok = self.delete_under_lock(dataset_id)
            except ValueError:
                # JD-SEC-01: reject traversal attempts without failing the
                # entire batch — classify as not_found so the response still
                # returns cleanly and legitimate IDs in the same request are
                # not penalised.
                not_found.append(dataset_id)
                continue
            if ok:
                deleted.append(dataset_id)
            else:
                not_found.append(dataset_id)
        return deleted, not_found

    def get_stats(self) -> dict[str, object]:
        """Get aggregate statistics about stored datasets.

        Returns:
            Dictionary with statistics.
        """
        all_meta = self._list_all_metadata_cached()

        if not all_meta:
            return {
                "total_datasets": 0,
                "total_samples": 0,
                "by_generator": {},
                "by_tag": {},
                "oldest_created_at": None,
                "newest_created_at": None,
                "expired_count": 0,
            }

        by_generator: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        total_samples = 0
        expired_count = 0
        created_times = []

        for meta in all_meta:
            by_generator[meta.generator] = by_generator.get(meta.generator, 0) + 1
            for tag in meta.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1
            total_samples += meta.n_samples
            created_times.append(meta.created_at)
            if self.is_expired(meta):
                expired_count += 1

        return {
            "total_datasets": len(all_meta),
            "total_samples": total_samples,
            "by_generator": by_generator,
            "by_tag": by_tag,
            "oldest_created_at": min(created_times),
            "newest_created_at": max(created_times),
            "expired_count": expired_count,
        }
