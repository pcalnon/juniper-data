"""Abstract base class for dataset storage."""

import base64
import contextlib
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

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
        in the same process. The plan's caveat applies: this is a
        per-process lock, so multi-process deployments accept best-effort
        counting (see BUG-JD-05). Access counting is informational so
        this is an acceptable trade-off.
        """
        with self._version_lock, self._meta_write_lock(dataset_id):
            meta = self.get_meta(dataset_id)
            if meta is not None:
                meta.last_accessed_at = datetime.now(UTC)
                meta.access_count += 1
                self.update_meta(dataset_id, meta)

    def update_tags(self, dataset_id: str, add_tags: list[str], remove_tags: list[str]) -> DatasetMeta | None:
        """Atomically add and/or remove tags on a dataset's metadata.

        Args:
            dataset_id: Unique identifier for the dataset.
            add_tags: Tags to add.
            remove_tags: Tags to remove. Applied after ``add_tags``.

        Returns:
            The updated metadata, or ``None`` if the dataset does not exist.

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

        The per-process caveat from :meth:`record_access` applies unchanged:
        this serialises threads within one process, not across a multi-process
        deployment (BUG-JD-05).
        """
        with self._version_lock, self._meta_write_lock(dataset_id):
            meta = self.get_meta(dataset_id)
            if meta is None:
                return None
            tags = set(meta.tags)
            tags.update(add_tags)
            tags -= set(remove_tags)
            meta.tags = sorted(tags)
            self.update_meta(dataset_id, meta)
            return meta

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
        durable state overrides it. Both call sites take ``_version_lock`` first and
        this second, so the acquisition order is uniform and cannot deadlock.

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
        """
        deleted: list[str] = []
        deleted.extend(meta.dataset_id for meta in self._list_all_metadata_cached() if self.is_expired(meta) and self.delete(meta.dataset_id))
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
    ) -> None:
        """Atomically allocate a version number and save.

        If ``meta.dataset_name`` is set and ``meta.dataset_version`` is None,
        the next version number is computed and assigned under a lock so that
        concurrent callers cannot receive the same version.

        Args:
            dataset_id: Unique identifier for the dataset.
            meta: Dataset metadata. ``dataset_version`` is set in-place.
            arrays: Dictionary of numpy arrays.
        """
        if meta.dataset_name is not None and meta.dataset_version is None:
            with self._version_lock:
                meta.dataset_version = self.next_version_number(meta.dataset_name)
                self.save(dataset_id, meta, arrays)
        else:
            self.save(dataset_id, meta, arrays)

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
                ok = self.delete(dataset_id)
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
