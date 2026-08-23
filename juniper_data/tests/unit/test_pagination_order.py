"""APD-DATA-012 (total order) and APD-DATA-011 (keyset pagination).

``filter_datasets`` used to sort on ``created_at`` alone. ``list.sort`` is stable, so
rows sharing a timestamp came back in whatever order the enumeration produced --
``Path.glob`` for the filesystem store, which specifies no ordering and, measured on
ext4, does not deliver sorted order either. Two facts followed:

* the same data enumerated differently produced different pages (``-012``); and
* ``filtered[offset:offset+limit]`` re-slices a collection that may have changed since
  the previous page, so an insert repeats a row across pages and a delete skips one
  (``-011``) -- reproduced, an insert between two fetches returned dataset ``0003`` on
  both pages.

The fix is one ordering and two ways to walk it: a total order
``(created_at DESC, dataset_id ASC)``, and an opt-in cursor naming a position in that
order rather than a count of rows before it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from juniper_data.core.models import DatasetMeta
from juniper_data.storage.base import decode_cursor, encode_cursor
from juniper_data.storage.local_fs import LocalFSDatasetStore
from juniper_data.storage.memory import InMemoryDatasetStore

SHARED_TS = datetime(2026, 8, 23, 12, 0, 0, tzinfo=UTC)
ARRAYS = {k: np.zeros((2, 2), dtype=np.float32) for k in ("X_train", "y_train", "X_test", "y_test", "X_full", "y_full")}


def _meta(ds_id: str, ts: datetime = SHARED_TS) -> DatasetMeta:
    return DatasetMeta(
        dataset_id=ds_id,
        generator="spiral",
        generator_version="1.0.0",
        params={},
        n_samples=4,
        n_features=2,
        n_train=2,
        n_test=2,
        tags=[],
        created_at=ts,
    )


def _ids(metas: list[DatasetMeta]) -> list[str]:
    return [m.dataset_id for m in metas]


@pytest.mark.unit
class TestTotalOrder:
    """APD-DATA-012 — the sort must not depend on enumeration order."""

    def test_tie_order_is_independent_of_insertion_order(self) -> None:
        """Same six datasets, two insertion orders, one output.

        This is the defect in its purest form: with a stable sort keyed on
        ``created_at`` alone, these two stores returned exactly reversed pages.
        """
        ids = [f"spiral-1.0.0-{i:016d}" for i in range(6)]
        outputs = []
        for order in (ids, list(reversed(ids))):
            store = InMemoryDatasetStore()
            for ds in order:
                store.save(ds, _meta(ds), ARRAYS)
            store._invalidate_metadata_cache()
            page, _ = store.filter_datasets(limit=100)
            outputs.append(_ids(page))

        assert outputs[0] == outputs[1], "page order still depends on enumeration order"

    def test_ties_are_broken_by_dataset_id_ascending(self) -> None:
        """The declared tiebreak, asserted -- not merely 'some deterministic order'."""
        store = InMemoryDatasetStore()
        for ds in ["spiral-1.0.0-cccccccccccccccc", "spiral-1.0.0-aaaaaaaaaaaaaaaa", "spiral-1.0.0-bbbbbbbbbbbbbbbb"]:
            store.save(ds, _meta(ds), ARRAYS)
        store._invalidate_metadata_cache()

        page, _ = store.filter_datasets(limit=100)

        assert _ids(page) == sorted(_ids(page))

    def test_newest_first_still_governs_across_timestamps(self) -> None:
        """The tiebreak must not disturb the primary key: newest still comes first."""
        store = InMemoryDatasetStore()
        store.save("spiral-1.0.0-zzzzzzzzzzzzzzzz", _meta("spiral-1.0.0-zzzzzzzzzzzzzzzz", datetime(2026, 1, 1, tzinfo=UTC)), ARRAYS)
        store.save("spiral-1.0.0-aaaaaaaaaaaaaaaa", _meta("spiral-1.0.0-aaaaaaaaaaaaaaaa", datetime(2026, 6, 1, tzinfo=UTC)), ARRAYS)
        store._invalidate_metadata_cache()

        page, _ = store.filter_datasets(limit=100)

        assert _ids(page) == ["spiral-1.0.0-aaaaaaaaaaaaaaaa", "spiral-1.0.0-zzzzzzzzzzzzzzzz"]

    def test_filesystem_enumeration_is_sorted(self, tmp_path) -> None:
        """``list_all_metadata`` must not hand out directory-entry order.

        The total sort above is what fixes tie order; this pins the enumeration beneath
        it so any other consumer is deterministic too.
        """
        store = LocalFSDatasetStore(tmp_path)
        for i in range(6):
            ds = f"spiral-1.0.0-{i:016d}"
            store.save(ds, _meta(ds), ARRAYS)

        ids = [m.dataset_id for m in store.list_all_metadata()]

        assert ids == sorted(ids)


@pytest.mark.unit
class TestKeysetPagination:
    """APD-DATA-011 — a cursor names a position, not a count."""

    @staticmethod
    def _seed(store: InMemoryDatasetStore, n: int = 6) -> None:
        for i in range(n):
            ds = f"spiral-1.0.0-{i:016d}"
            store.save(ds, _meta(ds, datetime(2026, 8, 23, 12, 0, i, tzinfo=UTC)), ARRAYS)
        store._invalidate_metadata_cache()

    def test_insert_between_pages_does_not_duplicate_a_row(self) -> None:
        """The reproduced defect: with ``offset`` this returned ``0003`` on both pages."""
        store = InMemoryDatasetStore()
        self._seed(store)
        page1, _ = store.filter_datasets(limit=3)

        newest = "spiral-1.0.0-9999999999999999"
        store.save(newest, _meta(newest, datetime(2026, 8, 23, 13, 0, 0, tzinfo=UTC)), ARRAYS)
        store._invalidate_metadata_cache()

        page2, _ = store.filter_datasets(limit=3, cursor=encode_cursor(page1[-1]))

        assert not set(_ids(page1)) & set(_ids(page2)), "a row was returned on two pages"

    def test_delete_between_pages_does_not_skip_a_row(self) -> None:
        """The other half. An offset would shift every later row up by one."""
        store = InMemoryDatasetStore()
        self._seed(store)
        page1, _ = store.filter_datasets(limit=3)

        store.delete(_ids(page1)[0])
        store._invalidate_metadata_cache()

        page2, _ = store.filter_datasets(limit=3, cursor=encode_cursor(page1[-1]))

        assert _ids(page2) == ["spiral-1.0.0-0000000000000002", "spiral-1.0.0-0000000000000001", "spiral-1.0.0-0000000000000000"]

    def test_cursor_walks_a_tie_group_without_repeating_or_skipping(self) -> None:
        """Every row shares one timestamp, so only the id half of the key advances.

        This is the arm that catches a reversed ``dataset_id`` comparison: with the
        wrong direction the second page re-serves the first page's rows, and with no
        tiebreak at all the walk never terminates.
        """
        store = InMemoryDatasetStore()
        ids = [f"spiral-1.0.0-{i:016d}" for i in range(6)]
        for ds in ids:
            store.save(ds, _meta(ds), ARRAYS)
        store._invalidate_metadata_cache()

        seen: list[str] = []
        cursor: str | None = None
        for _ in range(4):
            page, _total = store.filter_datasets(limit=2, cursor=cursor)
            if not page:
                break
            seen.extend(_ids(page))
            cursor = encode_cursor(page[-1])

        assert seen == sorted(ids), f"tie-group walk was not exhaustive/ordered: {seen}"

    def test_cursor_round_trips_through_encode_decode(self) -> None:
        meta = _meta("spiral-1.0.0-0000000000000042")

        created_at, dataset_id = decode_cursor(encode_cursor(meta))

        assert created_at == meta.created_at
        assert dataset_id == meta.dataset_id

    @pytest.mark.parametrize("bad", ["not-base64!!", "", "///", "YWJj"])
    def test_malformed_cursor_raises_value_error(self, bad: str) -> None:
        """A token we did not issue is a caller error, surfaced as ValueError -> 400."""
        with pytest.raises(ValueError):
            decode_cursor(bad)

    def test_offset_mode_is_unchanged(self) -> None:
        """Back-compat: the existing contract still behaves exactly as before."""
        store = InMemoryDatasetStore()
        self._seed(store)

        page1, total = store.filter_datasets(limit=3, offset=0)
        page2, _ = store.filter_datasets(limit=3, offset=3)

        assert total == 6
        assert _ids(page1) + _ids(page2) == sorted([f"spiral-1.0.0-{i:016d}" for i in range(6)], reverse=True)

    def test_total_is_the_full_match_count_not_the_page_size(self) -> None:
        """``total`` keeps its meaning in cursor mode, or callers lose their progress bar."""
        store = InMemoryDatasetStore()
        self._seed(store)
        page1, _ = store.filter_datasets(limit=2)

        _page2, total = store.filter_datasets(limit=2, cursor=encode_cursor(page1[-1]))

        assert total == 6
