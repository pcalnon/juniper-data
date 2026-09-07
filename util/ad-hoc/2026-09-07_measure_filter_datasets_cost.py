"""
Measure the real per-page cost of ``DatasetStore.filter_datasets`` (APD-DATA-019).

Project: juniper-data
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-07
Status: ad-hoc -- investigation (adversarial validation of APD-DATA-019)
Retire when: APD-DATA-019 is closed or the owner rules on the `total` response shape.
Related: juniper-ml/notes/JUNIPER_2026-08-14_JUNIPER-ECOSYSTEM_DEFECT-REGISTER.md (APD-DATA-019)

The register claims "every page does full-population work; exact `total` recomputed
per page" and files it as a PERFORMANCE defect with no measurement. This measures it.

Populates a real LocalFSDatasetStore (the store the service actually wires at
api/app.py:42) with N `*.meta.json` files and times a single page fetch on both the
offset and cursor paths, with and without filters. Also times InMemoryDatasetStore --
the ONLY store whose JD-PERF-02 TTL cache is live, because it is the only one that
calls ``super().__init__()``.

Run:
    /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/2026-09-07_measure_filter_datasets_cost.py
"""

from __future__ import annotations

import json
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from juniper_data.core.models import DatasetMeta  # noqa: E402
from juniper_data.storage.base import encode_cursor  # noqa: E402
from juniper_data.storage.local_fs import LocalFSDatasetStore  # noqa: E402
from juniper_data.storage.memory import InMemoryDatasetStore  # noqa: E402

GENERATORS = ["spiral", "two_moons", "xor", "sine", "mackey_glass", "lorenz"]
TAG_POOL = [["prod"], ["dev"], ["prod", "large"], ["scratch"], []]


def make_meta(i: int, base_time: datetime) -> DatasetMeta:
    """Build one realistic DatasetMeta."""
    return DatasetMeta(
        dataset_id=f"ds-{i:07d}-abcdef0123456789",
        generator=GENERATORS[i % len(GENERATORS)],
        generator_version="2.0.0",
        params={"n_samples": 1000 + i, "noise": 0.1, "seed": i, "n_classes": 2},
        n_samples=1000 + i,
        n_features=2,
        task_type="classification",
        n_classes=2,
        n_train=800,
        n_val=100,
        n_test=100,
        created_at=base_time + timedelta(seconds=i),
        tags=TAG_POOL[i % len(TAG_POOL)],
        dataset_name=f"bench-{i % 50}",
        dataset_version=(i // 50) + 1,
        checksum="0" * 64,
        description="benchmark fixture row for APD-DATA-019 cost measurement",
        created_by="bench",
    )


def _json_default(obj: object) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def populate_fs(base: Path, n: int) -> None:
    """Write N .meta.json files directly (bypasses save() so no NPZ is written)."""
    base.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        meta = make_meta(i, t0)
        (base / f"{meta.dataset_id}.meta.json").write_text(
            json.dumps(meta.model_dump(), default=_json_default),
            encoding="utf-8",
        )


def populate_mem(store: InMemoryDatasetStore, n: int) -> None:
    import numpy as np

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    arrays = {
        "X_train": np.zeros((1, 2), dtype=np.float32),
        "y_train": np.zeros((1,), dtype=np.float32),
    }
    for i in range(n):
        meta = make_meta(i, t0)
        store.save(meta.dataset_id, meta, arrays)


def timeit(fn, reps: int) -> tuple[float, float]:
    """Return (median_ms, min_ms) over `reps` calls."""
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(samples), min(samples)


def bench_store(label: str, store, n: int, reps: int) -> None:
    limit = 100

    # Warm anything lazily built (and, for InMemory, prime the TTL cache).
    first_page, total = store.filter_datasets(limit=limit, offset=0)
    cursor = encode_cursor(first_page[-1]) if first_page else None

    cases = {
        "offset=0, no filter": lambda: store.filter_datasets(limit=limit, offset=0),
        "offset=deep, no filter": lambda: store.filter_datasets(limit=limit, offset=max(0, n - limit)),
        "cursor page 2, no filter": lambda: store.filter_datasets(limit=limit, cursor=cursor),
        "offset=0, generator filter": lambda: store.filter_datasets(limit=limit, offset=0, generator="spiral"),
        "offset=0, tags+range filter": lambda: store.filter_datasets(
            limit=limit,
            offset=0,
            tags=["prod"],
            tags_match="any",
            min_samples=1500,
        ),
        "list_datasets(limit=100)": lambda: store.list_datasets(limit=limit, offset=0),
        "list_datasets(deep offset)": lambda: store.list_datasets(limit=limit, offset=max(0, n - limit)),
    }

    print(f"\n  {label}  (N={n}, total reported={total}, reps={reps})")
    print(f"    {'case':<32} {'median ms':>12} {'min ms':>12}")
    for name, fn in cases.items():
        med, mn = timeit(fn, reps)
        print(f"    {name:<32} {med:>12.2f} {mn:>12.2f}")


def bench_cache_effect(base: Path, n: int, reps: int) -> None:
    """Show what the JD-PERF-02 cache WOULD buy LocalFS if it were wired."""
    import threading

    store = LocalFSDatasetStore(base)
    print(f"    LocalFS has _metadata_cache_lock (cache live?): {hasattr(store, '_metadata_cache_lock')}")
    med_off, _ = timeit(lambda: store.filter_datasets(limit=100, offset=0), reps)

    # Manually wire the cache attrs the subclass never initialises.
    store._metadata_cache_lock = threading.Lock()  # noqa: SLF001
    store._metadata_cache = None  # noqa: SLF001
    store._metadata_cache_at = 0.0  # noqa: SLF001
    store.filter_datasets(limit=100, offset=0)  # prime
    med_on, _ = timeit(lambda: store.filter_datasets(limit=100, offset=0), reps)

    print(f"    N={n}: cache INERT (as shipped) = {med_off:.2f} ms   cache LIVE (hand-wired) = {med_on:.2f} ms   speedup = {med_off / med_on:.1f}x")


def main() -> None:
    ns = [100, 1000, 10000]
    tmp_root = Path(tempfile.mkdtemp(prefix="apd_data_019_"))
    try:
        print("=" * 88)
        print("APD-DATA-019 cost measurement -- filter_datasets / list_datasets per-page cost")
        print("=" * 88)

        for n in ns:
            reps = 20 if n <= 1000 else 5
            base = tmp_root / f"fs_{n}"
            t = time.perf_counter()
            populate_fs(base, n)
            print(f"\n[populate] LocalFS N={n} in {(time.perf_counter() - t):.1f}s -> {base}")

            bench_store("LocalFSDatasetStore (production store, api/app.py:42)", LocalFSDatasetStore(base), n, reps)

            mem = InMemoryDatasetStore()
            populate_mem(mem, n)
            bench_store("InMemoryDatasetStore (JD-PERF-02 cache LIVE)", mem, n, reps)

            print("\n  [JD-PERF-02 cache effect on LocalFS]")
            bench_cache_effect(base, n, reps)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
