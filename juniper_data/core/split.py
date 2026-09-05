"""Split and shuffle utilities for dataset partitioning.

This module provides pure NumPy utilities for shuffling and splitting datasets
into train/test sets with reproducible random number generation.
"""

from typing import Any

import numpy as np


def shuffle_data(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    extras: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle X and y arrays together using the same permutation.

    Args:
        X: Feature array of shape (n_samples, ...).
        y: Label array of shape (n_samples, ...).
        rng: NumPy random generator for reproducibility.
        extras: Optional mapping of name -> row-aligned array, permuted IN PLACE with
            the SAME permutation. This is for per-row metadata that describes the rows
            rather than being part of them -- ARC-AGI's ``task_ids`` is the case that
            motivated it. Metadata carried outside the shuffle keeps its GENERATION
            order, so after the shuffle every entry names a different sample's origin,
            and nothing downstream can tell. There is no third return value because a
            varying arity would be a breaking change for the existing callers.

    Returns:
        Tuple of shuffled (X, y) arrays with the same permutation applied.

    Raises:
        ValueError: If X and y -- or any array in ``extras`` -- have different numbers
            of samples.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. Got X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}")

    permutation = rng.permutation(X.shape[0])
    if extras is not None:
        for name, array in list(extras.items()):
            if array.shape[0] != X.shape[0]:
                raise ValueError(f"extras[{name!r}] must have the same number of samples as X. Got {array.shape[0]}, expected {X.shape[0]}")
            extras[name] = array[permutation]
    return X[permutation], y[permutation]


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    test_ratio: float,
) -> dict[str, np.ndarray]:
    """Split arrays into train and test sets based on ratios.

    Args:
        X: Feature array of shape (n_samples, ...).
        y: Label array of shape (n_samples, ...).
        train_ratio: Fraction of data for training (0.0 to 1.0).
        test_ratio: Fraction of data for testing (0.0 to 1.0).

    Returns:
        Dictionary with keys "X_train", "y_train", "X_test", "y_test".

    Raises:
        ValueError: If ratios are invalid or X and y have different sample counts.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. Got X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}")

    if not (0.0 <= train_ratio <= 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1. Got {train_ratio}")

    if not (0.0 <= test_ratio <= 1.0):
        raise ValueError(f"test_ratio must be between 0 and 1. Got {test_ratio}")

    if train_ratio + test_ratio > 1.0:
        raise ValueError(f"train_ratio + test_ratio must not exceed 1.0. Got {train_ratio} + {test_ratio} = {train_ratio + test_ratio}")

    n_samples = X.shape[0]
    n_train = int(np.round(n_samples * train_ratio))
    n_test = int(np.round(n_samples * test_ratio))

    if n_train + n_test > n_samples:
        n_test = n_samples - n_train

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train : n_train + n_test],
        "y_test": y[n_train : n_train + n_test],
    }


def shuffle_and_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    test_ratio: float,
    seed: int | None = None,
    shuffle: bool = True,
) -> dict[str, np.ndarray]:
    """Optionally shuffle and then split data into train/test sets.

    High-level function that combines shuffling and splitting operations.
    Uses np.random.Generator for reproducible randomness.

    Args:
        X: Feature array of shape (n_samples, ...).
        y: Label array of shape (n_samples, ...).
        train_ratio: Fraction of data for training (0.0 to 1.0).
        test_ratio: Fraction of data for testing (0.0 to 1.0).
        seed: Random seed for reproducibility. If None, uses non-deterministic seed.
        shuffle: Whether to shuffle data before splitting. Defaults to True.

    Returns:
        Dictionary with keys "X_train", "y_train", "X_test", "y_test".

    Raises:
        ValueError: If ratios are invalid or X and y have different sample counts.
    """
    if shuffle:
        rng = np.random.default_rng(seed)
        X, y = shuffle_data(X, y, rng)

    return split_data(X, y, train_ratio, test_ratio)


# --- Three-partition sizing (train / val / test) -----------------------------
#
# Decisions 2 and 8 of the partition design of record
# (notes/JUNIPER_2026-08-29_JUNIPER-ECOSYSTEM_TRAIN-EVAL-TEST-PARTITION-DESIGN.md
# in juniper-ml, sections 6.3 and 9.2):
#
#   * The requested TRAIN count is honoured literally -- ``val`` and ``test``
#     are ADDITIONAL rows, not a carve-up of the requested N.
#   * Percentages are expressed relative to ``train``, which starts at 100 %.
#     The default breakdown 100/40/30 at n_train=1000 yields 1000/400/300.
#   * The percentages denote absolute ROWS of the realised dataset, identically
#     for every generator regardless of its native size knob. They are never
#     per-spiral / per-quadrant / per-class units.
#
# Note what this does NOT buy: asking a generator for N+M rows does not
# reproduce the first N rows it would have produced for N (V-1, measured
# 2026-08-30 -- 6/6 generators differ). The train COUNT is preserved; the train
# CONTENT is not. Existing baselines move under this model exactly as they would
# under a carve-up, so do not cite baseline preservation as a reason for it.

DEFAULT_VAL_PERCENT: float = 40.0
DEFAULT_TEST_PERCENT: float = 30.0

#: Upper bound on a single partition's percentage of train.
#:
#: These percentages are request parameters on a public endpoint and they
#: MULTIPLY the row count a generator allocates, so leaving them unbounded is a
#: resource-exhaustion vector (CWE-770): ``val_percent=1e6`` turns a 200-row
#: request into a 4,000,200-row allocation. Ten times the train count is far
#: beyond any legitimate breakdown -- the default is 40 -- while still leaving
#: the "percentages are relative to train and may exceed 100" property intact.
MAX_PARTITION_PERCENT: float = 1000.0


def partition_row_counts(
    n_train: int,
    val_percent: float = DEFAULT_VAL_PERCENT,
    test_percent: float = DEFAULT_TEST_PERCENT,
) -> dict[str, int]:
    """Absolute row counts for an additively-sized three-way partition.

    Implements the sizing model of design decisions 2 and 8: ``n_train`` is
    taken literally and the other two partitions are sized as percentages
    *of it*, so the realised dataset has ``n_train + n_val + n_test`` rows.

    Args:
        n_train: Requested number of training rows. Honoured exactly.
        val_percent: Validation rows as a percentage of ``n_train``.
        test_percent: Test rows as a percentage of ``n_train``.

    Returns:
        Dictionary with keys ``n_train``, ``n_val``, ``n_test`` and
        ``n_total``, where ``n_total`` is the sum of the other three.

    Raises:
        ValueError: If ``n_train`` is below 1, or either percentage is
            negative or not finite.
    """
    if n_train < 1:
        raise ValueError(f"n_train must be at least 1. Got {n_train}")

    for name, pct in (("val_percent", val_percent), ("test_percent", test_percent)):
        if not np.isfinite(pct):
            raise ValueError(f"{name} must be finite. Got {pct}")
        if pct < 0.0:
            raise ValueError(f"{name} must not be negative. Got {pct}")

    n_val = int(np.round(n_train * val_percent / 100.0))
    n_test = int(np.round(n_train * test_percent / 100.0))

    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_total": n_train + n_val + n_test,
    }


def split_three_way(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
) -> dict[str, np.ndarray]:
    """Cut arrays into contiguous train / val / test blocks by row count.

    The three blocks are contiguous and non-overlapping, so the partitions are
    **index-disjoint by construction** -- the property design section 9.6.1
    relies on in place of a duplicate-row guard. Callers wanting a shuffled
    partition must shuffle before cutting; see :func:`shuffle_and_split_three_way`.

    Args:
        X: Feature array of shape ``(n_samples, ...)``.
        y: Label array of shape ``(n_samples, ...)``.
        n_train: Rows assigned to ``train``.
        n_val: Rows assigned to ``val``.
        n_test: Rows assigned to ``test``.

    Returns:
        Dictionary with keys ``X_train``, ``y_train``, ``X_val``, ``y_val``,
        ``X_test`` and ``y_test``.

    Raises:
        ValueError: If ``X`` and ``y`` disagree on sample count, if any count
            is negative, or if the three counts together exceed the rows
            available. Rows beyond ``n_train + n_val + n_test`` are left
            unused rather than silently folded into a partition.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. Got X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}")

    for name, count in (("n_train", n_train), ("n_val", n_val), ("n_test", n_test)):
        if count < 0:
            raise ValueError(f"{name} must not be negative. Got {count}")

    n_needed = n_train + n_val + n_test
    if n_needed > X.shape[0]:
        raise ValueError(f"Not enough rows to partition: need n_train + n_val + n_test = {n_train} + {n_val} + {n_test} = {n_needed}, got {X.shape[0]}")

    val_start = n_train
    test_start = n_train + n_val
    test_end = test_start + n_test

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_val": X[val_start:test_start],
        "y_val": y[val_start:test_start],
        "X_test": X[test_start:test_end],
        "y_test": y[test_start:test_end],
    }


def shuffle_and_split_three_way(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int | None = None,
    shuffle: bool = True,
    *,
    extras: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Optionally shuffle, then cut into train / val / test by row count.

    The three-partition counterpart of :func:`shuffle_and_split`. Shuffling
    happens once over the whole array, so the resulting partitions remain
    index-disjoint whatever the permutation.

    Args:
        X: Feature array of shape ``(n_samples, ...)``.
        y: Label array of shape ``(n_samples, ...)``.
        n_train: Rows assigned to ``train``.
        n_val: Rows assigned to ``val``.
        n_test: Rows assigned to ``test``.
        seed: Random seed for reproducibility. If None, uses a
            non-deterministic seed.
        shuffle: Whether to shuffle before cutting. Defaults to True.
        extras: Optional per-row metadata, permuted IN PLACE alongside the rows it
            describes. See :func:`shuffle_data`. Untouched when ``shuffle`` is False,
            because the rows are not reordered either.

    Returns:
        Dictionary with keys ``X_train``, ``y_train``, ``X_val``, ``y_val``,
        ``X_test`` and ``y_test``.

    Raises:
        ValueError: Propagated from :func:`split_three_way`.
    """
    if shuffle:
        rng = np.random.default_rng(seed)
        X, y = shuffle_data(X, y, rng, extras=extras)

    return split_three_way(X, y, n_train, n_val, n_test)


def temporal_split_index(n_samples: int, train_ratio: float) -> int:
    """Row index of the chronological (non-shuffled) train/test boundary.

    Rows ``[0, idx)`` are train (earliest dates) and ``[idx, n)`` are test
    (latest). This is the date-ordered split the equities time-series generators
    use; promoting it here lets any time-series generator -- and the sequence
    windower, which maps the returned index to its target-date cut -- reuse one
    tested boundary instead of re-deriving it. For ``n_samples >= 2`` the index
    is clamped to ``[1, n_samples - 1]`` so both splits can be non-empty.

    Args:
        n_samples: total number of date-ordered rows.
        train_ratio: fraction of the earliest rows used for training, in ``(0, 1]``.

    Returns:
        The boundary row index.

    Raises:
        ValueError: if ``train_ratio`` is not in ``(0, 1]``.

    Note:
        A walk-forward (multi-fold rolling) variant is a planned extension
        (WS-1 / juniper-data#168 scope C) and is intentionally not implemented yet.
    """
    if not (0.0 < train_ratio <= 1.0):
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
    idx = int(round(n_samples * train_ratio))
    if n_samples >= 2:
        return min(max(idx, 1), n_samples - 1)
    return max(idx, 0)


def temporal_split_indices(n_samples: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    """The two chronological boundaries of a three-way time-ordered split.

    The three-partition counterpart of :func:`temporal_split_index`. Rows
    ``[0, train_end)`` are train (earliest), ``[train_end, val_end)`` are
    validation, and ``[val_end, n)`` are test (latest).

    **The order is the point.** Validation sits BETWEEN train and test in time,
    so every train target precedes every validation target and every validation
    target precedes every test target. That keeps the no-future-leak guarantee
    transitive: early stopping never sees data from after the reported window,
    and the reported score is measured on the most recent rows -- which is what a
    forecaster is actually asked to do.

    Test takes the remainder rather than its own rounded share, so no window is
    dropped to independent rounding (the same defect the tabular carve had at
    0.8 / 0.1 / 0.1 over four rows).

    For ``n_samples >= 3`` the boundaries are clamped so all three partitions are
    non-empty; below that a three-way split is not expressible and the caller is
    expected to have refused earlier.

    Args:
        n_samples: total number of time-ordered rows (windows).
        train_ratio: fraction of the earliest rows used for training, ``(0, 1]``.
        val_ratio: fraction used for validation, ``[0, 1)``.

    Returns:
        ``(train_end, val_end)``.

    Raises:
        ValueError: if ``train_ratio`` is not in ``(0, 1]``, ``val_ratio`` is not
            in ``[0, 1)``, or the two together leave no room for a test split.
    """
    if not (0.0 < train_ratio <= 1.0):
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0 to leave a test split, got {train_ratio + val_ratio}")

    train_end = int(round(n_samples * train_ratio))
    val_end = train_end + int(round(n_samples * val_ratio))

    if n_samples >= 3:
        # Clamp so each partition keeps at least one row, preserving order.
        train_end = min(max(train_end, 1), n_samples - 2)
        val_end = min(max(val_end, train_end + 1), n_samples - 1)
    else:
        train_end = min(max(train_end, 0), n_samples)
        val_end = min(max(val_end, train_end), n_samples)

    return train_end, val_end


# --- Sizing-mode resolution ---------------------------------------------------
#
# Design section 6.3 requires TWO sizing models, not one:
#
#   * ADDITIVE (the default) -- the generator's native size knob denotes the
#     TRAIN row count, and val/test are generated as ADDITIONAL rows sized as
#     percentages of it. Decisions 2 and 8.
#   * CARVE -- the conventional split of a fixed N by ratios. Section 6.3 admits
#     it "when any of these holds: an explicit CLI switch, environment variable
#     or config setting; the dataset has no generator or no generator specs; or
#     the dataset type is not amenable to synthetic generation". The last clause
#     is why the real-data generators (mnist, csv_import, arc_agi) are
#     carve-only: they cannot conjure additional rows to honour a train count.

SIZING_MODE_ADDITIVE: str = "additive"
SIZING_MODE_CARVE: str = "carve"

#: Every legal value of a generator's ``sizing_mode`` parameter.
SIZING_MODES: tuple[str, ...] = (SIZING_MODE_ADDITIVE, SIZING_MODE_CARVE)


def resolve_partition_counts(
    *,
    sizing_mode: str,
    n_native: int,
    train_ratio: float = 1.0,
    val_ratio: float = 0.0,
    test_ratio: float = 0.0,
    val_percent: float = DEFAULT_VAL_PERCENT,
    test_percent: float = DEFAULT_TEST_PERCENT,
) -> dict[str, int]:
    """Resolve a generator's partition row counts under either sizing model.

    This is the single place the two models differ, so a generator does not
    branch on the mode itself -- it asks for counts and a raw row target, then
    generates that many rows and cuts them.

    Args:
        sizing_mode: ``"additive"`` or ``"carve"``.
        n_native: rows the generator's size knob natively describes. Under
            ``additive`` this IS the train count; under ``carve`` it is the
            total to be divided.
        train_ratio: carve mode only -- train's share of ``n_native``.
        val_ratio: carve mode only -- val's share of ``n_native``.
        test_ratio: carve mode only -- test's share of ``n_native``.
        val_percent: additive mode only -- val rows as a percentage of train.
        test_percent: additive mode only -- test rows as a percentage of train.

    Returns:
        Dictionary with ``n_train``, ``n_val``, ``n_test``, ``n_total`` and
        ``n_raw_required`` -- the number of rows the generator must produce
        before cutting. Under ``additive`` that is ``n_total``; under ``carve``
        it is ``n_native``, because a carve invents no rows.

    Raises:
        ValueError: If ``sizing_mode`` is not a known mode, ``n_native`` is
            below 1, or a carve's ratios are outside ``[0, 1]``.
    """
    if sizing_mode not in SIZING_MODES:
        raise ValueError(f"sizing_mode must be one of {SIZING_MODES}. Got {sizing_mode!r}")

    if n_native < 1:
        raise ValueError(f"n_native must be at least 1. Got {n_native}")

    if sizing_mode == SIZING_MODE_ADDITIVE:
        counts = partition_row_counts(n_native, val_percent, test_percent)
        counts["n_raw_required"] = counts["n_total"]
        return counts

    for name, ratio in (("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)):
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"{name} must be between 0 and 1. Got {ratio}")

    n_train = int(np.round(n_native * train_ratio))
    n_val = int(np.round(n_native * val_ratio))
    n_test = int(np.round(n_native * test_ratio))

    # When the ratios account for the WHOLE dataset, the last partition absorbs
    # the rounding remainder instead of being rounded independently.
    #
    # Rounding all three separately loses rows. At 0.8 / 0.1 / 0.1 over four
    # rows it yields 3 + 0 + 0 -- a quarter of the dataset silently discarded,
    # and small real-data fixtures are exactly where that bites. The two-way
    # code this replaces had the same protection in the form
    # ``n_test = n_samples - n_train``; dropping it while adding a third
    # partition would have been a regression, not a new limitation.
    #
    # The guard is deliberately conditional. A caller who asks for 0.5 / 0.0 /
    # 0.2 has asked for 70 % of the rows and must keep getting a 20 % test
    # partition -- silently inflating it to 50 % to "use everything" would be
    # answering a question they did not ask.
    if abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
        n_test = n_native - n_train - n_val

        # Independently-rounded train and val can TOGETHER exceed n_native, which makes the
        # absorbed remainder negative. 0.7 / 0.3 / 0.0 over 5 rows is the concrete case:
        # np.round(3.5) = 4 and np.round(1.5) = 2, so n_test = 5 - 4 - 2 = -1 and
        # `split_three_way` raises on a request the params model accepted as valid.
        #
        # The overflow trim below cannot catch it: with n_test already negative the sum is
        # exactly n_native, so `overflow` is 0 and the trim never fires. A guard that keys on
        # the TOTAL cannot see a negative PART -- the sum is the wrong unit for the question.
        #
        # Absorb the shortfall from the END, val before train, matching the trim order below
        # and for the same reason: shrinking train silently moves the partition every existing
        # baseline is measured against.
        if n_test < 0:
            deficit = -n_test
            n_test = 0
            take = min(deficit, n_val)
            n_val -= take
            deficit -= take
            n_train -= deficit

    # A carve invents no rows, so an over-subscribed request is trimmed from the
    # END -- test first, then val. Trimming train would silently shrink the
    # partition every existing baseline is measured against.
    overflow = n_train + n_val + n_test - n_native
    if overflow > 0:
        take = min(overflow, n_test)
        n_test -= take
        overflow -= take
    if overflow > 0:
        n_val -= min(overflow, n_val)

    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_total": n_train + n_val + n_test,
        "n_raw_required": n_native,
    }


def per_unit_count(n_required: int, n_units: int) -> int:
    """Per-unit size knob that yields at least ``n_required`` rows.

    Generators whose size knob is per-spiral / per-quadrant / per-class need to
    be asked for a larger unit under additive sizing. Rounding is UP so the
    generator never comes up short; the surplus rows are dropped by
    :func:`split_three_way`, which cuts exactly the counts it is given.

    Scaling every unit equally is what keeps class balance intact -- the surplus
    is discarded after shuffling, so it is not taken from any one class.

    Args:
        n_required: total rows needed across all units.
        n_units: number of units (spirals, quadrants, classes).

    Returns:
        The per-unit count.

    Raises:
        ValueError: If ``n_units`` is below 1.
    """
    if n_units < 1:
        raise ValueError(f"n_units must be at least 1. Got {n_units}")
    return int(-(-n_required // n_units))


def resolve_counts_for_params(params: Any, n_native: int) -> dict[str, int]:
    """:func:`resolve_partition_counts` driven by a generator's params model.

    Duck-typed on purpose: it reads the fields
    ``juniper_data.core.partition_params.PartitionParams`` contributes plus the
    generator's own ``train_ratio`` / ``test_ratio``, without importing the
    model (which would make ``split`` depend on the layer that depends on it).

    Args:
        params: a generator params model carrying the partition vocabulary.
        n_native: rows the generator's size knob natively describes.

    Returns:
        The dict :func:`resolve_partition_counts` returns.
    """
    return resolve_partition_counts(
        sizing_mode=params.sizing_mode,
        n_native=n_native,
        train_ratio=getattr(params, "train_ratio", 1.0),
        val_ratio=getattr(params, "val_ratio", 0.0),
        test_ratio=getattr(params, "test_ratio", 0.0),
        val_percent=getattr(params, "val_percent", DEFAULT_VAL_PERCENT),
        test_percent=getattr(params, "test_percent", DEFAULT_TEST_PERCENT),
    )


def partition_and_assemble(
    X: np.ndarray,
    y: np.ndarray,
    counts: dict[str, int],
    seed: int | None,
    shuffle: bool,
    *,
    extras: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Cut into train / val / test and assemble the legacy ``*_full`` pair.

    ``X_full`` is the vstack of the three partitions rather than the raw
    generated array, and that difference is load-bearing. Additive sizing rounds
    a per-unit size knob UP, so a generator can produce a few more rows than the
    partitions need. Reporting the raw array as ``X_full`` would then break the
    ``n_full == n_train + n_val + n_test`` length identity that
    ``test_e2e_workflow`` asserts. Assembling it from the partitions keeps the
    identity exact, and -- because the surplus is discarded from the SHUFFLED
    tail rather than the raw array's end -- the rows dropped are random instead
    of coming out of whichever class the generator emits last.

    Args:
        X: generated feature array, at least ``n_train + n_val + n_test`` rows.
        y: generated label array, same row count as ``X``.
        counts: the dict from :func:`resolve_counts_for_params`.
        seed: random seed for the shuffle.
        shuffle: whether to shuffle before cutting.
        extras: optional per-row metadata arrays, keyed by the name they should carry
            in the result. Each is permuted with the SAME permutation as the rows and
            then truncated to ``X_full``'s length, so ``result[name][i]`` describes
            ``result["X_full"][i]``. A generator that attaches such metadata AFTER the
            split instead is attaching generation-order values to shuffled rows.

    Returns:
        The six partition keys plus ``X_full`` / ``y_full``, and one key per ``extras``
        entry.
    """
    aligned = dict(extras or {})
    split = shuffle_and_split_three_way(
        X,
        y,
        counts["n_train"],
        counts["n_val"],
        counts["n_test"],
        seed=seed,
        shuffle=shuffle,
        extras=aligned,
    )
    split["X_full"] = np.vstack([split["X_train"], split["X_val"], split["X_test"]])
    split["y_full"] = np.vstack([split["y_train"], split["y_val"], split["y_test"]])
    # `split_three_way` slices contiguously in train | val | test order, so `X_full` is
    # exactly the first `n_train + n_val + n_test` shuffled rows. Any surplus the
    # generator produced is dropped from the shuffled TAIL, and the metadata is cut at
    # the same place for the same reason.
    for name, array in aligned.items():
        split[name] = array[: split["X_full"].shape[0]]
    return split
