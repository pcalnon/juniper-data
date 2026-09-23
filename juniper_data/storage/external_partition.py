"""Three-way partitioning and identity for the external-source store adapters.

juniper-data#411. ``HuggingFaceDatasetStore`` and ``KaggleDatasetStore`` cut their data
two-way (``X[:n_train]`` / ``X[n_train:]``), emitted the retired ``X_full`` / ``y_full``
pair, and stamped ``generator_version="1.0.0"`` -- below decision 11's 3.0.0 floor, on a
path the generator floor guard never enumerated because it is not under
``juniper_data.generators``.

They now partition the way the other real-data sources do (``mnist``, ``csv_import``,
``arc_agi``). A fixed corpus cannot synthesise rows, so the carve model of design section
6.3 is the only one available, with the same 0.8 / 0.1 / 0.1 default.

Their dataset ID now also carries the version and the request parameters, through the
same :func:`generate_dataset_id` the generator path uses. The old ID was
``hf-<name>-<rows>``: it held neither, so bumping the version could not stop a cached
two-way artifact being returned for a three-way request under the same ID, and two
different partitionings of one dataset collided on it.
"""

from typing import Any

import numpy as np

from juniper_data.core.dataset_id import generate_dataset_id
from juniper_data.core.partition_params import DEFAULT_CARVE_ONLY_VAL_RATIO
from juniper_data.core.split import SIZING_MODE_CARVE, resolve_partition_counts, split_three_way

#: Contract version of the arrays the external stores emit. 3.0.0 is decision 11's
#: floor: three partitions, no ``*_full``. It is hashed into the dataset ID.
EXTERNAL_STORE_VERSION: str = "3.0.0"

#: Carve defaults, matching the other real-data sources (``MNIST_DEFAULT_TRAIN_RATIO``,
#: ``DEFAULT_CARVE_ONLY_VAL_RATIO``, ``MNIST_DEFAULT_TEST_RATIO``).
EXTERNAL_STORE_DEFAULT_TRAIN_RATIO: float = 0.8
EXTERNAL_STORE_DEFAULT_VAL_RATIO: float = DEFAULT_CARVE_ONLY_VAL_RATIO
EXTERNAL_STORE_DEFAULT_TEST_RATIO: float = 0.1

# Float tolerance for the ratio-sum check: 0.56 + 0.34 + 0.1 evaluates to
# 1.0000000000000002, which is not a request for more rows than exist. The same
# tolerance is what makes resolve_partition_counts treat such a sum as a whole-dataset
# carve and let test absorb the rounding remainder.
_RATIO_SUM_TOLERANCE: float = 1e-9


def carve_three_way(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Carve a fixed set of rows into contiguous train / val / test blocks.

    Rows are cut in their current order. A store that shuffles does so before calling
    this, so the partitions are index-disjoint by construction whatever the order.

    Args:
        X: Feature array of shape ``(n_samples, ...)``.
        y: Label array with the same number of rows.
        train_ratio: Train's share of the rows. Must be greater than 0.
        val_ratio: Val's share of the rows.
        test_ratio: Test's share of the rows.

    Returns:
        ``(arrays, counts)``: the six partition keys, and the dict
        :func:`resolve_partition_counts` returned. When the three ratios sum to 1 the
        last partition absorbs the rounding remainder; when they sum to less, the
        rows beyond ``counts["n_total"]`` are left out rather than folded in.

    Raises:
        ValueError: If a ratio is outside ``[0, 1]``, ``train_ratio`` is 0, or the
            three together ask for more rows than exist.
    """
    for name, ratio in (("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)):
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1. Got {ratio}")
    if train_ratio <= 0.0:
        raise ValueError(f"train_ratio must be greater than 0. Got {train_ratio}")
    total = train_ratio + val_ratio + test_ratio
    if total > 1.0 + _RATIO_SUM_TOLERANCE:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must be <= 1.0, got {total}")

    counts = resolve_partition_counts(
        sizing_mode=SIZING_MODE_CARVE,
        n_native=X.shape[0],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    arrays = split_three_way(X, y, counts["n_train"], counts["n_val"], counts["n_test"])
    return arrays, counts


def external_dataset_id(prefix: str, generator: str, params: dict[str, Any]) -> str:
    """Build a store's dataset ID: a readable prefix plus the canonical hashed ID.

    Args:
        prefix: Human-readable stem, e.g. ``"hf-mnist"`` or ``"kaggle-owner-iris"``.
        generator: The ``DatasetMeta.generator`` value, e.g. ``"huggingface"``.
        params: The parameters recorded in ``DatasetMeta.params``. A ``None`` seed
            adds a per-call nonce, exactly as it does for a generator (BUG-JD-04).

    Returns:
        ``"<prefix>-<generator>-<version>-<hash>"``.
    """
    return f"{prefix}-{generate_dataset_id(generator, EXTERNAL_STORE_VERSION, params)}"
