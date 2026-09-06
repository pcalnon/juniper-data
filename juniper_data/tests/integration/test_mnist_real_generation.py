"""Real-generation integration test for the MNIST generator (plan unit D2).

Unlike the unit suite (which mocks ``hf_load_dataset``), this test exercises
``MnistGenerator.generate`` for real against the Hugging Face Hub with a tiny
``n_samples``, asserting the NPZ data contract on the result. It is the
regression that would have caught the missing ``datasets`` dependency
(juniper-ml training-runtime-defects plan §4 I-5: 71 masked ImportError
tracebacks -> HTTP 500) and the bare-repo-id break under ``datasets>=5``.

Skip posture (must never turn CI red on runners without the capability):

- ``datasets`` not installed -> module-level importorskip.
- No network AND no seeded HF cache -> skip with the underlying error as the
  reason (connectivity errors are detected across the exception chain).

A missing Pillow (image decode) deliberately FAILS instead of skipping: the
``mnist`` extra ships ``datasets[vision]``, so a Pillow-less environment means
the packaging contract regressed.
"""

import numpy as np
import pytest

pytest.importorskip("datasets", reason="mnist extra not installed (pip install 'juniper-data[mnist]')")

from juniper_data.generators.mnist.generator import MnistGenerator  # noqa: E402
from juniper_data.generators.mnist.params import MnistParams  # noqa: E402
from juniper_data.tests.partitions import whole

# Class-name fragments that identify "the Hub is unreachable and no cache can
# serve the request" failures anywhere in the exception chain: builtin/requests
# ConnectionError family, socket.gaierror, read/connect timeouts, urllib3
# MaxRetryError/NameResolutionError, huggingface_hub LocalEntryNotFoundError /
# OfflineModeIsEnabled, and datasets' DatasetNotFoundError (also raised when
# the Hub cannot be reached to resolve the repo).
_OFFLINE_ERROR_NAME_FRAGMENTS = (
    "Connection",
    "Timeout",
    "gaierror",
    "LocalEntryNotFound",
    "OfflineMode",
    "DatasetNotFound",
    "MaxRetry",
    "NameResolution",
)


def _offline_reason(exc: BaseException) -> str | None:
    """Return a skip reason when the exception chain indicates the Hub is unavailable, else None."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for klass in type(current).__mro__:
            if any(fragment in klass.__name__ for fragment in _OFFLINE_ERROR_NAME_FRAGMENTS):
                return f"MNIST data unavailable (no network and no seeded HF cache): {type(exc).__name__}: {exc}"
        current = current.__cause__ or current.__context__
    return None


def _generate_or_skip(params: MnistParams) -> dict[str, np.ndarray]:
    """Run the real generation, skipping (never failing) when the Hub is unreachable without a cache."""
    try:
        return MnistGenerator.generate(params)
    except Exception as exc:
        reason = _offline_reason(exc)
        if reason is not None:
            pytest.skip(reason)
        raise


@pytest.mark.integration
@pytest.mark.generators
@pytest.mark.slow
@pytest.mark.timeout(300)
class TestMnistRealGeneration:
    """Real (non-mocked) MNIST generation against the Hugging Face Hub."""

    def test_real_mnist_generation_satisfies_npz_contract(self) -> None:
        """Default params (flatten + one-hot + normalize) yield contract-compliant float32 arrays."""
        n_samples = 64
        result = _generate_or_skip(MnistParams(n_samples=n_samples, seed=42))

        assert set(result) == {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
        for key, array in result.items():
            assert array.dtype == np.float32, f"{key} must be float32, got {array.dtype}"

        # Flattened 28x28 images -> rank-2 (n, 784); one-hot labels -> (n, 10).
        assert whole(result, "X").shape == (n_samples, 784)
        assert whole(result, "y").shape == (n_samples, 10)
        assert result["X_train"].shape[1] == 784
        assert result["y_train"].shape[1] == 10

        # Train/val/test partition the requested samples (carve-only generator,
        # default 0.8 / 0.1 / 0.1). The identity spans THREE partitions now --
        # over train + test alone it would pass only while val is empty, which is
        # the regression it exists to catch. mnist reads a fixed corpus, so no
        # row may be dropped to rounding either.
        assert result["X_val"].shape[0] > 0, "X_val must be non-empty, or the identity below holds vacuously"
        assert result["X_train"].shape[0] + result["X_val"].shape[0] + result["X_test"].shape[0] == n_samples
        assert result["X_train"].shape[0] == result["y_train"].shape[0]
        assert result["X_val"].shape[0] == result["y_val"].shape[0]
        assert result["X_test"].shape[0] == result["y_test"].shape[0]

        # One-hot labels: exactly one class per row.
        np.testing.assert_array_almost_equal(whole(result, "y").sum(axis=1), np.ones(n_samples))

        # Normalized pixels live in [0, 1] and the digits are not a constant image.
        assert whole(result, "X").min() >= 0.0
        assert whole(result, "X").max() <= 1.0
        assert whole(result, "X").std() > 0.0
