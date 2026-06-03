"""Background full-build driver for the equities S&P 500 dataset.

Generates the complete S&P 500 equities dataset (default universe, 2000 ->
today) via the equities generator and writes the NPZ artifact plus a metadata
sidecar under ``data/datasets/``. Intended to be launched as a long-running
background job, e.g.::

    PYTHONPATH=<worktree> \
        /opt/miniforge3/envs/JuniperData/bin/python util/ad-hoc/build_equities_full.py

Progress is logged at INFO (one line per ticker) so the run can be tailed.
"""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     build_equities_full.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from juniper_data.core.artifacts import compute_checksum, save_npz
from juniper_data.generators.equities import VERSION, EquitiesGenerator, EquitiesParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_equities_full")

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "datasets"
_CANONICAL = {"X_train", "y_train", "X_test", "y_test", "X_full", "y_full"}


def main() -> None:
    """Build the full S&P 500 dataset and persist the artifact + metadata."""
    start = time.monotonic()
    params = EquitiesParams(start_date="2000-01-01", end_date=None, fundamentals_fill="zero")
    log.info("starting full S&P 500 equities build (symbols=ALL, 2000 -> today)")

    arrays = EquitiesGenerator.generate(params)
    elapsed = time.monotonic() - start

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    npz_path = OUT_DIR / f"equities_sp500_full_{stamp}.npz"
    save_npz(npz_path, arrays)

    meta = {
        "generator": "equities",
        "generator_version": VERSION,
        "params": params.model_dump(),
        "n_samples": int(arrays["X_full"].shape[0]),
        "n_features": int(arrays["X_full"].shape[1]),
        "n_classes": int(arrays["y_full"].shape[1]),
        "n_train": int(arrays["X_train"].shape[0]),
        "n_test": int(arrays["X_test"].shape[0]),
        "n_tickers": int(len(arrays["ticker_vocab"])),
        "checksum": compute_checksum(arrays),
        "elapsed_seconds": round(elapsed, 1),
        "artifact": str(npz_path),
        "size_mb": round(npz_path.stat().st_size / 1e6, 1),
        "extra_arrays": sorted(key for key in arrays if key not in _CANONICAL),
    }
    (OUT_DIR / f"equities_sp500_full_{stamp}.meta.json").write_text(json.dumps(meta, indent=2, default=str))

    log.info("DONE: %d tickers, %d rows, %.1fs -> %s (%.1f MB)", meta["n_tickers"], meta["n_samples"], elapsed, npz_path, meta["size_mb"])
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
