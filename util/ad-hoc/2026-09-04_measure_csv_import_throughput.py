#!/usr/bin/env python3
"""Measure csv_import parse throughput, to pick a defensible byte cap.

Project:     Juniper
Sub-Project: juniper-data
Application: APD-DATA-018 -- csv_import input bound
Author:      Paul Calnon
Version:     0.1.0
License:     MIT
Status:      ad-hoc (one-shot measurement supporting the cap constant)

``APD-DATA-018``'s remedy is to bound the inputs so generation fits inside the
client's ~30 s request budget. The owner's decision (2026-09-04) is a **byte**
cap for ``csv_import``. A byte cap is only defensible if it is derived from
measured throughput rather than picked round -- so this measures the real
``CsvImportGenerator`` path on synthetic CSVs of increasing size and reports
MB/s plus the byte figure that lands at a chosen wall-clock target.

Deliberately measures the WHOLE ``generate()`` call, not just ``_load_csv``:
the float conversion in ``_convert_to_arrays`` is per-cell and is a large part
of the cost, so timing the read alone would understate the bound and produce a
cap that still blows the budget.

Run:
    python3 util/ad-hoc/2026-09-04_measure_csv_import_throughput.py
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Wall-clock target for the parse itself. The client budget is 30 s; the parse
# must leave room for splitting, checksumming and NPZ persistence, so aim well
# under it.
TARGET_SECONDS = 10.0

ROW_COUNTS = (20_000, 80_000, 200_000)
N_FEATURES = 20


def write_csv(path: Path, n_rows: int, n_features: int) -> int:
    """Write a synthetic numeric CSV; return its size in bytes."""
    rng = random.Random(20260904)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"f{i}" for i in range(n_features)] + ["label"])
        for _ in range(n_rows):
            writer.writerow([f"{rng.random():.6f}" for _ in range(n_features)] + [rng.randint(0, 3)])
    return path.stat().st_size


def main() -> int:
    # Import settings FIRST. Importing the generator module cold pulls
    # juniper_data.api -> app -> routes -> generators -> csv_import, which
    # re-enters the partially-initialized csv_import package and raises
    # ImportError on VERSION. Letting the api package finish first breaks that.
    import juniper_data.api.settings as settings_mod  # noqa: F401

    from juniper_data.generators.csv_import.generator import CsvImportGenerator
    from juniper_data.generators.csv_import.params import CsvImportParams

    print(f"{'rows':>9} {'bytes':>12} {'MB':>7} {'seconds':>9} {'MB/s':>8}")
    rates: list[float] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for n_rows in ROW_COUNTS:
            path = tmpdir / f"bench_{n_rows}.csv"
            size = write_csv(path, n_rows, N_FEATURES)

            # The generator resolves file_path against settings.import_dir, so
            # point that at our temp dir for the duration of the call.
            import juniper_data.api.settings as settings_mod

            settings = settings_mod.get_settings()
            original = settings.import_dir
            object.__setattr__(settings, "import_dir", str(tmpdir)) if hasattr(settings, "__dataclass_fields__") else setattr(settings, "import_dir", str(tmpdir))
            try:
                params = CsvImportParams(file_path=path.name, label_column="label")
                start = time.monotonic()
                CsvImportGenerator.generate(params)
                elapsed = time.monotonic() - start
            finally:
                setattr(settings, "import_dir", original)

            mb = size / (1024 * 1024)
            rate = mb / elapsed
            rates.append(rate)
            print(f"{n_rows:>9} {size:>12} {mb:>7.2f} {elapsed:>9.2f} {rate:>8.2f}")

    median_rate = statistics.median(rates)
    budget_mb = median_rate * TARGET_SECONDS
    print()
    print(f"median throughput : {median_rate:.2f} MB/s")
    print(f"at {TARGET_SECONDS:.0f} s target : {budget_mb:.1f} MB")
    print(f"suggested cap     : {int(budget_mb) // 16 * 16} MB (rounded down to a 16 MB step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
