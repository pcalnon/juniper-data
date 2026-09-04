"""Default constants for CSV/JSON import generator.

Mirrors the structure of ``generators/spiral/defaults.py``: per-field
defaults for the Pydantic ``CsvImportParams`` model.
"""

# Format & Parsing Defaults
CSV_IMPORT_DEFAULT_FILE_FORMAT: str = "auto"
CSV_IMPORT_FILE_FORMAT_CHOICES: tuple[str, ...] = ("csv", "json", "auto")
CSV_IMPORT_DEFAULT_LABEL_COLUMN: str = "label"
CSV_IMPORT_DEFAULT_DELIMITER: str = ","
CSV_IMPORT_DEFAULT_HEADER: bool = True

# Preprocessing Defaults
CSV_IMPORT_DEFAULT_ONE_HOT_LABELS: bool = True
CSV_IMPORT_DEFAULT_NORMALIZE_FEATURES: bool = False

# Dataset Splitting Defaults
CSV_IMPORT_DEFAULT_TRAIN_RATIO: float = 0.8
CSV_IMPORT_DEFAULT_TEST_RATIO: float = 0.2

# Input Bound (APD-DATA-018) -- re-exported, not defined here.
#
# These two live in ``juniper_data/core/limits.py`` because ``api/settings.py``
# needs them as its deployment defaults, and settings cannot import from this
# package without a cycle (importing any csv_import submodule runs
# ``__init__.py`` -> ``generator.py`` -> ``api.settings``). They are re-exported
# here so every csv_import default is still discoverable in one place; the
# rationale for the 128 MiB figure is at the definition.
from juniper_data.core.limits import (  # noqa: E402  (re-export, kept beside the other defaults)
    CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION,
    CSV_IMPORT_DEFAULT_MAX_BYTES,
)

__all__ = [
    "CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION",
    "CSV_IMPORT_DEFAULT_DELIMITER",
    "CSV_IMPORT_DEFAULT_FILE_FORMAT",
    "CSV_IMPORT_DEFAULT_HEADER",
    "CSV_IMPORT_DEFAULT_LABEL_COLUMN",
    "CSV_IMPORT_DEFAULT_MAX_BYTES",
    "CSV_IMPORT_DEFAULT_NORMALIZE_FEATURES",
    "CSV_IMPORT_DEFAULT_ONE_HOT_LABELS",
    "CSV_IMPORT_DEFAULT_TEST_RATIO",
    "CSV_IMPORT_DEFAULT_TRAIN_RATIO",
    "CSV_IMPORT_FILE_FORMAT_CHOICES",
]
