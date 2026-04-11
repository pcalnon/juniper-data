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
