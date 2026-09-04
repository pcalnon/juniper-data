"""Parameters for the CSV/JSON import generator."""

from typing import Literal

from pydantic import BaseModel, Field

from juniper_data.core.constants import DEFAULT_GENERATOR_SEED

from .defaults import (
    CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION,
    CSV_IMPORT_DEFAULT_DELIMITER,
    CSV_IMPORT_DEFAULT_FILE_FORMAT,
    CSV_IMPORT_DEFAULT_HEADER,
    CSV_IMPORT_DEFAULT_LABEL_COLUMN,
    CSV_IMPORT_DEFAULT_MAX_BYTES,
    CSV_IMPORT_DEFAULT_NORMALIZE_FEATURES,
    CSV_IMPORT_DEFAULT_ONE_HOT_LABELS,
    CSV_IMPORT_DEFAULT_TEST_RATIO,
    CSV_IMPORT_DEFAULT_TRAIN_RATIO,
)


class CsvImportParams(BaseModel):
    """Configuration parameters for CSV/JSON data import.

    Loads and preprocesses data from CSV or JSON files.
    """

    file_path: str = Field(
        description="Path to the CSV or JSON file to import",
    )
    file_format: Literal["csv", "json", "auto"] = Field(
        default=CSV_IMPORT_DEFAULT_FILE_FORMAT,
        description="File format: 'csv', 'json', or 'auto' (detect from extension)",
    )
    feature_columns: list[str] | None = Field(
        default=None,
        description="Column names for features (None = all except label column)",
    )
    label_column: str = Field(
        default=CSV_IMPORT_DEFAULT_LABEL_COLUMN,
        description="Column name for labels",
    )
    delimiter: str = Field(
        default=CSV_IMPORT_DEFAULT_DELIMITER,
        description="CSV delimiter character",
    )
    header: bool = Field(
        default=CSV_IMPORT_DEFAULT_HEADER,
        description="Whether the file has a header row",
    )
    one_hot_labels: bool = Field(
        default=CSV_IMPORT_DEFAULT_ONE_HOT_LABELS,
        description="One-hot encode labels",
    )
    normalize_features: bool = Field(
        default=CSV_IMPORT_DEFAULT_NORMALIZE_FEATURES,
        description="Normalize features to [0, 1]",
    )
    seed: int | None = Field(default=DEFAULT_GENERATOR_SEED, ge=0, description="Random seed for reproducibility. Defaults to DEFAULT_GENERATOR_SEED so the documented default configuration is REPRODUCIBLE (juniper-data#319); pass None explicitly to opt into a fresh draw per call.")
    max_bytes: int = Field(
        default=CSV_IMPORT_DEFAULT_MAX_BYTES,
        gt=0,
        description="Maximum bytes to read from the source file (APD-DATA-018). A source larger than this is REFUSED unless allow_truncation is set. Omit to use the deployment default (JUNIPER_DATA_CSV_IMPORT_MAX_BYTES).",
    )
    allow_truncation: bool = Field(
        default=CSV_IMPORT_DEFAULT_ALLOW_TRUNCATION,
        description="Accept a partial import when the source exceeds max_bytes. Default false: an oversized source is refused with 422 rather than silently truncated. When true, the import stops at the last complete record inside the cap and the dataset is PERMANENTLY annotated as truncated in its metadata. Can also be enabled deployment-wide via JUNIPER_DATA_CSV_IMPORT_ALLOW_TRUNCATION or the matching .env entry.",
    )
    train_ratio: float = Field(default=CSV_IMPORT_DEFAULT_TRAIN_RATIO, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=CSV_IMPORT_DEFAULT_TEST_RATIO, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
