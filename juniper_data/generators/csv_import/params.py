"""Parameters for the CSV/JSON import generator."""

from typing import Literal

from pydantic import BaseModel, Field

from juniper_data.core.constants import DEFAULT_GENERATOR_SEED

from .defaults import (
    CSV_IMPORT_DEFAULT_DELIMITER,
    CSV_IMPORT_DEFAULT_FILE_FORMAT,
    CSV_IMPORT_DEFAULT_HEADER,
    CSV_IMPORT_DEFAULT_LABEL_COLUMN,
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
    train_ratio: float = Field(default=CSV_IMPORT_DEFAULT_TRAIN_RATIO, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=CSV_IMPORT_DEFAULT_TEST_RATIO, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
