"""Parameters for the CSV/JSON import generator."""

from typing import Literal

from pydantic import BaseModel, Field


class CsvImportParams(BaseModel):
    """Configuration parameters for CSV/JSON data import.

    Loads and preprocesses data from CSV or JSON files.
    """

    file_path: str = Field(
        description="Path to the CSV or JSON file to import",
    )
    file_format: Literal["csv", "json", "auto"] = Field(
        default="auto",
        description="File format: 'csv', 'json', or 'auto' (detect from extension)",
    )
    feature_columns: list[str] | None = Field(
        default=None,
        description="Column names for features (None = all except label column)",
    )
    label_column: str = Field(
        default="label",
        description="Column name for labels",
    )
    delimiter: str = Field(
        default=",",
        description="CSV delimiter character",
    )
    header: bool = Field(
        default=True,
        description="Whether the file has a header row",
    )
    one_hot_labels: bool = Field(
        default=True,
        description="One-hot encode labels",
    )
    normalize_features: bool = Field(
        default=False,
        description="Normalize features to [0, 1]",
    )
    seed: int | None = Field(default=None, ge=0, description="Random seed for reproducibility")
    train_ratio: float = Field(default=0.8, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=0.2, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
