"""Parameters for the ARC-AGI dataset generator."""

from typing import Literal

from pydantic import BaseModel, Field


class ArcAgiParams(BaseModel):
    """Configuration parameters for ARC-AGI dataset loading.

    Loads ARC-AGI tasks from Hugging Face Hub or local JSON files.
    The ARC (Abstraction and Reasoning Corpus) contains grid-based
    reasoning tasks with input/output pairs.
    """

    source: Literal["huggingface", "local"] = Field(
        default="huggingface",
        description="Data source: 'huggingface' or 'local' JSON files",
    )
    local_path: str | None = Field(
        default=None,
        description="Path to local ARC JSON files (required if source='local')",
    )
    subset: Literal["training", "evaluation", "all"] = Field(
        default="training",
        description="Which subset to load: 'training', 'evaluation', or 'all'",
    )
    n_tasks: int | None = Field(
        default=None,
        ge=1,
        description="Limit number of tasks to load (None for all)",
    )
    pad_to: int = Field(
        default=30,
        ge=1,
        le=50,
        description="Pad all grids to this size (max ARC grid is 30x30)",
    )
    pad_value: int = Field(
        default=-1,
        ge=-1,
        le=9,
        description="Value to use for padding (-1 recommended for masking)",
    )
    include_test: bool = Field(
        default=True,
        description="Include test input/output pairs (in addition to train pairs)",
    )
    flatten_pairs: bool = Field(
        default=True,
        description="Flatten all input/output pairs into single arrays",
    )
    seed: int | None = Field(default=None, ge=0, description="Random seed for reproducibility")
    train_ratio: float = Field(default=0.8, gt=0, le=1, description="Fraction of data for training")
    test_ratio: float = Field(default=0.2, ge=0, le=1, description="Fraction of data for testing")
    shuffle: bool = Field(default=True, description="Shuffle before splitting")
