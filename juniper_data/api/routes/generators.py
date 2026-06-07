"""Generator endpoints for listing and describing available generators."""

from typing import Any

from fastapi import APIRouter, HTTPException
from starlette import status

from juniper_data.core.models import GeneratorInfo
from juniper_data.generators.arc_agi import VERSION as ARC_AGI_VERSION
from juniper_data.generators.arc_agi import ArcAgiGenerator, ArcAgiParams
from juniper_data.generators.checkerboard import VERSION as CHECKERBOARD_VERSION
from juniper_data.generators.checkerboard import CheckerboardGenerator, CheckerboardParams
from juniper_data.generators.circles import VERSION as CIRCLES_VERSION
from juniper_data.generators.circles import CirclesGenerator, CirclesParams
from juniper_data.generators.csv_import import VERSION as CSV_IMPORT_VERSION
from juniper_data.generators.csv_import import CsvImportGenerator, CsvImportParams
from juniper_data.generators.equities import VERSION as EQUITIES_VERSION
from juniper_data.generators.equities import EquitiesGenerator, EquitiesParams
from juniper_data.generators.gaussian import VERSION as GAUSSIAN_VERSION
from juniper_data.generators.gaussian import GaussianGenerator, GaussianParams
from juniper_data.generators.mnist import VERSION as MNIST_VERSION
from juniper_data.generators.mnist import MnistGenerator, MnistParams
from juniper_data.generators.moon import VERSION as MOON_VERSION
from juniper_data.generators.moon import MoonGenerator, MoonParams
from juniper_data.generators.spiral import VERSION as SPIRAL_VERSION
from juniper_data.generators.spiral import SpiralGenerator, SpiralParams
from juniper_data.generators.xor import VERSION as XOR_VERSION
from juniper_data.generators.xor import XorGenerator, XorParams

router = APIRouter(prefix="/generators", tags=["generators"])

GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    # ``task_type`` is declared per generator and drives the dataset route's
    # metadata dispatch (WS-1 / juniper-data#168): "classification" generators
    # get n_classes + class_distribution from their one-hot y; future
    # "regression" generators leave those None. All current generators are
    # classification (equities carries an auxiliary y_reg_* rider, but its
    # canonical target is the one-hot next-day direction).
    "spiral": {
        "generator": SpiralGenerator,
        "params_class": SpiralParams,
        "version": SPIRAL_VERSION,
        "task_type": "classification",
        "description": "Multi-spiral classification dataset generator. Generates N interleaved spiral arms with configurable points, rotations, and noise.",
    },
    "xor": {
        "generator": XorGenerator,
        "params_class": XorParams,
        "version": XOR_VERSION,
        "task_type": "classification",
        "description": "XOR classification dataset generator. Generates points in 4 quadrants with opposite classes in diagonal quadrants.",
    },
    "gaussian": {
        "generator": GaussianGenerator,
        "params_class": GaussianParams,
        "version": GAUSSIAN_VERSION,
        "task_type": "classification",
        "description": "Gaussian blobs classification dataset generator. Generates mixture-of-Gaussians with configurable centers and covariance.",
    },
    "circles": {
        "generator": CirclesGenerator,
        "params_class": CirclesParams,
        "version": CIRCLES_VERSION,
        "task_type": "classification",
        "description": "Concentric circles classification dataset generator. Generates binary classification with inner and outer circle classes.",
    },
    # XREPO-01b / DC-02 (2026-04-24): added to align with the
    # juniper-data-client ``GENERATOR_MOON`` constant, which previously
    # referenced a server generator that did not exist.
    "moon": {
        "generator": MoonGenerator,
        "params_class": MoonParams,
        "version": MOON_VERSION,
        "task_type": "classification",
        "description": "Two interleaving half-moons classification dataset generator. Generates binary classification with upper and lower half-circle classes.",
    },
    "checkerboard": {
        "generator": CheckerboardGenerator,
        "params_class": CheckerboardParams,
        "version": CHECKERBOARD_VERSION,
        "task_type": "classification",
        "description": "Checkerboard pattern classification dataset generator. Generates 2D grid with alternating class squares.",
    },
    "csv_import": {
        "generator": CsvImportGenerator,
        "params_class": CsvImportParams,
        "version": CSV_IMPORT_VERSION,
        "task_type": "classification",
        "description": "CSV/JSON import generator for custom datasets. Import data from CSV or JSON files with configurable feature and label columns.",
    },
    "equities": {
        "generator": EquitiesGenerator,
        "params_class": EquitiesParams,
        "version": EQUITIES_VERSION,
        "task_type": "classification",
        "description": "S&P 500 equities time-series generator. Daily OHLCV (2000->present) from Yahoo Finance plus SEC EDGAR shares/market-cap, with 52-week high/low, configurable-purchase-date cost basis, and dual targets: next-day direction (one-hot y_*) and next-day close (y_reg_*).",
    },
    "mnist": {
        "generator": MnistGenerator,
        "params_class": MnistParams,
        "version": MNIST_VERSION,
        "task_type": "classification",
        "description": "MNIST and Fashion-MNIST dataset generator. Downloads and prepares standard handwritten digit or fashion item classification datasets.",
    },
    "arc_agi": {
        "generator": ArcAgiGenerator,
        "params_class": ArcAgiParams,
        "version": ARC_AGI_VERSION,
        "task_type": "classification",
        "description": "ARC-AGI (Abstraction and Reasoning Corpus) dataset generator. Generates visual reasoning tasks from the ARC benchmark.",
    },
}


@router.get("", response_model=list[GeneratorInfo])
async def list_generators() -> list[GeneratorInfo]:
    """List all available dataset generators with their info.

    Returns:
        List of generator information objects including name, version,
        description, and parameter schema.
    """
    generators: list[GeneratorInfo] = []
    generators.extend(
        GeneratorInfo(
            name=name,
            version=info["version"],
            description=info["description"],
            schema=info["params_class"].model_json_schema(),
        )
        for name, info in GENERATOR_REGISTRY.items()
    )
    return generators


@router.get("/{name}/schema")
async def get_generator_schema(name: str) -> dict[str, Any]:
    """Get the JSON schema for a generator's parameters.

    Args:
        name: Generator name (e.g., "spiral").

    Returns:
        JSON schema dictionary describing the generator's parameters.

    Raises:
        HTTPException: 404 if generator not found.
    """
    if name not in GENERATOR_REGISTRY:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Generator '{name}' not found")

    params_class = GENERATOR_REGISTRY[name]["params_class"]
    return params_class.model_json_schema()
