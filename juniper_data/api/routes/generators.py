"""Generator endpoints for listing and describing available generators."""

from typing import Any

from fastapi import APIRouter, HTTPException

from juniper_data.core.models import GeneratorInfo
from juniper_data.generators.arc_agi import VERSION as ARC_AGI_VERSION
from juniper_data.generators.arc_agi import ArcAgiGenerator, ArcAgiParams
from juniper_data.generators.checkerboard import VERSION as CHECKERBOARD_VERSION
from juniper_data.generators.checkerboard import CheckerboardGenerator, CheckerboardParams
from juniper_data.generators.circles import VERSION as CIRCLES_VERSION
from juniper_data.generators.circles import CirclesGenerator, CirclesParams
from juniper_data.generators.csv_import import VERSION as CSV_IMPORT_VERSION
from juniper_data.generators.csv_import import CsvImportGenerator, CsvImportParams
from juniper_data.generators.gaussian import VERSION as GAUSSIAN_VERSION
from juniper_data.generators.gaussian import GaussianGenerator, GaussianParams
from juniper_data.generators.mnist import VERSION as MNIST_VERSION
from juniper_data.generators.mnist import MnistGenerator, MnistParams
from juniper_data.generators.spiral import VERSION as SPIRAL_VERSION
from juniper_data.generators.spiral import SpiralGenerator, SpiralParams
from juniper_data.generators.xor import VERSION as XOR_VERSION
from juniper_data.generators.xor import XorGenerator, XorParams

router = APIRouter(prefix="/generators", tags=["generators"])

GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "spiral": {
        "generator": SpiralGenerator,
        "params_class": SpiralParams,
        "version": SPIRAL_VERSION,
        "description": "Multi-spiral classification dataset generator. "
        "Generates N interleaved spiral arms with configurable points, rotations, and noise.",
    },
    "xor": {
        "generator": XorGenerator,
        "params_class": XorParams,
        "version": XOR_VERSION,
        "description": "XOR classification dataset generator. "
        "Generates points in 4 quadrants with opposite classes in diagonal quadrants.",
    },
    "gaussian": {
        "generator": GaussianGenerator,
        "params_class": GaussianParams,
        "version": GAUSSIAN_VERSION,
        "description": "Gaussian blobs classification dataset generator. "
        "Generates mixture-of-Gaussians with configurable centers and covariance.",
    },
    "circles": {
        "generator": CirclesGenerator,
        "params_class": CirclesParams,
        "version": CIRCLES_VERSION,
        "description": "Concentric circles classification dataset generator. "
        "Generates binary classification with inner and outer circle classes.",
    },
    "checkerboard": {
        "generator": CheckerboardGenerator,
        "params_class": CheckerboardParams,
        "version": CHECKERBOARD_VERSION,
        "description": "Checkerboard pattern classification dataset generator. "
        "Generates 2D grid with alternating class squares.",
    },
    "csv_import": {
        "generator": CsvImportGenerator,
        "params_class": CsvImportParams,
        "version": CSV_IMPORT_VERSION,
        "description": "CSV/JSON import generator for custom datasets. "
        "Import data from CSV or JSON files with configurable feature and label columns.",
    },
    "mnist": {
        "generator": MnistGenerator,
        "params_class": MnistParams,
        "version": MNIST_VERSION,
        "description": "MNIST and Fashion-MNIST dataset generator. "
        "Downloads and prepares standard handwritten digit or fashion item classification datasets.",
    },
    "arc_agi": {
        "generator": ArcAgiGenerator,
        "params_class": ArcAgiParams,
        "version": ARC_AGI_VERSION,
        "description": "ARC-AGI (Abstraction and Reasoning Corpus) dataset generator. "
        "Generates visual reasoning tasks from the ARC benchmark.",
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
        raise HTTPException(status_code=404, detail=f"Generator '{name}' not found")

    params_class = GENERATOR_REGISTRY[name]["params_class"]
    return params_class.model_json_schema()
