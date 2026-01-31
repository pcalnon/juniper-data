"""Generator endpoints for listing and describing available generators."""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from juniper_data.core.models import GeneratorInfo
from juniper_data.generators.spiral import VERSION as SPIRAL_VERSION
from juniper_data.generators.spiral import SpiralGenerator, SpiralParams

router = APIRouter(prefix="/generators", tags=["generators"])

GENERATOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "spiral": {
        "generator": SpiralGenerator,
        "params_class": SpiralParams,
        "version": SPIRAL_VERSION,
        "description": "Multi-spiral classification dataset generator. "
        "Generates N interleaved spiral arms with configurable points, rotations, and noise.",
    },
}


@router.get("", response_model=List[GeneratorInfo])
async def list_generators() -> List[Dict[str, Any]]:
    """List all available dataset generators with their info.

    Returns:
        List of generator information objects including name, version,
        description, and parameter schema.
    """
    generators = []
    for name, info in GENERATOR_REGISTRY.items():
        generators.append(
            {
                "name": name,
                "version": info["version"],
                "description": info["description"],
                "schema": info["params_class"].model_json_schema(),
            }
        )
    return generators


@router.get("/{name}/schema")
async def get_generator_schema(name: str) -> Dict[str, Any]:
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
