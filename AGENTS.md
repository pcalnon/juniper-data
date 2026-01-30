# AGENTS.md - Juniper Data Project Guide

**Project**: Juniper Data - Dataset Generation Service  
**Version**: 0.1.0  
**License**: MIT License  
**Author**: Paul Calnon  
**Last Updated**: 2026-01-29

---

## Quick Reference

### Essential Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Install with API support
pip install -e ".[api]"

# Install everything
pip install -e ".[all]"

# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run tests with coverage
pytest --cov=juniper_data --cov-report=html

# Run a specific test file
pytest tests/unit/test_spiral_generator.py -v

# Type checking with mypy
mypy juniper_data

# Linting with flake8
flake8 juniper_data --max-line-length=120 --extend-ignore=E203,E266,E501,W503

# Format checking with black
black --check --diff juniper_data tests

# Import sorting check with isort
isort --check-only --diff juniper_data tests

# Start API server (development)
uvicorn juniper_data.api.main:app --reload --host 0.0.0.0 --port 8000

# Start API server (production)
uvicorn juniper_data.api.main:app --host 0.0.0.0 --port 8000
```

---

## Project Architecture

### Directory Structure

```
JuniperData/
├── juniper_data/              # Main package
│   ├── __init__.py            # Package init with version
│   ├── core/                  # Core functionality
│   │   └── __init__.py
│   ├── generators/            # Dataset generators
│   │   ├── __init__.py
│   │   └── spiral/            # Spiral dataset generator
│   │       └── __init__.py
│   ├── storage/               # Dataset persistence
│   │   └── __init__.py
│   └── api/                   # FastAPI application
│       ├── __init__.py
│       └── routes/            # API route handlers
│           └── __init__.py
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── unit/                  # Unit tests
│   │   └── __init__.py
│   └── integration/           # Integration tests
│       └── __init__.py
├── pyproject.toml             # Project configuration
├── README.md                  # Project documentation
└── AGENTS.md                  # This file
```

### Component Overview

| Component | Purpose |
|-----------|---------|
| `core/` | Base classes, exceptions, configuration |
| `generators/` | Dataset generation implementations |
| `generators/spiral/` | Two-spiral classification dataset |
| `storage/` | Dataset persistence and retrieval |
| `api/` | FastAPI REST service |
| `api/routes/` | API endpoint handlers |

---

## Code Style Conventions

Following JuniperCascor patterns:

### Naming Conventions

**Constants**:
- Uppercase with underscores, prefixed by component: `_DATA_DEFAULT_NOISE`
- Hierarchical naming: `_SPIRAL_GENERATOR_DEFAULT_POINTS`

**Classes**:
- PascalCase: `SpiralGenerator`, `DatasetStorage`

**Methods/Functions**:
- snake_case: `generate_dataset`, `get_configuration`

**Private Members**:
- Single underscore prefix: `_internal_method`, `_private_attribute`

**Dunder Methods**:
- Double underscore: `__init__`, `__repr__`

### Code Formatting

- Line length: 120 characters
- Black formatter
- isort for imports (profile: black)
- Type hints required for all public methods

### Documentation

- Docstrings for all public classes and methods
- Google-style docstring format
- Type annotations in signatures, not docstrings

---

## Dependencies

### Core Dependencies

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `pydantic` | Data validation and settings |

### API Dependencies (Optional)

| Library | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |

### Development Dependencies

| Library | Purpose |
|---------|---------|
| `pytest` | Testing framework |
| `pytest-cov` | Coverage reporting |
| `black` | Code formatting |
| `isort` | Import sorting |
| `mypy` | Static type checking |

---

## Testing

### Test Organization

- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for full workflows

### Test Markers

```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.spiral        # Spiral generator tests
@pytest.mark.api           # API endpoint tests
@pytest.mark.generators    # Generator tests
@pytest.mark.storage       # Storage tests
```

### Test Naming

- Files: `test_<component>.py`
- Classes: `Test<ComponentName>`
- Methods: `test_<behavior_under_test>`

---

## API Design

### REST Conventions

- Use nouns for resources: `/datasets`, `/generators`
- Use HTTP methods appropriately: GET, POST, PUT, DELETE
- Return proper status codes
- Include pagination for list endpoints

### Response Format

```python
{
    "status": "success",
    "data": { ... },
    "meta": {
        "timestamp": "...",
        "version": "0.1.0"
    }
}
```

---

## Security Notes

- No secrets or API keys in codebase
- Validate all input data with Pydantic
- Sensitive files excluded via `.gitignore`

---

## Development Workflow

### Adding New Features

1. Create feature in appropriate module
2. Add Pydantic models for validation
3. Add tests in `tests/unit/` or `tests/integration/`
4. Update documentation
5. Run tests and type checking

### Adding New Generators

1. Create new subpackage under `generators/`
2. Implement generator class following `SpiralGenerator` pattern
3. Add API routes in `api/routes/`
4. Add comprehensive tests
