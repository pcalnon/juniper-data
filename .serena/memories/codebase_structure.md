# Codebase Structure - Juniper Data

## Directory Layout

```
juniper_data/                   # Project root (also a subdir of JuniperData)
├── juniper_data/               # Main Python package
│   ├── __init__.py             # Package init with version
│   ├── __main__.py             # Module entry point (runs API server)
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── artifacts.py        # Artifact management
│   │   ├── dataset_id.py       # Dataset ID generation
│   │   ├── models.py           # Core Pydantic models
│   │   └── split.py            # Dataset splitting utilities
│   ├── generators/             # Dataset generators
│   │   ├── __init__.py
│   │   └── spiral/             # Spiral dataset generator
│   │       ├── __init__.py
│   │       ├── defaults.py     # Default values/constants
│   │       ├── generator.py    # Main generator class
│   │       └── params.py       # Parameter models
│   ├── storage/                # Dataset persistence
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base storage
│   │   ├── local_fs.py         # Filesystem storage
│   │   └── memory.py           # In-memory storage
│   ├── api/                    # FastAPI application
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI app factory
│   │   ├── settings.py         # API configuration
│   │   └── routes/             # API route handlers
│   │       ├── __init__.py
│   │       ├── datasets.py     # Dataset endpoints
│   │       ├── generators.py   # Generator endpoints
│   │       └── health.py       # Health check endpoint
│   └── tests/                  # Test suite
│       ├── __init__.py
│       ├── conftest.py         # Pytest fixtures
│       ├── fixtures/           # Test fixtures
│       │   ├── generate_golden_datasets.py
│       │   └── golden_datasets/  # Reference datasets
│       ├── unit/               # Unit tests
│       │   ├── test_spiral_generator.py
│       │   ├── test_storage.py
│       │   ├── test_api_*.py
│       │   └── ...
│       └── integration/        # Integration tests
│           ├── test_api.py
│           └── test_storage_workflow.py
├── conf/                       # Configuration files
├── data/                       # Data directory (gitignored)
├── docs/                       # Documentation
├── logs/                       # Log files (gitignored)
├── reports/                    # Test/coverage reports
├── util/                       # Utility scripts
│   ├── run.bash
│   ├── run_tests.bash
│   └── run_all_tests.bash
├── pyproject.toml              # Project configuration
├── .pre-commit-config.yaml     # Pre-commit hooks
├── README.md                   # Project documentation
├── AGENTS.md / CLAUDE.md       # AI agent guidance
└── CHANGELOG.md                # Version history
```

## Component Responsibilities

| Component | Description |
|-----------|-------------|
| `core/` | Base classes, exceptions, configuration, utilities |
| `core/models.py` | Pydantic models for datasets and metadata |
| `core/dataset_id.py` | Deterministic dataset ID generation |
| `core/split.py` | Train/validation/test splitting |
| `generators/` | Dataset generation implementations |
| `generators/spiral/` | Two-spiral classification dataset generator |
| `storage/` | Dataset persistence layer |
| `storage/base.py` | Abstract storage interface |
| `storage/local_fs.py` | Filesystem-based storage |
| `storage/memory.py` | In-memory storage for testing |
| `api/` | FastAPI REST service |
| `api/app.py` | App factory and lifespan management |
| `api/routes/` | Endpoint handlers |
| `tests/` | Comprehensive test suite |

## Key Files

- `juniper_data/__main__.py` - Entry point for `python -m juniper_data`
- `juniper_data/api/app.py` - FastAPI application creation
- `juniper_data/generators/spiral/generator.py` - Main spiral generator
- `juniper_data/core/models.py` - Core data models
- `pyproject.toml` - All project configuration (build, tools, dependencies)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/datasets` | GET | List datasets |
| `/datasets/{id}` | GET | Get specific dataset |
| `/generators/spiral` | POST | Generate spiral dataset |
| `/generators/spiral/config` | GET | Get generator configuration |
