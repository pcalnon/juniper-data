# Juniper Data

Dataset generation and management service for the Juniper ecosystem.

## Overview

Juniper Data provides a centralized service for generating, storing, and serving datasets used by the Juniper neural network projects. It supports various dataset types including the classic two-spiral classification problem.

## Installation

### Basic Installation

```bash
pip install -e .
```

### With API Support

```bash
pip install -e ".[api]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Full Installation

```bash
pip install -e ".[all]"
```

## Quick Start

### Generate a Spiral Dataset

```python
from juniper_data.generators.spiral import SpiralGenerator

generator = SpiralGenerator()
dataset = generator.generate(n_points=100, n_spirals=2, noise=0.1)
```

### Start the API Server

```bash
uvicorn juniper_data.api.app:app --reload
```

## API Endpoints

| Endpoint                    | Method | Description                        |
| --------------------------- | ------ | ---------------------------------- |
| `/health`                   | GET    | Health check endpoint              |
| `/datasets`                 | GET    | List available datasets            |
| `/datasets/{id}`            | GET    | Get a specific dataset             |
| `/generators/spiral`        | POST   | Generate a new spiral dataset      |
| `/generators/spiral/config` | GET    | Get spiral generator configuration |

## Project Structure

```bash
JuniperData/
├── juniper_data/
│   ├── core/           # Core functionality and base classes
│   ├── generators/     # Dataset generators
│   │   └── spiral/     # Spiral dataset generator
│   ├── storage/        # Dataset persistence layer
│   └── api/            # FastAPI application
│       └── routes/     # API route handlers
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── pyproject.toml      # Project configuration
└── README.md           # This file
```

## Development

### Running Tests

```bash
pytest
```

### Running Tests with Coverage

```bash
pytest --cov=juniper_data --cov-report=html
```

### Code Formatting

```bash
black juniper_data tests
isort juniper_data tests
```

### Type Checking

```bash
mypy juniper_data
```

## License

MIT License - Copyright (c) 2024-2026 Paul Calnon

## Git Leaks

```bash
<img alt="gitleaks badge" src="https://img.shields.io/badge/protected%20by-gitleaks-blue">
```
