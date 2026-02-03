# Juniper Data - Project Overview

## Purpose
Juniper Data is a dataset generation and management service for the Juniper neural network ecosystem. It provides a centralized service for generating, storing, and serving datasets used by Juniper neural network projects.

## Primary Features
- Dataset generation (spiral classification problem, extensible for other types)
- Dataset storage and persistence
- REST API for dataset access and generation
- Integration with Juniper neural network training projects

## Tech Stack

### Core
- **Python**: 3.11+ (targeting 3.14)
- **NumPy**: Numerical computations for dataset generation
- **Pydantic**: Data validation, settings, and schema management (v2.0+)

### API (Optional)
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server

### Development Tools
- **pytest**: Testing framework with pytest-cov, pytest-asyncio, pytest-timeout
- **black**: Code formatter (line length 512)
- **isort**: Import sorting (black profile)
- **mypy**: Static type checking
- **flake8**: Linting with bugbear, comprehensions, simplify plugins
- **bandit**: Security scanning (SAST)
- **pip-audit**: Dependency vulnerability scanning
- **pre-commit**: Git hooks for code quality

## Project Context
- Part of the larger Juniper ecosystem
- Follows patterns established in JuniperCascor
- MIT Licensed
- Author: Paul Calnon

## Related Projects
- JuniperCascor: Cascade correlation neural network implementation
- JuniperCanopy: Related project in the ecosystem
