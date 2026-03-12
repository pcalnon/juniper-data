# Testing Quick Start

## Get juniper-data Tests Running in 5 Minutes

**Version:** 0.4.2
**Status:** Active
**Last Updated:** March 3, 2026
**Project:** Juniper - Dataset Generation Service

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Run All Tests](#run-all-tests)
- [Run by Category](#run-by-category)
- [Run by Component](#run-by-component)
- [Coverage Report](#coverage-report)
- [Common Issues](#common-issues)
- [Next Steps](#next-steps)

---

## Prerequisites

- [ ] Python >= 3.12 installed
- [ ] Conda environment `JuniperData` activated (`conda activate JuniperData`)
- [ ] Repository cloned and at the project root

---

## Installation

```bash
# Install all dependencies including test extras
pip install -e ".[all]"
```

This installs pytest, pytest-cov, pytest-timeout, pytest-asyncio, pytest-benchmark, httpx, coverage, and juniper-data-client.

---

## Run All Tests

```bash
# Run all tests (default: quiet mode, short tracebacks)
pytest

# Run with verbose output
pytest -v

# Run with full tracebacks
pytest --tb=long
```

---

## Run by Category

```bash
# Unit tests only (fast, isolated)
pytest juniper_data/tests/unit/

# Integration tests only (requires service dependencies)
pytest juniper_data/tests/integration/

# Performance benchmarks (disabled by default)
pytest juniper_data/tests/performance/ --benchmark-enable
```

---

## Run by Component

```bash
# Spiral generator tests
pytest -m spiral -v

# All generator tests
pytest -m generators -v

# Storage tests
pytest -m storage -v

# API endpoint tests
pytest -m api -v

# Exclude slow tests (for quick feedback)
pytest -m "not slow"
```

---

## Coverage Report

```bash
# Run tests with terminal coverage report
pytest --cov=juniper_data --cov-report=term-missing --cov-fail-under=80

# Generate HTML coverage report (opens in browser)
pytest --cov=juniper_data --cov-report=html
# open htmlcov/index.html

# Per-module coverage check (CI-level enforcement)
python scripts/check_module_coverage.py --run-tests
# Enforces: 80% aggregate, 85% per-module
```

---

## Common Issues

**`ModuleNotFoundError: juniper_data`**: Install in editable mode with `pip install -e ".[all]"`.

**Tests timing out**: Default timeout is 60 seconds. Override with `pytest --timeout=120`.

**Benchmark warnings**: Benchmarks are disabled by default. Use `--benchmark-enable` to run them, or ignore the disabled message.

**Missing juniper-data-client**: Install it separately if needed: `pip install juniper-data-client>=0.3.0`.

---

## Next Steps

- [Testing Manual](TESTING_MANUAL.md) -- comprehensive testing guide with test architecture and writing conventions
- [Testing Reference](TESTING_REFERENCE.md) -- complete marker, fixture, and configuration reference
- [Reference](../REFERENCE.md) -- project-wide configuration and tooling reference

---

**Last Updated:** March 3, 2026
**Version:** 0.4.2
**Status:** Active
