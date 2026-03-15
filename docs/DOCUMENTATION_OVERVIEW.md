# Documentation Overview

## Complete Navigation Guide to Juniper Data Documentation

**Version:** 0.4.2
**Last Updated:** March 3, 2026
**Project:** Juniper Data - Dataset Generation Service

---

## Table of Contents

- [Quick Navigation](#quick-navigation)
- [Getting Started](#getting-started)
- [Core Documentation](#core-documentation)
- [Technical Guides](#technical-guides)
- [Development Resources](#development-resources)
- [Document Index](#document-index)
- [Documentation Standards](#documentation-standards)

---

## Quick Navigation

### I'm New Here - Where Do I Start?

```bash
1. README.md              -> Project overview, what is this?
2. QUICK_START.md         -> Get running in 5 minutes
3. ENVIRONMENT_SETUP.md   -> Set up your environment
4. AGENTS.md              -> Development conventions and guides
```

### I Want To

| Goal | Document | Location |
|------|----------|----------|
| **Get the service running** | [QUICK_START.md](QUICK_START.md) | docs/ |
| **Understand the project** | [README.md](../README.md) | Root |
| **Set up my environment** | [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) | docs/ |
| **Use the service** | [USER_MANUAL.md](USER_MANUAL.md) | docs/ |
| **Look up API endpoints** | [JUNIPER_DATA_API.md](api/JUNIPER_DATA_API.md) | docs/api/ |
| **Find technical reference** | [REFERENCE.md](REFERENCE.md) | docs/ |
| **Run tests** | [TESTING_QUICK_START.md](testing/TESTING_QUICK_START.md) | docs/testing/ |
| **Learn testing** | [TESTING_MANUAL.md](testing/TESTING_MANUAL.md) | docs/testing/ |
| **Testing reference** | [TESTING_REFERENCE.md](testing/TESTING_REFERENCE.md) | docs/testing/ |
| **Get CI/CD running** | [CICD_QUICK_START.md](ci_cd/CICD_QUICK_START.md) | docs/ci_cd/ |
| **Learn CI/CD workflow** | [CICD_MANUAL.md](ci_cd/CICD_MANUAL.md) | docs/ci_cd/ |
| **CI/CD reference** | [CICD_REFERENCE.md](ci_cd/CICD_REFERENCE.md) | docs/ci_cd/ |
| **See version history** | [CHANGELOG.md](../CHANGELOG.md) | Root |
| **Quick-reference dev tasks** | [DEVELOPER_CHEATSHEET.md](DEVELOPER_CHEATSHEET.md) | docs/ |
| **Contribute code** | [AGENTS.md](../AGENTS.md) | Root |

---

## Getting Started

### Essential Documents (Read First)

#### 1. [README.md](../README.md)

**What:** Project overview, ecosystem context, architecture diagram, quick start
**Audience:** Everyone -- first-time users and returning developers
**Key contents:** Ecosystem compatibility matrix, service architecture, API endpoint table, installation options

#### 2. [QUICK_START.md](QUICK_START.md)

**What:** Get Juniper Data running in 5 minutes
**Audience:** New users who want to start immediately
**Key contents:** Clone, install, start service, generate dataset, retrieve data

#### 3. [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)

**What:** Complete environment configuration from scratch
**Audience:** New developers and CI/CD setup
**Key contents:** Conda environment, dependency groups, environment variables, verification

#### 4. [AGENTS.md](../AGENTS.md)

**What:** Development conventions, testing, code style
**Audience:** Contributors and developers
**Key contents:** Essential commands, architecture, code conventions, test markers, worktree procedures

---

## Core Documentation

### Application Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [USER_MANUAL.md](USER_MANUAL.md) | Comprehensive usage guide -- generators, API, storage, data contract | Users, integrators |
| [REFERENCE.md](REFERENCE.md) | Technical reference -- configuration, commands, tool settings | Developers |
| [JUNIPER_DATA_API.md](api/JUNIPER_DATA_API.md) | Full REST API documentation with schemas and examples | API consumers |

### Architecture

Juniper Data is the **foundational data layer** of the Juniper ecosystem:

```
juniper-canopy (8050) ──REST──> JuniperData (8100) <──REST── juniper-cascor (8200)
                                     │
                              Generates datasets
                              Stores as NPZ artifacts
                              Serves via REST API
```

**Key components:**

- **8 generators** -- spiral, xor, gaussian, circles, checkerboard, csv_import, mnist, arc_agi
- **8 storage backends** -- local filesystem, PostgreSQL, Redis, HuggingFace, Kaggle, in-memory, cached
- **FastAPI REST service** -- Full CRUD operations on port 8100
- **NPZ data contract** -- `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`)

### Client Library

**juniper-data-client** provides a Python HTTP client for consuming the Juniper Data REST API programmatically:

- **PyPI:** `pip install juniper-data-client`
- **Repository:** [pcalnon/juniper-data-client](https://github.com/pcalnon/juniper-data-client)
- **Consumers:** juniper-cascor (SpiralDataProvider), juniper-canopy (DemoMode, CascorIntegration)

See the [juniper-data-client documentation](https://github.com/pcalnon/juniper-data-client) for usage and API reference.

---

## Technical Guides

### Testing Documentation

| Document | Purpose | Lines |
|----------|---------|-------|
| [TESTING_QUICK_START.md](testing/TESTING_QUICK_START.md) | Get tests running in 5 minutes | ~120 |
| [TESTING_MANUAL.md](testing/TESTING_MANUAL.md) | Comprehensive testing guide | ~400 |
| [TESTING_REFERENCE.md](testing/TESTING_REFERENCE.md) | Test markers, fixtures, configuration | ~280 |

### CI/CD Documentation

| Document | Purpose | Lines |
|----------|---------|-------|
| [CICD_QUICK_START.md](ci_cd/CICD_QUICK_START.md) | Run CI locally in 5 minutes | ~120 |
| [CICD_MANUAL.md](ci_cd/CICD_MANUAL.md) | Full CI/CD pipeline guide | ~400 |
| [CICD_REFERENCE.md](ci_cd/CICD_REFERENCE.md) | Jobs, hooks, environment variables reference | ~250 |

### API Documentation

| Document | Purpose | Lines |
|----------|---------|-------|
| [JUNIPER_DATA_API.md](api/JUNIPER_DATA_API.md) | Complete REST API reference | ~620 |

---

## Development Resources

### Key Files for Developers

| File | Location | Purpose |
|------|----------|---------|
| `pyproject.toml` | Root | Project config, dependencies, tool settings |
| `AGENTS.md` | Root | Development guide and conventions |
| `CHANGELOG.md` | Root | Version history |
| `.pre-commit-config.yaml` | Root | Pre-commit hook configuration |
| `.github/workflows/ci.yml` | Root | GitHub Actions CI pipeline |
| `.github/workflows/publish.yml` | Root | PyPI publishing workflow |

### Source Code Map

```
juniper_data/
├── __init__.py           # Package init with version
├── __main__.py           # CLI entry point
├── core/                 # Base classes, models, ID generation, splitting
├── generators/           # 8 dataset generators
│   ├── spiral/           # Multi-spiral classification (primary)
│   ├── xor/              # XOR classification
│   ├── gaussian/         # Mixture of Gaussians
│   ├── circles/          # Concentric circles
│   ├── checkerboard/     # 2D checkerboard pattern
│   ├── csv_import/       # CSV/JSON file import
│   ├── mnist/            # MNIST / Fashion-MNIST
│   └── arc_agi/          # ARC-AGI visual reasoning
├── storage/              # 8 storage backend implementations
│   ├── base.py           # Abstract DatasetStore interface
│   ├── local_fs.py       # Local filesystem (primary)
│   ├── memory.py         # In-memory (testing)
│   ├── cached.py         # Cached wrapper
│   ├── postgres_store.py # PostgreSQL
│   ├── redis_store.py    # Redis
│   ├── hf_store.py       # HuggingFace Hub
│   └── kaggle_store.py   # Kaggle Datasets
├── api/                  # FastAPI REST application
│   ├── app.py            # Factory-pattern app creation
│   ├── settings.py       # Pydantic BaseSettings
│   ├── middleware.py      # SecurityMiddleware
│   ├── security.py       # API key auth, rate limiting
│   ├── observability.py  # Prometheus, Sentry, logging
│   └── routes/           # health, generators, datasets
└── tests/                # Test suite (~9,000 lines)
    ├── unit/             # 29 test files
    ├── integration/      # 5 test files
    ├── performance/      # 2 benchmark files
    └── fixtures/         # Golden dataset generation
```

---

## Document Index

| Document | Location | Type | Audience | Status |
|----------|----------|------|----------|--------|
| [README.md](../README.md) | Root | Overview | Everyone | Active |
| [AGENTS.md](../AGENTS.md) | Root | Dev Guide | Developers | Active |
| [CHANGELOG.md](../CHANGELOG.md) | Root | History | Everyone | Active |
| [QUICK_START.md](QUICK_START.md) | docs/ | Quick Start | New Users | Active |
| [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) | docs/ | Setup | Developers | Active |
| [USER_MANUAL.md](USER_MANUAL.md) | docs/ | Manual | Users | Active |
| [REFERENCE.md](REFERENCE.md) | docs/ | Reference | Developers | Active |
| [JUNIPER_DATA_API.md](api/JUNIPER_DATA_API.md) | docs/api/ | API Ref | API Consumers | Active |
| [TESTING_QUICK_START.md](testing/TESTING_QUICK_START.md) | docs/testing/ | Quick Start | Developers | Active |
| [TESTING_MANUAL.md](testing/TESTING_MANUAL.md) | docs/testing/ | Manual | Developers | Active |
| [TESTING_REFERENCE.md](testing/TESTING_REFERENCE.md) | docs/testing/ | Reference | Developers | Active |
| [CICD_QUICK_START.md](ci_cd/CICD_QUICK_START.md) | docs/ci_cd/ | Quick Start | DevOps | Active |
| [CICD_MANUAL.md](ci_cd/CICD_MANUAL.md) | docs/ci_cd/ | Manual | DevOps | Active |
| [CICD_REFERENCE.md](ci_cd/CICD_REFERENCE.md) | docs/ci_cd/ | Reference | DevOps | Active |

### notes/ Directory

| Document | Location | Type | Audience | Status |
|----------|----------|------|----------|--------|
| [DEVELOPER_CHEATSHEET.md](DEVELOPER_CHEATSHEET.md) | docs/ | Cheatsheet | Developers | Active |

---

## Documentation Standards

### Naming Conventions

- **UPPER_CASE** filenames with `.md` extension
- **Subdirectories**: lowercase with underscores (`testing/`, `ci_cd/`, `api/`)
- **Prefixed**: Subdirectory docs use prefix (`TESTING_*.md`, `CICD_*.md`)

### Document Metadata

Every doc includes:

```markdown
**Version:** X.Y.Z
**Last Updated:** Month DD, YYYY
**Project:** Juniper Data - Dataset Generation Service
```

### Cross-References

- Use relative paths for internal links
- Use descriptive link text
- Link to specific sections with anchors where helpful

### Full Standard

See [DOCUMENTATION_TEMPLATE_STANDARD.md](../../notes/DOCUMENTATION_TEMPLATE_STANDARD.md) for the complete cross-ecosystem documentation standard.

---

**Last Updated:** March 3, 2026
**Version:** 0.4.2
**Maintainer:** Paul Calnon
