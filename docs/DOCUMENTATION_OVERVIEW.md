# Documentation Overview

## Complete Navigation Guide to Juniper Canopy Documentation

**Version:** 0.4.1  
**Last Updated:** November 11, 2025  
**Project:** Juniper Canopy - Real-Time CasCor Monitoring Frontend

---

## Table of Contents

- [Quick Navigation](#quick-navigation)
- [Getting Started](#getting-started)
- [Core Documentation](#core-documentation)
- [Technical Guides](#technical-guides)
- [Development Resources](#development-resources)
- [Historical Documentation](#historical-documentation)
- [Document Index](#document-index)
- [Documentation Standards](#documentation-standards)

---

## Quick Navigation

### I'm New Here - Where Do I Start?

```bash
1. README.md              → Project overview, what is this?
2. QUICK_START.md         → Get running in 5 minutes
3. ENVIRONMENT_SETUP.md   → Set up your environment
4. AGENTS.md              → Development conventions and guides
```

### I Want To

| Goal | Document | Location |
| ------ | ---------- | ---------- |
| **Get the app running** | [QUICK_START.md](QUICK_START.md) | Root |
| **Understand the project** | [README.md](README.md) | Root |
| **Set up my environment** | [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) | Root |
| **Run tests** | [TESTING_QUICK_START.md](TESTING_QUICK_START.md) | Root |
| **Set up test environment** | [TESTING_ENVIRONMENT_SETUP.md](TESTING_ENVIRONMENT_SETUP.md) | Root |
| **Learn testing** | [TESTING_MANUAL.md](TESTING_MANUAL.md) | Root |
| **View coverage reports** | [TESTING_REPORTS_COVERAGE.md](TESTING_REPORTS_COVERAGE.md) | Root |
| **Testing reference** | [TESTING_REFERENCE.md](TESTING_REFERENCE.md) | Root |
| **Get CI/CD running** | [CICD_QUICK_START.md](docs/ci_cd/CICD_QUICK_START.md) | docs/ci_cd/ |
| **Set up CI/CD environment** | [CICD_ENVIRONMENT_SETUP.md](docs/ci_cd/CICD_ENVIRONMENT_SETUP.md) | docs/ci_cd/ |
| **Learn CI/CD workflow** | [CICD_MANUAL.md](docs/ci_cd/CICD_MANUAL.md) | docs/ci_cd/ |
| **CI/CD reference** | [CICD_REFERENCE.md](docs/ci_cd/CICD_REFERENCE.md) | docs/ci_cd/ |
| **See version history** | [CHANGELOG.md](CHANGELOG.md) | Root |
| **Contribute code** | [AGENTS.md](AGENTS.md) | Root |
| **Find external links** | [references_and_links.md](docs/references_and_links.md) | docs/ |

---

## Getting Started

### Essential Documents (Read First)

#### 1. [README.md](README.md)

**Location:** Root directory  
**Purpose:** Project overview, features, quick start  
**Audience:** Everyone  
**Key Sections:**

- What is Juniper Canopy?
- Quick start (60 seconds)
- Key features
- Installation
- Usage
- Testing
- API reference

**When to Read:** First time visiting the project

---

#### 2. [QUICK_START.md](QUICK_START.md)

**Location:** Root directory  
**Purpose:** Get running in 5 minutes  
**Audience:** New users, developers  
**Key Sections:**

- Prerequisites checklist
- Step-by-step setup
- Demo mode launch
- Production mode setup
- First-time verification
- Common startup issues

**When to Read:** When you want to run the application immediately

---

#### 3. [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)

**Location:** Root directory  
**Purpose:** Complete environment configuration guide  
**Audience:** Developers setting up for the first time  
**Key Sections:**

- Conda environment setup
- Python dependencies
- Configuration files
- Environment variables
- Troubleshooting
- Verification steps

**When to Read:** Before starting development, when environment issues occur

---

#### 4. [AGENTS.md](AGENTS.md)

**Location:** Root directory (also duplicated in docs/)  
**Purpose:** AI agent development guide and conventions  
**Audience:** Developers, AI assistants  
**Key Sections:**

- Quick start commands
- Architecture overview
- Code style guidelines
- Thread safety patterns
- Testing requirements
- Common issues and solutions
- Definition of Done

**When to Read:** Before writing any code, when debugging issues

---

## Core Documentation

### Project Information

#### [CHANGELOG.md](CHANGELOG.md)

**Location:** Root directory  
**Purpose:** Version history and release notes  
**Format:** Keep a Changelog standard  
**Audience:** All users  
**Key Sections:**

- Unreleased changes
- Version history (0.4.0, 0.3.0, 0.2.1, 0.2.0, 0.1.4)
- Breaking changes
- Migration guides
- Testing procedures

**When to Read:**

- After updates/upgrades
- When investigating when a feature was added
- When troubleshooting regressions

**Update Frequency:** Every release, every significant change

---

### Architecture & Design

#### Project Structure

```bash
juniper_canopy/
├── README.md                      ← Start here
├── QUICK_START.md                 ← Get running fast
├── ENVIRONMENT_SETUP.md           ← Environment setup
├── DOCUMENTATION_OVERVIEW.md      ← You are here
├── AGENTS.md                      ← Development guide
├── CHANGELOG.md                   ← Version history
├── conf/                          ← Configuration
│   ├── app_config.yaml
│   ├── requirements.txt
│   └── conda_environment.yaml
├── docs/                          ← Technical documentation
│   ├── ci_cd/                     ← CI/CD documentation (4 files)
│   │   ├── CICD_QUICK_START.md
│   │   ├── CICD_ENVIRONMENT_SETUP.md
│   │   ├── CICD_MANUAL.md
│   │   └── CICD_REFERENCE.md
│   ├── testing/                   ← Testing documentation
│   ├── deployment/                ← Deployment guides
│   ├── references_and_links.md
│   └── history/                   ← Historical docs (75+ files)
├── src/                           ← Source code
│   ├── main.py                    ← Entry point
│   ├── demo_mode.py
│   ├── config_manager.py
│   ├── backend/
│   ├── communication/
│   ├── frontend/
│   ├── logger/
│   └── tests/                     ← Test suite
├── util/                          ← Utility scripts
│   └── run_demo.bash
├── demo                           ← Demo mode launcher
└── try                            ← Production launcher
```

---

## Technical Guides

### CI/CD & Quality

> **Note:** CI/CD documentation consolidated on 2025-11-11. All CI/CD guides now in [docs/ci_cd/](docs/ci_cd/).  
> Legacy files archived to [docs/history/](docs/history/).

#### [docs/ci_cd/CICD_QUICK_START.md](docs/ci_cd/CICD_QUICK_START.md)

**Lines:** ~400  
**Purpose:** Get CI/CD running in 5 minutes  
**Audience:** New developers  
**Key Sections:**

- Prerequisites
- Pre-commit installation
- Run tests locally
- GitHub secrets setup
- First commit
- View CI results

**When to Read:**

- First-time developer setup
- Need to get tests running quickly

---

#### [docs/ci_cd/CICD_ENVIRONMENT_SETUP.md](docs/ci_cd/CICD_ENVIRONMENT_SETUP.md)

**Lines:** ~870  
**Purpose:** Complete CI/CD environment configuration  
**Audience:** DevOps, maintainers  
**Key Sections:**

- GitHub Actions configuration
- Environment variables and secrets
- Python matrix setup
- Dependencies and caching
- Workflow triggers
- Artifact management

**When to Read:**

- Setting up GitHub Actions for first time
- Modifying CI/CD pipeline
- Troubleshooting environment issues

---

#### [docs/ci_cd/CICD_MANUAL.md](docs/ci_cd/CICD_MANUAL.md)

**Lines:** ~1,688  
**Purpose:** Comprehensive CI/CD usage guide  
**Audience:** Developers, reviewers  
**Key Sections:**

- Daily developer workflow
- Writing and running tests
- Coverage workflow
- Pre-commit hooks usage
- Debugging CI failures
- Emergency procedures

**When to Read:**

- Learning CI/CD workflow
- Debugging test failures
- Reviewing pull requests
- Managing pipeline issues

---

#### [docs/ci_cd/CICD_REFERENCE.md](docs/ci_cd/CICD_REFERENCE.md)

**Lines:** ~1,058  
**Purpose:** Technical CI/CD reference  
**Audience:** All developers  
**Key Sections:**

- Pipeline architecture
- Workflow specifications
- Tool configurations
- Environment variables
- Command reference
- Troubleshooting reference

**When to Read:**

- Need quick command reference
- Understanding pipeline details
- Configuring tools

---

### Configuration & Setup

#### [conf/app_config.yaml](conf/app_config.yaml)

**Purpose:** Application configuration  
**Format:** YAML  
**Key Sections:**

- Server settings (port, host)
- Demo mode configuration
- Backend integration paths
- Logging levels
- WebSocket settings

**Environment Variable Overrides:**

- Format: `CASCOR_<SECTION>_<KEY>`
- Example: `CASCOR_SERVER_PORT=8051`
- Supports `${VAR}` and `$VAR` expansion

---

#### [conf/requirements.txt](conf/requirements.txt)

**Purpose:** Python dependencies  
**Format:** pip requirements file  
**Key Dependencies:**

- fastapi
- uvicorn
- dash
- plotly
- pytest
- pytest-cov
- websockets

---

#### [conf/conda_environment.yaml](conf/conda_environment.yaml)

**Purpose:** Conda environment specification  
**Environment Name:** JuniperPython  
**Python Version:** 3.11+

---

### Reference Documentation

#### [docs/references_and_links.md](docs/references_and_links.md)

**Lines:** ~176  
**Purpose:** External resources and references  
**Audience:** All users  
**Key Sections:**

- FastAPI documentation
- Dash/Plotly guides
- WebSocket specifications
- Testing frameworks
- CI/CD tools
- Python best practices

---

## Development Resources

### Code Style & Conventions

**Source:** [AGENTS.md](AGENTS.md) - Code Style Guidelines section

**Key Conventions:**

- **File Headers:** Standard project header with author, version, date
- **Naming:** PascalCase (classes), snake_case (functions), UPPER_SNAKE_CASE (constants)
- **Metric Naming:** snake_case with `train_`/`val_` prefixes
- **Thread Safety:** Locks for shared state, Events for signaling
- **Path Resolution:** Use pathlib, no hardcoded absolute paths
- **Error Handling:** Appropriate logging levels (debug, info, warning, error)

---

### Testing Standards

**Source:** [AGENTS.md](AGENTS.md) - Testing Guidelines section

**Requirements:**

- **No PR without tests** for new/changed behavior
- **Unit tests:** >80% coverage
- **Integration tests:** Core workflows covered
- **Critical paths:** 100% coverage
- **Regression tests:** For all fixed bugs

**Test Organization:**

```bash
src/tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
├── integration/             # Integration tests
└── performance/             # Performance tests
```

---

### API & WebSocket Contracts

**Source:** [AGENTS.md](AGENTS.md) - API and WebSocket Contracts section

**REST Endpoints:**

- `GET /api/metrics` - Current training metrics
- `GET /api/metrics/history` - Historical metrics
- `GET /api/network/topology` - Network structure
- `GET /api/decision_boundary` - Decision boundary data
- `GET /api/dataset` - Current dataset points
- `GET /health` - Health check

**WebSocket Channels:**

- `/ws/training` - Stream metrics and state updates
- `/ws/control` - Send commands (start, stop, pause, resume, reset)

**Message Format:**

```json
{
  "type": "metrics | state | topology | event",
  "timestamp": 1234567890.123,
  "data": { ... }
}
```

---

### Definition of Done

**Source:** [AGENTS.md](AGENTS.md) - Definition of Done section

**All new/modified code must meet:**

**Code Quality:**

- [ ] Thread safety preserved
- [ ] Bounded collections for streaming/history buffers
- [ ] Metric naming follows standard
- [ ] Proper path resolution
- [ ] Error handling with appropriate logging

**Testing:**

- [ ] Unit tests added for new functionality
- [ ] Integration tests for component interactions
- [ ] Regression tests for fixed bugs
- [ ] Coverage maintained/increased
- [ ] All tests passing

**API/Interface:**

- [ ] API/WebSocket changes backward compatible or versioned
- [ ] Payload schemas documented
- [ ] No breaking changes without migration plan

**Documentation:**

- [ ] CHANGELOG.md updated
- [ ] README.md reflects current instructions
- [ ] Roadmap status updated
- [ ] Code comments only where complexity requires

---

## Historical Documentation

### Archive Location

**Path:** [docs/history/](docs/history/)  
**Files:** 67 documents  
**Total Lines:** ~33,000  
**Purpose:** Historical record of development, fixes, analyses

### Categories

#### MVP/Implementation Reports (15 files)

- FINAL_STATUS_2025-11-03.md
- MVP_COMPLETE_SUMMARY.md
- MVP_ACHIEVEMENT_REPORT.md
- PHASE_1_MVP_COMPLETE.md
- Implementation summaries

**Purpose:** Track MVP completion milestones

---

#### Testing Reports (10 files)

- TEST_FIXES_2025-11-03.md
- FRONTEND_TESTING_2025-11-03.md
- TESTING_VERIFICATION_REPORT.md
- Coverage improvement reports

**Purpose:** Document testing evolution and fixes

---

#### Bug Fix Reports (12 files)

- REGRESSION_FIX_REPORT.md
- MISSING_DATA_FIX_2025-10-29.md
- DASH_FASTAPI_INTEGRATION_FIX.md
- Frontend fix implementations

**Purpose:** Record bug investigations and solutions

---

#### Analysis/Design Documents (12 files)

- architecture_design.md
- juniper_canopy_design.md
- technical_specifications.md
- logging_framework_design.md

**Purpose:** Design decisions and architectural analysis

---

#### Integration/Planning (8 files)

- DEVELOPMENT_ROADMAP.md
- cascor_backend_integration_plan.md
- INTEGRATION_PATTERNS.md

**Purpose:** Planning and integration strategies

---

### When to Consult Historical Docs

**Scenarios:**

1. **Investigating a regression** - Check bug fix reports
2. **Understanding design decisions** - Review design documents
3. **Seeing project evolution** - Read MVP completion reports
4. **Troubleshooting similar issues** - Search fix reports
5. **Learning implementation patterns** - Study integration documents

---

## Document Index

### Root Directory

| File | Lines | Type | Audience | Status |
| ------ | ------- | ------ | ---------- | -------- |
| README.md | ~550 | Overview | All | ✅ Active |
| QUICK_START.md | ~250 | Tutorial | New users | ✅ Active |
| ENVIRONMENT_SETUP.md | ~400 | Guide | Developers | ✅ Active |
| DOCUMENTATION_OVERVIEW.md | ~700 | Navigation | All | ✅ Active (this file) |
| AGENTS.md | 829 | Reference | Developers, AI | ✅ Active |
| CHANGELOG.md | 392 | History | All | ✅ Active |

### docs/ Directory

| File | Lines | Type | Audience | Status |
| ------ | ------- | ------ | ---------- | -------- |
| **ci_cd/CICD_QUICK_START.md** | ~400 | Tutorial | Developers | ✅ **Active** |
| **ci_cd/CICD_ENVIRONMENT_SETUP.md** | ~870 | Guide | DevOps | ✅ **Active** |
| **ci_cd/CICD_MANUAL.md** | ~1,688 | Guide | Developers | ✅ **Active** |
| **ci_cd/CICD_REFERENCE.md** | ~1,058 | Reference | All | ✅ **Active** |
| references_and_links.md | 176 | Reference | All | ✅ Active |
| AGENTS.md | 829 | Reference | Developers, AI | ✅ Active (duplicate) |
| README.md | 142 | Overview | All | 🟡 Superseded by root README |
| CHANGELOG.md | 392 | History | All | 🟡 Superseded by root CHANGELOG |
| **testing/TESTING_QUICK_START.md** | ~180 | Tutorial | Developers | ✅ **Active** |
| **testing/TESTING_ENVIRONMENT_SETUP.md** | ~550 | Guide | Developers | ✅ **Active** |
| **testing/TESTING_MANUAL.md** | ~900 | Guide | Developers | ✅ **Active** |
| **testing/TESTING_REFERENCE.md** | ~1,200 | Reference | Developers | ✅ **Active** |
| **testing/TESTING_REPORTS_COVERAGE.md** | ~900 | Guide | Developers | ✅ **Active** |

> **Note:** CI/CD documentation consolidated 2025-11-11 (12→4 files). See [docs/ci_cd/CONSOLIDATION_SUMMARY.md](docs/ci_cd/CONSOLIDATION_SUMMARY.md).

### docs/history/ Directory

**75+ files, ~36,000+ lines** - See [Historical Documentation](#historical-documentation) section

> **Recent Additions (2025-11-11):** 8 CI/CD files archived during consolidation

---

## Documentation Standards

### File Naming Conventions

**Active Documentation:**

- Use clear, descriptive names: `QUICK_START.md`, `ENVIRONMENT_SETUP.md`
- All caps for major guides: `README.md`, `CHANGELOG.md`, `AGENTS.md`
- Lowercase with underscores for references: `references_and_links.md`

**Historical Documentation:**

- Include dates for time-sensitive docs: `FINAL_STATUS_2025-11-03.md`
- Use descriptive names indicating purpose: `REGRESSION_FIX_REPORT.md`

---

### Markdown Formatting

**Required Elements:**

- Title (# heading)
- Table of contents (for docs >200 lines)
- Clear section headings (##, ###)
- Code blocks with language specification
- Links to related documents
- Last updated date
- Author/version information

**Example:**

```markdown
# Document Title

**Version:** 0.4.0  
**Last Updated:** November 7, 2025  
**Author:** Paul Calnon

## Table of Contents

- [Section 1](#section-1)
- [Section 2](#section-2)

## Section 1

Content...

## Section 2

Content...
```

---

### Cross-Referencing

**Internal Links:**

- Use relative paths: `[AGENTS.md](AGENTS.md)`, `[CI_CD.md](docs/CI_CD.md)`
- Include section anchors: `[Testing](#testing)`, `[Quick Start](README.md#quick-start)`

**External Links:**

- Use descriptive text: `[FastAPI Documentation](https://fastapi.tiangolo.com/)`
- Collect in references_and_links.md for reuse

---

### Update Requirements

**On Every Change:**

1. **CHANGELOG.md** - Summarize changes and impact
2. **README.md** - Update if run/test instructions change
3. **DEVELOPMENT_ROADMAP.md** - Update status (if in docs/history/)
4. **Relevant technical docs** - Update affected guides

**Version Bumps:**

- Update version numbers in README.md, CHANGELOG.md
- Add release notes to CHANGELOG.md
- Update "Last Updated" dates

---

## Documentation Gaps & Future Work

### Missing Documentation (To Be Created)

1. **ARCHITECTURE.md** - Complete system architecture with diagrams
2. **API_REFERENCE.md** - Complete API endpoint specifications
3. **TROUBLESHOOTING.md** - Common issues and solutions extracted from AGENTS.md
4. **DEVELOPMENT_ROADMAP.md** - Current and future features (currently in history/)
5. **CONTRIBUTING.md** - Contribution guidelines
6. **SECURITY.md** - Security policies and reporting

---

### Consolidation Opportunities

**High Priority:**

1. Merge CI/CD quick references into CI_CD.md
2. Merge pre-commit quick reference into PRE_COMMIT_GUIDE.md
3. Consolidate MVP reports into single IMPLEMENTATION_HISTORY.md
4. Extract troubleshooting from AGENTS.md to TROUBLESHOOTING.md

**See:** [docs/DOCUMENTATION_ANALYSIS_2025-11-05.md](docs/DOCUMENTATION_ANALYSIS_2025-11-05.md) for detailed analysis

---

## Finding Information

### Search Strategies

**By Topic:**

1. Check this overview's "I Want To..." table
2. Search AGENTS.md for development topics
3. Search docs/ for technical guides
4. Search docs/history/ for historical context

**By Keyword:**

```bash
# Search all markdown files
grep -r "keyword" *.md docs/*.md

# Search with context
grep -r -C 3 "keyword" *.md docs/*.md

# Search specific directory
grep -r "keyword" docs/history/
```

**By Recent Changes:**

1. Check CHANGELOG.md for version history
2. Review git log for recent commits
3. Check "Recent Changes" section in AGENTS.md

---

## Quick Reference Card

### Essential Commands

```bash
# Get running
./demo

# Run tests
cd src && pytest tests/ -v

# Run with coverage
cd src && pytest tests/ --cov=. --cov-report=html

# Pre-commit checks
pre-commit run --all-files

# Format code
black src/ && isort src/

# Check syntax
python -m py_compile src/**/*.py
```

### Essential Files

```bash
# Start here
README.md              # What is this?
QUICK_START.md         # Get running now
ENVIRONMENT_SETUP.md   # Set up environment

# Development
AGENTS.md              # Development guide
docs/CI_CD.md          # Testing & CI/CD
docs/PRE_COMMIT_GUIDE.md  # Code quality

# Reference
CHANGELOG.md           # Version history
docs/references_and_links.md  # External links
```

---

## Contact & Support

- **Author:** Paul Calnon
- **Project:** Juniper
- **Prototype:** juniper_canopy (Juniper Canopy)

**For Documentation Issues:**

1. Check this overview first
2. Search existing docs
3. Consult AGENTS.md for conventions
4. Check CHANGELOG.md for recent changes

---

**Last Updated:** November 11, 2025  
**Version:** 0.4.1  
**Maintainer:** Paul Calnon

---

## Recent Updates

### 2025-11-11: CI/CD Documentation Consolidation

- **Consolidated:** 12 CI/CD files → 4 focused documents
- **New location:** docs/ci_cd/ (single directory)
- **New structure:**
  - CICD_QUICK_START.md - Get running in 5 minutes
  - CICD_ENVIRONMENT_SETUP.md - Complete environment setup
  - CICD_MANUAL.md - Comprehensive usage guide
  - CICD_REFERENCE.md - Technical reference
- **Archived:** 8 legacy files to docs/history/ (2025-11-11)
- **See:** [docs/ci_cd/CONSOLIDATION_SUMMARY.md](docs/ci_cd/CONSOLIDATION_SUMMARY.md)
