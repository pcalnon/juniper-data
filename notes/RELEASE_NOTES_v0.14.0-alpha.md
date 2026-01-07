# JuniperCanopy v0.14.0-alpha

Real-time monitoring and diagnostics for Cascade Correlation Neural Networks (CasCor).

This **alpha** release delivers a major refactor of the Bash-based infrastructure that underpins JuniperCanopy's tooling and configuration. The primary goal is to make the shell/CLI layer more modular, testable, and maintainable, while preserving core Python behavior.

> **Status:** ALPHA – not yet production-hardened. Expect rough edges in the shell tooling and configuration layer.

---

## Highlights

- **Major Bash infrastructure refactor** (36 commits) with a more modular configuration system.
- **25+ new `.conf` files** for clearer separation of concerns and easier script reuse.
- **Improved logging, date utilities, and path resolution**, reducing subtle shell bugs.
- **Leaner `conda_environment.yaml`** (−75 lines) for a more focused environment definition.

---

## What's New

### Added

- **Comprehensive Bash Script Configuration Infrastructure**
  - 25+ new modular `.conf` files for:
    - Common environment and path handling
    - Logging and diagnostic behavior
    - Date/time utilities and related helpers
  - Designed to replace large monolithic script utility files with smaller, composable configs.

- **New utility scripts**
  - `color_display_codes.bash` – Displays terminal color codes to help debug and standardize color output in shell-based tools.
  - `color_table.py` – Python helper for generating and visualizing color tables, useful for tuning terminal/UI color schemes.
  - `mv2_bash_n_back.bash` – Workflow utility for moving files and then restoring them, easing iterative script development and local experimentation.

### Changed

- **Major Bash Infrastructure Refactoring (36 commits)**
  - Restructured how scripts load configuration and utilities.
  - Consolidated shared logic into modular `.conf` components.
  - Reduced duplication between different script entry points.

- **Introduced `CALLING_PID` for accurate parent path resolution**
  - More reliable detection of caller context and script location.
  - Reduces brittle behavior when scripts are invoked from different working directories or wrapper scripts.

- **Enhanced date functions and logging mechanisms**
  - Date/time utilities moved into dedicated config modules for reuse.
  - Logging behavior centralized with clearer configuration, improving:
    - Log formatting consistency
    - Log level handling
    - Future extensibility for more advanced logging targets

- **`conf/common.conf` expanded (+488 lines)**
  - Now the central hub for shared configuration:
    - Common environment variables
    - Path and directory resolution logic
    - Shared utility function references
  - Heavily commented to ease debugging and extension.

- **`conda_environment.yaml` streamlined (−75 lines)**
  - Removed unused or redundant dependencies.
  - Tightened environment definition to reduce environment size and setup time.
  - Makes it easier to track necessary vs. optional dependencies.

### Fixed

- **Corrected inverted logic in `is_defined` checks**
  - Shell helpers now correctly detect defined variables and options.
  - Prevents subtle misconfigurations where environment checks were silently inverted.

- **Improved path resolution for `init.conf` sourcing**
  - More robust discovery and loading of initialization configuration.
  - Reduces failures when running scripts from non-standard working directories or via symlinks.

- **Fixed method and TODO counting with proper whitespace handling**
  - More accurate counts in tooling that scans code for methods and `TODO` markers.
  - Handles whitespace and formatting edge cases more reliably.

### Removed

- **Configuration files consolidated or replaced**
  - `conf/script_util.conf` (329 lines) – Functionality distributed across new modular `.conf` files.
  - `conf/util_logging.conf` (266 lines) – Replaced by `logging.conf` and `logging_functions.conf` for clearer separation of configuration and logic.

- **Legacy utility scripts superseded**
  - `util/__date_functions.bash` (155 lines) – Functionality moved into `conf/__date_functions.conf`.
  - `util/run_demo.bash`, `util/try.bash` – Removed as obsolete; their responsibilities have been superseded by the new infrastructure and/or are no longer aligned with current workflows.

---

## Breaking / Behavioral Changes

Because this release restructures the shell infrastructure, **any custom scripts or tooling that directly source JuniperCanopy's old Bash utilities or configs may break**.

### Potentially Affected Areas

- Direct use of:
  - `conf/script_util.conf`
  - `conf/util_logging.conf`
  - `util/__date_functions.bash`
  - `util/run_demo.bash`
  - `util/try.bash`
- Custom wrappers that:
  - Assume specific paths or file layouts for `.conf` files.
  - Depend on old logging or date-function behaviors or names.
  - Rely on previous (now-corrected) `is_defined` logic.

### Action Required

If you have custom integrations:

1. **Review the new config layout** in `conf/` (especially `common.conf`, `logging.conf`, `logging_functions.conf`, and `__date_functions.conf`).
2. **Update any `source`/`.` statements** to reference the new `.conf` files and functions.
3. **Re-test script workflows** that:
   - Depend on environment detection / path resolution.
   - Parse logs or rely on specific log formats.
   - Use date/time utilities or `is_defined` behavior.

Core Python functionality is not expected to change, but shell-based tooling and workflows very likely need minor adjustments.

---

## Installation & Upgrade Notes

### Fresh Installation

1. Clone or download the repository at tag **`v0.14.0-alpha`**.
2. Create or update your environment using the updated `conda_environment.yaml`.
3. Follow the setup instructions in the project's README for environment activation and initial configuration.

### Upgrading from a Previous Version

1. **Backup any local modifications** – Especially within `conf/` and `util/` if you customized scripts or configs.
2. **Update to `v0.14.0-alpha`** – Pull the tag or checkout the release branch.
3. **Recreate or update your Conda environment** – Use the new `conda_environment.yaml`. Remove deprecated dependencies if you previously added them manually.
4. **Review and adapt shell integrations** – Replace references to removed `.conf` and `util/*.bash` files with their new counterparts. Validate your workflow using the new logging and date utilities.
5. **Smoke test your typical workflows** – Run your standard CasCor experiments and monitoring setups. Confirm that visualization and monitoring work as expected.

---

## Known Issues

- **Test suite collection errors**
  - The current test run shows **58 collection errors**.
  - These errors are primarily related to the **new infrastructure and test discovery** rather than to core Python functionality.
  - At this stage:
    - The Python core (CasCor monitoring/visualization logic) is expected to behave as in prior versions.
    - Shell-level and infrastructure coverage is still being stabilized.

If you encounter issues that appear related to the new Bash infrastructure or configuration layout, please:

- Include details about your environment (OS, shell, Conda/Python versions).
- Provide any custom scripts or overrides you are using.
- Open an issue referencing **`v0.14.0-alpha`** and, if possible, attach logs demonstrating the failure.

---

Thank you for trying out **JuniperCanopy v0.14.0-alpha**. Feedback on the new infrastructure (especially from users with custom shell workflows or large experiments) is particularly valuable at this stage.
