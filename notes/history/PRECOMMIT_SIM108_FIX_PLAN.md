# Pre-commit SIM108 Fix Plan

**Date**: 2026-03-11
**Issue**: `pre-commit run --all` fails with 4 Ruff SIM108 violations
**Rule**: [SIM108](https://docs.astral.sh/ruff/rules/if-else-block-instead-of-if-exp/) — Use ternary operator instead of if-else-block

---

## Root Cause Analysis

The Ruff linter configuration in `pyproject.toml` enables the `SIM` rule group (flake8-simplify). While
`SIM102`, `SIM105`, and `SIM117` are explicitly ignored as style preferences, `SIM108` (ternary operator
suggestion) is **not** ignored. Four if-else blocks in two generator files assign a single variable
in both branches — the exact pattern SIM108 targets.

### Affected Files

| # | File | Line | Description |
|---|------|------|-------------|
| 1 | `juniper_data/generators/arc_agi/generator.py` | 63 | Load tasks from HuggingFace vs local |
| 2 | `juniper_data/generators/csv_import/generator.py` | 99 | Load data from CSV vs JSON |
| 3 | `juniper_data/generators/csv_import/generator.py` | 132 | Parse JSON array vs JSONL |
| 4 | `juniper_data/generators/csv_import/generator.py` | 147 | Use explicit feature columns vs auto-detect |

---

## Approach Evaluation

### Option A: Convert to ternary operators (SELECTED)

Convert each 4-line if-else block to a single-line ternary expression.

**Pros:**
- Directly resolves the lint violations
- All ternaries are under 145 characters (well within the 320-char line limit)
- Keeps the SIM108 rule active for future code
- Ternary is idiomatic Python for simple conditional assignment

**Cons:**
- Conversion 3 has a list comprehension with its own `if` filter in the else branch, creating
  two `if` keywords on one line (purely visual complexity, syntactically unambiguous)

### Option B: Add SIM108 to ignore list (REJECTED)

Add `"SIM108"` to `[tool.ruff.lint] ignore` in `pyproject.toml`.

**Pros:**
- Zero code changes required
- Consistent with existing SIM102/SIM105/SIM117 ignores

**Cons:**
- Suppresses a useful lint rule globally — future trivial if-else blocks won't be flagged
- The existing SIM ignores are qualitatively different (collapsible-if, contextlib.suppress,
  multi-with are more opinionated than ternary suggestions)

### Decision

**Option A** — Convert all 4 violations to ternary operators. All conversions are semantically
identical, fit within line limits, and maintain readability.

---

## Implementation Plan

### Step 1: Apply ternary conversions

**File: `juniper_data/generators/arc_agi/generator.py` (line 63)**

Before:
```python
        if params.source == "huggingface":
            tasks = ArcAgiGenerator._load_from_huggingface(params)
        else:
            tasks = ArcAgiGenerator._load_from_local(params)
```

After:
```python
        tasks = ArcAgiGenerator._load_from_huggingface(params) if params.source == "huggingface" else ArcAgiGenerator._load_from_local(params)
```

**File: `juniper_data/generators/csv_import/generator.py` (line 99)**

Before:
```python
        if file_format == "csv":
            data = CsvImportGenerator._load_csv(path, params)
        else:
            data = CsvImportGenerator._load_json(path, params)
```

After:
```python
        data = CsvImportGenerator._load_csv(path, params) if file_format == "csv" else CsvImportGenerator._load_json(path, params)
```

**File: `juniper_data/generators/csv_import/generator.py` (line 132)**

Before:
```python
            if content.startswith("["):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.split("\n") if line.strip()]
```

After:
```python
            data = json.loads(content) if content.startswith("[") else [json.loads(line) for line in content.split("\n") if line.strip()]
```

**File: `juniper_data/generators/csv_import/generator.py` (line 147)**

Before:
```python
        if params.feature_columns is not None:
            feature_cols = params.feature_columns
        else:
            feature_cols = [c for c in all_columns if c != params.label_column]
```

After:
```python
        feature_cols = params.feature_columns if params.feature_columns is not None else [c for c in all_columns if c != params.label_column]
```

### Step 2: Verify lint passes

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-data
ruff check juniper_data/generators/arc_agi/generator.py juniper_data/generators/csv_import/generator.py
```

### Step 3: Verify formatting passes

```bash
ruff format --check juniper_data/generators/arc_agi/generator.py juniper_data/generators/csv_import/generator.py
```

### Step 4: Run unit tests

```bash
pytest juniper_data/tests/unit/test_arc_agi_generator.py juniper_data/tests/unit/test_csv_import_generator.py -v
```

### Step 5: Run full pre-commit suite

```bash
pre-commit run --all
```

---

## Risk Assessment

- **Semantic risk**: None. All conversions are mechanically equivalent.
- **Test coverage**: Both generators have comprehensive test suites covering the affected methods.
- **Scope**: Only 2 files modified, no API or behavioral changes.

---

## Additional Note

A 5th SIM108 violation exists in `scripts/check_doc_links.py:256`, but this file falls outside
the pre-commit Ruff hook's file pattern (`^juniper_data/.*\.py$`) and does not cause pre-commit
failures. It can be addressed separately if desired.
