# Golden Reference Datasets

This directory contains golden reference datasets generated from the JuniperCascor `SpiralProblem` implementation. These datasets are used for parity testing to ensure the new JuniperData spiral generator produces equivalent results.

## Purpose

- Validate that the JuniperData spiral generator produces identical or equivalent output to the original JuniperCascor implementation
- Provide regression testing for the data generation pipeline
- Document the expected data format and characteristics

## Dataset Files

### 2_spiral.npz
A 2-spiral classification dataset.

**Parameters:**
| Parameter | Value |
|-----------|-------|
| n_spirals | 2 |
| n_points | 100 (per spiral) |
| noise | 0.1 |
| seed | 42 |
| train_ratio | 0.8 |
| test_ratio | 0.2 |

**Expected Shapes:**
| Array | Shape |
|-------|-------|
| X_train | (160, 2) |
| y_train | (160, 2) |
| X_test | (40, 2) |
| y_test | (40, 2) |

**Expected Dtypes:**
- All arrays: `float32`

### 3_spiral.npz
A 3-spiral classification dataset.

**Parameters:**
| Parameter | Value |
|-----------|-------|
| n_spirals | 3 |
| n_points | 50 (per spiral) |
| noise | 0.05 |
| seed | 42 |
| train_ratio | 0.8 |
| test_ratio | 0.2 |

**Expected Shapes:**
| Array | Shape |
|-------|-------|
| X_train | (120, 2) |
| y_train | (120, 3) |
| X_test | (30, 2) |
| y_test | (30, 3) |

**Expected Dtypes:**
- All arrays: `float32`

## Array Descriptions

| Array | Description |
|-------|-------------|
| X_train | Training input features (x, y coordinates) |
| y_train | Training labels (one-hot encoded) |
| X_test | Test input features (x, y coordinates) |
| y_test | Test labels (one-hot encoded) |

## Class Distribution

For balanced datasets, each class should have approximately equal representation:
- **2-spiral**: ~50% per class
- **3-spiral**: ~33.3% per class

Note: Due to shuffling and train/test split, exact distributions may vary slightly.

## Regenerating Datasets

To regenerate the golden datasets from JuniperCascor:

```bash
# Set these to the appropriate locations on your system or in CI:
export JUNIPER_CASCOR_SRC=/path/to/JuniperCascor/juniper_cascor/src
export JUNIPER_DATA_ROOT=/path/to/JuniperData

# From the JuniperCascor source directory, run the generator script from this repo:
cd "$JUNIPER_CASCOR_SRC"
python "$JUNIPER_DATA_ROOT/juniper_data/tests/fixtures/generate_golden_datasets.py"
```

## Usage in Tests

```python
import numpy as np

def load_golden_dataset(name: str):
    data = np.load(f"tests/fixtures/golden_datasets/{name}.npz")
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
    }

def test_spiral_parity():
    golden = load_golden_dataset("2_spiral")
    generated = generate_spiral_dataset(n_spirals=2, n_points=100, noise=0.1, seed=42)

    np.testing.assert_allclose(generated["X_train"], golden["X_train"], rtol=1e-5)
    np.testing.assert_allclose(generated["y_train"], golden["y_train"], rtol=1e-5)
```

## Metadata Files

Each `.npz` file has a corresponding `_metadata.json` file containing:
- Generation parameters
- Array shapes and dtypes
- Class distribution statistics
- Value ranges
