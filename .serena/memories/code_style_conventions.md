# Code Style and Conventions - Juniper Data

## Formatting Rules

### Line Length
- **512 characters** (configured in black, isort, flake8)
- This is intentionally long to allow horizontal code layout

### Formatter: Black
- Profile used for isort compatibility
- Target versions: py311, py312, py313

### Import Sorting: isort
- Profile: `black`
- Known first party: `juniper_data`
- Section order: FUTURE, STDLIB, THIRDPARTY, FIRSTPARTY, LOCALFOLDER

## Naming Conventions

### Constants
- Uppercase with underscores
- Prefix by component: `_DATA_DEFAULT_NOISE`, `_SPIRAL_GENERATOR_DEFAULT_POINTS`
- Private constants use leading underscore

### Classes
- PascalCase: `SpiralGenerator`, `DatasetStorage`, `Settings`

### Methods/Functions
- snake_case: `generate_dataset`, `get_configuration`, `validate_params`

### Private Members
- Single underscore prefix: `_internal_method`, `_private_attribute`

### Dunder Methods
- Double underscore: `__init__`, `__repr__`, `__str__`

### Variables
- snake_case: `n_points`, `noise_level`, `dataset_id`

## Type Hints
- **Required** for all public methods and functions
- Use modern typing syntax (Python 3.11+)
- Type annotations in function signatures, not in docstrings
- mypy configured with `ignore_missing_imports = true`

## Documentation

### Docstrings
- **Required** for all public classes and methods
- Google-style docstring format
- Example:
```python
def generate(self, n_points: int, noise: float = 0.1) -> Dataset:
    """Generate a spiral dataset.
    
    Args:
        n_points: Number of points per spiral.
        noise: Standard deviation of Gaussian noise.
    
    Returns:
        Dataset containing generated points and labels.
    
    Raises:
        ValueError: If n_points is not positive.
    """
```

### File Headers
- Files include standard header block with:
  - Project, Sub-Project, Application
  - File Name, Author, Version
  - Date Created, Last Modified
  - License, Copyright
  - Description
  - References, TODO, COMPLETED sections

## Pydantic Patterns
- Use Pydantic models for:
  - Configuration classes
  - API request/response schemas
  - Data validation
- BaseSettings for configuration management
- Field validators for complex validation

## Error Handling
- Raise appropriate exceptions with clear messages
- Use custom exception classes when appropriate
- Validate input data at API boundaries

## Test Patterns

### File Naming
- `test_<component>.py`

### Class Naming
- `Test<ComponentName>`

### Method Naming
- `test_<behavior_under_test>`

### Markers
Use pytest markers for test categorization:
```python
@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.spiral
@pytest.mark.api
@pytest.mark.generators
@pytest.mark.storage
```
