# PyTecplot Development Setup

This guide will help you set up your development environment for PyTecplot.

## Prerequisites

1. **Python 3.10 or higher**
   ```bash
   python --version  # Should be 3.10+
   ```

2. **Tecplot 360**
   - Must be installed on your system
   - The `libtecio` shared library must be accessible
   - Tecplot executable should be in your PATH

3. **Git** (for version control)

## Initial Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd pytecplot
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install package in editable mode with dev dependencies
make install-dev

# Or manually:
pip install -e ".[dev]"
```

This installs:
- **Runtime dependencies**: NumPy
- **Development tools**: Black, isort, Pylint, Pyflakes, MyPy

### 4. Verify Installation

```bash
# Check that Tecplot can be found
python -c "from tecutils import get_tec_exe; print(get_tec_exe())"

# Check tool versions
make versions
```

## Development Workflow

### Code Formatting

```bash
# Format all Python files
make format

# This runs:
# - isort (sorts imports)
# - black (formats code)
```

### Linting

```bash
# Run linters
make lint

# This runs:
# - pyflakes (error checking)
# - pylint (code quality)
```

### Type Checking

```bash
# Run type checker
make typecheck

# This runs mypy
```

### All Checks

```bash
# Run formatting + linting + type checking
make check
```

### Running Tests

```bash
# Run unit tests (when implemented)
make test
```

## Code Style Guidelines

### PEP 8 Compliance

- Line length: 88 characters (Black default)
- Use 4 spaces for indentation
- Use type hints for all functions

### Naming Conventions

- **Variables/functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private attributes**: `_leading_underscore`

### Import Organization (isort)

Imports are automatically organized in this order:
1. Future imports
2. Standard library
3. Third-party packages (e.g., numpy)
4. First-party (project modules)
5. Local folder imports

Example:
```python
from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from szlio import ZoneType
from auxdata import AuxData
```

### Type Hints

Always use type hints:
```python
def get_variable_data(self, var_index: int) -> Optional[npt.NDArray]:
    """Get variable data array."""
    ...
```

### Docstrings

Use docstrings for public APIs:
```python
def normalize_variable(self, var_index: int, reference_value: float) -> None:
    """
    Normalize a variable by a reference value.
    
    Args:
        var_index: Variable index (0-based)
        reference_value: Value to divide by
    """
    ...
```

## Common Development Tasks

### Adding a New Module

1. Create the Python file
2. Add appropriate imports and type hints
3. Format code: `make format`
4. Check code: `make check`
5. Add tests (when test framework is set up)

### Modifying Existing Code

1. Make your changes
2. Run `make format` to auto-format
3. Run `make check` to verify quality
4. Test your changes
5. Commit

### Before Committing

```bash
# Always run before committing
make check

# Clean up any generated files
make clean
```

## Editor Configuration

### VS Code

Add to `.vscode/settings.json`:
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.pyflakesEnabled": true,
    "editor.formatOnSave": true,
    "python.analysis.typeCheckingMode": "basic",
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

### PyCharm

1. Go to `Settings > Tools > Black`
2. Enable "On save"
3. Set line length to 88

For isort:
1. Go to `Settings > Tools > File Watchers`
2. Add isort file watcher

## Troubleshooting

### Tecplot Not Found

If you get errors about Tecplot not being found:

```bash
# Check if Tecplot is in PATH
which tec360  # Linux/macOS
where tec360  # Windows

# On macOS, if installed but not in PATH:
export PATH="/Applications/Tecplot 360 EX 2023 R1/Tecplot 360 EX 2023 R1.app/Contents/MacOS:$PATH"

# Test again
python -c "from tecutils import get_tec_exe; print(get_tec_exe())"
```

### Import Errors

If you get import errors:

```bash
# Make sure package is installed in editable mode
pip install -e .

# Verify installation
pip list | grep pytecplot
```

### Linter Errors

If you get many linter errors:

```bash
# Auto-format first
make format

# Then check what's left
make lint

# You can disable specific warnings in pyproject.toml
```

## Getting Help

- Check the README.md for usage examples
- Review existing code for patterns
- Ask questions in issues/discussions

## Contributing

1. Create a feature branch
2. Make your changes
3. Run `make check`
4. Commit with clear messages
5. Submit pull request

## Useful Commands Reference

```bash
make help          # Show all available commands
make install-dev   # Install with dev dependencies
make format        # Format code
make lint          # Run linters
make typecheck     # Run type checker
make check         # Run all checks
make clean         # Clean generated files
make versions      # Show tool versions
```
