.PHONY: help install install-dev format lint check test clean

# Default target
help:
	@echo "Tecio Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install package dependencies"
	@echo "  make install-cli      Install command line scripts"
	@echo "  make install-dev      Install package + dev dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format           Format code ruff"
	@echo "  make lint             Run code linting with ruff"
	@echo "  make check            Run all checks (format + lint)"
	@echo "  make typecheck        Run mypy type checking"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run unit tests (not yet implemented)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove generated files"

# Install package dependencies
install:
	pip install -e .
	pipx install -e .
	pipx ensurepath

# Install only command line scripts
install-cli:
	pipx install -e .
	pipx ensurepath

# Install package + development dependencies
install-dev:
	pip install -e ".[dev]"

# Uninstall package
uninstall:
	pip uninstall tecio
	pipx uninstall tecio

# Uninstall command line scripts only
uninstall-cli:
	pipx uninstall tecio

# Format code with black and isort
format:
	ruff format .

# Run linters
lint:
	ruff check . --fix

# Run type checker
typecheck:
	mypy .

# Run all checks
check: format lint typecheck
	@echo ""
	@echo "✓ All checks passed"

# Run unit tests (when implemented)
test:
	pytest
	@echo "✓ Tests complete"
# Clean up generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "test*.plt" -delete
	find . -type f -name "test*.szplt" -delete
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "✓ Cleanup complete"

# Show current versions of tools
versions:
	@echo "Tool Versions:"
	@echo "=============="
	@python --version
	@ruff --version | head -n1
	@mypy --version
	@pytest --version
