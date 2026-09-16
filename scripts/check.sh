#!/bin/bash
# Run code quality checks: formatting check and test suite.
set -e

cd "$(dirname "$0")/.."

echo "Checking formatting with black..."
uv run black --check --diff backend main.py

echo "Running tests..."
uv run pytest

echo "All checks passed."
