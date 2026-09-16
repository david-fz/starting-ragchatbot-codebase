#!/bin/bash
# Auto-format the codebase with black.
set -e

cd "$(dirname "$0")/.."

echo "Running black..."
uv run black backend main.py
