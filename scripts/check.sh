#!/usr/bin/env bash
set -euo pipefail

# Canonical local gate: full test suite, lint, and the blocking
# independence gate. Run from the repository root:
#   bash scripts/check.sh

cd "$(dirname "$0")/.."

# Prefer the project venv when present so `python` resolves to the
# interpreter with the dev dependencies installed.
if [ -x ".venv/bin/python" ]; then
  export PATH="$PWD/.venv/bin:$PATH"
fi

echo "== python -m pytest -q =="
python -m pytest -q

echo "== ruff check src tests =="
if command -v ruff >/dev/null 2>&1; then
  ruff check src tests
else
  echo "error: ruff not found on PATH (declared dev dependency in pyproject.toml [dev]); refusing to skip lint" >&2
  exit 1
fi

echo "== pytest tests/independence -q =="
python -m pytest tests/independence -q
