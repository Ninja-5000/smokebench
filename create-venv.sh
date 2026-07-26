#!/usr/bin/env bash
# Quick dev setup: create venv, install package with dev + full deps
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev,full]"

echo "✅ Dev environment ready."
echo "Activate with: source .venv/bin/activate"
echo "Run TUI:       smokebench"
echo "Run tests:     python -m pytest"