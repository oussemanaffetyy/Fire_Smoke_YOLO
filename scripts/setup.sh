#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="python3.11"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required.")
if sys.version_info >= (3, 13):
    raise SystemExit("Python 3.13 is not recommended for this project. Use Python 3.10 or 3.11.")
print(f"Using Python {sys.version.split()[0]}")
PY

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete."
echo "Run: source .venv/bin/activate && python src/run_camera.py"
