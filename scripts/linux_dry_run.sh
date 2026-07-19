#!/usr/bin/env bash
# Fresh-environment scorer dry-run (matches submission checklist).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-/tmp/aignition-venv}"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -r requirements.txt
bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv
python src/verify_submission.py
python check.py
echo "Linux dry-run OK"
