#!/usr/bin/env bash
set -euo pipefail

# Scoring contract:
#   ./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
# Defaults support local no-arg runs.

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

# Prefer python3 when available (Linux); fall back to python (Windows).
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "ERROR: neither python3 nor python found on PATH" >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: DATA_DIR is not a directory: $DATA_DIR" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH not found: $MODEL_PATH" >&2
  exit 1
fi

OUT_DIR="$(dirname "$OUTPUT_PATH")"
mkdir -p "$OUT_DIR"

FEATURES_PATH="${OUT_DIR}/features.parquet"
REPORT_PATH="${OUT_DIR}/reconcile_report.json"

# 1. Features from whatever is in DATA_DIR (no hardcoded filenames beyond patterns)
"$PYTHON" src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out "$FEATURES_PATH"

# 2. Predict + hierarchy reconcile -> OUTPUT_PATH (overwrite, never append)
"$PYTHON" src/predict.py \
    --features "$FEATURES_PATH" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH" \
    --report-out "$REPORT_PATH"

echo "Done. Predictions written to $OUTPUT_PATH"
