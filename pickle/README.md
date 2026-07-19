# Trained model artifact

`model.pkl` is the **committed** LightGBM quantile bundle used by `run.sh`.

- **Do not retrain inside `run.sh`** — scorers load this file offline.
- **Retrain locally (optional):** `python src/train.py --data-dir ./data --model-out ./pickle/model.pkl`
- **Format:** `RevenueQuantileModel` (P10 / P50 / P90 boosters + `conformal_rel` interval scale)
- **Expected size:** ~1–2 MB

If the file is missing or empty, `predict.py` falls back to a schema-valid historical-ROAS stub (for local dev only — commit the real artifact before submission).
