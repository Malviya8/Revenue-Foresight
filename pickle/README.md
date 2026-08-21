# Trained model artifact

`model.pkl` is the **committed** LightGBM quantile bundle used by `run.sh`.

- **Do not retrain inside `run.sh`** — the scoring CLI loads this file offline.
- **Retrain locally (optional):** `python src/train.py --data-dir ./data --model-out ./pickle/model.pkl`
- **Format:** `RevenueQuantileModel` — P10/P50/P90 LightGBM boosters + Mondrian `conformal_adjustments`
- **Expected size:** ~6–7 MB (LightGBM boosters + conformal strata)

If the file is missing, empty, or not a usable `RevenueQuantileModel`, `predict.py`
**raises** (no silent stub). Commit the real artifact before scoring.
