# S3 backtest notes

Holdout window: last ~20% of time cutoffs (`2025-11-11` → `2026-03-06`),
n=894 labeled rows.

| Metric | Value |
|--------|-------|
| Model wMAPE (P50 revenue) | **0.409** |
| Run-rate baseline wMAPE | 1.234 |
| Relative lift vs baseline | **~67.5%** |
| P10–P90 coverage (holdout, with 0.40 half-width cap) | **~0.36** |

## Setup

- LightGBM quantile models on `log1p(target_revenue)`, alphas 0.1 / 0.5 / 0.9
- Time-based split (no random shuffle)
- Intervals: prefer model quantiles; mild conformal expand; **hard cap** half-width
  at 0.40 so P90/P10 ≈ **2.3×** for planning usefulness (was ~12× with 0.85)
- Output ROAS is **not** clipped to 100 (feature winsor of 50 is training-only)
- Soft floor: if P50 ROAS ≥ 2, P10 ROAS ≥ 1.0

Strongest levels: aggregate / channel. Hardest: campaign (sparse / volatile).

**Inference stabilizer:** sparse campaigns (`hist_spend_sum_28` &lt; 800) blend model P50 toward hist run-rate before hierarchy reconcile (`src/predict.py`).

See `output/backtest_metrics.json` for per-horizon / per-level detail.
