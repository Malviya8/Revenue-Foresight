# Holdout (TEST) backtest results

**Authoritative metrics** come from `python scripts/evaluate.py` on the final
20% chronological TEST window. Snapshot: [`evaluation_snapshot.json`](evaluation_snapshot.json).

| Field | Value |
|-------|-------|
| Holdout window | `2025-11-11` → `2026-03-06` |
| Labeled TEST rows | n=894 |
| Train / calib (fit only) | n=3141 / n=1191 |

| Metric | Value |
|--------|-------|
| Model wMAPE (P50 revenue, excl. Bing) | **0.500** |
| Run-rate baseline wMAPE | 1.234 |
| Relative lift vs run-rate | **59.5%** |
| MAPE (P50, excl. Bing, secondary) | 90.3% |
| P10–P90 coverage (TEST, Mondrian split-conformal) | **74.0%** |

Headline point accuracy is **wMAPE** (`sum|e| / sum|y|`). Simple MAPE is secondary.
Bing is excluded from headline wMAPE / lift (data sparsity — see methodology).

## Calib vs TEST (do not mix)

| Artifact | What it measures |
|----------|------------------|
| `output/backtest_metrics.json` (from `src/train.py`) | **Calibration-window diagnostics only** — not the holdout score |
| `python scripts/evaluate.py` → `output/evaluation_metrics.json` | **TEST-only** metrics used in README / this doc |
| [`evaluation_snapshot.json`](evaluation_snapshot.json) | Committed copy of the TEST headline numbers |

## Setup

- LightGBM quantile models on `log1p(target_revenue)`, alphas 0.1 / 0.5 / 0.9
- Chronological 60% train / 20% calib / 20% test (never random)
- Intervals: **Mondrian split-conformal** additive adjustment from calib nonconformity
  scores; per-channel × holiday/non-holiday strata, plus Google brand vs non-brand;
  scores trimmed at the 95th percentile; holiday adj capped at 5× non-holiday
- Output ROAS is **not** clipped to 100 (feature winsor of 50 is training-only)
- Soft floor: if P50 ROAS ≥ 2, P10 ROAS ≥ 1.0

Strongest levels: aggregate / channel. Hardest: campaign (sparse / volatile).
Bing is excluded from headline accuracy — prefer sMAPE / MASE from `evaluate.py`.

**Inference stabilizer:** sparse campaigns (`hist_spend_sum_28` &lt; 800) blend model P50
toward hist run-rate before hierarchy reconcile (`src/predict.py`).

Reproduce:

```bash
python scripts/evaluate.py
```
