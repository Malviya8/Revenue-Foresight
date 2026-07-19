# Modeling decision (expert stack for this dataset)

## Verdict

**Primary model:** LightGBM quantile regression (P10 / P50 / P90) on
hierarchical *period* targets (sum of revenue over next H ∈ {30,60,90} days),
conditioned on recent performance + **planned spend**.

**Surrounding toolkit:** scikit-learn for time-based CV, metrics (wMAPE,
coverage), baselines (Ridge / seasonal-naive), and optional conformal
calibration of intervals.

**Not primary:** Prophet. Strong for univariate seasonality, weak for
budget-conditional, multi-entity ecommerce panel forecasting under this brief.

## Why this fits *this* data

Sample panel reality (S1 QA):

- ~25k daily campaign rows across Google / Meta / Bing
- Strong **Nov–Dec seasonality** (all channels)
- Wide ROAS tails (daily ROAS > 50 is common) → need robust tabular learners
  + Winsorized features, not fragile ARIMA/Prophet on sparse series
- Bing is sparse (~32% zero-activity rows) → shared model with hierarchy
  features beats thousands of per-series Prophets
- Brief requires **budget simulation** → spend/budget must be covariates;
  Prophet does not model spend→revenue response well
- Scorers need one pickled artifact + offline `run.sh` → GBDT pickle is
  standard and stable when versions are pinned

## Architecture

```
cleaned panel
  -> S2 hierarchical period features (as_of + horizon)
  -> LightGBM quantile models (revenue | features, planned_spend)
  -> ROAS = predicted_revenue / assumed_spend
  -> optional sklearn conformal adjust on validation residuals
  -> reconcile hierarchy (campaign -> type -> channel -> aggregate)
```

Prophet (or a seasonal naive) may still appear as a **channel-level baseline**
in the writeup to prove lift — not as the scored `model.pkl` path.

## Prophet benchmark (TEST only)

`python scripts/evaluate.py --with-prophet` performs a fair walk-forward
benchmark: Prophet is refit at every TEST `as_of_date`, sees only history
available at that cutoff, uses planned spend as a regressor, and is scored on
exactly the same Google/Meta channel-level rows as LightGBM.

| Model | Channel-level TEST wMAPE (Google + Meta, n=48) |
|-------|-----------------------------------------------|
| LightGBM quantile P50 | **48.7%** |
| Prophet + spend regressor | 98.5% |
| Run-rate | 114.3% |

LightGBM improves on Prophet by **50.5%** on this matched slice. Per channel:
Google 50.7% vs Prophet 92.1%; Meta 39.5% vs Prophet 128.3%. This validates
Prophet as a useful benchmark, but not as the primary submission model.

## Alternatives considered

| Option | Role |
|--------|------|
| sklearn `HistGradientBoostingRegressor` (quantile) | Acceptable LGBM-free fallback |
| Prophet per channel | Baseline / ensemble component only |
| Pure linear sklearn | Too weak for seasonality + saturation |
| Deep sequence models | Overkill for timeline + data size |
