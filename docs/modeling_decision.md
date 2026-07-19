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

## Alternatives considered

| Option | Role |
|--------|------|
| sklearn `HistGradientBoostingRegressor` (quantile) | Acceptable LGBM-free fallback |
| Prophet per channel | Baseline / ensemble component only |
| Pure linear sklearn | Too weak for seasonality + saturation |
| Deep sequence models | Overkill for timeline + data size |
