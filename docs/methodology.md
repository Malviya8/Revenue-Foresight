# Technical documentation — AIgnition Forecasting

Probabilistic revenue and ROAS forecasting for ecommerce marketing agencies
(Google Ads, Meta Ads, Microsoft Ads). Aggregate planning horizons: **30 / 60 / 90** days.

---

## 1. Forecasting methodology

### Problem framing

Predict **period-aggregate** revenue and blended ROAS with **P10 / P50 / P90** ranges at:

- ecommerce aggregate  
- channel  
- campaign type  
- campaign  

conditioned on **planned media spend** (budget simulation). Daily series are only an intermediate representation.

### Target & features

- **Target:** sum of attributed revenue over the next *H* days (`target_revenue`).
- **Primary covariates:** trailing 7/14/28/60d spend & revenue, Winsor-aware ROAS (cap 50), trends, volatility, budget fill, seasonality (month / Q4 / holiday), hierarchy codes, log-spend saturation, and **`planned_spend`**.
- **ROAS output:** `predicted_revenue / assumed_spend` (where assumed spend = planned spend after scenario overlays).

### Uncertainty

1. LightGBM **quantile** models (α = 0.1 / 0.5 / 0.9) on `log1p(revenue)`.  
2. **Relative conformal** expand on holdout (`conformal_rel`, capped at `MAX_INTERVAL_HALF_WIDTH = 0.40`) so bands stay planning-useful (~2.3× P90/P10) while coverage ≈ 80%.  
3. Hard half-width cap at **0.40** even when raw quantiles are wider.  
4. Quantile order enforced (`P10 ≤ P50 ≤ P90`).

### Hierarchy reconciliation

Children are scaled to their **channel** parent; **aggregate = Σ channels**.  
This keeps budget what-ifs coherent (channel spend shocks roll up to ecommerce totals).

### What we explicitly do *not* build

- Custom attribution / MMM (out of scope; platform attribution is source of truth)  
- Daily scored forecasts as the primary artifact  
- Online / in-`run.sh` retraining or network calls  

---

## 2. Model selection

| Component | Choice | Role |
|-----------|--------|------|
| Primary | LightGBM quantile GBDT | Best accuracy on this hierarchical panel + spend response |
| Toolkit | scikit-learn metrics / time split | wMAPE, coverage, baselines |
| Baseline | `planned_spend × hist_roas_28` | Prove lift |
| Not primary | Prophet | Weak for budget-conditional, sparse multi-entity panels |

Decision rationale: [`modeling_decision.md`](modeling_decision.md).

### Holdout results (sample data)

| Metric | Value |
|--------|-------|
| Model wMAPE (P50) | **~0.41** |
| Run-rate baseline wMAPE | ~1.23 |
| Lift vs baseline | **~67%** |
| P10–P90 coverage | **~0.36** (tradeoff: tight ~2.3× bands for planning) |

Details: [`backtest_results.md`](backtest_results.md), `output/backtest_metrics.json`.

---

## 3. Data preprocessing

```
data/*.csv
  -> pattern match per channel (google/meta/bing)
  -> unify schema (micros→currency, Meta type inference)
  -> QA (duplicates, negatives, sparsity, peaks)
  -> clean (dedupe, clip, derive ROAS/share/rolls)
  -> hierarchical period features
```

Policies & quirks: [`data_assumptions.md`](data_assumptions.md).

---

## 4. Assumptions

1. Platform-reported conversion value = revenue for that channel (as-is attribution).  
2. Cross-channel sums may overlap users; acceptable under brief constraints.  
3. Future spend defaults to recent run-rate × horizon unless a scenario overrides it.  
4. Sparse campaigns borrow strength via shared GBDT + type/channel features.  
5. Output column schema in [`output_contract.md`](output_contract.md) is our contract until organizers publish a different official format.

---

## 5. Limitations

1. **Campaign-level error** is higher than channel/aggregate (sparsity, creative/auction shocks). Prefer channel or type for planning.  
2. **Spend response** is learned from historical covariation — not a controlled experiment; large budget shocks can show diminishing (or noisy) returns.  
3. **Seasonality** is calendar-feature based; brand-new peak events outside history are under-represented.  
4. **Meta `daily_budget`** often missing → budget-fill features partially undefined.  
5. **Provisional prediction schema** — confirm scored CSV columns with organizers if published separately.  
6. Intervals are calibrated globally (relative); a single conformal radius cannot perfectly fit every entity scale.

---

## 6. AI integration strategy

| Surface | Behavior |
|---------|----------|
| `run.sh` scoring | **No LLM / no network** |
| `app.py` + `llm_layer.py` | Optional OpenAI briefing grounded in forecast + QA JSON |
| Fallback | Deterministic heuristic memo citing the same numbers |

LLM is instructed to **only cite provided figures** (causal “why”, risks, reallocation ideas).  
Heuristic path works fully offline for demos without keys.

Architecture: [`architecture.md`](architecture.md). Walkthrough: [`demo_walkthrough.md`](demo_walkthrough.md).

---

## 7. How to reproduce

```bash
# Score path
pip install -r requirements.txt
bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv
python src/verify_submission.py

# Retrain offline (not done by scorers)
python src/train.py --data-dir ./data --model-out ./pickle/model.pkl

# Demo
pip install -r requirements-demo.txt
streamlit run app.py
```

Python **3.12.x**. Pin set in `requirements.txt` must match the pickled LightGBM/sklearn versions.
