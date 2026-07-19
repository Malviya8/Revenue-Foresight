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
2. **Mondrian split-conformal prediction** (train 60% / calib 20% / test 20% by date):
   nonconformity `score = max(p10−y, y−p90)` on calibration; scores are **clipped at the
   95th percentile** before the finite-sample quantile (holiday strata also use the
   matching non-holiday 95th as an upper clip, and holiday adjustments are capped at
   **5×** the non-holiday adjustment) so a single outlier period cannot inflate the
   band. Strata: channel × season (holiday Nov–Dec), plus for Google a **brand vs
   non-brand** cut (TM Search vs other). Additive adjustment at level
   `ceil((n+1)·0.80)/n`. At inference: `p10' = p10 − adj`, `p90' = p90 + adj`,
   **p50 unchanged**.  
3. Soft floor: if P50 ROAS ≥ 2, P10 ROAS ≥ 1.0 (p50 unchanged).  
4. Quantile order: `p10 ≤ 0.95·p50`, `p90 ≥ 1.05·p50`, revenue ≥ 0.

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

### Prophet baseline benchmark

On the same 48 Google/Meta channel-level TEST rows, using walk-forward refits
with no target leakage:

| Model | wMAPE |
|-------|------:|
| LightGBM quantile P50 | **48.7%** |
| Prophet + planned-spend regressor | 98.5% |
| Run-rate | 114.3% |

LightGBM achieves **50.5% lower wMAPE than Prophet** on this matched comparison.
The optional benchmark is reproducible with
`pip install -r requirements-eval.txt` and
`python scripts/evaluate.py --with-prophet`; Prophet is never used by `run.sh`.

### Holdout results (sample data — TEST only)

| Metric | Value |
|--------|-------|
| Model wMAPE (P50, excl. Bing) | **0.500** |
| Run-rate baseline wMAPE | 1.234 |
| Lift vs baseline | **59.5%** |
| MAPE (P50, excl. Bing, secondary) | 90.3% |
| P10–P90 coverage | **74.0%** (Mondrian split-conformal on TEST) |

Headline point accuracy is **wMAPE** (volume-weighted). Simple MAPE is reported for diagnostics but is not the primary score — equal-weight MAPE lets tiny campaigns dominate. Bing is excluded from headline accuracy (see *Bing Channel Limitations*).

Details: [`backtest_results.md`](backtest_results.md), [`evaluation_snapshot.json`](evaluation_snapshot.json).  
`output/backtest_metrics.json` (from `train.py`) is **calib diagnostics only** — not the holdout score.

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

### Bing Channel Limitations

Bing (Microsoft Ads) revenue in the sample panel is **75%+ sparse** (zero-revenue rows on the large majority of day×campaign observations). That sparsity makes reliable supervised ML forecasting for Bing effectively impossible: the target is mostly zeros, gradients are dominated by rare non-zero days, and row-wise error metrics (especially simple MAPE) explode whenever a near-zero actual sits in the denominator.

**How we handle it:**

- Bing rows still flow through the shared LightGBM + conformal path so the scored CSV remains complete for all three channels.
- At inference, sparse entities (Bing is almost entirely in this regime) **blend toward hist run-rate** (`planned_spend × hist_roas_28`) before hierarchy reconcile — see `src/predict.py` `_shrink_sparse_campaigns`.
- **Bing is excluded from headline accuracy metrics** (wMAPE / lift reported in README and `evaluation_snapshot.json`). Channel diagnostics for Bing use sMAPE / MASE instead of MAPE.

This is a **data quality constraint**, not a modeling failure: Google and Meta carry the planning signal; Bing forecasts are best treated as run-rate planning stubs until denser conversion history is available.

---

## 6. AI integration strategy

| Surface | Behavior |
|---------|----------|
| `run.sh` scoring | **No LLM / no network** |
| `app.py` + `llm_layer.py` | Optional LLM briefing (Groq first, then OpenAI) grounded in forecast + QA JSON |
| Fallback | Deterministic heuristic memo citing the same numbers |

LLM is instructed to **only cite provided figures** (causal “why”, risks, reallocation ideas).  
Heuristic path works fully offline for demos without keys. Set `GROQ_API_KEY` (or `OPENAI_API_KEY`) for live insights.

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
