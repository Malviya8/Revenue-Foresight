# Demo walkthrough (5–7 minutes)

Use this as the live script for judges / final presentation.
Run beforehand: `streamlit run app.py`

---

## 0. Positioning (30 sec)

> “Agencies need forward-looking revenue and ROAS ranges under budget
> scenarios — not another dashboard of yesterday’s spend. We ship a
> probabilistic, multi-channel forecasting utility with explainable AI.”

Show: repo layout + `run.sh` (scoring) vs Streamlit (prototype).

---

## 1. Data ingestion & QA (60–90 sec)

1. Open **Data QA** tab (after **Run forecast**).  
2. Point out:
   - three channels unified despite different schemas  
   - Bing sparsity vs Google density  
   - Nov–Dec seasonality peaks  
3. Line: “We treat platform attribution as ground truth — no custom MMM.”

---

## 2. Baseline forecast (90 sec)

1. Horizon = **30**. Multipliers all **1.0**. Run forecast.  
2. **Forecast** tab: P50 revenue, P10–P90 fan, blended ROAS.  
3. Mention holdout lift: ~**67%** wMAPE improvement vs run-rate baseline.  
4. Note intervals are capped for planning (~2.3× P90/P10); holdout coverage is ~**36%**
   (tighter bands trade coverage for usable ranges — see `docs/backtest_results.md`).

---

## 3. Budget simulation (90 sec)

1. Set **Meta → 1.20**, re-run.  
2. Compare scenario vs baseline metrics.  
3. Call out **diminishing returns** if Meta ROAS falls while spend rises —
   “the model refuses linear ROAS fantasy.”  
4. Optional CLI proof:  
   `python src/simulate.py --budget-json scenarios/meta_plus20.json`

---

## 4. Hierarchy drill-down (60 sec)

1. **Channels** — contribution and reconciliation (aggregate = Σ channels).  
2. **Campaigns** — top contributors / wide intervals = operational risk.

---

## 5. AI causal memo (60–90 sec)

1. **AI Insights** → Generate (API key optional; heuristic works offline).  
2. Show: why / risks / reallocation bullets citing the same numbers.  
3. Emphasize: “LLM is demo-only — scorers never hit the network.”

---

## 6. Close (30 sec)

> “Technically: LightGBM quantiles + spend features + hierarchy reconcile.  
> Operationally: 30/60/90 planning, budget what-ifs, explainable ranges.  
> Submission: one command — `./run.sh ./data ./pickle/model.pkl ./output/predictions.csv`.”

---

## Slide checklist (if using slides)

1. Problem (fragmented channels, ROAS-aware planning)  
2. Architecture diagram (`docs/architecture.md`)  
3. Model + holdout metrics  
4. Demo screenshots / live  
5. Limitations & assumptions  
6. Submission command  
