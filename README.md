# AIgnition Forecasting

Probabilistic ecommerce **revenue** and **ROAS** forecasting across Google Ads, Meta Ads, and Microsoft Ads — built for NetElixir AIgnition 3.0.

**Python:** 3.12.x  

Holdout (sample data): model wMAPE **~0.41** vs run-rate baseline **~1.23** (~**67%** lift); interval coverage **~0.36** with tight ~2.3× P10–P90 bands.

---

## Quick start (scorers)

```bash
pip install -r requirements.txt
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
# Windows: .\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv
python src/verify_submission.py
```

Contract: three args (`DATA_DIR`, `MODEL_PATH`, `OUTPUT_PATH`) with those defaults.  
No network, no prompts, no retraining at score time.

---

## Repository layout

```
├── run.sh / run.ps1       # Scoring entrypoint
├── app.py                 # Streamlit demo UI
├── llm_layer.py           # Optional AI causal insights
├── requirements.txt       # Pinned score-time deps
├── requirements-demo.txt  # Streamlit (+ score deps) for demo only
├── data/                  # Channel CSVs (replaced at test time)
├── pickle/model.pkl       # Trained LightGBM quantile bundle
├── scenarios/             # Example budget what-ifs
├── src/                   # Ingest → features → model → predict
├── demo/                  # Streamlit prototype + LLM insights
├── docs/                  # Methodology, architecture, walkthrough
└── output/                # Local artifacts (gitignored except .gitkeep)
```

---

## Documentation (deliverables)

| Doc | Purpose |
|-----|---------|
| [`docs/methodology.md`](docs/methodology.md) | Forecasting method, model, preprocessing, assumptions, limits, AI |
| [`docs/architecture.md`](docs/architecture.md) | Frontend / backend / pipeline / LLM flow |
| [`docs/demo_walkthrough.md`](docs/demo_walkthrough.md) | 5–7 min presentation script |
| [`docs/output_contract.md`](docs/output_contract.md) | `predictions.csv` schema |
| [`docs/submission_checklist.md`](docs/submission_checklist.md) | Pre-submit checks |
| [`docs/SUBMISSION_EMAIL.md`](docs/SUBMISSION_EMAIL.md) | Email template to NetElixir |
| [`docs/data_assumptions.md`](docs/data_assumptions.md) | Cleaning + attribution policy |
| [`docs/modeling_decision.md`](docs/modeling_decision.md) | Why LightGBM (+ sklearn), not Prophet-primary |
| [`docs/backtest_results.md`](docs/backtest_results.md) | Holdout metrics summary |

---

## Schema map (raw → unified)

| Unified | Google | Meta | Bing |
|---------|--------|------|------|
| `date` | `segments_date` | `date_start` | `TimePeriod` |
| `spend` | `metrics_cost_micros / 1e6` | `spend` | `Spend` |
| `revenue` | `metrics_conversions_value` | `conversion` | `Revenue` |
| `campaign_type` | channel type col | inferred from name | `CampaignType` |

Platform attribution is used as-is (no custom MMM).

---

## Local workflows

**Train (offline only):**
```bash
python src/train.py --data-dir ./data --model-out ./pickle/model.pkl
```

**Budget simulation:**
```bash
python src/simulate.py --budget-json scenarios/meta_plus20.json \
  --out ./output/predictions_meta_plus20.csv
```

**Demo:**
```bash
pip install -r requirements-demo.txt
streamlit run app.py
```
Uses root `app.py` + `llm_layer.py`. Free LLM: set `GROQ_API_KEY` from
[console.groq.com](https://console.groq.com) (or paste in the sidebar). Falls
back to offline heuristic with no key.

---

## Status

| Step | Status |
|------|--------|
| S0–S6 Build + demo | Complete |
| S7 Docs + submission package | Complete |

---

## Submission

Email **public GitHub URL**, the exact `./run.sh ...` command, and team details to  
`sunitha.k@netelixir.us` by **19 July 2026, 10:00 PM IST**.  
Template: [`docs/SUBMISSION_EMAIL.md`](docs/SUBMISSION_EMAIL.md).
