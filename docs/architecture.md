# Architecture overview

## System diagram

```
┌─────────────────────────────────────────────────────────────┐
│  SCORING (run.sh) — offline, no network                     │
│  data/ → generate_features → predict(+reconcile) → CSV      │
│  model: pickle/model.pkl                                    │
└─────────────────────────────────────────────────────────────┘
          ▲ same core modules
┌─────────────────────────────────────────────────────────────┐
│  DEMO (Streamlit) — optional network for LLM only           │
│  sidebar budgets → baseline + scenario → charts → insights  │
└─────────────────────────────────────────────────────────────┘
```

## Frontend

- **Streamlit** (`app.py`)
- Tabs: Data QA · Forecast · Channels · Campaigns · AI Insights
- Budget multipliers per channel; horizon 30/60/90
- Downloads scenario `predictions.csv`

## Backend / core (`src/`)

| Module | Responsibility |
|--------|----------------|
| `schema.py` | Contracts, mappings, horizons |
| `ingest.py` / `clean.py` / `validate.py` | Unify + QA + clean panel |
| `features.py` | Hierarchical period features |
| `model.py` / `train.py` | LightGBM quantiles + conformal |
| `predict.py` / `reconcile.py` | Inference + hierarchy consistency |
| `budget.py` / `simulate.py` | Spend scenarios |
| `verify_submission.py` | Submission contract checks |
| `metrics.py` | wMAPE / coverage / baseline |

## Forecasting pipeline

```
CSV (Google / Meta / Bing)
  -> unify + validate + clean
  -> as_of × horizon features (+ planned_spend)
  -> LightGBM P10/P50/P90 on log1p(revenue)
  -> relative conformal bands
  -> reconcile (type/campaign → channel; aggregate = Σ channel)
  -> predictions.csv
```

## LLM integration workflow

```
forecast tables + QA inventory + multipliers
  -> llm_layer.build_insight_context  (numbers only)
  -> if OPENAI_API_KEY: Chat Completions (gpt-4o-mini default)
     else: heuristic_insights()
  -> markdown causal briefing
```

Constraints:

- Never imported by `run.sh`
- Prompt forbids inventing metrics not in context JSON
- Failures soft-fallback to heuristic so the demo never crashes

## Deployment mental model for judges

1. Clone public repo  
2. `pip install -r requirements.txt`  
3. Replace `data/` with held-out CSVs (same schema)  
4. `./run.sh ./data ./pickle/model.pkl ./output/predictions.csv`  
5. Optionally: `streamlit run app.py` for the live walkthrough  
