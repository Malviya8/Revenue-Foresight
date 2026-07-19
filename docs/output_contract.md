# Prediction output contract (S4)

## File

Default scored artifact: `output/predictions.csv` (path from `run.sh` `OUTPUT_PATH`).

Written fresh every run (never appended).

## Columns

| Column | Description |
|--------|-------------|
| `horizon_days` | 30 / 60 / 90 |
| `level` | `aggregate` \| `channel` \| `campaign_type` \| `campaign` |
| `channel` | `all` for aggregate; else `google` / `meta` / `bing` |
| `campaign_type` | Type label (blank at aggregate/channel) |
| `campaign_id` / `campaign_name` | Campaign identity (campaign level) |
| `assumed_spend` | Period media spend assumed for the forecast |
| `p10_revenue` / `p50_revenue` / `p90_revenue` | Probabilistic revenue |
| `p10_roas` / `p50_roas` / `p90_roas` | `revenue / assumed_spend` |

**Note:** Confirm with organizers if the official scored schema differs.

## Hierarchy reconciliation

After model inference:

1. Within each channel, scale `campaign_type` / `campaign` so children sum to
   the **channel** P50 / assumed spend.
2. Rebuild `aggregate` as the **sum of channels** (so budget what-ifs roll up).

Disable with `--no-reconcile` on `predict.py` / `simulate.py`.

## Budget simulation

Baseline `run.sh` uses run-rate planned spend (no scenario file).

What-if scenarios:

```bash
python src/simulate.py --data-dir ./data --model ./pickle/model.pkl \
  --budget-json scenarios/meta_plus20.json \
  --out ./output/predictions_meta_plus20.csv

# or CLI multipliers
python src/simulate.py --budget-mult meta=1.2 --budget-mult google=0.9 \
  --out ./output/predictions_reallocate.csv
```

Scenario JSON keys:

- `channel_spend_multipliers`: scale vs run-rate (`meta: 1.2`)
- `channel_daily_spend`: absolute daily $ → period spend = daily × horizon
