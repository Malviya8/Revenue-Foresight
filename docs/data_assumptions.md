# Data assumptions & cleaning policy (S1)

## Attribution (source of truth)

Platform attribution is used **as provided**. We do **not** build a custom
attribution engine or Media Mix Model.

| Channel | Revenue field (treated as ecommerce revenue) | Spend field |
|---------|-----------------------------------------------|-------------|
| Google  | `metrics_conversions_value`                   | `metrics_cost_micros / 1e6` |
| Meta    | `conversion` (conversion value)               | `spend` |
| Bing    | `Revenue`                                     | `Spend` |

Cross-channel totals are **sums of platform-attributed revenue**, which may
double-count users touched by multiple channels. That is acceptable under the
brief constraint.

## Campaign type

- Google / Bing: platform `campaign_type` / channel type columns.
- Meta: inferred from `campaign_name` keywords (`Prospecting`, `DPA`,
  `Search`, `Brand`, …). Unmatched names → `Other`.

## Cleaning rules

1. Drop rows with unparseable dates.
2. Aggregate duplicate `(channel, campaign_id, date)` rows:
   - Sum: spend, revenue, clicks, impressions, conversions
   - Last: campaign_name, campaign_type, daily_budget
3. Clip negative metrics to 0.
4. Fill NaN spend/revenue/clicks/impressions with 0 after clip.
5. Keep zero-activity rows (calendar continuity / seasonality).
6. Do **not** invent missing `daily_budget` (frequent on Meta).
   `budget_fill_rate` is undefined when budget is missing or ≤ 0.

## Derived fields

- `roas = revenue / spend` (null when spend ≤ 0)
- `ctr = clicks / impressions`
- `cpc = spend / clicks`
- `spend_share` = campaign spend / channel-day spend
- `budget_fill_rate = spend / daily_budget`
- `is_active` = 1 if any of spend, revenue, clicks > 0
- Rolling 7 / 14 / 28 day means for spend & revenue; rolling ROAS from
  rolling sums (more stable than mean of daily ratios)

## Known data quirks observed on sample

- Meta `daily_budget` often null; many zero-activity days in inactive windows.
- Google costs arrive in micros and are converted to currency units.
- Date coverage differs by channel (do not assume identical calendars).
- Holiday / peak months appear in Meta (e.g. Nov–Dec); see `qa_report.json`
  `seasonality.peak_months`.
