"""
Clean the unified panel and attach derived metrics for forecasting.

Cleaning policy (documented in docs/data_assumptions.md):
- Aggregate duplicate (channel, campaign_id, date) rows
- Clip negative metrics to 0
- Keep zero-activity rows (calendar continuity)
- Do not invent missing budgets; budget_fill_rate is NaN when budget missing
- Platform attribution revenue fields treated as ground truth
"""

from __future__ import annotations

import pandas as pd

from schema import CLEANED_COLUMNS, ROLLING_WINDOWS, UNIFIED_COLUMNS


def _safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = numer.astype("float64") / denom.astype("float64")
    out = out.where(denom.astype("float64") > 0)
    return out


def deduplicate_campaign_days(panel: pd.DataFrame) -> pd.DataFrame:
    """Sum additive metrics for duplicate campaign-days; keep last name/type/budget."""
    key = ["channel", "campaign_id", "date"]
    if not panel.duplicated(key).any():
        return panel.reset_index(drop=True)

    sorted_panel = panel.sort_values(key).copy()
    agg = (
        sorted_panel.groupby(key, as_index=False)
        .agg(
            campaign_name=("campaign_name", "last"),
            campaign_type=("campaign_type", "last"),
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            conversions=("conversions", "sum"),
            daily_budget=("daily_budget", "last"),
        )
    )
    return agg[UNIFIED_COLUMNS]


def clip_negatives(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for col in ("spend", "revenue", "clicks", "impressions", "conversions", "daily_budget"):
        if col in out.columns:
            out[col] = out[col].clip(lower=0)
    return out


def add_derived_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["roas"] = _safe_divide(out["revenue"], out["spend"])
    out["ctr"] = _safe_divide(out["clicks"], out["impressions"])
    out["cpc"] = _safe_divide(out["spend"], out["clicks"])
    out["budget_fill_rate"] = _safe_divide(out["spend"], out["daily_budget"])
    out["is_active"] = (
        (out["spend"].fillna(0) > 0)
        | (out["revenue"].fillna(0) > 0)
        | (out["clicks"].fillna(0) > 0)
    ).astype(int)

    # Spend share within channel-day (campaign contribution that day)
    channel_day_spend = out.groupby(["channel", "date"])["spend"].transform("sum")
    out["spend_share"] = _safe_divide(out["spend"], channel_day_spend)
    return out


def add_rolling_windows(panel: pd.DataFrame) -> pd.DataFrame:
    """Trailing mean spend/revenue/roas by campaign (requires sorted dates)."""
    out = panel.sort_values(["channel", "campaign_id", "date"]).copy()
    group = out.groupby(["channel", "campaign_id"], sort=False)

    for window in ROLLING_WINDOWS:
        # min_periods=1 so early history isn't all-null
        out[f"spend_roll_{window}"] = group["spend"].transform(
            lambda s, w=window: s.rolling(w, min_periods=1).mean()
        )
        out[f"revenue_roll_{window}"] = group["revenue"].transform(
            lambda s, w=window: s.rolling(w, min_periods=1).mean()
        )
        # Rolling ROAS from rolling sums (more stable than mean of ratios)
        roll_rev = group["revenue"].transform(
            lambda s, w=window: s.rolling(w, min_periods=1).sum()
        )
        roll_spend = group["spend"].transform(
            lambda s, w=window: s.rolling(w, min_periods=1).sum()
        )
        out[f"roas_roll_{window}"] = _safe_divide(roll_rev, roll_spend)

    return out


def clean_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Full S1 clean pipeline: dedupe -> clip -> derive -> rolls."""
    cleaned = deduplicate_campaign_days(panel)
    cleaned = clip_negatives(cleaned)
    # Fill NaN numerics for additive cols after clip (conversions may be NaN on Meta)
    for col in ("spend", "revenue", "clicks", "impressions"):
        cleaned[col] = cleaned[col].fillna(0.0)
    cleaned = add_derived_metrics(cleaned)
    cleaned = add_rolling_windows(cleaned)
    cleaned = cleaned.sort_values(["channel", "campaign_id", "date"]).reset_index(drop=True)

    # Ensure column order; keep any extras if schema drifts
    ordered = [c for c in CLEANED_COLUMNS if c in cleaned.columns]
    extras = [c for c in cleaned.columns if c not in ordered]
    return cleaned[ordered + extras]
