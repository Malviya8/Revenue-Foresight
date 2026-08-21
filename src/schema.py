"""
Canonical schema contract for Revenue Foresight.

All channels are normalized into UNIFIED_COLUMNS before feature generation.
Raw platform column names differ; CHANNEL_FILE_PATTERNS + CHANNEL_MAPPINGS
are the single source of truth for ingest.
"""

from __future__ import annotations

from typing import TypedDict


# ---------------------------------------------------------------------------
# Canonical (unified) panel
# ---------------------------------------------------------------------------

UNIFIED_COLUMNS: list[str] = [
    "date",
    "channel",
    "campaign_id",
    "campaign_name",
    "campaign_type",
    "spend",
    "revenue",
    "clicks",
    "impressions",
    "conversions",
    "daily_budget",
]

# Derived / cleaned panel columns added in S1 (after QA + normalize).
DERIVED_COLUMNS: list[str] = [
    "roas",
    "ctr",
    "cpc",
    "spend_share",
    "budget_fill_rate",
    "is_active",
]

# Optional rolling windows (by channel + campaign_id, sorted by date).
ROLLING_WINDOWS: tuple[int, ...] = (7, 14, 28)

CLEANED_COLUMNS: list[str] = (
    UNIFIED_COLUMNS
    + DERIVED_COLUMNS
    + [f"spend_roll_{w}" for w in ROLLING_WINDOWS]
    + [f"revenue_roll_{w}" for w in ROLLING_WINDOWS]
    + [f"roas_roll_{w}" for w in ROLLING_WINDOWS]
)

CHANNELS: tuple[str, ...] = ("google", "meta", "bing")

# Filename patterns under data/ (matched case-insensitively). Prefer patterns
# over hardcoding so held-out test swaps still resolve.
CHANNEL_FILE_PATTERNS: dict[str, tuple[str, ...]] = {
    "google": ("*google*.csv", "*google_ads*.csv"),
    "meta": ("*meta*.csv", "*facebook*.csv"),
    "bing": ("*bing*.csv", "*microsoft*.csv", "*msads*.csv"),
}


class ColumnMapping(TypedDict):
    """Maps a unified field to a raw CSV column (or None if derived)."""

    date: str
    campaign_id: str
    campaign_name: str
    campaign_type: str | None
    spend: str
    revenue: str
    clicks: str
    impressions: str
    conversions: str | None
    daily_budget: str | None
    # Extra transform flags handled in ingest
    spend_micros: bool


CHANNEL_MAPPINGS: dict[str, ColumnMapping] = {
    "google": {
        "date": "segments_date",
        "campaign_id": "campaign_id",
        "campaign_name": "campaign_name",
        "campaign_type": "campaign_advertising_channel_type",
        "spend": "metrics_cost_micros",
        "revenue": "metrics_conversions_value",
        "clicks": "metrics_clicks",
        "impressions": "metrics_impressions",
        "conversions": "metrics_conversions",
        "daily_budget": "campaign_budget_amount",
        "spend_micros": True,
    },
    "meta": {
        "date": "date_start",
        "campaign_id": "campaign_id",
        "campaign_name": "campaign_name",
        "campaign_type": None,  # inferred from campaign_name
        "spend": "spend",
        "revenue": "conversion",  # platform conversion value (attribution as-is)
        "clicks": "clicks",
        "impressions": "impressions",
        "conversions": None,
        "daily_budget": "daily_budget",
        "spend_micros": False,
    },
    "bing": {
        "date": "TimePeriod",
        "campaign_id": "CampaignId",
        "campaign_name": "CampaignName",
        "campaign_type": "CampaignType",
        "spend": "Spend",
        "revenue": "Revenue",
        "clicks": "Clicks",
        "impressions": "Impressions",
        "conversions": "Conversions",
        "daily_budget": "DailyBudget",
        "spend_micros": False,
    },
}


# ---------------------------------------------------------------------------
# Forecast horizons + hierarchy levels (product / brief contract)
# ---------------------------------------------------------------------------

HORIZONS_DAYS: tuple[int, ...] = (30, 60, 90)
HIERARCHY_LEVELS: tuple[str, ...] = (
    "aggregate",
    "channel",
    "campaign_type",
    "campaign",
)
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

# Feature windows + ROAS winsor for *training features only* (not output ROAS).
FEATURE_WINDOWS: tuple[int, ...] = (7, 14, 28, 60)
ROAS_WINSOR_CAP: float = 50.0
# Split-conformal target coverage (1 - alpha). Adjustment is the empirical
# quantile of calibration nonconformity scores — never grid-searched.
CONFORMAL_TARGET_COVERAGE: float = 0.80
# Winsorize nonconformity scores at this percentile before the conformal
# quantile so a single outlier cannot inflate the band (esp. small strata
# where ceil((n+1)*0.80)/n approaches 1.0).
CONFORMAL_SCORE_TRIM_Q: float = 0.95
# Holiday stratum adjustment is also clipped to this multiple of the matching
# non-holiday adjustment. Holiday residual scales can have a heavy right tail
# even after 95th winsorization (the 80% conformal level sits below 95%).
CONFORMAL_HOLIDAY_MAX_RATIO: float = 5.0
# Mondrian strata with fewer than this many score points fall back to parent
# group; holiday strata may augment scores from train (same channel+season).
CONFORMAL_MIN_STRATUM: int = 12
# Chronological split fractions: train / calibration / test.
SPLIT_TRAIN_FRAC: float = 0.60
SPLIT_CALIB_FRAC: float = 0.20
SPLIT_TEST_FRAC: float = 0.20
# Soft floor so aggregate/channel P10 is not "lose money" when history is healthy.
# Applied only when P50 ROAS > 2.0 (see model.predict) — does not change p50.
MIN_P10_ROAS_WHEN_HEALTHY: float = 1.0
# Low-spend ROAS post-process (predict path).
LOW_SPEND_THRESHOLD: float = 50.0
LOW_SPEND_ROAS_CAP: float = 30.0
MIN_P10_ROAS_GLOBAL: float = 0.1

# Columns that identify an inference / training row (not model inputs).
FEATURE_ID_COLUMNS: list[str] = [
    "as_of_date",
    "horizon_days",
    "level",
    "channel",
    "campaign_type",
    "campaign_id",
    "campaign_name",
]

# Numeric / categorical model inputs produced by S2 (extend carefully in S3).
FEATURE_VALUE_COLUMNS: list[str] = [
    "horizon_days",
    "planned_spend",
    "planned_spend_log",
    "hist_spend_sum_7",
    "hist_spend_sum_14",
    "hist_spend_sum_28",
    "hist_spend_sum_60",
    "hist_revenue_sum_7",
    "hist_revenue_sum_14",
    "hist_revenue_sum_28",
    "hist_revenue_sum_60",
    "hist_roas_7",
    "hist_roas_14",
    "hist_roas_28",
    "hist_roas_60",
    "hist_spend_mean_7",
    "hist_spend_mean_28",
    "hist_revenue_mean_7",
    "hist_revenue_mean_28",
    "hist_spend_std_28",
    "hist_revenue_std_28",
    "hist_roas_std_28",
    "trend_spend_7",
    "trend_revenue_7",
    "trend_roas_7",
    "budget_mean_28",
    "budget_fill_mean_28",
    "active_day_rate_28",
    "age_days",
    "active_days_lifetime",
    "month",
    "weekofyear",
    "is_q4",
    "is_holiday_season",
    "channel_code",
    "campaign_type_code",
    "level_code",
    "saturation_spend_log",
    "saturation_spend_log_sq",
]

TARGET_COLUMNS: list[str] = [
    "target_revenue",
    "target_spend",
    "target_roas",
]


# ---------------------------------------------------------------------------
# Scored output schema (see docs/output_contract.md).
# ---------------------------------------------------------------------------

PREDICTION_COLUMNS: list[str] = [
    "horizon_days",
    "level",
    "channel",
    "campaign_type",
    "campaign_id",
    "campaign_name",
    "assumed_spend",
    "p10_revenue",
    "p50_revenue",
    "p90_revenue",
    "p10_roas",
    "p50_roas",
    "p90_roas",
]


# Keywords used to infer Meta (and sparse) campaign types from names
CAMPAIGN_TYPE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("prospecting", "Prospecting"),
    ("dpa", "DPA"),
    ("retarget", "Retargeting"),
    ("remarket", "Remarketing"),
    ("brand", "Brand"),
    ("search_tm", "Search_TM"),
    ("search", "Search"),
    ("shopping", "Shopping"),
    ("pmax", "Performance_Max"),
    ("performance_max", "Performance_Max"),
    ("display", "Display"),
    ("video", "Video"),
    ("generic", "Generic"),
)
DEFAULT_CAMPAIGN_TYPE = "Other"
