"""
Hierarchical period feature engineering (S2).

Builds as_of × horizon feature rows for aggregate / channel / campaign_type /
campaign entities. Designed for LightGBM quantile models conditioned on
planned spend (see docs/modeling_decision.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from schema import (
    CHANNELS,
    FEATURE_WINDOWS,
    HIERARCHY_LEVELS,
    HORIZONS_DAYS,
    ROAS_WINSOR_CAP,
)


CHANNEL_CODES = {ch: i for i, ch in enumerate(CHANNELS)}
LEVEL_CODES = {lvl: i for i, lvl in enumerate(HIERARCHY_LEVELS)}


@dataclass(frozen=True)
class EntityKey:
    level: str
    channel: str
    campaign_type: str
    campaign_id: str
    campaign_name: str


def _winsor_roas(revenue: pd.Series, spend: pd.Series, cap: float = ROAS_WINSOR_CAP) -> pd.Series:
    roas = revenue.astype("float64") / spend.astype("float64").replace(0, np.nan)
    return roas.clip(upper=cap)


def _safe_div(a: float, b: float) -> float:
    if b is None or b <= 0 or np.isnan(b):
        return np.nan
    return float(a) / float(b)


def aggregate_hierarchy(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Collapse campaign-day panel to daily series per hierarchy level.

    Returns dict level -> DataFrame with columns:
      date, channel, campaign_type, campaign_id, campaign_name,
      spend, revenue, clicks, impressions, daily_budget, n_campaigns
    """
    base = panel.copy()
    base["date"] = pd.to_datetime(base["date"])

    frames: dict[str, pd.DataFrame] = {}

    # Campaign level (already daily)
    camp = (
        base.groupby(
            ["date", "channel", "campaign_type", "campaign_id", "campaign_name"],
            as_index=False,
        )
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            daily_budget=("daily_budget", "sum"),
        )
    )
    camp["n_campaigns"] = 1
    camp["level"] = "campaign"
    frames["campaign"] = camp

    # Campaign type within channel
    ctype = (
        base.groupby(["date", "channel", "campaign_type"], as_index=False)
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            daily_budget=("daily_budget", "sum"),
            n_campaigns=("campaign_id", "nunique"),
        )
    )
    ctype["campaign_id"] = ""
    ctype["campaign_name"] = ""
    ctype["level"] = "campaign_type"
    frames["campaign_type"] = ctype

    # Channel
    channel = (
        base.groupby(["date", "channel"], as_index=False)
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            daily_budget=("daily_budget", "sum"),
            n_campaigns=("campaign_id", "nunique"),
        )
    )
    channel["campaign_type"] = ""
    channel["campaign_id"] = ""
    channel["campaign_name"] = ""
    channel["level"] = "channel"
    frames["channel"] = channel

    # Aggregate ecommerce (all channels)
    agg = (
        base.groupby(["date"], as_index=False)
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            daily_budget=("daily_budget", "sum"),
            n_campaigns=("campaign_id", "nunique"),
        )
    )
    agg["channel"] = "all"
    agg["campaign_type"] = ""
    agg["campaign_id"] = ""
    agg["campaign_name"] = ""
    agg["level"] = "aggregate"
    frames["aggregate"] = agg

    return frames


def _entity_id_cols(level: str) -> list[str]:
    if level == "aggregate":
        return []
    if level == "channel":
        return ["channel"]
    if level == "campaign_type":
        return ["channel", "campaign_type"]
    return ["channel", "campaign_type", "campaign_id", "campaign_name"]


def _window_slice(hist: pd.DataFrame, as_of: pd.Timestamp, window: int) -> pd.DataFrame:
    start = as_of - pd.Timedelta(days=window - 1)
    return hist[(hist["date"] >= start) & (hist["date"] <= as_of)]


def _features_from_history(
    hist: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int,
    level: str,
    channel: str,
    campaign_type: str,
    campaign_id: str,
    campaign_name: str,
    campaign_type_codes: dict[str, int],
    planned_spend_override: float | None = None,
) -> dict[str, object] | None:
    if hist.empty:
        return None

    hist = hist[hist["date"] <= as_of].sort_values("date")
    if hist.empty:
        return None

    age_days = int((as_of - hist["date"].min()).days) + 1
    active_days_lifetime = int(((hist["spend"] > 0) | (hist["revenue"] > 0)).sum())

    feats: dict[str, object] = {
        "as_of_date": as_of.normalize(),
        "horizon_days": horizon,
        "level": level,
        "channel": channel,
        "campaign_type": campaign_type,
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "age_days": age_days,
        "active_days_lifetime": active_days_lifetime,
        "month": int(as_of.month),
        "weekofyear": int(as_of.isocalendar().week),
        "is_q4": int(as_of.month in (10, 11, 12)),
        "is_holiday_season": int(as_of.month in (11, 12)),
        "channel_code": CHANNEL_CODES.get(channel, -1),
        "campaign_type_code": campaign_type_codes.get(campaign_type, -1),
        "level_code": LEVEL_CODES.get(level, -1),
    }

    for w in FEATURE_WINDOWS:
        sl = _window_slice(hist, as_of, w)
        spend_sum = float(sl["spend"].sum())
        rev_sum = float(sl["revenue"].sum())
        n = max(len(sl), 1)
        roas_w = _winsor_roas(sl["revenue"], sl["spend"])
        feats[f"hist_spend_sum_{w}"] = spend_sum
        feats[f"hist_revenue_sum_{w}"] = rev_sum
        roas_val = _safe_div(rev_sum, spend_sum)
        feats[f"hist_roas_{w}"] = (
            float(min(roas_val, ROAS_WINSOR_CAP)) if roas_val == roas_val else float("nan")
        )
        feats[f"hist_spend_mean_{w}"] = spend_sum / n
        feats[f"hist_revenue_mean_{w}"] = rev_sum / n
        if w == 28:
            feats["hist_spend_std_28"] = float(sl["spend"].std(ddof=0)) if len(sl) else 0.0
            feats["hist_revenue_std_28"] = float(sl["revenue"].std(ddof=0)) if len(sl) else 0.0
            feats["hist_roas_std_28"] = float(roas_w.std(ddof=0)) if len(sl) else 0.0
            budget = sl["daily_budget"].replace(0, np.nan)
            feats["budget_mean_28"] = float(budget.mean()) if budget.notna().any() else np.nan
            fill = sl["spend"] / budget
            feats["budget_fill_mean_28"] = float(fill.replace([np.inf, -np.inf], np.nan).mean())
            active = ((sl["spend"] > 0) | (sl["revenue"] > 0)).mean()
            feats["active_day_rate_28"] = float(active)

    # Trends: recent 7 vs prior 7
    recent = _window_slice(hist, as_of, 7)
    prior_end = as_of - pd.Timedelta(days=7)
    prior = _window_slice(hist, prior_end, 7)
    feats["trend_spend_7"] = float(recent["spend"].mean() - prior["spend"].mean()) if len(prior) else 0.0
    feats["trend_revenue_7"] = (
        float(recent["revenue"].mean() - prior["revenue"].mean()) if len(prior) else 0.0
    )
    r_roas = _safe_div(float(recent["revenue"].sum()), float(recent["spend"].sum()))
    p_roas = _safe_div(float(prior["revenue"].sum()), float(prior["spend"].sum()))
    feats["trend_roas_7"] = (
        (r_roas - p_roas) if (r_roas == r_roas and p_roas == p_roas) else 0.0
    )

    # Planned spend for horizon: override (budget sim) or run-rate * H
    daily_run_rate = float(feats["hist_spend_mean_28"])
    if daily_run_rate <= 0:
        daily_run_rate = float(feats["hist_spend_mean_7"])
    planned = (
        float(planned_spend_override)
        if planned_spend_override is not None
        else daily_run_rate * horizon
    )
    feats["planned_spend"] = max(planned, 0.0)
    feats["planned_spend_log"] = float(np.log1p(feats["planned_spend"]))
    feats["saturation_spend_log"] = float(np.log1p(daily_run_rate))
    feats["saturation_spend_log_sq"] = float(feats["saturation_spend_log"] ** 2)

    # Ensure all window mean aliases exist for 7/28 even if listed twice in schema
    return feats


def _future_targets(hist_full: pd.DataFrame, as_of: pd.Timestamp, horizon: int) -> dict[str, float]:
    future_start = as_of + pd.Timedelta(days=1)
    future_end = as_of + pd.Timedelta(days=horizon)
    fut = hist_full[(hist_full["date"] >= future_start) & (hist_full["date"] <= future_end)]
    spend = float(fut["spend"].sum())
    revenue = float(fut["revenue"].sum())
    return {
        "target_spend": spend,
        "target_revenue": revenue,
        "target_roas": _safe_div(revenue, spend),
    }


def _iter_entities(level_df: pd.DataFrame, level: str) -> list[EntityKey]:
    id_cols = _entity_id_cols(level)
    if not id_cols:
        return [
            EntityKey(
                level=level,
                channel="all",
                campaign_type="",
                campaign_id="",
                campaign_name="",
            )
        ]
    keys = level_df[id_cols].drop_duplicates()
    out: list[EntityKey] = []
    for row in keys.itertuples(index=False):
        vals = dict(zip(id_cols, row))
        out.append(
            EntityKey(
                level=level,
                channel=str(vals.get("channel", "all")),
                campaign_type=str(vals.get("campaign_type", "")),
                campaign_id=str(vals.get("campaign_id", "")),
                campaign_name=str(vals.get("campaign_name", "")),
            )
        )
    return out


def _filter_entity(df: pd.DataFrame, entity: EntityKey) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if entity.level != "aggregate":
        mask &= df["channel"] == entity.channel
    if entity.level in ("campaign_type", "campaign"):
        mask &= df["campaign_type"] == entity.campaign_type
    if entity.level == "campaign":
        mask &= df["campaign_id"] == entity.campaign_id
    return df.loc[mask].copy()


def build_feature_table(
    panel: pd.DataFrame,
    *,
    mode: str = "inference",
    as_of: pd.Timestamp | None = None,
    cutoff_freq_days: int = 14,
    min_history_days: int = 28,
    horizons: tuple[int, ...] = HORIZONS_DAYS,
    levels: tuple[str, ...] = HIERARCHY_LEVELS,
    planned_spend_by_channel: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build feature rows.

    mode='inference': one as_of (default max date), no targets.
    mode='training': rolling cutoffs where future horizon labels exist.
    """
    if mode not in {"inference", "training"}:
        raise ValueError("mode must be 'inference' or 'training'")

    hierarchy = aggregate_hierarchy(panel)
    # Campaign-type code map from full panel
    types = sorted(panel["campaign_type"].dropna().astype(str).unique().tolist())
    campaign_type_codes = {t: i for i, t in enumerate(types)}
    campaign_type_codes[""] = -1

    global_max = pd.to_datetime(panel["date"]).max()
    if as_of is None:
        as_of = pd.Timestamp(global_max).normalize()
    else:
        as_of = pd.Timestamp(as_of).normalize()

    rows: list[dict[str, object]] = []

    for level in levels:
        level_df = hierarchy[level]
        level_df = level_df.copy()
        level_df["date"] = pd.to_datetime(level_df["date"])

        for entity in _iter_entities(level_df, level):
            series = _filter_entity(level_df, entity).sort_values("date")
            if series.empty:
                continue

            series_max = series["date"].max()
            series_min = series["date"].min()

            if mode == "inference":
                cutoffs = [min(as_of, pd.Timestamp(series_max).normalize())]
            else:
                # Need room for max horizon labels
                last_cutoff = pd.Timestamp(series_max).normalize() - pd.Timedelta(
                    days=max(horizons)
                )
                first_cutoff = pd.Timestamp(series_min).normalize() + pd.Timedelta(
                    days=min_history_days
                )
                if last_cutoff < first_cutoff:
                    continue
                cutoffs = list(
                    pd.date_range(first_cutoff, last_cutoff, freq=f"{cutoff_freq_days}D")
                )

            for cutoff in cutoffs:
                hist = series[series["date"] <= cutoff]
                if hist.empty:
                    continue
                if (cutoff - hist["date"].min()).days + 1 < min_history_days:
                    continue

                for horizon in horizons:
                    override = None
                    if planned_spend_by_channel and entity.channel in planned_spend_by_channel:
                        # Interpret override as *total period spend* for that channel entity
                        if entity.level in ("channel", "aggregate"):
                            override = planned_spend_by_channel[entity.channel]

                    feat = _features_from_history(
                        hist,
                        cutoff,
                        horizon,
                        entity.level,
                        entity.channel,
                        entity.campaign_type,
                        entity.campaign_id,
                        entity.campaign_name,
                        campaign_type_codes,
                        planned_spend_override=override,
                    )
                    if feat is None:
                        continue
                    if mode == "training":
                        feat.update(_future_targets(series, cutoff, horizon))
                    rows.append(feat)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Stable column order
    id_front = [
        "as_of_date",
        "horizon_days",
        "level",
        "channel",
        "campaign_type",
        "campaign_id",
        "campaign_name",
    ]
    ordered = id_front + [c for c in out.columns if c not in id_front]
    return out[ordered].reset_index(drop=True)


def build_inference_features(
    panel: pd.DataFrame,
    planned_spend_by_channel: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Features at the latest date for run.sh / demo prediction."""
    return build_feature_table(
        panel,
        mode="inference",
        planned_spend_by_channel=planned_spend_by_channel,
    )


def build_training_features(
    panel: pd.DataFrame,
    cutoff_freq_days: int = 14,
    min_history_days: int = 28,
) -> pd.DataFrame:
    """Labeled feature table for S3 model training."""
    return build_feature_table(
        panel,
        mode="training",
        cutoff_freq_days=cutoff_freq_days,
        min_history_days=min_history_days,
    )
