"""
Hierarchy reconciliation for probabilistic forecasts (S4).

Strategy (marketing-standard):
  1. Within each channel, top-down scale campaign_type -> channel and
     campaign -> campaign_type so children sum to the parent channel.
  2. Rebuild aggregate as the sum of reconciled channels.

This keeps budget simulations coherent: channel spend changes flow up to
ecommerce totals instead of being crushed by a stale aggregate prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from schema import PREDICTION_COLUMNS


REV_COLS = ("p10_revenue", "p50_revenue", "p90_revenue")
SPEND_COL = "assumed_spend"


def _roas_from_revenue(df: pd.DataFrame) -> pd.DataFrame:
    from schema import MIN_P10_ROAS_WHEN_HEALTHY

    out = df.copy()
    spend = out[SPEND_COL].astype(float).to_numpy()
    for q in ("p10", "p50", "p90"):
        rev = out[f"{q}_revenue"].astype(float).to_numpy()
        roas = np.divide(rev, spend, out=np.zeros_like(rev), where=spend > 0)
        # Do not clamp output ROAS to 100 (old ROAS_WINSOR_CAP*2 bug).
        out[f"{q}_roas"] = np.maximum(roas, 0.0)

    # Healthy median → soft-floor P10 so planning bands aren't "losing money"
    p50_roas = out["p50_roas"].to_numpy()
    healthy = p50_roas >= 2.0
    floor_rev = MIN_P10_ROAS_WHEN_HEALTHY * spend
    out.loc[healthy, "p10_revenue"] = np.maximum(
        out.loc[healthy, "p10_revenue"].to_numpy(), floor_rev[healthy]
    )
    # Keep quantile order after floor
    out["p10_revenue"] = np.minimum(out["p10_revenue"], out["p50_revenue"])
    out["p10_roas"] = np.divide(
        out["p10_revenue"].to_numpy(),
        spend,
        out=np.zeros(len(out)),
        where=spend > 0,
    )
    out.loc[healthy, "p10_roas"] = np.maximum(
        out.loc[healthy, "p10_roas"].to_numpy(), MIN_P10_ROAS_WHEN_HEALTHY
    )
    return out


def _scale_group_to_parent(
    children: pd.DataFrame,
    parent_p50: float,
    parent_spend: float,
) -> pd.DataFrame:
    out = children.copy()
    child_sum = float(out["p50_revenue"].sum())
    spend_sum = float(out[SPEND_COL].sum())

    if child_sum > 0 and parent_p50 == parent_p50:
        rev_scale = parent_p50 / child_sum
        for col in REV_COLS:
            out[col] = out[col].astype(float) * rev_scale
    if spend_sum > 0 and parent_spend == parent_spend and parent_spend > 0:
        spend_scale = parent_spend / spend_sum
        out[SPEND_COL] = out[SPEND_COL].astype(float) * spend_scale
    return out


def _sum_to_aggregate(channels: pd.DataFrame, template: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build / overwrite the aggregate row as the sum of channel rows."""
    if channels.empty:
        return template.copy()

    row = {
        "horizon_days": horizon,
        "level": "aggregate",
        "channel": "all",
        "campaign_type": "",
        "campaign_id": "",
        "campaign_name": "",
        SPEND_COL: float(channels[SPEND_COL].sum()),
    }
    for col in REV_COLS:
        row[col] = float(channels[col].sum())

    # Preserve any extra columns from template if present
    if not template.empty:
        for col in template.columns:
            if col not in row:
                row[col] = template.iloc[0][col]
    return pd.DataFrame([row])


def reconcile_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return preds.copy()

    frames: list[pd.DataFrame] = []
    base = preds.copy()
    for col in REV_COLS + (SPEND_COL,):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)

    for horizon, hdf in base.groupby("horizon_days", sort=True):
        horizon = int(horizon)
        agg = hdf[hdf["level"] == "aggregate"].copy()
        channels = hdf[hdf["level"] == "channel"].copy()
        types = hdf[hdf["level"] == "campaign_type"].copy()
        camps = hdf[hdf["level"] == "campaign"].copy()

        # Campaign types -> channel
        type_parts: list[pd.DataFrame] = []
        if not types.empty and not channels.empty:
            for _, ch_row in channels.iterrows():
                ch = str(ch_row["channel"])
                subset = types[types["channel"] == ch]
                if subset.empty:
                    continue
                type_parts.append(
                    _scale_group_to_parent(
                        subset,
                        float(ch_row["p50_revenue"]),
                        float(ch_row[SPEND_COL]),
                    )
                )
            types = pd.concat(type_parts, ignore_index=True) if type_parts else types

        # Campaigns -> campaign_type (fallback channel)
        camp_parts: list[pd.DataFrame] = []
        if not camps.empty:
            if not types.empty:
                for _, t_row in types.iterrows():
                    ch = str(t_row["channel"])
                    ct = str(t_row.get("campaign_type", "") or "")
                    subset = camps[
                        (camps["channel"] == ch)
                        & (camps["campaign_type"].fillna("").astype(str) == ct)
                    ]
                    if subset.empty:
                        continue
                    camp_parts.append(
                        _scale_group_to_parent(
                            subset,
                            float(t_row["p50_revenue"]),
                            float(t_row[SPEND_COL]),
                        )
                    )
            elif not channels.empty:
                for _, ch_row in channels.iterrows():
                    ch = str(ch_row["channel"])
                    subset = camps[camps["channel"] == ch]
                    if subset.empty:
                        continue
                    camp_parts.append(
                        _scale_group_to_parent(
                            subset,
                            float(ch_row["p50_revenue"]),
                            float(ch_row[SPEND_COL]),
                        )
                    )
            camps = pd.concat(camp_parts, ignore_index=True) if camp_parts else camps

        # Aggregate := sum of channels (budget-sim friendly)
        agg = _sum_to_aggregate(channels, agg, horizon)

        piece = pd.concat([agg, channels, types, camps], ignore_index=True)
        frames.append(piece)

    out = pd.concat(frames, ignore_index=True)
    # Enforce revenue order after top-down scaling (guards against numerical drift)
    stacked = np.vstack([out[c].astype(float).to_numpy() for c in REV_COLS])
    ordered = np.sort(stacked, axis=0)
    for i, c in enumerate(REV_COLS):
        out[c] = ordered[i]
    out = _roas_from_revenue(out)
    out = out.sort_values(
        ["horizon_days", "level", "channel", "campaign_type", "campaign_id"]
    ).reset_index(drop=True)
    return out[PREDICTION_COLUMNS]


def reconciliation_report(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, object]:
    report: dict[str, object] = {"horizons": {}, "mode": "channel_up_aggregate"}
    for horizon in sorted(after["horizon_days"].unique()):
        b = before[before["horizon_days"] == horizon]
        a = after[after["horizon_days"] == horizon]
        agg = float(a[a["level"] == "aggregate"]["p50_revenue"].sum())
        ch = float(a[a["level"] == "channel"]["p50_revenue"].sum())
        report["horizons"][str(int(horizon))] = {
            "aggregate_p50": agg,
            "channels_p50_sum": ch,
            "abs_gap_after": abs(agg - ch),
            "aggregate_p50_before": float(
                b[b["level"] == "aggregate"]["p50_revenue"].sum()
            ),
            "channels_p50_sum_before": float(
                b[b["level"] == "channel"]["p50_revenue"].sum()
            ),
        }
    return report
