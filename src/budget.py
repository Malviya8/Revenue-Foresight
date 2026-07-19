"""
Budget scenario application for period forecasts (S4).

Supports channel spend multipliers and/or absolute daily spend targets.
Scenarios are JSON files consumed by generate_features / simulate / predict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from schema import CHANNELS


def load_budget_scenario(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Budget scenario must be a JSON object")
    return raw


def parse_multiplier_cli(values: list[str] | None) -> dict[str, float]:
    """Parse channel=1.2 strings from CLI."""
    out: dict[str, float] = {}
    if not values:
        return out
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected channel=multiplier, got: {item}")
        channel, mult = item.split("=", 1)
        channel = channel.strip().lower()
        out[channel] = float(mult)
    return out


def apply_budget_scenario(features: pd.DataFrame, scenario: dict[str, Any]) -> pd.DataFrame:
    """
    Adjust planned_spend on an inference feature table.

    Recognized keys:
      - channel_spend_multipliers: {"meta": 1.2, "google": 1.0, ...}
      - channel_daily_spend: {"meta": 5000}  # absolute daily $; period = daily * H
      - mode: "multiplier" | "daily" (informational)
    """
    out = features.copy()
    out["planned_spend"] = pd.to_numeric(out["planned_spend"], errors="coerce").fillna(0.0)

    multipliers = scenario.get("channel_spend_multipliers") or {}
    # allow shorthand "channels" when mode is multiplier
    if not multipliers and scenario.get("mode", "multiplier") == "multiplier":
        maybe = scenario.get("channels") or {}
        if maybe and all(isinstance(v, (int, float)) for v in maybe.values()):
            # Heuristic: values near 1.0 look like multipliers; large values = daily
            sample = list(maybe.values())[0]
            if sample <= 5:
                multipliers = maybe

    daily = dict(scenario.get("channel_daily_spend") or {})
    if not daily and scenario.get("mode") == "daily":
        daily = dict(scenario.get("channels") or {})

    # 1) Absolute daily spend: set channel totals, redistribute children by share
    for channel, daily_spend in daily.items():
        channel = str(channel).lower()
        daily_spend = float(daily_spend)
        for horizon, hmask in out.groupby("horizon_days").groups.items():
            horizon = int(horizon)
            target_period = daily_spend * horizon
            idx = out.index[
                (out["channel"] == channel)
                & (out["horizon_days"] == horizon)
                & (out["level"] != "aggregate")
            ]
            if not len(idx):
                continue
            # Set channel row exactly; scale type/campaign by prior share of channel
            ch_idx = out.index[
                (out["channel"] == channel)
                & (out["horizon_days"] == horizon)
                & (out["level"] == "channel")
            ]
            child_idx = out.index[
                (out["channel"] == channel)
                & (out["horizon_days"] == horizon)
                & (out["level"].isin(["campaign_type", "campaign"]))
            ]
            if len(ch_idx):
                out.loc[ch_idx, "planned_spend"] = target_period
            if len(child_idx):
                shares = out.loc[child_idx, "planned_spend"].astype(float)
                total = float(shares.sum())
                if total > 0:
                    out.loc[child_idx, "planned_spend"] = shares / total * target_period
                else:
                    out.loc[child_idx, "planned_spend"] = target_period / max(len(child_idx), 1)

    # 2) Multipliers (applied after daily so you can combine carefully)
    for channel, mult in multipliers.items():
        channel = str(channel).lower()
        mult = float(mult)
        if channel not in CHANNELS and channel != "all":
            # still allow unknown labels
            pass
        mask = out["channel"] == channel
        out.loc[mask, "planned_spend"] = out.loc[mask, "planned_spend"] * mult

    # 3) Rebuild aggregate planned_spend as sum of channels per horizon
    for horizon in out["horizon_days"].unique():
        horizon = int(horizon)
        ch_sum = float(
            out.loc[
                (out["horizon_days"] == horizon) & (out["level"] == "channel"),
                "planned_spend",
            ].sum()
        )
        agg_idx = out.index[
            (out["horizon_days"] == horizon) & (out["level"] == "aggregate")
        ]
        if len(agg_idx) and ch_sum >= 0:
            out.loc[agg_idx, "planned_spend"] = ch_sum

    out["planned_spend"] = out["planned_spend"].clip(lower=0.0)
    if "planned_spend_log" in out.columns:
        out["planned_spend_log"] = np.log1p(out["planned_spend"])
    return out


def scenario_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Channel × horizon planned spend pivot for CLI printing."""
    ch = features[features["level"] == "channel"][
        ["horizon_days", "channel", "planned_spend"]
    ].copy()
    return ch.sort_values(["horizon_days", "channel"]).reset_index(drop=True)
