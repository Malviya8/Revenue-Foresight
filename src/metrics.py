"""Forecast accuracy and interval metrics for S3 backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE: sum|e| / sum|y|."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom <= 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def interval_coverage(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p10 = np.asarray(p10, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_true >= p10) & (y_true <= p90)))


def run_rate_baseline(frame: pd.DataFrame) -> np.ndarray:
    """Seasonal-naive-ish: planned_spend * hist_roas_28 (winsor-aware features)."""
    spend = frame["planned_spend"].astype(float).fillna(0.0).to_numpy()
    roas = frame["hist_roas_28"].astype(float)
    if "hist_roas_14" in frame.columns:
        roas = roas.fillna(frame["hist_roas_14"].astype(float))
    roas = roas.fillna(0.0).clip(lower=0.0, upper=50.0).to_numpy()
    return spend * roas


def summarize_backtest(
    frame: pd.DataFrame,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, object]:
    y = frame["target_revenue"].astype(float).to_numpy()
    summary: dict[str, object] = {
        "n": int(len(frame)),
        "wmape_model": wmape(y, p50),
        "wmape_baseline": wmape(y, baseline),
        "smape_model": smape(y, p50),
        "coverage_p10_p90": interval_coverage(y, p10, p90),
        "lift_wmape_vs_baseline": None,
    }
    if summary["wmape_baseline"] and summary["wmape_baseline"] == summary["wmape_baseline"]:
        base = float(summary["wmape_baseline"])
        model = float(summary["wmape_model"])
        summary["lift_wmape_vs_baseline"] = (base - model) / base if base > 0 else None

    by_horizon: dict[str, object] = {}
    for h, grp in frame.groupby("horizon_days"):
        idx = grp.index.to_numpy()
        # align with positional arrays via .loc positions
        pos = frame.index.get_indexer(grp.index)
        by_horizon[str(int(h))] = {
            "n": int(len(grp)),
            "wmape_model": wmape(y[pos], p50[pos]),
            "wmape_baseline": wmape(y[pos], baseline[pos]),
            "coverage_p10_p90": interval_coverage(y[pos], p10[pos], p90[pos]),
        }
    summary["by_horizon"] = by_horizon

    by_level: dict[str, object] = {}
    for level, grp in frame.groupby("level"):
        pos = frame.index.get_indexer(grp.index)
        by_level[str(level)] = {
            "n": int(len(grp)),
            "wmape_model": wmape(y[pos], p50[pos]),
            "wmape_baseline": wmape(y[pos], baseline[pos]),
            "coverage_p10_p90": interval_coverage(y[pos], p10[pos], p90[pos]),
        }
    summary["by_level"] = by_level
    return summary
