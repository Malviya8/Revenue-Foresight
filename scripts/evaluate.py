"""
AIgnition Forecasting — split-conformal evaluation on the held-out TEST set.

Chronological 60/20/20 split (same as train.py):
  train / calib used only if rebuilding; scoring is always on TEST (final 20%).

Prefer loading pickle/model.pkl (train+calib baked in) and predicting on TEST.

Usage (from repo root):
  python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features import build_training_features  # noqa: E402
from ingest import load_cleaned_panel  # noqa: E402
from metrics import mase, run_rate_baseline, smape, wmape  # noqa: E402
from model import (  # noqa: E402
    RevenueQuantileModel,
    chronological_three_way_split,
)
from schema import FEATURE_VALUE_COLUMNS  # noqa: E402

SEED = 42
TARGET_COVERAGE = 0.80
COVERAGE_PASS_AT = 0.70
LIFT_KEEP_ABOVE = 0.50


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) == 0:
        return float("nan")
    err = y - pred
    loss = np.where(err >= 0, q * err, (1.0 - q) * (-err))
    return float(np.mean(loss))


def winkler_score(y: np.ndarray, p10: np.ndarray, p90: np.ndarray, alpha: float = 0.2) -> float:
    y = np.asarray(y, dtype=float)
    p10 = np.asarray(p10, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    if len(y) == 0:
        return float("nan")
    width = p90 - p10
    below = np.maximum(0.0, p10 - y)
    above = np.maximum(0.0, y - p90)
    return float(np.mean(width + (2.0 / alpha) * below + (2.0 / alpha) * above))


def interval_coverage(y: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p10 = np.asarray(p10, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((y >= p10) & (y <= p90)))


def mape_pct(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.abs(y) > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y[mask] - pred[mask]) / np.abs(y[mask])) * 100.0)


def wmape_pct(y: np.ndarray, pred: np.ndarray) -> float:
    """Volume-weighted MAPE as a percentage (100 * sum|e| / sum|y|)."""
    w = wmape(y, pred)
    if w != w:
        return float("nan")
    return float(w * 100.0)


def spread_ratio(p10: np.ndarray, p90: np.ndarray) -> float:
    """Honest raw mean p90/p10 (no winsorization)."""
    p10 = np.asarray(p10, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    mask = p10 > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(p90[mask] / p10[mask]))


def naive_last30_baseline(frame: pd.DataFrame) -> np.ndarray:
    horizon = frame["horizon_days"].astype(float).to_numpy()
    if "hist_revenue_mean_28" in frame.columns:
        daily = frame["hist_revenue_mean_28"].astype(float).fillna(0.0).to_numpy()
    elif "hist_revenue_sum_28" in frame.columns:
        daily = frame["hist_revenue_sum_28"].astype(float).fillna(0.0).to_numpy() / 28.0
    else:
        return np.full(len(frame), np.nan)
    return np.maximum(daily, 0.0) * horizon


def lift_vs_baseline(y: np.ndarray, model_pred: np.ndarray, baseline_pred: np.ndarray) -> float:
    base = wmape(y, baseline_pred)
    model = wmape(y, model_pred)
    if base != base or base <= 0:
        return float("nan")
    return float((base - model) / base)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"])
    out["target_revenue"] = out["target_revenue"].astype(float).clip(lower=0.0)
    for col in FEATURE_VALUE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _channel_mask(frame: pd.DataFrame, channel: str) -> np.ndarray:
    ch = frame["channel"].astype(str).str.lower()
    lvl = frame["level"].astype(str)
    is_ch = (ch == channel) & (lvl == "channel")
    if is_ch.any():
        return is_ch.to_numpy()
    return ((ch == channel) & (lvl != "aggregate")).to_numpy()


def load_test_predictions(
    data_dir: Path,
    *,
    model_path: Path,
    features_cache: Path | None,
    cutoff_freq_days: int,
    num_boost_round: int,
    seed: int,
    rebuild: bool,
) -> dict[str, Any]:
    """
    Score the chronological TEST split (final 20% of dates).

    Default: load pickle/model.pkl (already calibrated on calib) and predict TEST.
    ``rebuild=True`` refits train+calib from scratch (does not touch test during fit).
    """
    if features_cache is not None and features_cache.exists():
        print(f"Loading cached training features: {features_cache}")
        frame = (
            pd.read_parquet(features_cache)
            if features_cache.suffix.lower() == ".parquet"
            else pd.read_csv(features_cache)
        )
        frame = _prepare_frame(frame)
    else:
        print(f"Loading cleaned panel from {data_dir} ...")
        panel, _ = load_cleaned_panel(str(data_dir))
        if panel is None or panel.empty:
            return {"ok": False, "reason": "insufficient data: empty panel after ingest"}
        print("Building training features...")
        frame = build_training_features(panel, cutoff_freq_days=cutoff_freq_days)
        if frame.empty:
            return {"ok": False, "reason": "insufficient data: no labeled training rows"}
        frame = _prepare_frame(frame)

    train_df, calib_df, test_df = chronological_three_way_split(frame)
    if test_df.empty:
        return {"ok": False, "reason": "insufficient data: test split is empty"}

    print(
        f"  chronological 60/20/20:\n"
        f"    train n={len(train_df)} "
        f"({train_df['as_of_date'].min().date()}->{train_df['as_of_date'].max().date()})\n"
        f"    calib n={len(calib_df)} "
        f"({calib_df['as_of_date'].min().date()}->{calib_df['as_of_date'].max().date()})\n"
        f"    TEST  n={len(test_df)} "
        f"({test_df['as_of_date'].min().date()}->{test_df['as_of_date'].max().date()}) "
        f"<-- metrics use this only"
    )

    if rebuild or not model_path.is_file():
        print("Rebuilding model on train + calibrating on calib (test untouched)...")
        model = RevenueQuantileModel(feature_columns=list(FEATURE_VALUE_COLUMNS))
        model.fit(train_df, valid=None, num_boost_round=num_boost_round, random_state=seed)
        model.calibrate_conformal(calib_df, train=train_df)
    else:
        print(f"Loading pickled model: {model_path}")
        with model_path.open("rb") as fh:
            model = pickle.load(fh)
        if not getattr(model, "conformal_adjustments", None):
            print("  pickle missing conformal_adjustments — calibrating on calib...")
            model.calibrate_conformal(calib_df, train=train_df)

    p10, p50, p90 = model.predict_quantiles(test_df)
    test_df = test_df.reset_index(drop=True)
    y = test_df["target_revenue"].astype(float).to_numpy()

    return {
        "ok": True,
        "frame": test_df,
        "y": y,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "n_train": int(len(train_df)),
        "n_calib": int(len(calib_df)),
        "n_test": int(len(test_df)),
        "train_start": str(train_df["as_of_date"].min().date()),
        "train_end": str(train_df["as_of_date"].max().date()),
        "calib_start": str(calib_df["as_of_date"].min().date()),
        "calib_end": str(calib_df["as_of_date"].max().date()),
        "test_start": str(test_df["as_of_date"].min().date()),
        "test_end": str(test_df["as_of_date"].max().date()),
        "conformal_adjustments": dict(getattr(model, "conformal_adjustments", {})),
        "conformal_stratum_counts": dict(getattr(model, "conformal_stratum_counts", {})),
    }


def evaluate_prophet_channel_baseline(
    panel: pd.DataFrame,
    holdout: dict[str, Any],
) -> dict[str, Any]:
    """
    Fair channel-level Prophet benchmark on the same chronological TEST rows.

    Prophet is refit for each TEST ``as_of_date`` using only observations
    available up to that date. Future daily spend is the row's planned period
    spend divided by its horizon. Google and Meta are benchmarked; Bing remains
    excluded because its revenue target is mostly zero.
    """
    try:
        from prophet import Prophet
    except ImportError:
        return {
            "available": False,
            "reason": "Prophet is not installed; run: pip install -r requirements-eval.txt",
        }

    if not holdout.get("ok") or panel is None or panel.empty:
        return {"available": False, "reason": "holdout or cleaned panel is unavailable"}

    frame = holdout["frame"].reset_index(drop=True).copy()
    p50 = np.asarray(holdout["p50"], dtype=float)
    eligible = (
        frame["level"].astype(str).eq("channel")
        & frame["channel"].astype(str).str.lower().isin(("google", "meta"))
    )
    positions = np.flatnonzero(eligible.to_numpy())
    if len(positions) == 0:
        return {"available": False, "reason": "no Google/Meta channel TEST rows"}

    daily = panel.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["channel"] = daily["channel"].astype(str).str.lower()
    daily = (
        daily[daily["channel"].isin(("google", "meta"))]
        .groupby(["channel", "date"], as_index=False)[["spend", "revenue"]]
        .sum()
        .sort_values(["channel", "date"])
    )

    prophet_pred = np.full(len(frame), np.nan, dtype=float)
    failures: list[str] = []

    for (channel, as_of), rows in frame.loc[eligible].groupby(
        [frame.loc[eligible, "channel"].astype(str).str.lower(), "as_of_date"]
    ):
        as_of = pd.Timestamp(as_of).normalize()
        hist = daily[(daily["channel"] == channel) & (daily["date"] <= as_of)].copy()
        if len(hist) < 60 or hist["revenue"].sum() <= 0:
            failures.append(f"{channel}@{as_of.date()}: insufficient history")
            continue

        train = hist.rename(columns={"date": "ds", "revenue": "y"})[["ds", "y", "spend"]]
        max_horizon = int(rows["horizon_days"].max())
        # All horizons are generated from the same recent run-rate; averaging
        # planned_spend / horizon handles harmless floating-point differences.
        daily_spend = float(
            np.mean(
                rows["planned_spend"].astype(float).to_numpy()
                / rows["horizon_days"].astype(float).to_numpy()
            )
        )
        future = pd.DataFrame(
            {
                "ds": pd.date_range(as_of + pd.Timedelta(days=1), periods=max_horizon),
                "spend": max(daily_spend, 0.0),
            }
        )

        try:
            model = Prophet(
                interval_width=0.80,
                weekly_seasonality=True,
                yearly_seasonality="auto",
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05,
            )
            model.add_regressor("spend", prior_scale=5.0, standardize=True)
            model.fit(train)
            daily_pred = np.maximum(
                model.predict(future)["yhat"].to_numpy(dtype=float), 0.0
            )
            cumulative = np.cumsum(daily_pred)
            for idx, row in rows.iterrows():
                horizon = int(row["horizon_days"])
                prophet_pred[int(idx)] = float(cumulative[horizon - 1])
        except Exception as exc:  # benchmark must not break primary evaluation
            failures.append(f"{channel}@{as_of.date()}: {type(exc).__name__}: {exc}")

    scored = positions[np.isfinite(prophet_pred[positions])]
    if len(scored) == 0:
        return {
            "available": False,
            "reason": "Prophet produced no benchmark predictions",
            "failures": failures,
        }

    y = frame["target_revenue"].astype(float).to_numpy()
    runrate = run_rate_baseline(frame)
    result: dict[str, Any] = {
        "available": True,
        "scope": "Google/Meta channel-level TEST rows only",
        "protocol": "walk-forward refit at each as_of_date; no TEST target leakage",
        "n": int(len(scored)),
        "wmape_lightgbm": wmape(y[scored], p50[scored]),
        "wmape_prophet": wmape(y[scored], prophet_pred[scored]),
        "wmape_runrate": wmape(y[scored], runrate[scored]),
        "prophet_lift_vs_runrate": lift_vs_baseline(
            y[scored], prophet_pred[scored], runrate[scored]
        ),
        "lightgbm_lift_vs_prophet": lift_vs_baseline(
            y[scored], p50[scored], prophet_pred[scored]
        ),
        "per_channel": {},
        "failures": failures,
    }
    for channel in ("google", "meta"):
        mask = (
            frame.iloc[scored]["channel"].astype(str).str.lower().eq(channel).to_numpy()
        )
        channel_pos = scored[mask]
        if len(channel_pos):
            result["per_channel"][channel] = {
                "n": int(len(channel_pos)),
                "wmape_lightgbm": wmape(y[channel_pos], p50[channel_pos]),
                "wmape_prophet": wmape(y[channel_pos], prophet_pred[channel_pos]),
                "wmape_runrate": wmape(y[channel_pos], runrate[channel_pos]),
            }
    return result


def compute_metrics(holdout: dict[str, Any], predictions_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "seed": SEED,
        "method": "mondrian_split_conformal_test_set",
        "insufficient_data": False,
        "notes": [],
    }

    spread_from_csv: float | None = None
    if predictions_path.exists():
        preds = pd.read_csv(predictions_path)
        if {"p10_revenue", "p90_revenue"}.issubset(preds.columns):
            spread_from_csv = spread_ratio(
                preds["p10_revenue"].to_numpy(), preds["p90_revenue"].to_numpy()
            )
            out["predictions_csv"] = str(predictions_path.resolve())
            out["predictions_csv_rows"] = int(len(preds))

    if not holdout.get("ok"):
        out["insufficient_data"] = True
        out["notes"].append(holdout.get("reason", "insufficient data"))
        out["overall"] = {}
        out["per_channel"] = {}
        out["pass_fail"] = {}
        return out

    y = holdout["y"]
    p10, p50, p90 = holdout["p10"], holdout["p50"], holdout["p90"]
    frame = holdout["frame"]

    naive = naive_last30_baseline(frame)
    runrate = run_rate_baseline(frame)
    coverage = interval_coverage(y, p10, p90)

    ch_col = frame["channel"].astype(str).str.lower()
    non_bing = ch_col.ne("bing").to_numpy()
    bing_mask = ch_col.eq("bing").to_numpy()

    # Headline point accuracy: wMAPE excl. Bing (volume-weighted).
    # Simple MAPE is reported as a secondary diagnostic only.
    y_nb, p50_nb = y[non_bing], p50[non_bing]
    naive_nb = naive[non_bing] if non_bing.any() else naive
    runrate_nb = runrate[non_bing] if non_bing.any() else runrate

    wmape_excl = wmape(y_nb, p50_nb) if non_bing.any() else float("nan")
    wmape_excl_pct = wmape_pct(y_nb, p50_nb) if non_bing.any() else float("nan")
    mape = mape_pct(y_nb, p50_nb) if non_bing.any() else float("nan")
    mape_bing = mape_pct(y[bing_mask], p50[bing_mask]) if bing_mask.any() else float("nan")
    # Raw MAPE on Bing is misleading when many rows have near-zero actual revenue
    # (denominator → 0 inflates % error). Prefer sMAPE / MASE for Bing reporting.
    smape_bing = smape(y[bing_mask], p50[bing_mask]) if bing_mask.any() else float("nan")
    mase_bing = (
        mase(y[bing_mask], p50[bing_mask], naive[bing_mask]) if bing_mask.any() else float("nan")
    )

    spread = spread_ratio(p10, p90)  # raw, no winsor
    # Lifts on the same excl-Bing slice used for the headline wMAPE.
    lift_naive = (
        lift_vs_baseline(y_nb, p50_nb, naive_nb) if non_bing.any() else float("nan")
    )
    lift_runrate = (
        lift_vs_baseline(y_nb, p50_nb, runrate_nb) if non_bing.any() else float("nan")
    )

    overall = {
        "n": int(len(y)),
        "n_headline_excl_bing": int(non_bing.sum()),
        "n_mape_excl_bing": int(non_bing.sum()),
        "interval_coverage": coverage,
        "interval_coverage_pct": None if coverage != coverage else coverage * 100.0,
        "pinball_p10": pinball_loss(y, p10, 0.1),
        "pinball_p50": pinball_loss(y, p50, 0.5),
        "pinball_p90": pinball_loss(y, p90, 0.9),
        "winkler_score": winkler_score(y, p10, p90, alpha=0.2),
        # Headline point metric (volume-weighted); Bing excluded (data sparsity).
        "wmape_p50": wmape_excl,
        "wmape_p50_pct": wmape_excl_pct,
        "wmape_p50_excl_bing": wmape_excl,
        "wmape_p50_pct_excl_bing": wmape_excl_pct,
        # Secondary unweighted MAPE (equal weight per row — tiny campaigns dominate).
        "mape_p50_pct": mape,
        "mape_p50_pct_excl_bing": mape,
        "mape_p50_pct_bing_only": mape_bing,
        "smape_p50_bing_only": smape_bing,
        "smape_p50_pct_bing_only": None if smape_bing != smape_bing else smape_bing * 100.0,
        "mase_p50_bing_only": mase_bing,
        "spread_ratio": spread,
        "spread_ratio_raw_mean": spread,
        "spread_ratio_predictions_csv": spread_from_csv,
        "lift_vs_naive": lift_naive,
        "lift_vs_naive_pct": None if lift_naive != lift_naive else lift_naive * 100.0,
        "lift_vs_runrate": lift_runrate,
        "lift_vs_runrate_pct": None if lift_runrate != lift_runrate else lift_runrate * 100.0,
        "wmape_model": wmape_excl,  # headline alias (excl Bing)
        "wmape_model_all_channels": wmape(y, p50),
        "wmape_naive": wmape(y_nb, naive_nb) if non_bing.any() else float("nan"),
        "wmape_runrate": wmape(y_nb, runrate_nb) if non_bing.any() else float("nan"),
    }

    per_channel: dict[str, Any] = {}
    for ch in ("google", "meta", "bing"):
        mask = _channel_mask(frame, ch)
        if not mask.any():
            per_channel[ch] = {
                "n": 0,
                "wmape_p50": None,
                "wmape_p50_pct": None,
                "mape_p50_pct": None,
                "pinball_p50": None,
                "coverage_pct": None,
                "note": "insufficient data",
            }
            continue
        yc, p10c, p50c, p90c = y[mask], p10[mask], p50[mask], p90[mask]
        cov = interval_coverage(yc, p10c, p90c)
        w = wmape(yc, p50c)
        ch_stats: dict[str, Any] = {
            "n": int(mask.sum()),
            "wmape_p50": w,
            "wmape_p50_pct": None if w != w else w * 100.0,
            "mape_p50_pct": mape_pct(yc, p50c),
            "pinball_p50": pinball_loss(yc, p50c, 0.5),
            "coverage": cov,
            "coverage_pct": None if cov != cov else cov * 100.0,
        }
        if ch == "bing":
            naive_c = naive[mask]
            ch_stats["smape_p50_pct"] = (
                None if smape(yc, p50c) != smape(yc, p50c) else smape(yc, p50c) * 100.0
            )
            ch_stats["mase_p50"] = mase(yc, p50c, naive_c)
            ch_stats["mape_note"] = (
                "Bing excluded from headline wMAPE; raw MAPE inflated by near-zero "
                "revenue — use sMAPE/MASE; model uses run-rate fallback"
            )
            ch_stats["excluded_from_headline"] = True
        per_channel[ch] = ch_stats

    cov_pct = overall["interval_coverage_pct"]
    lift_pct = overall["lift_vs_naive_pct"]

    def _ok(flag: bool | None) -> str:
        if flag is None:
            return "insufficient data"
        return "pass" if flag else "fail"

    pass_fail = {
        "interval_coverage": _ok(
            None if cov_pct is None else cov_pct >= COVERAGE_PASS_AT * 100.0
        ),
        "pinball_loss": _ok(
            all(overall[k] == overall[k] for k in ("pinball_p10", "pinball_p50", "pinball_p90"))
        ),
        "winkler_score": _ok(overall["winkler_score"] == overall["winkler_score"]),
        "wmape_p50": _ok(overall["wmape_p50"] == overall["wmape_p50"]),
        "mape_p50": _ok(overall["mape_p50_pct"] == overall["mape_p50_pct"]),
        "spread_ratio": _ok(overall["spread_ratio"] == overall["spread_ratio"]),
        "lift_vs_naive": _ok(
            None if lift_pct is None else lift_pct >= LIFT_KEEP_ABOVE * 100.0
        ),
        "lift_vs_runrate_verified": _ok(
            None
            if overall["lift_vs_runrate_pct"] is None
            else overall["lift_vs_runrate_pct"] >= LIFT_KEEP_ABOVE * 100.0
        ),
    }

    out["overall"] = overall
    out["per_channel"] = per_channel
    out["pass_fail"] = pass_fail
    out["holdout_meta"] = {
        "n_train": holdout["n_train"],
        "n_calib": holdout["n_calib"],
        "n_test": holdout["n_test"],
        "train_start": holdout["train_start"],
        "train_end": holdout["train_end"],
        "calib_start": holdout["calib_start"],
        "calib_end": holdout["calib_end"],
        "test_start": holdout["test_start"],
        "test_end": holdout["test_end"],
        "conformal_adjustments": holdout.get("conformal_adjustments", {}),
        "conformal_stratum_counts": holdout.get("conformal_stratum_counts", {}),
        "eval_split": "test",
    }
    return out


def _fmt(v: Any, kind: str = "float") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "insufficient data"
    if kind == "pct":
        return f"{float(v):.1f}%"
    if kind == "spread":
        return f"{float(v):.2f}x"
    if kind == "lift":
        return f"{float(v):+.1f}%"
    return f"{float(v):.1f}"


def _mark(status: str) -> str:
    if status == "pass":
        return "✅"
    if status == "fail":
        return "❌"
    return "—"


def strongest_weakest(per_channel: dict[str, Any]) -> tuple[str, str]:
    scores: list[tuple[str, float]] = []
    for ch, stats in per_channel.items():
        if ch == "bing" or stats.get("excluded_from_headline"):
            continue
        m = stats.get("wmape_p50")
        if m is None or m != m:
            m = stats.get("mape_p50_pct")
            if m is not None and m == m:
                m = float(m) / 100.0
        if m is not None and m == m:
            scores.append((ch, float(m)))
    if not scores:
        return "insufficient data", "insufficient data"
    scores.sort(key=lambda t: t[1])
    return scores[0][0], scores[-1][0]


def format_report(metrics: dict[str, Any]) -> str:
    o = metrics.get("overall", {})
    pf = metrics.get("pass_fail", {})
    ch = metrics.get("per_channel", {})

    lines: list[str] = []
    lines.append("═══════════════════════════════════════")
    lines.append(" AIGNITION FORECAST EVALUATION REPORT")
    lines.append("═══════════════════════════════════════")
    lines.append("")

    if metrics.get("insufficient_data"):
        lines.append("STATUS: insufficient data for backtest")
        for n in metrics.get("notes", []):
            lines.append(f"  - {n}")
        lines.append("")

    meta = metrics.get("holdout_meta")
    if meta:
        lines.append(
            f"TEST set: {meta['test_start']} → {meta['test_end']} "
            f"(n={meta['n_test']}; train n={meta['n_train']}; calib n={meta['n_calib']})"
        )
        lines.append(
            f"Seed: {SEED}  |  Method: Mondrian split-conformal (TEST only)"
        )
        adj = meta.get("conformal_adjustments") or {}
        if adj:
            lines.append(f"Conformal adjustments: {adj}")
        strata = meta.get("conformal_stratum_counts") or {}
        if strata:
            lines.append(f"Conformal stratum counts: {strata}")
        lines.append("")

    lines.append("OVERALL METRICS")
    lines.append("───────────────")
    cov = o.get("interval_coverage_pct")
    lines.append(
        f"Interval Coverage (p10-p90) : {_fmt(cov, 'pct'):<18} "
        f"[target: 80%, pass≥70%]  {_mark(pf.get('interval_coverage', ''))}"
    )
    lines.append(f"Pinball Loss  p10            : {_fmt(o.get('pinball_p10'))}")
    lines.append(f"Pinball Loss  p50            : {_fmt(o.get('pinball_p50'))}")
    lines.append(f"Pinball Loss  p90            : {_fmt(o.get('pinball_p90'))}")
    lines.append(
        f"Winkler Score                : {_fmt(o.get('winkler_score')):<18} [lower=better]"
    )
    lines.append(
        f"wMAPE (p50, excl. Bing)      : {_fmt(o.get('wmape_p50_pct'), 'pct'):<18} "
        f"[HEADLINE; volume-weighted]  {_mark(pf.get('wmape_p50', ''))}"
    )
    lines.append(
        f"MAPE  (p50, excl. Bing)      : {_fmt(o.get('mape_p50_pct'), 'pct'):<18} "
        f"[secondary; equal-weight rows]"
    )
    lines.append(
        f"MAPE (Bing only, sparse)     : {_fmt(o.get('mape_p50_pct_bing_only'), 'pct'):<18} "
        f"[unreliable — see sMAPE/MASE]"
    )
    lines.append(
        f"sMAPE (Bing only)            : {_fmt(o.get('smape_p50_pct_bing_only'), 'pct'):<18} "
        f"[symmetric, near-zero safe]"
    )
    lines.append(
        f"MASE (Bing vs naive)         : {_fmt(o.get('mase_p50_bing_only')):<18} "
        f"[<1 beats baseline]"
    )
    lines.append(
        f"Spread Ratio (raw mean)      : {_fmt(o.get('spread_ratio'), 'spread'):<18} "
        f"[no winsorization]"
    )
    lines.append(
        f"Lift vs Naive Baseline       : {_fmt(o.get('lift_vs_naive_pct'), 'lift'):<18} "
        f"[target: >50%]  {_mark(pf.get('lift_vs_naive', ''))}"
    )
    lines.append(
        f"Lift vs Run-rate Baseline    : {_fmt(o.get('lift_vs_runrate_pct'), 'lift'):<18} "
        f"[target: >50%] {_mark(pf.get('lift_vs_runrate_verified', ''))}"
    )
    lines.append("")

    lines.append("PER-CHANNEL BREAKDOWN")
    lines.append("─────────────────────")
    lines.append(f"{'Channel':<10}{'wMAPE':<10}{'MAPE':<10}{'Pinball(p50)':<16}{'Coverage'}")
    for name in ("google", "meta", "bing"):
        s = ch.get(name, {})
        if s.get("n", 0) == 0 or s.get("note") == "insufficient data":
            lines.append(f"{name:<10}{'—':<10}{'—':<10}{'—':<16}insufficient data")
            continue
        extra = ""
        if name == "bing":
            extra = (
                f"  sMAPE={_fmt(s.get('smape_p50_pct'), 'pct')} "
                f"MASE={_fmt(s.get('mase_p50'))}  [excluded from headline]"
            )
        lines.append(
            f"{name:<10}"
            f"{_fmt(s.get('wmape_p50_pct'), 'pct'):<10}"
            f"{_fmt(s.get('mape_p50_pct'), 'pct'):<10}"
            f"{_fmt(s.get('pinball_p50')):<16}"
            f"{_fmt(s.get('coverage_pct'), 'pct')} (n={s.get('n', 0)}){extra}"
        )
    lines.append("")

    prophet = metrics.get("benchmarks", {}).get("prophet_channel", {})
    if prophet:
        lines.append("CHANNEL-LEVEL BASELINE BENCHMARK")
        lines.append("────────────────────────────────")
        if prophet.get("available"):
            lines.append(
                "Same Google/Meta channel TEST rows "
                f"(n={prophet.get('n', 0)}; walk-forward, no target leakage)"
            )
            lines.append(
                f"LightGBM wMAPE : {_fmt(prophet.get('wmape_lightgbm', float('nan')) * 100, 'pct')}"
            )
            lines.append(
                f"Prophet wMAPE  : {_fmt(prophet.get('wmape_prophet', float('nan')) * 100, 'pct')}"
            )
            lines.append(
                f"Run-rate wMAPE : {_fmt(prophet.get('wmape_runrate', float('nan')) * 100, 'pct')}"
            )
            lines.append(
                "LightGBM lift vs Prophet: "
                f"{_fmt(prophet.get('lightgbm_lift_vs_prophet', float('nan')) * 100, 'lift')}"
            )
            for name, stats in prophet.get("per_channel", {}).items():
                lines.append(
                    f"  {name:<8} LightGBM={_fmt(stats['wmape_lightgbm'] * 100, 'pct')} "
                    f"Prophet={_fmt(stats['wmape_prophet'] * 100, 'pct')} "
                    f"Run-rate={_fmt(stats['wmape_runrate'] * 100, 'pct')}"
                )
        else:
            lines.append(f"Unavailable: {prophet.get('reason', 'unknown reason')}")
        lines.append("")

    primary_keys = [
        "interval_coverage",
        "pinball_loss",
        "winkler_score",
        "wmape_p50",
        "spread_ratio",
        "lift_vs_naive",
        "lift_vs_runrate_verified",
    ]
    n_pass = sum(1 for k in primary_keys if pf.get(k) == "pass")
    strong, weak = strongest_weakest(ch)

    lines.append("SUMMARY")
    lines.append("───────")
    lines.append(f"{n_pass}/7 metrics passing target thresholds")
    lines.append(f"Strongest channel : {strong}")
    lines.append(f"Weakest channel   : {weak}")
    lines.append("═══════════════════════════════════════")
    return "\n".join(lines)


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if v != v or np.isinf(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Split-conformal evaluation on TEST set")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model", default="./pickle/model.pkl")
    parser.add_argument("--predictions", default="./output/predictions.csv")
    parser.add_argument("--report-out", default="./output/evaluation_report.txt")
    parser.add_argument("--metrics-out", default="./output/evaluation_metrics.json")
    parser.add_argument("--features-cache", default="./output/features_train.parquet")
    parser.add_argument("--cutoff-freq-days", type=int, default=14)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--with-prophet",
        action="store_true",
        help=(
            "Run optional walk-forward Prophet benchmark on the same Google/Meta "
            "channel-level TEST rows (requires requirements-eval.txt)"
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Refit train+calib instead of loading pickle (test still untouched)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    cache = None if args.no_cache else Path(args.features_cache)
    model_path = Path(args.model)

    if not data_dir.is_dir() and (cache is None or not cache.exists()):
        metrics = {
            "insufficient_data": True,
            "notes": ["insufficient data: no data dir and no features cache"],
            "overall": {},
            "per_channel": {},
            "pass_fail": {},
            "seed": args.seed,
        }
    else:
        holdout = load_test_predictions(
            data_dir,
            model_path=model_path,
            features_cache=cache,
            cutoff_freq_days=args.cutoff_freq_days,
            num_boost_round=args.num_boost_round,
            seed=args.seed,
            rebuild=args.rebuild,
        )
        metrics = compute_metrics(holdout, Path(args.predictions))
        if args.with_prophet:
            if not data_dir.is_dir():
                metrics.setdefault("benchmarks", {})["prophet_channel"] = {
                    "available": False,
                    "reason": "Prophet benchmark requires --data-dir",
                }
            else:
                panel, _ = load_cleaned_panel(str(data_dir))
                metrics.setdefault("benchmarks", {})["prophet_channel"] = (
                    evaluate_prophet_channel_baseline(panel, holdout)
                )

    report = format_report(metrics)
    _safe_print(report)

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nWrote report -> {report_path}")

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in metrics.items() if k != "frame"}
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"Wrote metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
