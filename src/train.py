"""
Train LightGBM quantile models, backtest, and pickle pickle/model.pkl.

Usage:
  python src/train.py --data-dir ./data --model-out ./pickle/model.pkl
  python src/train.py --features ./output/features_train.parquet --model-out ./pickle/model.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import build_training_features
from ingest import load_cleaned_panel
from metrics import run_rate_baseline, summarize_backtest
from model import RevenueQuantileModel
from schema import FEATURE_VALUE_COLUMNS, MAX_INTERVAL_HALF_WIDTH


def _time_split(frame: pd.DataFrame, valid_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values("as_of_date").reset_index(drop=True)
    cutoffs = np.sort(frame["as_of_date"].unique())
    if len(cutoffs) < 5:
        n = max(1, int(len(frame) * (1 - valid_frac)))
        return frame.iloc[:n].copy(), frame.iloc[n:].copy()
    split_i = max(1, int(len(cutoffs) * (1 - valid_frac)))
    split_date = cutoffs[split_i - 1]
    train = frame[frame["as_of_date"] <= split_date].copy()
    valid = frame[frame["as_of_date"] > split_date].copy()
    if valid.empty:
        # fall back to row split
        n = max(1, int(len(frame) * (1 - valid_frac)))
        return frame.iloc[:n].copy(), frame.iloc[n:].copy()
    return train, valid


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"])
    out["target_revenue"] = out["target_revenue"].astype(float).clip(lower=0.0)
    # Ensure model columns exist
    for col in FEATURE_VALUE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def load_or_build_training(
    data_dir: str | None,
    features_path: str | None,
    cutoff_freq_days: int,
) -> pd.DataFrame:
    if features_path:
        path = Path(features_path)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        return _prepare_frame(frame)

    if not data_dir:
        raise ValueError("Provide --data-dir or --features")
    panel, _ = load_cleaned_panel(data_dir)
    frame = build_training_features(panel, cutoff_freq_days=cutoff_freq_days)
    if frame.empty:
        raise RuntimeError("Training feature table is empty")
    return _prepare_frame(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train quantile revenue models (S3)")
    parser.add_argument("--data-dir", default="./data", help="Channel CSV folder")
    parser.add_argument(
        "--features",
        default=None,
        help="Optional prebuilt training features parquet/csv (skips rebuild)",
    )
    parser.add_argument("--model-out", default="./pickle/model.pkl")
    parser.add_argument("--metrics-out", default="./output/backtest_metrics.json")
    parser.add_argument("--features-out", default="./output/features_train.parquet")
    parser.add_argument("--cutoff-freq-days", type=int, default=14)
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-boost-round", type=int, default=500)
    args = parser.parse_args()

    print("Loading / building training features...")
    frame = load_or_build_training(args.data_dir, args.features, args.cutoff_freq_days)
    frame = frame.reset_index(drop=True)
    feats_out = Path(args.features_out)
    feats_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(feats_out, index=False)
    print(f"  training rows={len(frame)} -> {feats_out}")

    train_df, valid_df = _time_split(frame, valid_frac=args.valid_frac)
    print(
        f"  time split: train={len(train_df)} "
        f"({train_df['as_of_date'].min().date()}->{train_df['as_of_date'].max().date()}) "
        f"valid={len(valid_df)} "
        f"({valid_df['as_of_date'].min().date() if len(valid_df) else 'n/a'}"
        f"->{valid_df['as_of_date'].max().date() if len(valid_df) else 'n/a'})"
    )

    model = RevenueQuantileModel(feature_columns=list(FEATURE_VALUE_COLUMNS))
    print("Fitting LightGBM quantile models (P10/P50/P90)...")
    model.fit(
        train_df,
        valid=valid_df if len(valid_df) else None,
        num_boost_round=args.num_boost_round,
        random_state=args.seed,
    )
    if len(valid_df):
        model.calibrate_intervals(valid_df)

    # Holdout backtest
    if len(valid_df):
        p10, p50, p90 = model.predict_quantiles(valid_df)
        baseline = run_rate_baseline(valid_df)
        metrics = summarize_backtest(valid_df.reset_index(drop=True), p10, p50, p90, baseline)
        metrics["interval_scale"] = {
            "conformal_rel": model.conformal_rel,
            "conformal_radius": model.conformal_radius,
            "max_interval_half_width": MAX_INTERVAL_HALF_WIDTH,
        }
    else:
        p10, p50, p90 = model.predict_quantiles(train_df)
        baseline = run_rate_baseline(train_df)
        metrics = summarize_backtest(train_df.reset_index(drop=True), p10, p50, p90, baseline)
        metrics["note"] = "no holdout; metrics on train"

    # Refit on full data for the committed artifact (best accuracy for submission)
    print("Refitting on full training table for submission artifact...")
    final = RevenueQuantileModel(feature_columns=list(FEATURE_VALUE_COLUMNS))
    # Use last 15% as early-stopping monitor even for final fit
    full_train, full_valid = _time_split(frame, valid_frac=0.15)
    final.fit(
        full_train if len(full_valid) else frame,
        valid=full_valid if len(full_valid) else None,
        num_boost_round=args.num_boost_round,
        random_state=args.seed,
    )
    # Carry calibration from holdout evaluation model
    final.conformal_rel = model.conformal_rel
    final.conformal_radius = model.conformal_radius
    final.lower_width_scale = model.lower_width_scale
    final.upper_width_scale = model.upper_width_scale
    final.train_metrics = {
        "holdout": metrics,
        "n_train_rows": int(len(frame)),
        "feature_columns": list(FEATURE_VALUE_COLUMNS),
        "seed": args.seed,
        "interval_scale": {
            "conformal_rel": final.conformal_rel,
            "conformal_radius": final.conformal_radius,
            "max_interval_half_width": MAX_INTERVAL_HALF_WIDTH,
        },
    }

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(final, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote model -> {model_path} ({model_path.stat().st_size} bytes)")

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(final.train_metrics, indent=2, default=str), encoding="utf-8")
    print(f"Wrote metrics -> {metrics_path}")

    hold = final.train_metrics["holdout"]
    print(
        f"Holdout wMAPE model={hold.get('wmape_model'):.4f} "
        f"baseline={hold.get('wmape_baseline'):.4f} "
        f"lift={hold.get('lift_wmape_vs_baseline')} "
        f"coverage={hold.get('coverage_p10_p90'):.3f}"
    )


if __name__ == "__main__":
    main()
