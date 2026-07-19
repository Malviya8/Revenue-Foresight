"""
Train LightGBM quantile models with Mondrian split-conformal calibration.

Chronological split (by as_of_date):
  train 60% → fit P10/P50/P90
  calib 20% → channel × season conformal adjustments (holiday Nov–Dec)
  test  20% → reserved for evaluate.py only (never used here)

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
from model import RevenueQuantileModel, chronological_three_way_split
from schema import FEATURE_VALUE_COLUMNS


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"])
    out["target_revenue"] = out["target_revenue"].astype(float).clip(lower=0.0)
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
    parser = argparse.ArgumentParser(description="Train split-conformal quantile models")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-boost-round", type=int, default=500)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading / building training features...")
    frame = load_or_build_training(args.data_dir, args.features, args.cutoff_freq_days)
    frame = frame.reset_index(drop=True)
    feats_out = Path(args.features_out)
    feats_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(feats_out, index=False)
    print(f"  training rows={len(frame)} -> {feats_out}")

    train_df, calib_df, test_df = chronological_three_way_split(frame)
    print(
        f"  chronological 60/20/20 split:\n"
        f"    train  n={len(train_df)} "
        f"({train_df['as_of_date'].min().date()}->{train_df['as_of_date'].max().date()})\n"
        f"    calib  n={len(calib_df)} "
        f"({calib_df['as_of_date'].min().date()}->{calib_df['as_of_date'].max().date()})\n"
        f"    test   n={len(test_df)} "
        f"({test_df['as_of_date'].min().date()}->{test_df['as_of_date'].max().date()}) "
        f"[held out — evaluate.py only]"
    )

    model = RevenueQuantileModel(feature_columns=list(FEATURE_VALUE_COLUMNS))
    print("Fitting LightGBM quantile models on TRAIN only (P10/P50/P90)...")
    # No early-stopping on calib — keeps conformal scores honest.
    model.fit(
        train_df,
        valid=None,
        num_boost_round=args.num_boost_round,
        random_state=args.seed,
    )

    print("Mondrian split-conformal calibration on CALIB (channel × season × Google brand)...")
    model.calibrate_conformal(calib_df, train=train_df)
    print(f"  conformal_adjustments: {model.conformal_adjustments}")
    print(f"  conformal_stratum_counts: {model.conformal_stratum_counts}")

    # Diagnostics on calibration window (not the test set).
    p10, p50, p90 = model.predict_quantiles(calib_df)
    baseline = run_rate_baseline(calib_df)
    metrics = summarize_backtest(calib_df.reset_index(drop=True), p10, p50, p90, baseline)
    metrics["split"] = {
        "train_n": int(len(train_df)),
        "calib_n": int(len(calib_df)),
        "test_n": int(len(test_df)),
        "train_start": str(train_df["as_of_date"].min().date()),
        "train_end": str(train_df["as_of_date"].max().date()),
        "calib_start": str(calib_df["as_of_date"].min().date()),
        "calib_end": str(calib_df["as_of_date"].max().date()),
        "test_start": str(test_df["as_of_date"].min().date()),
        "test_end": str(test_df["as_of_date"].max().date()),
        "note": "coverage below is on CALIB; evaluate.py reports TEST",
    }
    metrics["conformal_adjustments"] = dict(model.conformal_adjustments)
    metrics["conformal_stratum_counts"] = dict(model.conformal_stratum_counts)
    metrics["method"] = "mondrian_split_conformal"

    model.train_metrics = {
        "calibration_diagnostics": metrics,
        "n_train_rows": int(len(train_df)),
        "n_calib_rows": int(len(calib_df)),
        "n_test_rows": int(len(test_df)),
        "feature_columns": list(FEATURE_VALUE_COLUMNS),
        "seed": args.seed,
        "conformal_adjustments": dict(model.conformal_adjustments),
        "conformal_stratum_counts": dict(model.conformal_stratum_counts),
        "version": model.version,
    }

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote model -> {model_path} ({model_path.stat().st_size} bytes)")

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(model.train_metrics, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote metrics -> {metrics_path}")

    print(
        f"Calib wMAPE model={metrics.get('wmape_model'):.4f} "
        f"baseline={metrics.get('wmape_baseline'):.4f} "
        f"lift={metrics.get('lift_wmape_vs_baseline')} "
        f"coverage={metrics.get('coverage_p10_p90'):.3f} "
        f"(calib only — see evaluate.py for test)"
    )


if __name__ == "__main__":
    main()
