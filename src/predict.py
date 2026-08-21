"""
Load pickled RevenueQuantileModel and write predictions.csv.

Split-conformal adjustments live inside the pickle (per-channel). This module
applies optional hierarchy reconciliation + ROAS post-process rules.
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

from model import RevenueQuantileModel  # noqa: F401
from reconcile import reconcile_predictions, reconciliation_report
from schema import (
    LOW_SPEND_ROAS_CAP,
    LOW_SPEND_THRESHOLD,
    MIN_P10_ROAS_GLOBAL,
    MIN_P10_ROAS_WHEN_HEALTHY,
    PREDICTION_COLUMNS,
    ROAS_WINSOR_CAP,
)

HEALTHY_P50_ROAS = 2.0
SHRINK_SPEND_THRESHOLD = 800.0
SHRINK_MIN_MODEL_WEIGHT = 0.35


def _enforce_revenue_order(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee p10_revenue <= p50_revenue <= p90_revenue without moving p50 via sort.

    Uses explicit clamps so p50 stays fixed.
    """
    out = df.copy()
    need = ["p10_revenue", "p50_revenue", "p90_revenue"]
    if not all(c in out.columns for c in need):
        return out
    p10 = out["p10_revenue"].astype(float).to_numpy()
    p50 = out["p50_revenue"].astype(float).to_numpy()
    p90 = out["p90_revenue"].astype(float).to_numpy()
    p10 = np.maximum(p10, 0.0)
    p10 = np.minimum(p10, p50 * 0.95)
    p90 = np.maximum(p90, p50 * 1.05)
    p10 = np.minimum(p10, p50)
    p90 = np.maximum(p90, p50)
    out["p10_revenue"] = p10
    out["p90_revenue"] = p90
    return out


def _enforce_roas_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = ["p10_roas", "p50_roas", "p90_roas"]
    if not all(c in out.columns for c in cols):
        return out
    p10 = out["p10_roas"].astype(float).to_numpy()
    p50 = out["p50_roas"].astype(float).to_numpy()
    p90 = out["p90_roas"].astype(float).to_numpy()
    p10 = np.minimum(p10, p50)
    p90 = np.maximum(p90, p50)
    out["p10_roas"] = p10
    out["p90_roas"] = p90
    return out


def _shrink_sparse_campaigns(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Blend low-history campaign P50 toward hist run-rate; scale P10/P90 to keep shape.
    """
    if raw.empty or features.empty:
        return raw

    need = {
        "horizon_days",
        "level",
        "channel",
        "campaign_type",
        "campaign_id",
        "planned_spend",
        "hist_roas_28",
        "hist_spend_sum_28",
    }
    if not need.issubset(features.columns):
        return raw

    keys = ["horizon_days", "level", "channel", "campaign_type", "campaign_id"]
    str_keys = ["level", "channel", "campaign_type", "campaign_id"]
    feat = features[keys + ["planned_spend", "hist_roas_28", "hist_spend_sum_28"]].copy()
    feat["horizon_days"] = feat["horizon_days"].astype(int)
    for col in str_keys:
        feat[col] = feat[col].fillna("").astype(str)
    out = raw.copy()
    out["horizon_days"] = out["horizon_days"].astype(int)
    for col in str_keys:
        out[col] = out[col].fillna("").astype(str)

    merged = out.merge(feat, on=keys, how="left", suffixes=("", "_feat"))
    camp = merged["level"] == "campaign"
    if not camp.any():
        return raw

    spend = merged.loc[camp, "planned_spend"].astype(float).fillna(0.0).to_numpy()
    hist_roas = merged.loc[camp, "hist_roas_28"].astype(float).fillna(0.0).to_numpy()
    hist_roas = np.clip(hist_roas, 0.0, ROAS_WINSOR_CAP)
    hist_spend = merged.loc[camp, "hist_spend_sum_28"].astype(float).fillna(0.0).to_numpy()

    p10 = merged.loc[camp, "p10_revenue"].astype(float).to_numpy()
    p50 = merged.loc[camp, "p50_revenue"].astype(float).to_numpy()
    p90 = merged.loc[camp, "p90_revenue"].astype(float).to_numpy()

    baseline_p50 = spend * hist_roas
    w = np.minimum(1.0, hist_spend / SHRINK_SPEND_THRESHOLD)
    w = np.maximum(SHRINK_MIN_MODEL_WEIGHT, w)
    sparse = hist_spend < SHRINK_SPEND_THRESHOLD
    w = np.where(sparse, w, 1.0)

    p50_new = w * p50 + (1.0 - w) * baseline_p50
    eps = 1e-9
    ratio_lo = np.divide(p10, np.maximum(p50, eps), out=np.full_like(p10, 0.7), where=p50 > 0)
    ratio_hi = np.divide(p90, np.maximum(p50, eps), out=np.full_like(p90, 1.3), where=p50 > 0)
    ratio_lo = np.clip(ratio_lo, 0.0, 1.0)
    ratio_hi = np.maximum(ratio_hi, 1.0)
    p10_new = np.maximum(p50_new * ratio_lo, 0.0)
    p90_new = np.maximum(p50_new * ratio_hi, p50_new)

    idx = merged.index[camp]
    out.loc[idx, "p10_revenue"] = np.round(p10_new, 4)
    out.loc[idx, "p50_revenue"] = np.round(p50_new, 4)
    out.loc[idx, "p90_revenue"] = np.round(p90_new, 4)
    out = _enforce_revenue_order(out)
    return out[PREDICTION_COLUMNS] if all(c in out.columns for c in PREDICTION_COLUMNS) else out


def postprocess_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Post-conformal inference rules:

    - final_p10 = max(final_p10, 0)
    - final_p10 = min(final_p10, final_p50 * 0.95)
    - final_p90 = max(final_p90, final_p50 * 1.05)
    - assumed_spend < 50 → p10_roas = 0.1, p90_roas = min(p90_roas, 30)
    """
    out = preds.copy()
    for col in (
        "assumed_spend",
        "p10_revenue",
        "p50_revenue",
        "p90_revenue",
        "p10_roas",
        "p50_roas",
        "p90_roas",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out = _enforce_revenue_order(out)

    # Recompute ROAS from revenue after revenue clamps.
    spend = out["assumed_spend"].to_numpy(dtype=float)
    for q in ("p10", "p50", "p90"):
        rev = out[f"{q}_revenue"].to_numpy(dtype=float)
        roas = np.divide(rev, spend, out=np.zeros_like(rev), where=spend > 0)
        out[f"{q}_roas"] = roas

    low = out["assumed_spend"] < LOW_SPEND_THRESHOLD
    out.loc[low, "p10_roas"] = MIN_P10_ROAS_GLOBAL
    out.loc[low, "p90_roas"] = out.loc[low, "p90_roas"].clip(upper=LOW_SPEND_ROAS_CAP)

    healthy = out["p50_roas"] >= HEALTHY_P50_ROAS
    out.loc[healthy, "p10_roas"] = out.loc[healthy, "p10_roas"].clip(
        lower=MIN_P10_ROAS_WHEN_HEALTHY
    )

    out = _enforce_roas_order(out)
    return out


# Back-compat name used by older callers
def postprocess_roas(preds: pd.DataFrame) -> pd.DataFrame:
    return postprocess_predictions(preds)


def load_model(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model pickle not found: {model_path}. "
            "Train offline with src/train.py or restore pickle/model.pkl."
        )
    if model_path.stat().st_size <= 0:
        raise ValueError(f"Model pickle is empty: {model_path}")
    with model_path.open("rb") as fh:
        try:
            model = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 — surface corrupt pickles clearly
            raise ValueError(f"Failed to unpickle model at {model_path}: {exc}") from exc
    return model


def _require_predictable(model: object, model_path: Path):
    if hasattr(model, "predict") and callable(model.predict):
        return model
    if isinstance(model, dict) and callable(model.get("predict")):
        return model
    raise TypeError(
        f"Unsupported model artifact at {model_path}: expected RevenueQuantileModel "
        "with a callable .predict (or a dict with a 'predict' callable). "
        "Silent hist-ROAS stubs are disabled so the CLI cannot emit fake CSVs."
    )


def predict_frame(features: pd.DataFrame, model_path: Path, reconcile: bool = True) -> pd.DataFrame:
    model = _require_predictable(load_model(model_path), model_path)
    if hasattr(model, "predict") and callable(getattr(model, "predict", None)):
        raw = model.predict(features)
    else:
        raw = model["predict"](features)

    if reconcile:
        raw = _shrink_sparse_campaigns(raw, features)
        preds = reconcile_predictions(raw)
    else:
        preds = raw[PREDICTION_COLUMNS] if all(c in raw.columns for c in PREDICTION_COLUMNS) else raw
    return postprocess_predictions(preds)


def predict(
    features_path: Path,
    model_path: Path,
    reconcile: bool = True,
) -> pd.DataFrame:
    features = (
        pd.read_parquet(features_path)
        if features_path.suffix.lower() == ".parquet"
        else pd.read_csv(features_path)
    )
    return predict_frame(features, model_path, reconcile=reconcile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce predictions from features + model")
    parser.add_argument("--features", required=True, help="Feature file from generate_features")
    parser.add_argument("--model", required=True, help="Path to pickle/model.pkl")
    parser.add_argument("--output", required=True, help="Output predictions.csv path")
    parser.add_argument(
        "--no-reconcile",
        action="store_true",
        help="Skip top-down hierarchy reconciliation",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help="Optional JSON path for reconciliation gap report",
    )
    args = parser.parse_args()

    features = (
        pd.read_parquet(args.features)
        if Path(args.features).suffix.lower() == ".parquet"
        else pd.read_csv(args.features)
    )
    model_path = Path(args.model)
    model = _require_predictable(load_model(model_path), model_path)
    if hasattr(model, "predict") and callable(getattr(model, "predict", None)):
        raw = model.predict(features)
    else:
        raw = model["predict"](features)

    if args.no_reconcile:
        preds = raw[PREDICTION_COLUMNS] if all(c in raw.columns for c in PREDICTION_COLUMNS) else raw
    else:
        raw = _shrink_sparse_campaigns(raw, features)
        preds = reconcile_predictions(raw)
        if args.report_out:
            report = reconciliation_report(raw, preds)
            Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Wrote reconcile report -> {args.report_out}")

    preds = postprocess_predictions(preds)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"Wrote predictions ({len(preds)} rows) -> {out_path}")


if __name__ == "__main__":
    main()
