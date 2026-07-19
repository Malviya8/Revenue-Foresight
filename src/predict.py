"""
Load pickled RevenueQuantileModel and write predictions.csv.

S4: optional hierarchy reconciliation + budget-aware feature inputs.
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
from schema import PREDICTION_COLUMNS, ROAS_WINSOR_CAP

# Low-spend campaigns: ROAS is statistically unreliable → hard cap.
LOW_SPEND_THRESHOLD = 50.0
LOW_SPEND_ROAS_CAP = 30.0
# P10 ROAS floors (do not mutate p50/p90 for these rules).
HEALTHY_P50_ROAS = 2.0
MIN_P10_ROAS_WHEN_HEALTHY = 1.0  # planning floor; check.py flags p10_roas < 1
MIN_P10_ROAS_GLOBAL = 0.1       # replace zeros / near-zeros
# Sparse campaigns: blend model quantiles toward hist run-rate before reconcile.
SHRINK_SPEND_THRESHOLD = 800.0
SHRINK_MIN_MODEL_WEIGHT = 0.35


def _enforce_revenue_order(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee p10_revenue <= p50_revenue <= p90_revenue row-wise."""
    out = df.copy()
    cols = ["p10_revenue", "p50_revenue", "p90_revenue"]
    if not all(c in out.columns for c in cols):
        return out
    stacked = np.vstack([out[c].astype(float).to_numpy() for c in cols])
    ordered = np.sort(stacked, axis=0)
    for i, c in enumerate(cols):
        out[c] = ordered[i]
    return out


def _enforce_roas_order(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee p10_roas <= p50_roas <= p90_roas row-wise."""
    out = df.copy()
    cols = ["p10_roas", "p50_roas", "p90_roas"]
    if not all(c in out.columns for c in cols):
        return out
    stacked = np.vstack([out[c].astype(float).to_numpy() for c in cols])
    ordered = np.sort(stacked, axis=0)
    for i, c in enumerate(cols):
        out[c] = ordered[i]
    return out


def _shrink_sparse_campaigns(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Blend low-history campaign P50 toward hist run-rate; scale P10/P90 to keep shape.

    Reconcile still top-down scales children → parents; this only stabilizes
    volatile campaign rows before hierarchy enforcement.
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
    # More trailing spend → trust model more; very sparse → lean on run-rate.
    w = np.minimum(1.0, hist_spend / SHRINK_SPEND_THRESHOLD)
    w = np.maximum(SHRINK_MIN_MODEL_WEIGHT, w)
    sparse = hist_spend < SHRINK_SPEND_THRESHOLD
    w = np.where(sparse, w, 1.0)

    p50_new = w * p50 + (1.0 - w) * baseline_p50
    # Use true p50 (eps), not max(p50, 1.0) — that broke ratios when p50 < 1.
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


def postprocess_roas(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Business ROAS guards applied after model + reconcile:

    1. assumed_spend < 50 → clip p10/p50/p90 ROAS at 30
    2. p50_roas >= 2 → p10_roas = max(p10_roas, 1.0)  [p50/p90 unchanged]
    3. p10_roas < 0.1 → raise to 0.1 when it will not cross p50
    4. Enforce p10 <= p50 <= p90 on both revenue and ROAS
    """
    out = preds.copy()
    for col in ("assumed_spend", "p10_revenue", "p50_revenue", "p90_revenue",
                "p10_roas", "p50_roas", "p90_roas"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    low_spend = out["assumed_spend"] < LOW_SPEND_THRESHOLD
    for col in ("p10_roas", "p50_roas", "p90_roas"):
        out.loc[low_spend, col] = out.loc[low_spend, col].clip(upper=LOW_SPEND_ROAS_CAP)

    # Near-zero floor only when it preserves quantile order
    can_floor = out["p50_roas"] >= MIN_P10_ROAS_GLOBAL
    needs_floor = out["p10_roas"] < MIN_P10_ROAS_GLOBAL
    out.loc[can_floor & needs_floor, "p10_roas"] = MIN_P10_ROAS_GLOBAL

    # Healthy median → p10 floor at 1.0
    healthy = out["p50_roas"] >= HEALTHY_P50_ROAS
    out.loc[healthy, "p10_roas"] = out.loc[healthy, "p10_roas"].clip(
        lower=MIN_P10_ROAS_WHEN_HEALTHY
    )

    out = _enforce_revenue_order(out)
    out = _enforce_roas_order(out)
    return out


def _placeholder_predictions(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, feat in features.iterrows():
        assumed_spend = float(feat.get("planned_spend", 0.0) or 0.0)
        roas = feat.get("hist_roas_28", feat.get("hist_roas_14", feat.get("hist_roas_7", 0.0)))
        roas = float(roas) if pd.notna(roas) else 0.0
        roas = min(max(roas, 0.0), 50.0)
        p50_rev = assumed_spend * roas
        vol = feat.get("hist_roas_std_28", 0.0)
        vol = float(vol) if pd.notna(vol) else 0.0
        width = min(0.35, 0.15 + vol / max(roas, 1.0) * 0.1)
        rows.append(
            {
                "horizon_days": int(feat["horizon_days"]),
                "level": str(feat.get("level", "channel")),
                "channel": str(feat.get("channel", "")),
                "campaign_type": str(feat.get("campaign_type", "") or ""),
                "campaign_id": str(feat.get("campaign_id", "") or ""),
                "campaign_name": str(feat.get("campaign_name", "") or ""),
                "assumed_spend": round(assumed_spend, 4),
                "p10_revenue": round(p50_rev * (1 - width), 4),
                "p50_revenue": round(p50_rev, 4),
                "p90_revenue": round(p50_rev * (1 + width), 4),
                "p10_roas": round(roas * (1 - width), 6),
                "p50_roas": round(roas, 6),
                "p90_roas": round(roas * (1 + width), 6),
            }
        )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def load_model(model_path: Path):
    with model_path.open("rb") as fh:
        return pickle.load(fh)


def predict_frame(features: pd.DataFrame, model_path: Path, reconcile: bool = True) -> pd.DataFrame:
    raw: pd.DataFrame
    if model_path.is_file() and model_path.stat().st_size > 0:
        model = load_model(model_path)
        if hasattr(model, "predict") and callable(model.predict):
            raw = model.predict(features)
        elif isinstance(model, dict) and callable(model.get("predict")):
            raw = model["predict"](features)
        else:
            raw = _placeholder_predictions(features)
    else:
        raw = _placeholder_predictions(features)

    if reconcile:
        raw = _shrink_sparse_campaigns(raw, features)
        preds = reconcile_predictions(raw)
    else:
        preds = raw[PREDICTION_COLUMNS] if all(c in raw.columns for c in PREDICTION_COLUMNS) else raw
    return postprocess_roas(preds)


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

    # Raw then optional reconcile (need both for report)
    if model_path.is_file() and model_path.stat().st_size > 0:
        model = load_model(model_path)
        if hasattr(model, "predict") and callable(model.predict):
            raw = model.predict(features)
        else:
            raw = _placeholder_predictions(features)
    else:
        raw = _placeholder_predictions(features)

    if args.no_reconcile:
        preds = raw[PREDICTION_COLUMNS] if all(c in raw.columns for c in PREDICTION_COLUMNS) else raw
    else:
        raw = _shrink_sparse_campaigns(raw, features)
        preds = reconcile_predictions(raw)
        if args.report_out:
            report = reconciliation_report(raw, preds)
            Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Wrote reconcile report -> {args.report_out}")

    preds = postprocess_roas(preds)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Always rewrite fresh (submission contract)
    preds.to_csv(out_path, index=False)
    print(f"Wrote predictions ({len(preds)} rows) -> {out_path}")


if __name__ == "__main__":
    main()
