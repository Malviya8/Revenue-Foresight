"""
Verify submission contract artifacts (S5).

Usage (from repo root, after a successful run):
  python src/verify_submission.py
  python src/verify_submission.py --predictions ./output/predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema import HORIZONS_DAYS, HIERARCHY_LEVELS, PREDICTION_COLUMNS


REQUIRED_ROOT = [
    "run.sh",
    "requirements.txt",
    "README.md",
    "pickle/model.pkl",
    "data",
    "src/generate_features.py",
    "src/predict.py",
]


def check_repo(root: Path) -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_ROOT:
        path = root / rel
        if not path.exists():
            errors.append(f"missing required path: {rel}")
    model = root / "pickle" / "model.pkl"
    if model.exists() and model.stat().st_size < 1000:
        errors.append("pickle/model.pkl looks too small to be a trained model")
    req = root / "requirements.txt"
    if req.exists():
        text = req.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" not in line:
                errors.append(f"unpinned dependency: {line}")
    return errors


def check_predictions(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"predictions file missing: {path}"]
    df = pd.read_csv(path)
    if df.empty:
        errors.append("predictions file is empty")
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    extra_front = [c for c in df.columns if c not in PREDICTION_COLUMNS]
    # extras allowed but warn via stderr only - not hard errors
    if extra_front:
        print(f"NOTE: extra columns present (ignored): {extra_front}", file=sys.stderr)

    if "horizon_days" in df.columns:
        got = set(int(x) for x in df["horizon_days"].dropna().unique())
        need = set(HORIZONS_DAYS)
        if not need.issubset(got):
            errors.append(f"missing horizons: {sorted(need - got)}")
    if "level" in df.columns:
        got_lvl = set(df["level"].astype(str).unique())
        need_lvl = set(HIERARCHY_LEVELS)
        if not need_lvl.issubset(got_lvl):
            errors.append(f"missing levels: {sorted(need_lvl - got_lvl)}")

    for col in ("p10_revenue", "p50_revenue", "p90_revenue", "assumed_spend"):
        if col in df.columns and (df[col] < 0).any():
            errors.append(f"negative values in {col}")

    if {"p10_revenue", "p50_revenue", "p90_revenue"}.issubset(df.columns):
        bad = (
            (df["p10_revenue"] - 1e-6 > df["p50_revenue"])
            | (df["p50_revenue"] - 1e-6 > df["p90_revenue"])
        ).sum()
        if bad:
            errors.append(f"quantile crossing on {int(bad)} rows")

    # Hierarchy consistency: aggregate ≈ sum(channels) per horizon
    if {"level", "horizon_days", "p50_revenue"}.issubset(df.columns):
        for h, grp in df.groupby("horizon_days"):
            agg = grp.loc[grp["level"] == "aggregate", "p50_revenue"].sum()
            ch = grp.loc[grp["level"] == "channel", "p50_revenue"].sum()
            if abs(agg - ch) > max(1.0, 0.01 * abs(agg)):
                errors.append(
                    f"horizon {h}: aggregate p50 ({agg:.2f}) != channel sum ({ch:.2f})"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify AIgnition submission contract")
    parser.add_argument("--root", default=".", help="Repo root")
    parser.add_argument(
        "--predictions",
        default="./output/predictions.csv",
        help="Predictions CSV to validate",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Only check repo layout / requirements pins",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    errors = check_repo(root)
    if not args.skip_predictions:
        errors.extend(check_predictions(Path(args.predictions)))

    if errors:
        print("VERIFY FAILED")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("VERIFY OK")
    print(f"  root={root}")
    if not args.skip_predictions:
        print(f"  predictions={Path(args.predictions).resolve()}")


if __name__ == "__main__":
    main()
