"""
Sanity checks on a scored predictions.csv (ROAS floors, spread, aggregate peek).

Usage (from repo root, after run.sh / run.ps1):
  python scripts/check_predictions.py
  python scripts/check_predictions.py --predictions ./output/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="ROAS/spread sanity checks on predictions.csv")
    parser.add_argument(
        "--predictions",
        default="./output/predictions.csv",
        help="Path to predictions.csv",
    )
    args = parser.parse_args()
    path = Path(args.predictions)
    if not path.is_file():
        raise SystemExit(f"Missing predictions file: {path}")

    df = pd.read_csv(path)

    print("=== AGGREGATE ===")
    agg = df[df["level"] == "aggregate"][
        [
            "horizon_days",
            "p10_revenue",
            "p50_revenue",
            "p90_revenue",
            "p10_roas",
            "p50_roas",
            "p90_roas",
        ]
    ]
    print(agg.to_string())

    print("\n=== ROAS CAP CHECK ===")
    print("Rows with p50_roas >= 50:", int((df["p50_roas"] >= 50).sum()))
    print("Rows with p90_roas == 100:", int((df["p90_roas"] == 100).sum()))

    print("\n=== SPREAD CHECK ===")
    spread = df["p90_revenue"] / df["p10_revenue"].replace(0, 0.01)
    print("Mean spread ratio:", round(float(spread.mean()), 2))
    print("Max spread ratio:", round(float(spread.max()), 2))

    print("\n=== P10 ROAS FLOOR CHECK ===")
    # Floor applies when median outlook is healthy (p50_roas >= 2).
    healthy = df["p50_roas"] >= 2.0
    print("Healthy rows with p10_roas < 1:", int(((healthy) & (df["p10_roas"] < 1)).sum()))
    print(
        "Unhealthy rows with p10_roas < 1 (p50_roas < 2):",
        int(((~healthy) & (df["p10_roas"] < 1)).sum()),
    )
    print("Min p10_roas:", round(float(df["p10_roas"].min()), 3))
    print(
        "Min p10_roas (healthy only):",
        round(float(df.loc[healthy, "p10_roas"].min()), 3) if healthy.any() else "n/a",
    )


if __name__ == "__main__":
    main()
