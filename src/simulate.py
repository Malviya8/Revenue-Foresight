"""
One-shot budget simulation: features + model + reconcile -> predictions.

Example:
  python src/simulate.py --data-dir ./data --model ./pickle/model.pkl \\
      --budget-json scenarios/meta_plus20.json \\
      --out ./output/predictions_meta_plus20.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from budget import apply_budget_scenario, load_budget_scenario, parse_multiplier_cli, scenario_summary
from features import build_inference_features
from ingest import load_cleaned_panel
from predict import predict_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-scenario forecast simulation")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model", default="./pickle/model.pkl")
    parser.add_argument("--out", default="./output/predictions_scenario.csv")
    parser.add_argument("--budget-json", default=None)
    parser.add_argument(
        "--budget-mult",
        action="append",
        default=None,
        help="channel=multiplier (repeatable), e.g. meta=1.2",
    )
    parser.add_argument("--no-reconcile", action="store_true")
    parser.add_argument(
        "--features-out",
        default=None,
        help="Optional path to save scenario feature table",
    )
    args = parser.parse_args()

    panel, _ = load_cleaned_panel(args.data_dir)
    features = build_inference_features(panel)

    scenario: dict = {}
    if args.budget_json:
        scenario.update(load_budget_scenario(args.budget_json))
    cli_mult = parse_multiplier_cli(args.budget_mult)
    if cli_mult:
        merged = dict(scenario.get("channel_spend_multipliers") or {})
        merged.update(cli_mult)
        scenario["channel_spend_multipliers"] = merged
    if scenario:
        features = apply_budget_scenario(features, scenario)
        print("Scenario planned spend (channel x horizon):")
        print(scenario_summary(features).to_string(index=False))
    else:
        print("No budget overrides — using run-rate planned spend")

    if args.features_out:
        path = Path(args.features_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".parquet":
            features.to_parquet(path, index=False)
        else:
            features.to_csv(path, index=False)

    preds = predict_frame(
        features, Path(args.model), reconcile=not args.no_reconcile
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out, index=False)
    print(f"Wrote scenario predictions ({len(preds)} rows) -> {out}")

    # Compact channel view for demos
    ch = preds[preds["level"] == "channel"][
        ["horizon_days", "channel", "assumed_spend", "p10_revenue", "p50_revenue", "p90_revenue", "p50_roas"]
    ]
    print(ch.sort_values(["horizon_days", "channel"]).to_string(index=False))


if __name__ == "__main__":
    main()
