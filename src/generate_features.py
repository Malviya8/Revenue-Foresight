"""
Generate model features from data/.

S2 hierarchical period features + S4 optional budget scenario overlays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from budget import apply_budget_scenario, load_budget_scenario, parse_multiplier_cli, scenario_summary
from features import build_inference_features, build_training_features
from ingest import load_cleaned_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate features from data/")
    parser.add_argument("--data-dir", required=True, help="Folder containing channel CSVs")
    parser.add_argument("--out", required=True, help="Output path (.parquet or .csv)")
    parser.add_argument(
        "--mode",
        choices=("inference", "training"),
        default="inference",
        help="inference = run.sh path; training = labeled cutoffs for model fit",
    )
    parser.add_argument(
        "--cutoff-freq-days",
        type=int,
        default=14,
        help="Spacing between training cutoffs (training mode only)",
    )
    parser.add_argument(
        "--budget-json",
        default=None,
        help="Optional budget scenario JSON (inference mode)",
    )
    parser.add_argument(
        "--budget-mult",
        action="append",
        default=None,
        help="Channel multiplier override, e.g. --budget-mult meta=1.2 (repeatable)",
    )
    args = parser.parse_args()

    panel, _report = load_cleaned_panel(args.data_dir)

    if args.mode == "training":
        features = build_training_features(
            panel, cutoff_freq_days=args.cutoff_freq_days
        )
    else:
        features = build_inference_features(panel)
        scenario: dict = {}
        if args.budget_json:
            scenario.update(load_budget_scenario(args.budget_json))
        cli_mult = parse_multiplier_cli(args.budget_mult)
        if cli_mult:
            merged = dict(scenario.get("channel_spend_multipliers") or {})
            merged.update(cli_mult)
            scenario["channel_spend_multipliers"] = merged
            scenario.setdefault("mode", "multiplier")
        if scenario:
            features = apply_budget_scenario(features, scenario)
            print("Budget scenario applied:")
            print(scenario_summary(features).to_string(index=False))

    if features.empty:
        raise RuntimeError("Feature table is empty — check data coverage / history length")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        features.to_parquet(out_path, index=False)
    else:
        features.to_csv(out_path, index=False)

    levels = (
        features.groupby("level").size().to_dict() if "level" in features.columns else {}
    )
    print(
        f"Wrote features mode={args.mode} rows={len(features)} "
        f"levels={levels} -> {out_path}"
    )


if __name__ == "__main__":
    main()
