"""
Load channel CSVs from data/, validate, and emit a cleaned unified panel.

Pipeline: raw CSV -> normalize (schema) -> QA report -> clean/derive -> cleaned panel
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clean import clean_panel
from schema import (
    CAMPAIGN_TYPE_KEYWORDS,
    CHANNEL_FILE_PATTERNS,
    CHANNEL_MAPPINGS,
    CHANNELS,
    DEFAULT_CAMPAIGN_TYPE,
    UNIFIED_COLUMNS,
)
from validate import print_eda, validate_panel


def infer_campaign_type(name: object) -> str:
    """Derive a campaign type from free-text campaign names (Meta etc.)."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return DEFAULT_CAMPAIGN_TYPE
    text = str(name).lower()
    for needle, label in CAMPAIGN_TYPE_KEYWORDS:
        if needle in text:
            return label
    return DEFAULT_CAMPAIGN_TYPE


def _resolve_channel_file(data_dir: Path, channel: str) -> Path:
    patterns = CHANNEL_FILE_PATTERNS[channel]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(data_dir.glob(pattern))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in matches:
        resolved = path.resolve()
        if resolved not in seen and path.is_file():
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise FileNotFoundError(
            f"No CSV found for channel '{channel}' in {data_dir} "
            f"(tried patterns: {patterns})"
        )
    if len(unique) > 1:
        unique.sort(key=lambda p: (channel not in p.name.lower(), len(p.name)))
    return unique[0]


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _normalize_channel(raw: pd.DataFrame, channel: str) -> pd.DataFrame:
    mapping = CHANNEL_MAPPINGS[channel]
    missing = [
        raw_col
        for key, raw_col in mapping.items()
        if key not in ("spend_micros",) and raw_col is not None and raw_col not in raw.columns
    ]
    if missing:
        raise ValueError(
            f"Channel '{channel}' missing expected columns: {missing}. "
            f"Found: {list(raw.columns)}"
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw[mapping["date"]], errors="coerce")
    out["channel"] = channel
    out["campaign_id"] = raw[mapping["campaign_id"]].astype(str)
    out["campaign_name"] = raw[mapping["campaign_name"]].astype(str)

    if mapping["campaign_type"] is not None:
        out["campaign_type"] = (
            raw[mapping["campaign_type"]]
            .astype(str)
            .replace({"nan": DEFAULT_CAMPAIGN_TYPE, "None": DEFAULT_CAMPAIGN_TYPE})
        )
        # Soft-normalize empty/unknown Google types
        blank = out["campaign_type"].str.strip().isin(["", "nan", "None", "NaT"])
        if blank.any():
            out.loc[blank, "campaign_type"] = raw.loc[blank, mapping["campaign_name"]].map(
                infer_campaign_type
            )
    else:
        out["campaign_type"] = raw[mapping["campaign_name"]].map(infer_campaign_type)

    spend = pd.to_numeric(raw[mapping["spend"]], errors="coerce")
    if mapping["spend_micros"]:
        spend = spend / 1_000_000.0
    out["spend"] = spend

    out["revenue"] = pd.to_numeric(raw[mapping["revenue"]], errors="coerce")
    out["clicks"] = pd.to_numeric(raw[mapping["clicks"]], errors="coerce")
    out["impressions"] = pd.to_numeric(raw[mapping["impressions"]], errors="coerce")

    if mapping["conversions"] is not None:
        out["conversions"] = pd.to_numeric(raw[mapping["conversions"]], errors="coerce")
    else:
        out["conversions"] = pd.Series(pd.NA, index=raw.index, dtype="Float64")

    if mapping["daily_budget"] is not None:
        out["daily_budget"] = pd.to_numeric(raw[mapping["daily_budget"]], errors="coerce")
    else:
        out["daily_budget"] = pd.Series(pd.NA, index=raw.index, dtype="Float64")

    for col in ("spend", "revenue", "clicks", "impressions", "conversions", "daily_budget"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    return out[UNIFIED_COLUMNS]


def load_unified_panel(data_dir: str | Path) -> pd.DataFrame:
    """Read all channel CSVs under data_dir and return a raw unified panel."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise NotADirectoryError(f"DATA_DIR is not a directory: {data_path}")

    frames: list[pd.DataFrame] = []
    for channel in CHANNELS:
        path = _resolve_channel_file(data_path, channel)
        raw = pd.read_csv(path)
        raw = _drop_unnamed(raw)
        frames.append(_normalize_channel(raw, channel))

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["date"])
    panel = panel.sort_values(["channel", "campaign_id", "date"]).reset_index(drop=True)
    return panel


def load_cleaned_panel(
    data_dir: str | Path,
    qa_path: str | Path | None = None,
) -> tuple[pd.DataFrame, object]:
    """
    Full S1 path: unify -> validate -> clean.
    Returns (cleaned_panel, qa_report). Optionally writes QA JSON.
    """
    raw_panel = load_unified_panel(data_dir)
    report = validate_panel(raw_panel)
    if qa_path is not None:
        report.save(qa_path)
    cleaned = clean_panel(raw_panel)
    return cleaned, report


def panel_summary(panel: pd.DataFrame) -> dict[str, object]:
    """Lightweight inventory for smoke checks."""
    summary: dict[str, object] = {
        "rows": int(len(panel)),
        "date_min": str(panel["date"].min().date()) if len(panel) else None,
        "date_max": str(panel["date"].max().date()) if len(panel) else None,
        "channels": {},
    }
    for channel, grp in panel.groupby("channel"):
        spend = float(grp["spend"].sum(skipna=True))
        revenue = float(grp["revenue"].sum(skipna=True))
        summary["channels"][channel] = {
            "rows": int(len(grp)),
            "campaigns": int(grp["campaign_id"].nunique()),
            "date_min": str(grp["date"].min().date()),
            "date_max": str(grp["date"].max().date()),
            "spend": spend,
            "revenue": revenue,
            "blended_roas": (revenue / spend) if spend > 0 else 0.0,
        }
    return summary


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest + validate + clean channel CSVs into a unified panel"
    )
    parser.add_argument("--data-dir", default="./data", help="Folder with channel CSVs")
    parser.add_argument(
        "--out",
        default="./output/cleaned_panel.parquet",
        help="Path for cleaned panel (.parquet or .csv)",
    )
    parser.add_argument(
        "--raw-out",
        default=None,
        help="Optional path to also write the pre-clean unified panel",
    )
    parser.add_argument(
        "--qa-out",
        default="./output/qa_report.json",
        help="Path for QA / EDA JSON report",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Only unify + validate; write raw panel to --out",
    )
    args = parser.parse_args()

    raw_panel = load_unified_panel(args.data_dir)
    report = validate_panel(raw_panel)
    report.save(args.qa_out)

    print("Unified panel OK")
    summary = panel_summary(raw_panel)
    print(f"  rows={summary['rows']}  range={summary['date_min']} -> {summary['date_max']}")
    for channel, stats in summary["channels"].items():  # type: ignore[union-attr]
        print(
            f"  {channel}: rows={stats['rows']} campaigns={stats['campaigns']} "
            f"spend={stats['spend']:.2f} revenue={stats['revenue']:.2f} "
            f"ROAS={stats['blended_roas']:.3f}"
        )

    print_eda(report)
    print(f"Wrote QA report -> {args.qa_out}")

    if args.raw_out:
        _write_frame(raw_panel, Path(args.raw_out))
        print(f"Wrote raw unified panel -> {args.raw_out}")

    if args.skip_clean:
        _write_frame(raw_panel, Path(args.out))
        print(f"Wrote raw panel (--skip-clean) -> {args.out}")
        return

    cleaned = clean_panel(raw_panel)
    _write_frame(cleaned, Path(args.out))
    print(
        f"Cleaned panel: rows={len(cleaned)} cols={len(cleaned.columns)} -> {args.out}"
    )
    # Quick derived-metric sanity
    active = cleaned[cleaned["is_active"] == 1]
    print(
        f"  active_rows={len(active)} "
        f"median_roas={active['roas'].median(skipna=True):.3f} "
        f"median_spend_share={active['spend_share'].median(skipna=True):.3f}"
    )


if __name__ == "__main__":
    main()
