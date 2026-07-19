"""
Campaign consistency checks and EDA inventory for the unified panel.

Does not mutate data — returns structured Issue / QAReport objects that clean.py
and the ingest CLI consume.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from schema import CHANNELS


@dataclass
class Issue:
    code: str
    severity: str  # info | warn | error
    channel: str | None
    count: int
    detail: str


@dataclass
class QAReport:
    row_count_raw: int
    issues: list[Issue] = field(default_factory=list)
    inventory: dict[str, Any] = field(default_factory=dict)
    seasonality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count_raw": self.row_count_raw,
            "issues": [asdict(i) for i in self.issues],
            "inventory": self.inventory,
            "seasonality": self.seasonality,
        }

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    def summary_lines(self) -> list[str]:
        lines = [
            f"QA report: {self.row_count_raw} raw rows, {len(self.issues)} issue groups",
        ]
        for issue in self.issues:
            ch = issue.channel or "all"
            lines.append(
                f"  [{issue.severity}] {issue.code} ({ch}): n={issue.count} - {issue.detail}"
            )
        return lines


def validate_panel(panel: pd.DataFrame) -> QAReport:
    """Run consistency checks on a unified (pre-clean) panel."""
    report = QAReport(row_count_raw=int(len(panel)))
    if panel.empty:
        report.issues.append(
            Issue("empty_panel", "error", None, 0, "No rows after ingest")
        )
        return report

    # --- Schema / nulls ---
    null_dates = int(panel["date"].isna().sum())
    if null_dates:
        report.issues.append(
            Issue("null_dates", "error", None, null_dates, "Rows with unparseable dates")
        )

    for col in ("spend", "revenue", "clicks", "impressions"):
        n = int(panel[col].isna().sum())
        if n:
            report.issues.append(
                Issue(f"null_{col}", "warn", None, n, f"Null {col} values")
            )

    # --- Duplicates ---
    key = ["channel", "campaign_id", "date"]
    dup_mask = panel.duplicated(key, keep=False)
    dup_n = int(dup_mask.sum())
    if dup_n:
        report.issues.append(
            Issue(
                "duplicate_campaign_date",
                "warn",
                None,
                dup_n,
                "Duplicate (channel, campaign_id, date) rows - will be aggregated in clean",
            )
        )

    # --- Negatives ---
    for col in ("spend", "revenue", "clicks", "impressions", "conversions", "daily_budget"):
        if col not in panel.columns:
            continue
        neg = int((panel[col] < 0).sum())
        if neg:
            report.issues.append(
                Issue(
                    f"negative_{col}",
                    "warn",
                    None,
                    neg,
                    f"Negative {col}; clipped to 0 in clean",
                )
            )

    # --- Per-channel checks ---
    for channel in CHANNELS:
        grp = panel[panel["channel"] == channel]
        if grp.empty:
            report.issues.append(
                Issue("missing_channel", "error", channel, 0, "No rows for channel")
            )
            continue

        budget_na = int(grp["daily_budget"].isna().sum())
        if budget_na:
            report.issues.append(
                Issue(
                    "missing_daily_budget",
                    "info" if channel == "meta" else "warn",
                    channel,
                    budget_na,
                    "Missing daily_budget (common on Meta); left as NaN, fill rate undefined",
                )
            )

        # Spend with zero/missing budget
        has_spend = grp["spend"] > 0
        bad_budget = has_spend & (grp["daily_budget"].isna() | (grp["daily_budget"] <= 0))
        n_bad = int(bad_budget.sum())
        if n_bad:
            report.issues.append(
                Issue(
                    "spend_without_budget",
                    "info",
                    channel,
                    n_bad,
                    "Active spend days with missing or non-positive daily_budget",
                )
            )

                # Zero-activity calendars (kept - useful for seasonality)
        zero_day = (
            (grp["spend"].fillna(0) == 0)
            & (grp["revenue"].fillna(0) == 0)
            & (grp["clicks"].fillna(0) == 0)
        )
        n_zero = int(zero_day.sum())
        if n_zero:
            report.issues.append(
                Issue(
                    "zero_activity_rows",
                    "info",
                    channel,
                    n_zero,
                    "All-zero metric rows retained for calendar continuity",
                )
            )

        # Extreme daily ROAS (tiny spend, large revenue) - informative for modeling caps
        spend_pos = grp["spend"] > 0
        if spend_pos.any():
            daily_roas = grp.loc[spend_pos, "revenue"] / grp.loc[spend_pos, "spend"]
            extreme = int((daily_roas > 50).sum())
            if extreme:
                report.issues.append(
                    Issue(
                        "extreme_daily_roas",
                        "info",
                        channel,
                        extreme,
                        "Daily ROAS > 50; consider Winsorizing in features/model",
                    )
                )

        # Google micros sanity: typical CPC shouldn't look like millions after /1e6
        if channel == "google":
            active = grp[grp["clicks"] > 0]
            if not active.empty:
                cpc = active["spend"] / active["clicks"]
                absurd = int((cpc > 500).sum())
                if absurd:
                    report.issues.append(
                        Issue(
                            "google_cpc_outlier",
                            "warn",
                            channel,
                            absurd,
                            "CPC > 500 after micros->currency; check cost scale",
                        )
                    )

    report.inventory = build_inventory(panel)
    report.seasonality = build_seasonality_hints(panel)
    return report


def build_inventory(panel: pd.DataFrame) -> dict[str, Any]:
    """Channel / type / campaign coverage for EDA."""
    inv: dict[str, Any] = {
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "channels": {},
        "campaign_types": {},
    }

    for channel, grp in panel.groupby("channel"):
        active_days = int(((grp["spend"] > 0) | (grp["revenue"] > 0)).sum())
        inv["channels"][channel] = {
            "rows": int(len(grp)),
            "campaigns": int(grp["campaign_id"].nunique()),
            "campaign_types": sorted(grp["campaign_type"].dropna().unique().tolist()),
            "date_min": str(grp["date"].min().date()),
            "date_max": str(grp["date"].max().date()),
            "n_calendar_days": int(grp["date"].nunique()),
            "active_metric_rows": active_days,
            "sparsity_pct": round(100.0 * (1 - active_days / max(len(grp), 1)), 2),
            "spend": float(grp["spend"].sum(skipna=True)),
            "revenue": float(grp["revenue"].sum(skipna=True)),
            "blended_roas": float(
                grp["revenue"].sum(skipna=True) / grp["spend"].sum(skipna=True)
                if grp["spend"].sum(skipna=True) > 0
                else 0.0
            ),
        }

    type_counts = (
        panel.groupby(["channel", "campaign_type"], dropna=False)
        .agg(
            rows=("campaign_id", "size"),
            campaigns=("campaign_id", "nunique"),
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
        )
        .reset_index()
    )
    inv["campaign_types"] = type_counts.to_dict(orient="records")
    return inv


def build_seasonality_hints(panel: pd.DataFrame) -> dict[str, Any]:
    """
    Cheap seasonality peek: monthly revenue share per channel.
    Flags months that are >1.5x that channel's median monthly revenue.
    """
    tmp = panel.copy()
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    hints: dict[str, Any] = {"peak_months": {}, "monthly_revenue": {}}

    for channel, grp in tmp.groupby("channel"):
        monthly = grp.groupby("month", sort=True)["revenue"].sum()
        hints["monthly_revenue"][channel] = {
            str(k): float(v) for k, v in monthly.items()
        }
        if monthly.empty:
            continue
        median = float(monthly.median())
        if median <= 0:
            continue
        peaks = monthly[monthly > 1.5 * median]
        hints["peak_months"][channel] = [
            {"month": str(idx), "revenue": float(val), "vs_median": round(float(val / median), 2)}
            for idx, val in peaks.items()
        ]
    return hints


def print_eda(report: QAReport) -> None:
    print("\n".join(report.summary_lines()))
    inv = report.inventory
    if not inv:
        return
    print(
        f"Inventory: {inv.get('date_min')} -> {inv.get('date_max')}"
    )
    for channel, stats in inv.get("channels", {}).items():
        print(
            f"  {channel}: campaigns={stats['campaigns']} "
            f"sparsity={stats['sparsity_pct']}% "
            f"ROAS={stats['blended_roas']:.3f} "
            f"types={stats['campaign_types']}"
        )
    peaks = report.seasonality.get("peak_months", {})
    for channel, peak_list in peaks.items():
        if peak_list:
            labels = ", ".join(
                f"{p['month']}({p['vs_median']}x)" for p in peak_list[:6]
            )
            print(f"  seasonality peaks [{channel}]: {labels}")
