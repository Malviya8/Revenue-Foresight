"""
LLM causal-insight layer (demo only; never used by run.sh).

Providers (free-first):
  - Groq   → GROQ_API_KEY  (https://console.groq.com)  [default in auto]
  - OpenAI → OPENAI_API_KEY
  - heuristic offline fallback (no network)
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


def build_insight_context(
    panel: pd.DataFrame,
    baseline: pd.DataFrame,
    scenario: pd.DataFrame | None,
    qa_inventory: dict[str, Any] | None,
    horizon: int,
    multipliers: dict[str, float],
) -> dict[str, Any]:
    """Pack numeric facts the LLM (or fallback) must cite — no invention."""
    b = baseline[baseline["horizon_days"] == horizon].copy()
    s = (
        scenario[scenario["horizon_days"] == horizon].copy()
        if scenario is not None
        else None
    )

    agg_b = b[b["level"] == "aggregate"]
    channels_b = b[b["level"] == "channel"].sort_values("p50_revenue", ascending=False)

    def _agg_block(df: pd.DataFrame) -> dict[str, float]:
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            "assumed_spend": float(row["assumed_spend"]),
            "p10_revenue": float(row["p10_revenue"]),
            "p50_revenue": float(row["p50_revenue"]),
            "p90_revenue": float(row["p90_revenue"]),
            "p50_roas": float(row["p50_roas"]),
        }

    channel_rows = []
    for _, row in channels_b.iterrows():
        ch = str(row["channel"])
        item = {
            "channel": ch,
            "assumed_spend": float(row["assumed_spend"]),
            "p50_revenue": float(row["p50_revenue"]),
            "p10_revenue": float(row["p10_revenue"]),
            "p90_revenue": float(row["p90_revenue"]),
            "p50_roas": float(row["p50_roas"]),
            "multiplier": float(multipliers.get(ch, 1.0)),
        }
        if s is not None:
            sr = s[(s["level"] == "channel") & (s["channel"] == ch)]
            if not sr.empty:
                item["scenario_assumed_spend"] = float(sr.iloc[0]["assumed_spend"])
                item["scenario_p50_revenue"] = float(sr.iloc[0]["p50_revenue"])
                item["scenario_p50_roas"] = float(sr.iloc[0]["p50_roas"])
                if item["assumed_spend"] > 0:
                    item["spend_delta_pct"] = 100.0 * (
                        item["scenario_assumed_spend"] / item["assumed_spend"] - 1.0
                    )
                if item["p50_revenue"] > 0:
                    item["revenue_delta_pct"] = 100.0 * (
                        item["scenario_p50_revenue"] / item["p50_revenue"] - 1.0
                    )
        channel_rows.append(item)

    # Uncertainty drivers: widest relative interval at campaign level
    camps = b[b["level"] == "campaign"].copy()
    if not camps.empty and (camps["p50_revenue"] > 0).any():
        camps["width_rel"] = (camps["p90_revenue"] - camps["p10_revenue"]) / camps[
            "p50_revenue"
        ].clip(lower=1.0)
        top_uncertain = (
            camps.sort_values("width_rel", ascending=False)
            .head(5)[
                [
                    "channel",
                    "campaign_name",
                    "p10_revenue",
                    "p50_revenue",
                    "p90_revenue",
                    "width_rel",
                ]
            ]
            .to_dict(orient="records")
        )
    else:
        top_uncertain = []

    # Simple anomaly hints from recent panel
    anomalies: list[dict[str, Any]] = []
    if panel is not None and not panel.empty:
        tmp = panel.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        cutoff = tmp["date"].max() - pd.Timedelta(days=28)
        recent = tmp[tmp["date"] >= cutoff]
        for ch, grp in recent.groupby("channel"):
            daily = grp.groupby("date")["roas"].mean()
            if len(daily) < 7:
                continue
            mu, sigma = float(daily.mean()), float(daily.std(ddof=0))
            if sigma <= 0:
                continue
            last = float(daily.iloc[-1]) if len(daily) else mu
            z = (last - mu) / sigma
            if abs(z) >= 2.0:
                anomalies.append(
                    {
                        "channel": ch,
                        "metric": "daily_roas_28d",
                        "z_score": round(z, 2),
                        "last": round(last, 3),
                        "mean": round(mu, 3),
                    }
                )

    ctx: dict[str, Any] = {
        "horizon_days": horizon,
        "baseline_aggregate": _agg_block(agg_b),
        "scenario_aggregate": _agg_block(s[s["level"] == "aggregate"]) if s is not None else None,
        "channels": channel_rows,
        "budget_multipliers": multipliers,
        "top_uncertain_campaigns": top_uncertain,
        "anomalies": anomalies,
        "qa_channel_inventory": (qa_inventory or {}).get("channels", {}),
        "assumptions": [
            "Platform attribution used as-is (no custom MMM).",
            "Forecasts are probabilistic period aggregates (P10/P50/P90).",
            "Budget scenarios adjust planned_spend features; model may show diminishing ROAS.",
            "Aggregate revenue is reconciled as the sum of channel forecasts.",
        ],
    }
    if s is not None and ctx["baseline_aggregate"] and ctx["scenario_aggregate"]:
        b50 = ctx["baseline_aggregate"].get("p50_revenue") or 0.0
        s50 = ctx["scenario_aggregate"].get("p50_revenue") or 0.0
        if b50 > 0:
            ctx["aggregate_revenue_delta_pct"] = 100.0 * (s50 / b50 - 1.0)
    return ctx


def heuristic_insights(ctx: dict[str, Any]) -> str:
    """Deterministic briefing when no LLM key is available."""
    h = ctx["horizon_days"]
    agg = ctx.get("baseline_aggregate") or {}
    lines = [
        f"## Causal briefing ({h}-day horizon)",
        "",
        "### Baseline outlook",
        (
            f"- Aggregate assumed spend **${agg.get('assumed_spend', 0):,.0f}** with "
            f"P50 revenue **${agg.get('p50_revenue', 0):,.0f}** "
            f"(P10–P90: ${agg.get('p10_revenue', 0):,.0f} – ${agg.get('p90_revenue', 0):,.0f}), "
            f"blended P50 ROAS **{agg.get('p50_roas', 0):.2f}**."
        ),
        "",
        "### Channel contribution",
    ]
    for ch in ctx.get("channels", []):
        lines.append(
            f"- **{ch['channel']}**: spend ${ch['assumed_spend']:,.0f} → "
            f"P50 ${ch['p50_revenue']:,.0f} @ ROAS {ch['p50_roas']:.2f} "
            f"(range ${ch['p10_revenue']:,.0f}–${ch['p90_revenue']:,.0f})."
        )

    if ctx.get("scenario_aggregate"):
        d = ctx.get("aggregate_revenue_delta_pct")
        sagg = ctx["scenario_aggregate"]
        lines += [
            "",
            "### Budget scenario effect",
            f"- Multipliers: `{ctx.get('budget_multipliers')}`",
            (
                f"- Scenario aggregate P50 **${sagg.get('p50_revenue', 0):,.0f}** "
                f"(Δ {d:+.2f}% vs baseline)."
                if d is not None
                else f"- Scenario aggregate P50 **${sagg.get('p50_revenue', 0):,.0f}**."
            ),
        ]
        for ch in ctx.get("channels", []):
            if "revenue_delta_pct" in ch:
                lines.append(
                    f"- {ch['channel']}: spend {ch.get('spend_delta_pct', 0):+.1f}% → "
                    f"revenue {ch['revenue_delta_pct']:+.1f}% "
                    f"(ROAS {ch['p50_roas']:.2f} → {ch.get('scenario_p50_roas', ch['p50_roas']):.2f})."
                )
                if ch.get("spend_delta_pct", 0) > 1 and ch.get("revenue_delta_pct", 0) < ch.get(
                    "spend_delta_pct", 0
                ):
                    lines.append(
                        f"  - Interpretation: **diminishing returns** on {ch['channel']} — "
                        f"incremental spend is not matching baseline efficiency."
                    )

    if ctx.get("anomalies"):
        lines += ["", "### Recent anomalies (z≥2 on 28d ROAS)"]
        for a in ctx["anomalies"]:
            lines.append(
                f"- {a['channel']}: z={a['z_score']} (last {a['last']} vs mean {a['mean']})"
            )

    if ctx.get("top_uncertain_campaigns"):
        lines += ["", "### Uncertainty concentration"]
        for c in ctx["top_uncertain_campaigns"][:3]:
            lines.append(
                f"- {c.get('channel')} / {c.get('campaign_name')}: "
                f"width/P50={float(c.get('width_rel', 0)):.2f}"
            )

    lines += [
        "",
        "### Operational risks",
        "- Sparse / volatile campaigns can dominate interval width — prefer channel or type planning first.",
        "- Holiday seasonality (Nov–Dec) can shift ROAS; treat out-of-season extrapolations cautiously.",
        "- Attribution is platform-native; cross-channel totals may overlap users.",
        "",
        "### Assumptions",
    ]
    for a in ctx.get("assumptions", []):
        lines.append(f"- {a}")
    lines += [
        "",
        "_Generated by heuristic engine (no LLM key). Numbers cited from model outputs only._",
    ]
    return "\n".join(lines)


def _chat_completions(
    ctx: dict[str, Any],
    *,
    api_key: str,
    model: str,
    base_url: str,
) -> str:
    from urllib import request

    system = (
        "You are a senior ecommerce media strategist. Write a concise causal briefing. "
        "ONLY cite numbers present in the JSON context. Do not invent metrics. "
        "Explain why performance may rise/fall, call out diminishing returns, "
        "uncertainty drivers, and 2-3 concrete budget reallocation ideas. "
        "Use markdown with short sections. Include an Assumptions bullet list from context."
    )
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "Context JSON:\n" + json.dumps(ctx, default=str)[:120000],
            },
        ],
    }
    endpoint = base_url.rstrip("/") + "/chat/completions"
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


# Free-tier default: Groq (https://console.groq.com) — OpenAI-compatible API.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


def generate_insights(
    ctx: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str | None = None,
    provider: str = "auto",
) -> tuple[str, str]:
    """
    Returns (markdown, engine_label).

    provider:
      - auto      → Groq free tier, then OpenAI, then heuristic
      - groq      → Groq only (GROQ_API_KEY or sidebar key)
      - openai    → OpenAI only
      - heuristic → no network
    """
    provider = (provider or "auto").lower().strip()
    if provider == "heuristic":
        return heuristic_insights(ctx), "heuristic"

    env_groq = (os.environ.get("GROQ_API_KEY") or "").strip()
    env_openai = (os.environ.get("OPENAI_API_KEY") or "").strip()
    sidebar = (api_key or "").strip()

    # (engine_label, key, base_url, default_model)
    candidates: list[tuple[str, str, str, str]] = []

    if provider in ("auto", "groq"):
        groq_key = env_groq or (sidebar if provider == "groq" else "")
        if provider == "auto" and not groq_key and sidebar and not env_openai:
            groq_key = sidebar  # free path: treat lone sidebar key as Groq
        if groq_key:
            candidates.append(("groq", groq_key, GROQ_BASE_URL, GROQ_DEFAULT_MODEL))

    if provider in ("auto", "openai"):
        openai_key = env_openai or (sidebar if provider == "openai" else "")
        if openai_key:
            candidates.append(
                ("openai", openai_key, OPENAI_BASE_URL, OPENAI_DEFAULT_MODEL)
            )

    errors: list[str] = []
    for label, key, base, default_model in candidates:
        use_model = model or default_model
        try:
            text = _chat_completions(ctx, api_key=key, model=use_model, base_url=base)
            return text, label
        except Exception as exc:  # noqa: BLE001 — demo must not crash
            errors.append(f"{label}: {type(exc).__name__}")

    fallback = heuristic_insights(ctx)
    if errors:
        fallback += (
            "\n\n---\n_LLM call failed ("
            + "; ".join(errors)
            + "); showing heuristic briefing._"
        )
    return fallback, "heuristic"
