"""
Revenue Foresight — Streamlit demo entrypoint.

Run:
  pip install -r requirements-demo.txt
  streamlit run app.py

Scoring path (run.sh) does not use this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from budget import apply_budget_scenario  # noqa: E402
from features import build_inference_features  # noqa: E402
from ingest import load_cleaned_panel  # noqa: E402
from llm_layer import (  # noqa: E402
    build_insight_context,
    generate_insights,
    heuristic_insights,
)
from predict import predict_frame  # noqa: E402
from validate import validate_panel  # noqa: E402

ORANGE = "#FF6A3D"
ORANGE_SOFT = "#FF9A74"
MUTED = "#9A9AA3"
WHITE = "#ECECEE"
GRID = "rgba(255,255,255,0.07)"
HOVER_BG = "rgba(16,16,20,0.94)"

CHANNEL_LABELS = {
    "google": "Google Ads",
    "meta": "Meta Ads",
    "bing": "Microsoft Ads",
}

ISSUE_COPY = {
    "missing_daily_budget": (
        "Some days have no daily budget",
        "The platform did not send a budget for those days. Forecasts still use actual spend and revenue.",
    ),
    "spend_without_budget": (
        "Spend with no budget number",
        "Ads ran, but the budget field was blank or zero. We treat this as a reporting gap, not a model error.",
    ),
    "zero_activity_rows": (
        "Quiet calendar days",
        "Days with no spend or revenue are kept so the calendar stays complete. Sparse channels (especially Bing) show more of these.",
    ),
    "extreme_daily_roas": (
        "Very high single-day ROAS",
        "A few days look like huge returns on tiny spend. The model caps these so they do not dominate the forecast.",
    ),
    "google_cpc_outlier": (
        "Unusual Google cost per click",
        "A handful of days have CPC far above a normal range after converting micros to currency. Worth a quick data check.",
    ),
    "null_dates": (
        "Rows with missing dates",
        "Those rows cannot be placed on the timeline and are dropped in cleaning.",
    ),
    "duplicate_campaign_date": (
        "Duplicate campaign-day rows",
        "Cleaning will add them up so each campaign has one row per day.",
    ),
    "empty_panel": (
        "No rows loaded",
        "Check that the data folder has Google, Meta, and Microsoft Ads CSVs.",
    ),
    "missing_channel": (
        "A channel is missing",
        "That ads platform has no rows in this dataset.",
    ),
}


st.set_page_config(
    page_title="Revenue Foresight",
    page_icon=str(ROOT / "static" / "logo-mark.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _read_asset(name: str, stamp: float) -> str:
    # stamp is part of the cache key on purpose. It must not be named with a
    # leading underscore, which is how Streamlit marks an argument as unhashable.
    del stamp
    return (ROOT / "assets" / name).read_text(encoding="utf-8")


def _asset(name: str) -> str:
    """Streamlit only watches .py files, so key the cache on mtime instead."""
    path = ROOT / "assets" / name
    return _read_asset(name, path.stat().st_mtime)


st.markdown(f"<style>{_asset('styles.css')}</style>", unsafe_allow_html=True)


# Streamlit runs components in a sandboxed iframe that is torn down on rerun, so
# motion.js is copied into the app document itself where its observers survive.
_MOTION_BOOTSTRAP = """
<script>
(function () {
  var win = window.parent;
  if (!win || win.__rfMotionInjected) return;
  win.__rfMotionInjected = true;
  var el = win.document.createElement("script");
  el.textContent = __SOURCE__;
  win.document.head.appendChild(el);
})();
</script>
"""

def _logo(*, large: bool = False) -> str:
    """Served from ./static so the browser caches it instead of re-parsing base64."""
    return (
        f'<span class="rf-logo{" rf-logo-lg" if large else ""}">'
        '<img src="app/static/logo.png" alt="Revenue Foresight" draggable="false">'
        "</span>"
    )


_BOOT_VEIL = f"""
<div class="rf-veil">
  <div class="rf-veil-inner">
    {_logo(large=True)}
    <div class="rf-veil-bar"></div>
    <p class="rf-veil-note">Warming up the outlook…</p>
  </div>
</div>
"""


def _boot_motion() -> None:
    source = json.dumps(_asset("motion.js")).replace("</", "<\\/")
    components.html(_MOTION_BOOTSTRAP.replace("__SOURCE__", source), height=0, width=0)


def _boot_veil() -> None:
    """Cover the first paint once per session; the fade-out is pure CSS."""
    if st.session_state.get("rf_veil_shown"):
        return
    st.session_state.rf_veil_shown = True
    st.markdown(_BOOT_VEIL, unsafe_allow_html=True)


_NAV_JS = r"""
(function () {
  const win = window.parent;
  const doc = win.document;
  const cmd = "__CMD__";
  const mobileMq = win.matchMedia("(max-width: 768px)");

  function isMobile() {
    return mobileMq.matches;
  }

  function sidebar() {
    return doc.querySelector('[data-testid="stSidebar"]');
  }

  function isExpanded() {
    const sb = sidebar();
    return !!(sb && sb.getAttribute("aria-expanded") === "true");
  }

  function collapse() {
    const sb = sidebar();
    if (!sb || !isExpanded()) return;
    const btn =
      sb.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
      doc.querySelector('[data-testid="stSidebarCollapseButton"] button');
    if (btn) btn.click();
  }

  function expand() {
    if (isExpanded()) return;
    const btn = doc.querySelector('[data-testid="stSidebarCollapsedControl"] button');
    if (btn) btn.click();
  }

  function hydrate() {
    doc.documentElement.classList.add("rf-hydrated");
  }

  if (!win.__rfNav) {
    win.__rfNav = { booted: false };
    const start = Date.now();
    const boot = win.setInterval(function () {
      const sb = sidebar();
      if (!sb && Date.now() - start < 4000) return;
      win.clearInterval(boot);
      if (isMobile()) {
        collapse();
        win.setTimeout(collapse, 80);
      } else {
        expand();
      }
      win.setTimeout(hydrate, 120);
    }, 40);
    win.setTimeout(hydrate, 1600);
  }

  if (cmd === "close" && isMobile()) {
    doc.documentElement.classList.add("rf-force-close");
    collapse();
    win.setTimeout(collapse, 60);
    win.setTimeout(collapse, 180);
    win.setTimeout(function () {
      collapse();
      doc.documentElement.classList.remove("rf-force-close");
      hydrate();
    }, 480);
  }
})();
"""

def _queue_forecast() -> None:
    st.session_state["_forecast_phase"] = "close"


def _inject_nav(cmd: str) -> None:
    tick = int(st.session_state.get("_nav_tick", 0))
    script = _NAV_JS.replace("__CMD__", cmd)
    components.html(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
        f"<body><script>/* {cmd}-{tick} */{script}</script></body></html>",
        height=0,
        width=0,
    )


def _close_sidebar_now() -> None:
    # No inline <style> here: an injected block lives for the rest of the session and
    # would pin the sidebar shut, so reopening it via Plan would silently do nothing.
    # styles.css handles the slide-out, gated on the html.rf-force-close class that
    # _NAV_JS adds and then removes.
    st.session_state["_nav_tick"] = int(st.session_state.get("_nav_tick", 0)) + 1
    _inject_nav("close")


@st.cache_data(show_spinner=False)
def _load_panel(data_dir: str):
    panel, _report = load_cleaned_panel(data_dir)
    from ingest import load_unified_panel

    unified = load_unified_panel(data_dir)
    qa = validate_panel(unified)
    return panel, qa


@st.cache_resource(show_spinner=False)
def _model_path(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return str(p)


def _run_forecast(panel: pd.DataFrame, model_path: str, multipliers: dict[str, float]) -> pd.DataFrame:
    features = build_inference_features(panel)
    if any(abs(v - 1.0) > 1e-9 for v in multipliers.values()):
        features = apply_budget_scenario(
            features, {"mode": "multiplier", "channel_spend_multipliers": multipliers}
        )
    return predict_frame(features, Path(model_path), reconcile=True)


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def _channel_name(code: str) -> str:
    return CHANNEL_LABELS.get(str(code).lower(), str(code).title())


def _ordered_channels(codes) -> list[str]:
    """Google, Meta, Microsoft first so tables and charts never disagree."""
    present = [str(c).lower() for c in codes]
    known = [c for c in CHANNEL_LABELS if c in present]
    return known + sorted(c for c in present if c not in CHANNEL_LABELS)


def _is_baseline_mix(multipliers: dict[str, float]) -> bool:
    return all(abs(float(v) - 1.0) <= 1e-9 for v in multipliers.values())


def _plotly_theme(fig: go.Figure, *, height: int = 320, y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=64, r=16, t=56, b=48),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="DM Sans, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0, font=dict(size=12)),
        bargap=0.32,
        barcornerradius=6,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=HOVER_BG,
            bordercolor="rgba(255,106,61,0.45)",
            font=dict(color=WHITE, family="DM Sans, sans-serif", size=13),
        ),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, color=MUTED, title_standoff=12)
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        color=MUTED,
        title=dict(text=y_title, font=dict(size=12, color=MUTED)),
        tickprefix="$",
        separatethousands=True,
        tickformat="~s",
        rangemode="tozero",
        automargin=True,
    )
    return fig


def _band_chart(p10: float, p50: float, p90: float, horizon: int) -> go.Figure:
    """Horizontal P10–P90 band with the P50 marked, for the no-scenario view."""
    fig = go.Figure()
    half = 0.17

    fig.add_scatter(
        x=[p10, p90, p90, p10, p10],
        y=[-half, -half, half, half, -half],
        mode="lines",
        line=dict(color="rgba(255,154,116,0.50)", width=1),
        fill="toself",
        fillgradient=dict(
            type="horizontal",
            colorscale=[
                [0.0, "rgba(224,62,18,0.34)"],
                [0.5, "rgba(255,106,61,0.46)"],
                [1.0, "rgba(255,154,116,0.32)"],
            ],
        ),
        hoverinfo="skip",
        showlegend=False,
    )

    # Layered strokes stand in for a glow around the planning number.
    for width, color in (
        (14, "rgba(255,106,61,0.10)"),
        (7, "rgba(255,106,61,0.26)"),
        (2.6, ORANGE_SOFT),
    ):
        fig.add_scatter(
            x=[p50, p50],
            y=[-half * 1.45, half * 1.45],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            showlegend=False,
        )

    fig.add_scatter(
        x=[p10, p50, p90],
        y=[0, 0, 0],
        mode="markers",
        marker=dict(size=20, color="rgba(0,0,0,0)"),
        customdata=[
            f"P10 · cautious floor {_fmt_money(p10)}",
            f"P50 · planning number {_fmt_money(p50)}",
            f"P90 · optimistic ceiling {_fmt_money(p90)}",
        ],
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    )

    for value, label, color, shift in (
        (p10, f"P10 {_fmt_money(p10)}", MUTED, -46),
        (p50, f"P50 {_fmt_money(p50)}", "#F5F5F8", 46),
        (p90, f"P90 {_fmt_money(p90)}", MUTED, -46),
    ):
        fig.add_annotation(
            x=value,
            y=0,
            yshift=shift,
            text=label,
            showarrow=False,
            font=dict(size=13, color=color, family="DM Sans, sans-serif"),
        )

    span = max(p90 - p10, 1.0)
    fig.update_layout(
        height=235,
        margin=dict(l=16, r=16, t=38, b=52),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="DM Sans, sans-serif", size=13),
        hoverlabel=dict(
            bgcolor=HOVER_BG,
            bordercolor="rgba(255,106,61,0.45)",
            font=dict(color=WHITE, family="DM Sans, sans-serif", size=13),
        ),
    )
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, range=[-0.62, 0.62])
    fig.update_xaxes(
        range=[p10 - span * 0.35, p90 + span * 0.35],
        gridcolor=GRID,
        zeroline=False,
        color=MUTED,
        tickprefix="$",
        tickformat="~s",
        title=dict(
            text=f"Forecasted revenue (USD) · next {horizon} days",
            font=dict(size=12, color=MUTED),
        ),
    )
    return fig


def _hero(*, horizon: int | None = None, multipliers: dict[str, float] | None = None) -> None:
    """Full hero on the landing; a compact status band once a forecast exists."""
    if horizon is None or multipliers is None:
        st.markdown(
            f"""
            <div class="rf-hero">
              {_logo()}
              <h1>See the next 30–90 days before you spend.</h1>
              <p class="rf-lede">Probabilistic revenue and ROAS across Google, Meta, and Microsoft Ads — a likely number, plus a cautious and optimistic range.</p>
              <div class="rf-status">
                <span class="rf-chip">Google Ads</span>
                <span class="rf-chip">Meta Ads</span>
                <span class="rf-chip">Microsoft Ads</span>
                <span class="rf-chip"><em>P10–P50–P90</em></span>
                <span class="rf-chip">30 / 60 / 90 days</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if _is_baseline_mix(multipliers):
        mix = '<span class="rf-chip">Current spend mix</span>'
    else:
        mix = "".join(
            f'<span class="rf-chip">{_channel_name(code)} '
            f'<em>{float(multipliers[code]):.2f}×</em></span>'
            for code in _ordered_channels(multipliers)
            if abs(float(multipliers[code]) - 1.0) > 1e-9
        )
    st.markdown(
        f"""
        <div class="rf-hero rf-compact">
          {_logo()}
          <h1>The next {horizon} days, before you spend.</h1>
          <div class="rf-status">
            <span class="rf-chip"><em>{horizon}-day</em> outlook</span>
            {mix}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _issue_title(issue) -> tuple[str, str]:
    title, why = ISSUE_COPY.get(issue.code, (issue.code.replace("_", " ").title(), issue.detail))
    return title, why


def _render_empty_state(data_dir: str) -> None:
    """Fallback view when no forecast exists yet (or scoring failed)."""
    try:
        _panel, qa = _load_panel(data_dir)
        inv = qa.inventory.get("channels", {}) if qa else {}
    except Exception:
        inv = {}

    if inv:
        snaps = "".join(
            (
                '<div class="rf-card">'
                f'<p class="rf-kicker">{_channel_name(code)}</p>'
                f'<p class="rf-big">{int(inv[code]["campaigns"])} campaigns</p>'
                '<p class="rf-muted">'
                f'Historic ROAS <span class="rf-orange">{inv[code]["blended_roas"]:.2f}</span>'
                f' · {100.0 - float(inv[code].get("sparsity_pct", 0)):.0f}% active days'
                "</p></div>"
            )
            for code in _ordered_channels(inv)
        )
        st.markdown(f'<div class="rf-landing-stats">{snaps}</div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="rf-cta-note">Sample exports are already loaded. Open '
        '<span class="rf-orange">Plan</span> to change the horizon or spend mix.</p>',
        unsafe_allow_html=True,
    )
    st.button(
        "Run forecast",
        type="primary",
        use_container_width=True,
        on_click=_queue_forecast,
        key="run_main",
    )


def _render_qa(qa) -> None:
    st.subheader("Is the data in good shape?")
    st.caption("A quick health check before you trust the forecast. Notes are not always problems — many are how ads platforms report data.")

    issues = list(qa.issues)
    n_error = sum(1 for i in issues if i.severity == "error")
    n_warn = sum(1 for i in issues if i.severity == "warn")
    n_info = sum(1 for i in issues if i.severity == "info")
    inv = qa.inventory.get("channels", {})

    if n_error:
        headline = "Fix these before planning"
        pill = "error"
        pill_label = "Blocked"
    elif n_warn:
        headline = "Ready to forecast — a few items to notice"
        pill = "warn"
        pill_label = "Good, with notes"
    else:
        headline = "Ready to forecast"
        pill = "ok"
        pill_label = "Healthy"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows loaded", f"{qa.row_count_raw:,}")
    k2.metric("Needs a look", str(n_warn + n_error))
    k3.metric("Context notes", str(n_info))
    k4.metric("Channels", str(len(inv) or 0))

    st.markdown(
        f'<span class="rf-pill {pill}">{pill_label}</span> <span class="rf-muted">{headline}</span>',
        unsafe_allow_html=True,
    )

    if inv:
        st.markdown("##### How complete is each channel?")
        codes = _ordered_channels(inv)
        cols = st.columns(len(codes))
        for col, ch in zip(cols, codes):
            stats = inv[ch]
            active = 100.0 - float(stats.get("sparsity_pct", 0))
            with col:
                st.markdown(
                    f"""
                    <div class="rf-card">
                      <p class="rf-kicker">{_channel_name(ch)}</p>
                      <p class="rf-big">{int(stats['campaigns'])} campaigns</p>
                      <p class="rf-muted">
                        Historic ROAS <span class="rf-orange">{stats['blended_roas']:.2f}</span>
                        · {active:.0f}% active days<br/>
                        {stats['date_min']} → {stats['date_max']}
                      </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if issues:
        st.markdown("##### What we noticed")
        left, right = st.columns(2)
        priority = [i for i in issues if i.severity in ("error", "warn")]
        notes = [i for i in issues if i.severity == "info"]
        buckets = [(left, priority, "Worth a look"), (right, notes, "Context, not blockers")]
        for col, bucket, heading in buckets:
            with col:
                st.markdown(f"**{heading}**")
                if not bucket:
                    st.markdown(
                        '<div class="rf-card"><p class="rf-muted">Nothing in this group.</p></div>',
                        unsafe_allow_html=True,
                    )
                    continue
                for issue in bucket:
                    title, why = _issue_title(issue)
                    ch = _channel_name(issue.channel) if issue.channel else "All channels"
                    st.markdown(
                        f"""
                        <div class="rf-issue {issue.severity}">
                          <span class="rf-pill {issue.severity}">{issue.severity}</span>
                          <strong>{title}</strong>
                          <p class="rf-muted">{ch} · {issue.count:,} rows<br/>{why}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    peaks = qa.seasonality.get("peak_months", {})
    if peaks:
        st.markdown("##### Seasonal peaks")
        st.caption("Months where revenue was at least 1.5× that channel’s typical month — useful when you read a Q4-heavy forecast.")
        bits = []
        for ch, plist in peaks.items():
            if not plist:
                continue
            labels = ", ".join(f"{p['month']} ({p['vs_median']}×)" for p in plist[:5])
            bits.append(f"**{_channel_name(ch)}:** {labels}")
        if bits:
            st.markdown("  \n".join(bits))


def _render_forecast(h: int, agg_b, agg_s, *, whatif: bool) -> None:
    st.subheader(f"What the next {h} days look like")
    if whatif:
        st.caption(
            "P50 is the most likely outcome. P10 is a cautious floor. P90 is an optimistic ceiling. "
            "The what-if column reflects the spend mix set in **Plan**."
        )
    else:
        st.caption(
            "P50 is the most likely outcome. P10 is a cautious floor. P90 is an optimistic ceiling. "
            "Move a spend slider in **Plan** to compare a what-if against this baseline."
        )

    hover = " %{y:$,.0f}<extra>%{fullData.name}</extra>"

    if whatif:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Likely revenue · current plan", _fmt_money(float(agg_b["p50_revenue"])))
        c2.metric(
            "Likely revenue · what-if",
            _fmt_money(float(agg_s["p50_revenue"])),
            delta=f"{100 * (float(agg_s['p50_revenue']) / max(float(agg_b['p50_revenue']), 1) - 1):+.1f}%",
        )
        c3.metric("Likely ROAS · current plan", f"{float(agg_b['p50_roas']):.2f}")
        c4.metric(
            "Likely ROAS · what-if",
            f"{float(agg_s['p50_roas']):.2f}",
            delta=f"{float(agg_s['p50_roas']) - float(agg_b['p50_roas']):+.2f}",
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f'<div class="rf-card"><h4>Cautious (P10)</h4><p class="rf-big">{_fmt_money(float(agg_s["p10_revenue"]))}</p>'
                f'<p class="rf-muted">Downside if the period is weak. What-if spend {_fmt_money(float(agg_s["assumed_spend"]))}.</p></div>',
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f'<div class="rf-card"><h4>Most likely (P50)</h4><p class="rf-big">{_fmt_money(float(agg_s["p50_revenue"]))}</p>'
                f'<p class="rf-muted">Planning number. Current plan was {_fmt_money(float(agg_b["p50_revenue"]))}.</p></div>',
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f'<div class="rf-card"><h4>Optimistic (P90)</h4><p class="rf-big">{_fmt_money(float(agg_s["p90_revenue"]))}</p>'
                f'<p class="rf-muted">Upside if the period is strong. Not a target — a ceiling.</p></div>',
                unsafe_allow_html=True,
            )

        fig = go.Figure()
        for name, key, color in (
            ("Cautious (P10)", "p10_revenue", "#6A6A6A"),
            ("Likely (P50)", "p50_revenue", ORANGE),
            ("Optimistic (P90)", "p90_revenue", ORANGE_SOFT),
        ):
            fig.add_bar(
                name=name,
                x=["Current plan", "What-if"],
                y=[float(agg_b[key]), float(agg_s[key])],
                marker_color=color,
                hovertemplate=hover,
            )
    else:
        p10 = float(agg_b["p10_revenue"])
        p50 = float(agg_b["p50_revenue"])
        p90 = float(agg_b["p90_revenue"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Likely revenue (P50)", _fmt_money(p50))
        c2.metric("Planning band (P10–P90)", f"{_fmt_money(p10)} – {_fmt_money(p90)}")
        c3.metric("Likely ROAS", f"{float(agg_b['p50_roas']):.2f}")
        st.caption(
            f"On assumed spend of {_fmt_money(float(agg_b['assumed_spend']))}. "
            "Plan on P50, stress-test with P10, and treat P90 as a ceiling rather than a target."
        )

        st.plotly_chart(
            _band_chart(p10, p50, p90, h),
            use_container_width=True,
        )

    if whatif:
        st.plotly_chart(
            _plotly_theme(
                fig,
                height=380,
                y_title=f"Forecasted revenue (USD) · next {h} days",
            ),
            use_container_width=True,
        )
        st.caption(
            "Y-axis is **store-level attributed revenue**, not spend and not ROAS. "
            "A 1.25× Google spend slider does not lift total revenue by 25% — only Google’s share of the mix changes, and returns diminish."
        )
    else:
        st.caption(
            "The bar is **store-level attributed revenue**, not spend and not ROAS. "
            "The shaded span is the P10–P90 planning band; the line is P50."
        )

    st.download_button(
        "Download this forecast as CSV" if not whatif else "Download this what-if as CSV",
        data=st.session_state.scenario.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_h{h}_{'baseline' if not whatif else 'scenario'}.csv",
        mime="text/csv",
    )


def _channel_table(
    base_ch: pd.DataFrame, scen_ch: pd.DataFrame, *, whatif: bool
) -> pd.DataFrame:
    labels = [_channel_name(c) for c in base_ch.index]
    if not whatif:
        return pd.DataFrame(
            {
                "Channel": labels,
                "Assumed spend": base_ch["assumed_spend"].to_numpy(),
                "Low (P10)": base_ch["p10_revenue"].to_numpy(),
                "Likely (P50)": base_ch["p50_revenue"].to_numpy(),
                "High (P90)": base_ch["p90_revenue"].to_numpy(),
                "Likely ROAS": base_ch["p50_roas"].to_numpy(),
            }
        )
    now = base_ch["p50_revenue"].to_numpy()
    whatif_rev = scen_ch["p50_revenue"].to_numpy()
    return pd.DataFrame(
        {
            "Channel": labels,
            "Spend now": base_ch["assumed_spend"].to_numpy(),
            "Spend what-if": scen_ch["assumed_spend"].to_numpy(),
            "Likely now": now,
            "Likely what-if": whatif_rev,
            "Change": [
                (float(w) / float(n) - 1.0) if float(n) else 0.0
                for n, w in zip(now, whatif_rev)
            ],
            "Band what-if (P10–P90)": [
                f"{_fmt_money(float(lo))} – {_fmt_money(float(hi))}"
                for lo, hi in zip(scen_ch["p10_revenue"], scen_ch["p90_revenue"])
            ],
            "ROAS what-if": scen_ch["p50_roas"].to_numpy(),
        }
    )


def _render_channels(b_h: pd.DataFrame, s_h: pd.DataFrame, *, whatif: bool) -> None:
    st.subheader("Where the revenue comes from")
    st.caption("Channel totals are the planning layer. They always add up to the store-level number.")

    order = _ordered_channels(b_h.loc[b_h["level"] == "channel", "channel"].unique())
    base_ch = b_h[b_h["level"] == "channel"].set_index("channel").reindex(order)
    scen_ch = s_h[s_h["level"] == "channel"].set_index("channel").reindex(order)

    table = _channel_table(base_ch, scen_ch, whatif=whatif)
    config: dict[str, object] = {}
    for col in table.columns:
        if col in ("Channel", "Band what-if (P10–P90)"):
            continue
        if col == "Change":
            config[col] = st.column_config.NumberColumn("Change", format="%+.1f%%")
        elif "ROAS" in col:
            config[col] = st.column_config.NumberColumn(col, format="%.2f")
        else:
            config[col] = st.column_config.NumberColumn(col, format="$%.0f")
    if "Change" in table.columns:
        table = table.assign(Change=table["Change"] * 100.0)
    st.dataframe(table, use_container_width=True, hide_index=True, column_config=config)

    labels = [_channel_name(c) for c in order]
    fig = go.Figure()
    if whatif:
        fig.add_bar(
            name="Current plan (P50)",
            x=labels,
            y=base_ch["p50_revenue"].tolist(),
            marker_color="#6A6A6A",
            hovertemplate=" %{y:$,.0f}<extra>Current plan</extra>",
        )
        fig.add_bar(
            name="What-if (P50)",
            x=labels,
            y=scen_ch["p50_revenue"].tolist(),
            marker_color=ORANGE,
            hovertemplate=" %{y:$,.0f}<extra>What-if</extra>",
        )
    else:
        p50 = base_ch["p50_revenue"].astype(float)
        fig.add_bar(
            name="Likely (P50)",
            x=labels,
            y=p50.tolist(),
            marker_color=ORANGE,
            error_y=dict(
                type="data",
                symmetric=False,
                array=(base_ch["p90_revenue"].astype(float) - p50).tolist(),
                arrayminus=(p50 - base_ch["p10_revenue"].astype(float)).tolist(),
                color=ORANGE_SOFT,
                thickness=1.6,
                width=18,
            ),
            hovertemplate=" %{y:$,.0f}<extra>Likely (P50)</extra>",
        )
        fig.update_layout(showlegend=False)

    st.plotly_chart(
        _plotly_theme(
            fig,
            height=360,
            y_title=f"Likely revenue (USD) · next {int(s_h['horizon_days'].iloc[0])} days",
        ),
        use_container_width=True,
    )
    st.caption(
        "Same Y-axis: likely (P50) attributed revenue. Microsoft Ads is small on this scale "
        "(a few thousand dollars vs Google’s hundreds of thousands) — it is not zero."
    )


def _render_campaigns(s_h: pd.DataFrame) -> None:
    st.subheader("Campaigns that move the number")
    st.caption("Use this to spot contributors and wide ranges. Sparse campaigns (often Bing) can look noisy — prefer channel totals for the decision.")
    channels = ["All channels"] + [
        _channel_name(c)
        for c in _ordered_channels(s_h.loc[s_h["level"] == "channel", "channel"].unique())
    ]
    picker, _spacer = st.columns([1, 2])
    with picker:
        pick = st.selectbox("Show", options=channels)
    code = {v: k for k, v in CHANNEL_LABELS.items()}.get(pick)

    camps = s_h[s_h["level"] == "campaign"].copy()
    if code:
        camps = camps[camps["channel"] == code]
    camps = camps.sort_values("p50_revenue", ascending=False).head(40)
    spend = camps["assumed_spend"].astype(float)
    show = pd.DataFrame(
        {
            "Campaign": camps["campaign_name"],
            "Channel": camps["channel"].map(_channel_name),
            "Type": camps["campaign_type"],
            "Planned spend": spend,
            "Low (P10)": camps["p10_revenue"],
            "Likely (P50)": camps["p50_revenue"],
            "High (P90)": camps["p90_revenue"],
            "Cautious ROAS": camps["p10_revenue"].astype(float) / spend.where(spend > 0),
            "Likely ROAS": camps["p50_roas"],
        }
    )
    config = {
        col: st.column_config.NumberColumn(col, format="$%.0f")
        for col in ("Planned spend", "Low (P10)", "Likely (P50)", "High (P90)")
    }
    config["Cautious ROAS"] = st.column_config.NumberColumn("Cautious ROAS", format="%.2f")
    config["Likely ROAS"] = st.column_config.NumberColumn("Likely ROAS", format="%.2f")
    st.dataframe(show, use_container_width=True, hide_index=True, column_config=config)
    st.caption(
        "**Cautious ROAS** is the P10 revenue divided by planned spend. Where it sits near 1.00, "
        "the cautious case for that campaign is roughly break-even — that is why some rows show a "
        "Low (P10) close to their planned spend."
    )

    types = s_h[s_h["level"] == "campaign_type"].copy()
    if code:
        types = types[types["channel"] == code]
    if not types.empty:
        types = types.sort_values("p50_revenue", ascending=False)
        fig = go.Figure(
            go.Bar(
                x=types["campaign_type"].astype(str),
                y=types["p50_revenue"],
                marker_color=ORANGE,
            )
        )
        fig.update_layout(title_text="Likely revenue by campaign type")
        st.plotly_chart(_plotly_theme(fig, height=300), use_container_width=True)


def _insight_context(panel, baseline, scenario, qa, h, multipliers) -> dict:
    mix = st.session_state.get("multipliers", multipliers)
    return build_insight_context(
        panel=panel,
        baseline=baseline,
        # A no-op scenario would only add "+0.00% vs baseline" noise to the briefing.
        scenario=None if _is_baseline_mix(mix) else scenario,
        qa_inventory=qa.inventory if qa else {},
        horizon=h,
        multipliers=mix,
    )


def _briefing_markdown(md: str) -> str:
    """Streamlit reads $…$ as LaTeX, which mangles dollar amounts."""
    return md.replace("$", r"\$")


def _render_ai(panel, baseline, scenario, qa, h, multipliers, api_key, model_name, provider) -> None:
    st.subheader("Plain-language briefing")
    st.caption(
        "Written from the numbers above — nothing invented. The offline summary needs no API key; "
        "an LLM only rephrases the same facts. The scoring CLI never calls out to a network."
    )
    if st.button("Rewrite with an LLM", type="primary"):
        ctx = _insight_context(panel, baseline, scenario, qa, h, multipliers)
        with st.spinner("Drafting briefing..."):
            md, engine = generate_insights(
                ctx,
                api_key=api_key or None,
                model=model_name or None,
                provider=provider,
            )
        st.session_state.insights_md = md
        st.session_state.insights_engine = engine

    with st.expander("Numbers behind this briefing (for trust)", expanded=False):
        st.json(_insight_context(panel, baseline, scenario, qa, h, multipliers))

    if st.session_state.get("insights_md"):
        engine = st.session_state.get("insights_engine")
        label = {
            "heuristic": "Offline summary",
            "groq": "Groq",
            "openai": "OpenAI",
        }.get(engine, engine)
        st.markdown(f'<span class="rf-pill ok">{label}</span>', unsafe_allow_html=True)
        st.markdown(_briefing_markdown(st.session_state.insights_md))


def main() -> None:
    _boot_veil()
    phase = st.session_state.get("_forecast_phase")

    with st.sidebar:
        st.markdown(
            f"""
            <div class="rf-sidebar-brand">
              {_logo()}
              <p class="rf-muted">Set the horizon and spend mix, then score the outlook.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("##### Plan")
        horizon = st.selectbox(
            "How far ahead?",
            options=[30, 60, 90],
            index=0,
            format_func=lambda d: f"{d} days",
        )
        st.markdown("##### What if spend changes?")
        st.caption("1.00 = keep the current mix.")
        m_google = st.slider("Google Ads", 0.5, 2.0, 1.0, 0.05)
        m_meta = st.slider("Meta Ads", 0.5, 2.0, 1.0, 0.05)
        m_bing = st.slider("Microsoft Ads", 0.5, 2.0, 1.0, 0.05)
        multipliers = {"google": m_google, "meta": m_meta, "bing": m_bing}

        with st.expander("AI briefing (optional)"):
            provider = st.selectbox(
                "Writer",
                options=["auto", "groq", "openai", "heuristic"],
                index=0,
                help="auto uses Groq if a key is present, then OpenAI, then a local summary.",
            )
            api_key = st.text_input(
                "API key",
                type="password",
                help="Groq: console.groq.com. Or set GROQ_API_KEY / OPENAI_API_KEY.",
            )
            default_model = (
                "llama-3.3-70b-versatile"
                if provider in ("auto", "groq")
                else "gpt-4o-mini"
            )
            model_name = st.text_input("Model", value=default_model)

        st.button(
            "Run forecast",
            type="primary",
            use_container_width=True,
            on_click=_queue_forecast,
            key="run_sidebar",
        )

        with st.expander("Advanced"):
            data_dir = st.text_input("Data folder", value=str(ROOT / "data"))
            model_path = st.text_input("Model file", value=str(ROOT / "pickle" / "model.pkl"))

    if "baseline" not in st.session_state:
        st.session_state.baseline = None
        st.session_state.scenario = None
        st.session_state.panel = None
        st.session_state.qa = None

    # Land on real numbers instead of instructions: score the sample plan once on open.
    if "rf_autorun" not in st.session_state:
        st.session_state.rf_autorun = True
        if st.session_state.baseline is None and phase is None:
            phase = "score"

    if "rf_motion_booted" not in st.session_state:
        st.session_state.rf_motion_booted = True
        _boot_motion()

    if phase == "close":
        _close_sidebar_now()
        st.session_state["_forecast_phase"] = "score"
    elif phase == "score":
        _close_sidebar_now()
    elif "rf_nav_booted" not in st.session_state:
        st.session_state.rf_nav_booted = True
        _inject_nav("boot")

    if phase == "score":
        st.session_state["_forecast_phase"] = None
        try:
            with st.spinner("Scoring the current plan…"):
                panel, qa = _load_panel(data_dir)
                _model_path(model_path)
                baseline = _run_forecast(
                    panel, model_path, {"google": 1.0, "meta": 1.0, "bing": 1.0}
                )
                scenario = (
                    baseline
                    if _is_baseline_mix(multipliers)
                    else _run_forecast(panel, model_path, multipliers)
                )
                st.session_state.panel = panel
                st.session_state.qa = qa
                st.session_state.baseline = baseline
                st.session_state.scenario = scenario
                st.session_state.multipliers = multipliers
                st.session_state.horizon = horizon
                st.session_state.insights_md = heuristic_insights(
                    _insight_context(panel, baseline, scenario, qa, horizon, multipliers)
                )
                st.session_state.insights_engine = "heuristic"
        except Exception as exc:  # surface a readable card instead of a raw traceback
            st.session_state.forecast_error = str(exc)

    baseline = st.session_state.baseline
    scenario = st.session_state.scenario
    panel = st.session_state.panel
    qa = st.session_state.qa

    if baseline is None:
        _hero()
        if st.session_state.get("forecast_error"):
            st.error(
                "Could not score this data folder: "
                f"{st.session_state['forecast_error']}\n\n"
                "Check the paths under **Plan → Advanced**."
            )
        _render_empty_state(data_dir)
        if phase == "close":
            st.rerun()
        return

    h = horizon
    scored = st.session_state.get("multipliers", multipliers)
    whatif = not _is_baseline_mix(scored)
    b_h = baseline[baseline["horizon_days"] == h]
    s_h = scenario[scenario["horizon_days"] == h]
    agg_b = b_h[b_h["level"] == "aggregate"].iloc[0]
    agg_s = s_h[s_h["level"] == "aggregate"].iloc[0]

    _hero(horizon=h, multipliers=scored)

    # Horizon re-filters instantly; a new spend mix has to be scored.
    if any(abs(float(multipliers[k]) - float(scored.get(k, 1.0))) > 1e-9 for k in multipliers):
        st.warning("Spend sliders changed. Click **Run forecast** to score the new mix.", icon="⚠️")

    tab_forecast, tab_channels, tab_campaigns, tab_qa, tab_ai = st.tabs(
        ["Outlook", "Channels", "Campaigns", "Data check", "Briefing"]
    )

    with tab_forecast:
        _render_forecast(h, agg_b, agg_s, whatif=whatif)
    with tab_channels:
        _render_channels(b_h, s_h, whatif=whatif)
    with tab_campaigns:
        _render_campaigns(s_h)
    with tab_qa:
        if qa is not None:
            _render_qa(qa)
        else:
            st.caption("Run a forecast to load the data check.")
    with tab_ai:
        _render_ai(panel, baseline, scenario, qa, h, multipliers, api_key, model_name, provider)

    if phase == "close":
        st.rerun()


if __name__ == "__main__":
    main()
