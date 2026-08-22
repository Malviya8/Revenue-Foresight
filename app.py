"""
Revenue Foresight — Streamlit demo entrypoint.

Run:
  pip install -r requirements-demo.txt
  streamlit run app.py

Scoring path (run.sh) does not use this file.
"""

from __future__ import annotations

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
from llm_layer import build_insight_context, generate_insights  # noqa: E402
from predict import predict_frame  # noqa: E402
from validate import validate_panel  # noqa: E402

ORANGE = "#FF6A3D"
ORANGE_SOFT = "#FF9A74"
MUTED = "#9A9A9A"
WHITE = "#ECECEC"

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
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

      html, body, [class*="css"], .stApp, p, span, label, button {
        font-family: 'DM Sans', sans-serif !important;
      }
      .stApp { background: #0C0C0C; color: #ECECEC; }
      h1, h2, h3, h4 {
        font-family: 'DM Sans', sans-serif !important;
        letter-spacing: -0.025em;
        font-weight: 650 !important;
        color: #F2F2F2 !important;
      }
      .block-container { padding-top: 2.25rem; padding-bottom: 3.5rem; max-width: 1120px; }

      header[data-testid="stHeader"] {
        background: transparent !important;
        border: none !important;
        height: 0 !important;
        min-height: 0 !important;
      }
      #MainMenu { visibility: hidden; }
      .stAppDeployButton, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
      [data-testid="stDecoration"] { display: none; }

      iframe[height="0"] {
        position: absolute !important;
        left: -9999px !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
        border: 0 !important;
      }
      div[data-testid="stIFrame"]:has(iframe[height="0"]),
      .stElementContainer:has(iframe[height="0"]) {
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
      }

      .stSpinner, [data-testid="stSpinner"] {
        padding: 0.45rem 0 0.7rem;
        overflow: visible !important;
        line-height: 1.5 !important;
      }
      .stSpinner > div, [data-testid="stSpinner"] > div {
        overflow: visible !important;
        height: auto !important;
      }
      [data-testid="stStatusWidget"] {
        overflow: visible !important;
        height: auto !important;
        line-height: 1.5;
        padding: 0.35rem 0.5rem;
      }

      div[data-testid="stMetricValue"] {
        font-family: 'DM Sans', sans-serif; color: #F2F2F2; font-size: 1.45rem; font-weight: 650;
      }
      div[data-testid="stMetricLabel"] { color: #8F8F8F; letter-spacing: 0.01em; }
      div[data-testid="stMetricDelta"] { font-weight: 600; }

      [data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid #242424;
        transition: transform 0.45s cubic-bezier(0.32, 0.72, 0, 1),
                    min-width 0.45s cubic-bezier(0.32, 0.72, 0, 1),
                    max-width 0.45s cubic-bezier(0.32, 0.72, 0, 1) !important;
      }
      [data-testid="stSidebar"] .block-container { padding-top: 1.15rem; }

      [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        align-items: center !important;
        gap: 0.05rem;
        top: 0.7rem !important;
        left: 0.7rem !important;
        z-index: 1000002 !important;
        background: #161616 !important;
        border: 1px solid #2E2E2E !important;
        border-radius: 12px !important;
        padding: 0.12rem 0.75rem 0.12rem 0.12rem !important;
        box-shadow: 0 10px 28px rgba(0,0,0,0.38) !important;
      }
      [data-testid="stSidebarCollapsedControl"]::after {
        content: "Plan";
        color: #FF8A66;
        font-family: "DM Sans", sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        padding-right: 0.15rem;
      }

      .stButton > button[kind="primary"] {
        background: #FF6A3D; color: #0C0C0C; border: 0; font-weight: 700;
        border-radius: 10px; box-shadow: 0 0 0 1px rgba(255,106,61,0.25);
      }
      .stButton > button[kind="primary"]:hover { background: #FF815C; border: 0; color: #0C0C0C; }
      .stButton > button[kind="secondary"] {
        border: 1px solid #2E2E2E; background: #191919; color: #ECECEC; border-radius: 10px;
      }

      div[data-testid="stTabs"] { margin-top: 0.15rem; }
      .stTabs [data-baseweb="tab-list"] {
        gap: 1.65rem; padding-left: 0; margin-bottom: 0.35rem;
        border-bottom: 1px solid #262626;
      }
      .stTabs [data-baseweb="tab"] {
        color: #8F8F8F; font-weight: 600; padding: 0.7rem 0 0.95rem; font-size: 0.95rem;
      }
      .stTabs [aria-selected="true"] { color: #FF8A66 !important; }
      .stTabs [data-baseweb="tab-highlight"] { background-color: #FF6A3D; height: 2px; }
      .stTabs [data-baseweb="tab-border"] { background-color: transparent; }

      [data-testid="stDataFrame"] {
        border: 1px solid #262626; border-radius: 12px; overflow: hidden; background: #141414;
      }

      .rf-hero {
        background:
          radial-gradient(720px 280px at 92% -10%, rgba(255,106,61,0.10), transparent 58%),
          linear-gradient(180deg, #161616 0%, #101010 100%);
        border: 1px solid #2A2A2A;
        box-shadow: 0 18px 40px rgba(0,0,0,0.28);
        color: #ECECEC;
        padding: 1.6rem 1.75rem 1.45rem;
        border-radius: 18px;
        margin: 0 0 1.85rem 0;
      }
      .rf-kicker {
        color: #FF8A66; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.18em;
        text-transform: uppercase; margin: 0 0 0.7rem 0;
      }
      .rf-hero h1 {
        color: #F7F7F7 !important; margin: 0 0 0.7rem 0; font-size: 1.72rem !important;
        line-height: 1.25; font-weight: 650 !important; max-width: 34rem;
      }
      .rf-lede { margin: 0; color: #B4B4B4; font-size: 0.98rem; line-height: 1.55; max-width: 40rem; }
      .rf-status {
        display: flex; flex-wrap: wrap; align-items: center; gap: 0.55rem 0.7rem;
        margin-top: 1.3rem; padding: 1rem 0 0;
        border-top: 1px solid #2A2A2A; color: #9A9A9A; font-size: 0.86rem;
      }
      .rf-chip {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: #1C1C1C; border: 1px solid #333; color: #E6E6E6;
        border-radius: 999px; padding: 0.22rem 0.65rem; font-weight: 600; font-size: 0.8rem;
      }
      .rf-chip em { color: #FF8A66; font-style: normal; font-weight: 700; }

      .rf-card {
        background: #161616; border: 1px solid #2A2A2A; border-radius: 14px;
        padding: 1.1rem 1.15rem; height: 100%;
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
      }
      .rf-card h4 { margin: 0 0 0.4rem 0; font-size: 0.82rem; color: #9A9A9A !important; font-weight: 600 !important; letter-spacing: 0.04em; text-transform: uppercase; }
      .rf-card .rf-big { font-size: 1.45rem; color: #F4F4F4; margin: 0.1rem 0 0.45rem; font-weight: 650; letter-spacing: -0.03em; }
      .rf-muted { color: #9A9A9A; font-size: 0.88rem; line-height: 1.5; }
      .rf-orange { color: #FF8A66; font-weight: 650; }

      .rf-issue {
        background: #161616; border: 1px solid #2A2A2A; border-left: 3px solid #3A3A3A;
        border-radius: 12px; padding: 0.85rem 1rem; margin-bottom: 0.7rem;
      }
      .rf-issue.warn { border-left-color: #FF6A3D; }
      .rf-issue.error { border-left-color: #E03E12; }
      .rf-issue.info { border-left-color: #3A3A3A; }
      .rf-issue strong { color: #F0F0F0; display: block; margin-bottom: 0.2rem; font-weight: 650; }
      .rf-pill {
        display: inline-block; font-size: 0.66rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; padding: 0.16rem 0.5rem; border-radius: 999px; margin-bottom: 0.4rem;
      }
      .rf-pill.warn { background: rgba(255,106,61,0.14); color: #FF8A66; }
      .rf-pill.info { background: #222; color: #B0B0B0; }
      .rf-pill.error { background: rgba(224,62,18,0.18); color: #FF8A66; }
      .rf-pill.ok { background: rgba(255,106,61,0.12); color: #FF8A66; }

      .rf-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; margin: 0 0 1.4rem; }
      .rf-step { background: #161616; border: 1px solid #2A2A2A; border-radius: 14px; padding: 1.05rem 1.1rem; }
      .rf-step span { color: #FF8A66; font-weight: 700; font-size: 0.72rem; letter-spacing: 0.1em; }
      .rf-step h4 { margin: 0.35rem 0 0.35rem; font-size: 1.02rem !important; }

      .rf-landing-stats {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; margin: 0 0 1.25rem;
      }
      .rf-cta-note { color: #9A9A9A; font-size: 0.9rem; line-height: 1.5; margin: 0.15rem 0 0.85rem; }
      .rf-sidebar-brand { margin: 0 0 1.1rem; padding: 0 0 0.95rem; border-bottom: 1px solid #2A2A2A; }
      .rf-sidebar-brand p { margin: 0; }

      @media (max-width: 768px) {
        html:not(.rf-hydrated) [data-testid="stSidebar"] {
          transform: translateX(-100%) !important;
          min-width: 0 !important;
          max-width: 0 !important;
          box-shadow: none !important;
        }
        html.rf-force-close [data-testid="stSidebar"] {
          transform: translateX(-100%) !important;
          min-width: 0 !important;
          max-width: 0 !important;
          box-shadow: none !important;
        }
        .block-container {
          padding-top: 4.1rem !important;
          padding-left: 1.05rem !important;
          padding-right: 1.05rem !important;
          padding-bottom: 2.4rem !important;
        }
        .rf-hero {
          padding: 1.35rem 1.15rem 1.2rem;
          margin-bottom: 1.15rem;
          border-radius: 16px;
        }
        .rf-hero h1 { font-size: 1.48rem !important; max-width: none; }
        .rf-lede { font-size: 0.95rem; max-width: none; }
        .rf-landing-stats { grid-template-columns: 1fr 1fr; }
        .rf-landing-stats .rf-card:last-child { grid-column: 1 / -1; }
        .rf-steps { grid-template-columns: 1fr; margin-bottom: 1.1rem; }
        .stTabs [data-baseweb="tab-list"] {
          gap: 0.85rem; overflow-x: auto; flex-wrap: nowrap;
          -webkit-overflow-scrolling: touch;
        }
        .stTabs [data-baseweb="tab"] {
          font-size: 0.86rem; padding: 0.55rem 0 0.8rem; white-space: nowrap;
        }
        div[data-testid="stMetricValue"] { font-size: 1.18rem; }
        [data-testid="stSidebarCollapsedControl"] {
          top: 0.55rem !important;
          left: 0.55rem !important;
        }
        .stButton > button[kind="primary"] {
          min-height: 3rem;
          font-size: 1.02rem;
        }
        .stApp { overflow-x: hidden; }
      }
      @media (max-width: 900px) and (min-width: 769px) {
        .rf-steps, .rf-landing-stats { grid-template-columns: 1fr; }
        .rf-hero h1 { font-size: 1.4rem !important; }
      }
      @media (prefers-reduced-motion: reduce) {
        [data-testid="stSidebar"] { transition: none !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


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

  if (cmd === "close") {
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

_FORCE_SIDEBAR_CSS = """
<style>
  @media (max-width: 768px) {
    [data-testid="stSidebar"] {
      transform: translateX(-100%) !important;
      min-width: 0 !important;
      max-width: 0 !important;
      box-shadow: none !important;
    }
  }
</style>
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
    st.session_state["_nav_tick"] = int(st.session_state.get("_nav_tick", 0)) + 1
    st.markdown(_FORCE_SIDEBAR_CSS, unsafe_allow_html=True)
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


def _plotly_theme(fig: go.Figure, *, height: int = 320, y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=64, r=16, t=56, b=48),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="DM Sans, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0, font=dict(size=12)),
        bargap=0.28,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#2A2A2A", zeroline=False, color=MUTED, title_standoff=12)
    fig.update_yaxes(
        gridcolor="#2A2A2A",
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


def _hero(*, horizon: int | None = None, multipliers: dict[str, float] | None = None) -> None:
    if horizon is not None and multipliers is not None:
        chips = "".join(
            f'<span class="rf-chip">{_channel_name(k)} {v:.2f}×</span>'
            for k, v in multipliers.items()
        )
        status = (
            f'<div class="rf-status">'
            f'<span class="rf-chip"><em>{horizon}-day</em> outlook</span>'
            f"{chips}"
            f"</div>"
        )
    else:
        status = (
            '<div class="rf-status">'
            '<span class="rf-chip">Google Ads</span>'
            '<span class="rf-chip">Meta Ads</span>'
            '<span class="rf-chip">Microsoft Ads</span>'
            '<span class="rf-chip"><em>P10–P50–P90</em></span>'
            '<span class="rf-chip">30 / 60 / 90 days</span>'
            "</div>"
        )
    st.markdown(
        f"""
        <div class="rf-hero">
          <p class="rf-kicker">Revenue Foresight</p>
          <h1>See the next 30–90 days before you spend.</h1>
          <p class="rf-lede">Probabilistic revenue and ROAS across Google, Meta, and Microsoft Ads — a likely number, plus a cautious and optimistic range.</p>
          {status}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _issue_title(issue) -> tuple[str, str]:
    title, why = ISSUE_COPY.get(issue.code, (issue.code.replace("_", " ").title(), issue.detail))
    return title, why


def _render_empty_state(data_dir: str) -> None:
    cards = [
        ("Channels", "3", "Google, Meta, and Microsoft Ads in one outlook."),
        ("Horizon", "30–90d", "Pick the window in Plan, then score both mixes."),
        ("Range", "P10–P90", "Cautious floor, likely number, optimistic ceiling."),
    ]
    try:
        _panel, qa = _load_panel(data_dir)
        inv = qa.inventory.get("channels", {}) if qa else {}
        if inv:
            cards[0] = (
                "Channels loaded",
                str(len(inv)),
                " · ".join(_channel_name(ch) for ch in inv),
            )
    except Exception:
        inv = {}

    stats = "".join(
        f'<div class="rf-card"><h4>{title}</h4><p class="rf-big">{big}</p>'
        f'<p class="rf-muted">{note}</p></div>'
        for title, big, note in cards
    )
    st.markdown(f'<div class="rf-landing-stats">{stats}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="rf-steps">
          <div class="rf-step"><span>1 · PLAN</span><h4>Keep spend as-is, or try a what-if</h4>
          <p class="rf-muted">Open Plan to move a slider. 1.00 is today’s mix. 1.20 means +20% on that channel.</p></div>
          <div class="rf-step"><span>2 · FORECAST</span><h4>Run the outlook</h4>
          <p class="rf-muted">We score a baseline and your scenario together. No retraining, no network.</p></div>
          <div class="rf-step"><span>3 · DECIDE</span><h4>Compare range, channels, campaigns</h4>
          <p class="rf-muted">P50 is the most likely outcome. P10–P90 is the planning band.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if inv:
        snaps = "".join(
            (
                '<div class="rf-card">'
                f'<p class="rf-kicker">{_channel_name(ch)}</p>'
                f'<p class="rf-big">{int(stats_row["campaigns"])} campaigns</p>'
                '<p class="rf-muted">'
                f'Historic ROAS <span class="rf-orange">{stats_row["blended_roas"]:.2f}</span>'
                f' · {100.0 - float(stats_row.get("sparsity_pct", 0)):.0f}% active days'
                "</p></div>"
            )
            for ch, stats_row in inv.items()
        )
        st.markdown(f'<div class="rf-landing-stats">{snaps}</div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="rf-cta-note">Sample exports are already selected. Tap <span class="rf-orange">Plan</span> '
        "at the top left to change the mix, or run the default outlook now.</p>",
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
        cols = st.columns(len(inv))
        for col, (ch, stats) in zip(cols, inv.items()):
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


def _render_forecast(h: int, agg_b, agg_s) -> None:
    st.subheader(f"What the next {h} days look like")
    st.caption("P50 is the most likely outcome. P10 is a cautious floor. P90 is an optimistic ceiling. Spend changes in the sidebar update the what-if column.")

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
    hover = " %{y:$,.0f}<extra>%{fullData.name}</extra>"
    fig.add_bar(
        name="Cautious (P10)",
        x=["Current plan", "What-if"],
        y=[float(agg_b["p10_revenue"]), float(agg_s["p10_revenue"])],
        marker_color="#4A4A4A",
        hovertemplate=hover,
    )
    fig.add_bar(
        name="Likely (P50)",
        x=["Current plan", "What-if"],
        y=[float(agg_b["p50_revenue"]), float(agg_s["p50_revenue"])],
        marker_color=ORANGE,
        hovertemplate=hover,
    )
    fig.add_bar(
        name="Optimistic (P90)",
        x=["Current plan", "What-if"],
        y=[float(agg_b["p90_revenue"]), float(agg_s["p90_revenue"])],
        marker_color=ORANGE_SOFT,
        hovertemplate=hover,
    )
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

    st.download_button(
        "Download this what-if as CSV",
        data=st.session_state.scenario.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_h{h}_scenario.csv",
        mime="text/csv",
    )


def _channel_table(b_h: pd.DataFrame, s_h: pd.DataFrame) -> pd.DataFrame:
    ch = b_h[b_h["level"] == "channel"][
        ["channel", "assumed_spend", "p10_revenue", "p50_revenue", "p90_revenue", "p50_roas"]
    ].copy()
    chs = s_h[s_h["level"] == "channel"][
        ["channel", "assumed_spend", "p10_revenue", "p50_revenue", "p90_revenue", "p50_roas"]
    ].copy()
    ch["channel"] = ch["channel"].map(_channel_name)
    chs["channel"] = chs["channel"].map(_channel_name)
    merged = ch.merge(chs, on="channel", suffixes=("_now", "_whatif"))
    out = pd.DataFrame(
        {
            "Channel": merged["channel"],
            "Spend now": merged["assumed_spend_now"],
            "Spend what-if": merged["assumed_spend_whatif"],
            "Likely revenue now": merged["p50_revenue_now"],
            "Likely revenue what-if": merged["p50_revenue_whatif"],
            "Low (P10) what-if": merged["p10_revenue_whatif"],
            "High (P90) what-if": merged["p90_revenue_whatif"],
            "ROAS now": merged["p50_roas_now"],
            "ROAS what-if": merged["p50_roas_whatif"],
        }
    )
    return out


def _render_channels(b_h: pd.DataFrame, s_h: pd.DataFrame) -> None:
    st.subheader("Where the revenue comes from")
    st.caption("Channel totals are the planning layer. They always add up to the store-level number.")
    table = _channel_table(b_h, s_h)
    money_cols = [c for c in table.columns if c not in ("Channel", "ROAS now", "ROAS what-if")]
    config = {
        col: st.column_config.NumberColumn(col, format="$%.0f")
        for col in money_cols
    }
    config["ROAS now"] = st.column_config.NumberColumn("ROAS now", format="%.2f")
    config["ROAS what-if"] = st.column_config.NumberColumn("ROAS what-if", format="%.2f")
    st.dataframe(table, use_container_width=True, hide_index=True, column_config=config)

    order = [c for c in ("google", "meta", "bing") if c in set(b_h.loc[b_h["level"] == "channel", "channel"])]
    base_ch = b_h[b_h["level"] == "channel"].set_index("channel").reindex(order)
    scen_ch = s_h[s_h["level"] == "channel"].set_index("channel").reindex(order)
    labels = [_channel_name(c) for c in order]
    fig = go.Figure()
    fig.add_bar(
        name="Current plan (P50)",
        x=labels,
        y=base_ch["p50_revenue"].tolist(),
        marker_color="#4A4A4A",
        hovertemplate=" %{y:$,.0f}<extra>Current plan</extra>",
    )
    fig.add_bar(
        name="What-if (P50)",
        x=labels,
        y=scen_ch["p50_revenue"].tolist(),
        marker_color=ORANGE,
        hovertemplate=" %{y:$,.0f}<extra>What-if</extra>",
    )
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
        _channel_name(c) for c in sorted(s_h[s_h["level"] == "channel"]["channel"].unique().tolist())
    ]
    pick = st.selectbox("Show", options=channels)
    code = {v: k for k, v in CHANNEL_LABELS.items()}.get(pick)

    camps = s_h[s_h["level"] == "campaign"].copy()
    if code:
        camps = camps[camps["channel"] == code]
    camps = camps.sort_values("p50_revenue", ascending=False).head(40)
    show = pd.DataFrame(
        {
            "Campaign": camps["campaign_name"],
            "Channel": camps["channel"].map(_channel_name),
            "Type": camps["campaign_type"],
            "Planned spend": camps["assumed_spend"],
            "Low (P10)": camps["p10_revenue"],
            "Likely (P50)": camps["p50_revenue"],
            "High (P90)": camps["p90_revenue"],
            "Likely ROAS": camps["p50_roas"],
        }
    )
    config = {
        col: st.column_config.NumberColumn(col, format="$%.0f")
        for col in ("Planned spend", "Low (P10)", "Likely (P50)", "High (P90)")
    }
    config["Likely ROAS"] = st.column_config.NumberColumn("Likely ROAS", format="%.2f")
    st.dataframe(show, use_container_width=True, hide_index=True, column_config=config)

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


def _render_ai(panel, baseline, scenario, qa, h, multipliers, api_key, model_name, provider) -> None:
    st.subheader("Plain-language briefing")
    st.caption("Optional. Grounded in the numbers above. If there is no API key, you still get an offline summary — never used by the scoring CLI.")
    if st.button("Write the briefing", type="primary"):
        ctx = build_insight_context(
            panel=panel,
            baseline=baseline,
            scenario=scenario,
            qa_inventory=qa.inventory if qa else {},
            horizon=h,
            multipliers=st.session_state.get("multipliers", multipliers),
        )
        with st.spinner("Drafting briefing..."):
            md, engine = generate_insights(
                ctx,
                api_key=api_key or None,
                model=model_name or None,
                provider=provider,
            )
        st.session_state.insights_md = md
        st.session_state.insights_engine = engine
        with st.expander("Numbers sent to the briefing (for trust)", expanded=False):
            st.json(ctx)

    if st.session_state.get("insights_md"):
        engine = st.session_state.get("insights_engine")
        label = {
            "heuristic": "Offline summary",
            "groq": "Groq",
            "openai": "OpenAI",
        }.get(engine, engine)
        st.markdown(f'<span class="rf-pill ok">{label}</span>', unsafe_allow_html=True)
        st.markdown(st.session_state.insights_md)


def main() -> None:
    phase = st.session_state.get("_forecast_phase")

    with st.sidebar:
        st.markdown(
            """
            <div class="rf-sidebar-brand">
              <p class="rf-kicker">Revenue Foresight</p>
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

    if phase == "close":
        _close_sidebar_now()
        st.session_state["_forecast_phase"] = "score"
    elif phase == "score":
        _close_sidebar_now()
    elif "rf_nav_booted" not in st.session_state:
        st.session_state.rf_nav_booted = True
        _inject_nav("boot")

    if "baseline" not in st.session_state:
        st.session_state.baseline = None
        st.session_state.scenario = None
        st.session_state.panel = None
        st.session_state.qa = None

    if phase == "score":
        st.session_state["_forecast_phase"] = None
        with st.spinner("Scoring current plan and your what-if…"):
            panel, qa = _load_panel(data_dir)
            _model_path(model_path)
            baseline = _run_forecast(panel, model_path, {"google": 1.0, "meta": 1.0, "bing": 1.0})
            scenario = _run_forecast(panel, model_path, multipliers)
            st.session_state.panel = panel
            st.session_state.qa = qa
            st.session_state.baseline = baseline
            st.session_state.scenario = scenario
            st.session_state.multipliers = multipliers
            st.session_state.horizon = horizon

    baseline = st.session_state.baseline
    scenario = st.session_state.scenario
    panel = st.session_state.panel
    qa = st.session_state.qa

    if baseline is None:
        _hero()
        _render_empty_state(data_dir)
        if phase == "close":
            st.rerun()
        return

    h = horizon
    b_h = baseline[baseline["horizon_days"] == h]
    s_h = scenario[scenario["horizon_days"] == h]
    agg_b = b_h[b_h["level"] == "aggregate"].iloc[0]
    agg_s = s_h[s_h["level"] == "aggregate"].iloc[0]

    _hero(horizon=h, multipliers=multipliers)

    tab_forecast, tab_channels, tab_campaigns, tab_qa, tab_ai = st.tabs(
        ["Outlook", "Channels", "Campaigns", "Data check", "Briefing"]
    )

    with tab_forecast:
        _render_forecast(h, agg_b, agg_s)
    with tab_channels:
        _render_channels(b_h, s_h)
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
