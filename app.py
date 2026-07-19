"""
AIgnition Forecasting — Streamlit demo entrypoint.

Run:
  pip install -r requirements-demo.txt
  streamlit run app.py

Scoring path (run.sh) does not use this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

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

st.set_page_config(
    page_title="AIgnition Forecasting",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.02em; }
      .block-container { padding-top: 1.5rem; max-width: 1200px; }
      div[data-testid="stMetricValue"] { font-family: 'Fraunces', Georgia, serif; }
      .aignite-banner {
        background: linear-gradient(135deg, #0b1f1c 0%, #143d36 48%, #1c4f45 100%);
        color: #e8f2ef; padding: 1.25rem 1.5rem; border-radius: 12px; margin-bottom: 1rem;
      }
      .aignite-banner h1 { color: #f4faf8 !important; margin: 0 0 0.25rem 0; font-size: 1.85rem; }
      .aignite-banner p { margin: 0; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def main() -> None:
    st.markdown(
        """
        <div class="aignite-banner">
          <h1>AIgnition Forecasting</h1>
          <p>Probabilistic ecommerce revenue &amp; ROAS — Google · Meta · Microsoft Ads</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        data_dir = st.text_input("Data folder", value=str(ROOT / "data"))
        model_path = st.text_input("Model pickle", value=str(ROOT / "pickle" / "model.pkl"))
        horizon = st.selectbox("Forecast horizon (days)", options=[30, 60, 90], index=0)
        st.subheader("Budget multipliers")
        m_google = st.slider("Google", 0.5, 2.0, 1.0, 0.05)
        m_meta = st.slider("Meta", 0.5, 2.0, 1.0, 0.05)
        m_bing = st.slider("Bing", 0.5, 2.0, 1.0, 0.05)
        multipliers = {"google": m_google, "meta": m_meta, "bing": m_bing}
        st.subheader("AI insights")
        provider = st.selectbox(
            "LLM provider",
            options=["auto", "groq", "openai", "heuristic"],
            index=0,
            help="auto prefers free Groq, then OpenAI, then offline heuristic.",
        )
        api_key = st.text_input(
            "API key (optional)",
            type="password",
            help="Groq: console.groq.com (free). Or set GROQ_API_KEY / OPENAI_API_KEY in the environment.",
        )
        default_model = (
            "llama-3.3-70b-versatile"
            if provider in ("auto", "groq")
            else "gpt-4o-mini"
        )
        model_name = st.text_input("Model", value=default_model)
        run = st.button("Run forecast", type="primary", use_container_width=True)

    if "baseline" not in st.session_state:
        st.session_state.baseline = None
        st.session_state.scenario = None
        st.session_state.panel = None
        st.session_state.qa = None

    if run:
        with st.spinner("Loading data, scoring baseline + scenario..."):
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
        st.info("Set controls in the sidebar and click **Run forecast** to begin the demo.")
        st.markdown(
            """
            **Demo flow (5–7 min)**  
            1. Confirm data QA  
            2. Inspect baseline 30/60/90d forecast  
            3. Raise Meta (or other) budget multiplier  
            4. Drill into campaign contributors  
            5. Generate AI causal memo  
            """
        )
        return

    h = horizon
    b_h = baseline[baseline["horizon_days"] == h]
    s_h = scenario[scenario["horizon_days"] == h]
    agg_b = b_h[b_h["level"] == "aggregate"].iloc[0]
    agg_s = s_h[s_h["level"] == "aggregate"].iloc[0]

    tab_qa, tab_forecast, tab_channels, tab_campaigns, tab_ai = st.tabs(
        ["Data QA", "Forecast", "Channels", "Campaigns", "AI Insights"]
    )

    with tab_qa:
        st.subheader("Campaign consistency & coverage")
        if qa is not None:
            for line in qa.summary_lines():
                st.text(line)
            inv = qa.inventory.get("channels", {})
            rows = []
            for ch, stats in inv.items():
                rows.append(
                    {
                        "channel": ch,
                        "campaigns": stats["campaigns"],
                        "sparsity_%": stats["sparsity_pct"],
                        "blended_roas": round(stats["blended_roas"], 3),
                        "date_min": stats["date_min"],
                        "date_max": stats["date_max"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            peaks = qa.seasonality.get("peak_months", {})
            if peaks:
                st.caption("Seasonality peaks (vs channel median monthly revenue)")
                for ch, plist in peaks.items():
                    if plist:
                        st.write(
                            f"**{ch}**: "
                            + ", ".join(f"{p['month']} ({p['vs_median']}x)" for p in plist[:5])
                        )

    with tab_forecast:
        st.subheader(f"{h}-day probabilistic outlook")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline P50 revenue", _fmt_money(float(agg_b["p50_revenue"])))
        c2.metric(
            "Scenario P50 revenue",
            _fmt_money(float(agg_s["p50_revenue"])),
            delta=f"{100*(float(agg_s['p50_revenue'])/max(float(agg_b['p50_revenue']),1)-1):+.1f}%",
        )
        c3.metric("Baseline P50 ROAS", f"{float(agg_b['p50_roas']):.2f}")
        c4.metric(
            "Scenario P50 ROAS",
            f"{float(agg_s['p50_roas']):.2f}",
            delta=f"{float(agg_s['p50_roas'])-float(agg_b['p50_roas']):+.2f}",
        )

        st.caption(
            f"Baseline range P10–P90: {_fmt_money(float(agg_b['p10_revenue']))} – "
            f"{_fmt_money(float(agg_b['p90_revenue']))} · "
            f"Assumed spend {_fmt_money(float(agg_b['assumed_spend']))} → "
            f"scenario {_fmt_money(float(agg_s['assumed_spend']))}"
        )

        fan = pd.DataFrame(
            {
                "scenario": ["Baseline", "Budget scenario"],
                "p10": [float(agg_b["p10_revenue"]), float(agg_s["p10_revenue"])],
                "p50": [float(agg_b["p50_revenue"]), float(agg_s["p50_revenue"])],
                "p90": [float(agg_b["p90_revenue"]), float(agg_s["p90_revenue"])],
            }
        )
        st.bar_chart(fan.set_index("scenario")[["p10", "p50", "p90"]])

        st.download_button(
            "Download scenario predictions CSV",
            data=scenario.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_h{h}_scenario.csv",
            mime="text/csv",
        )

    with tab_channels:
        st.subheader("Channel breakdown")
        ch = b_h[b_h["level"] == "channel"][
            ["channel", "assumed_spend", "p10_revenue", "p50_revenue", "p90_revenue", "p50_roas"]
        ].rename(columns=lambda c: f"base_{c}" if c != "channel" else c)
        chs = s_h[s_h["level"] == "channel"][
            ["channel", "assumed_spend", "p10_revenue", "p50_revenue", "p90_revenue", "p50_roas"]
        ].rename(columns=lambda c: f"scen_{c}" if c != "channel" else c)
        merged = ch.merge(chs, on="channel")
        st.dataframe(merged, use_container_width=True, hide_index=True)
        st.bar_chart(
            s_h[s_h["level"] == "channel"].set_index("channel")[["p50_revenue"]]
        )

    with tab_campaigns:
        st.subheader("Campaign-level contributors")
        channel_filter = st.selectbox(
            "Channel filter",
            options=["all"] + sorted(s_h[s_h["level"] == "channel"]["channel"].unique().tolist()),
        )
        camps = s_h[s_h["level"] == "campaign"].copy()
        if channel_filter != "all":
            camps = camps[camps["channel"] == channel_filter]
        camps = camps.sort_values("p50_revenue", ascending=False)
        show = camps[
            [
                "channel",
                "campaign_type",
                "campaign_name",
                "assumed_spend",
                "p10_revenue",
                "p50_revenue",
                "p90_revenue",
                "p50_roas",
            ]
        ].head(40)
        st.dataframe(show, use_container_width=True, hide_index=True)

        types = s_h[s_h["level"] == "campaign_type"].copy()
        if channel_filter != "all":
            types = types[types["channel"] == channel_filter]
        if not types.empty:
            st.caption("Campaign type P50 revenue")
            st.bar_chart(
                types.set_index("campaign_type")[["p50_revenue"]].sort_values(
                    "p50_revenue", ascending=False
                )
            )

    with tab_ai:
        st.subheader("AI-assisted causal summary")
        st.caption(
            "Grounded in model outputs + QA via llm_layer.py. "
            "Heuristic fallback if no API key. Not used by run.sh."
        )
        if st.button("Generate insights", type="primary"):
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
            with st.expander("Context JSON sent to the reasoner", expanded=False):
                st.json(ctx)

        if st.session_state.get("insights_md"):
            st.caption(f"Engine: `{st.session_state.get('insights_engine')}`")
            st.markdown(st.session_state.insights_md)


if __name__ == "__main__":
    main()
