"""Reports — aggregated intelligence across the trading system.

Four reports (AMD scans · trade performance · system activity · market/macro),
each with deterministic rule-based analytics and an optional Claude-written
narrative summary (enabled when an Anthropic API key is configured).

Report data is built in src/core/reporting.py; the AI layer is
src/core/ai_reports.py. This page is presentation only.
"""
import streamlit as st
from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
from src.core import reporting, ai_reports
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Reports · Trading System",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

st.markdown("""
<style>
    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer{visibility:hidden;}
    .block-container{padding-top:1.5rem;max-width:1500px;}
    .rp-hero{border:1px solid #2a2a2a;background:#0a0a0a;padding:14px 18px;margin-bottom:14px;}
    .rp-verdict{border-left:3px solid #00ff41;background:#001a08;padding:8px 12px;margin:8px 0;
                font-family:'JetBrains Mono',monospace;font-size:12px;color:#e6e6e6;}
    .rp-ai{border:1px solid #00a32a;background:#04140a;padding:12px 14px;margin-top:8px;
           white-space:pre-wrap;font-size:13px;line-height:1.5;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📑 Reports")
    st.caption("Cross-system intelligence — scans, performance, activity, macro.")
    window = st.slider("Trade window (closed trades)", 20, 500, 100, step=20)
    if st.button("🔄 Refresh", use_container_width=True, type="primary"):
        st.rerun()

    ai_on = ai_reports.is_available()
    if ai_on:
        st.success("✨ AI summaries enabled (Claude)", icon="🤖")
    else:
        st.caption("✨ AI summaries off — " + ai_reports.unavailable_reason())

    st.divider()
    render_sidebar_nav()

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="rp-hero">
  <div style="font-size:24px;font-weight:700;color:#e6e6e6;">📑 Reports</div>
  <div style="color:#9a9a9a;font-size:13px;margin-top:4px;">
    AMD Scans · Trade Performance · System Activity · Market &amp; Macro Snapshot
  </div>
  <div style="font-size:12px;color:#00ff41;margin-top:6px;">
    🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)


def _verdict(text: str) -> None:
    st.markdown(f'<div class="rp-verdict">◆ {text}</div>', unsafe_allow_html=True)


def _highlights(items) -> None:
    for h in items or []:
        st.markdown(f"- {h}")


def _ai_block(title: str, data: dict, key: str) -> None:
    """Render the optional AI-summary control + output for one report."""
    if not ai_reports.is_available():
        return
    if st.button(f"✨ AI summary — {title}", key=f"ai_{key}",
                 use_container_width=True):
        with st.spinner("Asking Claude to read the report…"):
            try:
                st.session_state[f"ai_out_{key}"] = ai_reports.summarize_report(title, data)
            except Exception as exc:
                st.session_state[f"ai_out_{key}"] = f"AI summary failed: {exc}"
    out = st.session_state.get(f"ai_out_{key}")
    if out:
        st.markdown(f'<div class="rp-ai">{out}</div>', unsafe_allow_html=True)


tab_amd, tab_perf, tab_sys, tab_macro = st.tabs(
    ["🧩 AMD Scans", "📓 Performance", "🩺 System Activity", "🌐 Market & Macro"]
)

# ── AMD scans ──────────────────────────────────────────────────────────────
with tab_amd:
    rep = reporting.amd_scan_report()
    _verdict(rep["verdict"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Instruments scanned", rep["scanned_instruments"])
    c2.metric("Scans recorded", rep["total_scans_recorded"])
    c3.metric("Phases seen", len(rep["phase_counts"]))
    _highlights(rep["highlights"])

    if rep["latest_by_instrument"]:
        st.markdown("**Latest phase by instrument**")
        df = pd.DataFrame(list(rep["latest_by_instrument"].values()))
        st.dataframe(df[[c for c in ("instrument", "phase", "bias",
                                     "last_close", "ts") if c in df]],
                     use_container_width=True, height=320)
    else:
        st.info("No AMD scans recorded yet. Open the **AMD Scanner** page and run "
                "a few instruments — each scan is recorded here automatically.")
    _ai_block("AMD Scans", rep, "amd")

# ── Trade performance ──────────────────────────────────────────────────────
with tab_perf:
    rep = reporting.trade_performance_report(window=window)
    _verdict(rep["verdict"])
    if not rep.get("available"):
        st.warning(rep.get("reason", "Trade journal unavailable."))
    else:
        s = rep["stats"]
        m = st.columns(5)
        m[0].metric("Win rate", f"{s['win_rate']:.0f}%")
        m[1].metric("Trades", s["total"])
        m[2].metric("Net R", f"{rep['net_r']:+.1f}R")
        m[3].metric("Profit factor", f"{s['profit_factor']:.2f}")
        m[4].metric("Expectancy", f"{s['expectancy']:+.2f}R")
        _highlights(rep["highlights"])

        if rep["equity_curve"]:
            eq = pd.DataFrame(rep["equity_curve"])
            st.markdown("**Equity curve (cumulative R)**")
            st.line_chart(eq.set_index("logged_at")["cum_r"], height=240)
        if rep["by_instrument"]:
            st.markdown("**Net R by instrument**")
            st.bar_chart(pd.Series(rep["by_instrument"]), height=240)
    _ai_block("Trade Performance", rep, "perf")

# ── System activity ────────────────────────────────────────────────────────
with tab_sys:
    rep = reporting.system_activity_report()
    _verdict(rep["verdict"])
    m = st.columns(4)
    m[0].metric("Events", rep["total_events"])
    m[1].metric("Sessions", rep["sessions"])
    m[2].metric("Data fetches", rep["data_fetches"]["total"])
    m[3].metric("Errors", rep["errors"])
    _highlights(rep["highlights"])

    if rep["page_views_by_page"]:
        st.markdown("**Page views**")
        st.bar_chart(pd.Series(rep["page_views_by_page"]), height=240)
    if rep["recent_errors"]:
        st.markdown("**Recent errors**")
        st.dataframe(pd.DataFrame(rep["recent_errors"]),
                     use_container_width=True, height=240)
    _ai_block("System Activity", rep, "sys")

# ── Market & macro snapshot ────────────────────────────────────────────────
with tab_macro:
    with st.spinner("Building macro snapshot…"):
        rep = reporting.market_snapshot_report()
    _verdict(rep["verdict"])
    if not rep["macro_live"]:
        st.warning("FRED live data unavailable — macro bias uses fallback values.")
    _highlights(rep["highlights"])

    if rep["macro_bias"]:
        st.markdown("**Macro bias by currency**")
        bias_df = pd.DataFrame(
            [{"currency": c, "bias": b,
              **{k: rep["macro_raw"].get(c, {}).get(k)
                 for k in ("GDP", "Inflation", "Rates", "Unemployment")}}
             for c, b in rep["macro_bias"].items()]
        )
        st.dataframe(bias_df, use_container_width=True, height=320)
    if rep["amd_phase_counts"]:
        st.markdown("**Current AMD phase distribution**")
        st.bar_chart(pd.Series(rep["amd_phase_counts"]), height=220)
    _ai_block("Market & Macro Snapshot", rep, "macro")
