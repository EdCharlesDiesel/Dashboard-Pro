"""
pages/swing_playbook_tab.py
============================
Swing Playbook — weekly pre-flight. One row per instrument: price, COT
crowdedness, gold's real-yield disconnect read, nearby swing-pivot supply/
demand levels, and vol-scaled position sizing, plus a per-instrument weekly
thesis journal. All data comes from services this app already has — see
src/services/swing_playbook_service.py.

Audit-only (not a trade_setups signal): the "bias" per row is a hand-typed
weekly thesis, not a computed directional read, so playbook snapshots are
logged to tool_usage_log (like Disconnect Monitor / Quant Models) rather than
persist_signals.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src.instruments.registry import INSTRUMENTS
from src.pages_lib.navigation import render_sidebar_nav
from src.services import account_state, open_positions
from src.services.alert_service import NotifyCache
from src.services.swing_playbook_service import (
    build_playbook,
    load_thesis,
    save_thesis,
    week_start,
)
from src.services.surprise_service import event_gate
from src.services.tool_log import log_tool_usage
from src.ui.theme import BloombergTheme

T = BloombergTheme

st.set_page_config(
    page_title="Swing Playbook",
    page_icon="📔",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

DEFAULT_INSTRUMENTS = ["EUR/USD", "GBP/USD", "AUD/USD", "USD/JPY", "XAU/USD", "XAG/USD"]

with st.sidebar:
    st.markdown("### 📔 Swing Playbook")

    all_instruments = INSTRUMENTS.keys()
    selected = st.multiselect(
        "Instruments", all_instruments,
        default=[i for i in DEFAULT_INSTRUMENTS if i in all_instruments],
    )

    st.divider()
    # Sized off the balance the MT5 sync stores. This page used a hardcoded
    # $10,000 and never touched session state at all, so unlike the other
    # sizing pages it could not even inherit the right number by arriving from
    # one - every weekly plan came out 2.6x too large against a real $3,844.
    acct = account_state.get()
    live_bal = account_state.get_balance(0.0)
    has_live = bool(acct) and live_bal > 0

    use_live = st.checkbox(
        "🔗 Use live balance from MT5",
        value=bool(st.session_state.get("swpb_use_live_bal", has_live)) and has_live,
        disabled=not has_live,
        help="Reads the balance the MT5 sync stores. Uncheck to plan against a "
             "hypothetical account size.",
    )
    st.session_state["swpb_use_live_bal"] = use_live

    if use_live and has_live:
        # Assigned every run, never setdefault: a value seeded once goes stale
        # in silence as the account moves, which is the same bug in a new coat.
        account = live_bal
        st.session_state["account_bal"] = live_bal
        _age = open_positions.age_minutes()
        if _age is not None and _age > 15:
            st.error(f"⚠ Balance is **{_age:.0f} min old** - the MT5 sync is "
                     f"not running. See logs/mt5_sync.log.")
        st.caption(f"Account size: **${live_bal:,.2f}** live · "
                   f"{acct.get('source', '')}")
    else:
        account = st.number_input(
            "Account size", min_value=0.0, step=500.0,
            value=float(st.session_state.get("account_bal", 10_000.0)))
        st.session_state["account_bal"] = account
    risk_pct = st.number_input(
        "Risk per trade (%)", value=1.0, min_value=0.25, max_value=3.0, step=0.25,
    ) / 100

    st.divider()
    if st.button("🔄 Refresh Data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} local")

    st.divider()
    render_sidebar_nav()

from src.ui import tv_charts
_tv_engine = tv_charts.engine_toggle("swingpb")

st.markdown(f"""
<div style="background:linear-gradient(135deg,#000000 0%,#0a0a0a 50%,#000000 100%);
            border:1px solid {T.BORDER}; border-radius:16px; padding:24px 28px;
            margin-bottom:20px;">
  <div style="font-size:24px; font-weight:700; color:{T.WHITE};">📔 Swing Playbook</div>
  <div style="color:{T.GREY}; font-size:13px; margin-top:4px;">
    Weekly pre-flight — positioning, disconnect, zones, sizing, and your written thesis
  </div>
  <div style="font-size:12px; color:{T.AMBER}; margin-top:6px;">
    Week of {week_start().strftime('%A %d %B %Y')}
  </div>
</div>
""", unsafe_allow_html=True)

if not selected:
    st.info("Select at least one instrument in the sidebar.")
    st.stop()

with st.spinner("Building playbook…"):
    rows = build_playbook(selected, account, risk_pct, event_gate_fn=event_gate)

df = pd.DataFrame([{
    "Instrument": r.instrument,
    "Price": f"{r.price:,.5f}" if r.price and abs(r.price) < 100 else (f"{r.price:,.2f}" if r.price else "N/A"),
    "Bias": r.bias,
    "COT %ile": f"{r.cot_pctile:.0f}" if r.cot_pctile is not None else "N/A",
    "Disc z": f"{r.disconnect_z:+.1f}" if r.disconnect_z is not None else "—",
    "Gate": r.gate,
    "Supply": f"{r.supply:,.5f}" if r.supply else "—",
    "Demand": f"{r.demand:,.5f}" if r.demand else "—",
    "Stop %": f"{r.stop_pct:.2f}" if r.stop_pct else "N/A",
    "Units @ risk": f"{r.units:,.2f}" if r.units else "N/A",
    "Flags": " | ".join(r.flags) if r.flags else "",
} for r in rows])


def _style(row):
    if "NO THESIS" in row["Flags"]:
        return [f"background-color: {T.RED}26"] * len(row)
    if row["Gate"] == "BLOCKED":
        return [f"background-color: {T.YELLOW}26"] * len(row)
    return [""] * len(row)


st.dataframe(df.style.apply(_style, axis=1), width="stretch", hide_index=True)
st.caption(
    "Red = no written thesis. Yellow = event gate active. "
    "Units sized so a 2·sigma·sqrt(8d) adverse move = your risk %."
)

# --- price & supply/demand chart -------------------------------------------
# Visualises the row's swing zones on the actual candles so the numbers in the
# table above become actionable levels. Dual engine (TradingView pilot /
# Terminal Plotly) — same opt-in toggle as the other price-action pages.
st.subheader("📊 Price & swing zones")
chart_inst = st.selectbox("Chart instrument", selected, key="pb_chart_inst")
_crow = next((r for r in rows if r.instrument == chart_inst), None)
from src.services.market_data import daily_ohlc
_cdf = daily_ohlc(INSTRUMENTS[chart_inst]["ticker"])
if _cdf is None or _cdf.empty:
    st.info(f"No price history available for {chart_inst}.")
else:
    _view = _cdf.tail(120)
    _sup = _crow.supply if _crow else None
    _dem = _crow.demand if _crow else None
    _px = _crow.price if _crow else None
    _times = list(_view.index)
    _used_tv = False
    if _tv_engine:
        try:
            _series = [tv_charts.candle_series(_view)]
            if _sup:
                _series.append(tv_charts.level_series(_times, _sup, T.RED,
                               f"Supply {_sup:,.5f}", dash="dash"))
            if _dem:
                _series.append(tv_charts.level_series(_times, _dem, T.GREEN,
                               f"Demand {_dem:,.5f}", dash="dash"))
            if _px:
                _series.append(tv_charts.level_series(_times, _px, T.WHITE,
                               f"{_px:,.5f}", dash="solid"))
            tv_charts.render([tv_charts.price_chart(_series, height=460)],
                             key=f"tv_swingpb_{chart_inst}")
            st.caption("🅣 TradingView pilot · daily candles + supply (red) / "
                       "demand (green) swing zones + current price.")
            _used_tv = True
        except Exception as exc:
            st.warning(f"TradingView render failed ({exc}); showing Terminal chart.")
    if not _used_tv:
        from src.ui.charts import ChartKit
        _fig = ChartKit.panels([f"{chart_inst} — Daily + swing zones"])
        ChartKit.candles(_fig, _view)
        if _sup:
            ChartKit.hline(_fig, _sup, T.RED, label=f"Supply {_sup:,.5f}")
        if _dem:
            ChartKit.hline(_fig, _dem, T.GREEN, label=f"Demand {_dem:,.5f}")
        if _px:
            ChartKit.hline(_fig, _px, T.WHITE, dash="solid", label=f"{_px:,.5f}")
        st.plotly_chart(ChartKit.finish(_fig, height=460), width="stretch",
                        config=ChartKit.PLOTLY_CONFIG)

# --- audit log (not a trade_setups signal — see module docstring) ----------
_log_key = f"{week_start()}_{account}_{risk_pct}"
_new = NotifyCache("swing_playbook_log").filter_new([f"{r.instrument}_{_log_key}" for r in rows])
if _new:
    for r in rows:
        if f"{r.instrument}_{_log_key}" in _new:
            log_tool_usage("swing_playbook", {
                "instrument": r.instrument, "week": str(week_start()),
                "price": r.price, "bias": r.bias, "cot_pctile": r.cot_pctile,
                "disconnect_z": r.disconnect_z, "gate": r.gate,
                "supply": r.supply, "demand": r.demand,
                "stop_pct": r.stop_pct, "units": r.units, "flags": r.flags,
            })

# --- weekly thesis entry -----------------------------------------------
st.subheader("Weekly thesis")
inst = st.selectbox("Instrument", selected)
existing = load_thesis(inst)
bias_options = ["long", "short", "flat"]
default_bias_idx = bias_options.index(existing["bias"]) if existing and existing.get("bias") in bias_options else 2
bias = st.radio("Bias", bias_options, index=default_bias_idx, horizontal=True)
inval = st.text_area(
    "Invalidation (a condition, not a price)",
    value=(existing.get("invalidation") or "") if existing else "",
    placeholder="e.g. short gold invalid if Warsh signals no hike "
                "and year-end hike odds drop below 60%",
)
if st.button("Save thesis"):
    if bias != "flat" and not inval.strip():
        st.error("A directional thesis needs an invalidation condition.")
    elif save_thesis(inst, bias, inval.strip()):
        st.success(f"{inst} thesis saved for this week.")
        st.rerun()
    else:
        st.warning("Database not configured — thesis wasn't saved. Check the sidebar DB connection.")
