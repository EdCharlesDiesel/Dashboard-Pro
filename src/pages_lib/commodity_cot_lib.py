"""
commodity_cot_lib.py
====================
Shared render logic for disaggregated-COT commodities (gold, WTI). Thin
per-commodity pages live in pages/gold_cot_tab.py and pages/oil_cot_tab.py.

Layout mirrors the FX COT pages:
    header metrics -> price vs Managed Money net overlay -> z/percentile
    history -> Producer vs Managed Money divergence -> signal table
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from src.services.alert_service import NotifyCache
from src.services.cot_disaggregated import (
    CONTRACTS,
    compute_signals,
    fetch_disaggregated,
    latest_snapshot,
)
from src.services.tool_log import log_tool_usage

SIGNAL_BADGE = {
    "CROWDED_LONG": "🔴 CROWDED LONG (fade / pullback risk)",
    "CROWDED_SHORT": "🟢 CROWDED SHORT (squeeze / bounce risk)",
    "NEUTRAL": "⚪ NEUTRAL",
    "WARMUP": "⏳ insufficient history",
}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _price_history(ticker: str, years: int) -> pd.Series:
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    px = yf.download(ticker, start=start, interval="1d",
                     auto_adjust=False, progress=False)
    if px.empty:
        return pd.Series(dtype=float)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    s = px["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def render_commodity_cot_tab(commodity: str) -> None:
    meta = CONTRACTS[commodity]
    st.header(f"🛢️ {meta['name']} — Disaggregated COT" if commodity == "WTI"
              else f"🥇 {meta['name']} — Disaggregated COT")
    st.caption(
        f"Tradeable as **{meta['spot_ticker']}** on MT4/MT5. "
        "Speculators = Managed Money; commercials = Producer/Merchant. "
        "Report is as of Tuesday, published Friday ~21:30 SAST."
    )

    years = st.slider("History (years)", 3, 10, 6, key=f"{commodity}_years")

    try:
        raw = fetch_disaggregated(commodity, years)
    except Exception as exc:  # network / API failure
        st.error(f"CFTC API fetch failed: {exc}")
        return
    if raw.empty:
        st.warning("No COT rows returned — check contract code / API status.")
        return

    sig = compute_signals(raw)
    snap = latest_snapshot(sig)
    price = _price_history(meta["price_ticker"], years)

    # Log this read to Postgres (audit trail, not trade_setups — CROWDED_* is
    # explicitly caveated below as "a risk flag, not a standalone fade
    # signal"). Deduped per commodity+report via NotifyCache, since CFTC data
    # only updates weekly.
    _cot_key = f"{commodity}|{snap['report_date']}"
    if NotifyCache("commodity_cot_log").filter_new([_cot_key]):
        log_tool_usage(f"{commodity.lower()}_cot", {
            "commodity": commodity, "report_date": str(snap["report_date"]),
            "mm_net": snap["mm_net"], "mm_net_chg_wow": snap["mm_net_chg_wow"],
            "mm_net_pct_oi": snap["mm_net_pct_oi"], "pctl_3y": snap["pctl_3y"],
            "z_156w": snap["z_156w"], "signal": snap["signal"],
        })

    # ---- header metrics ----------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MM net (contracts)", f"{snap['mm_net']:,}",
              delta=f"{snap['mm_net_chg_wow']:+,} w/w")
    c2.metric("MM net %OI", f"{snap['mm_net_pct_oi']}%")
    c3.metric("3y percentile", f"{snap['pctl_3y']}"
              if snap["pctl_3y"] is not None else "–")
    c4.metric("Z-score (156w)", f"{snap['z_156w']}"
              if snap["z_156w"] is not None else "–")
    st.subheader(SIGNAL_BADGE.get(snap["signal"], snap["signal"]))
    st.caption(f"Latest report: {snap['report_date']}")

    # ---- price vs MM net overlay --------------------------------------------
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not price.empty:
        fig.add_trace(go.Scatter(x=price.index, y=price.values,
                                 name=f"{meta['price_ticker']} close",
                                 line=dict(color="#888")),
                      secondary_y=False)
    fig.add_trace(go.Scatter(x=sig["date"], y=sig["mm_net"],
                             name="Managed Money net",
                             line=dict(color="#1f77b4")),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=sig["date"], y=sig["prod_net"],
                             name="Producer/Merchant net",
                             line=dict(color="#d62728", dash="dot")),
                  secondary_y=True)
    fig.update_layout(height=420, title="Price vs positioning",
                      legend=dict(orientation="h"))
    fig.update_yaxes(title_text="price", secondary_y=False)
    fig.update_yaxes(title_text="net contracts", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # ---- percentile history with crowded bands ------------------------------
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sig["date"], y=sig["pctl_3y"],
                              name="MM net %OI percentile (3y)",
                              line=dict(color="#2ca02c")))
    fig2.add_hrect(y0=90, y1=100, fillcolor="red", opacity=0.12, line_width=0)
    fig2.add_hrect(y0=0, y1=10, fillcolor="green", opacity=0.12, line_width=0)
    fig2.update_layout(height=280, title="Positioning percentile",
                       yaxis_range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)

    # ---- signal history table ------------------------------------------------
    with st.expander("Recent report history"):
        cols = ["date", "oi", "mm_net", "mm_net_pct_oi",
                "prod_net_pct_oi", "z_156w", "pctl_3y", "signal"]
        st.dataframe(sig[cols].tail(26).iloc[::-1],
                     use_container_width=True, hide_index=True)

    if commodity == "WTI":
        st.info(
            "Oil caveat: WTI trends on supply shocks (OPEC, inventories, "
            "geopolitics) and crowded positioning can stay crowded far longer "
            "than in gold. Treat CROWDED_* here as a *risk flag*, not a "
            "standalone fade signal — confirm with your regime filter."
        )
    else:
        st.info(
            "Gold pairs naturally with your bonds/gold/DXY page: a "
            "CROWDED_LONG print plus rising real yields is the classic "
            "pullback setup; CROWDED_SHORT plus falling real yields the "
            "classic squeeze."
        )