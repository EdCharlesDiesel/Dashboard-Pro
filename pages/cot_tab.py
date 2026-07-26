"""COT Positioning — CFTC Commitment of Traders analysis.

Weekly institutional/speculative positioning for forex majors, gold, and DXY.
Harmonised to the Bloomberg terminal look (`BloombergTheme`) and wired into
the shared sidebar nav, following the pattern used by the other legacy pages
(e.g. `vwap-ema-gold.py`).

Educational/context tool — a positioning-crowdedness read, not a trade
signal on its own, so it isn't wired into the signal store.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.pages_lib.navigation import render_sidebar_nav
from src.services.alert_service import NotifyCache
from src.services.cot_fetcher import (
    INSTRUMENTS,
    compute_extremeness,
    get_all_snapshot,
    get_instrument_series,
)
from src.services.tool_log import log_tool_usage
from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="COT Positioning · Trading System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

T = BloombergTheme  # short alias for palette tokens

st.title("🏛️ COT Positioning")
st.caption(
    "Weekly institutional/speculative positioning from CFTC COT reports. "
    "FX majors use the Leveraged Funds category (Traders in Financial Futures "
    "report); Gold and DXY use the Non-Commercial category (Legacy report), "
    "since TFF doesn't cover them. Data lags the market by a few days — "
    "CFTC publishes Friday afternoon for the Tuesday close."
)

# Instrument -> yfinance ticker for the optional price overlay (mirrors the
# registry's tickers where the instrument is a registry entry; DXY isn't).
PRICE_TICKERS = {
    "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "USDJPY=X", "CHF": "USDCHF=X",
    "AUD": "AUDUSD=X", "NZD": "NZDUSD=X", "CAD": "USDCAD=X",
    "GOLD": "GC=F", "DXY": "DX-Y.NYB",
}

# ── Plotly terminal styling ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=T.BG_PANEL,
    font=dict(family=T.FONT_MONO, color=T.GREY, size=11),
    legend=dict(font=dict(color=T.GREY, size=11), bgcolor=T.BG_ELEV,
                bordercolor=T.BORDER, orientation="h", x=0, y=1.08),
    margin=dict(l=10, r=10, t=30, b=10),
)


def style_axes(fig: go.Figure, **kwargs) -> None:
    fig.update_xaxes(showgrid=False, tickfont=dict(color=T.GREY, size=9),
                     linecolor=T.BORDER, **kwargs)
    fig.update_yaxes(showgrid=True, gridcolor=T.BORDER,
                     tickfont=dict(color=T.GREY, size=9), **kwargs)


@st.cache_data(ttl=600, show_spinner=False)
def _price_series(ticker: str, years_back: int) -> pd.DataFrame:
    """Cached daily close series for the price overlay only — the COT data
    itself is fetched/cached separately by `cot_fetcher` (CSV-on-disk)."""
    from src.db.market_cache import cached_ohlc
    try:
        df = cached_ohlc(ticker, start=datetime.now() - timedelta(days=years_back * 365),
                         interval="1d", ttl=600)
        if df is None or df.empty or "Close" not in df.columns:
            return pd.DataFrame()
        s = df["Close"].dropna()
        return pd.DataFrame({"date": s.index, "close": s.values})
    except Exception:
        return pd.DataFrame()


def _positioning_chart(df: pd.DataFrame, instrument: str, price_df: pd.DataFrame | None) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["net_spec"],
            name="Net Speculative Position",
            marker_color=[T.GREEN if v >= 0 else T.RED for v in df["net_spec"]],
            opacity=0.7,
        ),
        secondary_y=False,
    )

    if price_df is not None and not price_df.empty:
        fig.add_trace(
            go.Scatter(
                x=price_df["date"], y=price_df["close"],
                name=f"{instrument} price", line=dict(color=T.CYAN, width=1.5),
            ),
            secondary_y=True,
        )

    fig.update_yaxes(title_text="Net Contracts", secondary_y=False)
    fig.update_yaxes(title_text="Price", secondary_y=True, showgrid=False)
    style_axes(fig)
    fig.update_layout(
        **PLOTLY_LAYOUT, height=450,
        title=dict(text=f"{instrument} — Net Speculative Positioning vs Price",
                   font=dict(color=T.WHITE, size=13)),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — symbol/params first, then nav (sidebar contract)
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("COT Settings")
    instrument = st.selectbox("Instrument", sorted(INSTRUMENTS.keys()), index=0)
    years_back = st.slider("Years of history", 1, 8, 4)
    refresh = st.button("🔄 Force refresh from CFTC", width="stretch")
    if refresh:
        _price_series.clear()

    st.divider()
    render_sidebar_nav()

try:
    with st.spinner(f"Loading {instrument} COT data..."):
        series = get_instrument_series(instrument, years_back=years_back, force_refresh=refresh)
except Exception as e:
    st.error(f"Couldn't load COT data for {instrument}: {e}")
    st.stop()

ext = compute_extremeness(series)
price_df = _price_series(PRICE_TICKERS[instrument], years_back)

# Log this positioning read to Postgres (audit trail, not trade_setups — this
# is a crowdedness read, not a directional pair+bias trade signal; see the
# module docstring). Deduped per instrument+week via NotifyCache, since the
# CFTC data only updates weekly — no point logging the same read every rerun.
_latest_date = series["date"].iloc[-1] if not series.empty else None
_cot_key = f"{instrument}|{_latest_date}"
if NotifyCache("cot_tab_log").filter_new([_cot_key]):
    log_tool_usage("cot_tab", {
        "instrument": instrument, "week": str(_latest_date),
        "latest_net": ext.get("latest_net"), "percentile": ext.get("percentile"),
        "z_score": ext.get("z_score"), "label": ext.get("label"),
        "net_change_wow": float(series["net_change_wow"].iloc[-1]) if not series.empty else None,
    })

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest Net Position", f"{ext.get('latest_net', 0):,.0f}")
m2.metric("Week-over-Week Change", f"{series['net_change_wow'].iloc[-1]:,.0f}")
m3.metric("3Y Percentile", f"{ext.get('percentile', 0):.0f}th")
m4.metric("Z-score", f"{ext.get('z_score', 0):.2f}")

label = ext.get("label", "")
if "Extreme" in label:
    st.warning(f"**{label}** — positioning is at a historical extreme; crowded positioning can precede reversals, but timing is unreliable on its own.")
elif "Stretched" in label:
    st.info(f"**{label}**")
else:
    st.write(f"Read: **{label}**")

st.plotly_chart(_positioning_chart(series, instrument, price_df), width="stretch")

with st.expander("Raw weekly data"):
    st.dataframe(
        series.tail(52).sort_values("date", ascending=False),
        width="stretch",
        hide_index=True,
    )

st.divider()
st.markdown("**Cross-market snapshot (latest week)**")
with st.spinner("Building snapshot across all instruments..."):
    snap = get_all_snapshot(years_back=years_back)
if not snap.empty:
    st.dataframe(
        snap.sort_values("percentile_3y", ascending=False),
        width="stretch",
        hide_index=True,
    )
else:
    st.warning("Snapshot unavailable.")
