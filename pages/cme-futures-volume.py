"""CME Currency Futures — Volume Module.

Spot forex tickers on yfinance (e.g. EURUSD=X) report 0 volume because spot FX
is OTC with no consolidated tape — that's why Smart Money (archived, see
archive/pages/smart_money_tab.py) blocks spot pairs outright. CME currency
futures (e.g. 6E=F) trade on a centralized exchange, so they carry real,
exchange-reported volume — a legitimate series to drive OBV/CMF instead of
the flat-zero series spot data gives you.

Caveats:
  - Futures price carries a small basis vs spot (interest-rate differential).
  - Contracts roll quarterly (Mar/Jun/Sep/Dec); yfinance's "=F" tickers are
    continuous front-month series and absorb most of this, but volume can
    still spike/drop sharply in roll weeks — that's normal, not a data bug.
  - 6J=F (JPY) is quoted as USD per 100 JPY — inverted vs how USD/JPY spot is
    usually read. Don't directly compare price levels to USDJPY=X.

Entry point: runs top-level (Streamlit multipage convention).
Standalone:  streamlit run cme-futures-volume.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.core.config import CANDLE_STYLE, CHART_LAYOUT
from src.pages_lib.navigation import render_sidebar_nav
from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="CME FX Futures · Trading System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

# Continuous front-month CME currency futures tickers on yfinance.
CME_FUTURES = {
    "EUR (6E=F)": "6E=F",
    "GBP (6B=F)": "6B=F",
    "JPY (6J=F)": "6J=F",
    "AUD (6A=F)": "6A=F",
    "CAD (6C=F)": "6C=F",
    "NZD (6N=F)": "6N=F",
    "CHF (6S=F)": "6S=F",
    "USD Index (DX=F)": "DX=F",
}

QUOTE_NOTE = {
    "6J=F": "Quoted as USD per 100 JPY — inverted vs spot USDJPY. Lower "
            "price = weaker USD/stronger JPY (same direction as spot JPY "
            "strength), but don't compare raw price levels.",
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA + INDICATORS (pure / testable)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def fetch_futures(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Futures OHLCV via the shared read-through cache (Postgres + pool)."""
    from src.db.market_cache import cached_ohlc
    df = cached_ohlc(ticker, period=period, interval=interval, ttl=900)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative volume signed by the bar's direction."""
    direction = df["Close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * df["Volume"]).fillna(0).cumsum()


def compute_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow: n-period money-flow volume / volume. >0 buying pressure."""
    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, pd.NA)
    mfv = (mfm * vol).fillna(0)
    return mfv.rolling(period).sum() / vol.rolling(period).sum()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — symbol picker first, per the sidebar contract
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📦 CME FX Futures")
    st.caption("Volume-backed money flow for forex, via real exchange volume")

    label = st.selectbox("Currency", list(CME_FUTURES.keys()))
    ticker = CME_FUTURES[label]
    period = st.selectbox("Lookback", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1h"], index=0)
    cmf_window = st.slider("CMF window", 5, 40, 20)

    if st.button("🔄 Refresh", width="stretch", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    render_sidebar_nav()


# ═══════════════════════════════════════════════════════════════════════════
# BODY
# ═══════════════════════════════════════════════════════════════════════════

st.header("📦 CME FX Futures — Volume-Backed Money Flow")
st.caption(
    "Spot forex (=X tickers) reports 0 volume on yfinance — there's no "
    "consolidated OTC tape. These are CME futures, which trade on a real "
    "exchange and carry real volume, so OBV/CMF here are actually meaningful."
)

if ticker in QUOTE_NOTE:
    st.info(QUOTE_NOTE[ticker])

df = fetch_futures(ticker, period, interval)

if df.empty or "Volume" not in df.columns or df["Volume"].fillna(0).sum() <= 0:
    st.warning(f"No volume data returned for {ticker}. Try a different lookback/interval.")
    st.stop()

total_vol = int(df["Volume"].sum())
avg_vol = int(df["Volume"].mean())
last_close = float(df["Close"].iloc[-1])
pct_change = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("Last Close", f"{last_close:,.4f}")
m2.metric("Period Change", f"{pct_change:+.2f}%")
m3.metric("Total Volume", f"{total_vol:,}")
m4.metric("Avg Volume/Bar", f"{avg_vol:,}")

df["OBV"] = compute_obv(df)
df["CMF"] = compute_cmf(df, cmf_window)

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
    vertical_spacing=0.03,
    subplot_titles=(f"{label} Price", "Volume", "OBV / CMF"),
)
fig.add_trace(
    go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        showlegend=False, **CANDLE_STYLE,
    ),
    row=1, col=1,
)
vol_colors = [CANDLE_STYLE["increasing_line_color"] if c >= o else CANDLE_STYLE["decreasing_line_color"]
              for o, c in zip(df["Open"], df["Close"])]
fig.add_trace(
    go.Bar(x=df.index, y=df["Volume"], name="Volume",
           marker_color=vol_colors, showlegend=False),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(width=1.4)),
    row=3, col=1,
)
chart_layout = dict(CHART_LAYOUT)
chart_layout["legend"] = {**chart_layout["legend"], "orientation": "h",
                          "yanchor": "bottom", "y": 1.02}
fig.update_layout(height=750, margin=dict(l=10, r=10, t=40, b=10), **chart_layout)
st.plotly_chart(fig, width="stretch")

with st.expander("CMF (Chaikin Money Flow)"):
    st.line_chart(df["CMF"])
    st.caption(
        "Positive CMF = buying pressure dominant, negative = selling pressure. "
        "This is only meaningful because the underlying volume is real "
        "(futures) — the same calc on spot yfinance data is noise."
    )

with st.expander("Raw data"):
    st.dataframe(df.tail(50), width="stretch")
