import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
from src.services.signal_store import persist_signals
from src.instruments import INSTRUMENTS as _REGISTRY_INSTRUMENTS

# Page configurations
st.set_page_config(page_title="Forex Analytics Dashboard", layout="wide", page_icon="💱")
BloombergTheme.apply()

st.title("💱 Forex Analytics & Market Dashboard")
st.markdown("Monitor real-time currency trends and track key technical indicators.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Dashboard Settings")

# Sourced from the shared registry (src/instruments/registry.py) so this page
# covers the same universe, with the same tickers, as every other page.
forex_pairs = {name: info.ticker for name, info in _REGISTRY_INSTRUMENTS.items()}

selected_pair_label = st.sidebar.selectbox("Select Currency Pair", list(forex_pairs.keys()))
ticker_symbol = forex_pairs[selected_pair_label]

# Timeframe selection
interval_mapping = {
    "1 Day": {"period": "1mo", "interval": "15m"},
    "1 Week": {"period": "3mo", "interval": "1h"},
    "1 Month": {"period": "6mo", "interval": "1d"},
    "1 Year": {"period": "1y", "interval": "1d"},
    "5 Years": {"period": "5y", "interval": "1wk"}
}
timeframe = st.sidebar.selectbox("Timeframe", list(interval_mapping.keys()), index=2)
period = interval_mapping[timeframe]["period"]
interval = interval_mapping[timeframe]["interval"]

# Moving Average Indicators
st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Simple Moving Average (SMA)", value=True)
sma_period = st.sidebar.slider("SMA Period", min_value=5, max_value=100, value=20)

# --- FETCH DATA ---
@st.cache_data(ttl=60)  # Cache data for 1 minute to avoid spamming the API
def load_forex_data(ticker, p, i):
    from src.db.market_cache import cached_ohlc
    return cached_ohlc(ticker, period=p, interval=i, ttl=60)

try:
    df = load_forex_data(ticker_symbol, period, interval)
    
    if df.empty:
        st.error("No data found for this pair/timeframe combination.")
    else:
        # Flatten columns if multi-indexed by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Calculate metrics
        latest_price = df['Close'].iloc[-1]
        previous_price = df['Close'].iloc[-2] if len(df) > 1 else latest_price
        price_change = latest_price - previous_price
        pct_change = (price_change / previous_price) * 100

        # --- MAIN METRICS ---
        col1, col2, col3 = st.columns(3)
        col1.metric(label=f"Current {selected_pair_label} Rate", value=f"{latest_price:.4f}")
        col2.metric(label="Session Change", value=f"{price_change:+.4f}", delta=f"{pct_change:+.2f}%")
        col3.metric(label="Highest in Period", value=f"{df['High'].max():.4f}")

        # Persist a price-vs-SMA directional read (source='predictive').
        _sma_last = df['Close'].rolling(sma_period).mean().iloc[-1]
        if pd.notna(_sma_last):
            _bull = latest_price > _sma_last
            persist_signals("predictive", [{
                "pair": selected_pair_label,
                "bias": "Long" if _bull else "Short",
                "entry": float(latest_price),
                "conviction": f"Price {'>' if _bull else '<'} {sma_period}SMA",
                "thesis": (f"Analytics — price {latest_price:.4f} "
                           f"{'>' if _bull else '<'} {sma_period}-SMA {_sma_last:.4f}"),
            }])

        # --- TECHNICAL INDICATORS ---
        if show_sma:
            df['SMA'] = df['Close'].rolling(window=sma_period).mean()

        # --- PLOT CHART ---
        st.subheader(f"{selected_pair_label} Performance Chart ({timeframe})")
        
        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Market Price'
        ))

        # Add SMA if selected
        if show_sma:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['SMA'], 
                line=dict(color='orange', width=1.5), 
                name=f'{sma_period} SMA'
            ))

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=500,
            margin=dict(l=10, r=10, t=20, b=20),
            yaxis_title="Exchange Rate",
            xaxis_title="Date/Time"
        )

        st.plotly_chart(fig, width="stretch")

        # --- DATA TABLE VIEW ---
        with st.expander("View Raw Historical Data"):
            st.dataframe(df.sort_index(ascending=False), width="stretch")

except Exception as e:
    st.error(f"An error occurred while loading data: {e}")

with st.sidebar:
    st.divider()
    render_sidebar_nav()
