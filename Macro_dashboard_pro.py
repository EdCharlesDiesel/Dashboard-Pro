"""
McKinsey / Hedge-Fund Grade Forex Macro Dashboard
================================================

Features:
- QuantConnect-ready data layer
- Gold (XAU/USD)
- FRED macro data
- Retry-safe fetching
- Technical indicators
- Trading Ideas (multi-factor signals)

Run:
pip install streamlit plotly pandas numpy yfinance ta fredapi tenacity requests
streamlit run forex_macro_dashboard_pro.py
"""

import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import ta
from fredapi import Fred
from tenacity import retry, stop_after_attempt, wait_fixed

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Macro Dashboard Pro", layout="wide")

ASSETS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/ZAR": "USDZAR=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "XAU/USD": "XAUUSD=X",
}

# ─────────────────────────────────────────────────────────────
# DATA PROVIDERS
# ─────────────────────────────────────────────────────────────

class MarketDataProvider:
    def get_data(self, symbol: str, timeframe: str):
        raise NotImplementedError


class QuantConnectProvider(MarketDataProvider):
    BASE_URL = "http://localhost:5000/api/marketdata"

    def get_data(self, symbol, timeframe):
        url = f"{self.BASE_URL}/{symbol}?tf={timeframe}"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df


class FallbackProvider(MarketDataProvider):
    def get_data(self, symbol, timeframe):
        return yf.Ticker(symbol).history(period=timeframe)


def get_provider():
    try:
        provider = QuantConnectProvider()
        test = provider.get_data("EURUSD", "1d")
        if not test.empty:
            return provider
    except:
        pass
    return FallbackProvider()


provider = get_provider()

# ─────────────────────────────────────────────────────────────
# RETRY
# ─────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def safe_fetch(symbol, timeframe):
    return provider.get_data(symbol, timeframe)

# ─────────────────────────────────────────────────────────────
# MACRO (FRED)
# ─────────────────────────────────────────────────────────────

fred = Fred(api_key=os.getenv("FRED_API_KEY"))

@st.cache_data(ttl=3600)
def get_macro_data():
    try:
        return {
            "USD": {
                "GDP": fred.get_series_latest_release("GDP").iloc[-1],
                "Inflation": fred.get_series_latest_release("CPIAUCSL").pct_change().iloc[-1] * 100,
                "Rates": fred.get_series_latest_release("DFF").iloc[-1]
            }
        }
    except:
        return {"USD": {"GDP": 0, "Inflation": 0, "Rates": 0}}

# ─────────────────────────────────────────────────────────────
# TECHNICALS
# ─────────────────────────────────────────────────────────────

def add_indicators(df):
    if df.empty or len(df) < 20:
        return df

    df = df.copy()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    return df

# ─────────────────────────────────────────────────────────────
# GOLD SIGNAL
# ─────────────────────────────────────────────────────────────

def gold_signal(dxy_df, rates):
    if dxy_df.empty:
        return "Neutral"

    trend = dxy_df['Close'].pct_change().rolling(5).mean().iloc[-1]

    if trend > 0 and rates > 3:
        return "Bearish ❌"
    elif trend < 0 and rates < 3:
        return "Bullish ✅"
    return "Neutral ⚖️"

# ─────────────────────────────────────────────────────────────
# TRADING IDEAS ENGINE
# ─────────────────────────────────────────────────────────────

def generate_trading_ideas(data, macro, dxy):
    ideas = []

    for name, df in data.items():
        if df is None or df.empty or len(df) < 50:
            continue

        df = add_indicators(df)

        last = df.iloc[-1]
        price = last['Close']
        rsi = last.get('RSI', 50)

        sma20 = df['Close'].rolling(20).mean().iloc[-1]
        macd = last.get('MACD', 0)
        macd_signal = last.get('MACD_Signal', 0)

        bias = None
        conviction = "Low"
        thesis = []

        # RSI
        if rsi < 30:
            bias = "Long"
            conviction = "High" if rsi < 25 else "Medium"
            thesis.append("Oversold RSI")
        elif rsi > 70:
            bias = "Short"
            conviction = "High" if rsi > 75 else "Medium"
            thesis.append("Overbought RSI")

        # Trend
        if price > sma20 * 1.01:
            bias = bias or "Long"
            thesis.append("Uptrend (SMA20)")
        elif price < sma20 * 0.99:
            bias = bias or "Short"
            thesis.append("Downtrend (SMA20)")

        # MACD
        if macd > macd_signal:
            thesis.append("MACD Bullish")
        else:
            thesis.append("MACD Bearish")

        # Gold Macro
        if name == "XAU/USD":
            macro_bias = gold_signal(dxy, macro["USD"]["Rates"])
            thesis.append(f"Macro: {macro_bias}")

            if "Bullish" in macro_bias:
                bias = "Long"
            elif "Bearish" in macro_bias:
                bias = "Short"

        if not bias:
            continue

        # Risk (ATR-style)
        atr = df['High'].rolling(14).max().iloc[-1] - df['Low'].rolling(14).min().iloc[-1]
        atr = atr if atr > 0 else price * 0.01

        if bias == "Long":
            entry = price
            stop = price - atr * 0.5
            target = price + atr
        else:
            entry = price
            stop = price + atr * 0.5
            target = price - atr

        ideas.append({
            "pair": name,
            "bias": bias,
            "conviction": conviction,
            "thesis": " | ".join(thesis),
            "entry": entry,
            "target": target,
            "stop": stop
        })

    return ideas

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y"])

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    result = {}
    for name, symbol in ASSETS.items():
        try:
            df = safe_fetch(symbol, timeframe)
            if not df.empty:
                result[name] = df
        except:
            continue
    return result

data = load_data()
macro = get_macro_data()
dxy = safe_fetch("DX-Y.NYB", timeframe)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.title("💹 Macro Dashboard Pro")

# ─────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────

cols = st.columns(5)
pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/ZAR", "XAU/USD"]

for i, pair in enumerate(pairs):
    with cols[i]:
        df = data.get(pair)
        if df is not None and not df.empty:
            price = df['Close'].iloc[-1]
            change = df['Close'].pct_change().iloc[-1] * 100
            st.metric(pair, f"{price:.4f}", f"{change:.2f}%")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Technicals",
    "🥇 Gold",
    "🎯 Trading Ideas"
])

# OVERVIEW
with tab1:
    perf = []
    for name, df in data.items():
        if len(df) > 1:
            ret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
            perf.append({"Asset": name, "Return %": ret})

    st.plotly_chart(px.bar(pd.DataFrame(perf), x="Asset", y="Return %"), use_container_width=True)

# TECHNICALS
with tab2:
    pair = st.selectbox("Pair", list(data.keys()))
    df = add_indicators(data[pair])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))
    st.plotly_chart(fig, use_container_width=True)

# GOLD
with tab3:
    gold = data.get("XAU/USD")
    if gold is not None:
        signal = gold_signal(dxy, macro["USD"]["Rates"])
        st.metric("Gold Signal", signal)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gold.index, y=gold['Close']))
        st.plotly_chart(fig, use_container_width=True)

# TRADING IDEAS
with tab4:
    ideas = generate_trading_ideas(data, macro, dxy)

    col1, col2 = st.columns([3, 1])

    with col1:
        for idea in ideas[:8]:
            txt = f"**{idea['pair']} — {idea['bias']}** ({idea['conviction']})  \n{idea['thesis']}  \nEntry: {idea['entry']:.4f} | Target: {idea['target']:.4f} | Stop: {idea['stop']:.4f}"
            if idea["bias"] == "Long":
                st.success(txt)
            else:
                st.error(txt)

    with col2:
        if ideas:
            long_count = sum(1 for i in ideas if i["bias"] == "Long")
            short_count = sum(1 for i in ideas if i["bias"] == "Short")

            st.plotly_chart(go.Figure(go.Bar(
                x=["Long", "Short"],
                y=[long_count, short_count]
            )), use_container_width=True)

# FOOTER
st.divider()
st.caption(f"Last updated: {datetime.now()}")