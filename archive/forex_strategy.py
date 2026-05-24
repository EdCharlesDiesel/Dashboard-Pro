import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------- Page Configuration --------------------
st.set_page_config(layout="wide", page_title="Supertrend + QQE + DEMA Strategy")
st.title("📈 Supertrend + QQE + DEMA Strategy Dashboard")


# -------------------- Manual Indicator Functions (No pandas-ta needed) --------------------
def calculate_atr(high, low, close, period=14):
    """Manual ATR calculation"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()


def calculate_rsi(close, period=14):
    """Manual RSI calculation"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def supertrend(high, low, close, period, multiplier):
    """Manual Supertrend calculation"""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    trend = pd.Series(1, index=close.index)  # 1 for uptrend, -1 for downtrend
    supertrend_line = pd.Series(index=close.index, dtype='float64')

    for i in range(1, len(close)):
        if close.iloc[i] <= supertrend_line.iloc[i - 1] if not pd.isna(supertrend_line.iloc[i - 1]) else False:
            trend.iloc[i] = -1
        elif close.iloc[i] >= supertrend_line.iloc[i - 1] if not pd.isna(supertrend_line.iloc[i - 1]) else False:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = 1 if close.iloc[i] > lower_band.iloc[i] else -1

        if trend.iloc[i] == 1:
            supertrend_line.iloc[i] = max(lower_band.iloc[i],
                                          supertrend_line.iloc[i - 1] if not pd.isna(supertrend_line.iloc[i - 1]) else
                                          lower_band.iloc[i])
        else:
            supertrend_line.iloc[i] = min(upper_band.iloc[i],
                                          supertrend_line.iloc[i - 1] if not pd.isna(supertrend_line.iloc[i - 1]) else
                                          upper_band.iloc[i])

    return trend, supertrend_line


def calculate_dema(close, period):
    """Manual DEMA calculation"""
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    return 2 * ema1 - ema2


def calculate_qqe(close, rsi_period=14, smooth=5, factor=2.618):
    """Manual QQE calculation"""
    # Calculate RSI
    rsi = calculate_rsi(close, rsi_period)

    # Smooth RSI
    rsi_smooth = rsi.ewm(span=smooth, adjust=False).mean()

    # ATR of smoothed RSI
    atr_rsi = abs(rsi_smooth.diff()).ewm(span=smooth, adjust=False).mean()

    # Trailing levels
    upper = rsi_smooth + factor * atr_rsi
    lower = rsi_smooth - factor * atr_rsi

    # QQE line (simplified - captures direction)
    qqe_line = pd.Series(0.0, index=close.index)
    trend = pd.Series(0, index=close.index)

    for i in range(1, len(close)):
        if qqe_line.iloc[i - 1] == upper.iloc[i - 1]:
            if rsi_smooth.iloc[i] < upper.iloc[i]:
                qqe_line.iloc[i] = upper.iloc[i]
                trend.iloc[i] = -1
            else:
                qqe_line.iloc[i] = lower.iloc[i]
                trend.iloc[i] = 1
        elif qqe_line.iloc[i - 1] == lower.iloc[i - 1]:
            if rsi_smooth.iloc[i] > lower.iloc[i]:
                qqe_line.iloc[i] = lower.iloc[i]
                trend.iloc[i] = 1
            else:
                qqe_line.iloc[i] = upper.iloc[i]
                trend.iloc[i] = -1
        else:
            qqe_line.iloc[i] = upper.iloc[i] if rsi_smooth.iloc[i] > 50 else lower.iloc[i]
            trend.iloc[i] = 1 if rsi_smooth.iloc[i] > 50 else -1

    return rsi_smooth, qqe_line, trend


# -------------------- Sidebar Controls --------------------
st.sidebar.header("⚙️ Configuration")

# Ticker Input (Forex Pairs on Yahoo Finance)
ticker = st.sidebar.text_input("Yahoo Ticker (e.g., EURUSD=X)", "EURUSD=X").upper()

# Timeframe & Period
interval = st.sidebar.selectbox("Candlestick Interval",
                                ["15m", "30m", "1h", "4h", "1d"], index=3)
period = st.sidebar.selectbox("Data Period",
                              ["7d", "1mo", "3mo", "6mo", "1y", "2y"], index=3)

# Chart Type Toggle
use_heikin_ashi = st.sidebar.checkbox("Use Heikin Ashi Charts", value=True,
                                      help="Strategy is designed for Heikin Ashi. Disabling may degrade performance.")

# Strategy Parameters
st.sidebar.subheader("Supertrend Settings")
st_atr_period_1 = st.sidebar.slider("Supertrend 1 ATR Period", 5, 20, 10)
st_multiplier_1 = st.sidebar.slider("Supertrend 1 Multiplier", 1.0, 5.0, 3.0, step=0.5)

st_atr_period_2 = st.sidebar.slider("Supertrend 2 ATR Period", 5, 20, 12)
st_multiplier_2 = st.sidebar.slider("Supertrend 2 Multiplier", 1.0, 5.0, 3.5, step=0.5)

st.sidebar.subheader("DEMA Settings")
dema_length = st.sidebar.slider("DEMA Period", 50, 300, 200, step=10)

st.sidebar.subheader("QQE Settings")
qqe_rsi_period = st.sidebar.slider("QQE RSI Period", 6, 21, 14)
qqe_smooth = st.sidebar.slider("QQE Smooth Factor", 3, 10, 5)
qqe_factor = st.sidebar.slider("QQE Fast Factor", 1.0, 5.0, 2.618, step=0.1)


# -------------------- Data Fetching --------------------
@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check the symbol and try again.")
            st.stop()
        df.columns = df.columns.droplevel(1)  # Drop 'Ticker' level from multi-index
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()


df = fetch_data(ticker, period, interval)


# -------------------- Heikin Ashi Conversion --------------------
def calculate_heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha['HA_Open'] = 0.0
    ha['HA_Open'].iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha['HA_Open'].iloc[i] = (ha['HA_Open'].iloc[i - 1] + ha['HA_Close'].iloc[i - 1]) / 2
    ha['HA_High'] = df[['High', 'Open', 'Close']].max(axis=1)
    ha['HA_Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    return ha


if use_heikin_ashi:
    ha = calculate_heikin_ashi(df)
    ohlc = {'Open': ha['HA_Open'], 'High': ha['HA_High'],
            'Low': ha['HA_Low'], 'Close': ha['HA_Close']}
    ohlc_df = pd.DataFrame(ohlc)
else:
    ohlc_df = df[['Open', 'High', 'Low', 'Close']].copy()

# -------------------- Indicator Calculation --------------------
st_trend_1, st_line_1 = supertrend(ohlc_df['High'], ohlc_df['Low'], ohlc_df['Close'],
                                   st_atr_period_1, st_multiplier_1)
st_trend_2, st_line_2 = supertrend(ohlc_df['High'], ohlc_df['Low'], ohlc_df['Close'],
                                   st_atr_period_2, st_multiplier_2)
dema = calculate_dema(ohlc_df['Close'], dema_length)
qqe_rsi, qqe_line, qqe_trend = calculate_qqe(ohlc_df['Close'], qqe_rsi_period, qqe_smooth, qqe_factor)


# -------------------- Signal Generation --------------------
def generate_signals(df, st_trend_1, st_trend_2, dema, qqe_trend):
    """
    Generate buy/sell signals based on confluence:
    - Both Supertrends must agree
    - Price must be on correct side of DEMA
    - QQE trend must confirm
    """
    signals = pd.Series(0, index=df.index)  # 0 = no signal, 1 = buy, -1 = sell

    # Conditions
    st_bullish = (st_trend_1 == 1) & (st_trend_2 == 1)
    st_bearish = (st_trend_1 == -1) & (st_trend_2 == -1)
    price_above_dema = df['Close'] > dema
    price_below_dema = df['Close'] < dema
    qqe_bullish = qqe_trend == 1
    qqe_bearish = qqe_trend == -1

    # Buy Signal
    buy_condition = st_bullish & price_above_dema & qqe_bullish
    signals[buy_condition] = 1

    # Sell Signal
    sell_condition = st_bearish & price_below_dema & qqe_bearish
    signals[sell_condition] = -1

    return signals


signals = generate_signals(ohlc_df, st_trend_1, st_trend_2, dema, qqe_trend)


# -------------------- Performance Metrics --------------------
def calculate_performance(df, signals):
    """Calculate strategy performance metrics"""
    position = 0
    entry_price = 0
    trades = []

    for i in range(1, len(df)):
        # Entry
        if position == 0 and signals.iloc[i] == 1:
            position = 1
            entry_price = df['Close'].iloc[i]
            entry_time = df.index[i]
        elif position == 0 and signals.iloc[i] == -1:
            position = -1
            entry_price = df['Close'].iloc[i]
            entry_time = df.index[i]

        # Exit (signal flip or opposite signal)
        elif position == 1 and signals.iloc[i] == -1:
            exit_price = df['Close'].iloc[i]
            trades.append({'Entry': entry_time, 'Exit': df.index[i],
                           'Direction': 'Long', 'Entry_Price': entry_price,
                           'Exit_Price': exit_price,
                           'PnL': exit_price - entry_price,
                           'Return': (exit_price - entry_price) / entry_price})
            position = 0
        elif position == -1 and signals.iloc[i] == 1:
            exit_price = df['Close'].iloc[i]
            trades.append({'Entry': entry_time, 'Exit': df.index[i],
                           'Direction': 'Short', 'Entry_Price': entry_price,
                           'Exit_Price': exit_price,
                           'PnL': entry_price - exit_price,
                           'Return': (entry_price - exit_price) / entry_price})
            position = 0

    if trades:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['PnL'] > 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        total_return = trades_df['Return'].sum() * 100
        avg_return = trades_df['Return'].mean() * 100

        return {
            'Total Trades': total_trades,
            'Win Rate (%)': round(win_rate, 2),
            'Total Return (%)': round(total_return, 2),
            'Average Return per Trade (%)': round(avg_return, 3),
            'Trades': trades_df
        }
    return {
        'Total Trades': 0,
        'Win Rate (%)': 0,
        'Total Return (%)': 0,
        'Average Return per Trade (%)': 0,
        'Trades': pd.DataFrame()
    }


performance = calculate_performance(ohlc_df, signals)

# -------------------- Main Dashboard --------------------
# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Trades", performance['Total Trades'])
with col2:
    st.metric("Win Rate", f"{performance['Win Rate (%)']}%")
with col3:
    st.metric("Total Return", f"{performance['Total Return (%)']}%")
with col4:
    st.metric("Avg Return/Trade", f"{performance['Average Return per Trade (%)']}%")

# -------------------- Chart --------------------
st.subheader("Price Chart with Indicators")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"{ticker} - {'Heikin Ashi' if use_heikin_ashi else 'Standard'} Chart",
                                    "QQE Indicator"))

# Candlestick chart
fig.add_trace(go.Candlestick(
    x=ohlc_df.index,
    open=ohlc_df['Open'],
    high=ohlc_df['High'],
    low=ohlc_df['Low'],
    close=ohlc_df['Close'],
    name='Price'
), row=1, col=1)

# Supertrend lines
fig.add_trace(go.Scatter(
    x=ohlc_df.index, y=st_line_1,
    mode='lines', name=f'Supertrend 1 ({st_atr_period_1},{st_multiplier_1})',
    line=dict(color='rgba(0, 255, 0, 0.7)', width=1)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=ohlc_df.index, y=st_line_2,
    mode='lines', name=f'Supertrend 2 ({st_atr_period_2},{st_multiplier_2})',
    line=dict(color='rgba(0, 0, 255, 0.7)', width=1)
), row=1, col=1)

# DEMA line
fig.add_trace(go.Scatter(
    x=ohlc_df.index, y=dema,
    mode='lines', name=f'DEMA ({dema_length})',
    line=dict(color='orange', width=2)
), row=1, col=1)

# Buy/Sell Signals on chart
buy_signals = signals == 1
sell_signals = signals == -1

fig.add_trace(go.Scatter(
    x=ohlc_df.index[buy_signals],
    y=ohlc_df['Low'][buy_signals] * 0.999,
    mode='markers',
    name='Buy Signal',
    marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1, color='darkgreen'))
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=ohlc_df.index[sell_signals],
    y=ohlc_df['High'][sell_signals] * 1.001,
    mode='markers',
    name='Sell Signal',
    marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='darkred'))
), row=1, col=1)

# QQE Panel
fig.add_trace(go.Scatter(
    x=ohlc_df.index, y=qqe_rsi,
    mode='lines', name='QQE Line',
    line=dict(color='purple', width=1.5)
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=ohlc_df.index, y=qqe_line,
    mode='lines', name='QQE Signal',
    line=dict(color='black', width=1, dash='dot')
), row=2, col=1)

# Add 50 level to QQE
fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

# Layout updates
fig.update_layout(
    height=800,
    showlegend=True,
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)

fig.update_xaxes(title_text="Date/Time", row=2, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="QQE", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# -------------------- Strategy Logic Display --------------------
st.subheader("📋 Strategy Confluence Conditions")
col_buy, col_sell = st.columns(2)

with col_buy:
    st.markdown("**Long (Buy) Conditions**")
    st.markdown("""
    ✅ **Dual Supertrend Bullish**
    - Supertrend 1: Green/Uptrend
    - Supertrend 2: Green/Uptrend

    ✅ **Price Above DEMA**
    - Close > 200-period DEMA

    ✅ **QQE Bullish Momentum**
    - QQE Trend = Bullish
    """)

with col_sell:
    st.markdown("**Short (Sell) Conditions**")
    st.markdown("""
    ❌ **Dual Supertrend Bearish**
    - Supertrend 1: Red/Downtrend
    - Supertrend 2: Red/Downtrend

    ❌ **Price Below DEMA**
    - Close < 200-period DEMA

    ❌ **QQE Bearish Momentum**
    - QQE Trend = Bearish
    """)

# -------------------- Trades Table --------------------
if not performance['Trades'].empty:
    st.subheader("📊 Recent Trades")
    st.dataframe(performance['Trades'].tail(10), use_container_width=True)

# -------------------- Disclaimer --------------------
st.warning("""
⚠️ **Disclaimer**: This is a simplified implementation for educational purposes. The strategy logic and indicator 
calculations are approximations and may differ from TradingView implementations. Always backtest thoroughly on 
a demo account before live trading. Past performance does not guarantee future results. Win rates of 50-70% 
are typical for this strategy in optimal conditions.
""")