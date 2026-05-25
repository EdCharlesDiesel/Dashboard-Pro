import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta

# Page config
st.set_page_config(
    page_title="Forex Trend & Pivot Signals",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Forex Trend Trading & Pivot Points Signals with Entry/TP/SL")
st.markdown("---")

# Sidebar for parameters
st.sidebar.header("⚙️ Configuration")

# Major forex pairs
forex_pairs = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "USDCAD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X"
}

# Sidebar selections
selected_pair = st.sidebar.selectbox("Select Currency Pair", list(forex_pairs.keys()))
timeframe = st.sidebar.selectbox("Select Timeframe", ["1h", "4h", "1d"], index=1)

# Map timeframe to yfinance interval
interval_map = {"1h": "60m", "4h": "4h", "1d": "1d"}
data_period_map = {"1h": "7d", "4h": "30d", "1d": "90d"}

# Technical indicator parameters
st.sidebar.subheader("📊 Indicator Parameters")
ema_fast = st.sidebar.slider("EMA Fast", 5, 30, 12)
ema_slow = st.sidebar.slider("EMA Slow", 20, 50, 26)
rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)
rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 80, 70)
rsi_oversold = st.sidebar.slider("RSI Oversold", 20, 40, 30)
adx_threshold = st.sidebar.slider("ADX Threshold", 15, 35, 25)

# Risk Management Parameters
st.sidebar.subheader("⚠️ Risk Management")
risk_percent = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
reward_risk_ratio = st.sidebar.selectbox("Risk:Reward Ratio", ["1:1", "1:1.5", "1:2", "1:2.5", "1:3"], index=2)
atr_multiplier_sl = st.sidebar.slider("ATR Multiplier for SL", 1.0, 3.0, 1.5, 0.1)
atr_multiplier_tp = st.sidebar.slider("ATR Multiplier for TP", 1.5, 5.0, 3.0, 0.1)


def clean_dataframe(df):
    """Clean and flatten the dataframe to ensure 1-dimensional columns"""
    if df.empty:
        return df

    # If columns are MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            matching_cols = [c for c in df.columns if c.lower() == col.lower()]
            if matching_cols:
                df[col] = df[matching_cols[0]]
            elif col != 'Volume':
                return None

    # Convert all columns to float and ensure they're 1D
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].values.flatten(), errors='coerce')

    # Drop any NaN rows
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    return df


def fetch_forex_data(pair, interval, period):
    """Fetch forex data from Yahoo Finance"""
    try:
        ticker = forex_pairs[pair]

        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True
        )

        if data.empty:
            st.warning(f"No data received for {pair}. Trying alternative method...")
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False
            )

            if data.empty:
                return None

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

        data = clean_dataframe(data)

        if data is None or data.empty:
            st.error("Failed to process data properly")
            return None

        return data

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def calculate_pivot_points(df):
    """Calculate pivot points using multiple methods"""
    if len(df) < 2:
        return df

    try:
        # Get previous period's data
        prev_high = float(df['High'].iloc[-2])
        prev_low = float(df['Low'].iloc[-2])
        prev_close = float(df['Close'].iloc[-2])

        # Classic Pivot Points
        pivot = (prev_high + prev_low + prev_close) / 3

        r1 = 2 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)

        s1 = 2 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)

        # Fibonacci Pivot Points
        fib_range = prev_high - prev_low
        fib_pivot = (prev_high + prev_low + prev_close) / 3

        fib_r1 = fib_pivot + 0.382 * fib_range
        fib_r2 = fib_pivot + 0.618 * fib_range
        fib_r3 = fib_pivot + 1.000 * fib_range

        fib_s1 = fib_pivot - 0.382 * fib_range
        fib_s2 = fib_pivot - 0.618 * fib_range
        fib_s3 = fib_pivot - 1.000 * fib_range

        # Woodie's Pivot Points
        woodie_pivot = (prev_high + prev_low + 2 * prev_close) / 4

        woodie_r1 = 2 * woodie_pivot - prev_low
        woodie_r2 = woodie_pivot + (prev_high - prev_low)

        woodie_s1 = 2 * woodie_pivot - prev_high
        woodie_s2 = woodie_pivot - (prev_high - prev_low)

        # Add to dataframe
        df['Pivot'] = pivot
        df['R1'] = r1
        df['R2'] = r2
        df['R3'] = r3
        df['S1'] = s1
        df['S2'] = s2
        df['S3'] = s3

        # Fibonacci levels
        df['Fib_Pivot'] = fib_pivot
        df['Fib_R1'] = fib_r1
        df['Fib_R2'] = fib_r2
        df['Fib_R3'] = fib_r3
        df['Fib_S1'] = fib_s1
        df['Fib_S2'] = fib_s2
        df['Fib_S3'] = fib_s3

        # Woodie's levels
        df['Woodie_Pivot'] = woodie_pivot
        df['Woodie_R1'] = woodie_r1
        df['Woodie_R2'] = woodie_r2
        df['Woodie_S1'] = woodie_s1
        df['Woodie_S2'] = woodie_s2

        return df

    except Exception as e:
        st.warning(f"Could not calculate pivot points: {e}")
        return df


def calculate_indicators(df):
    """Calculate technical indicators"""
    try:
        close_prices = df['Close'].values.flatten()
        high_prices = df['High'].values.flatten()
        low_prices = df['Low'].values.flatten()

        close_series = pd.Series(close_prices, index=df.index)
        high_series = pd.Series(high_prices, index=df.index)
        low_series = pd.Series(low_prices, index=df.index)

        # EMAs
        df['EMA_Fast'] = ta.trend.ema_indicator(close_series, window=ema_fast)
        df['EMA_Slow'] = ta.trend.ema_indicator(close_series, window=ema_slow)

        # RSI
        df['RSI'] = ta.momentum.rsi(close_series, window=rsi_period)

        # MACD
        macd = ta.trend.MACD(close_series)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        # ADX
        df['ADX'] = ta.trend.adx(high_series, low_series, close_series, window=14)
        df['DI+'] = ta.trend.adx_pos(high_series, low_series, close_series, window=14)
        df['DI-'] = ta.trend.adx_neg(high_series, low_series, close_series, window=14)

        # ATR for volatility
        df['ATR'] = ta.volatility.average_true_range(high_series, low_series, close_series, window=14)

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close_series)
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['BB_Lower'] = bollinger.bollinger_lband()

        # Support and Resistance from recent price action
        df['Recent_High_20'] = high_series.rolling(window=20).max()
        df['Recent_Low_20'] = low_series.rolling(window=20).min()

        # Average True Range Percentage
        df['ATR_Pct'] = (df['ATR'] / close_series) * 100

        return df

    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df


def calculate_entry_tp_sl(df, signal):
    """Calculate entry, take profit, and stop loss levels"""
    try:
        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        atr = float(latest['ATR'])
        atr_pct = float(latest['ATR_Pct'])

        entry_price = current_price

        # Parse risk:reward ratio
        rr_ratio = float(reward_risk_ratio.split(':')[1]) / float(reward_risk_ratio.split(':')[0])

        # Calculate SL and TP based on ATR
        sl_distance = atr * atr_multiplier_sl
        tp_distance = atr * atr_multiplier_tp

        # Initialize levels
        entry_levels = []
        stop_loss_levels = []
        take_profit_levels = []

        if "BUY" in signal.upper():
            # Long position setup

            # Entry options
            entry_levels = [
                {
                    "type": "Market Entry",
                    "price": round(current_price, 5),
                    "description": "Immediate entry at current price"
                },
                {
                    "type": "Pivot Support Entry",
                    "price": round(float(latest.get('S1', current_price * 0.999)), 5),
                    "description": "Entry on pullback to S1 pivot support"
                },
                {
                    "type": "Fibonacci Support Entry",
                    "price": round(float(latest.get('Fib_S1', current_price * 0.998)), 5),
                    "description": "Entry on pullback to Fib S1 support"
                }
            ]

            # Stop Loss options
            stop_loss_levels = [
                {
                    "type": "ATR Stop Loss",
                    "price": round(current_price - sl_distance, 5),
                    "description": f"{atr_multiplier_sl}x ATR ({atr_pct * atr_multiplier_sl:.2f}% risk)",
                    "risk_pct": round(atr_pct * atr_multiplier_sl, 2)
                },
                {
                    "type": "Pivot S1 Stop",
                    "price": round(float(latest.get('S1', current_price * 0.999)), 5),
                    "description": "Stop below S1 pivot",
                    "risk_pct": round(
                        abs(current_price - float(latest.get('S1', current_price * 0.999))) / current_price * 100, 2)
                },
                {
                    "type": "Recent Low Stop",
                    "price": round(float(latest['Recent_Low_20']), 5),
                    "description": "Stop below 20-period low",
                    "risk_pct": round(abs(current_price - float(latest['Recent_Low_20'])) / current_price * 100, 2)
                }
            ]

            # Take Profit options
            take_profit_levels = [
                {
                    "type": f"TP1 (1:{rr_ratio} R:R)",
                    "price": round(current_price + (sl_distance * rr_ratio), 5),
                    "description": f"First target at {rr_ratio}:1 reward:risk",
                    "rr_ratio": f"1:{rr_ratio}"
                },
                {
                    "type": "Pivot R1 TP",
                    "price": round(float(latest.get('R1', current_price * 1.001)), 5),
                    "description": "Take profit at R1 resistance",
                    "rr_ratio": f"1:{round(abs(float(latest.get('R1', current_price * 1.001)) - current_price) / sl_distance, 1)}"
                },
                {
                    "type": "Pivot R2 TP",
                    "price": round(float(latest.get('R2', current_price * 1.002)), 5),
                    "description": "Take profit at R2 resistance",
                    "rr_ratio": f"1:{round(abs(float(latest.get('R2', current_price * 1.002)) - current_price) / sl_distance, 1)}"
                },
                {
                    "type": "Fibonacci R1 TP",
                    "price": round(float(latest.get('Fib_R1', current_price * 1.001)), 5),
                    "description": "Take profit at Fib R1",
                    "rr_ratio": f"1:{round(abs(float(latest.get('Fib_R1', current_price * 1.001)) - current_price) / sl_distance, 1)}"
                }
            ]

        elif "SELL" in signal.upper():
            # Short position setup

            # Entry options
            entry_levels = [
                {
                    "type": "Market Entry",
                    "price": round(current_price, 5),
                    "description": "Immediate entry at current price"
                },
                {
                    "type": "Pivot Resistance Entry",
                    "price": round(float(latest.get('R1', current_price * 1.001)), 5),
                    "description": "Entry on rally to R1 pivot resistance"
                },
                {
                    "type": "Fibonacci Resistance Entry",
                    "price": round(float(latest.get('Fib_R1', current_price * 1.002)), 5),
                    "description": "Entry on rally to Fib R1 resistance"
                }
            ]

            # Stop Loss options
            stop_loss_levels = [
                {
                    "type": "ATR Stop Loss",
                    "price": round(current_price + sl_distance, 5),
                    "description": f"{atr_multiplier_sl}x ATR ({atr_pct * atr_multiplier_sl:.2f}% risk)",
                    "risk_pct": round(atr_pct * atr_multiplier_sl, 2)
                },
                {
                    "type": "Pivot R1 Stop",
                    "price": round(float(latest.get('R1', current_price * 1.001)), 5),
                    "description": "Stop above R1 pivot",
                    "risk_pct": round(
                        abs(float(latest.get('R1', current_price * 1.001)) - current_price) / current_price * 100, 2)
                },
                {
                    "type": "Recent High Stop",
                    "price": round(float(latest['Recent_High_20']), 5),
                    "description": "Stop above 20-period high",
                    "risk_pct": round(abs(float(latest['Recent_High_20']) - current_price) / current_price * 100, 2)
                }
            ]

            # Take Profit options
            take_profit_levels = [
                {
                    "type": f"TP1 (1:{rr_ratio} R:R)",
                    "price": round(current_price - (sl_distance * rr_ratio), 5),
                    "description": f"First target at {rr_ratio}:1 reward:risk",
                    "rr_ratio": f"1:{rr_ratio}"
                },
                {
                    "type": "Pivot S1 TP",
                    "price": round(float(latest.get('S1', current_price * 0.999)), 5),
                    "description": "Take profit at S1 support",
                    "rr_ratio": f"1:{round(abs(current_price - float(latest.get('S1', current_price * 0.999))) / sl_distance, 1)}"
                },
                {
                    "type": "Pivot S2 TP",
                    "price": round(float(latest.get('S2', current_price * 0.998)), 5),
                    "description": "Take profit at S2 support",
                    "rr_ratio": f"1:{round(abs(current_price - float(latest.get('S2', current_price * 0.998))) / sl_distance, 1)}"
                },
                {
                    "type": "Fibonacci S1 TP",
                    "price": round(float(latest.get('Fib_S1', current_price * 0.999)), 5),
                    "description": "Take profit at Fib S1",
                    "rr_ratio": f"1:{round(abs(current_price - float(latest.get('Fib_S1', current_price * 0.999))) / sl_distance, 1)}"
                }
            ]

        return {
            "entry": entry_levels,
            "stop_loss": stop_loss_levels,
            "take_profit": take_profit_levels,
            "atr": atr,
            "atr_pct": atr_pct
        }

    except Exception as e:
        st.error(f"Error calculating entry/TP/SL: {e}")
        return None


def generate_signals(df):
    """Generate trading signals based on trend and pivot points"""
    if len(df) < 2:
        return df, "HOLD", [], "gray"

    try:
        latest = df.iloc[-1]
        signals = []
        score = 0
        signal_details = []

        # Trend Strength Analysis
        trend_bullish = latest['EMA_Fast'] > latest['EMA_Slow']
        trend_bearish = latest['EMA_Fast'] < latest['EMA_Slow']

        # MACD Signal
        macd_bullish = latest['MACD'] > latest['MACD_Signal']
        macd_bearish = latest['MACD'] < latest['MACD_Signal']

        # ADX Trend Strength
        strong_trend = latest['ADX'] > adx_threshold
        di_plus = latest['DI+']
        di_minus = latest['DI-']
        di_bullish = di_plus > di_minus

        # RSI Conditions
        rsi_value = latest['RSI']

        # Price vs Pivot Points
        price = float(latest['Close'])
        pivot = float(latest.get('Pivot', price))

        above_pivot = price > pivot
        below_pivot = price < pivot

        # Price relative to Bollinger Bands
        bb_position = (price - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])

        # Signal scoring system
        if trend_bullish:
            score += 2
            signals.append("✅ EMA Trend: Bullish")
            signal_details.append({"indicator": "EMA Trend", "signal": "Bullish", "weight": 2})
        else:
            score -= 2
            signals.append("❌ EMA Trend: Bearish")
            signal_details.append({"indicator": "EMA Trend", "signal": "Bearish", "weight": -2})

        if macd_bullish:
            score += 2
            signals.append("✅ MACD: Bullish")
            signal_details.append({"indicator": "MACD", "signal": "Bullish", "weight": 2})
        else:
            score -= 2
            signals.append("❌ MACD: Bearish")
            signal_details.append({"indicator": "MACD", "signal": "Bearish", "weight": -2})

        if strong_trend and di_bullish:
            score += 1
            signals.append("✅ ADX: Strong Bullish Trend")
            signal_details.append({"indicator": "ADX/DI", "signal": "Strong Bullish", "weight": 1})
        elif strong_trend and not di_bullish:
            score -= 1
            signals.append("❌ ADX: Strong Bearish Trend")
            signal_details.append({"indicator": "ADX/DI", "signal": "Strong Bearish", "weight": -1})
        else:
            signals.append("⚠️ ADX: Weak/No Trend")
            signal_details.append({"indicator": "ADX/DI", "signal": "Weak Trend", "weight": 0})

        if rsi_oversold < rsi_value < rsi_overbought:
            score += 1
            signals.append("✅ RSI: Neutral")
            signal_details.append({"indicator": "RSI", "signal": "Neutral", "weight": 1})
        elif rsi_value < rsi_oversold:
            score += 1
            signals.append("✅ RSI: Oversold (Potential Buy)")
            signal_details.append({"indicator": "RSI", "signal": "Oversold", "weight": 1})
        else:
            score -= 1
            signals.append("⚠️ RSI: Overbought (Caution)")
            signal_details.append({"indicator": "RSI", "signal": "Overbought", "weight": -1})

        # Bollinger Band position
        if bb_position < 0.2:
            score += 0.5
            signals.append("✅ BB: Near Lower Band (Support)")
        elif bb_position > 0.8:
            score -= 0.5
            signals.append("⚠️ BB: Near Upper Band (Resistance)")

        # Pivot point analysis
        if 'Pivot' in df.columns:
            if above_pivot:
                score += 1
                signals.append(f"✅ Price > Pivot ({pivot:.5f})")
            else:
                score -= 1
                signals.append(f"❌ Price < Pivot ({pivot:.5f})")

        # Determine final signal
        if score >= 5:
            final_signal = "STRONG BUY"
            signal_color = "#00C853"
        elif 3 <= score < 5:
            final_signal = "BUY"
            signal_color = "#69F0AE"
        elif -2 <= score < 3:
            final_signal = "HOLD/NEUTRAL"
            signal_color = "#9E9E9E"
        elif -4 <= score < -2:
            final_signal = "SELL"
            signal_color = "#FF6E40"
        else:
            final_signal = "STRONG SELL"
            signal_color = "#FF1744"

        return df, final_signal, signals, signal_color

    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return df, "ERROR", [f"Error: {str(e)}"], "gray"


def plot_chart_with_levels(df, pair, timeframe, trade_levels=None):
    """Create interactive chart with entry, TP, and SL levels"""
    try:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{pair} - {timeframe}", "RSI", "MACD")
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price",
                showlegend=False
            ),
            row=1, col=1
        )

        # Add EMAs
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_Fast'],
                name=f"EMA {ema_fast}",
                line=dict(color='blue', width=1),
                showlegend=True
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_Slow'],
                name=f"EMA {ema_slow}",
                line=dict(color='orange', width=1),
                showlegend=True
            ),
            row=1, col=1
        )

        # Add Pivot Points
        if 'Pivot' in df.columns:
            # Main pivot levels
            pivot_levels = {
                'R3': ('darkred', 'dash'),
                'R2': ('red', 'dash'),
                'R1': ('lightcoral', 'dash'),
                'Pivot': ('yellow', 'dash'),
                'S1': ('lightgreen', 'dash'),
                'S2': ('green', 'dash'),
                'S3': ('darkgreen', 'dash')
            }

            for level, (color, style) in pivot_levels.items():
                if level in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=[df.index[0], df.index[-1]],
                            y=[float(df[level].iloc[-1]), float(df[level].iloc[-1])],
                            name=f"{level}",
                            line=dict(color=color, width=1.5, dash=style),
                            showlegend=True
                        ),
                        row=1, col=1
                    )

        # Add Trade Levels if available
        if trade_levels:
            current_price = float(df['Close'].iloc[-1])

            # Entry levels
            for entry in trade_levels.get('entry', [])[:2]:  # Show top 2 entries
                fig.add_hline(
                    y=entry['price'],
                    line_dash="dot",
                    line_color="white",
                    opacity=0.7,
                    row=1, col=1
                )
                fig.add_annotation(
                    x=df.index[-1],
                    y=entry['price'],
                    text=f"Entry: {entry['price']:.5f}",
                    showarrow=True,
                    arrowhead=1,
                    ax=40,
                    ay=0,
                    bgcolor="white",
                    font=dict(size=10)
                )

            # Stop Loss levels
            for sl in trade_levels.get('stop_loss', [])[:1]:  # Show primary SL
                fig.add_hline(
                    y=sl['price'],
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    opacity=0.8,
                    row=1, col=1
                )
                fig.add_annotation(
                    x=df.index[-1],
                    y=sl['price'],
                    text=f"SL: {sl['price']:.5f}",
                    showarrow=True,
                    arrowhead=1,
                    ax=40,
                    ay=20,
                    bgcolor="red",
                    font=dict(size=10, color="white")
                )

            # Take Profit levels
            tp_colors = ['lime', 'green', 'darkgreen']
            for i, tp in enumerate(trade_levels.get('take_profit', [])[:3]):  # Show top 3 TPs
                fig.add_hline(
                    y=tp['price'],
                    line_dash="dash",
                    line_color=tp_colors[i],
                    line_width=2,
                    opacity=0.8,
                    row=1, col=1
                )
                fig.add_annotation(
                    x=df.index[-1],
                    y=tp['price'],
                    text=f"TP{i + 1}: {tp['price']:.5f}",
                    showarrow=True,
                    arrowhead=1,
                    ax=40,
                    ay=-20,
                    bgcolor="green",
                    font=dict(size=10, color="white")
                )

        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name="RSI",
                    line=dict(color='purple', width=1.5),
                    showlegend=False
                ),
                row=2, col=1
            )

            fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)

        # MACD
        if 'MACD' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    name="MACD",
                    line=dict(color='blue', width=1.5),
                    showlegend=False
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD_Signal'],
                    name="Signal",
                    line=dict(color='red', width=1),
                    showlegend=False
                ),
                row=3, col=1
            )

            if 'MACD_Hist' in df.columns:
                macd_hist = df['MACD_Hist'].values.flatten()
                colors = ['green' if val >= 0 else 'red' for val in macd_hist]
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=macd_hist,
                        name="Histogram",
                        marker_color=colors,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=3, col=1
                )

        # Update layout
        fig.update_layout(
            height=900,
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            )
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=3, col=1)

        return fig

    except Exception as e:
        st.error(f"Error creating chart: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig


def display_trade_plan(trade_levels, signal):
    """Display detailed trade plan with entry, TP, and SL levels"""
    if not trade_levels:
        return

    st.markdown("---")
    st.header("📋 Trading Plan")

    # Trade Summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"{float(trade_levels['entry'][0]['price']):.5f}"
        )

    with col2:
        st.metric(
            label="📊 ATR (Volatility)",
            value=f"{trade_levels['atr']:.5f}",
            delta=f"{trade_levels['atr_pct']:.2f}%"
        )

    with col3:
        st.metric(
            label="🎯 Signal",
            value=signal
        )

    # Entry Points
    st.subheader("🚀 Entry Points")
    entry_df = pd.DataFrame(trade_levels['entry'])
    entry_df.columns = ['Entry Type', 'Price', 'Description']
    st.dataframe(entry_df.style.format({'Price': '{:.5f}'}), use_container_width=True)

    # Stop Loss and Take Profit
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛑 Stop Loss Levels")
        sl_df = pd.DataFrame(trade_levels['stop_loss'])
        sl_df.columns = ['SL Type', 'Price', 'Description', 'Risk %']
        st.dataframe(sl_df.style.format({'Price': '{:.5f}', 'Risk %': '{:.2f}%'}), use_container_width=True)

        # Calculate and display position sizing
        st.subheader("💎 Position Sizing")
        account_balance = st.number_input("Account Balance ($)", min_value=100, value=10000, step=1000)

        if account_balance > 0 and len(trade_levels['stop_loss']) > 0:
            risk_amount = account_balance * (risk_percent / 100)
            sl_risk_pct = trade_levels['stop_loss'][0]['risk_pct'] / 100
            position_size = risk_amount / sl_risk_pct if sl_risk_pct > 0 else 0

            col1_1, col1_2, col1_3 = st.columns(3)
            with col1_1:
                st.metric("Risk Amount ($)", f"${risk_amount:.2f}")
            with col1_2:
                st.metric("Position Size ($)", f"${position_size:.2f}")
            with col1_3:
                leverage = position_size / account_balance
                st.metric("Effective Leverage", f"{leverage:.2f}x")

    with col2:
        st.subheader("🎯 Take Profit Levels")
        tp_df = pd.DataFrame(trade_levels['take_profit'])
        tp_df.columns = ['TP Type', 'Price', 'Description', 'R:R Ratio']
        st.dataframe(tp_df.style.format({'Price': '{:.5f}'}), use_container_width=True)

        # Risk/Reward Analysis
        if len(trade_levels['stop_loss']) > 0 and len(trade_levels['take_profit']) > 0:
            st.subheader("📊 Risk/Reward Analysis")

            entry_price = trade_levels['entry'][0]['price']
            sl_price = trade_levels['stop_loss'][0]['price']

            for tp in trade_levels['take_profit']:
                tp_price = tp['price']
                risk = abs(entry_price - sl_price)
                reward = abs(tp_price - entry_price)
                rr = reward / risk if risk > 0 else 0

                st.progress(min(rr / 3, 1.0), text=f"{tp['type']}: R:R = 1:{rr:.1f}")

    # Trade Management Tips
    with st.expander("📚 Trade Management Tips"):
        st.markdown("""
        ### Best Practices:

        1. **Entry**: Wait for confirmation before entering (candle close above/below key levels)
        2. **Scaling In**: Consider entering 50% at market, 50% at limit orders near pivot levels
        3. **Stop Loss**: Always use stop loss! The ATR-based SL provides good protection
        4. **Trailing Stop**: Once in profit, consider trailing stop to lock in gains
        5. **Partial Profits**: Take 50% profit at TP1, move SL to breakeven, let rest run to TP2/TP3

        ### Risk Management:
        - Never risk more than {0}% per trade
        - Maximum correlation exposure should be limited
        - Avoid trading during major news events
        - Keep a trading journal to track performance
        """.format(risk_percent))


# Main application
try:
    # Fetch data
    with st.spinner('Fetching forex data...'):
        df = fetch_forex_data(selected_pair, interval_map[timeframe], data_period_map[timeframe])

    if df is not None and not df.empty:
        st.success(f"✅ Data loaded: {len(df)} candles for {selected_pair} ({timeframe})")

        # Calculate indicators and pivot points
        df = calculate_indicators(df)
        df = calculate_pivot_points(df)
        df, signal, signals, signal_color = generate_signals(df)

        # Calculate trade levels
        trade_levels = calculate_entry_tp_sl(df, signal)

        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        current_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100

        with col1:
            st.metric(
                label=f"💹 {selected_pair}",
                value=f"{current_price:.5f}",
                delta=f"{price_change:.5f} ({price_change_pct:.2f}%)"
            )

        with col2:
            st.metric(
                label="📊 Signal",
                value=signal,
                delta=None
            )

        with col3:
            rsi_val = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else 0
            st.metric(
                label="📈 RSI",
                value=f"{rsi_val:.1f}",
                delta="Overbought" if rsi_val > rsi_overbought else (
                    "Oversold" if rsi_val < rsi_oversold else "Neutral")
            )

        with col4:
            adx_val = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns else 0
            st.metric(
                label="💪 ADX",
                value=f"{adx_val:.1f}",
                delta="Strong" if adx_val > adx_threshold else "Weak"
            )

        with col5:
            atr_val = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else 0
            st.metric(
                label="📏 ATR",
                value=f"{atr_val:.5f}",
                delta=f"{float(df['ATR_Pct'].iloc[-1]):.2f}%" if 'ATR_Pct' in df.columns else None
            )

        # Signal box
        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {signal_color}; opacity: 0.85;'>
            <h2 style='text-align: center; color: white; margin: 0;'>
                {signal} - {selected_pair}
            </h2>
        </div>
        """, unsafe_allow_html=True)

        # Display chart with trade levels
        fig = plot_chart_with_levels(df, selected_pair, timeframe, trade_levels)
        st.plotly_chart(fig, use_container_width=True)

        # Signal Analysis
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Signal Analysis")
            for s in signals:
                st.write(s)

        with col2:
            st.subheader("🎯 Pivot Points Summary")
            if 'Pivot' in df.columns:
                pivot_levels = {
                    "🟥 Resistance 3": float(df['R3'].iloc[-1]),
                    "🟧 Resistance 2": float(df['R2'].iloc[-1]),
                    "🟨 Resistance 1": float(df['R1'].iloc[-1]),
                    "⬜ Pivot Point": float(df['Pivot'].iloc[-1]),
                    "🟩 Support 1": float(df['S1'].iloc[-1]),
                    "🟢 Support 2": float(df['S2'].iloc[-1]),
                    "🟦 Support 3": float(df['S3'].iloc[-1])
                }

                for level, value in pivot_levels.items():
                    distance = abs(current_price - value)
                    distance_pct = (distance / current_price) * 100
                    st.markdown(f"{level}: **{value:.5f}** ({distance_pct:.2f}% away)")

        # Display complete trade plan
        display_trade_plan(trade_levels, signal)

        # Recent price action table
        st.subheader("📈 Recent Price Action")
        display_cols = ['Open', 'High', 'Low', 'Close', 'RSI', 'ADX', 'ATR']
        available_cols = [col for col in display_cols if col in df.columns]
        if available_cols:
            recent_df = df.tail(20)[available_cols]
            st.dataframe(
                recent_df.style.background_gradient(cmap='RdYlGn', subset=['RSI'] if 'RSI' in available_cols else None),
                use_container_width=True)

        # Educational section
        with st.expander("📚 How to Use This Trading System"):
            st.markdown(f"""
            ### Trading Strategy Overview

            This system combines **trend following** with **pivot point analysis** for {timeframe} timeframe trading.

            #### Signal Components:
            1. **EMA Crossover** ({ema_fast}/{ema_slow}): Defines the trend direction
            2. **MACD**: Confirms momentum
            3. **RSI** ({rsi_period}): Identifies overbought/oversold conditions
            4. **ADX** (>{adx_threshold}): Confirms trend strength
            5. **Pivot Points**: Key support/resistance levels

            #### Trade Execution:
            - **Entry**: Market price or limit orders near pivot levels
            - **Stop Loss**: ATR-based ({atr_multiplier_sl}x ATR) or below recent swing points
            - **Take Profit**: Multiple targets using pivot levels and R:R ratio ({reward_risk_ratio})

            #### Risk Management:
            - Risk {risk_percent}% per trade
            - Use proper position sizing
            - Trail stops as trade moves in your favor
            - Take partial profits at key levels

            #### Best Trading Times:
            - London Session: 08:00-17:00 GMT
            - New York Session: 13:00-22:00 GMT
            - Avoid major news events
            """)

        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Refresh Data & Signals", use_container_width=True):
                st.rerun()

    else:
        st.error("❌ No data available for this pair. Please try another pair or timeframe.")
        st.info("💡 Tip: Most reliable data is available for major pairs on daily timeframe.")

        # Show available pairs
        with st.expander("📋 Troubleshooting"):
            st.markdown("""
            Try these steps:
            1. Select a major pair (EUR/USD, GBP/USD, USD/JPY)
            2. Use daily (1d) timeframe first
            3. Check your internet connection
            4. Wait a few minutes and try again (API rate limits)
            """)

except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")

    # Debug information
    with st.expander("🔍 Debug Information"):
        st.write("Error Type:", type(e).__name__)
        st.write("Error Details:", str(e))
        st.write("Selected Pair:", selected_pair)
        st.write("Timeframe:", timeframe)
        st.write("Try refreshing or selecting different parameters.")