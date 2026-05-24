"""
╔══════════════════════════════════════════════════════════════╗
║  📈 Forex & Commodities Trend Following Signal Alert App    ║
║  Strategy: 50 EMA / 200 EMA + RSI + MACD + ADX Confirmation ║
║  FILTER: Strong Buy & Strong Sell Signals Only              ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, List
import io

# ── Try importing autorefresh (gracefully degrade if missing) ──
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forex & Commodities Signal Alerts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
#  CUSTOM CSS - Enhanced with distinct colors for each signal
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* STRONG BUY - Vibrant Green */
    .card-strong-buy {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white; padding: 18px; border-radius: 14px;
        text-align: center; margin-bottom: 10px;
        box-shadow: 0 4px 20px rgba(0,184,148,0.5);
        border: 2px solid #00e6a0;
    }
    /* STRONG SELL - Intense Red */
    .card-strong-sell {
        background: linear-gradient(135deg, #d63031 0%, #e17055 100%);
        color: white; padding: 18px; border-radius: 14px;
        text-align: center; margin-bottom: 10px;
        box-shadow: 0 4px 20px rgba(214,48,49,0.5);
        border: 2px solid #ff4757;
    }
    /* NEUTRAL */
    .card-neutral {
        background: linear-gradient(135deg, #636e72 0%, #b2bec3 100%);
        color: white; padding: 18px; border-radius: 14px;
        text-align: center; margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(99,110,114,0.25);
    }
    .pair-name   { font-size: 1.0rem; font-weight: 600; opacity: 0.9; }
    .sig-text    { font-size: 1.7rem; font-weight: 800; letter-spacing: 1px; margin: 4px 0; }
    .price-text  { font-size: 1.0rem; font-family: monospace; }
    .score-text  { font-size: 0.78rem; opacity: 0.85; margin-top: 4px; }

    /* Condition list */
    .cond-met   { color: #00b894; font-weight: 600; }
    .cond-unmet { color: #d63031; font-weight: 600; }

    /* Alert banners with distinct colors */
    .alert-strong-buy  { background:#d4f7e8; border-left:5px solid #00b894;
                  padding:12px 18px; border-radius:8px; margin:8px 0; }
    .alert-strong-sell { background:#fdf2f2; border-left:5px solid #d63031;
                  padding:12px 18px; border-radius:8px; margin:8px 0; }

    /* Commodity indicators */
    .commodity-tag {
        display: inline-block;
        background: #fdcb6e;
        color: #2d3436;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .forex-tag {
        display: inline-block;
        background: #74b9ff;
        color: #2d3436;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Multi-TF Table Styling */
    .mtf-strong-buy {
        background: #00b894;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        text-align: center;
        display: inline-block;
        min-width: 90px;
    }
    .mtf-strong-sell {
        background: #d63031;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        text-align: center;
        display: inline-block;
        min-width: 90px;
    }
    .mtf-buy {
        background: #6c5ce7;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        text-align: center;
        display: inline-block;
        min-width: 90px;
        opacity: 0.7;
    }
    .mtf-sell {
        background: #f39c12;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        text-align: center;
        display: inline-block;
        min-width: 90px;
        opacity: 0.7;
    }
    .mtf-neutral {
        background: #636e72;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.8rem;
        text-align: center;
        display: inline-block;
        min-width: 90px;
        opacity: 0.5;
    }
    .mtf-consensus-strong {
        background: linear-gradient(135deg, #00b894, #d63031);
        color: white;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 0.9rem;
        text-align: center;
        letter-spacing: 1px;
    }

    /* Backtesting styles */
    .bt-profit {
        color: #00b894;
        font-weight: 700;
    }
    .bt-loss {
        color: #d63031;
        font-weight: 700;
    }
    .bt-metric-card {
        background: #1a1c23;
        border: 1px solid #2d3143;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 4px 0;
    }
    .bt-metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .bt-metric-label {
        font-size: 0.75rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sidebar polish */
    [data-testid="stSidebar"] { background: #0f1117; }
    [data-testid="stSidebar"] h2 { color: #fdcb6e; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1c23;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d3143 !important;
        border-bottom: 3px solid #fdcb6e !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  CONSTANTS - Added Commodities
# ──────────────────────────────────────────────────────────────
PAIRS = {
    # Forex Pairs
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "USDCAD=X",
    "EUR/GBP": "EURGBP=X",
    "USD/ZAR": "USDZAR=X",
    # Precious Metals / Commodities
    "🥇 Gold": "GC=F",
    "🥈 Silver": "SI=F",
    "🪙 Platinum": "PL=F",
}

# Separate commodity list for styling
COMMODITIES = {"🥇 Gold", "🥈 Silver", "🪙 Platinum"}

# (yfinance interval, yfinance period, resample_rule or None)
TIMEFRAMES = {
    "1 Hour":  ("60m", "59d",  "1h"),
    "4 Hours": ("60m", "59d",  "4h"),
    "Daily":   ("1d",  "2y",   None),
    "Weekly":  ("1d",  "2y",   "1W")
}

# Sessions (UTC hours)
SESSIONS = {
    "London":        (7,  16),
    "New York":      (12, 21),
    "London+NY":     (12, 16),
    "Asia":          (23, 8),
}

ALERT_HISTORY_KEY = "alert_history"
MULTI_TF_CACHE_KEY = "multi_tf_results"

# ──────────────────────────────────────────────────────────────
#  TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    e_fast = ema(series, fast)
    e_slow = ema(series, slow)
    macd_line = e_fast - e_slow
    sig_line = ema(macd_line, sig)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist

def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high, low, close = df["High"], df["Low"], df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0.0)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx_val, plus_di, minus_di

# ──────────────────────────────────────────────────────────────
#  DATA FETCHING & PREPARATION
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_prepare(ticker: str, interval: str, period: str, resample: Optional[str]) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_cols):
        return None

    df = df[required_cols].copy()
    df.dropna(inplace=True)

    if resample:
        if resample == "1W":
            df = df.resample("W").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna()
        else:
            df = df.resample(resample).agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna()

    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACDSig"], df["MACDHist"] = macd(df["Close"])
    df["ADX"], df["PlusDI"], df["MinusDI"] = adx(df)

    return df.dropna()

# ──────────────────────────────────────────────────────────────
#  SIGNAL ENGINE
# ──────────────────────────────────────────────────────────────
def evaluate_signal(df: pd.DataFrame, min_conditions: int = 4) -> Tuple[str, int, int, Dict[str, bool], str]:
    """
    Returns (signal_label, score, max_score, conditions_dict, direction)
    direction: 'STRONG_BUY' | 'BUY' | 'STRONG_SELL' | 'SELL' | 'NEUTRAL'
    """
    if len(df) < 10:
        return "⏳ NEUTRAL", 0, 6, {}, "NEUTRAL"

    r = df.iloc[-1]
    close = float(r["Close"])
    e50 = float(r["EMA50"])
    e200 = float(r["EMA200"])
    rsi_val = float(r["RSI"])
    macd_v = float(r["MACD"])
    macd_s = float(r["MACDSig"])
    adx_v = float(r["ADX"])

    buy_conds = {
        "Price above 200 EMA": close > e200,
        "50 EMA above 200 EMA (Golden X)": e50 > e200,
        "Price at/above 50 EMA": close >= e50 * 0.9985,
        "RSI 45–70 (bullish momentum)": 45 <= rsi_val <= 70,
        "MACD above Signal line": macd_v > macd_s,
        "Strong trend (ADX > 25)": adx_v > 25,
    }
    sell_conds = {
        "Price below 200 EMA": close < e200,
        "50 EMA below 200 EMA (Death X)": e50 < e200,
        "Price at/below 50 EMA": close <= e50 * 1.0015,
        "RSI 30–55 (bearish momentum)": 30 <= rsi_val <= 55,
        "MACD below Signal line": macd_v < macd_s,
        "Strong trend (ADX > 25)": adx_v > 25,
    }

    bs = sum(buy_conds.values())
    ss = sum(sell_conds.values())

    if bs >= min_conditions and bs >= ss:
        if bs >= 5:
            return "🚀 STRONG BUY", bs, 6, buy_conds, "STRONG_BUY"
        else:
            return "📈 BUY", bs, 6, buy_conds, "BUY"
    elif ss >= min_conditions and ss > bs:
        if ss >= 5:
            return "🔻 STRONG SELL", ss, 6, sell_conds, "STRONG_SELL"
        else:
            return "📉 SELL", ss, 6, sell_conds, "SELL"
    else:
        return "⏳ NEUTRAL", max(bs, ss), 6, {}, "NEUTRAL"

def evaluate_signal_row(row: pd.Series, min_conditions: int = 4) -> Tuple[str, int, int, str]:
    """
    Evaluates signal from a single row of indicator data.
    Returns (signal_label, score, max_score, direction)
    """
    close = float(row["Close"])
    e50 = float(row["EMA50"])
    e200 = float(row["EMA200"])
    rsi_val = float(row["RSI"])
    macd_v = float(row["MACD"])
    macd_s = float(row["MACDSig"])
    adx_v = float(row["ADX"])

    buy_conds = {
        "Price above 200 EMA": close > e200,
        "50 EMA above 200 EMA (Golden X)": e50 > e200,
        "Price at/above 50 EMA": close >= e50 * 0.9985,
        "RSI 45–70 (bullish momentum)": 45 <= rsi_val <= 70,
        "MACD above Signal line": macd_v > macd_s,
        "Strong trend (ADX > 25)": adx_v > 25,
    }
    sell_conds = {
        "Price below 200 EMA": close < e200,
        "50 EMA below 200 EMA (Death X)": e50 < e200,
        "Price at/below 50 EMA": close <= e50 * 1.0015,
        "RSI 30–55 (bearish momentum)": 30 <= rsi_val <= 55,
        "MACD below Signal line": macd_v < macd_s,
        "Strong trend (ADX > 25)": adx_v > 25,
    }

    bs = sum(buy_conds.values())
    ss = sum(sell_conds.values())

    if bs >= min_conditions and bs >= ss:
        if bs >= 5:
            return "🚀 STRONG BUY", bs, 6, "STRONG_BUY"
        else:
            return "📈 BUY", bs, 6, "BUY"
    elif ss >= min_conditions and ss > bs:
        if ss >= 5:
            return "🔻 STRONG SELL", ss, 6, "STRONG_SELL"
        else:
            return "📉 SELL", ss, 6, "SELL"
    else:
        return "⏳ NEUTRAL", max(bs, ss), 6, "NEUTRAL"

# ──────────────────────────────────────────────────────────────
#  MULTI-TIMEFRAME SCANNER
# ──────────────────────────────────────────────────────────────
def scan_all_timeframes(pairs: list, min_conditions: int = 4) -> pd.DataFrame:
    """
    Scans all selected pairs across all timeframes.
    Returns a DataFrame with multi-TF results.
    """
    results = []
    tf_names = list(TIMEFRAMES.keys())

    progress_bar = st.progress(0, text="Scanning all timeframes...")
    total_ops = len(pairs) * len(tf_names)
    completed = 0

    for pair in pairs:
        row = {"Market": pair}
        has_strong = False

        for tf in tf_names:
            interval, period, resample = TIMEFRAMES[tf]
            ticker = PAIRS[pair]
            df = fetch_and_prepare(ticker, interval, period, resample)

            if df is None or df.empty:
                row[tf] = "❌ Error"
                row[f"{tf}_direction"] = "ERROR"
                row[f"{tf}_score"] = 0
                row[f"{tf}_price"] = 0
            else:
                signal, score, max_score, conds, direction = evaluate_signal(df, min_conditions)
                last = df.iloc[-1]
                row[tf] = signal
                row[f"{tf}_direction"] = direction
                row[f"{tf}_score"] = score
                row[f"{tf}_price"] = float(last["Close"])

                if direction in ["STRONG_BUY", "STRONG_SELL"]:
                    has_strong = True

            completed += 1
            progress_bar.progress(completed / total_ops,
                                  text=f"Scanning {pair} — {tf}")

        row["Has_Strong"] = has_strong
        results.append(row)

    progress_bar.empty()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Has_Strong", ascending=False)
    return df_results

# ──────────────────────────────────────────────────────────────
#  BACKTESTING ENGINE
# ──────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, min_conditions: int = 4,
                 initial_capital: float = 10000,
                 position_size_pct: float = 2.0,
                 use_strong_only: bool = True) -> Dict[str, Any]:
    """
    Runs backtest on prepared DataFrame with indicator columns.

    Returns dict with:
    - trades: List of trade dictionaries
    - equity_curve: pd.Series of equity over time
    - metrics: Dict of performance metrics
    - df: DataFrame with signals marked
    """
    df = df.copy()

    # Generate signals for each row
    signals = []
    directions = []
    scores = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        if pd.isna(row["EMA50"]) or pd.isna(row["EMA200"]) or pd.isna(row["RSI"]):
            signals.append("⏳ NEUTRAL")
            directions.append("NEUTRAL")
            scores.append(0)
        else:
            signal, score, max_score, direction = evaluate_signal_row(row, min_conditions)
            signals.append(signal)
            directions.append(direction)
            scores.append(score)

    df["Signal"] = signals
    df["Direction"] = directions
    df["Signal_Score"] = scores

    # Track trades
    trades = []
    in_position = False
    position_type = None  # 'long' or 'short'
    entry_price = 0
    entry_date = None
    entry_idx = 0
    capital = initial_capital
    equity_curve = pd.Series(index=df.index, dtype=float)
    equity_curve.iloc[0] = initial_capital

    for i in range(1, len(df)):
        current_direction = df["Direction"].iloc[i]
        prev_direction = df["Direction"].iloc[i-1]
        current_price = df["Close"].iloc[i]

        # Determine if we should trade based on filter
        if use_strong_only:
            valid_buy = current_direction == "STRONG_BUY"
            valid_sell = current_direction == "STRONG_SELL"
            exit_buy = current_direction != "STRONG_BUY"
            exit_sell = current_direction != "STRONG_SELL"
        else:
            valid_buy = current_direction in ["STRONG_BUY", "BUY"]
            valid_sell = current_direction in ["STRONG_SELL", "SELL"]
            exit_buy = current_direction not in ["STRONG_BUY", "BUY"]
            exit_sell = current_direction not in ["STRONG_SELL", "SELL"]

        # Exit logic
        if in_position:
            exit_signal = False
            exit_reason = ""

            if position_type == 'long' and exit_buy:
                exit_signal = True
                exit_reason = "Signal reversed from Buy"
            elif position_type == 'short' and exit_sell:
                exit_signal = True
                exit_reason = "Signal reversed from Sell"

            # Stop loss / Take profit (optional)
            if in_position:
                pnl_pct = ((current_price - entry_price) / entry_price * 100)
                if position_type == 'short':
                    pnl_pct = -pnl_pct

                # 5% stop loss
                if pnl_pct <= -5:
                    exit_signal = True
                    exit_reason = "Stop Loss (-5%)"
                # 15% take profit
                elif pnl_pct >= 15:
                    exit_signal = True
                    exit_reason = "Take Profit (+15%)"

            if exit_signal:
                # Calculate P&L
                if position_type == 'long':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100

                position_value = capital * (position_size_pct / 100)
                pnl_amount = position_value * (pnl_pct / 100)
                capital += pnl_amount

                trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": df.index[i],
                    "Type": position_type.upper(),
                    "Entry Price": round(entry_price, 5),
                    "Exit Price": round(current_price, 5),
                    "P&L %": round(pnl_pct, 2),
                    "P&L $": round(pnl_amount, 2),
                    "Exit Reason": exit_reason,
                    "Bars Held": i - entry_idx,
                })

                in_position = False
                position_type = None

        # Entry logic
        if not in_position:
            if valid_buy and prev_direction != "STRONG_BUY":
                in_position = True
                position_type = 'long'
                entry_price = current_price
                entry_date = df.index[i]
                entry_idx = i
            elif valid_sell and prev_direction != "STRONG_SELL":
                in_position = True
                position_type = 'short'
                entry_price = current_price
                entry_date = df.index[i]
                entry_idx = i

        # Update equity curve
        if in_position:
            if position_type == 'long':
                unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                unrealized_pnl_pct = (entry_price - current_price) / entry_price * 100
            position_value = capital * (position_size_pct / 100)
            unrealized_pnl = position_value * (unrealized_pnl_pct / 100)
            equity_curve.iloc[i] = capital + unrealized_pnl
        else:
            equity_curve.iloc[i] = capital

    # Close any open position at end
    if in_position:
        final_price = df["Close"].iloc[-1]
        if position_type == 'long':
            pnl_pct = (final_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - final_price) / entry_price * 100

        position_value = capital * (position_size_pct / 100)
        pnl_amount = position_value * (pnl_pct / 100)
        capital += pnl_amount

        trades.append({
            "Entry Date": entry_date,
            "Exit Date": df.index[-1],
            "Type": position_type.upper(),
            "Entry Price": round(entry_price, 5),
            "Exit Price": round(final_price, 5),
            "P&L %": round(pnl_pct, 2),
            "P&L $": round(pnl_amount, 2),
            "Exit Reason": "End of data",
            "Bars Held": len(df) - entry_idx,
        })
        equity_curve.iloc[-1] = capital

    # Forward fill equity curve
    equity_curve = equity_curve.ffill()

    # Calculate metrics
    total_return = ((capital - initial_capital) / initial_capital) * 100

    # Win rate
    winning_trades = [t for t in trades if t["P&L $"] > 0]
    losing_trades = [t for t in trades if t["P&L $"] <= 0]
    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0

    # Average win/loss
    avg_win = np.mean([t["P&L %"] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t["P&L %"] for t in losing_trades]) if losing_trades else 0

    # Profit factor
    total_wins = sum(t["P&L $"] for t in winning_trades)
    total_losses = abs(sum(t["P&L $"] for t in losing_trades))
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    # Max drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio (simplified)
    returns = equity_curve.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() > 0 else 0

    metrics = {
        "Initial Capital": initial_capital,
        "Final Capital": round(capital, 2),
        "Total Return %": round(total_return, 2),
        "Total Trades": len(trades),
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win Rate %": round(win_rate, 1),
        "Avg Win %": round(avg_win, 2),
        "Avg Loss %": round(avg_loss, 2),
        "Profit Factor": round(profit_factor, 2),
        "Max Drawdown %": round(max_drawdown, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
    }

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "metrics": metrics,
        "df": df,
        "drawdown": drawdown,
    }

# ──────────────────────────────────────────────────────────────
#  CHART
# ──────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, pair: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.21, 0.21],
        subplot_titles=[f"{pair} — Price with EMAs", "RSI (14)", "MACD (12 / 26 / 9)"],
    )

    # ── Candlestick ──────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#00b894",
        decreasing_line_color="#d63031",
        increasing_fillcolor="#00b894",
        decreasing_fillcolor="#d63031",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA50"],
        name="50 EMA", line=dict(color="#fdcb6e", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA200"],
        name="200 EMA", line=dict(color="#a29bfe", width=2.5, dash="dash"),
    ), row=1, col=1)

    # Shade area between EMAs
    fig.add_trace(go.Scatter(
        x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
        y=pd.concat([df["EMA50"], df["EMA200"][::-1]]),
        fill="toself",
        fillcolor="rgba(253,203,110,0.08)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False, name="EMA band",
    ), row=1, col=1)

    # ── RSI ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        name="RSI", line=dict(color="#74b9ff", width=1.8),
    ), row=2, col=1)

    for level, color in [(70, "rgba(214,48,49,0.6)"), (30, "rgba(0,184,148,0.6)"), (50, "rgba(255,255,255,0.25)")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, row=2, col=1)

    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(214,48,49,0.07)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,184,148,0.07)", line_width=0, row=2, col=1)

    # ── MACD ─────────────────────────────────────────────────
    hist_colors = ["#00b894" if v >= 0 else "#d63031" for v in df["MACDHist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACDHist"],
        name="Histogram", marker_color=hist_colors, opacity=0.7,
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        name="MACD", line=dict(color="#0984e3", width=1.8),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACDSig"],
        name="Signal", line=dict(color="#e17055", width=1.8),
    ), row=3, col=1)

    # ── Layout ───────────────────────────────────────────────
    fig.update_layout(
        height=700,
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=10, t=40, b=0),
        font=dict(family="monospace", size=11),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig

def build_backtest_chart(bt_results: Dict[str, Any]) -> go.Figure:
    """Builds comprehensive backtest visualization."""
    df = bt_results["df"]
    equity = bt_results["equity_curve"]
    drawdown = bt_results["drawdown"]
    trades = bt_results["trades"]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=[
            "Price with Signals",
            "Equity Curve",
            "Drawdown %",
            "RSI (14)"
        ],
    )

    # ── Price chart with signals ─────────────────────────────
    # Background
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close", line=dict(color="#a0a0a0", width=1),
        opacity=0.7,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA50"],
        name="50 EMA", line=dict(color="#fdcb6e", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA200"],
        name="200 EMA", line=dict(color="#a29bfe", width=1.5, dash="dash"),
    ), row=1, col=1)

    # Mark signals
    strong_buy_mask = df["Direction"] == "STRONG_BUY"
    strong_sell_mask = df["Direction"] == "STRONG_SELL"
    buy_mask = df["Direction"] == "BUY"
    sell_mask = df["Direction"] == "SELL"

    fig.add_trace(go.Scatter(
        x=df.index[strong_buy_mask], y=df["Close"][strong_buy_mask],
        mode='markers', name='Strong Buy',
        marker=dict(symbol='triangle-up', size=12, color='#00b894', line=dict(width=1, color='white')),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index[strong_sell_mask], y=df["Close"][strong_sell_mask],
        mode='markers', name='Strong Sell',
        marker=dict(symbol='triangle-down', size=12, color='#d63031', line=dict(width=1, color='white')),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index[buy_mask], y=df["Close"][buy_mask],
        mode='markers', name='Buy',
        marker=dict(symbol='triangle-up', size=8, color='#6c5ce7', opacity=0.6),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index[sell_mask], y=df["Close"][sell_mask],
        mode='markers', name='Sell',
        marker=dict(symbol='triangle-down', size=8, color='#f39c12', opacity=0.6),
    ), row=1, col=1)

    # Mark trades
    for trade in trades:
        entry_date = trade["Entry Date"]
        exit_date = trade["Exit Date"]
        if entry_date in df.index and exit_date in df.index:
            color = '#00b894' if trade["P&L $"] > 0 else '#d63031'
            fig.add_trace(go.Scatter(
                x=[entry_date, exit_date],
                y=[trade["Entry Price"], trade["Exit Price"]],
                mode='lines+markers',
                line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=6, color=color),
                showlegend=False,
                opacity=0.6,
            ), row=1, col=1)

    # ── Equity Curve ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name='Equity', fill='tozeroy',
        line=dict(color='#00cec9', width=2),
        fillcolor='rgba(0,206,201,0.1)',
    ), row=2, col=1)

    # Add initial capital line
    initial_cap = bt_results["metrics"]["Initial Capital"]
    fig.add_hline(y=initial_cap, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=2, col=1)

    # ── Drawdown ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        name='Drawdown', fill='tozeroy',
        line=dict(color='#d63031', width=1.5),
        fillcolor='rgba(214,48,49,0.15)',
    ), row=3, col=1)

    # ── RSI ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        name='RSI', line=dict(color='#74b9ff', width=1.5),
    ), row=4, col=1)

    for level, color in [(70, "rgba(214,48,49,0.6)"), (30, "rgba(0,184,148,0.6)"), (50, "rgba(255,255,255,0.25)")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, row=4, col=1)

    # ── Layout ───────────────────────────────────────────────
    fig.update_layout(
        height=900,
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=10, t=40, b=0),
        font=dict(family="monospace", size=10),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")

    return fig

# ──────────────────────────────────────────────────────────────
#  SOUND ALERT  (browser Web Audio API)
# ──────────────────────────────────────────────────────────────
def play_sound(direction: str):
    if "BUY" in direction:
        tones = [(600, 0.0, 0.25), (900, 0.3, 0.25)]
    else:
        tones = [(900, 0.0, 0.25), (600, 0.3, 0.25)]

    tone_js = "\n".join(
        f"""
        var o{i}=ctx.createOscillator(), g{i}=ctx.createGain();
        o{i}.connect(g{i}); g{i}.connect(ctx.destination);
        o{i}.frequency.value={freq}; o{i}.type='sine';
        g{i}.gain.setValueAtTime(0.35, ctx.currentTime+{start});
        g{i}.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime+{start}+{dur});
        o{i}.start(ctx.currentTime+{start}); o{i}.stop(ctx.currentTime+{start}+{dur}+0.05);
        """
        for i, (freq, start, dur) in enumerate(tones)
    )
    st.components.v1.html(f"""
    <script>
    (function(){{
        var ctx = new (window.AudioContext || window.webkitAudioContext)();
        {tone_js}
    }})();
    </script>""", height=0)

# ──────────────────────────────────────────────────────────────
#  RENDER MULTI-TF CELL
# ──────────────────────────────────────────────────────────────
def render_mtf_cell(direction: str, signal_text: str, score: int, price: float) -> str:
    """Returns HTML for a multi-timeframe cell based on signal strength."""
    if direction == "STRONG_BUY":
        css_class = "mtf-strong-buy"
        display = f"🚀 {score}/6"
    elif direction == "STRONG_SELL":
        css_class = "mtf-strong-sell"
        display = f"🔻 {score}/6"
    elif direction == "BUY":
        css_class = "mtf-buy"
        display = f"📈 {score}/6"
    elif direction == "SELL":
        css_class = "mtf-sell"
        display = f"📉 {score}/6"
    else:
        css_class = "mtf-neutral"
        display = f"⏳ {score}/6"

    return f'<span class="{css_class}">{display}</span>'

# ──────────────────────────────────────────────────────────────
#  VALIDATE UPLOADED DATA
# ──────────────────────────────────────────────────────────────
def validate_and_prepare_upload(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Validates uploaded CSV and prepares it for backtesting."""
    required_cols = ["Open", "High", "Low", "Close"]

    # Check if we have the required columns
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        # Try case-insensitive match
        df_cols_lower = {col.lower(): col for col in df.columns}
        rename_map = {}
        for req in required_cols:
            if req.lower() in df_cols_lower:
                rename_map[df_cols_lower[req.lower()]] = req

        if rename_map:
            df = df.rename(columns=rename_map)
            missing = [col for col in required_cols if col not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}. Required: Open, High, Low, Close")
        return None

    # Try to parse datetime index
    datetime_cols = ["Date", "Datetime", "DateTime", "Time", "Timestamp", "date", "datetime", "time", "timestamp"]
    date_col = None

    for col in datetime_cols:
        if col in df.columns:
            date_col = col
            break

    if date_col is None and df.index.name in datetime_cols:
        date_col = df.index.name

    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        except:
            pass

    # If no date column found, try using the index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            st.warning("⚠️ Could not parse dates. Using row numbers as index.")

    # Ensure numeric columns
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate indicators
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACDSig"], df["MACDHist"] = macd(df["Close"])

    # ADX requires High, Low, Close
    if "High" in df.columns and "Low" in df.columns:
        df["ADX"], df["PlusDI"], df["MinusDI"] = adx(df)
    else:
        df["ADX"] = 0

    df = df.dropna()

    if len(df) < 50:
        st.warning("⚠️ Less than 50 rows of data after processing. Results may be unreliable.")

    return df

# ──────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ──────────────────────────────────────────────────────────────
if ALERT_HISTORY_KEY not in st.session_state:
    st.session_state[ALERT_HISTORY_KEY] = []

if MULTI_TF_CACHE_KEY not in st.session_state:
    st.session_state[MULTI_TF_CACHE_KEY] = None

if "bt_results" not in st.session_state:
    st.session_state["bt_results"] = None

if "bt_df" not in st.session_state:
    st.session_state["bt_df"] = None

# ──────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Forex & Commodities")
    st.caption("Trend Following Strategy")
    st.caption("**🔴 Strong Buy & Strong Sell Only**")
    st.divider()

    st.subheader("🌐 Markets & Timeframe")

    # Quick select buttons for categories
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 All Forex", use_container_width=True):
            st.session_state.selected_pairs = [p for p in PAIRS.keys() if p not in COMMODITIES]
    with col2:
        if st.button("🪙 All Metals", use_container_width=True):
            st.session_state.selected_pairs = [p for p in PAIRS.keys() if p in COMMODITIES]

    # Initialize selected_pairs in session state if not exists
    if "selected_pairs" not in st.session_state:
        st.session_state.selected_pairs = ["EUR/USD", "GBP/USD", "USD/JPY"]

    selected_pairs = st.multiselect(
        "Select Markets",
        list(PAIRS.keys()),
        default=st.session_state.selected_pairs,
        key="pair_selector"
    )

    timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=1)

    st.divider()
    st.subheader("🔔 Alerts")
    sound_on = st.toggle("🔊 Sound Alert", value=True)
    toast_on = st.toggle("💬 Toast Popup", value=True)
    banner_on = st.toggle("🚨 Alert Banner", value=True)

    st.divider()
    st.subheader("⚙️ Signal Sensitivity")
    min_conds = st.slider(
        "Min conditions to trigger",
        min_value=3, max_value=6, value=4,
        help="Lower = more signals (more noise). Higher = fewer but stronger signals."
    )

    st.divider()
    st.subheader("🔄 Auto Refresh")
    auto_refresh = st.toggle("Enable Auto Refresh", value=AUTOREFRESH_AVAILABLE)
    refresh_mins = st.slider("Interval (minutes)", 1, 60, 15,
                             disabled=not auto_refresh)

    if st.button("⚡ Refresh Now", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.session_state[MULTI_TF_CACHE_KEY] = None
        st.rerun()

    st.divider()
    st.caption(f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    if AUTOREFRESH_AVAILABLE and auto_refresh:
        st.caption(f"Next refresh in ~{refresh_mins} min")

    # Legend for signal colors
    st.divider()
    st.subheader("🎨 Signal Colors")
    st.markdown("""
    <div style="font-size:0.8rem;">
        <div style="background:#00b894;color:white;padding:4px 8px;border-radius:4px;margin:2px 0;">🚀 Strong Buy (5-6/6)</div>
        <div style="background:#d63031;color:white;padding:4px 8px;border-radius:4px;margin:2px 0;">🔻 Strong Sell (5-6/6)</div>
        <div style="background:#6c5ce7;color:white;padding:4px 8px;border-radius:4px;margin:2px 0;">📈 Buy (4/6) - Filtered</div>
        <div style="background:#f39c12;color:white;padding:4px 8px;border-radius:4px;margin:2px 0;">📉 Sell (4/6) - Filtered</div>
        <div style="background:#636e72;color:white;padding:4px 8px;border-radius:4px;margin:2px 0;">⏳ Neutral</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  AUTO REFRESH
# ──────────────────────────────────────────────────────────────
if AUTOREFRESH_AVAILABLE and auto_refresh:
    count = st_autorefresh(interval=refresh_mins * 60_000, limit=None, key="autorefresh_counter")

# ──────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────
st.title("📈 Forex & Commodities — Strong Signal Dashboard")
st.caption(
    "**Strategy:** 50 EMA / 200 EMA crossover · RSI (14) · MACD · ADX  |  "
    f"**Sensitivity:** {min_conds}/6 conditions required  |  "
    "**Filter:** 🚀 STRONG BUY & 🔻 STRONG SELL only"
)

# ──────────────────────────────────────────────────────────────
#  TABS
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Single Timeframe View",
    "🔍 Multi-Timeframe Scanner",
    "🧪 Backtesting Lab"
])

# ╔══════════════════════════════════════════════════════════════╗
# ║                    TAB 1: SINGLE TIMEFRAME                  ║
# ╚══════════════════════════════════════════════════════════════╝
with tab1:
    st.caption(f"**Active Timeframe:** {timeframe}")
    st.divider()

    # ──────────────────────────────────────────────────────────
    #  LOAD DATA + GENERATE SIGNALS
    # ──────────────────────────────────────────────────────────
    if not selected_pairs:
        st.warning("⚠️ Please select at least one market in the sidebar.")
        st.stop()

    interval, period, resample = TIMEFRAMES[timeframe]
    pair_results = {}
    new_alerts = []

    progress = st.progress(0, text="Fetching market data…")
    for idx, pair in enumerate(selected_pairs):
        ticker = PAIRS[pair]
        with st.spinner(f"Loading {pair}…"):
            df = fetch_and_prepare(ticker, interval, period, resample)
        progress.progress((idx + 1) / len(selected_pairs), text=f"Processed {pair}")

        if df is None or df.empty:
            pair_results[pair] = {"error": True}
            continue

        signal, score, max_score, conds, direction = evaluate_signal(df, min_conds)
        last = df.iloc[-1]

        # ── FILTER: Only keep STRONG_BUY and STRONG_SELL ─────────
        if direction not in ["STRONG_BUY", "STRONG_SELL"]:
            pair_results[pair] = {
                "error": False,
                "df": df,
                "signal": signal,
                "score": score,
                "max_score": max_score,
                "conds": conds,
                "direction": direction,
                "close": float(last["Close"]),
                "ema50": float(last["EMA50"]),
                "ema200": float(last["EMA200"]),
                "rsi": float(last["RSI"]),
                "macd": float(last["MACD"]),
                "adx": float(last["ADX"]),
                "filtered": True,
            }
            continue  # Skip alert generation for non-strong signals

        pair_results[pair] = {
            "error": False,
            "df": df,
            "signal": signal,
            "score": score,
            "max_score": max_score,
            "conds": conds,
            "direction": direction,
            "close": float(last["Close"]),
            "ema50": float(last["EMA50"]),
            "ema200": float(last["EMA200"]),
            "rsi": float(last["RSI"]),
            "macd": float(last["MACD"]),
            "adx": float(last["ADX"]),
            "filtered": False,
        }

        # Only STRONG signals trigger alerts
        new_alerts.append((pair, signal, direction, float(last["Close"])))

    progress.empty()

    # ──────────────────────────────────────────────────────────
    #  FIRE ALERTS (Only for STRONG signals)
    # ──────────────────────────────────────────────────────────
    if new_alerts:
        if sound_on:
            play_sound(new_alerts[0][2])

        if toast_on:
            for pair, signal, direction, price in new_alerts:
                icon = "🚀" if "BUY" in direction else "🔻"
                st.toast(f"{icon} **{pair}** — {signal}  @ {price:.5f}", icon=icon)

        if banner_on:
            for pair, signal, direction, price in new_alerts:
                css_cls = {
                    "STRONG_BUY": "alert-strong-buy",
                    "STRONG_SELL": "alert-strong-sell"
                }.get(direction, "alert-strong-buy")

                emoji = {"STRONG_BUY": "🚀", "STRONG_SELL": "🔻"}.get(direction, "")

                st.markdown(f"""
                <div class="{css_cls}">
                    {emoji} <strong>{pair}</strong> — <strong>{signal}</strong>
                    — Entry price: <code>{price:.5f}</code>
                    — {datetime.now(timezone.utc).strftime('%H:%M UTC')}
                </div>
                """, unsafe_allow_html=True)

        for pair, signal, direction, price in new_alerts:
            st.session_state[ALERT_HISTORY_KEY].append({
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "pair": pair,
                "signal": signal,
                "direction": direction,
                "price": f"{price:.5f}",
            })
        st.session_state[ALERT_HISTORY_KEY] = st.session_state[ALERT_HISTORY_KEY][-50:]

    # ──────────────────────────────────────────────────────────
    #  SIGNAL CARDS GRID
    # ──────────────────────────────────────────────────────────
    st.subheader("🎯 Strong Signals Only")

    strong_pairs = [p for p in selected_pairs
                    if not pair_results.get(p, {}).get("error")
                    and not pair_results.get(p, {}).get("filtered", True)]

    filtered_pairs = [p for p in selected_pairs
                      if not pair_results.get(p, {}).get("error")
                      and pair_results.get(p, {}).get("filtered", False)]

    if strong_pairs:
        cols = st.columns(min(len(strong_pairs), 4))
        for idx, pair in enumerate(strong_pairs):
            r = pair_results.get(pair, {})
            with cols[idx % 4]:
                direction = r["direction"]
                css_cls = {
                    "STRONG_BUY": "card-strong-buy",
                    "STRONG_SELL": "card-strong-sell"
                }.get(direction, "card-neutral")

                pct_bar = int(r["score"] / r["max_score"] * 100)
                tag = '<span class="commodity-tag">COMMODITY</span>' if pair in COMMODITIES else '<span class="forex-tag">FOREX</span>'

                st.markdown(f"""
                <div class="{css_cls}">
                    <div class="pair-name">{pair} {tag}</div>
                    <div class="sig-text">{r["signal"]}</div>
                    <div class="price-text">Price: {r["close"]:.5f}</div>
                    <div class="score-text">✦ Conditions: {r["score"]}/{r["max_score"]}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(pct_bar, text=f"Signal strength {pct_bar}%")
    else:
        st.info("💤 No STRONG BUY or STRONG SELL signals detected at this time. "
                "Try adjusting sensitivity or timeframe in the sidebar.")

    if filtered_pairs:
        with st.expander(f"🔇 Filtered Out — {len(filtered_pairs)} market(s) with weak/neutral signals"):
            for pair in filtered_pairs:
                r = pair_results[pair]
                st.markdown(f"• **{pair}** — {r['signal']} ({r['score']}/{r['max_score']} conditions met)")

    # ──────────────────────────────────────────────────────────
    #  DETAILED CHART VIEW
    # ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Detailed Chart & Analysis")

    valid_pairs = [p for p in selected_pairs if not pair_results.get(p, {}).get("error")]
    strong_valid = [p for p in valid_pairs if not pair_results.get(p, {}).get("filtered", True)]

    if not valid_pairs:
        st.error("No valid data available. Try refreshing.")
        st.stop()

    if strong_valid:
        default_pair = strong_valid[0]
        selected_detail = st.selectbox("Choose pair to inspect:", strong_valid, index=0)

        r = pair_results[selected_detail]

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Close", f"{r['close']:.5f}")
        m2.metric("50 EMA", f"{r['ema50']:.5f}",
                  delta=f"{((r['close']-r['ema50'])/r['ema50']*100):+.2f}%")
        m3.metric("200 EMA", f"{r['ema200']:.5f}",
                  delta=f"{((r['close']-r['ema200'])/r['ema200']*100):+.2f}%")
        m4.metric("RSI (14)", f"{r['rsi']:.1f}",
                  delta="Overbought" if r['rsi'] > 70 else ("Oversold" if r['rsi'] < 30 else "Neutral"))
        m5.metric("MACD", f"{r['macd']:.5f}")
        m6.metric("ADX", f"{r['adx']:.1f}",
                  delta="Trending" if r['adx'] > 25 else "Ranging")

        st.plotly_chart(build_chart(r["df"], selected_detail), use_container_width=True)

        if r["conds"]:
            st.subheader(f"📋 Signal Conditions — {selected_detail}")
            col_a, col_b = st.columns(2)
            items = list(r["conds"].items())
            half = (len(items) + 1) // 2
            for col, chunk in zip([col_a, col_b], [items[:half], items[half:]]):
                with col:
                    for cond, met in chunk:
                        icon = "✅" if met else "❌"
                        st.markdown(f"{icon} {cond}")
        else:
            st.info("📋 No active signal conditions — market is neutral / ranging on this timeframe.")
    else:
        st.info("🔇 No strong signals to display. All markets are currently showing weak or neutral signals.")

# ╔══════════════════════════════════════════════════════════════╗
# ║                 TAB 2: MULTI-TIMEFRAME SCANNER              ║
# ╚══════════════════════════════════════════════════════════════╝
with tab2:
    st.caption("**Scanning all pairs across all timeframes for Strong Buy & Strong Sell signals**")
    st.divider()

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        scan_clicked = st.button("🔍 Scan All Timeframes", type="primary", use_container_width=True)

    with col_btn2:
        st.caption("Click to scan all markets across 1H, 4H, Daily & Weekly timeframes")

    if scan_clicked or st.session_state[MULTI_TF_CACHE_KEY] is not None:
        if scan_clicked:
            st.cache_data.clear()
            with st.spinner("🔍 Scanning all timeframes... This may take a moment."):
                df_mtf = scan_all_timeframes(selected_pairs, min_conds)
            st.session_state[MULTI_TF_CACHE_KEY] = df_mtf
        else:
            df_mtf = st.session_state[MULTI_TF_CACHE_KEY]

        st.divider()

        tf_names = list(TIMEFRAMES.keys())
        strong_count = df_mtf["Has_Strong"].sum()
        total_pairs = len(df_mtf)

        strong_per_tf = {}
        for tf in tf_names:
            strong_per_tf[tf] = sum(1 for _, row in df_mtf.iterrows()
                                    if row[f"{tf}_direction"] in ["STRONG_BUY", "STRONG_SELL"])

        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        with col_s1:
            st.metric("Total Markets", total_pairs)
        with col_s2:
            st.metric("With Strong Signal", strong_count,
                      delta=f"{strong_count/total_pairs*100:.0f}%" if total_pairs > 0 else "0%")
        with col_s3:
            st.metric("1H Strong", strong_per_tf.get("1 Hour", 0))
        with col_s4:
            st.metric("4H Strong", strong_per_tf.get("4 Hours", 0))
        with col_s5:
            st.metric("Daily Strong", strong_per_tf.get("Daily", 0))

        st.divider()

        st.subheader("📋 Multi-Timeframe Signal Matrix")

        table_html = """
        <table style="width:100%; border-collapse: collapse; font-family: monospace; font-size: 0.9rem;">
        <thead>
            <tr style="background: #1a1c23; color: #fdcb6e;">
                <th style="padding: 12px 16px; text-align: left; border-bottom: 2px solid #fdcb6e;">Market</th>
        """
        for tf in tf_names:
            table_html += f'<th style="padding: 12px 16px; text-align: center; border-bottom: 2px solid #fdcb6e;">{tf}</th>'
        table_html += '<th style="padding: 12px 16px; text-align: center; border-bottom: 2px solid #fdcb6e;">Consensus</th>'
        table_html += '</tr></thead><tbody>'

        for _, row in df_mtf.iterrows():
            pair = row["Market"]
            has_strong = row["Has_Strong"]

            row_bg = "#0d1117" if not has_strong else "#0f1a0f" if "BUY" in str(row.get(f"{tf_names[0]}_direction", "")) else "#1a0f0f"

            table_html += f'<tr style="background: {row_bg}; border-bottom: 1px solid #2d3143;">'
            table_html += f'<td style="padding: 10px 16px; font-weight: 600; color: white;">{pair}</td>'

            strong_buy_count = sum(1 for tf in tf_names if row[f"{tf}_direction"] == "STRONG_BUY")
            strong_sell_count = sum(1 for tf in tf_names if row[f"{tf}_direction"] == "STRONG_SELL")

            for tf in tf_names:
                direction = row[f"{tf}_direction"]
                score = row[f"{tf}_score"]
                price = row[f"{tf}_price"]
                cell_html = render_mtf_cell(direction, row[tf], score, price)
                table_html += f'<td style="padding: 8px 10px; text-align: center;">{cell_html}</td>'

            if strong_buy_count >= 3:
                consensus = '<span class="mtf-consensus-strong" style="background: #00b894;">🚀 STRONG BUY</span>'
            elif strong_sell_count >= 3:
                consensus = '<span class="mtf-consensus-strong" style="background: #d63031;">🔻 STRONG SELL</span>'
            elif strong_buy_count >= 2:
                consensus = '<span class="mtf-buy">📈 BULLISH</span>'
            elif strong_sell_count >= 2:
                consensus = '<span class="mtf-sell">📉 BEARISH</span>'
            elif strong_buy_count >= 1:
                consensus = '<span class="mtf-buy" style="opacity: 0.5;">📈 Weak Bull</span>'
            elif strong_sell_count >= 1:
                consensus = '<span class="mtf-sell" style="opacity: 0.5;">📉 Weak Bear</span>'
            else:
                consensus = '<span class="mtf-neutral">⏳ No Signal</span>'

            table_html += f'<td style="padding: 8px 10px; text-align: center;">{consensus}</td>'
            table_html += '</tr>'

        table_html += '</tbody></table>'

        st.markdown(table_html, unsafe_allow_html=True)

        st.divider()

        st.subheader("🎯 All Strong Signals Detected")

        strong_signals_found = []
        for _, row in df_mtf.iterrows():
            pair = row["Market"]
            for tf in tf_names:
                direction = row[f"{tf}_direction"]
                if direction in ["STRONG_BUY", "STRONG_SELL"]:
                    score = row[f"{tf}_score"]
                    price = row[f"{tf}_price"]
                    strong_signals_found.append({
                        "Market": pair,
                        "Timeframe": tf,
                        "Signal": row[tf],
                        "Direction": direction,
                        "Score": f"{score}/6",
                        "Price": f"{price:.5f}"
                    })

        if strong_signals_found:
            df_strong = pd.DataFrame(strong_signals_found)
            st.dataframe(
                df_strong,
                column_config={
                    "Market": st.column_config.TextColumn("Market"),
                    "Timeframe": st.column_config.TextColumn("TF"),
                    "Signal": st.column_config.TextColumn("Signal"),
                    "Direction": st.column_config.TextColumn("Dir"),
                    "Score": st.column_config.TextColumn("Score"),
                    "Price": st.column_config.TextColumn("Price"),
                },
                hide_index=True,
                use_container_width=True,
            )
            st.caption(f"✅ **{len(strong_signals_found)}** strong signals found across all timeframes")
        else:
            st.info("💤 No strong signals found across any timeframe. Try adjusting sensitivity.")

# ╔══════════════════════════════════════════════════════════════╗
# ║                   TAB 3: BACKTESTING LAB                    ║
# ╚══════════════════════════════════════════════════════════════╝
with tab3:
    st.caption("**Backtest the strategy on historical data — upload CSV or use Yahoo Finance**")
    st.divider()

    # ── Data Source Selection ─────────────────────────────────
    st.subheader("📁 Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["📥 Upload CSV File", "📊 Download from Yahoo Finance"],
        horizontal=True
    )

    bt_df = None

    if data_source == "📥 Upload CSV File":
        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            st.markdown("""
            **CSV Requirements:**
            - Columns: `Open`, `High`, `Low`, `Close` (required)
            - Optional: `Date`/`Datetime` column (or use index)
            - Volume column is optional
            """)

        with col_up2:
            st.download_button(
                "📄 Download Sample CSV",
                data="""Date,Open,High,Low,Close,Volume
2023-01-02,1.2050,1.2085,1.2030,1.2060,125000
2023-01-03,1.2060,1.2090,1.2045,1.2075,118000
2023-01-04,1.2075,1.2100,1.2050,1.2080,132000""",
                file_name="sample_forex_data.csv",
                mime="text/csv",
                use_container_width=True
            )

        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=["csv"],
            help="CSV with Open, High, Low, Close columns"
        )

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded: {uploaded_file.name} — {len(raw_df)} rows")

                with st.expander("🔍 Preview Uploaded Data"):
                    st.dataframe(raw_df.head(10), use_container_width=True)

                bt_df = validate_and_prepare_upload(raw_df)
                if bt_df is not None:
                    st.session_state["bt_df"] = bt_df
                    st.success(f"✅ Data prepared: {len(bt_df)} rows after processing")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
        else:
            if st.session_state["bt_df"] is not None:
                bt_df = st.session_state["bt_df"]
                st.info(f"📊 Using previously uploaded data: {len(bt_df)} rows")

    else:  # Yahoo Finance
        col_bt1, col_bt2 = st.columns([1, 1])
        with col_bt1:
            bt_pair = st.selectbox(
                "Select Market",
                list(PAIRS.keys()),
                key="bt_pair_select"
            )
        with col_bt2:
            bt_timeframe = st.selectbox(
                "Select Timeframe",
                list(TIMEFRAMES.keys()),
                index=1,
                key="bt_tf_select"
            )

        col_bt3, col_bt4 = st.columns([1, 1])
        with col_bt3:
            bt_period_options = {
                "1 Hour": "59d",
                "4 Hours": "59d",
                "Daily": "2y",
                "Weekly": "2y"
            }
            bt_period = st.selectbox(
                "Data Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=3,
                key="bt_period_select"
            )
        with col_bt4:
            if st.button("📥 Fetch Data", type="primary", use_container_width=True):
                ticker = PAIRS[bt_pair]
                interval, period_default, resample = TIMEFRAMES[bt_timeframe]

                with st.spinner(f"Downloading {bt_pair} data..."):
                    bt_df = fetch_and_prepare(ticker, interval, bt_period, resample)

                if bt_df is not None and not bt_df.empty:
                    st.session_state["bt_df"] = bt_df
                    st.success(f"✅ Downloaded {len(bt_df)} rows for {bt_pair}")
                else:
                    st.error(f"❌ Failed to download data for {bt_pair}")

        if st.session_state["bt_df"] is not None:
            bt_df = st.session_state["bt_df"]

    # ── Backtest Configuration ───────────────────────────────
    if bt_df is not None and not bt_df.empty:
        st.divider()
        st.subheader("⚙️ Backtest Configuration")

        col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
        with col_cfg1:
            bt_initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000, max_value=1000000, value=10000, step=1000
            )
        with col_cfg2:
            bt_position_size = st.slider(
                "Position Size (% of capital)",
                min_value=0.5, max_value=10.0, value=2.0, step=0.5
            )
        with col_cfg3:
            bt_min_conds = st.slider(
                "Min Conditions",
                min_value=3, max_value=6, value=4,
                key="bt_min_conds"
            )
        with col_cfg4:
            bt_strong_only = st.toggle(
                "Strong Signals Only",
                value=True,
                help="Only trade STRONG BUY/SELL (5-6 conditions)"
            )

        col_cfg5, col_cfg6 = st.columns([1, 3])
        with col_cfg5:
            run_bt = st.button(
                "🚀 Run Backtest",
                type="primary",
                use_container_width=True
            )

        # ── Run Backtest ─────────────────────────────────────
        if run_bt:
            with st.spinner("🔄 Running backtest..."):
                bt_results = run_backtest(
                    bt_df,
                    min_conditions=bt_min_conds,
                    initial_capital=bt_initial_capital,
                    position_size_pct=bt_position_size,
                    use_strong_only=bt_strong_only
                )
                st.session_state["bt_results"] = bt_results

            st.success("✅ Backtest complete!")
            st.balloons()

        # ── Display Results ──────────────────────────────────
        if st.session_state["bt_results"] is not None:
            bt_results = st.session_state["bt_results"]
            metrics = bt_results["metrics"]
            trades = bt_results["trades"]

            st.divider()
            st.subheader("📊 Performance Metrics")

            # Metrics cards
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

            return_val = metrics["Total Return %"]
            return_color = "#00b894" if return_val > 0 else "#d63031"

            with col_m1:
                st.markdown(f"""
                <div class="bt-metric-card">
                    <div class="bt-metric-value" style="color: {return_color};">{return_val:+.2f}%</div>
                    <div class="bt-metric-label">Total Return</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m2:
                st.markdown(f"""
                <div class="bt-metric-card">
                    <div class="bt-metric-value">${metrics['Final Capital']:,.2f}</div>
                    <div class="bt-metric-label">Final Capital</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m3:
                wr_color = "#00b894" if metrics["Win Rate %"] >= 50 else "#f39c12"
                st.markdown(f"""
                <div class="bt-metric-card">
                    <div class="bt-metric-value" style="color: {wr_color};">{metrics['Win Rate %']}%</div>
                    <div class="bt-metric-label">Win Rate</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m4:
                pf = metrics["Profit Factor"]
                pf_color = "#00b894" if pf >= 1.5 else ("#f39c12" if pf >= 1.0 else "#d63031")
                st.markdown(f"""
                <div class="bt-metric-card">
                    <div class="bt-metric-value" style="color: {pf_color};">{pf:.2f}</div>
                    <div class="bt-metric-label">Profit Factor</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m5:
                dd_val = metrics["Max Drawdown %"]
                st.markdown(f"""
                <div class="bt-metric-card">
                    <div class="bt-metric-value" style="color: #d63031;">{dd_val:.2f}%</div>
                    <div class="bt-metric-label">Max Drawdown</div>
                </div>
                """, unsafe_allow_html=True)

            # Second row of metrics
            col_m6, col_m7, col_m8, col_m9, col_m10 = st.columns(5)

            with col_m6:
                st.metric("Total Trades", metrics["Total Trades"])
            with col_m7:
                st.metric("Winning", metrics["Winning Trades"])
            with col_m8:
                st.metric("Losing", metrics["Losing Trades"])
            with col_m9:
                st.metric("Avg Win", f"{metrics['Avg Win %']:.2f}%")
            with col_m10:
                st.metric("Avg Loss", f"{metrics['Avg Loss %']:.2f}%")

            st.divider()

            # ── Charts ───────────────────────────────────────
            st.subheader("📈 Backtest Visualization")
            st.plotly_chart(build_backtest_chart(bt_results), use_container_width=True)

            # ── Trade List ───────────────────────────────────
            st.divider()
            st.subheader("📋 Trade History")

            if trades:
                df_trades = pd.DataFrame(trades)

                # Format columns
                df_trades["P&L"] = df_trades["P&L $"].apply(
                    lambda x: f'<span class="bt-profit">+${x:,.2f}</span>' if x > 0
                    else f'<span class="bt-loss">-${abs(x):,.2f}</span>'
                )

                # Summary stats
                total_pnl = sum(t["P&L $"] for t in trades)
                pnl_color = "#00b894" if total_pnl > 0 else "#d63031"

                st.markdown(f"""
                **Total P&L:** <span style="color: {pnl_color}; font-size: 1.2rem; font-weight: 700;">
                ${total_pnl:+,.2f}
                </span> | **Trades:** {len(trades)} | **Win Rate:** {metrics['Win Rate %']}%
                """, unsafe_allow_html=True)

                # Display trade table
                display_cols = ["Entry Date", "Exit Date", "Type", "Entry Price", "Exit Price", "P&L %", "Exit Reason"]
                st.dataframe(
                    df_trades[display_cols],
                    column_config={
                        "Entry Date": st.column_config.DatetimeColumn("Entry Date"),
                        "Exit Date": st.column_config.DatetimeColumn("Exit Date"),
                        "Type": st.column_config.TextColumn("Type"),
                        "Entry Price": st.column_config.NumberColumn("Entry", format="%.5f"),
                        "Exit Price": st.column_config.NumberColumn("Exit", format="%.5f"),
                        "P&L %": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
                        "Exit Reason": st.column_config.TextColumn("Reason"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

                # Download trades CSV
                csv_trades = df_trades.to_csv(index=False)
                st.download_button(
                    "📥 Download Trade History (CSV)",
                    csv_trades,
                    f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No trades were executed during this backtest period.")
    else:
        st.info("👆 Upload a CSV file or download data from Yahoo Finance to start backtesting.")

        # Example placeholder
        st.divider()
        st.markdown("""
        ### 📖 Backtesting Guide
        
        **How to use the Backtesting Lab:**
        
        1. **Upload your data** (CSV with Open/High/Low/Close) or download from Yahoo Finance
        2. **Configure parameters**: Initial capital, position size, signal sensitivity
        3. **Run backtest** to see performance metrics and trade history
        4. **Analyze results**: Equity curve, drawdown, win rate, profit factor
        
        **Strategy Rules Tested:**
        - Enters LONG on Strong Buy signals
        - Enters SHORT on Strong Sell signals
        - Exits when signal reverses or at stop loss/take profit
        - 5% stop loss and 15% take profit built in
        """)

# ──────────────────────────────────────────────────────────────
#  STRATEGY REFERENCE (shared between tabs)
# ──────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖 Strategy Reference — Trend Following Rules"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🟢 STRONG BUY Setup (5-6 conditions met)
        - Price **above** 200 EMA
        - 50 EMA **above** 200 EMA (Golden Cross)
        - Price pulling back toward / holding above 50 EMA
        - RSI between **45–70** (bullish zone)
        - MACD line **above** signal line
        - ADX **> 25** (trend strength confirmed)

        ### 🔴 STRONG SELL Setup (5-6 conditions met)
        - Price **below** 200 EMA
        - 50 EMA **below** 200 EMA (Death Cross)
        - Price pulling back toward / below 50 EMA
        - RSI between **30–55** (bearish zone)
        - MACD line **below** signal line
        - ADX **> 25** (trend strength confirmed)
        """)
    with col2:
        st.markdown("""
        ### ⚙️ Multi-TF Confirmation
        | Timeframe | Weight |
        |-----------|--------|
        | Weekly    | Highest |
        | Daily     | High |
        | 4 Hours   | Medium |
        | 1 Hour    | Entry timing |

        ### 💡 Risk Management
        - Best trades: **≥ 3 timeframes aligned**
        - Risk **≤ 1–2%** per trade
        - Target **≥ 1:2** risk:reward ratio
        - Best sessions: **London** & **NY overlap**
        - Best markets: **EUR/USD · GBP/USD · Gold**
        """)

# ──────────────────────────────────────────────────────────────
#  ALERT HISTORY
# ──────────────────────────────────────────────────────────────
st.divider()
with st.expander(f"🔔 Alert History ({len(st.session_state[ALERT_HISTORY_KEY])} strong alerts this session)"):
    hist = st.session_state[ALERT_HISTORY_KEY]
    if hist:
        df_hist = pd.DataFrame(hist[::-1])
        st.dataframe(
            df_hist,
            column_config={
                "direction": st.column_config.TextColumn("Dir"),
                "pair": st.column_config.TextColumn("Market"),
                "signal": st.column_config.TextColumn("Signal"),
                "price": st.column_config.TextColumn("Price"),
                "time": st.column_config.TextColumn("Time (UTC)"),
            },
            hide_index=True,
            use_container_width=True,
        )
        if st.button("🗑️ Clear History"):
            st.session_state[ALERT_HISTORY_KEY] = []
            st.rerun()
    else:
        st.info("No strong alerts triggered yet this session.")

# ──────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is for educational and informational purposes only. "
    "It does not constitute financial advice."
)