import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import psycopg2
import psycopg2.extras
import yfinance as yf
import json
from typing import Optional, Tuple, Dict, Any

# ── Instrument registry ────────────────────────────────────────────────────────
INSTRUMENTS = {
    "EUR/USD": {"ticker": "EURUSD=X", "pip": 10.0, "pip_size": 0.0001, "corr": "DXY ↑ = bearish"},
    "GBP/USD": {"ticker": "GBPUSD=X", "pip": 10.0, "pip_size": 0.0001, "corr": "DXY ↑ = bearish"},
    "AUD/USD": {"ticker": "AUDUSD=X", "pip": 10.0, "pip_size": 0.0001, "corr": "Gold ↑ = bullish"},
    "NZD/USD": {"ticker": "NZDUSD=X", "pip": 10.0, "pip_size": 0.0001, "corr": "AUD/USD alignment"},
    "USD/JPY": {"ticker": "USDJPY=X", "pip": 9.09, "pip_size": 0.01, "corr": "US10Y yields aligned"},
    "USD/CHF": {"ticker": "USDCHF=X", "pip": 10.8, "pip_size": 0.0001, "corr": "EUR/USD inverse"},
    "USD/CAD": {"ticker": "USDCAD=X", "pip": 7.4, "pip_size": 0.0001, "corr": "Oil inverse"},
    "EUR/GBP": {"ticker": "EURGBP=X", "pip": 12.5, "pip_size": 0.0001, "corr": "EUR/USD vs GBP/USD"},
    "EUR/JPY": {"ticker": "EURJPY=X", "pip": 9.09, "pip_size": 0.01, "corr": "Risk-on sentiment"},
    "GBP/JPY": {"ticker": "GBPJPY=X", "pip": 9.09, "pip_size": 0.01, "corr": "Volatility proxy"},
    "AUD/JPY": {"ticker": "AUDJPY=X", "pip": 9.09, "pip_size": 0.01, "corr": "Risk appetite gauge"},
    "EUR/AUD": {"ticker": "EURAUD=X", "pip": 6.3, "pip_size": 0.0001, "corr": "EUR/USD vs AUD/USD"},
    "GBP/AUD": {"ticker": "GBPAUD=X", "pip": 6.3, "pip_size": 0.0001, "corr": "GBP/USD vs AUD/USD"},
    "EUR/CAD": {"ticker": "EURCAD=X", "pip": 7.4, "pip_size": 0.0001, "corr": "EUR/USD vs Oil"},
    "GBP/CAD": {"ticker": "GBPCAD=X", "pip": 7.4, "pip_size": 0.0001, "corr": "GBP/USD vs Oil"},
    "USD/ZAR": {"ticker": "USDZAR=X", "pip": 0.55, "pip_size": 0.0001, "corr": "Gold & risk sentiment"},
    "EUR/ZAR": {"ticker": "EURZAR=X", "pip": 0.55, "pip_size": 0.0001, "corr": "EUR/USD + Gold"},
    "GBP/ZAR": {"ticker": "GBPZAR=X", "pip": 0.55, "pip_size": 0.0001, "corr": "GBP/USD + Gold"},
    "🥇 Gold": {"ticker": "GC=F", "pip": 10.0, "pip_size": 0.10, "corr": "DXY inverse, VIX"},
    "🥈 Silver": {"ticker": "SI=F", "pip": 10.0, "pip_size": 0.01, "corr": "Gold correlation"},
    "🪙 Platinum": {"ticker": "PL=F", "pip": 10.0, "pip_size": 0.10, "corr": "Industrial demand"},
}

# Correlated instrument groups — pairs in the same group share directional exposure
CORR_GROUPS = {
    "USD vs majors (same dir = stacked USD risk)": {"EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD"},
    "USD-base pairs (same dir = stacked USD risk)": {"USD/JPY", "USD/CHF", "USD/CAD"},
    "JPY crosses (same dir = stacked JPY risk)": {"USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY"},
    "Gold / Silver (highly correlated)": {"🥇 Gold", "🥈 Silver"},
    "ZAR pairs (same dir = stacked ZAR risk)": {"USD/ZAR", "EUR/ZAR", "GBP/ZAR"},
}

# Trend Following specific commodities
TREND_COMMODITIES = {"🥇 Gold", "🥈 Silver", "🪙 Platinum"}
TREND_TIMEFRAMES = {
    "1 Hour": ("60m", "59d", "1h"),
    "4 Hours": ("60m", "59d", "4h"),
    "Daily": ("1d", "2y", None),
    "Weekly": ("1d", "2y", "1W")
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Daily Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    /* Theme-adaptive layout vars */
    .stApp {
      --border: color-mix(in srgb, var(--text-color) 12%, transparent);
      --muted:  color-mix(in srgb, var(--text-color) 55%, transparent);
    }
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    .stApp{background:var(--background-color);}
    section[data-testid="stSidebar"]{background:var(--secondary-background-color)!important;border-right:1px solid var(--border,#21262d);}
    .card{background:var(--secondary-background-color);border:1px solid var(--border,#21262d);border-radius:12px;padding:20px;margin-bottom:16px;}
    .card-header{font-size:13px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#8b949e;margin-bottom:14px;}
    .chip-go  {background:#0d4a2f;color:#3fb950;border:1px solid #238636;border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600;display:inline-block;}
    .chip-wait{background:#4a2d0d;color:#e3b341;border:1px solid #9e6a03;border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600;display:inline-block;}
    .chip-no  {background:#4a0d0d;color:#f85149;border:1px solid #8b2d2d;border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600;display:inline-block;}
    .metric-box{background:var(--background-color);border:1px solid var(--border,#21262d);border-radius:8px;padding:14px;text-align:center;}
    .metric-value{font-size:22px;font-weight:700;color:#c9d1d9;}
    .metric-label{font-size:11px;color:#8b949e;margin-top:2px;font-weight:500;letter-spacing:.04em;text-transform:uppercase;}
    .section-title{font-size:16px;font-weight:700;color:var(--text-color);margin:24px 0 12px 0;padding-left:4px;border-left:3px solid #388bfd;}
    .hero{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);border:1px solid #21262d;border-radius:16px;padding:28px 32px;margin-bottom:24px;position:relative;overflow:hidden;}
    .hero::before{content:'';position:absolute;top:-50%;right:-10%;width:300px;height:300px;background:radial-gradient(circle,rgba(56,139,253,.08) 0%,transparent 70%);border-radius:50%;}
    .prog-track{background:#21262d;border-radius:8px;height:10px;margin:6px 0 2px 0;overflow:hidden;}
    .ticker-tag{display:inline-block;background:#0d1117;border:1px solid #30363d;border-radius:4px;padding:2px 7px;font-size:11px;color:#58a6ff;font-family:monospace;margin-left:6px;}
    .atr-box{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:12px 16px;margin:8px 0;font-size:13px;}
    .atr-row{display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid #21262d;}
    .atr-row:last-child{border-bottom:none;}
    .atr-label{color:#8b949e;font-size:12px;}
    .atr-val{color:#e6edf3;font-weight:600;font-family:monospace;}
    .atr-val-green{color:#3fb950;font-weight:700;font-family:monospace;}
    .atr-val-red{color:#f85149;font-weight:700;font-family:monospace;}
    .db-badge-ok {background:#0d4a2f;color:#3fb950;border:1px solid #238636;border-radius:6px;padding:3px 10px;font-size:11px;font-weight:600;}
    .db-badge-err{background:#4a0d0d;color:#f85149;border:1px solid #8b2d2d;border-radius:6px;padding:3px 10px;font-size:11px;font-weight:600;}

    /* Trend Following Signal Cards */
    .trend-card-strong-buy {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white; padding: 18px; border-radius: 14px; text-align: center;
        box-shadow: 0 4px 20px rgba(0,184,148,0.5);
        border: 2px solid #00e6a0;
    }
    .trend-card-buy {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        color: white; padding: 18px; border-radius: 14px; text-align: center;
        box-shadow: 0 4px 15px rgba(108,92,231,0.35);
    }
    .trend-card-strong-sell {
        background: linear-gradient(135deg, #d63031 0%, #e17055 100%);
        color: white; padding: 18px; border-radius: 14px; text-align: center;
        box-shadow: 0 4px 20px rgba(214,48,49,0.5);
        border: 2px solid #ff4757;
    }
    .trend-card-sell {
        background: linear-gradient(135deg, #fdcb6e 0%, #f39c12 100%);
        color: white; padding: 18px; border-radius: 14px; text-align: center;
        box-shadow: 0 4px 15px rgba(253,203,110,0.35);
    }
    .trend-card-neutral {
        background: linear-gradient(135deg, #636e72 0%, #b2bec3 100%);
        color: white; padding: 18px; border-radius: 14px; text-align: center;
        box-shadow: 0 4px 12px rgba(99,110,114,0.25);
    }
    .trend-pair-name { font-size: 1.0rem; font-weight: 600; opacity: 0.9; }
    .trend-sig-text { font-size: 1.5rem; font-weight: 800; letter-spacing: 1px; margin: 4px 0; }
    .trend-price-text { font-size: 0.9rem; font-family: monospace; }

    .commodity-tag { background:#fdcb6e; color:#2d3436; padding:2px 8px; border-radius:12px; font-size:0.7rem; font-weight:700; display:inline-block; margin-top:4px; }
    .forex-tag { background:#74b9ff; color:#2d3436; padding:2px 8px; border-radius:12px; font-size:0.7rem; font-weight:700; display:inline-block; margin-top:4px; }

    #MainMenu,footer,header{visibility:hidden;}
    /* Keep sidebar toggle visible regardless of header being hidden */
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;display:flex !important;}
    [data-testid="stSidebarCollapseButton"]{visibility:visible !important;display:flex !important;}
    /* Force sidebar to always be present in layout */
    section[data-testid="stSidebar"]{display:block !important;background:#161b22 !important;border-right:1px solid #21262d;}
    [data-testid="stSidebarNav"]{display:none;}
    .block-container{padding-top:1.5rem;max-width:1380px;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TREND FOLLOWING TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_val, plus_di, minus_di


@st.cache_data(ttl=300, show_spinner=False)
def fetch_trend_data(ticker: str, interval: str, period: str, resample: Optional[str]) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    except Exception as e:
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
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
            }).dropna()
        else:
            df = df.resample(resample).agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
            }).dropna()
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACDSig"], df["MACDHist"] = macd(df["Close"])
    df["ADX"], df["PlusDI"], df["MinusDI"] = adx(df)
    return df.dropna()


def evaluate_trend_signal(df: pd.DataFrame, min_conditions: int = 4) -> Tuple[str, int, int, Dict[str, bool], str]:
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


def build_trend_chart(df: pd.DataFrame, pair: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.58, 0.21, 0.21],
        subplot_titles=[f"{pair} — Price with EMAs", "RSI (14)", "MACD (12 / 26 / 9)"],
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#00b894", decreasing_line_color="#d63031",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="50 EMA", line=dict(color="#fdcb6e", width=2)), row=1,
                  col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df["EMA200"], name="200 EMA", line=dict(color="#a29bfe", width=2.5, dash="dash")),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#74b9ff", width=1.8)), row=2, col=1)
    for level, color in [(70, "rgba(214,48,49,0.6)"), (30, "rgba(0,184,148,0.6)"), (50, "rgba(255,255,255,0.25)")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, row=2, col=1)
    hist_colors = ["#00b894" if v >= 0 else "#d63031" for v in df["MACDHist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACDHist"], name="Histogram", marker_color=hist_colors, opacity=0.7), row=3,
                  col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#0984e3", width=1.8)), row=3,
                  col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACDSig"], name="Signal", line=dict(color="#e17055", width=1.8)), row=3,
                  col=1)
    fig.update_layout(height=700, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f1117",
                      showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


# ══════════════════════════════════════════════════════════════════
# DATABASE HELPERS (same as before)
# ══════════════════════════════════════════════════════════════════

def get_db_connection(cfg):
    return psycopg2.connect(
        host=cfg["host"], port=cfg["port"],
        dbname=cfg["dbname"], user=cfg["user"], password=cfg["password"]
    )


def init_db(cfg):
    """Create trades table and migrate new outcome-tracking columns if absent."""
    try:
        conn = get_db_connection(cfg)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_setups (
                id            SERIAL PRIMARY KEY,
                logged_at     TIMESTAMP NOT NULL DEFAULT NOW(),
                instrument    VARCHAR(30),
                ticker        VARCHAR(20),
                direction     VARCHAR(10),
                session       VARCHAR(20),
                score         VARCHAR(20),
                verdict       VARCHAR(20),
                atr14         FLOAT,
                atr20         FLOAT,
                sl_pips       FLOAT,
                tp1_pips      FLOAT,
                tp2_pips      FLOAT,
                lot_size      FLOAT,
                risk_amount   FLOAT,
                rr_tp1        FLOAT,
                rr_tp2        FLOAT,
                account_bal   FLOAT,
                risk_pct      FLOAT,
                checks_passed INT,
                checks_total  INT,
                checks_detail JSONB,
                notes         TEXT
            )
        """)
        # Outcome-tracking columns — safe to run on existing tables
        for col_def in [
            "entry_price FLOAT",
            "outcome     VARCHAR(10)",
            "close_price FLOAT",
            "pips_gained FLOAT",
            "r_multiple  FLOAT",
            "is_open     BOOLEAN DEFAULT TRUE",
        ]:
            cur.execute(f"ALTER TABLE trade_setups ADD COLUMN IF NOT EXISTS {col_def}")
        conn.commit()
        cur.close()
        conn.close()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def save_trade(cfg, row: dict):
    conn = get_db_connection(cfg)
    cur = conn.cursor()
    cur.execute("""
                INSERT INTO trade_setups (logged_at, instrument, ticker, direction, session,
                                          score, verdict, atr14, atr20, sl_pips, tp1_pips, tp2_pips,
                                          lot_size, risk_amount, rr_tp1, rr_tp2, account_bal, risk_pct,
                                          checks_passed, checks_total, checks_detail, notes)
                VALUES (%(logged_at)s, %(instrument)s, %(ticker)s, %(direction)s, %(session)s,
                        %(score)s, %(verdict)s, %(atr14)s, %(atr20)s, %(sl_pips)s, %(tp1_pips)s, %(tp2_pips)s,
                        %(lot_size)s, %(risk_amount)s, %(rr_tp1)s, %(rr_tp2)s, %(account_bal)s, %(risk_pct)s,
                        %(checks_passed)s, %(checks_total)s, %(checks_detail)s, %(notes)s)
                """, row)
    conn.commit()
    cur.close()
    conn.close()


def load_trades(cfg, limit=50):
    conn = get_db_connection(cfg)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
                SELECT id,
                       logged_at,
                       instrument,
                       ticker,
                       direction, session, score, verdict, atr14, atr20, sl_pips, tp1_pips, tp2_pips, lot_size, risk_amount, rr_tp1, rr_tp2, account_bal, risk_pct, checks_passed, checks_total, notes
                FROM trade_setups
                ORDER BY logged_at DESC
                    LIMIT %s
                """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def delete_trade(cfg, trade_id):
    conn = get_db_connection(cfg)
    cur = conn.cursor()
    cur.execute("DELETE FROM trade_setups WHERE id = %s", (trade_id,))
    conn.commit()
    cur.close()
    conn.close()


def close_trade(cfg, trade_id: int, entry_price: float, close_price: float,
                pips_gained: float, r_multiple: float, outcome: str):
    conn = get_db_connection(cfg)
    cur  = conn.cursor()
    cur.execute("""
        UPDATE trade_setups
        SET entry_price = %s, close_price = %s, pips_gained = %s,
            r_multiple = %s, outcome = %s, is_open = FALSE
        WHERE id = %s
    """, (entry_price, close_price, pips_gained, r_multiple, outcome, trade_id))
    conn.commit()
    cur.close()
    conn.close()


def load_daily_losses(cfg, max_losses: int = 2) -> dict:
    """Count today's closed losing trades and return limit status."""
    try:
        conn = get_db_connection(cfg)
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT COUNT(*) AS cnt
            FROM trade_setups
            WHERE outcome = 'LOSS'
              AND is_open = FALSE
              AND DATE(logged_at) = CURRENT_DATE
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        count = int(row["cnt"]) if row else 0
        return {
            "losses_today": count,
            "limit": max_losses,
            "blocked": count >= max_losses,
        }
    except Exception:
        return {"losses_today": 0, "limit": max_losses, "blocked": False}


def load_open_trades(cfg):
    conn = get_db_connection(cfg)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, logged_at, instrument, direction, sl_pips, tp1_pips, tp2_pips, lot_size
        FROM trade_setups
        WHERE is_open IS TRUE OR is_open IS NULL
        ORDER BY logged_at DESC LIMIT 20
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def load_stats(cfg, n: int = 20) -> Optional[Dict]:
    conn = get_db_connection(cfg)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT outcome, r_multiple, pips_gained
        FROM trade_setups
        WHERE is_open = FALSE AND outcome IS NOT NULL
        ORDER BY logged_at DESC LIMIT %s
    """, (n,))
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    if not rows:
        return None
    wins      = [r for r in rows if r["outcome"] == "WIN"]
    losses    = [r for r in rows if r["outcome"] == "LOSS"]
    be_trades = [r for r in rows if r["outcome"] == "BE"]
    total     = len(rows)
    win_rate  = len(wins) / total * 100 if total else 0
    win_rs    = [r["r_multiple"] for r in wins   if r["r_multiple"]]
    loss_rs   = [abs(r["r_multiple"]) for r in losses if r["r_multiple"]]
    avg_win_r  = sum(win_rs)  / len(win_rs)  if win_rs  else 0
    avg_loss_r = sum(loss_rs) / len(loss_rs) if loss_rs else 1.0
    loss_rate  = len(losses) / total if total else 0
    expectancy = (win_rate / 100 * avg_win_r) - (loss_rate * avg_loss_r)
    pf         = (len(wins) * avg_win_r) / (len(losses) * avg_loss_r) \
                 if losses and avg_loss_r else 0.0
    return {
        "total": total, "wins": len(wins), "losses": len(losses),
        "be": len(be_trades), "win_rate": win_rate,
        "avg_win_r": avg_win_r, "avg_loss_r": avg_loss_r,
        "expectancy": expectancy, "profit_factor": pf,
    }


# ══════════════════════════════════════════════════════════════════
# ATR FETCHER
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_atr(ticker: str, pip_size: float):
    """Fetch daily OHLCV and compute ATR(14) and ATR(20). Returns dict."""
    try:
        df = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 22:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        def _atr(high, low, close, period):
            prev = close.shift(1)
            tr = pd.concat([high - low,
                            (high - prev).abs(),
                            (low - prev).abs()], axis=1).max(axis=1)
            return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        df["atr14"] = _atr(df["High"], df["Low"], df["Close"], 14)
        df["atr20"] = _atr(df["High"], df["Low"], df["Close"], 20)
        df = df.dropna(subset=["atr14", "atr20"])
        if df.empty:
            return None
        a14 = float(df["atr14"].iloc[-1])
        a20 = float(df["atr20"].iloc[-1])
        a14_pips = round(a14 / pip_size, 1)
        a20_pips = round(a20 / pip_size, 1)
        sl_pips = round(a14_pips * 1.5, 1)
        tp1_pips = round(sl_pips * 2.0, 1)
        tp2_pips = round(sl_pips * 3.0, 1)
        return {
            "atr14": round(a14, 5), "atr20": round(a20, 5),
            "atr14_pips": a14_pips, "atr20_pips": a20_pips,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
            "atr_ok": a14 > a20,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# SESSION KILL ZONE DETECTOR
# ══════════════════════════════════════════════════════════════════

def get_session_status() -> dict:
    now = datetime.now(timezone.utc)
    h = now.hour + now.minute / 60
    t = now.strftime("%H:%M UTC")
    if 7.0 <= h < 9.0:
        return {"window": "London Kill Zone", "prime": True, "color": "#3fb950",
                "icon": "🟢", "desc": "07:00–09:00 UTC — highest order flow", "time": t}
    if 12.0 <= h < 14.0:
        return {"window": "NY Kill Zone", "prime": True, "color": "#3fb950",
                "icon": "🟢", "desc": "12:00–14:00 UTC — NY open sweep", "time": t}
    if 15.0 <= h < 17.0:
        return {"window": "London Close", "prime": False, "color": "#e3b341",
                "icon": "🟡", "desc": "15:00–17:00 UTC — trend continuation", "time": t}
    if 0.0 <= h < 3.0:
        return {"window": "Tokyo Session", "prime": False, "color": "#f85149",
                "icon": "🔴", "desc": "00:00–03:00 UTC — low probability, avoid", "time": t}
    # Compute wait to next prime window
    next_windows = [(7.0, "London KZ"), (12.0, "NY KZ"), (15.0, "London Close"), (24.0, "London KZ")]
    next_name, wait_mins = "London KZ", 0
    for start, name in next_windows:
        if h < start:
            wait_mins = int((start - h) * 60)
            next_name = name
            break
    return {"window": "Dead Zone", "prime": False, "color": "#484f58",
            "icon": "⚫", "desc": f"Between sessions — {wait_mins}m until {next_name}", "time": t}


def check_correlation_exposure(open_trades: list, direction: str, instrument: str) -> list:
    """
    Check if the proposed trade stacks correlated exposure with existing open trades.
    Returns a list of warning dicts: {group, conflicting, count}
    """
    warnings = []
    for group_name, group_instruments in CORR_GROUPS.items():
        if instrument not in group_instruments:
            continue
        conflicts = [
            t for t in open_trades
            if t.get("instrument") in group_instruments
            and t.get("direction") == direction
            and t.get("instrument") != instrument
        ]
        if conflicts:
            warnings.append({
                "group": group_name,
                "conflicting": ", ".join(t["instrument"] for t in conflicts),
                "count": len(conflicts),
            })
    return warnings


def get_mtf_alignment(ticker: str, direction: str) -> dict:
    """
    Check Weekly / Daily / 4H EMA alignment for the selected direction.
    Reuses fetch_trend_data() so no extra API calls if data is already cached.
    """
    res = {}
    try:
        df_w = fetch_trend_data(ticker, "1d", "2y", "1W")
        if df_w is not None and len(df_w) > 10:
            r = df_w.iloc[-1]
            c, e50, e200 = float(r["Close"]), float(r["EMA50"]), float(r["EMA200"])
            res["weekly"] = "BULL" if c > e50 > e200 else ("BEAR" if c < e50 < e200 else "NEUTRAL")
        else:
            res["weekly"] = None
    except Exception:
        res["weekly"] = None
    try:
        df_d = fetch_trend_data(ticker, "1d", "2y", None)
        if df_d is not None and len(df_d) > 10:
            r = df_d.iloc[-1]
            c, e50 = float(r["Close"]), float(r["EMA50"])
            res["daily"] = "BULL" if c > e50 else "BEAR"
        else:
            res["daily"] = None
    except Exception:
        res["daily"] = None
    try:
        df_4h = fetch_trend_data(ticker, "60m", "59d", "4h")
        if df_4h is not None and len(df_4h) > 10:
            r = df_4h.iloc[-1]
            c, e50 = float(r["Close"]), float(r["EMA50"])
            res["4h"] = "BULL" if c > e50 else "BEAR"
        else:
            res["4h"] = None
    except Exception:
        res["4h"] = None
    target = "BULL" if direction == "LONG" else "BEAR"
    aligned = sum(1 for v in res.values() if v == target)
    return {**res, "aligned": aligned, "total": 3, "target": target}


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

CHECKS_TOTAL = 18
for i in range(1, CHECKS_TOTAL + 1):
    if f"check_{i}" not in st.session_state:
        st.session_state[f"check_{i}"] = False

# Trend Following session state
if "trend_selected_pairs" not in st.session_state:
    st.session_state.trend_selected_pairs = ["EUR/USD", "GBP/USD", "🥇 Gold"]
if "trend_timeframe" not in st.session_state:
    st.session_state.trend_timeframe = "4 Hours"
if "trend_min_conds" not in st.session_state:
    st.session_state.trend_min_conds = 4

defaults = {
    "selected_instrument": "EUR/USD",
    "trade_direction": "LONG",
    "session": "London",
    "notes": "",
    "account_bal": 10000.0,
    "risk_pct": 1.0,
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "trading",
    "db_user": "postgres",
    "db_pass": "",
    "db_ok": False,
    "db_msg": "Not connected",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# PAGE NAVIGATION
# ══════════════════════════════════════════════════════════════════

if "current_page" not in st.session_state:
    st.session_state.current_page = "checklist"

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Navigation")

    # Page selection
    page = st.radio(
        "Select Module",
        ["📋 Checklist", "📈 Trend Following Signals"],
        index=0 if st.session_state.current_page == "checklist" else 1,
        horizontal=True
    )

    if page == "📋 Checklist":
        st.session_state.current_page = "checklist"
        st.markdown("---")
        st.markdown("### 📋 Trade Setup")
    else:
        st.session_state.current_page = "trend"
        st.markdown("---")
        st.markdown("### 📈 Trend Following")

    # Conditional sidebar content based on page
    if st.session_state.current_page == "checklist":
        # Original checklist sidebar
        st.page_link("daily-trading-checklist.py",    label="00. Checklist",           icon="📋")
        st.page_link("pages/macro-bias.py",           label="01. Macro Bias",           icon="🌐")
        st.page_link("pages/news-filter.py",          label="02. News Filter",          icon="📰")
        st.page_link("pages/correlations.py",         label="03. Correlations",         icon="🔗")
        st.page_link("pages/atr-volatility.py",       label="04. ATR Volatility",       icon="📊")
        st.page_link("pages/weekly-ema.py",           label="05. Weekly EMA",           icon="📉")
        st.page_link("pages/weekly-rsi.py",           label="06. Weekly RSI",           icon="📡")
        st.page_link("pages/weekly-swing.py",         label="07. Weekly Swing",         icon="🔄")
        st.page_link("pages/daily-trend.py",          label="08. Daily Trend",          icon="📈")
        st.page_link("pages/daily-macd.py",           label="09. Daily MACD",           icon="📊")
        st.page_link("pages/4H-confluence-zone.py",   label="10. 4H Confluence Zone",   icon="🎯")
        st.page_link("pages/confluence-checker.py",   label="11. 2/3 Confluence Check", icon="🔀")
        st.page_link("pages/15m-rejection.py",        label="12. 15M Rejection",        icon="🕯️")
        st.page_link("pages/15m-entry-signal.py",     label="13. 15M Entry Signal",     icon="⚡")
        st.page_link("pages/stop-structure.py",       label="14. Stop Structure",       icon="🛡️")
        st.page_link("pages/rr-calculator.py",        label="15. R:R Calculator",       icon="⚖️")
        st.page_link("pages/trade-journal.py",        label="16. Trade Journal",         icon="📓")
        st.page_link("pages/market-structure.py",     label="17. Market Structure",     icon="🏗️")
        st.page_link("pages/setup-ranker.py",         label="18. Setup Ranker",         icon="🎰")
        st.page_link("pages/backtest-workflow.py",    label="19. Backtest Workflow",    icon="🧪")
        st.divider()

        # Instrument selection
        inst_keys = list(INSTRUMENTS.keys())
        if st.session_state.selected_instrument not in inst_keys:
            st.session_state.selected_instrument = inst_keys[0]

        selected_key = st.selectbox("Instrument", inst_keys,
                                    index=inst_keys.index(st.session_state.selected_instrument))
        st.session_state.selected_instrument = selected_key
        inst_data = INSTRUMENTS[selected_key]

        st.markdown(
            f'<span class="ticker-tag">{inst_data["ticker"]}</span>'
            f'<br><span style="font-size:11px;color:#8b949e;margin-top:4px;display:block;">'
            f'📡 {inst_data["corr"]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        st.session_state.trade_direction = st.radio("Direction", ["LONG", "SHORT"], horizontal=True)
        st.session_state.session = st.selectbox("Session", ["London", "New York", "Tokyo", "Sydney"])

        st.markdown("---")
        st.markdown("**📐 Position Sizing**")
        st.session_state.account_bal = st.number_input("Account Balance ($)", value=float(st.session_state.account_bal),
                                                       step=500.0, format="%.2f")
        st.session_state.risk_pct = st.slider("Risk per Trade (%)", 0.25, 3.0, float(st.session_state.risk_pct), 0.25)
        pip_val = inst_data["pip"]

        st.markdown("---")
        st.markdown("**📊 ATR & Levels (Auto)**")
        if st.button("🔄 Fetch ATR + Calculate Levels", use_container_width=True):
            with st.spinner("Fetching market data…"):
                fetch_atr.clear()
                result = fetch_atr(inst_data["ticker"], inst_data["pip_size"])
                st.session_state[f"atr_data_{selected_key}"] = result

        atr_data = st.session_state.get(f"atr_data_{selected_key}", None)
        if atr_data is None:
            atr_data = fetch_atr(inst_data["ticker"], inst_data["pip_size"])
            st.session_state[f"atr_data_{selected_key}"] = atr_data

        if atr_data:
            atr_color = "#3fb950" if atr_data["atr_ok"] else "#f85149"
            atr_label = "✅ Volatile" if atr_data["atr_ok"] else "⚠️ Low Vol"
            sl_pips = atr_data["sl_pips"]
            tp1_pips = atr_data["tp1_pips"]
            tp2_pips = atr_data["tp2_pips"]
            atr14 = atr_data["atr14"]
            atr20 = atr_data["atr20"]
            atr_ok = atr_data["atr_ok"]
            st.session_state.trend_sl_pips = sl_pips
            st.session_state.trend_tp1_pips = tp1_pips
            st.session_state.trend_tp2_pips = tp2_pips
        else:
            st.warning("⚠️ Could not fetch live data. Enter manually.")
            sl_pips = st.number_input("Stop Loss (pips)", value=20.0, step=1.0)
            tp1_pips = st.number_input("TP1 (pips)", value=40.0, step=1.0)
            tp2_pips = st.number_input("TP2 (pips)", value=60.0, step=1.0)
            atr14 = atr20 = 0.0
            atr_ok = None

        # PostgreSQL config
        st.markdown("---")
        st.markdown("**🗄️ PostgreSQL Database**")
        st.session_state.db_host = st.text_input("Host", value=st.session_state.db_host)
        st.session_state.db_port = st.number_input("Port", value=int(st.session_state.db_port), step=1)
        st.session_state.db_name = st.text_input("Database", value=st.session_state.db_name)
        st.session_state.db_user = st.text_input("User", value=st.session_state.db_user)
        st.session_state.db_pass = st.text_input("Password", value=st.session_state.db_pass, type="password")

        db_cfg = {
            "host": st.session_state.db_host,
            "port": int(st.session_state.db_port),
            "dbname": st.session_state.db_name,
            "user": st.session_state.db_user,
            "password": st.session_state.db_pass,
        }

        if st.button("🔌 Connect & Init DB", use_container_width=True):
            ok, msg = init_db(db_cfg)
            st.session_state.db_ok = ok
            st.session_state.db_msg = msg

        badge_cls = "db-badge-ok" if st.session_state.db_ok else "db-badge-err"
        badge_txt = "✅ Connected" if st.session_state.db_ok else f"❌ {st.session_state.db_msg[:30]}"
        st.markdown(f'<span class="{badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ Reset All Checks", use_container_width=True):
            for i in range(1, CHECKS_TOTAL + 1):
                st.session_state[f"check_{i}"] = False
            st.rerun()

    else:  # Trend Following sidebar
        # Quick select buttons for categories
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 All Forex", use_container_width=True):
                st.session_state.trend_selected_pairs = [p for p in INSTRUMENTS.keys() if p not in TREND_COMMODITIES]
        with col2:
            if st.button("🪙 All Metals", use_container_width=True):
                st.session_state.trend_selected_pairs = [p for p in INSTRUMENTS.keys() if p in TREND_COMMODITIES]

        selected_pairs = st.multiselect(
            "Select Markets",
            list(INSTRUMENTS.keys()),
            default=st.session_state.trend_selected_pairs,
            key="trend_pair_selector"
        )
        st.session_state.trend_selected_pairs = selected_pairs

        timeframe = st.selectbox("Timeframe", list(TREND_TIMEFRAMES.keys()),
                                 index=list(TREND_TIMEFRAMES.keys()).index(st.session_state.trend_timeframe))
        st.session_state.trend_timeframe = timeframe

        st.divider()
        st.subheader("⚙️ Signal Sensitivity")
        min_conds = st.slider(
            "Min conditions to trigger",
            min_value=3, max_value=6, value=st.session_state.trend_min_conds,
            help="Lower = more signals (more noise). Higher = fewer but stronger signals."
        )
        st.session_state.trend_min_conds = min_conds

        st.divider()
        st.subheader("🔄 Auto Refresh")
        auto_refresh = st.toggle("Enable Auto Refresh", value=True)
        refresh_mins = st.slider("Interval (minutes)", 1, 60, 15, disabled=not auto_refresh)

        if st.button("⚡ Refresh Now", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Legend for signal colors
        st.divider()
        st.subheader("🎨 Signal Colors")
        st.markdown("""
        <div style="font-size:0.8rem;">
            <div style="background:#00b894;color:white;padding:4px 8px;border-radius:4px;margin:2px 0;">🚀 Strong Buy</div>
            <div style="background:#6c5ce7;color:white;padding:4px 8px;border-radius:4px;margin:2px 0;">📈 Buy</div>
            <div style="background:#d63031;color:white;padding:4px 8px;border-radius:4px;margin:2px 0;">🔻 Strong Sell</div>
            <div style="background:#f39c12;color:white;padding:4px 8px;border-radius:4px;margin:2px 0;">📉 Sell</div>
            <div style="background:#636e72;color:white;padding:4px 8px;border-radius:4px;margin:2px 0;">⏳ Neutral</div>
        </div>
        """, unsafe_allow_html=True)

        # Auto refresh logic
        if auto_refresh:
            st.caption(f"Next refresh in ~{refresh_mins} min")

# ══════════════════════════════════════════════════════════════════
# MAIN PAGE CONTENT
# ══════════════════════════════════════════════════════════════════

if st.session_state.current_page == "checklist":
    # ──────────────────────────────────────────────────────────────
    # ORIGINAL CHECKLIST PAGE CONTENT
    # ──────────────────────────────────────────────────────────────

    # Get current values
    inst_data = INSTRUMENTS[st.session_state.selected_instrument]
    pip_val = inst_data["pip"]
    sl_pips = st.session_state.get("trend_sl_pips", 20.0)
    tp1_pips = st.session_state.get("trend_tp1_pips", 40.0)
    tp2_pips = st.session_state.get("trend_tp2_pips", 60.0)
    atr_data = st.session_state.get(f"atr_data_{st.session_state.selected_instrument}", None)
    atr14 = atr_data["atr14"] if atr_data else 0.0
    atr20 = atr_data["atr20"] if atr_data else 0.0
    atr_ok = atr_data["atr_ok"] if atr_data else None

    # Computed values
    checked = sum(st.session_state[f"check_{i}"] for i in range(1, CHECKS_TOTAL + 1))
    pct = int(checked / CHECKS_TOTAL * 100)
    risk_amount = st.session_state.account_bal * (st.session_state.risk_pct / 100)
    lot_size = round(risk_amount / sl_pips / pip_val, 2) if (sl_pips and pip_val) else 0.0
    rr_tp1 = round(tp1_pips / sl_pips, 2) if sl_pips else 0.0
    rr_tp2 = round(tp2_pips / sl_pips, 2) if sl_pips else 0.0

    critical_ok = all(st.session_state[f"check_{i}"] for i in [11, 12, 13, 14, 15, 16])
    if checked >= 16 and critical_ok:
        signal = ("🟢 GO", "chip-go", "#3fb950")
    elif pct >= 55:
        signal = ("🟡 WAIT", "chip-wait", "#e3b341")
    else:
        signal = ("🔴 PASS", "chip-no", "#f85149")

    prog_color = "#3fb950" if (checked >= 16 and critical_ok) else "#e3b341" if pct >= 55 else "#f85149"
    direction_icon = "🔼" if st.session_state.trade_direction == "LONG" else "🔽"
    direction_col = "#3fb950" if st.session_state.trade_direction == "LONG" else "#f85149"
    display_name = st.session_state.selected_instrument.split(" ", 1)[-1] if st.session_state.selected_instrument[
                                                                                 0] in "🥇🪙" else st.session_state.selected_instrument

    # Hero Header
    st.markdown(f"""
    <div class="hero">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
        <div>
          <div style="font-size:26px;font-weight:700;color:#e6edf3;">📈 Daily Trading System</div>
          <div style="color:#8b949e;font-size:14px;margin-top:6px;">18-Point Pre-Trade Checklist · Multi-Timeframe Confluence</div>
          <div style="font-size:13px;color:#388bfd;font-weight:500;margin-top:4px;">
            🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M')} · {st.session_state.session} Session
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:30px;font-weight:700;color:#e6edf3;line-height:1.1;">{display_name}</div>
          <div style="font-size:12px;color:#58a6ff;font-family:monospace;margin-top:2px;">{inst_data['ticker']}</div>
          <div style="font-size:15px;font-weight:600;color:{direction_col};margin-top:4px;">{direction_icon} {st.session_state.trade_direction}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    kpis = [
        (k1, f"{checked}/{CHECKS_TOTAL}", "Checks Passed", prog_color),
        (k2, f"{pct}%", "Confidence", prog_color),
        (k3, f"{atr14:.5f}" if atr14 else "—", "ATR(14)", "#c9d1d9"),
        (k4, f"{atr20:.5f}" if atr20 else "—", "ATR(20)", "#c9d1d9"),
        (k5, f"{sl_pips}", "SL Pips", "#f85149"),
        (k6, f"{rr_tp1}:1", "R:R TP1", "#3fb950" if rr_tp1 >= 2 else "#f85149"),
        (k7, f"{lot_size:.2f}", "Lot Size", "#388bfd"),
    ]
    for col, val, lbl, color in kpis:
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value" style="color:{color};">{val}</div><div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True)

    # Progress bar
    st.markdown(f"""
    <div style="margin:16px 0 24px 0;">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="font-size:12px;color:#8b949e;font-weight:500;">CHECKLIST PROGRESS</span>
        <span style="font-size:12px;color:{prog_color};font-weight:600;">{pct}% complete · {signal[0]}</span>
      </div>
      <div class="prog-track"><div style="background:{prog_color};width:{pct}%;height:100%;border-radius:8px;transition:width .4s;"></div></div>
    </div>
    """, unsafe_allow_html=True)

    # MTF Alignment Strip
    with st.spinner("Checking MTF alignment…"):
        mtf = get_mtf_alignment(inst_data["ticker"], st.session_state.trade_direction)

    def _mtf_badge(label, val, target):
        if val is None:
            return f'<span style="background:#1c2128;color:#484f58;border:1px solid #30363d;border-radius:6px;padding:3px 10px;font-size:12px;font-weight:600;margin-right:6px;">{label} —</span>'
        color = "#3fb950" if val == target else ("#f85149" if val != "NEUTRAL" else "#8b949e")
        icon  = "✅" if val == target else ("❌" if val != "NEUTRAL" else "⚪")
        bg    = "#0d4a2f" if val == target else ("#4a0d0d" if val != "NEUTRAL" else "#1c2128")
        disp  = "BULL" if val == "BULL" else ("BEAR" if val == "BEAR" else "~")
        return (f'<span style="background:{bg};color:{color};border:1px solid {color}55;'
                f'border-radius:6px;padding:3px 10px;font-size:12px;font-weight:600;margin-right:6px;">'
                f'{icon} {label} {disp}</span>')

    mtf_score = mtf["aligned"]
    mtf_bonus_badge = ""
    if mtf_score == 3:
        mtf_bonus_badge = '<span style="background:#2a1e00;color:#e3b341;border:1px solid #e3b34155;border-radius:6px;padding:3px 12px;font-size:12px;font-weight:700;">⭐ FULLY ALIGNED</span>'
    elif mtf_score == 2:
        mtf_bonus_badge = '<span style="background:#1c2128;color:#fdcb6e;border:1px solid #fdcb6e55;border-radius:6px;padding:3px 12px;font-size:12px;font-weight:700;">🔶 2/3 ALIGNED</span>'
    else:
        mtf_bonus_badge = '<span style="background:#1c2128;color:#484f58;border:1px solid #30363d;border-radius:6px;padding:3px 12px;font-size:12px;font-weight:600;">⬛ {}/3 ALIGNED</span>'.format(mtf_score)

    target = mtf["target"]
    st.markdown(
        f'<div style="background:#161b22;border:1px solid #21262d;border-radius:10px;'
        f'padding:12px 18px;margin-bottom:16px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        f'<span style="font-size:12px;color:#8b949e;font-weight:600;letter-spacing:.06em;margin-right:4px;">MTF ALIGNMENT</span>'
        f'{_mtf_badge("Weekly", mtf.get("weekly"), target)}'
        f'{_mtf_badge("Daily", mtf.get("daily"), target)}'
        f'{_mtf_badge("4H", mtf.get("4h"), target)}'
        f'<span style="margin-left:auto;">{mtf_bonus_badge}</span>'
        f'</div>',
        unsafe_allow_html=True)

    # Main Layout
    left, right = st.columns([3, 2], gap="large")


    def render_check(num, title, desc):
        col_a, col_b = st.columns([0.07, 0.93])
        with col_a:
            val = st.checkbox("", value=st.session_state[f"check_{num}"], key=f"cb_{num}")
            st.session_state[f"check_{num}"] = val
        with col_b:
            icon = "✅" if st.session_state[f"check_{num}"] else "⬜"
            st.markdown(f"**{icon} #{num} — {title}**  \n<span style='font-size:12px;color:#8b949e;'>{desc}</span>",
                        unsafe_allow_html=True)
        st.markdown("---")


    # LEFT: Checklist
    with left:
        atr_hint = (f"✅ ATR14 {atr14:.5f} > ATR20 {atr20:.5f} — volatility supports follow-through" if atr_ok
                    else f"❌ ATR14 {atr14:.5f} ≤ ATR20 {atr20:.5f} — low volatility caution" if atr_ok is False
        else "Fetch ATR data from sidebar")

        st.markdown('<div class="section-title">🌐 Macro Bias & News Filter</div>', unsafe_allow_html=True)
        render_check(1, "Macro bias confirmed", "Rates, GDP, inflation all favour trade direction")
        render_check(2, "No red-folder news", "No high-impact news within 1 hour of entry window")
        render_check(3, "Correlated markets align", f"Expected correlation: {inst_data['corr']}")
        render_check(4, "ATR above 20-period avg", atr_hint)

        st.markdown('<div class="section-title">📅 Weekly Timeframe Bias</div>', unsafe_allow_html=True)
        if st.button("🔗 Auto-fill from Trend Signals", key="auto_trend"):
            with st.spinner("Fetching trend signals…"):
                _dir = st.session_state.trade_direction
                _df_w = fetch_trend_data(inst_data["ticker"], "1d", "2y", "1W")
                _df_d = fetch_trend_data(inst_data["ticker"], "1d", "2y", None)
                if _df_w is not None and len(_df_w) > 10:
                    _, _, _, _, _dw = evaluate_trend_signal(_df_w)
                    if (_dw in ("BUY", "STRONG_BUY") and _dir == "LONG") or \
                       (_dw in ("SELL", "STRONG_SELL") and _dir == "SHORT"):
                        for i in [5, 6, 7]:
                            st.session_state[f"check_{i}"] = True
                if _df_d is not None and len(_df_d) > 10:
                    _, _, _, _, _dd = evaluate_trend_signal(_df_d)
                    if (_dd in ("BUY", "STRONG_BUY") and _dir == "LONG") or \
                       (_dd in ("SELL", "STRONG_SELL") and _dir == "SHORT"):
                        for i in [8, 9]:
                            st.session_state[f"check_{i}"] = True
            st.rerun()
        render_check(5, "Weekly EMA aligned", "Weekly EMA direction matches your trade direction")
        render_check(6, "Weekly RSI has room", "RSI not already overbought (>70) or oversold (<30)")
        render_check(7, "Weekly Swing ✅ Aligned", "Weekly Swing tab daily confirmation shows ✅ Aligned · <a href='/weekly-swing' target='_self' style='color:#58a6ff;'>Open Weekly Swing →</a>")

        st.markdown('<div class="section-title">📆 Daily Trend Confirmation</div>', unsafe_allow_html=True)
        render_check(8, "Daily trend intact", "EMA20 > EMA50 for longs / EMA20 < EMA50 for shorts")
        render_check(9, "Daily MACD momentum", "MACD histogram turning in the direction of the trade · <a href='/daily-macd' target='_self' style='color:#58a6ff;'>Open MACD Scanner →</a>")
        sess = get_session_status()
        render_check(10, "Entry in session kill zone",
                     f"London KZ: 07–09 UTC · NY KZ: 12–14 UTC · London Close: 15–17 UTC · "
                     f"Now: {sess['icon']} {sess['window']} ({sess['time']}) — {sess['desc']}")

        st.markdown('<div class="section-title">⏱️ 4H Confluence Zone</div>', unsafe_allow_html=True)
        render_check(11, "Price at 4H confluence zone", "Price at overlap of Fibonacci + Pivot + EMA on 4H chart")
        render_check(12, "Min 2/3 confluence elements", "At least 2 of 3 elements confirmed at zone · <a href='/confluence-checker' target='_self' style='color:#58a6ff;'>Open Confluence Checker →</a>")
        render_check(13, "Candlestick rejection on 15M", "Rejection candle or structural confirmation on 15M · <a href='/15m-rejection' target='_self' style='color:#58a6ff;'>Open 15M Scanner →</a>")

        st.markdown('<div class="section-title">🎯 Entry Signal & Risk Management</div>', unsafe_allow_html=True)
        render_check(14, "15M entry signal fired", "Stochastic crossover + RSI reset on 15M chart · <a href='/15m-entry-signal' target='_self' style='color:#58a6ff;'>Open Entry Scanner →</a>")
        render_check(15, "Stop below/above structure", f"SL = {sl_pips} pips (1.5 × ATR14) below/above structure · <a href='/stop-structure' target='_self' style='color:#58a6ff;'>Open Stop Calculator →</a>")
        render_check(16, "R:R ≥ 2:1 to TP1",
                     f"TP1 = {tp1_pips} pips → R:R {rr_tp1}:1 {'✅' if rr_tp1 >= 2 else '❌ Below minimum'} · <a href='/rr-calculator' target='_self' style='color:#58a6ff;'>Open R:R Calculator →</a>")
        render_check(17, "Partial TP + breakeven rule set",
                     f"50% close at TP1 ({tp1_pips} pips) → SL immediately to entry (BE) · "
                     f"Runner 50% trails to TP2 ({tp2_pips} pips) · Never hold through −1R on open trade")
        render_check(18, "Position size within limit",
                     f"Lot size: {lot_size:.2f} · Risk: ${risk_amount:.2f} ({st.session_state.risk_pct}% of ${st.session_state.account_bal:,.0f})")

    # RIGHT: Dashboard
    with right:
        # ── Daily Loss Limit Banner ──────────────────────────────────
        db_cfg = {
            "host": st.session_state.db_host,
            "port": int(st.session_state.db_port),
            "dbname": st.session_state.db_name,
            "user": st.session_state.db_user,
            "password": st.session_state.db_pass,
        }
        daily_loss_limit = 2
        if st.session_state.db_ok:
            try:
                dll = load_daily_losses(db_cfg, max_losses=daily_loss_limit)
                losses_today = dll["losses_today"]
                dl_blocked = dll["blocked"]
            except Exception:
                dll = {"losses_today": 0, "limit": daily_loss_limit, "blocked": False}
                losses_today, dl_blocked = 0, False
        else:
            dll = {"losses_today": 0, "limit": daily_loss_limit, "blocked": False}
            losses_today, dl_blocked = 0, False

        if dl_blocked:
            st.markdown(f"""
            <div style="background:#4a0d0d;border:2px solid #f85149;border-radius:10px;
                        padding:14px 20px;margin-bottom:14px;text-align:center;">
              <div style="font-size:20px;font-weight:800;color:#f85149;letter-spacing:1px;">
                🚫 DAILY LOSS LIMIT HIT
              </div>
              <div style="font-size:13px;color:#8b949e;margin-top:4px;">
                {losses_today}/{daily_loss_limit} losses today — no new trades allowed
              </div>
            </div>
            """, unsafe_allow_html=True)
        elif losses_today == daily_loss_limit - 1:
            st.markdown(f"""
            <div style="background:#2a1500;border:1px solid #e3b341;border-radius:10px;
                        padding:10px 18px;margin-bottom:14px;text-align:center;">
              <div style="font-size:14px;font-weight:700;color:#e3b341;">
                ⚠️ {losses_today}/{daily_loss_limit} losses today — 1 more triggers daily limit
              </div>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.db_ok:
            st.markdown(f"""
            <div style="background:#0a1a0a;border:1px solid #238636;border-radius:10px;
                        padding:8px 18px;margin-bottom:14px;text-align:center;">
              <div style="font-size:12px;color:#3fb950;">
                ✅ {losses_today}/{daily_loss_limit} losses today — daily limit clear
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Correlation Exposure Check ───────────────────────────────
        if st.session_state.db_ok:
            try:
                _open = load_open_trades(db_cfg)
                _corr_warns = check_correlation_exposure(
                    _open, st.session_state.trade_direction,
                    st.session_state.selected_instrument
                )
                if _corr_warns:
                    warns_html = "".join(
                        f'<div style="margin:4px 0;font-size:12px;color:#fdcb6e;">'
                        f'⚠️ <b>{w["group"]}</b><br>'
                        f'&nbsp;&nbsp;Also open: {w["conflicting"]}</div>'
                        for w in _corr_warns
                    )
                    st.markdown(f"""
                    <div style="background:#2a1f00;border:1px solid #fdcb6e55;border-radius:10px;
                                padding:10px 16px;margin-bottom:14px;">
                      <div style="font-size:12px;font-weight:700;color:#fdcb6e;margin-bottom:4px;">
                        🔗 CORRELATED EXPOSURE
                      </div>
                      {warns_html}
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                pass

        # Verdict
        if checked >= 16 and critical_ok:
            verdict_text = f"All {checked}/18 checks passed + critical path clear — trade has edge"
        elif not critical_ok:
            miss = [i for i in [11, 12, 13, 14, 15, 16] if not st.session_state[f"check_{i}"]]
            verdict_text = f"Critical path incomplete: checks {', '.join(f'#{i}' for i in miss)} — no entry"
        elif pct >= 55:
            verdict_text = "Building confluence — complete critical checks 11–16 before entry"
        else:
            verdict_text = "Insufficient confirmation — stay flat, wait for setup"
        verdict_icon = "🟢" if (checked >= 16 and critical_ok) else "🟡" if pct >= 55 else "🔴"
        critical_badge = "✅ Critical clear" if critical_ok else "⚠️ Critical incomplete"
        st.markdown(f"""
        <div class="card" style="border-color:{prog_color}33;">
          <div class="card-header">🧭 Trade Verdict</div>
          <div style="text-align:center;padding:8px 0;">
            <div style="font-size:46px;margin-bottom:6px;">{verdict_icon}</div>
            <div style="font-size:22px;font-weight:700;color:{prog_color};margin-bottom:10px;">{signal[0].split(None, 1)[1]}</div>
            <div class="{signal[1]}" style="font-size:14px;padding:5px 18px;">{checked}/18 · {critical_badge}</div>
            <div style="margin-top:12px;font-size:12px;color:#8b949e;">{verdict_text}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Levels card
        st.markdown(f"""
        <div class="card">
          <div class="card-header">📐 Auto-Calculated Levels — {display_name}</div>
          <table style="width:100%;border-collapse:collapse;font-size:13px;color:#c9d1d9;">
            <tr style="border-bottom:1px solid #21262d;">
              <td style="padding:8px 4px;color:#8b949e;">ATR(14)</td>
              <td style="padding:8px 4px;text-align:right;font-weight:600;font-family:monospace;">{atr14:.5f} &nbsp;|&nbsp; {atr_data['atr14_pips'] if atr_data else '—'} pips</td>
            </tr>
            <tr style="border-bottom:1px solid #21262d;">
              <td style="padding:8px 4px;color:#8b949e;">ATR(20)</td>
              <td style="padding:8px 4px;text-align:right;font-weight:600;font-family:monospace;">{atr20:.5f} &nbsp;|&nbsp; {atr_data['atr20_pips'] if atr_data else '—'} pips</td>
            </tr>
            <tr style="border-bottom:1px solid #21262d;">
              <td style="padding:8px 4px;color:#8b949e;">Stop Loss <span style="font-size:10px;">(1.5×ATR14)</span></td>
              <td style="padding:8px 4px;text-align:right;font-weight:700;color:#f85149;">{sl_pips} pips</td>
            </tr>
            <tr style="border-bottom:1px solid #21262d;">
              <td style="padding:8px 4px;color:#8b949e;">TP1 <span style="font-size:10px;">(2:1 R:R)</span></td>
              <td style="padding:8px 4px;text-align:right;font-weight:700;color:#3fb950;">{tp1_pips} pips</td>
            </tr>
            <tr style="border-bottom:1px solid #21262d;">
              <td style="padding:8px 4px;color:#8b949e;">TP2 <span style="font-size:10px;">(3:1 R:R)</span></td>
              <td style="padding:8px 4px;text-align:right;font-weight:700;color:#3fb950;">{tp2_pips} pips</td>
            </tr>
            <tr style="border-bottom:1px solid #21262d;">
              <td style="padding:8px 4px;color:#8b949e;">Lot Size</td>
              <td style="padding:8px 4px;text-align:right;font-weight:700;color:#388bfd;font-size:15px;">{lot_size:.2f} lots</td>
            </tr>
            <tr>
              <td style="padding:8px 4px;color:#8b949e;">Risk Amount</td>
              <td style="padding:8px 4px;text-align:right;font-weight:600;color:#e3b341;">${risk_amount:.2f} ({st.session_state.risk_pct}%)</td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        # Radar
        cats = ["Macro", "No News", "Corr.", "ATR", "Wk EMA", "Wk RSI", "Wk Swing",
                "Daily", "MACD", "Session", "4H Zone", "2/3 Conf", "15M Cndle",
                "15M Sig", "SL Str", "R:R", "Part TP", "Pos Size"]
        vals = [1 if st.session_state[f"check_{i}"] else 0 for i in range(1, 19)]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], fill='toself',
            fillcolor='rgba(56,139,253,0.15)',
            line=dict(color='#388bfd', width=2), marker=dict(color='#388bfd', size=5),
        ))
        fig.update_layout(
            polar=dict(bgcolor='#0d1117',
                       angularaxis=dict(tickfont=dict(size=9, color='#8b949e'), linecolor='#21262d',
                                        gridcolor='#21262d'),
                       radialaxis=dict(range=[0, 1], tickvals=[0, .5, 1], ticktext=['', '', ''], gridcolor='#21262d',
                                       linecolor='#21262d')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22',
            showlegend=False, margin=dict(l=40, r=40, t=20, b=20), height=270,
        )
        st.markdown('<div class="card"><div class="card-header">🕸️ Confluence Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))
        st.markdown('</div>', unsafe_allow_html=True)

        # Section breakdown
        sections = {
            "Macro & News (1–4)": [1, 2, 3, 4],
            "Weekly (5–7)": [5, 6, 7],
            "Daily (8–10)": [8, 9, 10],
            "4H Zone (11–13)": [11, 12, 13],
            "Entry & Risk (14–18)": [14, 15, 16, 17, 18],
        }
        st.markdown('<div class="card"><div class="card-header">📊 Section Breakdown</div>', unsafe_allow_html=True)
        for sec_name, ids in sections.items():
            n = sum(st.session_state[f"check_{i}"] for i in ids)
            t = len(ids)
            p = int(n / t * 100)
            c = "#3fb950" if p == 100 else "#e3b341" if p >= 50 else "#f85149"
            st.markdown(f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:12px;color:#c9d1d9;">{sec_name}</span>
                <span style="font-size:12px;font-weight:600;color:{c};">{n}/{t}</span>
              </div>
              <div class="prog-track"><div style="background:{c};width:{p}%;height:100%;border-radius:8px;"></div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Notes
        st.markdown("**📝 Trade Notes**")
        st.session_state.notes = st.text_area("", value=st.session_state.notes,
                                              placeholder="Key levels, bias, confluences, news context…",
                                              height=90, label_visibility="collapsed")

        # Log button
        if st.button("💾 Save Trade Setup to PostgreSQL", type="primary", use_container_width=True):
            if not st.session_state.db_ok:
                st.error("❌ Not connected to PostgreSQL. Configure & connect in the sidebar first.")
            elif dl_blocked:
                st.error(f"🚫 Daily loss limit hit ({losses_today}/{daily_loss_limit}) — no new trades allowed today. Come back tomorrow.")
            else:
                checks_detail = {f"check_{i}": st.session_state[f"check_{i}"] for i in range(1, CHECKS_TOTAL + 1)}
                row = {
                    "logged_at": datetime.now(),
                    "instrument": display_name,
                    "ticker": inst_data["ticker"],
                    "direction": st.session_state.trade_direction,
                    "session": st.session_state.session,
                    "score": f"{checked}/{CHECKS_TOTAL}",
                    "verdict": signal[0],
                    "atr14": atr14,
                    "atr20": atr20,
                    "sl_pips": sl_pips,
                    "tp1_pips": tp1_pips,
                    "tp2_pips": tp2_pips,
                    "lot_size": lot_size,
                    "risk_amount": round(risk_amount, 2),
                    "rr_tp1": rr_tp1,
                    "rr_tp2": rr_tp2,
                    "account_bal": st.session_state.account_bal,
                    "risk_pct": st.session_state.risk_pct,
                    "checks_passed": checked,
                    "checks_total": CHECKS_TOTAL,
                    "checks_detail": json.dumps(checks_detail),
                    "notes": st.session_state.notes,
                }
                try:
                    save_trade(db_cfg, row)
                    st.success(
                        f"✅ Saved to PostgreSQL — {display_name} · {signal[0]} · {datetime.now().strftime('%H:%M:%S')}")
                except Exception as e:
                    st.error(f"❌ Save failed: {e}")

    # ── Performance Stats ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Performance Stats — Last 20 Closed Trades")

    if st.session_state.db_ok:
        try:
            stats = load_stats(db_cfg, n=20)
            if stats:
                target = 66.0
                wr = stats["win_rate"]
                banner_color = "#0d4a2f" if wr >= target else "#4a0d0d"
                banner_border = "#238636" if wr >= target else "#8b2d2d"
                banner_text_color = "#3fb950" if wr >= target else "#f85149"
                banner_icon = "🎯" if wr >= target else "⚠️"
                st.markdown(f"""
                <div style="background:{banner_color};border:1px solid {banner_border};border-radius:10px;
                            padding:12px 20px;margin-bottom:16px;display:flex;align-items:center;gap:12px;">
                  <span style="font-size:22px;">{banner_icon}</span>
                  <div>
                    <span style="color:{banner_text_color};font-weight:700;font-size:15px;">
                      {wr:.1f}% Win Rate {'≥' if wr >= target else '<'} 66% target
                    </span>
                    <span style="color:#8b949e;font-size:12px;margin-left:12px;">
                      {stats['wins']}W / {stats['losses']}L / {stats['be']}BE — last {stats['total']} closed trades
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                stat_items = [
                    (sc1, f"{wr:.1f}%", "Win Rate", banner_text_color),
                    (sc2, f"{stats['wins']}/{stats['total']}", "W / Total", "#c9d1d9"),
                    (sc3, f"{stats['expectancy']:+.2f}R", "Expectancy", "#3fb950" if stats["expectancy"] > 0 else "#f85149"),
                    (sc4, f"{stats['profit_factor']:.2f}", "Profit Factor", "#3fb950" if stats["profit_factor"] >= 1.5 else "#e3b341"),
                    (sc5, f"{stats['avg_win_r']:.2f}R", "Avg Win R", "#c9d1d9"),
                ]
                for col, val, lbl, color in stat_items:
                    with col:
                        st.markdown(
                            f'<div class="metric-box"><div class="metric-value" style="color:{color};">{val}</div>'
                            f'<div class="metric-label">{lbl}</div></div>',
                            unsafe_allow_html=True)
            else:
                st.info("No closed trades yet. Close a trade below to begin tracking performance.")
        except Exception as e:
            st.error(f"❌ Could not load stats: {e}")
    else:
        st.info("🔌 Connect to PostgreSQL in the sidebar to view stats.")

    # ── Close Trade Form ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ✅ Close Trade — Record Outcome")

    if st.session_state.db_ok:
        try:
            open_trades = load_open_trades(db_cfg)
            if open_trades:
                trade_labels = {
                    f"#{t['id']} · {t['instrument']} {t['direction']} · logged {str(t['logged_at'])[:16]}": t["id"]
                    for t in open_trades
                }
                chosen_label = st.selectbox("Select open trade to close:", list(trade_labels.keys()),
                                            key="close_trade_select")
                chosen_id = trade_labels[chosen_label]
                chosen = next(t for t in open_trades if t["id"] == chosen_id)

                cf1, cf2, cf3 = st.columns(3)
                with cf1:
                    ct_entry = st.number_input("Entry Price", value=0.0, format="%.5f", step=0.00001,
                                               key="ct_entry")
                with cf2:
                    ct_close = st.number_input("Close Price", value=0.0, format="%.5f", step=0.00001,
                                               key="ct_close")
                with cf3:
                    ct_pips = st.number_input("Pips Gained (−ve = loss)", value=0.0, step=0.1,
                                              key="ct_pips")

                cf4, cf5 = st.columns(2)
                with cf4:
                    ct_outcome = st.selectbox("Outcome", ["WIN", "LOSS", "BE"], key="ct_outcome")
                with cf5:
                    sl_p = chosen.get("sl_pips") or 1.0
                    ct_r = round(ct_pips / sl_p, 2) if sl_p else 0.0
                    st.markdown(
                        f'<div class="metric-box" style="margin-top:8px;">'
                        f'<div class="metric-value" style="color:{"#3fb950" if ct_r >= 0 else "#f85149"};">'
                        f'{ct_r:+.2f}R</div><div class="metric-label">R Multiple</div></div>',
                        unsafe_allow_html=True)

                if st.button("💾 Save Trade Outcome", type="primary", use_container_width=True,
                             key="save_outcome_btn"):
                    close_trade(db_cfg, chosen_id, ct_entry, ct_close, ct_pips, ct_r, ct_outcome)
                    st.success(f"✅ Trade #{chosen_id} closed — {ct_outcome} · {ct_r:+.2f}R · {ct_pips:+.1f} pips")
                    st.rerun()
            else:
                st.info("No open trades to close. Save a trade setup above first.")
        except Exception as e:
            st.error(f"❌ Could not load open trades: {e}")
    else:
        st.info("🔌 Connect to PostgreSQL in the sidebar to close trades.")

    # ── Trade Log Table ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📚 Trade Log — PostgreSQL")

    if st.session_state.db_ok:
        try:
            trades = load_trades(db_cfg, limit=100)
            if trades:
                df_log = pd.DataFrame(trades)
                df_log["logged_at"] = pd.to_datetime(df_log["logged_at"]).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(df_log, use_container_width=True, column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "logged_at": st.column_config.TextColumn("Logged", width="medium"),
                    "instrument": st.column_config.TextColumn("Pair"),
                    "ticker": st.column_config.TextColumn("Ticker"),
                    "direction": st.column_config.TextColumn("Dir", width="small"),
                    "session": st.column_config.TextColumn("Session"),
                    "score": st.column_config.TextColumn("Score"),
                    "verdict": st.column_config.TextColumn("Verdict"),
                    "atr14": st.column_config.NumberColumn("ATR14", format="%.5f"),
                    "atr20": st.column_config.NumberColumn("ATR20", format="%.5f"),
                    "sl_pips": st.column_config.NumberColumn("SL pips", format="%.1f"),
                    "tp1_pips": st.column_config.NumberColumn("TP1 pips", format="%.1f"),
                    "tp2_pips": st.column_config.NumberColumn("TP2 pips", format="%.1f"),
                    "lot_size": st.column_config.NumberColumn("Lots", format="%.2f"),
                    "risk_amount": st.column_config.NumberColumn("Risk $", format="%.2f"),
                    "rr_tp1": st.column_config.NumberColumn("R:R TP1", format="%.2f"),
                    "account_bal": st.column_config.NumberColumn("Balance", format="%.2f"),
                    "risk_pct": st.column_config.NumberColumn("Risk %", format="%.2f"),
                    "checks_passed": st.column_config.NumberColumn("✅ Checks"),
                    "notes": st.column_config.TextColumn("Notes", width="large"),
                })
            else:
                st.info("No trade setups saved yet. Complete the checklist and hit Save.")
        except Exception as e:
            st.error(f"❌ Could not load trades: {e}")
    else:
        st.info("🔌 Connect to PostgreSQL in the sidebar to view saved trades.")


else:
    # ──────────────────────────────────────────────────────────────
    # TREND FOLLOWING SIGNAL PAGE
    # ──────────────────────────────────────────────────────────────

    st.title("📈 Trend Following Signal Dashboard")
    st.caption(
        "**Strategy:** 50 EMA / 200 EMA crossover · RSI (14) · MACD · ADX  |  "
        f"**Timeframe:** {st.session_state.trend_timeframe}  |  "
        f"**Sensitivity:** {st.session_state.trend_min_conds}/6 conditions required"
    )
    st.divider()

    selected_pairs = st.session_state.trend_selected_pairs

    if not selected_pairs:
        st.warning("⚠️ Please select at least one market in the sidebar.")
        st.stop()

    interval, period, resample = TREND_TIMEFRAMES[st.session_state.trend_timeframe]
    pair_results = {}

    progress = st.progress(0, text="Fetching market data…")
    for idx, pair in enumerate(selected_pairs):
        ticker = INSTRUMENTS[pair]["ticker"]
        with st.spinner(f"Loading {pair}…"):
            df = fetch_trend_data(ticker, interval, period, resample)
        progress.progress((idx + 1) / len(selected_pairs), text=f"Processed {pair}")

        if df is None or df.empty:
            pair_results[pair] = {"error": True}
            continue

        signal, score, max_score, conds, direction = evaluate_trend_signal(df, st.session_state.trend_min_conds)
        last = df.iloc[-1]

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
        }
    progress.empty()

    # Signal Cards Grid
    st.subheader("🎯 Live Signals")
    cols = st.columns(min(len(selected_pairs), 4))

    for idx, pair in enumerate(selected_pairs):
        r = pair_results.get(pair, {})
        with cols[idx % 4]:
            if r.get("error"):
                st.error(f"❌ {pair}\nFailed to load")
                continue

            direction = r["direction"]
            css_cls = {
                "STRONG_BUY": "trend-card-strong-buy",
                "BUY": "trend-card-buy",
                "STRONG_SELL": "trend-card-strong-sell",
                "SELL": "trend-card-sell"
            }.get(direction, "trend-card-neutral")

            pct_bar = int(r["score"] / r["max_score"] * 100)

            tag = '<span class="commodity-tag">COMMODITY</span>' if pair in TREND_COMMODITIES else '<span class="forex-tag">FOREX</span>'

            st.markdown(f"""
            <div class="{css_cls}">
                <div class="trend-pair-name">{pair} {tag}</div>
                <div class="trend-sig-text">{r["signal"]}</div>
                <div class="trend-price-text">Price: {r["close"]:.5f}</div>
                <div class="score-text" style="font-size:0.7rem;opacity:0.85;">✦ Conditions: {r["score"]}/{r["max_score"]}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(pct_bar, text=f"Signal strength {pct_bar}%")

    # Detailed Chart View
    st.divider()
    st.subheader("📊 Detailed Chart & Analysis")

    valid_pairs = [p for p in selected_pairs if not pair_results.get(p, {}).get("error")]
    if not valid_pairs:
        st.error("No valid data available. Try refreshing.")
        st.stop()

    default_pair = next(
        (p for p in valid_pairs if pair_results[p]["direction"] != "NEUTRAL"),
        valid_pairs[0],
    )
    selected_detail = st.selectbox("Choose pair to inspect:", valid_pairs,
                                   index=valid_pairs.index(default_pair))

    r = pair_results[selected_detail]

    # Key metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Close", f"{r['close']:.5f}")
    m2.metric("50 EMA", f"{r['ema50']:.5f}",
              delta=f"{((r['close'] - r['ema50']) / r['ema50'] * 100):+.2f}%")
    m3.metric("200 EMA", f"{r['ema200']:.5f}",
              delta=f"{((r['close'] - r['ema200']) / r['ema200'] * 100):+.2f}%")
    m4.metric("RSI (14)", f"{r['rsi']:.1f}",
              delta="Overbought" if r['rsi'] > 70 else ("Oversold" if r['rsi'] < 30 else "Neutral"))
    m5.metric("MACD", f"{r['macd']:.5f}")
    m6.metric("ADX", f"{r['adx']:.1f}",
              delta="Trending" if r['adx'] > 25 else "Ranging")

    # Chart
    st.plotly_chart(build_trend_chart(r["df"], selected_detail), use_container_width=True)

    # Conditions breakdown
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

    # Strategy Reference
    with st.expander("📖 Strategy Reference — Trend Following Rules"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🟢 BUY Setup
            - Price **above** 200 EMA
            - 50 EMA **above** 200 EMA (Golden Cross)
            - Price pulling back toward / holding above 50 EMA
            - RSI between **45–70** (bullish zone)
            - MACD line **above** signal line
            - ADX **> 25** (trend strength confirmed)
            """)
        with col2:
            st.markdown("""
            ### 🔴 SELL Setup
            - Price **below** 200 EMA
            - 50 EMA **below** 200 EMA (Death Cross)
            - Price pulling back toward / below 50 EMA
            - RSI between **30–55** (bearish zone)
            - MACD line **below** signal line
            - ADX **> 25** (trend strength confirmed)
            """)

        st.markdown("""
        ### ⚙️ Recommended Settings
        | Parameter | Value |
        |-----------|-------|
        | Timeframe | 4H or Daily |
        | EMA Fast  | 50 periods |
        | EMA Slow  | 200 periods |
        | RSI       | 14 periods |
        | MACD      | 12 / 26 / 9 |
        | ADX       | 14 periods |
        """)

    # Alert History
    if "trend_alert_history" not in st.session_state:
        st.session_state.trend_alert_history = []

    # Check for new alerts
    new_alerts = []
    for pair, r in pair_results.items():
        if not r.get("error") and r["direction"] in ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"]:
            new_alerts.append((pair, r["signal"], r["direction"], r["close"]))

    if new_alerts and st.session_state.get("trend_last_alert_count", 0) != len(new_alerts):
        for pair, signal, direction, price in new_alerts:
            st.session_state.trend_alert_history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
                "pair": pair,
                "signal": signal,
                "price": f"{price:.5f}",
            })
        st.session_state.trend_alert_history = st.session_state.trend_alert_history[-50:]
        st.session_state.trend_last_alert_count = len(new_alerts)

    st.divider()
    with st.expander(f"🔔 Alert History ({len(st.session_state.trend_alert_history)} alerts this session)"):
        if st.session_state.trend_alert_history:
            df_hist = pd.DataFrame(st.session_state.trend_alert_history[::-1])
            st.dataframe(df_hist, hide_index=True, use_container_width=True)
            if st.button("🗑️ Clear History", key="clear_trend_history"):
                st.session_state.trend_alert_history = []
                st.rerun()
        else:
            st.info("No alerts triggered yet this session.")

# Footer
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid #21262d;">
  📈 Daily Trading System · 18-Point Confluence Checklist · Trend Following Signals · For educational purposes only
</div>
""", unsafe_allow_html=True)