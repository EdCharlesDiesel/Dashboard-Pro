import streamlit as st
from src.pages_lib.navigation import render_sidebar_nav
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="15M Entry Signal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — matches main app dark theme ─────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    /* Theme-adaptive layout vars */
    .stApp {
      --border: color-mix(in srgb, var(--text-color) 12%, transparent);
      --muted:  color-mix(in srgb, var(--text-color) 55%, transparent);
    }
    html,body,[class*="css"]{ font-family:'Inter',sans-serif; }
    .stApp { background:var(--background-color); }
    section[data-testid="stSidebar"]{ background:var(--secondary-background-color)!important; border-right:1px solid var(--border,#21262d); }

    .card{ background:var(--secondary-background-color); border:1px solid var(--border,#21262d); border-radius:12px; padding:20px; margin-bottom:16px; }
    .card-header{ font-size:13px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--muted,#8b949e); margin-bottom:14px; }
    .metric-box{ background:var(--background-color); border:1px solid var(--border,#21262d); border-radius:8px; padding:14px; text-align:center; }
    .metric-value{ font-size:22px; font-weight:700; color:var(--text-color); }
    .metric-label{ font-size:11px; color:var(--muted,#8b949e); margin-top:2px; font-weight:500; letter-spacing:.04em; text-transform:uppercase; }
    .section-title{ font-size:16px; font-weight:700; color:var(--text-color); margin:24px 0 12px 0;
                    padding-left:4px; border-left:3px solid #388bfd; }
    .prog-track{ background:var(--border,#21262d); border-radius:8px; height:10px; margin:6px 0 2px 0; overflow:hidden; }
    #MainMenu,footer,header{ visibility:hidden; }
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    .block-container{ padding-top:1.5rem; max-width:1380px; }

    /* Signal verdict banners */
    .verdict-long { background:linear-gradient(135deg,#0d3a1f,#0d5e32); border:2px solid #238636;
                    border-radius:14px; padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-short{ background:linear-gradient(135deg,#3a0d0d,#5e1414); border:2px solid #8b2d2d;
                    border-radius:14px; padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-wait { background:linear-gradient(135deg,#1c1c0d,#2e2a0d); border:2px solid #9e6a03;
                    border-radius:14px; padding:22px 28px; text-align:center; margin-bottom:18px; }
    .verdict-none { background:linear-gradient(135deg,#161b22,#1c2128); border:1px solid #30363d;
                    border-radius:14px; padding:22px 28px; text-align:center; margin-bottom:18px; }

    /* Condition checklist rows */
    .cond-row{ display:flex; align-items:flex-start; gap:10px; padding:10px 14px;
               border-bottom:1px solid #21262d; font-size:13px; }
    .cond-row:last-child{ border-bottom:none; }
    .cond-icon{ font-size:16px; flex-shrink:0; margin-top:1px; }
    .cond-text{ color:#c9d1d9; }
    .cond-sub { font-size:11px; color:#8b949e; margin-top:2px; font-family:monospace; }

    /* Signal log table pills */
    .pill-long { display:inline-block; background:#0d4a2f; color:#3fb950; border:1px solid #238636;
                 border-radius:20px; padding:2px 10px; font-size:11px; font-weight:700; }
    .pill-short{ display:inline-block; background:#4a0d0d; color:#f85149; border:1px solid #8b2d2d;
                 border-radius:20px; padding:2px 10px; font-size:11px; font-weight:700; }

    /* Explanation box */
    .explainer{ background:var(--background-color); border:1px solid #1e3a5f; border-left:3px solid #388bfd;
                border-radius:8px; padding:14px 18px; margin:8px 0; font-size:13px; color:var(--muted,#8b949e); line-height:1.7; }
    [data-testid="stSidebarNav"]{ display:none; }
</style>
""", unsafe_allow_html=True)

# ── Instrument registry ────────────────────────────────────────────
INSTRUMENTS = {
    "EUR/USD": {"ticker": "EURUSD=X", "pip_size": 0.0001},
    "GBP/USD": {"ticker": "GBPUSD=X", "pip_size": 0.0001},
    "AUD/USD": {"ticker": "AUDUSD=X", "pip_size": 0.0001},
    "NZD/USD": {"ticker": "NZDUSD=X", "pip_size": 0.0001},
    "USD/JPY": {"ticker": "USDJPY=X", "pip_size": 0.01},
    "USD/CHF": {"ticker": "USDCHF=X", "pip_size": 0.0001},
    "USD/CAD": {"ticker": "USDCAD=X", "pip_size": 0.0001},
    "EUR/GBP": {"ticker": "EURGBP=X", "pip_size": 0.0001},
    "EUR/JPY": {"ticker": "EURJPY=X", "pip_size": 0.01},
    "GBP/JPY": {"ticker": "GBPJPY=X", "pip_size": 0.01},
    "AUD/JPY": {"ticker": "AUDJPY=X", "pip_size": 0.01},
    "EUR/AUD": {"ticker": "EURAUD=X", "pip_size": 0.0001},
    "GBP/AUD": {"ticker": "GBPAUD=X", "pip_size": 0.0001},
    "EUR/CAD": {"ticker": "EURCAD=X", "pip_size": 0.0001},
    "GBP/CAD": {"ticker": "GBPCAD=X", "pip_size": 0.0001},
    "USD/ZAR": {"ticker": "USDZAR=X", "pip_size": 0.0001},
    "EUR/ZAR": {"ticker": "EURZAR=X", "pip_size": 0.0001},
    "GBP/ZAR": {"ticker": "GBPZAR=X", "pip_size": 0.0001},
    "🥇 Gold": {"ticker": "GC=F", "pip_size": 0.10},
    "🥈 Silver": {"ticker": "SI=F", "pip_size": 0.01},
    "🪙 Platinum": {"ticker": "PL=F", "pip_size": 0.10},
}


# ══════════════════════════════════════════════════════════════════
# DATA & INDICATOR CALCULATIONS
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=180, show_spinner=False)
def fetch_15m(ticker: str, days: int = 5) -> pd.DataFrame:
    try:
        df = yf.download(ticker, interval="15m", period=f"{days}d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def calc_stochastic(df: pd.DataFrame, k_period: int = 14,
                    d_period: int = 3, smooth_k: int = 3) -> tuple[pd.Series, pd.Series]:
    """Full Stochastic Oscillator — smoothed %K and %D."""
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    raw_k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    pct_k = raw_k.rolling(smooth_k).mean()  # smoothed %K
    pct_d = pct_k.rolling(d_period).mean()  # %D signal line
    return pct_k, pct_d


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def detect_signals(df: pd.DataFrame,
                   stoch_os: float = 20.0, stoch_ob: float = 80.0,
                   rsi_os: float = 40.0, rsi_ob: float = 60.0,
                   rsi_reset_band: float = 5.0,
                   vol_period: int = 20, vol_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Detect LONG / SHORT entry signals:

    LONG  — Stochastic %K crosses above %D in oversold zone (< stoch_os)
             AND RSI was oversold / reset (crossed back above rsi_os from below)
    SHORT — Stochastic %K crosses below %D in overbought zone (> stoch_ob)
             AND RSI was overbought / reset (crossed back below rsi_ob from above)

    Volume spike column added: volume > vol_multiplier × vol_period MA
    """
    df = df.copy()
    K, D = calc_stochastic(df)
    rsi_vals = calc_rsi(df["Close"])

    df["stoch_k"] = K
    df["stoch_d"] = D
    df["rsi"] = rsi_vals

    # Volume spike — vectorized
    vol_raw = df["Volume"].values.astype(float)
    vol_ma_arr = pd.Series(vol_raw).rolling(vol_period, min_periods=1).mean().values
    with np.errstate(invalid="ignore"):
        vol_spike_arr = np.where(vol_ma_arr > 0, vol_raw > vol_multiplier * vol_ma_arr, False)
    df["vol_spike"] = vol_spike_arr.astype(bool)
    df["vol_ma"] = pd.Series(vol_ma_arr, index=df.index)

    n = len(df)

    # Stochastic crossovers
    k = K.values
    d = D.values
    r = rsi_vals.values

    stoch_cross_up = np.zeros(n, bool)  # K crosses D upward
    stoch_cross_down = np.zeros(n, bool)  # K crosses D downward
    rsi_reset_up = np.zeros(n, bool)  # RSI rising from oversold into neutral
    rsi_reset_down = np.zeros(n, bool)  # RSI falling from overbought into neutral
    signal_long = np.zeros(n, bool)
    signal_short = np.zeros(n, bool)

    for i in range(1, n):
        if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(r[i]):
            continue

        # Stoch crossovers
        if k[i] > d[i] and k[i - 1] <= d[i - 1]:
            stoch_cross_up[i] = True
        if k[i] < d[i] and k[i - 1] >= d[i - 1]:
            stoch_cross_down[i] = True

        # RSI resets — look back up to 4 candles for a recent reset
        lookback = min(i, 4)
        prev_r = r[i - lookback: i]

        rsi_was_oversold = any(v < rsi_os for v in prev_r if not np.isnan(v))
        rsi_was_overbought = any(v > rsi_ob for v in prev_r if not np.isnan(v))

        # RSI reset up: was in oversold, now recovering (within reset band above os level)
        if rsi_was_oversold and rsi_os <= r[i] <= rsi_os + rsi_reset_band * 3:
            rsi_reset_up[i] = True

        # RSI reset down: was in overbought, now retreating
        if rsi_was_overbought and rsi_ob - rsi_reset_band * 3 <= r[i] <= rsi_ob:
            rsi_reset_down[i] = True

        # Combined entry conditions
        # LONG: Stoch crosses up in/near oversold zone + RSI resetting from low
        stoch_in_os = k[i] < (stoch_os + 20)  # within 20pts of oversold zone
        if stoch_cross_up[i] and stoch_in_os and rsi_reset_up[i]:
            signal_long[i] = True

        # SHORT: Stoch crosses down in/near overbought zone + RSI resetting from high
        stoch_in_ob = k[i] > (stoch_ob - 20)
        if stoch_cross_down[i] and stoch_in_ob and rsi_reset_down[i]:
            signal_short[i] = True

    df["stoch_cross_up"] = stoch_cross_up
    df["stoch_cross_down"] = stoch_cross_down
    df["rsi_reset_up"] = rsi_reset_up
    df["rsi_reset_down"] = rsi_reset_down
    df["signal_long"] = signal_long
    df["signal_short"] = signal_short

    return df


def latest_signal_status(df: pd.DataFrame,
                         stoch_os: float, stoch_ob: float,
                         rsi_os: float, rsi_ob: float,
                         vol_multiplier: float = 1.5) -> dict:
    """Analyse the last 5 candles for a live entry signal."""
    tail = df.tail(5)
    last = df.iloc[-1]

    k_now = float(last["stoch_k"])
    d_now = float(last["stoch_d"])
    rsi_now = float(last["rsi"])
    close = float(last["Close"])

    recent_long = int(tail["signal_long"].sum())
    recent_short = int(tail["signal_short"].sum())

    # Volume spike checks on recent candles
    has_vol = "vol_spike" in tail.columns
    bullish_vol_spike = False
    bearish_vol_spike = False
    vol_spike_now = False
    if has_vol:
        spikes = tail[tail["vol_spike"]]
        bullish_vol_spike = bool(len(spikes[spikes["Close"] >= spikes["Open"]]) > 0)
        bearish_vol_spike = bool(len(spikes[spikes["Close"] < spikes["Open"]]) > 0)
        vol_spike_now = bool(last.get("vol_spike", False))

    # Per-condition status for display
    cond_long = {
        "Stoch %K crossed above %D": bool(tail["stoch_cross_up"].any()),
        f"Stoch in oversold zone (K < {stoch_os + 20:.0f})": k_now < (stoch_os + 20),
        f"RSI reset from oversold (RSI ≥ {rsi_os:.0f})": bool(tail["rsi_reset_up"].any()),
        f"Volume spike (>{vol_multiplier:.1f}× avg) on up candle": bullish_vol_spike,
    }
    cond_short = {
        "Stoch %K crossed below %D": bool(tail["stoch_cross_down"].any()),
        f"Stoch in overbought zone (K > {stoch_ob - 20:.0f})": k_now > (stoch_ob - 20),
        f"RSI reset from overbought (RSI ≤ {rsi_ob:.0f})": bool(tail["rsi_reset_down"].any()),
        f"Volume spike (>{vol_multiplier:.1f}× avg) on down candle": bearish_vol_spike,
    }

    long_score = sum(cond_long.values())
    short_score = sum(cond_short.values())

    if recent_long and long_score == 4:
        verdict = "LONG_CONFIRMED"
    elif recent_short and short_score == 4:
        verdict = "SHORT_CONFIRMED"
    elif recent_long and long_score >= 3:
        verdict = "LONG"
    elif recent_short and short_score >= 3:
        verdict = "SHORT"
    elif long_score >= 3:
        verdict = "WAIT_LONG"
    elif short_score >= 3:
        verdict = "WAIT_SHORT"
    else:
        verdict = "NONE"

    return {
        "verdict": verdict,
        "k_now": k_now,
        "d_now": d_now,
        "rsi_now": rsi_now,
        "close": close,
        "last_time": df.index[-1],
        "cond_long": cond_long,
        "cond_short": cond_short,
        "long_score": long_score,
        "short_score": short_score,
        "recent_long": recent_long,
        "recent_short": recent_short,
        "vol_spike_now": vol_spike_now,
    }


# ══════════════════════════════════════════════════════════════════
# CHART BUILDER
# ══════════════════════════════════════════════════════════════════

def build_chart(df: pd.DataFrame, pair: str,
                stoch_os: float, stoch_ob: float,
                rsi_os: float, rsi_ob: float,
                show_n: int = 80, vol_period: int = 20) -> go.Figure:
    plot = df.tail(show_n).copy()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.44, 0.19, 0.19, 0.18],
        vertical_spacing=0.02,
        subplot_titles=[
            f"{pair} — 15M Price",
            "Stochastic Oscillator (%K / %D)",
            "RSI (14)",
            "Volume",
        ],
    )

    # ── Row 1: Candlesticks ────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=plot.index,
        open=plot["Open"], high=plot["High"],
        low=plot["Low"], close=plot["Close"],
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="rgba(63,185,80,0.20)", decreasing_fillcolor="rgba(248,81,73,0.20)",
        name="Price", showlegend=False,
    ), row=1, col=1)

    # Long entry arrows
    long_pts = plot[plot["signal_long"]]
    if not long_pts.empty:
        fig.add_trace(go.Scatter(
            x=long_pts.index,
            y=long_pts["Low"] - (long_pts["High"] - long_pts["Low"]) * 0.4,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=18, color="#3fb950",
                        line=dict(color="#0d1117", width=1)),
            text=["⚡ LONG"] * len(long_pts),
            textposition="bottom center",
            textfont=dict(size=9, color="#3fb950"),
            name="LONG Signal",
        ), row=1, col=1)

    # Short entry arrows
    short_pts = plot[plot["signal_short"]]
    if not short_pts.empty:
        fig.add_trace(go.Scatter(
            x=short_pts.index,
            y=short_pts["High"] + (short_pts["High"] - short_pts["Low"]) * 0.4,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=18, color="#f85149",
                        line=dict(color="#0d1117", width=1)),
            text=["⚡ SHORT"] * len(short_pts),
            textposition="top center",
            textfont=dict(size=9, color="#f85149"),
            name="SHORT Signal",
        ), row=1, col=1)

    # Stoch cross-up markers on price (subtle dots)
    cross_up_pts = plot[plot["stoch_cross_up"] & ~plot["signal_long"]]
    if not cross_up_pts.empty:
        fig.add_trace(go.Scatter(
            x=cross_up_pts.index,
            y=cross_up_pts["Low"] - (cross_up_pts["High"] - cross_up_pts["Low"]) * 0.2,
            mode="markers",
            marker=dict(symbol="circle", size=6, color="rgba(0,212,255,0.27)",
                        line=dict(color="#00d4ff", width=1)),
            name="Stoch Cross ↑ (partial)", showlegend=True,
        ), row=1, col=1)

    cross_dn_pts = plot[plot["stoch_cross_down"] & ~plot["signal_short"]]
    if not cross_dn_pts.empty:
        fig.add_trace(go.Scatter(
            x=cross_dn_pts.index,
            y=cross_dn_pts["High"] + (cross_dn_pts["High"] - cross_dn_pts["Low"]) * 0.2,
            mode="markers",
            marker=dict(symbol="circle", size=6, color="rgba(253,203,110,0.27)",
                        line=dict(color="#fdcb6e", width=1)),
            name="Stoch Cross ↓ (partial)", showlegend=True,
        ), row=1, col=1)

    # ── Row 2: Stochastic ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=plot.index, y=plot["stoch_k"],
        mode="lines", line=dict(color="#00d4ff", width=2),
        name="%K",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=plot.index, y=plot["stoch_d"],
        mode="lines", line=dict(color="#fdcb6e", width=1.5, dash="dot"),
        name="%D",
    ), row=2, col=1)

    # Oversold / overbought bands
    fig.add_hrect(y0=0, y1=stoch_os, fillcolor="rgba(63,185,80,0.07)",
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=stoch_ob, y1=100, fillcolor="rgba(248,81,73,0.07)",
                  line_width=0, row=2, col=1)
    fig.add_hline(y=stoch_os, line_color="rgba(63,185,80,0.40)", line_dash="dash",
                  line_width=1, row=2, col=1,
                  annotation_text=f"OS {stoch_os:.0f}",
                  annotation_font=dict(size=9, color="#3fb950"),
                  annotation_position="right")
    fig.add_hline(y=stoch_ob, line_color="rgba(248,81,73,0.40)", line_dash="dash",
                  line_width=1, row=2, col=1,
                  annotation_text=f"OB {stoch_ob:.0f}",
                  annotation_font=dict(size=9, color="#f85149"),
                  annotation_position="right")

    # Mark stoch crossover points on the stoch panel
    if not long_pts.empty:
        fig.add_trace(go.Scatter(
            x=long_pts.index, y=plot.loc[long_pts.index, "stoch_k"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="#3fb950"),
            name="Cross↑ (stoch)", showlegend=False,
        ), row=2, col=1)
    if not short_pts.empty:
        fig.add_trace(go.Scatter(
            x=short_pts.index, y=plot.loc[short_pts.index, "stoch_k"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#f85149"),
            name="Cross↓ (stoch)", showlegend=False,
        ), row=2, col=1)

    # ── Row 3: RSI ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=plot.index, y=plot["rsi"],
        mode="lines", line=dict(color="#a29bfe", width=2),
        name="RSI (14)",
    ), row=3, col=1)

    # RSI bands
    fig.add_hrect(y0=0, y1=rsi_os, fillcolor="rgba(63,185,80,0.07)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=rsi_ob, y1=100, fillcolor="rgba(248,81,73,0.07)", line_width=0, row=3, col=1)
    fig.add_hline(y=rsi_os, line_color="rgba(63,185,80,0.33)", line_dash="dash", line_width=1,
                  row=3, col=1,
                  annotation_text=f"OS {rsi_os:.0f}",
                  annotation_font=dict(size=9, color="#3fb950"),
                  annotation_position="right")
    fig.add_hline(y=rsi_ob, line_color="rgba(248,81,73,0.33)", line_dash="dash", line_width=1,
                  row=3, col=1,
                  annotation_text=f"OB {rsi_ob:.0f}",
                  annotation_font=dict(size=9, color="#f85149"),
                  annotation_position="right")
    fig.add_hline(y=50, line_color="rgba(255,255,255,0.13)", line_dash="dot", line_width=1, row=3, col=1)

    # RSI reset markers
    reset_up_pts = plot[plot["rsi_reset_up"]]
    reset_dn_pts = plot[plot["rsi_reset_down"]]
    if not reset_up_pts.empty:
        fig.add_trace(go.Scatter(
            x=reset_up_pts.index, y=plot.loc[reset_up_pts.index, "rsi"],
            mode="markers",
            marker=dict(symbol="circle-open", size=10, color="#3fb950",
                        line=dict(width=2, color="#3fb950")),
            name="RSI Reset ↑", showlegend=True,
        ), row=3, col=1)
    if not reset_dn_pts.empty:
        fig.add_trace(go.Scatter(
            x=reset_dn_pts.index, y=plot.loc[reset_dn_pts.index, "rsi"],
            mode="markers",
            marker=dict(symbol="circle-open", size=10, color="#f85149",
                        line=dict(width=2, color="#f85149")),
            name="RSI Reset ↓", showlegend=True,
        ), row=3, col=1)

    # ── Row 4: Volume ─────────────────────────────────────────────
    if "vol_spike" in plot.columns:
        is_bull = plot["Close"].values >= plot["Open"].values
        is_spk  = plot["vol_spike"].values.astype(bool)
        vol_colors = [
            "#3fb950" if bull and spk else
            "#f85149" if (not bull) and spk else
            "rgba(63,185,80,0.30)" if bull else
            "rgba(248,81,73,0.30)"
            for bull, spk in zip(is_bull, is_spk)
        ]
        fig.add_trace(go.Bar(
            x=plot.index, y=plot["Volume"],
            marker_color=vol_colors,
            name="Volume", showlegend=False,
        ), row=4, col=1)
        if "vol_ma" in plot.columns:
            fig.add_trace(go.Scatter(
                x=plot.index, y=plot["vol_ma"],
                mode="lines", line=dict(color="#fdcb6e", width=1.2, dash="dot"),
                name=f"Vol MA({vol_period})",
            ), row=4, col=1)

    # Global layout
    fig.update_layout(
        height=780,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", size=11, color="#8b949e"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(13,17,23,0.60)", bordercolor="#21262d", borderwidth=1,
            font=dict(size=10), orientation="h",
            yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        margin=dict(l=10, r=60, t=10, b=10),
    )
    fig.update_yaxes(gridcolor="#21262d", zeroline=False)
    fig.update_xaxes(gridcolor="#21262d", showgrid=False)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚡ 15M Entry Signal")
    render_sidebar_nav()
    st.divider()

    inst_keys = list(INSTRUMENTS.keys())
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]

    selected_pair = st.selectbox("Instrument", inst_keys,
                                 index=inst_keys.index(default_inst))

    lookback_days = st.slider("Data lookback (days)", 1, 7, 3)
    show_candles = st.slider("Candles on chart", 30, 200, 80, step=10)

    st.divider()
    st.markdown("**📐 Stochastic Settings**")
    k_period = st.number_input("%K Period", value=14, min_value=3, max_value=50, step=1)
    d_period = st.number_input("%D Smooth", value=3, min_value=1, max_value=10, step=1)
    smooth_k = st.number_input("%K Smooth", value=3, min_value=1, max_value=10, step=1)
    stoch_os = st.slider("Oversold threshold", 5, 40, 20)
    stoch_ob = st.slider("Overbought threshold", 60, 95, 80)

    st.divider()
    st.markdown("**📐 RSI Settings**")
    rsi_os = st.slider("RSI oversold level", 20, 50, 40)
    rsi_ob = st.slider("RSI overbought level", 50, 80, 60)
    rsi_reset_band = st.slider("RSI reset band (pts)", 1, 15, 5)

    st.divider()
    st.markdown("**📊 Volume Spike Settings**")
    vol_period = st.slider("Vol MA period (bars)", 5, 50, 20)
    vol_multiplier = st.slider("Spike multiplier (×avg)", 1.1, 3.0, 1.5, step=0.1)

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} local")

    st.divider()
    st.markdown("""
    <div style='font-size:11px;color:#484f58;line-height:1.8;'>
    ⚡ <b style='color:#8b949e;'>LONG signal</b><br>
    &nbsp;&nbsp;%K crosses above %D<br>
    &nbsp;&nbsp;in/near oversold zone<br>
    &nbsp;&nbsp;+ RSI reset from low<br><br>
    ⚡ <b style='color:#8b949e;'>SHORT signal</b><br>
    &nbsp;&nbsp;%K crosses below %D<br>
    &nbsp;&nbsp;in/near overbought zone<br>
    &nbsp;&nbsp;+ RSI reset from high
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════

ticker = INSTRUMENTS[selected_pair]["ticker"]

# Header
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);
            border:1px solid #21262d; border-radius:16px; padding:24px 28px;
            margin-bottom:20px; position:relative; overflow:hidden;">
  <div style="font-size:24px; font-weight:700; color:#e6edf3;">
    ⚡ 15M Entry Signal Scanner
  </div>
  <div style="color:#8b949e; font-size:13px; margin-top:4px;">
    Stochastic %K/%D crossover + RSI reset · 15-minute chart · {selected_pair}
  </div>
  <div style="font-size:12px; color:#388bfd; margin-top:6px;">
    Check #13 — 15M entry signal fired · {datetime.now().strftime('%A %d %B %Y  |  %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)

VERDICT_CFG = {
    "LONG_CONFIRMED": ("verdict-long",  "⚡ LONG SIGNAL CONFIRMED (4/4)", "#3fb950",
                       "All 4 conditions met including volume spike — high-conviction LONG entry"),
    "SHORT_CONFIRMED":("verdict-short", "⚡ SHORT SIGNAL CONFIRMED (4/4)", "#f85149",
                       "All 4 conditions met including volume spike — high-conviction SHORT entry"),
    "LONG":           ("verdict-long",  "⚡ LONG ENTRY SIGNAL FIRED",     "#3fb950",
                       "3 of 4 conditions met — Stochastic crossover ↑ + RSI reset from oversold (no vol spike)"),
    "SHORT":          ("verdict-short", "⚡ SHORT ENTRY SIGNAL FIRED",    "#f85149",
                       "3 of 4 conditions met — Stochastic crossover ↓ + RSI reset from overbought (no vol spike)"),
    "WAIT_LONG":      ("verdict-wait",  "🟡 LONG SETUP FORMING",          "#e3b341",
                       "≥3 long conditions forming — wait for stochastic crossover confirmation"),
    "WAIT_SHORT":     ("verdict-wait",  "🟡 SHORT SETUP FORMING",         "#e3b341",
                       "≥3 short conditions forming — wait for stochastic crossover confirmation"),
    "NONE":           ("verdict-none",  "⏳ NO ENTRY SIGNAL",              "#8b949e",
                       "Conditions not yet aligned — monitor for stochastic crossover + RSI reset"),
}

SCAN_COLOR = {
    "LONG_CONFIRMED": "#3fb950", "SHORT_CONFIRMED": "#f85149",
    "LONG": "#3fb950", "SHORT": "#f85149",
    "WAIT_LONG": "#e3b341", "WAIT_SHORT": "#e3b341", "NONE": "#484f58",
}
SCAN_LABEL = {
    "LONG_CONFIRMED": "LONG ✓VOL", "SHORT_CONFIRMED": "SHORT ✓VOL",
    "LONG": "LONG", "SHORT": "SHORT",
    "WAIT_LONG": "WAIT ↑", "WAIT_SHORT": "WAIT ↓", "NONE": "NONE",
}


def render_conditions(title, color, conds, score):
    items_html = ""
    for cond_text, met in conds.items():
        icon  = "✅" if met else "❌"
        shade = "#c9d1d9" if met else "#484f58"
        items_html += (
            f'<div class="cond-row">'
            f'<div class="cond-icon">{icon}</div>'
            f'<div><div class="cond-text" style="color:{shade};">{cond_text}</div></div>'
            f'</div>'
        )
    max_score = max(len(conds), 1)
    prog_pct = int(score / max_score * 100)
    prog_col = color if score == max_score else "#e3b341" if score >= max_score - 1 else "#484f58"
    return (
        f'<div class="card">'
        f'<div class="card-header" style="color:{color};">{title}</div>'
        f'{items_html}'
        f'<div style="margin-top:12px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
        f'<span style="font-size:11px;color:#8b949e;">Conditions met</span>'
        f'<span style="font-size:11px;font-weight:700;color:{prog_col};">{score}/3</span>'
        f'</div>'
        f'<div class="prog-track">'
        f'<div style="background:{prog_col};width:{prog_pct}%;height:100%;border-radius:8px;"></div>'
        f'</div></div></div>'
    )


# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────
tab_scan, tab_detail = st.tabs(["📊 All-Pairs Scanner", "🔍 Pair Detail"])

# ══════════════════════════════════════════════
#  TAB 1 — All-Pairs Scanner
# ══════════════════════════════════════════════
with tab_scan:
    st.markdown(
        f"Scanning all 21 instruments for 15M entry signals "
        f"(Stoch {stoch_os}/{stoch_ob} · RSI {rsi_os}/{rsi_ob} · Vol >{vol_multiplier:.1f}×) …"
    )
    prog = st.progress(0)
    all_instruments = list(INSTRUMENTS.items())
    n_inst = len(all_instruments)
    scan_results: list[dict] = []

    for idx, (pair_name, info) in enumerate(all_instruments):
        prog.progress((idx + 1) / n_inst, text=f"Scanning {pair_name} …")
        try:
            df_s = fetch_15m(info["ticker"], 3)
            if df_s.empty or len(df_s) < 30:
                scan_results.append({"pair": pair_name, "ok": False})
                continue
            df_s = detect_signals(
                df_s,
                stoch_os=stoch_os, stoch_ob=stoch_ob,
                rsi_os=rsi_os, rsi_ob=rsi_ob,
                rsi_reset_band=rsi_reset_band,
                vol_period=vol_period, vol_multiplier=vol_multiplier,
            )
            st_s = latest_signal_status(df_s, stoch_os, stoch_ob, rsi_os, rsi_ob,
                                        vol_multiplier=vol_multiplier)
            scan_results.append({
                "pair":      pair_name,
                "ok":        True,
                "verdict":   st_s["verdict"],
                "k_now":     st_s["k_now"],
                "rsi_now":   st_s["rsi_now"],
                "close":     st_s["close"],
                "l_score":   st_s["long_score"],
                "s_score":   st_s["short_score"],
                "vol_spike": st_s.get("vol_spike_now", False),
            })
        except Exception:
            scan_results.append({"pair": pair_name, "ok": False})

    prog.empty()

    # ── Summary counts ──────────────────────────
    cnt_long     = sum(1 for r in scan_results if r.get("verdict") in ("LONG", "LONG_CONFIRMED"))
    cnt_short    = sum(1 for r in scan_results if r.get("verdict") in ("SHORT", "SHORT_CONFIRMED"))
    cnt_conf     = sum(1 for r in scan_results if r.get("verdict") in ("LONG_CONFIRMED", "SHORT_CONFIRMED"))
    cnt_wait     = sum(1 for r in scan_results if r.get("verdict") in ("WAIT_LONG", "WAIT_SHORT"))
    cnt_none     = sum(1 for r in scan_results if r.get("ok") and r.get("verdict") == "NONE")

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    for col_, val_, lbl_, c_ in [
        (sc1, cnt_long,  "LONG Signal",   "#3fb950"),
        (sc2, cnt_short, "SHORT Signal",  "#f85149"),
        (sc3, cnt_conf,  "Vol Confirmed", "#00d4ff"),
        (sc4, cnt_wait,  "Setup Forming", "#e3b341"),
        (sc5, cnt_none,  "No Signal",     "#484f58"),
    ]:
        with col_:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="color:{c_};font-size:28px;">{val_}</div>'
                f'<div class="metric-label">{lbl_}</div>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Card grid ───────────────────────────────
    cols3 = st.columns(3)
    for i, r in enumerate(scan_results):
        pair_name = r["pair"]
        is_sel    = pair_name == selected_pair
        border    = "border:2px solid #388bfd;" if is_sel else "border:1px solid #21262d;"

        if not r.get("ok"):
            card_body = '<span style="font-size:11px;color:#f85149;">No data</span>'
        else:
            vrd   = r["verdict"]
            color = SCAN_COLOR.get(vrd, "#484f58")
            label = SCAN_LABEL.get(vrd, vrd)
            card_body = (
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:13px;font-weight:600;color:#e6edf3;">{pair_name}</span>'
                f'<span style="background:{color}18;border:1px solid {color}55;border-radius:4px;'
                f'padding:2px 8px;font-size:11px;color:{color};font-weight:700;">{label}</span>'
                f'</div>'
                f'<div style="display:flex;gap:14px;margin-top:6px;">'
                f'<span style="font-size:11px;color:#8b949e;">%K <span style="color:#00d4ff;font-family:monospace;">{r["k_now"]:.1f}</span></span>'
                f'<span style="font-size:11px;color:#8b949e;">RSI <span style="color:#a29bfe;font-family:monospace;">{r["rsi_now"]:.1f}</span></span>'
                f'<span style="font-size:11px;color:#8b949e;font-family:monospace;">{r["close"]:,.5f}</span>'
                f'</div>'
            )

        with cols3[i % 3]:
            st.markdown(
                f'<div style="background:#161b22;{border}border-radius:8px;'
                f'padding:12px 14px;margin-bottom:8px;">{card_body}</div>',
                unsafe_allow_html=True)

    # ── Expander table ───────────────────────────
    with st.expander("📋 Full Scanner Results"):
        table_rows = []
        for r in scan_results:
            table_rows.append({
                "Pair":    r["pair"],
                "Signal":  SCAN_LABEL.get(r.get("verdict",""), "Error") if r.get("ok") else "Error",
                "%K":      f'{r["k_now"]:.1f}'   if r.get("ok") else "—",
                "RSI":     f'{r["rsi_now"]:.1f}' if r.get("ok") else "—",
                "L-Score": r.get("l_score", "—") if r.get("ok") else "—",
                "S-Score": r.get("s_score", "—") if r.get("ok") else "—",
                "Price":   f'{r["close"]:,.5f}'  if r.get("ok") else "—",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  TAB 2 — Pair Detail
# ══════════════════════════════════════════════
with tab_detail:
    detail_ok = True

    with st.spinner(f"Loading 15M data for {selected_pair}…"):
        df_raw = fetch_15m(ticker, lookback_days)

    if df_raw.empty or len(df_raw) < 30:
        st.error(f"⚠️ Could not load enough 15M data for **{selected_pair}**. "
                 "Try reducing lookback or a different instrument.")
        detail_ok = False

    if detail_ok:
        df = detect_signals(
            df_raw,
            stoch_os=stoch_os, stoch_ob=stoch_ob,
            rsi_os=rsi_os, rsi_ob=rsi_ob,
            rsi_reset_band=rsi_reset_band,
            vol_period=vol_period, vol_multiplier=vol_multiplier,
        )
        status = latest_signal_status(df, stoch_os, stoch_ob, rsi_os, rsi_ob,
                                      vol_multiplier=vol_multiplier)

        # ── Verdict banner ───────────────────────────
        vcls, vtitle, vcolor, vdesc = VERDICT_CFG[status["verdict"]]
        st.markdown(f"""
        <div class="{vcls}">
          <div style="font-size:22px; font-weight:800; color:{vcolor}; letter-spacing:1px;">{vtitle}</div>
          <div style="font-size:13px; color:#8b949e; margin:6px 0 10px 0;">{vdesc}</div>
          <div style="display:flex; gap:24px; justify-content:center; flex-wrap:wrap;
                      font-size:13px; color:#c9d1d9; margin-top:10px;">
            <div>%K &nbsp;<code style="color:#00d4ff;">{status['k_now']:.1f}</code></div>
            <div>%D &nbsp;<code style="color:#fdcb6e;">{status['d_now']:.1f}</code></div>
            <div>RSI &nbsp;<code style="color:#a29bfe;">{status['rsi_now']:.1f}</code></div>
            <div>Close &nbsp;<code style="color:#e6edf3;">{status['close']:.5f}</code></div>
            <div>⏰ &nbsp;<code style="color:#8b949e;">{status['last_time'].strftime('%H:%M')}</code></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI row ──────────────────────────────────
        total_long      = int(df["signal_long"].sum())
        total_short     = int(df["signal_short"].sum())
        total_cross_up  = int(df["stoch_cross_up"].sum())
        total_cross_dn  = int(df["stoch_cross_down"].sum())
        total_rsi_reset = int(df["rsi_reset_up"].sum() + df["rsi_reset_down"].sum())
        total_vol_spk   = int(df["vol_spike"].sum()) if "vol_spike" in df.columns else 0

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        for col_, val_, lbl_, color_ in [
            (k1, total_long,      "LONG Signals",  "#3fb950"),
            (k2, total_short,     "SHORT Signals", "#f85149"),
            (k3, total_cross_up,  "Stoch Cross ↑", "#00d4ff"),
            (k4, total_cross_dn,  "Stoch Cross ↓", "#fdcb6e"),
            (k5, total_rsi_reset, "RSI Resets",    "#a29bfe"),
            (k6, total_vol_spk,   "Vol Spikes",    "#58a6ff"),
        ]:
            with col_:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value" style="color:{color_};">{val_}</div>'
                    f'<div class="metric-label">{lbl_}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # ── Condition breakdown ──────────────────────
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(render_conditions(
                "🟢 LONG Entry Conditions", "#3fb950",
                status["cond_long"], status["long_score"],
            ), unsafe_allow_html=True)
        with col_right:
            st.markdown(render_conditions(
                "🔴 SHORT Entry Conditions", "#f85149",
                status["cond_short"], status["short_score"],
            ), unsafe_allow_html=True)

        # ── Chart ────────────────────────────────────
        st.markdown('<div class="section-title">📊 15M Chart — Stochastic + RSI</div>',
                    unsafe_allow_html=True)
        fig = build_chart(
            df, selected_pair,
            stoch_os=stoch_os, stoch_ob=stoch_ob,
            rsi_os=rsi_os, rsi_ob=rsi_ob,
            show_n=show_candles, vol_period=vol_period,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Signal log ───────────────────────────────
        st.markdown('<div class="section-title">🗂️ Signal Log — All Fired Signals</div>',
                    unsafe_allow_html=True)
        sig_rows = []
        for ts, row in df.iterrows():
            if row["signal_long"]:
                sig_rows.append({
                    "Time": ts.strftime("%m-%d %H:%M"), "Direction": "LONG",
                    "Close": round(float(row["Close"]), 5),
                    "Stoch %K": round(float(row["stoch_k"]), 1),
                    "Stoch %D": round(float(row["stoch_d"]), 1),
                    "RSI": round(float(row["rsi"]), 1),
                    "Stoch Cross": "↑ Up", "RSI Reset": "↑ From Low",
                })
            if row["signal_short"]:
                sig_rows.append({
                    "Time": ts.strftime("%m-%d %H:%M"), "Direction": "SHORT",
                    "Close": round(float(row["Close"]), 5),
                    "Stoch %K": round(float(row["stoch_k"]), 1),
                    "Stoch %D": round(float(row["stoch_d"]), 1),
                    "RSI": round(float(row["rsi"]), 1),
                    "Stoch Cross": "↓ Down", "RSI Reset": "↓ From High",
                })

        if sig_rows:
            st.dataframe(
                pd.DataFrame(sig_rows[::-1]),
                use_container_width=True, hide_index=True,
                column_config={
                    "Direction": st.column_config.TextColumn("Dir",   width="small"),
                    "Stoch %K":  st.column_config.NumberColumn("%K",  format="%.1f"),
                    "Stoch %D":  st.column_config.NumberColumn("%D",  format="%.1f"),
                    "RSI":       st.column_config.NumberColumn("RSI", format="%.1f"),
                })
        else:
            st.info("No complete entry signals detected in the loaded data window. "
                    "Try extending the lookback or adjusting the sensitivity thresholds in the sidebar.")

        st.markdown("---")

        # ── Strategy explainer ───────────────────────
        st.markdown('<div class="section-title">📖 Entry Signal Logic</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="explainer">
            <b style="color:#3fb950;">🟢 LONG Entry — all 3 required</b><br><br>
            <b>1. Stochastic Crossover ↑</b><br>
            &nbsp;&nbsp;%K line crosses <em>above</em> %D signal line.<br>
            &nbsp;&nbsp;Momentum is shifting from bearish to bullish.<br><br>
            <b>2. In / Near Oversold Zone</b><br>
            &nbsp;&nbsp;%K is below the OS threshold + 20 pts.<br>
            &nbsp;&nbsp;Confirms the crossover happens from a low base,<br>
            &nbsp;&nbsp;not from a mid-range noise cross.<br><br>
            <b>3. RSI Reset from Oversold</b><br>
            &nbsp;&nbsp;RSI was below the oversold level within the last 4 candles<br>
            &nbsp;&nbsp;and is now recovering back toward neutral.<br>
            &nbsp;&nbsp;Confirms momentum behind the stochastic cross.
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="explainer">
            <b style="color:#f85149;">🔴 SHORT Entry — all 3 required</b><br><br>
            <b>1. Stochastic Crossover ↓</b><br>
            &nbsp;&nbsp;%K line crosses <em>below</em> %D signal line.<br>
            &nbsp;&nbsp;Momentum is shifting from bullish to bearish.<br><br>
            <b>2. In / Near Overbought Zone</b><br>
            &nbsp;&nbsp;%K is above the OB threshold − 20 pts.<br>
            &nbsp;&nbsp;Confirms the cross happens at elevated levels,<br>
            &nbsp;&nbsp;not a mid-range false signal.<br><br>
            <b>3. RSI Reset from Overbought</b><br>
            &nbsp;&nbsp;RSI was above the overbought level within the last 4 candles<br>
            &nbsp;&nbsp;and is now retreating back toward neutral.<br>
            &nbsp;&nbsp;Confirms exhaustion of the prior move.
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="explainer" style="margin-top:12px;">
        <b style="color:#388bfd;">⚙️ Indicator Parameters</b><br><br>
        <b>Stochastic:</b> %K Period {int(k_period)} · Smooth %K {int(smooth_k)} · %D Signal {int(d_period)} &nbsp;
        (adjustable in sidebar)<br>
        <b>RSI:</b> Period 14 · Oversold &lt; {rsi_os} · Overbought &gt; {rsi_ob} &nbsp;
        (use 30/70 for stricter signals)<br>
        <b>Volume:</b> Spike when volume &gt; {vol_multiplier:.1f}× its {vol_period}-bar MA · shown on chart as bright bars<br>
        &nbsp;&nbsp;<em>CONFIRMED</em> (4/4) requires vol spike on same-direction candle within signal window.<br>
        <b>Best used with:</b> 4H confluence zone confirmed + candlestick rejection on 15M already identified.<br>
        This is <em>Check #13</em> — the final entry trigger after all higher-timeframe confluences align.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;
                    padding-top:16px;border-top:1px solid #21262d;">
          ⚡ 15M Entry Signal Scanner · Check #13 · Stochastic + RSI · For educational purposes only
        </div>
        """, unsafe_allow_html=True)
