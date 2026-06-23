"""VWAP-EMA Regime-Filtered Intraday Trading Framework for XAU/USD (Gold).

Harmonised to the Bloomberg terminal look (`BloombergTheme`) and wired into the
shared sidebar nav. Backtests a regime-filtered VWAP + EMA pullback strategy on
GC=F intraday bars.

Improvements over the original standalone script:
  • Terminal green-on-black theme + shared sidebar navigation (symbol/params
    first, per the sidebar contract).
  • Robust yfinance loading (MultiIndex flatten, cached, intraday-window guard)
    instead of the brittle fixed 6-column rename that breaks under
    auto_adjust=True.
  • Daily-anchored VWAP (resets each session) rather than a single cumulative
    line over the whole history.
  • A real hard stop at 1.5×ATR (repo risk model) so 1R is well-defined and
    losing trades are modelled — the original had no initial stop, leaving the
    R-multiple denominator arbitrary.
  • EMA columns keyed to the configured periods (the original hard-coded
    EMA_50/200 yet indexed f"EMA_{ema_short}" in the backtest — a KeyError the
    moment you changed the pullback period).
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import warnings

from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="VWAP-EMA Gold · Trading System",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

T = BloombergTheme  # short alias for palette tokens

# ── Plotly terminal styling ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=T.BG,
    font=dict(family="JetBrains Mono, monospace", color=T.GREY, size=11),
    legend=dict(font=dict(color=T.GREY, size=11), bgcolor=T.BG_ELEV,
                bordercolor=T.BORDER, orientation="h", x=0, y=1.06),
    margin=dict(l=10, r=10, t=30, b=10),
)


def style_axes(fig: go.Figure, rows: int) -> None:
    for r in range(1, rows + 1):
        fig.update_xaxes(showgrid=False, tickfont=dict(color=T.GREY, size=9),
                         linecolor=T.BORDER, row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=T.BORDER,
                         tickfont=dict(color=T.GREY, size=9), row=r, col=1)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — symbol/params first, then nav (sidebar contract)
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🟡 VWAP-EMA Gold")

    st.markdown("#### ⚙️ Strategy Parameters")

    st.caption("Risk Management")
    risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.0, 0.5) / 100
    initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)

    st.caption("Indicators")
    ema_pull = st.number_input("EMA Pullback Period", value=50, min_value=20, max_value=100)
    ema_regime = st.number_input("EMA Regime Period", value=200, min_value=100, max_value=500)
    ema_final = st.number_input("EMA Final-Leg Period", value=20, min_value=10, max_value=50)
    atr_period = st.number_input("ATR Period", value=14, min_value=5, max_value=30)
    atr_sl_mult = st.slider("Stop = ATR ×", 1.0, 3.0, 1.5, 0.25)

    st.caption("Targets")
    target_r = st.slider("Profit Target (R)", 2.0, 5.0, 3.0, 0.5)
    trail_tighten_threshold = st.slider("Final-Leg Tighten At (R)", 1.0, 4.0, 2.5, 0.25)

    st.caption("Backtest")
    interval_options = {"15-Minute": "15m", "30-Minute": "30m", "1-Hour": "1h", "Daily": "1d"}
    sel_interval = st.selectbox("Timeframe", list(interval_options.keys()), index=0)
    fetch_interval = interval_options[sel_interval]
    lookback_days = st.slider("Backtest Period (Days)", 5, 365, 60, 5)

    if st.button("🔄 Refresh Market Data", width="stretch", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    render_sidebar_nav()


# ══════════════════════════════════════════════════════════════════
# DATA + INDICATORS
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_gold_data(period_days: int, interval: str) -> pd.DataFrame:
    """Fetch XAU/USD (GC=F) OHLCV, robust to yfinance's MultiIndex columns."""
    df = yf.download("GC=F", period=f"{period_days}d", interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        # Intraday windows are capped (~60d for <1h); fall back to daily.
        df = yf.download("GC=F", period=f"{period_days}d", interval="1d",
                         progress=False, auto_adjust=True)
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Daily-anchored VWAP — cumulative sums reset each trading session."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = tp * df["Volume"]
    session = pd.Index(df.index).normalize()
    cum_pv = pv.groupby(session).cumsum()
    cum_vol = df["Volume"].groupby(session).cumsum().replace(0, np.nan)
    df["VWAP"] = (cum_pv / cum_vol).ffill()
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA_pull"] = df["Close"].ewm(span=ema_pull, adjust=False).mean()
    df["EMA_regime"] = df["Close"].ewm(span=ema_regime, adjust=False).mean()
    df["EMA_final"] = df["Close"].ewm(span=ema_final, adjust=False).mean()

    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=atr_period).mean()
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
    return df


def identify_entry_signals(df: pd.DataFrame) -> list:
    """Signals satisfying all six regime/VWAP/EMA/rejection/volume/momentum rules."""
    signals = []
    for i in range(2, len(df)):
        cur, prev = df.iloc[i], df.iloc[i - 1]
        if pd.isna(cur["ATR"]) or pd.isna(cur["EMA_regime"]) or pd.isna(cur["VWAP"]):
            continue

        bullish_regime = cur["Close"] > cur["EMA_regime"]
        bearish_regime = cur["Close"] < cur["EMA_regime"]
        above_vwap = cur["Close"] > cur["VWAP"]
        below_vwap = cur["Close"] < cur["VWAP"]

        ema_prox_long = min(cur["Low"], prev["Low"]) <= cur["EMA_pull"] <= cur["Close"]
        ema_prox_short = max(cur["High"], prev["High"]) >= cur["EMA_pull"] >= cur["Close"]

        body = abs(cur["Close"] - cur["Open"])
        lower_wick = min(cur["Open"], cur["Close"]) - cur["Low"]
        upper_wick = cur["High"] - max(cur["Open"], cur["Close"])
        pin_long = lower_wick >= 2 * body and upper_wick <= 0.5 * lower_wick
        pin_short = upper_wick >= 2 * body and lower_wick <= 0.5 * upper_wick
        engulf_long = cur["Close"] > prev["Open"] and cur["Open"] < prev["Close"]
        engulf_short = cur["Close"] < prev["Open"] and cur["Open"] > prev["Close"]
        rejection_long = pin_long or engulf_long
        rejection_short = pin_short or engulf_short

        vol_ok = cur["Volume"] > 1.1 * cur["Volume_MA20"]
        momentum_ok = (cur["High"] - cur["Low"]) >= 0.8 * cur["ATR"]

        if bullish_regime and above_vwap and ema_prox_long and rejection_long and vol_ok and momentum_ok:
            signals.append(("LONG", i, cur["Close"], cur["ATR"]))
        elif bearish_regime and below_vwap and ema_prox_short and rejection_short and vol_ok and momentum_ok:
            signals.append(("SHORT", i, cur["Close"], cur["ATR"]))
    return signals


def backtest_strategy(df: pd.DataFrame, signals: list) -> tuple:
    """Single-position backtest: hard ATR stop, EMA trail, R-target.

    1R is the entry-to-stop distance (atr_sl_mult × ATR at entry), so every
    trade's R-multiple is comparable and losing trades are fully modelled.
    """
    signal_at = {idx: (side, price, atr) for side, idx, price, atr in signals}
    trades, capital = [], float(initial_capital)
    equity_curve = [capital]
    pos = None  # (entry_idx, side, entry_price, sl_dist, stop_price)

    for idx in range(len(df)):
        bar = df.iloc[idx]

        if pos is not None:
            entry_idx, side, entry_price, sl_dist, stop_price = pos

            if side == "LONG":
                pnl_r = (bar["Close"] - entry_price) / sl_dist
                trail = bar["EMA_final"] if pnl_r >= trail_tighten_threshold else bar["EMA_pull"]
                stop_hit = bar["Low"] <= stop_price
                trail_hit = bar["Close"] < trail
            else:
                pnl_r = (entry_price - bar["Close"]) / sl_dist
                trail = bar["EMA_final"] if pnl_r >= trail_tighten_threshold else bar["EMA_pull"]
                stop_hit = bar["High"] >= stop_price
                trail_hit = bar["Close"] > trail

            exit_reason, exit_r, exit_price = None, None, None
            if stop_hit:
                exit_reason, exit_r, exit_price = "Stop Loss", -1.0, stop_price
            elif pnl_r >= target_r:
                exit_reason, exit_r, exit_price = "Target Hit", target_r, bar["Close"]
            elif trail_hit:
                exit_reason, exit_r, exit_price = "Trailing Stop", pnl_r, bar["Close"]

            if exit_reason is not None:
                pnl_amount = capital * risk_per_trade * exit_r
                capital += pnl_amount
                trades.append({
                    "entry_date": df.index[entry_idx], "exit_date": df.index[idx],
                    "type": side, "entry_price": entry_price, "exit_price": exit_price,
                    "r_multiple": exit_r, "pnl": pnl_amount, "exit_reason": exit_reason,
                    "bars_held": idx - entry_idx,
                })
                pos = None

        if pos is None and idx in signal_at:
            side, price, atr = signal_at[idx]
            sl_dist = atr * atr_sl_mult
            if sl_dist > 0:
                stop_price = price - sl_dist if side == "LONG" else price + sl_dist
                pos = (idx, side, price, sl_dist, stop_price)

        equity_curve.append(capital)

    return trades, equity_curve


def calculate_metrics(trades: list, equity_curve: list):
    if not trades:
        return None, None
    dft = pd.DataFrame(trades)
    wins = dft[dft["r_multiple"] > 0]
    losses = dft[dft["r_multiple"] <= 0]

    win_rate = len(wins) / len(dft) * 100
    avg_win = wins["r_multiple"].mean() if len(wins) else 0
    avg_loss = abs(losses["r_multiple"].mean()) if len(losses) else 0
    expectancy = dft["r_multiple"].mean()
    loss_sum = abs(losses["r_multiple"].sum())
    profit_factor = wins["r_multiple"].sum() / loss_sum if loss_sum else np.inf

    eq = pd.Series(equity_curve)
    rets = eq.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    cum_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
    running_max = eq.expanding().max()
    max_dd = abs(((eq - running_max) / running_max * 100).min())
    calmar = (cum_return / 100) / (max_dd / 100) if max_dd > 0 else 0

    return {
        "Total Trades": len(dft),
        "Win Rate (%)": round(win_rate, 1),
        "Average Win (R)": round(avg_win, 2),
        "Average Loss (R)": round(avg_loss, 2),
        "Expectancy (R)": round(expectancy, 3),
        "Profit Factor": round(profit_factor, 2),
        "Total Return (%)": round(cum_return, 1),
        "Annualised Sharpe Ratio": round(sharpe, 2),
        "Maximum Drawdown (%)": round(max_dd, 1),
        "Calmar Ratio": round(calmar, 2),
    }, dft


# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
now_str = datetime.now().strftime("%A, %d %B %Y · %H:%M")
st.markdown(f"""
<div class="bb-hero">
  <div>
    <div class="bb-hero-title">🟡 VWAP-EMA Regime Framework</div>
    <div class="bb-hero-sub">XAU/USD (Gold Spot · GC=F) · {sel_interval} · {lookback_days}d lookback · {now_str}</div>
  </div>
  <div class="bb-hero-right">
    <div class="bb-hero-symbol">XAU/USD</div>
    <div class="bb-hero-ticker">SL = {atr_sl_mult}×ATR{atr_period} · TGT = {target_r}R</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load + compute ──────────────────────────────────────────────────────────────
with st.spinner("📡 Loading market data…"):
    df = fetch_gold_data(lookback_days, fetch_interval)

if df.empty:
    st.error("❌ Failed to load market data. Check the connection and try refreshing.")
    st.stop()

if fetch_interval in {"15m", "30m", "1h"} and len(df) > 0:
    span_days = (df.index[-1] - df.index[0]).days
    if span_days < lookback_days - 2:
        st.warning(
            f"⚠️ yfinance caps {sel_interval} history — loaded {span_days}d of the "
            f"{lookback_days}d requested.", icon="⚠️")

df = calculate_vwap(df)
df = calculate_indicators(df)
signals = identify_entry_signals(df)
trades, equity_curve = backtest_strategy(df, signals)
metrics, df_trades = calculate_metrics(trades, equity_curve)

c1, c2, c3 = st.columns(3)
c1.success(f"✅ {len(df)} bars loaded")
c2.info(f"📊 {len(signals)} entry signals")
c3.info(f"🎯 {len(trades)} trades executed")

# ══════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════
if metrics:
    st.markdown("#### 📊 Performance Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Return", f"{metrics['Total Return (%)']}%")
    k1.metric("Win Rate", f"{metrics['Win Rate (%)']}%")
    k2.metric("Sharpe Ratio", f"{metrics['Annualised Sharpe Ratio']}")
    k2.metric("Profit Factor", f"{metrics['Profit Factor']}")
    k3.metric("Expectancy", f"{metrics['Expectancy (R)']}R")
    k3.metric("Max Drawdown", f"-{metrics['Maximum Drawdown (%)']}%")
    k4.metric("Total Trades", f"{metrics['Total Trades']}")
    k4.metric("Calmar Ratio", f"{metrics['Calmar Ratio']}")

    s1, s2, s3 = st.columns(3)
    s1.metric("Average Win", f"+{metrics['Average Win (R)']}R")
    s2.metric("Average Loss", f"-{metrics['Average Loss (R)']}R")
    s3.metric("Risk per Trade", f"{risk_per_trade * 100:.1f}%")

    # ── Price + indicators + equity ──────────────────────────────
    st.markdown("#### 📉 Strategy Visualisation")
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("XAU/USD with VWAP + EMAs", "Volume", "Equity Curve"),
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color=T.GREEN, decreasing_line_color=T.RED,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_pull"], name=f"EMA {ema_pull}",
                             line=dict(color=T.YELLOW, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_regime"], name=f"EMA {ema_regime}",
                             line=dict(color=T.CYAN, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                             line=dict(color=T.PURPLE, width=1, dash="dash")), row=1, col=1)

    for side, idx, price, _ in signals:
        fig.add_trace(go.Scatter(
            x=[df.index[idx]], y=[price], mode="markers",
            marker=dict(symbol="triangle-up" if side == "LONG" else "triangle-down",
                        size=11, color=T.GREEN if side == "LONG" else T.RED),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    vol_colors = [T.GREEN if df["Close"].iloc[i] >= df["Open"].iloc[i] else T.RED
                  for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=vol_colors, showlegend=False), row=2, col=1)

    eq_series = pd.Series(equity_curve, index=[df.index[0]] + list(df.index))
    fig.add_trace(go.Scatter(x=eq_series.index, y=eq_series, name="Equity",
                             line=dict(color=T.AMBER, width=2), showlegend=False), row=3, col=1)

    fig.update_layout(height=760, xaxis_rangeslider_visible=False, **PLOTLY_LAYOUT)
    style_axes(fig, 3)
    fig.update_annotations(font_color=T.GREY, font_size=11)
    st.plotly_chart(fig, width="stretch", config=dict(displayModeBar=False))

    # ── R-distribution + monthly returns ─────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        rfig = go.Figure(go.Histogram(
            x=df_trades["r_multiple"], nbinsx=30,
            marker_color=T.AMBER_DIM, marker_line_color=T.AMBER, marker_line_width=1,
        ))
        mean_r = df_trades["r_multiple"].mean()
        rfig.add_vline(x=mean_r, line_dash="dash", line_color=T.GREEN,
                       annotation_text=f"Mean {mean_r:.2f}R", annotation_position="top",
                       annotation_font_color=T.GREEN)
        rfig.add_vline(x=0, line_color=T.RED, line_width=2)
        rfig.update_layout(title="R-Multiple Distribution", height=360, **PLOTLY_LAYOUT)
        rfig.update_xaxes(showgrid=False, linecolor=T.BORDER)
        rfig.update_yaxes(showgrid=True, gridcolor=T.BORDER)
        st.plotly_chart(rfig, width="stretch", config=dict(displayModeBar=False))

    with col_b:
        mdf = df_trades.copy()
        mdf["month"] = pd.to_datetime(mdf["exit_date"]).dt.to_period("M")
        monthly_pct = mdf.groupby("month")["pnl"].sum() / initial_capital * 100
        mfig = go.Figure(go.Bar(
            x=monthly_pct.index.astype(str), y=monthly_pct.values,
            marker_color=[T.GREEN if v > 0 else T.RED for v in monthly_pct.values],
        ))
        mfig.update_layout(title="Monthly Returns (% of Capital)", height=360, **PLOTLY_LAYOUT)
        mfig.update_xaxes(showgrid=False, linecolor=T.BORDER)
        mfig.update_yaxes(showgrid=True, gridcolor=T.BORDER, ticksuffix="%")
        st.plotly_chart(mfig, width="stretch", config=dict(displayModeBar=False))

    # ── Trade table + download ───────────────────────────────────
    st.markdown("#### 📋 Trade Summary")
    disp = df_trades.copy()
    disp["entry_date"] = pd.to_datetime(disp["entry_date"]).dt.strftime("%Y-%m-%d %H:%M")
    disp["exit_date"] = pd.to_datetime(disp["exit_date"]).dt.strftime("%Y-%m-%d %H:%M")
    for c in ("entry_price", "exit_price", "r_multiple", "pnl"):
        disp[c] = disp[c].round(2)
    disp = disp[["entry_date", "exit_date", "type", "entry_price", "exit_price",
                 "r_multiple", "pnl", "exit_reason", "bars_held"]]
    st.dataframe(disp, width="stretch", hide_index=True, height=380)
    st.download_button("📥 Download Trade Data (CSV)", disp.to_csv(index=False),
                       file_name="vwap_ema_gold_trades.csv", mime="text/csv")
else:
    st.warning("No trades were generated for this period. Try a longer lookback, a "
               "different timeframe, or looser indicator parameters.")

# ══════════════════════════════════════════════════════════════════
# RULES
# ══════════════════════════════════════════════════════════════════
with st.expander("📖 Strategy Rules"):
    st.markdown(f"""
**Entry (all six must hold):**

1. **Regime** — price on the correct side of EMA {ema_regime}
2. **VWAP** — price on the correct side of the daily-anchored VWAP
3. **EMA proximity** — bar touches EMA {ema_pull} intrabar but closes back across it
4. **Rejection** — pin bar or engulfing candle
5. **Volume** — volume > 1.1× its 20-period average
6. **Momentum** — candle range ≥ 0.8× ATR({atr_period})

**Exits:**

- **Hard stop** — {atr_sl_mult}×ATR({atr_period}) from entry (defines 1R, = −1R)
- **Target** — +{target_r}R
- **Trailing stop** — EMA {ema_pull} (close-only), tightening to EMA {ema_final} once +{trail_tighten_threshold}R

**Risk:** {risk_per_trade * 100:.1f}% of equity risked per trade · volatility-sized via ATR.
""")

st.markdown(f"""
<div style="text-align:center;color:{T.BORDER};font-size:11px;margin-top:28px;
     padding-top:14px;border-top:1px solid {T.BORDER};">
  🟡 VWAP-EMA Gold · after 'A Regime-Filtered Intraday Trading Framework for Gold',
  A. Bhatti (2026) · Not financial advice
</div>
""", unsafe_allow_html=True)
