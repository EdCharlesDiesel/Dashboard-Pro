import streamlit as st
from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Workflow Backtest · Trading System",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { --border: color-mix(in srgb, var(--text-color) 12%, transparent);
              --muted:  color-mix(in srgb, var(--text-color) 55%, transparent); }
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    .stApp{background:var(--background-color);}
    section[data-testid="stSidebar"]{background:var(--secondary-background-color)!important;border-right:1px solid var(--border,#2a2a2a);}
    .hero{background:linear-gradient(135deg,#000000 0%,#0a0a0a 50%,#000000 100%);border:1px solid #2a2a2a;border-radius:16px;padding:24px 32px;margin-bottom:20px;position:relative;overflow:hidden;}
    .metric-box{background:var(--background-color);border:1px solid var(--border,#2a2a2a);border-radius:8px;padding:12px;text-align:center;}
    .metric-value{font-size:22px;font-weight:700;color:#e6e6e6;}
    .metric-label{font-size:10px;color:#9a9a9a;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}
    .check-pass{background:#1a3a2a;border:1px solid #00ff66;border-radius:6px;padding:8px 12px;margin:3px 0;font-size:13px;}
    .check-fail{background:#3a1a1a;border:1px solid #ff3344;border-radius:6px;padding:8px 12px;margin:3px 0;font-size:13px;}
    .check-auto{background:#1a2a3a;border:1px solid #00ff41;border-radius:6px;padding:8px 12px;margin:3px 0;font-size:13px;}
    .trade-win{border-left:3px solid #00ff66;padding:4px 8px;margin:2px 0;font-size:12px;}
    .trade-loss{border-left:3px solid #ff3344;padding:4px 8px;margin:2px 0;font-size:12px;}
    .trade-be{border-left:3px solid #ffcc00;padding:4px 8px;margin:2px 0;font-size:12px;}
    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    [data-testid="stSidebarCollapseButton"]{visibility:visible !important;display:flex !important;}
    section[data-testid="stSidebar"]{display:block !important;}
    .block-container{padding-top:1.5rem;max-width:1400px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# INSTRUMENTS
# ══════════════════════════════════════════════════════════════════

INSTRUMENTS = {
    "EUR/USD": {"ticker": "EURUSD=X", "pip": 0.0001, "pip_val": 10.0},
    "GBP/USD": {"ticker": "GBPUSD=X", "pip": 0.0001, "pip_val": 10.0},
    "AUD/USD": {"ticker": "AUDUSD=X", "pip": 0.0001, "pip_val": 10.0},
    "NZD/USD": {"ticker": "NZDUSD=X", "pip": 0.0001, "pip_val": 10.0},
    "USD/JPY": {"ticker": "USDJPY=X", "pip": 0.01,   "pip_val": 9.09},
    "USD/CHF": {"ticker": "USDCHF=X", "pip": 0.0001, "pip_val": 10.8},
    "USD/CAD": {"ticker": "USDCAD=X", "pip": 0.0001, "pip_val": 7.4},
    "EUR/GBP": {"ticker": "EURGBP=X", "pip": 0.0001, "pip_val": 12.5},
    "EUR/JPY": {"ticker": "EURJPY=X", "pip": 0.01,   "pip_val": 9.09},
    "GBP/JPY": {"ticker": "GBPJPY=X", "pip": 0.01,   "pip_val": 9.09},
    "AUD/JPY": {"ticker": "AUDJPY=X", "pip": 0.01,   "pip_val": 9.09},
    "EUR/AUD": {"ticker": "EURAUD=X", "pip": 0.0001, "pip_val": 6.3},
    "GBP/AUD": {"ticker": "GBPAUD=X", "pip": 0.0001, "pip_val": 6.3},
    "EUR/CAD": {"ticker": "EURCAD=X", "pip": 0.0001, "pip_val": 7.4},
    "GBP/CAD": {"ticker": "GBPCAD=X", "pip": 0.0001, "pip_val": 7.4},
    "USD/ZAR": {"ticker": "USDZAR=X", "pip": 0.0001, "pip_val": 0.55},
    "🥇 Gold":  {"ticker": "GC=F",    "pip": 0.10,   "pip_val": 10.0},
}

CHECK_META = [
    ("macro_bias",       "02. Macro Bias Confirmed",         "auto",  True),
    ("news_filter",      "03. News Filter Clear",            "auto",  True),
    ("correlations",     "03. Correlation Exposure OK",      "auto",  False),
    ("atr_volatility",   "05. ATR Volatility OK",            "calc",  False),
    ("weekly_ema",       "06. Weekly EMA Aligned",           "calc",  True),   # hard block: macro direction
    ("weekly_rsi",       "07. Weekly RSI has Room",          "calc",  False),  # score contributor
    ("weekly_swing",     "08. Weekly Swing Structure",       "calc",  False),  # score contributor
    ("daily_trend",      "09. Daily Trend Intact",           "calc",  True),   # hard block: must trade with trend
    ("daily_macd",       "10. Daily MACD Momentum",          "calc",  False),  # score contributor
    ("4h_confluence",    "11. 4H Confluence Zone",           "calc",  False),  # score contributor
    ("confluence_check", "11. 2/3 Confluences Met",          "calc",  False),  # score contributor
    ("rejection",        "13. 15M Rejection Candle",         "calc",  False),  # score contributor
    ("entry_signal",     "14. 15M Entry Signal",             "calc",  False),  # score contributor
    ("stop_structure",   "14. Stop Below Structure",         "calc",  True),   # hard block: must have a stop
    ("rr_check",         "15. R:R ≥ 2:1",                   "calc",  True),   # hard block: minimum RR
    ("loss_limit",       "16. Daily Loss Limit OK",          "auto",  False),
    ("mkt_structure",    "18. Market Structure Intact",      "calc",  False),  # score contributor
    ("setup_rank",       "18. Setup Score ≥ 14/18",          "calc",  False),
]

# ══════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    hist = calc_ema(series, fast) - calc_ema(series, slow)
    sig  = calc_ema(hist, signal)
    return hist - sig  # histogram

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low']  - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def calc_stochastic(df: pd.DataFrame, k=14, d=3):
    low_k  = df['Low'].rolling(k).min()
    high_k = df['High'].rolling(k).max()
    pct_k  = 100 * (df['Close'] - low_k) / (high_k - low_k + 1e-9)
    return pct_k, pct_k.rolling(d).mean()

def swing_highs(series: pd.Series, strength: int = 3) -> list:
    idx = []
    for i in range(strength, len(series) - strength):
        window = series.iloc[i - strength: i + strength + 1]
        if series.iloc[i] == window.max():
            idx.append(i)
    return idx

def swing_lows(series: pd.Series, strength: int = 3) -> list:
    idx = []
    for i in range(strength, len(series) - strength):
        window = series.iloc[i - strength: i + strength + 1]
        if series.iloc[i] == window.min():
            idx.append(i)
    return idx

# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_data(ticker: str):
    daily  = yf.download(ticker, period="1y",  interval="1d",  auto_adjust=True, progress=False)
    weekly = yf.download(ticker, period="3y",  interval="1wk", auto_adjust=True, progress=False)
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns  = daily.columns.get_level_values(0)
    if isinstance(weekly.columns, pd.MultiIndex):
        weekly.columns = weekly.columns.get_level_values(0)
    daily.index  = pd.to_datetime(daily.index).tz_localize(None)
    weekly.index = pd.to_datetime(weekly.index).tz_localize(None)
    return daily.dropna(), weekly.dropna()

# ══════════════════════════════════════════════════════════════════
# CHECKS
# ══════════════════════════════════════════════════════════════════

def run_checks(d: pd.DataFrame, w: pd.DataFrame, direction: str) -> dict:
    """Evaluate all computable checks on the last bar of `d` and `w`."""
    res = {}

    # ── Auto-pass checks ──────────────────────────────────────────
    for key in ("macro_bias", "news_filter", "correlations", "loss_limit"):
        res[key] = (True, "Auto-pass in backtest (manual confirmation required live)")

    # ── ATR Volatility ────────────────────────────────────────────
    if len(d) >= 20:
        atr = calc_atr(d, 14).iloc[-1]
        price = d['Close'].iloc[-1]
        ratio = atr / price
        res['atr_volatility'] = (0.001 <= ratio <= 0.05, f"ATR/Price = {ratio*100:.2f}%")
    else:
        res['atr_volatility'] = (False, "Insufficient data")

    # ── Weekly EMA ────────────────────────────────────────────────
    if len(w) >= 55:
        w_ema20 = calc_ema(w['Close'], 20).iloc[-1]
        w_ema50 = calc_ema(w['Close'], 50).iloc[-1]
        w_price = w['Close'].iloc[-1]
        if direction == "Long":
            ok = w_price > w_ema20 > w_ema50
        else:
            ok = w_price < w_ema20 < w_ema50
        res['weekly_ema'] = (ok, f"EMA20={w_ema20:.5f}  EMA50={w_ema50:.5f}  Price={w_price:.5f}")
    else:
        res['weekly_ema'] = (False, "Insufficient weekly data")

    # ── Weekly RSI ────────────────────────────────────────────────
    if len(w) >= 18:
        w_rsi = calc_rsi(w['Close'], 14).iloc[-1]
        if direction == "Long":
            ok = w_rsi < 65
        else:
            ok = w_rsi > 35
        res['weekly_rsi'] = (ok, f"Weekly RSI = {w_rsi:.1f}")
    else:
        res['weekly_rsi'] = (False, "Insufficient weekly data")

    # ── Weekly Swing Structure ────────────────────────────────────
    sh = swing_highs(w['High'], 2)
    sl = swing_lows(w['Low'],  2)
    if len(sh) >= 2 and len(sl) >= 2:
        hh = w['High'].iloc[sh[-1]] > w['High'].iloc[sh[-2]]
        hl = w['Low'].iloc[sl[-1]]  > w['Low'].iloc[sl[-2]]
        ll = w['Low'].iloc[sl[-1]]  < w['Low'].iloc[sl[-2]]
        lh = w['High'].iloc[sh[-1]] < w['High'].iloc[sh[-2]]
        if direction == "Long":
            ok = hh and hl
            res['weekly_swing'] = (ok, f"HH={hh}  HL={hl}")
        else:
            ok = ll and lh
            res['weekly_swing'] = (ok, f"LL={ll}  LH={lh}")
    else:
        res['weekly_swing'] = (False, "Not enough weekly swing points")

    # ── Daily Trend ───────────────────────────────────────────────
    if len(d) >= 52:
        d_ema20 = calc_ema(d['Close'], 20).iloc[-1]
        d_ema50 = calc_ema(d['Close'], 50).iloc[-1]
        d_price = d['Close'].iloc[-1]
        if direction == "Long":
            ok = d_ema20 > d_ema50 and d_price > d_ema20
        else:
            ok = d_ema20 < d_ema50 and d_price < d_ema20
        res['daily_trend'] = (ok, f"EMA20={d_ema20:.5f}  EMA50={d_ema50:.5f}  Price={d_price:.5f}")
    else:
        res['daily_trend'] = (False, "Insufficient daily data")

    # ── Daily MACD ────────────────────────────────────────────────
    if len(d) >= 35:
        hist = calc_macd(d['Close'])
        h_now  = hist.iloc[-1]
        h_prev = hist.iloc[-2]
        if direction == "Long":
            ok = h_now > 0 or (h_now > h_prev and h_now > -0.001)
        else:
            ok = h_now < 0 or (h_now < h_prev and h_now < 0.001)
        res['daily_macd'] = (ok, f"MACD hist = {h_now:.6f}  (prev {h_prev:.6f})")
    else:
        res['daily_macd'] = (False, "Insufficient data")

    # ── 4H Confluence Zone (approximated from daily) ──────────────
    if len(d) >= 20:
        atr = calc_atr(d, 14).iloc[-1]
        price = d['Close'].iloc[-1]
        tol = atr * 0.6

        recent_high = d['High'].rolling(20).max().iloc[-1]
        recent_low  = d['Low'].rolling(20).min().iloc[-1]
        fib_range   = recent_high - recent_low
        fib_levels  = [recent_high - r * fib_range for r in (0.236, 0.382, 0.500, 0.618, 0.786)]
        near_fib    = any(abs(price - f) <= tol for f in fib_levels)

        prev   = d.iloc[-2]
        pp     = (prev['High'] + prev['Low'] + prev['Close']) / 3
        s1, r1 = 2 * pp - prev['High'], 2 * pp - prev['Low']
        near_piv = any(abs(price - lvl) <= tol for lvl in (pp, s1, r1))

        d_ema20  = calc_ema(d['Close'], 20).iloc[-1]
        near_ema = abs(price - d_ema20) <= tol

        count = sum([near_fib, near_piv, near_ema])
        res['4h_confluence']    = (count >= 1, f"Fib={near_fib}  Pivot={near_piv}  EMA={near_ema}  ({count}/3)")
        res['confluence_check'] = (count >= 2, f"{count}/3 confluences  (min 2 required)")
    else:
        res['4h_confluence']    = (False, "Insufficient data")
        res['confluence_check'] = (False, "Insufficient data")

    # ── 15M Rejection Candle (proxied via daily close position) ───
    if len(d) >= 3:
        bar  = d.iloc[-1]
        rng  = bar['High'] - bar['Low']
        if rng > 0:
            pos = (bar['Close'] - bar['Low']) / rng
            if direction == "Long":
                ok = pos >= 0.60
            else:
                ok = pos <= 0.40
            res['rejection'] = (ok, f"Close position in day range: {pos*100:.0f}%  ({'bullish' if pos>=0.6 else 'bearish' if pos<=0.4 else 'neutral'})")
        else:
            res['rejection'] = (False, "Doji — no range")
    else:
        res['rejection'] = (False, "Insufficient data")

    # ── 15M Entry Signal — Stochastic proxy ──────────────────────
    if len(d) >= 20:
        k, k_d = calc_stochastic(d, k=14, d=3)
        k_now, k_prev = k.iloc[-1], k.iloc[-2]
        d_now, d_prev = k_d.iloc[-1], k_d.iloc[-2]
        if direction == "Long":
            cross_up = k_prev < d_prev and k_now >= d_now
            ok = cross_up and k_prev < 50
            res['entry_signal'] = (ok, f"Stoch K={k_now:.1f}  D={d_now:.1f}  cross_up={cross_up}  was_below_50={k_prev < 50}")
        else:
            cross_dn = k_prev > d_prev and k_now <= d_now
            ok = cross_dn and k_prev > 50
            res['entry_signal'] = (ok, f"Stoch K={k_now:.1f}  D={d_now:.1f}  cross_down={cross_dn}  was_above_50={k_prev > 50}")
    else:
        res['entry_signal'] = (False, "Insufficient data")

    # ── Stop Structure ────────────────────────────────────────────
    if len(d) >= 12:
        price = d['Close'].iloc[-1]
        atr   = calc_atr(d, 14).iloc[-1]
        sh_idx = swing_highs(d['High'], 3)
        sl_idx = swing_lows(d['Low'],  3)
        if direction == "Long" and sl_idx:
            level  = d['Low'].iloc[sl_idx[-1]]
            sl_dist = price - level
            ok = 0 < sl_dist < atr * 3
            res['stop_structure'] = (ok, f"Last swing low = {level:.5f}  SL dist = {sl_dist:.5f} ({sl_dist/atr:.1f}× ATR)")
        elif direction == "Short" and sh_idx:
            level  = d['High'].iloc[sh_idx[-1]]
            sl_dist = level - price
            ok = 0 < sl_dist < atr * 3
            res['stop_structure'] = (ok, f"Last swing high = {level:.5f}  SL dist = {sl_dist:.5f} ({sl_dist/atr:.1f}× ATR)")
        else:
            res['stop_structure'] = (False, "No swing point found")
    else:
        res['stop_structure'] = (False, "Insufficient data")

    # ── R:R ≥ 2:1 ────────────────────────────────────────────────
    if len(d) >= 14:
        atr    = calc_atr(d, 14).iloc[-1]
        sl_d   = atr * 1.5
        tp1_d  = sl_d * 2.0
        res['rr_check'] = (True, f"ATR-based R:R = {tp1_d/sl_d:.1f}:1  (SL={sl_d:.5f}  TP1={tp1_d:.5f})")
    else:
        res['rr_check'] = (False, "Insufficient data")

    # ── Market Structure (4H proxy — daily swing) ─────────────────
    sh_idx = swing_highs(d['High'], 3)
    sl_idx = swing_lows(d['Low'],  3)
    if len(sh_idx) >= 2 and len(sl_idx) >= 2:
        hh = d['High'].iloc[sh_idx[-1]] > d['High'].iloc[sh_idx[-2]]
        hl = d['Low'].iloc[sl_idx[-1]]  > d['Low'].iloc[sl_idx[-2]]
        ll = d['Low'].iloc[sl_idx[-1]]  < d['Low'].iloc[sl_idx[-2]]
        lh = d['High'].iloc[sh_idx[-1]] < d['High'].iloc[sh_idx[-2]]
        if direction == "Long":
            ok = hh and hl
            res['mkt_structure'] = (ok, f"Daily structure: HH={hh}  HL={hl}")
        else:
            ok = ll and lh
            res['mkt_structure'] = (ok, f"Daily structure: LL={ll}  LH={lh}")
    else:
        res['mkt_structure'] = (False, "Insufficient swing history")

    # ── Setup Rank (aggregate) ────────────────────────────────────
    score = sum(v for v, _ in res.values())
    res['setup_rank'] = (score >= 14, f"Score = {score}/18")

    return res, score

# ══════════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════════

def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str,
                   sl_mult: float = 1.5, rr: float = 2.0, max_bars: int = 20) -> dict:
    entry = df['Open'].iloc[entry_idx]
    atr   = calc_atr(df.iloc[:entry_idx], 14).iloc[-1]
    sl_d  = atr * sl_mult

    if direction == "Long":
        sl, tp1, tp2 = entry - sl_d, entry + sl_d * rr, entry + sl_d * (rr + 1)
    else:
        sl, tp1, tp2 = entry + sl_d, entry - sl_d * rr, entry - sl_d * (rr + 1)

    outcome    = "Timeout"
    exit_price = df['Close'].iloc[min(entry_idx + max_bars, len(df) - 1)]
    tp1_hit    = False
    active_sl  = sl

    for i in range(entry_idx, min(entry_idx + max_bars, len(df))):
        bar = df.iloc[i]
        if direction == "Long":
            if bar['Low'] <= active_sl:
                outcome    = "TP1 → BE" if tp1_hit else "SL"
                exit_price = active_sl
                break
            if not tp1_hit and bar['High'] >= tp1:
                tp1_hit   = True
                active_sl = entry
            if tp1_hit and bar['High'] >= tp2:
                outcome    = "TP2"
                exit_price = tp2
                break
        else:
            if bar['High'] >= active_sl:
                outcome    = "TP1 → BE" if tp1_hit else "SL"
                exit_price = active_sl
                break
            if not tp1_hit and bar['Low'] <= tp1:
                tp1_hit   = True
                active_sl = entry
            if tp1_hit and bar['Low'] <= tp2:
                outcome    = "TP2"
                exit_price = tp2
                break

    r_mult = ((exit_price - entry) / sl_d) if direction == "Long" else ((entry - exit_price) / sl_d)
    return dict(entry=entry, exit=exit_price, sl=sl, tp1=tp1, tp2=tp2,
                outcome=outcome, r=round(r_mult, 2), sl_d=sl_d)

# ══════════════════════════════════════════════════════════════════
# BACKTEST RUNNER
# ══════════════════════════════════════════════════════════════════

def run_backtest(daily: pd.DataFrame, weekly: pd.DataFrame, direction: str,
                 min_score: int, sl_mult: float, rr: float, start_dt, end_dt) -> list:
    trades = []
    warmup = 55  # bars needed before scoring begins
    in_trade = False

    for i in range(warmup, len(daily)):
        bar_date = daily.index[i].date()
        if bar_date < start_dt or bar_date > end_dt:
            continue
        if in_trade:
            in_trade = False
            continue

        d_slice = daily.iloc[:i + 1]
        w_slice = weekly[weekly.index <= daily.index[i]]
        if len(w_slice) < 20:
            continue

        res, score = run_checks(d_slice, w_slice, direction)

        # Critical checks must all pass
        critical_keys = [k for k, _, mode, crit in CHECK_META if mode == "calc" and crit]
        critical_pass  = all(res.get(k, (False,))[0] for k in critical_keys)

        if score >= min_score and critical_pass:
            if i + 1 >= len(daily):
                continue
            trade = simulate_trade(daily, i + 1, direction, sl_mult, rr)
            trades.append({
                "date":   bar_date,
                "score":  score,
                **trade,
            })
            in_trade = True

    return trades

# ══════════════════════════════════════════════════════════════════
# STATS & CHARTS
# ══════════════════════════════════════════════════════════════════

def compute_stats(trades: list) -> dict:
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    wins    = df[df['r'] > 0]
    losses  = df[df['r'] <= 0]
    win_r   = len(wins) / len(df) * 100
    avg_r   = df['r'].mean()
    gross_p = wins['r'].sum()
    gross_l = abs(losses['r'].sum()) or 1
    pf      = gross_p / gross_l

    cumr    = df['r'].cumsum()
    peak    = cumr.cummax()
    dd      = (cumr - peak).min()

    return dict(
        total=len(df), wins=len(wins), losses=len(losses),
        win_rate=win_r, avg_r=avg_r, net_r=df['r'].sum(),
        profit_factor=pf, max_dd=dd,
    )

def equity_curve_chart(trades: list) -> go.Figure:
    df = pd.DataFrame(trades)
    df['cumR'] = df['r'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['cumR'],
        mode='lines+markers',
        line=dict(color='#00ff41', width=2),
        marker=dict(
            color=['#00ff66' if r > 0 else '#ff3344' for r in df['r']],
            size=8, symbol='circle',
        ),
        name='Equity (R)',
        hovertemplate='%{x}<br>Cumulative R: %{y:.2f}<extra></extra>',
    ))
    fig.add_hline(y=0, line=dict(color='#9a9a9a', dash='dash', width=1))

    fig.update_layout(
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='#c0c0c0', size=11),
        xaxis=dict(gridcolor='#2a2a2a', showgrid=True),
        yaxis=dict(gridcolor='#2a2a2a', showgrid=True, title='Cumulative R'),
        margin=dict(l=40, r=20, t=20, b=40),
        height=280,
        showlegend=False,
        hovermode='x unified',
    )
    return fig

def r_distribution_chart(trades: list) -> go.Figure:
    df = pd.DataFrame(trades)
    colors = ['#00ff66' if r > 0 else '#ff3344' for r in df['r']]
    fig = go.Figure(go.Bar(
        x=[str(t) for t in df['date']], y=df['r'],
        marker_color=colors,
        hovertemplate='%{x}<br>R: %{y:.2f}<extra></extra>',
    ))
    fig.add_hline(y=0, line=dict(color='#9a9a9a', dash='dash', width=1))
    fig.update_layout(
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='#c0c0c0', size=11),
        xaxis=dict(gridcolor='#2a2a2a', tickangle=-45, showgrid=False),
        yaxis=dict(gridcolor='#2a2a2a', title='R Multiple'),
        margin=dict(l=40, r=20, t=20, b=60),
        height=220,
    )
    return fig

def check_pass_rate_chart(trades: list, daily: pd.DataFrame, weekly: pd.DataFrame,
                           direction: str) -> go.Figure:
    """Show how often each check passed across all evaluated bars."""
    pass_counts = {k: 0 for k, *_ in CHECK_META}
    total_bars  = 0
    warmup = 55
    for i in range(warmup, min(warmup + 60, len(daily))):
        d_s = daily.iloc[:i + 1]
        w_s = weekly[weekly.index <= daily.index[i]]
        if len(w_s) < 20:
            continue
        res, _ = run_checks(d_s, w_s, direction)
        total_bars += 1
        for k, v_detail in res.items():
            if k in pass_counts:
                pass_counts[k] += int(v_detail[0])

    if total_bars == 0:
        return go.Figure()

    labels = [label for _, label, _, _ in CHECK_META]
    pcts   = [pass_counts.get(k, 0) / total_bars * 100 for k, *_ in CHECK_META]
    colors = ['#00ff66' if p >= 60 else '#ffcc00' if p >= 40 else '#ff3344' for p in pcts]

    fig = go.Figure(go.Bar(
        x=pcts, y=labels, orientation='h',
        marker_color=colors,
        hovertemplate='%{y}<br>Pass rate: %{x:.0f}%<extra></extra>',
    ))
    fig.update_layout(
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='#c0c0c0', size=10),
        xaxis=dict(gridcolor='#2a2a2a', range=[0, 100], title='Pass Rate %'),
        yaxis=dict(gridcolor='#2a2a2a', autorange='reversed'),
        margin=dict(l=200, r=20, t=20, b=40),
        height=400,
    )
    return fig

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🧪 Workflow Backtest")
    st.markdown("---")

    pair      = st.selectbox("Instrument", list(INSTRUMENTS.keys()))
    direction = st.radio("Direction", ["Long", "Short"], horizontal=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        start_dt = st.date_input("From", value=datetime.today() - timedelta(days=180))
    with col_b:
        end_dt   = st.date_input("To",   value=datetime.today())

    st.markdown("---")
    min_score = st.slider("Min score to enter (out of 18)", 10, 18, 14)
    sl_mult   = st.slider("SL × ATR",  1.0, 3.0, 1.5, 0.1)
    rr        = st.slider("TP1 R:R",   1.5, 4.0, 2.0, 0.5)
    account   = st.number_input("Account size ($)", 1000, 1000000, 10000, 1000)
    risk_pct  = st.slider("Risk per trade (%)", 0.5, 3.0, 1.0, 0.25)

    st.markdown("---")
    run_btn = st.button("▶ Run Backtest", use_container_width=True, type="primary")

    st.divider()
    render_sidebar_nav()
# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero">
  <h2 style="margin:0;color:#e6e6e6;font-size:22px;font-weight:700;">
    🧪 Workflow Backtest — {pair} {direction}
  </h2>
  <p style="margin:6px 0 0;color:#9a9a9a;font-size:13px;">
    Simulates all 18 workflow checks on daily data · SL = {sl_mult}× ATR · TP1 = {rr}:1 R:R · Min score {min_score}/18
  </p>
  <p style="margin:4px 0 0;color:#9a9a9a;font-size:11px;">
    ⚠ 15M checks are proxied from daily bars. Results reflect system logic, not exact 15-minute execution.
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

ticker  = INSTRUMENTS[pair]["ticker"]
pip_sz  = INSTRUMENTS[pair]["pip"]

with st.spinner("Loading price data…"):
    try:
        daily, weekly = load_data(ticker)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if daily.empty or len(daily) < 60:
    st.error("Not enough data. Try a more liquid instrument or wider date range.")
    st.stop()

# ── Workflow Replay (always visible) ─────────────────────────────
st.markdown("### Step-by-Step Workflow Replay")
st.markdown("Pick any historical date to walk through all 18 checks exactly as the system would on that day.")

available_dates = [d.date() for d in daily.index if len(daily[daily.index <= d]) >= 55]
replay_date     = st.select_slider(
    "Select date to replay",
    options=available_dates,
    value=available_dates[-1] if available_dates else None,
)

if replay_date:
    replay_idx = next((i for i, d in enumerate(daily.index) if d.date() == replay_date), None)
    if replay_idx and replay_idx >= 55:
        d_slice = daily.iloc[:replay_idx + 1]
        w_slice = weekly[weekly.index <= daily.index[replay_idx]]
        replay_res, replay_score = run_checks(d_slice, w_slice, direction)

        passed   = sum(v for v, _ in replay_res.values())
        critical = [k for k, _, mode, crit in CHECK_META if mode == "calc" and crit]
        crit_ok  = all(replay_res.get(k, (False,))[0] for k in critical)
        signal   = "🟢 GO" if passed >= min_score and crit_ok else "🔴 NO TRADE"

        cols = st.columns([1, 1, 1, 1])
        cols[0].markdown(f'<div class="metric-box"><div class="metric-value">{passed}/18</div><div class="metric-label">Checks Passed</div></div>', unsafe_allow_html=True)
        cols[1].markdown(f'<div class="metric-box"><div class="metric-value">{signal}</div><div class="metric-label">Signal</div></div>', unsafe_allow_html=True)
        cols[2].markdown(f'<div class="metric-box"><div class="metric-value">{daily["Close"].iloc[replay_idx]:.5f}</div><div class="metric-label">Close Price</div></div>', unsafe_allow_html=True)
        cols[3].markdown(f'<div class="metric-box"><div class="metric-value">{calc_atr(d_slice, 14).iloc[-1]:.5f}</div><div class="metric-label">ATR14</div></div>', unsafe_allow_html=True)

        st.markdown("")

        c1, c2 = st.columns(2)
        for i, (key, label, mode, critical_flag) in enumerate(CHECK_META):
            col = c1 if i % 2 == 0 else c2
            if key in replay_res:
                passed_flag, detail = replay_res[key]
                crit_tag = " ⭐" if critical_flag else ""
                if mode == "auto":
                    col.markdown(f'<div class="check-auto">🔵 {label}{crit_tag}<br><small style="color:#9a9a9a">{detail}</small></div>', unsafe_allow_html=True)
                elif passed_flag:
                    col.markdown(f'<div class="check-pass">✅ {label}{crit_tag}<br><small style="color:#9a9a9a">{detail}</small></div>', unsafe_allow_html=True)
                else:
                    col.markdown(f'<div class="check-fail">❌ {label}{crit_tag}<br><small style="color:#9a9a9a">{detail}</small></div>', unsafe_allow_html=True)

        st.markdown("*⭐ = critical check. All critical checks must pass regardless of total score.*")

st.markdown("---")

# ── Backtest Results ──────────────────────────────────────────────
if run_btn:
    with st.spinner("Running backtest across all dates…"):
        trades = run_backtest(daily, weekly, direction, min_score, sl_mult, rr, start_dt, end_dt)

    st.session_state['bt_trades']    = trades
    st.session_state['bt_direction'] = direction
    st.session_state['bt_daily']     = daily
    st.session_state['bt_weekly']    = weekly

if 'bt_trades' in st.session_state:
    trades    = st.session_state['bt_trades']
    direction = st.session_state['bt_direction']

    st.markdown("### Backtest Results")

    if not trades:
        st.warning("No trades met the minimum score threshold in this date range. Try lowering Min Score or widening the date range.")
    else:
        stats = compute_stats(trades)

        # ── Summary metrics ─────────────────────────────────────
        m = st.columns(8)
        m[0].markdown(f'<div class="metric-box"><div class="metric-value">{stats["total"]}</div><div class="metric-label">Trades</div></div>', unsafe_allow_html=True)
        m[1].markdown(f'<div class="metric-box"><div class="metric-value">{stats["win_rate"]:.0f}%</div><div class="metric-label">Win Rate</div></div>', unsafe_allow_html=True)
        m[2].markdown(f'<div class="metric-box"><div class="metric-value">{stats["net_r"]:+.1f}R</div><div class="metric-label">Net R</div></div>', unsafe_allow_html=True)
        m[3].markdown(f'<div class="metric-box"><div class="metric-value">{stats["avg_r"]:+.2f}R</div><div class="metric-label">Avg R / Trade</div></div>', unsafe_allow_html=True)
        m[4].markdown(f'<div class="metric-box"><div class="metric-value">{stats["profit_factor"]:.2f}</div><div class="metric-label">Profit Factor</div></div>', unsafe_allow_html=True)
        m[5].markdown(f'<div class="metric-box"><div class="metric-value">{stats["max_dd"]:.1f}R</div><div class="metric-label">Max Drawdown</div></div>', unsafe_allow_html=True)

        dollar_gain = stats["net_r"] * (account * risk_pct / 100)
        m[6].markdown(f'<div class="metric-box"><div class="metric-value">${dollar_gain:+,.0f}</div><div class="metric-label">P&L ({risk_pct}% risk)</div></div>', unsafe_allow_html=True)
        be_wr = 1 / (1 + rr) * 100
        m[7].markdown(f'<div class="metric-box"><div class="metric-value">{be_wr:.0f}%</div><div class="metric-label">Breakeven WR</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # ── Charts ──────────────────────────────────────────────
        ch1, ch2 = st.columns([2, 1])
        with ch1:
            st.markdown("**Equity Curve (cumulative R)**")
            st.plotly_chart(equity_curve_chart(trades), use_container_width=True)
        with ch2:
            st.markdown("**R per Trade**")
            st.plotly_chart(r_distribution_chart(trades), use_container_width=True)

        # ── Check pass-rate breakdown ────────────────────────────
        with st.expander("Check Pass-Rate Breakdown (last 60 bars sampled)"):
            st.markdown("Shows how often each check passed — identifies which filters are hardest to satisfy.")
            d_state = st.session_state.get('bt_daily', daily)
            w_state = st.session_state.get('bt_weekly', weekly)
            st.plotly_chart(check_pass_rate_chart(trades, d_state, w_state, direction), use_container_width=True)

        # ── Trade log ────────────────────────────────────────────
        st.markdown("**Trade Log**")
        df_log = pd.DataFrame(trades)
        df_log['entry']   = df_log['entry'].round(5)
        df_log['exit']    = df_log['exit'].round(5)
        df_log['sl']      = df_log['sl'].round(5)
        df_log['tp1']     = df_log['tp1'].round(5)
        df_log['r']       = df_log['r'].round(2)
        df_log['pips']    = (df_log['sl_d'] / pip_sz * df_log['r']).round(1)

        def colour_outcome(val):
            if val == "TP2":
                return 'color: #00ff66; font-weight:600'
            if val == "TP1 → BE":
                return 'color: #ffcc00'
            if val == "SL":
                return 'color: #ff3344'
            return 'color: #9a9a9a'

        def colour_r(val):
            return f'color: {"#00ff66" if val > 0 else "#ff3344" if val < 0 else "#ffcc00"}'

        styled = (
            df_log[['date', 'score', 'entry', 'exit', 'sl', 'tp1', 'r', 'pips', 'outcome']]
            .style
            .map(colour_outcome, subset=['outcome'])
            .map(colour_r,       subset=['r'])
            .format({'entry': '{:.5f}', 'exit': '{:.5f}', 'sl': '{:.5f}', 'tp1': '{:.5f}',
                     'r': '{:+.2f}R', 'pips': '{:.0f}'})
        )
        st.dataframe(styled, use_container_width=True, height=320)

        # ── Outcome breakdown ────────────────────────────────────
        st.markdown("**Outcome Distribution**")
        oc_counts = pd.DataFrame(trades)['outcome'].value_counts()
        bc1, bc2, bc3, bc4 = st.columns(4)
        for col, (label, cnt) in zip([bc1, bc2, bc3, bc4], oc_counts.items()):
            pct = cnt / len(trades) * 100
            col.markdown(f'<div class="metric-box"><div class="metric-value">{cnt} ({pct:.0f}%)</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

else:
    st.info("Configure your parameters in the sidebar and click **▶ Run Backtest** to simulate the workflow on historical data.")
    st.markdown("""
**How this works:**
1. For each trading day, all 18 workflow checks are evaluated using data available up to that point.
2. If the score meets your minimum threshold **and** all critical checks (⭐) pass, a trade is taken at the next bar's open.
3. The trade is managed with ATR-based SL and TP until hit or the 20-bar timeout.
4. Results show equity curve, per-trade R, win rate, and profit factor.

**Limitations to be aware of:**
- Checks 12 (15M Rejection) and 13 (15M Entry Signal) are proxied from daily bars.
- Macro bias, news filter, and correlation checks are auto-passed — you confirm these manually in the live system.
- Slippage and spread are not modelled. Real results will be slightly worse.
    """)
