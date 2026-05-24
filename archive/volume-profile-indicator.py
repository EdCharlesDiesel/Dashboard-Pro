import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forex Volume Profile",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:       #060a12;
    --panel:    #0c1220;
    --card:     #101827;
    --border:   #1a2640;
    --border2:  #243350;
    --cyan:     #06d6e8;
    --green:    #0eed8a;
    --red:      #ff3b6b;
    --amber:    #f59e0b;
    --purple:   #a78bfa;
    --text:     #cdd6f4;
    --muted:    #4a5568;
    --muted2:   #64748b;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { background: var(--bg) !important; color: var(--text) !important; }
.stApp { background: var(--bg) !important; }
[data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebarContent"] { padding-top: 1rem; }

/* suppress default streamlit metric styling */
div[data-testid="metric-container"] { display: none; }

.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 32px; font-weight: 800;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--purple) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em; margin: 0;
}
.app-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: var(--muted2); letter-spacing: 0.15em;
    text-transform: uppercase; margin-top: 2px;
}

.card-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin: 12px 0; }
.kpi {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px;
    transition: border-color .2s, transform .15s;
}
.kpi:hover { border-color: var(--border2); transform: translateY(-1px); }
.kpi-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    color: var(--muted); text-transform: uppercase; letter-spacing: .12em; margin-bottom: 5px;
}
.kpi-val {
    font-family: 'IBM Plex Mono', monospace; font-size: 17px; font-weight: 700; line-height: 1;
}
.kpi-sub { font-family: 'IBM Plex Mono', monospace; font-size: 10px; margin-top: 4px; }

.c  { color: var(--cyan); }
.g  { color: var(--green); }
.r  { color: var(--red); }
.a  { color: var(--amber); }
.p  { color: var(--purple); }
.m  { color: var(--muted2); }

.level-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 7px 12px; border-radius: 7px; margin: 4px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
}
.level-poc  { background: rgba(245,158,11,.10); border: 1px solid rgba(245,158,11,.25); }
.level-hvn  { background: rgba(14,237,138,.08); border: 1px solid rgba(14,237,138,.20); }
.level-lvn  { background: rgba(255,59,107,.08); border: 1px solid rgba(255,59,107,.18); }
.level-va   { background: rgba(6,214,232,.07);  border: 1px solid rgba(6,214,232,.18); }

.badge {
    display: inline-block; padding: 2px 8px; border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    font-weight: 700; letter-spacing: .08em;
}
.badge-poc { background: rgba(245,158,11,.2); color: var(--amber); }
.badge-hvn { background: rgba(14,237,138,.15); color: var(--green); }
.badge-lvn { background: rgba(255,59,107,.15); color: var(--red); }
.badge-sup { background: rgba(14,237,138,.12); color: #6ee7b7; }
.badge-res { background: rgba(255,59,107,.12); color: #fca5a5; }

.section-title {
    font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 600;
    color: var(--muted2); text-transform: uppercase; letter-spacing: .1em;
    margin: 16px 0 8px;
}
.info-box {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; margin: 8px 0;
}

hr { border-color: var(--border) !important; margin: 8px 0 !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

label { color: var(--muted2) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; }
.stSelectbox [data-baseweb="select"] { background: var(--card) !important; border-color: var(--border) !important; }
.stTextInput input { background: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────

FOREX_PAIRS = {
    "── Majors ──": None,
    "EUR/USD":  ("EURUSD=X", 5, "EUR/USD",  1.0850, 0.0003),
    "GBP/USD":  ("GBPUSD=X", 5, "GBP/USD",  1.2720, 0.0004),
    "USD/JPY":  ("USDJPY=X", 3, "USD/JPY",  149.50, 0.05),
    "USD/CHF":  ("USDCHF=X", 5, "USD/CHF",  0.9020, 0.0003),
    "AUD/USD":  ("AUDUSD=X", 5, "AUD/USD",  0.6540, 0.0003),
    "USD/CAD":  ("USDCAD=X", 5, "USD/CAD",  1.3620, 0.0004),
    "NZD/USD":  ("NZDUSD=X", 5, "NZD/USD",  0.6080, 0.0003),
    "── Minors ──": None,
    "EUR/GBP":  ("EURGBP=X", 5, "EUR/GBP",  0.8520, 0.0003),
    "EUR/JPY":  ("EURJPY=X", 3, "EUR/JPY",  162.50, 0.05),
    "GBP/JPY":  ("GBPJPY=X", 3, "GBP/JPY",  190.20, 0.07),
    "EUR/CHF":  ("EURCHF=X", 5, "EUR/CHF",  0.9620, 0.0003),
    "AUD/JPY":  ("AUDJPY=X", 3, "AUD/JPY",  97.80,  0.05),
    "CHF/JPY":  ("CHFJPY=X", 3, "CHF/JPY",  165.60, 0.05),
    "── Metals ──": None,
    "XAU/USD":  ("XAUUSD=X", 2, "Gold",     2320.0, 0.50),
    "XAG/USD":  ("XAGUSD=X", 3, "Silver",   27.50,  0.05),
    "── Exotics ──": None,
    "USD/MXN":  ("USDMXN=X", 4, "USD/MXN",  17.20,  0.005),
    "USD/ZAR":  ("USDZAR=X", 4, "USD/ZAR",  18.65,  0.005),
    "USD/TRY":  ("USDTRY=X", 4, "USD/TRY",  32.10,  0.01),
}

SESSIONS = {
    "Tokyo":   (0,  9,  "rgba(167,139,250,0.06)"),   # UTC 00:00-09:00
    "London":  (8,  17, "rgba(6,214,232,0.06)"),     # UTC 08:00-17:00
    "New York":(13, 22, "rgba(14,237,138,0.05)"),    # UTC 13:00-22:00
}

PERIOD_OPTS = {
    "1 Day":"1d", "5 Days":"5d", "1 Month":"1mo",
    "3 Months":"3mo", "6 Months":"6mo", "1 Year":"1y",
}
INTERVAL_OPTS = {
    "1 Minute":"1m","5 Minutes":"5m","15 Minutes":"15m",
    "30 Minutes":"30m","1 Hour":"1h","4 Hours":"4h","1 Day":"1d",
}
INTERVAL_LIMITS = {    # max period allowed per interval
    "1m":"1d","5m":"5d","15m":"1mo","30m":"1mo",
    "1h":"3mo","4h":"6mo","1d":"1y",
}


# ─── Data Functions ───────────────────────────────────────────────────────────

def generate_synthetic(pair_name: str, base_price: float, vol_per_bar: float,
                       n_bars: int = 600, interval_min: int = 60) -> pd.DataFrame:
    """Realistic synthetic forex OHLCV with trend + mean-reversion + sessions."""
    np.random.seed(hash(pair_name) % 2**31)
    freq = f"{interval_min}min" if interval_min < 1440 else "1D"
    idx = pd.date_range(end=pd.Timestamp.utcnow().floor("h"), periods=n_bars, freq=freq)

    price = base_price
    closes = []
    # add a few regime changes
    trend_str = np.random.choice([-1, 0, 1], n_bars,
                                 p=[0.15, 0.70, 0.15]).cumsum() * vol_per_bar * 0.15

    for i in range(n_bars):
        session_hour = idx[i].hour
        # higher volatility during London/NY overlap (13-17 UTC)
        mult = 1.8 if 13 <= session_hour <= 17 else (1.3 if 8 <= session_hour <= 22 else 0.7)
        ret = np.random.normal(trend_str[i] * 0.0001, vol_per_bar * mult)
        price = max(price * (1 + ret), base_price * 0.5)
        closes.append(price)

    closes = np.array(closes)
    spread = vol_per_bar * 0.3
    hl_range = np.abs(np.random.normal(0, vol_per_bar * 1.5, n_bars))
    highs = closes + hl_range * np.random.uniform(0.4, 1.0, n_bars)
    lows  = closes - hl_range * np.random.uniform(0.4, 1.0, n_bars)
    opens = np.roll(closes, 1); opens[0] = closes[0]

    volume = np.abs(np.random.normal(vol_per_bar * 1e6, vol_per_bar * 3e5, n_bars))
    # higher volume on bigger moves
    volume *= (1 + hl_range / (vol_per_bar * 2))

    return pd.DataFrame({"Open": opens, "High": highs,
                         "Low": lows,  "Close": closes,
                         "Volume": volume}, index=idx)


@st.cache_data(ttl=60, show_spinner=False)
def load_data(yf_sym: str, pair_name: str, base_price: float, vol_per_bar: float,
              period: str, interval: str, demo: bool) -> tuple[pd.DataFrame, bool]:
    if demo:
        iv_min = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}
        n = {"1d":390,"5d":500,"1mo":600,"3mo":600,"6mo":500,"1y":400}.get(period, 500)
        return generate_synthetic(pair_name, base_price, vol_per_bar, n, iv_min.get(interval, 60)), True
    try:
        import yfinance as yf
        df = yf.download(yf_sym, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("Empty data")
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        return df, False
    except Exception:
        iv_min = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}
        n = {"1d":390,"5d":500,"1mo":600,"3mo":600,"6mo":500,"1y":400}.get(period, 500)
        return generate_synthetic(pair_name, base_price, vol_per_bar, n, iv_min.get(interval, 60)), True


def compute_volume_profile(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    lo, hi = df["Low"].min(), df["High"].max()
    bins = np.linspace(lo, hi, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    vol = np.zeros(n_bins)
    for row in df.itertuples():
        mask = (bins[1:] >= row.Low) & (bins[:-1] <= row.High)
        n_hit = mask.sum()
        if n_hit:
            vol[mask] += row.Volume / n_hit
    return pd.DataFrame({"price": centers, "volume": vol})


def detect_levels(vp: pd.DataFrame, top_n: int) -> dict:
    from scipy.signal import argrelextrema
    vol, prices = vp["volume"].values, vp["price"].values
    poc_i = vol.argmax()
    poc   = prices[poc_i]

    lmax = argrelextrema(vol, np.greater, order=4)[0]
    hvn_cands = [(prices[i], vol[i]) for i in lmax
                 if i != poc_i and vol[i] >= vol.max() * 0.45]
    hvn_cands.sort(key=lambda x: -x[1])
    hvns = [p for p, _ in hvn_cands[:top_n]]

    lmin = argrelextrema(vol, np.less, order=3)[0]
    lvns = sorted([prices[i] for i in lmin if vol[i] <= vol.max() * 0.18],
                  key=lambda p: abs(p - poc))[:top_n]
    return {"poc": poc, "hvns": sorted(hvns), "lvns": sorted(lvns)}


def value_area(vp: pd.DataFrame, pct: float = 0.70) -> tuple:
    total = vp["volume"].sum()
    target = total * pct
    poc_i = vp["volume"].idxmax()
    lo = hi = poc_i
    acc = vp.loc[poc_i, "volume"]
    while acc < target:
        lo2 = max(lo - 1, 0); hi2 = min(hi + 1, len(vp) - 1)
        a_lo = vp.loc[lo2, "volume"] if lo2 != lo else 0
        a_hi = vp.loc[hi2, "volume"] if hi2 != hi else 0
        if a_lo == 0 and a_hi == 0: break
        if a_hi >= a_lo: hi = hi2; acc += a_hi
        else: lo = lo2; acc += a_lo
    return vp.loc[lo, "price"], vp.loc[hi, "price"]


def pip_value(price: float, decimals: int) -> float:
    return 10 ** (-decimals + 1) if decimals >= 3 else 0.01


def fmt_price(price: float, decimals: int) -> str:
    return f"{price:.{decimals}f}"


def pips(p1: float, p2: float, decimals: int) -> float:
    pv = pip_value(p1, decimals)
    return round(abs(p2 - p1) / pv, 1)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:18px;font-weight:800;
                background:linear-gradient(135deg,#06d6e8,#a78bfa);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                margin-bottom:4px;'>💱 FOREX VP</div>
    <div style='font-family:IBM Plex Mono,monospace;font-size:9px;color:#4a5568;
                text-transform:uppercase;letter-spacing:.15em;margin-bottom:16px;'>
    Volume Profile · S&R</div>
    """, unsafe_allow_html=True)

    pair_label = st.selectbox(
        "Currency Pair",
        [k for k in FOREX_PAIRS if FOREX_PAIRS[k] is not None],
        format_func=lambda k: k,
        index=0,
    )
    pair_info = FOREX_PAIRS[pair_label]
    yf_sym, pip_dec, pair_display, base_px, bar_vol = pair_info

    st.markdown("---")
    period_label  = st.selectbox("Lookback Period",  list(PERIOD_OPTS.keys()), index=2)
    interval_label = st.selectbox("Candle Interval", list(INTERVAL_OPTS.keys()), index=4)
    period   = PERIOD_OPTS[period_label]
    interval = INTERVAL_OPTS[interval_label]

    # guard incompatible combos
    max_period = INTERVAL_LIMITS.get(interval, "1y")
    period_rank = list(PERIOD_OPTS.values())
    if period_rank.index(period) > period_rank.index(max_period):
        period = max_period
        st.caption(f"⚠️ Interval limited to {max_period}")

    st.markdown("---")
    st.markdown("**Profile**")
    n_bins    = st.slider("Price Bins", 40, 200, 80, 10)
    top_n     = st.slider("S/R Levels (each)", 1, 5, 3)
    va_pct    = st.slider("Value Area %", 50, 90, 70, 5)

    st.markdown("**Overlays**")
    show_va      = st.checkbox("Value Area band", True)
    show_vwap    = st.checkbox("VWAP", True)
    show_ema     = st.checkbox("EMA 20 / 50", True)
    show_sessions= st.checkbox("Session shading", True)

    st.markdown("---")
    demo_mode = st.toggle("🎭 Demo Mode (synthetic data)", value=False,
                          help="Uses realistic generated data when live feed unavailable")

    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:10px;line-height:2.2;color:#4a5568;'>
    <span style='color:#f59e0b;'>━━</span> POC &nbsp;|&nbsp;
    <span style='color:#0eed8a;'>━━</span> HVN<br>
    <span style='color:#ff3b6b;'>╌╌</span> LVN &nbsp;|&nbsp;
    <span style='color:#06d6e8;'>░░</span> Value Area<br>
    <span style='color:#a78bfa;'>···</span> VWAP &nbsp;|&nbsp;
    <span style='color:#fbbf24;'>──</span> EMA20<br>
    <span style='color:#06d6e8;'>──</span> EMA50
    </div>
    """, unsafe_allow_html=True)


# ─── Load Data ────────────────────────────────────────────────────────────────

with st.spinner(""):
    df, is_demo = load_data(yf_sym, pair_label, base_px, bar_vol,
                            period, interval, demo_mode)

if df.empty:
    st.error("No data available. Try Demo Mode or another pair.")
    st.stop()

# ─── Compute Indicators ───────────────────────────────────────────────────────

vp     = compute_volume_profile(df, n_bins)
levels = detect_levels(vp, top_n)
va_lo, va_hi = value_area(vp, va_pct / 100)

if show_vwap:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (df["Volume"] * typical).cumsum() / df["Volume"].cumsum()

if show_ema:
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

cur  = float(df["Close"].iloc[-1])
prev = float(df["Close"].iloc[-2]) if len(df) > 1 else cur
chg  = cur - prev
pct  = chg / prev * 100
hi24 = float(df["High"].max())
lo24 = float(df["Low"].max())
poc  = levels["poc"]
spread_pips = round(np.random.uniform(0.5, 1.5), 1)   # simulated

# ─── Header ───────────────────────────────────────────────────────────────────

demo_badge = " <span style='font-size:10px;background:rgba(245,158,11,.15);color:#f59e0b;padding:2px 8px;border-radius:10px;font-family:IBM Plex Mono,monospace;'>DEMO</span>" if is_demo else ""
st.markdown(f"""
<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:2px;'>
  <div class='app-title'>FOREX VOLUME PROFILE</div>
  <div class='app-sub'>{pair_display} · {interval_label} · {period_label}{demo_badge}</div>
</div>
<hr/>
""", unsafe_allow_html=True)

# ─── KPI Row ──────────────────────────────────────────────────────────────────

chg_cls = "g" if chg >= 0 else "r"
arrow   = "▲" if chg >= 0 else "▼"
above_va = cur > va_hi
below_va = cur < va_lo
zone_lbl = ("Above VA 📈" if above_va else "Below VA 📉" if below_va else "Inside VA ◆")
zone_cls = "g" if above_va else ("r" if below_va else "c")
dist_poc  = pips(cur, poc, pip_dec)
poc_side  = "↑ above" if cur > poc else "↓ below"
va_size   = pips(va_lo, va_hi, pip_dec)

st.markdown(f"""
<div class='card-grid'>
  <div class='kpi'>
    <div class='kpi-label'>Last Price</div>
    <div class='kpi-val c'>{fmt_price(cur, pip_dec)}</div>
    <div class='kpi-sub {chg_cls}'>{arrow} {fmt_price(abs(chg), pip_dec)} ({pct:+.3f}%)</div>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Point of Control</div>
    <div class='kpi-val a'>{fmt_price(poc, pip_dec)}</div>
    <div class='kpi-sub m'>{dist_poc:.1f} pips {poc_side}</div>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Value Area Hi</div>
    <div class='kpi-val r'>{fmt_price(va_hi, pip_dec)}</div>
    <div class='kpi-sub m'>VA {va_pct}% band</div>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Value Area Lo</div>
    <div class='kpi-val g'>{fmt_price(va_lo, pip_dec)}</div>
    <div class='kpi-sub m'>{va_size:.0f} pip range</div>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Price Zone</div>
    <div class='kpi-val {zone_cls}' style='font-size:13px;'>{zone_lbl}</div>
    <div class='kpi-sub m'>{len(df):,} candles</div>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Session Range</div>
    <div class='kpi-val' style='font-size:13px;color:var(--text);'>{fmt_price(df["High"].iloc[-20:].max(), pip_dec)}</div>
    <div class='kpi-sub m'>/ {fmt_price(df["Low"].iloc[-20:].min(), pip_dec)}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Main Chart ───────────────────────────────────────────────────────────────

fig = make_subplots(
    rows=2, cols=2,
    row_heights=[0.78, 0.22],
    column_widths=[0.82, 0.18],
    shared_yaxes=True,
    shared_xaxes=True,
    horizontal_spacing=0.004,
    vertical_spacing=0.02,
)

# ── Session shading
if show_sessions and interval in ("1h", "4h", "30m", "15m", "5m", "1m"):
    dates = pd.Series(df.index.date).unique()
    for date in dates[-60:]:
        for sess, (sh, eh, col) in SESSIONS.items():
            s = pd.Timestamp(f"{date} {sh:02d}:00", tz="UTC")
            e = pd.Timestamp(f"{date} {eh:02d}:00", tz="UTC")
            fig.add_vrect(x0=s, x1=e, fillcolor=col, line_width=0, row=1, col=1)

# ── Candlesticks
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name=pair_display,
    increasing_line_color="#0eed8a", decreasing_line_color="#ff3b6b",
    increasing_fillcolor="rgba(14,237,138,0.75)", decreasing_fillcolor="rgba(255,59,107,0.75)",
    line_width=1,
), row=1, col=1)

# ── VWAP
if show_vwap:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["VWAP"], name="VWAP", mode="lines",
        line=dict(color="#a78bfa", width=1.4, dash="dot"), opacity=0.9,
    ), row=1, col=1)

# ── EMAs
if show_ema:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA20"], name="EMA 20", mode="lines",
        line=dict(color="#fbbf24", width=1.2), opacity=0.85,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA50"], name="EMA 50", mode="lines",
        line=dict(color="#06d6e8", width=1.2), opacity=0.75,
    ), row=1, col=1)

# ── Value Area band
if show_va:
    fig.add_hrect(y0=va_lo, y1=va_hi,
                  fillcolor="rgba(6,214,232,0.05)", line_width=0, row=1, col=1)
    for y, lbl in [(va_lo, "VAL"), (va_hi, "VAH")]:
        fig.add_hline(y=y, line=dict(color="#06d6e8", width=0.8, dash="dot"),
                      annotation_text=f" {lbl} {fmt_price(y, pip_dec)}",
                      annotation_font_size=9, annotation_font_color="#06d6e8",
                      row=1, col=1)

# ── POC
fig.add_hline(y=poc, line=dict(color="#f59e0b", width=2, dash="dash"),
              annotation_text=f" POC  {fmt_price(poc, pip_dec)}",
              annotation_font_size=10, annotation_font_color="#f59e0b",
              row=1, col=1)

# ── HVNs
for hvn in levels["hvns"]:
    role = "Resistance" if hvn > cur else "Support"
    fig.add_hline(y=hvn, line=dict(color="#0eed8a", width=1.2, dash="dash"),
                  annotation_text=f" HVN {role}  {fmt_price(hvn, pip_dec)}",
                  annotation_font_size=9, annotation_font_color="#0eed8a",
                  row=1, col=1)

# ── LVNs
for lvn in levels["lvns"]:
    fig.add_hline(y=lvn, line=dict(color="#ff3b6b", width=0.9, dash="longdash"),
                  annotation_text=f" LVN  {fmt_price(lvn, pip_dec)}",
                  annotation_font_size=9, annotation_font_color="#ff3b6b",
                  row=1, col=1)

# ── Volume bars (bottom)
colors_vol = ["#0eed8a" if c >= o else "#ff3b6b"
              for c, o in zip(df["Close"], df["Open"])]
fig.add_trace(go.Bar(
    x=df.index, y=df["Volume"], name="Volume",
    marker=dict(color=colors_vol, opacity=0.6, line_width=0),
    showlegend=False,
), row=2, col=1)

# ── Volume Profile histogram (right)
vp_norm = vp["volume"] / vp["volume"].max()
bar_colors = []
bin_step = (vp["price"].iloc[-1] - vp["price"].iloc[0]) / n_bins * 2.5
for p, v in zip(vp["price"], vp["volume"]):
    if abs(p - poc) < bin_step:
        bar_colors.append("#f59e0b")
    elif any(abs(p - h) < bin_step for h in levels["hvns"]):
        bar_colors.append("#0eed8a")
    elif any(abs(p - l) < bin_step for l in levels["lvns"]):
        bar_colors.append("#ff3b6b")
    else:
        bar_colors.append("rgba(6,214,232,0.45)")

fig.add_trace(go.Bar(
    x=vp["volume"], y=vp["price"], orientation="h",
    name="Volume Profile", showlegend=False,
    marker=dict(color=bar_colors, line_width=0),
    hovertemplate=f"Price: {fmt_price(0, pip_dec).replace('0','%{{y:,.{pip_dec}f}}')}<br>Vol: %{{x:,.0f}}<extra></extra>",
), row=1, col=2)

# ── Layout
BG = "#060a12"
GRID = "#111e35"
fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="IBM Plex Mono", color="#94a3b8", size=10),
    margin=dict(l=8, r=8, t=8, b=8),
    height=660,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(bgcolor="rgba(12,18,32,0.95)", bordercolor="#1a2640",
                borderwidth=1, font_size=10, x=0.01, y=0.99),
)

tick_fmt = f".{pip_dec}f"
for ax in ["yaxis", "yaxis2", "yaxis3", "yaxis4"]:
    fig.update_layout(**{ax: dict(
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=False, tickformat=tick_fmt,
        tickfont=dict(family="IBM Plex Mono", size=9),
    )})
for ax in ["xaxis", "xaxis2"]:
    fig.update_layout(**{ax: dict(
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=False, showspikes=True, spikecolor="#1e3a5f",
        tickfont=dict(family="IBM Plex Mono", size=9),
    )})

fig.update_xaxes(showticklabels=False, row=1, col=2)
fig.update_yaxes(showticklabels=False, row=1, col=2)
fig.update_xaxes(showticklabels=False, row=2, col=2)
fig.update_yaxes(showticklabels=False, row=2, col=2)
fig.update_yaxes(title_text="Volume", row=2, col=1,
                 title_font=dict(size=9, color="#4a5568"))

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})


# ─── S/R Levels Table ─────────────────────────────────────────────────────────

st.markdown("---")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='section-title'>🎯 Key Levels</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='info-box'>
      <div class='level-row level-poc'>
        <span><span class='badge badge-poc'>POC</span>&nbsp; {fmt_price(poc, pip_dec)}</span>
        <span style='color:#f59e0b;font-size:11px;'>{dist_poc:.1f} pip {poc_side}</span>
      </div>
      <div class='level-row level-va'>
        <span><span class='badge' style='background:rgba(6,214,232,.15);color:#06d6e8;'>VAH</span>&nbsp; {fmt_price(va_hi, pip_dec)}</span>
        <span style='color:#06d6e8;font-size:11px;'>{pips(cur, va_hi, pip_dec):.1f} pips</span>
      </div>
      <div class='level-row level-va'>
        <span><span class='badge' style='background:rgba(6,214,232,.12);color:#06d6e8;'>VAL</span>&nbsp; {fmt_price(va_lo, pip_dec)}</span>
        <span style='color:#06d6e8;font-size:11px;'>{pips(cur, va_lo, pip_dec):.1f} pips</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("<div class='section-title'>🟢 High Volume Nodes</div>", unsafe_allow_html=True)
    rows = ""
    for hvn in sorted(levels["hvns"], reverse=True):
        role = "RES" if hvn > cur else "SUP"
        cls  = "badge-res" if hvn > cur else "badge-sup"
        d    = pips(cur, hvn, pip_dec)
        rows += f"""
        <div class='level-row level-hvn'>
          <span><span class='badge {cls}'>{role}</span>&nbsp;
          <span style='color:#0eed8a;'>{fmt_price(hvn, pip_dec)}</span></span>
          <span style='color:#64748b;font-size:11px;'>{d:.1f} pips</span>
        </div>"""
    st.markdown(f"<div class='info-box'>{rows or '<span style=color:#4a5568;font-size:12px;>None detected</span>'}</div>",
                unsafe_allow_html=True)

with c3:
    st.markdown("<div class='section-title'>🔴 Low Volume Nodes</div>", unsafe_allow_html=True)
    rows = ""
    for lvn in sorted(levels["lvns"], reverse=True):
        d = pips(cur, lvn, pip_dec)
        rows += f"""
        <div class='level-row level-lvn'>
          <span><span class='badge badge-lvn'>LVN</span>&nbsp;
          <span style='color:#ff6b8a;'>{fmt_price(lvn, pip_dec)}</span></span>
          <span style='color:#64748b;font-size:11px;'>{d:.1f} pips · fast-move zone</span>
        </div>"""
    st.markdown(f"<div class='info-box'>{rows or '<span style=color:#4a5568;font-size:12px;>None detected</span>'}</div>",
                unsafe_allow_html=True)


# ─── Trade Bias ───────────────────────────────────────────────────────────────

st.markdown("<div class='section-title'>📊 Market Bias</div>", unsafe_allow_html=True)

# compute bias score
score = 0
reasons = []
if cur > poc: score += 1; reasons.append(("Bullish", "Price above POC", "g"))
else:         score -= 1; reasons.append(("Bearish", "Price below POC", "r"))
if cur > va_hi: score += 2; reasons.append(("Bullish", f"Price above Value Area Hi ({fmt_price(va_hi, pip_dec)})", "g"))
elif cur < va_lo: score -= 2; reasons.append(("Bearish", f"Price below Value Area Lo ({fmt_price(va_lo, pip_dec)})", "r"))
if show_ema and len(df) > 50:
    e20, e50 = float(df["EMA20"].iloc[-1]), float(df["EMA50"].iloc[-1])
    if e20 > e50: score += 1; reasons.append(("Bullish", "EMA20 > EMA50 (uptrend)", "g"))
    else:         score -= 1; reasons.append(("Bearish", "EMA20 < EMA50 (downtrend)", "r"))

bias_lbl = "STRONG BULLISH 🐂" if score >= 3 else \
    "BULLISH 📈" if score > 0 else \
        "STRONG BEARISH 🐻" if score <= -3 else \
            "BEARISH 📉" if score < 0 else "NEUTRAL ◆"
bias_col = "#0eed8a" if score > 0 else "#ff3b6b" if score < 0 else "#06d6e8"

reason_html = "".join(
    f"<span style='color:var(--{'green' if c=='g' else 'red'});margin-right:16px;font-size:11px;'>"
    f"{'▲' if c=='g' else '▼'} {r}</span>"
    for lbl, r, c in reasons
)
st.markdown(f"""
<div class='info-box' style='display:flex;align-items:center;gap:24px;flex-wrap:wrap;'>
  <div>
    <div class='kpi-label'>Composite Bias</div>
    <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:{bias_col};'>{bias_lbl}</div>
  </div>
  <div style='flex:1;'>{reason_html}</div>
  <div style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#4a5568;'>
    Score: {score:+d} / 4
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
data_src = "⚠️ Synthetic Demo Data" if is_demo else "✅ Live — Yahoo Finance"
st.markdown(f"""
<div style='text-align:center;margin-top:20px;padding:10px;border-top:1px solid #1a2640;
            font-family:IBM Plex Mono,monospace;font-size:10px;color:#2d3f5a;'>
  {pair_display} · {period_label} · {interval_label} · {len(df):,} bars ·
  {data_src} · Not financial advice
</div>
""", unsafe_allow_html=True)