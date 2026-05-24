import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import psycopg2
import psycopg2.extras
import yfinance as yf
import json

# ── Instrument registry ────────────────────────────────────────────────────────
INSTRUMENTS = {
    "EUR/USD": {"ticker": "EURUSD=X", "pip": 10.0,  "pip_size": 0.0001, "corr": "DXY ↑ = bearish"},
    "GBP/USD": {"ticker": "GBPUSD=X", "pip": 10.0,  "pip_size": 0.0001, "corr": "DXY ↑ = bearish"},
    "AUD/USD": {"ticker": "AUDUSD=X", "pip": 10.0,  "pip_size": 0.0001, "corr": "Gold ↑ = bullish"},
    "NZD/USD": {"ticker": "NZDUSD=X", "pip": 10.0,  "pip_size": 0.0001, "corr": "AUD/USD alignment"},
    "USD/JPY": {"ticker": "USDJPY=X", "pip": 9.09,  "pip_size": 0.01,   "corr": "US10Y yields aligned"},
    "USD/CHF": {"ticker": "USDCHF=X", "pip": 10.8,  "pip_size": 0.0001, "corr": "EUR/USD inverse"},
    "USD/CAD": {"ticker": "USDCAD=X", "pip": 7.4,   "pip_size": 0.0001, "corr": "Oil inverse"},
    "EUR/GBP": {"ticker": "EURGBP=X", "pip": 12.5,  "pip_size": 0.0001, "corr": "EUR/USD vs GBP/USD"},
    "EUR/JPY": {"ticker": "EURJPY=X", "pip": 9.09,  "pip_size": 0.01,   "corr": "Risk-on sentiment"},
    "GBP/JPY": {"ticker": "GBPJPY=X", "pip": 9.09,  "pip_size": 0.01,   "corr": "Volatility proxy"},
    "AUD/JPY": {"ticker": "AUDJPY=X", "pip": 9.09,  "pip_size": 0.01,   "corr": "Risk appetite gauge"},
    "EUR/AUD": {"ticker": "EURAUD=X", "pip": 6.3,   "pip_size": 0.0001, "corr": "EUR/USD vs AUD/USD"},
    "GBP/AUD": {"ticker": "GBPAUD=X", "pip": 6.3,   "pip_size": 0.0001, "corr": "GBP/USD vs AUD/USD"},
    "EUR/CAD": {"ticker": "EURCAD=X", "pip": 7.4,   "pip_size": 0.0001, "corr": "EUR/USD vs Oil"},
    "GBP/CAD": {"ticker": "GBPCAD=X", "pip": 7.4,   "pip_size": 0.0001, "corr": "GBP/USD vs Oil"},
    "USD/ZAR": {"ticker": "USDZAR=X", "pip": 0.55,  "pip_size": 0.0001, "corr": "Gold & risk sentiment"},
    "EUR/ZAR": {"ticker": "EURZAR=X", "pip": 0.55,  "pip_size": 0.0001, "corr": "EUR/USD + Gold"},
    "GBP/ZAR": {"ticker": "GBPZAR=X", "pip": 0.55,  "pip_size": 0.0001, "corr": "GBP/USD + Gold"},
    "🥇 Gold":  {"ticker": "GC=F",    "pip": 10.0,  "pip_size": 0.10,   "corr": "DXY inverse, VIX"},
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
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    .stApp{background:#0d1117;}
    section[data-testid="stSidebar"]{background:#161b22!important;border-right:1px solid #21262d;}
    .card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:20px;margin-bottom:16px;}
    .card-header{font-size:13px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#8b949e;margin-bottom:14px;}
    .chip-go  {background:#0d4a2f;color:#3fb950;border:1px solid #238636;border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600;display:inline-block;}
    .chip-wait{background:#4a2d0d;color:#e3b341;border:1px solid #9e6a03;border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600;display:inline-block;}
    .chip-no  {background:#4a0d0d;color:#f85149;border:1px solid #8b2d2d;border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600;display:inline-block;}
    .metric-box{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px;text-align:center;}
    .metric-value{font-size:22px;font-weight:700;color:#c9d1d9;}
    .metric-label{font-size:11px;color:#8b949e;margin-top:2px;font-weight:500;letter-spacing:.04em;text-transform:uppercase;}
    .section-title{font-size:16px;font-weight:700;color:#e6edf3;margin:24px 0 12px 0;padding-left:4px;border-left:3px solid #388bfd;}
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
    #MainMenu,footer,header{visibility:hidden;}
    .block-container{padding-top:1.5rem;max-width:1380px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════

def get_db_connection(cfg):
    return psycopg2.connect(
        host=cfg["host"], port=cfg["port"],
        dbname=cfg["dbname"], user=cfg["user"], password=cfg["password"]
    )

def init_db(cfg):
    """Create trades table if it doesn't exist."""
    try:
        conn = get_db_connection(cfg)
        cur = conn.cursor()
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_setups (
                                                                id             SERIAL PRIMARY KEY,
                                                                logged_at      TIMESTAMP NOT NULL DEFAULT NOW(),
                        instrument     VARCHAR(30),
                        ticker         VARCHAR(20),
                        direction      VARCHAR(10),
                        session        VARCHAR(20),
                        score          VARCHAR(20),
                        verdict        VARCHAR(20),
                        atr14          FLOAT,
                        atr20          FLOAT,
                        sl_pips        FLOAT,
                        tp1_pips       FLOAT,
                        tp2_pips       FLOAT,
                        lot_size       FLOAT,
                        risk_amount    FLOAT,
                        rr_tp1         FLOAT,
                        rr_tp2         FLOAT,
                        account_bal    FLOAT,
                        risk_pct       FLOAT,
                        checks_passed  INT,
                        checks_total   INT,
                        checks_detail  JSONB,
                        notes          TEXT
                        );
                    """)
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
                INSERT INTO trade_setups (
                    logged_at, instrument, ticker, direction, session,
                    score, verdict, atr14, atr20, sl_pips, tp1_pips, tp2_pips,
                    lot_size, risk_amount, rr_tp1, rr_tp2, account_bal, risk_pct,
                    checks_passed, checks_total, checks_detail, notes
                ) VALUES (
                             %(logged_at)s, %(instrument)s, %(ticker)s, %(direction)s, %(session)s,
                             %(score)s, %(verdict)s, %(atr14)s, %(atr20)s, %(sl_pips)s, %(tp1_pips)s, %(tp2_pips)s,
                             %(lot_size)s, %(risk_amount)s, %(rr_tp1)s, %(rr_tp2)s, %(account_bal)s, %(risk_pct)s,
                             %(checks_passed)s, %(checks_total)s, %(checks_detail)s, %(notes)s
                         )
                """, row)
    conn.commit()
    cur.close()
    conn.close()

def load_trades(cfg, limit=50):
    conn = get_db_connection(cfg)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
                SELECT id, logged_at, instrument, ticker, direction, session,
                    score, verdict, atr14, atr20, sl_pips, tp1_pips, tp2_pips,
                    lot_size, risk_amount, rr_tp1, rr_tp2, account_bal, risk_pct,
                    checks_passed, checks_total, notes
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
                            (low  - prev).abs()], axis=1).max(axis=1)
            return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        df["atr14"] = _atr(df["High"], df["Low"], df["Close"], 14)
        df["atr20"] = _atr(df["High"], df["Low"], df["Close"], 20)
        df = df.dropna(subset=["atr14", "atr20"])
        if df.empty:
            return None
        a14 = float(df["atr14"].iloc[-1])
        a20 = float(df["atr20"].iloc[-1])
        # Convert ATR to pips
        a14_pips = round(a14 / pip_size, 1)
        a20_pips = round(a20 / pip_size, 1)
        # SL = 1.5 × ATR14, TP1 = 2:1, TP2 = 3:1
        sl_pips   = round(a14_pips * 1.5, 1)
        tp1_pips  = round(sl_pips  * 2.0, 1)
        tp2_pips  = round(sl_pips  * 3.0, 1)
        return {
            "atr14": round(a14, 5), "atr20": round(a20, 5),
            "atr14_pips": a14_pips, "atr20_pips": a20_pips,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
            "atr_ok": a14 > a20,
        }
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

CHECKS_TOTAL = 18
for i in range(1, CHECKS_TOTAL + 1):
    if f"check_{i}" not in st.session_state:
        st.session_state[f"check_{i}"] = False

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
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Trade Setup")
    st.page_link("daliy-trading-checklist.py", label="Checklist", icon="📋")
    st.page_link("pages/correlations.py", label="Correlations", icon="🔗")
    st.page_link("pages/macro-bias.py", label="Macro Bias", icon="🌐")
    st.page_link("pages/news-filter.py", label="News Filter", icon="📰")
    st.page_link("pages/atr-volatility.py", label="ATR Volatility", icon="📊")
    st.page_link("pages/weekly-ema.py", label="Weekly EMA", icon="📉")
    st.page_link("pages/weekly-rsi.py", label="Weekly RSI", icon="📡")
    st.page_link("pages/daily-trend.py", label="📈 Daily Trend", icon="📈")
    st.divider()

    # Instrument
    inst_keys    = list(INSTRUMENTS.keys())
    if st.session_state.selected_instrument not in inst_keys:
        st.session_state.selected_instrument = inst_keys[0]

    selected_key = st.selectbox("Instrument", inst_keys,
                                index=inst_keys.index(st.session_state.selected_instrument))
    st.session_state.selected_instrument = selected_key
    inst_data    = INSTRUMENTS[selected_key]

    st.markdown(
        f'<span class="ticker-tag">{inst_data["ticker"]}</span>'
        f'<br><span style="font-size:11px;color:#8b949e;margin-top:4px;display:block;">'
        f'📡 {inst_data["corr"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    st.session_state.trade_direction = st.radio("Direction", ["LONG", "SHORT"], horizontal=True)
    st.session_state.session         = st.selectbox("Session", ["London", "New York", "Tokyo", "Sydney"])

    st.markdown("---")
    st.markdown("**📐 Position Sizing**")
    st.session_state.account_bal = st.number_input("Account Balance ($)", value=float(st.session_state.account_bal), step=500.0, format="%.2f")
    st.session_state.risk_pct    = st.slider("Risk per Trade (%)", 0.25, 3.0, float(st.session_state.risk_pct), 0.25)
    pip_val = inst_data["pip"]

    # ATR Fetch button
    st.markdown("---")
    st.markdown("**📊 ATR & Levels (Auto)**")
    if st.button("🔄 Fetch ATR + Calculate Levels", use_container_width=True):
        with st.spinner("Fetching market data…"):
            result = fetch_atr.clear() or fetch_atr(inst_data["ticker"], inst_data["pip_size"])
            st.session_state[f"atr_data_{selected_key}"] = result

    atr_data = st.session_state.get(f"atr_data_{selected_key}", None)
    if atr_data is None:
        # Auto-fetch on first load
        atr_data = fetch_atr(inst_data["ticker"], inst_data["pip_size"])
        st.session_state[f"atr_data_{selected_key}"] = atr_data

    if atr_data:
        atr_color = "#3fb950" if atr_data["atr_ok"] else "#f85149"
        atr_label = "✅ Volatile" if atr_data["atr_ok"] else "⚠️ Low Vol"
        st.markdown(f"""
        <div class="atr-box">
          <div class="atr-row"><span class="atr-label">ATR(14)</span><span class="atr-val">{atr_data['atr14']} &nbsp;·&nbsp; {atr_data['atr14_pips']} pips</span></div>
          <div class="atr-row"><span class="atr-label">ATR(20)</span><span class="atr-val">{atr_data['atr20']} &nbsp;·&nbsp; {atr_data['atr20_pips']} pips</span></div>
          <div class="atr-row"><span class="atr-label">ATR Filter</span><span style="color:{atr_color};font-weight:700;font-size:12px;">{atr_label}</span></div>
          <div class="atr-row"><span class="atr-label">Stop Loss</span><span class="atr-val-red">{atr_data['sl_pips']} pips (1.5×ATR14)</span></div>
          <div class="atr-row"><span class="atr-label">TP1</span><span class="atr-val-green">{atr_data['tp1_pips']} pips (2:1)</span></div>
          <div class="atr-row"><span class="atr-label">TP2</span><span class="atr-val-green">{atr_data['tp2_pips']} pips (3:1)</span></div>
        </div>
        """, unsafe_allow_html=True)
        sl_pips  = atr_data["sl_pips"]
        tp1_pips = atr_data["tp1_pips"]
        tp2_pips = atr_data["tp2_pips"]
        atr14    = atr_data["atr14"]
        atr20    = atr_data["atr20"]
        atr_ok   = atr_data["atr_ok"]
    else:
        st.warning("⚠️ Could not fetch live data. Enter manually.")
        sl_pips  = st.number_input("Stop Loss (pips)", value=20.0, step=1.0)
        tp1_pips = st.number_input("TP1 (pips)",       value=40.0, step=1.0)
        tp2_pips = st.number_input("TP2 (pips)",       value=60.0, step=1.0)
        atr14 = atr20 = 0.0
        atr_ok = None

    # PostgreSQL config
    st.markdown("---")
    st.markdown("**🗄️ PostgreSQL Database**")
    st.session_state.db_host = st.text_input("Host",     value=st.session_state.db_host)
    st.session_state.db_port = st.number_input("Port",   value=int(st.session_state.db_port), step=1)
    st.session_state.db_name = st.text_input("Database", value=st.session_state.db_name)
    st.session_state.db_user = st.text_input("User",     value=st.session_state.db_user)
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
        st.session_state.db_ok  = ok
        st.session_state.db_msg = msg

    badge_cls = "db-badge-ok" if st.session_state.db_ok else "db-badge-err"
    badge_txt = "✅ Connected" if st.session_state.db_ok else f"❌ {st.session_state.db_msg[:30]}"
    st.markdown(f'<span class="{badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Reset All Checks", use_container_width=True):
        for i in range(1, CHECKS_TOTAL + 1):
            st.session_state[f"check_{i}"] = False
        st.rerun()

# ══════════════════════════════════════════════════════════════════
# COMPUTED VALUES
# ══════════════════════════════════════════════════════════════════
checked     = sum(st.session_state[f"check_{i}"] for i in range(1, CHECKS_TOTAL + 1))
pct         = int(checked / CHECKS_TOTAL * 100)
risk_amount = st.session_state.account_bal * (st.session_state.risk_pct / 100)
lot_size    = round(risk_amount / sl_pips / pip_val, 2) if (sl_pips and pip_val) else 0.0
rr_tp1      = round(tp1_pips / sl_pips, 2) if sl_pips else 0.0
rr_tp2      = round(tp2_pips / sl_pips, 2) if sl_pips else 0.0

if   pct >= 80: signal = ("🟢 GO",   "chip-go",   "#3fb950")
elif pct >= 55: signal = ("🟡 WAIT", "chip-wait", "#e3b341")
else:           signal = ("🔴 PASS", "chip-no",   "#f85149")

prog_color = "#3fb950" if pct >= 80 else "#e3b341" if pct >= 55 else "#f85149"
direction_icon = "🔼" if st.session_state.trade_direction == "LONG" else "🔽"
direction_col  = "#3fb950" if st.session_state.trade_direction == "LONG" else "#f85149"
display_name   = selected_key.split(" ", 1)[-1] if selected_key[0] in "🥇" else selected_key

# ══════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════
k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
kpis = [
    (k1, f"{checked}/{CHECKS_TOTAL}", "Checks Passed", prog_color),
    (k2, f"{pct}%",                   "Confidence",    prog_color),
    (k3, f"{atr14:.5f}" if atr14 else "—", "ATR(14)", "#c9d1d9"),
    (k4, f"{atr20:.5f}" if atr20 else "—", "ATR(20)", "#c9d1d9"),
    (k5, f"{sl_pips}",                "SL Pips",       "#f85149"),
    (k6, f"{rr_tp1}:1",               "R:R TP1", "#3fb950" if rr_tp1 >= 2 else "#f85149"),
    (k7, f"{lot_size:.2f}",           "Lot Size",      "#388bfd"),
]
for col, val, lbl, color in kpis:
    with col:
        st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{color};">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

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

# ══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════
left, right = st.columns([3, 2], gap="large")

def render_check(num, title, desc):
    col_a, col_b = st.columns([0.07, 0.93])
    with col_a:
        val = st.checkbox("", value=st.session_state[f"check_{num}"], key=f"cb_{num}")
        st.session_state[f"check_{num}"] = val
    with col_b:
        icon = "✅" if st.session_state[f"check_{num}"] else "⬜"
        st.markdown(f"**{icon} #{num} — {title}**  \n<span style='font-size:12px;color:#8b949e;'>{desc}</span>", unsafe_allow_html=True)
    st.markdown("---")

# ── LEFT: Checklist ────────────────────────────────────────────────────────────
with left:
    atr_hint = (f"✅ ATR14 {atr14:.5f} > ATR20 {atr20:.5f} — volatility supports follow-through" if atr_ok
                else f"❌ ATR14 {atr14:.5f} ≤ ATR20 {atr20:.5f} — low volatility caution" if atr_ok is False
    else "Fetch ATR data from sidebar")

    st.markdown('<div class="section-title">🌐 Macro Bias & News Filter</div>', unsafe_allow_html=True)
    render_check(1,  "Macro bias confirmed",     "Rates, GDP, inflation all favour trade direction")
    render_check(2,  "No red-folder news",        "No high-impact news within 1 hour of entry window")
    render_check(3,  "Correlated markets align",  f"Expected correlation: {inst_data['corr']}")
    render_check(4,  "ATR above 20-period avg",   atr_hint)

    st.markdown('<div class="section-title">📅 Weekly Timeframe Bias</div>', unsafe_allow_html=True)
    render_check(5,  "Weekly EMA aligned",        "Weekly EMA direction matches your trade direction")
    render_check(6,  "Weekly RSI has room",        "RSI not already overbought (>70) or oversold (<30)")
    render_check(7,  "Weekly Swing ✅ Aligned",    "Weekly Swing tab daily confirmation shows ✅ Aligned")

    st.markdown('<div class="section-title">📆 Daily Trend Confirmation</div>', unsafe_allow_html=True)
    render_check(8,  "Daily trend intact",         "EMA20 > EMA50 for longs / EMA20 < EMA50 for shorts")
    render_check(9,  "Daily MACD momentum",         "MACD histogram turning in the direction of the trade")
    render_check(10, "Entry within session open",   "Entry window within first 2 hours of preferred session")

    st.markdown('<div class="section-title">⏱️ 4H Confluence Zone</div>', unsafe_allow_html=True)
    render_check(11, "Price at 4H confluence zone", "Price at overlap of Fibonacci + Pivot + EMA on 4H chart")
    render_check(12, "Min 2/3 confluence elements",  "At least 2 of 3 elements confirmed at zone")
    render_check(13, "Candlestick rejection on 15M", "Rejection candle or structural confirmation on 15M")

    st.markdown('<div class="section-title">🎯 Entry Signal & Risk Management</div>', unsafe_allow_html=True)
    render_check(14, "15M entry signal fired",      "Stochastic crossover + RSI reset on 15M chart")
    render_check(15, "Stop below/above structure",  f"SL = {sl_pips} pips (1.5 × ATR14) below/above structure")
    render_check(16, "R:R ≥ 2:1 to TP1",           f"TP1 = {tp1_pips} pips → R:R {rr_tp1}:1 {'✅' if rr_tp1 >= 2 else '❌ Below minimum'}")
    render_check(17, "Partial TP plan defined",      f"TP1 = {tp1_pips} pips · TP2 = {tp2_pips} pips · 50% closed at TP1")
    render_check(18, "Position size within limit",   f"Lot size: {lot_size:.2f} · Risk: ${risk_amount:.2f} ({st.session_state.risk_pct}% of ${st.session_state.account_bal:,.0f})")

# ── RIGHT: Dashboard ───────────────────────────────────────────────────────────
with right:
    # Verdict
    verdict_text = ("All minimum checks met — trade has edge" if pct >= 80 else
                    "Partial confluence — reassess key levels" if pct >= 55 else
                    "Insufficient confirmation — stay flat")
    verdict_icon = "🟢" if pct >= 80 else "🟡" if pct >= 55 else "🔴"
    st.markdown(f"""
    <div class="card" style="border-color:{prog_color}33;">
      <div class="card-header">🧭 Trade Verdict</div>
      <div style="text-align:center;padding:8px 0;">
        <div style="font-size:46px;margin-bottom:6px;">{verdict_icon}</div>
        <div style="font-size:22px;font-weight:700;color:{prog_color};margin-bottom:10px;">{signal[0].split(None,1)[1]}</div>
        <div class="{signal[1]}" style="font-size:14px;padding:5px 18px;">{pct}% confluence met</div>
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
    cats = ["Macro","No News","Corr.","ATR","Wk EMA","Wk RSI","Wk Swing",
            "Daily","MACD","Session","4H Zone","2/3 Conf","15M Cndle",
            "15M Sig","SL Str","R:R","Part TP","Pos Size"]
    vals = [1 if st.session_state[f"check_{i}"] else 0 for i in range(1,19)]
    fig  = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself',
        fillcolor='rgba(56,139,253,0.15)',
        line=dict(color='#388bfd', width=2), marker=dict(color='#388bfd', size=5),
    ))
    fig.update_layout(
        polar=dict(bgcolor='#0d1117',
                   angularaxis=dict(tickfont=dict(size=9,color='#8b949e'),linecolor='#21262d',gridcolor='#21262d'),
                   radialaxis=dict(range=[0,1],tickvals=[0,.5,1],ticktext=['','',''],gridcolor='#21262d',linecolor='#21262d')),
        paper_bgcolor='#161b22', plot_bgcolor='#161b22',
        showlegend=False, margin=dict(l=40,r=40,t=20,b=20), height=270,
    )
    st.markdown('<div class="card"><div class="card-header">🕸️ Confluence Radar</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))
    st.markdown('</div>', unsafe_allow_html=True)

    # Section breakdown
    sections = {
        "Macro & News (1–4)":   [1,2,3,4],
        "Weekly (5–7)":         [5,6,7],
        "Daily (8–10)":         [8,9,10],
        "4H Zone (11–13)":      [11,12,13],
        "Entry & Risk (14–18)": [14,15,16,17,18],
    }
    st.markdown('<div class="card"><div class="card-header">📊 Section Breakdown</div>', unsafe_allow_html=True)
    for sec_name, ids in sections.items():
        n = sum(st.session_state[f"check_{i}"] for i in ids)
        t = len(ids)
        p = int(n/t*100)
        c = "#3fb950" if p==100 else "#e3b341" if p>=50 else "#f85149"
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
        else:
            checks_detail = {f"check_{i}": st.session_state[f"check_{i}"] for i in range(1, CHECKS_TOTAL+1)}
            row = {
                "logged_at":     datetime.now(),
                "instrument":    display_name,
                "ticker":        inst_data["ticker"],
                "direction":     st.session_state.trade_direction,
                "session":       st.session_state.session,
                "score":         f"{checked}/{CHECKS_TOTAL}",
                "verdict":       signal[0],
                "atr14":         atr14,
                "atr20":         atr20,
                "sl_pips":       sl_pips,
                "tp1_pips":      tp1_pips,
                "tp2_pips":      tp2_pips,
                "lot_size":      lot_size,
                "risk_amount":   round(risk_amount, 2),
                "rr_tp1":        rr_tp1,
                "rr_tp2":        rr_tp2,
                "account_bal":   st.session_state.account_bal,
                "risk_pct":      st.session_state.risk_pct,
                "checks_passed": checked,
                "checks_total":  CHECKS_TOTAL,
                "checks_detail": json.dumps(checks_detail),
                "notes":         st.session_state.notes,
            }
            try:
                save_trade(db_cfg, row)
                st.success(f"✅ Saved to PostgreSQL — {display_name} · {signal[0]} · {datetime.now().strftime('%H:%M:%S')}")
            except Exception as e:
                st.error(f"❌ Save failed: {e}")

# ══════════════════════════════════════════════════════════════════
# TRADE LOG FROM POSTGRES
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📚 Trade Log — PostgreSQL")

col_reload, col_del, _ = st.columns([2, 2, 6])
with col_reload:
    reload = st.button("🔄 Refresh Log", use_container_width=True)
with col_del:
    del_id = st.number_input("Delete by ID", min_value=0, step=1, value=0, label_visibility="collapsed")

if st.session_state.db_ok:
    try:
        trades = load_trades(db_cfg, limit=100)
        if trades:
            df = pd.DataFrame(trades)
            df["logged_at"] = pd.to_datetime(df["logged_at"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(df, use_container_width=True, column_config={
                "id":           st.column_config.NumberColumn("ID",        width="small"),
                "logged_at":    st.column_config.TextColumn("Logged",      width="medium"),
                "instrument":   st.column_config.TextColumn("Pair"),
                "ticker":       st.column_config.TextColumn("Ticker"),
                "direction":    st.column_config.TextColumn("Dir",         width="small"),
                "session":      st.column_config.TextColumn("Session"),
                "score":        st.column_config.TextColumn("Score"),
                "verdict":      st.column_config.TextColumn("Verdict"),
                "atr14":        st.column_config.NumberColumn("ATR14",     format="%.5f"),
                "atr20":        st.column_config.NumberColumn("ATR20",     format="%.5f"),
                "sl_pips":      st.column_config.NumberColumn("SL pips",   format="%.1f"),
                "tp1_pips":     st.column_config.NumberColumn("TP1 pips",  format="%.1f"),
                "tp2_pips":     st.column_config.NumberColumn("TP2 pips",  format="%.1f"),
                "lot_size":     st.column_config.NumberColumn("Lots",      format="%.2f"),
                "risk_amount":  st.column_config.NumberColumn("Risk $",    format="%.2f"),
                "rr_tp1":       st.column_config.NumberColumn("R:R TP1",   format="%.2f"),
                "account_bal":  st.column_config.NumberColumn("Balance",   format="%.2f"),
                "risk_pct":     st.column_config.NumberColumn("Risk %",    format="%.2f"),
                "checks_passed":st.column_config.NumberColumn("✅ Checks"),
                "notes":        st.column_config.TextColumn("Notes",       width="large"),
            })
            if del_id > 0:
                if st.button(f"🗑️ Delete Row ID {del_id}", type="secondary"):
                    try:
                        delete_trade(db_cfg, del_id)
                        st.success(f"Deleted trade ID {del_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
        else:
            st.info("No trade setups saved yet. Complete the checklist and hit Save.")
    except Exception as e:
        st.error(f"❌ Could not load trades: {e}")
else:
    st.info("🔌 Connect to PostgreSQL in the sidebar to view saved trades.")

# Footer
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid #21262d;">
  📈 Daily Trading System · 18-Point Confluence Checklist · For educational purposes only
</div>
""", unsafe_allow_html=True)