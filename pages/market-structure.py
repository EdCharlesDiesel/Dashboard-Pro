import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Market Structure · Trading System",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .card{background:var(--secondary-background-color);border:1px solid var(--border,#21262d);border-radius:12px;padding:18px 20px;margin-bottom:14px;}
    .card-header{font-size:12px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:#8b949e;margin-bottom:12px;}
    .hero{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);border:1px solid #21262d;border-radius:16px;padding:24px 32px;margin-bottom:20px;position:relative;overflow:hidden;}
    .hero::before{content:'';position:absolute;top:-40%;right:-5%;width:300px;height:300px;background:radial-gradient(circle,rgba(56,139,253,.07) 0%,transparent 70%);border-radius:50%;}
    .metric-box{background:var(--background-color);border:1px solid var(--border,#21262d);border-radius:8px;padding:12px;text-align:center;}
    .metric-value{font-size:20px;font-weight:700;color:#c9d1d9;}
    .metric-label{font-size:10px;color:#8b949e;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}
    .section-title{font-size:15px;font-weight:700;color:#e6edf3;margin:20px 0 12px 0;padding-left:4px;border-left:3px solid #388bfd;}

    .struct-card{border-radius:10px;padding:13px 15px;margin-bottom:8px;border:1px solid #21262d;background:#161b22;}
    .struct-bull{border-left:4px solid #3fb950;}
    .struct-bear{border-left:4px solid #f85149;}
    .struct-choch{border-left:4px solid #e3b341;}
    .struct-neutral{border-left:4px solid #484f58;}

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
    "EUR/USD":    {"ticker": "EURUSD=X", "pip_size": 0.0001},
    "GBP/USD":    {"ticker": "GBPUSD=X", "pip_size": 0.0001},
    "AUD/USD":    {"ticker": "AUDUSD=X", "pip_size": 0.0001},
    "NZD/USD":    {"ticker": "NZDUSD=X", "pip_size": 0.0001},
    "USD/JPY":    {"ticker": "USDJPY=X", "pip_size": 0.01},
    "USD/CHF":    {"ticker": "USDCHF=X", "pip_size": 0.0001},
    "USD/CAD":    {"ticker": "USDCAD=X", "pip_size": 0.0001},
    "EUR/GBP":    {"ticker": "EURGBP=X", "pip_size": 0.0001},
    "EUR/JPY":    {"ticker": "EURJPY=X", "pip_size": 0.01},
    "GBP/JPY":    {"ticker": "GBPJPY=X", "pip_size": 0.01},
    "AUD/JPY":    {"ticker": "AUDJPY=X", "pip_size": 0.01},
    "EUR/AUD":    {"ticker": "EURAUD=X", "pip_size": 0.0001},
    "GBP/AUD":    {"ticker": "GBPAUD=X", "pip_size": 0.0001},
    "EUR/CAD":    {"ticker": "EURCAD=X", "pip_size": 0.0001},
    "GBP/CAD":    {"ticker": "GBPCAD=X", "pip_size": 0.0001},
    "USD/ZAR":    {"ticker": "USDZAR=X", "pip_size": 0.0001},
    "EUR/ZAR":    {"ticker": "EURZAR=X", "pip_size": 0.0001},
    "GBP/ZAR":    {"ticker": "GBPZAR=X", "pip_size": 0.0001},
    "🥇 Gold":    {"ticker": "GC=F",     "pip_size": 0.10},
    "🥈 Silver":  {"ticker": "SI=F",     "pip_size": 0.01},
    "🪙 Platinum":{"ticker": "PL=F",     "pip_size": 0.10},
}


# ══════════════════════════════════════════════════════════════════
# MARKET STRUCTURE DETECTION
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(ticker: str, interval: str, days: int) -> pd.DataFrame:
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=days)
        df    = yf.download(ticker, start=start, end=end,
                            interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def find_swing_points(df: pd.DataFrame, strength: int = 3) -> pd.DataFrame:
    """
    Detect swing highs and lows using a left/right window of `strength` bars.
    Returns df with SH (swing high) and SL (swing low) boolean columns.
    """
    highs = df["High"].values
    lows  = df["Low"].values
    n     = len(df)
    sh    = [False] * n
    sl    = [False] * n
    for i in range(strength, n - strength):
        # Swing high: higher than all bars left and right within window
        if highs[i] == max(highs[i - strength: i + strength + 1]):
            sh[i] = True
        # Swing low: lower than all bars left and right within window
        if lows[i] == min(lows[i - strength: i + strength + 1]):
            sl[i] = True
    df = df.copy()
    df["swing_high"] = sh
    df["swing_low"]  = sl
    return df


def analyse_structure(df: pd.DataFrame) -> dict:
    """
    Identify HH/HL (bullish), LH/LL (bearish), or CHoCH/BOS patterns from swing points.
    Returns structure dict with: trend, last_hh, last_hl, last_lh, last_ll, choch, bos.
    """
    sh_df = df[df["swing_high"]].copy()
    sl_df = df[df["swing_low"]].copy()

    if len(sh_df) < 2 or len(sl_df) < 2:
        return {
            "trend": "NEUTRAL", "label": "⏳ Insufficient data",
            "color": "#484f58", "card_cls": "struct-neutral",
            "hh": None, "hl": None, "lh": None, "ll": None,
            "choch": False, "bos": False,
            "sh_prices": [], "sh_dates": [],
            "sl_prices": [], "sl_dates": [],
        }

    sh_vals = sh_df["High"].values
    sl_vals = sl_df["Low"].values
    sh_dates = sh_df.index.tolist()
    sl_dates = sl_df.index.tolist()

    # Last two swing highs and lows
    hh = sh_vals[-1] > sh_vals[-2]   # last SH > prev SH → Higher High
    hl = sl_vals[-1] > sl_vals[-2]   # last SL > prev SL → Higher Low
    lh = sh_vals[-1] < sh_vals[-2]   # last SH < prev SH → Lower High
    ll = sl_vals[-1] < sl_vals[-2]   # last SL < prev SL → Lower Low

    # Break of Structure — price broke above last SH (bullish BOS) or below last SL (bearish BOS)
    last_close  = float(df["Close"].iloc[-1])
    prev_sh_val = sh_vals[-2] if len(sh_vals) >= 2 else None
    prev_sl_val = sl_vals[-2] if len(sl_vals) >= 2 else None
    bos_bull    = last_close > float(sh_vals[-1]) if prev_sh_val else False
    bos_bear    = last_close < float(sl_vals[-1]) if prev_sl_val else False
    bos         = bos_bull or bos_bear

    # Change of Character — recent structure shift: was bearish, now HH formed (or vice versa)
    choch = (hh and ll) or (lh and hl)   # conflicting signals = character change

    if hh and hl:
        trend, label, color, cls = "BULLISH", "📈 HH + HL — Bullish Structure", "#3fb950", "struct-bull"
    elif lh and ll:
        trend, label, color, cls = "BEARISH", "📉 LH + LL — Bearish Structure", "#f85149", "struct-bear"
    elif choch:
        trend, label, color, cls = "CHOCH", "🔄 CHoCH — Character Change", "#e3b341", "struct-choch"
    elif hh:
        trend, label, color, cls = "BULLISH", "📈 HH forming — Potential Bullish", "#3fb950", "struct-bull"
    elif ll:
        trend, label, color, cls = "BEARISH", "📉 LL forming — Potential Bearish", "#f85149", "struct-bear"
    else:
        trend, label, color, cls = "NEUTRAL", "⏳ Ranging / No clear structure", "#484f58", "struct-neutral"

    if bos_bull and trend == "BULLISH":
        label += " + BOS ▲"
    elif bos_bear and trend == "BEARISH":
        label += " + BOS ▼"

    return {
        "trend":      trend,
        "label":      label,
        "color":      color,
        "card_cls":   cls,
        "hh":         bool(hh),
        "hl":         bool(hl),
        "lh":         bool(lh),
        "ll":         bool(ll),
        "choch":      bool(choch),
        "bos":        bool(bos),
        "last_sh":    float(sh_vals[-1]),
        "prev_sh":    float(sh_vals[-2]) if len(sh_vals) >= 2 else None,
        "last_sl":    float(sl_vals[-1]),
        "prev_sl":    float(sl_vals[-2]) if len(sl_vals) >= 2 else None,
        "sh_prices":  [float(v) for v in sh_vals[-8:]],
        "sh_dates":   sh_dates[-8:],
        "sl_prices":  [float(v) for v in sl_vals[-8:]],
        "sl_dates":   sl_dates[-8:],
        "close":      float(df["Close"].iloc[-1]),
    }


def build_structure_chart(df: pd.DataFrame, pair: str, struct: dict,
                           show_n: int = 100) -> go.Figure:
    plot = df.tail(show_n).copy()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.80, 0.20], vertical_spacing=0.03)

    # Candles
    fig.add_trace(go.Candlestick(
        x=plot.index,
        open=plot["Open"], high=plot["High"],
        low=plot["Low"],   close=plot["Close"],
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="rgba(63,185,80,0.15)",
        decreasing_fillcolor="rgba(248,81,73,0.15)",
        name="Price", showlegend=False,
    ), row=1, col=1)

    # Swing highs (triangles down)
    sh_in_window = [(d, p) for d, p in zip(struct["sh_dates"], struct["sh_prices"])
                    if d in plot.index]
    if sh_in_window:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in sh_in_window],
            y=[p * 1.0003 for _, p in sh_in_window],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=10, color="#3fb950",
                        line=dict(width=1, color="#0d1117")),
            text=["SH" for _ in sh_in_window],
            textposition="top center",
            textfont=dict(size=9, color="#3fb950"),
            name="Swing High", showlegend=True,
        ), row=1, col=1)

    # Swing lows (triangles up)
    sl_in_window = [(d, p) for d, p in zip(struct["sl_dates"], struct["sl_prices"])
                    if d in plot.index]
    if sl_in_window:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in sl_in_window],
            y=[p * 0.9997 for _, p in sl_in_window],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=10, color="#f85149",
                        line=dict(width=1, color="#0d1117")),
            text=["SL" for _ in sl_in_window],
            textposition="bottom center",
            textfont=dict(size=9, color="#f85149"),
            name="Swing Low", showlegend=True,
        ), row=1, col=1)

    # Structure lines connecting swing highs and lows
    color = struct["color"]
    if len(sh_in_window) >= 2:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in sh_in_window],
            y=[p for _, p in sh_in_window],
            mode="lines", line=dict(color="#3fb950", width=1.2, dash="dot"),
            name="SH trendline", showlegend=False, opacity=0.6,
        ), row=1, col=1)
    if len(sl_in_window) >= 2:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in sl_in_window],
            y=[p for _, p in sl_in_window],
            mode="lines", line=dict(color="#f85149", width=1.2, dash="dot"),
            name="SL trendline", showlegend=False, opacity=0.6,
        ), row=1, col=1)

    # Current price
    fig.add_hline(
        y=struct["close"], line_dash="solid",
        line_color="#ffffff", line_width=1, opacity=0.7,
        annotation_text=f"  {struct['close']:.5f}",
        annotation_position="right",
        annotation_font=dict(size=10, color="#ffffff"),
        row=1, col=1,
    )

    # Volume
    vol_colors = ["rgba(63,185,80,0.4)" if c >= o else "rgba(248,81,73,0.4)"
                  for c, o in zip(plot["Close"], plot["Open"])]
    fig.add_trace(go.Bar(
        x=plot.index, y=plot["Volume"],
        marker_color=vol_colors, name="Volume", showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
        font=dict(family="Inter,sans-serif", size=11, color="#8b949e"),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(13,17,23,.6)", bordercolor="#21262d", borderwidth=1,
                    font=dict(size=10), orientation="h", y=1.02, x=0),
        margin=dict(l=10, r=100, t=10, b=10),
    )
    fig.update_yaxes(gridcolor="#21262d", zeroline=False)
    fig.update_xaxes(showgrid=False, linecolor="#21262d")
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🏗️ Market Structure")
    st.page_link("daily-trading-checklist.py",    label="00. Checklist",           icon="📋")
    st.page_link("pages/setup-ranker.py",         label="01. Setup Ranker",         icon="🎰")
    st.page_link("pages/macro-bias.py",           label="02. Macro Bias",           icon="🌐")
    st.page_link("pages/news-filter.py",          label="03. News Filter",          icon="📰")
    st.page_link("pages/correlations.py",         label="04. Correlations",         icon="🔗")
    st.page_link("pages/atr-volatility.py",       label="05. ATR Volatility",       icon="📊")
    st.page_link("pages/weekly-ema.py",           label="06. Weekly EMA",           icon="📉")
    st.page_link("pages/weekly-rsi.py",           label="07. Weekly RSI",           icon="📡")
    st.page_link("pages/weekly-swing.py",         label="08. Weekly Swing",         icon="🔄")
    st.page_link("pages/daily-trend.py",          label="09. Daily Trend",          icon="📈")
    st.page_link("pages/daily-macd.py",           label="10. Daily MACD",           icon="📊")
    st.page_link("pages/4H-confluence-zone.py",   label="11. 4H Confluence Zone",   icon="🎯")
    st.page_link("pages/confluence-checker.py",   label="12. 2/3 Confluence Check", icon="🔀")
    st.page_link("pages/15m-rejection.py",        label="13. 15M Rejection",        icon="🕯️")
    st.page_link("pages/15m-entry-signal.py",     label="14. 15M Entry Signal",     icon="⚡")
    st.page_link("pages/stop-structure.py",       label="15. Stop Structure",       icon="🛡️")
    st.page_link("pages/rr-calculator.py",        label="16. R:R Calculator",       icon="⚖️")
    st.page_link("pages/trade-journal.py",        label="17. Trade Journal",        icon="📓")
    st.page_link("pages/market-structure.py",     label="18. Market Structure",     icon="🏗️")
    st.page_link("pages/backtest-workflow.py",    label="19. Backtest Workflow",    icon="🧪")
    st.divider()

    tf_options = {
        "4H (recommended)": ("1h",  90, "4h"),
        "Daily":            ("1d", 300, None),
        "Weekly":           ("1d", 730, "1W"),
        "1H":               ("1h",  30, None),
    }
    selected_tf  = st.selectbox("Timeframe", list(tf_options.keys()), index=0)
    interval, days, resample = tf_options[selected_tf]

    swing_strength = st.slider("Swing strength (bars each side)", 2, 8, 3, step=1,
                                help="Higher = fewer but more significant swing points")
    show_candles   = st.slider("Candles on chart", 60, 200, 100, step=20)

    inst_keys = list(INSTRUMENTS.keys())
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]
    focus_pair = st.selectbox("Detail Pair", inst_keys,
                               index=inst_keys.index(default_inst))

    st.divider()
    if st.button("🔄 Refresh All", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero">
  <div style="font-size:26px;font-weight:700;color:#e6edf3;">🏗️ Market Structure Scanner</div>
  <div style="color:#8b949e;font-size:13px;margin-top:4px;">
    HH/HL · LH/LL · CHoCH · Break of Structure · {selected_tf} · 21 Instruments
  </div>
  <div style="font-size:12px;color:#388bfd;margin-top:6px;">
    🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M')} · Swing strength: {swing_strength} bars
  </div>
</div>
""", unsafe_allow_html=True)

tab_scan, tab_detail = st.tabs(["📊 All-Pairs Scanner", "🔍 Pair Detail"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — SCANNER
# ══════════════════════════════════════════════════════════════════

with tab_scan:
    prog = st.progress(0, text="Scanning market structure…")
    scan_results = []
    all_items    = list(INSTRUMENTS.items())
    n_inst       = len(all_items)

    for idx, (pair_name, info) in enumerate(all_items):
        prog.progress((idx + 1) / n_inst, text=f"Scanning {pair_name}…")
        df_s = fetch_ohlcv(info["ticker"], interval, days)

        if df_s.empty or len(df_s) < 20:
            scan_results.append({"pair": pair_name, "ok": False})
            continue

        if resample:
            rule = "4h" if resample == "4h" else resample
            df_s = df_s.resample(rule).agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last",  "Volume": "sum",
            }).dropna()

        if len(df_s) < 20:
            scan_results.append({"pair": pair_name, "ok": False})
            continue

        df_s    = find_swing_points(df_s, swing_strength)
        struct  = analyse_structure(df_s)
        scan_results.append({"pair": pair_name, "ok": True, "struct": struct, "df": df_s})

    prog.empty()

    # ── Summary counts ──────────────────────────────────────────
    cnt_bull    = sum(1 for r in scan_results if r.get("ok") and r["struct"]["trend"] == "BULLISH")
    cnt_bear    = sum(1 for r in scan_results if r.get("ok") and r["struct"]["trend"] == "BEARISH")
    cnt_choch   = sum(1 for r in scan_results if r.get("ok") and r["struct"]["trend"] == "CHOCH")
    cnt_neutral = sum(1 for r in scan_results if r.get("ok") and r["struct"]["trend"] == "NEUTRAL")
    cnt_err     = sum(1 for r in scan_results if not r.get("ok"))
    cnt_bos     = sum(1 for r in scan_results if r.get("ok") and r["struct"]["bos"])

    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    for col_, val_, lbl_, c_ in [
        (sc1, cnt_bull,    "Bullish (HH/HL)",  "#3fb950"),
        (sc2, cnt_bear,    "Bearish (LH/LL)",  "#f85149"),
        (sc3, cnt_choch,   "CHoCH",            "#e3b341"),
        (sc4, cnt_neutral, "Neutral/Ranging",  "#484f58"),
        (sc5, cnt_bos,     "Break of Struct",  "#388bfd"),
        (sc6, cnt_err,     "No Data",          "#484f58"),
    ]:
        with col_:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="color:{c_};">{val_}</div>'
                f'<div class="metric-label">{lbl_}</div>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Card grid ────────────────────────────────────────────────
    cols3 = st.columns(3)
    for i, r in enumerate(scan_results):
        pair_name = r["pair"]
        is_sel    = pair_name == focus_pair
        border    = "border:2px solid #388bfd;" if is_sel else "border:1px solid #21262d;"

        if not r.get("ok"):
            card_content = '<span style="font-size:11px;color:#484f58;">No data</span>'
        else:
            s = r["struct"]
            color = s["color"]
            label = s["label"]
            bos_badge = (
                '<span style="background:rgba(56,139,253,.15);border:1px solid rgba(56,139,253,.5);'
                'border-radius:4px;padding:2px 7px;font-size:10px;color:#388bfd;font-weight:600;margin-left:6px;">'
                'BOS</span>'
            ) if s["bos"] else ""
            price_fmt = f"{s['close']:.5f}" if s["close"] < 100 else f"{s['close']:.3f}"
            hh_dot = f'<span style="color:#3fb950;font-size:10px;margin-right:4px;">{"●" if s["hh"] else "○"}HH</span>'
            hl_dot = f'<span style="color:#3fb950;font-size:10px;margin-right:4px;">{"●" if s["hl"] else "○"}HL</span>'
            lh_dot = f'<span style="color:#f85149;font-size:10px;margin-right:4px;">{"●" if s["lh"] else "○"}LH</span>'
            ll_dot = f'<span style="color:#f85149;font-size:10px;margin-right:4px;">{"●" if s["ll"] else "○"}LL</span>'

            card_content = (
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:13px;font-weight:700;color:#e6edf3;">{pair_name}</span>'
                f'<span style="font-size:11px;font-weight:600;color:{color};">'
                f'{label.split(" — ")[0] if " — " in label else label}'
                f'</span>{bos_badge}'
                f'</div>'
                f'<div style="margin-top:6px;font-size:11px;color:#8b949e;">'
                f'{hh_dot}{hl_dot}{lh_dot}{ll_dot}'
                f'</div>'
                f'<div style="margin-top:4px;font-size:11px;color:#484f58;font-family:monospace;">{price_fmt}</div>'
            )

        with cols3[i % 3]:
            st.markdown(
                f'<div style="background:#161b22;{border}border-radius:8px;'
                f'padding:12px 14px;margin-bottom:8px;">{card_content}</div>',
                unsafe_allow_html=True)

    # ── Summary table ────────────────────────────────────────────
    with st.expander("📋 Full Scanner Results"):
        table_rows = []
        for r in scan_results:
            if not r.get("ok"):
                table_rows.append({"Pair": r["pair"], "Structure": "No Data",
                                   "HH": "—", "HL": "—", "LH": "—", "LL": "—",
                                   "CHoCH": "—", "BOS": "—", "Price": "—"})
            else:
                s = r["struct"]
                table_rows.append({
                    "Pair":      r["pair"],
                    "Structure": s["trend"],
                    "Label":     s["label"],
                    "HH":        "✅" if s["hh"] else "",
                    "HL":        "✅" if s["hl"] else "",
                    "LH":        "✅" if s["lh"] else "",
                    "LL":        "✅" if s["ll"] else "",
                    "CHoCH":     "🔄" if s["choch"] else "",
                    "BOS":       "⚡" if s["bos"] else "",
                    "Price":     f"{s['close']:.5f}" if s["close"] < 100 else f"{s['close']:.3f}",
                })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — PAIR DETAIL
# ══════════════════════════════════════════════════════════════════

with tab_detail:
    detail_ok = True
    info      = INSTRUMENTS[focus_pair]

    with st.spinner(f"Loading {focus_pair} — {selected_tf}…"):
        df = fetch_ohlcv(info["ticker"], interval, days)

    if df.empty or len(df) < 20:
        st.error(f"Not enough data for {focus_pair}. Try a different timeframe.")
        detail_ok = False

    if detail_ok:
        if resample:
            rule = "4h" if resample == "4h" else resample
            df = df.resample(rule).agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last",  "Volume": "sum",
            }).dropna()

        df     = find_swing_points(df, swing_strength)
        struct = analyse_structure(df)
        color  = struct["color"]

        # ── Structure verdict banner ──────────────────────────────
        banner_bg = {"BULLISH": "#0d2f1a", "BEARISH": "#2f0d0d",
                     "CHOCH": "#2a1f00", "NEUTRAL": "#1a1a1a"}.get(struct["trend"], "#1a1a1a")
        banner_bd = {"BULLISH": "#238636", "BEARISH": "#8b2d2d",
                     "CHOCH": "#9e6a03", "NEUTRAL": "#30363d"}.get(struct["trend"], "#30363d")

        bos_text = ""
        if struct["bos"]:
            bos_text = '<span style="background:rgba(56,139,253,.15);border:1px solid #388bfd;border-radius:4px;padding:3px 10px;font-size:12px;color:#388bfd;font-weight:700;margin-left:10px;">⚡ BREAK OF STRUCTURE</span>'

        st.markdown(f"""
        <div style="background:{banner_bg};border:2px solid {banner_bd};border-radius:14px;
                    padding:20px 24px;margin-bottom:20px;">
          <div style="font-size:22px;font-weight:800;color:{color};letter-spacing:1px;">
            {struct["label"]} {bos_text}
          </div>
          <div style="font-size:12px;color:#8b949e;margin-top:8px;">
            {focus_pair} · {selected_tf} · Swing strength: {swing_strength} bars each side ·
            {datetime.now().strftime('%H:%M UTC')}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI row ───────────────────────────────────────────────
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        for col_, val_, lbl_, c_ in [
            (k1, "✅ HH" if struct["hh"] else "❌ HH", "Higher High", "#3fb950" if struct["hh"] else "#484f58"),
            (k2, "✅ HL" if struct["hl"] else "❌ HL", "Higher Low",  "#3fb950" if struct["hl"] else "#484f58"),
            (k3, "✅ LH" if struct["lh"] else "❌ LH", "Lower High",  "#f85149" if struct["lh"] else "#484f58"),
            (k4, "✅ LL" if struct["ll"] else "❌ LL", "Lower Low",   "#f85149" if struct["ll"] else "#484f58"),
            (k5, "🔄 YES" if struct["choch"] else "NO", "CHoCH",      "#e3b341" if struct["choch"] else "#484f58"),
            (k6, "⚡ YES" if struct["bos"] else "NO",   "BOS",        "#388bfd" if struct["bos"] else "#484f58"),
        ]:
            with col_:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value" style="color:{c_};font-size:16px;">{val_}</div>'
                    f'<div class="metric-label">{lbl_}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

        # ── Last swing levels ─────────────────────────────────────
        st.markdown('<div class="section-title">📐 Last Swing Levels</div>', unsafe_allow_html=True)
        lv1, lv2, lv3, lv4 = st.columns(4)
        for col_, val_, lbl_, c_ in [
            (lv1, f"{struct['last_sh']:.5f}" if struct["last_sh"] < 100 else f"{struct['last_sh']:.3f}",
             "Last Swing High", "#3fb950"),
            (lv2, f"{struct['prev_sh']:.5f}" if struct.get("prev_sh") and struct["prev_sh"] < 100
             else (f"{struct['prev_sh']:.3f}" if struct.get("prev_sh") else "—"),
             "Prev Swing High", "#8b949e"),
            (lv3, f"{struct['last_sl']:.5f}" if struct["last_sl"] < 100 else f"{struct['last_sl']:.3f}",
             "Last Swing Low", "#f85149"),
            (lv4, f"{struct['prev_sl']:.5f}" if struct.get("prev_sl") and struct["prev_sl"] < 100
             else (f"{struct['prev_sl']:.3f}" if struct.get("prev_sl") else "—"),
             "Prev Swing Low", "#8b949e"),
        ]:
            with col_:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-value" style="color:{c_};font-size:15px;">{val_}</div>'
                    f'<div class="metric-label">{lbl_}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

        # ── Chart ─────────────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Structure Chart</div>', unsafe_allow_html=True)
        fig = build_structure_chart(df, focus_pair, struct, show_candles)
        st.plotly_chart(fig, use_container_width=True)

        # ── Interpretation ────────────────────────────────────────
        with st.expander("📖 Structure Interpretation Guide"):
            st.markdown("""
| Pattern | Meaning | Trade bias |
|---------|---------|-----------|
| **HH + HL** | Higher High + Higher Low — textbook uptrend | LONG only |
| **LH + LL** | Lower High + Lower Low — textbook downtrend | SHORT only |
| **CHoCH** | Change of Character — structure is shifting | Wait for confirmation |
| **BOS** | Break of Structure — price took out last swing level | Direction of break |
| **HH only** | New high but no HL confirmation yet | Cautiously bullish |
| **LL only** | New low but no LH confirmation yet | Cautiously bearish |
| **Neutral** | Neither HH/HL nor LH/LL | Avoid — ranging market |

**Rule:** Only trade in the direction of the prevailing structure.
LONG only when HH + HL on 4H. SHORT only when LH + LL on 4H.
Use CHoCH as an early warning to switch bias — wait for BOS to confirm.
            """)

# Footer
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid #21262d;">
  🏗️ Market Structure Scanner · HH/HL · LH/LL · CHoCH · BOS · Not financial advice
</div>
""", unsafe_allow_html=True)
