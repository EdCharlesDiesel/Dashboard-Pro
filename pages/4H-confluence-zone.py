import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="4H Confluence Zone · Trading System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS — dark terminal / trading aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{ font-family:'Inter',sans-serif; }
.stApp{ background:#0d1117; }
section[data-testid="stSidebar"]{ background:#161b22!important; border-right:1px solid #21262d; }
#MainMenu,footer,header{ visibility:hidden; }
[data-testid="stSidebarCollapsedControl"]{ visibility:visible!important; }
[data-testid="stSidebarNav"]{ display:none; }
.block-container{ padding-top:1.5rem; max-width:1380px; }

/* Confluence zone cards — teal accent kept intentionally */
.metric-card{
    background:linear-gradient(135deg,#111827 0%,#1a2235 100%);
    border:1px solid #1e3a5f;
    border-left:3px solid #00d4ff;
    border-radius:8px;
    padding:16px 20px;
    margin-bottom:10px;
}

.confluence-badge{
    display:inline-block;
    background:linear-gradient(90deg,rgba(0,212,255,.13),rgba(255,107,53,.13));
    border:1px solid rgba(0,212,255,.33);
    border-radius:4px;
    padding:4px 10px;
    font-size:.78em;
    color:#00d4ff;
    font-weight:600;
    letter-spacing:.05em;
}
.confluence-badge.strong-badge{
    background:linear-gradient(90deg,rgba(255,107,53,.13),rgba(255,107,53,.27));
    border-color:rgba(255,107,53,.53);
    color:#ff6b35;
}

hr{ border-color:#21262d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_4h_data(ticker: str, days: int = 90) -> pd.DataFrame:
    end = datetime.now(pytz.utc)
    start = end - timedelta(days=days)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1h",
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        return pd.DataFrame()

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    df_4h = (
        df.resample("4h")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )

    return df_4h


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_fibonacci_levels(high: float, low: float) -> dict[str, float]:
    diff = high - low

    return {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.500 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100%": low,
    }


def calc_pivot_points(df: pd.DataFrame) -> dict[str, float]:
    if len(df) < 2:
        last = df.iloc[-1]
    else:
        last = df.iloc[-2]

    high = float(last["High"])
    low = float(last["Low"])
    close = float(last["Close"])

    pp = (high + low + close) / 3

    return {
        "PP": pp,
        "R1": 2 * pp - low,
        "R2": pp + (high - low),
        "R3": high + 2 * (pp - low),
        "S1": 2 * pp - high,
        "S2": pp - (high - low),
        "S3": low - 2 * (high - pp),
    }


def find_confluence_zones(
        price: float,
        fib_levels: dict[str, float],
        pivots: dict[str, float],
        ema_values: dict[str, float],
        tolerance_pct: float = 0.005,
) -> list[dict]:
    tolerance = price * tolerance_pct
    all_levels = []

    for label, value in fib_levels.items():
        all_levels.append({"source": "Fib", "label": label, "value": float(value)})

    for label, value in pivots.items():
        all_levels.append({"source": "Pivot", "label": label, "value": float(value)})

    for label, value in ema_values.items():
        all_levels.append({"source": "EMA", "label": label, "value": float(value)})

    zones = []
    used = set()

    for i, level_a in enumerate(all_levels):
        if i in used:
            continue

        cluster = [level_a]

        for j, level_b in enumerate(all_levels):
            if j <= i or j in used:
                continue

            if abs(level_a["value"] - level_b["value"]) <= tolerance:
                cluster.append(level_b)
                used.add(j)

        if len(cluster) >= 2:
            used.add(i)

            avg_value = float(np.mean([item["value"] for item in cluster]))
            sources = sorted({item["source"] for item in cluster})
            labels = [f"{item['source']}:{item['label']}" for item in cluster]
            distance_pct = (avg_value - price) / price * 100

            zones.append(
                {
                    "value": avg_value,
                    "sources": sources,
                    "labels": labels,
                    "strength": len(cluster),
                    "distance_pct": distance_pct,
                }
            )

    zones.sort(key=lambda item: abs(item["distance_pct"]))

    return zones


def safe_rgba(hex_color: str, opacity: float) -> str:
    value = hex_color.strip().lstrip("#")

    if len(value) != 6:
        raise ValueError("Expected 6-digit hex color like #00d4ff")

    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)

    opacity = max(0.0, min(1.0, opacity))

    return f"rgba({r}, {g}, {b}, {opacity})"


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
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

with st.sidebar:
    st.markdown("### 🎯 4H Confluence Zone")
    st.page_link("daily-trading-checklist.py", label="00. Checklist", icon="📋")
    st.page_link("pages/macro-bias.py", label="01. Macro Bias", icon="🌐")
    st.page_link("pages/news-filter.py", label="02. News Filter", icon="📰")
    st.page_link("pages/correlations.py", label="03. Correlations", icon="🔗")
    st.page_link("pages/atr-volatility.py", label="04. ATR Volatility", icon="📊")
    st.page_link("pages/weekly-ema.py", label="05. Weekly EMA", icon="📉")
    st.page_link("pages/weekly-rsi.py", label="06. Weekly RSI", icon="📡")
    st.page_link("pages/weekly-swing.py", label="07. Weekly Swing", icon="🔄")
    st.page_link("pages/daily-trend.py", label="08. Daily Trend", icon="📈")
    st.page_link("pages/daily-macd.py", label="09. Daily MACD", icon="📊")
    st.page_link("pages/4H-confluence-zone.py", label="10. 4H Confluence Zone", icon="🎯")
    st.page_link("pages/confluence-checker.py", label="11. 2/3 Confluence Check", icon="🔀")
    st.page_link("pages/15m-rejection.py", label="12. 15M Rejection", icon="🕯️")
    st.page_link("pages/15m-entry-signal.py", label="13. 15M Entry Signal", icon="⚡")
    st.page_link("pages/stop-structure.py", label="14. Stop Structure", icon="🛡️")
    st.page_link("pages/rr-calculator.py", label="15. R:R Calculator", icon="⚖️")
    st.divider()
    st.markdown("### ⚙️ Settings")

    inst_keys = list(INSTRUMENTS.keys())
    default_inst = st.session_state.get("selected_instrument", "EUR/USD")
    if default_inst not in inst_keys:
        default_inst = inst_keys[0]
    selected_pair = st.selectbox("Instrument", inst_keys, index=inst_keys.index(default_inst))
    ticker = INSTRUMENTS[selected_pair]["ticker"]
    lookback_days = st.slider("Lookback (days)", 30, 180, 90, step=15)

    st.markdown("---")
    st.markdown("**EMA Periods**")

    ema_short = st.number_input("EMA Short", value=20, min_value=5, max_value=200, step=1)
    ema_mid = st.number_input("EMA Mid", value=50, min_value=10, max_value=500, step=1)
    ema_long = st.number_input("EMA Long", value=200, min_value=20, max_value=800, step=5)

    st.markdown("---")

    confluence_tol = st.slider(
        "Confluence Tolerance (%)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
    ) / 100

    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────
#  Hero card
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);
            border:1px solid #21262d; border-radius:16px; padding:24px 28px;
            margin-bottom:20px; position:relative; overflow:hidden;">
  <div style="font-size:24px; font-weight:700; color:#e6edf3;">🎯 4H Confluence Zone</div>
  <div style="color:#8b949e; font-size:13px; margin-top:4px;">
    Fibonacci · Pivot Points · EMA overlap · {selected_pair} · 4H chart
  </div>
  <div style="font-size:12px; color:#388bfd; margin-top:6px;">
    Check #10 — 4H Confluence Zone · {datetime.now().strftime('%A %d %B %Y  |  %H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────
tab_scan, tab_detail = st.tabs(["📊 All-Pairs Scanner", "🔍 Pair Detail"])

# ══════════════════════════════════════════════
#  TAB 1 — All-Pairs Scanner
# ══════════════════════════════════════════════
with tab_scan:
    st.markdown("Scanning all 21 instruments for 4H confluence zones …")
    prog = st.progress(0)
    all_instruments = list(INSTRUMENTS.items())
    n_inst = len(all_instruments)
    scan_results: list[dict] = []

    for idx, (pair_name, info) in enumerate(all_instruments):
        prog.progress((idx + 1) / n_inst, text=f"Scanning {pair_name} …")
        try:
            df_s = fetch_4h_data(info["ticker"], 90)
            if df_s.empty or len(df_s) < 10:
                scan_results.append({"pair": pair_name, "ok": False})
                continue
            df_s["EMA_short"] = calc_ema(df_s["Close"], 20)
            df_s["EMA_mid"]   = calc_ema(df_s["Close"], 50)
            df_s["EMA_long"]  = calc_ema(df_s["Close"], 200)
            cp_s  = float(df_s["Close"].iloc[-1])
            sh_s  = float(df_s["High"].rolling(50, min_periods=1).max().iloc[-1])
            sl_s  = float(df_s["Low"].rolling(50, min_periods=1).min().iloc[-1])
            fib_s = calc_fibonacci_levels(sh_s, sl_s)
            piv_s = calc_pivot_points(df_s)
            ema_s = {
                "EMA20":  float(df_s["EMA_short"].iloc[-1]),
                "EMA50":  float(df_s["EMA_mid"].iloc[-1]),
                "EMA200": float(df_s["EMA_long"].iloc[-1]),
            }
            zones_s  = find_confluence_zones(cp_s, fib_s, piv_s, ema_s, 0.005)
            strong_s = sum(1 for z in zones_s if z["strength"] >= 3)
            scan_results.append({
                "pair":          pair_name,
                "ok":            True,
                "current_price": cp_s,
                "zones":         zones_s,
                "num_zones":     len(zones_s),
                "strong":        strong_s,
            })
        except Exception:
            scan_results.append({"pair": pair_name, "ok": False})

    prog.empty()

    # ── Summary counts ──────────────────────────
    cnt_strong   = sum(1 for r in scan_results if r.get("ok") and r.get("strong", 0) > 0)
    cnt_moderate = sum(1 for r in scan_results if r.get("ok") and r.get("strong", 0) == 0 and r.get("num_zones", 0) > 0)
    cnt_none     = sum(1 for r in scan_results if r.get("ok") and r.get("num_zones", 0) == 0)
    cnt_error    = sum(1 for r in scan_results if not r.get("ok"))

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col_, val_, lbl_, c_ in [
        (sc1, cnt_strong,   "Strong Zones",   "#ff6b35"),
        (sc2, cnt_moderate, "Moderate Zones", "#00d4ff"),
        (sc3, cnt_none,     "No Zones",       "#484f58"),
        (sc4, cnt_error,    "Errors",         "#f85149" if cnt_error else "#484f58"),
    ]:
        with col_:
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
                f'padding:14px;text-align:center;">'
                f'<div style="font-size:28px;font-weight:700;color:{c_};font-family:monospace;">{val_}</div>'
                f'<div style="font-size:11px;color:#8b949e;margin-top:2px;font-weight:500;'
                f'letter-spacing:.04em;text-transform:uppercase;">{lbl_}</div>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Card grid ───────────────────────────────
    cols3 = st.columns(3)
    for i, r in enumerate(scan_results):
        pair_name = r["pair"]
        is_selected = pair_name == selected_pair
        border_style = "border:2px solid #388bfd;" if is_selected else "border:1px solid #21262d;"

        if not r.get("ok"):
            badge_html = '<span style="font-size:11px;color:#f85149;">No data</span>'
            price_html = "—"
            zone_html  = "—"
        else:
            num_z  = r["num_zones"]
            strong = r["strong"]
            if strong > 0:
                badge_html = (
                    f'<span style="background:rgba(255,107,53,.15);border:1px solid rgba(255,107,53,.5);'
                    f'border-radius:4px;padding:2px 8px;font-size:11px;color:#ff6b35;font-weight:600;">'
                    f'{strong} STRONG</span>'
                )
            elif num_z > 0:
                badge_html = (
                    f'<span style="background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.4);'
                    f'border-radius:4px;padding:2px 8px;font-size:11px;color:#00d4ff;font-weight:600;">'
                    f'{num_z} MODERATE</span>'
                )
            else:
                badge_html = (
                    '<span style="background:rgba(72,79,88,.2);border:1px solid #484f58;'
                    'border-radius:4px;padding:2px 8px;font-size:11px;color:#484f58;font-weight:600;">'
                    'NO ZONES</span>'
                )
            price_html = f'{r["current_price"]:,.4f}'
            zone_html  = str(num_z)

        with cols3[i % 3]:
            st.markdown(
                f'<div style="background:#161b22;{border_style}border-radius:8px;'
                f'padding:12px 14px;margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:13px;font-weight:600;color:#e6edf3;">{pair_name}</span>'
                f'{badge_html}'
                f'</div>'
                f'<div style="display:flex;gap:16px;margin-top:8px;">'
                f'<span style="font-size:11px;color:#8b949e;">Price: <span style="color:#e6edf3;font-family:monospace;">{price_html}</span></span>'
                f'<span style="font-size:11px;color:#8b949e;">Zones: <span style="color:#00d4ff;font-family:monospace;">{zone_html}</span></span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True)

    # ── Expander table ───────────────────────────
    with st.expander("📋 Full Scanner Results"):
        table_rows = []
        for r in scan_results:
            table_rows.append({
                "Pair":        r["pair"],
                "Price":       f'{r["current_price"]:,.4f}' if r.get("ok") else "—",
                "Zones":       r.get("num_zones", 0) if r.get("ok") else "—",
                "Strong":      r.get("strong", 0)    if r.get("ok") else "—",
                "Status":      "Strong" if r.get("strong", 0) > 0
                               else "Moderate" if r.get("num_zones", 0) > 0
                               else "None" if r.get("ok") else "Error",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  TAB 2 — Pair Detail
# ══════════════════════════════════════════════
with tab_detail:
    detail_ok = True

    if not ticker:
        st.error("Please select a valid instrument.")
        detail_ok = False

    if detail_ok:
        with st.spinner(f"Fetching 4H data for {ticker} …"):
            df = fetch_4h_data(ticker, lookback_days)

        if df.empty or len(df) < 10:
            st.error(f"Could not fetch enough 4H data for **{ticker}**. Try a different symbol.")
            detail_ok = False

    if detail_ok:
        # ── Indicators ───────────────────────────
        df["EMA_short"] = calc_ema(df["Close"], int(ema_short))
        df["EMA_mid"]   = calc_ema(df["Close"], int(ema_mid))
        df["EMA_long"]  = calc_ema(df["Close"], int(ema_long))

        current_price = float(df["Close"].iloc[-1])
        swing_high    = float(df["High"].rolling(50, min_periods=1).max().iloc[-1])
        swing_low     = float(df["Low"].rolling(50, min_periods=1).min().iloc[-1])

        fib_levels = calc_fibonacci_levels(swing_high, swing_low)
        pivots     = calc_pivot_points(df)

        ema_values = {
            f"EMA{int(ema_short)}": float(df["EMA_short"].iloc[-1]),
            f"EMA{int(ema_mid)}":   float(df["EMA_mid"].iloc[-1]),
            f"EMA{int(ema_long)}":  float(df["EMA_long"].iloc[-1]),
        }

        zones = find_confluence_zones(
            price=current_price,
            fib_levels=fib_levels,
            pivots=pivots,
            ema_values=ema_values,
            tolerance_pct=confluence_tol,
        )

        # ── Top metrics ──────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        for col_, val_, lbl_, c_ in [
            (col1, f"{current_price:,.4f}", "Current Price",      "#e6edf3"),
            (col2, f"{swing_high:,.4f}",    "Swing High (50)",    "#3fb950"),
            (col3, f"{swing_low:,.4f}",     "Swing Low (50)",     "#f85149"),
            (col4, str(len(zones)),         "Confluence Zones",   "#00d4ff" if zones else "#484f58"),
        ]:
            with col_:
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
                    f'padding:14px;text-align:center;">'
                    f'<div style="font-size:20px;font-weight:700;color:{c_};font-family:monospace;">{val_}</div>'
                    f'<div style="font-size:11px;color:#8b949e;margin-top:2px;font-weight:500;'
                    f'letter-spacing:.04em;text-transform:uppercase;">{lbl_}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # ── Confluence zone cards ─────────────────
        if zones:
            st.markdown("### 🔥 Active Confluence Zones")

            for zone in zones[:6]:
                direction    = "▲" if zone["distance_pct"] > 0 else "▼"
                extra_badge  = "strong-badge" if zone["strength"] >= 3 else ""
                strength_label = "STRONG" if zone["strength"] >= 3 else "MODERATE"
                labels_text  = "  ·  ".join(zone["labels"])

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span style="font-size:1.2em; font-weight:700; color:#e2e8f0;">
                                    {zone["value"]:,.4f}
                                </span>
                                &nbsp;&nbsp;
                                <span style="color:#6b7a99; font-size:0.85em;">
                                    {direction} {abs(zone["distance_pct"]):.2f}% from price
                                </span>
                            </div>
                            <span class="confluence-badge {extra_badge}">
                                {strength_label} · {len(zone["sources"])} sources
                            </span>
                        </div>
                        <div style="margin-top:8px; font-size:0.78em; color:#4a6080;">
                            {labels_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "No confluence zones detected within the current tolerance. "
                "Try increasing the tolerance in the sidebar."
            )

        st.markdown("---")

        # ── Plotly chart ─────────────────────────
        st.markdown("### 📊 4H Chart — Fibonacci · Pivot · EMA")

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.78, 0.22],
            vertical_spacing=0.03,
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing=dict(
                    line=dict(color="#3fb950"),
                    fillcolor=safe_rgba("#3fb950", 0.13),
                ),
                decreasing=dict(
                    line=dict(color="#f85149"),
                    fillcolor=safe_rgba("#f85149", 0.13),
                ),
                name="4H Candles",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        ema_colors = {
            "EMA_short": "#f59e0b",
            "EMA_mid":   "#a78bfa",
            "EMA_long":  "#34d399",
        }
        ema_names = {
            "EMA_short": f"EMA {int(ema_short)}",
            "EMA_mid":   f"EMA {int(ema_mid)}",
            "EMA_long":  f"EMA {int(ema_long)}",
        }

        for column_name, color in ema_colors.items():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column_name],
                    mode="lines",
                    line=dict(color=color, width=1.4),
                    name=ema_names[column_name],
                ),
                row=1,
                col=1,
            )

        fib_colors = {
            "0.0%":  "#ffffff",
            "23.6%": "#00d4ff",
            "38.2%": "#a78bfa",
            "50.0%": "#f59e0b",
            "61.8%": "#ff6b35",
            "78.6%": "#ff4560",
            "100%":  "#ffffff",
        }

        for label, value in fib_levels.items():
            color = fib_colors.get(label, "#888888")
            fig.add_hline(
                y=value,
                line_dash="dot",
                line_color=color,
                line_width=1,
                opacity=0.55,
                annotation_text=f"Fib {label}",
                annotation_position="right",
                annotation_font=dict(size=9, color=color),
                row=1,
                col=1,
            )

        pivot_colors = {
            "PP": "#ffffff",
            "R1": "#00d4ff",
            "R2": "#0077ff",
            "R3": "#003f8a",
            "S1": "#ff6b35",
            "S2": "#ff3300",
            "S3": "#8a1a00",
        }

        for label, value in pivots.items():
            color = pivot_colors.get(label, "#888888")
            fig.add_hline(
                y=value,
                line_dash="dash",
                line_color=color,
                line_width=1,
                opacity=0.6,
                annotation_text=f"  {label}",
                annotation_position="left",
                annotation_font=dict(size=9, color=color),
                row=1,
                col=1,
            )

        for zone in zones:
            band  = zone["value"] * confluence_tol
            color = "#ff6b35" if zone["strength"] >= 3 else "#00d4ff"
            fig.add_hrect(
                y0=zone["value"] - band,
                y1=zone["value"] + band,
                fillcolor=safe_rgba(color, 0.08),
                line_width=0,
                row=1,
                col=1,
            )
            fig.add_hline(
                y=zone["value"],
                line_dash="longdash",
                line_color=color,
                line_width=1.5,
                opacity=0.8,
                row=1,
                col=1,
            )

        fig.add_hline(
            y=current_price,
            line_dash="solid",
            line_color="#ffffff",
            line_width=1,
            opacity=0.9,
            annotation_text=f"  Price: {current_price:,.4f}",
            annotation_position="right",
            annotation_font=dict(size=10, color="#ffffff"),
            row=1,
            col=1,
        )

        colors_vol = [
            safe_rgba("#f85149", 0.53) if close < open_ else safe_rgba("#3fb950", 0.53)
            for close, open_ in zip(df["Close"], df["Open"])
        ]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=colors_vol,
                name="Volume",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(family="Inter, sans-serif", size=11, color="#8b949e"),
            xaxis_rangeslider_visible=False,
            legend=dict(
                bgcolor=safe_rgba("#0d1117", 0.60),
                bordercolor="#1e3a5f",
                borderwidth=1,
                font=dict(size=10),
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=10, r=120, t=10, b=10),
            height=680,
            xaxis2=dict(showgrid=True, gridcolor="#21262d"),
            yaxis=dict(showgrid=True, gridcolor="#21262d", side="right"),
            yaxis2=dict(showgrid=False, side="right"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Raw level table ───────────────────────
        with st.expander("📋 All Levels Reference Table"):
            rows = []
            for label, value in fib_levels.items():
                rows.append({"Type": "Fibonacci", "Label": label,
                              "Level": round(float(value), 6),
                              "Δ% from Price": round((float(value) - current_price) / current_price * 100, 3)})
            for label, value in pivots.items():
                rows.append({"Type": "Pivot", "Label": label,
                              "Level": round(float(value), 6),
                              "Δ% from Price": round((float(value) - current_price) / current_price * 100, 3)})
            for label, value in ema_values.items():
                rows.append({"Type": "EMA", "Label": label,
                              "Level": round(float(value), 6),
                              "Δ% from Price": round((float(value) - current_price) / current_price * 100, 3)})

            ref_df = pd.DataFrame(rows).sort_values("Level", ascending=False)
            st.dataframe(ref_df, use_container_width=True, hide_index=True)

        st.caption(
            f"Data: yfinance · 4H candles · Last refreshed "
            f"{datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M')} UTC"
        )
