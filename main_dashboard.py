import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import json
import logging
import traceback
from typing import Dict, Tuple, List, Optional

from src.core.config import default_config as config, CANDLE_STYLE, CHART_LAYOUT, EMA_COLORS, RSI_LINE, RSI_OB, RSI_OS
from src.core.analyzer import TechnicalAnalyzer as analyzer
from src.core.data_provider import fetch_data, get_macro_data, fetch_fred_series
from src.core.signals import generate_trading_ideas, safe_get, entry_generator

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Macro Dashboard Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger("ForexDashboard")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.sig-buy  { background:#0d3b2a; border:1px solid #1a7a55; color:#4af0c4; padding:4px 12px; border-radius:3px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }
.sig-sell { background:#3b0d16; border:1px solid #7a1a2a; color:#f04a6a; padding:4px 12px; border-radius:3px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# NOTIFICATION PERSISTENCE
# ============================================================================
NOTIFY_FILE = os.path.join(os.getcwd(), "forex_notify_cache.json")


def load_notified_keys() -> set:
    try:
        if os.path.exists(NOTIFY_FILE):
            with open(NOTIFY_FILE) as fh:
                data = json.load(fh)
            return set(data.get("keys", []))
    except Exception:
        pass
    return set()


def save_notified_keys(keys: set) -> None:
    try:
        with open(NOTIFY_FILE, "w") as fh:
            json.dump({"keys": sorted(keys)}, fh)
    except Exception:
        pass


# ============================================================================
# NOTIFICATION SYSTEM
# ============================================================================
def init_notification_state() -> None:
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'notified_keys' not in st.session_state:
        st.session_state.notified_keys = load_notified_keys()
    if 'notification_log' not in st.session_state:
        st.session_state.notification_log = []
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()


def check_and_notify(ideas: List[Dict]) -> List[Dict]:
    init_notification_state()
    new_alerts: List[Dict] = []

    for idea in ideas:
        # Check for High conviction or high score (>=8)
        if idea['conviction'] != 'High' and idea.get('strength_score', 0) < 8:
            continue

        # Unique key based on pair, bias AND entry price (to allow new alerts for the same pair if price moved)
        # Use rounded price to avoid jitter
        price_key = f"{idea['entry']:.4f}"
        key = f"{idea['pair']}_{idea['bias']}_{price_key}"

        if key not in st.session_state.notified_keys:
            st.session_state.notified_keys.add(key)
            new_alerts.append(idea)

    if new_alerts:
        save_notified_keys(st.session_state.notified_keys)

    for idea in new_alerts:
        direction = "📈 LONG" if idea['bias'] == 'Long' else "📉 SHORT"
        st.toast(
            f"🚨 NEW SIGNAL: {idea['pair']} {direction}\n"
            f"Score: {idea['strength_score']}/10 | Entry: {idea['entry']:.5f}",
            icon="🔔",
        )
        st.session_state.notification_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "pair": idea["pair"],
            "bias": idea["bias"],
            "entry": idea["entry"],
            "rr": idea["risk_reward_1"],
        })
    return new_alerts


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(ttl=config.cache_ttl, show_spinner=False)
def load_all_market_data() -> Dict:
    """Fetches all market data. Cached across reruns for config.cache_ttl seconds."""
    data = {tf: {} for tf in config.timeframes}
    for tf_name, tf_cfg in config.timeframes.items():
        for pair_name, symbol in config.assets.items():
            try:
                df = fetch_data(symbol, tf_cfg["interval"], tf_cfg["period"])
                if not df.empty:
                    data[tf_name][pair_name] = df
            except Exception as e:
                logger.warning(f"Failed {pair_name} ({tf_name}): {e}")
    return data


def clear_data_cache() -> None:
    st.cache_data.clear()
    st.session_state.data_loaded = False
    st.session_state.last_refresh = datetime.now()


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_sidebar() -> str:
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")

        # Load key from secrets or environment — never hardcode keys in source
        fred_api_key = st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY", ""))
        if fred_api_key:
            st.success("✅ FRED API Key loaded")
        else:
            st.warning("⚠️ FRED API Key missing — add FRED_API_KEY to .streamlit/secrets.toml or your environment")

        st.divider()

        selected_tf = st.selectbox(
            "Default Chart Timeframe",
            ["Daily", "4 Hour", "Hourly", "15 Minute"],
        )

        st.divider()

        st.subheader("🔄 Data Refresh")
        if st.button("↺ Refresh Now", use_container_width=True):
            clear_data_cache()
            st.rerun()

        elapsed = int((datetime.now() - st.session_state.get('last_refresh', datetime.now())).total_seconds())
        st.caption(f"Age: {elapsed}s / {config.cache_ttl}s")

        st.divider()

        st.subheader("🔔 Alert Log")
        log = st.session_state.get('notification_log', [])
        if log:
            for entry in reversed(log[-10:]):
                icon = "📈" if entry['bias'] == 'Long' else "📉"
                st.markdown(f"**{entry['time']}** {icon} **{entry['pair']}** {entry['bias']} — R:R {entry['rr']:.2f}")
            if st.button("🗑️ Clear Alerts"):
                st.session_state.notification_log = []
                st.session_state.notified_keys = set()
                save_notified_keys(set())
                st.rerun()
        else:
            st.caption("No alerts yet.")

    return selected_tf


def render_kpis(daily_data: Dict) -> None:
    kpi_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "XAU/USD", "BTC/USD"]
    cols = st.columns(len(kpi_pairs))
    for i, pair in enumerate(kpi_pairs):
        df = daily_data.get(pair)
        with cols[i]:
            if df is not None and not df.empty:
                price = df["Close"].iloc[-1]
                change = df["Close"].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0
                fmt = f"{price:,.2f}" if pair in ("BTC/USD", "XAU/USD") else f"{price:.4f}"
                st.metric(pair, fmt, f"{change:+.2f}%")
            else:
                st.metric(pair, "N/A", "—")


def render_macro_table(macro_data: Dict, is_live: bool) -> None:
    from src.core.data_provider import MACRO_FALLBACKS_DATE
    if is_live:
        st.success("✅ Live FRED data")
    else:
        st.warning(
            f"⚠️ **Static fallback data (as of {MACRO_FALLBACKS_DATE})** — "
            "Values may be significantly out of date. Add a FRED API key in the sidebar for live values."
        )

    rows = [{"Currency": ccy, **vals} for ccy, vals in macro_data.items()]
    df = pd.DataFrame(rows).set_index("Currency")
    st.dataframe(df.style.background_gradient(cmap="RdYlGn", subset=["GDP", "Inflation", "Rates", "Unemployment"]),
                 use_container_width=True)


def render_overview_tab(daily_data: Dict):
    st.subheader("Market Overview")
    if daily_data:
        rows = []
        for pair, df in daily_data.items():
            if not df.empty:
                price = df['Close'].iloc[-1]
                change = df['Close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0
                rows.append({"Pair": pair, "Price": price, "Change %": change, "Bars": len(df)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_mtf_matrix_tab(data_by_timeframe: Dict):
    st.subheader("🧭 Multi-Timeframe Matrix")
    mtf_rows = []
    # Explicitly requested timeframes
    target_tfs = ["Weekly", "Daily", "4 Hour", "Hourly"]

    for pair in config.assets.keys():
        sentiments = analyzer.get_mtf_sentiment(data_by_timeframe, pair)
        row = {"Pair": pair}
        for tf in target_tfs:
            row[tf] = sentiments.get(tf, "N/A")
        mtf_rows.append(row)

    mtf_df = pd.DataFrame(mtf_rows).set_index("Pair")

    def color_sentiment(val):
        if val == "Bullish": return "background-color: #249d53; color: white;"
        if val == "Bearish": return "background-color: #ca2427; color: white;"
        return ""

    st.table(mtf_df.style.map(color_sentiment))


def render_technical_chart_tab(data_by_timeframe: Dict):
    st.subheader("📈 Technical Analysis Chart")
    daily_data = data_by_timeframe.get('Daily', {})
    avail = [p for p, d in daily_data.items() if not d.empty]
    if avail:
        c1, c2 = st.columns(2)
        pair = c1.selectbox("Pair", avail, key="chart_pair")
        tf = c2.selectbox("Timeframe", list(config.timeframes.keys()), key="chart_tf")
        df = data_by_timeframe.get(tf, {}).get(pair, pd.DataFrame())
        if not df.empty:
            df = analyzer.add_indicators(df)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3],
                                subplot_titles=(f"{pair} — {tf}", "RSI"))
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                         name="Price", **CANDLE_STYLE), row=1, col=1)
            for ma, color in [(k, v) for k, v in EMA_COLORS.items() if k in df.columns]:
                fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=color, width=1.4)), row=1, col=1)
            if 'RSI' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=RSI_LINE), row=2, col=1)
                fig.add_hline(y=70, line=RSI_OB, row=2, col=1)
                fig.add_hline(y=30, line=RSI_OS, row=2, col=1)
            fig.update_layout(height=600, **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


def render_trading_view_tab(data_by_timeframe: Dict):
    st.subheader("🛒 Trading View (Pivots & Fibonacci)")
    daily_data = data_by_timeframe.get('Daily', {})
    avail_tv = [p for p, d in daily_data.items() if not d.empty]
    if avail_tv:
        col1, col2 = st.columns([1, 3])
        with col1:
            tv_pair = st.selectbox("Select Pair", avail_tv, key="tv_pair")
            tv_tf = st.selectbox("Anchor Timeframe", ["Weekly", "Daily", "4 Hour"], index=1, key="tv_tf")

            df_anchor = data_by_timeframe.get(tv_tf, {}).get(tv_pair, pd.DataFrame())
            if not df_anchor.empty:
                pivots = analyzer.calculate_pivots(df_anchor)
                fibs = analyzer.calculate_fibonacci(df_anchor)

                st.markdown("### 📏 Pivot Points")
                for k, v in pivots.items():
                    st.text(f"{k:5}: {v:.5f}")

                st.markdown("### 🔢 Fibonacci")
                for k, v in fibs.items():
                    st.text(f"{k:6}: {v:.5f}")

        with col2:
            df_chart = data_by_timeframe.get(tv_tf, {}).get(tv_pair, pd.DataFrame())
            if not df_chart.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                                   low=df_chart['Low'], close=df_chart['Close'],
                                   name="Price", **CANDLE_STYLE))

                # Add Pivot Lines
                pivots = analyzer.calculate_pivots(df_chart)
                colors = {"R": "rgba(239,83,80,0.5)", "S": "rgba(38,166,154,0.5)", "P": "rgba(200,200,200,0.5)"}
                for level, val in pivots.items():
                    color = colors.get(level[0], "rgba(200,200,200,0.5)")
                    fig.add_hline(y=val, line_dash="dot", line_color=color, annotation_text=level)

                fig.update_layout(height=600, title=f"{tv_pair} - {tv_tf} with Pivot Points", **CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)


def render_15m_entry_tab(data_by_timeframe: Dict):
    st.subheader("⏱️ 15-Minute Entry Signal")
    daily_data = data_by_timeframe.get('Daily', {})
    avail_pairs = [p for p in daily_data if not daily_data[p].empty]
    if avail_pairs:
        pair_e = st.selectbox("Pair", avail_pairs, key="entry_pair")
        df_15m = data_by_timeframe.get('15 Minute', {}).get(pair_e, pd.DataFrame())
        df_d = data_by_timeframe.get('Daily', {}).get(pair_e, pd.DataFrame())

        if not df_15m.empty and not df_d.empty:
            df_d = analyzer.add_indicators(df_d)
            di = df_d.iloc[-1]
            adx_v = safe_get(di, "ADX", 0.0)
            close_v = safe_get(di, "Close", 0.0)
            ema20_v = safe_get(di, "EMA_20", close_v)

            bias_v = ("Long" if close_v > ema20_v else "Short") if adx_v > config.adx_trend_min else "Neutral"
            st.write(f"**Daily Trend Bias:** `{bias_v}` | ADX = {adx_v:.1f}")

            sig = entry_generator.get_entry_signal(df_15m, bias_v)
            c1, c2, c3 = st.columns(3)
            with c1:
                if sig["signal"] == 1:
                    st.success("### 🟢 LONG")
                elif sig["signal"] == -1:
                    st.error("### 🔴 SHORT")
                else:
                    st.info("### ⚪ NO SIGNAL")
            c2.metric("Confidence", f"{sig['confidence']}/5")
            c3.metric("Price", f"{sig.get('price', 0):.5f}")

            for r in sig.get("reasons", []):
                st.info(f"ℹ️ {r}")
        else:
            st.warning("Insufficient data for 15-minute analysis")


def render_signal_pro_tab(data_by_timeframe: Dict):
    st.subheader("⚡ Signal Dashboard (QuantConnect-Style)")
    daily_data = data_by_timeframe.get('Daily', {})
    avail = [p for p, d in daily_data.items() if not d.empty]
    if avail:
        c1, c2 = st.columns([1, 4])
        with c1:
            pair = st.selectbox("Select Asset", avail, key="pro_pair")
            tf = st.selectbox("Timeframe", ["Daily", "4 Hour", "Hourly", "15 Minute"], key="pro_tf")
            pivot_tf = st.selectbox("Pivot Lookback", ["Weekly", "Daily"], index=0, key="pro_piv_tf")

        df = data_by_timeframe.get(tf, {}).get(pair, pd.DataFrame())
        df_piv = data_by_timeframe.get(pivot_tf, {}).get(pair, pd.DataFrame())

        if not df.empty and not df_piv.empty:
            pivots = analyzer.calculate_expanded_pivots(df_piv)
            df = analyzer.generate_pro_signals(df, pivots)

            # Metrics
            latest = df["Close"].iloc[-1]
            last_signal = df["Signal"].iloc[-1]
            rsi_val = df["RSI"].iloc[-1] if "RSI" in df.columns else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price", f"{latest:.5f}")
            m2.metric("Signal", last_signal)
            m3.metric("RSI", f"{rsi_val:.1f}")
            m4.metric("PP Level", f"{pivots['PP']:.5f}")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                         name="Price", **CANDLE_STYLE), row=1, col=1)

            # Add Pivot Lines
            colors = {"R": "rgba(239,83,80,0.3)", "S": "rgba(38,166,154,0.3)", "P": "rgba(200,200,200,0.3)"}
            for level, val in pivots.items():
                if level in ["PP", "R1", "S1", "R2", "S2", "R3", "S3"]:
                    color = colors.get(level[0], "rgba(200,200,200,0.3)")
                    fig.add_hline(y=val, line_dash="dot", line_color=color, annotation_text=level, row=1, col=1)

            # Add Signal markers
            for sig, color, sym in [("STRONG BUY", "#26a69a", "triangle-up"), ("BUY", "#66bb6a", "triangle-up"),
                                    ("STRONG SELL", "#ef5350", "triangle-down"), ("SELL", "#ffa726", "triangle-down")]:
                mask = df["Signal"] == sig
                if mask.any():
                    fig.add_trace(go.Scatter(x=df.index[mask], y=df.loc[mask, "High" if "SELL" in sig else "Low"],
                                             mode="markers", marker=dict(symbol=sym, color=color, size=10), name=sig),
                                  row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=RSI_LINE), row=2, col=1)
            fig.add_hline(y=70, line=RSI_OB, row=2, col=1)
            fig.add_hline(y=30, line=RSI_OS, row=2, col=1)

            fig.update_layout(height=600, **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Signal Score Timeline")
            fig_score = go.Figure(
                go.Bar(x=df.index, y=df["Signal_Score"], marker_color=df["Signal_Score"], marker_colorscale="RdYlGn"))
            fig_score.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_score, use_container_width=True)


def render_macro_pro_tab(fred_key: str):
    st.subheader("🏛 FRED Macro Dashboard (8-Grid)")
    if not fred_key:
        st.warning("Enter a FRED API key in the sidebar to view the macro grid.")
        return

    fred_series_map = {
        "Fed Funds Rate": "FEDFUNDS",
        "CPI YoY": "CPIAUCSL",
        "10Y Treasury": "DGS10",
        "2Y Treasury": "DGS2",
        "Unemployment": "UNRATE",
        "GDP Growth": "A191RL1Q225SBEA",
        "DXY Index": "DTWEXBGS",
        "VIX": "VIXCLS",
    }

    with st.spinner("Fetching macro series..."):
        loaded_data = {name: fetch_fred_series(sid, fred_key) for name, sid in fred_series_map.items()}

    valid_data = {k: v for k, v in loaded_data.items() if v is not None and not v.empty}

    if valid_data:
        fig = make_subplots(rows=2, cols=4, subplot_titles=list(valid_data.keys()))
        for i, (name, df) in enumerate(valid_data.items()):
            row, col = divmod(i, 4)
            fig.add_trace(go.Scatter(x=df["date"], y=df["value"], name=name, fill="tozeroy"), row=row + 1, col=col + 1)

        fig.update_layout(height=400, showlegend=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Macro Regime
        st.markdown("### Macro Regime Assessment")
        score = 0
        notes = []
        if "Fed Funds Rate" in valid_data:
            val = valid_data["Fed Funds Rate"]["value"].iloc[-1]
            if val > 4.5:
                notes.append("🔴 High Rates (Restrictive)"); score -= 1
            else:
                notes.append("🟢 Moderate Rates"); score += 1

        regime = "BULLISH 🟢" if score > 0 else ("BEARISH 🔴" if score < 0 else "NEUTRAL ⚪")
        st.metric("Overall Macro Regime", regime)
        for n in notes: st.write(n)
    else:
        st.error("Could not fetch FRED data. Please check your API key.")


@st.fragment(run_every=300)
def render_trading_ideas_tab():
    """
    Decorated with @st.fragment(run_every=300) so Streamlit re-executes this
    block automatically every 5 minutes without re-running the full page.
    Reads data from session_state so each auto-run picks up the current dataset.
    """
    st.subheader("🎯 Live Trading Ideas")

    data = st.session_state.get("data_by_timeframe", {})
    if not data:
        st.warning("Market data not loaded yet — waiting for initial fetch.")
        return

    # ── Run analysis ─────────────────────────────────────────────────────────
    with st.spinner("Scanning all pairs…"):
        for tf in data:
            for p in data[tf]:
                if not data[tf][p].empty:
                    data[tf][p] = analyzer.add_indicators(data[tf][p])

        ideas, skipped = generate_trading_ideas(data)
        st.session_state.latest_ideas = ideas
        check_and_notify(ideas)

    # ── Header row ────────────────────────────────────────────────────────────
    now = datetime.now()
    h_left, h_right = st.columns([3, 1])
    h_left.caption(
        f"🕐 Last scanned: **{now.strftime('%H:%M:%S')}**  ·  "
        f"Auto-refreshes every 5 min"
    )
    if skipped:
        h_right.caption(f"⚠️ {len(skipped)} pairs skipped")

    if not ideas:
        st.info("No qualifying setups found on this scan. Next scan in 5 min.")
        return

    st.success(f"✅ {len(ideas)} setup{'s' if len(ideas) != 1 else ''} found")

    for idx, idea in enumerate(ideas):
        direction = "📈" if idea["bias"] == "Long" else "📉"
        color     = "green" if idea["bias"] == "Long" else "red"
        header    = (f"### {idx + 1}. {idea['pair']} — "
                     f"<span style='color:{color}'>{idea['bias'].upper()} {direction}</span>")

        if idea["conviction"] == "High":
            st.markdown(
                f"{header} &nbsp;"
                f"<span style='background:#ffd700;color:#000;padding:2px 8px;"
                f"border-radius:3px;font-size:12px'>🔔 HIGH CONVICTION</span>",
                unsafe_allow_html=True)
        else:
            st.markdown(header, unsafe_allow_html=True)

        cols = st.columns(6)
        cols[0].metric("Entry",          f"{idea['entry']:.5f}")
        cols[1].metric("TP1",            f"{idea['take_profit_1']:.5f}",
                       delta=f"R:R {idea['risk_reward_1']:.2f}")
        cols[2].metric("TP2",            f"{idea['take_profit_2']:.5f}",
                       delta=f"R:R {idea['risk_reward_2']:.2f}")
        cols[3].metric("Stop Loss",      f"{idea['stop_loss']:.5f}")
        cols[4].metric("Multi-TF Score", f"{idea['strength_score']}/10",
                       help="Multi-timeframe alignment strength")
        cols[5].metric("Entry Quality",  f"{idea['confidence']}/10",
                       help="15-min entry trigger confidence (Stoch, RSI, BB)")

        st.markdown(f"**Thesis:** {idea['thesis']}")
        st.caption(
            f"Stop method: {idea['stop_loss_method']} | "
            f"Distance: {idea['stop_loss_pips']} pips"
        )
        st.divider()


def render_weekly_swing_tab(data_by_timeframe: Dict):
    st.subheader("📅 Weekly Swing Trading Ideas")
    st.caption("Pivot-point driven setups anchored to the weekly timeframe, confirmed by the daily chart.")

    weekly_data = data_by_timeframe.get("Weekly", {})
    daily_data  = data_by_timeframe.get("Daily",  {})

    if not weekly_data:
        st.warning("No weekly data available — click ↺ Refresh in the sidebar.")
        return

    ideas = []

    for pair in config.assets:
        df_w = weekly_data.get(pair, pd.DataFrame())
        df_d = daily_data.get(pair,  pd.DataFrame())

        if df_w.empty or len(df_w) < 20:
            continue

        df_w = analyzer.add_indicators(df_w.copy())
        last = df_w.iloc[-1]

        price  = float(last["Close"])
        ema20  = float(last.get("EMA_20", price))
        ema50  = float(last.get("EMA_50", price))
        rsi    = float(last.get("RSI",   50.0))
        adx    = float(last.get("ADX",    0.0))
        atr    = float(last.get("ATR",  price * 0.01))

        # ── Weekly pivots (based on previous completed weekly candle) ─────────
        pivots = analyzer.calculate_pivots(df_w)
        if not pivots:
            continue
        pp = pivots["Pivot"]
        r1 = pivots["R1"]; r2 = pivots["R2"]; r3 = pivots["R3"]
        s1 = pivots["S1"]; s2 = pivots["S2"]; s3 = pivots["S3"]

        # ── Fibonacci levels over the last 12 weeks ───────────────────────────
        fibs = analyzer.calculate_fibonacci(df_w.tail(12))

        # ── Bias scoring ──────────────────────────────────────────────────────
        bull_s = bear_s = 0
        reasons: List[str] = []

        if price > ema20:  bull_s += 2
        else:              bear_s += 2

        if ema20 > ema50:  bull_s += 2; reasons.append("EMA20 > EMA50 — weekly uptrend")
        else:              bear_s += 2; reasons.append("EMA20 < EMA50 — weekly downtrend")

        if rsi > 55:       bull_s += 1; reasons.append(f"RSI {rsi:.0f} — bullish momentum")
        elif rsi < 45:     bear_s += 1; reasons.append(f"RSI {rsi:.0f} — bearish momentum")

        if price > pp:     bull_s += 1; reasons.append(f"Price above weekly pivot ({pp:.5f})")
        else:              bear_s += 1; reasons.append(f"Price below weekly pivot ({pp:.5f})")

        if adx > 20:
            if bull_s > bear_s: bull_s += 1; reasons.append(f"ADX {adx:.0f} — trend is strong")
            else:               bear_s += 1; reasons.append(f"ADX {adx:.0f} — trend is strong")

        if bull_s == bear_s or adx < 15:
            continue  # skip flat / trendless markets

        bias = "Long" if bull_s > bear_s else "Short"

        # ── Entry / SL / TP ───────────────────────────────────────────────────
        fib_382 = fibs.get("38.2%", price)
        fib_500 = fibs.get("50.0%", price)
        fib_618 = fibs.get("61.8%", price)

        if bias == "Long":
            # Entry: current price (or tighten to S1 pullback zone)
            entry    = price
            # SL: below S1 or below 61.8% fib — whichever is lower
            sl       = min(s1, fib_618) - atr * 0.3
            tp1      = r1
            tp2      = r2
            tp3      = r3
        else:
            entry    = price
            # SL: above R1 or above 38.2% fib — whichever is higher
            sl       = max(r1, fib_382) + atr * 0.3
            tp1      = s1
            tp2      = s2
            tp3      = s3

        stop_dist = abs(entry - sl)
        if stop_dist == 0:
            continue

        rr1 = round(abs(tp1 - entry) / stop_dist, 2)
        rr2 = round(abs(tp2 - entry) / stop_dist, 2)
        rr3 = round(abs(tp3 - entry) / stop_dist, 2)

        if rr1 < 1.5:
            continue  # reject low-quality setups

        # ── Daily confirmation ────────────────────────────────────────────────
        daily_conf = "—"
        if not df_d.empty:
            df_d_ind     = analyzer.add_indicators(df_d.copy())
            daily_sent   = analyzer.get_sentiment(df_d_ind)
            if (bias == "Long"  and daily_sent == "Bullish") or \
               (bias == "Short" and daily_sent == "Bearish"):
                daily_conf = "✅ Aligned"
            elif daily_sent == "Neutral":
                daily_conf = "⚪ Neutral"
            else:
                daily_conf = "⚠️ Conflicting"

        ideas.append({
            "pair": pair, "bias": bias, "price": price,
            "entry": entry, "sl": sl,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "rr1": rr1, "rr2": rr2, "rr3": rr3,
            "rsi": rsi, "adx": adx, "atr": atr,
            "pivots": pivots, "fibs": fibs,
            "reasons": reasons, "daily_conf": daily_conf,
            "score": bull_s if bias == "Long" else bear_s,
            "df_w": df_w,
        })

    # Best R:R first, then highest score
    ideas.sort(key=lambda x: (x["rr1"], x["score"]), reverse=True)

    if not ideas:
        st.warning("No qualifying swing setups found. Markets may be ranging — ADX threshold is 15.")
        return

    aligned   = [i for i in ideas if i["daily_conf"] == "✅ Aligned"]
    other     = [i for i in ideas if i["daily_conf"] != "✅ Aligned"]

    def _render_idea(idea):
        direction = "📈 LONG" if idea["bias"] == "Long" else "📉 SHORT"
        header = (f"{idea['pair']}  —  {direction}  |  "
                  f"R:R {idea['rr1']:.2f}  |  Daily: {idea['daily_conf']}")

        with st.expander(header, expanded=False):
            col_info, col_chart = st.columns([1, 2])

            with col_info:
                st.markdown("##### Trade levels")
                st.metric("Entry",      f"{idea['entry']:.5f}")
                st.metric("Stop Loss",  f"{idea['sl']:.5f}")
                st.metric("Target 1",   f"{idea['tp1']:.5f}", delta=f"R:R {idea['rr1']:.2f}")
                st.metric("Target 2",   f"{idea['tp2']:.5f}", delta=f"R:R {idea['rr2']:.2f}")
                st.metric("Target 3",   f"{idea['tp3']:.5f}", delta=f"R:R {idea['rr3']:.2f}")
                st.divider()
                st.markdown("##### Weekly signals")
                for r in idea["reasons"]:
                    st.caption(f"• {r}")
                st.caption(f"RSI {idea['rsi']:.1f}  |  ADX {idea['adx']:.1f}  |  ATR {idea['atr']:.5f}")

            with col_chart:
                df_w    = idea["df_w"]
                pivots  = idea["pivots"]
                fibs    = idea["fibs"]

                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05, row_heights=[0.75, 0.25],
                    subplot_titles=[f"{idea['pair']} — Weekly", "RSI (14)"],
                )

                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=df_w.index,
                    open=df_w["Open"], high=df_w["High"],
                    low=df_w["Low"],   close=df_w["Close"],
                    name="Price", showlegend=False, **CANDLE_STYLE,
                ), row=1, col=1)

                # EMAs
                for col_name, color in EMA_COLORS.items():
                    if col_name in df_w.columns:
                        fig.add_trace(go.Scatter(
                            x=df_w.index, y=df_w[col_name],
                            line=dict(color=color, width=1.4),
                            name=col_name,
                        ), row=1, col=1)

                # Pivot dotted lines
                piv_style = {
                    "Pivot": ("#c0c0c0", 1.2), "R1": ("#ef5350", 0.9),
                    "R2": ("#e53935", 0.9),     "S1": ("#26a69a", 0.9),
                    "S2": ("#00897b", 0.9),
                }
                for lbl, (col_c, lw) in piv_style.items():
                    val = pivots.get(lbl)
                    if val:
                        fig.add_hline(y=val, line_dash="dot", line_color=col_c,
                                      line_width=lw, annotation_text=lbl,
                                      annotation_font=dict(size=9, color=col_c),
                                      row=1, col=1)

                # Trade levels (solid, bolder)
                tp_lines = [
                    (idea["tp3"], "#1de9b6", "TP3"),
                    (idea["tp2"], "#26a69a", "TP2"),
                    (idea["tp1"], "#66bb6a", "TP1"),
                    (idea["sl"],  "#ef5350", "SL"),
                ]
                for val, col_c, lbl in tp_lines:
                    fig.add_hline(y=val, line_dash="dash", line_color=col_c,
                                  line_width=1.8, annotation_text=lbl,
                                  annotation_font=dict(size=10, color=col_c),
                                  row=1, col=1)

                # RSI
                if "RSI" in df_w.columns:
                    fig.add_trace(go.Scatter(
                        x=df_w.index, y=df_w["RSI"],
                        line=RSI_LINE, name="RSI",
                    ), row=2, col=1)
                    fig.add_hline(y=70, line=RSI_OB, row=2, col=1)
                    fig.add_hline(y=30, line=RSI_OS, row=2, col=1)

                fig.update_layout(height=420, **CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

    if aligned:
        st.markdown(f"#### ✅ Daily-Confirmed Setups ({len(aligned)})")
        for idea in aligned:
            _render_idea(idea)

    if other:
        st.markdown(f"#### Other Setups ({len(other)})")
        for idea in other:
            _render_idea(idea)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title(f"💹 Macro Dashboard Pro v{config.version}")

    init_notification_state()

    fred_api_key = st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY", ""))

    selected_tf = render_sidebar()

    if not st.session_state.data_loaded:
        bar = st.progress(0, text="Fetching market data…")
        st.session_state.data_by_timeframe = load_all_market_data()
        bar.progress(1.0, text="Done!")
        bar.empty()
        st.session_state.data_loaded = True
        st.session_state.last_refresh = datetime.now()

    # Get macro data reactively based on FRED key in sidebar
    macro_data, macro_live = get_macro_data(fred_api_key)

    data_by_timeframe = st.session_state.data_by_timeframe

    # BACKGROUND ALERT ENGINE
    # This ensures high-conviction ideas are checked even if the user is on another tab
    if st.session_state.data_loaded:
        # Check every 5 minutes (config.cache_ttl)
        # We don't want to re-run heavy analysis on every interaction,
        # but we do want to ensure ideas are processed.
        # If latest_ideas exists, we re-verify notifications.
        ideas = st.session_state.get('latest_ideas', [])
        if ideas:
            check_and_notify(ideas)

    daily_data = data_by_timeframe.get('Daily', {})
    if daily_data:
        render_kpis(daily_data)

    tabs = st.tabs([
        "📊 Market Overview",           # Step 0 — morning scan
        "🌍 Macro Fundamentals",        # Step 1 — fundamental backdrop
        "🏛 Macro Dashboard",           # Step 2 — deep FRED macro
        "📅 Weekly Swing",              # Step 3 — weekly direction filter
        "🧭 Multi-Timeframe Matrix",    # Step 4 — timeframe alignment
        "📈 Technical Chart",           # Step 5 — chart analysis per pair
        "🛒 Pivots & Fibonacci",        # Step 6 — session key levels
        "⚡ Signal Pro",                # Step 7 — signal confirmation
        "🎯 Trading Ideas",             # Step 8 — auto-refreshed setups
        "⏱️ 15-Min Entry",              # Step 9 — execution timing
    ])

    with tabs[0]:
        render_overview_tab(daily_data)
    with tabs[1]:
        render_macro_table(macro_data, macro_live)
    with tabs[2]:
        render_macro_pro_tab(fred_api_key)
    with tabs[3]:
        render_weekly_swing_tab(data_by_timeframe)
    with tabs[4]:
        render_mtf_matrix_tab(data_by_timeframe)
    with tabs[5]:
        render_technical_chart_tab(data_by_timeframe)
    with tabs[6]:
        render_trading_view_tab(data_by_timeframe)
    with tabs[7]:
        render_signal_pro_tab(data_by_timeframe)
    with tabs[8]:
        render_trading_ideas_tab()
    with tabs[9]:
        render_15m_entry_tab(data_by_timeframe)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(traceback.format_exc())