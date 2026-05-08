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

from src.core.config import default_config as config
from src.core.analyzer import TechnicalAnalyzer as analyzer
from src.core.data_provider import fetch_data, get_macro_data
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
        if idea['conviction'] != 'High':
            continue
        key = f"{idea['pair']}_{idea['bias']}"
        if key not in st.session_state.notified_keys:
            st.session_state.notified_keys.add(key)
            new_alerts.append(idea)

    save_notified_keys(st.session_state.notified_keys)

    for idea in new_alerts:
        direction = "📈 LONG" if idea['bias'] == 'Long' else "📉 SHORT"
        st.toast(
            f"🚨 HIGH CONVICTION\n{idea['pair']} {direction}\n"
            f"Entry {idea['entry']:.5f} | R:R 1:{idea['risk_reward_1']:.2f}",
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
def load_all_market_data() -> Dict:
    bar = st.progress(0, text="Fetching market data…")
    data = {tf: {} for tf in config.timeframes}

    total_steps = len(config.timeframes) * len(config.assets)
    step = 0

    for tf_name, tf_cfg in config.timeframes.items():
        for pair_name, symbol in config.assets.items():
            try:
                df = fetch_data(symbol, tf_cfg["interval"], tf_cfg["period"])
                if not df.empty:
                    # Indicators are added later or we can do it here for performance
                    data[tf_name][pair_name] = df
            except Exception as e:
                logger.warning(f"Failed {pair_name} ({tf_name}): {e}")
            step += 1
            bar.progress(step / total_steps)

    bar.empty()
    return data

def clear_data_cache() -> None:
    st.cache_data.clear()
    st.session_state.data_loaded = False
    st.session_state.last_refresh = datetime.now()

# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_sidebar(default_key: str) -> Tuple[str, str]:
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")

        st.subheader("🔑 FRED API Key")
        fred_api_key = st.text_input(
            "API Key", value=default_key, type="password",
            help="Free key at https://fred.stlouisfed.org/docs/api/api_key.html",
        )
        if fred_api_key:
            st.success("✅ FRED key loaded")
        else:
            st.warning("⚠️ No key — using static fallback data")

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

    return selected_tf, fred_api_key

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
    if is_live:
        st.success("✅ Live FRED data")
    else:
        st.warning("⚠️ **Static fallback data** — Enter a FRED API key in the sidebar for live values.")

    rows = [{"Currency": ccy, **vals} for ccy, vals in macro_data.items()]
    df = pd.DataFrame(rows).set_index("Currency")
    st.dataframe(df.style.background_gradient(cmap="RdYlGn", subset=["GDP", "Inflation", "Rates", "Unemployment"]), use_container_width=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title(f"💹 Macro Dashboard Pro v{config.version}")

    init_notification_state()

    # Get default FRED key
    default_key = ""
    try:
        if hasattr(st, "secrets") and "FRED_API_KEY" in st.secrets:
            default_key = st.secrets["FRED_API_KEY"]
        else:
            default_key = os.environ.get("FRED_API_KEY", "")
    except Exception:
        pass

    selected_tf, fred_api_key = render_sidebar(default_key)

    if not st.session_state.data_loaded:
        with st.spinner("Loading market data…"):
            st.session_state.data_by_timeframe = load_all_market_data()
            st.session_state.macro_data, st.session_state.macro_live = get_macro_data(fred_api_key)
            st.session_state.data_loaded = True
            st.session_state.last_refresh = datetime.now()

    data_by_timeframe = st.session_state.data_by_timeframe
    macro_data = st.session_state.macro_data
    macro_live = st.session_state.macro_live

    daily_data = data_by_timeframe.get('Daily', {})
    if daily_data:
        render_kpis(daily_data)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🌍 Macro Fundamentals",
        "📈 Technical Chart",
        "⏱️ 15-Min Entry",
        "🎯 Trading Ideas",
        "📅 Weekly Swing",
    ])

    with tab1:
        st.subheader("Market Overview")
        if daily_data:
            rows = []
            for pair, df in daily_data.items():
                if not df.empty:
                    price = df['Close'].iloc[-1]
                    change = df['Close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0
                    rows.append({"Pair": pair, "Price": price, "Change %": change, "Bars": len(df)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab2:
        st.subheader("🌍 Macro Fundamentals")
        render_macro_table(macro_data, macro_live)

    with tab3:
        st.subheader("📈 Technical Analysis Chart")
        avail = [p for p, d in daily_data.items() if not d.empty]
        if avail:
            c1, c2 = st.columns(2)
            pair = c1.selectbox("Pair", avail, key="chart_pair")
            tf = c2.selectbox("Timeframe", list(config.timeframes.keys()), key="chart_tf")
            df = data_by_timeframe.get(tf, {}).get(pair, pd.DataFrame())
            if not df.empty:
                df = analyzer.add_indicators(df)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3], subplot_titles=(f"{pair} — {tf}", "RSI"))
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                for ma, color in [('EMA_20', 'orange'), ('EMA_50', 'blue')]:
                    if ma in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=color)), row=1, col=1)
                if 'RSI' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                fig.update_layout(height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("⏱️ 15-Minute Entry Signal")
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
                    if sig["signal"] == 1: st.success("### 🟢 LONG")
                    elif sig["signal"] == -1: st.error("### 🔴 SHORT")
                    else: st.info("### ⚪ NO SIGNAL")
                c2.metric("Confidence", f"{sig['confidence']}/5")
                c3.metric("Price", f"{sig.get('price', 0):.5f}")

                for r in sig.get("reasons", []):
                    st.info(f"ℹ️ {r}")
            else:
                st.warning("Insufficient data for 15-minute analysis")

    with tab5:
        st.subheader("🎯 Trading Ideas")
        if st.button("🔄 Generate Trading Ideas", type="primary", key="gen_ideas_main"):
            with st.spinner("Analysing pairs..."):
                for tf in data_by_timeframe:
                    for p in data_by_timeframe[tf]:
                        data_by_timeframe[tf][p] = analyzer.add_indicators(data_by_timeframe[tf][p])

                ideas, skipped = generate_trading_ideas(data_by_timeframe)
                st.session_state.latest_ideas = ideas
                check_and_notify(ideas)

            if ideas:
                st.success(f"✅ Generated {len(ideas)} trading ideas")
                for idx, idea in enumerate(ideas):
                    direction = "📈" if idea['bias'] == 'Long' else "📉"
                    header = f"### {idx+1}. {idea['pair']} — {idea['bias'].upper()} {direction}"

                    if idea['conviction'] == "High":
                        st.success(header + " 🔔 HIGH CONVICTION")
                    else:
                        st.info(header)

                    cols = st.columns(5)
                    cols[0].metric("Entry", f"{idea['entry']:.5f}")
                    cols[1].metric("TP1", f"{idea['take_profit_1']:.5f}", delta=f"R:R {idea['risk_reward_1']:.2f}")
                    cols[2].metric("TP2", f"{idea['take_profit_2']:.5f}", delta=f"R:R {idea['risk_reward_2']:.2f}")
                    cols[3].metric("Stop Loss", f"{idea['stop_loss']:.5f}")
                    risk_pct = (abs(idea["entry"] - idea["stop_loss"]) / idea["entry"]) * 100
                    cols[4].metric("Risk %", f"{risk_pct:.2f}%")

                    st.markdown(f"**Thesis:** {idea['thesis']}")
                    st.caption(f"Stop method: {idea['stop_loss_method']} | Distance: {idea['stop_loss_pips']} pips")
                    st.divider()
            else:
                st.info("No trading ideas found with current market conditions.")

    with tab6:
        st.subheader("📅 Weekly Swing Trading")
        st.info("Coming soon: Swing analysis based on Weekly/Daily confluence.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
