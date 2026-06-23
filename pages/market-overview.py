import streamlit as st
from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import logging
import traceback
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from streamlit_autorefresh import st_autorefresh
from src.core import secrets
from src.services import alert_service
from src.core.config import default_config as config, CANDLE_STYLE, CHART_LAYOUT, EMA_COLORS, RSI_LINE, RSI_OB, RSI_OS
from src.core.analyzer import TechnicalAnalyzer as analyzer
from src.core.data_provider import fetch_data, get_macro_data, fetch_fred_series
from src.core.signals import (
    generate_trading_ideas,
    generate_weekly_swing_ideas, evaluate_trend_following_signal
)

st.set_page_config(
    page_title="Market Overview · Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

logger = logging.getLogger("ForexDashboard")

# ============================================================================
# EMAIL ALERTS  — sending + dedupe handled by the shared src/services/alert_service.
# This page only describes what a signal is and how to render it.
# ============================================================================

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
h1, h2, h3 { font-family: 'JetBrains Mono', monospace !important; }
.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.sig-buy  { background:#0d3b2a; border:1px solid #1a7a55; color:#4af0c4; padding:4px 12px; border-radius:3px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }
.sig-sell { background:#3b0d16; border:1px solid #7a1a2a; color:#f04a6a; padding:4px 12px; border-radius:3px; font-size:11px; font-family:'JetBrains Mono',monospace; display:inline-block; }
.alert-buy  { background:#0d3b2a; border-left:5px solid #4af0c4; padding:12px 18px; border-radius:8px; margin:8px 0; color:#e0e0e0; }
.alert-sell { background:#3b0d16; border-left:5px solid #f04a6a; padding:12px 18px; border-radius:8px; margin:8px 0; color:#e0e0e0; }
</style>
""", unsafe_allow_html=True)

def play_alert_sound(bias: str):
    """Browser-based audio alert via Web Audio API."""
    if bias == "Long":
        tones = [(600, 0.0, 0.25), (900, 0.3, 0.25)]
    else:
        tones = [(900, 0.0, 0.25), (600, 0.3, 0.25)]

    tone_js = "\n".join(
        f"""
        var o{i}=ctx.createOscillator(), g{i}=ctx.createGain();
        o{i}.connect(g{i}); g{i}.connect(ctx.destination);
        o{i}.frequency.value={freq}; o{i}.type='sine';
        g{i}.gain.setValueAtTime(0.35, ctx.currentTime+{start});
        g{i}.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime+{start}+{dur});
        o{i}.start(ctx.currentTime+{start}); o{i}.stop(ctx.currentTime+{start}+{dur}+0.05);
        """
        for i, (freq, start, dur) in enumerate(tones)
    )
    st.components.v1.html(f"""
    <script>
    (function(){{
        var ctx = new (window.AudioContext || window.webkitAudioContext)();
        {tone_js}
    }})();
    </script>""", height=0)

# ============================================================================
# NOTIFICATION PERSISTENCE  — file-backed dedupe via the shared service.
# Namespace "forex" preserves the existing forex_notify_cache.json file, so
# previously-alerted signals stay deduped across this change.
# ============================================================================
_NOTIFY = alert_service.NotifyCache("forex")


# ============================================================================
# EMAIL ALERT SENDER
# ============================================================================
def _build_email_html(ideas: List[Dict]) -> str:
    """Render a clean HTML email body for a list of new signal ideas."""
    rows_html = ""
    for idea in ideas:
        direction  = "📈 LONG" if idea["bias"] == "Long" else "📉 SHORT"
        bias_color = "#26a69a"  if idea["bias"] == "Long" else "#ef5350"
        rows_html += f"""
        <tr style="border-bottom:1px solid #2d3148;">
          <td style="padding:10px 14px;font-weight:700;color:#e0e0e0;">{idea['pair']}</td>
          <td style="padding:10px 14px;color:{bias_color};font-weight:700;">{direction}</td>
          <td style="padding:10px 14px;color:#e0e0e0;">{idea['entry']:.5f}</td>
          <td style="padding:10px 14px;color:#26a69a;">{idea['take_profit_1']:.5f}
              <span style="font-size:11px;color:#8b8fa8;"> R:R {idea['risk_reward_1']:.2f}</span></td>
          <td style="padding:10px 14px;color:#ef5350;">{idea['stop_loss']:.5f}</td>
          <td style="padding:10px 14px;color:#ffa726;">{idea['strength_score']}/10</td>
          <td style="padding:10px 14px;color:#c0c0c0;font-size:11px;">{idea['conviction']}</td>
        </tr>
        <tr>
          <td colspan="7" style="padding:4px 14px 12px;color:#8b8fa8;font-size:12px;">
            {idea.get('thesis', '')[:180]}
          </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<body style="background:#0e1117;font-family:'Courier New',monospace;color:#c0c0c0;margin:0;padding:20px;">
  <div style="max-width:760px;margin:0 auto;">
    <h2 style="color:#4af0c4;border-bottom:1px solid #2d3148;padding-bottom:10px;margin-bottom:20px;">
      🚨 Macro Dashboard — New Trading Signals
    </h2>
    <p style="color:#8b8fa8;font-size:13px;margin-bottom:20px;">
      {len(ideas)} new high-conviction signal{'s' if len(ideas) != 1 else ''} detected at
      <strong style="color:#e0e0e0;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</strong>
    </p>
    <table style="width:100%;border-collapse:collapse;background:#1a1d27;border-radius:8px;overflow:hidden;">
      <thead>
        <tr style="background:#000000;">
          <th style="padding:10px 14px;text-align:left;color:#8b8fa8;font-size:11px;letter-spacing:1px;">PAIR</th>
          <th style="padding:10px 14px;text-align:left;color:#8b8fa8;font-size:11px;letter-spacing:1px;">BIAS</th>
          <th style="padding:10px 14px;text-align:left;color:#8b8fa8;font-size:11px;letter-spacing:1px;">ENTRY</th>
          <th style="padding:10px 14px;text-align:left;color:#8b8fa8;font-size:11px;letter-spacing:1px;">TP1</th>
          <th style="padding:10px 14px;text-align:left;color:#8b8fa8;font-size:11px;letter-spacing:1px;">STOP LOSS</th>
          <th style="padding:10px 14px;text-align:left;color:#8b8fa8;font-size:11px;letter-spacing:1px;">SCORE</th>
          <th style="padding:10px 14px;text-align:left;color:#8b8fa8;font-size:11px;letter-spacing:1px;">CONVICTION</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p style="color:#8b8fa8;font-size:11px;margin-top:24px;border-top:1px solid #2d3148;padding-top:12px;">
      Macro Dashboard Pro · Auto-alerts fire only for High conviction or score ≥ 8 · Each unique entry fires once.
    </p>
  </div>
</body>
</html>"""


def send_email_alert(new_ideas: List[Dict]) -> None:
    """Send a formatted HTML email for newly detected high-conviction signals.

    SMTP delivery and config checks live in alert_service; this only builds the
    macro-signal subject / plain / HTML bodies.
    """
    if not new_ideas:
        return

    subject = (
        f"🚨 {len(new_ideas)} New Signal{'s' if len(new_ideas) != 1 else ''} — "
        f"{', '.join(i['pair'] for i in new_ideas[:3])}"
        f"{'…' if len(new_ideas) > 3 else ''}"
    )

    plain_lines = [f"NEW TRADING SIGNALS — {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    for idea in new_ideas:
        direction = "LONG" if idea["bias"] == "Long" else "SHORT"
        plain_lines.append(
            f"{idea['pair']} {direction}  |  Entry: {idea['entry']:.5f}  |  "
            f"TP1: {idea['take_profit_1']:.5f}  |  SL: {idea['stop_loss']:.5f}  |  "
            f"Score: {idea['strength_score']}/10  |  {idea['conviction']}"
        )
        plain_lines.append(f"  Thesis: {idea.get('thesis','')[:160]}")
        plain_lines.append("")

    ok, msg = alert_service.send_email(subject, _build_email_html(new_ideas),
                                       "\n".join(plain_lines))
    if ok:
        logger.info("Signal email sent (%d ideas)", len(new_ideas))
    else:
        logger.warning("Signal email not sent: %s", msg)


# ============================================================================
# NOTIFICATION SYSTEM
# ============================================================================
def init_notification_state() -> None:
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'notified_keys' not in st.session_state:
        st.session_state.notified_keys = _NOTIFY.load()
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
        _NOTIFY.save(st.session_state.notified_keys)
        send_email_alert(new_alerts)   # ← email fired once per unique signal
        play_alert_sound(new_alerts[0]['bias'])

    for idea in new_alerts:
        direction = "📈 LONG" if idea['bias'] == 'Long' else "📉 SHORT"

        # Toast
        st.toast(
            f"🚨 NEW SIGNAL: {idea['pair']} {direction}\n"
            f"Score: {idea['strength_score']}/10 | Entry: {idea['entry']:.5f}",
            icon="🔔",
        )

        # Visual Banner
        cls = "alert-buy" if idea['bias'] == "Long" else "alert-sell"
        st.markdown(f"""
        <div class="{cls}">
            <strong>{idea['pair']}</strong> — {direction} detected at <strong>{idea['entry']:.5f}</strong>
            (Score: {idea['strength_score']}/10)
        </div>
        """, unsafe_allow_html=True)
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
def render_sidebar(fred_api_key: str = "") -> str:
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")

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
        if st.button("↺ Refresh Now", width="stretch"):
            clear_data_cache()
            st.rerun()

        elapsed = int((datetime.now() - st.session_state.get('last_refresh', datetime.now())).total_seconds())
        st.caption(f"Age: {elapsed}s / {config.cache_ttl}s")

        st.divider()

        # ── Email alert status ────────────────────────────────────────────────
        st.subheader("📧 Email Alerts")
        if alert_service.email_configured():
            st.success(f"✅ Alerts → {alert_service.email_recipient()}")
            st.caption(f"Sending via {secrets.email_config().get('user', '')}")
            if st.button("✉ Send test email", width="stretch"):
                ok, msg = alert_service.send_email(
                    "Market Overview — test alert",
                    "<p style='font-family:monospace;color:#4af0c4;'>✅ Market Overview email "
                    "alerts are wired up correctly.</p>",
                    "Market Overview email alerts are wired up correctly.")
                (st.success if ok else st.error)(msg)
        else:
            st.warning("⚠️ Email not configured — add [gmail] to .streamlit/secrets.toml "
                       "(or set GMAIL_* env vars)")

        st.divider()
        log = st.session_state.get('notification_log', [])
        if log:
            for entry in reversed(log[-10:]):
                icon = "📈" if entry['bias'] == 'Long' else "📉"
                st.markdown(f"**{entry['time']}** {icon} **{entry['pair']}** {entry['bias']} — R:R {entry['rr']:.2f}")
            if st.button("🗑️ Clear Alerts"):
                st.session_state.notification_log = []
                st.session_state.notified_keys = set()
                _NOTIFY.reset()
                st.rerun()
        else:
            st.caption("No alerts yet.")

        st.divider()
        render_sidebar_nav()

    return selected_tf


def render_kpis(daily_data: Dict) -> None:
    # Representative headline instruments, all drawn from the registry universe.
    kpi_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "XAU/USD", "XAG/USD"]
    cols = st.columns(len(kpi_pairs))
    for i, pair in enumerate(kpi_pairs):
        df = daily_data.get(pair)
        with cols[i]:
            if df is not None and not df.empty:
                price = df["Close"].iloc[-1]
                change = df["Close"].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0
                # Domain convention: 5 decimals for FX (<100), 2–3 for JPY
                # crosses / metals / indices.
                fmt = f"{price:.5f}" if abs(price) < 100 else f"{price:,.2f}"
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
                 width="stretch")


def render_overview_tab(daily_data: Dict):
    st.subheader("Market Overview")
    if daily_data:
        rows = []
        for pair, df in daily_data.items():
            if not df.empty:
                price = df['Close'].iloc[-1]
                change = df['Close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0
                rows.append({"Pair": pair, "Price": price, "Change %": change, "Bars": len(df)})
        st.dataframe(pd.DataFrame(rows), width="stretch")


def render_mtf_matrix_tab(data_by_timeframe: Dict):
    st.subheader("🧭 Multi-Timeframe Matrix")
    mtf_rows = []
    # Explicitly requested timeframes
    target_tfs = ["Weekly", "Daily", "4 Hour", "Hourly"]

    for pair in config.assets.keys():
        sentiments = analyzer.get_mtf_sentiment(data_by_timeframe, pair)
        tf_vals = {tf: sentiments.get(tf, "N/A") for tf in target_tfs}

        # Overall = net score across the timeframes (+1 Bullish, -1 Bearish,
        # 0 for Neutral/N/A). Sign of the net decides the bias label; the score
        # is shown alongside so a 4/0 read is distinguishable from a 2/1.
        bull = sum(1 for v in tf_vals.values() if v == "Bullish")
        bear = sum(1 for v in tf_vals.values() if v == "Bearish")
        net = bull - bear
        if net > 0:
            overall = f"🟢 {net:+d}"
        elif net < 0:
            overall = f"🔴 {net:+d}"
        else:
            overall = "⚪ 0"

        # "Overall" sits right after the Pair index (dict insertion order).
        row = {"Pair": pair, "Overall": overall, **tf_vals}
        mtf_rows.append(row)

    mtf_df = pd.DataFrame(mtf_rows).set_index("Pair")

    def color_sentiment(val):
        # Matches both the per-timeframe word cells ("Bullish"/"Bearish") and the
        # icon-based "Overall" cell ("🟢 +3" / "🔴 -2") so all colour consistently.
        text = str(val)
        if text.startswith("Bullish") or "🟢" in text: return "background-color: #249d53; color: white;"
        if text.startswith("Bearish") or "🔴" in text: return "background-color: #ca2427; color: white;"
        return ""

    st.table(mtf_df.style.map(color_sentiment))


# ── Perfect Order (Lien, Ch.16) ──────────────────────────────────────────────
# A "perfect order" is the 10/20/50/100/200 SMA stack in strict sequence:
#   bullish → SMA10 > SMA20 > SMA50 > SMA100 > SMA200
#   bearish → SMA10 < SMA20 < SMA50 < SMA100 < SMA200
# It signals a strong trend (confirm with ADX > 20 and rising). Entry is 5
# candles after the order first forms; stop is the formation day's low (long) /
# high (short); exit when the order breaks (10 crosses back through 20).
PERFECT_ORDER_SMAS = [10, 20, 50, 100, 200]
PERFECT_ORDER_COLORS = {10: "#00e0ff", 20: "#00ff41", 50: "#ffcc00",
                        100: "#ff8c00", 200: "#ff3366"}


def _perfect_order_state(values) -> str:
    """bullish / bearish / none for one ordered list of [SMA10..SMA200]."""
    if any(pd.isna(v) for v in values):
        return "none"
    if values[0] > values[1] > values[2] > values[3] > values[4]:
        return "bullish"
    if values[0] < values[1] < values[2] < values[3] < values[4]:
        return "bearish"
    return "none"


def _perfect_order_runs(states):
    """Contiguous [start, end, state] spans where the order holds (for shading)."""
    runs, i, n = [], 0, len(states)
    while i < n:
        if states[i] != "none":
            j = i
            while j + 1 < n and states[j + 1] == states[i]:
                j += 1
            runs.append((i, j, states[i]))
            i = j + 1
        else:
            i += 1
    return runs


@st.cache_data(ttl=config.cache_ttl, show_spinner=False)
def _fetch_perfect_order_daily(symbol: str) -> pd.DataFrame:
    """2y of daily bars — enough history to seed the 200-day SMA."""
    return fetch_data(symbol, "1d", "2y")


def render_technical_chart_tab(data_by_timeframe: Dict):
    st.subheader("📈 Technical Analysis Chart — Perfect Order")
    st.caption("Moving averages in **perfect order** (10>20>50>100>200 for an "
               "uptrend, reversed for a downtrend) flag a strong trend. Confirm "
               "with ADX > 20 and rising; enter 5 candles after the order forms, "
               "stop at the formation day's low/high, exit when the order breaks.")

    daily_data = data_by_timeframe.get('Daily', {})
    avail = [p for p, d in daily_data.items() if not d.empty]
    if not avail:
        st.info("No daily data available.")
        return

    pair = st.selectbox("Pair", avail, key="chart_pair")
    symbol = config.assets.get(pair)

    df = _fetch_perfect_order_daily(symbol) if symbol else pd.DataFrame()
    if df.empty or len(df) < PERFECT_ORDER_SMAS[-1]:
        st.warning(f"Need ≥{PERFECT_ORDER_SMAS[-1]} daily bars for the 200-SMA — "
                   f"only {len(df)} returned for {pair}.")
        return

    df = analyzer.add_indicators(df)
    for p in PERFECT_ORDER_SMAS:
        df[f"SMA_{p}"] = df["Close"].rolling(p).mean()
    states = [_perfect_order_state([row[f"SMA_{p}"] for p in PERFECT_ORDER_SMAS])
              for _, row in df.iterrows()]
    df["po_state"] = states
    runs = _perfect_order_runs(states)

    # ── Current read + entry plan ────────────────────────────────────────────
    current = states[-1]
    adx = float(df["ADX"].iloc[-1]) if "ADX" in df.columns and pd.notna(df["ADX"].iloc[-1]) else 0.0
    adx_prev = float(df["ADX"].iloc[-5]) if "ADX" in df.columns and len(df) > 5 and pd.notna(df["ADX"].iloc[-5]) else adx
    adx_rising = adx > adx_prev
    icon = {"bullish": "🟢 BULLISH", "bearish": "🔴 BEARISH", "none": "⚪ NONE"}[current]

    m1, m2, m3 = st.columns(3)
    m1.metric("Perfect Order", icon)
    m2.metric("ADX", f"{adx:.1f}", delta=("rising" if adx_rising else "falling"),
              delta_color="normal" if adx_rising else "inverse",
              help="Strategy wants ADX > 20 and rising.")

    entry_marker = stop_marker = None
    plan = "—"
    if current != "none" and runs:
        f_start, f_end, f_state = runs[-1]
        bars_held = (len(df) - 1) - f_start
        entry_ok = adx > 20 and adx_rising
        if bars_held >= 5:
            entry_idx = f_start + 5
            entry_px = float(df["Close"].iloc[entry_idx])
            # Stop = formation day's low (long) / high (short).
            stop_px = (float(df["Low"].iloc[f_start]) if f_state == "bullish"
                       else float(df["High"].iloc[f_start]))
            side = "LONG" if f_state == "bullish" else "SHORT"
            risk = abs(entry_px - stop_px)
            plan = (f"{side} · entry {entry_px:.5f} (5 bars after formation) · "
                    f"stop {stop_px:.5f} · risk {risk:.5f}")
            entry_marker = (df.index[entry_idx], entry_px, side)
            stop_marker = (df.index[f_start], stop_px)
        else:
            plan = (f"Order forming ({bars_held}/5 bars) — wait for the 5th candle "
                    f"after formation before entering.")
        m3.metric("Entry Trigger", "✅ READY" if (entry_ok and bars_held >= 5)
                  else ("⏳ WAIT" if bars_held < 5 else "⚠ ADX weak"))
    else:
        m3.metric("Entry Trigger", "—")

    if plan != "—":
        st.info(f"**Plan:** {plan}")

    # ── Chart: price + 5-SMA stack + shaded perfect-order regions, ADX below ──
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.72, 0.28], subplot_titles=(f"{pair} — Daily", "ADX"))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Price",
                                 **CANDLE_STYLE), row=1, col=1)
    for p in PERFECT_ORDER_SMAS:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{p}"], name=f"SMA{p}",
                                 line=dict(color=PERFECT_ORDER_COLORS[p], width=1.3)),
                      row=1, col=1)
    # shade where the order holds (green bullish / red bearish)
    for s_idx, e_idx, s_state in runs:
        fig.add_vrect(x0=df.index[s_idx], x1=df.index[e_idx], row=1, col=1,
                      line_width=0, layer="below",
                      fillcolor="rgba(0,255,65,0.10)" if s_state == "bullish"
                      else "rgba(255,51,102,0.10)")
    if entry_marker:
        ex, ey, eside = entry_marker
        fig.add_trace(go.Scatter(x=[ex], y=[ey], mode="markers", name=f"Entry ({eside})",
                                 marker=dict(symbol="triangle-up" if eside == "LONG"
                                 else "triangle-down", size=14, color="#ffffff",
                                 line=dict(width=1.5, color="#00ff41"))), row=1, col=1)
    if stop_marker:
        sx, sy = stop_marker
        fig.add_hline(y=sy, line=dict(color="#ff3366", width=1, dash="dot"), row=1, col=1)

    if "ADX" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX",
                                 line=dict(color="#00e0ff", width=1.4)), row=2, col=1)
        fig.add_hline(y=20, line=dict(color="#ffcc00", width=1, dash="dot"), row=2, col=1)
    fig.update_layout(height=640, **CHART_LAYOUT)
    st.plotly_chart(fig, width="stretch")


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
                st.plotly_chart(fig, width="stretch")


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
        fig = make_subplots(rows=2, cols=4, subplot_titles=list(valid_data.keys()),
                            vertical_spacing=0.18, horizontal_spacing=0.08)
        colors = ["#4af0c4", "#f04a6a", "#f0a030", "#5090e0", "#c878e0", "#e05050", "#50c878", "#e0d050"]

        for i, (name, df) in enumerate(valid_data.items()):
            row, col = divmod(i, 4)
            hex_c = colors[i % len(colors)].lstrip('#')
            r_c, g_c, b_c = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["value"], name=name,
                line=dict(color=colors[i % len(colors)], width=1.5),
                fill="tozeroy", fillcolor=f"rgba({r_c},{g_c},{b_c},0.08)",
            ), row=row + 1, col=col + 1)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e17",
            font=dict(family="JetBrains Mono", size=9, color="#8899bb"),
            showlegend=False, height=400, margin=dict(l=10, r=10, t=35, b=10),
        )
        for ann in fig.layout.annotations:
            ann.font = dict(size=9, color="#8899bb", family="JetBrains Mono")
        fig.update_xaxes(gridcolor="#1e2d45")
        fig.update_yaxes(gridcolor="#1e2d45")

        st.plotly_chart(fig, width="stretch")

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

        if "10Y Treasury" in valid_data and "2Y Treasury" in valid_data:
            t10 = valid_data["10Y Treasury"]["value"].iloc[-1]
            t2 = valid_data["2Y Treasury"]["value"].iloc[-1]
            if t10 - t2 < 0:
                notes.append("🔴 Inverted yield curve — recession signal"); score -= 1
            else:
                notes.append("🟢 Normal yield curve"); score += 1

        regime = "BULLISH 🟢" if score > 0 else ("BEARISH 🔴" if score < 0 else "NEUTRAL ⚪")
        st.metric("Overall Macro Regime", regime)
        for n in notes:
            st.markdown(f"<div style='font-size:11px;font-family:JetBrains Mono,monospace;color:#8899bb;margin:3px 0'>{n}</div>", unsafe_allow_html=True)
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

    if not weekly_data:
        st.warning("No weekly data available — click ↺ Refresh in the sidebar.")
        return


    ideas = generate_weekly_swing_ideas(data_by_timeframe)


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
                st.plotly_chart(fig, width="stretch")

    if aligned:
        st.markdown(f"#### ✅ Daily-Confirmed Setups ({len(aligned)})")
        for idea in aligned:
            _render_idea(idea)

    if other:
        st.markdown(f"#### Other Setups ({len(other)})")
        for idea in other:
            _render_idea(idea)


# ============================================================================
# FOREX TREND FOLLOWING TAB
# ============================================================================



def render_volume_profile_tab(data_by_timeframe: Dict) -> None:
    st.subheader("🔊 Volume Profile Analysis")
    st.caption("Analyzes the distribution of volume over price to find Point of Control (POC) and Value Areas.")

    avail_pairs = list(config.assets.keys())
    c1, c2, c3 = st.columns(3)
    pair = c1.selectbox("Select Pair", avail_pairs, key="vp_pair")
    tf = c2.selectbox("Timeframe", ["Daily", "4 Hour", "Hourly"], key="vp_tf")
    n_bins = c3.slider("Price Bins", 40, 200, 80, 10, key="vp_bins")

    df = data_by_timeframe.get(tf, {}).get(pair, pd.DataFrame())
    if df.empty:
        st.warning(f"No data available for {pair} on {tf} timeframe.")
        return

    vp = analyzer.compute_volume_profile(df, n_bins)
    levels = analyzer.detect_levels(vp)
    va_lo, va_hi = analyzer.value_area(vp)

    cur = float(df["Close"].iloc[-1])
    poc = levels["poc"]

    # KPIs
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last Price", f"{cur:.5f}")
    m2.metric("Point of Control", f"{poc:.5f}")
    m3.metric("Value Area High", f"{va_hi:.5f}")
    m4.metric("Value Area Low", f"{va_lo:.5f}")

    # Plot
    fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2], shared_yaxes=True, horizontal_spacing=0.01)

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name="Price", **CANDLE_STYLE), row=1, col=1)

    # POC and VA Lines
    fig.add_hline(y=poc, line=dict(color="#f59e0b", width=2, dash="dash"), annotation_text="POC", row=1, col=1)
    fig.add_hrect(y0=va_lo, y1=va_hi, fillcolor="rgba(6,214,232,0.05)", line_width=0, row=1, col=1)

    # Volume Profile Bar Chart
    fig.add_trace(go.Bar(x=vp["volume"], y=vp["price"], orientation="h", name="Volume Profile",
                         marker=dict(color="rgba(6,214,232,0.45)", line_width=0)), row=1, col=2)

    fig.update_layout(height=600, **CHART_LAYOUT)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    st.plotly_chart(fig, width="stretch")


def render_trend_following_tab(data_by_timeframe: Dict) -> None:
    st.subheader("📈 Forex Trend Following")
    st.caption(
        "Scans all pairs for clean trending conditions — ADX strength, EMA alignment, and momentum. "
        "Use as a pair-selection filter before drilling into the Technical Chart or Pivots & Fibonacci tabs."
    )

    # ── Controls ─────────────────────────────────────────────────────────────
    col_tf, col_adx, col_dir = st.columns(3)
    scan_tf  = col_tf.selectbox("Scan Timeframe", ["Daily", "4 Hour", "Weekly"], key="trf_tf")
    adx_min  = col_adx.slider("Min ADX (trend strength)", 15, 40, 20, key="trf_adx")
    dir_filt = col_dir.radio("Direction", ["All", "Uptrend", "Downtrend"], horizontal=True, key="trf_dir")

    tf_data = data_by_timeframe.get(scan_tf, {})
    if not tf_data:
        st.warning(f"No data for {scan_tf}. Try refreshing.")
        return

    # ── Build scanner rows ────────────────────────────────────────────────────
    rows = []
    for pair, df_raw in tf_data.items():
        if df_raw.empty or len(df_raw) < 55:
            continue
        df = analyzer.add_indicators(df_raw.copy())
        if df.empty:
            continue

        last = df.iloc[-1]
        price  = float(last.get("Close",  0.0) or 0.0)
        ema20  = float(last.get("EMA_20", price) or price)
        ema50  = float(last.get("EMA_50", price) or price)
        adx    = float(last.get("ADX",    0.0) or 0.0)
        adx_p  = float(last.get("ADX_Pos", 0.0) or 0.0)
        adx_n  = float(last.get("ADX_Neg", 0.0) or 0.0)
        rsi    = float(last.get("RSI",    50.0) or 50.0)
        atr    = float(last.get("ATR",    0.0) or 0.0)

        if adx < adx_min:
            continue

        trend_dir = "Uptrend" if ema20 > ema50 else "Downtrend"
        if dir_filt != "All" and trend_dir != dir_filt:
            continue

        # Price change vs previous close
        change_pct = 0.0
        if len(df) >= 2:
            prev = float(df.iloc[-2].get("Close", price) or price)
            change_pct = (price - prev) / prev * 100 if prev else 0.0

        # Use consolidated signal logic
        label, score, max_s, conds, direction = evaluate_trend_following_signal(df, min_conditions=3)

        # EMA alignment: +1 each for price>EMA20, EMA20>EMA50
        ema_align = int(price > ema20) + int(ema20 > ema50)

        rows.append({
            "Pair":        pair,
            "Direction":   trend_dir,
            "Price":       price,
            "Change %":    round(change_pct, 3),
            "ADX":         round(adx, 1),
            "+DI":         round(adx_p, 1),
            "-DI":         round(adx_n, 1),
            "EMA Align":   f"{'✅' * ema_align}{'⬜' * (2 - ema_align)}",
            "RSI":         round(rsi, 1),
            "ATR":         round(atr, 5),
            "Score":       score,
            "Signal":      label,
            "_df":         df,          # hidden, used for chart
        })

    if not rows:
        st.info(f"No pairs meet ADX ≥ {adx_min} on the {scan_tf} timeframe. Lower the ADX threshold or try another timeframe.")
        return

    rows.sort(key=lambda r: r["Score"], reverse=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    up_count   = sum(1 for r in rows if r["Direction"] == "Uptrend")
    down_count = len(rows) - up_count
    avg_adx    = sum(r["ADX"] for r in rows) / len(rows)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trending Pairs",  len(rows))
    m2.metric("Uptrends",        up_count,   delta=f"+{up_count}")
    m3.metric("Downtrends",      down_count, delta=f"-{down_count}", delta_color="inverse")
    m4.metric("Avg ADX",         f"{avg_adx:.1f}")

    st.divider()

    # ── Scanner table ─────────────────────────────────────────────────────────
    display_cols = ["Pair", "Direction", "Price", "Change %", "ADX", "+DI", "-DI", "EMA Align", "RSI", "ATR", "Score", "Signal"]
    display_df = pd.DataFrame([{k: r[k] for k in display_cols} for r in rows])

    def _style_row(row):
        base = [""] * len(row)
        if row["Direction"] == "Uptrend":
            base[1] = "background-color:#0d2d1f; color:#4af0c4;"
        else:
            base[1] = "background-color:#2d0d0d; color:#f04a6a;"
        if row["Score"] >= 8:
            base[-1] = "color:#ffd700; font-weight:700;"
        elif row["Score"] >= 6:
            base[-1] = "color:#66bb6a;"
        return base

    st.dataframe(
        display_df.style.apply(_style_row, axis=1),
        width="stretch",
        height=min(38 * len(rows) + 38, 450),
    )

    st.divider()

    # ── Per-pair chart ────────────────────────────────────────────────────────
    st.markdown("#### 📊 Trend Chart")
    pair_names = [r["Pair"] for r in rows]
    selected_pair = st.selectbox("Select pair to chart", pair_names, key="trf_chart_pair")
    row_data = next(r for r in rows if r["Pair"] == selected_pair)
    df_chart  = row_data["_df"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=[
            f"{selected_pair} — {scan_tf}  |  ADX {row_data['ADX']}  |  {row_data['Direction']}",
            "ADX / ±DI",
            "RSI (14)",
        ],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart["Open"], high=df_chart["High"],
        low=df_chart["Low"],   close=df_chart["Close"],
        name="Price", showlegend=False, **CANDLE_STYLE,
    ), row=1, col=1)

    # EMAs
    for ema_col, color in EMA_COLORS.items():
        if ema_col in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart[ema_col],
                line=dict(color=color, width=1.4), name=ema_col,
            ), row=1, col=1)

    # Bollinger Bands (light shading)
    if all(c in df_chart.columns for c in ["BB_Upper", "BB_Lower", "BB_Middle"]):
        for band, color, dash in [("BB_Upper", "#4fc3f7", "dot"), ("BB_Middle", "#90a4ae", "solid"), ("BB_Lower", "#4fc3f7", "dot")]:
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart[band],
                line=dict(color=color, width=0.8, dash=dash),
                name=band, opacity=0.5,
            ), row=1, col=1)

    # ADX panel
    if "ADX" in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["ADX"],
            line=dict(color="#ffa726", width=1.8), name="ADX",
        ), row=2, col=1)
    if "ADX_Pos" in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["ADX_Pos"],
            line=dict(color="#26a69a", width=1.2, dash="dot"), name="+DI",
        ), row=2, col=1)
    if "ADX_Neg" in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["ADX_Neg"],
            line=dict(color="#ef5350", width=1.2, dash="dot"), name="-DI",
        ), row=2, col=1)
    # Trend-strength reference lines
    fig.add_hline(y=25, line=dict(color="#ffa726", width=0.7, dash="dash"), row=2, col=1)
    fig.add_hline(y=adx_min, line=dict(color="#78909c", width=0.7, dash="dot"), row=2, col=1)

    # RSI panel
    if "RSI" in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["RSI"],
            line=RSI_LINE, name="RSI",
        ), row=3, col=1)
        fig.add_hline(y=70, line=RSI_OB, row=3, col=1)
        fig.add_hline(y=50, line=dict(color="#78909c", width=0.6, dash="dot"), row=3, col=1)
        fig.add_hline(y=30, line=RSI_OS, row=3, col=1)

    fig.update_layout(height=680, **CHART_LAYOUT)
    st.plotly_chart(fig, width="stretch")

    # ── Trend read-out ────────────────────────────────────────────────────────
    st.markdown("##### Trend Read-out")
    rd = row_data
    direction_icon = "📈" if rd["Direction"] == "Uptrend" else "📉"
    adx_label = (
        "Very Strong (>35)" if rd["ADX"] > 35 else
        "Strong (25–35)"    if rd["ADX"] > 25 else
        "Developing (20–25)" if rd["ADX"] > 20 else
        "Weak (<20)"
    )
    st.markdown(
        f"{direction_icon} **{rd['Direction']}** &nbsp;|&nbsp; "
        f"ADX **{rd['ADX']}** — {adx_label} &nbsp;|&nbsp; "
        f"+DI {rd['+DI']} / −DI {rd['-DI']} &nbsp;|&nbsp; "
        f"RSI **{rd['RSI']}** &nbsp;|&nbsp; "
        f"Score **{rd['Score']}/10**"
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title(f"💹 Macro Dashboard Pro v{config.version}")

    # ── Auto-refresh every 5 minutes ─────────────────────────────────────────
    # st_autorefresh returns a monotonically increasing counter (0 on first load).
    # On every tick after the first we bust the data cache so fresh prices are
    # fetched and signals re-evaluated — identical to clicking ↺ Refresh.
    _refresh_count = st_autorefresh(interval=300_000, key="dashboard_autorefresh")
    if _refresh_count > 0 and st.session_state.get("_last_refresh_count", -1) != _refresh_count:
        st.session_state["_last_refresh_count"] = _refresh_count
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.session_state.last_refresh = datetime.now()

    init_notification_state()

    fred_api_key = secrets.fred_api_key()

    selected_tf = render_sidebar(fred_api_key)

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
        "🏛 Macro Dashboard",            # Step 2 — deep FRED macro
        "📅 Weekly Swing",              # Step 3 — weekly direction filter
        "🧭 Multi-Timeframe Matrix",    # Step 4 — timeframe alignment
        "📈 Technical Chart",           # Step 5 — chart analysis per pair
        "🛒 Pivots & Fibonacci",        # Step 6 — session key levels
        "🎯 Trading Ideas",             # Step 7 — auto-refreshed setups
        "🔥 Trend Following",           # Step 8 — trending pair scanner
        "🔊 Volume Profile",            # Step 9 — volume distribution
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
        render_trading_ideas_tab()
    with tabs[8]:
        render_trend_following_tab(data_by_timeframe)
    with tabs[9]:
        render_volume_profile_tab(data_by_timeframe)

# Streamlit runs this file as a page (imported, not __main__), so call
# unconditionally — matching every other page in pages/.
try:
    main()
except Exception as e:
    st.error(f"Application error: {e}")
    logger.error(traceback.format_exc())