import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import pytz

from src.core.signals import score_setup

st.set_page_config(
    page_title="Setup Ranker · Trading System",
    page_icon="🎰",
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
    .hero{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);border:1px solid #21262d;border-radius:16px;padding:24px 32px;margin-bottom:20px;position:relative;overflow:hidden;}
    .metric-box{background:var(--background-color);border:1px solid var(--border,#21262d);border-radius:8px;padding:12px;text-align:center;}
    .metric-value{font-size:20px;font-weight:700;color:#c9d1d9;}
    .metric-label{font-size:10px;color:#8b949e;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}
    .rank-card{border-radius:12px;padding:16px 18px;margin-bottom:10px;border:1px solid #21262d;background:#161b22;transition:border-color .2s;}
    .rank-card-a{border-left:4px solid #3fb950;}
    .rank-card-b{border-left:4px solid #388bfd;}
    .rank-card-c{border-left:4px solid #e3b341;}
    .rank-card-d{border-left:4px solid #f85149;}
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
    "EUR/USD":    {"ticker": "EURUSD=X", "pip_size": 0.0001, "pip": 10.0},
    "GBP/USD":    {"ticker": "GBPUSD=X", "pip_size": 0.0001, "pip": 10.0},
    "AUD/USD":    {"ticker": "AUDUSD=X", "pip_size": 0.0001, "pip": 10.0},
    "NZD/USD":    {"ticker": "NZDUSD=X", "pip_size": 0.0001, "pip": 10.0},
    "USD/JPY":    {"ticker": "USDJPY=X", "pip_size": 0.01,   "pip": 9.09},
    "USD/CHF":    {"ticker": "USDCHF=X", "pip_size": 0.0001, "pip": 10.8},
    "USD/CAD":    {"ticker": "USDCAD=X", "pip_size": 0.0001, "pip": 7.4},
    "EUR/GBP":    {"ticker": "EURGBP=X", "pip_size": 0.0001, "pip": 12.5},
    "EUR/JPY":    {"ticker": "EURJPY=X", "pip_size": 0.01,   "pip": 9.09},
    "GBP/JPY":    {"ticker": "GBPJPY=X", "pip_size": 0.01,   "pip": 9.09},
    "AUD/JPY":    {"ticker": "AUDJPY=X", "pip_size": 0.01,   "pip": 9.09},
    "EUR/AUD":    {"ticker": "EURAUD=X", "pip_size": 0.0001, "pip": 6.3},
    "GBP/AUD":    {"ticker": "GBPAUD=X", "pip_size": 0.0001, "pip": 6.3},
    "EUR/CAD":    {"ticker": "EURCAD=X", "pip_size": 0.0001, "pip": 7.4},
    "GBP/CAD":    {"ticker": "GBPCAD=X", "pip_size": 0.0001, "pip": 7.4},
    "USD/ZAR":    {"ticker": "USDZAR=X", "pip_size": 0.0001, "pip": 0.55},
    "EUR/ZAR":    {"ticker": "EURZAR=X", "pip_size": 0.0001, "pip": 0.55},
    "GBP/ZAR":    {"ticker": "GBPZAR=X", "pip_size": 0.0001, "pip": 0.55},
    "🥇 Gold":    {"ticker": "GC=F",     "pip_size": 0.10,   "pip": 10.0},
    "🥈 Silver":  {"ticker": "SI=F",     "pip_size": 0.01,   "pip": 10.0},
    "🪙 Platinum":{"ticker": "PL=F",     "pip_size": 0.10,   "pip": 10.0},
}

TYPICAL_SPREADS = {
    "EUR/USD": 1.2, "GBP/USD": 1.5, "AUD/USD": 1.5, "NZD/USD": 2.0,
    "USD/JPY": 1.2, "USD/CHF": 2.0, "USD/CAD": 2.0, "EUR/GBP": 1.5,
    "EUR/JPY": 2.0, "GBP/JPY": 3.0, "AUD/JPY": 2.5, "EUR/AUD": 3.0,
    "GBP/AUD": 3.5, "EUR/CAD": 3.0, "GBP/CAD": 3.5, "USD/ZAR": 80.0,
    "EUR/ZAR": 90.0, "GBP/ZAR": 90.0,
    "🥇 Gold": 3.0, "🥈 Silver": 5.0, "🪙 Platinum": 8.0,
}


# ══════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily(ticker: str, days: int = 300) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_4h(ticker: str) -> pd.DataFrame:
    try:
        end   = datetime.now(pytz.utc)
        start = end - timedelta(days=90)
        df    = yf.download(ticker, start=start, end=end,
                            interval="1h", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df.resample("4h").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last",  "Volume": "sum",
        }).dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_weekly(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="2y", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df.resample("W").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last",  "Volume": "sum",
        }).dropna()
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# SCORING  (delegates to the shared engine in src/core/signals.py so this
# page and the app's Trading Ideas tab always agree for a given pair/direction)
# ══════════════════════════════════════════════════════════════════

def score_pair(pair: str, info: dict, direction: str) -> dict:
    """Fetch this page's data and score it with the unified checklist."""
    ticker   = info["ticker"]
    pip_size = info["pip_size"]

    df_w  = fetch_weekly(ticker)
    df_d  = fetch_daily(ticker)
    df_4h = fetch_4h(ticker)

    spread = TYPICAL_SPREADS.get(pair, 0.0)
    result = score_setup(df_w, df_d, df_4h, direction, pip_size, spread)
    result["pair"] = pair
    return result


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🎰 Setup Ranker")
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

    direction = st.radio("Scan direction", ["LONG", "SHORT", "Both"], horizontal=True)
    min_score = st.slider("Min score to show", 0, 10, 5)
    st.divider()
    if st.button("🔄 Rescan All Pairs", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"🕐 {datetime.now().strftime('%H:%M')} · TTL 5min")


# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero">
  <div style="font-size:26px;font-weight:700;color:#e6edf3;">🎰 Auto Setup Ranker</div>
  <div style="color:#8b949e;font-size:13px;margin-top:4px;">
    Scores all 21 instruments across 10 criteria · Weekly + Daily + 4H · Ranks by readiness
  </div>
  <div style="font-size:12px;color:#388bfd;margin-top:6px;">
    🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M')} · Direction: {direction}
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SCAN
# ══════════════════════════════════════════════════════════════════

directions_to_scan = (["LONG", "SHORT"] if direction == "Both"
                       else [direction])

prog  = st.progress(0, text="Scoring pairs…")
all_results = []
pairs_list  = list(INSTRUMENTS.items())
total_scans = len(pairs_list) * len(directions_to_scan)
done        = 0

for pair_name, info in pairs_list:
    for d in directions_to_scan:
        done += 1
        prog.progress(done / total_scans, text=f"Scoring {pair_name} {d}…")
        try:
            r = score_pair(pair_name, info, d)
            if r["score"] >= min_score:
                all_results.append(r)
        except Exception:
            pass

prog.empty()
all_results.sort(key=lambda x: -x["score"])

# ══════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════

grade_counts = {g: sum(1 for r in all_results if r["grade"] == g) for g in ["A", "B", "C", "D"]}
k1, k2, k3, k4, k5 = st.columns(5)
for col_, val_, lbl_, c_ in [
    (k1, grade_counts.get("A", 0), "Grade A (8–10)", "#3fb950"),
    (k2, grade_counts.get("B", 0), "Grade B (6–7)",  "#388bfd"),
    (k3, grade_counts.get("C", 0), "Grade C (4–5)",  "#e3b341"),
    (k4, grade_counts.get("D", 0), "Grade D (<4)",   "#484f58"),
    (k5, len(all_results),         f"Total (≥{min_score})", "#c9d1d9"),
]:
    with col_:
        st.markdown(
            f'<div class="metric-box"><div class="metric-value" style="color:{c_};">{val_}</div>'
            f'<div class="metric-label">{lbl_}</div></div>',
            unsafe_allow_html=True)

if not all_results:
    st.info(f"No pairs scored ≥{min_score}. Lower the minimum score or change direction.")
    st.stop()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════

tab_cards, tab_table, tab_chart = st.tabs(["🃏 Ranked Cards", "📋 Score Table", "📊 Score Chart"])

# ── CARDS ─────────────────────────────────────────────────────────
with tab_cards:
    GRADE_INFO = {
        "A": ("#3fb950", "#0d2f1a", "#238636", "struct-bull",   "🟢 PRIME SETUP"),
        "B": ("#388bfd", "#0d1e3a", "#1a3a6e", "rank-card-b",   "🔵 GOOD SETUP"),
        "C": ("#e3b341", "#2a1f00", "#9e6a03", "rank-card-c",   "🟡 PARTIAL SETUP"),
        "D": ("#484f58", "#161b22", "#30363d", "rank-card-d",   "⚫ WEAK SETUP"),
    }

    for rank, r in enumerate(all_results, 1):
        color, bg, border_c, _, grade_label = GRADE_INFO[r["grade"]]
        d_icon  = "🔼" if r["direction"] == "LONG" else "🔽"
        d_color = "#388bfd" if r["direction"] == "LONG" else "#a371f7"
        bar_pct = r["pct"]
        price_fmt = f"{r['close']:.5f}" if r["close"] < 100 else f"{r['close']:.3f}"

        # Score dots
        dots = "".join(
            f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;'
            f'background:{color if i < r["score"] else "#21262d"};margin:0 2px;"></span>'
            for i in range(10)
        )

        # Criterion pills
        pills = "".join(
            f'<span style="display:inline-block;background:{"rgba(63,185,80,.12)" if v else "rgba(248,81,73,.08)"};'
            f'border:1px solid {"#238636" if v else "#30363d"};border-radius:4px;'
            f'padding:2px 7px;font-size:10px;color:{"#3fb950" if v else "#484f58"};margin:2px;">'
            f'{k}: {r["details"][k]}</span>'
            for k, v in r["scores"].items()
        )

        spread_warn = (
            f'<span style="background:rgba(248,81,73,.15);border:1px solid rgba(248,81,73,.5);'
            f'border-radius:4px;padding:2px 7px;font-size:10px;color:#f85149;font-weight:600;margin-left:8px;">'
            f'⚠️ Spread {r["spread_pct"]:.1f}%</span>'
        ) if r["spread_pct"] > 5 else ""

        card_html = f"""
        <div style="background:{bg};border:1px solid {border_c};border-left:4px solid {color};
                    border-radius:10px;padding:16px 18px;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
            <div>
              <span style="font-size:11px;color:#484f58;font-weight:600;">#{rank}</span>
              <span style="font-size:18px;font-weight:700;color:#e6edf3;margin-left:8px;">{r["pair"]}</span>
              <span style="font-size:14px;font-weight:600;color:{d_color};margin-left:8px;">{d_icon} {r["direction"]}</span>
              {spread_warn}
            </div>
            <div style="text-align:right;">
              <span style="font-size:15px;font-weight:700;color:{color};">{r["score"]}/10</span>
              <span style="font-size:11px;color:#8b949e;margin-left:8px;">{grade_label}</span>
            </div>
          </div>
          <div style="margin:10px 0 6px 0;">{dots}</div>
          <div style="font-size:11px;color:#8b949e;margin-bottom:8px;">
            Price: <span style="color:#e6edf3;font-family:monospace;">{price_fmt}</span>
            &nbsp;·&nbsp; SL: <span style="color:#f85149;">{r["sl_pips"]} pips</span>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:2px;">{pills}</div>
        </div>
        """

        card_html = "\n".join(line.strip() for line in card_html.splitlines())
        st.markdown(card_html, unsafe_allow_html=True)

# ── TABLE ─────────────────────────────────────────────────────────
with tab_table:
    criteria_keys = list(all_results[0]["scores"].keys()) if all_results else []
    table_rows = []
    for r in all_results:
        row = {
            "Rank":      all_results.index(r) + 1,
            "Pair":      r["pair"],
            "Direction": r["direction"],
            "Score":     f"{r['score']}/10",
            "Grade":     r["grade"],
            "Price":     f"{r['close']:.5f}" if r["close"] < 100 else f"{r['close']:.3f}",
            "SL Pips":   r["sl_pips"],
        }
        for k in criteria_keys:
            row[k] = "✅" if r["scores"][k] else "❌"
        table_rows.append(row)

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ── CHART ─────────────────────────────────────────────────────────
with tab_chart:
    top = all_results[:20]
    labels = [f"{r['pair']} {r['direction']}" for r in top]
    values = [r["score"] for r in top]
    bar_colors = ["#3fb950" if r["grade"] == "A"
                  else "#388bfd" if r["grade"] == "B"
                  else "#e3b341" if r["grade"] == "C"
                  else "#484f58"
                  for r in top]

    fig = go.Figure(go.Bar(
        y=labels[::-1], x=values[::-1],
        orientation="h",
        marker_color=bar_colors[::-1],
        text=[f"{v}/10" for v in values[::-1]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x}/10<extra></extra>",
    ))
    fig.add_vline(x=8, line_dash="dash", line_color="#3fb950", line_width=1.5,
                  annotation_text="Grade A", annotation_font_color="#3fb950")
    fig.add_vline(x=6, line_dash="dot", line_color="#388bfd", line_width=1,
                  annotation_text="Grade B", annotation_font_color="#388bfd")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
        height=max(400, len(top) * 30),
        xaxis=dict(range=[0, 12], showgrid=True, gridcolor="#21262d", title="Score / 10"),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=80, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

# Footer
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid #21262d;">
  🎰 Setup Ranker · 10-point multi-timeframe scoring · Not financial advice
</div>
""", unsafe_allow_html=True)
