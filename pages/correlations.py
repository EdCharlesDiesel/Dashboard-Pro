import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Correlations · Trading System",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
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
    .card{background:var(--secondary-background-color);border:1px solid var(--border,#21262d);border-radius:12px;padding:20px;margin-bottom:16px;}
    .card-header{font-size:13px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#8b949e;margin-bottom:14px;}
    .hero{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);border:1px solid #21262d;border-radius:16px;padding:24px 32px;margin-bottom:24px;}
    .metric-box{background:var(--background-color);border:1px solid var(--border,#21262d);border-radius:8px;padding:12px;text-align:center;}
    .metric-value{font-size:20px;font-weight:700;color:#c9d1d9;}
    .metric-label{font-size:10px;color:#8b949e;margin-top:2px;font-weight:500;letter-spacing:.04em;text-transform:uppercase;}
    .corr-legend{display:flex;align-items:center;gap:20px;font-size:12px;color:#8b949e;padding:8px 0;}
    .legend-dot{width:12px;height:12px;border-radius:50%;display:inline-block;margin-right:5px;}
    .insight-row{display:flex;justify-content:space-between;align-items:center;padding:8px 4px;border-bottom:1px solid #21262d;font-size:13px;}
    .insight-row:last-child{border-bottom:none;}
    .badge{border-radius:6px;padding:2px 9px;font-size:11px;font-weight:700;display:inline-block;}
    .badge-pos-strong{background:#0d4a2f;color:#3fb950;border:1px solid #238636;}
    .badge-pos-mod   {background:#1a3a1a;color:#7ee787;border:1px solid #2ea043;}
    .badge-neg-strong{background:#4a0d0d;color:#f85149;border:1px solid #8b2d2d;}
    .badge-neg-mod   {background:#3a1a1a;color:#ffa198;border:1px solid #8b2d2d;}
    .badge-neutral   {background:#21262d;color:#8b949e;border:1px solid #30363d;}
    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    .block-container{padding-top:1.5rem;max-width:1400px;}
</style>
""", unsafe_allow_html=True)

# ── Instruments ────────────────────────────────────────────────────────────────
ALL_INSTRUMENTS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "USD/CAD": "USDCAD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/AUD": "EURAUD=X",
    "GBP/AUD": "GBPAUD=X",
    "EUR/CAD": "EURCAD=X",
    "GBP/CAD": "GBPCAD=X",
    "USD/ZAR": "USDZAR=X",
    "EUR/ZAR": "EURZAR=X",
    "GBP/ZAR": "GBPZAR=X",
    "🥇 Gold":    "GC=F",
    "🥈 Silver":  "SI=F",
    "🪙 Platinum":"PL=F",
    "💵 DXY":     "DX-Y.NYB",
    "📈 S&P 500": "^GSPC",
}

KNOWN_RELATIONSHIPS = [
    ("EUR/USD", "GBP/USD",  "Positive",  "Both inversely correlated to DXY. Tend to move together."),
    ("EUR/USD", "USD/CHF",  "Negative",  "Strong inverse — CHF pairs often mirror EUR/USD closely."),
    ("EUR/USD", "USD/JPY",  "Negative",  "Risk-on dollar pairs diverge from EUR when safe-haven flows activate."),
    ("EUR/USD", "🥇 Gold",   "Positive",  "Both price in USD — dollar weakness lifts EUR and Gold simultaneously."),
    ("GBP/USD", "GBP/JPY",  "Positive",  "GBP leg drives both; JPY adds volatility amplification."),
    ("AUD/USD", "🥇 Gold",   "Positive",  "Australia is a major gold exporter — AUD tracks gold prices closely."),
    ("AUD/USD", "NZD/USD",  "Positive",  "Commodity-currency bloc — high correlation, similar drivers."),
    ("USD/CAD", "🥇 Gold",   "Negative",  "CAD benefits from commodity strength; Gold USD-priced = inverse."),
    ("USD/JPY", "EUR/JPY",  "Positive",  "JPY leg dominates both pairs during risk-off events."),
    ("USD/ZAR", "🥇 Gold",   "Negative",  "ZAR strengthens with Gold; USD/ZAR falls when Gold rises."),
    ("EUR/ZAR", "USD/ZAR",  "Positive",  "Both ZAR pairs — correlated through shared EM/risk-off dynamics."),
    ("GBP/JPY", "AUD/JPY",  "Positive",  "JPY crosses move together in risk-on/off regimes."),
    ("💵 DXY",  "EUR/USD",  "Negative",  "EUR makes up 57% of the DXY basket — almost a perfect inverse."),
    ("💵 DXY",  "🥇 Gold",  "Negative",  "Gold is USD-priced; dollar strength directly suppresses Gold."),
    ("💵 DXY",  "AUD/USD",  "Negative",  "Commodity currencies weaken when the dollar strengthens."),
    ("📈 S&P 500","AUD/USD","Positive",  "Risk-on flows lift equities and commodity currencies simultaneously."),
    ("📈 S&P 500","USD/JPY","Positive",  "JPY weakens in risk-on environments as safe-haven demand fades."),
    ("📈 S&P 500","🥇 Gold","Negative",  "Gold competes with equities for safe-haven capital — often inverse."),
]

def corr_badge(val):
    if pd.isna(val):
        return "badge-neutral", "Neutral"
    if val >=  0.70: return "badge-pos-strong", "Strong +"
    if val >=  0.40: return "badge-pos-mod",    "Moderate +"
    if val <= -0.70: return "badge-neg-strong", "Strong −"
    if val <= -0.40: return "badge-neg-mod",    "Moderate −"
    return "badge-neutral", "Neutral"

# ── Data fetch ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_prices(tickers: tuple, period: str, interval: str):
    try:
        raw = yf.download(list(tickers), period=period, interval=interval,
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            # Multiple tickers: raw["Close"] gives DataFrame with ticker names as columns
            close = raw["Close"]
        else:
            # Single ticker: flat columns — rename "Close" to the ticker symbol
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        returns = close.pct_change().dropna()
        return returns
    except Exception:
        return None

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔗 Correlation Settings")
    st.page_link("daily-trading-checklist.py", label="00. Checklist", icon="📋")
    st.page_link("pages/macro-bias.py", label="02. Macro Bias", icon="🌐")
    st.page_link("pages/news-filter.py", label="03. News Filter", icon="📰")
    st.page_link("pages/correlations.py", label="04. Correlations", icon="🔗")
    st.page_link("pages/atr-volatility.py", label="05. ATR Volatility", icon="📊")
    st.page_link("pages/weekly-ema.py", label="06. Weekly EMA", icon="📉")
    st.page_link("pages/weekly-rsi.py", label="07. Weekly RSI", icon="📡")
    st.page_link("pages/weekly-swing.py", label="08. Weekly Swing", icon="🔄")
    st.page_link("pages/daily-trend.py", label="09. Daily Trend", icon="📈")
    st.page_link("pages/daily-macd.py", label="10. Daily MACD", icon="📊")
    st.page_link("pages/4H-confluence-zone.py", label="11. 4H Confluence Zone", icon="🎯")
    st.page_link("pages/confluence-checker.py", label="12. 2/3 Confluence Check", icon="🔀")
    st.page_link("pages/15m-rejection.py", label="13. 15M Rejection", icon="🕯️")
    st.page_link("pages/15m-entry-signal.py", label="14. 15M Entry Signal", icon="⚡")
    st.page_link("pages/stop-structure.py", label="15. Stop Structure", icon="🛡️")
    st.page_link("pages/rr-calculator.py", label="16. R:R Calculator", icon="⚖️")
    st.page_link("pages/trade-journal.py",    label="17. Trade Journal",     icon="📓")
    st.page_link("pages/market-structure.py",  label="18. Market Structure",  icon="🏗️")
    st.page_link("pages/setup-ranker.py",      label="01. Setup Ranker",      icon="🏆")
    st.page_link("pages/backtest-workflow.py",    label="19. Backtest Workflow",    icon="🧪")
    st.divider()
    st.markdown("")

    period_map = {
        "1 Month":  ("1mo",  "1d"),
        "3 Months": ("3mo",  "1d"),
        "6 Months": ("6mo",  "1d"),
        "1 Year":   ("1y",   "1d"),
    }
    selected_period = st.selectbox("Lookback Period", list(period_map.keys()), index=1)
    yf_period, yf_interval = period_map[selected_period]

    selected_pairs = st.multiselect(
        "Instruments to include",
        list(ALL_INSTRUMENTS.keys()),
        default=list(ALL_INSTRUMENTS.keys()),
    )

    corr_method = st.radio("Correlation Method", ["Pearson", "Spearman"], horizontal=True)

    rolling_window = st.slider("Rolling Window (days)", 10, 60, 20)

    focus_pair = st.selectbox("Focus Pair (Rolling Chart)", selected_pairs if selected_pairs else ["EUR/USD"])

    st.divider()
    fetch_btn = st.button("🔄 Refresh Data", use_container_width=True, type="primary")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:26px;font-weight:700;color:#e6edf3;">🔗 Instrument Correlation Matrix</div>
      <div style="color:#8b949e;font-size:14px;margin-top:6px;">Return-based correlations · {selected_period} lookback · {yf_interval} bars</div>
      <div style="font-size:13px;color:#388bfd;font-weight:500;margin-top:4px;">
        🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M')}
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:13px;color:#8b949e;">Pairs selected</div>
      <div style="font-size:36px;font-weight:700;color:#e6edf3;">{len(selected_pairs)}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Fetch data ─────────────────────────────────────────────────────────────────
if not selected_pairs:
    st.warning("Select at least 2 instruments in the sidebar.")
    st.stop()

tickers_tuple = tuple(ALL_INSTRUMENTS[p] for p in selected_pairs)

if fetch_btn:
    st.cache_data.clear()

with st.spinner("📡 Fetching market data…"):
    returns = fetch_prices(tickers_tuple, yf_period, yf_interval)

if returns is None or returns.empty:
    st.error("❌ Could not fetch price data. Check your internet connection.")
    st.stop()

# Map ticker columns back to pair names
ticker_to_pair = {v: k for k, v in ALL_INSTRUMENTS.items()}
returns.columns = [ticker_to_pair.get(c, c) for c in returns.columns]

# Keep only selected pairs that actually loaded
available = [p for p in selected_pairs if p in returns.columns]
returns   = returns[available]

if len(available) < 2:
    st.error("Not enough data loaded. Try a shorter period or fewer pairs.")
    st.stop()

# Correlation matrix
method = corr_method.lower()
corr_matrix = returns.corr(method=method)

# ── KPI strip ──────────────────────────────────────────────────────────────────
flat    = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
# Remove NaN values from flat
flat = flat.dropna()
avg_c   = flat.mean() if not flat.empty else 0
max_c   = flat.max() if not flat.empty else 0
min_c   = flat.min() if not flat.empty else 0
strong_pos = (flat >= 0.7).sum()
strong_neg = (flat <= -0.7).sum()

k1,k2,k3,k4,k5,k6 = st.columns(6)
kpis = [
    (k1, len(available),          "Pairs Loaded",      "#c9d1d9"),
    (k2, f"{avg_c:.2f}",          "Avg Correlation",   "#e3b341"),
    (k3, f"{max_c:.2f}",          "Strongest +",       "#3fb950"),
    (k4, f"{min_c:.2f}",          "Strongest −",       "#f85149"),
    (k5, int(strong_pos),         "Strong + Pairs",    "#3fb950"),
    (k6, int(strong_neg),         "Strong − Pairs",    "#f85149"),
]
for col, val, lbl, color in kpis:
    with col:
        st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{color};">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HEATMAP
# ══════════════════════════════════════════════════════════════════
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="card"><div class="card-header">🗺️ Correlation Heatmap</div>', unsafe_allow_html=True)

    labels = corr_matrix.columns.tolist()
    z      = corr_matrix.values
    text   = [[f"{v:.2f}" for v in row] for row in z]

    fig_heat = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, text=text,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        colorscale=[
            [0.0,  "#8b2d2d"],
            [0.25, "#c0392b"],
            [0.45, "#1a1a2e"],
            [0.50, "#161b22"],
            [0.55, "#1a2e1a"],
            [0.75, "#238636"],
            [1.0,  "#0d4a2f"],
        ],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(
            title=dict(
                text="r",
                font=dict(color="#8b949e")
            ),
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1", "-0.5", "0", "0.5", "1"],
            tickfont=dict(color="#8b949e", size=11),
            bgcolor="#161b22",
            bordercolor="#21262d",
        ),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: <b>%{z:.3f}</b><extra></extra>",
    ))
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161b22",
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
        xaxis=dict(tickfont=dict(color="#8b949e", size=10), side="bottom",
                   showgrid=False, tickangle=-45),
        yaxis=dict(tickfont=dict(color="#8b949e", size=10), showgrid=False, autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config=dict(displayModeBar=False))

    st.markdown("""
    <div class="corr-legend">
      <span><span class="legend-dot" style="background:#0d4a2f;border:1px solid #238636;"></span>Strong positive ≥ 0.70</span>
      <span><span class="legend-dot" style="background:#238636;"></span>Moderate + (0.40–0.70)</span>
      <span><span class="legend-dot" style="background:#21262d;"></span>Neutral</span>
      <span><span class="legend-dot" style="background:#c0392b;"></span>Moderate − (−0.40–−0.70)</span>
      <span><span class="legend-dot" style="background:#8b2d2d;border:1px solid #f85149;"></span>Strong negative ≤ −0.70</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# RIGHT PANELS
# ══════════════════════════════════════════════════════════════════
with right:

    # Top correlated pairs
    pair_corrs = flat.reset_index()
    pair_corrs.columns = ["Pair A", "Pair B", "Correlation"]
    pair_corrs["abs"] = pair_corrs["Correlation"].abs()
    # Remove NaN correlations before processing
    pair_corrs = pair_corrs.dropna(subset=["Correlation"])
    pair_corrs = pair_corrs.sort_values("Correlation", ascending=False)

    st.markdown('<div class="card"><div class="card-header">🔝 Strongest Correlations</div>', unsafe_allow_html=True)
    top10 = pair_corrs.head(8)
    for _, row in top10.iterrows():
        bc, bl = corr_badge(row["Correlation"])
        bar_w = int(abs(row["Correlation"]) * 100)
        bar_c = "#3fb950" if row["Correlation"] > 0 else "#f85149"
        st.markdown(f"""
        <div class="insight-row">
          <div style="flex:1;">
            <span style="color:#e6edf3;font-weight:600;">{row['Pair A']}</span>
            <span style="color:#8b949e;font-size:11px;"> vs </span>
            <span style="color:#e6edf3;font-weight:600;">{row['Pair B']}</span>
            <div style="background:#21262d;border-radius:4px;height:4px;margin-top:4px;width:100%;">
              <div style="background:{bar_c};width:{bar_w}%;height:4px;border-radius:4px;"></div>
            </div>
          </div>
          <div style="margin-left:12px;text-align:right;">
            <span class="badge {bc}">{row['Correlation']:+.2f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Bottom (most negative)
    st.markdown('<div class="card"><div class="card-header">🔻 Most Negative Correlations</div>', unsafe_allow_html=True)
    bot8 = pair_corrs.tail(8).iloc[::-1]
    for _, row in bot8.iterrows():
        bc, bl = corr_badge(row["Correlation"])
        bar_w = int(abs(row["Correlation"]) * 100)
        st.markdown(f"""
        <div class="insight-row">
          <div style="flex:1;">
            <span style="color:#e6edf3;font-weight:600;">{row['Pair A']}</span>
            <span style="color:#8b949e;font-size:11px;"> vs </span>
            <span style="color:#e6edf3;font-weight:600;">{row['Pair B']}</span>
            <div style="background:#21262d;border-radius:4px;height:4px;margin-top:4px;width:100%;">
              <div style="background:#f85149;width:{bar_w}%;height:4px;border-radius:4px;"></div>
            </div>
          </div>
          <div style="margin-left:12px;text-align:right;">
            <span class="badge {bc}">{row['Correlation']:+.2f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ROLLING CORRELATION CHART
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"### 📈 Rolling {rolling_window}-Day Correlation — {focus_pair} vs All")

if focus_pair in returns.columns:
    others = [p for p in available if p != focus_pair]
    rolling_corrs = {}
    for pair in others:
        rc = returns[focus_pair].rolling(rolling_window).corr(returns[pair]).dropna()
        if not rc.empty:
            rolling_corrs[pair] = rc

    if rolling_corrs:
        fig_roll = go.Figure()
        palette = [
            "#388bfd","#3fb950","#e3b341","#f85149","#a371f7",
            "#58a6ff","#7ee787","#ffa657","#ff7b72","#d2a8ff",
            "#56d364","#79c0ff","#ff9550","#f778ba","#b392f0",
            "#6e7681","#21262d","#30363d",
        ]
        for idx, (pair, rc) in enumerate(rolling_corrs.items()):
            color = palette[idx % len(palette)]
            fig_roll.add_trace(go.Scatter(
                x=rc.index, y=rc.values, name=pair,
                mode="lines", line=dict(color=color, width=1.5),
                hovertemplate=f"<b>{focus_pair} vs {pair}</b><br>Date: %{{x|%d %b %Y}}<br>Corr: %{{y:.3f}}<extra></extra>",
            ))

        # Zero line & bands
        fig_roll.add_hline(y=0,    line_dash="dot", line_color="#8b949e", line_width=1)
        fig_roll.add_hline(y=0.7,  line_dash="dash", line_color="#238636", line_width=1, annotation_text="Strong +0.7", annotation_font_color="#238636", annotation_position="right")
        fig_roll.add_hline(y=-0.7, line_dash="dash", line_color="#8b2d2d", line_width=1, annotation_text="Strong −0.7", annotation_font_color="#8b2d2d", annotation_position="right")
        fig_roll.add_hrect(y0=0.7,  y1=1,   fillcolor="#238636", opacity=0.05, line_width=0)
        fig_roll.add_hrect(y0=-1,   y1=-0.7, fillcolor="#8b2d2d", opacity=0.05, line_width=0)

        fig_roll.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            margin=dict(l=20, r=80, t=20, b=20), height=400,
            legend=dict(font=dict(color="#8b949e", size=10), bgcolor="#161b22",
                        bordercolor="#21262d", borderwidth=1, orientation="v",
                        x=1.02, y=1),
            xaxis=dict(showgrid=False, tickfont=dict(color="#8b949e"),
                       tickformat="%d %b", linecolor="#21262d"),
            yaxis=dict(showgrid=True, gridcolor="#21262d",
                       tickfont=dict(color="#8b949e"), range=[-1.05, 1.05],
                       tickvals=[-1,-0.7,-0.4,0,0.4,0.7,1],
                       ticktext=["-1.0","-0.7","-0.4","0","+0.4","+0.7","+1.0"],
                       zerolinecolor="#21262d"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_roll, use_container_width=True, config=dict(displayModeBar=False))
    else:
        st.info("Not enough data for rolling correlation.")
else:
    st.info(f"{focus_pair} not in loaded data.")

# ══════════════════════════════════════════════════════════════════
# KNOWN RELATIONSHIP INSIGHTS
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 💡 Key Relationship Insights")

cols = st.columns(2)
for idx, (pa, pb, rel_type, desc) in enumerate(KNOWN_RELATIONSHIPS):
    live_corr = None
    if pa in corr_matrix.columns and pb in corr_matrix.columns:
        live_corr = corr_matrix.loc[pa, pb]

    with cols[idx % 2]:
        bc, _ = corr_badge(live_corr) if live_corr is not None else ("badge-neutral", "")
        live_str = f'<span class="badge {bc}">{live_corr:+.2f}</span>' if live_corr is not None and not pd.isna(live_corr) else '<span class="badge badge-neutral">N/A</span>'
        rel_color = "#3fb950" if rel_type == "Positive" else "#f85149"
        st.markdown(f"""
        <div class="card" style="padding:14px 16px;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
            <div>
              <span style="font-weight:700;color:#e6edf3;">{pa}</span>
              <span style="color:#8b949e;font-size:11px;margin:0 6px;">vs</span>
              <span style="font-weight:700;color:#e6edf3;">{pb}</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="font-size:11px;color:{rel_color};font-weight:600;">{rel_type}</span>
              {live_str}
            </div>
          </div>
          <div style="font-size:12px;color:#8b949e;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Full sortable table ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Full Correlation Table")

display_df = pair_corrs.drop(columns=["abs"]).copy()
display_df["Correlation"] = display_df["Correlation"].round(4)
display_df.index = range(1, len(display_df)+1)

st.dataframe(display_df, use_container_width=True, column_config={
    "Pair A":      st.column_config.TextColumn("Pair A"),
    "Pair B":      st.column_config.TextColumn("Pair B"),
    "Correlation": st.column_config.NumberColumn("Correlation (r)", format="%.4f"),
})

# Footer
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid #21262d;">
  🔗 Correlations · Return-based (daily close-to-close) · Refreshes every 10 minutes · Not financial advice
</div>
""", unsafe_allow_html=True)