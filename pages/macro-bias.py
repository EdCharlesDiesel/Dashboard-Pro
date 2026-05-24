import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Macro Bias · Trading System",
    page_icon="🌐",
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
    .card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:18px 20px;margin-bottom:14px;}
    .card-header{font-size:12px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:#8b949e;margin-bottom:12px;}
    .hero{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d1117 100%);border:1px solid #21262d;border-radius:16px;padding:24px 32px;margin-bottom:20px;position:relative;overflow:hidden;}
    .hero::before{content:'';position:absolute;top:-40%;right:-5%;width:280px;height:280px;background:radial-gradient(circle,rgba(56,139,253,.07) 0%,transparent 70%);border-radius:50%;}

    /* Currency cards */
    .ccy-card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:16px;margin-bottom:12px;transition:border-color .2s;}
    .ccy-card:hover{border-color:#388bfd55;}
    .ccy-flag{font-size:22px;margin-right:8px;}
    .ccy-name{font-size:16px;font-weight:700;color:#e6edf3;}
    .ccy-bank{font-size:11px;color:#8b949e;margin-top:1px;}

    /* Metric pills */
    .pill{display:inline-flex;align-items:center;gap:5px;border-radius:6px;padding:3px 10px;font-size:12px;font-weight:600;margin:2px;}
    .pill-bull{background:#0d2f1a;color:#3fb950;border:1px solid #238636;}
    .pill-bear{background:#2f0d0d;color:#f85149;border:1px solid #8b2d2d;}
    .pill-neut{background:#1c1c24;color:#8b949e;border:1px solid #30363d;}
    .pill-warn{background:#2f1f0d;color:#e3b341;border:1px solid #9e6a03;}

    /* Bias score bar */
    .bias-track{background:#21262d;border-radius:8px;height:8px;margin:6px 0 2px;overflow:hidden;}
    .bias-bull{background:linear-gradient(90deg,#238636,#3fb950);height:100%;border-radius:8px;}
    .bias-bear{background:linear-gradient(90deg,#8b2d2d,#f85149);height:100%;border-radius:8px;}
    .bias-neut{background:linear-gradient(90deg,#30363d,#484f58);height:100%;border-radius:8px;}

    /* Data rows */
    .data-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #21262d;font-size:13px;}
    .data-row:last-child{border-bottom:none;}
    .data-label{color:#8b949e;font-size:12px;}
    .data-val{font-weight:600;color:#e6edf3;font-family:monospace;}
    .data-val-up{font-weight:700;color:#3fb950;font-family:monospace;}
    .data-val-dn{font-weight:700;color:#f85149;font-family:monospace;}
    .data-val-warn{font-weight:700;color:#e3b341;font-family:monospace;}

    /* Pair bias table */
    .pair-row{display:flex;justify-content:space-between;align-items:center;padding:10px 4px;border-bottom:1px solid #21262d;}
    .pair-row:last-child{border-bottom:none;}
    .pair-name{font-size:15px;font-weight:700;color:#e6edf3;}
    .pair-sub{font-size:11px;color:#8b949e;margin-top:2px;}

    .badge{border-radius:20px;padding:4px 14px;font-size:12px;font-weight:700;display:inline-block;}
    .badge-bull{background:#0d4a2f;color:#3fb950;border:1px solid #238636;}
    .badge-bear{background:#4a0d0d;color:#f85149;border:1px solid #8b2d2d;}
    .badge-neut{background:#21262d;color:#8b949e;border:1px solid #30363d;}
    .badge-watch{background:#4a2d0d;color:#e3b341;border:1px solid #9e6a03;}

    .metric-box{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:12px 14px;text-align:center;}
    .metric-value{font-size:20px;font-weight:700;color:#c9d1d9;}
    .metric-label{font-size:10px;color:#8b949e;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}

    #MainMenu,footer,header{visibility:hidden;}
    .block-container{padding-top:1.5rem;max-width:1420px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CURRENCY CONFIG
# ══════════════════════════════════════════════════════════════════
CURRENCIES = {
    "USD": {
        "flag": "🇺🇸", "name": "US Dollar",       "bank": "Federal Reserve",
        "wb_code": "US",  "rate_ticker": "^IRX",
        "yield_ticker": "^TNX",
        "inflation_target": 2.0,
        "cb_rate": 5.25,   # manually maintained — update as needed
        "rate_trend": "Holding",  # Hiking / Cutting / Holding
    },
    "EUR": {
        "flag": "🇪🇺", "name": "Euro",             "bank": "ECB",
        "wb_code": "XC",  "rate_ticker": None,
        "yield_ticker": None,
        "inflation_target": 2.0,
        "cb_rate": 3.65,
        "rate_trend": "Cutting",
    },
    "GBP": {
        "flag": "🇬🇧", "name": "British Pound",    "bank": "Bank of England",
        "wb_code": "GB",  "rate_ticker": None,
        "yield_ticker": None,
        "inflation_target": 2.0,
        "cb_rate": 5.00,
        "rate_trend": "Cutting",
    },
    "AUD": {
        "flag": "🇦🇺", "name": "Australian Dollar","bank": "RBA",
        "wb_code": "AU",  "rate_ticker": None,
        "yield_ticker": None,
        "inflation_target": 2.5,
        "cb_rate": 4.35,
        "rate_trend": "Holding",
    },
    "NZD": {
        "flag": "🇳🇿", "name": "New Zealand Dollar","bank": "RBNZ",
        "wb_code": "NZ",  "rate_ticker": None,
        "yield_ticker": None,
        "inflation_target": 2.0,
        "cb_rate": 4.75,
        "rate_trend": "Cutting",
    },
    "JPY": {
        "flag": "🇯🇵", "name": "Japanese Yen",     "bank": "Bank of Japan",
        "wb_code": "JP",  "rate_ticker": None,
        "yield_ticker": None,
        "inflation_target": 2.0,
        "cb_rate": 0.25,
        "rate_trend": "Hiking",
    },
    "CHF": {
        "flag": "🇨🇭", "name": "Swiss Franc",      "bank": "SNB",
        "wb_code": "CH",  "rate_ticker": None,
        "yield_ticker": None,
        "inflation_target": 2.0,
        "cb_rate": 1.00,
        "rate_trend": "Cutting",
    },
    "CAD": {
        "flag": "🇨🇦", "name": "Canadian Dollar",  "bank": "Bank of Canada",
        "wb_code": "CA",  "rate_ticker": None,
        "yield_ticker": None,
        "inflation_target": 2.0,
        "cb_rate": 4.25,
        "rate_trend": "Cutting",
    },
    "ZAR": {
        "flag": "🇿🇦", "name": "South African Rand","bank": "SARB",
        "wb_code": "ZA",  "rate_ticker": None,
        "yield_ticker": None,
        "inflation_target": 4.5,
        "cb_rate": 8.25,
        "rate_trend": "Cutting",
    },
}

# ── Trading pairs with base/quote ──────────────────────────────────────────────
PAIRS = [
    ("EUR/USD","EUR","USD"), ("GBP/USD","GBP","USD"), ("AUD/USD","AUD","USD"),
    ("NZD/USD","NZD","USD"), ("USD/JPY","USD","JPY"), ("USD/CHF","USD","CHF"),
    ("USD/CAD","USD","CAD"), ("EUR/GBP","EUR","GBP"), ("EUR/JPY","EUR","JPY"),
    ("GBP/JPY","GBP","JPY"), ("AUD/JPY","AUD","JPY"), ("EUR/AUD","EUR","AUD"),
    ("GBP/AUD","GBP","AUD"), ("EUR/CAD","EUR","CAD"), ("GBP/CAD","GBP","CAD"),
    ("USD/ZAR","USD","ZAR"), ("EUR/ZAR","EUR","ZAR"), ("GBP/ZAR","GBP","ZAR"),
    ("🥇 Gold","USD","—"),
]

# ══════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_wb_indicator(country_code: str, indicator: str, years: int = 5):
    """Fetch World Bank indicator for a country. Returns list of (year, value) sorted asc."""
    try:
        url = (f"https://api.worldbank.org/v2/country/{country_code}"
               f"/indicator/{indicator}?format=json&mrv={years}&per_page=10")
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or len(data) < 2 or not data[1]:
            return None
        results = [(d["date"], d["value"]) for d in data[1] if d["value"] is not None]
        return sorted(results, key=lambda x: x[0])  # oldest first
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_macro():
    """Fetch GDP and CPI for all currencies from World Bank."""
    macro = {}
    for ccy, cfg in CURRENCIES.items():
        code = cfg["wb_code"]
        gdp  = fetch_wb_indicator(code, "NY.GDP.MKTP.KD.ZG", 5)  # GDP growth %
        cpi  = fetch_wb_indicator(code, "FP.CPI.TOTL.ZG",    5)  # Inflation %
        macro[ccy] = {"gdp": gdp, "cpi": cpi}
    return macro

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yields():
    """Fetch 10Y government bond yields via yfinance."""
    yield_tickers = {
        "USD": "^TNX",   # US 10Y
        "GBP": "^TNX",   # placeholder — use UK gilt ETF proxy
        "JPY": "^TNX",   # placeholder
    }
    results = {}
    for ccy, tkr in yield_tickers.items():
        try:
            df = yf.download(tkr, period="5d", interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                results[ccy] = round(float(df["Close"].iloc[-1]), 3)
        except Exception:
            pass
    return results

# ══════════════════════════════════════════════════════════════════
# BIAS SCORING
# ══════════════════════════════════════════════════════════════════

def score_currency(ccy: str, macro: dict) -> dict:
    """Score a currency's macro bias. Returns score -3 to +3 and signals."""
    cfg    = CURRENCIES[ccy]
    score  = 0
    signals = []

    # 1. Rate level vs peers (compare to average)
    all_rates  = [c["cb_rate"] for c in CURRENCIES.values()]
    avg_rate   = np.mean(all_rates)
    rate_score = 1 if cfg["cb_rate"] > avg_rate else -1 if cfg["cb_rate"] < avg_rate * 0.6 else 0
    score += rate_score
    signals.append(("Rate Level", f"{cfg['cb_rate']}%",
                    "bull" if rate_score > 0 else "bear" if rate_score < 0 else "neut"))

    # 2. Rate trend
    trend_score = {"Hiking": 1, "Holding": 0, "Cutting": -1}.get(cfg["rate_trend"], 0)
    score += trend_score
    signals.append(("Rate Trend", cfg["rate_trend"],
                    "bull" if trend_score > 0 else "bear" if trend_score < 0 else "neut"))

    # 3. GDP growth
    gdp_data = macro.get(ccy, {}).get("gdp")
    if gdp_data and len(gdp_data) >= 2:
        latest_gdp = gdp_data[-1][1]
        prev_gdp   = gdp_data[-2][1]
        gdp_score  = 1 if latest_gdp > 2.0 else -1 if latest_gdp < 0.5 else 0
        trend_ok   = latest_gdp >= prev_gdp
        score += gdp_score + (0.5 if trend_ok else -0.5)
        signals.append(("GDP Growth", f"{latest_gdp:.1f}%",
                        "bull" if gdp_score > 0 else "bear" if gdp_score < 0 else "neut"))
    else:
        signals.append(("GDP Growth", "N/A", "neut"))

    # 4. Inflation vs target
    cpi_data = macro.get(ccy, {}).get("cpi")
    if cpi_data and len(cpi_data) >= 1:
        latest_cpi  = cpi_data[-1][1]
        target      = cfg["inflation_target"]
        # Hawkish pressure (high inflation) = rate hike expectation = bullish for now
        # But hyper-inflation = bearish
        if latest_cpi > target * 2.5:
            cpi_score = -1
        elif latest_cpi > target * 1.2:
            cpi_score = 1
        elif latest_cpi < target * 0.5:
            cpi_score = -1
        else:
            cpi_score = 0
        score += cpi_score
        signals.append(("Inflation", f"{latest_cpi:.1f}% (tgt {target}%)",
                        "bull" if cpi_score > 0 else "bear" if cpi_score < 0 else "warn" if abs(latest_cpi - target) > 1 else "neut"))
    else:
        signals.append(("Inflation", "N/A", "neut"))

    # Clamp score
    score = max(-4, min(4, score))
    bias  = "Bullish" if score >= 1.5 else "Bearish" if score <= -1.5 else "Neutral"
    return {"score": score, "bias": bias, "signals": signals}

def pair_macro_bias(base: str, quote: str, scores: dict) -> dict:
    """Derive pair bias from base vs quote currency scores."""
    if base == "—" or quote == "—":
        return {"direction": "Watch", "strength": 0, "reason": "Commodity — check DXY"}
    bs = scores.get(base, {}).get("score", 0)
    qs = scores.get(quote, {}).get("score", 0)
    diff = bs - qs
    if   diff >= 2:  direction, strength = "LONG",  min(int(abs(diff)), 4)
    elif diff >= 1:  direction, strength = "LONG",  1
    elif diff <= -2: direction, strength = "SHORT", min(int(abs(diff)), 4)
    elif diff <= -1: direction, strength = "SHORT", 1
    else:            direction, strength = "NEUTRAL", 0
    bb = scores.get(base,  {}).get("bias", "?")
    qb = scores.get(quote, {}).get("bias", "?")
    reason = f"{base} is {bb} · {quote} is {qb}"
    return {"direction": direction, "strength": strength, "reason": reason}

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌐 Macro Bias")
    st.page_link("daliy-trading-checklist.py", label="Checklist", icon="📋")
    st.page_link("pages/correlations.py", label="Correlations", icon="🔗")
    st.page_link("pages/macro-bias.py", label="Macro Bias", icon="🌐")
    st.page_link("pages/news-filter.py", label="News Filter", icon="📰")
    st.page_link("pages/atr-volatility.py", label="ATR Volatility", icon="📊")
    st.page_link("pages/weekly-ema.py", label="Weekly EMA", icon="📉")
    st.page_link("pages/weekly-rsi.py", label="Weekly RSI", icon="📡")
    st.page_link("pages/daily-trend.py", label="📈 Daily Trend", icon="📈")
    st.divider()

    st.markdown("**⚡ Manual Rate Override**")
    st.caption("Update if CB rates have changed since last release.")
    with st.expander("Edit Central Bank Rates", expanded=False):
        for ccy in CURRENCIES:
            CURRENCIES[ccy]["cb_rate"] = st.number_input(
                f"{CURRENCIES[ccy]['flag']} {ccy} rate (%)",
                value=float(CURRENCIES[ccy]["cb_rate"]),
                step=0.25, format="%.2f", key=f"rate_{ccy}"
            )
            CURRENCIES[ccy]["rate_trend"] = st.selectbox(
                f"{ccy} trend", ["Hiking","Holding","Cutting"],
                index=["Hiking","Holding","Cutting"].index(CURRENCIES[ccy]["rate_trend"]),
                key=f"trend_{ccy}"
            )

    st.divider()
    if st.button("🔄 Refresh Macro Data", use_container_width=True, type="primary"):
        st.cache_data.clear()

    show_pairs = st.multiselect("Filter Pairs", [p[0] for p in PAIRS], default=[p[0] for p in PAIRS])

# ══════════════════════════════════════════════════════════════════
# FETCH
# ══════════════════════════════════════════════════════════════════
with st.spinner("📡 Loading macro data from World Bank…"):
    macro_data = fetch_all_macro()

# Score each currency
scores = {ccy: score_currency(ccy, macro_data) for ccy in CURRENCIES}
bull_ccys  = [c for c, s in scores.items() if s["bias"] == "Bullish"]
bear_ccys  = [c for c, s in scores.items() if s["bias"] == "Bearish"]
neut_ccys  = [c for c, s in scores.items() if s["bias"] == "Neutral"]

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:26px;font-weight:700;color:#e6edf3;">🌐 Macro Bias Dashboard</div>
      <div style="color:#8b949e;font-size:14px;margin-top:6px;">Interest Rates · GDP Growth · Inflation · Currency Bias Scoring</div>
      <div style="font-size:13px;color:#388bfd;font-weight:500;margin-top:4px;">
        🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M')} · World Bank API + CB Rates
      </div>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <div class="metric-box" style="min-width:90px;">
        <div class="metric-value" style="color:#3fb950;">{len(bull_ccys)}</div>
        <div class="metric-label">🟢 Bullish</div>
      </div>
      <div class="metric-box" style="min-width:90px;">
        <div class="metric-value" style="color:#8b949e;">{len(neut_ccys)}</div>
        <div class="metric-label">⚪ Neutral</div>
      </div>
      <div class="metric-box" style="min-width:90px;">
        <div class="metric-value" style="color:#f85149;">{len(bear_ccys)}</div>
        <div class="metric-label">🔴 Bearish</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 1: INTEREST RATES COMPARISON
# ══════════════════════════════════════════════════════════════════
st.markdown("## 💹 Central Bank Interest Rates")

rate_names  = [f"{CURRENCIES[c]['flag']} {c}" for c in CURRENCIES]
rate_vals   = [CURRENCIES[c]["cb_rate"] for c in CURRENCIES]
rate_trends = [CURRENCIES[c]["rate_trend"] for c in CURRENCIES]
rate_colors = ["#3fb950" if t == "Hiking" else "#f85149" if t == "Cutting" else "#e3b341" for t in rate_trends]

fig_rates = go.Figure()
fig_rates.add_trace(go.Bar(
    x=rate_names, y=rate_vals,
    marker_color=rate_colors,
    text=[f"{v:.2f}%<br><span style='font-size:9px'>{t}</span>" for v, t in zip(rate_vals, rate_trends)],
    textposition="outside",
    textfont=dict(color="#c9d1d9", size=11),
    hovertemplate="<b>%{x}</b><br>Rate: %{y:.2f}%<extra></extra>",
))
fig_rates.add_hline(
    y=float(np.mean(rate_vals)),
    line_dash="dot", line_color="#8b949e", line_width=1.5,
    annotation_text=f"Avg {np.mean(rate_vals):.2f}%",
    annotation_font_color="#8b949e", annotation_position="right",
)
fig_rates.update_layout(
    paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
    margin=dict(l=10, r=80, t=30, b=10), height=300,
    xaxis=dict(tickfont=dict(color="#c9d1d9", size=12), showgrid=False, linecolor="#21262d"),
    yaxis=dict(tickfont=dict(color="#8b949e"), showgrid=True, gridcolor="#21262d",
               ticksuffix="%", range=[0, max(rate_vals) * 1.3]),
    showlegend=False,
    bargap=0.25,
)
st.plotly_chart(fig_rates, use_container_width=True, config=dict(displayModeBar=False))

# Rate trend legend
st.markdown("""
<div style="display:flex;gap:20px;font-size:12px;color:#8b949e;padding:4px 0 16px 0;">
  <span>🟢 Hiking = hawkish = currency bullish pressure</span>
  <span>🟡 Holding = neutral</span>
  <span>🔴 Cutting = dovish = currency bearish pressure</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 2: CURRENCY CARDS (Rates + GDP + CPI)
# ══════════════════════════════════════════════════════════════════
st.markdown("## 🏦 Currency Macro Scorecard")

ccy_list = list(CURRENCIES.keys())
rows = [ccy_list[i:i+3] for i in range(0, len(ccy_list), 3)]

for row in rows:
    cols = st.columns(3)
    for col, ccy in zip(cols, row):
        cfg    = CURRENCIES[ccy]
        sc     = scores[ccy]
        bias   = sc["bias"]
        s      = sc["score"]
        signals= sc["signals"]

        bias_color = "#3fb950" if bias == "Bullish" else "#f85149" if bias == "Bearish" else "#8b949e"
        bias_icon  = "🟢" if bias == "Bullish" else "🔴" if bias == "Bearish" else "⚪"
        bar_pct    = int((s + 4) / 8 * 100)
        bar_class  = "bias-bull" if s > 0 else "bias-bear" if s < 0 else "bias-neut"

        gdp_data = macro_data.get(ccy, {}).get("gdp")
        cpi_data = macro_data.get(ccy, {}).get("cpi")
        gdp_val  = f"{gdp_data[-1][1]:.1f}%" if gdp_data else "N/A"
        gdp_yr   = gdp_data[-1][0] if gdp_data else ""
        cpi_val  = f"{cpi_data[-1][1]:.1f}%" if cpi_data else "N/A"
        cpi_yr   = cpi_data[-1][0] if cpi_data else ""

        gdp_num  = gdp_data[-1][1] if gdp_data else None
        cpi_num  = cpi_data[-1][1] if cpi_data else None
        gdp_cls  = "data-val-up" if gdp_num and gdp_num > 2 else "data-val-dn" if gdp_num and gdp_num < 0.5 else "data-val-warn"
        cpi_cls  = "data-val-up" if cpi_num and abs(cpi_num - cfg['inflation_target']) < 0.5 else "data-val-warn"

        # Trend arrow
        trend_icon = {"Hiking": "⬆️", "Cutting": "⬇️", "Holding": "➡️"}.get(cfg["rate_trend"], "➡️")

        # Signals as pills
        pill_html = ""
        for lbl, val, stype in signals:
            pill_cls = f"pill-{stype}"
            icon = "▲" if stype == "bull" else "▼" if stype == "bear" else "●"
            pill_html += f'<span class="pill {pill_cls}">{icon} {lbl}: {val}</span>'

        with col:
            st.markdown(f"""
            <div class="ccy-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                <div>
                  <span class="ccy-flag">{cfg['flag']}</span>
                  <span class="ccy-name">{ccy}</span>
                  <div class="ccy-bank">{cfg['bank']}</div>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:18px;font-weight:700;color:{bias_color};">{bias_icon} {bias}</div>
                  <div style="font-size:11px;color:#8b949e;">Score {s:+.1f}</div>
                </div>
              </div>

              <!-- Bias bar -->
              <div class="bias-track">
                <div class="{bar_class}" style="width:{bar_pct}%;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:10px;color:#484f58;margin-bottom:10px;">
                <span>Bearish</span><span>Neutral</span><span>Bullish</span>
              </div>

              <!-- Data rows -->
              <div class="data-row">
                <span class="data-label">🏦 CB Rate</span>
                <span class="data-val">{cfg['cb_rate']:.2f}% &nbsp;{trend_icon}</span>
              </div>
              <div class="data-row">
                <span class="data-label">📈 GDP Growth ({gdp_yr})</span>
                <span class="{gdp_cls}">{gdp_val}</span>
              </div>
              <div class="data-row">
                <span class="data-label">🔥 Inflation ({cpi_yr})</span>
                <span class="{cpi_cls}">{cpi_val} <span style="color:#484f58;font-size:10px;">tgt {cfg['inflation_target']}%</span></span>
              </div>

              <!-- Signal pills -->
              <div style="margin-top:10px;line-height:1.8;">{pill_html}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 3: GDP & INFLATION CHARTS
# ══════════════════════════════════════════════════════════════════
st.markdown("## 📊 GDP Growth vs Inflation by Economy")

chart_col1, chart_col2 = st.columns(2)

# GDP scatter
gdp_plot = []
for ccy, cfg in CURRENCIES.items():
    gd = macro_data.get(ccy, {}).get("gdp")
    cd = macro_data.get(ccy, {}).get("cpi")
    if gd and cd:
        gdp_plot.append({
            "Currency": ccy, "Flag": cfg["flag"],
            "GDP Growth (%)": round(gd[-1][1], 2),
            "Inflation (%)":  round(cd[-1][1], 2),
            "CB Rate (%)":    cfg["cb_rate"],
            "Bias":           scores[ccy]["bias"],
        })

with chart_col1:
    st.markdown('<div class="card"><div class="card-header">📈 Latest GDP Growth Rate (%)</div>', unsafe_allow_html=True)
    if gdp_plot:
        df_gdp = pd.DataFrame(gdp_plot).sort_values("GDP Growth (%)", ascending=True)
        colors_gdp = ["#3fb950" if v > 2 else "#f85149" if v < 0.5 else "#e3b341" for v in df_gdp["GDP Growth (%)"]]
        fig_gdp = go.Figure(go.Bar(
            y=[f"{r['Flag']} {r['Currency']}" for _, r in df_gdp.iterrows()],
            x=df_gdp["GDP Growth (%)"],
            orientation="h",
            marker_color=colors_gdp,
            text=[f"{v:.1f}%" for v in df_gdp["GDP Growth (%)"]],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=11),
            hovertemplate="<b>%{y}</b><br>GDP Growth: %{x:.2f}%<extra></extra>",
        ))
        fig_gdp.add_vline(x=2.0, line_dash="dot", line_color="#238636", line_width=1.5,
                          annotation_text="Solid +2%", annotation_font_color="#238636")
        fig_gdp.add_vline(x=0.0, line_dash="dot", line_color="#8b2d2d", line_width=1,
                          annotation_text="Recession", annotation_font_color="#8b2d2d")
        fig_gdp.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            margin=dict(l=10, r=60, t=10, b=10), height=280,
            xaxis=dict(tickfont=dict(color="#8b949e"), showgrid=True,
                       gridcolor="#21262d", ticksuffix="%"),
            yaxis=dict(tickfont=dict(color="#c9d1d9", size=12), showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_gdp, use_container_width=True, config=dict(displayModeBar=False))
    else:
        st.info("GDP data loading… check internet connection.")
    st.markdown('</div>', unsafe_allow_html=True)

with chart_col2:
    st.markdown('<div class="card"><div class="card-header">🔥 Inflation Rate vs Target (%)</div>', unsafe_allow_html=True)
    if gdp_plot:
        df_cpi = pd.DataFrame(gdp_plot).sort_values("Inflation (%)", ascending=True)
        colors_cpi = []
        for _, r in df_cpi.iterrows():
            tgt = CURRENCIES[r["Currency"]]["inflation_target"]
            v   = r["Inflation (%)"]
            if abs(v - tgt) < 0.5:       colors_cpi.append("#3fb950")
            elif v > tgt * 1.5:           colors_cpi.append("#f85149")
            elif v < tgt * 0.5:           colors_cpi.append("#8b949e")
            else:                          colors_cpi.append("#e3b341")

        fig_cpi = go.Figure()
        fig_cpi.add_trace(go.Bar(
            y=[f"{r['Flag']} {r['Currency']}" for _, r in df_cpi.iterrows()],
            x=df_cpi["Inflation (%)"],
            orientation="h",
            marker_color=colors_cpi,
            text=[f"{v:.1f}%" for v in df_cpi["Inflation (%)"]],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=11),
            name="Inflation",
            hovertemplate="<b>%{y}</b><br>Inflation: %{x:.2f}%<extra></extra>",
        ))
        fig_cpi.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            margin=dict(l=10, r=60, t=10, b=10), height=280,
            xaxis=dict(tickfont=dict(color="#8b949e"), showgrid=True,
                       gridcolor="#21262d", ticksuffix="%"),
            yaxis=dict(tickfont=dict(color="#c9d1d9", size=12), showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_cpi, use_container_width=True, config=dict(displayModeBar=False))
    else:
        st.info("Inflation data loading…")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 4: GDP TREND SPARKLINES
# ══════════════════════════════════════════════════════════════════
st.markdown("## 📉 GDP Growth Trend (Last 5 Years)")

sparkline_cols = st.columns(3)
ccy_items = list(CURRENCIES.items())
for idx, (ccy, cfg) in enumerate(ccy_items):
    with sparkline_cols[idx % 3]:
        gdp_hist = macro_data.get(ccy, {}).get("gdp")
        if gdp_hist and len(gdp_hist) >= 2:
            years = [d[0] for d in gdp_hist]
            vals  = [d[1] for d in gdp_hist]
            latest = vals[-1]
            trend  = vals[-1] - vals[-2]
            trend_str = f"▲ {trend:+.1f}pp" if trend > 0 else f"▼ {trend:+.1f}pp"
            trend_col = "#3fb950" if trend > 0 else "#f85149"
            line_col  = "#3fb950" if latest > 2 else "#f85149" if latest < 0.5 else "#e3b341"

            fig_sp = go.Figure()
            fig_sp.add_trace(go.Scatter(
                x=years,
                y=vals,
                mode="lines+markers",
                line=dict(color="#3fb950", width=3),
                marker=dict(size=8, color="#3fb950"),
                fill="tozeroy",
                fillcolor='rgba(63, 185, 80, 0.08)',
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ))
            fig_sp.add_hline(y=0, line_dash="dot", line_color="#30363d", line_width=1)
            fig_sp.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                margin=dict(l=5, r=5, t=5, b=5), height=120,
                xaxis=dict(showgrid=False, showticklabels=True,
                           tickfont=dict(color="#484f58", size=9), linecolor="#21262d"),
                yaxis=dict(showgrid=False, showticklabels=True,
                           tickfont=dict(color="#484f58", size=9), ticksuffix="%"),
                showlegend=False,
            )
            st.markdown(f"""
            <div class="card" style="padding:12px 14px;margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-weight:700;color:#e6edf3;">{cfg['flag']} {ccy}</span>
                <span style="font-size:12px;color:{trend_col};font-weight:600;">{latest:.1f}% &nbsp;{trend_str}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig_sp, use_container_width=True, config=dict(displayModeBar=False))
        else:
            st.markdown(f"""
            <div class="card" style="padding:12px 14px;">
              <span style="font-weight:700;color:#e6edf3;">{cfg['flag']} {ccy}</span>
              <div style="color:#484f58;font-size:12px;margin-top:6px;">GDP data unavailable</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 5: PAIR MACRO BIAS TABLE
# ══════════════════════════════════════════════════════════════════
st.markdown("## 🎯 Pair-by-Pair Macro Bias")
st.caption("Derived from base vs quote currency macro score differential. Use to confirm checklist item #1.")

filtered_pairs = [p for p in PAIRS if p[0] in show_pairs]

bias_list = []
for pair, base, quote in filtered_pairs:
    pb = pair_macro_bias(base, quote, scores)
    bias_list.append((pair, base, quote, pb))

bull_pairs = [(p,b,q,pb) for p,b,q,pb in bias_list if pb["direction"] == "LONG"]
bear_pairs = [(p,b,q,pb) for p,b,q,pb in bias_list if pb["direction"] == "SHORT"]
neut_pairs = [(p,b,q,pb) for p,b,q,pb in bias_list if pb["direction"] == "NEUTRAL"]
watch_pairs= [(p,b,q,pb) for p,b,q,pb in bias_list if pb["direction"] == "Watch"]

def strength_stars(n):
    return "★" * n + "☆" * (4 - n)

def render_pair_group(pairs, badge_cls, badge_lbl, dir_icon):
    if not pairs:
        return
    for pair, base, quote, pb in pairs:
        stars = strength_stars(pb["strength"])
        st.markdown(f"""
        <div class="pair-row">
          <div style="flex:1;">
            <div class="pair-name">{pair}</div>
            <div class="pair-sub">{pb['reason']}</div>
          </div>
          <div style="text-align:right;margin-left:16px;">
            <div class="badge {badge_cls}">{dir_icon} {badge_lbl}</div>
            <div style="font-size:12px;color:#e3b341;margin-top:4px;letter-spacing:1px;">{stars}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

pcol1, pcol2, pcol3 = st.columns(3)

with pcol1:
    st.markdown(f'<div class="card"><div class="card-header">🟢 Macro Long Bias ({len(bull_pairs)})</div>', unsafe_allow_html=True)
    render_pair_group(bull_pairs, "badge-bull", "LONG", "🔼")
    if not bull_pairs:
        st.markdown('<div style="color:#484f58;font-size:12px;padding:8px 0;">No pairs with long bias</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with pcol2:
    st.markdown(f'<div class="card"><div class="card-header">🔴 Macro Short Bias ({len(bear_pairs)})</div>', unsafe_allow_html=True)
    render_pair_group(bear_pairs, "badge-bear", "SHORT", "🔽")
    if not bear_pairs:
        st.markdown('<div style="color:#484f58;font-size:12px;padding:8px 0;">No pairs with short bias</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with pcol3:
    st.markdown(f'<div class="card"><div class="card-header">⚪ Neutral / Watch ({len(neut_pairs)+len(watch_pairs)})</div>', unsafe_allow_html=True)
    render_pair_group(neut_pairs,  "badge-neut",  "NEUTRAL", "➡️")
    render_pair_group(watch_pairs, "badge-watch", "WATCH",   "👁️")
    if not neut_pairs and not watch_pairs:
        st.markdown('<div style="color:#484f58;font-size:12px;padding:8px 0;">No neutral pairs</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 6: RATE DIFFERENTIAL HEATMAP
# ══════════════════════════════════════════════════════════════════
st.markdown("## ⚖️ Interest Rate Differential Matrix")
st.caption("Base rate minus quote rate. Positive = carry trade favours base currency.")

ccy_names = list(CURRENCIES.keys())
rates_arr = [CURRENCIES[c]["cb_rate"] for c in ccy_names]
diff_matrix = [[round(r1 - r2, 2) for r2 in rates_arr] for r1 in rates_arr]

flags = [f"{CURRENCIES[c]['flag']} {c}" for c in ccy_names]
fig_diff = go.Figure(go.Heatmap(
    z=diff_matrix, x=flags, y=flags,
    text=[[f"{v:+.2f}%" for v in row] for row in diff_matrix],
    texttemplate="%{text}",
    textfont=dict(size=10, color="white"),
    colorscale=[
        [0.0,  "#8b2d2d"], [0.35, "#c0392b"],
        [0.45, "#1a1a2e"], [0.50, "#161b22"],
        [0.55, "#1a2e1a"], [0.65, "#238636"],
        [1.0,  "#0d4a2f"],
    ],
    zmid=0,
    colorbar=dict(
        title=dict(
            text="Rate diff",
            font=dict(color="#8b949e")
        ),
        tickfont=dict(color="#8b949e", size=11),
        bgcolor="#161b22",
        bordercolor="#21262d",
        borderwidth=1
    ),
    hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Rate diff: %{z:+.2f}%<extra></extra>",
))
fig_diff.update_layout(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    margin=dict(l=10, r=10, t=10, b=10), height=380,
    xaxis=dict(tickfont=dict(color="#8b949e", size=10), tickangle=-45, showgrid=False),
    yaxis=dict(tickfont=dict(color="#8b949e", size=10), showgrid=False, autorange="reversed"),
)
st.plotly_chart(fig_diff, use_container_width=True, config=dict(displayModeBar=False))
st.caption("🟢 Green = base currency has higher rate (carry advantage) · 🔴 Red = quote has higher rate")

# Footer
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid #21262d;">
  🌐 Macro Bias · World Bank API (GDP & CPI) · CB Rates manually maintained · Not financial advice
</div>
""", unsafe_allow_html=True)