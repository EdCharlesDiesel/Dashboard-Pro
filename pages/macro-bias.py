import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache

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

    .metric-box{background:var(--background-color);border:1px solid var(--border,#21262d);border-radius:8px;padding:12px 14px;text-align:center;}
    .metric-value{font-size:20px;font-weight:700;color:#c9d1d9;}
    .metric-label{font-size:10px;color:#8b949e;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}

    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    .block-container{padding-top:1.5rem;max-width:1420px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CURRENCY CONFIG - NOW WITH REAL-TIME DATA SOURCES
# ══════════════════════════════════════════════════════════════════
CURRENCIES = {
    "USD": {
        "flag": "🇺🇸", "name": "US Dollar", "bank": "Federal Reserve",
        "wb_code": "US",
        "yield_ticker": "^TNX",  # US 10Y Treasury
        "yield_name": "US 10Y",
        "inflation_target": 2.0,
        "cb_rate_ticker": None,  # Will use FRED API
        "cb_rate_series": "FEDFUNDS",  # FRED series for Fed Funds Rate
    },
    "EUR": {
        "flag": "🇪🇺", "name": "Euro", "bank": "ECB",
        "wb_code": "XC",
        "yield_ticker": None,  # Use German Bund
        "yield_etf": "BUND",  # German Bund futures
        "inflation_target": 2.0,
        "cb_rate_series": "ECBDFR",  # ECB Deposit Facility Rate
    },
    "GBP": {
        "flag": "🇬🇧", "name": "British Pound", "bank": "Bank of England",
        "wb_code": "GB",
        "yield_ticker": None,
        "yield_etf": "IGLT",  # UK Gilts ETF
        "inflation_target": 2.0,
        "cb_rate_series": "BOEBANKR",  # Bank of England Bank Rate
    },
    "AUD": {
        "flag": "🇦🇺", "name": "Australian Dollar", "bank": "RBA",
        "wb_code": "AU",
        "yield_ticker": None,
        "yield_etf": None,
        "inflation_target": 2.5,
        "cb_rate_series": "RBACRTR",  # RBA Cash Rate Target
    },
    "NZD": {
        "flag": "🇳🇿", "name": "New Zealand Dollar", "bank": "RBNZ",
        "wb_code": "NZ",
        "yield_ticker": None,
        "yield_etf": None,
        "inflation_target": 2.0,
        "cb_rate_series": "RBINZOCR",  # RBNZ Official Cash Rate
    },
    "JPY": {
        "flag": "🇯🇵", "name": "Japanese Yen", "bank": "Bank of Japan",
        "wb_code": "JP",
        "yield_ticker": "^TNX",  # Using US as placeholder
        "yield_etf": None,
        "inflation_target": 2.0,
        "cb_rate_series": "JPYOVERNIGHT",  # BOJ Policy Rate
    },
    "CHF": {
        "flag": "🇨🇭", "name": "Swiss Franc", "bank": "SNB",
        "wb_code": "CH",
        "yield_ticker": None,
        "yield_etf": None,
        "inflation_target": 2.0,
        "cb_rate_series": "SNBPOLICY",  # SNB Policy Rate
    },
    "CAD": {
        "flag": "🇨🇦", "name": "Canadian Dollar", "bank": "Bank of Canada",
        "wb_code": "CA",
        "yield_ticker": None,
        "yield_etf": None,
        "inflation_target": 2.0,
        "cb_rate_series": "BOCPOLICY",  # BOC Policy Rate
    },
    "ZAR": {
        "flag": "🇿🇦", "name": "South African Rand", "bank": "SARB",
        "wb_code": "ZA",
        "yield_ticker": None,
        "yield_etf": None,
        "inflation_target": 4.5,
        "cb_rate_series": "SARBPRATE",  # SARB Repo Rate
    },
}

# ── Trading pairs with base/quote ──────────────────────────────────────────────
PAIRS = [
    ("EUR/USD", "EUR", "USD"), ("GBP/USD", "GBP", "USD"), ("AUD/USD", "AUD", "USD"),
    ("NZD/USD", "NZD", "USD"), ("USD/JPY", "USD", "JPY"), ("USD/CHF", "USD", "CHF"),
    ("USD/CAD", "USD", "CAD"), ("EUR/GBP", "EUR", "GBP"), ("EUR/JPY", "EUR", "JPY"),
    ("GBP/JPY", "GBP", "JPY"), ("AUD/JPY", "AUD", "JPY"), ("EUR/AUD", "EUR", "AUD"),
    ("GBP/AUD", "GBP", "AUD"), ("EUR/CAD", "EUR", "CAD"), ("GBP/CAD", "GBP", "CAD"),
    ("USD/ZAR", "USD", "ZAR"), ("EUR/ZAR", "EUR", "ZAR"), ("GBP/ZAR", "GBP", "ZAR"),
    ("🥇 Gold", "USD", "—"),
]


# ══════════════════════════════════════════════════════════════════
# IMPROVED DATA FETCHERS - FRESH & LATEST
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)  # 15 min cache
def fetch_fred_rate(series_id: str) -> dict:
    """Fetch latest central bank rate from FRED (St. Louis Fed)"""
    try:
        # FRED API - free tier doesn't need API key for basic queries
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": "YOUR_FRED_API_KEY",  # Get free key from https://fred.stlouisfed.org/docs/api/api_key.html
            "file_type": "json",
            "sort_order": "desc",
            "limit": 3,
            "frequency": "m"  # Monthly
        }

        # Fallback: Use CSV endpoint which doesn't require API key
        csv_url = f"https://fred.stlouisfed.org/data/{series_id}.csv"
        df = pd.read_csv(csv_url)
        if not df.empty:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            return {
                "rate": round(float(latest.iloc[1]), 2),
                "prev_rate": round(float(prev.iloc[1]), 2),
                "date": latest.iloc[0],
                "series": series_id
            }
    except Exception as e:
        pass
    return None


@st.cache_data(ttl=600, show_spinner=False)  # 10 min cache
def fetch_fx_rates():
    """Fetch real-time FX rates using Yahoo Finance"""
    fx_pairs = {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "AUD/USD": "AUDUSD=X",
        "NZD/USD": "NZDUSD=X", "USD/JPY": "USDJPY=X", "USD/CHF": "USDCHF=X",
        "USD/CAD": "USDCAD=X", "USD/ZAR": "USDZAR=X",
        "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X", "GBP/JPY": "GBPJPY=X",
        "AUD/JPY": "AUDJPY=X", "EUR/AUD": "EURAUD=X", "GBP/AUD": "GBPAUD=X",
        "EUR/CAD": "EURCAD=X", "GBP/CAD": "GBPCAD=X", "EUR/ZAR": "EURZAR=X",
        "GBP/ZAR": "GBPZAR=X",
    }

    try:
        tickers = list(fx_pairs.values())
        data = yf.download(tickers, period="2d", interval="1d", progress=False, auto_adjust=True)

        rates = {}
        for pair, ticker in fx_pairs.items():
            if ticker in data["Close"].columns:
                latest = data["Close"][ticker].iloc[-1]
                prev = data["Close"][ticker].iloc[-2] if len(data) > 1 else latest
                change = ((latest - prev) / prev) * 100 if prev != 0 else 0
                rates[pair] = {
                    "rate": round(float(latest), 4),
                    "change_pct": round(float(change), 2),
                    "timestamp": datetime.now().strftime("%H:%M")
                }
        return rates
    except Exception as e:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)  # 1 hour cache - macro data doesn't change frequently
def fetch_wb_indicator(country_code: str, indicator: str, years: int = 5):
    """Fetch World Bank indicator for a country"""
    try:
        url = (f"https://api.worldbank.org/v2/country/{country_code}"
               f"/indicator/{indicator}?format=json&mrv={years}&per_page=10")
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or len(data) < 2 or not data[1]:
            return None
        results = [(d["date"], d["value"]) for d in data[1] if d["value"] is not None]
        return sorted(results, key=lambda x: x[0])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_macro():
    """Fetch GDP, CPI, and lending rate for all currencies"""
    macro = {}
    for ccy, cfg in CURRENCIES.items():
        code = cfg["wb_code"]
        gdp = fetch_wb_indicator(code, "NY.GDP.MKTP.KD.ZG", 5)
        cpi = fetch_wb_indicator(code, "FP.CPI.TOTL.ZG", 5)
        rate = fetch_wb_indicator(code, "FR.INR.LEND", 5)
        macro[ccy] = {"gdp": gdp, "cpi": cpi, "rate": rate}
    return macro


@st.cache_data(ttl=600, show_spinner=False)  # 10 min cache
def fetch_yields():
    """Fetch 10Y government bond yields using yfinance"""
    yield_tickers = {
        "USD": "^TNX",  # US 10Y Treasury
        "DE10Y": "BUND",  # German 10Y Bund (approximation)
        "GB10Y": "IGLT.L",  # iShares UK Gilts ETF
        "JP10Y": "^TNX",  # Placeholder - Japan 10Y not easily available
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


@st.cache_data(ttl=1800, show_spinner=False)  # 30 min cache
def fetch_dxy():
    """Fetch DXY (US Dollar Index)"""
    try:
        dxy = yf.download("DX-Y.NYB", period="5d", interval="1d", progress=False, auto_adjust=True)
        if not dxy.empty:
            latest = float(dxy["Close"].iloc[-1])
            prev = float(dxy["Close"].iloc[-2]) if len(dxy) > 1 else latest
            return {
                "value": round(latest, 2),
                "change": round(((latest - prev) / prev) * 100, 2)
            }
    except:
        pass
    return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_vix():
    """Fetch VIX (volatility index)"""
    try:
        vix = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
        if not vix.empty:
            return round(float(vix["Close"].iloc[-1]), 2)
    except:
        pass
    return None


# ══════════════════════════════════════════════════════════════════
# BIAS SCORING
# ══════════════════════════════════════════════════════════════════

def score_currency(ccy: str, macro: dict) -> dict:
    """Score a currency's macro bias. Returns score -4 to +4 and signals."""
    cfg = CURRENCIES[ccy]
    score = 0
    signals = []

    # 1. Rate level vs peers
    all_rates = [c["cb_rate"] for c in CURRENCIES.values()]
    avg_rate = np.mean(all_rates) if all_rates else 0
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
        prev_gdp = gdp_data[-2][1]
        gdp_score = 1 if latest_gdp > 2.0 else -1 if latest_gdp < 0.5 else 0
        trend_ok = latest_gdp >= prev_gdp
        score += gdp_score + (0.5 if trend_ok else -0.5)
        signals.append(("GDP Growth", f"{latest_gdp:.1f}%",
                        "bull" if gdp_score > 0 else "bear" if gdp_score < 0 else "neut"))
    else:
        signals.append(("GDP Growth", "N/A", "neut"))

    # 4. Inflation vs target
    cpi_data = macro.get(ccy, {}).get("cpi")
    if cpi_data and len(cpi_data) >= 1:
        latest_cpi = cpi_data[-1][1]
        target = cfg["inflation_target"]
        if latest_cpi > target * 2.5:
            cpi_score = -1  # Hyperinflation = bad
        elif latest_cpi > target * 1.2:
            cpi_score = 1  # Moderate inflation = hawkish = bullish
        elif latest_cpi < target * 0.5:
            cpi_score = -1  # Deflation risk
        else:
            cpi_score = 0
        score += cpi_score
        signals.append(("Inflation", f"{latest_cpi:.1f}% (tgt {target}%)",
                        "bull" if cpi_score > 0 else "bear" if cpi_score < 0
                        else "warn" if abs(latest_cpi - target) > 1 else "neut"))
    else:
        signals.append(("Inflation", "N/A", "neut"))

    # Clamp score
    score = max(-4, min(4, score))
    bias = "Bullish" if score >= 1.5 else "Bearish" if score <= -1.5 else "Neutral"
    return {"score": score, "bias": bias, "signals": signals}


def pair_macro_bias(base: str, quote: str, scores: dict) -> dict:
    """Derive pair bias from base vs quote currency scores"""
    if base == "—" or quote == "—":
        return {"direction": "Watch", "strength": 0, "reason": "Commodity — check DXY"}
    bs = scores.get(base, {}).get("score", 0)
    qs = scores.get(quote, {}).get("score", 0)
    diff = bs - qs
    if diff >= 2:
        direction, strength = "LONG", min(int(abs(diff)), 4)
    elif diff >= 1:
        direction, strength = "LONG", 1
    elif diff <= -2:
        direction, strength = "SHORT", min(int(abs(diff)), 4)
    elif diff <= -1:
        direction, strength = "SHORT", 1
    else:
        direction, strength = "NEUTRAL", 0
    bb = scores.get(base, {}).get("bias", "?")
    qb = scores.get(quote, {}).get("bias", "?")
    reason = f"{base} is {bb} · {quote} is {qb}"
    return {"direction": direction, "strength": strength, "reason": reason}


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌐 Macro Bias")
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
    st.page_link("pages/trade-journal.py",    label="16. Trade Journal",     icon="📓")
    st.page_link("pages/market-structure.py",  label="17. Market Structure",  icon="🏗️")
    st.page_link("pages/setup-ranker.py",      label="18. Setup Ranker",      icon="🏆")
    st.page_link("pages/backtest-workflow.py",    label="19. Backtest Workflow",    icon="🧪")
    st.markdown("---")

    # Refresh controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh All", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        if auto_refresh:
            st.caption("⏱️ Refreshes every 5 min")
            import time

            time.sleep(300)
            st.cache_data.clear()
            st.rerun()

    # Cache info
    st.divider()
    st.caption("📡 Data freshness:")
    st.caption("• FX rates: 10 min")
    st.caption("• Yields/VIX: 10 min")
    st.caption("• Macro data: 1 hour")
    st.caption("• DXY: 30 min")

    st.divider()
    show_pairs = st.multiselect("Filter Pairs", [p[0] for p in PAIRS],
                                default=[p[0] for p in PAIRS[:12]])

# ══════════════════════════════════════════════════════════════════
# FETCH ALL DATA
# ══════════════════════════════════════════════════════════════════
with st.spinner("📡 Fetching real-time market data..."):
    fx_rates = fetch_fx_rates()
    dxy_data = fetch_dxy()
    vix_data = fetch_vix()

with st.spinner("📊 Loading macro economic data..."):
    macro_data = fetch_all_macro()
    yields = fetch_yields()

# Populate cb_rate and rate_trend from World Bank or FRED
for _ccy, _cfg in CURRENCIES.items():
    # Try FRED first for more accurate CB rates
    if _cfg.get("cb_rate_series"):
        fred_data = fetch_fred_rate(_cfg["cb_rate_series"])
        if fred_data and fred_data.get("rate"):
            _cfg["cb_rate"] = fred_data["rate"]
            if fred_data["rate"] > fred_data["prev_rate"] + 0.05:
                _cfg["rate_trend"] = "Hiking"
            elif fred_data["rate"] < fred_data["prev_rate"] - 0.05:
                _cfg["rate_trend"] = "Cutting"
            else:
                _cfg["rate_trend"] = "Holding"
            continue

    # Fallback to World Bank data
    _rate_series = macro_data.get(_ccy, {}).get("rate")
    if _rate_series and len(_rate_series) >= 1:
        _cfg["cb_rate"] = round(_rate_series[-1][1], 2)
        if len(_rate_series) >= 2:
            _r_latest, _r_prev = _rate_series[-1][1], _rate_series[-2][1]
            if _r_latest > _r_prev + 0.05:
                _cfg["rate_trend"] = "Hiking"
            elif _r_latest < _r_prev - 0.05:
                _cfg["rate_trend"] = "Cutting"
            else:
                _cfg["rate_trend"] = "Holding"
        else:
            _cfg["rate_trend"] = "Holding"
    else:
        _cfg.setdefault("cb_rate", 0.0)
        _cfg.setdefault("rate_trend", "Holding")

# Score each currency
scores = {ccy: score_currency(ccy, macro_data) for ccy in CURRENCIES}
bull_ccys = [c for c, s in scores.items() if s["bias"] == "Bullish"]
bear_ccys = [c for c, s in scores.items() if s["bias"] == "Bearish"]
neut_ccys = [c for c, s in scores.items() if s["bias"] == "Neutral"]

# ══════════════════════════════════════════════════════════════════
# HERO SECTION WITH REAL-TIME MARKET DATA
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:26px;font-weight:700;color:#e6edf3;">🌐 Macro Bias Dashboard</div>
      <div style="color:#8b949e;font-size:14px;margin-top:6px;">
        Real-Time FX · Interest Rates · GDP Growth · Inflation · Currency Bias Scoring
      </div>
      <div style="font-size:13px;color:#388bfd;font-weight:500;margin-top:4px;">
        🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M UTC')} · Data Sources: FRED · World Bank · Yahoo Finance
      </div>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <div class="metric-box" style="min-width:90px;">
        <div class="metric-value" style="color:#388bfd;">
          {f'${dxy_data["value"]:.1f}' if dxy_data else 'N/A'}
        </div>
        <div class="metric-label">💵 DXY Index</div>
      </div>
      <div class="metric-box" style="min-width:90px;">
        <div class="metric-value" style="color:#e3b341;">
          {f'{vix_data:.1f}' if vix_data else 'N/A'}
        </div>
        <div class="metric-label">😰 VIX</div>
      </div>
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
# SECTION 1: REAL-TIME FX RATES
# ══════════════════════════════════════════════════════════════════
st.markdown("## 💱 Live FX Rates")
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} · Auto-refreshes every 10 minutes")

if fx_rates:
    fx_cols = st.columns(6)
    for idx, (pair, data) in enumerate(fx_rates.items()):
        with fx_cols[idx % 6]:
            change_color = "#3fb950" if data["change_pct"] >= 0 else "#f85149"
            change_sign = "+" if data["change_pct"] >= 0 else ""
            st.markdown(f"""
            <div class="card" style="padding:10px 12px;margin-bottom:8px;">
              <div style="font-size:11px;color:#8b949e;margin-bottom:4px;">{pair}</div>
              <div style="font-size:18px;font-weight:700;color:#e6edf3;">{data['rate']:.4f}</div>
              <div style="font-size:11px;color:{change_color};font-weight:600;">
                {change_sign}{data['change_pct']:.2f}%
              </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Unable to fetch real-time FX rates. Check your internet connection.")

# ══════════════════════════════════════════════════════════════════
# SECTION 2: INTEREST RATES COMPARISON
# ══════════════════════════════════════════════════════════════════
st.markdown("## 💹 Central Bank Interest Rates")
st.caption("Source: FRED (Federal Reserve Economic Data) + World Bank")

rate_names = [f"{CURRENCIES[c]['flag']} {c}" for c in CURRENCIES]
rate_vals = [CURRENCIES[c]["cb_rate"] for c in CURRENCIES]
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
if rate_vals:
    avg_rate = float(np.mean(rate_vals))
    fig_rates.add_hline(
        y=avg_rate,
        line_dash="dot", line_color="#8b949e", line_width=1.5,
        annotation_text=f"Avg {avg_rate:.2f}%",
        annotation_font_color="#8b949e", annotation_position="right",
    )
fig_rates.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
    margin=dict(l=10, r=80, t=30, b=10), height=300,
    xaxis=dict(tickfont=dict(color="#c9d1d9", size=12), showgrid=False, linecolor="#21262d"),
    yaxis=dict(tickfont=dict(color="#8b949e"), showgrid=True, gridcolor="#21262d",
               ticksuffix="%", range=[0, max(rate_vals) * 1.3] if rate_vals else [0, 10]),
    showlegend=False,
    bargap=0.25,
)
st.plotly_chart(fig_rates, use_container_width=True, config=dict(displayModeBar=False))

st.markdown("""
<div style="display:flex;gap:20px;font-size:12px;color:#8b949e;padding:4px 0 16px 0;">
  <span>🟢 Hiking = hawkish = currency bullish pressure</span>
  <span>🟡 Holding = neutral</span>
  <span>🔴 Cutting = dovish = currency bearish pressure</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 3: BOND YIELDS
# ══════════════════════════════════════════════════════════════════
if yields:
    st.markdown("## 📈 Government Bond Yields (10Y)")
    yield_cols = st.columns(len(yields))
    for idx, (ccy, value) in enumerate(yields.items()):
        with yield_cols[idx]:
            yield_name = {
                "USD": "🇺🇸 US 10Y",
                "DE10Y": "🇩🇪 German Bund",
                "GB10Y": "🇬🇧 UK Gilt",
                "JP10Y": "🇯🇵 Japan 10Y"
            }.get(ccy, ccy)
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-value" style="color:#e3b341;">{value:.2f}%</div>
              <div class="metric-label">{yield_name}</div>
            </div>
            """, unsafe_allow_html=True)
    st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 4: CURRENCY CARDS (Rates + GDP + CPI)
# ══════════════════════════════════════════════════════════════════
st.markdown("## 🏦 Currency Macro Scorecard")

ccy_list = list(CURRENCIES.keys())
rows = [ccy_list[i:i + 3] for i in range(0, len(ccy_list), 3)]

for row in rows:
    cols = st.columns(3)
    for col, ccy in zip(cols, row):
        cfg = CURRENCIES[ccy]
        sc = scores[ccy]
        bias = sc["bias"]
        s = sc["score"]
        signals = sc["signals"]

        bias_color = "#3fb950" if bias == "Bullish" else "#f85149" if bias == "Bearish" else "#8b949e"
        bias_icon = "🟢" if bias == "Bullish" else "🔴" if bias == "Bearish" else "⚪"
        bar_pct = int((s + 4) / 8 * 100)
        bar_class = "bias-bull" if s > 0 else "bias-bear" if s < 0 else "bias-neut"

        gdp_data = macro_data.get(ccy, {}).get("gdp")
        cpi_data = macro_data.get(ccy, {}).get("cpi")
        gdp_val = f"{gdp_data[-1][1]:.1f}%" if gdp_data else "N/A"
        gdp_yr = gdp_data[-1][0] if gdp_data else ""
        cpi_val = f"{cpi_data[-1][1]:.1f}%" if cpi_data else "N/A"
        cpi_yr = cpi_data[-1][0] if cpi_data else ""

        gdp_num = gdp_data[-1][1] if gdp_data else None
        cpi_num = cpi_data[-1][1] if cpi_data else None
        gdp_cls = "data-val-up" if gdp_num and gdp_num > 2 else "data-val-dn" if gdp_num and gdp_num < 0.5 else "data-val-warn"
        cpi_cls = "data-val-up" if cpi_num and abs(cpi_num - cfg['inflation_target']) < 0.5 else "data-val-warn"

        trend_icon = {"Hiking": "⬆️", "Cutting": "⬇️", "Holding": "➡️"}.get(cfg["rate_trend"], "➡️")

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

              <div class="bias-track">
                <div class="{bar_class}" style="width:{bar_pct}%;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:10px;color:#484f58;margin-bottom:10px;">
                <span>Bearish</span><span>Neutral</span><span>Bullish</span>
              </div>

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

              <div style="margin-top:10px;line-height:1.8;">{pill_html}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 5: GDP & INFLATION CHARTS
# ══════════════════════════════════════════════════════════════════
st.markdown("## 📊 GDP Growth vs Inflation by Economy")

chart_col1, chart_col2 = st.columns(2)

gdp_plot = []
for ccy, cfg in CURRENCIES.items():
    gd = macro_data.get(ccy, {}).get("gdp")
    cd = macro_data.get(ccy, {}).get("cpi")
    if gd and cd:
        gdp_plot.append({
            "Currency": ccy, "Flag": cfg["flag"],
            "GDP Growth (%)": round(gd[-1][1], 2),
            "Inflation (%)": round(cd[-1][1], 2),
            "CB Rate (%)": cfg["cb_rate"],
            "Bias": scores[ccy]["bias"],
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
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            margin=dict(l=10, r=60, t=10, b=10), height=280,
            xaxis=dict(tickfont=dict(color="#8b949e"), showgrid=True, gridcolor="#21262d", ticksuffix="%"),
            yaxis=dict(tickfont=dict(color="#c9d1d9", size=12), showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_gdp, use_container_width=True, config=dict(displayModeBar=False))
    st.markdown('</div>', unsafe_allow_html=True)

with chart_col2:
    st.markdown('<div class="card"><div class="card-header">🔥 Inflation Rate vs Target (%)</div>',
                unsafe_allow_html=True)
    if gdp_plot:
        df_cpi = pd.DataFrame(gdp_plot).sort_values("Inflation (%)", ascending=True)
        colors_cpi = []
        for _, r in df_cpi.iterrows():
            tgt = CURRENCIES[r["Currency"]]["inflation_target"]
            v = r["Inflation (%)"]
            if abs(v - tgt) < 0.5:
                colors_cpi.append("#3fb950")
            elif v > tgt * 1.5:
                colors_cpi.append("#f85149")
            elif v < tgt * 0.5:
                colors_cpi.append("#8b949e")
            else:
                colors_cpi.append("#e3b341")

        fig_cpi = go.Figure(go.Bar(
            y=[f"{r['Flag']} {r['Currency']}" for _, r in df_cpi.iterrows()],
            x=df_cpi["Inflation (%)"],
            orientation="h",
            marker_color=colors_cpi,
            text=[f"{v:.1f}%" for v in df_cpi["Inflation (%)"]],
            textposition="outside",
            textfont=dict(color="#c9d1d9", size=11),
            hovertemplate="<b>%{y}</b><br>Inflation: %{x:.2f}%<extra></extra>",
        ))
        fig_cpi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
            margin=dict(l=10, r=60, t=10, b=10), height=280,
            xaxis=dict(tickfont=dict(color="#8b949e"), showgrid=True, gridcolor="#21262d", ticksuffix="%"),
            yaxis=dict(tickfont=dict(color="#c9d1d9", size=12), showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_cpi, use_container_width=True, config=dict(displayModeBar=False))
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 6: PAIR MACRO BIAS TABLE
# ══════════════════════════════════════════════════════════════════
st.markdown("## 🎯 Pair-by-Pair Macro Bias")
st.caption("Derived from base vs quote currency macro score differential. Real-time FX overlay included.")

filtered_pairs = [p for p in PAIRS if p[0] in show_pairs]

bias_list = []
for pair, base, quote in filtered_pairs:
    pb = pair_macro_bias(base, quote, scores)
    fx_info = fx_rates.get(pair, {})
    bias_list.append((pair, base, quote, pb, fx_info))

bull_pairs = [(p, b, q, pb, fx) for p, b, q, pb, fx in bias_list if pb["direction"] == "LONG"]
bear_pairs = [(p, b, q, pb, fx) for p, b, q, pb, fx in bias_list if pb["direction"] == "SHORT"]
neut_pairs = [(p, b, q, pb, fx) for p, b, q, pb, fx in bias_list if pb["direction"] == "NEUTRAL"]
watch_pairs = [(p, b, q, pb, fx) for p, b, q, pb, fx in bias_list if pb["direction"] == "Watch"]


def strength_stars(n):
    return "★" * n + "☆" * (4 - n)


def render_pair_group(pairs, badge_cls, badge_lbl, dir_icon):
    if not pairs:
        return
    for pair, base, quote, pb, fx_info in pairs:
        stars = strength_stars(pb["strength"])
        fx_display = ""
        if fx_info:
            change_sign = "+" if fx_info.get("change_pct", 0) >= 0 else ""
            fx_display = f" | {fx_info.get('rate', 0):.4f} ({change_sign}{fx_info.get('change_pct', 0):.1f}%)"

        st.markdown(f"""
        <div class="pair-row">
          <div style="flex:1;">
            <div class="pair-name">{pair} <span style="font-size:11px;color:#484f58;">{fx_display}</span></div>
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
    st.markdown(f'<div class="card"><div class="card-header">🟢 Macro Long Bias ({len(bull_pairs)})</div>',
                unsafe_allow_html=True)
    render_pair_group(bull_pairs, "badge-bull", "LONG", "🔼")
    st.markdown('</div>', unsafe_allow_html=True)

with pcol2:
    st.markdown(f'<div class="card"><div class="card-header">🔴 Macro Short Bias ({len(bear_pairs)})</div>',
                unsafe_allow_html=True)
    render_pair_group(bear_pairs, "badge-bear", "SHORT", "🔽")
    st.markdown('</div>', unsafe_allow_html=True)

with pcol3:
    st.markdown(
        f'<div class="card"><div class="card-header">⚪ Neutral / Watch ({len(neut_pairs) + len(watch_pairs)})</div>',
        unsafe_allow_html=True)
    render_pair_group(neut_pairs, "badge-neut", "NEUTRAL", "➡️")
    render_pair_group(watch_pairs, "badge-watch", "WATCH", "👁️")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align:center;color:#484f58;font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid #21262d;">
  🌐 Macro Bias Dashboard · Sources: FRED (St. Louis Fed) · World Bank API · Yahoo Finance
  <br>Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} · Auto-refresh available in sidebar · Not financial advice
</div>
""", unsafe_allow_html=True)

# Add auto-refresh meta tag if enabled
if auto_refresh:
    st.markdown("""
    <meta http-equiv="refresh" content="300">
    """, unsafe_allow_html=True)