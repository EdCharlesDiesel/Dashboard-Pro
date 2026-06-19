"""
forex_fundamentals_tab.py
=========================
A drop-in Streamlit tab for forex FUNDAMENTAL analysis.

Pure-data, no AI APIs. Sources:
  - FRED  (no API key needed — uses the public fredgraph.csv endpoint)
  - yfinance (currency pair prices for the differential overlay)
  - Central bank statement pages (plain HTTP fetch + deterministic text analysis)

Three sections:
  1. Rate Differentials  — the #1 driver of FX. Short- & long-term rate
     spreads between the two economies in a pair, overlaid on price.
  2. Economic Surprise   — a FREE proxy: latest macro prints vs their own
     recent trend (beat / miss), aggregated into a per-currency momentum
     score. Clearly labelled as a proxy (true consensus needs a paid feed).
  3. Statement Diff      — pulls the latest central-bank statement, diffs it
     against the previous one, and scores hawkish vs dovish wording shifts.

Entry point:  render()
Standalone:   streamlit run forex_fundamentals_tab.py
"""

from __future__ import annotations

import difflib
import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import requests

try:
    import streamlit as st
except Exception:  # allow importing the pure helpers without streamlit
    st = None

try:
    import yfinance as yf
except Exception:
    yf = None


# ===========================================================================
# CONFIG
# ===========================================================================

# OECD comparable rate series on FRED (monthly, M156N = percent per annum).
# Short-term = 3-month interbank; Long-term = 10y government bond yield.
SHORT_RATE_FRED = {
    "USD": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N",
    "CAD": "IR3TIB01CAM156N",
    "AUD": "IR3TIB01AUM156N",
}
LONG_RATE_FRED = {
    "USD": "IRLTLT01USM156N",
    "EUR": "IRLTLT01DEM156N",  # Germany as euro-area benchmark
    "JPY": "IRLTLT01JPM156N",
    "GBP": "IRLTLT01GBM156N",
    "CAD": "IRLTLT01CAM156N",
    "AUD": "IRLTLT01AUM156N",
}

# Macro indicators for the economic-surprise proxy.
# direction = +1 if "higher print is hawkish / currency-supportive",
#             -1 if "higher print is dovish / currency-negative" (e.g. unemployment).
MACRO_INDICATORS = {
    "USD": {
        "CPI YoY":        ("CPIAUCSL", "yoy", +1),
        "Unemployment":   ("UNRATE",   "level", -1),
        "Retail Sales":   ("RSAFS",    "yoy", +1),
    },
    "EUR": {
        "CPI (Euro HICP)": ("CP0000EZ19M086NEST", "yoy", +1),
        "Unemployment":    ("LRHUTTTTEZM156S",     "level", -1),
    },
    "GBP": {
        "CPI":          ("GBRCPIALLMINMEI", "yoy", +1),
        "Unemployment": ("LRHUTTTTGBM156S",  "level", -1),
    },
    "JPY": {
        "CPI":          ("JPNCPIALLMINMEI", "yoy", +1),
        "Unemployment": ("LRHUTTTTJPM156S",  "level", -1),
    },
    "CAD": {
        "CPI":          ("CANCPIALLMINMEI", "yoy", +1),
        "Unemployment": ("LRHUTTTTCAM156S",  "level", -1),
    },
    "AUD": {
        "CPI":          ("AUSCPIALLQINMEI", "yoy", +1),
        "Unemployment": ("LRHUTTTTAUM156S",  "level", -1),
    },
}

PAIRS = {
    "EUR/USD": ("EUR", "USD", "EURUSD=X"),
    "GBP/USD": ("GBP", "USD", "GBPUSD=X"),
    "USD/JPY": ("USD", "JPY", "JPY=X"),
    "AUD/USD": ("AUD", "USD", "AUDUSD=X"),
    "USD/CAD": ("USD", "CAD", "CAD=X"),
}

# Risk-on / risk-off signals. Each is oriented so a POSITIVE z-score = risk-ON.
#   kind:   'momentum' (z-score of ~1m return) or 'level' (z-score of the level)
#   orient: +1 if rising = risk-on, -1 if rising = risk-off (havens / fear)
RISK_SIGNALS = {
    "S&P 500":          {"ticker": "^GSPC",        "src": "yf",    "kind": "momentum", "orient": +1},
    "VIX (fear gauge)": {"ticker": "^VIX",         "src": "yf",    "kind": "level",    "orient": -1},
    "AUD/JPY":          {"ticker": "AUDJPY=X",     "src": "yf",    "kind": "momentum", "orient": +1},
    "Copper/Gold":      {"ticker": "_CUAU",        "src": "ratio", "kind": "momentum", "orient": +1},
    "HY credit spread": {"ticker": "BAMLH0A0HYM2", "src": "fred",  "kind": "level",    "orient": -1},
}

# Central bank statement pages. These DOM structures change occasionally;
# the fetcher degrades gracefully and you can override the URL in the UI.
CB_STATEMENTS = {
    "Federal Reserve (FOMC)":
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260129a.htm",
    "ECB":
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2026/html/index.en.html",
    "Bank of England":
        "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2026",
}

# Deterministic hawkish / dovish lexicon (lowercased, substring match).
HAWKISH_TERMS = [
    "raise", "raising", "increase", "tighten", "tightening", "restrictive",
    "elevated inflation", "inflationary pressure", "upside risk", "robust",
    "strong labor", "strong labour", "overheating", "vigilant", "further increases",
    "higher for longer", "firm", "resilient", "persistent inflation", "above target",
]
DOVISH_TERMS = [
    "cut", "cutting", "lower", "ease", "easing", "accommodative", "downside risk",
    "slowing", "slowdown", "weaken", "weakening", "softening", "soft", "moderating",
    "cooling", "patient", "support employment", "below target", "subdued",
    "disinflation", "loosen", "loosening",
]

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
HEADERS = {"User-Agent": "Mozilla/5.0 (forex-dashboard)"}


# ===========================================================================
# DATA HELPERS (pure / testable)
# ===========================================================================

def _fetch_fred_csv(series_id: str, timeout: int = 20) -> pd.Series:
    """Pull a FRED series as a date-indexed float Series. No API key needed."""
    url = FRED_CSV.format(sid=series_id)
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # First col is the date; second is the value. Column names vary by series.
    date_col, val_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    s = df.dropna(subset=[date_col]).set_index(date_col)[val_col].dropna()
    s.name = series_id
    return s


def compute_surprise(series: pd.Series, mode: str, lookback: int = 12) -> dict | None:
    """
    FREE surprise proxy: compare the latest print against a naive expectation
    built from the indicator's own recent history.

    expectation = mean of the prior `lookback` observations of the transformed
    series (YoY change for 'yoy', the level itself for 'level').
    surprise_z  = (latest - expectation) / std(prior window)

    Returns None if not enough data. This is a *proxy* for a real
    consensus-vs-actual surprise, which requires a paid economic-calendar feed.
    """
    if series is None or len(series) < lookback + 2:
        return None

    if mode == "yoy":
        # Detect periodicity to compute year-over-year correctly.
        freq = pd.infer_freq(series.index)
        periods = 12
        if freq and freq.startswith("Q"):
            periods = 4
        transformed = series.pct_change(periods=periods).dropna() * 100.0
    else:
        transformed = series.dropna()

    if len(transformed) < lookback + 1:
        return None

    latest = float(transformed.iloc[-1])
    window = transformed.iloc[-(lookback + 1):-1]
    expectation = float(window.mean())
    std = float(window.std(ddof=1))

    surprise = latest - expectation
    # Robust scale: prefer std; fall back to mean-abs-deviation; then to a
    # small floor so a genuine break in a near-flat series still registers.
    if std and not np.isnan(std) and std > 1e-9:
        scale = std
    else:
        mad = float((window - expectation).abs().mean())
        scale = mad if mad > 1e-9 else max(abs(expectation) * 0.01, 1e-6)
    z = surprise / scale
    return {
        "latest": latest,
        "expectation": expectation,
        "surprise": surprise,
        "z": z,
        "as_of": transformed.index[-1],
    }


def lexicon_score(text: str) -> dict:
    """Deterministic hawkish/dovish tone from a statement's text."""
    low = (text or "").lower()
    hawk = sum(low.count(t) for t in HAWKISH_TERMS)
    dove = sum(low.count(t) for t in DOVISH_TERMS)
    total = hawk + dove
    net = (hawk - dove) / total if total else 0.0  # -1 dovish .. +1 hawkish
    if net > 0.15:
        label = "Hawkish"
    elif net < -0.15:
        label = "Dovish"
    else:
        label = "Neutral / Mixed"
    return {"hawkish": hawk, "dovish": dove, "net": net, "label": label}


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def diff_statements(old_text: str, new_text: str) -> dict:
    """Sentence-level diff: what was added / removed between two statements."""
    old_s, new_s = _sentences(old_text), _sentences(new_text)
    sm = difflib.SequenceMatcher(a=old_s, b=new_s)
    added, removed = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            added.extend(new_s[j1:j2])
        if tag in ("replace", "delete"):
            removed.extend(old_s[i1:i2])
    return {
        "added": added,
        "removed": removed,
        "similarity": sm.ratio(),
        "old_tone": lexicon_score(old_text),
        "new_tone": lexicon_score(new_text),
    }


def extract_text_from_html(html: str) -> str:
    """Crude but dependency-light HTML -> text (no bs4 required)."""
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rolling_z(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score of a series (how unusual the latest value is vs ~1y)."""
    s = series.astype(float).dropna()
    mu = s.rolling(window, min_periods=60).mean()
    sd = s.rolling(window, min_periods=60).std(ddof=1)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def oriented_component(series: pd.Series, kind: str, orient: int,
                       lookback: int = 21, window: int = 252) -> pd.Series:
    """
    Turn a raw market series into a risk-on-oriented z-score series.
      kind 'momentum' -> z-score of the ~1m return (rising price)
      kind 'level'    -> z-score of the level itself (e.g. VIX, credit spread)
      orient +1/-1    -> flip so that POSITIVE always means risk-ON
    """
    base = series.astype(float).pct_change(lookback) if kind == "momentum" \
        else series.astype(float)
    return rolling_z(base, window) * orient


def composite_from_components(component_series: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Align oriented component z-scores and average them into one regime line."""
    clean = {k: v for k, v in component_series.items()
             if v is not None and len(v.dropna())}
    if not clean:
        return pd.DataFrame(), pd.Series(dtype=float)
    df = pd.DataFrame(clean).dropna(how="all")
    comp = df.mean(axis=1, skipna=True)
    return df, comp


def classify_regime(score: float) -> str:
    """Map a composite risk score to a regime label."""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "Unknown"
    if score > 0.4:
        return "Risk-On"
    if score < -0.4:
        return "Risk-Off"
    return "Neutral / Mixed"


def risk_fx_implications(label: str) -> str:
    """Plain-English FX tilt for the current regime."""
    if label == "Risk-Off":
        return ("**Capital flees to safety.** Favoured: **JPY** and **CHF** (top "
                "havens), then **USD** (haven vs commodity/EM FX). Pressured: "
                "AUD, NZD, CAD and EM (ZAR, MXN). Cleanest expressions: "
                "**short AUD/JPY, short NZD/JPY**. Note USD/JPY is muddy — both "
                "sides are havens.")
    if label == "Risk-On":
        return ("**Capital chases yield and growth.** Favoured: **AUD, NZD, CAD** "
                "and EM. Pressured: JPY and CHF (funding currencies get sold to "
                "fund risk trades). Cleanest expression: **long AUD/JPY**, the "
                "classic risk-barometer pair.")
    return ("**No dominant regime.** Risk flows aren't in the driver's seat — "
            "defer to the rate-differential, economic-surprise and priced-in "
            "tabs, which carry more signal when sentiment is neutral.")


def curve_implied_priced_in(short_rate: float, long_rate: float,
                            hike_size: float = 0.25) -> dict:
    """
    How much policy change the market has 'priced in', read from the yield curve.

    The short rate (~3m) reflects the current policy stance; a longer market
    rate reflects the average expected policy rate over its horizon. Their
    spread is what the market expects policy to do, in net terms.

        priced_bps  = (long_rate - short_rate) * 100
        moves_priced = priced_bps / (hike_size * 100)   # in 25bp units

    NOTE: the long rate also contains a term premium, so this overstates the
    pure rate-expectation component somewhat. It is reliable for *direction*
    and rough magnitude, not a precise probability. For a clean single-meeting
    probability use implied_hike_probability() with rate futures.
    """
    priced_bps = (long_rate - short_rate) * 100.0
    moves = priced_bps / (hike_size * 100.0)
    if priced_bps > 5:
        stance = "hikes priced in"
    elif priced_bps < -5:
        stance = "cuts priced in"
    else:
        stance = "roughly on hold"
    return {"priced_bps": priced_bps, "moves_priced": moves, "stance": stance}


def implied_hike_probability(futures_price: float, current_rate: float,
                             meeting_day: int, days_in_month: int,
                             hike_size: float = 0.25) -> dict:
    """
    Probability of a rate move at the next meeting, from a 30-day rate future
    (e.g. Fed funds future ZQ=F). Simplified CME-FedWatch methodology.

    A 30-day fed funds future settles on the AVERAGE daily effective rate over
    its contract month. If the meeting falls mid-month, the month's average is
    a blend of the rate before the meeting (= current_rate) and the rate after:

        avg_implied = 100 - futures_price
        avg_implied = (current_rate*days_before + rate_after*days_after) / N

    Solve for rate_after, then the fraction of a full hike priced in is:

        prob = (rate_after - current_rate) / hike_size

    Assumptions (documented, and they matter):
      * only one meeting in the contract month
      * the effective rate equals the target before the meeting
      * settlement = simple average of daily rates
    For a CUT, pass a negative hike_size or read a negative probability as the
    fraction of a cut priced in.
    """
    avg_implied = 100.0 - futures_price
    days_before = max(meeting_day - 1, 0)
    days_after = max(days_in_month - days_before, 1)
    rate_after = (avg_implied * days_in_month - current_rate * days_before) / days_after
    bp_priced = rate_after - current_rate
    prob = bp_priced / hike_size if hike_size else 0.0
    return {
        "rate_after": rate_after,
        "bp_priced": bp_priced * 100.0,   # in basis points
        "prob": prob,                     # fraction of a full move (can exceed 1)
        "avg_implied": avg_implied,
    }


# ===========================================================================
# CACHED LOADERS (streamlit)
# ===========================================================================

def _cache(ttl):
    """Return st.cache_data decorator if streamlit present, else a no-op."""
    if st is not None:
        return st.cache_data(ttl=ttl, show_spinner=False)
    return lambda f: f


@_cache(ttl=3600)
def load_rate(series_id: str) -> pd.Series:
    return _fetch_fred_csv(series_id)


@_cache(ttl=3600)
def load_macro(series_id: str) -> pd.Series:
    return _fetch_fred_csv(series_id)


@_cache(ttl=900)
def load_pair_price(ticker: str, period: str = "2y") -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float)
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    col = "Close" if "Close" in df.columns else df.columns[0]
    s = df[col].copy()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s.dropna()


@_cache(ttl=1800)
def load_last_price(ticker: str) -> float | None:
    """Latest price for a futures/quote ticker via yfinance (e.g. ZQ=F)."""
    if yf is None:
        return None
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        col = "Close" if "Close" in df.columns else df.columns[0]
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return float(s.dropna().iloc[-1])
    except Exception:
        return None


@_cache(ttl=3600)
def load_us_effective_rate() -> float | None:
    """Current effective Fed funds rate (FRED DFF), latest value."""
    try:
        return float(_fetch_fred_csv("DFF").iloc[-1])
    except Exception:
        return None


@_cache(ttl=1800)
def load_yf_series(ticker: str, period: str = "3y") -> pd.Series:
    """Generic daily close series via yfinance."""
    if yf is None:
        return pd.Series(dtype=float)
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    col = "Close" if "Close" in df.columns else df.columns[0]
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s.dropna()


@_cache(ttl=1800)
def load_risk_signal(name: str) -> pd.Series:
    """Load one risk signal (yfinance, FRED, or the copper/gold ratio)."""
    cfg = RISK_SIGNALS[name]
    src = cfg["src"]
    if src == "yf":
        return load_yf_series(cfg["ticker"])
    if src == "fred":
        return _fetch_fred_csv(cfg["ticker"])
    if src == "ratio":  # copper / gold
        cu = load_yf_series("HG=F")
        au = load_yf_series("GC=F")
        df = pd.concat([cu.rename("cu"), au.rename("au")], axis=1).dropna()
        return (df["cu"] / df["au"]).dropna() if not df.empty else pd.Series(dtype=float)
    return pd.Series(dtype=float)


@_cache(ttl=1800)
def fetch_statement(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return extract_text_from_html(r.text)


# ===========================================================================
# UI SECTIONS
# ===========================================================================

def _section_rate_differentials():
    st.subheader("📊 Rate Differentials")
    st.caption(
        "Currencies track *relative* rate expectations. A widening yield "
        "spread in favour of the base currency is fundamentally bullish for the pair."
    )

    c1, c2 = st.columns([1, 1])
    pair_name = c1.selectbox("Pair", list(PAIRS.keys()), key="rd_pair")
    horizon = c2.radio("Rate horizon", ["Short (3m)", "Long (10y)"],
                       horizontal=True, key="rd_horizon")

    base, quote, ticker = PAIRS[pair_name]
    table = SHORT_RATE_FRED if horizon.startswith("Short") else LONG_RATE_FRED

    try:
        base_rate = load_rate(table[base])
        quote_rate = load_rate(table[quote])
    except Exception as e:
        st.error(f"Could not load FRED rates: {e}")
        return

    df = pd.concat([base_rate.rename(base), quote_rate.rename(quote)], axis=1)
    df = df.dropna()
    if df.empty:
        st.warning("No overlapping rate data returned.")
        return
    df["Differential"] = df[base] - df[quote]

    latest = df.iloc[-1]
    d_now = latest["Differential"]
    d_6m = df["Differential"].iloc[-7] if len(df) > 7 else d_now
    trend = d_now - d_6m

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{base} rate", f"{latest[base]:.2f}%")
    m2.metric(f"{quote} rate", f"{latest[quote]:.2f}%")
    m3.metric(f"{base}-{quote} differential", f"{d_now:+.2f}%",
              delta=f"{trend:+.2f} vs ~6m ago")

    bias = (f"Differential favouring **{base}** and widening → fundamentally "
            f"supportive of {pair_name}." if trend > 0.02 else
            f"Differential favouring **{base}** but narrowing → fundamental "
            f"tailwind fading." if d_now > 0 else
            f"Differential favours **{quote}** → fundamental pressure on {pair_name}.")
    st.markdown(bias)

    # Overlay differential vs price (normalised to compare shapes).
    price = load_pair_price(ticker)
    if not price.empty:
        merged = pd.concat(
            [df["Differential"].rename("Rate differential"),
             price.rename("Price")], axis=1
        ).dropna()
        if not merged.empty:
            norm = (merged - merged.mean()) / merged.std()
            st.line_chart(norm, height=300)
            st.caption("Z-normalised so the *shape* of the differential and the "
                       "price can be compared. Divergences are where fundamentals "
                       "and price disagree — often the interesting trades.")
    else:
        st.line_chart(df[[base, quote, "Differential"]], height=300)


def _section_economic_surprise():
    st.subheader("⚡ Economic Surprise (proxy)")
    st.caption(
        "Latest macro print vs the indicator's own recent trend, expressed as a "
        "z-score. Positive = beating trend. This is a FREE proxy — a true "
        "consensus-vs-actual surprise needs a paid calendar feed; the hook is "
        "in compute_surprise() if you add one."
    )

    rows = []
    prog = st.progress(0.0)
    items = [(ccy, name, cfg) for ccy, ind in MACRO_INDICATORS.items()
             for name, cfg in ind.items()]
    for i, (ccy, name, (sid, mode, direction)) in enumerate(items):
        try:
            series = load_macro(sid)
            res = compute_surprise(series, mode)
        except Exception:
            res = None
        if res:
            signed_z = res["z"] * direction
            rows.append({
                "Currency": ccy,
                "Indicator": name,
                "Latest": round(res["latest"], 2),
                "Trend exp.": round(res["expectation"], 2),
                "Surprise (z)": round(res["z"], 2),
                "Currency impact (z)": round(signed_z, 2),
                "As of": res["as_of"].date().isoformat(),
            })
        prog.progress((i + 1) / len(items))
    prog.empty()

    if not rows:
        st.warning("No macro data returned (FRED may be unreachable).")
        return

    detail = pd.DataFrame(rows)
    st.dataframe(detail, use_container_width=True, hide_index=True)

    score = (detail.groupby("Currency")["Currency impact (z)"]
             .mean().sort_values(ascending=False).round(2))
    st.markdown("**Aggregate currency momentum (mean signed surprise):**")
    st.bar_chart(score, height=260)
    strongest, weakest = score.index[0], score.index[-1]
    st.markdown(
        f"Data momentum is strongest for **{strongest}** ({score.iloc[0]:+.2f}) "
        f"and weakest for **{weakest}** ({score.iloc[-1]:+.2f}). "
        f"All else equal this biases **{strongest}/{weakest}** higher — "
        f"cross-check against the rate differential before acting."
    )


def _section_statement_diff():
    st.subheader("🏛 Central Bank Statement Diff")
    st.caption(
        "Markets trade the *change* in central-bank language. Paste the previous "
        "and latest statements (or fetch the latest), and this surfaces added / "
        "removed sentences plus a deterministic hawkish–dovish tone shift."
    )

    bank = st.selectbox("Central bank", list(CB_STATEMENTS.keys()), key="cb_bank")
    default_url = CB_STATEMENTS[bank]
    url = st.text_input("Latest statement URL (editable)", value=default_url,
                        key="cb_url")

    st.markdown("**Option A — fetch latest, paste previous:**")
    fetch_new = st.button("Fetch latest statement", key="cb_fetch")
    new_text = st.session_state.get("cb_new_text", "")
    if fetch_new:
        try:
            new_text = fetch_statement(url)
            st.session_state["cb_new_text"] = new_text
            st.success(f"Fetched {len(new_text):,} characters.")
        except Exception as e:
            st.error(f"Fetch failed ({e}). Paste the text manually below.")

    new_text = st.text_area("Latest statement text", value=new_text, height=160,
                            key="cb_new_area")
    old_text = st.text_area("Previous statement text (paste)", height=160,
                            key="cb_old_area")

    if not (old_text.strip() and new_text.strip()):
        st.info("Provide both statements to run the diff.")
        return

    d = diff_statements(old_text, new_text)
    ot, nt = d["old_tone"], d["new_tone"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Previous tone", ot["label"], f"net {ot['net']:+.2f}")
    c2.metric("Latest tone", nt["label"], f"net {nt['net']:+.2f}")
    shift = nt["net"] - ot["net"]
    arrow = "↗ more hawkish" if shift > 0.05 else "↘ more dovish" if shift < -0.05 else "→ little change"
    c3.metric("Tone shift", arrow, f"{shift:+.2f}")
    st.caption(f"Statement similarity: {d['similarity']*100:.0f}% "
               f"(hawkish/dovish keyword counts — latest: {nt['hawkish']}/{nt['dovish']}).")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**🟢 New / changed language**")
        if d["added"]:
            for s in d["added"][:12]:
                st.markdown(f"- {s}")
        else:
            st.caption("No additions detected.")
    with cc2:
        st.markdown("**🔴 Removed language**")
        if d["removed"]:
            for s in d["removed"][:12]:
                st.markdown(f"- {s}")
        else:
            st.caption("No removals detected.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def _section_priced_in():
    st.subheader("🎯 Priced-In Analyzer")
    st.caption(
        "What a currency reacts to is the *surprise* — the gap between what "
        "happens and what the market already expects. This reads that "
        "expectation straight from the rate market, so you can judge whether "
        "it's too high or too low."
    )

    # ---- Engine 1: curve-implied, every currency, FRED no-key ----
    st.markdown("#### Curve-implied expectations (all currencies)")
    st.caption("Spread of a longer market rate over the ~3m rate ≈ the net "
               "policy change the market has priced over that horizon.")
    rows = []
    for ccy in SHORT_RATE_FRED:
        try:
            sr = float(load_rate(SHORT_RATE_FRED[ccy]).iloc[-1])
            lr = float(load_rate(LONG_RATE_FRED[ccy]).iloc[-1])
            c = curve_implied_priced_in(sr, lr)
            rows.append({
                "Currency": ccy,
                "Short ~3m": round(sr, 2),
                "Long 10y": round(lr, 2),
                "Priced (bps)": round(c["priced_bps"], 0),
                "≈ 25bp moves": round(c["moves_priced"], 1),
                "Market expects": c["stance"],
            })
        except Exception:
            continue
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("⚠️ The 10y rate carries a term premium, so this reads "
                   "*direction* and rough magnitude, not a clean probability. "
                   "For a single-meeting probability use the Fed funds tool below.")
    else:
        st.warning("Could not load curve data from FRED.")

    st.divider()

    # ---- Engine 2: Fed funds futures probability, US ----
    st.markdown("#### Fed funds futures → single-meeting probability (US)")
    st.caption("Backs out the implied post-meeting rate from the 30-day Fed "
               "funds future (ZQ=F) using the CME-style settlement formula.")

    auto_rate = load_us_effective_rate()
    auto_px = load_last_price("ZQ=F")

    c1, c2 = st.columns(2)
    current_rate = c1.number_input(
        "Current effective Fed funds rate (%)",
        value=float(round(auto_rate, 2)) if auto_rate else 4.33, step=0.25,
        help="Auto-filled from FRED DFF; override if needed.")
    futures_price = c2.number_input(
        "Front-month ZQ=F price",
        value=float(round(auto_px, 4)) if auto_px else 95.6700, step=0.0025,
        format="%.4f",
        help="Auto-filled from yfinance ZQ=F; implied avg rate = 100 − price.")

    c3, c4, c5 = st.columns(3)
    meeting_date = c3.date_input("Next meeting date", key="pi_meeting")
    hike_size = c4.selectbox("Move size", [0.25, 0.50], key="pi_size")
    direction = c5.radio("Testing for", ["Hike", "Cut"], horizontal=True,
                         key="pi_dir")

    import calendar as _cal
    dim = _cal.monthrange(meeting_date.year, meeting_date.month)[1]
    size = hike_size if direction == "Hike" else -hike_size
    res = implied_hike_probability(futures_price, current_rate,
                                   meeting_date.day, dim, hike_size=size)

    prob_pct = max(min(res["prob"], 1.5), -0.5) * 100  # display clamp
    m1, m2, m3 = st.columns(3)
    m1.metric("Implied avg rate", f"{res['avg_implied']:.3f}%")
    m2.metric("Implied post-meeting rate", f"{res['rate_after']:.3f}%",
              f"{res['bp_priced']:+.0f} bp")
    m3.metric(f"P({direction.lower()} priced in)", f"{prob_pct:.0f}%")

    if abs(res["prob"]) >= 0.9:
        verdict = (f"A {direction.lower()} is **essentially fully priced**. If "
                   f"delivered as-is, the reaction lives in the *guidance*, not "
                   f"the decision — watch the statement diff. A no-change here "
                   f"would be a large surprise.")
    elif abs(res["prob"]) <= 0.2:
        verdict = (f"Almost **no {direction.lower()} priced**. If one is "
                   f"delivered, it's a genuine surprise and the move could be sharp.")
    else:
        verdict = (f"Partially priced (~{abs(prob_pct):.0f}%). The market is "
                   f"undecided — the outcome itself still carries information.")
    st.markdown(verdict)
    st.caption("Assumes one meeting in the contract month and effective = target "
               "pre-meeting. A rough but standard read; treat near 0%/100% edges "
               "as approximate.")


def _section_risk_sentiment():
    st.subheader("🌡️ Risk Sentiment & Capital Flows")
    st.caption(
        "Markets swing between **risk-on** (chase yield and growth) and "
        "**risk-off** (flee to safe havens). In risk-off, capital floods into "
        "USD / JPY / CHF *regardless of the data* — so the regime often "
        "overrides the micro fundamentals in the other tabs. This gauge reads "
        "the regime from the market itself."
    )

    comp_series, latest_z = {}, {}
    prog = st.progress(0.0)
    names = list(RISK_SIGNALS.keys())
    for i, name in enumerate(names):
        cfg = RISK_SIGNALS[name]
        try:
            raw = load_risk_signal(name)
            oc = oriented_component(raw, cfg["kind"], cfg["orient"])
            comp_series[name] = oc
            v = oc.dropna()
            latest_z[name] = float(v.iloc[-1]) if len(v) else None
        except Exception:
            latest_z[name] = None
        prog.progress((i + 1) / len(names))
    prog.empty()

    df, comp = composite_from_components(comp_series)
    if comp.dropna().empty:
        st.warning("Could not load enough market data for the regime gauge "
                   "(yfinance / FRED may be unreachable).")
        return

    score = float(comp.dropna().iloc[-1])
    label = classify_regime(score)
    emoji = {"Risk-On": "🟢", "Risk-Off": "🔴",
             "Neutral / Mixed": "🟡", "Unknown": "⚪"}[label]

    prev = comp.dropna()
    delta = score - float(prev.iloc[-6]) if len(prev) > 6 else 0.0
    m1, m2 = st.columns([1, 2])
    m1.metric("Current regime", f"{emoji} {label}", f"score {score:+.2f}")
    m2.metric("Composite risk score (−2 fear ↔ +2 greed)", f"{score:+.2f}",
              f"{delta:+.2f} vs ~1wk ago")

    st.markdown(risk_fx_implications(label))

    # What's driving it — latest oriented z per component (+ = pushing risk-on).
    drivers = pd.Series({k: v for k, v in latest_z.items() if v is not None})
    if not drivers.empty:
        st.markdown("**What's driving the regime** (bars right = risk-on, left = risk-off):")
        st.bar_chart(drivers.sort_values(), height=240)

    # Regime over time so you can see the swings.
    st.markdown("**Regime over the last ~2 years:**")
    st.line_chart(comp.dropna().tail(504), height=260)
    st.caption("Above +0.4 = risk-on, below −0.4 = risk-off, in between = "
               "neutral. Signals: S&P 500 momentum, VIX, AUD/JPY (the classic "
               "risk barometer), copper/gold, and high-yield credit spreads — "
               "each z-scored and oriented so positive = risk-on.")


def render():
    """Call this inside your dashboard tab."""
    st.header("🌍 Forex Fundamentals")
    st.caption("Rate differentials · economic surprise · central-bank tone · "
               "what's priced in · risk sentiment — the macro backdrop behind "
               "every pair. Data: FRED (no key) + yfinance.")
    t1, t2, t3, t4, t5 = st.tabs(["📊 Rate Differentials",
                                  "⚡ Economic Surprise",
                                  "🏛 Statement Diff",
                                  "🎯 Priced In",
                                  "🌡️ Risk Sentiment"])
    with t1:
        _section_rate_differentials()
    with t2:
        _section_economic_surprise()
    with t3:
        _section_statement_diff()
    with t4:
        _section_priced_in()
    with t5:
        _section_risk_sentiment()


if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run forex_fundamentals_tab.py")
    st.set_page_config(page_title="Forex Fundamentals", layout="wide")
    render()