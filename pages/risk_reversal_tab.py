"""
risk_reversal_tab.py
===================
FX risk-reversal SENTIMENT, after Kathy Lien (Chapter 23).

A 25-delta risk reversal = (IV of 25d call) − (IV of 25d put). Positive = the
market pays up for upside (bullish positioning); negative = pays up for downside
(bearish). At EXTREMES (±1 standard deviation from its own mean) it works as a
CONTRARIAN signal: everyone leaning one way makes the move harder in that
direction. This page builds the bands, fires the signals, overlays RR on spot
(like Figures 23.1/23.2), and forward-tests whether the contrarian edge is real.

⚠️ DATA HONESTY: true risk reversals come from options desks (FXCM / Bloomberg /
Reuters) and are NOT free. So this page offers three sources:
  • Paste / upload YOUR RR data (the real thing, from your broker/terminal)
  • Free PROXY — rolling return-skew sentiment (a rough stand-in, clearly labelled)
  • Demo — synthetic RR so the page renders and teaches the mechanics
The book also names the one FREE positioning source: the CFTC Commitment of
Traders report (weekly, 3-day delay) — a natural companion to build next.

Entry point: render()
Standalone:  streamlit run risk_reversal_tab.py
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None


PAIRS = {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
         "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X"}


# ===========================================================================
# PURE / TESTABLE
# ===========================================================================

def rr_bands(rr: pd.Series, k: float = 1.0) -> dict:
    """Mean and ±k·SD extreme bands of a risk-reversal series."""
    mu = float(rr.mean())
    sd = float(rr.std(ddof=1))
    return {"mean": mu, "sd": sd, "upper": mu + k * sd, "lower": mu - k * sd}


def rr_signals(rr: pd.Series, upper: float, lower: float) -> pd.Series:
    """
    Contrarian signal on CROSSING into an extreme zone:
      RR rises to/above upper band -> sentiment overbought -> 'sell'
      RR falls to/below lower band -> sentiment oversold   -> 'buy'
    Only the first bar of each excursion fires (no repeat spam).
    """
    above = rr >= upper
    below = rr <= lower
    sell = above & ~above.shift(1, fill_value=False)
    buy = below & ~below.shift(1, fill_value=False)
    sig = pd.Series("", index=rr.index, dtype=object)
    sig[sell] = "sell"
    sig[buy] = "buy"
    return sig


def forward_return_test(spot: pd.Series, signals: pd.Series,
                        horizon: int = 10) -> pd.DataFrame:
    """
    For each signal, the spot return over the next `horizon` bars, and whether
    the CONTRARIAN call was right (buy -> price up, sell -> price down).
    """
    spot = spot.sort_index()
    rows = []
    for d, s in signals[signals != ""].items():
        if d in spot.index:
            pos = spot.index.get_loc(d)
        else:
            ind = spot.index.get_indexer([d], method="ffill")
            pos = int(ind[0])
        if pos < 0 or pos + horizon >= len(spot):
            continue
        p0, p1 = float(spot.iloc[pos]), float(spot.iloc[pos + horizon])
        ret = p1 / p0 - 1.0
        correct = (ret > 0) if s == "buy" else (ret < 0)
        rows.append({"date": d, "signal": s, "fwd_return": ret, "correct": bool(correct)})
    return pd.DataFrame(rows)


def returns_skew_proxy(close: pd.Series, window: int = 20) -> pd.Series:
    """
    FREE proxy for directional sentiment (NOT a true risk reversal): rolling
    skew of log returns, z-scored. Negative skew (fat downside tail) ~ put
    demand ~ negative RR. A rough stand-in only.
    """
    r = np.log(close).diff()
    sk = r.rolling(window).skew()
    mu = sk.rolling(252, min_periods=30).mean()
    sd = sk.rolling(252, min_periods=30).std()
    return ((sk - mu) / sd).replace([np.inf, -np.inf], np.nan)


def parse_rr_csv(text: str) -> pd.Series:
    """Parse pasted 'date,value' rows (or CSV) into a date-indexed RR series."""
    if not text or not text.strip():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(io.StringIO(text))
        if df.shape[1] >= 2:
            d = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            v = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            s = pd.Series(v.values, index=d).dropna()
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    except Exception:
        pass
    return pd.Series(dtype=float)


def synthetic_rr(spot: pd.Series, seed: int = 0) -> pd.Series:
    """Illustrative RR that leads spot turns — for the demo only."""
    rng = np.random.default_rng(seed)
    mom = np.log(spot).diff().rolling(10).mean().fillna(0.0)
    base = (mom / (mom.std() or 1)) * 0.6
    noise = pd.Series(rng.normal(0, 0.35, len(spot)), index=spot.index)
    return (base + noise).rolling(3).mean().fillna(0.0)


# ===========================================================================
# LOADER / UI
# ===========================================================================

def _cache(ttl):
    return st.cache_data(ttl=ttl, show_spinner=False) if st is not None else (lambda f: f)


@_cache(ttl=3600)
def load_spot(ticker: str, period: str = "2y") -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float)
    from src.db.market_cache import cached_ohlc
    df = cached_ohlc(ticker, period=period, interval="1d", ttl=3600)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    col = "Close" if "Close" in df.columns else df.columns[0]
    s = df[col].copy()
    s.index = pd.to_datetime(s.index)
    return s.dropna()


def _chart(spot, rr, bands, signals, pair):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=spot.index, y=spot, name=f"{pair} spot",
                             line=dict(color="#90caf9", width=1.6)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=rr.index, y=rr, name="Risk reversal",
                             line=dict(color="#bdbdbd", width=1, dash="dot")),
                  secondary_y=True)
    for y, c in [(bands["upper"], "#ef5350"), (bands["lower"], "#26a69a"),
                 (bands["mean"], "#777777")]:
        fig.add_hline(y=y, line=dict(color=c, width=1, dash="dash"),
                      secondary_y=True)

    # buy/sell markers on the spot line
    buys = signals[signals == "buy"].index
    sells = signals[signals == "sell"].index
    sp = spot.reindex(spot.index)
    for dates, sym, col, nm in [(buys, "triangle-up", "#26a69a", "Buy (oversold)"),
                                (sells, "triangle-down", "#ef5350", "Sell (overbought)")]:
        pts = sp.reindex(dates, method="ffill")
        fig.add_trace(go.Scatter(x=pts.index, y=pts.values, mode="markers",
                                 name=nm, marker=dict(symbol=sym, size=11, color=col,
                                 line=dict(width=1, color="white"))),
                      secondary_y=False)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=20, b=10),
                      template="plotly_dark", legend=dict(orientation="h", y=1.05))
    fig.update_yaxes(title_text=f"{pair}", secondary_y=False)
    fig.update_yaxes(title_text="Risk reversal", secondary_y=True)
    return fig


def render():
    st.header("🔄 Risk Reversals — options sentiment (contrarian at extremes)")
    st.caption("After Lien Ch.23: at ±1 SD extremes, RR is a contrarian signal — "
               "crowded bullish positioning makes a rally harder, and vice versa.")
    if go is None:
        st.error("Needs plotly — `pip install plotly`.")
        return

    st.warning("⚠️ Real 25-delta risk reversals are an **options-desk feed** "
               "(FXCM / Bloomberg / Reuters) — not free. Paste your own below, use "
               "the free return-skew proxy, or the demo. The only *free* positioning "
               "data the book names is the CFTC COT report (weekly, 3-day delay).")

    c1, c2, c3 = st.columns(3)
    pair = c1.selectbox("Pair", list(PAIRS.keys()))
    period = c2.selectbox("History", ["1y", "2y", "5y"], index=1)
    source = c3.selectbox("RR data source",
                          ["Demo (synthetic)", "Free proxy (return-skew)",
                           "Paste my RR data"])

    spot = load_spot(PAIRS[pair], period)
    if spot.empty:
        st.warning("Could not load spot price (yfinance unreachable).")
        return

    if source == "Paste my RR data":
        st.caption("Format: two columns `date,rr_value` (CSV or one per line).")
        txt = st.text_area("Risk-reversal data", height=120,
                           placeholder="2025-01-02,0.35\n2025-01-03,0.41\n...")
        rr = parse_rr_csv(txt)
        if rr.empty:
            st.info("Paste RR data to run the analysis.")
            return
        rr = rr.reindex(spot.index, method="nearest").dropna()
        proxy_note = ""
    elif source == "Free proxy (return-skew)":
        win = st.slider("Skew window", 10, 40, 20)
        rr = returns_skew_proxy(spot, win).dropna()
        proxy_note = ("This is a **rough free proxy** (return skew), not a true "
                      "options risk reversal — directionally useful, not exact.")
    else:
        rr = synthetic_rr(spot)
        proxy_note = ("**Synthetic / illustrative** data so the mechanics are "
                      "visible. Swap in real RR or the proxy for live use.")

    rr = rr.reindex(spot.index).dropna()
    if len(rr) < 30:
        st.info("Not enough overlapping data.")
        return

    k = st.slider("Extreme band (× std dev)", 0.5, 2.5, 1.0, 0.25)
    bands = rr_bands(rr, k)
    signals = rr_signals(rr, bands["upper"], bands["lower"])

    last = float(rr.iloc[-1])
    if last >= bands["upper"]:
        state, emoji = "Overbought — contrarian SELL zone", "🔴"
    elif last <= bands["lower"]:
        state, emoji = "Oversold — contrarian BUY zone", "🟢"
    else:
        state, emoji = "Neutral — no strong bias", "🟡"
    m = st.columns(3)
    m[0].metric("Current RR", f"{last:+.2f}")
    m[1].metric("Sentiment", f"{emoji} {state.split(' — ')[0]}")
    m[2].metric("Extreme bands", f"{bands['lower']:+.2f} / {bands['upper']:+.2f}")
    if proxy_note:
        st.caption(proxy_note)

    # Persist the contrarian read — but NOT on synthetic demo data (data honesty).
    # Overbought → contrarian SELL (Short); Oversold → contrarian BUY (Long).
    if source != "Demo (synthetic)" and ("Overbought" in state or "Oversold" in state):
        try:
            from src.services.signal_store import persist_signals
            persist_signals("risk_reversal", [{
                "pair": pair,
                "bias": "Short" if "Overbought" in state else "Long",
                "entry": float(spot.iloc[-1]),
                "conviction": "RR contrarian",
                "thesis": f"Risk Reversal — {state} (RR {last:+.2f})",
            }])
        except Exception:
            pass

    st.plotly_chart(_chart(spot, rr, bands, signals, pair), width="stretch")
    st.caption("Dashed RR (right axis) vs spot (left). Red/green dashed lines are "
               "the ±1 SD extreme bands; triangles mark contrarian buy/sell signals "
               "where RR crossed into an extreme.")

    # forward-return test of the contrarian claim
    hz = st.slider("Test horizon (days after signal)", 5, 30, 10)
    res = forward_return_test(spot, signals, hz)
    if not res.empty:
        hit = res["correct"].mean()
        avg = res.loc[res["correct"], "fwd_return"].abs().mean() if res["correct"].any() else 0
        mm = st.columns(3)
        mm[0].metric("Signals", len(res))
        mm[1].metric(f"Contrarian hit rate ({hz}d)", f"{hit*100:.0f}%")
        mm[2].metric("Avg move when right", f"{avg*100:.1f}%")
        st.dataframe(res.assign(
            date=res["date"].dt.date.astype(str),
            fwd_return=(res["fwd_return"] * 100).round(2)).rename(
            columns={"fwd_return": "fwd_return_%"}),
            width="stretch", hide_index=True)
        st.caption("A hit rate meaningfully above 50% supports the contrarian read. "
                   "On the demo/proxy don't trust the number — it's only real once "
                   "you feed in genuine options RR data.")
    else:
        st.info("No extreme signals in this window — try a smaller band multiple.")


if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run risk_reversal_tab.py")

    # Page bootstrap — kept lazy so the pure helpers above can still be imported
    # headlessly. Mirrors the legacy-page contract used across the app:
    # set_page_config → BloombergTheme.apply() → sidebar (header → divider → nav).
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="Risk Reversals · Trading System",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()

    with st.sidebar:
        st.markdown("### 🎭 Risk Reversals")
        st.caption("FX options sentiment — contrarian at extremes")
        st.divider()
        render_sidebar_nav()

    render()