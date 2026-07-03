"""
regime.py
=========
One source of truth for market-regime detection, shared across the dashboard.

A regime filter decides WHAT KIND of market you're in — trending or balanced
(ranging) — so a strategy only trades the regime it's built for:
  • mean-reversion (POC fade, zone fade)  -> trade only when RANGING
  • trend-following (MA, breakout)         -> trade only when TRENDING

Any strategy that produces a +1 / 0 / -1 position Series can be gated with one
line:  position = apply_regime_filter(position, regime_mask(...))

Detectors:
  efficiency_ratio  — Kaufman: net move / total path; ~1 trend, ~0 chop
  adx               — Wilder's trend-strength index; >25 trend, <20 range
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# UI-only deps — guarded so the pure detectors below stay importable headlessly
# (e.g. `from regime import efficiency_ratio` in trading_lab_tab.py).
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


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman Efficiency Ratio: |net move over n| / sum of |bar moves| over n."""
    net = close.diff(n).abs()
    path = close.diff().abs().rolling(n).sum()
    return (net / path).replace([np.inf, -np.inf], np.nan)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's ADX — trend strength (not direction). >25 trending, <20 ranging."""
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = pd.concat([(high - low),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def regime_mask(method: str, threshold: float, want: str,
                close: pd.Series | None = None,
                high: pd.Series | None = None,
                low: pd.Series | None = None,
                n: int | None = None) -> pd.Series:
    """
    Boolean 'is the regime one we want to trade?' mask.

      method : 'Efficiency Ratio' | 'ADX'
      want   : 'ranging' (mean-reversion) | 'trending' (trend-following)
      threshold : ranging if value < threshold; trending if value >= threshold

    Unknown/early values resolve to False (stand aside), which is the safe
    default — comparisons against NaN are already False.
    """
    if method == "Efficiency Ratio":
        val = efficiency_ratio(close, n or 20)
    elif method == "ADX":
        val = adx(high, low, close, n or 14)
    else:
        raise ValueError(f"Unknown method: {method}")
    mask = (val < threshold) if want == "ranging" else (val >= threshold)
    return mask.reindex(val.index).fillna(False).astype(bool)


def regime_value(method: str, close=None, high=None, low=None, n=None) -> pd.Series:
    """Raw detector value (for plotting / inspecting the regime)."""
    if method == "Efficiency Ratio":
        return efficiency_ratio(close, n or 20)
    if method == "ADX":
        return adx(high, low, close, n or 14)
    raise ValueError(f"Unknown method: {method}")


def apply_regime_filter(signal: pd.Series, allow_mask: pd.Series) -> pd.Series:
    """Zero the signal wherever allow_mask is False (i.e. stand aside)."""
    out = signal.copy()
    mask = allow_mask.reindex(signal.index).fillna(False).astype(bool)
    out[~mask] = 0.0
    return out


# ===========================================================================
# DATA LOADER (UI only — the detectors above are the source of truth, unchanged)
# ===========================================================================

def _cache(ttl):
    return st.cache_data(ttl=ttl, show_spinner=False) if st is not None else (lambda f: f)


@_cache(ttl=3600)
def load_ohlc(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Daily OHLC via yfinance, with the project's standard MultiIndex flatten."""
    if yf is None:
        return pd.DataFrame()
    from src.db.market_cache import cached_ohlc
    df = cached_ohlc(ticker, period=period, interval="1d", ttl=3600)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.dropna(how="all")


# ===========================================================================
# UI
# ===========================================================================

# Sensible defaults + slider bounds per detector (presentation only).
DETECTOR_DEFAULTS = {
    "Efficiency Ratio": {"n": 20, "thr": 0.30, "lo": 0.05, "hi": 0.60, "step": 0.05},
    "ADX":              {"n": 14, "thr": 22.0, "lo": 10.0, "hi": 40.0, "step": 1.0},
}


def _true_runs(index, mask: pd.Series):
    """Contiguous [start, end] spans where a boolean series is True (for shading)."""
    runs, start = [], None
    vals = mask.values
    for i, v in enumerate(vals):
        if v and start is None:
            start = index[i]
        elif not v and start is not None:
            runs.append((start, index[i]))
            start = None
    if start is not None:
        runs.append((start, index[-1]))
    return runs


def _build_chart(df, value, mask, method, threshold, want):
    """Price (top, with the wanted regime shaded) + detector value vs threshold."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.05,
                        subplot_titles=(f"Price · shaded = {want} regime", method))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(width=1.3, color="#e6e6e6"), showlegend=False),
                  row=1, col=1)
    color = "rgba(0,255,65,0.14)" if want == "trending" else "rgba(0,224,255,0.14)"
    for x0, x1 in _true_runs(df.index, mask):
        fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor=color,
                      layer="below", row=1, col=1)
    fig.add_trace(go.Scatter(x=value.index, y=value, name=method,
                             line=dict(width=1.3, color="#00ff41"), showlegend=False),
                  row=2, col=1)
    fig.add_hline(y=threshold, line=dict(color="#ff8c00", width=1, dash="dot"),
                  row=2, col=1)
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10),
                      template="plotly_dark", xaxis_rangeslider_visible=False)
    return fig


def render():
    """Call this inside your dashboard tab."""
    st.header("🚦 Regime — trending vs ranging")
    st.caption("Classify the market's STATE before you pick a tactic: trend-following "
               "wants a trending regime, mean-reversion wants a ranging one. Same two "
               "detectors the Backtest Lab's regime filter uses — Kaufman Efficiency "
               "Ratio and Wilder's ADX.")

    if go is None:
        st.error("This page needs plotly — install it with `pip install plotly`.")
        return

    from src.instruments import INSTRUMENTS

    c1, c2, c3 = st.columns(3)
    inst_names = INSTRUMENTS.keys()
    default_name = st.session_state.get("selected_instrument", "EUR/USD")
    if default_name not in inst_names:
        default_name = inst_names[0]
    inst_name = c1.selectbox("Ticker", inst_names,
                             index=inst_names.index(default_name))
    ticker = INSTRUMENTS.get(inst_name).ticker
    period = c2.selectbox("History", ["1y", "2y", "5y"], index=1)
    method = c3.selectbox("Detector", ["Efficiency Ratio", "ADX"])

    d = DETECTOR_DEFAULTS[method]
    c4, c5, c6 = st.columns(3)
    n = c4.slider("Lookback (n)", 5, 50, d["n"])
    threshold = c5.slider("Threshold", d["lo"], d["hi"], d["thr"], d["step"],
                          help="Trending if the value ≥ threshold; ranging if below.")
    want = c6.radio("Highlight regime", ["trending", "ranging"], horizontal=True)

    df = load_ohlc(ticker, period)
    if df.empty or "Close" not in df:
        st.warning("Could not load price data (yfinance unreachable).")
        return
    close, high, low = df["Close"], df["High"], df["Low"]

    # All classification goes through the shared detectors above — nothing new.
    value = regime_value(method, close=close, high=high, low=low, n=n)
    trending = regime_mask(method, threshold, "trending",
                           close=close, high=high, low=low, n=n)
    want_mask = regime_mask(method, threshold, want,
                            close=close, high=high, low=low, n=n)

    now_trending = bool(trending.iloc[-1]) if len(trending) else False
    pct_trend = float(trending.mean()) * 100 if len(trending) else 0.0
    last_val = f"{float(value.dropna().iloc[-1]):.2f}" if value.notna().any() else "—"

    m = st.columns(3)
    m[0].metric("Current regime", "📈 TRENDING" if now_trending else "🔀 RANGING",
                help="Latest bar classified by the chosen detector & threshold.")
    m[1].metric(f"{method} now", last_val, help="Raw detector value on the latest bar.")
    m[2].metric("Time trending", f"{pct_trend:.0f}%",
                delta=f"{100 - pct_trend:.0f}% ranging", delta_color="off")

    st.plotly_chart(_build_chart(df, value, want_mask, method, threshold, want),
                    width="stretch")
    tactic = ("trend-following (MA, breakout)" if want == "trending"
              else "mean-reversion (POC / zone fade)")
    st.caption(f"Shaded bands mark **{want}** bars — where a {tactic} strategy is in "
               f"its element; outside them it should stand aside. The Backtest Lab "
               f"applies exactly this gate via `apply_regime_filter`.")


if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run regime.py")

    # Page bootstrap — kept lazy so the pure detectors above stay importable in a
    # headless context. Mirrors the legacy-page contract:
    # set_page_config → BloombergTheme.apply() → sidebar (header → divider → nav).
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="Regime · Trading System",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()

    with st.sidebar:
        st.markdown("### 🚦 Regime")
        st.caption("Trending vs ranging market state")
        st.divider()
        render_sidebar_nav()

    render()