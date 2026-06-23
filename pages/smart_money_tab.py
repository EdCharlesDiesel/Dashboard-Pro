"""
smart_money_tab.py
=================
"What is the 'smart money' doing?" — the honest, computable version.

No tool can see hidden institutional orders. What CAN be computed from public
price & volume are evidence-based PROXIES for institutional activity:

  • MONEY FLOW   — OBV, A/D line, Chaikin Money Flow, Money Flow Index: is volume
    accumulating (buying) or distributing (selling)?
  • SUPPLY/DEMAND ZONES (order blocks) — the consolidation a strong impulse left
    behind, where resting institutional orders are presumed to sit. Each zone is
    tagged FRESH or MITIGATED — the legitimate version of "avoid fake zones":
    a zone price has already traded back through is used up, not magic.

Needs REAL volume (stocks / ETFs / futures). Spot FX has none, so it's blocked.

Entry point: render()
Standalone:  streamlit run smart_money_tab.py
"""

from __future__ import annotations

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


# ===========================================================================
# MONEY-FLOW INDICATORS (pure / testable)
# ===========================================================================

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative volume signed by the day's direction."""
    return (np.sign(close.diff().fillna(0.0)) * volume).cumsum()


def adline(high, low, close, volume) -> pd.Series:
    """Accumulation/Distribution line — volume weighted by close location in range."""
    rng = (high - low).replace(0, np.nan)
    mfm = (((close - low) - (high - close)) / rng).fillna(0.0)
    return (mfm * volume).cumsum()


def cmf(high, low, close, volume, n: int = 20) -> pd.Series:
    """Chaikin Money Flow: n-period money-flow volume / volume. >0 buying pressure."""
    rng = (high - low).replace(0, np.nan)
    mfv = (((close - low) - (high - close)) / rng).fillna(0.0) * volume
    return mfv.rolling(n).sum() / volume.rolling(n).sum()


def mfi(high, low, close, volume, n: int = 14) -> pd.Series:
    """Money Flow Index — a volume-weighted RSI (0-100). >80 overbought, <20 oversold."""
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    up = tp.diff() > 0
    pos = rmf.where(up, 0.0).rolling(n).sum()
    neg = rmf.where(~up, 0.0).rolling(n).sum()
    mfr = pos / neg.replace(0, np.nan)
    return 100 - 100 / (1 + mfr)


def atr(high, low, close, n: int = 14) -> pd.Series:
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def money_flow_verdict(cmf_last: float, obv_series: pd.Series) -> dict:
    """Summarise net flow into an accumulation / distribution read."""
    slope = float(obv_series.diff(10).iloc[-1]) if len(obv_series) > 10 else 0.0
    if cmf_last > 0.05 and slope > 0:
        label, emoji = "Accumulation — net money flowing IN", "🟢"
    elif cmf_last < -0.05 and slope < 0:
        label, emoji = "Distribution — net money flowing OUT", "🔴"
    else:
        label, emoji = "Mixed / neutral flow", "🟡"
    return {"label": label, "emoji": emoji, "cmf": cmf_last, "obv_slope": slope}


# ===========================================================================
# SUPPLY / DEMAND ZONE DETECTION (pure / testable)
# ===========================================================================

def detect_zones(df: pd.DataFrame, lookahead: int = 5, impulse_atr: float = 2.0,
                 max_zones: int = 10) -> list[dict]:
    """
    Detect supply/demand zones (order blocks).

    A strong impulse (a move of >= impulse_atr ATRs over the next `lookahead`
    bars) implies institutions transacted at the consolidation it left behind:
      • up-impulse   -> DEMAND zone at the last down-candle before it
      • down-impulse -> SUPPLY zone at the last up-candle before it

    Each zone gets a strength (impulse size in ATR), a volume z-score at the
    base, and a status: 'fresh' (price has not traded back into it) or
    'mitigated' (it has — i.e. used up, the "fake"/weak ones to be wary of).
    """
    need = {"Open", "High", "Low", "Close", "Volume"}
    if df is None or df.empty or not need.issubset(df.columns):
        return []
    a = atr(df["High"], df["Low"], df["Close"])
    vmean = df["Volume"].rolling(20).mean()
    vstd = df["Volume"].rolling(20).std()
    O, H, L, C = (df["Open"].values, df["High"].values,
                  df["Low"].values, df["Close"].values)
    n = len(df)
    raw = {}
    for i in range(1, n - lookahead):
        ai = a.iloc[i]
        if not np.isfinite(ai) or ai <= 0:
            continue
        move = (C[i + lookahead] - C[i]) / ai
        if move >= impulse_atr:
            ztype = "demand"
            base = next((j for j in range(i, max(i - 3, -1), -1) if C[j] < O[j]), i)
        elif move <= -impulse_atr:
            ztype = "supply"
            base = next((j for j in range(i, max(i - 3, -1), -1) if C[j] > O[j]), i)
        else:
            continue
        vz = 0.0
        if np.isfinite(vstd.iloc[base]) and vstd.iloc[base] > 0:
            vz = float((df["Volume"].iloc[base] - vmean.iloc[base]) / vstd.iloc[base])
        key = (ztype, base)
        cand = {"type": ztype, "top": float(H[base]), "bottom": float(L[base]),
                "idx": base, "date": df.index[base], "strength": abs(move),
                "vol_z": vz}
        if key not in raw or cand["strength"] > raw[key]["strength"]:
            raw[key] = cand

    zones = list(raw.values())
    # status: has price re-entered the zone after it formed?
    for z in zones:
        after = df.iloc[z["idx"] + 1:]
        if after.empty:
            z["status"] = "fresh"
        else:
            hit = ((after["Low"] <= z["top"]) & (after["High"] >= z["bottom"])).any()
            z["status"] = "mitigated" if bool(hit) else "fresh"
        z["quality"] = z["strength"] + max(z["vol_z"], 0.0)
    zones.sort(key=lambda z: z["quality"], reverse=True)
    return zones[:max_zones]


# ===========================================================================
# LOADER
# ===========================================================================

def _cache(ttl):
    return st.cache_data(ttl=ttl, show_spinner=False) if st is not None else (lambda f: f)


@_cache(ttl=3600)
def load_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.dropna(how="all")


# ===========================================================================
# UI
# ===========================================================================

def _build_chart(df, zones, mf_name, mf_series):
    """Candlesticks + shaded zones, with a money-flow subplot below."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04,
                        subplot_titles=("Price + supply/demand zones", mf_name))
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
        close=df["Close"], name="Price", showlegend=False), row=1, col=1)

    x0, x1 = df.index[0], df.index[-1]
    for z in zones:
        demand = z["type"] == "demand"
        fresh = z["status"] == "fresh"
        color = ("rgba(38,166,154,%.2f)" if demand else "rgba(239,83,80,%.2f)")
        alpha = 0.28 if fresh else 0.10
        fig.add_shape(type="rect", x0=z["date"], x1=x1, y0=z["bottom"], y1=z["top"],
                      line=dict(width=0), fillcolor=color % alpha,
                      layer="below", row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=mf_series, name=mf_name,
                             line=dict(width=1.4), showlegend=False), row=2, col=1)
    if mf_name == "Chaikin Money Flow":
        fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False, template="plotly_dark")
    return fig


def render():
    st.header("🕵️ Smart Money — money flow & supply/demand zones")
    st.caption("The honest version of 'what the big money is doing': evidence-based "
               "proxies computed from public price & volume — not hidden orders. "
               "Anyone claiming to show you exactly what institutions are doing in "
               "real time is selling something.")

    if go is None:
        st.error("This page needs plotly — install it with `pip install plotly`.")
        return

    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker (needs real volume)", value="SPY")
    period = c2.selectbox("History", ["1y", "2y", "5y"], index=1)
    mf_name = c3.selectbox("Money-flow panel",
                           ["Chaikin Money Flow", "On-Balance Volume",
                            "Money Flow Index", "Accumulation/Distribution"])

    df = load_ohlcv(ticker, period)
    if df.empty or "Close" not in df:
        st.warning("Could not load price data (yfinance unreachable).")
        return
    if "Volume" not in df or df["Volume"].fillna(0).sum() <= 0:
        st.error(f"**{ticker} has no real volume** (spot FX shows zero/tick volume), "
                 "so money-flow and order-block analysis aren't meaningful. Use a "
                 "stock, ETF or futures contract — e.g. SPY, QQQ, AAPL, ES=F.")
        return

    c4, c5 = st.columns(2)
    sens = c4.slider("Zone impulse strength (ATR)", 1.0, 4.0, 2.0, 0.5,
                     help="Higher = only the most violent moves create zones.")
    mf_win = c5.slider("Money-flow window", 5, 40, 20)

    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]
    mf_map = {
        "Chaikin Money Flow": cmf(high, low, close, vol, mf_win),
        "On-Balance Volume": obv(close, vol),
        "Money Flow Index": mfi(high, low, close, vol, mf_win),
        "Accumulation/Distribution": adline(high, low, close, vol),
    }
    mf_series = mf_map[mf_name]

    # --- verdict ---
    cmf_last = float(cmf(high, low, close, vol, mf_win).dropna().iloc[-1])
    v = money_flow_verdict(cmf_last, obv(close, vol))
    m = st.columns(3)
    m[0].metric("Net flow", f"{v['emoji']} {v['label'].split(' — ')[0]}")
    m[1].metric("Chaikin Money Flow", f"{cmf_last:+.3f}",
                help="Sum of money-flow volume / volume. >0 buying, <0 selling.")
    m[2].metric("OBV 10-day slope", f"{v['obv_slope']:+,.0f}")
    st.caption(v["label"] + ". Flow leading price (rising flow, flat price = quiet "
               "accumulation; falling flow into a rally = distribution) is the tell "
               "that matters more than the absolute level.")

    zones = detect_zones(df, impulse_atr=sens)
    st.plotly_chart(_build_chart(df, zones, mf_name, mf_series),
                    width="stretch")
    st.caption("Solid zones are **fresh** (untested); faded zones are **mitigated** "
               "— price already traded back through them, so they're used up. That "
               "fresh/used distinction is the real 'fake zone' filter.")

    # --- nearest fresh zones ---
    px = float(close.iloc[-1])
    fresh = [z for z in zones if z["status"] == "fresh"]
    dem = sorted([z for z in fresh if z["type"] == "demand" and z["top"] < px],
                 key=lambda z: px - z["top"])
    sup = sorted([z for z in fresh if z["type"] == "supply" and z["bottom"] > px],
                 key=lambda z: z["bottom"] - px)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Nearest fresh demand (support) below**")
        if dem:
            z = dem[0]
            st.markdown(f"{z['bottom']:.2f} – {z['top']:.2f}  ·  "
                        f"{z['strength']:.1f} ATR impulse  ·  vol z {z['vol_z']:+.1f}")
        else:
            st.caption("None fresh below current price.")
    with cc2:
        st.markdown("**Nearest fresh supply (resistance) above**")
        if sup:
            z = sup[0]
            st.markdown(f"{z['bottom']:.2f} – {z['top']:.2f}  ·  "
                        f"{z['strength']:.1f} ATR impulse  ·  vol z {z['vol_z']:+.1f}")
        else:
            st.caption("None fresh above current price.")

    if zones:
        st.markdown("**All detected zones** (ranked by quality = impulse + volume)")
        zdf = pd.DataFrame([{
            "Type": z["type"], "Bottom": round(z["bottom"], 2),
            "Top": round(z["top"], 2), "Formed": z["date"].date().isoformat(),
            "Impulse (ATR)": round(z["strength"], 1),
            "Vol z": round(z["vol_z"], 1), "Status": z["status"],
        } for z in zones])
        st.dataframe(zdf, width="stretch", hide_index=True)
    st.caption("A high-quality zone = strong impulse **and** elevated volume at the "
               "base **and** still fresh. A zone on weak volume or already mitigated "
               "is exactly the 'fake' one to skip — now you can see which is which "
               "instead of taking a reel's word for it.")


if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run smart_money_tab.py")

    # Page bootstrap — kept lazy so the pure helpers above can still be imported
    # in a headless (no-Streamlit) context. Mirrors the legacy-page contract:
    # set_page_config → BloombergTheme.apply() → sidebar (header → divider → nav).
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="Smart Money · Trading System",
        page_icon="🕵️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()

    with st.sidebar:
        st.markdown("### 🕵️ Smart Money")
        st.caption("Money flow & supply/demand zones")
        st.divider()
        render_sidebar_nav()

    render()