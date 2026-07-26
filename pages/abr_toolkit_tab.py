"""
abr_toolkit_tab.py
==================
ABR Price Action Toolkit — standalone dashboard page (auto-registered from
pages/, see src/pages_lib/navigation.py's ABRT entry).

Port of the TradingView / MT5 indicator:
  swings -> BoS/CHoCH (with ATR strength) -> order blocks (mitigation + age)
  -> auto trendlines + break detection -> MTF EMA bias -> quality score
  -> trade plan (entry / SL / TP1-3, USD risk sizing)

Journaling: each instrument's latest read is upserted to the `abr_signals`
table via TradeRepository.save_abr_signal (src/db/trade_repository.py) —
the same pooled Postgres connection every other page uses, resolved from
session state the same way (auto_connect()). `render(repo=None)` skips
journaling when the DB isn't configured.

Data source: yfinance (1h fetched and resampled for H4/H6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.db.cache import pooled_repository
from src.db.connection import current_db_config
from src.instruments.registry import INSTRUMENTS as _REGISTRY
from src.instruments.registry import TREND_COMMODITIES as _COMMODITIES
from src.pages_lib.navigation import render_sidebar_nav
from src.ui.theme import BloombergTheme as T

# ----------------------------------------------------------------------------
# Instrument presets: yfinance ticker + USD value of a 1.0 price move per lot,
# derived from the shared registry (was previously a hardcoded 7-symbol
# subset). Symbol keys are compact ("XAUUSD" not "XAU/USD") to match this
# page's own abr_signals.symbol convention (MT5-style naming for the fetch
# fallback in _fetch_ohlc_mt5).
#
# point_value derivation differs by asset class:
#  - Forex: registry pip / pip_size (standard retail 100k-unit lot formula —
#    matches this dict's pre-existing EUR/USD and GBP/USD values exactly).
#  - Commodities: real futures-contract multipliers, NOT the registry's
#    generic pip formula (that's tuned for forex lot sizing and is simply
#    wrong for futures — e.g. it would give silver 1000 instead of the real
#    5000, a 5x position-sizing error for a COMEX SI=F contract).
# ----------------------------------------------------------------------------
_SYMBOL_OVERRIDES = {"WTI/USD": "WTI"}  # crude isn't quoted as "WTI/USD" by convention
_COMMODITY_POINT_VALUES = {
    "XAU/USD": 100.0,    # COMEX GC=F: 100 troy oz/contract
    "XAG/USD": 5000.0,   # COMEX SI=F: 5,000 troy oz/contract
    "XPT/USD": 50.0,     # NYMEX PL=F: 50 troy oz/contract
    "WTI/USD": 1000.0,   # NYMEX CL=F: 1,000 barrels/contract
}


def _point_value(name: str, inst) -> float:
    if name in _COMMODITIES:
        return _COMMODITY_POINT_VALUES[name]
    return inst.pip / inst.pip_size


INSTRUMENTS: dict[str, dict] = {
    _SYMBOL_OVERRIDES.get(name, name.replace("/", "")): {
        "yf": inst.ticker, "point_value": _point_value(name, inst),
    }
    for name, inst in _REGISTRY.items()
}
# BTC/USD isn't a registry instrument (crypto, no forex/metal pip convention
# applies) — kept as an ABR-only addition; point_value=1.0 since BTC-USD is
# quoted in whole dollars (a $1 move = $1/unit).
INSTRUMENTS["BTCUSD"] = {"yf": "BTC-USD", "point_value": 1.0}

TIMEFRAMES = {"H1": "1h", "H4": "4h", "H6": "6h", "D1": "1d"}


# ============================================================================
# Engine (pure pandas — unit-testable, no Streamlit imports needed)
# ============================================================================
@dataclass
class StructEvent:
    idx: int
    time: pd.Timestamp
    kind: str          # "BoS" | "CHoCH"
    direction: int     # +1 bull / -1 bear
    level: float
    strength: str      # "Trend Strong" | "Trend Weak"


@dataclass
class OrderBlock:
    idx: int
    time: pd.Timestamp
    bullish: bool
    top: float
    bottom: float
    mitigated: bool = False
    mitigated_idx: int | None = None


@dataclass
class ABRResult:
    trend: int = 0
    events: list[StructEvent] = field(default_factory=list)
    obs: list[OrderBlock] = field(default_factory=list)
    swing_highs: list[tuple[int, float]] = field(default_factory=list)
    swing_lows: list[tuple[int, float]] = field(default_factory=list)
    tl_hi: tuple[tuple[int, float], tuple[int, float]] | None = None
    tl_lo: tuple[tuple[int, float], tuple[int, float]] | None = None
    tl_break_active: bool = False
    # trade plan
    signal: int = 0
    entry: float = np.nan
    sl: float = np.nan
    tp1: float = np.nan
    tp2: float = np.nan
    tp3: float = np.nan
    lots: float = 0.0
    sl_usd: float = np.nan
    tp_usd: tuple[float, float, float] = (np.nan, np.nan, np.nan)
    quality: int = 0
    grade: str = "-"
    htf_bias: dict[str, int] = field(default_factory=dict)
    htf_aligned: int = 0
    cur_bias: int = 0


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _ema_bias(close: pd.Series, fast: int, slow: int) -> int:
    if len(close) < slow + 5:
        return 0
    f = close.ewm(span=fast, adjust=False).mean().iloc[-1]
    s = close.ewm(span=slow, adjust=False).mean().iloc[-1]
    c = close.iloc[-1]
    if f > s and c > f:
        return 1
    if f < s and c < f:
        return -1
    return 0


def _find_pivots(df: pd.DataFrame, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays: bar i is a swing high/low vs L bars each side (strict)."""
    h, lo = df["High"].values, df["Low"].values
    n = len(df)
    is_hi = np.zeros(n, dtype=bool)
    is_lo = np.zeros(n, dtype=bool)
    for i in range(L, n - L):
        win_h = h[i - L: i + L + 1]
        win_l = lo[i - L: i + L + 1]
        if h[i] == win_h.max() and (win_h == h[i]).sum() == 1:
            is_hi[i] = True
        if lo[i] == win_l.min() and (win_l == lo[i]).sum() == 1:
            is_lo[i] = True
    return is_hi, is_lo


def run_engine(
    df: pd.DataFrame,
    swing_len: int = 5,
    ob_search: int = 15,
    ob_max_age: int = 300,
    max_ob: int = 2,
    tl_lookback: int = 10,
    sl_buf_atr: float = 0.5,
    tp_r: tuple[float, float, float] = (1.0, 2.0, 2.83),
    risk_usd: float = 30.0,
    point_value: float = 100.0,
    htf_frames: dict[str, pd.DataFrame] | None = None,
    ma_fast: int = 21,
    ma_slow: int = 50,
) -> ABRResult:
    """df: OHLC indexed by datetime, oldest first."""
    res = ABRResult()
    n = len(df)
    if n < swing_len * 2 + 20:
        return res

    atr = _atr(df).values
    close = df["Close"].values
    opn = df["Open"].values
    high = df["High"].values
    low = df["Low"].values
    idx = df.index

    is_hi, is_lo = _find_pivots(df, swing_len)

    last_hi = last_lo = np.nan
    hi_act = lo_act = False
    trend = 0

    for j in range(swing_len, n):
        c = j - swing_len            # pivot candidate confirmed at bar j
        if is_hi[c]:
            last_hi, hi_act = high[c], True
            res.swing_highs.append((c, high[c]))
        if is_lo[c]:
            last_lo, lo_act = low[c], True
            res.swing_lows.append((c, low[c]))

        a = atr[j] if not np.isnan(atr[j]) and atr[j] > 0 else np.nan

        if hi_act and not np.isnan(last_hi) and close[j] > last_hi:
            pen = (close[j] - last_hi) / a if not np.isnan(a) else 0.0
            res.events.append(StructEvent(
                j, idx[j], "CHoCH" if trend == -1 else "BoS", 1, last_hi,
                "Trend Strong" if pen >= 0.5 else "Trend Weak"))
            # bullish OB = last down candle before the up impulse
            for b in range(j - 1, max(j - 1 - ob_search, -1), -1):
                if close[b] < opn[b]:
                    res.obs.append(OrderBlock(b, idx[b], True, high[b], low[b]))
                    break
            trend, hi_act = 1, False

        if lo_act and not np.isnan(last_lo) and close[j] < last_lo:
            pen = (last_lo - close[j]) / a if not np.isnan(a) else 0.0
            res.events.append(StructEvent(
                j, idx[j], "CHoCH" if trend == 1 else "BoS", -1, last_lo,
                "Trend Strong" if pen >= 0.5 else "Trend Weak"))
            for b in range(j - 1, max(j - 1 - ob_search, -1), -1):
                if close[b] > opn[b]:
                    res.obs.append(OrderBlock(b, idx[b], False, high[b], low[b]))
                    break
            trend, lo_act = -1, False

    res.trend = trend
    res.signal = trend

    # ---- OB mitigation + age ------------------------------------------------
    for ob in res.obs:
        for j in range(ob.idx + 1, n):
            if ob.bullish and close[j] < ob.bottom:
                ob.mitigated, ob.mitigated_idx = True, j
                break
            if not ob.bullish and close[j] > ob.top:
                ob.mitigated, ob.mitigated_idx = True, j
                break
        if not ob.mitigated and (n - 1 - ob.idx) > ob_max_age:
            ob.mitigated, ob.mitigated_idx = True, n - 1

    # cap visible per side (keep most recent)
    for bullish in (True, False):
        alive = [o for o in res.obs if o.bullish == bullish and not o.mitigated]
        for o in alive[:-max_ob]:
            o.mitigated, o.mitigated_idx = True, n - 1

    # ---- trendlines ----------------------------------------------------------
    def tl_value(p_old, p_new, j):
        (b1, v1), (b2, v2) = p_old, p_new
        if b1 == b2:
            return v2
        return v2 + (v2 - v1) / (b2 - b1) * (j - b2)

    def broke_recently(p_old, p_new, downward: bool) -> bool:
        for j in range(n - 1, max(n - 1 - tl_lookback, 1), -1):
            v, v_prev = tl_value(p_old, p_new, j), tl_value(p_old, p_new, j - 1)
            if downward and close[j] < v and close[j - 1] >= v_prev:
                return True
            if not downward and close[j] > v and close[j - 1] <= v_prev:
                return True
        return False

    if len(res.swing_highs) >= 2:
        res.tl_hi = (res.swing_highs[-2], res.swing_highs[-1])
    if len(res.swing_lows) >= 2:
        res.tl_lo = (res.swing_lows[-2], res.swing_lows[-1])

    # ---- MTF bias ------------------------------------------------------------
    if htf_frames:
        for name, hdf in htf_frames.items():
            res.htf_bias[name] = _ema_bias(hdf["Close"], ma_fast, ma_slow)
        res.htf_aligned = sum(1 for b in res.htf_bias.values()
                              if res.signal != 0 and b == res.signal)
    res.cur_bias = _ema_bias(df["Close"], ma_fast, ma_slow)

    # ---- trade plan ----------------------------------------------------------
    a_now = atr[-1] if not np.isnan(atr[-1]) else 0.0
    buf = a_now * sl_buf_atr

    def latest_ob(bullish: bool) -> OrderBlock | None:
        for o in reversed(res.obs):
            if o.bullish == bullish and not o.mitigated:
                return o
        return None

    if res.signal < 0:
        ob = latest_ob(False)
        swing_hi = res.swing_highs[-1][1] if res.swing_highs else high[-2]
        res.entry = close[-1] if ob is None else ob.bottom
        res.sl = (swing_hi if ob is None else max(ob.top, swing_hi)) + buf
        r = res.sl - res.entry
        if r > 0:
            res.tp1, res.tp2, res.tp3 = (res.entry - r * m for m in tp_r)
        res.tl_break_active = (res.tl_lo is not None
                               and broke_recently(*res.tl_lo, downward=True))
    elif res.signal > 0:
        ob = latest_ob(True)
        swing_lo = res.swing_lows[-1][1] if res.swing_lows else low[-2]
        res.entry = close[-1] if ob is None else ob.top
        res.sl = (swing_lo if ob is None else min(ob.bottom, swing_lo)) - buf
        r = res.entry - res.sl
        if r > 0:
            res.tp1, res.tp2, res.tp3 = (res.entry + r * m for m in tp_r)
        res.tl_break_active = (res.tl_hi is not None
                               and broke_recently(*res.tl_hi, downward=False))

    risk_dist = abs(res.sl - res.entry)
    if risk_dist > 0 and point_value > 0 and not np.isnan(risk_dist):
        lots = risk_usd / (risk_dist * point_value)
        res.lots = max(round(lots / 0.01) * 0.01, 0.01)
        res.sl_usd = risk_dist * point_value * res.lots
        res.tp_usd = tuple(
            abs(res.entry - tp) * point_value * res.lots if not np.isnan(tp) else np.nan
            for tp in (res.tp1, res.tp2, res.tp3))

    # ---- quality ---------------------------------------------------------------
    if res.signal != 0:
        q = 25 + res.htf_aligned * 15
        q += 15 if latest_ob(res.signal > 0) is not None else 0
        q += 15 if res.tl_break_active else 0
        res.quality = q
        res.grade = ("A+" if q >= 85 else "A" if q >= 70 else
                     "B" if q >= 55 else "C" if q >= 40 else "D")
    return res


# ============================================================================
# Data
# ============================================================================
_MT5_TIMEFRAMES = {"1h": "TIMEFRAME_H1", "4h": "TIMEFRAME_H4",
                   "6h": "TIMEFRAME_H6", "1d": "TIMEFRAME_D1"}


def _fetch_ohlc_mt5(symbol: str, tf: str) -> pd.DataFrame:
    """Candles straight from a running Deriv/MT5 terminal — matches your
    actual fills instead of yfinance's futures/spot basis offset. Empty
    DataFrame (never raises) if MetaTrader5 isn't installed, the terminal
    isn't running, or the symbol isn't found; caller falls back to yfinance.

    Deliberately not in requirements.txt: it's a Windows-only package with no
    Linux wheel/sdist, so pinning it breaks `pip install` in CI/on Linux. Install
    it yourself (`pip install MetaTrader5`) on the Windows box running the app."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return pd.DataFrame()
    try:
        if not mt5.initialize():
            return pd.DataFrame()
        timeframe = getattr(mt5, _MT5_TIMEFRAMES[tf])
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1500)
    except Exception:
        return pd.DataFrame()
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return (df.set_index("time")
              .rename(columns={"open": "Open", "high": "High",
                                "low": "Low", "close": "Close"})
              [["Open", "High", "Low", "Close"]])


def _fetch_ohlc_yfinance(ticker: str, tf: str) -> pd.DataFrame:
    """H1/D1 native; H4/H6 resampled from 1h. Returns OHLC, oldest first."""
    if tf == "1d":
        raw = yf.download(ticker, period="2y", interval="1d",
                          progress=False, auto_adjust=False)
    else:
        raw = yf.download(ticker, period="720d", interval="1h",
                          progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].dropna()
    if tf in ("4h", "6h"):
        df = df.resample(tf).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
    return df.tail(1500)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlc(symbol: str, yf_ticker: str, tf: str) -> pd.DataFrame:
    """Tries the local MT5 terminal first (exact broker candles, all four
    timeframes native); falls back to yfinance (H4/H6 resampled from H1)
    when MT5 isn't installed/running. Returns OHLC, oldest first."""
    df = _fetch_ohlc_mt5(symbol, tf)
    if not df.empty:
        return df.tail(1500)
    return _fetch_ohlc_yfinance(yf_ticker, tf)


# ============================================================================
# Journaling (optional — skipped when `repo` is None, i.e. DB not configured)
# ============================================================================
def journal_signal(repo, symbol: str, tf: str, r: ABRResult) -> None:
    """`repo`: a TradeRepository (see src/db/cache.py's pooled_repository)."""
    try:
        repo.save_abr_signal(
            symbol, tf,
            signal="BUY" if r.signal > 0 else "SELL" if r.signal < 0 else "NONE",
            entry=None if np.isnan(r.entry) else float(r.entry),
            sl=None if np.isnan(r.sl) else float(r.sl),
            tp3=None if np.isnan(r.tp3) else float(r.tp3),
            quality=int(r.quality), grade=r.grade, htf_aligned=int(r.htf_aligned),
        )
    except Exception as exc:                     # journaling must never break the page
        st.caption(f"journal skipped: {exc}")


# ============================================================================
# Chart
# ============================================================================
def build_chart(df: pd.DataFrame, r: ABRResult, symbol: str) -> go.Figure:
    view = df.tail(300)
    fig = go.Figure(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"],
        low=view["Low"], close=view["Close"],
        increasing_line_color=T.GREEN, decreasing_line_color=T.RED,
        name=symbol, showlegend=False))

    off = len(df) - len(view)

    # order blocks (active only)
    for ob in r.obs:
        if ob.mitigated or ob.idx < off:
            continue
        fig.add_shape(type="rect", x0=df.index[ob.idx], x1=view.index[-1],
                      y0=ob.bottom, y1=ob.top,
                      fillcolor="rgba(0,255,102,0.15)" if ob.bullish
                                 else "rgba(255,51,68,0.15)",
                      line=dict(width=0.5, color=T.GREEN if ob.bullish else T.RED))

    # structure event labels (last 8 within view)
    for ev in [e for e in r.events if e.idx >= off][-8:]:
        fig.add_annotation(
            x=df.index[ev.idx], y=ev.level,
            text=f"{ev.kind}-{ev.level:.3f}-{ev.strength}",
            showarrow=True, arrowhead=0, ax=0,
            ay=25 if ev.direction < 0 else -25,
            font=dict(size=9, color=T.CYAN if ev.direction > 0 else T.YELLOW))

    # trendlines
    for tl, col in ((r.tl_hi, T.RED), (r.tl_lo, T.CYAN)):
        if tl and tl[0][0] >= off:
            (b1, p1), (b2, p2) = tl
            fig.add_trace(go.Scatter(
                x=[df.index[b1], df.index[b2]], y=[p1, p2], mode="lines",
                line=dict(color=col, width=2), showlegend=False))

    # level lines
    if r.signal != 0 and not np.isnan(r.entry):
        side = "BUY" if r.signal > 0 else "SELL"
        for px, name, col, dash in (
                (r.entry, f"{side} ENTRY", T.GREEN, "solid"),
                (r.sl,    f"{side} SL",    T.RED,   "solid"),
                (r.tp1,   f"{side} TP1",   T.CYAN,  "dot"),
                (r.tp2,   f"{side} TP2",   T.CYAN,  "dash"),
                (r.tp3,   f"{side} TP3",   T.CYAN,  "solid")):
            if not np.isnan(px):
                fig.add_hline(y=px, line_color=col, line_dash=dash, line_width=1,
                              annotation_text=f"{name} {px:.3f}",
                              annotation_font_color=col,
                              annotation_position="right")

    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=25, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=T.FONT_MONO, color=T.WHITE, size=10),
        xaxis=dict(showgrid=True, gridcolor=T.BORDER, tickfont=dict(color=T.GREY)),
        yaxis=dict(showgrid=True, gridcolor=T.BORDER, tickfont=dict(color=T.GREY)))
    return fig


def _render_abr_tv(df: pd.DataFrame, r: "ABRResult", symbol: str) -> None:
    """TradingView (lightweight-charts) view of the ABR chart.

    Honest degradation vs the Plotly `build_chart`: order blocks are the
    toolkit's signature feature and are *filled rectangles* there — this engine
    has no rectangle primitive, so each active block is drawn as its top+bottom
    bound lines (spanning from where the block formed to now) instead of a
    shaded box. Structure events become markers; trendlines and the risk-plan
    levels map cleanly. Switch to Terminal for the filled order-block boxes."""
    from src.ui import tv_charts
    view = df.tail(300)
    off = len(df) - len(view)

    # structure-event markers on the candles (arrow up/down by direction)
    _markers = []
    for ev in [e for e in r.events if e.idx >= off][-8:]:
        up = ev.direction > 0
        _markers.append(tv_charts.marker(
            df.index[ev.idx], "belowBar" if up else "aboveBar",
            T.CYAN if up else T.YELLOW, "arrowUp" if up else "arrowDown",
            f"{ev.kind}"))
    _markers.sort(key=lambda m: m["time"])
    series = [tv_charts.candle_series(view, markers=_markers or None)]

    # order blocks (active, in view) → top+bottom bound lines from block origin
    for ob in r.obs:
        if ob.mitigated or ob.idx < off:
            continue
        _obt = list(df.index[ob.idx:])
        _oc = T.GREEN if ob.bullish else T.RED
        series.append(tv_charts.level_series(_obt, ob.top, _oc, "OB", dash="dot", last_value=False))
        series.append(tv_charts.level_series(_obt, ob.bottom, _oc, "", dash="dot", last_value=False))

    # trendlines (2-point lines)
    for tl, col in ((r.tl_hi, T.RED), (r.tl_lo, T.CYAN)):
        if tl and tl[0][0] >= off:
            (b1, p1), (b2, p2) = tl
            series.append(tv_charts.line_series(
                [df.index[b1], df.index[b2]], pd.Series([p1, p2]), col, "TL",
                width=2, last_value=False))

    # risk-plan levels
    if r.signal != 0 and not np.isnan(r.entry):
        side = "BUY" if r.signal > 0 else "SELL"
        for px, ttl, col, dash in (
                (r.entry, f"{side} ENTRY", T.GREEN, "solid"),
                (r.sl,    f"{side} SL",    T.RED,   "solid"),
                (r.tp1,   f"{side} TP1",   T.CYAN,  "dot"),
                (r.tp2,   f"{side} TP2",   T.CYAN,  "dash"),
                (r.tp3,   f"{side} TP3",   T.CYAN,  "solid")):
            if not np.isnan(px):
                series.append(tv_charts.level_series(list(view.index), px, col,
                                                     f"{ttl} {px:.3f}", dash=dash))

    tv_charts.render([tv_charts.price_chart(series, height=560)],
                     key=f"tv_abr_{symbol}")
    st.caption("🅣 TradingView pilot · candles + structure markers + trendlines "
               "+ risk-plan levels. Order blocks show as top/bottom bound lines "
               "(this engine has no shaded rectangle) — switch to Terminal for "
               "the filled order-block boxes.")


# ============================================================================
# Panel + page
# ============================================================================
def _bias_str(b: int) -> str:
    return "BULLISH" if b > 0 else "BEARISH" if b < 0 else "NEUTRAL"


def render_instrument(repo, name: str, cfg: dict, params: dict) -> None:
    tf_label = params["tf_label"]
    df = fetch_ohlc(name, cfg["yf"], TIMEFRAMES[tf_label])
    if df.empty:
        st.warning(f"No data returned for {name} ({cfg['yf']}) — check the MT5 symbol/ticker.")
        return

    # HTF frames for bias (resample from the base 1h/daily fetch)
    hourly = fetch_ohlc(name, cfg["yf"], "1h")
    htf = {}
    if not hourly.empty:
        for lbl, rule in (("H1", "1h"), ("H4", "4h"), ("H6", "6h")):
            htf[lbl] = (hourly if rule == "1h" else hourly.resample(rule).agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            ).dropna())

    r = run_engine(
        df,
        swing_len=params["swing_len"], ob_search=15,
        ob_max_age=params["ob_max_age"], max_ob=params["max_ob"],
        tl_lookback=10, sl_buf_atr=params["sl_buf"],
        tp_r=(params["tp1r"], params["tp2r"], params["tp3r"]),
        risk_usd=params["risk_usd"], point_value=cfg["point_value"],
        htf_frames=htf)

    left, right = st.columns([3, 1])
    with left:
        if params.get("tv_engine"):
            try:
                _render_abr_tv(df, r, name)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"TradingView render failed ({exc}); showing Terminal chart.")
                st.plotly_chart(build_chart(df, r, name), width="stretch",
                                key=f"abr_chart_{name}")
        else:
            st.plotly_chart(build_chart(df, r, name), width="stretch",
                            key=f"abr_chart_{name}")
    with right:
        st.markdown(f"#### {name} · {tf_label}")
        st.markdown(
            f"**Structure Bias:** :{'green' if r.trend > 0 else 'red' if r.trend < 0 else 'gray'}[{_bias_str(r.trend)}]")
        st.markdown(f"**Market State:** {_bias_str(r.cur_bias) if r.cur_bias else 'RANGING'}")
        n_bull = sum(1 for o in r.obs if o.bullish and not o.mitigated)
        n_bear = sum(1 for o in r.obs if not o.bullish and not o.mitigated)
        st.markdown(f"**Order Blocks:** {n_bull} Bull / {n_bear} Bear")
        st.markdown(f"**Trendline Break:** {'🟡 ACTIVE' if r.tl_break_active else '—'}")
        st.markdown("---")
        sig = "BUY" if r.signal > 0 else "SELL" if r.signal < 0 else "NONE"
        st.markdown(f"**Signal:** :{'green' if r.signal > 0 else 'red'}[{sig}]")
        bias_line = " / ".join(f"{k} {_bias_str(v)}" for k, v in r.htf_bias.items())
        st.markdown(f"**HTF:** {bias_line}  \n**Combined:** [{r.htf_aligned}/3]")
        st.markdown(f"**Quality:** {r.grade} ({r.quality}/100)")
        st.markdown("---")
        if r.signal != 0 and not np.isnan(r.entry):
            st.markdown(
                f"**Entry:** {r.entry:.3f}  \n**SL:** {r.sl:.3f}  \n"
                f"**TP1/2/3:** {r.tp1:.3f} / {r.tp2:.3f} / {r.tp3:.3f}")
            st.markdown(
                f"**Lots:** {r.lots:.2f} @ USD {params['risk_usd']:.0f} risk  \n"
                f"**SL:** −{r.sl_usd:.2f} USD  \n"
                f"**TPs:** +{r.tp_usd[0]:.0f} / +{r.tp_usd[1]:.0f} / +{r.tp_usd[2]:.0f} USD")
        else:
            st.caption("No active setup.")

    if len(r.events) > 0:
        with st.expander("Recent structure events"):
            ev_df = pd.DataFrame([{
                "time": e.time, "event": e.kind,
                "dir": "BULL" if e.direction > 0 else "BEAR",
                "level": round(e.level, 3), "strength": e.strength,
            } for e in r.events[-15:]])
            st.dataframe(ev_df, use_container_width=True, hide_index=True)

    if repo is not None:
        journal_signal(repo, name, tf_label, r)


def render(repo=None) -> None:
    st.title("ABR Price Action Toolkit")
    st.caption("Structure · Order Blocks · Trendlines · MTF Bias · Risk Plan")

    with st.sidebar:
        st.subheader("ABR settings")
        chosen = st.multiselect("Instruments", sorted(INSTRUMENTS.keys()),
                                default=["XAUUSD", "XAGUSD"])
        tf_label = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=0)
        risk_usd = st.number_input("Risk per trade (USD)", 1.0, 10000.0, 30.0)
        swing_len = st.slider("Swing length", 3, 12, 5)
        max_ob = st.slider("Max OBs per side", 1, 6, 2)
        ob_max_age = st.slider("OB max age (bars)", 50, 1000, 300, step=50)
        sl_buf = st.slider("SL buffer (x ATR)", 0.0, 2.0, 0.5, 0.1)
        c1, c2, c3 = st.columns(3)
        tp1r = c1.number_input("TP1 R", 0.5, 5.0, 1.0, 0.1)
        tp2r = c2.number_input("TP2 R", 0.5, 8.0, 2.0, 0.1)
        tp3r = c3.number_input("TP3 R", 0.5, 10.0, 2.83, 0.01)

        from src.ui import tv_charts
        _tv_engine = tv_charts.engine_toggle("abr")

        st.divider()
        render_sidebar_nav()

    if not chosen:
        st.info("Pick at least one instrument in the sidebar.")
        return

    params = dict(tf_label=tf_label, risk_usd=risk_usd, swing_len=swing_len,
                  max_ob=max_ob, ob_max_age=ob_max_age, sl_buf=sl_buf,
                  tp1r=tp1r, tp2r=tp2r, tp3r=tp3r, tv_engine=_tv_engine)

    tabs = st.tabs(chosen)
    for tab, name in zip(tabs, chosen):
        with tab:
            render_instrument(repo, name, INSTRUMENTS[name], params)


st.set_page_config(page_title="ABR Toolkit", layout="wide")
T.apply()
_repo = pooled_repository(current_db_config()) if st.session_state.get("journal_db_ok") else None
render(_repo)