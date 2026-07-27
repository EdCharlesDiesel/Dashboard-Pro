"""TradingView-style charts (pilot) via TradingView's open-source
`lightweight-charts`, wrapped by the `streamlit-lightweight-charts` component.

This is a **pilot**, currently wired only on the Leading Indicators page
behind an opt-in engine toggle — the terminal-styled Plotly stencil
(`src/ui/charts.py` `ChartKit`) remains the default and the system-wide look.
The point is to evaluate the genuine TradingView candle experience (crosshair
OHLC readout, smooth pan/zoom, price-axis levels) side by side before
committing the whole app to a third-party component.

Design:
- The **builders are pure** — they turn OHLC / indicator frames into the
  component's list-of-charts config (plain dicts), so the data conversion is
  unit-tested without a browser or the component installed.
- Only :func:`render` imports the component; it raises ``ImportError`` cleanly
  when the package isn't present, so the page can fall back to Plotly.
- Colours come from :class:`BloombergTheme` / ``CANDLE_STYLE`` so the pilot
  matches the terminal palette rather than TradingView's default light theme.

Known limitation (honest for a pilot): this wrapper renders each pane as a
separate chart, so the crosshair does **not** sync across the price / RSI /
flow panes the way Plotly's ``hovermode="x unified"`` does within one figure.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.config import CANDLE_STYLE
from src.ui.theme import BloombergTheme as T


# ── time / data conversion (pure) ───────────────────────────────────────────
def _epoch(ts) -> int:
    """UTC epoch seconds for a timestamp — lightweight-charts' UTCTimestamp.

    Handles tz-naive (spine frames) and tz-aware alike; monotonic and unique
    for daily/4H bars, which is all the library requires."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp())


def candle_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for ts, r in df.iterrows():
        if any(pd.isna(r[c]) for c in ("Open", "High", "Low", "Close")):
            continue
        out.append({"time": _epoch(ts), "open": float(r["Open"]),
                    "high": float(r["High"]), "low": float(r["Low"]),
                    "close": float(r["Close"])})
    return out


def line_data(index, series: pd.Series) -> List[Dict[str, Any]]:
    out = []
    for ts, v in zip(index, series):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out.append({"time": _epoch(ts), "value": float(v)})
    return out


def histogram_data(index, series: pd.Series, pos: str = T.GREEN,
                   neg: str = T.RED) -> List[Dict[str, Any]]:
    out = []
    for ts, v in zip(index, series):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out.append({"time": _epoch(ts), "value": float(v),
                    "color": pos if v >= 0 else neg})
    return out


def _flat(times: Sequence, level: float) -> List[Dict[str, Any]]:
    """A horizontal reference line: value ``level`` at the first and last time."""
    if len(times) < 2:
        return []
    return [{"time": _epoch(times[0]), "value": float(level)},
            {"time": _epoch(times[-1]), "value": float(level)}]


# ── general composable series builders (pure) ───────────────────────────────
# lightweight-charts lineStyle enum: 0 solid, 1 dotted, 2 dashed, 3 large-dash,
# 4 sparse-dotted. Named so page code reads like the Plotly `line_dash` it
# replaces.
_LINE_STYLE = {None: 0, "solid": 0, "dot": 1, "dotted": 1, "dash": 2,
               "dashed": 2, "longdash": 3, "sparse": 4}


def marker(time, position: str, color: str, shape: str, text: str = "") -> Dict[str, Any]:
    """One series marker (swing point, phase, divergence). ``position`` ∈
    aboveBar|belowBar|inBar; ``shape`` ∈ arrowUp|arrowDown|circle|square."""
    return {"time": _epoch(time), "position": position, "color": color,
            "shape": shape, "text": text}


def candle_series(df: pd.DataFrame,
                  markers: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    s: Dict[str, Any] = {"type": "Candlestick", "data": candle_data(df),
                         "options": _candle_opts()}
    if markers:
        s["markers"] = markers
    return s


def line_series(index, series: pd.Series, color: str, title: str = "",
                width: float = 1.4, dash: Optional[str] = None,
                last_value: bool = True) -> Dict[str, Any]:
    return {"type": "Line", "data": line_data(index, series),
            "options": {"color": color, "lineWidth": width,
                        "lineStyle": _LINE_STYLE.get(dash, 0), "title": title,
                        "lastValueVisible": last_value, "priceLineVisible": False,
                        "crosshairMarkerVisible": False}}


def level_series(times: Sequence, level: float, color: str, title: str = "",
                 dash: str = "dashed", last_value: bool = True) -> Dict[str, Any]:
    """A horizontal level as a flat two-point Line (this component has no
    declarative price lines)."""
    return {"type": "Line", "data": _flat(times, level),
            "options": {"color": color, "lineWidth": 1,
                        "lineStyle": _LINE_STYLE.get(dash, 2), "title": title,
                        "lastValueVisible": last_value, "priceLineVisible": False,
                        "crosshairMarkerVisible": False}}


def histogram_series(index, series: pd.Series, title: str = "",
                     pos: str = T.GREEN, neg: str = T.RED) -> Dict[str, Any]:
    return {"type": "Histogram", "data": histogram_data(index, series, pos, neg),
            "options": {"title": title}}


def price_chart(series: List[Dict[str, Any]], height: int = 420) -> Dict[str, Any]:
    """One themed pane holding an arbitrary list of series."""
    return {"chart": _chart_opts(height), "series": series}


# ── themed chart options (pure) ─────────────────────────────────────────────
def _chart_opts(height: int, right_pad: bool = False) -> Dict[str, Any]:
    return {
        "height": height,
        "layout": {"background": {"type": "solid", "color": T.BG},
                   "textColor": T.WHITE, "fontFamily": "JetBrains Mono, monospace"},
        "grid": {"vertLines": {"color": T.BORDER},
                 "horzLines": {"color": T.BORDER}},
        "crosshair": {"mode": 0},   # free crosshair (TradingView default feel)
        "rightPriceScale": {"borderColor": T.BORDER},
        "timeScale": {"borderColor": T.BORDER, "timeVisible": True,
                      "secondsVisible": False},
    }


def _candle_opts() -> Dict[str, Any]:
    return {
        "upColor": CANDLE_STYLE["increasing_fillcolor"],
        "downColor": CANDLE_STYLE["decreasing_fillcolor"],
        "borderUpColor": CANDLE_STYLE["increasing_line_color"],
        "borderDownColor": CANDLE_STYLE["decreasing_line_color"],
        "wickUpColor": CANDLE_STYLE["increasing_line_color"],
        "wickDownColor": CANDLE_STYLE["decreasing_line_color"],
    }


# ── the Leading-Indicators chart set (pure) ─────────────────────────────────
def leading_charts(
    df: pd.DataFrame,
    pivots: Dict[str, float],
    pivot_colors: Dict[str, str],
    rsi: pd.Series,
    flow: pd.Series,
    rsi_period: int = 14,
    direction: Optional[str] = None,
    at_level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Full `renderLightweightCharts` config: price+pivots, RSI, flow.

    Pure — returns plain dicts, so it's testable without the component."""
    times = list(df.index)

    # ── price pane: candles + each pivot as a flat dashed line ──
    price_series: List[Dict[str, Any]] = [
        {"type": "Candlestick", "data": candle_data(df), "options": _candle_opts()},
    ]
    for name, level in pivots.items():
        price_series.append({
            "type": "Line",
            "data": _flat(times, level),
            "options": {"color": pivot_colors.get(name, T.GREY), "lineWidth": 1,
                        "lineStyle": 2, "priceLineVisible": False,
                        "lastValueVisible": True, "title": name,
                        "crosshairMarkerVisible": False},
        })
    # divergence-at-pivot marker on the last bar
    if direction in ("Long", "Short") and at_level:
        last = df.iloc[-1]
        up = direction == "Long"
        price_series[0]["markers"] = [{
            "time": _epoch(df.index[-1]),
            "position": "belowBar" if up else "aboveBar",
            "color": T.GREEN if up else T.RED,
            "shape": "arrowUp" if up else "arrowDown",
            "text": f"DIV {at_level}",
        }]

    price_chart = {"chart": _chart_opts(400), "series": price_series}

    # ── RSI pane: line + 70/30 references ──
    rsi_chart = {"chart": _chart_opts(150), "series": [
        {"type": "Line", "data": line_data(df.index, rsi),
         "options": {"color": T.CYAN, "lineWidth": 2, "title": f"RSI {rsi_period}"}},
        {"type": "Line", "data": _flat(times, 70),
         "options": {"color": T.RED, "lineWidth": 1, "lineStyle": 2,
                     "lastValueVisible": False, "priceLineVisible": False}},
        {"type": "Line", "data": _flat(times, 30),
         "options": {"color": T.GREEN, "lineWidth": 1, "lineStyle": 2,
                     "lastValueVisible": False, "priceLineVisible": False}},
    ]}

    # ── flow pane: signed histogram ──
    flow_chart = {"chart": _chart_opts(150), "series": [
        {"type": "Histogram", "data": histogram_data(df.index, flow),
         "options": {"title": "Flow"}},
    ]}

    return [price_chart, rsi_chart, flow_chart]


# ── impure render (only place the component is imported) ─────────────────────
def available() -> bool:
    """True when the lightweight-charts component can be imported."""
    try:
        import streamlit_lightweight_charts  # noqa: F401
        return True
    except Exception:
        return False


def render(charts: List[Dict[str, Any]], key: str = "tv") -> None:
    """Render a list-of-charts config. Raises ImportError if the component
    isn't installed — callers catch it and fall back to Plotly."""
    from streamlit_lightweight_charts import renderLightweightCharts
    renderLightweightCharts(charts, key=key)


def engine_toggle(key: str, label: str = "Chart engine") -> bool:
    """Shared sidebar radio for the TradingView renderer. Returns True when the
    user picked TradingView **and** the component is importable; otherwise
    False (use the Plotly stencil).

    TradingView is the **default** when the component is installed; Terminal
    (Plotly) remains the fallback on every render (each call site wraps the tv
    render in try/except) and stays the default when the component is missing,
    so the radio never opens on a choice that immediately warns."""
    import streamlit as st
    ok = available()
    with st.sidebar:
        st.markdown(f"**📊 {label}**")
        choice = st.radio(
            label, ["Terminal (Plotly)", "TradingView (pilot)"],
            index=1 if ok else 0, key=f"engine_{key}",
            label_visibility="collapsed",
            help=("TradingView pilot uses the open-source lightweight-charts "
                  "engine (crosshair OHLC readout, smooth pan/zoom). Terminal "
                  "is the system-wide Plotly look." if ok else
                  "Install streamlit-lightweight-charts to enable the "
                  "TradingView pilot."))
        if choice.startswith("TradingView") and not ok:
            st.caption("⚠️ `streamlit-lightweight-charts` not installed — using Terminal.")
            return False
    return choice.startswith("TradingView")
