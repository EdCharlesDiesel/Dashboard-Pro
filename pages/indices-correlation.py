"""S&P 500 vs Dow Jones — index analysis & correlation (TradingView pilot).

Two US equity benchmarks side by side: candles for each, their rolling
return-correlation, and the relative-strength ratio (S&P/Dow). The indices
aren't in the forex registry, so they're fetched by their Yahoo tickers
(^GSPC / ^DJI) through the canonical spine. Correlation maths lives in
src/services/index_analysis.py (pure, unit-tested). Dual chart engine — the
TradingView (lightweight-charts) pilot with the terminal Plotly stencil as the
default and fallback, same opt-in toggle as the other price-action pages.

Educational cross-asset context (risk-on/risk-off backdrop), not a trade
signal — audit-only to tool_usage_log.
"""
import streamlit as st

from src.ui.theme import BloombergTheme
from src.ui.theme import BloombergTheme as T
from src.pages_lib.navigation import render_sidebar_nav

st.set_page_config(
    page_title="Indices Correlation · Trading System",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

import pandas as pd

from src.services.market_data import daily_ohlc, asof_caption
from src.services import index_analysis as ia
from src.ui import tv_charts
from src.ui.components import MetricCell, render_metric_row

# Cash indices (not registry pairs) — fetched by Yahoo ticker through the spine.
INDICES = {"S&P 500": "^GSPC", "Dow Jones": "^DJI"}

with st.sidebar:
    st.markdown("### 🇺🇸 Indices Correlation")
    period = st.selectbox("History", ["6mo", "1y", "2y", "5y"], index=1)
    win_short = st.slider("Rolling corr — short (days)", 10, 60, 20, step=5)
    win_long = st.slider("Rolling corr — long (days)", 40, 200, 60, step=10)
    st.divider()
    render_sidebar_nav()

_tv_engine = tv_charts.engine_toggle("indices")

st.markdown("## 🇺🇸 S&P 500 vs Dow Jones — Analysis & Correlation")

with st.spinner("Fetching index history…"):
    spx = daily_ohlc(INDICES["S&P 500"], period=period)
    dji = daily_ohlc(INDICES["Dow Jones"], period=period)

st.caption(
    "Two US equity benchmarks: their candles, how tightly their daily returns "
    "move together (rolling correlation), and which is leading (the S&P/Dow "
    "ratio). Cross-asset risk-on/risk-off context — a decoupling (corr falling "
    "from its usual ~0.95) or a ratio breakout is the signal to notice. Not a "
    "trade trigger. · " + asof_caption(spx, dji))

if spx is None or spx.empty or dji is None or dji.empty:
    st.error("Could not load index data (Yahoo unreachable). Try again shortly.")
    st.stop()

# ── Correlation headline ────────────────────────────────────────────────────
summary = ia.correlation_summary(spx["Close"], dji["Close"],
                                 windows=(int(win_short), int(win_long)))


def _corr_txt(v):
    return f"{v:+.3f}" if v is not None else "—"


def _corr_color(v):
    if v is None:
        return "white"
    return "green" if v >= 0.85 else "yellow" if v >= 0.5 else "red"


cells = [
    MetricCell(f"Corr {win_short}d", _corr_txt(summary.get(f"w{win_short}")),
               color=_corr_color(summary.get(f"w{win_short}"))),
    MetricCell(f"Corr {win_long}d", _corr_txt(summary.get(f"w{win_long}")),
               color=_corr_color(summary.get(f"w{win_long}"))),
    MetricCell("Corr (full)", _corr_txt(summary.get("full")),
               color=_corr_color(summary.get("full"))),
    MetricCell("β (S&P vs Dow)",
               f"{summary['beta']:.2f}" if summary.get("beta") is not None else "—",
               color="cyan"),
    MetricCell("Overlap (days)", str(summary.get("n", 0)), color="white"),
]
render_metric_row(cells)

_short_c = summary.get(f"w{win_short}")
if _short_c is not None and _short_c < 0.5:
    st.warning(f"⚠️ {win_short}-day correlation has fallen to {_short_c:+.2f} — "
               "the two indices are decoupling (unusual; watch for a rotation "
               "or a single-index shock).")

# ── Data prepped for both engines ───────────────────────────────────────────
spx_v, dji_v = spx.tail(250), dji.tail(250)
rc_short = ia.rolling_correlation(spx["Close"], dji["Close"], int(win_short))
rc_long = ia.rolling_correlation(spx["Close"], dji["Close"], int(win_long))
ratio_s = ia.ratio(spx["Close"], dji["Close"])

_used_tv = False
if _tv_engine:
    try:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**S&P 500 (^GSPC)**")
            tv_charts.render([tv_charts.price_chart(
                [tv_charts.candle_series(spx_v)], height=340)], key="tv_spx")
        with c2:
            st.markdown("**Dow Jones (^DJI)**")
            tv_charts.render([tv_charts.price_chart(
                [tv_charts.candle_series(dji_v)], height=340)], key="tv_dji")

        st.markdown("**Rolling return-correlation**")
        corr_series = [
            tv_charts.line_series(rc_short.index, rc_short, T.CYAN,
                                  f"{win_short}d", width=2),
            tv_charts.line_series(rc_long.index, rc_long, T.AMBER,
                                  f"{win_long}d", width=2),
            tv_charts.level_series(list(rc_short.dropna().index), 0.0, T.BORDER,
                                   "0", dash="dot", last_value=False),
        ]
        tv_charts.render([tv_charts.price_chart(corr_series, height=200)],
                         key="tv_corr")

        st.markdown("**Relative strength — S&P / Dow ratio**")
        tv_charts.render([tv_charts.price_chart(
            [tv_charts.line_series(ratio_s.index, ratio_s, T.GREEN, "S&P/Dow",
                                   width=2)], height=200)], key="tv_ratio")
        st.caption("🅣 TradingView pilot (lightweight-charts). Crosshairs are "
                   "per-pane (not synced) — a known limit vs the Terminal view.")
        _used_tv = True
    except Exception as exc:  # noqa: BLE001
        st.warning(f"TradingView render failed ({exc}); showing Terminal charts.")

if not _used_tv:
    from src.ui.charts import ChartKit
    # Two candle panes + correlation + ratio, one stacked terminal figure.
    fig = ChartKit.panels(
        ["S&P 500 (^GSPC)", "Dow Jones (^DJI)",
         f"Rolling return-correlation ({win_short}d / {win_long}d)",
         "Relative strength — S&P / Dow ratio"],
        heights=[0.32, 0.32, 0.18, 0.18],
    )
    ChartKit.candles(fig, spx_v, row=1)
    ChartKit.candles(fig, dji_v, row=2)
    ChartKit.line(fig, rc_short.index, rc_short, f"{win_short}d", color=T.CYAN, row=3)
    ChartKit.line(fig, rc_long.index, rc_long, f"{win_long}d", color=T.AMBER, row=3)
    ChartKit.hline(fig, 0, T.BORDER, row=3)
    ChartKit.line(fig, ratio_s.index, ratio_s, "S&P/Dow", color=T.GREEN, row=4)
    st.plotly_chart(ChartKit.finish(fig, height=820, showlegend=True),
                    width="stretch", config=ChartKit.PLOTLY_CONFIG)

# ── Audit log (research context, not a trade_setups signal) ─────────────────
try:
    from src.services.alert_service import NotifyCache
    from src.services.tool_log import log_tool_usage
    _key = f"{spx.index[-1].date()}|{win_short}|{win_long}"
    _cache = NotifyCache("indices_corr_log")
    if _key not in _cache.load():
        if log_tool_usage("indices_corr", {
            "date": str(spx.index[-1].date()),
            f"corr_{win_short}d": summary.get(f"w{win_short}"),
            f"corr_{win_long}d": summary.get(f"w{win_long}"),
            "corr_full": summary.get("full"), "beta": summary.get("beta"),
            "n": summary.get("n"),
        }):
            _cache.add([_key])
except Exception:
    pass
