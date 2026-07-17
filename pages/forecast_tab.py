"""
forecast_tab.py
===============
Forecast page — standalone dashboard page (auto-registered from pages/, see
src/pages_lib/navigation.py's FCST entry). GARCH(1,1) uncertainty cone +
driver-based directional score + narrative ("reasons for the forecast") +
forecast journal with self-scoring.

Design notes (honest-forecasting principles):
  * No point forecast theater. The projection is a volatility cone around a
    random-walk center (Meese-Rogoff: hard to beat RW at these horizons).
  * Direction comes from a transparent driver score, each driver shown.
  * Every forecast is journaled; realized outcomes are scored against the cone
    and against the directional call, so the page grades itself over time.

Narrative: deterministic template by default. If ANTHROPIC_API_KEY is present
in st.secrets or the environment, the template facts are polished by the
Claude API (fails soft back to the template).
"""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from arch import arch_model

from src.core.secrets import anthropic_api_key
from src.db.cache import pooled_repository
from src.db.connection import current_db_config
from src.instruments.registry import INSTRUMENTS as _REGISTRY
from src.pages_lib.navigation import render_sidebar_nav
from src.ui.theme import BloombergTheme as T

# Compact symbol -> yfinance ticker, derived from the shared registry so this
# page's instrument list always covers the full universe (was previously a
# hardcoded 6-symbol subset). Keys are compact ("XAUUSD" not "XAU/USD", "WTI"
# not "WTI/USD") to match abr_signals.symbol — pages/abr_toolkit_tab.py's
# MT5-style naming — so _latest_abr_signal's cross-reference keeps working.
_SYMBOL_OVERRIDES = {"WTI/USD": "WTI"}  # crude isn't quoted as "WTI/USD" by convention
INSTRUMENTS: dict[str, str] = {
    _SYMBOL_OVERRIDES.get(name, name.replace("/", "")): inst.ticker
    for name, inst in _REGISTRY.items()
}
# Compact symbol → registry pair name, for the shared house-view strip.
_COMPACT_TO_REGISTRY: dict[str, str] = {
    _SYMBOL_OVERRIDES.get(name, name.replace("/", "")): name
    for name in _REGISTRY.keys()
}

HORIZONS = {"1 week": 5, "1 month": 21, "1 quarter": 63}


# ============================================================================
# Engine (pure functions — testable without Streamlit)
# ============================================================================
def garch_cone(close: pd.Series, horizon: int) -> dict:
    """Random-walk center with GARCH(1,1)-t vol cone. Returns band arrays."""
    rets = np.log(close / close.shift()).dropna() * 100.0
    rets = rets[np.isfinite(rets)]
    am = arch_model(rets, vol="GARCH", p=1, q=1, dist="t", rescale=False)
    fit = am.fit(disp="off", show_warning=False)
    fc = fit.forecast(horizon=horizon, reindex=False)
    sigma_daily = np.sqrt(fc.variance.values[-1]) / 100.0          # decimal
    cum_sigma = np.sqrt(np.cumsum(sigma_daily ** 2))
    spot = float(close.iloc[-1])
    return {
        "spot": spot,
        "ann_vol": float(np.sqrt(252) * sigma_daily[0]) * 100.0,   # % annualized
        "center": np.full(horizon, spot),
        "hi68": spot * np.exp(1.0 * cum_sigma),
        "lo68": spot * np.exp(-1.0 * cum_sigma),
        "hi95": spot * np.exp(1.96 * cum_sigma),
        "lo95": spot * np.exp(-1.96 * cum_sigma),
    }


def compute_drivers(df: pd.DataFrame, abr_row: dict | None) -> list[dict]:
    """Transparent, price-based drivers (+ ABR structure bias if journaled).
    Each driver: name, value (+1/-1/0), and a human explanation."""
    close = df["Close"]
    drivers: list[dict] = []

    # EMA20/50 — the canonical fast/slow pair (src/core/bias.py), so this
    # driver can never disagree with the house view about the daily trend.
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    v = 1 if ema20 > ema50 else -1
    drivers.append({"name": "Trend (EMA20 vs EMA50)", "value": v,
                    "why": f"EMA20 {'above' if v > 0 else 'below'} EMA50 on the daily"})

    ma200 = close.rolling(200).mean().iloc[-1]
    if not np.isnan(ma200):
        v = 1 if close.iloc[-1] > ma200 else -1
        drivers.append({"name": "Long-term (vs 200d MA)", "value": v,
                        "why": f"price {'above' if v > 0 else 'below'} the 200-day average"})

    mom = close.iloc[-1] / close.iloc[-64] - 1 if len(close) > 64 else 0.0
    v = 1 if mom > 0.01 else -1 if mom < -0.01 else 0
    drivers.append({"name": "3-month momentum", "value": v,
                    "why": f"{mom:+.1%} over ~3 months"})

    hi20 = close.rolling(20).max().iloc[-2]
    lo20 = close.rolling(20).min().iloc[-2]
    v = 1 if close.iloc[-1] > hi20 else -1 if close.iloc[-1] < lo20 else 0
    drivers.append({"name": "20-day breakout", "value": v,
                    "why": "fresh 20-day high" if v > 0 else
                           "fresh 20-day low" if v < 0 else "inside the 20-day range"})

    if abr_row is not None and abr_row.get("signal") in ("BUY", "SELL"):
        v = 1 if abr_row["signal"] == "BUY" else -1
        drivers.append({"name": "ABR structure", "value": v,
                        "why": f"ABR {abr_row['signal']} grade {abr_row.get('grade', '?')} "
                               f"on {abr_row.get('timeframe', '?')}"})
    return drivers


def score_drivers(drivers: list[dict]) -> tuple[int, str]:
    s = sum(d["value"] for d in drivers)
    n = len(drivers)
    if s >= n * 0.6:
        lbl = "bullish"
    elif s >= 2:
        lbl = "mildly bullish"
    elif s <= -n * 0.6:
        lbl = "bearish"
    elif s <= -2:
        lbl = "mildly bearish"
    else:
        lbl = "neutral / mixed"
    return s, lbl


def template_narrative(sym: str, cone: dict, drivers: list[dict],
                       score: int, label: str, horizon_days: int,
                       chg_1m: float) -> str:
    pos = [d for d in drivers if d["value"] > 0]
    neg = [d for d in drivers if d["value"] < 0]
    parts = [
        f"{sym} trades at {cone['spot']:.4g}, {chg_1m:+.1%} over the past month, "
        f"with GARCH-implied annualized volatility of {cone['ann_vol']:.1f}%.",
        f"The driver score reads {label} ({score:+d}/{len(drivers)}).",
    ]
    if pos:
        parts.append("Supportive: " + "; ".join(d["why"] for d in pos) + ".")
    if neg:
        parts.append("Against: " + "; ".join(d["why"] for d in neg) + ".")
    parts.append(
        f"Over the next {horizon_days} trading days the 68% volatility band spans "
        f"{cone['lo68'][-1]:.4g}–{cone['hi68'][-1]:.4g} "
        f"(95%: {cone['lo95'][-1]:.4g}–{cone['hi95'][-1]:.4g}); no point forecast is "
        f"implied beyond this distribution.")
    return " ".join(parts)


def polished_narrative(template: str) -> str | None:
    """Optional Claude API polish. Returns None on any failure (fail soft)."""
    key = anthropic_api_key()
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content":
                "Rewrite this market summary as one flowing analyst paragraph. "
                "Keep every number exactly as given, add no data, no advice, "
                "neutral tone:\n\n" + template}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


# ============================================================================
# Journal (optional — skipped when `repo` is None, i.e. DB not configured)
# ============================================================================
def journal_forecast(repo, sym: str, horizon: int, cone: dict,
                     drivers: list[dict], score: int, label: str) -> None:
    """`repo`: a TradeRepository (see src/db/cache.py's pooled_repository)."""
    try:
        repo.save_forecast(
            sym, horizon, cone["spot"], score, label, json.dumps(drivers),
            float(cone["lo68"][-1]), float(cone["hi68"][-1]),
            float(cone["lo95"][-1]), float(cone["hi95"][-1]),
        )
    except Exception as exc:
        st.caption(f"forecast journal skipped: {exc}")


def evaluate_forecasts(repo, sym: str, close: pd.Series) -> pd.DataFrame:
    """Fill realized prices for matured forecasts; return scored history."""
    try:
        rows = repo.load_forecasts(sym, limit=40)
    except Exception:
        return pd.DataFrame()
    out = []
    idx = close.index.tz_localize(None) if close.index.tz else close.index
    for row in rows:
        row = dict(row)
        matured_at = pd.Timestamp(row["made_at"]).tz_localize(None) \
            + timedelta(days=row["horizon_days"] * 1.45)      # trading->calendar
        if row["realized"] is None and matured_at <= pd.Timestamp.now():
            pos = idx.searchsorted(matured_at)
            if pos < len(close):
                row["realized"] = float(close.iloc[pos])
                try:
                    repo.update_forecast_realized(row["id"], row["realized"])
                except Exception:
                    pass
        if row["realized"] is not None:
            row["in_68"] = row["lo68"] <= row["realized"] <= row["hi68"]
            row["in_95"] = row["lo95"] <= row["realized"] <= row["hi95"]
            move = row["realized"] - row["spot"]
            row["dir_hit"] = (np.sign(move) == np.sign(row["score"])
                              if row["score"] not in (0, None) else None)
        out.append(row)
    return pd.DataFrame(out)


# ============================================================================
# Data + chart
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily(ticker: str) -> pd.DataFrame:
    # Canonical spine (5y kept for the GARCH fit; interval/source are shared).
    from src.services.market_data import daily_ohlc
    return daily_ohlc(ticker, period="5y")


def build_chart(df: pd.DataFrame, cone: dict, horizon: int, sym: str) -> go.Figure:
    view = df.tail(250)
    fig = go.Figure(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"],
        low=view["Low"], close=view["Close"], name=sym, showlegend=False,
        increasing_line_color=T.GREEN, decreasing_line_color=T.RED))
    fut = pd.bdate_range(df.index[-1], periods=horizon + 1)[1:]
    for hi, lo, alpha in (("hi95", "lo95", 0.10), ("hi68", "lo68", 0.18)):
        fig.add_trace(go.Scatter(x=fut, y=cone[hi], mode="lines",
                                 line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fut, y=cone[lo], mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor=f"rgba(0,224,255,{alpha})",
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut, y=cone["center"], mode="lines",
                             line=dict(color=T.GREY, dash="dot", width=1),
                             showlegend=False))
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=25, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=T.FONT_MONO, color=T.WHITE, size=10),
        xaxis=dict(showgrid=True, gridcolor=T.BORDER, tickfont=dict(color=T.GREY)),
        yaxis=dict(showgrid=True, gridcolor=T.BORDER, tickfont=dict(color=T.GREY)))
    return fig


# ============================================================================
# Page
# ============================================================================
def _latest_abr_signal(repo, sym: str) -> dict | None:
    if repo is None:
        return None
    try:
        return repo.latest_abr_signal(sym)
    except Exception:
        return None


def render_instrument(repo, sym: str, ticker: str, horizon_lbl: str) -> None:
    df = fetch_daily(ticker)
    if df.empty or len(df) < 260:
        st.warning(f"Not enough daily data for {ticker}.")
        return
    horizon = HORIZONS[horizon_lbl]
    close = df["Close"]

    cone = garch_cone(close, horizon)
    drivers = compute_drivers(df, _latest_abr_signal(repo, sym))
    score, label = score_drivers(drivers)
    chg_1m = float(close.iloc[-1] / close.iloc[-22] - 1)

    # Shared house view — flags when this page's driver score opposes the
    # canonical Weekly/Daily/4H read.
    registry_pair = _COMPACT_TO_REGISTRY.get(sym)
    if registry_pair:
        from src.services.bias_service import show_house_view
        _read = ("Bullish" if "bullish" in label else
                 "Bearish" if "bearish" in label else None)
        show_house_view(registry_pair, page_read=_read, page_label="Forecast")

    st.plotly_chart(build_chart(df, cone, horizon, sym),
                    use_container_width=True, key=f"fc_{sym}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"{cone['spot']:.4g}", f"{chg_1m:+.1%} / 1m")
    c2.metric("Direction score", f"{score:+d}/{len(drivers)}", label)
    c3.metric("GARCH ann. vol", f"{cone['ann_vol']:.1f}%")
    c4.metric(f"68% band ({horizon_lbl})",
              f"{cone['lo68'][-1]:.4g} – {cone['hi68'][-1]:.4g}")

    st.markdown("##### Reasons")
    for d in drivers:
        icon = "🟢" if d["value"] > 0 else "🔴" if d["value"] < 0 else "⚪"
        st.markdown(f"{icon} **{d['name']}** — {d['why']}")

    st.markdown("##### Outlook")
    base = template_narrative(sym, cone, drivers, score, label, horizon, chg_1m)
    polished = polished_narrative(base)
    st.write(polished or base)
    if polished:
        st.caption("narrative polished via Claude API")

    if repo is not None:
        journal_forecast(repo, sym, horizon, cone, drivers, score, label)
        hist = evaluate_forecasts(repo, sym, close)
        if not hist.empty:
            with st.expander("Forecast track record"):
                scored = hist.dropna(subset=["realized"])
                if not scored.empty:
                    hit = scored["dir_hit"].dropna()
                    st.markdown(
                        f"Matured: **{len(scored)}** · in 68% band: "
                        f"**{scored['in_68'].mean():.0%}** (target ≈ 68%) · "
                        f"in 95% band: **{scored['in_95'].mean():.0%}** · "
                        f"direction hit-rate: "
                        f"**{hit.mean():.0%}** of {len(hit)} calls "
                        f"(coin flip = 50%)")
                st.dataframe(hist[["made_at", "horizon_days", "spot", "score",
                                   "label", "lo68", "hi68", "realized"]],
                             use_container_width=True, hide_index=True)


def render(repo=None) -> None:
    st.title("Forecast")
    st.caption("GARCH volatility cone · transparent driver score · self-scoring journal")

    with st.sidebar:
        st.subheader("Forecast settings")
        chosen = st.multiselect("Instruments", list(INSTRUMENTS.keys()),
                                default=["EURUSD", "XAUUSD"])
        horizon_keys = list(HORIZONS.keys())
        horizon_lbl = st.selectbox("Horizon", horizon_keys,
                                   index=horizon_keys.index("1 quarter"))

        st.divider()
        render_sidebar_nav()

    if not chosen:
        st.info("Pick at least one instrument.")
        return

    tabs = st.tabs(chosen)
    for tab, sym in zip(tabs, chosen):
        with tab:
            render_instrument(repo, sym, INSTRUMENTS[sym], horizon_lbl)


st.set_page_config(page_title="Forecast", layout="wide")
T.apply()
_repo = pooled_repository(current_db_config()) if st.session_state.get("journal_db_ok") else None
render(_repo)