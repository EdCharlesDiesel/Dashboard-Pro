"""Instrument Predictor — one composite directional read per instrument.

Gathers four of the dashboard's existing scorers for a single chosen
instrument — the 10-point Setup Score, the 6-condition Trend Signal, the
Currency Strength differential, and the COT Composite read (majors + gold
only) — normalizes each to [-1, +1], and combines them via
``src/services/prediction_service.aggregate()`` into one weighted composite
score, a STRONG_SELL…STRONG_BUY label, and a confidence level. A GARCH vol
regime read (best-effort; needs the optional ``arch`` package) caps
confidence in choppy conditions rather than voting on direction.

IMPORTANT: this is a heuristic combination of the dashboard's own existing
tools, not a new statistically validated model — same caveat as
``cot_composite_trade_signal``: backtest before sizing conviction on it.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytz
import streamlit as st

from src.core.signals import score_setup
from src.indicators.trend_signal import TrendSignalEvaluator
from src.instruments.registry import INSTRUMENTS, TYPICAL_SPREADS
from src.pages_lib.base import BloombergPage, PageContext
from src.services.prediction_service import Component, Prediction, aggregate
from src.services.signal_store import persist_signals
from src.ui.components import CommandBar, MetricCell, Panel, render_metric_row

_WEIGHTS = {
    "Setup Score": 0.30,
    "Trend Signal": 0.25,
    "Currency Strength": 0.25,
    "COT Composite": 0.20,
}

_LABEL_DISPLAY = {
    "STRONG_BUY": ("🚀 STRONG BUY", "green"),
    "BUY": ("📈 BUY", "green"),
    "NEUTRAL": ("⏳ NEUTRAL", "amber"),
    "SELL": ("📉 SELL", "red"),
    "STRONG_SELL": ("🔻 STRONG SELL", "red"),
}

_CONFIDENCE_COLOR = {"High": "green", "Medium": "amber", "Low": "red"}

# Currency -> (registry pair, inverted) for the COT Composite component —
# same mapping cot_composite_trade_signal.py persists with; only majors +
# gold have both a CFTC series and a tradable registry pair.
_COT_CCY_FOR_PAIR = {
    "EUR/USD": ("EUR", False), "GBP/USD": ("GBP", False),
    "AUD/USD": ("AUD", False), "NZD/USD": ("NZD", False),
    "USD/JPY": ("JPY", True), "USD/CHF": ("CHF", True),
    "USD/CAD": ("CAD", True), "XAU/USD": ("GOLD", False),
}


# ------------------------------------------------------------------ data --
@st.cache_data(ttl=300, show_spinner=False)
def _daily(ticker: str, days: int = 400) -> pd.DataFrame:
    from src.db.market_cache import cached_ohlc
    try:
        df = cached_ohlc(ticker, period=f"{days}d", interval="1d", ttl=300)
        if df.empty:
            return pd.DataFrame()
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def _four_hour(ticker: str) -> pd.DataFrame:
    from src.db.market_cache import cached_ohlc
    try:
        end = datetime.now(pytz.utc)
        start = end - timedelta(days=90)
        df = cached_ohlc(ticker, start=start, end=end, interval="1h", ttl=300)
        if df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df.resample("4h").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def _weekly(ticker: str) -> pd.DataFrame:
    from src.db.market_cache import cached_ohlc
    try:
        df = cached_ohlc(ticker, period="2y", interval="1d", ttl=300)
        if df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df.resample("W").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()
    except Exception:
        return pd.DataFrame()


# ------------------------------------------------------------ components --
def _setup_score_component(pair: str, ticker: str, pip_size: float) -> Component:
    df_w, df_d, df_4h = _weekly(ticker), _daily(ticker), _four_hour(ticker)
    if df_d.empty:
        return Component("Setup Score", 0.0, 0.0, "No price data")
    spread = TYPICAL_SPREADS.get(pair, 0.0)
    long_res = score_setup(df_w, df_d, df_4h, "LONG", pip_size, spread)
    short_res = score_setup(df_w, df_d, df_4h, "SHORT", pip_size, spread)
    if long_res["score"] >= short_res["score"]:
        best, sign = long_res, 1
    else:
        best, sign = short_res, -1
    norm = sign * (best["score"] / best["max_score"])
    detail = f"{best['direction']} {best['score']}/{best['max_score']} (grade {best['grade']})"
    return Component("Setup Score", norm, _WEIGHTS["Setup Score"], detail)


def _trend_signal_component(ticker: str) -> Component:
    df = TrendSignalEvaluator.fetch(ticker, interval="1d", period="2y", resample=None)
    if df is None or df.empty:
        return Component("Trend Signal", 0.0, 0.0, "No price data")
    sig = TrendSignalEvaluator.evaluate(df)
    scale = {"STRONG_BUY": 1.0, "BUY": 0.5, "NEUTRAL": 0.0,
             "SELL": -0.5, "STRONG_SELL": -1.0}[sig.direction]
    return Component("Trend Signal", scale, _WEIGHTS["Trend Signal"],
                     f"{sig.label} ({sig.score}/{sig.max_score})")


def _currency_strength_component(pair: str) -> Component:
    if INSTRUMENTS[pair].instrument.is_commodity:
        return Component("Currency Strength", 0.0, 0.0, "N/A for commodities")
    from src.pages_lib.currency_strength import _currency_returns, _fetch_pair_closes
    closes = _fetch_pair_closes()
    if closes.empty:
        return Component("Currency Strength", 0.0, 0.0, "No data")
    strength = _currency_returns(closes)
    base, quote = pair.split("/")
    base_s = strength.get(base, {}).get(20, 0.0)
    quote_s = strength.get(quote, {}).get(20, 0.0)
    diff = base_s - quote_s
    norm = max(-1.0, min(1.0, diff / 2.0))
    detail = f"{base} {base_s:+.2f}% vs {quote} {quote_s:+.2f}% (20d avg)"
    return Component("Currency Strength", norm, _WEIGHTS["Currency Strength"], detail)


def _cot_composite_component(pair: str) -> Component:
    mapping = _COT_CCY_FOR_PAIR.get(pair)
    if mapping is None:
        return Component("COT Composite", 0.0, 0.0, "No COT mapping for this pair")
    ccy, inverted = mapping
    try:
        from pages.cot_composite_trade_signal import generate_trade_signal
        from src.services.cot_fetcher import compute_extremeness, get_instrument_series
        series = get_instrument_series(ccy)
        ext = compute_extremeness(series)
        if "error" in ext:
            return Component("COT Composite", 0.0, 0.0, ext["error"])
        wow = float(series["net_change_wow"].iloc[-1])
        latest_pos = float(series["net_spec"].iloc[-1])
        sig = generate_trade_signal(ext["percentile"], ext["z_score"], "none",
                                    wow, latest_pos)
        bullish_ccy = sig.state in ("STRONG_BUY", "BUY")
        bullish_pair = (not bullish_ccy) if inverted else bullish_ccy
        norm = (sig.score / 3.0) * (-1 if inverted else 1)
        norm = max(-1.0, min(1.0, norm))
        detail = (f"{ccy} {sig.state} ({sig.confidence} conf) -> pair "
                 f"{'bullish' if bullish_pair else 'bearish'}")
        return Component("COT Composite", norm, _WEIGHTS["COT Composite"], detail)
    except Exception as exc:
        return Component("COT Composite", 0.0, 0.0, f"Unavailable ({exc})")


def _vol_regime(ticker: str) -> Optional[str]:
    """Best-effort GARCH vol regime (needs the optional `arch` package and
    ~1y of daily returns) — a confidence modifier, not a directional vote."""
    try:
        from src.core.quant_models import fit_garch
        px = _daily(ticker)["Close"]
        rets = px.pct_change(fill_method=None).dropna()
        return fit_garch(rets)["regime"]
    except Exception:
        return None


def gather_prediction(pair: str) -> Prediction:
    """Build every component for ``pair`` and return the aggregated Prediction."""
    ticker = INSTRUMENTS[pair]["ticker"]
    pip_size = INSTRUMENTS[pair]["pip_size"]
    components = [
        _setup_score_component(pair, ticker, pip_size),
        _trend_signal_component(ticker),
        _currency_strength_component(pair),
        _cot_composite_component(pair),
    ]
    vol_regime = _vol_regime(ticker)
    return aggregate(components, vol_regime=vol_regime)


# ------------------------------------------------------------------- page --
class InstrumentPredictorPage(BloombergPage):
    """Composite directional-read page for a single chosen instrument."""

    def configure(self) -> PageContext:
        st.session_state.setdefault("pred_pair", "EUR/USD")
        return PageContext(code="PRDX", title="Instrument Predictor", icon="🔮")

    def sidebar(self, ctx: PageContext) -> None:
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;">PREDICTOR SETTINGS</div>',
            unsafe_allow_html=True,
        )
        names = sorted(INSTRUMENTS.keys())  # alphabetical dropdown
        st.session_state.pred_pair = st.selectbox(
            "Instrument", names,
            index=names.index(st.session_state.pred_pair)
            if st.session_state.pred_pair in names else 0,
        )
        st.caption(
            "Combines Setup Score, Trend Signal, Currency Strength, and (for "
            "majors + gold) COT Composite into one weighted read. GARCH vol "
            "regime caps confidence in choppy conditions."
        )
        st.divider()
        run = st.button("🔮 Run Prediction", width="stretch", type="primary")
        st.session_state.pred_run = run or st.session_state.get("pred_run", False)

    def body(self, ctx: PageContext) -> None:
        pair = st.session_state.pred_pair
        if not st.session_state.get("pred_run"):
            st.info("Choose an instrument in the sidebar and click **Run Prediction**.")
            return

        with st.spinner(f"Gathering signals for {pair}..."):
            pred = gather_prediction(pair)

        label_disp, label_color = _LABEL_DISPLAY[pred.label]
        conf_color = _CONFIDENCE_COLOR[pred.confidence]

        # Shared house view — flags when this composite second opinion opposes
        # the canonical Weekly/Daily/4H read.
        from src.services.bias_service import show_house_view
        show_house_view(pair, page_read=pred.label,
                        page_label="Instrument Predictor")

        CommandBar(label="PRDX", cells=[
            (f"PAIR {pair}", ""),
            (f"READ {label_disp}", label_color),
            (f"CONF {pred.confidence.upper()}", conf_color),
        ]).show()

        render_metric_row([
            MetricCell("Composite Score", f"{pred.composite:+.2f}", label_color),
            MetricCell("Direction", label_disp, label_color),
            MetricCell("Confidence", pred.confidence, conf_color),
            MetricCell("Agreement", f"{pred.agreement * 100:.0f}%", "cyan"),
            MetricCell("Vol Regime", (pred.vol_regime or "n/a").upper(),
                      "red" if pred.vol_regime == "high" else "white"),
        ])

        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:8px 4px;border-bottom:1px solid #2a2a2a;">'
            f'<div><strong>{c.name}</strong> (weight {c.weight:.0%})'
            f'<div style="font-size:11px;color:#9a9a9a;">{c.detail}</div></div>'
            f'<div style="font-weight:700;color:{"#00ff66" if c.score > 0 else "#ff3344" if c.score < 0 else "#9a9a9a"};">'
            f'{c.score:+.2f}</div></div>'
            for c in pred.components
        )
        Panel("COMPONENT BREAKDOWN", tag=f"{len(pred.components)} of {len(_WEIGHTS)} available",
              body_html=rows or "<div>No components could be computed.</div>").show()

        st.caption(
            "Heuristic combination of the dashboard's own existing scorers — not "
            "a new validated model. A missing component (e.g. COT has no "
            "mapping for this pair) is left out and the remaining weights are "
            "renormalized, so it never silently drags the score toward zero. "
            "Backtest before sizing conviction on it, same as COT Composite."
        )

        if pred.label != "NEUTRAL":
            self._persist(pair, pred)

    @staticmethod
    def _persist(pair: str, pred: Prediction) -> None:
        ticker = INSTRUMENTS[pair]["ticker"]
        closes = _daily(ticker)["Close"] if not _daily(ticker).empty else None
        entry = float(closes.iloc[-1]) if closes is not None and not closes.empty else None
        bias = "Long" if pred.composite > 0 else "Short"
        persist_signals("instrument_predictor", [{
            "pair": pair,
            "bias": bias,
            "entry": entry,
            "bar_time": (closes.index[-1] if closes is not None and not closes.empty
                         else None),
            "strength_score": round(pred.composite * 10, 1),
            "conviction": pred.confidence,
            "thesis": (f"Instrument Predictor — {pred.label} composite "
                      f"{pred.composite:+.2f}, {pred.agreement * 100:.0f}% agreement "
                      f"across {len(pred.components)} components."),
        }])
