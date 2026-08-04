"""
COT Composite Trade Signal
---------------------------
Combines everything built so far (percentile/z-score extremes, price/positioning
divergence, and open-interest context) into one scored signal with discrete
states: STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL, and a separate
"COLLAPSE_WATCH" flag for the "everything unwinds fast" scenario.

IMPORTANT — read before trusting this live:
This is a rules-based heuristic, not a validated trading edge. COT data is
weekly, lagged, and covers speculative futures positioning only -- one input
among many. Treat this the way you've treated your Z(R,T) module: walk-forward
backtest it against your own price history per instrument before trusting it,
and expect false signals, especially around extremes that keep extending
(crowded trades can stay crowded a long time before they resolve).
This is not financial advice, and Claude is not a licensed financial advisor --
this is a decision-support tool you should validate empirically, consistent
with how you've approached every other module in this dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:  # keep this module importable for unit tests without streamlit
    st = None


@dataclass
class TradeSignal:
    state: str                 # STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL
    score: float                # -3 to +3 composite score
    collapse_watch: bool         # True if rapid unwind conditions are present
    headline: str
    reasoning: list             # list of contributing factor strings
    confidence: str              # "low" | "medium" | "high" -- based on signal agreement


def _score_percentile(pct: float) -> float:
    """Extreme low percentile (crowded short) -> contrarian bullish score. Extreme high -> bearish."""
    if np.isnan(pct):
        return 0.0
    if pct <= 5:
        return 1.5
    elif pct <= 15:
        return 1.0
    elif pct >= 95:
        return -1.5
    elif pct >= 85:
        return -1.0
    return 0.0


def _score_zscore(z: float) -> float:
    if np.isnan(z):
        return 0.0
    if z <= -2.0:
        return 1.0
    elif z <= -1.5:
        return 0.5
    elif z >= 2.0:
        return -1.0
    elif z >= 1.5:
        return -0.5
    return 0.0


def _score_divergence(divergence_type: str) -> float:
    if divergence_type == "bullish_divergence":
        return 1.0
    elif divergence_type == "bearish_divergence":
        return -1.0
    return 0.0


def _score_oi_context(oi_interpretation: str, position_direction: str) -> float:
    """
    OI context modulates conviction, not direction on its own.
    Liquidation/covering at an extreme = the unwind may already be *in progress*
    (leans toward the contrarian direction firming up).
    Fresh building at an extreme = conviction is still increasing (extreme may
    extend further before it breaks -- dampens the contrarian score).
    """
    if oi_interpretation == "short_covering" and position_direction == "short":
        return 0.5   # shorts already unwinding -- supports bullish contrarian case
    elif oi_interpretation == "long_liquidation" and position_direction == "long":
        return -0.5  # longs already unwinding -- supports bearish contrarian case
    elif oi_interpretation == "new_shorts_building" and position_direction == "short":
        return -0.3  # extreme short still gaining fresh conviction -- less reliable as a bottom signal yet
    elif oi_interpretation == "new_longs_building" and position_direction == "long":
        return 0.3   # extreme long still gaining fresh conviction -- less reliable as a top signal yet (inverted sign since this dampens bearish contrarian)
    return 0.0


def generate_trade_signal(
    percentile: float,
    zscore: float,
    divergence_type: str,             # "bullish_divergence" | "bearish_divergence" | "none"
    wow_change: float,
    latest_position: float,
    oi_interpretation: str | None = None,   # from oi_trend_context(), optional
    collapse_wow_threshold_std: float = 2.5,
    recent_wow_std: float | None = None,
) -> TradeSignal:
    """
    Combine signals into one composite trade signal.

    collapse_wow_threshold_std / recent_wow_std: if you pass the trailing
    standard deviation of week-over-week changes, a WoW move beyond this many
    std devs flags "collapse_watch" -- rapid, sudden unwinding rather than the
    slow grind extremes usually show. This is the "everything collapses"
    scenario: a violent single-week move out of a crowded position, which is
    a materially different (and riskier/faster) situation than a slow fade.
    """
    reasoning = []
    position_direction = "short" if latest_position < 0 else "long"

    pct_score = _score_percentile(percentile)
    z_score_component = _score_zscore(zscore)
    div_score = _score_divergence(divergence_type)
    oi_score = _score_oi_context(oi_interpretation, position_direction) if oi_interpretation else 0.0

    if pct_score != 0:
        reasoning.append(f"Percentile {percentile:.0f}th contributes {pct_score:+.1f}")
    if z_score_component != 0:
        reasoning.append(f"Z-score {zscore:.2f} contributes {z_score_component:+.1f}")
    if div_score != 0:
        reasoning.append(f"{divergence_type.replace('_', ' ')} contributes {div_score:+.1f}")
    if oi_score != 0:
        reasoning.append(f"OI context ({oi_interpretation}) contributes {oi_score:+.1f}")

    total_score = pct_score + z_score_component + div_score + oi_score
    total_score = max(-3.0, min(3.0, total_score))

    # --- Collapse watch: a violent single-week move rather than a slow fade ---
    collapse_watch = False
    if recent_wow_std and recent_wow_std > 0:
        wow_z = abs(wow_change) / recent_wow_std
        if wow_z >= collapse_wow_threshold_std and abs(percentile - 50) > 30:
            collapse_watch = True
            reasoning.append(
                f"Week-over-week change is {wow_z:.1f}x the recent typical weekly move "
                f"while positioning is already stretched -- this looks like a rapid unwind "
                f"in progress, not a slow fade."
            )

    # --- Map score to discrete state ---
    if total_score >= 2.0:
        state = "STRONG_BUY"
    elif total_score >= 0.8:
        state = "BUY"
    elif total_score <= -2.0:
        state = "STRONG_SELL"
    elif total_score <= -0.8:
        state = "SELL"
    else:
        state = "NEUTRAL"

    # --- Confidence: how many independent factors agree in direction ---
    components = [pct_score, z_score_component, div_score, oi_score]
    nonzero = [c for c in components if c != 0]
    same_sign = all(c > 0 for c in nonzero) or all(c < 0 for c in nonzero)
    if len(nonzero) >= 3 and same_sign:
        confidence = "high"
    elif len(nonzero) >= 2 and same_sign:
        confidence = "medium"
    else:
        confidence = "low"

    if collapse_watch:
        headline = f"⚡ COLLAPSE WATCH — rapid unwind from extreme positioning ({state} bias, {confidence} confidence)"
    elif state == "NEUTRAL":
        headline = "No actionable signal -- positioning within normal range or factors conflict"
    else:
        direction_word = "bullish" if total_score > 0 else "bearish"
        headline = f"{state.replace('_', ' ')} ({direction_word} contrarian signal, {confidence} confidence, score {total_score:+.1f})"

    return TradeSignal(
        state=state,
        score=total_score,
        collapse_watch=collapse_watch,
        headline=headline,
        reasoning=reasoning if reasoning else ["No individual factor crossed its threshold."],
        confidence=confidence,
    )


def render_streamlit_signal(st, signal: TradeSignal):
    """Drop-in Streamlit renderer."""
    if signal.collapse_watch:
        st.error(f"**{signal.headline}**")
    elif signal.state in ("STRONG_BUY", "STRONG_SELL"):
        st.warning(f"**{signal.headline}**")
    elif signal.state in ("BUY", "SELL"):
        st.info(f"**{signal.headline}**")
    else:
        st.caption(signal.headline)

    with st.expander("Why this signal"):
        for r in signal.reasoning:
            st.write(f"- {r}")
        st.caption(
            "This is a rules-based heuristic combining positioning extremes, "
            "divergence, and open interest context -- not a validated trading "
            "edge on its own. Backtest against price history before acting on it."
        )


# Currency -> (registry pair, inverted). "inverted" means the currency is the
# quote leg of a USD/XXX pair, so a bullish read on the currency is a bearish
# read on the pair (e.g. bullish JPY = short USD/JPY). DXY has no tradable
# registry pair, so it's left out and simply never persisted.
CURRENCY_TO_PAIR = {
    "EUR": ("EUR/USD", False),
    "GBP": ("GBP/USD", False),
    "AUD": ("AUD/USD", False),
    "NZD": ("NZD/USD", False),
    "JPY": ("USD/JPY", True),
    "CHF": ("USD/CHF", True),
    "CAD": ("USD/CAD", True),
    "GOLD": ("XAU/USD", False),
}


def _persist_composite_signal(instrument: str, signal: TradeSignal, entry_price: float) -> None:
    """Persist actionable reads to ``trade_setups`` (see ``signal_store``).

    Skips NEUTRAL reads, instruments with no tradable registry pair (DXY), and
    low-confidence reads that aren't also a collapse watch -- this heuristic
    explicitly isn't a validated edge, so only its higher-conviction reads are
    worth surfacing in the journal.
    """
    mapping = CURRENCY_TO_PAIR.get(instrument)
    if mapping is None or signal.state == "NEUTRAL":
        return
    if signal.confidence == "low" and not signal.collapse_watch:
        return

    from src.services.signal_store import persist_signals

    pair, inverted = mapping
    bullish_currency = signal.state in ("STRONG_BUY", "BUY")
    long_pair = (not bullish_currency) if inverted else bullish_currency
    # "cot_composite" (not the module's full name) -- trade_setups.source is
    # VARCHAR(20) and "cot_composite_trade_signal" (27 chars) silently failed
    # every insert until this was caught by an end-to-end AppTest run.
    persist_signals("cot_composite", [{
        "pair": pair,
        "bias": "Long" if long_pair else "Short",
        "entry": entry_price,
        # CFTC reports are weekly with a 3-day lag, so the report date is the only
        # honest freshness stamp — and it keeps one week's reading from being
        # re-persisted every day until the next report lands.
        "bar_time": merged["date"].iloc[-1],
        "conviction": "High" if (signal.confidence == "high" or signal.collapse_watch) else "Medium",
        "thesis": signal.headline,
    }])


if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run pages/cot_composite_trade_signal.py")

    # Page bootstrap — kept lazy so the pure functions above stay importable
    # headlessly (tests, or another page combining this signal with its own
    # data). Mirrors the sibling COT pages' contract: set_page_config ->
    # BloombergTheme.apply() -> sidebar (controls first, then nav).
    from datetime import datetime, timedelta

    from pages.cot_open_interest import fetch_cot_with_oi, oi_trend_context
    from pages.cot_signals import compute_cot_signal
    from src.pages_lib.navigation import render_sidebar_nav
    from src.services.cot_fetcher import INSTRUMENTS as _COT_INSTRUMENTS, get_instrument_series
    from src.ui.theme import BloombergTheme

    st.set_page_config(
        page_title="COT Composite Trade Signal · Trading System",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()

    T = BloombergTheme  # short alias for palette tokens

    # Instrument -> yfinance ticker for the price overlay (mirrors cot_signals.py).
    PRICE_TICKERS = {
        "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "USDJPY=X", "CHF": "USDCHF=X",
        "AUD": "AUDUSD=X", "NZD": "NZDUSD=X", "CAD": "USDCAD=X",
        "GOLD": "GC=F", "DXY": "DX-Y.NYB",
    }
    # Only these instruments are in the TFF report that fetch_cot_with_oi() reads;
    # GOLD/DXY are Legacy-report instruments with no OI feed here (see
    # cot_open_interest.py) -- the composite score just skips that factor for them.
    TFF_MARKET_NAMES = {
        name: market for name, (report_type, market, _spec) in _COT_INSTRUMENTS.items()
        if report_type == "traders_in_financial_futures_fut"
    }

    @st.cache_data(ttl=600, show_spinner=False)
    def _price_series(ticker: str, years_back: int) -> pd.DataFrame:
        """Cached daily close series for the divergence price overlay — the COT
        data itself is fetched/cached separately by `cot_fetcher` (CSV-on-disk)."""
        from src.db.market_cache import cached_ohlc
        try:
            raw = cached_ohlc(ticker, start=datetime.now() - timedelta(days=years_back * 365),
                              interval="1d", ttl=600)
            if raw is None or raw.empty or "Close" not in raw.columns:
                return pd.DataFrame()
            close = raw["Close"].dropna()
            return pd.DataFrame({"date": close.index, "price": close.values})
        except Exception:
            return pd.DataFrame()

    def _signal_chart(merged: pd.DataFrame, instrument: str):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=merged["date"], y=merged["net_position"],
                name="Net Speculative Position",
                marker_color=[T.GREEN if v >= 0 else T.RED for v in merged["net_position"]],
                opacity=0.7,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=merged["date"], y=merged["price"],
                name=f"{instrument} price", line=dict(color=T.CYAN, width=1.5),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Net Contracts", secondary_y=False, showgrid=True,
                         gridcolor=T.BORDER, tickfont=dict(color=T.GREY, size=9))
        fig.update_yaxes(title_text="Price", secondary_y=True, showgrid=False,
                         tickfont=dict(color=T.GREY, size=9))
        fig.update_xaxes(showgrid=False, tickfont=dict(color=T.GREY, size=9), linecolor=T.BORDER)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T.BG_PANEL,
            font=dict(family=T.FONT_MONO, color=T.GREY, size=11),
            legend=dict(font=dict(color=T.GREY, size=11), bgcolor=T.BG_ELEV,
                       bordercolor=T.BORDER, orientation="h", x=0, y=1.08),
            margin=dict(l=10, r=10, t=30, b=10), height=450,
            title=dict(text=f"{instrument} — Composite Signal Inputs",
                      font=dict(color=T.WHITE, size=13)),
        )
        return fig

    st.title("🧩 COT Composite Trade Signal")
    st.caption(
        "Combines positioning extremes (see **18. COT Signals**), price/positioning "
        "divergence, and open-interest context (see **20. COT Open Interest**) into "
        "one scored STRONG_BUY…STRONG_SELL read, plus a collapse-watch flag for a "
        "violent single-week unwind. A rules-based heuristic, not a validated "
        "trading edge — backtest per instrument before acting on it."
    )

    with st.sidebar:
        st.header("Composite Signal Settings")
        instrument = st.selectbox("Instrument", sorted(_COT_INSTRUMENTS.keys()), index=0)
        years_back = st.slider("Years of history", 1, 8, 4)
        lookback_weeks = st.slider("Trend lookback (weeks)", 4, 16, 8)
        refresh = st.button("🔄 Force refresh from CFTC", width="stretch")
        if refresh:
            _price_series.clear()

        st.divider()
        render_sidebar_nav()

    try:
        with st.spinner(f"Loading {instrument} COT data..."):
            series = get_instrument_series(instrument, years_back=years_back, force_refresh=refresh)
    except Exception as e:
        st.error(f"Couldn't load COT data for {instrument}: {e}")
        st.stop()

    price_df = _price_series(PRICE_TICKERS[instrument], years_back)
    if price_df.empty:
        st.warning(f"Couldn't load price data for {instrument} — the composite signal needs both series.")
        st.stop()

    # The two sources carry different datetime *resolutions* — CFTC weekly
    # reports parse to microseconds, yfinance bars to seconds — and pandas 3.x
    # refuses to merge_asof across them ("incompatible merge keys"). Both hold
    # date-level values, so pinning each to nanoseconds is lossless and is a
    # no-op on the pandas 2.2.3 pinned in requirements.txt.
    cot_left = series[["date", "net_spec"]].sort_values("date").copy()
    cot_left["date"] = cot_left["date"].astype("datetime64[ns]")
    price_right = price_df.sort_values("date").copy()
    price_right["date"] = price_right["date"].astype("datetime64[ns]")

    merged = pd.merge_asof(
        cot_left,
        price_right,
        on="date",
        direction="backward",
    ).dropna().rename(columns={"net_spec": "net_position"})

    if len(merged) < 20:
        st.warning("Not enough overlapping history yet for a meaningful signal read.")
        st.stop()

    cot_signal = compute_cot_signal(merged, lookback_weeks=lookback_weeks)

    oi_interpretation: Optional[str] = None
    if instrument in TFF_MARKET_NAMES:
        try:
            oi_df = fetch_cot_with_oi(TFF_MARKET_NAMES[instrument], weeks_back=years_back * 52)
            oi_interpretation = oi_trend_context(oi_df, lookback=lookback_weeks)["interpretation"]
        except Exception as e:
            st.caption(f"Open-interest context unavailable ({e}) — scoring without it.")

    recent_wow_std = merged["net_position"].diff().tail(52).std()

    trade_signal = generate_trade_signal(
        percentile=cot_signal.percentile_3y,
        zscore=cot_signal.zscore,
        divergence_type=cot_signal.divergence_type,
        wow_change=cot_signal.wow_change,
        latest_position=cot_signal.latest_position,
        oi_interpretation=oi_interpretation,
        recent_wow_std=recent_wow_std,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Net Position", f"{cot_signal.latest_position:,.0f}")
    m2.metric("3Y Percentile", f"{cot_signal.percentile_3y:.0f}th")
    m3.metric("Z-score", f"{cot_signal.zscore:.2f}")
    m4.metric("Composite Score", f"{trade_signal.score:+.1f}")

    render_streamlit_signal(st, trade_signal)

    st.plotly_chart(_signal_chart(merged, instrument), width="stretch")

    with st.expander("Merged weekly data (positioning + price)"):
        st.dataframe(
            merged.tail(52).sort_values("date", ascending=False),
            width="stretch",
            hide_index=True,
        )

    _persist_composite_signal(instrument, trade_signal, float(merged["price"].iloc[-1]))
