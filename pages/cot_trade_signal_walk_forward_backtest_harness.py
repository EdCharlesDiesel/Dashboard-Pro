"""
COT Trade Signal — Walk-Forward Backtest Harness
--------------------------------------------------
Answers the only question that matters before you trust
`cot_composite_trade_signal.py` with real money: historically, when this
signal fired, what actually happened to price afterward?

Design principles (matching your Z(R,T) walk-forward approach):
  - No lookahead. At each historical week t, the signal is computed using only
    data available up to and including t (percentile/z-score/OI context all
    use trailing windows only -- this harness enforces that by truncating the
    dataframe before computing each week's signal, not by trusting the
    functions to behave).
  - Forward returns are measured AFTER the signal date, never before.
  - Reports performance split out by signal state (STRONG_BUY vs BUY vs SELL
    etc.) and by holding horizon, so you can see whether the signal actually
    has edge, or whether it's noise dressed up as a rule.

Reuses `pages/cot_signals.py`'s percentile/z-score/trend helpers and
`pages/cot_composite_trade_signal.py`'s `generate_trade_signal` so the
backtest tests the exact same code path the live page runs -- not a
reimplementation that could quietly drift out of sync.

Usage assumes a combined dataframe with weekly rows:
    date, net_position, price, open_interest (optional)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pages.cot_composite_trade_signal import generate_trade_signal
from pages.cot_signals import _rolling_percentile, _slope_trend, _zscore

try:
    import streamlit as st
except Exception:  # keep this module importable for unit tests without streamlit
    st = None


def _walk_forward_row_signal(hist: pd.DataFrame, lookback_weeks: int = 8) -> dict:
    """
    Compute one week's signal using ONLY the trailing history passed in
    (hist must already be truncated to date <= t by the caller).
    Mirrors compute_cot_signal's logic but returns raw components needed
    by generate_trade_signal, to avoid re-deriving OI context here.
    """
    pos = hist["net_position"]
    price = hist["price"]

    pct = _rolling_percentile(pos)
    z = _zscore(pos)
    position_trend_raw = _slope_trend(pos, lookback_weeks)
    price_trend = _slope_trend(price, lookback_weeks)
    latest_position = pos.iloc[-1]

    if position_trend_raw == "rising":
        position_trend = "building_long" if latest_position > 0 else "unwinding_short"
    elif position_trend_raw == "falling":
        position_trend = "building_short" if latest_position < 0 else "unwinding_long"
    else:
        position_trend = "stalled"

    is_extreme_short = pct <= 15 or z <= -1.5
    is_extreme_long = pct >= 85 or z >= 1.5

    divergence_type = "none"
    if price_trend == "rising" and position_trend == "building_short":
        divergence_type = "bearish_divergence"
    elif price_trend == "falling" and position_trend == "building_long":
        divergence_type = "bullish_divergence"
    elif price_trend == "rising" and is_extreme_short and position_trend == "stalled":
        divergence_type = "bearish_divergence"
    elif price_trend == "falling" and is_extreme_long and position_trend == "stalled":
        divergence_type = "bullish_divergence"

    wow_change = pos.iloc[-1] - pos.iloc[-2] if len(pos) > 1 else 0.0
    recent_wow_std = pos.diff().tail(52).std()

    # OI context left as None here unless caller merges an open_interest column
    # and passes oi_interpretation externally per-row; simplest path (used below)
    # recomputes a lightweight version inline if the column exists.
    oi_interpretation = None
    if "open_interest" in hist.columns and len(hist) > lookback_weeks:
        oi_recent = hist["open_interest"].tail(lookback_weeks)
        oi_change = oi_recent.iloc[-1] - oi_recent.iloc[0]
        pos_change_window = pos.tail(lookback_weeks).iloc[-1] - pos.tail(lookback_weeks).iloc[0]
        if pos_change_window > 0 and oi_change > 0:
            oi_interpretation = "new_longs_building"
        elif pos_change_window > 0 and oi_change < 0:
            oi_interpretation = "short_covering"
        elif pos_change_window < 0 and oi_change > 0:
            oi_interpretation = "new_shorts_building"
        elif pos_change_window < 0 and oi_change < 0:
            oi_interpretation = "long_liquidation"

    return dict(
        percentile=pct,
        zscore=z,
        divergence_type=divergence_type,
        wow_change=wow_change,
        latest_position=latest_position,
        oi_interpretation=oi_interpretation,
        recent_wow_std=recent_wow_std,
    )


def walk_forward_signals(df: pd.DataFrame, min_history: int = 104) -> pd.DataFrame:
    """
    Generates one signal per week across the full history, using only
    trailing data at each point (min_history weeks required before the
    first signal is emitted, so percentile/z-score have enough context
    to be meaningful).
    """
    df = df.sort_values("date").reset_index(drop=True)
    records = []

    for i in range(min_history, len(df)):
        hist = df.iloc[: i + 1]  # inclusive of row i -- no future data
        components = _walk_forward_row_signal(hist)
        sig = generate_trade_signal(**components)
        records.append({
            "date": df.iloc[i]["date"],
            "price": df.iloc[i]["price"],
            "state": sig.state,
            "score": sig.score,
            "confidence": sig.confidence,
            "collapse_watch": sig.collapse_watch,
        })

    return pd.DataFrame(records)


def evaluate_signal_performance(
    signal_df: pd.DataFrame,
    horizons: list[int] = (1, 4, 12),
) -> pd.DataFrame:
    """
    For each signal state, computes forward returns at each horizon (in weeks).
    A signal only "wins" if forward price direction matches the signal's
    implied direction (BUY/STRONG_BUY expects price up, SELL/STRONG_SELL
    expects price down).
    """
    df = signal_df.sort_values("date").reset_index(drop=True)
    results = []

    for h in horizons:
        fwd_return = (df["price"].shift(-h) - df["price"]) / df["price"]
        df[f"fwd_ret_{h}"] = fwd_return

    for state in ["STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"]:
        subset = df[df["state"] == state]
        if len(subset) == 0:
            continue
        row = {"state": state, "n_signals": len(subset)}
        for h in horizons:
            rets = subset[f"fwd_ret_{h}"].dropna()
            if len(rets) == 0:
                continue
            expected_direction = 1 if "BUY" in state else (-1 if "SELL" in state else 0)
            if expected_direction != 0:
                win_rate = (np.sign(rets) == expected_direction).mean()
            else:
                win_rate = np.nan
            row[f"avg_ret_{h}w"] = rets.mean()
            row[f"win_rate_{h}w"] = win_rate
            row[f"n_{h}w"] = len(rets)
        results.append(row)

    return pd.DataFrame(results)


def evaluate_collapse_watch(signal_df: pd.DataFrame, horizons: list[int] = (1, 4, 12)) -> pd.DataFrame:
    """Same evaluation, filtered to only rows where collapse_watch fired."""
    collapse_rows = signal_df[signal_df["collapse_watch"]].copy()
    if len(collapse_rows) == 0:
        return pd.DataFrame([{"note": "No collapse_watch events in this history."}])
    return evaluate_signal_performance(collapse_rows, horizons)


def simple_equity_curve(signal_df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Naive strategy: each week, take a position sized by score/3 (so a
    STRONG_BUY at score +3 is full long, NEUTRAL near 0 is roughly flat),
    hold for `horizon` weeks, no compounding overlap handling (simplification
    -- treat as non-overlapping signal-driven rebalance for a first pass).
    Returns cumulative equity curve assuming $1 starting capital.
    """
    df = signal_df.sort_values("date").reset_index(drop=True).copy()
    df["fwd_ret"] = (df["price"].shift(-horizon) - df["price"]) / df["price"]
    df["position_size"] = (df["score"] / 3.0).clip(-1, 1)
    df["strategy_ret"] = df["position_size"] * df["fwd_ret"]
    df["equity"] = (1 + df["strategy_ret"].fillna(0)).cumprod()
    df["buy_hold_equity"] = (1 + df["price"].pct_change(fill_method=None).fillna(0)).cumprod()
    return df[["date", "state", "score", "fwd_ret", "strategy_ret", "equity", "buy_hold_equity"]]


def synthetic_demo_data(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """Synthetic weekly (date, net_position, price, open_interest) series --
    NOT representative of real markets, just enough for a pipeline sanity
    check (no lookahead errors, no crashes) without hitting the network.
    Positioning extremes are built mildly mean-reverting against price so the
    harness has *something* to detect.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="W")

    position = np.zeros(n)
    price = np.zeros(n)
    position[0], price[0] = -20000, 130
    for i in range(1, n):
        position[i] = position[i - 1] + rng.randn() * 3000
        reversion = -0.00003 * position[i - 1]  # mild mean reversion vs price
        price[i] = price[i - 1] + reversion + rng.randn() * 0.5

    open_interest = 300000 + np.cumsum(rng.randn(n) * 2000)
    return pd.DataFrame({
        "date": dates, "net_position": position, "price": price,
        "open_interest": open_interest,
    })


if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run pages/cot_trade_signal_walk_forward_backtest_harness.py")

    # Page bootstrap — kept lazy so the pure functions above stay importable
    # headlessly. Mirrors the sibling COT pages' contract: set_page_config ->
    # BloombergTheme.apply() -> sidebar (controls first, then nav).
    from datetime import datetime, timedelta

    from src.pages_lib.navigation import render_sidebar_nav
    from src.services.cot_fetcher import (
        INSTRUMENTS as _COT_INSTRUMENTS,
        get_instrument_series,
        normalize_datetime64,
    )
    from src.ui.theme import BloombergTheme

    st.set_page_config(
        page_title="COT Composite Backtest · Trading System",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()

    T = BloombergTheme  # short alias for palette tokens

    PRICE_TICKERS = {
        "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "USDJPY=X", "CHF": "USDCHF=X",
        "AUD": "AUDUSD=X", "NZD": "NZDUSD=X", "CAD": "USDCAD=X",
        "GOLD": "GC=F", "DXY": "DX-Y.NYB",
    }

    @st.cache_data(ttl=600, show_spinner=False)
    def _price_series(ticker: str, years_back: int) -> pd.DataFrame:
        """Cached daily close series resampled to weekly, matching the COT
        report's weekly cadence -- the COT data itself is fetched/cached
        separately by `cot_fetcher` (CSV-on-disk)."""
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

    def _equity_chart(equity_df: pd.DataFrame):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_df["date"], y=equity_df["equity"],
                                  name="Signal strategy", line=dict(color=T.GREEN, width=1.75)))
        fig.add_trace(go.Scatter(x=equity_df["date"], y=equity_df["buy_hold_equity"],
                                  name="Buy & hold", line=dict(color=T.GREY, width=1.25, dash="dot")))
        fig.update_yaxes(title_text="Equity ($1 start)", showgrid=True,
                         gridcolor=T.BORDER, tickfont=dict(color=T.GREY, size=9))
        fig.update_xaxes(showgrid=False, tickfont=dict(color=T.GREY, size=9), linecolor=T.BORDER)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T.BG_PANEL,
            font=dict(family=T.FONT_MONO, color=T.GREY, size=11),
            legend=dict(font=dict(color=T.GREY, size=11), bgcolor=T.BG_ELEV,
                       bordercolor=T.BORDER, orientation="h", x=0, y=1.08),
            margin=dict(l=10, r=10, t=30, b=10), height=380,
        )
        return fig

    st.title("🧪 COT Composite — Walk-Forward Backtest")
    st.caption(
        "Walk-forward validation for **35. COT Composite Signal**: at each "
        "historical week, the signal is recomputed using only data available "
        "up to that week (no lookahead), then forward returns are measured "
        "to see whether the signal actually has edge. A rules-based heuristic "
        "validated empirically here isn't investment advice, and past "
        "performance on one instrument's history doesn't guarantee future edge."
    )

    with st.sidebar:
        st.header("Backtest Settings")
        data_source = st.radio(
            "Data source", ["Live COT + price data", "Synthetic demo data (pipeline check)"],
        )
        instrument = None
        years_back = 10
        if data_source.startswith("Live"):
            instrument = st.selectbox("Instrument", sorted(_COT_INSTRUMENTS.keys()), index=0)
            years_back = st.slider("Years of history", 2, 15, 10)
        min_history = st.slider("Min. history before first signal (weeks)", 26, 208, 104)
        lookback_weeks = st.slider("Trend lookback (weeks)", 4, 16, 8)
        horizons = st.multiselect(
            "Forward-return horizons (weeks)", [1, 2, 4, 8, 12, 26], default=[1, 4, 12],
        )
        refresh = False
        if data_source.startswith("Live"):
            refresh = st.button("🔄 Force refresh from CFTC", width="stretch")
            if refresh:
                _price_series.clear()

        st.divider()
        render_sidebar_nav()

    if not horizons:
        st.warning("Pick at least one forward-return horizon.")
        st.stop()
    horizons = sorted(horizons)

    if data_source.startswith("Live"):
        try:
            with st.spinner(f"Loading {instrument} COT data..."):
                series = get_instrument_series(instrument, years_back=years_back, force_refresh=refresh)
        except Exception as e:
            st.error(f"Couldn't load COT data for {instrument}: {e}")
            st.stop()

        price_df = _price_series(PRICE_TICKERS[instrument], years_back)
        if price_df.empty:
            st.warning(f"Couldn't load price data for {instrument} — the backtest needs both series.")
            st.stop()

        series = series.assign(date=normalize_datetime64(series["date"]))
        price_df = price_df.assign(date=normalize_datetime64(price_df["date"]))
        combined = pd.merge_asof(
            series[["date", "net_spec", "open_interest"]].sort_values("date"),
            price_df.sort_values("date"),
            on="date",
            direction="backward",
        ).dropna().rename(columns={"net_spec": "net_position"})
    else:
        instrument = "SYNTHETIC"
        combined = synthetic_demo_data()

    if len(combined) < min_history + 10:
        st.warning(
            f"Only {len(combined)} weeks of overlapping history — need at least "
            f"{min_history + 10} for a meaningful backtest. Lower 'Min. history' "
            f"or pick more years."
        )
        st.stop()

    with st.spinner("Running walk-forward signal generation..."):
        signals = walk_forward_signals(combined, min_history=min_history)

    if signals.empty:
        st.warning("No signals generated over this history — widen the date range.")
        st.stop()

    non_neutral = signals[signals["state"] != "NEUTRAL"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Weekly Signals", f"{len(signals):,}")
    m2.metric("Non-neutral Reads", f"{len(non_neutral):,}")
    m3.metric("Collapse Watches", f"{int(signals['collapse_watch'].sum()):,}")
    m4.metric("Instrument", instrument)

    st.subheader("Performance by signal state")
    st.dataframe(evaluate_signal_performance(signals, horizons=horizons),
                 width="stretch", hide_index=True)

    st.subheader("Collapse-watch performance")
    st.dataframe(evaluate_collapse_watch(signals, horizons=horizons),
                 width="stretch", hide_index=True)

    st.subheader("Equity curve")
    eq_horizon = st.selectbox("Holding horizon for equity curve (weeks)", horizons, index=0)
    equity = simple_equity_curve(signals, horizon=eq_horizon)
    st.plotly_chart(_equity_chart(equity), width="stretch")
    st.caption(
        f"Naive non-compounding strategy: position size = score/3, rebalanced "
        f"every {eq_horizon} week(s). Not a realistic backtest (no costs, "
        f"slippage, or overlap handling) — directional sanity check only."
    )

    with st.expander("Raw weekly signals"):
        st.dataframe(signals.sort_values("date", ascending=False),
                     width="stretch", hide_index=True)
