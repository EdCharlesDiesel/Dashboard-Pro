"""
COT Positioning Signal Detector
--------------------------------
Adds a signal layer on top of raw COT net-positioning data:
  1. Extreme positioning detection (percentile + z-score, already partly in your dashboard)
  2. Price/positioning divergence detection (the more valuable, less obvious signal)
  3. A composite "squeeze watch" flag combining both, with a plain-English reason string

Assumes an input DataFrame with columns (rename via the `col_map` arg if yours differ):
    date            : datetime, weekly
    net_position    : float, net speculative contracts
    price            : float, underlying price (JPY, gold, DXY, etc.)

Drop into your Streamlit app and call `generate_cot_alert(df)` per instrument tab.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np

try:
    import streamlit as st
except Exception:  # keep this module importable for unit tests without streamlit
    st = None


@dataclass
class CotSignal:
    latest_position: float
    wow_change: float
    percentile_3y: float
    zscore: float
    position_trend: str        # "building" | "stalled" | "unwinding"
    price_trend: str           # "rising" | "falling" | "flat"
    divergence: bool
    divergence_type: str       # "bullish_divergence" | "bearish_divergence" | "none"
    squeeze_watch: bool
    headline: str
    detail: str


def _rolling_percentile(series: pd.Series, window: int = 156) -> float:
    """3Y percentile (156 weeks) of the latest value within its own trailing window."""
    recent = series.tail(window)
    if len(recent) < 10:
        return np.nan
    return (recent.rank(pct=True).iloc[-1]) * 100


def _zscore(series: pd.Series, window: int = 156) -> float:
    recent = series.tail(window)
    if len(recent) < 10 or recent.std() == 0:
        return np.nan
    return (recent.iloc[-1] - recent.mean()) / recent.std()


def _slope_trend(series: pd.Series, lookback: int = 8, flat_thresh: float = 0.15) -> str:
    """
    Classify recent trend direction using a simple normalized linear slope
    over `lookback` weeks. flat_thresh is in units of trailing std devs per period.
    """
    recent = series.tail(lookback).reset_index(drop=True)
    if len(recent) < 4 or recent.std() == 0:
        return "flat"
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values, 1)[0]
    normalized = slope / recent.std()
    if normalized > flat_thresh:
        return "rising"
    elif normalized < -flat_thresh:
        return "falling"
    return "flat"


def compute_cot_signal(
    df: pd.DataFrame,
    col_map: dict | None = None,
    extreme_percentile: float = 15.0,   # <=15th or >=85th percentile treated as "extreme"
    extreme_z: float = 1.5,
    lookback_weeks: int = 8,
) -> CotSignal:
    """
    Core computation. Returns a CotSignal with all derived stats plus a plain-English
    headline/detail string suitable for a Streamlit st.warning/st.info banner.
    """
    cols = {"date": "date", "net_position": "net_position", "price": "price"}
    if col_map:
        cols.update(col_map)

    d = df.rename(columns={v: k for k, v in cols.items()}) if col_map else df.copy()
    d = d.sort_values("date").reset_index(drop=True)

    pos = d["net_position"]
    price = d["price"]

    latest_position = pos.iloc[-1]
    wow_change = pos.iloc[-1] - pos.iloc[-2] if len(pos) > 1 else 0.0
    pct = _rolling_percentile(pos)
    z = _zscore(pos)

    position_trend_raw = _slope_trend(pos, lookback_weeks)
    price_trend = _slope_trend(price, lookback_weeks)

    # Map position slope to funds' behavior language
    if position_trend_raw == "rising":
        position_trend = "building_long" if latest_position > 0 else "unwinding_short"
    elif position_trend_raw == "falling":
        position_trend = "building_short" if latest_position < 0 else "unwinding_long"
    else:
        position_trend = "stalled"

    is_extreme_short = pct <= extreme_percentile or z <= -extreme_z
    is_extreme_long = pct >= (100 - extreme_percentile) or z >= extreme_z

    # --- Divergence logic ---
    # Bearish divergence: price rising while positioning gets MORE short (funds not buying the rally)
    # Bullish divergence: price falling while positioning gets MORE long (funds not selling the drop)
    divergence = False
    divergence_type = "none"

    if price_trend == "rising" and position_trend == "building_short":
        divergence = True
        divergence_type = "bearish_divergence"
    elif price_trend == "falling" and position_trend == "building_long":
        divergence = True
        divergence_type = "bullish_divergence"
    elif price_trend == "rising" and is_extreme_short and position_trend == "stalled":
        # crowded short, stalled, price still climbing -> price is out ahead of positioning
        divergence = True
        divergence_type = "bearish_divergence"
    elif price_trend == "falling" and is_extreme_long and position_trend == "stalled":
        divergence = True
        divergence_type = "bullish_divergence"

    squeeze_watch = bool((is_extreme_short or is_extreme_long) and divergence)

    # --- Headline / detail text ---
    direction = "short" if latest_position < 0 else "long"
    extreme_label = ""
    if is_extreme_short:
        extreme_label = "crowded short"
    elif is_extreme_long:
        extreme_label = "crowded long"

    if squeeze_watch:
        headline = f"⚠️ Squeeze watch — {extreme_label}, positioning stalled while price {price_trend}"
        squeeze_dir = "short squeeze (upside)" if is_extreme_short else "long squeeze (downside)"
        detail = (
            f"Positioning is at a {pct:.0f}th percentile extreme (z={z:.2f}) and largely flat "
            f"week-over-week, while price has been {price_trend} over the last {lookback_weeks} weeks. "
            f"That combination — an extreme, stalled position alongside price moving against it — "
            f"is the classic setup dashboards flag for a potential {squeeze_dir}. "
            f"Timing is not implied; this flags elevated risk, not a trade trigger."
        )
    elif extreme_label:
        headline = f"{extreme_label.capitalize()} ({pct:.0f}th percentile, z={z:.2f})"
        detail = (
            f"Net {direction} position of {latest_position:,.0f} sits at a historical extreme. "
            f"No active divergence against price detected — positioning and price direction are "
            f"currently consistent with each other."
        )
    elif divergence:
        headline = f"Divergence forming: price {price_trend}, funds {position_trend.replace('_', ' ')}"
        detail = (
            f"Price and speculative positioning are moving in ways that don't confirm each other. "
            f"Worth watching, though positioning is not yet at a historical extreme "
            f"({pct:.0f}th percentile)."
        )
    else:
        headline = "No notable positioning signal"
        detail = (
            f"Net position {latest_position:,.0f} ({pct:.0f}th percentile, z={z:.2f}) is within "
            f"normal historical range and broadly consistent with recent price action."
        )

    return CotSignal(
        latest_position=latest_position,
        wow_change=wow_change,
        percentile_3y=pct,
        zscore=z,
        position_trend=position_trend,
        price_trend=price_trend,
        divergence=divergence,
        divergence_type=divergence_type,
        squeeze_watch=squeeze_watch,
        headline=headline,
        detail=detail,
    )


def render_streamlit_banner(st, signal: CotSignal):
    """
    Call inside your Streamlit tab: render_streamlit_banner(st, signal)
    Drop-in replacement/extension for your existing yellow banner.
    """
    if signal.squeeze_watch:
        st.error(f"**{signal.headline}**\n\n{signal.detail}")
    elif signal.percentile_3y <= 15 or signal.percentile_3y >= 85:
        st.warning(f"**{signal.headline}**\n\n{signal.detail}")
    elif signal.divergence:
        st.info(f"**{signal.headline}**\n\n{signal.detail}")
    else:
        st.caption(f"{signal.headline} — {signal.detail}")


# ---------------------------------------------------------------------------
# Example usage inside your existing tab:
#
#   from cot_signals import compute_cot_signal, render_streamlit_banner
#
#   signal = compute_cot_signal(
#       jpy_df,
#       col_map={"date": "report_date", "net_position": "net_spec_position", "price": "jpy_price"},
#   )
#   render_streamlit_banner(st, signal)
#
#   # Optional: also show raw numbers alongside your existing metric tiles
#   st.metric("Percentile (3Y)", f"{signal.percentile_3y:.0f}th")
#   st.metric("Z-score", f"{signal.zscore:.2f}")
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run pages/cot_signals.py")

    # Page bootstrap — kept lazy so the pure functions above stay importable
    # headlessly (tests, or the smoke-test-style usage in the module docstring).
    # Mirrors the legacy-page contract: set_page_config -> BloombergTheme.apply()
    # -> sidebar (controls first, then nav).
    from datetime import datetime, timedelta

    from src.pages_lib.navigation import render_sidebar_nav
    from src.services.alert_service import NotifyCache
    from src.services.cot_fetcher import INSTRUMENTS, get_instrument_series
    from src.services.tool_log import log_tool_usage
    from src.ui.theme import BloombergTheme

    st.set_page_config(
        page_title="COT Signals · Trading System",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()

    T = BloombergTheme  # short alias for palette tokens

    # Instrument -> yfinance ticker for the price overlay used in divergence
    # detection (mirrors pages/cot_tab.py's mapping; DXY isn't a registry entry).
    PRICE_TICKERS = {
        "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "USDJPY=X", "CHF": "USDCHF=X",
        "AUD": "AUDUSD=X", "NZD": "NZDUSD=X", "CAD": "USDCAD=X",
        "GOLD": "GC=F", "DXY": "DX-Y.NYB",
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
            title=dict(text=f"{instrument} — Positioning vs Price (divergence view)",
                      font=dict(color=T.WHITE, size=13)),
        )
        return fig

    st.title("🧭 COT Positioning Signals")
    st.caption(
        "Signal layer on top of raw COT positioning (see **17. COT Positioning** for "
        "the underlying data): extreme-positioning detection plus price/positioning "
        "divergence and a composite squeeze-watch flag. A crowdedness and divergence "
        "read, not a trade trigger on its own — timing is not implied."
    )

    with st.sidebar:
        st.header("COT Signal Settings")
        instrument = st.selectbox("Instrument", sorted(INSTRUMENTS.keys()), index=0)
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
        st.warning(f"Couldn't load price data for {instrument} — divergence detection needs both series.")
        st.stop()

    merged = pd.merge_asof(
        series[["date", "net_spec"]].sort_values("date"),
        price_df.sort_values("date"),
        on="date",
        direction="backward",
    ).dropna().rename(columns={"net_spec": "net_position"})

    if len(merged) < 20:
        st.warning("Not enough overlapping history yet for a meaningful signal read.")
        st.stop()

    signal = compute_cot_signal(merged, lookback_weeks=lookback_weeks)

    # Log this read to Postgres (audit trail, not trade_setups — explicitly
    # "not a trade trigger on its own" per this page's own caption). Deduped
    # per instrument+week via NotifyCache, since CFTC data only updates weekly.
    _latest_date = merged["date"].iloc[-1]
    _cs_key = f"{instrument}|{_latest_date}"
    if NotifyCache("cot_signals_log").filter_new([_cs_key]):
        log_tool_usage("cot_signals", {
            "instrument": instrument, "week": str(_latest_date),
            "latest_position": signal.latest_position, "wow_change": signal.wow_change,
            "percentile_3y": signal.percentile_3y, "zscore": signal.zscore,
            "position_trend": signal.position_trend, "price_trend": signal.price_trend,
            "divergence": signal.divergence, "divergence_type": signal.divergence_type,
            "squeeze_watch": signal.squeeze_watch, "headline": signal.headline,
        })

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Net Position", f"{signal.latest_position:,.0f}")
    m2.metric("Week-over-Week Change", f"{signal.wow_change:,.0f}")
    m3.metric("3Y Percentile", f"{signal.percentile_3y:.0f}th")
    m4.metric("Z-score", f"{signal.zscore:.2f}")

    render_streamlit_banner(st, signal)

    c1, c2, c3 = st.columns(3)
    c1.write(f"**Position trend:** {signal.position_trend.replace('_', ' ')}")
    c2.write(f"**Price trend:** {signal.price_trend}")
    c3.write(f"**Divergence:** {signal.divergence_type.replace('_', ' ')}")

    st.plotly_chart(_signal_chart(merged, instrument), width="stretch")

    with st.expander("Merged weekly data (positioning + price)"):
        st.dataframe(
            merged.tail(52).sort_values("date", ascending=False),
            width="stretch",
            hide_index=True,
        )