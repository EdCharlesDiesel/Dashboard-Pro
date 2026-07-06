"""
Open Interest add-on for your existing COT module
----------------------------------------------------
Key insight: you don't need a new data source. The CFTC COT report you're
already pulling includes an official Open Interest field per contract, per
week ("open_int_all"). It's exchange-reported (not survey data), free, no
auth, same endpoint family you're likely already hitting.

CME's own free Volume & Open Interest pages exist too, but they're rendered
via JS and not built for programmatic pulls without a paid DataMine
subscription -- CFTC's Socrata API is the practical free route.

Endpoint used below: TFF (Traders in Financial Futures) Futures Only report,
which is the correct COT report family for FX/financial futures (matches
what your dashboard already uses for Leveraged Funds positioning).
Dataset id: jun7-fc8e  (publicreporting.cftc.gov)
"""

import requests
import pandas as pd

try:
    import streamlit as st
except Exception:  # keep this module importable for unit tests without streamlit
    st = None

TFF_FUTURES_ONLY_ENDPOINT = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"


def fetch_cot_with_oi(
    market_name_contains: str,
    weeks_back: int = 260,   # ~5 years of weekly data
) -> pd.DataFrame:
    """
    Pulls COT rows (TFF Futures Only report) for a given market, including
    open interest, sorted oldest -> newest.

    market_name_contains: substring match against CFTC's market_and_exchange_names
        field, e.g. "JAPANESE YEN", "EURO FX", "GOLD" (Legacy report for gold,
        see note below), "U.S. DOLLAR INDEX" for DXY.

    Returns columns:
        date, open_interest, lev_money_positions_long, lev_money_positions_short,
        net_lev_money (derived)
    """
    params = {
        # '%' is the SoQL wildcard; pass it literally and let `requests` do the
        # URL-encoding — pre-encoding it as %25 here caused a double-encoding
        # bug (server saw a literal "%25" substring instead of a wildcard,
        # so the LIKE clause never matched anything).
        "$where": f"market_and_exchange_names like '%{market_name_contains}%'",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": weeks_back,
    }
    resp = requests.get(TFF_FUTURES_ONLY_ENDPOINT, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        raise ValueError(
            f"No rows returned for '{market_name_contains}'. "
            f"Check the exact market_and_exchange_names spelling on "
            f"publicreporting.cftc.gov."
        )

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    df["open_interest"] = pd.to_numeric(df["open_interest_all"], errors="coerce")
    df["lev_money_positions_long"] = pd.to_numeric(
        df["lev_money_positions_long"], errors="coerce"
    )
    df["lev_money_positions_short"] = pd.to_numeric(
        df["lev_money_positions_short"], errors="coerce"
    )
    df["net_lev_money"] = df["lev_money_positions_long"] - df["lev_money_positions_short"]

    return df[
        ["date", "open_interest", "lev_money_positions_long",
         "lev_money_positions_short", "net_lev_money"]
    ].sort_values("date").reset_index(drop=True)


def oi_trend_context(df: pd.DataFrame, lookback: int = 8) -> dict:
    """
    Classic futures-analysis combo: direction of price/position change +
    direction of open interest change tells you whether a move is being
    driven by fresh positioning or by unwinding.

    Standard interpretation (applies whether you pair OI with price or
    with net positioning change):
        position up   + OI up   -> new positions being added (trend strength)
        position up   + OI down -> short covering (less durable move)
        position down + OI up   -> new shorts being added (trend strength)
        position down + OI down -> long liquidation (less durable move)
    """
    recent = df.tail(lookback)
    oi_change = recent["open_interest"].iloc[-1] - recent["open_interest"].iloc[0]
    pos_change = recent["net_lev_money"].iloc[-1] - recent["net_lev_money"].iloc[0]

    if pos_change > 0 and oi_change > 0:
        interp = "new_longs_building"
        note = "Net long positioning rising alongside open interest -- fresh money entering, trend has structural support."
    elif pos_change > 0 and oi_change < 0:
        interp = "short_covering"
        note = "Net position rising but open interest falling -- likely short covering rather than new buying. Less durable."
    elif pos_change < 0 and oi_change > 0:
        interp = "new_shorts_building"
        note = "Net position falling alongside rising open interest -- fresh shorts entering, trend has structural support."
    elif pos_change < 0 and oi_change < 0:
        interp = "long_liquidation"
        note = "Net position falling alongside falling open interest -- likely long liquidation rather than new selling. Less durable."
    else:
        interp = "flat"
        note = "No clear directional signal in the open interest / positioning combination over this window."

    return {
        "oi_change": oi_change,
        "position_change": pos_change,
        "interpretation": interp,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Example usage in your Streamlit tab:
#
#   from cot_open_interest import fetch_cot_with_oi, oi_trend_context
#
#   jpy_oi_df = fetch_cot_with_oi("JAPANESE YEN")
#   context = oi_trend_context(jpy_oi_df)
#
#   st.metric("Open Interest", f"{jpy_oi_df['open_interest'].iloc[-1]:,.0f}")
#   st.caption(context["note"])
#
#   # Add a second y-axis line to your existing plotly chart:
#   fig.add_trace(go.Scatter(x=jpy_oi_df["date"], y=jpy_oi_df["open_interest"],
#                             name="Open Interest", yaxis="y3", line=dict(color="orange")))
# ---------------------------------------------------------------------------

# Notes on market_name_contains values (check exact spelling if a query returns empty):
#   JPY  -> "JAPANESE YEN"
#   EUR  -> "EURO FX"
#   GBP  -> "BRITISH POUND"
#   AUD  -> "AUSTRALIAN DOLLAR"
#   CAD  -> "CANADIAN DOLLAR"
#   DXY  -> "USD INDEX" (Legacy report, non-commercial category -- use Legacy
#            endpoint 6dca-aqww instead of TFF, since DXY isn't in TFF)
#   GOLD -> "GOLD" (also Legacy report, not TFF -- metals aren't financial futures)

if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run pages/cot_open_interest.py")

    # Page bootstrap — kept lazy so the pure functions above stay importable
    # headlessly. Mirrors the legacy-page contract: set_page_config ->
    # BloombergTheme.apply() -> sidebar (controls first, then nav).
    from src.pages_lib.navigation import render_sidebar_nav
    from src.services.cot_fetcher import INSTRUMENTS as _COT_INSTRUMENTS
    from src.ui.theme import BloombergTheme

    st.set_page_config(
        page_title="COT Open Interest · Trading System",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()

    T = BloombergTheme  # short alias for palette tokens

    # This page only covers the TFF (Traders in Financial Futures) report family
    # that `fetch_cot_with_oi` queries above — Gold/DXY are Legacy-report
    # instruments (see the module notes) and aren't available here.
    TFF_INSTRUMENTS = {
        name: market for name, (report_type, market, _spec) in _COT_INSTRUMENTS.items()
        if report_type == "traders_in_financial_futures_fut"
    }

    def _oi_chart(df: pd.DataFrame, instrument: str):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=df["date"], y=df["net_lev_money"],
                name="Net Leveraged Funds Position",
                marker_color=[T.GREEN if v >= 0 else T.RED for v in df["net_lev_money"]],
                opacity=0.7,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["open_interest"],
                name="Open Interest", line=dict(color=T.CYAN, width=1.5),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Net Contracts", secondary_y=False, showgrid=True,
                         gridcolor=T.BORDER, tickfont=dict(color=T.GREY, size=9))
        fig.update_yaxes(title_text="Open Interest", secondary_y=True, showgrid=False,
                         tickfont=dict(color=T.GREY, size=9))
        fig.update_xaxes(showgrid=False, tickfont=dict(color=T.GREY, size=9), linecolor=T.BORDER)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T.BG_PANEL,
            font=dict(family=T.FONT_MONO, color=T.GREY, size=11),
            legend=dict(font=dict(color=T.GREY, size=11), bgcolor=T.BG_ELEV,
                       bordercolor=T.BORDER, orientation="h", x=0, y=1.08),
            margin=dict(l=10, r=10, t=30, b=10), height=450,
            title=dict(text=f"{instrument} — Positioning vs Open Interest",
                      font=dict(color=T.WHITE, size=13)),
        )
        return fig

    st.title("🧮 COT Open Interest")
    st.caption(
        "Open interest context on top of leveraged-funds positioning (TFF report, "
        "see **17. COT Positioning** / **18. COT Signals** for the broader read): "
        "whether a positioning move is fresh conviction or unwinding, per the "
        "classic position-change + open-interest-change combination. Structural "
        "context, not a trade trigger on its own."
    )

    with st.sidebar:
        st.header("Open Interest Settings")
        instrument = st.selectbox("Instrument", list(TFF_INSTRUMENTS.keys()), index=0)
        weeks_back = st.slider("Weeks of history", 52, 260, 260)
        lookback = st.slider("Trend lookback (weeks)", 4, 16, 8)
        st.divider()
        render_sidebar_nav()

    try:
        with st.spinner(f"Loading {instrument} open interest data..."):
            df = fetch_cot_with_oi(TFF_INSTRUMENTS[instrument], weeks_back=weeks_back)
    except Exception as e:
        st.error(f"Couldn't load open interest data for {instrument}: {e}")
        st.stop()

    if len(df) < lookback + 1:
        st.warning("Not enough history yet for a meaningful open-interest read.")
        st.stop()

    context = oi_trend_context(df, lookback=lookback)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Open Interest", f"{df['open_interest'].iloc[-1]:,.0f}")
    m2.metric(f"OI Change ({lookback}w)", f"{context['oi_change']:,.0f}")
    m3.metric("Net Lev. Money", f"{df['net_lev_money'].iloc[-1]:,.0f}")
    m4.metric(f"Position Change ({lookback}w)", f"{context['position_change']:,.0f}")

    label = context["interpretation"].replace("_", " ").title()
    if context["interpretation"] in ("new_longs_building", "new_shorts_building"):
        st.success(f"**{label}** — {context['note']}")
    elif context["interpretation"] in ("short_covering", "long_liquidation"):
        st.warning(f"**{label}** — {context['note']}")
    else:
        st.caption(f"{label} — {context['note']}")

    st.plotly_chart(_oi_chart(df, instrument), width="stretch")

    with st.expander("Raw weekly data"):
        st.dataframe(
            df.tail(52).sort_values("date", ascending=False),
            width="stretch",
            hide_index=True,
        )