"""Multi-Timeframe Matrix — house-view consensus board + walk-forward validation."""
import streamlit as st

from src.pages_lib import market_overview_lib as mol
from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="MTF Matrix · Trading System",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()
mol.inject_css()
mol.subpage_sidebar("🧭 MTF Matrix")


def _render_validation_tab() -> None:
    """Walk-forward validation of the house view (src/services/house_view_backtest)."""
    from src.instruments.registry import INSTRUMENTS
    from src.services.house_view_backtest import DEFAULT_HORIZONS, run_universe
    from src.services.market_data import daily_ohlc

    st.subheader("🧪 Walk-Forward Validation")
    st.caption(
        "Replays the house view over history with **no lookahead** (per-day "
        "recomputation is unit-tested to match the live engine on truncated "
        "data; Weekly+Daily legs — hourly history can't reach back far enough "
        "for the 4H leg, and the composite renormalizes exactly as it does "
        "live). Signed forward returns: for BEARISH days a falling market "
        "counts as a hit."
    )

    default_pairs = [p for p in ("EUR/USD", "XAU/USD", "WTI/USD", "GBP/USD")
                     if p in INSTRUMENTS]
    c1, c2 = st.columns([3, 1])
    pairs = c1.multiselect("Instruments", sorted(INSTRUMENTS.keys()),
                           default=default_pairs)
    period = c2.selectbox("History", ["2y", "5y"], index=1)

    if st.button("▶ Run walk-forward", type="primary"):
        bars_by_pair = {}
        prog = st.progress(0, text="Fetching daily history…")
        for i, pair in enumerate(pairs):
            prog.progress((i + 1) / max(len(pairs), 1),
                          text=f"Fetching daily history — {pair}…")
            bars_by_pair[pair] = daily_ohlc(INSTRUMENTS[pair]["ticker"],
                                            period=period)
        prog.progress(1.0, text="Replaying house view…")
        pooled, per_pair = run_universe(bars_by_pair)
        prog.empty()
        st.session_state["hv_bt"] = {"pooled": pooled, "per_pair": per_pair,
                                     "period": period}

    result = st.session_state.get("hv_bt")
    if not result:
        st.info("Pick instruments and run the walk-forward. First run fetches "
                "the daily history; reruns serve from cache.")
        return

    pooled = result["pooled"]
    if pooled.empty:
        st.warning("Not enough history to evaluate — try a longer period or "
                   "different instruments.")
        return

    _col_cfg = {"n": st.column_config.NumberColumn("Days")}
    for h in DEFAULT_HORIZONS:
        _col_cfg[f"hit_{h}"] = st.column_config.NumberColumn(
            f"Hit % ({h}d)", format="%.1f",
            help=f"Share of days where the signed {h}-day forward return was positive")
        _col_cfg[f"avg_{h}_bps"] = st.column_config.NumberColumn(
            f"Avg bps ({h}d)", format="%.1f",
            help=f"Mean signed {h}-day forward return in basis points")

    st.markdown(f"#### Pooled — {len(result['per_pair'])} instruments · "
                f"{result['period']} history")
    st.dataframe(pooled.reset_index(names="State"), width="stretch",
                 hide_index=True, column_config=_col_cfg)
    st.caption(
        "**How to read it:** the house view earns its keep if BULLISH/BEARISH "
        "(and especially ALIGNED) days show hit rates above 50% and positive "
        "signed returns, while NEUTRAL days show no edge — that's the engine "
        "telling you to stand aside. ⚠️ Daily observations with multi-day "
        "horizons overlap heavily, so effective sample sizes are far smaller "
        "than the day counts — treat this as descriptive evidence, not a "
        "significance test. Close-to-close, no spread/slippage. The replay "
        "uses expanding history rather than the live trailing windows; EMA "
        "differences from that decay to negligible within a few spans."
    )

    with st.expander("Per-instrument breakdown"):
        for pair, rep in result["per_pair"].items():
            st.markdown(f"**{pair}**")
            if rep.empty:
                st.caption("Not enough history.")
            else:
                st.dataframe(rep.reset_index(names="State"), width="stretch",
                             hide_index=True, column_config=_col_cfg)


tab_board, tab_validate = st.tabs(["🧭 Consensus Board", "🧪 Validation"])
with tab_board:
    # No eager ensure_loaded(): the consensus board renders from the spine
    # alone; the legacy sentiment expander lazy-loads the Market Overview
    # dataset itself if (and only if) the user asks for it.
    mol.render_mtf_matrix_tab()
with tab_validate:
    _render_validation_tab()
