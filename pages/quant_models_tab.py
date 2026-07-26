"""
quant_models_tab.py
====================
Streamlit page wiring src/core/quant_models.py's statistical models into the
trading dashboard: pair cointegration (Engle-Granger -> Kalman -> OU mean
reversion), GARCH volatility regime + vol-targeted sizing, UIP/carry
deviation, and a GBM null test for any backtest metric.

Each sub-tab logs its read to Postgres via tool_usage_log (see
src/services/tool_log.py) — these are statistical diagnostics (a
cointegration test, a vol forecast, a null-hypothesis check), not a clean
single-instrument pair+bias trade signal, so they stay in the audit trail
rather than trade_setups (see CLAUDE.md's data-layer section).

Entry point: render_quant_models_tab()
Standalone:  streamlit run pages/quant_models_tab.py
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from src.core.quant_models import (fit_ou, ou_signal, fit_garch,
                                   garch_position_size, gbm_null_test,
                                   uip_carry_analysis, kalman_dynamic_beta,
                                   engle_granger)
from src.core.secrets import fred_api_key
from src.instruments.registry import INSTRUMENTS
from src.services.alert_service import NotifyCache
from src.services.tool_log import log_tool_usage

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
_LOG_NS = "quant_models_log"
_YEARS_HISTORY = 6
_CUSTOM_YAHOO = "Custom Yahoo ticker…"
_CUSTOM_FRED = "Custom FRED series…"
_CUSTOM_RATE = "Custom FRED series…"


# ------------------------------------------------------------------------- #
# Data
# ------------------------------------------------------------------------- #
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _fetch_yf(ticker: str, start: str) -> pd.Series:
    px = yf.download(ticker, start=start, interval="1d",
                     auto_adjust=False, progress=False)
    if px.empty:
        return pd.Series(dtype=float)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    s = px["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _fetch_fred(series_id: str, start: str) -> pd.Series:
    import requests
    key = fred_api_key()
    if not key:
        raise RuntimeError("FRED_API_KEY not found in st.secrets or env.")
    r = requests.get(FRED_URL, params={
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": start}, timeout=30)
    r.raise_for_status()
    obs = r.json()["observations"]
    s = pd.Series(
        [float(o["value"]) if o["value"] != "." else np.nan for o in obs],
        index=pd.to_datetime([o["date"] for o in obs]), name=series_id)
    return s.dropna()


def _history_start() -> str:
    return (pd.Timestamp.today()
            - pd.DateOffset(years=_YEARS_HISTORY)).strftime("%Y-%m-%d")


def _symbol_picker(label: str, key: str, default: str = "XAU/USD"
                   ) -> Tuple[pd.Series, str]:
    """A registry instrument by default, or a free-text Yahoo ticker / FRED
    series id for macro drivers (real yields, DXY, ...) that aren't in the
    tradable registry. Returns (price series, display name)."""
    names = sorted(INSTRUMENTS.keys())  # alphabetical dropdown
    options = names + [_CUSTOM_YAHOO, _CUSTOM_FRED]
    idx = options.index(default) if default in options else 0
    choice = st.selectbox(label, options, index=idx, key=f"{key}_choice")

    if choice == _CUSTOM_YAHOO:
        ident = st.text_input(f"{label} — Yahoo ticker", "DX-Y.NYB",
                              key=f"{key}_yf")
        return _fetch_yf(ident, _history_start()).rename(ident), ident
    if choice == _CUSTOM_FRED:
        ident = st.text_input(f"{label} — FRED series id", "DFII5",
                              key=f"{key}_fred")
        return _fetch_fred(ident, _history_start()).rename(ident), ident
    ticker = INSTRUMENTS[choice]["ticker"]
    return _fetch_yf(ticker, _history_start()).rename(choice), choice


def _rate_picker(label: str, key: str, default_ccy: str = "USD") -> str:
    """Dropdown over the dashboard's known policy-rate FRED series
    (src/core/data_provider.py::FRED_SERIES — one entry per registry
    currency), or a free-text FRED id for anything not covered. Returns the
    FRED series id."""
    from src.core.data_provider import FRED_SERIES
    rate_options = {ccy: series["Rates"] for ccy, series in FRED_SERIES.items()}
    labels = [f"{ccy} — {sid}" for ccy, sid in rate_options.items()] + [_CUSTOM_RATE]
    default_label = next((l for l in labels if l.startswith(f"{default_ccy} —")), labels[0])
    choice = st.selectbox(label, labels, index=labels.index(default_label), key=f"{key}_choice")
    if choice == _CUSTOM_RATE:
        return st.text_input(f"{label} — FRED series id", "",
                             placeholder="e.g. IRSTCI01ZAM156N", key=f"{key}_custom")
    return choice.split(" — ")[1]


def _log(sub_key: str, dedupe_key: str, payload: dict) -> None:
    """Log one distinct calculation to tool_usage_log, deduped per sub-tab
    so a Streamlit rerun (widget touch) doesn't re-log an unchanged read."""
    if NotifyCache(f"{_LOG_NS}_{sub_key}").filter_new([dedupe_key]):
        log_tool_usage("quant_models", {"model": sub_key, **payload})


def render_quant_models_tab() -> None:
    st.header("🧮 Quant Models")
    st.caption("OU mean reversion · GARCH vol · GBM null test · UIP/carry · "
               "Kalman beta · Engle-Granger cointegration")

    sub = st.tabs(["Pair Pipeline", "Volatility (GARCH)",
                   "Carry / UIP", "Signal vs Luck (GBM)"])

    # ------------------------------------------------------------------ #
    # TAB 1: full pair pipeline (Engle-Granger -> Kalman -> OU)
    # ------------------------------------------------------------------ #
    with sub[0]:
        c1, c2 = st.columns(2)
        with c1:
            y, sym_y = _symbol_picker("Y (dependent)", "qm_y", "XAU/USD")
        with c2:
            x, sym_x = _symbol_picker("X (driver)", "qm_x", _CUSTOM_FRED)
        entry_z = st.slider("Entry z-score", 1.0, 3.0, 2.0, 0.25, key="qm_z")

        if st.button("Run pair analysis", key="qm_run_pair"):
            if y.empty or x.empty:
                st.warning("No data for one of the two series.")
            else:
                with st.spinner("Fitting Engle-Granger + Kalman + OU..."):
                    eg = engle_granger(y, x)
                    kf = kalman_dynamic_beta(y, x)
                    ou = fit_ou(eg["spread"])
                    sig = ou_signal(ou, entry_z=entry_z)

                cointegrated = bool(eg["cointegrated_5pct"])
                if not cointegrated:
                    st.error(f"NOT cointegrated (ADF p={eg['adf_pvalue']:.3f}). "
                             "The spread is a random walk — z-scores are "
                             "meaningless. Do not trade this pair as reversion.")
                else:
                    st.success(f"Cointegrated (ADF p={eg['adf_pvalue']:.4f}), "
                               f"error-correction α={eg['ec_alpha']:.3f} "
                               f"(t={eg['ec_alpha_tstat']:.1f})")

                beta_kalman = float(kf["beta"].iloc[-1])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("OU z-score", f"{ou.zscore:+.2f}")
                m2.metric("Half-life (bars)",
                          f"{ou.half_life:.1f}" if np.isfinite(ou.half_life) else "∞")
                m3.metric("Kalman β (now)", f"{beta_kalman:.3f}",
                          delta=f"{beta_kalman - eg['hedge_ratio']:+.3f} vs static")
                m4.metric("Signal", {1: "LONG spread", 0: "FLAT",
                                     -1: "SHORT spread"}[sig["signal"]])

                spread = eg["spread"]
                bands = pd.DataFrame({
                    "spread": spread,
                    "mean": ou.mu,
                    f"+{entry_z}σ": ou.mu + entry_z * ou.sigma_eq,
                    f"-{entry_z}σ": ou.mu - entry_z * ou.sigma_eq,
                })
                st.line_chart(bands)

                st.line_chart(kf[["beta"]].rename(columns={"beta": "Kalman β"}))
                st.caption("If Kalman β drifts far from the static hedge ratio, "
                           "the relationship is unstable — reduce confidence.")

                _log("pair_pipeline", f"{sym_y}|{sym_x}|{entry_z}|{spread.index[-1]}", {
                    "symbol_y": sym_y, "symbol_x": sym_x,
                    "cointegrated": cointegrated,
                    "adf_pvalue": eg["adf_pvalue"],
                    "ec_alpha": eg["ec_alpha"], "ec_alpha_tstat": eg["ec_alpha_tstat"],
                    "hedge_ratio_static": eg["hedge_ratio"], "hedge_ratio_kalman": beta_kalman,
                    "ou_zscore": ou.zscore, "ou_half_life": ou.half_life,
                    "signal": sig["signal"], "tradeable": bool(sig["tradeable"] and cointegrated),
                })

    # ------------------------------------------------------------------ #
    # TAB 2: GARCH vol
    # ------------------------------------------------------------------ #
    with sub[1]:
        c1, c2 = st.columns(2)
        with c1:
            px, sym = _symbol_picker("Instrument", "qm_g", "EUR/USD")
        risk = c2.number_input("Risk per trade (% of account)",
                               0.25, 5.0, 1.0, 0.25, key="qm_g_risk")

        if st.button("Fit GARCH(1,1)", key="qm_run_garch"):
            if px.empty:
                st.warning("No data for this instrument.")
            else:
                rets = px.pct_change().dropna()
                try:
                    g = fit_garch(rets)
                except (ImportError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Vol now (annual)",
                              f"{g['current_vol_annual_pct']:.1f}%")
                    m2.metric("Vol percentile", f"{g['vol_percentile']:.0f}",
                              help=">80 = high-vol regime, <20 = low-vol regime")
                    m3.metric("Persistence α+β", f"{g['persistence']:.3f}")

                    regime_msg = {
                        "high": "HIGH vol regime: widen stops, halve size, favour "
                                "breakout logic over mean reversion.",
                        "low": "LOW vol regime: compression often precedes expansion "
                               "— your 20-day breakout scanner matters more here.",
                        "normal": "Normal vol regime.",
                    }[g["regime"]]
                    st.info(regime_msg)

                    sizing = garch_position_size(risk, g["current_vol_daily_pct"])
                    st.write(f"**Vol-targeted sizing** — stop at "
                             f"{sizing['stop_distance_pct']:.2f}% "
                             f"(2× forecast daily vol) → notional leverage "
                             f"{sizing['notional_leverage']:.2f}× to risk {risk}%.")

                    fc = pd.DataFrame({
                        "forecast vol (annual %)": g["forecast_vol_annual_pct"],
                        "long-run vol": g["long_run_vol_annual_pct"],
                    }, index=[f"t+{i+1}" for i in range(len(g["forecast_vol_annual_pct"]))])
                    st.line_chart(fc)
                    st.caption("Forecast decays toward long-run vol; the gap tells "
                               "you whether current conditions are temporary.")

                    _log("garch", f"{sym}|{risk}|{rets.index[-1]}", {
                        "symbol": sym, "vol_annual": g["current_vol_annual_pct"],
                        "vol_percentile": g["vol_percentile"], "regime": g["regime"],
                        "persistence": g["persistence"],
                    })

    # ------------------------------------------------------------------ #
    # TAB 3: UIP / carry
    # ------------------------------------------------------------------ #
    with sub[2]:
        st.caption("Domestic/foreign policy rates are looked up from the "
                   "dashboard's known FRED series per currency — pick "
                   "'Custom FRED series…' for anything not covered.")
        c1, c2, c3 = st.columns(3)
        with c1:
            spot, sym_pair = _symbol_picker("Pair (spot)", "qm_u", "USD/ZAR")
        base_ccy, _, quote_ccy = sym_pair.partition("/")
        with c2:
            r_dom = _rate_picker("Domestic rate", "qm_u_d", base_ccy or "USD")
        with c3:
            r_for = _rate_picker("Foreign rate", "qm_u_f", quote_ccy or "ZAR")

        if st.button("Run UIP analysis", key="qm_run_uip"):
            if spot.empty:
                st.warning("No data for this pair.")
            elif not r_dom or not r_for:
                st.warning("Enter both FRED series ids.")
            else:
                try:
                    dom = _fetch_fred(r_dom, _history_start())
                    for_ = _fetch_fred(r_for, _history_start())
                except Exception as exc:
                    st.error(f"FRED fetch failed: {exc}")
                else:
                    df = uip_carry_analysis(spot, dom, for_).dropna()
                    if df.empty:
                        st.warning("Not enough overlapping history.")
                    else:
                        last = df.iloc[-1]
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Rate differential", f"{last['rate_diff_ann']:.2f}%")
                        m2.metric("Carry deviation (63d)",
                                  f"{last['carry_deviation']:+.2f}%")
                        m3.metric("Carry z-score", f"{last['carry_z']:+.2f}")
                        st.line_chart(df[["uip_expected_ret", "realized_ret"]])
                        st.caption("Persistent gap = the carry premium. When realized "
                                   "return chronically exceeds UIP prediction, the "
                                   "high-yielder is paying you twice (rate + spot).")
                        _log("uip", f"{sym_pair}|{r_dom}|{r_for}|{df.index[-1]}", {
                            "symbol": sym_pair, "rate_dom": r_dom, "rate_for": r_for,
                            "rate_diff": last["rate_diff_ann"], "carry_z": last["carry_z"],
                        })

    # ------------------------------------------------------------------ #
    # TAB 4: GBM null test
    # ------------------------------------------------------------------ #
    with sub[3]:
        st.caption("Test any backtest number against pure luck. Example "
                   "below tests the mean daily log-return.")
        px, sym = _symbol_picker("Instrument", "qm_n", "XAU/USD")
        observed = st.number_input("Observed metric from your backtest "
                                   "(e.g. mean overnight log-return)",
                                   value=0.0005, format="%.6f", key="qm_n_obs")
        n_sims = st.slider("Simulations", 500, 5000, 2000, 500, key="qm_n_sims")

        if st.button("Run null test", key="qm_run_null"):
            if px.empty:
                st.warning("No data for this instrument.")
            else:
                def metric(prices: pd.Series) -> float:
                    # stand-in metric: mean daily log return.
                    # Swap for your real one, e.g. overnight-session return.
                    return float(np.log(prices).diff().mean())

                with st.spinner(f"Simulating {n_sims} GBM paths..."):
                    res = gbm_null_test(px, observed, metric, n_sims=n_sims)

                m1, m2, m3 = st.columns(3)
                m1.metric("Percentile vs null", f"{res['percentile']:.1f}")
                m2.metric("p-value", f"{res['p_value_one_sided']:.3f}")
                m3.metric("Beats luck?", "YES" if res["beats_null"] else "NO")

                if res["beats_null"]:
                    st.success("Observed value sits outside what random GBM "
                               "paths produce. Encouraging — but GBM is a weak "
                               "null (no fat tails / vol clustering). Confirm "
                               "with a block bootstrap before sizing up.")
                else:
                    st.error("A random walk reproduces this result. "
                             "No evidence of edge — do not trade it.")

                _log("gbm_null", f"{sym}|{observed}|{n_sims}", {
                    "symbol": sym, "observed": observed,
                    "p_value": res["p_value_one_sided"],
                    "beats_null": bool(res["beats_null"]),
                })


# ===========================================================================
# PAGE ENTRY — multipage (auto-run) + standalone (streamlit run)
# ===========================================================================

def _page() -> None:
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="QNTM · Quant Models",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()
    with st.sidebar:
        st.markdown("### 🧮 Quant Models")
        st.caption("Cointegration, GARCH vol, carry, GBM null test")
        st.divider()
        render_sidebar_nav()
    render_quant_models_tab()


_page()
