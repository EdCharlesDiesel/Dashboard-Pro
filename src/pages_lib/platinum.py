"""
Platinum monitor tab.

Dashboard-Pro tab contract: expose render(engine), cache heavy work behind
st.cache_data, read from Postgres only, never fetch in the hot path.

Lives at: src/pages_lib/platinum.py
Engine at: src/core/platinum.py
Collector: src/data_backbone/platinum_jobs.py  (APScheduler, writes to ohlc)
Register in the tab router alongside the other render(engine) tabs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sqlalchemy import text

from src.core.platinum import (
    cross_correlation,
    disconnect_state,
    log_returns,
    nested_incremental_test,
    producer_margin,
    ratio_series,
    realised_vol,
    rolling_factor_model,
    zscore,
)

try:
    from src.ui.theme import BloombergTheme

    _TPL = getattr(BloombergTheme, "plotly_template", "plotly_dark")
    _ACCENT = getattr(BloombergTheme, "accent", "#c8102e")
    _MUTED = getattr(BloombergTheme, "muted", "#8a8f98")
except Exception:  # theme module absent in a bare checkout
    _TPL, _ACCENT, _MUTED = "plotly_dark", "#c8102e", "#8a8f98"

_BLUE = "#4a9eff"
_AMBER = "#e8a33d"
_GREEN = "#3fb950"

_LAYOUT = dict(template=_TPL, height=420, margin=dict(l=55, r=30, t=50, b=45),
               hovermode="x unified", legend=dict(orientation="h", y=1.06, x=0))

# Symbols expected in the ohlc table. Adjust to match your collector.
SYM_PT = "XPTUSD"
SYM_ZAR = "USDZAR"
SYM_DXY = "DXY"
SYM_AU = "XAUUSD"
SYM_PD = "XPDUSD"

_OHLC_SQL = """
    SELECT ts, symbol, close
    FROM ohlc
    WHERE symbol = ANY(:symbols) AND ts >= :since
    ORDER BY ts
"""

_COT_SQL = """
    SELECT report_date, noncomm_long, noncomm_short, open_interest
    FROM cot_reports
    WHERE cftc_code = :code AND report_date >= :since
    ORDER BY report_date
"""

# CFTC market code for Platinum (NYMEX). Gold 088691, Silver 084691.
COT_PLATINUM = "076651"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner="Loading platinum complex...")
def load_prices(_engine, lookback_days: int) -> pd.DataFrame:
    """Wide close-price frame for the platinum factor set.

    Falls back to correlated synthetic data so the tab is explorable before
    the collector has backfilled. Synthetic mode is flagged loudly — never
    let a demo series be mistaken for a live read.
    """
    since = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=lookback_days)
    symbols = [SYM_PT, SYM_ZAR, SYM_DXY, SYM_AU, SYM_PD]
    try:
        with _engine.connect() as conn:
            df = pd.read_sql(text(_OHLC_SQL), conn,
                             params={"symbols": symbols, "since": since})
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
            wide = df.pivot_table(index="ts", columns="symbol", values="close",
                                  aggfunc="last").sort_index()
            if SYM_PT in wide.columns and wide[SYM_PT].notna().sum() >= 120:
                wide.attrs["synthetic"] = False
                return wide
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        st.info(f"ohlc table unavailable ({type(exc).__name__}); showing synthetic data.")

    return _synthetic(lookback_days)


def _synthetic(days: int) -> pd.DataFrame:
    """Correlated GBM stand-in with a genuine common dollar factor.

    Built so the ZAR loads on the same factor as platinum but has NO
    incremental effect — i.e. the null the tab is designed to test is true by
    construction here. If the incremental panel lights up on synthetic data,
    the test is broken, not the market.
    """
    rng = np.random.default_rng(11)
    n = max(400, days)
    idx = pd.bdate_range(end=pd.Timestamp.utcnow().normalize(), periods=n)

    dollar = rng.normal(0, 0.004, n)                      # common factor
    pt = -1.5 * dollar + rng.normal(0, 0.011, n)
    au = -1.1 * dollar + rng.normal(0, 0.007, n)
    pd_ = -1.4 * dollar + rng.normal(0, 0.015, n)
    zar = 1.3 * dollar + rng.normal(0, 0.008, n)          # no path to pt

    out = pd.DataFrame({
        SYM_DXY: 98 * np.exp(np.cumsum(dollar)),
        SYM_PT: 1650 * np.exp(np.cumsum(pt)),
        SYM_AU: 4400 * np.exp(np.cumsum(au)),
        SYM_PD: 1250 * np.exp(np.cumsum(pd_)),
        SYM_ZAR: 17.4 * np.exp(np.cumsum(zar)),
    }, index=idx)
    out.attrs["synthetic"] = True
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_cot(_engine, lookback_days: int) -> pd.DataFrame:
    since = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=lookback_days)
    try:
        with _engine.connect() as conn:
            df = pd.read_sql(text(_COT_SQL), conn,
                             params={"code": COT_PLATINUM, "since": since})
        if not df.empty:
            df["report_date"] = pd.to_datetime(df["report_date"])
            df["net"] = df["noncomm_long"] - df["noncomm_short"]
            return df
    except Exception:  # noqa: BLE001
        pass
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def compute_factors(prices: pd.DataFrame, window: int, z_window: int) -> dict:
    """All the modelling in one cached call."""
    rets = pd.DataFrame({c: log_returns(prices[c]) for c in prices.columns
                         if prices[c].notna().sum() > 30}).dropna(how="all")

    have = set(rets.columns)
    base_cols = [c for c in (SYM_DXY, SYM_AU) if c in have]
    out: dict = {"rets": rets, "base_cols": base_cols}

    if SYM_PT not in have or not base_cols:
        return out

    y = rets[SYM_PT]
    out["factor"] = rolling_factor_model(y, rets[base_cols], window=window,
                                         z_window=z_window)

    if SYM_ZAR in have:
        out["incremental"] = nested_incremental_test(
            y, rets[base_cols], rets[[SYM_ZAR]]
        )
        out["leadlag"] = cross_correlation(y, rets[SYM_ZAR], max_lag=10)
        out["margin"] = producer_margin(prices[SYM_PT], prices[SYM_ZAR])

    if SYM_AU in have:
        out["pt_au"] = ratio_series(prices[SYM_PT], prices[SYM_AU])
    if SYM_PD in have:
        out["pt_pd"] = ratio_series(prices[SYM_PT], prices[SYM_PD])

    out["rv20"] = realised_vol(prices[SYM_PT], 20)
    out["rv60"] = realised_vol(prices[SYM_PT], 60)
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pct(s: pd.Series, n: int) -> float:
    s = s.dropna()
    if len(s) <= n:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1.0) * 100.0


def _last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _fmt(v: float, d: int = 2, suffix: str = "") -> str:
    return "—" if not np.isfinite(v) else f"{v:,.{d}f}{suffix}"


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def _header(prices: pd.DataFrame, f: dict) -> None:
    pt = prices[SYM_PT]
    c = st.columns(6)
    c[0].metric("XPTUSD", _fmt(_last(pt)), f"{_pct(pt, 1):+.2f}%")
    c[1].metric("1M", _fmt(_pct(pt, 21), 2, "%"))

    if "margin" in f and len(f["margin"]):
        m = f["margin"]
        c[2].metric("XPTZAR (producer)", _fmt(_last(m), 0),
                    f"{_pct(m, 21):+.2f}% 1M")
    if SYM_ZAR in prices:
        z = prices[SYM_ZAR]
        c[3].metric("USDZAR", _fmt(_last(z), 4), f"{_pct(z, 21):+.2f}% 1M")
    if "pt_au" in f:
        c[4].metric("Pt/Au", _fmt(_last(f["pt_au"]), 3))
    if "rv20" in f:
        c[5].metric("RV 20d", _fmt(_last(f["rv20"]) * 100, 1, "%"))


def _price_chart(prices: pd.DataFrame, f: dict) -> None:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=prices.index, y=prices[SYM_PT], name="XPTUSD",
                             line=dict(color=_ACCENT, width=2)), secondary_y=False)
    if "margin" in f and len(f["margin"]):
        fig.add_trace(go.Scatter(x=f["margin"].index, y=f["margin"],
                                 name="XPTZAR", line=dict(color=_AMBER, width=1.4,
                                                          dash="dot")),
                      secondary_y=True)
    fig.update_layout(title="Platinum — dollar price vs rand producer revenue", **_LAYOUT)
    fig.update_yaxes(title_text="USD/oz", secondary_y=False)
    fig.update_yaxes(title_text="ZAR/oz", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


def _disconnect(f: dict, entry: float) -> None:
    fac = f.get("factor")
    if fac is None or fac.empty:
        st.warning("Not enough overlapping history for the factor model.")
        return

    z = _last(fac["resid_z"])
    state = disconnect_state(z, entry=entry)
    colour = {"STRETCHED RICH": _ACCENT, "STRETCHED CHEAP": _GREEN,
              "EXTENDED": _AMBER}.get(state, _MUTED)

    a, b, c = st.columns([1, 1, 2])
    a.metric("Residual z", _fmt(z, 2))
    b.metric("Rolling R²", _fmt(_last(fac["r2"]) * 100, 1, "%"))
    c.markdown(
        f"<div style='padding:14px 0;font-size:1.35rem;font-weight:600;"
        f"color:{colour}'>{state}</div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fac.index, y=fac["resid_z"], name="residual z",
                             line=dict(color=_BLUE, width=1.6)))
    for lvl, dash in ((entry, "dash"), (-entry, "dash"), (0, "dot")):
        fig.add_hline(y=lvl, line=dict(color=_MUTED, width=1, dash=dash))
    fig.update_layout(title="Dollar-factor disconnect — residual z-score", **_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Residual of daily XPTUSD returns after regressing out the factor set. "
        "This measures how far platinum has drifted from its usual relationship — "
        "it is a stretch measure, not a direction call."
    )


def _zar_panel(f: dict) -> None:
    inc = f.get("incremental")
    if not inc or "error" in inc:
        st.info(inc.get("error", "USDZAR series unavailable.") if inc else
                "USDZAR series unavailable.")
        return

    d_r2 = inc["delta_r2"] * 100
    max_t = inc["max_abs_t_extra"]
    verdict = "ADDS NOTHING" if (max_t < 2.0 and d_r2 < 1.0) else "WORTH A LOOK"
    colour = _MUTED if verdict == "ADDS NOTHING" else _AMBER

    c = st.columns(4)
    c[0].metric("R² base (factors only)", _fmt(inc["r2_base"] * 100, 1, "%"))
    c[1].metric("R² + USDZAR", _fmt(inc["r2_full"] * 100, 1, "%"))
    c[2].metric("Incremental R²", _fmt(d_r2, 2, "%"))
    c[3].metric("HAC |t| on ZAR", _fmt(max_t, 2))

    st.markdown(
        f"<div style='font-size:1.15rem;font-weight:600;color:{colour};"
        f"padding:4px 0 10px'>USDZAR → XPTUSD: {verdict}</div>",
        unsafe_allow_html=True)

    st.dataframe(inc["full_table"].style.format(
        {"coef": "{:+.4f}", "hac_se": "{:.4f}", "t": "{:+.2f}", "p": "{:.3f}"}),
        use_container_width=True)

    st.caption(
        f"Nested test on {inc['nobs']} observations. Base factors: "
        f"{', '.join(inc['base_cols'])}. Standard errors are Newey-West, because "
        "plain OLS errors on daily returns are too small and will over-reject. "
        "The F-test assumes iid errors — where it disagrees with the HAC t-stats, "
        "trust the t-stats."
    )


def _leadlag(f: dict) -> None:
    cc = f.get("leadlag")
    if cc is None or cc.empty:
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cc["lag"], y=cc["corr"], name="corr",
                         marker_color=np.where(cc["significant"], _ACCENT, _MUTED)))
    fig.add_trace(go.Scatter(x=cc["lag"], y=cc["hi"], name="95% (bootstrap)",
                             line=dict(color=_BLUE, width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=cc["lag"], y=cc["lo"], showlegend=False,
                             line=dict(color=_BLUE, width=1, dash="dot")))
    fig.update_layout(title="Lead-lag: XPTUSD vs USDZAR", **_LAYOUT)
    fig.update_xaxes(title_text="lag (days) — positive = platinum leads rand")
    st.plotly_chart(fig, use_container_width=True)

    sig = cc[cc["significant"]]
    if sig.empty:
        st.caption("No lag survives the block-bootstrap bands. Expected result.")
    else:
        lead = sig[sig["lag"] > 0]["lag"].tolist()
        lagd = sig[sig["lag"] < 0]["lag"].tolist()
        st.caption(
            f"Significant lags — platinum leads: {lead or 'none'}; "
            f"rand leads: {lagd or 'none'}. Bands come from a moving-block "
            "bootstrap that preserves autocorrelation; naive ±1.96/√n bands "
            "would be far too tight here."
        )


def _ratios(f: dict) -> None:
    fig = go.Figure()
    added = False
    for key, name, col in (("pt_au", "Pt/Au", _ACCENT), ("pt_pd", "Pt/Pd", _AMBER)):
        s = f.get(key)
        if s is not None and len(s):
            fig.add_trace(go.Scatter(x=s.index, y=zscore(s, window=252),
                                     name=f"{name} (z)", line=dict(color=col, width=1.6)))
            added = True
    if not added:
        return
    for lvl in (2, -2, 0):
        fig.add_hline(y=lvl, line=dict(color=_MUTED, width=1,
                                       dash="dot" if lvl == 0 else "dash"))
    fig.update_layout(title="Cross-metal ratios (1y rolling z)", **_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def _cot(cot: pd.DataFrame) -> None:
    if cot.empty:
        st.info(f"No COT rows for platinum (code {COT_PLATINUM}). "
                "Add the code to your existing COT ingestion job.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cot["report_date"], y=cot["net"],
                             name="Non-comm net", line=dict(color=_ACCENT, width=1.8)))
    fig.add_hline(y=0, line=dict(color=_MUTED, width=1, dash="dot"))
    fig.update_layout(title="CFTC managed-money net position — platinum", **_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    z = zscore(cot.set_index("report_date")["net"], window=156)
    st.caption(f"Positioning z (3y): {_fmt(_last(z), 2)}. "
               "COT is weekly and published with a 3-day lag — treat it as slow "
               "context, never as a timing trigger.")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def render(engine) -> None:
    st.subheader("Platinum Monitor")

    c1, c2, c3 = st.columns([1, 1, 1])
    lookback = c1.selectbox("Lookback", [365, 730, 1095, 1825], index=1,
                            format_func=lambda d: f"{d // 365}y")
    window = c2.slider("Factor window (days)", 60, 250, 120, 10)
    entry = c3.slider("Disconnect threshold (z)", 1.0, 3.0, 2.0, 0.25)

    prices = load_prices(engine, lookback)
    if prices.attrs.get("synthetic"):
        st.warning(
            "SYNTHETIC DATA — the collector has not backfilled the ohlc table. "
            "Nothing on this page reflects the live market.", icon="⚠️")

    f = compute_factors(prices, window=window, z_window=60)
    if SYM_PT not in prices.columns:
        st.error(f"No {SYM_PT} rows found. Check the collector.")
        return

    _header(prices, f)
    st.divider()

    tabs = st.tabs(["Price & margin", "Dollar disconnect", "Rand test",
                    "Lead-lag", "Ratios", "COT"])
    with tabs[0]:
        _price_chart(prices, f)
        st.caption(
            "XPTZAR is the series that drives South African mining economics: "
            "revenue is dollar-denominated, costs are rand-denominated. A firm "
            "rand against flat dollar platinum compresses local margin even when "
            "the USD chart looks healthy."
        )
    with tabs[1]:
        _disconnect(f, entry)
    with tabs[2]:
        _zar_panel(f)
    with tabs[3]:
        _leadlag(f)
    with tabs[4]:
        _ratios(f)
    with tabs[5]:
        _cot(load_cot(engine, lookback))
