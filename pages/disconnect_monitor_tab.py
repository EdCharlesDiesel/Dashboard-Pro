"""
disconnect_monitor_tab.py
=========================
Empirical test of the "this disconnect should NOT happen" thesis
(Bravos-style charts: real yields vs DXY, real yields vs gold,
DXY vs commodities, real yields vs Nasdaq).

Method, per pair (A = driver, B = asset):
  1. Rolling correlation of daily changes  -> is the relationship alive?
  2. Rolling OLS: B_norm = a + b * A_norm over a 2y window
     -> residual = how far B sits from where A "says" it should be
     -> residual z-score = the DISCONNECT SCORE
  3. Event study: every past day |z| crossed the threshold, what did
     the asset do over the next 5/20/60 trading days?
     -> if returns after past disconnects are ~random, the "massive
        opportunity" is a story, not an edge.

Data: FRED for real yields (needs FRED_API_KEY in st.secrets or env),
yfinance for market prices.

NOTE ON THE REAL YIELD SERIES: FRED has no 2-year TIPS constant
maturity. Default proxy is DFII5 (5y real yield); switch to a
DGS2-minus-T5YIE spread in the sidebar if you want something
shorter-dated. The shape (2022 surge, 2025-26 rise) is the same.

Entry point: render_disconnect_monitor_tab()
Standalone:  streamlit run pages/disconnect_monitor_tab.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from src.core.secrets import fred_api_key
from src.services.alert_service import NotifyCache
from src.services.signal_store import persist_signals
from src.services.tool_log import log_tool_usage

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

PAIRS = {
    "Real yield vs DXY": {
        "driver": ("FRED", "REAL_YIELD"), "asset": ("YF", "DX-Y.NYB"),
        "expected_sign": +1,
        "note": "2022 regime: real yields up -> dollar up.",
    },
    "Real yield vs Gold": {
        "driver": ("FRED", "REAL_YIELD"), "asset": ("YF", "GC=F"),
        "expected_sign": -1,
        "note": "Classic: real yields up -> gold down. Broken since ~2022 "
                "(central bank buying / fiscal premium).",
    },
    "DXY vs Commodities (DBC)": {
        "driver": ("YF", "DX-Y.NYB"), "asset": ("YF", "DBC"),
        "expected_sign": -1,
        "note": "Dollar up -> commodities down, classically.",
    },
    "Real yield vs Nasdaq 100": {
        "driver": ("FRED", "REAL_YIELD"), "asset": ("YF", "^NDX"),
        "expected_sign": -1,
        "note": "Higher discount rates should compress long-duration tech.",
    },
}

# Only assets with a tradable registry pair get a persist_signals row (see
# render loop below) — DX-Y.NYB/DBC/^NDX have none, so those three stay
# audit-only (tool_usage_log), matching cot_composite_trade_signal's
# "skip when there's no tradable registry pair" rule.
ASSET_TO_REGISTRY_PAIR = {"GC=F": "XAU/USD"}

CORR_WINDOW = 60      # days, rolling correlation of changes
BETA_WINDOW = 504     # ~2 years, rolling regression window
Z_THRESHOLD = 2.0
HORIZONS = (5, 20, 60)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_fred(series_id: str, start: str) -> pd.Series:
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


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_yf(ticker: str, start: str) -> pd.Series:
    px = yf.download(ticker, start=start, interval="1d",
                     auto_adjust=False, progress=False)
    if px.empty:
        return pd.Series(dtype=float)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    s = px["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def real_yield_series(start: str, mode: str) -> pd.Series:
    """'DFII5' = 5y TIPS real yield. 'DGS2-T5YIE' = 2y nominal minus
    5y breakeven, a rough short-dated real-rate proxy."""
    if mode == "DFII5":
        return fetch_fred("DFII5", start)
    dgs2 = fetch_fred("DGS2", start)
    bei = fetch_fred("T5YIE", start)
    return (dgs2 - bei).dropna().rename("REAL2Y_PROXY")


# ---------------------------------------------------------------------------
# Disconnect math
# ---------------------------------------------------------------------------
def disconnect_frame(driver: pd.Series, asset: pd.Series) -> pd.DataFrame:
    """Align, normalise, rolling corr + rolling-beta residual z-score."""
    df = pd.concat([driver.rename("A"), asset.rename("B")], axis=1).dropna()
    if len(df) < BETA_WINDOW + 60:
        return pd.DataFrame()

    # normalise: yields stay in %-points, prices go to log
    df["A_n"] = df["A"] if df["A"].abs().max() < 50 else np.log(df["A"])
    df["B_n"] = np.log(df["B"]) if (df["B"] > 0).all() else df["B"]

    df["corr"] = df["A_n"].diff().rolling(CORR_WINDOW).corr(df["B_n"].diff())

    # rolling OLS of B_n on A_n -> residual z
    a, b = df["A_n"], df["B_n"]
    ra_m = a.rolling(BETA_WINDOW).mean()
    rb_m = b.rolling(BETA_WINDOW).mean()
    cov = (a * b).rolling(BETA_WINDOW).mean() - ra_m * rb_m
    var = (a * a).rolling(BETA_WINDOW).mean() - ra_m ** 2
    beta = cov / var
    alpha = rb_m - beta * ra_m
    df["resid"] = b - (alpha + beta * a)
    r_std = df["resid"].rolling(BETA_WINDOW).std()
    df["z"] = df["resid"] / r_std
    df["beta"] = beta
    return df.dropna(subset=["z"])


def event_study(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Forward asset returns after |z| first crosses the threshold.
    Crossings only (not every day above), to avoid double counting."""
    above = df["z"].abs() >= threshold
    crossings = above & ~above.shift(1, fill_value=False)
    events = df.index[crossings]
    rows = []
    for t in events:
        i = df.index.get_loc(t)
        sign = np.sign(df["z"].iloc[i])
        row = {"date": t.date().isoformat(),
               "z": round(float(df["z"].iloc[i]), 2)}
        for h in HORIZONS:
            if i + h < len(df):
                fwd = df["B"].iloc[i + h] / df["B"].iloc[i] - 1.0
                # convergence return: if asset is RICH (z>0) reversion
                # means it falls, so flip the sign for "reversion pnl"
                row[f"fwd_{h}d_%"] = round(100 * fwd, 2)
                row[f"reversion_pnl_{h}d_%"] = round(-100 * sign * fwd, 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_disconnect_monitor_tab() -> None:
    st.header("🔌 Disconnect Monitor — do 'impossible' gaps actually close?")
    st.caption(
        "Quantifies the Bravos-style divergence charts. Disconnect score = "
        "z-score of the asset's residual vs a rolling 2-year regression on "
        "its driver. The event study answers the only question that matters: "
        "did fading past disconnects make money?"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        years = st.slider("History (years)", 6, 20, 12)
    with c2:
        ry_mode = st.selectbox("Real-yield series",
                               ["DFII5", "DGS2-T5YIE"], index=0)
    with c3:
        thr = st.slider("Disconnect threshold |z|", 1.0, 3.0,
                        Z_THRESHOLD, 0.25)

    start = (pd.Timestamp.today()
             - pd.DateOffset(years=years)).strftime("%Y-%m-%d")

    try:
        ry = real_yield_series(start, ry_mode)
    except Exception as exc:
        st.error(f"FRED fetch failed: {exc}")
        return

    summary = []
    for pair_name, cfg in PAIRS.items():
        with st.expander(pair_name, expanded=(pair_name ==
                                              "Real yield vs DXY")):
            st.caption(cfg["note"])
            src, ident = cfg["driver"]
            driver = ry if ident == "REAL_YIELD" else fetch_yf(ident, start)
            asset = fetch_yf(cfg["asset"][1], start)
            if driver.empty or asset.empty:
                st.warning("Data unavailable for this pair.")
                continue

            df = disconnect_frame(driver, asset)
            if df.empty:
                st.warning("Not enough overlapping history.")
                continue

            z_now = float(df["z"].iloc[-1])
            corr_now = float(df["corr"].iloc[-1])
            regime = ("ALIVE" if np.sign(corr_now) == cfg["expected_sign"]
                      and abs(corr_now) > 0.2 else "BROKEN")
            m1, m2, m3 = st.columns(3)
            m1.metric("Disconnect z now", f"{z_now:+.2f}")
            m2.metric(f"{CORR_WINDOW}d corr (changes)", f"{corr_now:+.2f}")
            m3.metric("Regime check", regime)

            # Log this read to Postgres (audit trail). Deduped per pair+date
            # via NotifyCache — this loop reruns every widget touch.
            _latest_date = df.index[-1]
            _dm_key = f"{pair_name}|{_latest_date}"
            if NotifyCache("disconnect_monitor_log").filter_new([_dm_key]):
                log_tool_usage("disconnect_mon", {
                    "pair_name": pair_name, "date": str(_latest_date),
                    "z_now": z_now, "corr_now": corr_now, "regime": regime,
                    "expected_sign": cfg["expected_sign"],
                })

            # Only escalate to an actual trade_setups signal when there's a
            # tradable registry pair AND the disconnect clears the user's
            # threshold AND the driver/asset relationship is still ALIVE
            # (fading a BROKEN regime is exactly the trap this page warns
            # against — see the caption above).
            _asset_ticker = cfg["asset"][1]
            _reg_pair = ASSET_TO_REGISTRY_PAIR.get(_asset_ticker)
            if _reg_pair and regime == "ALIVE" and abs(z_now) >= thr:
                persist_signals("disconnect_mon", [{
                    "pair": _reg_pair,
                    "bias": "SHORT" if z_now > 0 else "LONG",
                    "entry": float(df["B"].iloc[-1]),
                    "strength_score": None,
                    "conviction": f"|z|={abs(z_now):.2f} (thr {thr:.2f})",
                    "thesis": (f"Disconnect Monitor — {pair_name}: "
                              f"z={z_now:+.2f}, regime ALIVE (corr={corr_now:+.2f}), "
                              f"fading the {'rich' if z_now > 0 else 'cheap'} side."),
                }])

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.55, 0.45],
                                vertical_spacing=0.06)
            fig.add_trace(go.Scatter(x=df.index, y=df["A_n"],
                                     name="driver (norm)",
                                     line=dict(color="#1f77b4")), 1, 1)
            fig.add_trace(go.Scatter(x=df.index, y=df["B_n"],
                                     name="asset (norm)", yaxis="y2",
                                     line=dict(color="#d62728")), 1, 1)
            fig.add_trace(go.Scatter(x=df.index, y=df["z"],
                                     name="disconnect z",
                                     line=dict(color="#2ca02c")), 2, 1)
            fig.add_hline(y=thr, line_dash="dot", line_color="red",
                          row=2, col=1)
            fig.add_hline(y=-thr, line_dash="dot", line_color="red",
                          row=2, col=1)
            fig.update_layout(height=460, legend=dict(orientation="h"),
                              margin=dict(t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            ev = event_study(df, thr)
            if ev.empty:
                st.info("No historical disconnects at this threshold.")
            else:
                st.markdown("**Event study — what happened after past "
                            "disconnects:**")
                st.dataframe(ev.iloc[::-1], use_container_width=True,
                             hide_index=True)
                rev_cols = [c for c in ev.columns
                            if c.startswith("reversion")]
                stats = {c: f"{ev[c].mean():+.2f}% "
                            f"(hit {100 * (ev[c] > 0).mean():.0f}%)"
                         for c in rev_cols if ev[c].notna().any()}
                st.write({"mean reversion pnl by horizon": stats})
                st.caption(
                    "Reversion pnl assumes you fade the gap (short the rich "
                    "side). Mean near zero or hit-rate ~50% = the disconnect "
                    "is a regime change, not an opportunity."
                )

            summary.append({"pair": pair_name, "z_now": round(z_now, 2),
                            "corr_now": round(corr_now, 2)})

    if summary:
        st.subheader("Cockpit summary")
        st.dataframe(pd.DataFrame(summary), use_container_width=True,
                     hide_index=True)
        st.caption(
            "Interpretation guide: big |z| with regime ALIVE = tension that "
            "historically resolved (tradeable with your regime filter). Big "
            "|z| with regime BROKEN = the relationship itself changed — the "
            "gap can widen for months. The reel's charts are the first case "
            "visually, but check whether the event study agrees."
        )


# ===========================================================================
# PAGE ENTRY — multipage (auto-run) + standalone (streamlit run)
# ===========================================================================

def _page() -> None:
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="DISC · Disconnect Monitor",
        page_icon="🔌",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()
    with st.sidebar:
        st.markdown("### 🔌 Disconnect Monitor")
        st.caption("Real yields vs DXY/gold/Nasdaq divergence")
        st.divider()
        render_sidebar_nav()
    render_disconnect_monitor_tab()


_page()