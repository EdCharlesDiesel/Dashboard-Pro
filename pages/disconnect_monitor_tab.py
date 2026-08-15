"""
disconnect_monitor_tab.py
=========================
Empirical test of the "this disconnect should NOT happen" thesis, over **every
tradable instrument**, not just the four Bravos-style macro charts.

Two modes:
  • Instrument — pick any registry pair/metal; the page shows its disconnect vs
    each economically-relevant macro driver (dollar, US rates, real yield,
    gold, oil, risk) from src/services/disconnect_drivers.py. The pair is
    tradable, so a stretched + ALIVE disconnect escalates to a trade_setups
    signal.
  • Macro board — the original four fixed relationships (real yields vs
    DXY/gold/commodities/Nasdaq); only gold maps to a tradable pair.

Method, per relationship (A = driver, B = asset):
  1. Rolling correlation of daily changes  -> is the relationship alive?
  2. Rolling OLS: B_norm = a + b * A_norm over a 2y window
     -> residual = how far B sits from where A "says" it should be
     -> residual z-score = the DISCONNECT SCORE
  3. Ornstein-Uhlenbeck fit on the residual -> half-life = does the gap
     revert on a tradeable horizon, or is it a multi-year regime story?
  4. Event study: every past day |z| crossed the threshold, what did
     the asset do over the next 5/20/60 trading days?
     -> if returns after past disconnects are ~random, the "massive
        opportunity" is a story, not an edge.

Data: FRED for real yields / US 2Y (needs FRED_API_KEY in st.secrets or env),
the canonical OHLC spine (src/services/market_data.py) for market prices.

NOTE ON THE REAL YIELD SERIES: FRED has no 2-year TIPS constant maturity.
Default proxy is DFII5 (5y real yield); switch to a DGS2-minus-T5YIE spread in
the sidebar if you want something shorter-dated.

Entry point: render_disconnect_monitor_tab()
Standalone:  streamlit run pages/disconnect_monitor_tab.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.core.quant_models import fit_ou
from src.services.alert_service import NotifyCache
from src.services.fred_data import fred_series
from src.services.market_data import daily_ohlc
from src.services.signal_store import persist_signals
from src.services.tool_log import log_tool_usage


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

# Only assets with a tradable registry pair get a persist_signals row — the
# macro board's DX-Y.NYB/DBC/^NDX have none, so those stay audit-only
# (tool_usage_log). In Instrument mode the chosen asset IS a registry pair, so
# it always maps.
ASSET_TO_REGISTRY_PAIR = {"GC=F": "XAU/USD"}

CORR_WINDOW = 60      # days, rolling correlation of changes
BETA_WINDOW = 504     # ~2 years, rolling regression window
Z_THRESHOLD = 2.0
HORIZONS = (5, 20, 60)

# OU half-life gate: the residual technically reverts, but only trade it when
# half of the gap decays within a sane holding period. ~120 trading days
# (~6 months) is generous for these slow macro relationships; a disconnect
# that reverts slower than this is a regime story, not an entry.
OU_MAX_HALF_LIFE = 120


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_fred(series_id: str, start: str) -> pd.Series:
    """FRED observations from `start`, through the canonical macro spine.

    Postgres when the worker holds the series, HTTP otherwise — and the spine
    persists that fetch, so the next reader is local. `start` stays inclusive.

    No API key needed any more: the spine's fallback uses FRED's public CSV
    endpoint, so the macro board renders on a machine with no FRED_API_KEY
    where this previously raised.
    """
    return fred_series(series_id, start=start).rename(series_id)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_yf(ticker: str, start: str) -> pd.Series:
    """Daily closes from `start`, through the canonical spine.

    The spine takes a `period`, not a start date, so the span is converted to
    days and the frame trimmed back to `start` — the residual-z and OU
    half-life fits below read the whole window, so its left edge has to be
    what the caller asked for, not near it.

    The index is forced tz-naive conditionally: `tz_localize(None)` raises on
    an already-naive index, and the spine returns either depending on whether
    the bars came from Postgres or straight from Yahoo.
    """
    begin = pd.Timestamp(start)
    # +7d of slack: `period="Nd"` measures back from today and can land just
    # inside `begin`, dropping a bar sitting exactly on it (measured: the
    # 2024-01-01 bar vanished at an exact 957d span). Over-fetch a little and
    # let the trim below define the left edge, not the fetch window.
    days = max((pd.Timestamp.today().normalize() - begin).days, 1) + 7
    df = daily_ohlc(ticker, period=f"{days}d")
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].dropna()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    return s[s.index >= begin]


def real_yield_series(start: str, mode: str) -> pd.Series:
    """'DFII5' = 5y TIPS real yield. 'DGS2-T5YIE' = 2y nominal minus
    5y breakeven, a rough short-dated real-rate proxy."""
    if mode == "DFII5":
        return fetch_fred("DFII5", start)
    dgs2 = fetch_fred("DGS2", start)
    bei = fetch_fred("T5YIE", start)
    return (dgs2 - bei).dropna().rename("REAL2Y_PROXY")


def _driver_series(key: str, start: str, ry_mode: str) -> pd.Series:
    """Resolve a driver key (src/services/disconnect_drivers.DRIVERS) to a live
    series. The real-yield driver honours the sidebar's series toggle."""
    from src.services.disconnect_drivers import DRIVERS
    d = DRIVERS[key]
    if d["source"] == "FRED":
        if d["ident"] == "REAL_YIELD":
            return real_yield_series(start, ry_mode)
        return fetch_fred(d["ident"], start)
    return fetch_yf(d["ident"], start)


# ---------------------------------------------------------------------------
# Disconnect math
# ---------------------------------------------------------------------------
def disconnect_frame(driver: pd.Series, asset: pd.Series) -> pd.DataFrame:
    """Align, normalise, rolling corr + rolling-beta residual z-score."""
    # sort=True keeps the union index chronological (the rolling math depends on
    # it) and silences pandas' future-default deprecation warning.
    df = pd.concat([driver.rename("A"), asset.rename("B")], axis=1, sort=True).dropna()
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
# Shared relationship renderer (used by both modes)
# ---------------------------------------------------------------------------
def _render_relationship(
    label: str,
    driver: pd.Series,
    asset: pd.Series,
    expected_sign: int,
    note: str,
    thr: float,
    reg_pair: str | None,
    expanded: bool = False,
) -> dict | None:
    """Render one driver→asset disconnect card and return a summary row.

    ``reg_pair`` is the tradable registry pair to escalate to (None = audit-only).
    Identical math/UI for the macro board and the per-instrument mode — the only
    difference is whether the asset maps to a tradable pair.
    """
    with st.expander(label, expanded=expanded):
        st.caption(note)
        if driver is None or driver.empty or asset is None or asset.empty:
            st.warning("Data unavailable for this relationship.")
            return None

        df = disconnect_frame(driver, asset)
        if df.empty:
            st.warning("Not enough overlapping history.")
            return None

        z_now = float(df["z"].iloc[-1])
        corr_now = float(df["corr"].iloc[-1])
        regime = ("ALIVE" if np.sign(corr_now) == expected_sign
                  and abs(corr_now) > 0.2 else "BROKEN")

        # Fit an Ornstein-Uhlenbeck process to the disconnect RESIDUAL itself
        # (fit_ou wants a stationary series — the rolling-regression residual is
        # exactly that, never raw price). This puts an expected HOLDING PERIOD on
        # the gap: half_life = trading days for half the disconnect to decay back
        # to its mean. Degenerate / no-reversion fits come back with
        # half_life = inf and are treated as untradeable.
        try:
            ou = fit_ou(df["resid"])
        except Exception:
            ou = None
        ou_ok = bool(ou is not None and np.isfinite(ou.half_life)
                     and ou.half_life <= OU_MAX_HALF_LIFE)

        m1, m2, m3 = st.columns(3)
        m1.metric("Disconnect z now", f"{z_now:+.2f}")
        m2.metric(f"{CORR_WINDOW}d corr (changes)", f"{corr_now:+.2f}")
        m3.metric("Regime check", regime)

        m4, m5, m6 = st.columns(3)
        m4.metric("OU half-life (days)",
                  f"{ou.half_life:.0f}"
                  if (ou is not None and np.isfinite(ou.half_life)) else "—")
        m5.metric("OU z-score", f"{ou.zscore:+.2f}" if ou is not None else "—")
        m6.metric("Reverts in time?", "YES" if ou_ok else "NO")
        st.caption(
            f"OU half-life = trading days for half of this disconnect to decay "
            f"back toward its mean (fit on the residual, not price). 'Reverts in "
            f"time?' is YES only when the fit shows genuine mean reversion with "
            f"half-life ≤ {OU_MAX_HALF_LIFE}d — the extra gate that stops you "
            f"fading a gap which technically closes but takes years. The OU "
            f"z-score is an independent read on the same residual (vs its "
            f"equilibrium std) to cross-check the rolling z above."
        )

        # Audit log (deduped per label+date; this loop reruns on every widget touch).
        _latest_date = df.index[-1]
        _dm_key = f"{label}|{_latest_date}"
        if NotifyCache("disconnect_monitor_log").filter_new([_dm_key]):
            log_tool_usage("disconnect_mon", {
                "pair_name": label, "date": str(_latest_date),
                "z_now": z_now, "corr_now": corr_now, "regime": regime,
                "expected_sign": expected_sign,
                "ou_half_life": (float(ou.half_life) if ou is not None
                                 and np.isfinite(ou.half_life) else None),
                "ou_zscore": (float(ou.zscore) if ou is not None else None),
                "ou_tradeable": ou_ok,
            })

        # Escalate to a trade_setups signal only when there's a tradable pair
        # AND the disconnect clears the threshold AND the relationship is still
        # ALIVE (fading a BROKEN regime is exactly the trap this page warns
        # against — see the caption above). OU half-life is the third gate:
        # only fade a disconnect that actually reverts within a sane horizon.
        _hl = ou.half_life if (ou is not None
                               and np.isfinite(ou.half_life)) else None
        if reg_pair and regime == "ALIVE" and abs(z_now) >= thr and ou_ok:
            _hl_txt = f", OU half-life {_hl:.0f}d" if _hl is not None else ""
            persist_signals("disconnect_mon", [{
                "pair": reg_pair,
                "bias": "SHORT" if z_now > 0 else "LONG",
                "entry": float(df["B"].iloc[-1]),
                "bar_time": df.index[-1],
                "strength_score": None,
                "conviction": (f"|z|={abs(z_now):.2f} (thr {thr:.2f})" + _hl_txt),
                "thesis": (f"Disconnect Monitor — {label}: z={z_now:+.2f}, "
                           f"regime ALIVE (corr={corr_now:+.2f}){_hl_txt}, "
                           f"fading the {'rich' if z_now > 0 else 'cheap'} side."),
            }])
            st.success(f"📌 Trade signal saved: {reg_pair} "
                       f"{'SHORT' if z_now > 0 else 'LONG'} (fading the "
                       f"{'rich' if z_now > 0 else 'cheap'} side).")
        elif reg_pair and regime == "ALIVE" and abs(z_now) >= thr and not ou_ok:
            st.caption(
                "⚠️ Disconnect is stretched and the regime is ALIVE, but the OU "
                f"fit shows no tradeable mean reversion (half-life too long / "
                f"none, > {OU_MAX_HALF_LIFE}d) — no trade signal escalated. The "
                "gap may still close eventually, just not on a tradeable horizon.")
        elif reg_pair and abs(z_now) >= thr and regime == "BROKEN":
            st.caption(
                "⚠️ Stretched, but the driver→asset correlation no longer holds "
                "its expected sign (regime BROKEN) — fading this is exactly the "
                "trap this page warns against. No signal escalated.")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.45], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=df.index, y=df["A_n"], name="driver (norm)",
                                 line=dict(color="#1f77b4")), 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df["B_n"], name="asset (norm)",
                                 yaxis="y2", line=dict(color="#d62728")), 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df["z"], name="disconnect z",
                                 line=dict(color="#2ca02c")), 2, 1)
        fig.add_hline(y=thr, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=-thr, line_dash="dot", line_color="red", row=2, col=1)
        fig.update_layout(height=460, legend=dict(orientation="h"),
                          margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        ev = event_study(df, thr)
        if ev.empty:
            st.info("No historical disconnects at this threshold.")
        else:
            st.markdown("**Event study — what happened after past disconnects:**")
            st.dataframe(ev.iloc[::-1], use_container_width=True, hide_index=True)
            rev_cols = [c for c in ev.columns if c.startswith("reversion")]
            stats = {c: f"{ev[c].mean():+.2f}% (hit {100 * (ev[c] > 0).mean():.0f}%)"
                     for c in rev_cols if ev[c].notna().any()}
            st.write({"mean reversion pnl by horizon": stats})
            st.caption(
                "Reversion pnl assumes you fade the gap (short the rich side). "
                "Mean near zero or hit-rate ~50% = the disconnect is a regime "
                "change, not an opportunity.")

        return {"relationship": label, "z_now": round(z_now, 2),
                "corr_now": round(corr_now, 2), "regime": regime,
                "reverts": "YES" if ou_ok else "NO"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_disconnect_monitor_tab() -> None:
    from src.instruments.registry import INSTRUMENTS
    from src.services.disconnect_drivers import driver_label, drivers_for

    st.header("🔌 Disconnect Monitor — do 'impossible' gaps actually close?")
    st.caption(
        "Disconnect score = z-score of an asset's residual vs a rolling 2-year "
        "regression on its macro driver. Pick any instrument to see its "
        "disconnect vs the dollar, rates, real yields, gold, oil or risk — or "
        "switch to the Macro board for the original real-yield charts. The event "
        "study answers the only question that matters: did fading past "
        "disconnects make money?"
    )

    c0, c1, c2, c3 = st.columns([1.4, 1, 1, 1])
    with c0:
        mode = st.radio("Mode", ["💱 Instrument", "🌍 Macro board"],
                        horizontal=True)
    with c1:
        years = st.slider("History (years)", 6, 20, 12)
    with c2:
        ry_mode = st.selectbox("Real-yield series", ["DFII5", "DGS2-T5YIE"], index=0)
    with c3:
        thr = st.slider("Disconnect threshold |z|", 1.0, 3.0, Z_THRESHOLD, 0.25)

    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    summary: list[dict] = []

    if mode == "💱 Instrument":
        pair = st.selectbox("Instrument", sorted(INSTRUMENTS.keys()))
        rels = drivers_for(pair)
        if not rels:
            st.info(f"No macro driver is mapped for {pair} yet.")
            return
        st.caption(f"**{pair}** — {len(rels)} macro relationship"
                   f"{'s' if len(rels) != 1 else ''}. A stretched, ALIVE "
                   "disconnect escalates to a trade signal.")
        asset = fetch_yf(INSTRUMENTS[pair]["ticker"], start)
        if asset.empty:
            st.error(f"Price data unavailable for {pair}.")
            return
        for i, rel in enumerate(rels):
            label = f"{pair} vs {driver_label(rel.driver)}"
            try:
                driver = _driver_series(rel.driver, start, ry_mode)
            except Exception as exc:
                with st.expander(label):
                    st.warning(f"Driver unavailable ({exc}).")
                continue
            row = _render_relationship(label, driver, asset, rel.expected_sign,
                                       rel.note, thr, reg_pair=pair,
                                       expanded=(i == 0))
            if row:
                summary.append(row)
    else:
        for i, (pair_name, cfg) in enumerate(PAIRS.items()):
            src, ident = cfg["driver"]
            try:
                driver = (real_yield_series(start, ry_mode)
                          if ident == "REAL_YIELD" else fetch_yf(ident, start))
            except Exception as exc:
                with st.expander(pair_name):
                    st.warning(f"Driver unavailable ({exc}).")
                continue
            asset = fetch_yf(cfg["asset"][1], start)
            reg_pair = ASSET_TO_REGISTRY_PAIR.get(cfg["asset"][1])
            row = _render_relationship(pair_name, driver, asset,
                                       cfg["expected_sign"], cfg["note"], thr,
                                       reg_pair=reg_pair, expanded=(i == 0))
            if row:
                summary.append(row)

    if summary:
        st.subheader("Cockpit summary")
        st.dataframe(pd.DataFrame(summary), use_container_width=True,
                     hide_index=True)
        st.caption(
            "Interpretation guide: big |z| with regime ALIVE **and reverts YES** "
            "= tension that historically resolved on a tradeable horizon (this "
            "is what escalates to a signal). Big |z| with regime BROKEN = the "
            "relationship itself changed — the gap can widen for months. "
            "Reverts NO = it may close eventually, but too slowly to trade. "
            "Check whether the event study agrees.")


# ===========================================================================
# PAGE ENTRY — multipage (auto-run) + standalone (streamlit run)
# ===========================================================================
# Guarded behind __main__ (Streamlit runs each page script as __main__) so the
# pure math above (real_yield_series, fetch_yf, disconnect_frame) stays
# importable headlessly — same contract as pages/cot_signals.py and
# pages/cot_composite_trade_signal.py. src/services/swing_playbook_service.py
# relies on this to reuse the gold disconnect-z math without re-running this
# page's full UI (and colliding on a second set_page_config()).

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
        st.caption("Every instrument vs its macro driver")
        st.divider()
        render_sidebar_nav()
    render_disconnect_monitor_tab()


if __name__ == "__main__":
    _page()
