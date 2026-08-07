"""
martingale_tab.py
=================
Martingale & concentration diagnostics — is the equity curve edge, or luck?

Renders `src/core/concentration.py` (pure numpy/scipy, unit-tested) against
the *real* closed-trade log:

  Doob decomposition   — split cumulative P&L into a predictable "compensator"
                         (edge) and a martingale remainder (luck).
  Azuma / Freedman     — bound how far the luck component can plausibly wander.
                         Azuma assumes every trade could be a maximal loser;
                         Freedman adapts to realised variance and is far tighter.
  Martingale tests     — t-test on drift, Ljung-Box, variance ratio.
  McDiarmid            — how much can ONE trade move a reported Sharpe?
  Optional stopping    — why "I'll exit when I'm up" is not an edge.

**Increments are R-multiples**, not currency: `trade_setups.r_multiple` on
closed trades. R is the account-size-independent unit the rest of the app
already reports (the journal's equity curve is in R), so the two agree and the
bounds stay meaningful as the balance changes.

Only **executed trades** feed this — `EXECUTED_SOURCES` from the repository,
never the auto-saved page signals, for the same reason the Trade Journal
separates them: signals have no outcome and would swamp the sample.

Audit-only: a diagnostic read, never a directional pair+bias call, so it logs
to `tool_usage_log` and never writes `trade_setups`.

Entry point: render_martingale_tab()
"""
from __future__ import annotations

import streamlit as st

from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="MTGL · Martingale Diagnostics",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

import numpy as np
import pandas as pd

from src.core.concentration import (
    azuma_band,
    azuma_risk_budget,
    bounded_differences_bound,
    doob_decomposition,
    freedman_band,
    martingale_tests,
    optional_stopping_mc,
)
from src.db.trade_repository import EXECUTED_SOURCES
from src.pages_lib.navigation import render_sidebar_nav
from src.ui.charts import ChartKit
from src.ui.components import MetricCell, render_metric_row
from src.ui.theme import BloombergTheme as T

MIN_TRADES = 30          # below this the bounds say almost nothing
SYNTHETIC_N = 240


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Loading closed trades…")
def load_increments(instrument: str = "ALL") -> tuple:
    """Closed executed trades as an R-multiple series, newest last.

    Returns ``(increments, label, is_synthetic)``. Falls back to a synthetic
    series — clearly flagged — so the page is explorable before the journal has
    enough history, which is the normal state early on.
    """
    try:
        from contextlib import closing

        from src.db.cache import pooled_repository
        from src.db.market_cache import _resolve_cfg
        cfg = _resolve_cfg()
        if cfg is not None:
            repo = pooled_repository(cfg)
            # Project idiom: closing() returns the leased connection to the
            # pool, the middle `conn` handles commit/rollback. Dropping the
            # closing() leaks a pooled connection on every rerun.
            with closing(repo._connect()) as conn, conn, conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT r_multiple
                    FROM trade_setups
                    WHERE is_open = false
                      AND r_multiple IS NOT NULL
                      AND COALESCE(source, 'checklist') IN %s
                      AND (%s = 'ALL' OR instrument = %s)
                    ORDER BY logged_at
                    """,
                    (tuple(EXECUTED_SOURCES), instrument, instrument),
                )
                rows = [float(r[0]) for r in cur.fetchall()]
            if len(rows) >= MIN_TRADES:
                return tuple(rows), f"{len(rows)} closed trades (R)", False
            real_n = len(rows)
        else:
            real_n = 0
    except Exception:
        real_n = 0

    rng = np.random.default_rng(11)
    # A small genuine edge buried in fat-tailed noise — so the diagnostics have
    # something to actually find, and you can see what "edge" looks like here.
    pnl = rng.standard_t(df=4, size=SYNTHETIC_N) * 0.9 + 0.11
    return (tuple(float(v) for v in pnl),
            f"SYNTHETIC ({real_n} real closed trades — need {MIN_TRADES})", True)


@st.cache_data(ttl=600, show_spinner=False)
def compute(increments: tuple, delta: float, method: str, window: int) -> dict:
    """All the maths, memoised on a hashable tuple of increments."""
    x = np.asarray(increments, dtype=float)
    R = float(np.max(np.abs(x)))
    return {
        "doob": doob_decomposition(x, method=method, window=window),
        "azuma": azuma_band(c=R, n=x.size, delta=delta),
        "freedman": freedman_band(x, delta=delta, R=R),
        "tests": martingale_tests(x),
        "R": R,
    }


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
def render_martingale_tab() -> None:
    st.markdown("## 🎲 Martingale & Concentration — edge or luck?")
    st.caption(
        "Doob's decomposition splits the equity curve into a predictable part "
        "(**edge**) and a martingale part (**luck**). Azuma and Freedman then "
        "bound how far the luck component can plausibly wander, so you can tell "
        "a real run from a lucky one. Increments are **R-multiples** of closed "
        "trades — the same unit the Trade Journal's equity curve uses."
    )

    instrument = st.session_state.get("mg_instrument", "ALL")
    delta = st.session_state.get("mg_delta", 0.05)
    method = st.session_state.get("mg_method", "expanding")
    window = int(st.session_state.get("mg_window", 50))

    increments, label, synthetic = load_increments(instrument)
    x = np.asarray(increments, dtype=float)

    if synthetic:
        st.warning(
            f"⚠️ **Showing synthetic data** — {label}. Every number below "
            "describes a simulated series, not your trading. It's here so the "
            "page is explorable; treat nothing on it as a read on your edge "
            "until you have a real closed-trade history."
        )

    if x.size < MIN_TRADES:
        st.info(f"Only {x.size} increments — these bounds need ~50+ to say much.")
        return

    res = compute(increments, delta, method, window)
    doob, tests = res["doob"], res["tests"]
    share = doob.edge_share

    render_metric_row([
        MetricCell("Trades", f"{x.size:,}", "cyan"),
        MetricCell("Net R", f"{doob.cumulative[-1]:+.2f}",
                   "green" if doob.cumulative[-1] > 0 else "red"),
        MetricCell("Edge (compensator)", f"{doob.compensator[-1]:+.2f}",
                   "green" if doob.compensator[-1] > 0 else "red"),
        MetricCell("Luck (martingale)", f"{doob.martingale[-1]:+.2f}", "yellow"),
        MetricCell("Edge share",
                   f"{share:.0%}" if np.isfinite(share) else "—", "white"),
    ])

    t1, t2, t3 = st.tabs(
        ["◆ DECOMPOSITION", "📏 CONCENTRATION BANDS", "🛡️ RISK & STOPPING"])

    # ---- 1. Doob -------------------------------------------------------
    with t1:
        i = np.arange(1, x.size + 1)
        fig = ChartKit.panels(["Doob decomposition of cumulative R"])
        ChartKit.line(fig, i, doob.cumulative, "Equity (R)",
                      color=T.GREEN, width=2.4, showlegend=True)
        ChartKit.line(fig, i, doob.compensator, "Compensator (edge)",
                      color=T.CYAN, width=1.8, dash="dash", showlegend=True)
        ChartKit.line(fig, i, doob.martingale, "Martingale (luck)",
                      color=T.GREY, width=1.4, showlegend=True)
        ChartKit.hline(fig, 0, T.BORDER)
        st.plotly_chart(ChartKit.finish(fig, height=430, showlegend=True),
                        width="stretch", config=ChartKit.PLOTLY_CONFIG)

        if np.isfinite(share):
            verdict = ("Luck ran *against* the edge here — the result is worse "
                       "than the drift implies." if share > 1 else
                       "A share far below 1 means the result leans on the "
                       "unpredictable part.")
            st.markdown(
                f"**{share:.0%}** of net R is attributable to predictable drift; "
                f"the remaining **{1 - share:.0%}** is the martingale component. "
                + verdict)

        st.markdown("**Is there an edge at all?**")
        v = tests.verdict
        (st.info if "consistent" in v else st.success)(v)
        st.dataframe(pd.DataFrame({
            "Test": ["Mean increment (t-test)", "Ljung-Box (serial corr.)",
                     f"Variance ratio q={tests.vr_q}"],
            "Statistic": [f"{tests.drift_t:.3f}", f"{tests.ljung_box_stat:.2f}",
                          f"{tests.vr:.3f}"],
            "p-value": [f"{tests.drift_p:.4f}", f"{tests.ljung_box_p:.4f}",
                        f"{tests.vr_p:.4f}"],
        }), hide_index=True, width="stretch")
        st.caption(
            "⚠️ The verdict runs **three tests at 5% each with no "
            "multiple-comparison correction**, so a genuinely edgeless series "
            "is 'rejected' in roughly 12–14% of samples by chance. Treat a "
            "single rejection as a hint to look closer, not a finding — and "
            "remember that failing to reject is not proof of no edge, only "
            "failure to find one at this sample size.")

    # ---- 2. bands ------------------------------------------------------
    with t2:
        az, fr = res["azuma"], res["freedman"]
        i = np.arange(1, x.size + 1)
        fig = ChartKit.panels(["How far can the luck component wander?"])
        ChartKit.band(fig, i, az.radius, -az.radius, "rgba(120,120,120,0.10)",
                      name=f"Azuma {1 - delta:.0%}")
        ChartKit.band(fig, i, fr.radius, -fr.radius, "rgba(0,224,255,0.12)",
                      name=f"Freedman {1 - delta:.0%}")
        ChartKit.line(fig, i, doob.martingale, "Martingale component",
                      color=T.GREEN, width=2.2, showlegend=True)
        ChartKit.hline(fig, 0, T.BORDER)
        st.plotly_chart(ChartKit.finish(fig, height=430, showlegend=True),
                        width="stretch", config=ChartKit.PLOTLY_CONFIG)

        tighter = fr.radius[-1] / az.radius[-1] - 1
        render_metric_row([
            MetricCell("Azuma envelope", f"±{az.radius[-1]:.2f}R", "yellow"),
            MetricCell("Freedman envelope", f"±{fr.radius[-1]:.2f}R", "cyan"),
            MetricCell("Freedman vs Azuma", f"{tighter:+.0%}", "green"),
            MetricCell("Largest single trade", f"{res['R']:.2f}R", "white"),
        ])

        breaches = fr.breaches(doob.martingale)
        if breaches.size:
            shown = ", ".join(map(str, (breaches + 1)[:8]))
            st.warning(
                f"Martingale component escaped the Freedman band at trade(s) "
                f"{shown}{'…' if breaches.size > 8 else ''} — at "
                f"{1 - delta:.0%} this should be rare. Either the increment "
                "bound is understated, or the process is not the martingale "
                "it's being modelled as.")
        else:
            st.success(f"Path stayed inside the {1 - delta:.0%} Freedman "
                       "envelope throughout.")
        st.caption(
            "Freedman replaces Azuma's worst-case sum of squared bounds with "
            "realised quadratic variation, so it's an estimate rather than a "
            "theorem — but it's the one worth acting on, since assuming every "
            "trade is a maximal loser wastes most of the bound.")

    # ---- 3. risk & stopping -------------------------------------------
    with t3:
        st.markdown("#### Position sizing from Azuma")
        r1, r2 = st.columns(2)
        default_dd = float(max(abs(doob.cumulative).max() * 2, 5.0))
        max_dd = r1.number_input("Max tolerable drawdown (R)", 1.0, 1000.0,
                                 default_dd, step=1.0)
        horizon = r2.number_input("Trades ahead", 10, 10_000, 250, step=10)

        rb = azuma_risk_budget(max_dd, int(horizon), delta)
        render_metric_row([
            MetricCell("Max risk / trade", f"{rb['risk_per_trade']:.3f}R", "cyan"),
            MetricCell("Current avg |R|", f"{np.abs(x).mean():.3f}R", "white"),
        ])
        st.caption(
            f"Assuming **no edge**, risking more than {rb['risk_per_trade']:.3f}R "
            f"per trade lets cumulative R breach ±{max_dd:.1f}R over {horizon} "
            f"trades with probability above {delta:.0%}. Note the √n: "
            "quadrupling the horizon only doubles the envelope.")

        st.divider()
        st.markdown("#### Optional stopping — does an exit rule create edge?")
        p1, p2 = st.columns(2)
        tp = p1.number_input("Take profit (R)", 0.1, 100.0,
                             float(np.abs(x).mean() * 2), step=0.1)
        sl = p2.number_input("Stop loss (R)", 0.1, 100.0,
                             float(np.abs(x).mean() * 8), step=0.1)

        if st.button("▶ Run bootstrap", type="primary"):
            with st.spinner("Simulating…"):
                null = optional_stopping_mc(x, tp, sl, demean=True)
                live = optional_stopping_mc(x, tp, sl, demean=False)
            render_metric_row([
                MetricCell("Win rate", f"{null['win_rate']:.1%}", "green"),
                MetricCell("EV — no edge", f"{null['mean_stopped_value']:+.3f}R",
                           "yellow"),
                MetricCell("EV — your data", f"{live['mean_stopped_value']:+.3f}R",
                           "cyan"),
            ])
            st.caption(
                f"A wide stop and tight target wins {null['win_rate']:.0%} of "
                "the time on a *driftless* series and still returns zero — the "
                "rare large losses exactly offset the frequent small wins. "
                "That's the optional stopping theorem, and it's why win rate "
                "alone says nothing. The third figure applies the same rule to "
                "your actual increments: compare it with the second to see "
                "whether the rule preserves your edge or spends it.")

        st.divider()
        st.markdown("#### Metric stability (McDiarmid)")
        if st.button("▶ Bound the Sharpe ratio"):
            def sharpe(v):
                s = v.std(ddof=1)
                return float(v.mean() / s * np.sqrt(252)) if s > 0 else 0.0

            mc = bounded_differences_bound(x, sharpe, delta=delta)
            render_metric_row([
                MetricCell("Sharpe", f"{mc['metric']:.2f}", "cyan"),
                MetricCell(f"±{1 - delta:.0%} band", f"±{mc['radius']:.2f}", "yellow"),
                MetricCell("Most influential trade",
                           f"#{mc['most_influential_index'] + 1}", "white"),
                MetricCell("Moves Sharpe by",
                           f"{mc['max_single_influence']:.3f}", "red"),
            ])
            if mc["lower"] < 0 < mc["upper"]:
                st.warning(
                    "The bound straddles zero: this Sharpe is not "
                    "distinguishable from no skill at this sample size. "
                    "McDiarmid is conservative, so read this as inconclusive "
                    "rather than as a verdict.")
            else:
                st.success(f"Sharpe stays in [{mc['lower']:.2f}, "
                           f"{mc['upper']:.2f}] under single-trade perturbation.")

    # ---- audit (diagnostic read, never a trade signal) -----------------
    if not synthetic:
        try:
            from src.services.alert_service import NotifyCache
            from src.services.tool_log import log_tool_usage
            key = f"{instrument}|{x.size}|{method}|{delta}"
            if NotifyCache("martingale_log").filter_new([key]):
                log_tool_usage("martingale", {
                    "instrument": instrument, "n_trades": int(x.size),
                    "method": method, "delta": delta,
                    "net_r": float(doob.cumulative[-1]),
                    "compensator_r": float(doob.compensator[-1]),
                    "martingale_r": float(doob.martingale[-1]),
                    "edge_share": None if not np.isfinite(share) else float(share),
                    "verdict": tests.verdict,
                    "freedman_radius": float(res["freedman"].radius[-1]),
                })
        except Exception:
            pass


# ===========================================================================
# PAGE ENTRY
# ===========================================================================
with st.sidebar:
    st.markdown("### 🎲 Martingale Diagnostics")
    from src.instruments.registry import INSTRUMENTS
    st.session_state.mg_instrument = st.selectbox(
        "Instrument", ["ALL"] + sorted(INSTRUMENTS.keys()),
        help="Filter the closed-trade log. ALL pools every instrument.")
    st.session_state.mg_delta = st.select_slider(
        "Confidence", options=[0.20, 0.10, 0.05, 0.01], value=0.05,
        format_func=lambda d: f"{1 - d:.0%}")
    st.session_state.mg_method = st.selectbox(
        "Drift estimator", ["expanding", "rolling", "constant"],
        help="expanding and rolling use only prior data; constant peeks at the "
             "full sample and is diagnostic only.")
    st.session_state.mg_window = st.number_input(
        "Rolling window", 10, 500, 50, step=10,
        disabled=(st.session_state.get("mg_method") != "rolling"))
    st.divider()
    render_sidebar_nav()

render_martingale_tab()
