"""
stochastic_tab.py
=================
Stochastic calculus — realised volatility, Itô drag, first passage, options.

Renders `src/core/stochastic.py` (pure numpy/scipy/pandas, spec-tested) against
the canonical OHLC spine:

  Quadratic variation  → the five volatility estimators (close-to-close,
                         Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)
                         and the Hurst check on whether variance really scales
                         linearly in time (the assumption behind every √21).
  Itô's correction     → volatility drag σ²/2, the log-drift ν = μ − σ²/2 that
                         actually compounds, and the Kelly leverage limit.
  Scale function       → P(target fills before stop), plus the Monte-Carlo
                         finite-horizon version and a price cone.
  Black-Scholes        → price, Greeks, and implied vol — where the real-world
                         drift μ is conspicuously absent.

Prices come from `src/services/market_data.daily_ohlc` (the canonical spine),
so this page can never disagree with the rest of the app about what a bar is.

Audit-only: a research read, never a directional pair+bias call, so it logs to
`tool_usage_log` and never writes `trade_setups`.

Entry point: render_stochastic_tab()
"""
from __future__ import annotations

import streamlit as st

from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="STOC · Stochastic Calculus",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

import numpy as np
import pandas as pd

from src.core.stochastic import (
    barrier_probabilities,
    black_scholes,
    calibrate_gbm,
    implied_vol,
    kelly_leverage,
    signature_plot,
    simulate_gbm,
    variance_scaling,
    volatility_estimators,
)
from src.instruments.registry import INSTRUMENTS
from src.pages_lib.navigation import render_sidebar_nav
from src.services.market_data import asof_caption, daily_ohlc
from src.ui.charts import ChartKit
from src.ui.components import MetricCell, render_metric_row
from src.ui.theme import BloombergTheme as T

MIN_BARS = 60


@st.cache_data(ttl=900, show_spinner="Loading price history…")
def load_ohlc(pair: str, lookback_days: int) -> pd.DataFrame:
    """OHLC bars for a registry instrument, straight off the canonical spine."""
    inst = INSTRUMENTS.get(pair)
    if inst is None:
        return pd.DataFrame()
    df = daily_ohlc(inst.ticker, period=f"{int(lookback_days)}d")
    if df is None or df.empty:
        return pd.DataFrame()
    # The engine lower-cases column names itself, so the spine's
    # Open/High/Low/Close frame can be handed over untouched.
    return df.dropna()


@st.cache_data(ttl=900, show_spinner=False)
def analyse(close: tuple, ppy: int) -> dict:
    px = np.asarray(close, dtype=float)
    out = {"gbm": calibrate_gbm(px, ppy)}
    try:
        out["scaling"] = variance_scaling(px)
    except ValueError:
        out["scaling"] = None
    return out


def render_stochastic_tab() -> None:
    st.markdown("## 🎓 Stochastic Calculus — drag, first passage, options")
    st.caption(
        "Quadratic variation gives the volatility estimators; Itô's correction "
        "gives the drag and the leverage limit; the scale function gives the "
        "probability your target fills before your stop. Research and teaching "
        "tool — nothing here writes a trade signal."
    )

    pair = st.session_state.get("stc_pair", "XAU/USD")
    lookback = int(st.session_state.get("stc_lookback", 730))
    ppy = int(st.session_state.get("stc_ppy", 252))

    df = load_ohlc(pair, lookback)
    if df.empty or len(df) < MIN_BARS:
        st.warning(f"Only {len(df)} bars for {pair} — need ~{MIN_BARS}+ for "
                   "stable estimates. Try a longer lookback.")
        return

    st.caption(f"{pair} · {len(df)} daily bars · {asof_caption(df)}")

    px = df["Close"].to_numpy(dtype=float)
    res = analyse(tuple(px), ppy)
    p = res["gbm"]

    render_metric_row([
        MetricCell("Volatility σ", f"{p.sigma:.1%}", "cyan"),
        MetricCell("Arithmetic drift μ", f"{p.mu:+.1%}",
                   "green" if p.mu > 0 else "red"),
        MetricCell("Volatility drag σ²/2", f"−{p.drag:.1%}", "red"),
        MetricCell("Log-drift ν", f"{p.nu:+.1%}",
                   "green" if p.nu > 0 else "red"),
        MetricCell("Kelly f*", f"{p.kelly:.2f}×" if np.isfinite(p.kelly) else "—",
                   "yellow"),
    ])
    st.caption("ν = μ − σ²/2 is what actually compounds. The gap between μ and "
               "ν is not a fee anyone charges — it's arithmetic.")

    t1, t2, t3, t4 = st.tabs(
        ["📊 VOLATILITY", "⚖️ DRAG & LEVERAGE", "🎯 TARGET vs STOP", "🧾 OPTIONS"])

    # ---- 1. volatility -------------------------------------------------
    with t1:
        try:
            ve = volatility_estimators(df, periods_per_year=ppy)
            names = ["close_to_close", "parkinson", "garman_klass",
                     "rogers_satchell", "yang_zhang"]
            labels = ["Close-to-close", "Parkinson", "Garman-Klass",
                      "Rogers-Satchell", "Yang-Zhang"]
            vals = [ve[k] for k in names]
            fig = ChartKit.panels(["Annualised volatility by estimator"])
            ChartKit.bars(fig, labels, vals,
                          colors=[T.GREY] * 4 + [T.GREEN])
            st.plotly_chart(ChartKit.finish(fig, height=380),
                            width="stretch", config=ChartKit.PLOTLY_CONFIG)
            st.caption(
                "Close-to-close discards everything that happened inside the "
                "bar; the range estimators don't, and are several times more "
                "efficient on the same sample. **Yang-Zhang** is the default to "
                "prefer — the only one here handling both overnight gaps and "
                "intraday drift. All range estimators read *low* if your bars "
                "are built from sparse ticks, since a discretely sampled high "
                "understates the true one.")
        except ValueError as exc:
            st.warning(str(exc))

        sc = res["scaling"]
        if sc:
            st.markdown("**Does variance scale linearly in time?**")
            fig = ChartKit.panels(["Variance vs horizon (log-log)"])
            ChartKit.line(fig, sc["horizons"], sc["variances"], "realised",
                          color=T.GREEN, width=2.2, showlegend=True)
            ref = (np.asarray(sc["variances"][0], dtype=float)
                   * np.asarray(sc["horizons"], dtype=float) / sc["horizons"][0])
            ChartKit.line(fig, sc["horizons"], ref, "Brownian (H=0.5)",
                          color=T.GREY, width=1.4, dash="dash", showlegend=True)
            out = ChartKit.finish(fig, height=360, showlegend=True)
            out.update_xaxes(type="log", title_text="horizon (bars)")
            out.update_yaxes(type="log", title_text="variance")
            st.plotly_chart(out, width="stretch", config=ChartKit.PLOTLY_CONFIG)
            render_metric_row([
                MetricCell("Hurst exponent H", f"{sc['hurst']:.3f}", "cyan"),
                MetricCell("Std error", f"±{sc['hurst_stderr']:.3f}", "grey"),
                MetricCell("Reading", sc["interpretation"], "white"),
            ])
            st.caption(
                "Brownian motion forces H = 0.5. That's the assumption you make "
                "every time you scale a daily vol by √21 to get a monthly one — "
                "if H ≠ 0.5 that scaling is wrong, in the direction this tells "
                "you. Mind the standard error: separating H = 0.55 from 0.50 "
                "needs a great deal of data.")

        with st.expander("🔬 Microstructure signature plot (intraday data only)"):
            st.caption(
                "If observed log price = efficient price + noise of variance q, "
                "then RV per period at interval k equals true_var + 2q/k. "
                "Regressing on 1/k separates them. **At daily sampling a "
                "realistic 1bp spread is invisible** and the fit will correctly "
                "return nothing — point this at minute or tick bars.")
            if st.button("▶ Fit noise model"):
                with st.spinner("Bootstrapping…"):
                    curve, fit = signature_plot(px, max_skip=30, n_boot=200)
                if not fit:
                    st.warning("Not enough data.")
                else:
                    fig = ChartKit.panels(
                        ["Realised variance vs sampling interval"])
                    ChartKit.line(fig, curve["skip"], curve["rv_per_period"],
                                  "measured", color=T.GREEN, width=2,
                                  showlegend=True)
                    ChartKit.line(fig, curve["skip"], curve["fitted"],
                                  "true_var + 2q/k", color=T.CYAN, width=1.6,
                                  dash="dash", showlegend=True)
                    st.plotly_chart(
                        ChartKit.finish(fig, height=340, showlegend=True),
                        width="stretch", config=ChartKit.PLOTLY_CONFIG)
                    render_metric_row([
                        MetricCell("Implied noise sd",
                                   f"{fit['implied_noise_sd']:.5f}", "cyan"),
                        MetricCell("RV inflation at k=1",
                                   f"{fit['inflation_at_skip1']:.2f}×", "yellow"),
                    ])
                    lo, hi = fit["slope_ci"]
                    if fit["significant"]:
                        st.warning(
                            f"Slope CI [{lo:.2e}, {hi:.2e}] sits above zero — "
                            "high-frequency RV here is measuring the spread as "
                            "well as the price. Sample where the curve flattens.")
                    else:
                        st.success(
                            f"Slope CI [{lo:.2e}, {hi:.2e}] straddles zero — no "
                            "detectable microstructure noise at this frequency.")

    # ---- 2. drag & leverage --------------------------------------------
    with t2:
        st.markdown(
            f"Itô's lemma applied to `ln S` turns a price drift of "
            f"**{p.mu:.2%}** into a log-drift of **{p.nu:.2%}**. The gap — "
            f"**{p.drag:.2%} per year** — is what volatility costs you before "
            "any spread, commission or slippage.")

        kdf = kelly_leverage(p)
        fstar = p.kelly
        fig = ChartKit.panels(["Expected log growth vs leverage"])
        ChartKit.line(fig, kdf["leverage"], kdf["log_growth"], "g(f)",
                      color=T.GREEN, width=2.4, showlegend=True)
        ChartKit.hline(fig, 0, T.BORDER)
        if np.isfinite(fstar):
            ChartKit.hline(fig, float(kdf["log_growth"].max()), T.CYAN,
                           dash="dot")
        out = ChartKit.finish(fig, height=380, showlegend=True)
        out.update_yaxes(tickformat=".1%")
        out.update_xaxes(title_text="leverage f")
        if np.isfinite(fstar):
            out.add_vline(x=fstar, line=dict(color=T.CYAN, width=1.4, dash="dash"),
                          annotation_text=f"f* = {fstar:.2f}×")
            out.add_vline(x=2 * fstar, line=dict(color=T.GREY, width=1.2, dash="dot"),
                          annotation_text=f"2f* = {2 * fstar:.2f}×")
        st.plotly_chart(out, width="stretch", config=ChartKit.PLOTLY_CONFIG)

        render_metric_row([
            MetricCell("Kelly f*", f"{fstar:.2f}×", "cyan"),
            MetricCell("Half Kelly", f"{fstar / 2:.2f}×", "green"),
            MetricCell("Zero-growth leverage", f"{2 * fstar:.2f}×", "red"),
        ])
        st.caption(
            "g(f) = fμ − f²σ²/2 is pure Itô: a linear return term against a "
            "quadratic drag. Past f* the drag wins, and at 2f* expected log "
            "growth is back to zero no matter how good the edge. Size **below** "
            "f* in practice — μ is estimated with far more error than σ, so an "
            "overstated μ pushes f* out to a leverage that loses money.")

        if np.isfinite(fstar) and fstar > 6:
            se = p.sigma / np.sqrt(p.n_obs / p.periods_per_year)
            st.warning(
                f"f* = {fstar:.1f}× is implausibly high. That's a sign μ is "
                "over-fitted to a lucky sample, not a licence to lever up: the "
                f"drift standard error here is roughly {se:.1%} a year.")

    # ---- 3. first passage ----------------------------------------------
    with t3:
        spot_default = float(px[-1])
        fmt = "%.5f" if spot_default < 100 else "%.2f"
        s1, s2, s3 = st.columns(3)
        spot = s1.number_input("Spot", 0.0001, 1e7, spot_default, format=fmt)
        tp = s2.number_input("Take profit", 0.0001, 1e7,
                             float(round(spot_default * 1.03, 5)), format=fmt)
        sl = s3.number_input("Stop loss", 0.0001, 1e7,
                             float(round(spot_default * 0.99, 5)), format=fmt)
        d1, d2 = st.columns(2)
        use_drift = d1.checkbox(
            "Use estimated drift μ", value=False,
            help="Off assumes zero arithmetic drift — the conservative "
                 "default, since μ is barely identified from price history.")
        hold = d2.number_input("Horizon (bars)", 1, 2000, 20, step=1)

        if not (sl < spot < tp):
            st.error("Need stop loss < spot < take profit (long convention).")
        else:
            try:
                b = barrier_probabilities(
                    spot, tp, sl, sigma=p.sigma,
                    mu=p.mu if use_drift else 0.0,
                    horizon_years=hold / ppy,
                    periods_per_year=ppy, n_sims=40_000)
            except ValueError as exc:
                st.error(str(exc))
            else:
                edge = b["p_take_profit"] - b["breakeven_win_rate"]
                render_metric_row([
                    MetricCell("Reward : risk", f"{b['reward_risk']:.2f}", "cyan"),
                    MetricCell("Breakeven win rate",
                               f"{b['breakeven_win_rate']:.1%}", "yellow"),
                    MetricCell("P(target first)",
                               f"{b['p_take_profit']:.1%}", "white"),
                    MetricCell("Edge vs breakeven", f"{edge:+.1%}",
                               "green" if edge > 0 else "red"),
                ])

                fig = ChartKit.panels([f"Outcome within {hold} bars"])
                ChartKit.bars(
                    fig, ["Target first", "Stop first", "Unresolved"],
                    [b["mc_p_take_profit"], b["mc_p_stop_loss"],
                     b["mc_p_unresolved"]],
                    colors=[T.GREEN, T.RED, T.GREY])
                out = ChartKit.finish(fig, height=330)
                out.update_yaxes(tickformat=".0%")
                st.plotly_chart(out, width="stretch",
                                config=ChartKit.PLOTLY_CONFIG)

                st.caption(
                    f"Expected log return {b['expected_log_return']:+.4f}, "
                    f"expected price change {b['expected_value_price']:+.4f}. "
                    "These don't vanish together: when ν = 0 the first is "
                    "exactly zero, but Jensen makes price a *submartingale* so "
                    "the second stays positive. Compounding happens in log space.")

                if not use_drift:
                    st.info(
                        "With zero drift this reduces to **gambler's ruin** — "
                        "P(target first) is just the log-distance ratio "
                        "l/(u+l), and no choice of target and stop creates "
                        "edge. Same optional-stopping result the Martingale "
                        "page bootstraps in discrete time. Any edge has to come "
                        "from entry timing, not exit geometry.")
                elif b["log_drift_nu"] < 0 < b["arith_drift_mu"]:
                    st.warning(
                        f"μ = {b['arith_drift_mu']:.1%} is positive but ν = "
                        f"{b['log_drift_nu']:.1%} is negative: volatility drag "
                        "has more than eaten the drift. Long holds compound "
                        "downward here despite the positive average return.")

                with st.expander("📈 Simulated price cone"):
                    paths = simulate_gbm(spot, p.mu if use_drift else 0.0,
                                         p.sigma, hold / ppy, n_paths=1200,
                                         periods_per_year=ppy)
                    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
                    xs = np.arange(paths.shape[1])
                    fig = ChartKit.panels(["GBM cone — 50% and 90% bands"])
                    ChartKit.band(fig, xs, q[4], q[0], "rgba(0,255,65,0.08)",
                                  name="90%")
                    ChartKit.band(fig, xs, q[3], q[1], "rgba(0,255,65,0.18)",
                                  name="50%")
                    ChartKit.line(fig, xs, q[2], "median", color=T.GREEN,
                                  width=2, showlegend=True)
                    ChartKit.levels(fig, {"target": tp, "stop": sl},
                                    colors={"target": T.CYAN, "stop": T.RED})
                    st.plotly_chart(
                        ChartKit.finish(fig, height=380, showlegend=True),
                        width="stretch", config=ChartKit.PLOTLY_CONFIG)

    # ---- 4. options -----------------------------------------------------
    with t4:
        st.caption(
            "The real-world drift μ is **absent** from Black-Scholes: the delta "
            "hedge cancels the dW term, and what's left must earn r. Two traders "
            "who disagree completely on direction still owe the same price given "
            "the same σ. Volatility is the only input carrying an opinion — "
            "which is what makes an options market a market in σ.")

        o = st.columns(4)
        S = o[0].number_input("Spot ", 0.0001, 1e7, float(px[-1]), format=fmt)
        K = o[1].number_input("Strike", 0.0001, 1e7,
                              float(round(px[-1], 5)), format=fmt)
        days = o[2].number_input("Days to expiry", 1, 3650, 30)
        rate = o[3].number_input("Risk-free rate", -0.05, 0.5, 0.045,
                                 step=0.005, format="%.3f")
        v1, v2 = st.columns(2)
        sig_in = v1.slider("Volatility σ", 0.01, 2.0,
                           float(round(p.sigma, 2)), 0.01)
        kind = v2.radio("Type", ["call", "put"], horizontal=True)

        T_yrs = days / 365.0
        g = black_scholes(S, K, T_yrs, rate, sig_in, kind)
        render_metric_row([
            MetricCell("Price", f"{g['price']:,.4f}", "white"),
            MetricCell("Delta", f"{g['delta']:.4f}", "cyan"),
            MetricCell("Gamma", f"{g['gamma']:.6f}", "cyan"),
            MetricCell("Vega / 1% σ", f"{g['vega'] / 100:,.4f}", "yellow"),
            MetricCell("Theta / day", f"{g['theta'] / 365:,.4f}", "red"),
        ])

        grid = np.linspace(max(S * 0.6, 1e-6), S * 1.4, 160)
        now = [black_scholes(s, K, T_yrs, rate, sig_in, kind)["price"]
               for s in grid]
        expiry = (np.maximum(grid - K, 0) if kind == "call"
                  else np.maximum(K - grid, 0))
        fig = ChartKit.panels(
            ["Option value vs payoff — the gap is time value"])
        ChartKit.line(fig, grid, now, f"{days}d to expiry", color=T.GREEN,
                      width=2.4, showlegend=True)
        ChartKit.line(fig, grid, expiry, "payoff at expiry", color=T.GREY,
                      width=1.5, dash="dash", showlegend=True)
        out = ChartKit.finish(fig, height=380, showlegend=True)
        out.add_vline(x=S, line=dict(color=T.CYAN, width=1, dash="dot"),
                      annotation_text="spot")
        st.plotly_chart(out, width="stretch", config=ChartKit.PLOTLY_CONFIG)

        st.markdown("**Implied volatility**")
        q1, q2 = st.columns([2, 3])
        mkt = q1.number_input("Market price", 0.0, 1e7,
                              float(round(g["price"], 4)), format="%.4f")
        iv = implied_vol(mkt, S, K, T_yrs, rate, kind)
        if np.isnan(iv):
            q2.error("No implied vol exists — that quote violates the "
                     "no-arbitrage bounds for this contract.")
        else:
            q2.metric("Implied σ", f"{iv:.2%}", f"{iv - p.sigma:+.2%} vs realised",
                      help="Positive means options are pricing more volatility "
                           "than the underlying has recently delivered.")

    # ---- audit (research read, never a trade signal) -------------------
    try:
        from src.services.alert_service import NotifyCache
        from src.services.tool_log import log_tool_usage
        key = f"{pair}|{len(df)}|{ppy}|{df.index[-1].date()}"
        if NotifyCache("stochastic_log").filter_new([key]):
            log_tool_usage("stochastic", {
                "instrument": pair, "bars": int(len(df)),
                "periods_per_year": ppy,
                "sigma": float(p.sigma), "mu": float(p.mu),
                "nu": float(p.nu), "drag": float(p.drag),
                "kelly": None if not np.isfinite(p.kelly) else float(p.kelly),
                "hurst": (float(res["scaling"]["hurst"])
                          if res.get("scaling") else None),
            })
    except Exception:
        pass


# ===========================================================================
# PAGE ENTRY
# ===========================================================================
with st.sidebar:
    st.markdown("### 🎓 Stochastic Calculus")
    _pairs = sorted(INSTRUMENTS.keys())
    st.session_state.stc_pair = st.selectbox(
        "Instrument", _pairs,
        index=_pairs.index("XAU/USD") if "XAU/USD" in _pairs else 0)
    st.session_state.stc_lookback = st.number_input(
        "Lookback (days)", 90, 3650, 730, step=30)
    st.session_state.stc_ppy = st.number_input(
        "Periods / year", 12, 8760, 252, step=1,
        help="252 for daily bars, 52 weekly, 8760 hourly.")
    st.divider()
    render_sidebar_nav()

render_stochastic_tab()
