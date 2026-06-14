"""Backtest Lab — a dedicated, production-grade backtesting workbench.

Forecasting lives on its own page now (Forecast Lab); this page is *only* about
testing the 18-point workflow on history. Four tabs:

  • BACKTEST   — run the strategy, full quant stat block, equity curve, trade log
  • ROBUSTNESS — Monte Carlo trade-order resampling, walk-forward consistency,
                 parameter sweep heatmap
  • PORTFOLIO  — run across many instruments, leaderboard + combined equity
  • REPLAY     — step through all 18 checks on any historical date

All compute lives in src/services/backtest_service.py.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
from src.instruments.registry import INSTRUMENTS
from src.services import backtest_service as bt

st.set_page_config(
    page_title="Backtest Lab · Trading System",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

st.markdown("""
<style>
    .stApp { --border: color-mix(in srgb, var(--text-color) 12%, transparent);
              --muted:  color-mix(in srgb, var(--text-color) 55%, transparent); }
    html,body,[class*="css"]{font-family:'JetBrains Mono','Fira Code',monospace;}
    .stApp{background:var(--background-color);}
    section[data-testid="stSidebar"]{background:var(--secondary-background-color)!important;border-right:1px solid var(--border,#2a2a2a);}
    .hero{background:linear-gradient(135deg,#000000 0%,#0a0a0a 50%,#000000 100%);border:1px solid #2a2a2a;border-radius:16px;padding:24px 32px;margin-bottom:20px;position:relative;overflow:hidden;}
    .metric-box{background:var(--background-color);border:1px solid var(--border,#2a2a2a);border-radius:8px;padding:12px;text-align:center;}
    .metric-value{font-size:20px;font-weight:700;color:#e6e6e6;}
    .metric-label{font-size:10px;color:#9a9a9a;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}
    .check-pass{background:#1a3a2a;border:1px solid #00ff66;border-radius:6px;padding:8px 12px;margin:3px 0;font-size:13px;}
    .check-fail{background:#3a1a1a;border:1px solid #ff3344;border-radius:6px;padding:8px 12px;margin:3px 0;font-size:13px;}
    .check-auto{background:#1a2a3a;border:1px solid #00ff41;border-radius:6px;padding:8px 12px;margin:3px 0;font-size:13px;}
    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    [data-testid="stSidebarCollapseButton"]{visibility:visible !important;display:flex !important;}
    section[data-testid="stSidebar"]{display:block !important;}
    .block-container{padding-top:1.5rem;max-width:1500px;}
</style>
""", unsafe_allow_html=True)

GREEN, RED, AMBER, GREY = "#00ff66", "#ff3344", "#ffcc00", "#9a9a9a"


def metric(col, value, label, color="#e6e6e6"):
    col.markdown(
        f'<div class="metric-box"><div class="metric-value" style="color:{color}">{value}</div>'
        f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)


def _dark(fig, height=300, **kw):
    fig.update_layout(plot_bgcolor="#0f0f0f", paper_bgcolor="#000000",
                      font=dict(color="#c0c0c0", size=11), height=height,
                      margin=dict(l=40, r=20, t=20, b=40), **kw)
    fig.update_xaxes(gridcolor="#2a2a2a")
    fig.update_yaxes(gridcolor="#2a2a2a")
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🧪 Backtest Lab")
    st.markdown("---")

    pair = st.selectbox("Instrument", INSTRUMENTS.keys())
    direction = st.radio("Direction", ["Long", "Short"], horizontal=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        start_dt = st.date_input("From", value=datetime.today() - timedelta(days=180))
    with col_b:
        end_dt = st.date_input("To", value=datetime.today())

    st.markdown("---")
    min_score = st.slider("Min score to enter (/18)", 10, 18, 14)
    sl_mult = st.slider("SL × ATR", 1.0, 3.0, 1.5, 0.1)
    rr = st.slider("TP1 R:R", 1.5, 4.0, 2.0, 0.5)
    account = st.number_input("Account size ($)", 1000, 1000000, 10000, 1000)
    risk_pct = st.slider("Risk per trade (%)", 0.5, 3.0, 1.0, 0.25)
    spread_pips = st.slider("Spread + slippage (pips)", 0.0, 10.0, 0.0, 0.5,
                            help="Round-trip cost applied as a worse entry fill. 0 = frictionless.")

    st.markdown("---")
    run_btn = st.button("▶ Run Backtest", use_container_width=True, type="primary")

    st.divider()
    render_sidebar_nav()


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════

ticker = INSTRUMENTS[pair]["ticker"]
pip_sz = INSTRUMENTS[pair]["pip_size"]
cost_price = spread_pips * pip_sz

st.markdown(f"""
<div class="hero">
  <h2 style="margin:0;color:#e6e6e6;font-size:22px;font-weight:700;">🧪 Backtest Lab — {pair} {direction}</h2>
  <p style="margin:6px 0 0;color:#9a9a9a;font-size:13px;">
    18-point workflow on daily data · SL = {sl_mult}× ATR · TP1 = {rr}:1 · Min score {min_score}/18 ·
    cost {spread_pips:g}p · {risk_pct:g}% risk on ${account:,.0f}
  </p>
  <p style="margin:4px 0 0;color:#9a9a9a;font-size:11px;">
    ⚠ 15M checks are proxied from daily bars — results reflect system logic, not exact 15-minute execution.
  </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading price data…"):
    try:
        daily, weekly = bt.load_data(ticker)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if daily.empty or len(daily) < 60:
    st.error("Not enough data. Try a more liquid instrument or wider date range.")
    st.stop()

if run_btn:
    with st.spinner("Running backtest…"):
        trades = bt.run_backtest(daily, weekly, direction, min_score, sl_mult, rr,
                                 start_dt, end_dt, cost_price=cost_price)
    st.session_state["bt_trades"] = trades
    st.session_state["bt_ctx"] = dict(pair=pair, direction=direction, min_score=min_score,
                                      sl_mult=sl_mult, rr=rr, account=account,
                                      risk_pct=risk_pct, start_dt=start_dt, end_dt=end_dt)

tab_bt, tab_rob, tab_pf, tab_replay = st.tabs(
    ["📊 BACKTEST", "🎲 ROBUSTNESS", "🌐 PORTFOLIO", "🔁 REPLAY"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — BACKTEST
# ══════════════════════════════════════════════════════════════════

with tab_bt:
    if "bt_trades" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **▶ Run Backtest**.")
        st.markdown("""
**How it works** — for each trading day all 18 checks are evaluated on data
available up to that bar. When the score clears your threshold **and** every
critical check passes, a trade is opened at the next bar's open and managed with
an ATR stop and R-multiple targets until hit or a 20-bar timeout. Spread and
slippage are modelled as a worse entry fill.
        """)
    else:
        trades = st.session_state["bt_trades"]
        if not trades:
            st.warning("No trades met the threshold in this window. Lower **Min score** or widen the dates.")
        else:
            stats = bt.compute_stats(trades, account, risk_pct, rr)

            st.markdown("#### Performance")
            m = st.columns(6)
            metric(m[0], stats["total"], "Trades")
            metric(m[1], f'{stats["win_rate"]:.0f}%', "Win Rate",
                   GREEN if stats["win_rate"] >= stats["breakeven_wr"] else RED)
            metric(m[2], f'{stats["net_r"]:+.1f}R', "Net R", GREEN if stats["net_r"] > 0 else RED)
            metric(m[3], f'{stats["expectancy"]:+.2f}R', "Expectancy",
                   GREEN if stats["expectancy"] > 0 else RED)
            metric(m[4], f'{stats["profit_factor"]:.2f}', "Profit Factor",
                   GREEN if stats["profit_factor"] >= 1 else RED)
            metric(m[5], f'{stats["breakeven_wr"]:.0f}%', "Breakeven WR")

            m2 = st.columns(6)
            sharpe = stats["sharpe"]
            sqn = stats["sqn"]
            metric(m2[0], f'{sharpe:.2f}' if np.isfinite(sharpe) else "—", "Sharpe (ann.)",
                   GREEN if np.isfinite(sharpe) and sharpe > 1 else AMBER)
            metric(m2[1], f'{stats["sortino"]:.2f}' if np.isfinite(stats["sortino"]) else "—", "Sortino")
            metric(m2[2], f'{stats["calmar"]:.2f}' if np.isfinite(stats["calmar"]) else "—", "Calmar")
            metric(m2[3], f'{sqn:.2f}' if np.isfinite(sqn) else "—", "SQN",
                   GREEN if np.isfinite(sqn) and sqn >= 2 else AMBER)
            metric(m2[4], f'{stats["max_dd_pct"]:.1f}%', "Max Drawdown", RED)
            metric(m2[5], f'{stats["cagr"]:+.0f}%' if np.isfinite(stats["cagr"]) else "—", "CAGR")

            m3 = st.columns(6)
            dollar = stats["final_equity"] - account
            metric(m3[0], f'${stats["final_equity"]:,.0f}', "Final Equity",
                   GREEN if dollar > 0 else RED)
            metric(m3[1], f'${dollar:+,.0f}', "Net P&L", GREEN if dollar > 0 else RED)
            metric(m3[2], f'{stats["avg_win"]:+.2f}R', "Avg Win", GREEN)
            metric(m3[3], f'{stats["avg_loss"]:+.2f}R', "Avg Loss", RED)
            metric(m3[4], f'{stats["max_loss_streak"]}', "Max Loss Streak")
            metric(m3[5], f'{stats["avg_bars"]:.0f}', "Avg Bars Held")

            st.markdown("")

            # ── Equity curve + R distribution ─────────────────────────
            eq = bt.build_equity(trades, account, risk_pct)
            dates = [t["date"] for t in trades]
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Compounded Equity Curve ($)**")
                peak = eq["equity"].cummax()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(eq))), y=eq["equity"],
                                         line=dict(color=GREEN, width=2), name="Equity",
                                         hovertemplate="Trade %{x}<br>$%{y:,.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=list(range(len(eq))), y=peak,
                                         line=dict(color=GREY, width=1, dash="dot"),
                                         name="Peak", hoverinfo="skip"))
                fig.add_hline(y=account, line=dict(color=GREY, dash="dash", width=1))
                st.plotly_chart(_dark(fig, 300, showlegend=False, hovermode="x unified",
                                      yaxis=dict(title="Equity ($)")),
                                use_container_width=True, config=dict(displayModeBar=False))
            with c2:
                st.markdown("**R per Trade**")
                rs = [t["r"] for t in trades]
                fig_r = go.Figure(go.Bar(
                    x=[str(d) for d in dates], y=rs,
                    marker_color=[GREEN if r > 0 else RED for r in rs],
                    hovertemplate="%{x}<br>R: %{y:.2f}<extra></extra>"))
                fig_r.add_hline(y=0, line=dict(color=GREY, dash="dash", width=1))
                st.plotly_chart(_dark(fig_r, 300, xaxis=dict(tickangle=-45, showgrid=False),
                                      yaxis=dict(title="R")),
                                use_container_width=True, config=dict(displayModeBar=False))

            # ── Outcome distribution ──────────────────────────────────
            st.markdown("**Outcome Distribution**")
            oc = pd.DataFrame(trades)["outcome"].value_counts()
            oc_cols = st.columns(max(len(oc), 1))
            for col, (label, cnt) in zip(oc_cols, oc.items()):
                metric(col, f"{cnt} ({cnt/len(trades)*100:.0f}%)", label)

            # ── Trade log ─────────────────────────────────────────────
            st.markdown("**Trade Log**")
            df_log = pd.DataFrame(trades)
            df_log["pips"] = (df_log["sl_d"] / pip_sz * df_log["r"]).round(1)

            def colour_outcome(v):
                return {"TP2": f"color:{GREEN};font-weight:600", "TP1 → BE": f"color:{AMBER}",
                        "SL": f"color:{RED}"}.get(v, f"color:{GREY}")

            styled = (df_log[["date", "score", "entry", "exit", "sl", "tp1", "r", "pips", "bars_held", "outcome"]]
                      .style
                      .map(colour_outcome, subset=["outcome"])
                      .map(lambda v: f'color:{GREEN if v > 0 else RED}', subset=["r"])
                      .format({"entry": "{:.5f}", "exit": "{:.5f}", "sl": "{:.5f}", "tp1": "{:.5f}",
                               "r": "{:+.2f}R", "pips": "{:.0f}"}))
            st.dataframe(styled, use_container_width=True, height=320)

            st.download_button("⬇ Download trade log (CSV)",
                               df_log.to_csv(index=False).encode(),
                               file_name=f"backtest_{pair.replace('/','')}_{direction}.csv",
                               mime="text/csv")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — ROBUSTNESS
# ══════════════════════════════════════════════════════════════════

with tab_rob:
    if "bt_trades" not in st.session_state or not st.session_state["bt_trades"]:
        st.info("Run a backtest first — robustness tools resample its trades.")
    else:
        trades = st.session_state["bt_trades"]

        # ── Monte Carlo trade-order resampling ────────────────────────
        st.markdown("#### Monte Carlo — trade-order resampling")
        st.caption("Bootstraps 2,000 alternate orderings of your trades (sampled with replacement) "
                   "to show the range of outcomes the same edge could plausibly produce.")
        mc = bt.monte_carlo_resample(trades, account, risk_pct, n_sims=2000)
        if mc is None:
            st.warning("Need at least 5 trades for a meaningful Monte Carlo.")
        else:
            k = st.columns(5)
            metric(k[0], f"{mc.p_profit*100:.0f}%", "P(profit)",
                   GREEN if mc.p_profit > 0.5 else RED)
            metric(k[1], f"{mc.median_return:+.1f}%", "Median return",
                   GREEN if mc.median_return > 0 else RED)
            metric(k[2], f"{mc.q05_return:+.1f}% / {mc.q95_return:+.1f}%", "5th / 95th")
            metric(k[3], f"{mc.median_dd:.1f}%", "Median max DD", RED)
            metric(k[4], f"{mc.risk_of_ruin*100:.1f}%", "Risk of ruin (−50%)",
                   GREEN if mc.risk_of_ruin < 0.05 else RED)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Equity-curve fan (200 sample paths)**")
                fig = go.Figure()
                for curve in mc.sample_curves:
                    fig.add_trace(go.Scatter(y=curve, mode="lines",
                                             line=dict(color="rgba(0,255,65,0.06)", width=1),
                                             hoverinfo="skip", showlegend=False))
                fig.add_hline(y=account, line=dict(color=GREY, dash="dash", width=1))
                st.plotly_chart(_dark(fig, 320, yaxis=dict(title="Equity ($)")),
                                use_container_width=True, config=dict(displayModeBar=False))
            with c2:
                st.markdown("**Distribution of final return (%)**")
                fig_h = go.Figure(go.Histogram(
                    x=mc.final_returns, nbinsx=50,
                    marker=dict(color=[GREEN if v > 0 else RED for v in mc.final_returns])))
                fig_h.add_vline(x=0, line_color="#e6e6e6", line_width=1.5)
                st.plotly_chart(_dark(fig_h, 320, showlegend=False, xaxis=dict(title="Final return %")),
                                use_container_width=True, config=dict(displayModeBar=False))

        st.markdown("---")

        # ── Walk-forward segments ─────────────────────────────────────
        st.markdown("#### Walk-forward consistency")
        st.caption("Splits the window into equal calendar segments. A real edge holds up across "
                   "all of them, not just one lucky stretch.")
        ctx = st.session_state.get("bt_ctx", {})
        segs = bt.walk_forward_segments(trades, 4, ctx.get("start_dt"), ctx.get("end_dt"))
        if segs:
            sc = st.columns(len(segs))
            for col, s in zip(sc, segs):
                colr = GREEN if s["net_r"] > 0 else RED if s["net_r"] < 0 else GREY
                col.markdown(
                    f'<div class="metric-box"><div class="metric-value" style="color:{colr}">{s["net_r"]:+.1f}R</div>'
                    f'<div class="metric-label">{s["label"]}<br>{s["trades"]} trades · {s["win_rate"]:.0f}% WR</div></div>',
                    unsafe_allow_html=True)
            pos = sum(1 for s in segs if s["net_r"] > 0)
            st.markdown(f"**{pos}/{len(segs)} segments profitable** — "
                        + ("consistent edge ✅" if pos >= len(segs) - 1 else
                           "patchy; treat with caution ⚠"))

        st.markdown("---")

        # ── Parameter sweep ───────────────────────────────────────────
        st.markdown("#### Parameter sweep — SL × ATR vs R:R")
        st.caption("Re-runs the backtest across a grid. A robust system shows a smooth plateau of "
                   "positive cells, not a single hot spot (which is usually overfit).")
        sweep_metric = st.selectbox("Heatmap metric", ["net_r", "profit_factor", "win_rate", "sqn"],
                                    format_func=lambda m: {"net_r": "Net R", "profit_factor": "Profit Factor",
                                                           "win_rate": "Win Rate %", "sqn": "SQN"}[m])
        if st.button("Run parameter sweep", key="sweep_btn"):
            sl_grid = (1.0, 1.25, 1.5, 2.0, 2.5)
            rr_grid = (1.5, 2.0, 2.5, 3.0)
            with st.spinner(f"Running {len(sl_grid)*len(rr_grid)} backtests…"):
                grid = bt.parameter_sweep(daily, weekly, direction, min_score, sl_grid, rr_grid,
                                          start_dt, end_dt, metric=sweep_metric)
            fig = go.Figure(go.Heatmap(
                z=grid.values, x=[f"{c}:1" for c in grid.columns], y=[f"{r}×" for r in grid.index],
                colorscale="RdYlGn", zmid=0 if sweep_metric in ("net_r",) else None,
                colorbar=dict(title=sweep_metric),
                hovertemplate="SL %{y} · RR %{x}<br>%{z:.2f}<extra></extra>",
                text=np.round(grid.values, 1), texttemplate="%{text}"))
            st.plotly_chart(_dark(fig, 360, xaxis=dict(title="R:R"), yaxis=dict(title="SL × ATR")),
                            use_container_width=True, config=dict(displayModeBar=False))


# ══════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO
# ══════════════════════════════════════════════════════════════════

with tab_pf:
    st.markdown("#### Portfolio backtest — same strategy, many instruments")
    st.caption("Runs the workflow on each selected instrument and combines fills chronologically on "
               "one account. Hits Yahoo for every pair, so keep the list focused.")

    default_pf = [p for p in ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "GBP/JPY", "XAU/USD"]
                  if p in INSTRUMENTS]
    pf_pairs = st.multiselect("Instruments", INSTRUMENTS.keys(), default=default_pf)
    pf_dir = st.radio("Direction", ["Long", "Short"], horizontal=True, key="pf_dir")

    if st.button("▶ Run Portfolio Backtest", type="primary"):
        if not pf_pairs:
            st.warning("Select at least one instrument.")
        else:
            tickers = {p: INSTRUMENTS[p]["ticker"] for p in pf_pairs}
            with st.spinner(f"Backtesting {len(pf_pairs)} instruments…"):
                pf = bt.portfolio_backtest(tickers, pf_dir, min_score, sl_mult, rr,
                                           start_dt, end_dt, account, risk_pct)
            st.session_state["pf_result"] = pf

    if "pf_result" in st.session_state:
        pf = st.session_state["pf_result"]
        k = st.columns(4)
        dollar = pf["final_equity"] - account
        metric(k[0], pf["total_trades"], "Total Trades")
        metric(k[1], f'{pf["combined_net_r"]:+.1f}R', "Combined Net R",
               GREEN if pf["combined_net_r"] > 0 else RED)
        metric(k[2], f'${pf["final_equity"]:,.0f}', "Final Equity",
               GREEN if dollar > 0 else RED)
        metric(k[3], f'${dollar:+,.0f}', "Net P&L", GREEN if dollar > 0 else RED)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Combined Equity Curve ($)**")
            curve = pf["equity_curve"]
            fig = go.Figure(go.Scatter(y=curve["equity"], line=dict(color=GREEN, width=2),
                                       hovertemplate="$%{y:,.0f}<extra></extra>"))
            fig.add_hline(y=account, line=dict(color=GREY, dash="dash", width=1))
            st.plotly_chart(_dark(fig, 320, yaxis=dict(title="Equity ($)")),
                            use_container_width=True, config=dict(displayModeBar=False))
        with c2:
            st.markdown("**Per-Instrument Leaderboard**")
            lb = pd.DataFrame(pf["leaderboard"])
            if not lb.empty:
                styled = (lb.style
                          .map(lambda v: f'color:{GREEN if v > 0 else RED}', subset=["net_r"])
                          .format({"net_r": "{:+.1f}R", "win_rate": "{:.0f}%", "profit_factor": "{:.2f}"}))
                st.dataframe(styled, use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — REPLAY
# ══════════════════════════════════════════════════════════════════

with tab_replay:
    st.markdown("#### Step-by-step workflow replay")
    st.caption("Pick any historical date to walk all 18 checks exactly as the system would have on that day.")

    available_dates = [d.date() for d in daily.index if len(daily[daily.index <= d]) >= 55]
    if not available_dates:
        st.warning("Not enough history to replay.")
    else:
        replay_date = st.select_slider("Date to replay", options=available_dates,
                                       value=available_dates[-1])
        replay_idx = next((i for i, d in enumerate(daily.index) if d.date() == replay_date), None)
        if replay_idx and replay_idx >= 55:
            d_slice = daily.iloc[:replay_idx + 1]
            w_slice = weekly[weekly.index <= daily.index[replay_idx]]
            res, score = bt.run_checks(d_slice, w_slice, direction)

            passed = sum(v for v, _ in res.values())
            crit_ok = all(res.get(k, (False,))[0] for k in bt.CRITICAL_KEYS)
            go_signal = passed >= min_score and crit_ok

            cols = st.columns(4)
            metric(cols[0], f"{passed}/18", "Checks Passed")
            metric(cols[1], "🟢 GO" if go_signal else "🔴 NO TRADE", "Signal",
                   GREEN if go_signal else RED)
            metric(cols[2], f'{daily["Close"].iloc[replay_idx]:.5f}', "Close")
            metric(cols[3], f'{bt.calc_atr(d_slice, 14).iloc[-1]:.5f}', "ATR14")

            st.markdown("")
            c1, c2 = st.columns(2)
            for i, (key, label, mode, crit) in enumerate(bt.CHECK_META):
                col = c1 if i % 2 == 0 else c2
                if key not in res:
                    continue
                ok, detail = res[key]
                tag = " ⭐" if crit else ""
                cls = "check-auto" if mode == "auto" else "check-pass" if ok else "check-fail"
                icon = "🔵" if mode == "auto" else "✅" if ok else "❌"
                col.markdown(f'<div class="{cls}">{icon} {label}{tag}<br>'
                             f'<small style="color:#9a9a9a">{detail}</small></div>',
                             unsafe_allow_html=True)
            st.caption("⭐ = critical check. All critical checks must pass regardless of total score.")
