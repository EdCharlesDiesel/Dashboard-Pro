"""
twenty_day_breakout_tab.py
=========================
Kathy Lien's "20-Day Breakout" failure-test strategy, as a scanner + annotated
chart + success-rate backtest.

The setup (long; short is the mirror):
  1. price makes a NEW 20-day high            -> on the radar
  2. same day or next, it reverses to a 2-DAY LOW   -> the shake-out
  3. it RE-TAKES the 20-day high within 3 days      -> entry (weak hands flushed)
  4. stop a few pips below the 2-day low
  5. take half at +1R, move stop to breakeven; trail the rest

This page finds every occurrence in the history, marks the radar high, the
2-day-low pivot, and the entry/stop/target, and reports how often the trade
reached +1R before stopping out — so you can check the "high success rate"
claim instead of taking the book's word for it. Pure price action: works on FX.

Entry point: render()
Standalone:  streamlit run twenty_day_breakout_tab.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    import plotly.graph_objects as go
except Exception:
    go = None


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in (pair or "").upper() else 0.0001


# ===========================================================================
# DETECTION (pure / testable)
# ===========================================================================

def _simulate(df, t0, entry, stop, target, side, max_bars=80):
    """
    Resolve a trade from its entry bar. Conservative path assumption: on each
    bar the STOP is checked before the target, so same-bar ambiguity never
    inflates the win rate. Returns (outcome, bars_to_resolve).
    """
    H, L = df["High"].values, df["Low"].values
    end = min(t0 + max_bars, len(df))
    for k in range(t0, end):
        if side == "long":
            if L[k] <= stop:
                return "stop", k - t0
            if H[k] >= target:
                return "target", k - t0
        else:
            if H[k] >= stop:
                return "stop", k - t0
            if L[k] <= target:
                return "target", k - t0
    return "open", None


def find_setups(df: pd.DataFrame, side: str = "both", n_high: int = 20,
                within: int = 3, buffer_pips: float = 2.0,
                pip: float = 0.0001) -> list[dict]:
    """Find all 20-day-breakout failure-test setups in an OHLC frame."""
    need = {"High", "Low", "Close"}
    if df is None or df.empty or not need.issubset(df.columns):
        return []
    H, L = df["High"], df["Low"]
    prev_high20 = H.shift(1).rolling(n_high).max()
    prev_low20 = L.shift(1).rolling(n_high).min()
    N = len(df)
    setups, used_entry = [], set()

    def add_long():
        for h in range(n_high, N - 1):
            if not (H.iloc[h] > prev_high20.iloc[h]):
                continue
            level = float(H.iloc[h])
            for Lp in (h, h + 1):
                if Lp < 1 or Lp >= N:
                    continue
                if L.iloc[Lp] < L.iloc[Lp - 1]:                      # 2-day low
                    two_low = float(L.iloc[Lp])
                    entry_idx = next((t for t in range(Lp + 1, min(Lp + 1 + within, N))
                                      if H.iloc[t] > level), None)   # re-break
                    if entry_idx is None or entry_idx in used_entry:
                        continue
                    stop = two_low - buffer_pips * pip
                    risk = level - stop
                    if risk <= 0:
                        continue
                    target = level + risk
                    outcome, bars = _simulate(df, entry_idx, level, stop, target, "long")
                    used_entry.add(entry_idx)
                    setups.append(dict(
                        side="long", radar_idx=h, radar_date=df.index[h],
                        pivot_idx=Lp, pivot_date=df.index[Lp],
                        entry_idx=entry_idx, entry_date=df.index[entry_idx],
                        entry=level, stop=stop, target=target,
                        risk_pips=risk / pip, outcome=outcome, bars=bars))
                    break

    def add_short():
        for h in range(n_high, N - 1):
            if not (L.iloc[h] < prev_low20.iloc[h]):
                continue
            level = float(L.iloc[h])
            for Lp in (h, h + 1):
                if Lp < 1 or Lp >= N:
                    continue
                if H.iloc[Lp] > H.iloc[Lp - 1]:                      # 2-day high
                    two_high = float(H.iloc[Lp])
                    entry_idx = next((t for t in range(Lp + 1, min(Lp + 1 + within, N))
                                      if L.iloc[t] < level), None)   # re-break down
                    if entry_idx is None or entry_idx in used_entry:
                        continue
                    stop = two_high + buffer_pips * pip
                    risk = stop - level
                    if risk <= 0:
                        continue
                    target = level - risk
                    outcome, bars = _simulate(df, entry_idx, level, stop, target, "short")
                    used_entry.add(entry_idx)
                    setups.append(dict(
                        side="short", radar_idx=h, radar_date=df.index[h],
                        pivot_idx=Lp, pivot_date=df.index[Lp],
                        entry_idx=entry_idx, entry_date=df.index[entry_idx],
                        entry=level, stop=stop, target=target,
                        risk_pips=risk / pip, outcome=outcome, bars=bars))
                    break

    if side in ("long", "both"):
        add_long()
    if side in ("short", "both"):
        add_short()
    setups.sort(key=lambda s: s["entry_idx"])
    return setups


def setup_stats(setups: list[dict]) -> dict:
    resolved = [s for s in setups if s["outcome"] in ("target", "stop")]
    wins = [s for s in resolved if s["outcome"] == "target"]
    n = len(resolved)
    return {
        "total": len(setups), "resolved": n, "open": len(setups) - n,
        "wins": len(wins), "win_rate": (len(wins) / n) if n else 0.0,
        "avg_bars_to_target": float(np.mean([s["bars"] for s in wins])) if wins else 0.0,
        "avg_risk_pips": float(np.mean([s["risk_pips"] for s in setups])) if setups else 0.0,
    }


# ===========================================================================
# LOADER / UI
# ===========================================================================

def _cache(ttl):
    return st.cache_data(ttl=ttl, show_spinner=False) if st is not None else (lambda f: f)


@_cache(ttl=3600)
def load_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    from src.db.market_cache import cached_ohlc
    df = cached_ohlc(ticker, period=period, interval="1d", ttl=3600)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.dropna(how="all")


def _chart(df, setups, selected):
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
        close=df["Close"], name="Price", showlegend=False))

    # all entries as small triangles, coloured by outcome
    colour = {"target": "#26a69a", "stop": "#ef5350", "open": "#9e9e9e"}
    for s in setups:
        up = s["side"] == "long"
        fig.add_trace(go.Scatter(
            x=[s["entry_date"]], y=[s["entry"]], mode="markers",
            marker=dict(symbol="triangle-up" if up else "triangle-down",
                        size=11, color=colour[s["outcome"]],
                        line=dict(width=1, color="white")),
            showlegend=False, hovertext=f"{s['side']} entry {s['entry']:.4f} "
            f"({s['outcome']})", hoverinfo="text"))

    # detailed annotation for the selected setup
    if selected is not None:
        s = selected
        for y, name, dash in [(s["entry"], "Entry", "solid"),
                              (s["stop"], "Stop", "dash"),
                              (s["target"], "Target +1R", "dot")]:
            fig.add_shape(type="line", x0=s["pivot_date"], x1=df.index[-1],
                          y0=y, y1=y, line=dict(width=1.2, dash=dash,
                          color="#ef5350" if name == "Stop" else
                          "#26a69a" if name.startswith("Target") else "#90caf9"))
            fig.add_annotation(x=df.index[-1], y=y, text=name, showarrow=False,
                               xanchor="right", font=dict(size=11))
        fig.add_annotation(x=s["radar_date"], y=df["High"].iloc[s["radar_idx"]],
                           text="20-day high", showarrow=True, arrowhead=2, ay=-30)
        fig.add_annotation(x=s["pivot_date"],
                           y=(df["Low"] if s["side"] == "long" else df["High"]).iloc[s["pivot_idx"]],
                           text="2-day low" if s["side"] == "long" else "2-day high",
                           showarrow=True, arrowhead=2, ay=30)

    fig.update_layout(height=600, margin=dict(l=10, r=10, t=20, b=10),
                      xaxis_rangeslider_visible=False, template="plotly_dark")
    return fig


def render():
    st.header("🎯 20-Day Breakout — failure-test scanner")
    st.caption("Lien's setup: new 20-day high → shake-out to a 2-day low → re-break "
               "of the high (weak hands flushed, real money re-enters). Finds every "
               "occurrence, marks the levels, and reports the +1R success rate.")
    if go is None:
        st.error("Needs plotly — `pip install plotly`.")
        return

    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker", value="GBPUSD=X")
    period = c2.selectbox("History", ["1y", "2y", "5y"], index=1)
    side = c3.selectbox("Side", ["both", "long", "short"])

    c4, c5, c6 = st.columns(3)
    n_high = c4.slider("Breakout lookback (days)", 10, 40, 20)
    within = c5.slider("Re-break window (days)", 1, 5, 3)
    buf = c6.slider("Stop buffer (pips)", 0.0, 20.0, 2.0, 1.0)

    df = load_ohlcv(ticker, period)
    if df.empty or "Close" not in df:
        st.warning("Could not load price data (yfinance unreachable).")
        return

    pip = pip_size(ticker.replace("=X", ""))
    setups = find_setups(df, side=side, n_high=n_high, within=within,
                         buffer_pips=buf, pip=pip)
    if not setups:
        st.info("No setups found in this window. Try a longer history or the other "
                "side.")
        st.plotly_chart(_chart(df, [], None), width="stretch")
        return

    # Persist currently-OPEN setups (unresolved → live signals) to the journal DB.
    # Resolved (target/stop) setups are historical backtest outcomes and are not saved.
    try:
        from src.services.signal_store import persist_signals
        from src.instruments import INSTRUMENTS
        _name = next((n for n, i in INSTRUMENTS.items() if i.ticker == ticker), ticker)

        # Shared house view — flags when the latest OPEN setup's direction
        # opposes the canonical Weekly/Daily/4H read (registry pairs only).
        from src.services.bias_service import show_house_view
        _open = [s for s in setups if s.get("outcome") == "open"]
        show_house_view(
            _name,
            page_read=("Long" if _open[-1]["side"] == "long" else "Short") if _open else None,
            page_label="20-Day Breakout")
        persist_signals("twenty_day_breakout", [{
            "pair": _name,
            "bias": "Long" if s["side"] == "long" else "Short",
            "entry": s["entry"],
            "stop_loss": s["stop"],
            "take_profit_1": s["target"],
            "stop_loss_pips": s.get("risk_pips"),
            # The bar the setup actually triggered on, not today. A setup stays
            # "open" for days, so keying on the trigger bar stops one breakout
            # being re-persisted on every sweep until it resolves.
            "bar_time": s["entry_date"],
            "conviction": "20-day breakout",
            "thesis": (f"20-Day Breakout failure-test ({s['side']}) — entry "
                       f"{s['entry']:.4f}, stop {s['stop']:.4f}, +1R {s['target']:.4f}"),
        } for s in setups if s.get("outcome") == "open"])
    except Exception:
        pass

    stats = setup_stats(setups)
    m = st.columns(4)
    m[0].metric("Setups found", stats["total"])
    m[1].metric("+1R win rate", f"{stats['win_rate']*100:.0f}%",
                help="Reached +1R target before the stop (conservative: stop "
                "checked first on same-bar ties).")
    m[2].metric("Resolved / open", f"{stats['resolved']} / {stats['open']}")
    m[3].metric("Avg risk", f"{stats['avg_risk_pips']:.0f} pips")

    labels = [f"{s['entry_date'].date()}  {s['side']}  ({s['outcome']})"
              for s in setups]
    pick = st.selectbox("Annotate a setup in detail", list(range(len(setups))),
                        index=len(setups) - 1, format_func=lambda i: labels[i])
    selected = setups[pick]

    st.plotly_chart(_chart(df, setups, selected), width="stretch")
    st.caption("Triangles mark every entry — green reached +1R, red stopped out, "
               "grey still open. The selected setup shows its 20-day high, 2-day "
               "pivot, and entry/stop/target lines.")

    s = selected
    st.markdown(f"**Selected setup** — {s['side']} on "
                f"{s['entry_date'].date()}: entry **{s['entry']:.4f}**, stop "
                f"**{s['stop']:.4f}**, target **{s['target']:.4f}** "
                f"(risk {s['risk_pips']:.0f} pips) → **{s['outcome']}**"
                + (f" in {s['bars']} bars" if s['bars'] is not None else ""))

    tbl = pd.DataFrame([{
        "Side": s["side"], "Radar (20d high/low)": s["radar_date"].date().isoformat(),
        "Pivot (2d)": s["pivot_date"].date().isoformat(),
        "Entry date": s["entry_date"].date().isoformat(),
        "Entry": round(s["entry"], 4), "Stop": round(s["stop"], 4),
        "Target": round(s["target"], 4), "Risk (pips)": round(s["risk_pips"], 0),
        "Outcome": s["outcome"], "Bars": s["bars"],
    } for s in setups])
    st.markdown("**All detected setups**")
    st.dataframe(tbl, width="stretch", hide_index=True)
    st.caption("Reality check on the book's 'high success rate': this measures a "
               "clean +1R-before-stop hit, not the full take-half-and-trail "
               "management — which can lift expectancy further but is harder to "
               "model. Run it across several pairs and years before trusting it.")


if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run twenty_day_breakout_tab.py")

    # Page bootstrap — kept lazy so the pure helpers above can still be imported
    # headlessly. Mirrors the legacy-page contract used across the app:
    # set_page_config → BloombergTheme.apply() → sidebar (header → divider → nav).
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="20-Day Breakout · Trading System",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()

    with st.sidebar:
        st.markdown("### 🚀 20-Day Breakout")
        st.caption("Lien's failure-test breakout scanner")
        st.divider()
        render_sidebar_nav()

    render()