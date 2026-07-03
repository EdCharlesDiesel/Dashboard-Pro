"""
trading_lab_tab.py
==================
Two tools that raise real trading odds more than any single book:

  📓 TRADE JOURNAL  — log every trade with its thesis, persist to CSV, and see
     the analytics that actually reveal edge: expectancy in R, profit factor,
     max drawdown, equity curve, and performance broken down by setup / pair.

  🔬 BACKTEST LAB   — a clean, vectorised daily backtest engine plus example
     strategies (MA trend, Donchian breakout) you can parameterise and extend.
     Test a rule's edge BEFORE risking money.

Everything is deterministic, pure-data (yfinance for prices), no AI. The journal
persists to a local CSV you control.

Entry points: render()  ·  render_journal()  ·  render_backtest()
Standalone:   streamlit run trading_lab_tab.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# regime.py lives in pages/ alongside this file; import it package-qualified so
# it resolves under Streamlit (where `pages/` is not itself on sys.path). The
# page UI in regime.py is guarded behind `__main__`, so this only pulls in the
# pure detector functions — it does not execute that page.
from pages.regime import (efficiency_ratio, adx, regime_mask, apply_regime_filter)

try:
    import streamlit as st
except Exception:
    st = None
try:
    import yfinance as yf
except Exception:
    yf = None


JOURNAL_COLUMNS = [
    "id", "opened", "closed", "pair", "direction", "setup",
    "entry", "stop", "target", "exit", "size", "status",
    "confidence", "thesis", "notes",
]
DEFAULT_JOURNAL_PATH = "trade_journal.csv"


# ===========================================================================
# PURE / TESTABLE HELPERS — journal analytics
# ===========================================================================

def pip_size(pair: str) -> float:
    """0.01 for JPY crosses, else 0.0001."""
    return 0.01 if "JPY" in (pair or "").upper() else 0.0001


def r_multiple(direction: str, entry: float, stop: float, exit_: float) -> float | None:
    """
    Trade result in units of initial risk (R). The single most useful
    normaliser: a +2R win and a -1R loss are comparable across pairs and sizes.
    """
    try:
        risk = abs(float(entry) - float(stop))
        if risk == 0:
            return None
        gain = (float(exit_) - float(entry)) if str(direction).lower().startswith("l") \
            else (float(entry) - float(exit_))
        return gain / risk
    except (TypeError, ValueError):
        return None


def pnl_pips(direction: str, entry: float, exit_: float, pair: str) -> float | None:
    try:
        raw = (float(exit_) - float(entry)) if str(direction).lower().startswith("l") \
            else (float(entry) - float(exit_))
        return raw / pip_size(pair)
    except (TypeError, ValueError):
        return None


def max_drawdown_additive(cum: pd.Series) -> float:
    """Largest peak-to-trough drop of an additive equity curve (e.g. cumulative R)."""
    if cum is None or len(cum) == 0:
        return 0.0
    return float((cum - cum.cummax()).min())


def enrich_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (R, pips) to a journal frame for closed trades."""
    df = df.copy()
    if df.empty:
        for c in ["R", "pips"]:
            df[c] = []
        return df
    df["R"] = df.apply(
        lambda r: r_multiple(r.get("direction"), r.get("entry"),
                             r.get("stop"), r.get("exit")), axis=1)
    df["pips"] = df.apply(
        lambda r: pnl_pips(r.get("direction"), r.get("entry"),
                           r.get("exit"), r.get("pair")), axis=1)
    return df


def journal_stats(df: pd.DataFrame) -> dict | None:
    """
    Edge metrics from closed trades. Expectancy (mean R) is the headline:
    positive = a real edge per trade; everything else explains it.
    """
    if df is None or df.empty:
        return None
    closed = df[(df.get("status").astype(str).str.lower() == "closed")
                if "status" in df else df.index].copy()
    closed = enrich_trades(closed)
    closed = closed.dropna(subset=["R"])
    if closed.empty:
        return None

    R = closed["R"].astype(float)
    wins, losses = R[R > 0], R[R <= 0]
    n = len(R)
    win_rate = len(wins) / n
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = float(R.mean())
    gross_win = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else np.inf

    order = closed.copy()
    order["_d"] = pd.to_datetime(order.get("closed"), errors="coerce")
    order = order.sort_values("_d")
    cum = order["R"].astype(float).cumsum()

    return {
        "n": n, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "expectancy": expectancy, "total_R": float(R.sum()),
        "profit_factor": profit_factor,
        "max_dd_R": max_drawdown_additive(cum),
        "equity_R": cum.reset_index(drop=True),
        "R_series": R.reset_index(drop=True),
        "by_setup": (closed.groupby("setup")["R"].agg(["count", "mean", "sum"])
                     if "setup" in closed else None),
        "by_pair": (closed.groupby("pair")["R"].agg(["count", "mean", "sum"])
                    if "pair" in closed else None),
    }


# ===========================================================================
# PURE / TESTABLE HELPERS — backtest engine
# ===========================================================================

def ma_signal(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """+1 long when fast SMA > slow SMA, -1 short otherwise. Classic trend filter."""
    f, s = close.rolling(fast).mean(), close.rolling(slow).mean()
    return pd.Series(np.where(f > s, 1.0, -1.0), index=close.index)


def donchian_signal(close: pd.Series, entry_n: int, exit_n: int) -> pd.Series:
    """Donchian breakout: long on a new entry_n-day high, flat on exit_n-day low."""
    hi = close.rolling(entry_n).max()
    lo = close.rolling(exit_n).min()
    pos = np.zeros(len(close))
    cur = 0.0
    for i in range(len(close)):
        c = close.iloc[i]
        if not np.isnan(hi.iloc[i]) and c >= hi.iloc[i]:
            cur = 1.0
        elif not np.isnan(lo.iloc[i]) and c <= lo.iloc[i]:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=close.index)


def volume_profile_poc(high: pd.Series, low: pd.Series, volume: pd.Series,
                       window: int, bins: int = 50) -> pd.Series:
    """
    Rolling Point of Control (POC) price per bar.

    For each bar, build a volume-by-price histogram over the trailing `window`
    bars: each bar spreads its volume uniformly across the price bins its
    high–low range spans (a daily-bar approximation of a true volume profile).
    The POC is the price of the most-traded bin. Uses only data up to the bar,
    so there's no look-ahead.
    """
    n = len(high)
    poc = np.full(n, np.nan)
    H, L, V = high.values, low.values, volume.values
    for t in range(window - 1, n):
        lo = np.nanmin(L[t - window + 1: t + 1])
        hi = np.nanmax(H[t - window + 1: t + 1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        vbin = np.zeros(bins)
        for i in range(t - window + 1, t + 1):
            dl, dh, dv = L[i], H[i], V[i]
            if not np.isfinite(dv) or dv <= 0:
                continue
            lo_idx = min(max(np.searchsorted(edges, dl) - 1, 0), bins - 1)
            hi_idx = min(max(np.searchsorted(edges, dh) - 1, 0), bins - 1)
            span = hi_idx - lo_idx + 1
            vbin[lo_idx:hi_idx + 1] += dv / span
        if vbin.sum() > 0:
            poc[t] = centers[int(np.argmax(vbin))]
    return pd.Series(poc, index=high.index)


def poc_reversion_signal(close: pd.Series, poc: pd.Series,
                         band_pct: float = 0.0) -> pd.Series:
    """
    Mean-reversion toward the POC: long when price sits below the POC (expect a
    move back up to fair value), short when above. `band_pct` is a deadband
    around the POC (as a % of price) to avoid churning when price hugs it.
    """
    diff = (poc - close)
    band = (band_pct / 100.0) * close
    sig = np.where(diff > band, 1.0, np.where(diff < -band, -1.0, 0.0))
    sig = pd.Series(sig, index=close.index)
    sig[poc.isna()] = 0.0
    return sig


def run_backtest(close: pd.Series, position: pd.Series, ann: int = 252) -> dict:
    """
    Vectorised daily close-to-close backtest. Position is shifted by one bar so
    today's return uses yesterday's signal (no look-ahead). Returns equity curve,
    summary stats, and per-trade results.
    """
    close = close.dropna()
    ret = close.pct_change().fillna(0.0)
    pos = position.reindex(close.index).fillna(0.0).shift(1).fillna(0.0)
    strat = pos * ret
    equity = (1.0 + strat).cumprod()

    # Per-trade returns: compound within runs of constant non-zero position.
    trades, cur_sign, cur = [], 0.0, 1.0
    for p, r in zip(pos.values, strat.values):
        sign = np.sign(p)
        if sign != cur_sign:
            if cur_sign != 0:
                trades.append(cur - 1.0)
            cur, cur_sign = 1.0, sign
        if sign != 0:
            cur *= (1.0 + r)
    if cur_sign != 0:
        trades.append(cur - 1.0)
    trades = np.array(trades, dtype=float)

    n = len(equity)
    total_return = float(equity.iloc[-1] - 1.0) if n else 0.0
    cagr = float(equity.iloc[-1] ** (ann / n) - 1.0) if n > 1 else 0.0
    sd = float(strat.std())
    sharpe = float(strat.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0
    maxdd = float((equity / equity.cummax() - 1.0).min()) if n else 0.0
    twins = trades[trades > 0]
    tloss = trades[trades <= 0]
    win_rate = len(twins) / len(trades) if len(trades) else 0.0
    pf = (twins.sum() / abs(tloss.sum())) if len(tloss) and tloss.sum() != 0 else np.inf

    return {
        "equity": equity, "strat_returns": strat,
        "total_return": total_return, "cagr": cagr, "sharpe": sharpe,
        "max_dd": maxdd, "n_trades": int(len(trades)),
        "win_rate": win_rate, "profit_factor": float(pf),
        "expectancy": float(trades.mean()) if len(trades) else 0.0,
        "trades": trades,
    }


# ===========================================================================
# PERSISTENCE
# ===========================================================================

def load_journal(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path)
            for c in JOURNAL_COLUMNS:
                if c not in df.columns:
                    df[c] = np.nan
            return df[JOURNAL_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=JOURNAL_COLUMNS)


def save_journal(df: pd.DataFrame, path: str) -> bool:
    try:
        df.to_csv(path, index=False)
        return True
    except Exception:
        return False


def _cache(ttl):
    return st.cache_data(ttl=ttl, show_spinner=False) if st is not None else (lambda f: f)


@_cache(ttl=3600)
def load_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Daily OHLCV via yfinance. Volume is real for stocks/ETFs/futures, but is
    tick-only or zero for spot FX (e.g. EURUSD=X) — POC needs real volume."""
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


# ===========================================================================
# UI — JOURNAL
# ===========================================================================

def render_journal():
    st.subheader("📓 Trade Journal")
    st.caption("Log the thesis and the outcome of every trade. Your edge — if you "
               "have one — lives in this data, the same way your dashboard's signals "
               "live in price and macro data.")

    path = st.text_input("Journal file (CSV path)", value=DEFAULT_JOURNAL_PATH)
    df = load_journal(path)

    with st.expander("➕ Log a trade", expanded=df.empty):
        c1, c2, c3 = st.columns(3)
        opened = c1.date_input("Opened")
        pair = c2.text_input("Pair", value="EUR/USD")
        direction = c3.selectbox("Direction", ["Long", "Short"])
        c4, c5, c6, c7 = st.columns(4)
        entry = c4.number_input("Entry", value=0.0, format="%.5f")
        stop = c5.number_input("Stop", value=0.0, format="%.5f")
        target = c6.number_input("Target", value=0.0, format="%.5f")
        size = c7.number_input("Size (lots/units)", value=0.0)
        c8, c9 = st.columns(2)
        setup = c8.text_input("Setup / strategy tag", value="")
        confidence = c9.slider("Confidence", 1, 5, 3)
        thesis = st.text_area("Thesis (why you took it)", height=70)

        cc1, cc2 = st.columns(2)
        closed = cc1.date_input("Closed (leave if open)", value=opened)
        exit_ = cc2.number_input("Exit (blank/0 if open)", value=0.0, format="%.5f")
        status = "closed" if exit_ and exit_ != 0 else "open"

        if st.button("Add trade", type="primary"):
            new = {
                "id": (int(df["id"].max()) + 1) if not df.empty
                       and df["id"].notna().any() else 1,
                "opened": str(opened), "closed": str(closed) if status == "closed" else "",
                "pair": pair, "direction": direction, "setup": setup,
                "entry": entry or np.nan, "stop": stop or np.nan,
                "target": target or np.nan, "exit": exit_ or np.nan,
                "size": size or np.nan, "status": status,
                "confidence": confidence, "thesis": thesis, "notes": "",
            }
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
            if save_journal(df, path):
                st.success(f"Logged trade #{new['id']} ({status}). Saved to {path}.")
            else:
                st.error("Could not write the CSV — check the path is writable.")

    st.markdown("**Your trades** (edit inline, then Save)")
    edited = st.data_editor(df, width="stretch", num_rows="dynamic",
                            hide_index=True, key="journal_editor")
    if st.button("💾 Save edits"):
        st.success("Saved.") if save_journal(edited, path) else st.error("Save failed.")

    stats = journal_stats(edited)
    if not stats:
        st.info("Add some **closed** trades (with entry, stop and exit) to see your "
                "edge metrics.")
        return

    st.divider()
    st.markdown("### 📈 Edge metrics (closed trades)")
    m = st.columns(4)
    m[0].metric("Expectancy", f"{stats['expectancy']:+.2f} R",
                help="Average R per trade. Positive = a real edge. THE headline number.")
    m[1].metric("Win rate", f"{stats['win_rate']*100:.0f}%")
    pf = stats["profit_factor"]
    m[2].metric("Profit factor", "∞" if np.isinf(pf) else f"{pf:.2f}",
                help="Gross win R ÷ gross loss R. >1 is profitable; >1.5 is strong.")
    m[3].metric("Total / Max DD", f"{stats['total_R']:+.1f}R",
                f"DD {stats['max_dd_R']:.1f}R")

    m2 = st.columns(2)
    m2[0].metric("Avg win", f"{stats['avg_win']:+.2f} R")
    m2[1].metric("Avg loss", f"{stats['avg_loss']:+.2f} R")
    st.caption("Reality check: you can be profitable with a sub-50% win rate if "
               "avg win ≫ avg loss. Expectancy is what matters, not win rate.")

    st.markdown("**Equity curve (cumulative R)**")
    st.line_chart(stats["equity_R"], height=240)
    st.markdown("**R-multiple distribution**")
    st.bar_chart(stats["R_series"], height=200)

    cc1, cc2 = st.columns(2)
    if stats["by_setup"] is not None and not stats["by_setup"].empty:
        with cc1:
            st.markdown("**By setup** (mean R, count)")
            st.dataframe(stats["by_setup"].round(2), width="stretch")
    if stats["by_pair"] is not None and not stats["by_pair"].empty:
        with cc2:
            st.markdown("**By pair** (mean R, count)")
            st.dataframe(stats["by_pair"].round(2), width="stretch")
    st.caption("The 'by setup' table is where edges hide: one tag with strong "
               "positive mean R and enough trades is your real strategy — do more "
               "of it, cut the rest.")


# ===========================================================================
# UI — BACKTEST
# ===========================================================================

def _trend_gate(position, close, ohlcv):
    """
    Shared UI toggle: gate a TREND strategy to trade only when the market is
    trending (stand aside in chop). Returns (gated_position, raw_or_None, note).
    """
    use = st.checkbox("🚦 Regime filter — only trade when the market is TRENDING "
                      "(stand aside in chop)", value=False, key="trend_gate")
    if not use:
        return position, None, ""
    f1, f2, f3 = st.columns(3)
    method = f1.selectbox("Filter", ["ADX", "Efficiency Ratio"], key="tg_method")
    flook = f2.slider("Filter lookback", 7, 50, 14, key="tg_look")
    if method == "ADX":
        thr = f3.slider("Trending if ADX >", 10, 40, 25, 1, key="tg_adx")
    else:
        thr = f3.slider("Trending if ER >", 0.10, 0.70, 0.35, 0.05, key="tg_er")
    allow = regime_mask(method, thr, "trending", close=close,
                        high=ohlcv["High"], low=ohlcv["Low"], n=flook)
    return apply_regime_filter(position, allow), position, \
        f"  Gated: only when {method} says trending."


def render_backtest():
    st.subheader("🔬 Backtest Lab")
    st.caption("Test a rule's edge on history before risking money. Daily "
               "close-to-close, signal shifted one bar so there's no look-ahead. "
               "A scaffold: the strategy functions are a few lines — extend them.")

    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker (yfinance)", value="SPY")
    period = c2.selectbox("History", ["2y", "5y", "10y", "max"], index=1)
    strat = c3.selectbox("Strategy",
                         ["MA trend", "Donchian breakout", "POC reversion"])

    ohlcv = load_ohlcv(ticker, period)
    if ohlcv.empty or "Close" not in ohlcv:
        st.warning("Could not load price data (yfinance unreachable).")
        return
    close = ohlcv["Close"].dropna()

    poc = None
    compare = None
    if strat == "MA trend":
        p1, p2 = st.columns(2)
        fast = p1.slider("Fast SMA", 5, 100, 20)
        slow = p2.slider("Slow SMA", 20, 300, 100)
        position = ma_signal(close, fast, slow)
        desc = f"Long when SMA{fast} > SMA{slow}, short otherwise."
        position, compare, note = _trend_gate(position, close, ohlcv)
        desc += note
    elif strat == "Donchian breakout":
        p1, p2 = st.columns(2)
        entry_n = p1.slider("Breakout lookback (days)", 10, 100, 20)
        exit_n = p2.slider("Exit lookback (days)", 5, 60, 10)
        position = donchian_signal(close, entry_n, exit_n)
        desc = f"Long on a new {entry_n}-day high, flat on a {exit_n}-day low."
        position, compare, note = _trend_gate(position, close, ohlcv)
        desc += note
    else:  # POC reversion
        vol = ohlcv["Volume"] if "Volume" in ohlcv else pd.Series(dtype=float)
        if vol.fillna(0).sum() <= 0:
            st.error(
                f"**{ticker} has no real volume** (spot FX shows tick or zero "
                "volume), so a volume-profile POC isn't meaningful here. Use an "
                "instrument with real exchange volume — e.g. SPY, QQQ, ES=F, NQ=F.")
            return
        p1, p2, p3 = st.columns(3)
        window = p1.slider("Profile window (days)", 10, 120, 40)
        bins = p2.slider("Price bins", 20, 100, 50)
        band = p3.slider("Deadband around POC (%)", 0.0, 2.0, 0.3, 0.1)
        poc = volume_profile_poc(ohlcv["High"], ohlcv["Low"], vol, window, bins)
        raw_signal = poc_reversion_signal(close, poc, band_pct=band)
        position = raw_signal
        desc = (f"Fade toward the {window}-day volume POC: long below it, short "
                f"above it, flat within ±{band:.1f}%.")

        use_filter = st.checkbox(
            "🚦 Regime filter — only fade in balanced (ranging) markets, stand "
            "aside in trends", value=True)
        if use_filter:
            f1, f2, f3 = st.columns(3)
            method = f1.selectbox("Filter", ["Efficiency Ratio", "ADX"])
            flook = f2.slider("Filter lookback", 7, 50, 20)
            if method == "Efficiency Ratio":
                thr = f3.slider("Ranging if ER <", 0.10, 0.60, 0.30, 0.05)
            else:
                thr = f3.slider("Ranging if ADX <", 10, 35, 22, 1)
            ranging = regime_mask(method, thr, "ranging", close=close,
                                  high=ohlcv["High"], low=ohlcv["Low"], n=flook)
            position = apply_regime_filter(raw_signal, ranging)
            compare = raw_signal  # keep the unfiltered version for side-by-side
            desc += f"  Filtered: only when {method} says ranging."

    res = run_backtest(close, position)
    bh = (close / close.iloc[0])
    st.caption(desc)

    m = st.columns(4)
    m[0].metric("Total return", f"{res['total_return']*100:+.0f}%")
    m[1].metric("CAGR", f"{res['cagr']*100:+.1f}%")
    m[2].metric("Sharpe", f"{res['sharpe']:.2f}")
    m[3].metric("Max drawdown", f"{res['max_dd']*100:.0f}%")
    m2 = st.columns(4)
    m2[0].metric("Trades", res["n_trades"])
    m2[1].metric("Win rate", f"{res['win_rate']*100:.0f}%")
    pf = res["profit_factor"]
    m2[2].metric("Profit factor", "∞" if np.isinf(pf) else f"{pf:.2f}")
    m2[3].metric("Expectancy", f"{res['expectancy']*100:+.2f}%/trade")

    comp = pd.concat([res["equity"].rename("Strategy"),
                      bh.rename("Buy & hold")], axis=1).dropna()
    st.markdown("**Equity curve vs buy & hold**")
    st.line_chart(comp, height=300)
    st.caption("Beating buy & hold on a risk-adjusted basis (Sharpe, drawdown) is "
               "the bar. A strategy that only wins by taking more risk hasn't found "
               "an edge. Beware overfitting: if great results need very specific "
               "parameters, it won't hold live — test on out-of-sample data.")

    if poc is not None:
        st.markdown("**Price vs rolling POC**")
        pv = pd.concat([close.rename("Price"), poc.rename("POC")], axis=1).dropna()
        st.line_chart(pv.tail(400), height=260)
        st.caption("Price oscillating around the POC is the mean-reversion premise. "
                   "Where price trends away and *stays* away, the fade loses — that's "
                   "the regime where POC reversion gets run over.")

    if compare is not None:
        raw_res = run_backtest(close, compare)
        def _row(name, r, pos):
            pf = r["profit_factor"]
            return {
                "Version": name,
                "Total %": round(r["total_return"] * 100, 1),
                "Sharpe": round(r["sharpe"], 2),
                "Max DD %": round(r["max_dd"] * 100, 1),
                "Win %": round(r["win_rate"] * 100, 0),
                "Profit factor": "∞" if np.isinf(pf) else round(pf, 2),
                "Trades": r["n_trades"],
                "Time in mkt %": round(float((pos != 0).mean()) * 100, 0),
            }
        cmp_df = pd.DataFrame([
            _row("Raw POC", raw_res, compare),
            _row("Filtered", res, position),
        ])
        st.markdown("### 🚦 Does the regime filter earn its keep?")
        st.dataframe(cmp_df, width="stretch", hide_index=True)
        st.caption("The filter should trade **less** (lower time in market) but "
                   "**better** (higher Sharpe and profit factor, shallower "
                   "drawdown). If filtering doesn't lift risk-adjusted return, the "
                   "POC edge isn't really there for this instrument — which is "
                   "itself a useful, money-saving answer.")

    st.info("Extension point: write any function returning a +1/0/−1 position "
            "Series indexed like `close` and pass it to run_backtest() — e.g. a "
            "rate-differential signal from your Forex Fundamentals tab, so you can "
            "backtest *fundamentals*, not just price.")


def render():
    """Call this inside your dashboard tab."""
    st.header("🧪 Trading Lab — journal + backtest")
    st.caption("The two tools that move real odds: a journal to find your edge in "
               "your own trades, and a backtest engine to test edges before you risk "
               "money.")
    t1, t2 = st.tabs(["📓 Trade Journal", "🔬 Backtest Lab"])
    with t1:
        render_journal()
    with t2:
        render_backtest()


if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run trading_lab_tab.py")

    # Page bootstrap — kept lazy so the pure helpers above can still be imported
    # headlessly. Mirrors the legacy-page contract used across the app:
    # set_page_config → BloombergTheme.apply() → sidebar (header → divider → nav).
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="Trading Lab · Trading System",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()

    with st.sidebar:
        st.markdown("### 🔬 Trading Lab")
        st.caption("Trade journal + backtest engine")
        st.divider()
        render_sidebar_nav()

    render()