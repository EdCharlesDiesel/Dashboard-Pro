"""Production backtest engine — single source of truth for the 18-point
workflow backtest used by the Backtest Lab page and the `run_backtest.py` CLI.

The strategy logic (checks + trade simulation) is preserved byte-for-byte from
the original `pages/backtest-workflow.py` so historical results don't move:

- `run_checks()` evaluates all 18 workflow checks on the data available up to a
  bar (15M checks are proxied from daily bars — documented on the page).
- `simulate_trade()` manages an ATR-stop / R-multiple-target trade bar-by-bar,
  with optional spread+slippage cost modelling (off by default to stay
  equivalent to the legacy engine).

On top of that it adds the quant layer pages render:

- `compute_stats()` — full performance stats (win rate, expectancy, profit
  factor, Sharpe/Sortino/Calmar, SQN, CAGR, max drawdown %, streaks, exposure).
- `monte_carlo_resample()` — bootstraps trade order to get equity-curve
  confidence bands, drawdown distribution and risk-of-ruin.
- `walk_forward_segments()` — period-by-period out-of-sample consistency.
- `parameter_sweep()` — grid over SL×ATR / R:R at a fixed entry score.
- `portfolio_backtest()` — run the strategy across many instruments and
  aggregate into one combined equity curve + per-pair leaderboard.

Everything heavy is `@st.cache_data`-wrapped and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ══════════════════════════════════════════════════════════════════
# INDICATORS  (preserved exactly from the legacy backtest engine —
# do not swap for src/indicators; the RSI/ATR formulas differ slightly
# and changing them would move every historical backtest result)
# ══════════════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    hist = calc_ema(series, fast) - calc_ema(series, slow)
    sig = calc_ema(hist, signal)
    return hist - sig  # histogram


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calc_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_k = df["Low"].rolling(k).min()
    high_k = df["High"].rolling(k).max()
    pct_k = 100 * (df["Close"] - low_k) / (high_k - low_k + 1e-9)
    return pct_k, pct_k.rolling(d).mean()


def swing_highs(series: pd.Series, strength: int = 3) -> List[int]:
    idx = []
    for i in range(strength, len(series) - strength):
        window = series.iloc[i - strength: i + strength + 1]
        if series.iloc[i] == window.max():
            idx.append(i)
    return idx


def swing_lows(series: pd.Series, strength: int = 3) -> List[int]:
    idx = []
    for i in range(strength, len(series) - strength):
        window = series.iloc[i - strength: i + strength + 1]
        if series.iloc[i] == window.min():
            idx.append(i)
    return idx


# ══════════════════════════════════════════════════════════════════
# CHECK DEFINITIONS
# ══════════════════════════════════════════════════════════════════

# key, label, mode (auto|calc), is_critical
CHECK_META: List[Tuple[str, str, str, bool]] = [
    ("macro_bias",       "02. Macro Bias Confirmed",     "auto", True),
    ("news_filter",      "03. News Filter Clear",        "auto", True),
    ("correlations",     "03. Correlation Exposure OK",  "auto", False),
    ("atr_volatility",   "05. ATR Volatility OK",        "calc", False),
    ("weekly_ema",       "06. Weekly EMA Aligned",       "calc", True),
    ("weekly_rsi",       "07. Weekly RSI has Room",      "calc", False),
    ("weekly_swing",     "08. Weekly Swing Structure",   "calc", False),
    ("daily_trend",      "09. Daily Trend Intact",       "calc", True),
    ("daily_macd",       "10. Daily MACD Momentum",      "calc", False),
    ("4h_confluence",    "11. 4H Confluence Zone",       "calc", False),
    ("confluence_check", "11. 2/3 Confluences Met",      "calc", False),
    ("rejection",        "13. 15M Rejection Candle",     "calc", False),
    ("entry_signal",     "14. 15M Entry Signal",         "calc", False),
    ("stop_structure",   "14. Stop Below Structure",     "calc", True),
    ("rr_check",         "15. R:R ≥ 2:1",           "calc", True),
    ("loss_limit",       "16. Daily Loss Limit OK",      "auto", False),
    ("mkt_structure",    "18. Market Structure Intact",  "calc", False),
    ("setup_rank",       "18. Setup Score ≥ 14/18", "calc", False),
]

CRITICAL_KEYS: List[str] = [k for k, _, mode, crit in CHECK_META if mode == "calc" and crit]


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_data(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Daily (1y) + weekly (3y) OHLC, MultiIndex-flattened and tz-naive."""
    from src.db.market_cache import cached_ohlc
    daily = cached_ohlc(ticker, period="1y", interval="1d", ttl=1800)
    weekly = cached_ohlc(ticker, period="3y", interval="1wk", ttl=1800)
    daily.index = _tz_naive(daily.index)
    weekly.index = _tz_naive(weekly.index)
    return daily.dropna(), weekly.dropna()


def _tz_naive(idx) -> pd.DatetimeIndex:
    """Drop tz info — works whether the source frame is tz-aware (live yfinance)
    or already tz-naive (reconstructed from Postgres)."""
    idx = pd.to_datetime(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx


# ══════════════════════════════════════════════════════════════════
# CHECKS
# ══════════════════════════════════════════════════════════════════

def run_checks(d: pd.DataFrame, w: pd.DataFrame, direction: str) -> Tuple[Dict[str, Tuple[bool, str]], int]:
    """Evaluate every computable check on the last bar of `d`/`w`."""
    res: Dict[str, Tuple[bool, str]] = {}

    for key in ("macro_bias", "news_filter", "correlations", "loss_limit"):
        res[key] = (True, "Auto-pass in backtest (manual confirmation required live)")

    # ── ATR Volatility ───────────────────────────────────────────────
    if len(d) >= 20:
        atr = calc_atr(d, 14).iloc[-1]
        price = d["Close"].iloc[-1]
        ratio = atr / price
        res["atr_volatility"] = (0.001 <= ratio <= 0.05, f"ATR/Price = {ratio*100:.2f}%")
    else:
        res["atr_volatility"] = (False, "Insufficient data")

    # ── Weekly EMA ───────────────────────────────────────────────────
    if len(w) >= 55:
        w_ema20 = calc_ema(w["Close"], 20).iloc[-1]
        w_ema50 = calc_ema(w["Close"], 50).iloc[-1]
        w_price = w["Close"].iloc[-1]
        ok = (w_price > w_ema20 > w_ema50) if direction == "Long" else (w_price < w_ema20 < w_ema50)
        res["weekly_ema"] = (ok, f"EMA20={w_ema20:.5f}  EMA50={w_ema50:.5f}  Price={w_price:.5f}")
    else:
        res["weekly_ema"] = (False, "Insufficient weekly data")

    # ── Weekly RSI ───────────────────────────────────────────────────
    if len(w) >= 18:
        w_rsi = calc_rsi(w["Close"], 14).iloc[-1]
        ok = (w_rsi < 65) if direction == "Long" else (w_rsi > 35)
        res["weekly_rsi"] = (ok, f"Weekly RSI = {w_rsi:.1f}")
    else:
        res["weekly_rsi"] = (False, "Insufficient weekly data")

    # ── Weekly Swing Structure ───────────────────────────────────────
    sh = swing_highs(w["High"], 2)
    sl = swing_lows(w["Low"], 2)
    if len(sh) >= 2 and len(sl) >= 2:
        hh = w["High"].iloc[sh[-1]] > w["High"].iloc[sh[-2]]
        hl = w["Low"].iloc[sl[-1]] > w["Low"].iloc[sl[-2]]
        ll = w["Low"].iloc[sl[-1]] < w["Low"].iloc[sl[-2]]
        lh = w["High"].iloc[sh[-1]] < w["High"].iloc[sh[-2]]
        if direction == "Long":
            res["weekly_swing"] = (hh and hl, f"HH={hh}  HL={hl}")
        else:
            res["weekly_swing"] = (ll and lh, f"LL={ll}  LH={lh}")
    else:
        res["weekly_swing"] = (False, "Not enough weekly swing points")

    # ── Daily Trend ──────────────────────────────────────────────────
    if len(d) >= 52:
        d_ema20 = calc_ema(d["Close"], 20).iloc[-1]
        d_ema50 = calc_ema(d["Close"], 50).iloc[-1]
        d_price = d["Close"].iloc[-1]
        if direction == "Long":
            ok = d_ema20 > d_ema50 and d_price > d_ema20
        else:
            ok = d_ema20 < d_ema50 and d_price < d_ema20
        res["daily_trend"] = (ok, f"EMA20={d_ema20:.5f}  EMA50={d_ema50:.5f}  Price={d_price:.5f}")
    else:
        res["daily_trend"] = (False, "Insufficient daily data")

    # ── Daily MACD ───────────────────────────────────────────────────
    if len(d) >= 35:
        hist = calc_macd(d["Close"])
        h_now, h_prev = hist.iloc[-1], hist.iloc[-2]
        if direction == "Long":
            ok = h_now > 0 or (h_now > h_prev and h_now > -0.001)
        else:
            ok = h_now < 0 or (h_now < h_prev and h_now < 0.001)
        res["daily_macd"] = (ok, f"MACD hist = {h_now:.6f}  (prev {h_prev:.6f})")
    else:
        res["daily_macd"] = (False, "Insufficient data")

    # ── 4H Confluence Zone (approximated from daily) ─────────────────
    if len(d) >= 20:
        atr = calc_atr(d, 14).iloc[-1]
        price = d["Close"].iloc[-1]
        tol = atr * 0.6
        recent_high = d["High"].rolling(20).max().iloc[-1]
        recent_low = d["Low"].rolling(20).min().iloc[-1]
        fib_range = recent_high - recent_low
        fib_levels = [recent_high - r * fib_range for r in (0.236, 0.382, 0.500, 0.618, 0.786)]
        near_fib = any(abs(price - f) <= tol for f in fib_levels)

        prev = d.iloc[-2]
        pp = (prev["High"] + prev["Low"] + prev["Close"]) / 3
        s1, r1 = 2 * pp - prev["High"], 2 * pp - prev["Low"]
        near_piv = any(abs(price - lvl) <= tol for lvl in (pp, s1, r1))

        d_ema20 = calc_ema(d["Close"], 20).iloc[-1]
        near_ema = abs(price - d_ema20) <= tol

        count = sum([near_fib, near_piv, near_ema])
        res["4h_confluence"] = (count >= 1, f"Fib={near_fib}  Pivot={near_piv}  EMA={near_ema}  ({count}/3)")
        res["confluence_check"] = (count >= 2, f"{count}/3 confluences  (min 2 required)")
    else:
        res["4h_confluence"] = (False, "Insufficient data")
        res["confluence_check"] = (False, "Insufficient data")

    # ── 15M Rejection Candle (proxied via daily close position) ──────
    if len(d) >= 3:
        bar = d.iloc[-1]
        rng = bar["High"] - bar["Low"]
        if rng > 0:
            pos = (bar["Close"] - bar["Low"]) / rng
            ok = (pos >= 0.60) if direction == "Long" else (pos <= 0.40)
            tag = "bullish" if pos >= 0.6 else "bearish" if pos <= 0.4 else "neutral"
            res["rejection"] = (ok, f"Close position in day range: {pos*100:.0f}%  ({tag})")
        else:
            res["rejection"] = (False, "Doji — no range")
    else:
        res["rejection"] = (False, "Insufficient data")

    # ── 15M Entry Signal — Stochastic proxy ──────────────────────────
    if len(d) >= 20:
        k, k_d = calc_stochastic(d, k=14, d=3)
        k_now, k_prev = k.iloc[-1], k.iloc[-2]
        d_now, d_prev = k_d.iloc[-1], k_d.iloc[-2]
        if direction == "Long":
            cross_up = k_prev < d_prev and k_now >= d_now
            ok = cross_up and k_prev < 50
            res["entry_signal"] = (ok, f"Stoch K={k_now:.1f}  D={d_now:.1f}  cross_up={cross_up}  was_below_50={k_prev < 50}")
        else:
            cross_dn = k_prev > d_prev and k_now <= d_now
            ok = cross_dn and k_prev > 50
            res["entry_signal"] = (ok, f"Stoch K={k_now:.1f}  D={d_now:.1f}  cross_down={cross_dn}  was_above_50={k_prev > 50}")
    else:
        res["entry_signal"] = (False, "Insufficient data")

    # ── Stop Structure ───────────────────────────────────────────────
    if len(d) >= 12:
        price = d["Close"].iloc[-1]
        atr = calc_atr(d, 14).iloc[-1]
        sh_idx = swing_highs(d["High"], 3)
        sl_idx = swing_lows(d["Low"], 3)
        if direction == "Long" and sl_idx:
            level = d["Low"].iloc[sl_idx[-1]]
            sl_dist = price - level
            res["stop_structure"] = (0 < sl_dist < atr * 3, f"Last swing low = {level:.5f}  SL dist = {sl_dist:.5f} ({sl_dist/atr:.1f}× ATR)")
        elif direction == "Short" and sh_idx:
            level = d["High"].iloc[sh_idx[-1]]
            sl_dist = level - price
            res["stop_structure"] = (0 < sl_dist < atr * 3, f"Last swing high = {level:.5f}  SL dist = {sl_dist:.5f} ({sl_dist/atr:.1f}× ATR)")
        else:
            res["stop_structure"] = (False, "No swing point found")
    else:
        res["stop_structure"] = (False, "Insufficient data")

    # ── R:R ≥ 2:1 ────────────────────────────────────────────────────
    if len(d) >= 14:
        atr = calc_atr(d, 14).iloc[-1]
        sl_d = atr * 1.5
        tp1_d = sl_d * 2.0
        res["rr_check"] = (True, f"ATR-based R:R = {tp1_d/sl_d:.1f}:1  (SL={sl_d:.5f}  TP1={tp1_d:.5f})")
    else:
        res["rr_check"] = (False, "Insufficient data")

    # ── Market Structure (daily swing) ───────────────────────────────
    sh_idx = swing_highs(d["High"], 3)
    sl_idx = swing_lows(d["Low"], 3)
    if len(sh_idx) >= 2 and len(sl_idx) >= 2:
        hh = d["High"].iloc[sh_idx[-1]] > d["High"].iloc[sh_idx[-2]]
        hl = d["Low"].iloc[sl_idx[-1]] > d["Low"].iloc[sl_idx[-2]]
        ll = d["Low"].iloc[sl_idx[-1]] < d["Low"].iloc[sl_idx[-2]]
        lh = d["High"].iloc[sh_idx[-1]] < d["High"].iloc[sh_idx[-2]]
        if direction == "Long":
            res["mkt_structure"] = (hh and hl, f"Daily structure: HH={hh}  HL={hl}")
        else:
            res["mkt_structure"] = (ll and lh, f"Daily structure: LL={ll}  LH={lh}")
    else:
        res["mkt_structure"] = (False, "Insufficient swing history")

    # ── Setup Rank (aggregate) ───────────────────────────────────────
    score = sum(v for v, _ in res.values())
    res["setup_rank"] = (score >= 14, f"Score = {score}/18")

    return res, score


# ══════════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════════

def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str,
                   sl_mult: float = 1.5, rr: float = 2.0, max_bars: int = 20,
                   cost_price: float = 0.0) -> dict:
    """Bar-by-bar trade management. `cost_price` is the round-trip cost
    (spread + slippage) in price units, applied as a worse entry fill."""
    raw_entry = df["Open"].iloc[entry_idx]
    atr = calc_atr(df.iloc[:entry_idx], 14).iloc[-1]
    sl_d = atr * sl_mult

    # Worse fill: pay the cost on entry (long fills higher, short fills lower).
    entry = raw_entry + cost_price if direction == "Long" else raw_entry - cost_price

    if direction == "Long":
        sl, tp1, tp2 = entry - sl_d, entry + sl_d * rr, entry + sl_d * (rr + 1)
    else:
        sl, tp1, tp2 = entry + sl_d, entry - sl_d * rr, entry - sl_d * (rr + 1)

    outcome = "Timeout"
    exit_price = df["Close"].iloc[min(entry_idx + max_bars, len(df) - 1)]
    tp1_hit = False
    active_sl = sl
    bars_held = 0

    for i in range(entry_idx, min(entry_idx + max_bars, len(df))):
        bars_held = i - entry_idx + 1
        bar = df.iloc[i]
        if direction == "Long":
            if bar["Low"] <= active_sl:
                outcome = "TP1 → BE" if tp1_hit else "SL"
                exit_price = active_sl
                break
            if not tp1_hit and bar["High"] >= tp1:
                tp1_hit = True
                active_sl = entry
            if tp1_hit and bar["High"] >= tp2:
                outcome = "TP2"
                exit_price = tp2
                break
        else:
            if bar["High"] >= active_sl:
                outcome = "TP1 → BE" if tp1_hit else "SL"
                exit_price = active_sl
                break
            if not tp1_hit and bar["Low"] <= tp1:
                tp1_hit = True
                active_sl = entry
            if tp1_hit and bar["Low"] <= tp2:
                outcome = "TP2"
                exit_price = tp2
                break

    r_mult = ((exit_price - entry) / sl_d) if direction == "Long" else ((entry - exit_price) / sl_d)
    return dict(entry=entry, exit=exit_price, sl=sl, tp1=tp1, tp2=tp2,
                outcome=outcome, r=round(r_mult, 2), sl_d=sl_d, bars_held=bars_held)


# ══════════════════════════════════════════════════════════════════
# BACKTEST RUNNER
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def run_backtest(daily: pd.DataFrame, weekly: pd.DataFrame, direction: str,
                 min_score: int, sl_mult: float, rr: float,
                 start_dt: date, end_dt: date, cost_price: float = 0.0) -> List[dict]:
    """Walk the daily series; take a trade at next-bar open whenever the score
    clears `min_score` and every critical check passes."""
    trades: List[dict] = []
    warmup = 55
    in_trade = False

    for i in range(warmup, len(daily)):
        bar_date = daily.index[i].date()
        if bar_date < start_dt or bar_date > end_dt:
            continue
        if in_trade:
            in_trade = False
            continue

        d_slice = daily.iloc[:i + 1]
        w_slice = weekly[weekly.index <= daily.index[i]]
        if len(w_slice) < 20:
            continue

        res, score = run_checks(d_slice, w_slice, direction)
        critical_pass = all(res.get(k, (False,))[0] for k in CRITICAL_KEYS)

        if score >= min_score and critical_pass:
            if i + 1 >= len(daily):
                continue
            trade = simulate_trade(daily, i + 1, direction, sl_mult, rr, cost_price=cost_price)
            trades.append({"date": bar_date, "score": score, **trade})
            in_trade = True

    return trades


# ══════════════════════════════════════════════════════════════════
# PERFORMANCE STATS  (the quant layer)
# ══════════════════════════════════════════════════════════════════

def build_equity(trades: List[dict], account: float, risk_pct: float) -> pd.DataFrame:
    """Compounded equity curve: each trade risks `risk_pct` of current equity
    and returns r × that risk. Returns a frame indexed 0..N with date/equity."""
    eq = account
    rows = [{"date": None, "equity": account, "ret": 0.0}]
    risk_frac = risk_pct / 100.0
    for t in trades:
        pnl = eq * risk_frac * t["r"]
        eq += pnl
        rows.append({"date": t["date"], "equity": eq, "ret": risk_frac * t["r"]})
    return pd.DataFrame(rows)


def _max_consecutive(mask: List[bool]) -> int:
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def compute_stats(trades: List[dict], account: float = 10000.0,
                  risk_pct: float = 1.0, rr: float = 2.0) -> dict:
    """Full performance statistics for a trade list."""
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    r = df["r"].to_numpy(dtype=float)
    wins = r[r > 0]
    losses = r[r <= 0]
    n = len(r)

    win_rate = len(wins) / n * 100
    avg_r = float(r.mean())
    std_r = float(r.std(ddof=1)) if n > 1 else 0.0
    net_r = float(r.sum())
    gross_p = float(wins.sum())
    gross_l = float(abs(losses.sum())) or 1.0
    profit_factor = gross_p / gross_l
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    expectancy = avg_r  # R per trade

    # Cumulative-R drawdown (additive, R units)
    cumr = np.cumsum(r)
    peak_r = np.maximum.accumulate(cumr)
    dd_r = float((cumr - peak_r).min())

    # Compounded equity stats
    eq = build_equity(trades, account, risk_pct)
    equity = eq["equity"].to_numpy(dtype=float)
    peak_e = np.maximum.accumulate(equity)
    dd_pct = float(((equity - peak_e) / peak_e).min() * 100)
    final_equity = float(equity[-1])
    total_return_pct = (final_equity / account - 1) * 100

    # Annualisation from the actual date span
    dates = [t["date"] for t in trades]
    span_days = max((dates[-1] - dates[0]).days, 1)
    years = span_days / 365.25
    trades_per_year = n / years if years > 0 else float(n)
    cagr = ((final_equity / account) ** (1 / years) - 1) * 100 if years > 0 and final_equity > 0 else float("nan")

    # Risk-adjusted (per-trade ratios annualised by trade frequency)
    sharpe = (avg_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else float("nan")
    downside = r[r < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (avg_r / dstd) * np.sqrt(trades_per_year) if dstd > 0 else float("nan")
    calmar = (cagr / abs(dd_pct)) if dd_pct < 0 else float("nan")
    sqn = (np.sqrt(n) * avg_r / std_r) if std_r > 0 else float("nan")  # Van Tharp

    # Streaks & exposure
    max_win_streak = _max_consecutive(list(r > 0))
    max_loss_streak = _max_consecutive(list(r <= 0))
    avg_bars = float(df["bars_held"].mean()) if "bars_held" in df else float("nan")

    return dict(
        total=n, wins=len(wins), losses=len(losses),
        win_rate=win_rate, avg_r=avg_r, std_r=std_r, net_r=net_r,
        profit_factor=profit_factor, max_dd_r=dd_r, max_dd_pct=dd_pct,
        avg_win=avg_win, avg_loss=avg_loss, payoff=payoff, expectancy=expectancy,
        final_equity=final_equity, total_return_pct=total_return_pct, cagr=cagr,
        sharpe=sharpe, sortino=sortino, calmar=calmar, sqn=sqn,
        trades_per_year=trades_per_year, max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak, avg_bars=avg_bars,
        breakeven_wr=1.0 / (1.0 + rr) * 100,
    )


# ══════════════════════════════════════════════════════════════════
# MONTE CARLO  — trade-order resampling
# ══════════════════════════════════════════════════════════════════

@dataclass
class MonteCarloResult:
    final_returns: np.ndarray       # terminal % return per simulation
    max_drawdowns: np.ndarray       # max DD % per simulation (negative)
    p_profit: float
    median_return: float
    q05_return: float
    q95_return: float
    median_dd: float
    worst_dd: float                 # 5th percentile DD
    risk_of_ruin: float             # P(equity ever drops below ruin level)
    sample_curves: List[np.ndarray] = field(default_factory=list)


def monte_carlo_resample(trades: List[dict], account: float = 10000.0,
                         risk_pct: float = 1.0, n_sims: int = 2000,
                         ruin_drawdown_pct: float = 50.0,
                         seed: int = 20260614) -> Optional[MonteCarloResult]:
    """Bootstrap the order (and selection, with replacement) of realised trades
    to map the distribution of outcomes the strategy could have produced."""
    if len(trades) < 5:
        return None

    r = np.array([t["r"] for t in trades], dtype=float)
    rng = np.random.default_rng(seed)
    risk_frac = risk_pct / 100.0
    n = len(r)

    finals = np.empty(n_sims)
    dds = np.empty(n_sims)
    ruin_hits = 0
    ruin_level = account * (1 - ruin_drawdown_pct / 100.0)
    sample_curves: List[np.ndarray] = []

    for s in range(n_sims):
        seq = rng.choice(r, size=n, replace=True)
        equity = account * np.cumprod(1 + risk_frac * seq)
        equity = np.concatenate([[account], equity])
        peak = np.maximum.accumulate(equity)
        dd = ((equity - peak) / peak).min() * 100
        finals[s] = (equity[-1] / account - 1) * 100
        dds[s] = dd
        if equity.min() <= ruin_level:
            ruin_hits += 1
        if s < 200:
            sample_curves.append(equity)

    return MonteCarloResult(
        final_returns=finals,
        max_drawdowns=dds,
        p_profit=float((finals > 0).mean()),
        median_return=float(np.median(finals)),
        q05_return=float(np.percentile(finals, 5)),
        q95_return=float(np.percentile(finals, 95)),
        median_dd=float(np.median(dds)),
        worst_dd=float(np.percentile(dds, 5)),
        risk_of_ruin=ruin_hits / n_sims,
        sample_curves=sample_curves,
    )


# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD  — out-of-sample consistency across periods
# ══════════════════════════════════════════════════════════════════

def walk_forward_segments(trades: List[dict], n_segments: int = 4,
                          start_dt: Optional[date] = None,
                          end_dt: Optional[date] = None) -> List[dict]:
    """Split the test window into equal calendar segments and report each
    segment's stats — a fixed-parameter system should perform consistently."""
    if not trades:
        return []
    dates = [t["date"] for t in trades]
    lo = start_dt or min(dates)
    hi = end_dt or max(dates)
    total_days = max((hi - lo).days, 1)
    seg_len = total_days / n_segments

    segments = []
    for s in range(n_segments):
        seg_lo = lo + timedelta(days=int(seg_len * s))
        seg_hi = lo + timedelta(days=int(seg_len * (s + 1))) if s < n_segments - 1 else hi
        seg_trades = [t for t in trades if seg_lo <= t["date"] <= seg_hi]
        if seg_trades:
            r = np.array([t["r"] for t in seg_trades])
            segments.append(dict(
                label=f"{seg_lo:%b %d} – {seg_hi:%b %d}",
                trades=len(seg_trades),
                net_r=float(r.sum()),
                win_rate=float((r > 0).mean() * 100),
                avg_r=float(r.mean()),
            ))
        else:
            segments.append(dict(label=f"{seg_lo:%b %d} – {seg_hi:%b %d}",
                                 trades=0, net_r=0.0, win_rate=0.0, avg_r=0.0))
    return segments


# ══════════════════════════════════════════════════════════════════
# PARAMETER SWEEP  — SL×ATR vs R:R grid at a fixed entry score
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def parameter_sweep(daily: pd.DataFrame, weekly: pd.DataFrame, direction: str,
                    min_score: int, sl_grid: Tuple[float, ...],
                    rr_grid: Tuple[float, ...], start_dt: date, end_dt: date,
                    metric: str = "net_r") -> pd.DataFrame:
    """Run the backtest over every (sl_mult, rr) combination and tabulate one
    metric. Returns a tidy frame: rows=sl_mult, cols=rr, values=metric."""
    grid = pd.DataFrame(index=list(sl_grid), columns=list(rr_grid), dtype=float)
    for sl_mult in sl_grid:
        for rr in rr_grid:
            trades = run_backtest(daily, weekly, direction, min_score,
                                  float(sl_mult), float(rr), start_dt, end_dt)
            if trades:
                stats = compute_stats(trades, rr=float(rr))
                grid.loc[sl_mult, rr] = stats.get(metric, np.nan)
            else:
                grid.loc[sl_mult, rr] = np.nan
    return grid


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO  — strategy across many instruments
# ══════════════════════════════════════════════════════════════════

def portfolio_backtest(tickers: Dict[str, str], direction: str, min_score: int,
                       sl_mult: float, rr: float, start_dt: date, end_dt: date,
                       account: float = 10000.0, risk_pct: float = 1.0) -> dict:
    """Run the strategy on each instrument, build a per-pair leaderboard and a
    combined, date-ordered equity curve (each trade risks `risk_pct`)."""
    leaderboard = []
    all_trades = []  # (date, pair, r)

    for pair, ticker in tickers.items():
        try:
            daily, weekly = load_data(ticker)
        except Exception:
            continue
        if daily.empty or len(daily) < 60:
            continue
        trades = run_backtest(daily, weekly, direction, min_score, sl_mult, rr, start_dt, end_dt)
        if not trades:
            leaderboard.append(dict(pair=pair, trades=0, net_r=0.0, win_rate=float("nan"),
                                    profit_factor=float("nan")))
            continue
        stats = compute_stats(trades, account, risk_pct, rr)
        leaderboard.append(dict(
            pair=pair, trades=stats["total"], net_r=stats["net_r"],
            win_rate=stats["win_rate"], profit_factor=stats["profit_factor"],
        ))
        for t in trades:
            all_trades.append((t["date"], pair, t["r"]))

    # Combined, chronological equity (sequential risk on one account)
    all_trades.sort(key=lambda x: x[0])
    eq = account
    risk_frac = risk_pct / 100.0
    curve = [{"date": None, "equity": account}]
    for dt, pair, r in all_trades:
        eq += eq * risk_frac * r
        curve.append({"date": dt, "pair": pair, "equity": eq})

    combined_r = sum(r for _, _, r in all_trades)
    return dict(
        leaderboard=sorted(leaderboard, key=lambda x: x["net_r"], reverse=True),
        equity_curve=pd.DataFrame(curve),
        total_trades=len(all_trades),
        combined_net_r=combined_r,
        final_equity=eq,
    )
