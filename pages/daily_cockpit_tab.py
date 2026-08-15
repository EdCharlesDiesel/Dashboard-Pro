"""
daily_cockpit_tab.py
===================
Your pre-market routine on ONE screen. Instead of clicking through seven tabs,
the cockpit assembles the key reads and tells you where they AGREE:

  1. Risk regime      — risk-on / risk-off master filter (the weather)
  2. Per-pair rate bias — fundamental lean from rate differentials
  3. Today's events    — NFP (first-Friday rule) + high-impact flag
  4. Fresh setups      — recent 20-day breakouts, each marked ALIGNED or not
  5. Today's ideas      — pairs where regime + bias + a fresh setup all line up

Self-contained (re-implements light versions of the other tabs' logic so it runs
on its own; FRED (authenticated API) + yfinance). Process aid, not trading advice.

Entry point: render()
Standalone:  streamlit run daily_cockpit_tab.py
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import requests

try:
    import streamlit as st
except Exception:
    st = None
try:
    import yfinance as yf
except Exception:
    yf = None

from src.instruments import INSTRUMENTS
from src.services.fred_data import fred_series
from src.services.parallel_fetch import run_parallel

# pair -> (base, quote, yfinance ticker) — every forex pair in the registry (18
# pairs; was 5 hardcoded majors). Metals have no base-currency rate concept,
# so they're kept separate (METALS below) and only feed the breakout scanner,
# not the rate-bias table.
PAIRS = {
    name: (*name.split("/"), INSTRUMENTS[name]["ticker"])
    for name in INSTRUMENTS.forex_pairs()
}
# name -> yfinance ticker, for the breakout scanner only (src/instruments has
# no per-metal "base currency" — XAU/XAG/XPT aren't in SHORT_RATE below).
METALS = {name: INSTRUMENTS[name]["ticker"] for name in INSTRUMENTS.commodities()}

SHORT_RATE = {  # 3-month interbank (FRED, monthly), a policy-stance proxy
    "USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N", "CAD": "IR3TIB01CAM156N", "AUD": "IR3TIB01AUM156N",
    "CHF": "IR3TIB01CHM156N", "NZD": "IR3TIB01NZM156N", "ZAR": "IR3TIB01ZAM156N",
}
# Breakout scanner universe: forex pairs + metals, uniform (base, quote,
# ticker) shape. Metals get "" for base/quote (no rate-differential concept),
# so is_aligned() naturally always reads them as not rate-aligned — they still
# surface as fresh setups, just never as an "aligned idea".
SCAN_UNIVERSE = {**PAIRS, **{name: ("", "", ticker) for name, ticker in METALS.items()}}
RISK_SIGNALS = {  # oriented so + = risk-on
    "S&P 500":   {"t": "^GSPC",  "src": "yf",    "kind": "mom",   "o": +1},
    "VIX":       {"t": "^VIX",   "src": "yf",    "kind": "level", "o": -1},
    "AUD/JPY":   {"t": "AUDJPY=X","src": "yf",   "kind": "mom",   "o": +1},
    "Copper/Gold": {"t": "_CUAU","src": "ratio", "kind": "mom",   "o": +1},
    "HY spread": {"t": "BAMLH0A0HYM2", "src": "fred", "kind": "level", "o": -1},
}
RISK_CCY = {"AUD", "NZD", "CAD"}      # risk-on currencies
HAVEN_CCY = {"USD", "JPY", "CHF"}     # risk-off havens


# ===========================================================================
# PURE / TESTABLE
# ===========================================================================

def rate_bias(base_now, quote_now, base_old, quote_old, eps=0.02) -> dict:
    """Differential, its ~6m trend, and a directional lean for the pair."""
    d_now = base_now - quote_now
    d_old = base_old - quote_old
    trend = d_now - d_old
    if d_now > 0:
        favours = "base"
        lean = "Long bias" if trend > eps else ("Weak long" if trend > -eps else "Fading")
    else:
        favours = "quote"
        lean = "Short bias" if trend < -eps else ("Weak short" if trend < eps else "Fading")
    return {"diff": d_now, "trend": trend, "favours": favours, "lean": lean}


def _zscore_latest(s, window=252):
    s = s.dropna()
    if len(s) < 40:
        return None
    w = s.iloc[-window:]
    mu, sd = w.mean(), w.std(ddof=1)
    if not sd or np.isnan(sd):
        return 0.0
    return float((s.iloc[-1] - mu) / sd)


def oriented_z(series, kind, orient, lookback=21):
    base = series.pct_change(lookback) if kind == "mom" else series
    z = _zscore_latest(base)
    return None if z is None else z * orient


def risk_composite(components: dict) -> dict:
    vals = [v for v in components.values() if v is not None]
    if not vals:
        return {"score": None, "label": "Unknown"}
    score = float(np.mean(vals))
    label = "Risk-On" if score > 0.4 else "Risk-Off" if score < -0.4 else "Neutral"
    return {"score": score, "label": label}


def risk_tilt(label: str) -> str:
    if label == "Risk-Off":
        return "Favour havens (USD/JPY/CHF); be wary of long AUD/NZD/CAD."
    if label == "Risk-On":
        return "Favour risk/commodity FX (AUD/NZD/CAD); havens softer."
    return "No strong risk bias — let rate differentials and setups lead."


def first_friday(year, month):
    d = dt.date(year, month, 1)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def event_flags(today: dt.date) -> list[str]:
    """Lightweight high-impact event flags (NFP rule + month-shape heuristics)."""
    flags = []
    ff = first_friday(today.year, today.month)
    if today == ff:
        flags.append("🔴 NFP today (first Friday) — expect ~2× range on USD pairs")
    elif 0 <= (ff - today).days <= 2:
        flags.append(f"🟠 NFP in {(ff - today).days}d ({ff.isoformat()})")
    if 10 <= today.day <= 14:
        flags.append("🟠 US CPI window (mid-month) — check exact date")
    return flags or ["🟢 No flagged top-tier US event today (still check the calendar)"]


def find_recent_setups(df, recent_days=12, n_high=20, within=3,
                       buffer_pips=2.0, pip=0.0001) -> list[dict]:
    """Compact 20-day breakout failure-test scan; returns setups triggered recently."""
    if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return []
    H, L = df["High"], df["Low"]
    ph = H.shift(1).rolling(n_high).max()
    pl = L.shift(1).rolling(n_high).min()
    N = len(df)
    out, used = [], set()

    def scan(longside):
        for h in range(n_high, N - 1):
            new_ext = (H.iloc[h] > ph.iloc[h]) if longside else (L.iloc[h] < pl.iloc[h])
            if not new_ext:
                continue
            level = float(H.iloc[h]) if longside else float(L.iloc[h])
            for Lp in (h, h + 1):
                if Lp < 1 or Lp >= N:
                    continue
                piv = (L.iloc[Lp] < L.iloc[Lp - 1]) if longside else (H.iloc[Lp] > H.iloc[Lp - 1])
                if not piv:
                    continue
                pivot = float(L.iloc[Lp]) if longside else float(H.iloc[Lp])
                rng = range(Lp + 1, min(Lp + 1 + within, N))
                cond = ((lambda t: H.iloc[t] > level) if longside
                        else (lambda t: L.iloc[t] < level))
                et = next((t for t in rng if cond(t)), None)
                if et is None or et in used:
                    continue
                stop = pivot - buffer_pips * pip if longside else pivot + buffer_pips * pip
                risk = (level - stop) if longside else (stop - level)
                if risk <= 0:
                    continue
                used.add(et)
                out.append(dict(side="long" if longside else "short",
                                entry=level, stop=stop,
                                target=level + risk if longside else level - risk,
                                entry_date=df.index[et], risk_pips=risk / pip))
                break
    scan(True)
    scan(False)
    cutoff = df.index[-1] - pd.Timedelta(days=recent_days)
    return [s for s in out if s["entry_date"] >= cutoff]


def is_aligned(side: str, lean: str, base: str, quote: str, risk_label: str) -> bool:
    """A setup is 'aligned' if its direction agrees with rate bias and isn't
    fighting the risk regime."""
    rate_ok = (side == "long" and "long" in lean.lower()) or \
              (side == "short" and "short" in lean.lower())
    # risk veto: don't long a risk currency in risk-off, etc.
    risk_ok = True
    longed_ccy = base if side == "long" else quote
    if risk_label == "Risk-Off" and longed_ccy in RISK_CCY:
        risk_ok = False
    if risk_label == "Risk-On" and longed_ccy in HAVEN_CCY and base in HAVEN_CCY:
        risk_ok = True  # neutral-ish; don't hard-veto
    return bool(rate_ok and risk_ok)


# Bars of follow-through a fresh breakout gets before it's stale, not "fresh"
# (signal-expiry framework: 1-3 bars typical, 3-5 max).
EXPIRY_BARS = 3


def setup_status(side: str, stop: float, target: float, current_price: float | None,
                 bars_since_trigger: int, expiry_bars: int = EXPIRY_BARS) -> str:
    """Live invalidation + staleness check. A stopped/target-hit read always wins
    over "expired" — it's the more informative fact about what actually happened."""
    if current_price is not None:
        if side == "short":
            if current_price >= stop:
                return "STOPPED"
            if current_price <= target:
                return "TARGET_HIT"
        else:
            if current_price <= stop:
                return "STOPPED"
            if current_price >= target:
                return "TARGET_HIT"
    if bars_since_trigger > expiry_bars:
        return "EXPIRED"
    return "LIVE"


def regime_agrees(side: str, base: str, quote: str, risk_label: str) -> bool:
    """True only when the regime actively favours this direction. A Neutral
    regime (or the veto-only check in is_aligned) is not a point of confluence."""
    if risk_label == "Risk-Off":
        return (side == "long" and base in HAVEN_CCY) or (side == "short" and quote in HAVEN_CCY) \
            or (side == "short" and base in RISK_CCY) or (side == "long" and quote in RISK_CCY)
    if risk_label == "Risk-On":
        return (side == "long" and base in RISK_CCY) or (side == "short" and quote in RISK_CCY) \
            or (side == "short" and base in HAVEN_CCY) or (side == "long" and quote in HAVEN_CCY)
    return False


def alignment_score(side: str, lean: str, base: str, quote: str, risk_label: str,
                    status: str) -> tuple[int, int]:
    """0-3 confluence score (rate bias, regime actively agreeing, setup still
    live) — replaces a binary tick that credited a Neutral regime as full
    conviction and never checked whether the setup was still tradeable."""
    rate_pt = 1 if ((side == "long" and "long" in lean.lower()) or
                    (side == "short" and "short" in lean.lower())) else 0
    regime_pt = 1 if regime_agrees(side, base, quote, risk_label) else 0
    live_pt = 1 if status == "LIVE" else 0
    return rate_pt + regime_pt + live_pt, 3


def dedupe_by_side(setups: list[dict]) -> list[dict]:
    """A pair/side re-triggering inside the lookback window (e.g. two EUR/GBP
    shorts a few days apart) is one idea re-confirming, not two — keep only the
    latest trigger per side so conviction isn't double-counted."""
    latest = {}
    for s in setups:
        key = s["side"]
        if key not in latest or s["entry_date"] > latest[key]["entry_date"]:
            latest[key] = s
    return sorted(latest.values(), key=lambda s: s["entry_date"])


# ===========================================================================
# LOADERS
# ===========================================================================

def _cache(ttl):
    return st.cache_data(ttl=ttl, show_spinner=False) if st is not None else (lambda f: f)


@_cache(ttl=3600)
def load_fred(sid):
    """A FRED series through the canonical macro spine.

    Postgres when the worker holds it, HTTP otherwise, and the spine persists
    what it fetches. The old private client returned an empty Series whenever
    FRED_API_KEY was unset, which silently blanked the whole rate-bias table;
    the spine's fallback needs no key.

    Still returns an empty Series rather than raising — this feeds a parallel
    batch (`run_parallel`), where one bad series must not sink the rest.
    """
    return fred_series(sid)


@_cache(ttl=1800)
def load_yf(ticker, period="3y", ohlc=False):
    # Routed through cached_ohlc (the shared Postgres read-through seam)
    # rather than yf.download directly — this is what makes the cockpit
    # debug-aware (source='duka'/asof time-machine, see src/debug/panel.py)
    # without changing this function's own signature or callers.
    from src.db.market_cache import cached_ohlc
    d = cached_ohlc(ticker, period=period, interval="1d")
    if d is None or d.empty:
        return pd.DataFrame() if ohlc else pd.Series(dtype=float)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.index = pd.to_datetime(d.index)
    if ohlc:
        return d
    return d["Close"].dropna()


def load_risk_signal(cfg):
    if cfg["src"] == "yf":
        return load_yf(cfg["t"])
    if cfg["src"] == "fred":
        return load_fred(cfg["t"])
    if cfg["src"] == "ratio":
        leg = {}
        run_parallel(
            load_yf, ["HG=F", "GC=F"],
            on_result=lambda ticker, series: leg.__setitem__(ticker, series),
            on_error=lambda ticker, exc: leg.__setitem__(ticker, pd.Series(dtype=float)),
        )
        cu, au = leg.get("HG=F", pd.Series(dtype=float)), leg.get("GC=F", pd.Series(dtype=float))
        df = pd.concat([cu.rename("cu"), au.rename("au")], axis=1).dropna()
        return (df["cu"] / df["au"]).dropna() if not df.empty else pd.Series(dtype=float)
    return pd.Series(dtype=float)


# ===========================================================================
# UI
# ===========================================================================

def render():
    st.header("🛫 Daily Cockpit")
    st.caption("Your pre-market routine on one screen: risk regime → rate bias → "
               "events → fresh setups, with the parts that AGREE flagged. A process "
               "aid, not trading advice.")

    # ---- 1. Risk regime ----
    # 5 independent network pulls (yfinance ×3, FRED, a 2-leg ratio) — fan them
    # out instead of paying for each round-trip in sequence.
    comps = {}

    def _fetch_component(item):
        _name, cfg = item
        return oriented_z(load_risk_signal(cfg), cfg["kind"], cfg["o"])

    run_parallel(
        _fetch_component, list(RISK_SIGNALS.items()),
        on_result=lambda item, z: comps.__setitem__(item[0], z),
        on_error=lambda item, exc: comps.__setitem__(item[0], None),
    )
    rc = risk_composite(comps)
    emoji = {"Risk-On": "🟢", "Risk-Off": "🔴", "Neutral": "🟡", "Unknown": "⚪"}[rc["label"]]
    a, b = st.columns([1, 2])
    a.metric("Risk regime", f"{emoji} {rc['label']}",
             f"score {rc['score']:+.2f}" if rc["score"] is not None else "n/a")
    b.info(risk_tilt(rc["label"]))

    # ---- 2. Per-pair rate bias ----
    st.markdown("#### Rate bias by pair")
    # Fetch each *currency's* rate series once in parallel (5 pairs share only
    # 6 distinct currencies), then build the per-pair rows from the cache.
    unique_ccy = sorted({c for base, quote, _ in PAIRS.values() for c in (base, quote)})
    rate_cache = {}
    run_parallel(
        lambda ccy: load_fred(SHORT_RATE[ccy]), unique_ccy,
        on_result=lambda ccy, series: rate_cache.__setitem__(ccy, series),
        on_error=lambda ccy, exc: rate_cache.__setitem__(ccy, pd.Series(dtype=float)),
    )

    def rate(ccy):
        return rate_cache.get(ccy, pd.Series(dtype=float))

    rows, biases = [], {}
    for name, (base, quote, _) in PAIRS.items():
        rb_s, rq_s = rate(base), rate(quote)
        if rb_s.empty or rq_s.empty or len(rb_s) < 8 or len(rq_s) < 8:
            continue
        rb = rate_bias(float(rb_s.iloc[-1]), float(rq_s.iloc[-1]),
                       float(rb_s.iloc[-7]), float(rq_s.iloc[-7]))
        biases[name] = rb
        rows.append({"Pair": name, f"{base} 3m": round(float(rb_s.iloc[-1]), 2),
                     f"{quote} 3m": round(float(rq_s.iloc[-1]), 2),
                     "Diff": round(rb["diff"], 2), "Trend": round(rb["trend"], 2),
                     "Lean": rb["lean"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Rate data unavailable (FRED unreachable).")

    # ---- 3. Events ----
    st.markdown("#### Today's events")
    for f in event_flags(dt.date.today()):
        st.markdown(f)
    st.caption("Heuristic flags only — always confirm exact release & central-bank "
               "times on a live economic calendar.")

    # ---- 4. Fresh setups + alignment ----
    st.markdown("#### Fresh 20-day breakout setups")
    # Fetch every instrument's 1y OHLC in parallel (forex pairs + metals); the
    # breakout scan itself is pure CPU on an already-fetched frame, so it
    # stays a plain serial loop.
    ohlc_cache = {}
    run_parallel(
        lambda item: load_yf(item[1][2], "1y", ohlc=True), list(SCAN_UNIVERSE.items()),
        on_result=lambda item, df: ohlc_cache.__setitem__(item[0], df),
        on_error=lambda item, exc: ohlc_cache.__setitem__(item[0], pd.DataFrame()),
    )

    setup_rows, ideas, aligned_signals = [], [], []
    for name, (base, quote, ticker) in SCAN_UNIVERSE.items():
        df = ohlc_cache.get(name, pd.DataFrame())
        current_price = float(df["Close"].iloc[-1]) if not df.empty and "Close" in df else None
        try:
            # pip size varies by instrument (0.0001 majors, 0.01 JPY crosses,
            # 0.10 XAU/XPT) — registry-sourced so risk_pips isn't off by
            # 100-1000x for the pairs beyond the original 5 majors.
            pip_size = INSTRUMENTS[name]["pip_size"]
            setups = dedupe_by_side(find_recent_setups(df, pip=pip_size))
        except Exception:
            setups = []
        for s in setups:
            bars_since = (len(df) - 1 - df.index.get_loc(s["entry_date"])) if not df.empty else 0
            status = setup_status(s["side"], s["stop"], s["target"], current_price, bars_since)
            lean = biases.get(name, {}).get("lean", "")
            score, max_score = alignment_score(s["side"], lean, base, quote, rc["label"], status)
            # "Aligned idea" gate: rate/regime confluence (is_aligned) AND still LIVE —
            # a stopped-out or expired setup is never a today's-idea, no matter the score.
            aligned = status == "LIVE" and is_aligned(s["side"], lean, base, quote, rc["label"])
            setup_rows.append({
                "Pair": name, "Side": s["side"],
                "Entry": round(s["entry"], 4), "Stop": round(s["stop"], 4),
                "Target": round(s["target"], 4), "Risk (pips)": round(s["risk_pips"], 0),
                "Triggered": s["entry_date"].date().isoformat(),
                "Rate lean": lean or "—",
                "Status": status,
                "Aligned": f"{score}/{max_score}" + (" ✅" if aligned else ""),
            })
            if aligned:
                ideas.append(f"**{name} {s['side']}** — entry {s['entry']:.4f}, "
                             f"stop {s['stop']:.4f}, target {s['target']:.4f} "
                             f"(risk {s['risk_pips']:.0f} pips); agrees with rate bias "
                             f"and {rc['label']} regime.")
                aligned_signals.append({
                    "pair": name,
                    "bias": "Long" if s["side"] == "long" else "Short",
                    "entry": s["entry"],
                    "stop_loss": s["stop"],
                    "take_profit_1": s["target"],
                    "stop_loss_pips": s.get("risk_pips"),
                    # Trigger bar of the breakout, not today — an aligned idea stays
                    # open for days and must not re-persist on every sweep.
                    "bar_time": s.get("entry_date"),
                    "conviction": f"{lean or 'rate-aligned'}, {rc['label']} regime",
                    "thesis": (f"Daily Cockpit — {name} {s['side']} fresh 20-day breakout, "
                              f"agrees with rate bias ({lean or '—'}) and "
                              f"{rc['label']} risk regime."),
                })
    # Persist today's aligned ideas (dedup'd per pair/bias/price, source-tagged,
    # silent no-op without a DB) — same pattern as twenty_day_breakout, whose
    # scanner this page's breakout logic is built on.
    if aligned_signals:
        try:
            from src.services.signal_store import persist_signals
            persist_signals("daily_cockpit", aligned_signals)
        except Exception:
            pass
    if setup_rows:
        st.dataframe(pd.DataFrame(setup_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No fresh breakout setups on the watchlist in the last ~12 days.")

    # ---- 5. Today's aligned ideas ----
    st.markdown("#### 🎯 Today's aligned ideas")
    if ideas:
        # Cross-check each idea against the canonical house view: an idea that
        # cleared the rate/regime gate but opposes the Weekly/Daily/4H read
        # gets flagged instead of silently contradicting the other pages.
        from src.core.bias import is_conflict
        from src.services.bias_service import get_house_view
        for i, sig in zip(ideas, aligned_signals):
            note = ""
            view = get_house_view(sig["pair"])
            if view is not None and is_conflict(view.direction, sig["bias"]):
                note = (f" ⚠️ **opposes the house view "
                        f"({view.direction.title()})** — see MTF Matrix.")
            st.markdown("- " + i + note)
        st.caption("These are where regime + rate bias + a fresh setup agree — the "
                   "highest-conviction confluence. Still size at fixed small risk and "
                   "check events before entering.")
    else:
        st.info("Nothing fully aligned today. Often the correct number of trades is "
                "zero — the cockpit earning its keep by keeping you out.")

    from src.debug import debug_panel
    debug_panel("Daily Cockpit", {
        "XAU/USD": ohlc_cache.get("XAU/USD"),
        "EUR/USD": ohlc_cache.get("EUR/USD"),
    })


# ===========================================================================
# PAGE ENTRY — multipage (auto-run) + standalone (streamlit run)
# ===========================================================================

def _page() -> None:
    """Wire the cockpit into the Bloomberg-terminal multipage shell: shared
    theme, the uniform sidebar logo + navigation, then the cockpit body."""
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="DCKP · Daily Cockpit",
        page_icon="🛫",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()
    with st.sidebar:
        st.markdown("### 🛫 Daily Cockpit")
        st.caption("Pre-market routine on one screen")
        st.divider()
        render_sidebar_nav()
    render()


# Guarded like twenty_day_breakout_tab.py / cot_composite_trade_signal.py so
# this module is safe to import elsewhere for its pure functions (rate_bias,
# risk_composite, event_flags, …) without re-running st.set_page_config — the
# Setup Ranker's cockpit-reconciliation panel relies on this.
if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run daily_cockpit_tab.py")
    _page()