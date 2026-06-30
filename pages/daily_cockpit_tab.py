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
on its own; FRED no-key + yfinance). Process aid, not trading advice.

Entry point: render()
Standalone:  streamlit run daily_cockpit_tab.py
"""
from __future__ import annotations

import io
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

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
HEADERS = {"User-Agent": "Mozilla/5.0 (cockpit)"}

# pair -> (base, quote, yfinance ticker)
PAIRS = {
    "EUR/USD": ("EUR", "USD", "EURUSD=X"),
    "GBP/USD": ("GBP", "USD", "GBPUSD=X"),
    "USD/JPY": ("USD", "JPY", "JPY=X"),
    "AUD/USD": ("AUD", "USD", "AUDUSD=X"),
    "USD/CAD": ("USD", "CAD", "CAD=X"),
}
SHORT_RATE = {  # 3-month interbank (FRED, monthly), a policy-stance proxy
    "USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N", "CAD": "IR3TIB01CAM156N", "AUD": "IR3TIB01AUM156N",
}
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


# ===========================================================================
# LOADERS
# ===========================================================================

def _cache(ttl):
    return st.cache_data(ttl=ttl, show_spinner=False) if st is not None else (lambda f: f)


@_cache(ttl=3600)
def load_fred(sid):
    r = requests.get(FRED_CSV.format(sid=sid), headers=HEADERS, timeout=20)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    dc, vc = d.columns[0], d.columns[1]
    d[dc] = pd.to_datetime(d[dc], errors="coerce")
    d[vc] = pd.to_numeric(d[vc], errors="coerce")
    return d.dropna(subset=[dc]).set_index(dc)[vc].dropna()


@_cache(ttl=1800)
def load_yf(ticker, period="3y", ohlc=False):
    if yf is None:
        return pd.DataFrame() if ohlc else pd.Series(dtype=float)
    d = yf.download(ticker, period=period, interval="1d", progress=False)
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
        cu, au = load_yf("HG=F"), load_yf("GC=F")
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
    comps = {}
    for name, cfg in RISK_SIGNALS.items():
        try:
            comps[name] = oriented_z(load_risk_signal(cfg), cfg["kind"], cfg["o"])
        except Exception:
            comps[name] = None
    rc = risk_composite(comps)
    emoji = {"Risk-On": "🟢", "Risk-Off": "🔴", "Neutral": "🟡", "Unknown": "⚪"}[rc["label"]]
    a, b = st.columns([1, 2])
    a.metric("Risk regime", f"{emoji} {rc['label']}",
             f"score {rc['score']:+.2f}" if rc["score"] is not None else "n/a")
    b.info(risk_tilt(rc["label"]))

    # ---- 2. Per-pair rate bias ----
    st.markdown("#### Rate bias by pair")
    rate_cache = {}

    def rate(ccy):
        if ccy not in rate_cache:
            try:
                rate_cache[ccy] = load_fred(SHORT_RATE[ccy])
            except Exception:
                rate_cache[ccy] = pd.Series(dtype=float)
        return rate_cache[ccy]

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
    setup_rows, ideas = [], []
    for name, (base, quote, ticker) in PAIRS.items():
        try:
            ohlc = load_yf(ticker, "1y", ohlc=True)
            setups = find_recent_setups(ohlc)
        except Exception:
            setups = []
        for s in setups:
            lean = biases.get(name, {}).get("lean", "")
            aligned = is_aligned(s["side"], lean, base, quote, rc["label"])
            setup_rows.append({
                "Pair": name, "Side": s["side"],
                "Entry": round(s["entry"], 4), "Stop": round(s["stop"], 4),
                "Target": round(s["target"], 4), "Risk (pips)": round(s["risk_pips"], 0),
                "Triggered": s["entry_date"].date().isoformat(),
                "Rate lean": lean or "—",
                "Aligned": "✅" if aligned else "—",
            })
            if aligned:
                ideas.append(f"**{name} {s['side']}** — entry {s['entry']:.4f}, "
                             f"stop {s['stop']:.4f}, target {s['target']:.4f} "
                             f"(risk {s['risk_pips']:.0f} pips); agrees with rate bias "
                             f"and {rc['label']} regime.")
    if setup_rows:
        st.dataframe(pd.DataFrame(setup_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No fresh breakout setups on the watchlist in the last ~12 days.")

    # ---- 5. Today's aligned ideas ----
    st.markdown("#### 🎯 Today's aligned ideas")
    if ideas:
        for i in ideas:
            st.markdown("- " + i)
        st.caption("These are where regime + rate bias + a fresh setup agree — the "
                   "highest-conviction confluence. Still size at fixed small risk and "
                   "check events before entering.")
    else:
        st.info("Nothing fully aligned today. Often the correct number of trades is "
                "zero — the cockpit earning its keep by keeping you out.")


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


# Streamlit executes a page module top-to-bottom on every run (the same way the
# other legacy pages self-initialise), so call the entry unconditionally when a
# Streamlit runtime is present.
if st is not None:
    _page()