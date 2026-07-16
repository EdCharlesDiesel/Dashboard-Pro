"""
liquidity_hunt_tab.py — Liquidity sweep detection + event-study validation.

Same spec as the Pine/MQL5 builds:
  pool    : pivot(N/N) merged within eq_tol x ATR14, expires after max_age bars
  sweep   : pierce + reject close, wick_ratio >= wick_min, volume >= vol_mult x SMA20
  confirm : close beyond sweep-bar opposite extreme within disp_bars
  fail    : close beyond level +/- fail_k x ATR

Extra vs the chart indicators: journals every event to PostgreSQL
(liquidity_sweeps) and runs a forward-return event study against a
bootstrap random-entry baseline. That's the part that tells you whether
the pattern actually pays on XAUUSD/XAGUSD or just looks good in hindsight.

Usage in app.py:
    import liquidity_hunt_tab
    with tab_lh:
        liquidity_hunt_tab.render(engine)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sqlalchemy import text

from src.pages_lib.navigation import render_sidebar_nav
from src.ui.theme import BloombergTheme

HORIZONS = [5, 15, 60]  # forward horizons, in bars
SAST = "Africa/Johannesburg"

SYMBOL_MAP = {"XAUUSD": "GC=F", "XAGUSD": "SI=F", "EURUSD": "EURUSD=X",
              "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X"}

# ────────────────────────────── data ──────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(SAST)
    return df.dropna(subset=["open", "high", "low", "close"])


def add_indicators(df: pd.DataFrame, pivot_n: int) -> pd.DataFrame:
    df = df.copy()
    pc = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()))
    df["atr"] = tr.rolling(14).mean()
    df["vol_sma"] = df["volume"].rolling(20).mean()
    w = 2 * pivot_n + 1
    df["piv_hi"] = df["high"] == df["high"].rolling(w, center=True).max()
    df["piv_lo"] = df["low"] == df["low"].rolling(w, center=True).min()
    return df


def sast_session(ts: pd.Timestamp) -> str:
    h = ts.hour + ts.minute / 60
    if 1 <= h < 9:      return "Asia"
    if 9 <= h < 14.5:   return "London"
    if 14.5 <= h < 18:  return "LDN/NY overlap"
    if 18 <= h < 23:    return "NY"
    return "Off-hours"

# ─────────────────────────── detection ────────────────────────────

@dataclass
class Pool:
    level: float
    touches: int
    born: int
    is_high: bool
    state: int = 0            # 0 active, 1 pending, 2 confirmed, 3 failed
    sweep_i: int = -1
    sweep_extreme: float = np.nan
    meta: dict = field(default_factory=dict)


def detect(df: pd.DataFrame, pivot_n=5, eq_tol=0.10, max_age=300,
           wick_min=0.60, use_vol=True, vol_mult=1.2,
           disp_bars=3, fail_k=0.5):
    """Single forward pass. Pivots only become known pivot_n bars late
    (rolling center window), which reproduces live confirmation lag —
    no lookahead in pool creation."""
    pools: list[Pool] = []
    events, pool_lines = [], []
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    v, atr, vsma = df["volume"].to_numpy(), df["atr"].to_numpy(), df["vol_sma"].to_numpy()
    ph, pl = df["piv_hi"].to_numpy(), df["piv_lo"].to_numpy()
    idx = df.index

    def add_pivot(px, born_i, is_high, a):
        for p in pools:
            if p.state == 0 and p.is_high == is_high and abs(p.level - px) <= eq_tol * a:
                p.level = (p.level * p.touches + px) / (p.touches + 1)
                p.touches += 1
                return
        pools.append(Pool(px, 1, born_i, is_high))

    for i in range(len(df)):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        pc = i - pivot_n                      # pivot confirmed with lag
        if pc >= 0:
            if ph[pc]:
                add_pivot(h[pc], pc, True, a)
            if pl[pc]:
                add_pivot(l[pc], pc, False, a)

        rng = max(h[i] - l[i], 1e-12)
        vol_ok = (not use_vol) or (vsma[i] > 0 and v[i] > vsma[i] * vol_mult)

        for p in pools[:]:
            if p.state == 0 and i - p.born > max_age:
                pool_lines.append((idx[p.born], idx[i], p.level, p.is_high, p.touches))
                pools.remove(p)
                continue

            if p.state == 0:
                swept = False
                if p.is_high and h[i] > p.level and c[i] < p.level:
                    wick = (h[i] - max(o[i], c[i])) / rng
                    swept = wick >= wick_min and vol_ok
                elif (not p.is_high) and l[i] < p.level and c[i] > p.level:
                    wick = (min(o[i], c[i]) - l[i]) / rng
                    swept = wick >= wick_min and vol_ok
                if swept:
                    p.state, p.sweep_i = 1, i
                    p.sweep_extreme = l[i] if p.is_high else h[i]
                    p.meta = {"wick_ratio": round(wick, 3),
                              "vol_ratio": round(v[i] / vsma[i], 2) if vsma[i] > 0 else np.nan}

            elif p.state == 1:
                done, outcome = False, None
                if p.is_high:
                    if c[i] < p.sweep_extreme:
                        done, outcome = True, "confirmed"
                    elif c[i] > p.level + fail_k * a or i - p.sweep_i > disp_bars:
                        done, outcome = True, "failed"
                else:
                    if c[i] > p.sweep_extreme:
                        done, outcome = True, "confirmed"
                    elif c[i] < p.level - fail_k * a or i - p.sweep_i > disp_bars:
                        done, outcome = True, "failed"
                if done:
                    events.append({
                        "ts": idx[p.sweep_i], "resolve_ts": idx[i],
                        "side": "buyside" if p.is_high else "sellside",
                        "level": round(p.level, 5), "touches": p.touches,
                        "session": sast_session(idx[p.sweep_i]),
                        "outcome": outcome, "sweep_i": p.sweep_i, **p.meta})
                    pool_lines.append((idx[p.born], idx[i], p.level, p.is_high, p.touches))
                    pools.remove(p)

    ev = pd.DataFrame(events)
    if not ev.empty:
        # signed forward returns in the reversal direction, from sweep-bar close
        sign = np.where(ev["side"] == "buyside", -1.0, 1.0)
        for hz in HORIZONS:
            exit_i = np.minimum(ev["sweep_i"] + hz, len(c) - 1)
            ev[f"fwd_r{hz}"] = sign * (c[exit_i.to_numpy()] / c[ev["sweep_i"].to_numpy()] - 1) * 1e4  # bps
    return ev, pool_lines


def bootstrap_baseline(df: pd.DataFrame, n_events: int, hz: int,
                       n_boot=500, seed=42) -> np.ndarray:
    """Distribution of mean |signed| returns from random entries with
    random direction — the null against which sweep edges are judged."""
    rng = np.random.default_rng(seed)
    c = df["close"].to_numpy()
    valid = len(c) - hz - 1
    means = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(20, valid, size=n_events)
        s = rng.choice([-1.0, 1.0], size=n_events)
        means[b] = np.mean(s * (c[i + hz] / c[i] - 1) * 1e4)
    return means

# ─────────────────────────── persistence ──────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS liquidity_sweeps (
    symbol     text NOT NULL,
    interval   text NOT NULL,
    ts         timestamptz NOT NULL,
    side       text NOT NULL,
    level      double precision,
    touches    integer,
    wick_ratio double precision,
    vol_ratio  double precision,
    session    text,
    outcome    text,
    fwd_r5     double precision,
    fwd_r15    double precision,
    fwd_r60    double precision,
    created_at timestamptz DEFAULT now(),
    PRIMARY KEY (symbol, interval, ts, side)
);
"""

UPSERT = """
INSERT INTO liquidity_sweeps
    (symbol, interval, ts, side, level, touches, wick_ratio, vol_ratio,
     session, outcome, fwd_r5, fwd_r15, fwd_r60)
VALUES
    (:symbol, :interval, :ts, :side, :level, :touches, :wick_ratio, :vol_ratio,
     :session, :outcome, :fwd_r5, :fwd_r15, :fwd_r60)
ON CONFLICT (symbol, interval, ts, side) DO UPDATE SET
    outcome = EXCLUDED.outcome,
    fwd_r5 = EXCLUDED.fwd_r5, fwd_r15 = EXCLUDED.fwd_r15, fwd_r60 = EXCLUDED.fwd_r60;
"""


def journal(conn, symbol: str, interval: str, ev: pd.DataFrame) -> int:
    rows = [{"symbol": symbol, "interval": interval,
             "ts": r.ts.to_pydatetime(), "side": r.side, "level": r.level,
             "touches": int(r.touches),
             "wick_ratio": float(r.wick_ratio), "vol_ratio": None if pd.isna(r.vol_ratio) else float(r.vol_ratio),
             "session": r.session, "outcome": r.outcome,
             "fwd_r5": float(r.fwd_r5), "fwd_r15": float(r.fwd_r15), "fwd_r60": float(r.fwd_r60)}
            for r in ev.itertuples()]
    with conn.begin() as tx:
        tx.execute(text(DDL))
        tx.execute(text(UPSERT), rows)
    return len(rows)

# ────────────────────────────── UI ────────────────────────────────

def render(get_conn):
    """get_conn: zero-arg callable returning an open SQLAlchemy connection,
    invoked lazily (only when the user actually asks to journal events) so a
    down/unconfigured DB never blocks the chart and event study above it."""
    st.title("Liquidity Hunt Lab")
    st.caption("Pool detection + sweep/confirm/fail event study vs a random-entry baseline")

    c1, c2, c3, c4 = st.columns(4)
    sym = c1.selectbox("Symbol", list(SYMBOL_MAP), index=0)
    interval = c2.selectbox("Interval", ["15m", "30m", "1h"], index=0)
    period = c3.selectbox("Lookback", ["5d", "1mo", "3mo"], index=1)
    pivot_n = c4.number_input("Pivot length", 2, 20, 5)

    with st.expander("Detection parameters"):
        d1, d2, d3, d4 = st.columns(4)
        eq_tol = d1.number_input("Eq tolerance (xATR)", 0.02, 0.5, 0.10, 0.02)
        wick_min = d2.number_input("Min wick ratio", 0.3, 0.9, 0.60, 0.05)
        vol_mult = d3.number_input("Volume multiple", 0.0, 3.0, 1.2, 0.1)
        disp_bars = d4.number_input("Displacement window", 1, 10, 3)

    with st.sidebar:
        st.subheader("Liquidity Hunt Lab")
        st.caption("Sweep/confirm/fail detection on pivot pools")
        st.divider()
        render_sidebar_nav()

    df = load_ohlcv(SYMBOL_MAP[sym], period, interval)
    if df.empty:
        st.warning("No data returned — futures proxies (GC=F/SI=F) have limited intraday history on yfinance.")
        return
    df = add_indicators(df, pivot_n)

    ev, pool_lines = detect(df, pivot_n=pivot_n, eq_tol=eq_tol,
                            wick_min=wick_min, use_vol=vol_mult > 0,
                            vol_mult=vol_mult, disp_bars=disp_bars)

    # ── chart ──
    fig = go.Figure(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                   low=df["low"], close=df["close"], name=sym))
    for t0, t1, lvl, is_high, touches in pool_lines[-80:]:
        fig.add_shape(type="line", x0=t0, x1=t1, y0=lvl, y1=lvl,
                      line=dict(color="crimson" if is_high else "teal",
                                width=min(touches, 3), dash="dot"))
    if not ev.empty:
        for outcome, symb, col in [("confirmed", "star", "gold"), ("failed", "x", "gray")]:
            sub = ev[ev["outcome"] == outcome]
            if not sub.empty:
                y = np.where(sub["side"] == "buyside",
                             df["high"].reindex(sub["ts"]).to_numpy() * 1.001,
                             df["low"].reindex(sub["ts"]).to_numpy() * 0.999)
                fig.add_trace(go.Scatter(x=sub["ts"], y=y, mode="markers",
                                         name=f"sweep {outcome}",
                                         marker=dict(symbol=symb, size=10, color=col)))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if ev.empty:
        st.info("No sweep events in this window with current parameters.")
        return

    # ── event study ──
    st.markdown(f"**{len(ev)} sweep events** — "
                f"{(ev['outcome'] == 'confirmed').mean():.0%} confirmed")

    stats = (ev.groupby(["outcome", "side"])
               .agg(n=("ts", "size"),
                    **{f"mean_r{h}": (f"fwd_r{h}", "mean") for h in HORIZONS},
                    **{f"win_r{h}": (f"fwd_r{h}", lambda s: (s > 0).mean()) for h in HORIZONS})
               .round(1))
    st.dataframe(stats, use_container_width=True)

    by_sess = (ev[ev["outcome"] == "confirmed"]
               .groupby("session")[[f"fwd_r{h}" for h in HORIZONS]]
               .agg(["count", "mean"]).round(1))
    st.markdown("**Confirmed sweeps by session (SAST)** — fwd returns in bps, reversal direction")
    st.dataframe(by_sess, use_container_width=True)

    # baseline test on confirmed events, 15-bar horizon
    conf = ev[ev["outcome"] == "confirmed"]
    if len(conf) >= 10:
        base = bootstrap_baseline(df, len(conf), 15)
        actual = conf["fwd_r15"].mean()
        pctl = (base < actual).mean() * 100
        st.metric("Confirmed mean fwd_r15 vs random baseline",
                  f"{actual:+.1f} bps", f"{pctl:.0f}th percentile of null")
        if pctl < 90:
            st.caption("Below the 90th percentile of the random-entry null — "
                       "no demonstrable edge at this horizon yet. More data or tighter filters.")
    else:
        st.caption("Fewer than 10 confirmed events — baseline test skipped; extend lookback.")

    with st.expander("Event log"):
        st.dataframe(ev.drop(columns=["sweep_i"]).sort_values("ts", ascending=False),
                     use_container_width=True)

    if st.button("Journal events to PostgreSQL"):
        try:
            with get_conn() as conn:
                n = journal(conn, sym, interval, ev)
            st.success(f"Upserted {n} events into liquidity_sweeps.")
        except Exception as e:
            st.error(f"Persistence failed: {e}")


# ===========================================================================
# PAGE ENTRY — multipage (auto-run) + standalone (streamlit run)
# ===========================================================================
# Guarded behind __main__ (Streamlit runs each page script as __main__) so the
# pure functions above (detect, add_indicators, bootstrap_baseline) stay
# importable headlessly — same contract as pages/disconnect_monitor_tab.py and
# pages/surprise_tab.py.

def _get_conn():
    """Opens a fresh SQLAlchemy connection against the app's resolved DB
    target (session-state db_* overrides, falling back to secrets.toml
    [database] — same resolution `auto_connect()` uses). Called lazily from
    the "Journal events" button so a down/unconfigured DB surfaces as an
    st.error there instead of crashing the whole page on load."""
    from sqlalchemy import create_engine
    from src.db.connection import current_db_config
    cfg = current_db_config()
    if not cfg.password:
        raise RuntimeError("No DB credentials configured (secrets.toml [database]).")
    url = (f"postgresql+psycopg2://{cfg.user}:{cfg.password}"
          f"@{cfg.host}:{cfg.port}/{cfg.dbname}")
    return create_engine(url, pool_pre_ping=True).connect()


if __name__ == "__main__":
    st.set_page_config(
        page_title="LQHT · Liquidity Hunt Lab",
        page_icon="🎣",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()
    from src.db.connection import auto_connect
    auto_connect()
    render(_get_conn)