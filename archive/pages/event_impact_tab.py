"""
event_impact_tab.py
===================
"Which economic releases actually move the market?" — a Streamlit tab inspired
by Figure 5.1 in Kathy Lien's *Day Trading and Swing Trading the Currency
Market*.

Two halves:
  1. BOOK REFERENCE — the historical ranking from the book (2004 EUR/USD ranges
     and the 1992/1997 FX-dealer importance rankings), rendered interactively.
     This half is the fixed EUR/USD benchmark from the study.
  2. LIVE RECOMPUTE — for any pair in the instrument registry, measures the
     ACTUAL average daily range (in that pair's pips) on real release days vs a
     normal-day baseline, and ranks releases by their "impact multiple". Nonfarm
     Payrolls is generated automatically from the first-Friday-of-month rule;
     any other release can be added by pasting its dates. Free data only
     (yfinance daily OHLC).

Entry point: render()
Standalone:  streamlit run event_impact_tab.py
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
    # Single source of truth for the pair universe + pip sizes. Guarded so the
    # pure helpers below can still be imported headlessly outside the project.
    from src.instruments import INSTRUMENTS
except Exception:
    INSTRUMENTS = None


# ===========================================================================
# BOOK REFERENCE DATA (Lien, Fig 5.1 — EUR/USD, 2004)
# ===========================================================================

BOOK_20MIN = {  # avg range (pips) in the 20 minutes after the release
    "Nonfarm Payrolls": 124, "FOMC Decision": 74, "Trade Balance": 64,
    "Inflation – CPI": 44, "Retail Sales": 43, "GDP": 43,
    "Current Account": 43, "Durable Goods": 39, "TICS": 33,
}
BOOK_DAILY = {  # avg total daily range (pips) on the release day
    "Nonfarm Payrolls": 193, "FOMC Decision": 140, "TICS": 132,
    "Trade Balance": 129, "Current Account": 127, "Durable Goods": 126,
    "Retail Sales": 125, "Inflation – CPI": 123, "GDP": 110,
}
DEALER_RANK = {  # FX dealer ranking of data importance
    "1997": ["Unemployment", "Interest rates", "Inflation", "Trade balance", "GDP"],
    "1992": ["Trade balance", "Interest rates", "Unemployment", "Inflation", "GDP"],
}
PIP = 1e-4  # EUR/USD pip size


# ===========================================================================
# PURE / TESTABLE HELPERS
# ===========================================================================

def first_friday_dates(start, end) -> list[pd.Timestamp]:
    """First Friday of each month in [start, end] — the NFP release rule."""
    out = []
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cur = start.replace(day=1)
    while cur <= end:
        offset = (4 - cur.weekday()) % 7        # Friday == weekday 4
        ff = (cur + pd.Timedelta(days=offset)).normalize()
        if start <= ff <= end:
            out.append(ff)
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def parse_dates(text: str) -> list[pd.Timestamp]:
    """Parse pasted dates (one per line / comma sep) into normalised Timestamps."""
    raw = [p.strip() for chunk in (text or "").splitlines()
           for p in chunk.replace(",", " ").split()]
    out = []
    for r in raw:
        try:
            out.append(pd.Timestamp(r).normalize())
        except Exception:
            continue
    return out


def daily_range_series(ohlc: pd.DataFrame, pip: float = PIP) -> pd.Series:
    """Per-day high–low range in pips, indexed by normalised date."""
    if ohlc is None or ohlc.empty:
        return pd.Series(dtype=float)
    df = ohlc.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="first")]
    return ((df["High"] - df["Low"]) / pip).dropna()


def avg_range_on_dates(ohlc: pd.DataFrame, dates, pip: float = PIP,
                       span_days: int = 1) -> dict | None:
    """
    Average range (in `pip` units) on the given release dates.
      span_days=1 -> the release day's own high–low range
      span_days=2 -> high–low across the release day + next trading day
    Snaps to the next available trading day if a date is a weekend/holiday.
    """
    if ohlc is None or ohlc.empty:
        return None
    df = ohlc.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    idx = df.index
    vals = []
    for d in pd.DatetimeIndex(pd.Index(list(dates))).normalize().unique():
        future = df.loc[idx >= d]
        if future.empty:
            continue
        win = future.iloc[:max(span_days, 1)]
        rng = (win["High"].max() - win["Low"].min()) / pip
        if np.isfinite(rng):
            vals.append(float(rng))
    if not vals:
        return None
    return {"avg_pips": float(np.mean(vals)), "n": len(vals), "values": vals}


def impact_table(ohlc: pd.DataFrame, releases: dict, span_days: int = 1,
                 pip: float = PIP) -> pd.DataFrame:
    """
    Build a ranked impact table for {release_name: [dates]} vs the all-day
    baseline. 'Impact ×' = release-day avg range / normal-day avg range.
    `pip` is the selected pair's pip size (from the instrument registry).
    """
    base = daily_range_series(ohlc, pip=pip)
    baseline = float(base.mean()) if not base.empty else np.nan
    rows = []
    for name, dates in releases.items():
        res = avg_range_on_dates(ohlc, dates, pip=pip, span_days=span_days)
        if res:
            rows.append({
                "Release": name,
                "Avg range (pips)": round(res["avg_pips"], 0),
                "Impact ×": round(res["avg_pips"] / baseline, 2) if baseline else np.nan,
                "n events": res["n"],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Avg range (pips)", ascending=False).reset_index(drop=True)
    return df, baseline


# ===========================================================================
# LOADERS
# ===========================================================================

def _cache(ttl):
    if st is not None:
        return st.cache_data(ttl=ttl, show_spinner=False)
    return lambda f: f


@_cache(ttl=3600)
def load_pair_ohlc(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Daily OHLC for any instrument ticker (e.g. EURUSD=X, GBPJPY=X, GC=F)."""
    if yf is None:
        return pd.DataFrame()
    from src.db.market_cache import cached_ohlc
    df = cached_ohlc(ticker, period=period, interval="1d", ttl=3600)
    if df is None or df.empty:
        return pd.DataFrame()
    # flatten possible multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    out = df[keep].copy()
    out.index = pd.to_datetime(out.index)
    return out.dropna()


def pair_universe() -> dict:
    """{display name: (ticker, pip_size)} for the dropdown — registry-sourced,
    with an EUR/USD-only fallback if the registry isn't importable."""
    if INSTRUMENTS is None:
        return {"EUR/USD": ("EURUSD=X", PIP)}
    return {name: (inst.ticker, inst.pip_size) for name, inst in INSTRUMENTS.items()}


# ===========================================================================
# UI
# ===========================================================================

def _ranked_df(d: dict, col: str) -> pd.DataFrame:
    return (pd.DataFrame({col: d}).sort_values(col, ascending=False))


def _section_book_reference():
    st.markdown("#### 📖 Book reference — Lien, Figure 5.1 (EUR/USD, 2004)")
    st.caption("The historical ranking that motivates the idea: not all releases "
               "move the market equally. Employment dominates; second-tier data "
               "barely registers.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**20-minute range after release (pips)**")
        st.bar_chart(_ranked_df(BOOK_20MIN, "pips"), height=300)
    with c2:
        st.markdown("**Total daily range on release day (pips)**")
        st.bar_chart(_ranked_df(BOOK_DAILY, "pips"), height=300)

    st.caption("Notice the 20-minute column is far more *concentrated* — NFP's "
               "immediate spike (124p) towers over everything, while daily ranges "
               "are flatter. The instant reaction is where the release's surprise "
               "lives; the rest of the day blends in everything else.")

    st.markdown("**FX dealer ranking of data importance (how it shifts over time)**")
    rank_df = pd.DataFrame({"As of 1997": DEALER_RANK["1997"],
                            "As of 1992": DEALER_RANK["1992"]},
                           index=[1, 2, 3, 4, 5])
    rank_df.index.name = "Rank"
    st.table(rank_df)
    st.caption("The ranking is not static — trade balance topped 1992, "
               "unemployment topped 1997. By 2014 Lien notes employment, monetary "
               "policy and inflation were the big three. The lesson: *which* data "
               "matters rotates with what the economy is worried about, so re-rank "
               "with live data rather than trusting a fixed list — which is what "
               "the next section does.")


def _section_live_recompute():
    st.markdown("#### ⚡ Live recompute — real ranges on release days")
    st.caption("Pick any pair and measure its actual average daily high–low range "
               "(in that pair's pips) on real release days vs a normal-day "
               "baseline, ranked by impact. NFP is generated from the first-Friday "
               "rule; add any other release by pasting its dates.")

    # ── Pair selector (registry-sourced) + refresh ───────────────────────────
    universe = pair_universe()
    names = list(universe.keys())
    default_ix = names.index("EUR/USD") if "EUR/USD" in names else 0
    c0, c1, c2 = st.columns([1.2, 1, 1])
    pair_name = c0.selectbox("Pair", names, index=default_ix, key="ei_pair")
    ticker, pip = universe[pair_name]
    years = c1.slider("Lookback (years)", 1, 5, 3)
    span = c2.radio("Window", ["Release day", "Release day + next"],
                    horizontal=True)
    span_days = 1 if span == "Release day" else 2

    if st.button("🔄 Refresh data", help="Clear cached prices and re-pull from yfinance"):
        st.cache_data.clear()
        st.rerun()

    ohlc = load_pair_ohlc(ticker)
    if ohlc.empty:
        st.warning(f"Could not load {pair_name} ({ticker}) price data "
                   f"(yfinance unreachable).")
        return

    end = ohlc.index.max().normalize()
    start = end - pd.Timedelta(days=365 * years)
    ohlc_win = ohlc.loc[ohlc.index >= start]

    releases = {"Nonfarm Payrolls (1st Friday)": first_friday_dates(start, end)}

    if "USD" not in pair_name:
        st.caption(f"ℹ️ {pair_name} has no USD leg — Nonfarm Payrolls is a *USD* "
                   f"release, so its effect here is indirect. Paste this pair's own "
                   f"currency releases below for a cleaner read.")

    with st.expander("➕ Add another release (paste its dates)"):
        name = st.text_input("Release name", value="CPI")
        pasted = st.text_area("Dates (YYYY-MM-DD, one per line or comma-separated)",
                              height=90, placeholder="2025-01-15\n2025-02-12\n...")
        if pasted.strip():
            ds = parse_dates(pasted)
            if ds:
                releases[name or "Custom"] = ds
                st.caption(f"Parsed {len(ds)} dates for {name or 'Custom'}.")

    table, baseline = impact_table(ohlc_win, releases, span_days=span_days, pip=pip)
    if table.empty:
        st.warning("No release days matched the price history.")
        return

    last_px = float(ohlc["Close"].iloc[-1])
    st.metric(f"{pair_name} · normal-day baseline range", f"{baseline:.0f} pips",
              delta=f"last {last_px:.5f}" if abs(last_px) < 100 else f"last {last_px:.2f}",
              delta_color="off",
              help=f"Average {pair_name} daily high–low range across all days in window.")
    st.dataframe(table, width="stretch", hide_index=True)
    st.bar_chart(table.set_index("Release")["Avg range (pips)"], height=300)

    top = table.iloc[0]
    st.markdown(
        f"Over the last {years}y, **{top['Release']}** days saw {pair_name} swing "
        f"~**{top['Avg range (pips)']:.0f} pips** — **{top['Impact ×']:.2f}×** a "
        f"normal day. That multiple is the practical takeaway: it's how much extra "
        f"range to expect, size for, and trade around on that release.")
    st.caption("Daily high–low only (free data can't reliably reconstruct the "
               "20-minute post-release spike historically). It ranks *which* "
               "releases matter now for this pair; for the instant reaction, watch "
               "them live.")


def render():
    """Call this inside your dashboard tab."""
    st.header("📅 Event Impact — which releases move the market")
    st.caption("After Lien's Figure 5.1: rank economic releases by how much they "
               "actually move price — the historical EUR/USD benchmark below, then "
               "recomputed live for any pair you choose.")
    _section_book_reference()
    st.divider()
    _section_live_recompute()


if __name__ == "__main__":
    if st is None:
        raise SystemExit("Run with: streamlit run event_impact_tab.py")

    # Page bootstrap — kept lazy so the pure helpers above can still be imported
    # in a headless (no-Streamlit) context. Mirrors the legacy-page contract:
    # set_page_config → BloombergTheme.apply() → sidebar (header → divider → nav).
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="Event Impact · Trading System",
        page_icon="📅",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()

    with st.sidebar:
        st.markdown("### 📅 Event Impact")
        st.caption("Which releases actually move your pair")
        st.divider()
        render_sidebar_nav()

    render()