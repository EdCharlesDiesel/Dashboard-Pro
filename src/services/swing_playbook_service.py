"""
src/services/swing_playbook_service.py
=======================================
One row per instrument: everything needed to approve or reject a swing trade
before the week starts — price, COT crowdedness, gold's real-yield disconnect
read, nearby swing-pivot supply/demand levels, and vol-scaled position sizing.

Every fetcher degrades gracefully: if a source isn't available for an
instrument (e.g. COT only covers majors + gold) it returns ``None`` and the
playbook shows N/A instead of crashing. No SQLAlchemy and no direct DB
connection here — persistence goes through ``TradeRepository`` (psycopg2) via
the pooled resolver, same contract as ``src/services/signal_store.py``: a
missing/unconfigured DB is a silent no-op, never an exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.instruments.registry import INSTRUMENTS

# Registry pair -> COT instrument key (src.services.cot_fetcher.INSTRUMENTS),
# inverted from pages/cot_composite_trade_signal.py's CURRENCY_TO_PAIR. Only
# majors + gold have a CFTC series; everything else stays N/A on that column.
COT_INSTRUMENT_FOR_PAIR: Dict[str, str] = {
    "EUR/USD": "EUR", "GBP/USD": "GBP", "AUD/USD": "AUD", "NZD/USD": "NZD",
    "USD/JPY": "JPY", "USD/CHF": "CHF", "USD/CAD": "CAD", "XAU/USD": "GOLD",
}

# Gold is the only instrument disconnect_monitor_tab.py maps to a tradable
# registry pair (ASSET_TO_REGISTRY_PAIR = {"GC=F": "XAU/USD"}).
DISCONNECT_TICKER_FOR_PAIR: Dict[str, str] = {"XAU/USD": "GC=F"}


def _safe(fn):
    def wrapped(*a, **k):
        try:
            return fn(*a, **k)
        except Exception:
            return None
    return wrapped


# ---------------------------------------------------------------------------
# Price + OHLC history (one fetch reused for price, zones, and sigma)
# ---------------------------------------------------------------------------

@_safe
def _daily_frame(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    from src.db.market_cache import cached_ohlc
    df = cached_ohlc(ticker, period=period, interval="1d", ttl=300)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


@_safe
def last_price(instrument: str) -> Optional[float]:
    inst = INSTRUMENTS.get(instrument)
    if inst is None:
        return None
    df = _daily_frame(inst.ticker, period="5d")
    if df is None or df.empty:
        return None
    return float(df["Close"].iloc[-1])


# ---------------------------------------------------------------------------
# COT crowdedness — reuses the same pipeline cot_composite_trade_signal.py
# and instrument_predictor.py already use.
# ---------------------------------------------------------------------------

@_safe
def cot_read(instrument: str, years_back: int = 4, lookback_weeks: int = 8) -> Optional[Dict[str, float]]:
    """3-year percentile + z-score of managed-money net positioning. >90 or
    <10 = crowded. ``None`` when the instrument has no CFTC series."""
    cot_key = COT_INSTRUMENT_FOR_PAIR.get(instrument)
    if cot_key is None:
        return None

    from src.db.market_cache import cached_ohlc
    from src.services.cot_fetcher import get_instrument_series
    from pages.cot_signals import compute_cot_signal

    series = get_instrument_series(cot_key, years_back=years_back)
    if series is None or series.empty:
        return None

    price_ticker = {"GOLD": "GC=F"}.get(cot_key, INSTRUMENTS[instrument]["ticker"])
    from datetime import timedelta
    raw = cached_ohlc(price_ticker, start=datetime.now() - timedelta(days=years_back * 365),
                      interval="1d", ttl=600)
    if raw is None or raw.empty or "Close" not in raw.columns:
        return None
    close = raw["Close"].dropna()
    price_df = pd.DataFrame({"date": close.index, "price": close.values})

    merged = pd.merge_asof(
        series[["date", "net_spec"]].sort_values("date"),
        price_df.sort_values("date"),
        on="date", direction="backward",
    ).dropna().rename(columns={"net_spec": "net_position"})
    if len(merged) < 20:
        return None

    sig = compute_cot_signal(merged, lookback_weeks=lookback_weeks)
    return {"percentile": float(sig.percentile_3y), "zscore": float(sig.zscore)}


# ---------------------------------------------------------------------------
# Gold real-yield disconnect z-score — reuses disconnect_monitor_tab.py's
# pure math functions (not its Streamlit UI).
# ---------------------------------------------------------------------------

@_safe
def disconnect_z(instrument: str, years: int = 12) -> Optional[float]:
    ticker = DISCONNECT_TICKER_FOR_PAIR.get(instrument)
    if ticker is None:
        return None

    from pages.disconnect_monitor_tab import real_yield_series, fetch_yf, disconnect_frame

    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    driver = real_yield_series(start, "DFII5")
    asset = fetch_yf(ticker, start)
    if driver.empty or asset.empty:
        return None
    df = disconnect_frame(driver, asset)
    if df.empty:
        return None
    return float(df["z"].iloc[-1])


# ---------------------------------------------------------------------------
# Supply/demand zones — nearest untested swing pivot above/below price.
# ---------------------------------------------------------------------------

def _swing_pivots(df: pd.DataFrame, lookback: int = 2) -> Dict[str, List[float]]:
    """2-bar lookback swing highs/lows, same technique pages/weekly-swing.py
    uses for its weekly pivot classification."""
    highs, lows = df["High"].values, df["Low"].values
    n = len(df)
    sh, sl = [], []
    for i in range(lookback, n - lookback):
        if all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and \
                all(highs[i] >= highs[i + j] for j in range(1, lookback + 1)):
            sh.append(float(highs[i]))
        if all(lows[i] <= lows[i - j] for j in range(1, lookback + 1)) and \
                all(lows[i] <= lows[i + j] for j in range(1, lookback + 1)):
            sl.append(float(lows[i]))
    return {"highs": sh, "lows": sl}


@_safe
def nearest_zones(instrument: str, price: float) -> Optional[Dict[str, Optional[float]]]:
    inst = INSTRUMENTS.get(instrument)
    if inst is None or not price:
        return None
    df = _daily_frame(inst.ticker, period="6mo")
    if df is None or len(df) < 20:
        return None
    piv = _swing_pivots(df)
    supply = [h for h in piv["highs"] if h > price]
    demand = [lo for lo in piv["lows"] if lo < price]
    return {
        "supply": min(supply) if supply else None,
        "demand": max(demand) if demand else None,
    }


# ---------------------------------------------------------------------------
# Volatility — GARCH(1,1) when available, EWMA fallback otherwise.
# ---------------------------------------------------------------------------

@_safe
def daily_sigma(instrument: str) -> Optional[float]:
    """Daily volatility (decimal, e.g. 0.008 = 0.8%) for sizing the stop."""
    inst = INSTRUMENTS.get(instrument)
    if inst is None:
        return None
    df = _daily_frame(inst.ticker, period="1y")
    if df is None or len(df) < 40:
        return None
    rets = df["Close"].pct_change().dropna()

    try:
        from src.core.quant_models import fit_garch
        fit = fit_garch(rets)
        return float(fit["current_vol_daily_pct"]) / 100.0
    except Exception:
        pass

    # EWMA (RiskMetrics) fallback when the 'arch' package or a stable fit
    # isn't available.
    lam, s2 = 0.94, float(rets.var())
    for r in rets.values[-60:]:
        s2 = lam * s2 + (1 - lam) * r ** 2
    return float(np.sqrt(s2))


# ---------------------------------------------------------------------------
# Sizing — the piece that decides whether a bad print is annoying or painful
# ---------------------------------------------------------------------------

def swing_size(account: float, risk_pct: float, price: float,
               daily_sigma_: float, k: float = 2.0,
               holding_days: int = 8) -> Dict[str, float]:
    """Stop = k * sigma * sqrt(holding_days) away (vol scales with sqrt(T)).
    Units = risk amount / stop distance. Volatile regime -> smaller size,
    automatically."""
    stop_dist = k * daily_sigma_ * np.sqrt(holding_days) * price
    risk_amount = account * risk_pct
    units = risk_amount / stop_dist if stop_dist > 0 else 0.0
    return {
        "stop_distance": stop_dist,
        "risk_amount": risk_amount,
        "units": units,
        "stop_pct": stop_dist / price * 100 if price else 0.0,
    }


# ---------------------------------------------------------------------------
# Weekly theses — persisted via TradeRepository (psycopg2), resolved the same
# way src/services/signal_store.py resolves its DB target. No-ops gracefully
# when the DB isn't configured.
# ---------------------------------------------------------------------------

def week_start(now: Optional[datetime] = None):
    now = now or datetime.now(timezone.utc)
    return (now - pd.Timedelta(days=now.weekday())).date()


def _repo():
    from src.db.cache import pooled_repository
    from src.db.market_cache import _resolve_cfg
    cfg = _resolve_cfg()
    if cfg is None:
        return None
    return pooled_repository(cfg)


def save_thesis(instrument: str, bias: str, invalidation: str) -> bool:
    """Upsert this week's thesis for ``instrument``. Returns whether it was
    saved (``False`` when the DB isn't configured)."""
    repo = _repo()
    if repo is None:
        return False
    repo.save_swing_thesis(instrument, week_start(), bias, invalidation)
    return True


def load_thesis(instrument: str) -> Optional[Dict[str, Any]]:
    repo = _repo()
    if repo is None:
        return None
    return repo.load_swing_thesis(instrument, week_start())


# ---------------------------------------------------------------------------
# The playbook
# ---------------------------------------------------------------------------

@dataclass
class PlaybookRow:
    instrument: str
    price: Optional[float] = None
    bias: str = "—"
    invalidation: str = "—"
    cot_pctile: Optional[float] = None
    disconnect_z: Optional[float] = None
    gate: str = "?"
    supply: Optional[float] = None
    demand: Optional[float] = None
    daily_sigma: Optional[float] = None
    stop_pct: Optional[float] = None
    units: Optional[float] = None
    flags: List[str] = field(default_factory=list)


def build_playbook(instruments: List[str], account: float,
                   risk_pct: float, event_gate_fn=None) -> List[PlaybookRow]:
    rows = []
    for inst in instruments:
        r = PlaybookRow(instrument=inst)
        r.price = last_price(inst)

        th = load_thesis(inst)
        if th:
            r.bias, r.invalidation = th["bias"], th["invalidation"]
        else:
            r.flags.append("NO THESIS — do not trade without one")

        cot = cot_read(inst)
        if cot is not None:
            r.cot_pctile = cot["percentile"]
            if r.cot_pctile > 90 or r.cot_pctile < 10:
                r.flags.append(f"COT crowded ({r.cot_pctile:.0f}th pctile)")

        r.disconnect_z = disconnect_z(inst)
        if r.disconnect_z is not None and abs(r.disconnect_z) > 2:
            r.flags.append(f"Disconnect z={r.disconnect_z:+.1f} — mean-reversion setup")

        if event_gate_fn is not None:
            g = event_gate_fn(inst)
            r.gate = "BLOCKED" if g["blocked"] else "clear"
            if g["blocked"]:
                r.flags.append(g["reason"])

        if r.price:
            z = nearest_zones(inst, r.price)
            if z:
                r.supply, r.demand = z["supply"], z["demand"]

        r.daily_sigma = daily_sigma(inst)
        if r.price and r.daily_sigma:
            s = swing_size(account, risk_pct, r.price, r.daily_sigma)
            r.stop_pct, r.units = s["stop_pct"], s["units"]

        rows.append(r)
    return rows
