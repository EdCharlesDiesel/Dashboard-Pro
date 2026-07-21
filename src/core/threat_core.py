"""
threat_core.py — pure logic for the Threat Board.

No Streamlit imports here: this module is shared by
  - threat_board_tab.py   (dashboard UI)
  - threat_sentry_hook.py (evening_sentry integration)

Threat model (position-aware, scored 0-100, banded green/amber/red):
  1. Concentration  — correlated stop-out $ vs equity          (weight 30)
  2. Intervention   — JPY MoF proximity/speed + verbal scan    (weight 30)
  3. Squeeze        — COT spec positioning percentile vs you   (weight 20)
  4. Calendar       — red-impact events on held currencies     (weight 10)
  5. Regime         — exposure vs risk-on/off state conflict   (weight 10)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Config defaults (overridable from the UI)
# ---------------------------------------------------------------------------
JPY_ZONE_LOW, JPY_ZONE_HIGH = 162.0, 163.0   # MoF sensitivity band — review as regime moves
ROC_AMBER_PCT, ROC_RED_PCT = 1.5, 2.5        # 5-day USDJPY rate-of-change thresholds (%)
CONC_AMBER, CONC_RED = 0.10, 0.20            # correlated stop-out as fraction of equity
COT_LOOKBACK_WEEKS = 156                     # 3y percentile window
LEGACY_COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
JPY_FUTURES_CODE = "097741"                  # CME JPY futures, legacy report

WEIGHTS = {"concentration": 30, "intervention": 30, "squeeze": 20,
           "calendar": 10, "regime": 10}

INTERVENTION_KEYWORDS = [
    "excessive", "one-sided", "disorderly", "won't rule out", "will not rule out",
    "all options", "decisive action", "speculative moves", "rapid moves",
    "closely watching", "high sense of urgency", "intervention", "fx moves",
]
MOF_SPEAKERS = ["katayama", "mof", "ministry of finance", "kanda", "mimura",
                "vice finance", "boj", "ueda"]


# ---------------------------------------------------------------------------
# Positions & exposure
# ---------------------------------------------------------------------------
@dataclass
class Position:
    pair: str          # "USDJPY", "AUDJPY", ...
    direction: str     # "long" | "short"
    lots: float        # standard lots (1.0 = 100k units)
    entry: float
    stop: float

    @property
    def base(self) -> str:
        return self.pair[:3].upper()

    @property
    def quote(self) -> str:
        return self.pair[3:6].upper()


def pip_size(pair: str) -> float:
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


def pip_value_usd(pos: Position, usdjpy: float) -> float:
    """USD value of one pip for this position. Handles JPY- and USD-quoted pairs;
    falls back to $10/lot for exotic quotes (close enough for a threat gauge)."""
    units = pos.lots * 100_000
    if pos.quote == "JPY":
        return units * 0.01 / usdjpy          # e.g. 0.20 lots -> ¥2 000/pip -> USD
    if pos.quote == "USD":
        return units * 0.0001
    return pos.lots * 10.0


def stop_risk_usd(pos: Position, usdjpy: float) -> float:
    pips = abs(pos.entry - pos.stop) / pip_size(pos.pair)
    return pips * pip_value_usd(pos, usdjpy)


def currency_exposure(positions: list[Position]) -> dict[str, float]:
    """Net lots per currency. Long AUDJPY = +AUD lots, -JPY lots."""
    exp: dict[str, float] = {}
    for p in positions:
        sign = 1 if p.direction == "long" else -1
        exp[p.base] = exp.get(p.base, 0.0) + sign * p.lots
        exp[p.quote] = exp.get(p.quote, 0.0) - sign * p.lots
    return {k: round(v, 2) for k, v in exp.items() if abs(v) > 1e-9}


def correlated_stop_risk(positions: list[Position], usdjpy: float) -> tuple[float, str]:
    """Worst-case correlated cluster: for each currency, sum stop risks of all
    positions exposed to it in the same direction. Returns (usd, currency)."""
    worst, worst_ccy = 0.0, ""
    currencies = {c for p in positions for c in (p.base, p.quote)}
    for ccy in currencies:
        cluster = 0.0
        signs = set()
        for p in positions:
            sign = 1 if p.direction == "long" else -1
            if p.base == ccy:
                signs.add(sign)
                cluster += stop_risk_usd(p, usdjpy)
            elif p.quote == ccy:
                signs.add(-sign)
                cluster += stop_risk_usd(p, usdjpy)
        if len(signs) == 1 and cluster > worst:   # only if all same direction
            worst, worst_ccy = cluster, ccy
    return worst, worst_ccy


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def usdjpy_snapshot() -> tuple[float, float]:  # pragma: no cover - live yfinance
    """(last_price, 5-day rate of change %). Raises on data failure."""
    import yfinance as yf
    df = yf.download("USDJPY=X", period="1mo", interval="1d",
                     progress=False, auto_adjust=True)
    close = df["Close"].dropna()
    if hasattr(close, "columns"):                 # yf sometimes returns a frame
        close = close.iloc[:, 0]
    last = float(close.iloc[-1])
    roc5 = (last / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0.0
    return last, roc5


def jpy_cot_net_percentile() -> float | None:  # pragma: no cover - live CFTC fetch
    """Percentile (0-100) of latest non-commercial net JPY futures position
    over the 3y window. Low percentile = specs deeply net short yen."""
    try:
        r = requests.get(LEGACY_COT_URL, params={
            "cftc_contract_market_code": JPY_FUTURES_CODE,
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": COT_LOOKBACK_WEEKS,
        }, timeout=20)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if df.empty:
            return None
        net = (pd.to_numeric(df["noncomm_positions_long_all"])
               - pd.to_numeric(df["noncomm_positions_short_all"]))
        return float((net.iloc[1:] < net.iloc[0]).mean() * 100)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Optional integrations (defensive imports — dashboard keeps working without)
# ---------------------------------------------------------------------------
def upcoming_red_events(currencies: set[str], days: int = 7):  # pragma: no cover - optional news_fetcher integration
    """Expects news_fetcher.get_upcoming_events() -> list of dicts with at least
    {date, currency, impact, title}. Returns filtered list, or None if unavailable."""
    try:
        from news_fetcher import get_upcoming_events
        events = get_upcoming_events()
        horizon = datetime.now(timezone.utc) + timedelta(days=days)
        out = []
        for e in events:
            when = pd.to_datetime(e.get("date"), utc=True, errors="coerce")
            if when is pd.NaT or when > horizon:
                continue
            if str(e.get("impact", "")).lower() in ("high", "red") \
                    and e.get("currency", "").upper() in currencies:
                out.append(e)
        return out
    except Exception:
        return None


def scan_intervention_headlines() -> list[str]:  # pragma: no cover - optional news_fetcher integration
    """Expects news_fetcher.get_recent_headlines() -> list[str]. Returns hits."""
    try:
        from news_fetcher import get_recent_headlines
        hits = []
        for h in get_recent_headlines():
            low = h.lower()
            if any(k in low for k in INTERVENTION_KEYWORDS) \
                    and any(s in low for s in MOF_SPEAKERS):
                hits.append(h)
        return hits
    except Exception:
        return []


def current_regime() -> str | None:  # pragma: no cover - optional regime integration
    """Expects regime.current_regime() -> 'risk_on' | 'risk_off' | 'neutral'."""
    try:
        from regime import current_regime as _cr
        return _cr()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Component scores (each 0-100)
# ---------------------------------------------------------------------------
def score_concentration(worst_usd: float, equity: float) -> float:
    if equity <= 0:
        return 0.0
    frac = worst_usd / equity
    if frac >= CONC_RED:
        return min(100.0, 70 + (frac - CONC_RED) * 300)
    if frac >= CONC_AMBER:
        return 40 + (frac - CONC_AMBER) / (CONC_RED - CONC_AMBER) * 30
    return frac / CONC_AMBER * 40


def score_intervention(last: float, roc5: float, jpy_net_lots: float,
                       zone=(JPY_ZONE_LOW, JPY_ZONE_HIGH),
                       headline_hits: int = 0) -> float:
    """Only threatens accounts net SHORT yen (long USDJPY/AUDJPY etc.)."""
    if jpy_net_lots >= 0:
        return 0.0
    lo, hi = zone
    if last >= hi:
        base = 85.0
    elif last >= lo:
        base = 70.0
    elif last >= lo - 1.0:
        base = 40.0
    else:
        base = 10.0
    if roc5 >= ROC_RED_PCT:
        base += 20
    elif roc5 >= ROC_AMBER_PCT:
        base += 10
    base += min(15, headline_hits * 8)
    return min(100.0, base)


def score_squeeze(cot_pct: float | None, jpy_net_lots: float) -> float:
    """Crowded with the herd = risk. Specs deeply short yen (low percentile)
    while you are also short yen -> squeeze takes everyone out together."""
    if cot_pct is None or jpy_net_lots == 0:
        return 0.0
    if jpy_net_lots < 0:            # you are short yen
        if cot_pct <= 10:
            return 80.0
        if cot_pct <= 25:
            return 55.0
        return 20.0
    else:                            # you are long yen, crowded longs squeeze down
        if cot_pct >= 90:
            return 80.0
        if cot_pct >= 75:
            return 55.0
        return 20.0


def score_calendar(events) -> float:
    if events is None:
        return 0.0                   # feed unavailable -> don't fake a signal
    n = len(events)
    return min(100.0, n * 25.0)


def score_regime(regime: str | None, exposure: dict[str, float]) -> float:
    """Risk-off while long risk currencies (AUD, NZD, CAD) or short JPY/CHF
    is a conflict; and vice versa."""
    if regime is None:
        return 0.0
    risk_ccys = {"AUD", "NZD", "CAD"}
    haven_ccys = {"JPY", "CHF"}
    long_risk = any(exposure.get(c, 0) > 0 for c in risk_ccys)
    short_haven = any(exposure.get(c, 0) < 0 for c in haven_ccys)
    if regime == "risk_off" and (long_risk or short_haven):
        return 75.0
    if regime == "risk_on" and (not long_risk and not short_haven
                                and any(exposure.get(c, 0) < 0 for c in risk_ccys)):
        return 60.0
    return 10.0


def band(score: float) -> str:
    return "green" if score < 40 else ("amber" if score < 70 else "red")


@dataclass
class ThreatReport:
    components: dict          # name -> score
    detail: dict              # supporting numbers for display/alerts
    score: float
    state: str


def build_report(positions: list[Position], equity: float,
                 zone=(JPY_ZONE_LOW, JPY_ZONE_HIGH)) -> ThreatReport:
    last, roc5 = usdjpy_snapshot()
    exposure = currency_exposure(positions)
    worst_usd, worst_ccy = correlated_stop_risk(positions, last)
    cot_pct = jpy_cot_net_percentile()
    held = {c for p in positions for c in (p.base, p.quote)}
    events = upcoming_red_events(held)
    hits = scan_intervention_headlines()
    regime = current_regime()

    comps = {
        "concentration": score_concentration(worst_usd, equity),
        "intervention": score_intervention(last, roc5, exposure.get("JPY", 0.0),
                                           zone, len(hits)),
        "squeeze": score_squeeze(cot_pct, exposure.get("JPY", 0.0)),
        "calendar": score_calendar(events),
        "regime": score_regime(regime, exposure),
    }
    total = sum(comps[k] * WEIGHTS[k] for k in comps) / sum(WEIGHTS.values())
    detail = {
        "usdjpy_last": round(last, 3), "usdjpy_roc5_pct": round(roc5, 2),
        "exposure": exposure, "worst_cluster_usd": round(worst_usd, 2),
        "worst_cluster_ccy": worst_ccy,
        "worst_cluster_pct_equity": round(worst_usd / equity * 100, 1) if equity else None,
        "jpy_cot_percentile": cot_pct,
        "red_events": events, "headline_hits": hits, "regime": regime,
    }
    return ThreatReport(comps, detail, round(total, 1), band(total))


# ---------------------------------------------------------------------------
# Persistence — follows the render(conn)/SQLAlchemy pattern
# ---------------------------------------------------------------------------
def ensure_tables(conn) -> None:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS threat_positions (
            id SERIAL PRIMARY KEY,
            pair TEXT NOT NULL, direction TEXT NOT NULL,
            lots NUMERIC NOT NULL, entry NUMERIC NOT NULL, stop NUMERIC NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )"""))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS threat_journal (
            ts TIMESTAMPTZ DEFAULT now(),
            score NUMERIC NOT NULL, state TEXT NOT NULL,
            concentration NUMERIC, intervention NUMERIC, squeeze NUMERIC,
            calendar NUMERIC, regime NUMERIC, detail JSONB
        )"""))
    conn.commit()


def load_positions(conn) -> list[Position]:
    rows = conn.execute(text(
        "SELECT pair, direction, lots, entry, stop FROM threat_positions ORDER BY id"
    )).fetchall()
    return [Position(r[0], r[1], float(r[2]), float(r[3]), float(r[4])) for r in rows]


def journal(conn, rep: ThreatReport) -> None:
    import json
    conn.execute(text("""
        INSERT INTO threat_journal
            (score, state, concentration, intervention, squeeze, calendar, regime, detail)
        VALUES (:s, :st, :c, :i, :q, :cal, :r, :d)
    """), {"s": rep.score, "st": rep.state,
           "c": rep.components["concentration"], "i": rep.components["intervention"],
           "q": rep.components["squeeze"], "cal": rep.components["calendar"],
           "r": rep.components["regime"],
           "d": json.dumps(rep.detail, default=str)})
    conn.commit()


def last_state(conn) -> str | None:
    row = conn.execute(text(
        "SELECT state FROM threat_journal ORDER BY ts DESC LIMIT 1")).fetchone()
    return row[0] if row else None