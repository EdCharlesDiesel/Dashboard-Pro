"""
event_playbook_service.py — pure logic for the Macro Event Playbook page.

Design notes
------------
The session window is 17:00–20:00 SAST. Most of the tier-1 macro prints this
week land *outside* that window, so the tradeable object is not the release
spike but the leg that follows it. Two distinct plans are produced:

  RETRACE  — release lands inside the window. Let the impulse complete, then
             enter on a 38.2–61.8% pullback with M15 structural confirmation.
  BREAKOUT — release landed before the window. The spike is gone; trade the
             post-release consolidation range break in the direction the
             playbook bias supports.

Regime note (Feb 2026 onward): the gold safe-haven relationship is inverted.
Hot inflation -> hike odds -> stronger dollar -> gold DOWN. The beta used to
map a USD bias onto XAUUSD/XAGUSD is therefore a parameter, not a constant,
and can be estimated from price history via estimate_metal_usd_beta().

No Streamlit imports here. Page code lives in pages/Event_Playbook.py.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

SAST = ZoneInfo("Africa/Johannesburg")
UTC = timezone.utc

SESSION_START = time(17, 0)
SESSION_END = time(20, 0)

# How long after a release the initial impulse is assumed to be complete.
DEFAULT_IMPULSE_MINUTES = 30


# --------------------------------------------------------------------------
# Event model
# --------------------------------------------------------------------------

class EventKind(str, Enum):
    PRINT = "print"        # numeric release with a forecast
    SPEECH = "speech"      # qualitative, scored on a hawkish/dovish ladder
    EARNINGS = "earnings"  # risk-sentiment spillover only


class SessionFit(str, Enum):
    IN_WINDOW = "in_window"
    PRE_WINDOW = "pre_window"      # same day, lands before 17:00 SAST
    OUT_OF_SESSION = "out_of_session"


@dataclass
class EventSpec:
    key: str
    name: str
    kind: EventKind
    when_utc: datetime
    currency: str
    # For PRINT events:
    forecast: Optional[float] = None
    previous: Optional[float] = None
    bank_low: Optional[float] = None
    bank_high: Optional[float] = None
    unit: str = "%"
    # +1 => a higher-than-forecast reading is bullish for `currency`.
    # -1 => a higher reading is bearish (e.g. unemployment rate).
    effect_sign: int = 1
    # Std dev of historical (actual - forecast). If None, derived from the
    # bank range as (high - low) / 4.
    surprise_sigma: Optional[float] = None
    tier: int = 1                      # 1 = tradeable, 2 = context only
    timing_confirmed: bool = True
    note: str = ""

    @property
    def when_sast(self) -> datetime:
        return self.when_utc.astimezone(SAST)

    @property
    def sigma(self) -> float:
        if self.surprise_sigma and self.surprise_sigma > 0:
            return self.surprise_sigma
        if self.bank_low is not None and self.bank_high is not None:
            spread = abs(self.bank_high - self.bank_low)
            if spread > 0:
                return spread / 4.0
        return 0.1  # last-resort default for a percent-denominated print

    @property
    def is_complete(self) -> bool:
        """True when the spec has everything the ladder needs."""
        if self.kind is not EventKind.PRINT:
            return True
        return self.forecast is not None


def session_fit(
        ev: EventSpec,
        start: time = SESSION_START,
        end: time = SESSION_END,
) -> SessionFit:
    """Where the release sits relative to the 17:00–20:00 SAST window."""
    local = ev.when_sast
    if start <= local.time() <= end:
        return SessionFit.IN_WINDOW
    # Pre-window means: same trading day, earlier than the open, and close
    # enough that the reaction is still developing when the window starts.
    if local.time() < start and (
            datetime.combine(local.date(), start, tzinfo=SAST) - local
    ) <= timedelta(hours=8):
        return SessionFit.PRE_WINDOW
    return SessionFit.OUT_OF_SESSION


def hours_until_session(ev: EventSpec, start: time = SESSION_START) -> float:
    local = ev.when_sast
    open_dt = datetime.combine(local.date(), start, tzinfo=SAST)
    return (open_dt - local).total_seconds() / 3600.0


# --------------------------------------------------------------------------
# This week's calendar (Mon 24 Aug – Fri 28 Aug 2026)
# --------------------------------------------------------------------------

def week_events() -> list[EventSpec]:
    """Seeded calendar. Edit forecasts in the page before the week starts."""
    return [
        EventSpec(
            key="au_cpi_jul",
            name="AU Monthly CPI Indicator (YoY, Jul)",
            kind=EventKind.PRINT,
            when_utc=datetime(2026, 8, 25, 1, 30, tzinfo=UTC),
            currency="AUD",
            forecast=3.2,
            previous=3.8,
            bank_low=3.0,
            bank_high=3.4,
            effect_sign=1,
            tier=2,
            note="RBA still above target. A hot print puts a hike back on the "
                 "table and supports AUD. Lands 03:30 SAST — overnight risk "
                 "for anything already held, not a session trade.",
        ),
        EventSpec(
            key="us_pce_jul_headline",
            name="US PCE Headline (YoY, Jul)",
            kind=EventKind.PRINT,
            when_utc=datetime(2026, 8, 26, 12, 30, tzinfo=UTC),
            currency="USD",
            forecast=3.6,
            previous=3.7,
            bank_low=3.5,
            bank_high=3.7,
            effect_sign=1,
            tier=1,
            note="The week's main number. Last inflation read the Committee "
                 "sees before Warsh speaks. Hot print forces him to address "
                 "it at Jackson Hole two days later.",
        ),
        EventSpec(
            key="us_pce_jul_core",
            name="US PCE Core (YoY, Jul)",
            kind=EventKind.PRINT,
            when_utc=datetime(2026, 8, 26, 12, 30, tzinfo=UTC),
            currency="USD",
            forecast=3.3,
            previous=3.3,
            bank_low=3.2,
            bank_high=3.4,
            effect_sign=1,
            tier=1,
            note="Core carries more weight than headline for rate pricing. "
                 "When the two disagree, trade core.",
        ),
        EventSpec(
            key="us_gdp_q2_2nd",
            name="US GDP Q2 (2nd estimate, QoQ ann.)",
            kind=EventKind.PRINT,
            when_utc=datetime(2026, 8, 26, 12, 30, tzinfo=UTC),
            currency="USD",
            forecast=None,
            previous=None,
            effect_sign=1,
            tier=2,
            note="Same 8:30 ET slot as PCE. Only matters if it contradicts "
                 "PCE — then expect a messy, two-way first 20 minutes.",
        ),
        EventSpec(
            key="nvda_q2",
            name="NVIDIA earnings (after US close)",
            kind=EventKind.EARNINGS,
            when_utc=datetime(2026, 8, 26, 20, 20, tzinfo=UTC),
            currency="USD",
            tier=2,
            timing_confirmed=False,
            note="Risk-sentiment spillover into Asia. A large miss can lift "
                 "gold on a flight-to-safety bid that has nothing to do with "
                 "rates — it will look like a false signal on rate-based "
                 "models. Flatten or size down anything held overnight.",
        ),
        EventSpec(
            key="tokyo_cpi_aug",
            name="Tokyo CPI (Aug)",
            kind=EventKind.PRINT,
            when_utc=datetime(2026, 8, 27, 23, 30, tzinfo=UTC),  # 08:30 JST Fri
            currency="JPY",
            forecast=None,
            previous=None,
            effect_sign=1,
            tier=2,
            note="National CPI ran 1.8%, a six-month high. Markets price "
                 "roughly 90% odds of a September BoJ hike; a hot Tokyo print "
                 "confirms it. Fill forecast/previous from the calendar. "
                 "Lands 01:30 SAST — overnight gap risk on JPY crosses.",
        ),
        EventSpec(
            key="jh_warsh",
            name="Jackson Hole keynote — Fed Chair Warsh",
            kind=EventKind.SPEECH,
            when_utc=datetime(2026, 8, 28, 14, 0, tzinfo=UTC),  # ~10:00 ET
            currency="USD",
            tier=1,
            timing_confirmed=False,
            note="First Jackson Hole address as Chair, 19 days before the "
                 "16 Sep FOMC. This Fed has cut back on forward guidance, so "
                 "the speech carries real information value. Exact time is "
                 "not published until the agenda drops the evening before — "
                 "confirm it before Friday.",
        ),
    ]


# --------------------------------------------------------------------------
# Surprise scoring and the scenario ladder
# --------------------------------------------------------------------------

@dataclass
class BiasResult:
    z: float                 # standardised surprise
    ccy_bias: float          # -1..+1 for the event currency
    metal_bias: float        # -1..+1 for XAUUSD / XAGUSD
    label: str               # human-readable bucket
    direction: str           # "long" | "short" | "flat" (for the metal)


_LABELS = [
    (1.5, "Very bullish"),
    (0.5, "Bullish"),
    (-0.5, "Ranging"),
    (-1.5, "Bearish"),
]


def _bucket_label(z: float) -> str:
    for threshold, label in _LABELS:
        if z >= threshold:
            return label
    return "Very bearish"


def score_surprise(
        ev: EventSpec,
        actual: float,
        metal_beta: float = -1.0,
        cap: float = 2.0,
) -> BiasResult:
    """Map an actual print onto a currency bias and a metals bias.

    metal_beta is the sensitivity of the metal to the currency bias. Under the
    post-Feb-2026 inverted regime it is negative and roughly -1: a hawkish USD
    surprise pushes gold down.
    """
    if ev.forecast is None:
        raise ValueError(f"{ev.key} has no forecast; cannot score a surprise.")
    z = (actual - ev.forecast) / ev.sigma
    signed = z * ev.effect_sign
    ccy_bias = max(-1.0, min(1.0, signed / cap))
    metal_bias = max(-1.0, min(1.0, ccy_bias * metal_beta))
    if abs(metal_bias) < 0.25:
        direction = "flat"
    else:
        direction = "long" if metal_bias > 0 else "short"
    return BiasResult(
        z=signed,
        ccy_bias=ccy_bias,
        metal_bias=metal_bias,
        label=_bucket_label(signed),
        direction=direction,
    )


def scenario_ladder(
        ev: EventSpec,
        metal_beta: float = -1.0,
        steps: Sequence[float] = (-2.0, -1.0, 0.0, 1.0, 2.0),
) -> list[dict]:
    """The pre-computed 'if it prints X, I do Y' table.

    Rungs are placed at z-multiples of the surprise sigma so the ladder widens
    automatically for noisy releases.
    """
    if ev.forecast is None:
        return []
    rows = []
    for z in steps:
        actual = round(ev.forecast + z * ev.sigma, 3)
        res = score_surprise(ev, actual, metal_beta=metal_beta)
        rows.append(
            {
                "scenario": _rung_name(z),
                "actual": actual,
                "z": round(res.z, 2),
                f"{ev.currency} bias": res.label,
                "metal": {"long": "Long", "short": "Short", "flat": "Stand aside"}[
                    res.direction
                ],
                "conviction": round(abs(res.metal_bias), 2),
            }
        )
    return rows


def _rung_name(z: float) -> str:
    if z <= -2:
        return "Deep miss"
    if z < 0:
        return "Below forecast"
    if z == 0:
        return "On forecast"
    if z < 2:
        return "Above forecast"
    return "Big beat"


# Qualitative ladder for the Jackson Hole speech.
SPEECH_LADDER = [
    {
        "scenario": "Opens the door to a September hike",
        "USD bias": "Very bullish",
        "metal": "Short",
        "conviction": 1.0,
        "tell": "Names inflation persistence as the binding constraint, or "
                "says the Committee is prepared to move if data holds.",
    },
    {
        "scenario": "Hawkish tone, no commitment",
        "USD bias": "Bullish",
        "metal": "Short",
        "conviction": 0.6,
        "tell": "Emphasises the 2% goal and credibility without a rate hint.",
    },
    {
        "scenario": "Sticks to the payments/innovation theme",
        "USD bias": "Ranging",
        "metal": "Stand aside",
        "conviction": 0.2,
        "tell": "Structural remarks only. Expect a spike both ways inside "
                "10 minutes, then mean reversion. The base case.",
    },
    {
        "scenario": "Flags labour-market softening",
        "USD bias": "Bearish",
        "metal": "Long",
        "conviction": 0.6,
        "tell": "Balance-of-risks language shifts toward employment.",
    },
    {
        "scenario": "Signals patience or an easing bias",
        "USD bias": "Very bearish",
        "metal": "Long",
        "conviction": 1.0,
        "tell": "Explicitly pushes back on hike pricing.",
    },
]


# --------------------------------------------------------------------------
# Trade construction
# --------------------------------------------------------------------------

@dataclass
class TradePlan:
    style: str                  # "RETRACE" | "BREAKOUT" | "NO_TRADE"
    direction: str              # "long" | "short" | "flat"
    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    stop: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    r_multiple_t1: Optional[float] = None
    r_multiple_t2: Optional[float] = None
    reasons: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.style != "NO_TRADE" and self.stop is not None


def retrace_plan(
        direction: str,
        impulse_high: float,
        impulse_low: float,
        atr_m15: float,
        fib_near: float = 0.382,
        fib_far: float = 0.618,
        stop_buffer_atr: float = 0.25,
        max_impulse_atr: float = 3.0,
) -> TradePlan:
    """Pullback entry after the release impulse completes.

    Refuses the trade when the impulse is already stretched beyond
    max_impulse_atr — at that point the pullback is as likely to be the start
    of a full retracement as a continuation, and the spread is still wide.
    """
    reasons: list[str] = []
    rng = impulse_high - impulse_low
    if rng <= 0:
        return TradePlan("NO_TRADE", "flat", reasons=["Impulse range is zero or inverted."])
    if atr_m15 > 0 and rng > max_impulse_atr * atr_m15:
        reasons.append(
            f"Impulse is {rng / atr_m15:.1f}x M15 ATR (limit {max_impulse_atr:.1f}x) — "
            "too extended to chase."
        )
        return TradePlan("NO_TRADE", direction, reasons=reasons)
    if direction not in ("long", "short"):
        return TradePlan("NO_TRADE", "flat", reasons=["No directional bias from the print."])

    buf = stop_buffer_atr * atr_m15
    if direction == "long":
        entry_high = impulse_high - fib_near * rng
        entry_low = impulse_high - fib_far * rng
        stop = impulse_low - buf
        anchor = (entry_low + entry_high) / 2
        risk = anchor - stop
        t1 = impulse_high
        t2 = impulse_low + 1.618 * rng
    else:
        entry_low = impulse_low + fib_near * rng
        entry_high = impulse_low + fib_far * rng
        stop = impulse_high + buf
        anchor = (entry_low + entry_high) / 2
        risk = stop - anchor
        t1 = impulse_low
        t2 = impulse_high - 1.618 * rng

    if risk <= 0:
        return TradePlan("NO_TRADE", direction, reasons=["Stop sits through the entry zone."])

    reasons.append(
        f"Impulse {rng:.2f} ({rng / atr_m15:.1f}x M15 ATR). "
        f"Entry on the {fib_near:.3f}–{fib_far:.3f} pullback."
    )
    reasons.append("Requires an M15 rejection candle or structure break in the entry zone.")
    return TradePlan(
        style="RETRACE",
        direction=direction,
        entry_low=round(min(entry_low, entry_high), 3),
        entry_high=round(max(entry_low, entry_high), 3),
        stop=round(stop, 3),
        target_1=round(t1, 3),
        target_2=round(t2, 3),
        r_multiple_t1=round(abs(t1 - anchor) / risk, 2),
        r_multiple_t2=round(abs(t2 - anchor) / risk, 2),
        reasons=reasons,
    )


def breakout_plan(
        direction: str,
        range_high: float,
        range_low: float,
        atr_m15: float,
        buffer_atr: float = 0.15,
        min_range_atr: float = 0.5,
        max_range_atr: float = 2.5,
) -> TradePlan:
    """Post-release consolidation break, for events that fired before 17:00.

    The range is the high/low of the consolidation that formed after the
    release impulse — typically the 15:30–17:00 SAST block for an 8:30 ET
    print. Too tight a range means no real balance formed; too wide means the
    break is already the move.
    """
    reasons: list[str] = []
    rng = range_high - range_low
    if rng <= 0:
        return TradePlan("NO_TRADE", "flat", reasons=["Range high must exceed range low."])
    if atr_m15 <= 0:
        return TradePlan("NO_TRADE", direction, reasons=["Need a positive M15 ATR."])
    ratio = rng / atr_m15
    if ratio < min_range_atr:
        reasons.append(f"Range is only {ratio:.2f}x ATR — no balance formed yet, keep waiting.")
        return TradePlan("NO_TRADE", direction, reasons=reasons)
    if ratio > max_range_atr:
        reasons.append(f"Range is {ratio:.2f}x ATR — the move already happened. Stand aside.")
        return TradePlan("NO_TRADE", direction, reasons=reasons)
    if direction not in ("long", "short"):
        return TradePlan("NO_TRADE", "flat", reasons=["No directional bias from the print."])

    buf = buffer_atr * atr_m15
    if direction == "long":
        entry = range_high + buf
        stop = range_low - buf
        risk = entry - stop
        t1 = entry + rng
        t2 = entry + 2 * rng
    else:
        entry = range_low - buf
        stop = range_high + buf
        risk = stop - entry
        t1 = entry - rng
        t2 = entry - 2 * rng

    reasons.append(
        f"Consolidation {rng:.2f} ({ratio:.2f}x M15 ATR). Break plus a "
        f"{buffer_atr:.2f} ATR buffer to clear the stop cluster."
    )
    reasons.append("Measured move targets: 1x and 2x the range.")
    return TradePlan(
        style="BREAKOUT",
        direction=direction,
        entry_low=round(entry, 3),
        entry_high=round(entry, 3),
        stop=round(stop, 3),
        target_1=round(t1, 3),
        target_2=round(t2, 3),
        r_multiple_t1=round(abs(t1 - entry) / risk, 2),
        r_multiple_t2=round(abs(t2 - entry) / risk, 2),
        reasons=reasons,
    )


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------

def size_from_stop(
        equity: float,
        risk_pct: float,
        entry: float,
        stop: float,
        contract_value: float = 100.0,
) -> dict:
    """Units implied by the actual stop distance.

    contract_value is the $ move per 1.0 price unit per 1 unit of position.
    For XAUUSD on a standard lot (100 oz) that is 100.
    """
    risk_cash = equity * risk_pct / 100.0
    stop_distance = abs(entry - stop)
    if stop_distance <= 0 or contract_value <= 0:
        return {"risk_cash": risk_cash, "stop_distance": stop_distance, "units": 0.0}
    units = risk_cash / (stop_distance * contract_value)
    return {
        "risk_cash": round(risk_cash, 2),
        "stop_distance": round(stop_distance, 3),
        "units": round(units, 3),
        "lots": round(units, 2),
    }


def vol_scaled_size(
        equity: float,
        risk_pct: float,
        price: float,
        daily_vol_pct: float,
        horizon_days: float = 1.0,
        sigma_multiple: float = 2.0,
) -> dict:
    """Size from a 2-sigma-root-T stop so a volatile regime shrinks the position.

    Same convention as the swing playbook: stop = k * sigma * sqrt(T) * price.
    """
    risk_cash = equity * risk_pct / 100.0
    stop_distance = sigma_multiple * (daily_vol_pct / 100.0) * math.sqrt(max(horizon_days, 1e-9)) * price
    if stop_distance <= 0:
        return {"risk_cash": risk_cash, "stop_distance": 0.0, "units": 0.0}
    return {
        "risk_cash": round(risk_cash, 2),
        "stop_distance": round(stop_distance, 3),
        "units": round(risk_cash / stop_distance, 3),
    }


# --------------------------------------------------------------------------
# Regime beta estimation
# --------------------------------------------------------------------------

def estimate_metal_usd_beta(
        conn,
        metal_symbol: str = "XAUUSD",
        dollar_symbol: str = "DXY",
        lookback_days: int = 120,
        table: str = "prices_daily",
        fallback: float = -1.0,
) -> tuple[float, str]:
    """OLS slope of metal log-returns on dollar log-returns.

    Returns (beta, source). Falls back to `fallback` and an explanatory string
    whenever the table is missing or there is not enough overlap, so the page
    still renders on a fresh database.
    """
    if conn is None:
        return fallback, "No database connection — using the default inverted-regime beta."
    try:
        from sqlalchemy import text
        import numpy as np
        import pandas as pd

        q = text(
            f"""
            SELECT ts::date AS d, symbol, close
            FROM {table}
            WHERE symbol IN (:m, :u)
              AND ts >= NOW() - (:days || ' days')::interval
            ORDER BY ts
            """
        )
        df = pd.read_sql(q, conn, params={"m": metal_symbol, "u": dollar_symbol, "days": lookback_days})
        if df.empty:
            return fallback, f"No rows in {table} for {metal_symbol}/{dollar_symbol}."
        wide = df.pivot_table(index="d", columns="symbol", values="close").dropna()
        if len(wide) < 30:
            return fallback, f"Only {len(wide)} overlapping days — not enough to estimate."
        r = np.log(wide).diff().dropna()
        x = r[dollar_symbol].to_numpy()
        y = r[metal_symbol].to_numpy()
        var = float(np.var(x, ddof=1))
        if var <= 0:
            return fallback, "Dollar series has no variance."
        beta = float(np.cov(y, x, ddof=1)[0, 1] / var)
        corr = float(np.corrcoef(y, x)[0, 1])
        return beta, (
            f"Estimated from {len(r)} daily returns: beta {beta:+.2f}, "
            f"correlation {corr:+.2f}."
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any schema gap
        return fallback, f"Could not estimate ({type(exc).__name__}: {exc}). Using default."


# --------------------------------------------------------------------------
# Persistence — build a reaction database over time
# --------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS event_playbook_log (
    id              BIGSERIAL PRIMARY KEY,
    fingerprint     TEXT UNIQUE NOT NULL,
    event_key       TEXT NOT NULL,
    event_name      TEXT NOT NULL,
    when_utc        TIMESTAMPTZ NOT NULL,
    currency        TEXT NOT NULL,
    instrument      TEXT NOT NULL,
    forecast        DOUBLE PRECISION,
    actual          DOUBLE PRECISION,
    surprise_z      DOUBLE PRECISION,
    metal_beta      DOUBLE PRECISION,
    planned_dir     TEXT,
    plan_style      TEXT,
    entry_low       DOUBLE PRECISION,
    entry_high      DOUBLE PRECISION,
    stop            DOUBLE PRECISION,
    target_1        DOUBLE PRECISION,
    target_2        DOUBLE PRECISION,
    realised_move   DOUBLE PRECISION,
    outcome         TEXT,
    thesis          TEXT,
    invalidation    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_event_playbook_when ON event_playbook_log (when_utc DESC);
"""


def fingerprint(event_key: str, when_utc: datetime, instrument: str) -> str:
    raw = f"{event_key}|{when_utc.astimezone(UTC).isoformat()}|{instrument}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def ensure_schema(conn) -> None:
    from sqlalchemy import text
    for stmt in filter(None, (s.strip() for s in DDL.split(";"))):
        conn.execute(text(stmt))
    if hasattr(conn, "commit"):
        conn.commit()


def log_plan(
        conn,
        ev: EventSpec,
        instrument: str,
        actual: Optional[float],
        bias: Optional[BiasResult],
        plan: TradePlan,
        metal_beta: float,
        thesis: str = "",
        invalidation: str = "",
) -> str:
    """Idempotent upsert keyed on a SHA-256 fingerprint, matching the signal queue."""
    from sqlalchemy import text

    fp = fingerprint(ev.key, ev.when_utc, instrument)
    conn.execute(
        text(
            """
            INSERT INTO event_playbook_log (
                fingerprint, event_key, event_name, when_utc, currency, instrument,
                forecast, actual, surprise_z, metal_beta, planned_dir, plan_style,
                entry_low, entry_high, stop, target_1, target_2, thesis, invalidation
            ) VALUES (
                :fp, :key, :name, :when_utc, :ccy, :inst,
                :forecast, :actual, :z, :beta, :dir, :style,
                :e_lo, :e_hi, :stop, :t1, :t2, :thesis, :inval
            )
            ON CONFLICT (fingerprint) DO UPDATE SET
                actual       = EXCLUDED.actual,
                surprise_z   = EXCLUDED.surprise_z,
                metal_beta   = EXCLUDED.metal_beta,
                planned_dir  = EXCLUDED.planned_dir,
                plan_style   = EXCLUDED.plan_style,
                entry_low    = EXCLUDED.entry_low,
                entry_high   = EXCLUDED.entry_high,
                stop         = EXCLUDED.stop,
                target_1     = EXCLUDED.target_1,
                target_2     = EXCLUDED.target_2,
                thesis       = EXCLUDED.thesis,
                invalidation = EXCLUDED.invalidation
            """
        ),
        {
            "fp": fp,
            "key": ev.key,
            "name": ev.name,
            "when_utc": ev.when_utc,
            "ccy": ev.currency,
            "inst": instrument,
            "forecast": ev.forecast,
            "actual": actual,
            "z": bias.z if bias else None,
            "beta": metal_beta,
            "dir": plan.direction,
            "style": plan.style,
            "e_lo": plan.entry_low,
            "e_hi": plan.entry_high,
            "stop": plan.stop,
            "t1": plan.target_1,
            "t2": plan.target_2,
            "thesis": thesis,
            "inval": invalidation,
        },
    )
    if hasattr(conn, "commit"):
        conn.commit()
    return fp


def alert_text(ev: EventSpec, bias: BiasResult, plan: TradePlan, instrument: str) -> str:
    """One-block summary for the evening sentry's Telegram push."""
    if not plan.valid:
        reason = plan.reasons[0] if plan.reasons else "conditions not met"
        return f"[{instrument}] {ev.name}: stand aside — {reason}"
    return (
        f"[{instrument}] {ev.name} @ {ev.when_sast:%a %H:%M} SAST\n"
        f"Surprise z {bias.z:+.2f} -> {ev.currency} {bias.label.lower()}\n"
        f"{plan.style} {plan.direction.upper()}  "
        f"entry {plan.entry_low}–{plan.entry_high}  stop {plan.stop}\n"
        f"T1 {plan.target_1} ({plan.r_multiple_t1}R)  "
        f"T2 {plan.target_2} ({plan.r_multiple_t2}R)"
    )