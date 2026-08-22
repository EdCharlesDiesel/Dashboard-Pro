"""Macro event reaction map — the pure core.

Four scheduled US releases (NFP, CPI, PPI, FOMC), each scored the same way:

1. Turn the release into a single standardised surprise score, positive =
   hawkish, by z-scoring every component the event declares and weighting them.
2. Run that score through a *regime-aware, event-specific* transmission chain
   instead of the naive one-way chain that gets passed around on social media.
3. Rank which instruments are actually exposed, with a conviction score that
   drops when the rate channel and the growth channel disagree.

Everything that differs between events lives on its :class:`EventSpec` — the
components, the chain, the betas, the release time, the session phases. There
is no module-level "the components" or "the betas" any more, because there is
no single answer to either question.

**Three facts this design turns on.**

*"Hawkish" does not mean the same thing for growth in every event.* A hot NFP
is hawkish and growth-positive. A hot CPI is hawkish and growth-*negative* — a
real-income squeeze plus tighter policy. Every risk asset's growth beta
therefore flips sign between NFP and the three price/policy events. That sign
is baked into each event's own exposure table rather than carried as a
separate flag, so there is exactly one place to read it.

*The chain is the point of the page, so the chain is per-event.* CPI never
passes through jobs, and its third node moves *against* the surprise. Chain
nodes carry their own sign for that reason.

*FOMC is not an 08:30 event and the presser is not the statement.* The decision
lands 14:00 New York (≈20:00 SAST — the desk is live for it, unlike the morning
releases), and the presser at +30 routinely reverses the statement move.

The betas are PRIORS — starting values, not findings, and the three
non-payrolls tables have less standing than NFP's since none of them has been
fitted to this desk's tape. What makes them falsifiable is that
:func:`board_to_signals` feeds each event's high-conviction calls to
``trade_setups`` under its *own* source tag, so the Trade Journal's Source
Scorecard grades each event separately — CPI betas being good says nothing
about NFP betas.

``pages/nfp_reaction.py`` is the UI over this module; the split is the same one
``src/core/fibo_ribbon.py`` has with its page, and for the same reason — the
scoring has to be testable on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from src.services.broker_symbols import normalize_symbol

NY = ZoneInfo("America/New_York")
JHB = ZoneInfo("Africa/Johannesburg")


# --------------------------------------------------------------------------
# The pieces an event is built from
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Component:
    """One measured input to an event's surprise score.

    ``sd`` is the rough standard deviation of the surprise in the component's
    own units — it is what sets the z scale, and it is the number to
    re-estimate first from a real release history.

    A **paired** component is scored as ``actual - consensus``; the caller
    supplies both under ``key`` and ``key + "_c"``. A ``delta_only`` component
    *is* the surprise already (a payrolls revision, the FOMC dot shift, the
    tone dial) and takes no consensus.

    ``invert`` flips the sign: for unemployment a *lower* print is hawkish.

    ``default`` / ``step`` / ``fmt`` describe how the input is *entered*, not
    how it is scored. They live here rather than in the page because they are
    facts about the statistic — U3 is published to one decimal, core CPI to
    two — and the page's job is to render whatever the event declares, not to
    know that payrolls are in thousands.
    """
    key: str
    label: str
    sd: float
    weight: float
    invert: bool = False
    delta_only: bool = False
    help: str = ""
    default: float = 0.0
    step: float = 0.1
    fmt: str = "%.2f"


@dataclass(frozen=True)
class Exposure:
    """One instrument's response to a 1σ surprise, split across two channels.

    ``beta_rate`` is the response through the Fed-path / real-yield channel,
    ``beta_growth`` through the risk-appetite / activity channel. ``unit`` is
    the typical absolute move at ``|score| = 1`` over the first 30 minutes, in
    the instrument's own units.
    """
    symbol: str
    beta_rate: float
    beta_growth: float
    unit: float
    unit_label: str
    decimals: int


# (node label, sign vs the surprise). +1 moves with a hawkish print, -1 against.
ChainNode = Tuple[str, int]


@dataclass(frozen=True)
class Phase:
    """One window of the session around a release.

    A phase is anchored **either** relative to the release (``start_min`` /
    ``end_min``) **or** to an absolute SAST wall clock (``start_sast`` /
    ``end_sast``), never both. The desk's own trading window is local and must
    not drift with US daylight saving, which is exactly what a relative offset
    would do to it.
    """
    name: str
    note: str
    start_min: Optional[int] = None
    end_min: Optional[int] = None
    start_sast: Optional[time] = None
    end_sast: Optional[time] = None


@dataclass(frozen=True)
class EventSpec:
    key: str
    label: str
    source_tag: str
    calendar_key: str          # the name src.core.event_calendar uses
    release_time_ny: time
    components: Tuple[Component, ...]
    chain: Tuple[ChainNode, ...]
    exposures: Tuple[Exposure, ...]
    phases: Tuple[Phase, ...]
    note: str


# --------------------------------------------------------------------------
# Surprise
# --------------------------------------------------------------------------

@dataclass
class Surprise:
    z: Dict[str, float]
    composite: float

    @property
    def label(self) -> str:
        a = abs(self.composite)
        if a < 0.35:
            return "In line"
        if a < 1.0:
            return "Mild"
        if a < 2.0:
            return "Significant"
        return "Outlier"

    @property
    def direction(self) -> str:
        if abs(self.composite) < 0.35:
            return "neutral"
        return "hawkish" if self.composite > 0 else "dovish"


def compute_surprise(spec: EventSpec, values: Dict[str, float]) -> Surprise:
    """Standardised composite for ``spec``. Positive = hawkish.

    A component is scored when its inputs are present and **dropped** when
    they are not, with the weights renormalised over whatever survived. That
    is what lets a headline-only entry still produce a usable score instead of
    one dragged toward zero by components nobody filled in. Note the
    consequence: leaving a consensus box empty is *not* the same as typing the
    actual into it — the first says "I don't know", the second says "it came
    in on forecast", and only one of those is an observation.

    An entry with nothing in it scores 0.0 rather than raising, because that is
    the page's opening state on every load.
    """
    z: Dict[str, float] = {}
    for c in spec.components:
        raw = values.get(c.key)
        if raw is None:
            continue
        if c.delta_only:
            diff = float(raw)
        else:
            consensus = values.get(c.key + "_c")
            if consensus is None:
                continue
            diff = float(raw) - float(consensus)
        z[c.key] = (-diff if c.invert else diff) / c.sd

    live = {c.key: c.weight for c in spec.components if c.key in z}
    wsum = sum(live.values())
    composite = (sum(live[k] * z[k] for k in live) / wsum) if wsum else 0.0
    return Surprise(z=z, composite=composite)


# --------------------------------------------------------------------------
# Regimes
# --------------------------------------------------------------------------
# The chain in the viral clip assumes one regime forever. In practice the
# market's reaction function rotates between "the Fed path is all that
# matters" and "growth is all that matters", and gold and equities change
# sign across that rotation. Pick the regime from what the market has been
# rewarding in the last few prints, not from what should be true.
#
# The regime is a property of the market, not of which statistic was released,
# so it is deliberately shared across all four events.

REGIMES: Dict[str, Dict[str, Any]] = {
    "Rates-led (good news is bad news)": {
        "w_rate": 1.00,
        "w_growth": 0.15,
        "note": "Inflation still the binding constraint. Strong data lifts yields "
                "and the dollar, and equities sell off with gold.",
    },
    "Balanced": {
        "w_rate": 0.70,
        "w_growth": 0.55,
        "note": "Fed near neutral. Dollar tracks the rate channel, equities "
                "track the growth channel, gold gets a muddled two-sided read.",
    },
    "Growth-scare (bad news is bad news)": {
        "w_rate": 0.35,
        "w_growth": 1.00,
        "note": "Recession risk dominates. A weak print sells equities AND bids "
                "gold, and the dollar's haven bid can beat its rate discount.",
    },
}


def score_instruments(
    spec: EventSpec,
    composite: float,
    regime: str,
    overrides: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Score every instrument for this surprise, under this event and regime.

    conviction = |a + b| / (|a| + |b|) where a is the rate-channel contribution
    and b the growth-channel contribution. It falls to 0 when the two channels
    pull equally hard in opposite directions — which is exactly when the
    30-minute reaction is a coin flip and the setup should be skipped.
    """
    w_rate = REGIMES[regime]["w_rate"]
    w_growth = REGIMES[regime]["w_growth"]
    overrides = overrides or {}

    rows = []
    for exp in spec.exposures:
        b_rate, b_growth = exp.beta_rate, exp.beta_growth
        if exp.symbol in overrides:
            b_rate, b_growth = overrides[exp.symbol]

        a = w_rate * b_rate
        b = w_growth * b_growth
        net = a + b
        score = composite * net

        denom = abs(a) + abs(b)
        conviction = abs(net) / denom if denom > 1e-9 else 0.0

        rows.append(
            {
                "symbol": exp.symbol,
                "score": score,
                "direction": "up" if score > 0 else ("down" if score < 0 else "flat"),
                "expected_move": abs(score) * exp.unit,
                "unit": exp.unit_label,
                "decimals": exp.decimals,
                "conviction": conviction,
                "rate_channel": composite * a,
                "growth_channel": composite * b,
            }
        )

    df = pd.DataFrame(rows)
    df["abs_score"] = df["score"].abs()
    return df.sort_values("abs_score", ascending=False).reset_index(drop=True)


def chain_leaves(
    spec: EventSpec,
    surprise: Surprise,
    board: pd.DataFrame,
) -> Tuple[List[Tuple[str, bool]], List[Tuple[str, bool]]]:
    """The transmission chain's node directions, for the page to draw.

    Trunk nodes follow the sign of the surprise *multiplied by their own sign*
    — which is how a hot CPI draws "prices" up and "real income" down on the
    same print. The three leaves are read back off the scored board instead,
    which is how a growth-scare regime ends up drawing equities up on a
    hawkish payrolls print while the trunk still points hawkish. Deriving the
    leaves from the surprise would reproduce exactly the naive chain this page
    exists to correct.
    """
    hawk = surprise.composite > 0
    chain = [(name, hawk if sign > 0 else not hawk) for name, sign in spec.chain]

    def dir_of(sym: str) -> bool:
        row = board.loc[board["symbol"] == sym]
        return bool(row["score"].iloc[0] > 0) if len(row) else hawk

    leaves = [("indices", dir_of("US500")), ("gold", dir_of("XAUUSD")),
              ("usd", dir_of("DXY"))]
    return chain, leaves


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def release_datetime_sast(spec: EventSpec, d: date) -> datetime:
    """The release moment in SAST, DST-aware.

    08:30 New York is 15:30 SAST in winter and 14:30 in summer; hard-coding
    either puts the desk an hour out for half the year.
    """
    return datetime.combine(d, spec.release_time_ny, tzinfo=NY).astimezone(JHB)


def timing_frame(spec: EventSpec, d: date) -> pd.DataFrame:
    t0 = release_datetime_sast(spec, d)
    rows = []
    for ph in spec.phases:
        if ph.start_min is not None:
            start = t0 + timedelta(minutes=ph.start_min)
            end = t0 + timedelta(minutes=ph.end_min or ph.start_min)
        else:
            start = datetime.combine(d, ph.start_sast, tzinfo=JHB)
            end = datetime.combine(d, ph.end_sast, tzinfo=JHB)
        rows.append({
            "Phase": ph.name,
            "SAST": "{0:%H:%M}–{1:%H:%M}".format(start, end),
            "What is happening": ph.note,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Board -> trade_setups
# --------------------------------------------------------------------------

# Below this the print is in line with consensus and the board is noise: the
# page is declining to forecast rather than forecasting "flat", the same policy
# forecast_dashboard applies to a neutral driver score.
MIN_ABS_COMPOSITE = 0.5

# Below this the rate and growth channels are fighting hard enough that the
# first fifteen minutes are a coin flip. Storing those rows would dilute the
# source's expectancy with trades the page itself says not to take.
MIN_CONVICTION = 0.45


def _conviction_label(ratio: float) -> str:
    if ratio >= 0.85:
        return "High"
    if ratio >= 0.65:
        return "Medium"
    return "Low"


def board_to_signals(
    spec: EventSpec,
    board: pd.DataFrame,
    release_date: date,
    regime: str,
    composite: float,
    *,
    min_abs_composite: float = MIN_ABS_COMPOSITE,
    min_conviction: float = MIN_CONVICTION,
) -> List[Dict[str, Any]]:
    """The rows of ``board`` worth persisting, as signal dicts for the store.

    Two filters and one translation:

    - **The composite gate.** An in-line print returns ``[]``. The page's
      inputs are typed by hand, so this is also what keeps a headless
      ``signal_sweep`` pass — which renders the page at its consensus-matching
      defaults — from storing a release that never happened.
    - **The conviction gate.** See ``MIN_CONVICTION``.
    - **The registry translation.** ``normalize_symbol`` is the app's single
      broker-symbol map; a symbol it cannot resolve (DXY, US500, NAS100,
      US10Y, BTCUSD) has no tradable registry pair, so it stays on the board
      for reading and never reaches ``trade_setups``. Same rule
      ``disconnect_mon`` applies to its audit-only relationships.

    ``bar_time`` is the release date: this read is new once per release, not
    once per day, so that is the period the dedupe ledger must key on — the
    same reason ``cot_composite`` keys on the CFTC report date.
    """
    if abs(composite) < min_abs_composite:
        return []

    signals: List[Dict[str, Any]] = []
    for _, row in board.iterrows():
        pair = normalize_symbol(str(row["symbol"]))
        if pair is None:
            continue

        ratio = float(row["conviction"])
        score = float(row["score"])
        if ratio < min_conviction or score == 0.0:
            continue

        signals.append({
            "pair": pair,
            "bias": "Bullish" if score > 0 else "Bearish",
            "bar_time": release_date,
            "conviction": _conviction_label(ratio),
            "conviction_ratio": round(ratio, 3),
            "strength_score": round(min(abs(score) * 5.0, 10.0), 1),
            "event": spec.key,
            "rate_channel": round(float(row["rate_channel"]), 3),
            "growth_channel": round(float(row["growth_channel"]), 3),
            "expected_move": round(float(row["expected_move"]), int(row["decimals"])),
            "expected_move_unit": str(row["unit"]),
            "thesis": (
                "{0} {1}: composite z {2:+.2f} under {3} — "
                "rate channel {4:+.2f}, growth channel {5:+.2f}, "
                "conviction {6:.0%}".format(
                    spec.label, release_date.isoformat(), composite, regime,
                    float(row["rate_channel"]), float(row["growth_channel"]), ratio)
            ),
        })
    return signals


# --------------------------------------------------------------------------
# The four events
# --------------------------------------------------------------------------

# Shared session shape for the three 08:30 New York releases. "Your window" is
# an absolute SAST clock, not an offset: it is the desk's own session and must
# not slide an hour when the US changes its clocks.
_MORNING_PHASES: Tuple[Phase, ...] = (
    Phase("Pre-release drain", start_min=-20, end_min=0,
          note="Books pulled, spreads widen, depth collapses. Stops sitting in "
               "the book are cheapest to take here."),
    Phase("Algo impulse", start_min=0, end_min=2,
          note="First move is headline-only and often reverses once the detail "
               "is read. Two-sided."),
    Phase("Repricing", start_min=2, end_min=15,
          note="The composite starts to matter. This is where the board is "
               "most likely to be right."),
    Phase("Fade window", start_min=15, end_min=60,
          note="Partial retracement of the impulse is the base case unless the "
               "composite is an outlier."),
    Phase("US cash open", start_min=60, end_min=120,
          note="09:30 New York. Equity flow can overwrite the FX read, "
               "especially for gold."),
    Phase("Your window", start_sast=time(17, 0), end_sast=time(20, 0),
          note="Post-release continuation or drift. You are trading the "
               "aftermath, not the event."),
)

_FOMC_PHASES: Tuple[Phase, ...] = (
    Phase("Pre-decision freeze", start_min=-30, end_min=0,
          note="Liquidity vanishes. Nobody wants inventory into a dot plot."),
    Phase("Statement + dots", start_min=0, end_min=2,
          note="Both land at once, and the algo read is the dots, not the prose."),
    Phase("First repricing", start_min=2, end_min=30,
          note="The curve moves before the equity market decides what it thinks. "
               "This is where the board is most likely to be right."),
    Phase("Presser", start_min=30, end_min=90,
          note="The chair speaks at +30. The most common FOMC pattern on the "
               "tape is the presser reversing the statement move outright — do "
               "not marry the first leg."),
    Phase("Cash close", start_min=90, end_min=120,
          note="16:00 New York. Positioning into the close overwrites the "
               "macro read."),
    Phase("Your window", start_sast=time(20, 0), end_sast=time(23, 0),
          note="Unlike the 08:30 releases you are awake for this one. That is a "
               "reason for smaller size, not larger."),
)


def _exposures(rows: List[Tuple[str, float, float, float, str, int]]) -> Tuple[Exposure, ...]:
    return tuple(Exposure(*r) for r in rows)


EVENTS: Dict[str, EventSpec] = {
    "NFP": EventSpec(
        key="NFP",
        label="Nonfarm Payrolls",
        source_tag="nfp_reaction",
        calendar_key="NFP",
        release_time_ny=time(8, 30),
        note="The month's labour read. Strong payrolls are hawkish AND "
             "growth-positive — the one event on this page where those two "
             "point the same way.",
        components=(
            Component("nfp", "NFP (k)", 65.0, 0.42,
                      help="Headline nonfarm payrolls, thousands.",
                      default=150.0, step=5.0, fmt="%.0f"),
            Component("rev", "Net 2m revision (k)", 60.0, 0.10, delta_only=True,
                      help="Revision to the prior two months; already a surprise.",
                      default=0.0, step=5.0, fmt="%.0f"),
            Component("ur", "U3 (%)", 0.14, 0.22, invert=True,
                      help="Unemployment rate. A LOWER print than forecast is hawkish.",
                      default=4.2, step=0.1, fmt="%.1f"),
            Component("ahe", "AHE m/m (%)", 0.11, 0.26,
                      help="Average hourly earnings, month on month.",
                      default=0.3, step=0.1, fmt="%.1f"),
        ),
        chain=(("jobs", +1), ("spending", +1), ("inflation", +1), ("rates", +1)),
        phases=_MORNING_PHASES,
        exposures=_exposures([
            ("XAUUSD", -1.00, -0.25,   11.0, "USD",       2),
            ("XAGUSD", -1.10,  0.30,    0.30, "USD",      3),
            # DXY's growth beta is negative: risk-on drains the haven bid even
            # as the rate channel pushes the other way. That conflict is the
            # whole point of the conviction score.
            ("DXY",     1.00, -0.30,    0.35, "index pts", 2),
            ("EURUSD", -0.85,  0.10,   45.0, "pips",      0),
            ("GBPUSD", -0.80,  0.15,   42.0, "pips",      0),
            ("USDJPY",  1.10,  0.40,   60.0, "pips",      0),
            ("AUDUSD", -0.90,  0.45,   40.0, "pips",      0),
            ("USDZAR",  1.15, -0.65,    0.14, "ZAR",      3),
            ("US500",  -0.55,  1.00,   28.0, "pts",       1),
            ("NAS100", -0.75,  1.10,  140.0, "pts",       0),
            ("US10Y",   1.00,  0.20,    0.06, "%",        3),
            ("BTCUSD", -0.65,  0.55,  900.0, "USD",       0),
            # Oil is a demand-driven, growth-cyclical commodity, not a
            # monetary/safe-haven one like gold — its growth beta tracks the
            # equity indices' sign (positive here), not XAUUSD's (-0.25), and
            # is the dominant channel: NFP's demand read-through matters far
            # more to crude than the secondary dollar-strength drag from the
            # (small, still-negative) rate channel. A rate beta close in
            # magnitude to growth would cancel it out near-completely under
            # "Balanced" weights — genuinely two-sided is honest for DXY,
            # which has no registry pair to protect; it would silence oil's
            # otherwise-real NFP signal instead.
            ("WTIUSD",  -0.20,  0.90,    0.35, "USD",      2),
        ]),
    ),

    "CPI": EventSpec(
        key="CPI",
        label="US CPI",
        source_tag="cpi_reaction",
        calendar_key="US_CPI",
        release_time_ny=time(8, 30),
        note="The inflation read the curve trades hardest. Core m/m is the "
             "number that reprices the front end; the y/y prints mostly confirm.",
        components=(
            Component("core_mm", "Core m/m (%)", 0.10, 0.45,
                      help="Core CPI month on month — the number that moves the curve.",
                      default=0.30, step=0.01, fmt="%.2f"),
            Component("head_mm", "Headline m/m (%)", 0.12, 0.20,
                      help="Headline CPI month on month.",
                      default=0.30, step=0.01, fmt="%.2f"),
            Component("core_yy", "Core y/y (%)", 0.12, 0.25,
                      help="Core CPI year on year.",
                      default=3.0, step=0.1, fmt="%.1f"),
            Component("head_yy", "Headline y/y (%)", 0.15, 0.10,
                      help="Headline CPI year on year.",
                      default=2.9, step=0.1, fmt="%.1f"),
        ),
        # Not jobs -> spending: CPI enters the chain at prices. And "real
        # income" moves AGAINST a hot print, which is the stagflation node.
        chain=(("prices", +1), ("inflation", +1), ("real income", -1), ("rates", +1)),
        phases=_MORNING_PHASES,
        exposures=_exposures([
            # Every risk asset's growth beta is negative here, unlike NFP: hot
            # prices squeeze real income and tighten policy at the same time.
            ("XAUUSD", -1.15, -0.30,   14.0, "USD",       2),
            ("XAGUSD", -1.20, -0.35,    0.38, "USD",      3),
            ("DXY",     1.10, -0.20,    0.42, "index pts", 2),
            ("EURUSD", -0.95, -0.10,   52.0, "pips",      0),
            ("GBPUSD", -0.90, -0.10,   48.0, "pips",      0),
            ("USDJPY",  1.20, -0.15,   70.0, "pips",      0),
            ("AUDUSD", -1.00, -0.45,   46.0, "pips",      0),
            ("USDZAR",  1.20, -0.55,    0.17, "ZAR",      3),
            ("US500",  -0.85, -0.55,   34.0, "pts",       1),
            ("NAS100", -1.05, -0.65,  175.0, "pts",       0),
            ("US10Y",   1.15,  0.10,    0.08, "%",        3),
            ("BTCUSD", -0.90, -0.40, 1100.0, "USD",       0),
            ("WTIUSD",  -0.85, -0.50,    0.42, "USD",      2),
        ]),
    ),

    "PPI": EventSpec(
        key="PPI",
        label="US PPI",
        source_tag="ppi_reaction",
        calendar_key="US_PPI",
        release_time_ny=time(8, 30),
        note="Producer prices, traded for the PCE read-through rather than in "
             "their own right — roughly 60% of a CPI's move for the same z.",
        components=(
            Component("core_mm", "Core m/m (%)", 0.20, 0.55,
                      help="Core PPI month on month; the PCE-relevant components.",
                      default=0.20, step=0.01, fmt="%.2f"),
            Component("head_mm", "Headline m/m (%)", 0.25, 0.45,
                      help="Headline PPI month on month.",
                      default=0.20, step=0.01, fmt="%.2f"),
        ),
        chain=(("input costs", +1), ("margins", -1), ("consumer prices", +1),
               ("rates", +1)),
        phases=_MORNING_PHASES,
        exposures=_exposures([
            ("XAUUSD", -0.90, -0.25,    8.0, "USD",       2),
            ("XAGUSD", -0.95, -0.30,    0.24, "USD",      3),
            ("DXY",     0.85, -0.15,    0.26, "index pts", 2),
            ("EURUSD", -0.75, -0.10,   32.0, "pips",      0),
            ("GBPUSD", -0.70, -0.10,   30.0, "pips",      0),
            ("USDJPY",  0.95, -0.10,   42.0, "pips",      0),
            ("AUDUSD", -0.80, -0.35,   28.0, "pips",      0),
            ("USDZAR",  0.95, -0.45,    0.10, "ZAR",      3),
            ("US500",  -0.65, -0.40,   20.0, "pts",       1),
            ("NAS100", -0.80, -0.50,  100.0, "pts",       0),
            ("US10Y",   0.90,  0.10,    0.04, "%",        3),
            ("BTCUSD", -0.70, -0.30,  650.0, "USD",       0),
            ("WTIUSD",  -0.65, -0.40,    0.28, "USD",      2),
        ]),
    ),

    "FOMC": EventSpec(
        key="FOMC",
        label="FOMC Decision",
        source_tag="fomc_reaction",
        calendar_key="FOMC",
        release_time_ny=time(14, 0),
        note="The purest rate event on the calendar, and the only one you are "
             "awake for. The surprise is not a statistic minus a forecast — it "
             "is the decision against what was priced, plus where the dots "
             "moved, plus how the presser read.",
        components=(
            # All three are delta_only: there is no "actual vs consensus"
            # scalar for a policy decision, only a distance from what was
            # already in the curve.
            Component("decision_bp", "Decision vs priced (bp)", 12.5, 0.35,
                      delta_only=True,
                      help="Positive = tighter than priced. A full 25bp "
                           "surprise is 2 sigma.",
                      default=0.0, step=5.0, fmt="%.0f"),
            Component("dots_bp", "Dot-plot median shift (bp)", 15.0, 0.35,
                      delta_only=True,
                      help="Current-year median dot, change from the last SEP. "
                           "Positive = higher for longer.",
                      default=0.0, step=5.0, fmt="%.0f"),
            Component("tone", "Statement / presser tone", 1.0, 0.30,
                      delta_only=True,
                      help="-2 clearly dovish to +2 clearly hawkish. This is a "
                           "judgement, not a measurement — it carries the "
                           "smallest weight of the three for that reason.",
                      default=0.0, step=0.5, fmt="%.1f"),
        ),
        chain=(("policy", +1), ("front end", +1), ("real yields", +1),
               ("liquidity", -1)),
        phases=_FOMC_PHASES,
        exposures=_exposures([
            ("XAUUSD", -1.30, -0.20,   18.0, "USD",       2),
            ("XAGUSD", -1.35, -0.30,    0.48, "USD",      3),
            ("DXY",     1.20, -0.15,    0.55, "index pts", 2),
            ("EURUSD", -1.05, -0.10,   65.0, "pips",      0),
            ("GBPUSD", -1.00, -0.10,   60.0, "pips",      0),
            ("USDJPY",  1.30, -0.15,   90.0, "pips",      0),
            ("AUDUSD", -1.10, -0.45,   58.0, "pips",      0),
            ("USDZAR",  1.30, -0.60,    0.22, "ZAR",      3),
            ("US500",  -1.00, -0.60,   45.0, "pts",       1),
            ("NAS100", -1.20, -0.70,  230.0, "pts",       0),
            ("US10Y",   1.30,  0.10,    0.11, "%",        3),
            ("BTCUSD", -1.00, -0.45, 1500.0, "USD",       0),
            ("WTIUSD",  -1.05, -0.70,    0.55, "USD",      2),
        ]),
    ),
}
