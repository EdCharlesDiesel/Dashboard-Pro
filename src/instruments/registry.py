"""Instrument registry — single source of truth.

Consolidates the INSTRUMENTS / CORR_GROUPS / TREND_TIMEFRAMES tables that were
previously duplicated across every page. Logic and values are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Instrument:
    """Single tradable asset descriptor."""
    name: str
    ticker: str
    pip: float
    pip_size: float
    corr: str

    @property
    def display_name(self) -> str:
        """Display form of the instrument name.

        Names are plain symbols (metals use XAU/USD, XAG/USD, XPT/USD), so
        this is the name itself; kept as a property because callers and the
        DB save path build on it.
        """
        return self.name

    @property
    def is_commodity(self) -> bool:
        return self.name in TREND_COMMODITIES


_INSTRUMENT_TABLE: Tuple[Tuple[str, str, float, float, str], ...] = (
    # name, ticker, pip, pip_size, corr
    ("EUR/USD", "EURUSD=X", 10.0,  0.0001, "DXY ↑ = bearish"),
    ("GBP/USD", "GBPUSD=X", 10.0,  0.0001, "DXY ↑ = bearish"),
    ("AUD/USD", "AUDUSD=X", 10.0,  0.0001, "Gold ↑ = bullish"),
    ("NZD/USD", "NZDUSD=X", 10.0,  0.0001, "AUD/USD alignment"),
    ("USD/JPY", "USDJPY=X",  9.09, 0.01,   "US10Y yields aligned"),
    ("USD/CHF", "USDCHF=X", 10.8,  0.0001, "EUR/USD inverse"),
    ("USD/CAD", "USDCAD=X",  7.4,  0.0001, "Oil inverse"),
    ("EUR/GBP", "EURGBP=X", 12.5,  0.0001, "EUR/USD vs GBP/USD"),
    ("EUR/JPY", "EURJPY=X",  9.09, 0.01,   "Risk-on sentiment"),
    ("GBP/JPY", "GBPJPY=X",  9.09, 0.01,   "Volatility proxy"),
    ("AUD/JPY", "AUDJPY=X",  9.09, 0.01,   "Risk appetite gauge"),
    ("EUR/AUD", "EURAUD=X",  6.3,  0.0001, "EUR/USD vs AUD/USD"),
    ("GBP/AUD", "GBPAUD=X",  6.3,  0.0001, "GBP/USD vs AUD/USD"),
    ("EUR/CAD", "EURCAD=X",  7.4,  0.0001, "EUR/USD vs Oil"),
    ("GBP/CAD", "GBPCAD=X",  7.4,  0.0001, "GBP/USD vs Oil"),
    # Every ZAR cross is quoted in rand, so one 0.0001 move is worth the same
    # in USD on all six: 10 ZAR per standard lot, /USDZAR. Measured 0.62 against
    # the live terminal on 2026-08-07 (USD/ZAR ~16.17). The long-standing 0.55
    # was set when the rand was weaker and understates risk by ~13% — position
    # size is derived from pip value, so a stale figure sizes every ZAR trade
    # too large. Re-measure if USD/ZAR moves far from ~16.
    ("USD/ZAR", "USDZAR=X",  0.62, 0.0001, "Gold & risk sentiment"),
    ("EUR/ZAR", "EURZAR=X",  0.62, 0.0001, "EUR/USD + Gold"),
    ("GBP/ZAR", "GBPZAR=X",  0.62, 0.0001, "GBP/USD + Gold"),
    ("CHF/ZAR", "CHFZAR=X",  0.62, 0.0001, "USD/CHF + Gold"),
    ("NZD/ZAR", "NZDZAR=X",  0.62, 0.0001, "NZD/USD + risk sentiment"),
    ("AUD/ZAR", "AUDZAR=X",  0.62, 0.0001, "AUD/USD + Gold"),
    # ZAR/JPY is the odd one out and none of the figures above apply to it:
    # the rand is the BASE, not the quote, so the pair rises when ZAR
    # STRENGTHENS — the exact inverse of the five entries above. It is quoted in
    # yen, so it takes the JPY convention: pip_size 0.01 and roughly $6.34 a pip
    # per lot (measured 2026-08-07), about 10x the ZAR-quote crosses. Sizing it
    # off the 0.62 above would put on ten times the intended risk.
    ("ZAR/JPY", "ZARJPY=X",  6.34, 0.01,   "Risk appetite + SA gold"),
    ("XAU/USD", "GC=F",  10.0, 0.10, "DXY inverse, VIX"),
    ("XAG/USD", "SI=F",  10.0, 0.01, "Gold correlation"),
    ("XPT/USD", "PL=F",  10.0, 0.10, "Industrial demand"),
    ("WTI/USD", "CL=F",  10.0, 0.01, "USD/CAD inverse, risk sentiment"),
)


TREND_COMMODITIES: FrozenSet[str] = frozenset({"XAU/USD", "XAG/USD", "XPT/USD", "WTI/USD"})

TREND_TIMEFRAMES: Dict[str, Tuple[str, str, Optional[str]]] = {
    "1 Hour":  ("60m", "59d", "1h"),
    "4 Hours": ("60m", "59d", "4h"),
    "Daily":   ("1d",  "2y",  None),
    "Weekly":  ("1d",  "2y",  "1W"),
}

CORR_GROUPS: Dict[str, FrozenSet[str]] = {
    "USD vs majors (same dir = stacked USD risk)":  frozenset({"EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD"}),
    "USD-base pairs (same dir = stacked USD risk)": frozenset({"USD/JPY", "USD/CHF", "USD/CAD"}),
    # ZAR/JPY belongs here and NOT in the ZAR group below. Long ZAR/JPY is short
    # yen, exactly like long USD/JPY, so the same-direction rule holds.
    "JPY crosses (same dir = stacked JPY risk)":    frozenset({"USD/JPY", "EUR/JPY", "GBP/JPY",
                                                               "AUD/JPY", "ZAR/JPY"}),
    "Gold / Silver (highly correlated)":            frozenset({"XAU/USD", "XAG/USD"}),
    # All six move together — measured 2026-08-07: every one of them topped on
    # 24-31 Jul and fell 1.9-4.7% into the same rand rally. Shorting the lot is
    # one macro bet at 6x size, not a diversified book, so the exposure guard
    # must see them as a single group.
    #
    # ZAR/JPY is deliberately EXCLUDED. These groups express "same direction =
    # stacked risk", and ZAR/JPY quotes the rand the other way up: long ZAR/JPY
    # is long rand, long USD/ZAR is short rand. Including it would invert every
    # verdict — flagging a genuine hedge as a stack, and missing the real stack
    # (long ZAR/JPY + short USD/ZAR) because the directions differ. The group
    # structure has no way to express an inverse member; see the note in
    # CorrelationService if that ever needs fixing properly.
    "ZAR pairs (same dir = stacked ZAR risk)":      frozenset({"USD/ZAR", "EUR/ZAR", "GBP/ZAR",
                                                               "CHF/ZAR", "NZD/ZAR", "AUD/ZAR"}),
}

TYPICAL_SPREADS: Dict[str, float] = {
    "EUR/USD": 1.2, "GBP/USD": 1.5, "AUD/USD": 1.5, "NZD/USD": 2.0,
    "USD/JPY": 1.2, "USD/CHF": 2.0, "USD/CAD": 2.0, "EUR/GBP": 1.5,
    "EUR/JPY": 2.0, "GBP/JPY": 3.0, "AUD/JPY": 2.5, "EUR/AUD": 3.0,
    "GBP/AUD": 3.5, "EUR/CAD": 3.0, "GBP/CAD": 3.5, "USD/ZAR": 80.0,
    # ZAR crosses are exotics and quote very wide. These are the historical
    # house figures; the new three are scaled to the same basis from live
    # broker spreads measured 2026-08-07 (CHF 964, NZD 668, AUD 510 pips —
    # roughly 9.5x these values on that day). Worth noting the whole ZAR block
    # looks understated against a live Exness feed, which flatters the
    # Spread/ATR quality gate on every rand pair.
    "EUR/ZAR": 90.0, "GBP/ZAR": 90.0, "CHF/ZAR": 100.0,
    "NZD/ZAR": 70.0, "AUD/ZAR": 55.0,
    # ZAR/JPY quotes far tighter than the ZAR-quote crosses - 0.3 pips live
    # on 2026-08-07 against their 510-1,280. It trades like a JPY cross.
    "ZAR/JPY": 3.0, "XAU/USD": 3.0, "XAG/USD": 5.0,
    "XPT/USD": 8.0, "WTI/USD": 4.0,
}


class InstrumentRegistry:
    """Object-oriented access to the instrument universe.

    Backwards-compatibility: dict-style lookup `registry[name]["ticker"]`
    still works because each instrument exposes a dict-like view.
    """

    def __init__(self, table: Iterable[Tuple[str, str, float, float, str]] = _INSTRUMENT_TABLE) -> None:
        self._by_name: Dict[str, Instrument] = {
            name: Instrument(name, ticker, pip, pip_size, corr)
            for name, ticker, pip, pip_size, corr in table
        }

    # ── dict-style access for legacy callers ────────────────────────────────
    def __getitem__(self, name: str) -> "_InstrumentView":
        return _InstrumentView(self._by_name[name])

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def keys(self) -> List[str]:
        return list(self._by_name.keys())

    def values(self) -> List[Instrument]:
        return list(self._by_name.values())

    def items(self) -> List[Tuple[str, Instrument]]:
        return list(self._by_name.items())

    def get(self, name: str, default=None) -> Optional[Instrument]:
        return self._by_name.get(name, default)

    # ── domain helpers ──────────────────────────────────────────────────────
    def forex_pairs(self) -> List[str]:
        return [n for n in self._by_name if n not in TREND_COMMODITIES]

    def commodities(self) -> List[str]:
        return [n for n in self._by_name if n in TREND_COMMODITIES]

    def correlated_groups(self, name: str) -> List[Tuple[str, FrozenSet[str]]]:
        return [(g, members) for g, members in CORR_GROUPS.items() if name in members]


class _InstrumentView(Mapping[str, object]):
    """Read-only dict view over an Instrument for legacy `inst['ticker']` access."""

    __slots__ = ("_inst",)

    def __init__(self, inst: Instrument) -> None:
        self._inst = inst

    def __getitem__(self, key: str):
        if key == "ticker":   return self._inst.ticker
        if key == "pip":      return self._inst.pip
        if key == "pip_size": return self._inst.pip_size
        if key == "corr":     return self._inst.corr
        if key == "name":     return self._inst.name
        raise KeyError(key)

    def __iter__(self):
        return iter(("ticker", "pip", "pip_size", "corr", "name"))

    def __len__(self) -> int:
        return 5

    @property
    def instrument(self) -> Instrument:
        return self._inst


# Module-level singleton — drop-in replacement for the old INSTRUMENTS dict
INSTRUMENTS: InstrumentRegistry = InstrumentRegistry()
