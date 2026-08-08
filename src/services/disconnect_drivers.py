"""Macro-driver map for the Disconnect Monitor — which economic driver moves
each tradable instrument, and in which direction.

The Disconnect Monitor's method (rolling-regression residual z-score + event
study + OU half-life) needs a *driver → asset* pair where a real economic link
exists. The original page hard-coded four macro relationships (real yields vs
DXY/Gold/Commodities/Nasdaq); this module generalizes it to **every registry
instrument** so the page can offer an instrument dropdown and fade genuine
forex/metals disconnects, not just gold.

Pure data + helpers — no Streamlit, no network. The page resolves each driver
key to a live series and reuses its existing residual-z machinery.

`expected_sign` is the sign of the correlation between *driver change* and
*asset change* when the relationship is intact (+1 = they move together, -1 =
inversely). The page's regime check flags the link "BROKEN" when the live
rolling correlation no longer matches this sign — fading a broken relationship
is the trap the page exists to warn against.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

# ── driver library ───────────────────────────────────────────────────────────
# key → how to fetch it. "REAL5Y" is special-cased by the page so the sidebar's
# real-yield-series toggle (DFII5 vs the DGS2−T5YIE proxy) still applies.
DRIVERS: Dict[str, Dict[str, str]] = {
    "DXY":    {"source": "YF",   "ident": "DX-Y.NYB", "label": "US Dollar Index (DXY)"},
    "US2Y":   {"source": "FRED", "ident": "DGS2",     "label": "US 2Y Treasury yield"},
    "REAL5Y": {"source": "FRED", "ident": "REAL_YIELD", "label": "US real yield"},
    "GOLD":   {"source": "YF",   "ident": "GC=F",     "label": "Gold"},
    "OIL":    {"source": "YF",   "ident": "CL=F",     "label": "WTI crude oil"},
    "SPX":    {"source": "YF",   "ident": "^GSPC",    "label": "S&P 500 (risk-on/off)"},
}


class Relationship(NamedTuple):
    driver: str          # key into DRIVERS
    expected_sign: int   # +1 driver & asset move together, -1 inversely
    note: str            # one-line economic rationale


# ── instrument → applicable drivers ──────────────────────────────────────────
# Every registry instrument maps to at least one economically-defensible
# driver. Signs are for driver-change vs asset(=pair price)-change:
#   • DXY up  → USD/XXX up (+1), EUR/USD-style XXX/USD down (-1), metals down (-1)
#   • US 2Y up → dollar strength, same pattern as DXY
#   • Gold up → commodity currencies (AUD/NZD) up, ZAR strengthens (USD/ZAR down)
#   • Oil up  → CAD strengthens (USD/CAD down)
#   • S&P up (risk-on) → JPY & CHF weaken, AUD/NZD/ZAR strengthen
#   • Real yield up → metals down (classic discount-rate link)
INSTRUMENT_DRIVERS: Dict[str, List[Relationship]] = {
    # ── USD majors (dollar & US-rate driven) ─────────────────────────────────
    "EUR/USD": [Relationship("DXY", -1, "Dollar up → EUR/USD down."),
                Relationship("US2Y", -1, "US rates up → dollar bid → EUR/USD down.")],
    "GBP/USD": [Relationship("DXY", -1, "Dollar up → GBP/USD down."),
                Relationship("US2Y", -1, "US rates up → dollar bid → GBP/USD down.")],
    "AUD/USD": [Relationship("DXY", -1, "Dollar up → AUD/USD down."),
                Relationship("US2Y", -1, "US rates up → dollar bid → AUD/USD down."),
                Relationship("GOLD", +1, "Commodity currency: gold up → AUD up."),
                Relationship("SPX", +1, "Risk-on → AUD (high-beta) up.")],
    "NZD/USD": [Relationship("DXY", -1, "Dollar up → NZD/USD down."),
                Relationship("US2Y", -1, "US rates up → dollar bid → NZD/USD down."),
                Relationship("GOLD", +1, "Commodity/risk currency: gold up → NZD up."),
                Relationship("SPX", +1, "Risk-on → NZD up.")],
    "USD/JPY": [Relationship("DXY", +1, "Dollar up → USD/JPY up."),
                Relationship("US2Y", +1, "US-JP rate gap widens → USD/JPY up."),
                Relationship("SPX", +1, "Risk-on → safe-haven JPY weakens → up.")],
    "USD/CHF": [Relationship("DXY", +1, "Dollar up → USD/CHF up."),
                Relationship("US2Y", +1, "US rates up → USD/CHF up.")],
    "USD/CAD": [Relationship("DXY", +1, "Dollar up → USD/CAD up."),
                Relationship("US2Y", +1, "US rates up → USD/CAD up."),
                Relationship("OIL", -1, "Petro-currency: oil up → CAD up → USD/CAD down.")],
    "USD/ZAR": [Relationship("DXY", +1, "Dollar up → USD/ZAR up."),
                Relationship("GOLD", -1, "SA gold exporter: gold up → ZAR up → USD/ZAR down."),
                Relationship("SPX", -1, "Risk-on → EM ZAR up → USD/ZAR down.")],
    # ── EUR/GBP crosses (weaker single driver — best-available) ───────────────
    "EUR/GBP": [Relationship("DXY", -1, "Soft link: EUR slightly more dollar-sensitive than GBP.")],
    # ── JPY crosses (risk & rate driven) ─────────────────────────────────────
    "EUR/JPY": [Relationship("SPX", +1, "Risk-on → JPY cross up."),
                Relationship("US2Y", +1, "Global rates up → JPY weak → cross up.")],
    "GBP/JPY": [Relationship("SPX", +1, "Risk barometer: risk-on → GBP/JPY up."),
                Relationship("US2Y", +1, "Global rates up → JPY weak → cross up.")],
    "AUD/JPY": [Relationship("SPX", +1, "Classic risk gauge: risk-on → AUD/JPY up."),
                Relationship("GOLD", +1, "Commodity + risk: gold up → AUD/JPY up.")],
    # ── AUD/CAD crosses ──────────────────────────────────────────────────────
    "EUR/AUD": [Relationship("GOLD", -1, "Gold up → AUD up → EUR/AUD down."),
                Relationship("SPX", -1, "Risk-on → AUD up → EUR/AUD down.")],
    "GBP/AUD": [Relationship("GOLD", -1, "Gold up → AUD up → GBP/AUD down."),
                Relationship("SPX", -1, "Risk-on → AUD up → GBP/AUD down.")],
    "EUR/CAD": [Relationship("OIL", -1, "Oil up → CAD up → EUR/CAD down.")],
    "GBP/CAD": [Relationship("OIL", -1, "Oil up → CAD up → GBP/CAD down.")],
    "EUR/ZAR": [Relationship("GOLD", -1, "Gold up → ZAR up → EUR/ZAR down."),
                Relationship("SPX", -1, "Risk-on → ZAR up → EUR/ZAR down.")],
    "GBP/ZAR": [Relationship("GOLD", -1, "Gold up → ZAR up → GBP/ZAR down."),
                Relationship("SPX", -1, "Risk-on → ZAR up → GBP/ZAR down.")],
    # The rand leg dominates every one of these crosses, so the gold and
    # risk-sentiment links carry through unchanged from the pairs above.
    "CHF/ZAR": [Relationship("GOLD", -1, "Gold up → ZAR up → CHF/ZAR down."),
                # CHF is the funding leg here, so risk-on hits it from both
                # sides: the franc weakens as the rand strengthens.
                Relationship("SPX", -1, "Risk-on → CHF weak & ZAR up → CHF/ZAR down.")],
    "NZD/ZAR": [Relationship("GOLD", -1, "Gold up → ZAR up → NZD/ZAR down."),
                # Both legs are commodity/risk currencies, so the SPX link is
                # genuinely weaker than on the majors — they tend to rise
                # together and partly cancel.
                Relationship("SPX", -1, "Soft link: both legs risk-on, ZAR usually the stronger mover.")],
    "AUD/ZAR": [Relationship("GOLD", -1, "Soft link: gold lifts both AUD and ZAR, ZAR usually more."),
                Relationship("SPX", -1, "Soft link: both legs risk-on; ZAR the more volatile leg.")],
    # Signs are FLIPPED versus every pair above, because the rand is the base
    # here: ZAR/JPY rises when the rand strengthens. Gold up lifts the rand and
    # risk-on sells the yen, so both drivers push the same way for once.
    "ZAR/JPY": [Relationship("GOLD", +1, "SA gold exporter: gold up -> ZAR up -> ZAR/JPY up."),
                Relationship("SPX", +1, "Risk-on -> yen sold & ZAR bid -> ZAR/JPY up.")],
    # ── metals & oil ─────────────────────────────────────────────────────────
    "XAU/USD": [Relationship("REAL5Y", -1, "Classic: real yields up → gold down (broken since ~2022)."),
                Relationship("DXY", -1, "Dollar up → gold down."),
                Relationship("US2Y", -1, "Nominal rates up → gold down.")],
    "XAG/USD": [Relationship("REAL5Y", -1, "Real yields up → silver down."),
                Relationship("DXY", -1, "Dollar up → silver down."),
                Relationship("GOLD", +1, "Silver tracks gold.")],
    "XPT/USD": [Relationship("DXY", -1, "Dollar up → platinum down."),
                Relationship("GOLD", +1, "Precious-metal complex: tracks gold."),
                Relationship("SPX", +1, "Industrial demand: risk-on → platinum up.")],
    "WTI/USD": [Relationship("DXY", -1, "Dollar up → oil down."),
                Relationship("SPX", +1, "Risk-on / demand: equities up → oil up.")],
}


def drivers_for(instrument: str) -> List[Relationship]:
    """Applicable macro relationships for a registry instrument (empty if none)."""
    return INSTRUMENT_DRIVERS.get(instrument, [])


def covered_instruments() -> List[str]:
    """Registry instruments that have at least one mapped driver."""
    return [k for k, v in INSTRUMENT_DRIVERS.items() if v]


def driver_label(key: str) -> str:
    d = DRIVERS.get(key)
    return d["label"] if d else key
