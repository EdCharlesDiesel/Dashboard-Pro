"""Instrument Predictor's composite-signal aggregator — pure logic, no I/O.

Combines independently-computed directional reads (Setup Score, Trend Signal,
Currency Strength, COT Composite, ...) into one weighted composite score,
label, and confidence. Each component is normalized to [-1, +1] (positive =
bullish) before weighting; a component that couldn't be computed for this
instrument (e.g. COT has no currency mapping, or currency strength doesn't
apply to a commodity) is simply left out and the remaining weights are
renormalized so they still sum to 1 — a missing component never silently
drags the composite toward zero.

This is a heuristic combination of the dashboard's own existing scorers, not
a new statistically validated model — same caveat as
``cot_composite_trade_signal``: backtest before sizing conviction on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Component:
    """One scorer's normalized vote."""
    name: str
    score: float          # normalized [-1, +1], + = bullish
    weight: float          # base weight before renormalization; 0 = unavailable
    detail: str = ""       # human-readable "why" for this component


@dataclass(frozen=True)
class Prediction:
    composite: float                  # weighted average, [-1, +1]
    label: str                        # STRONG_SELL/SELL/NEUTRAL/BUY/STRONG_BUY
    confidence: str                   # Low/Medium/High
    agreement: float                  # fraction of nonzero components agreeing in sign
    components: List[Component]
    vol_regime: Optional[str] = None  # GARCH regime modifier, if computed


def label_for(composite: float) -> str:
    """Map a composite score in [-1, +1] to a discrete directional label."""
    if composite >= 0.5:
        return "STRONG_BUY"
    if composite >= 0.15:
        return "BUY"
    if composite <= -0.5:
        return "STRONG_SELL"
    if composite <= -0.15:
        return "SELL"
    return "NEUTRAL"


def aggregate(components: List[Component], vol_regime: Optional[str] = None) -> Prediction:
    """Weighted-average the available components into one composite read.

    ``vol_regime`` (from GARCH: "high"/"low"/"normal") is a risk modifier, not
    a vote — a "high" regime caps confidence at Medium even when the vote
    count alone would say High, since directional reads are less reliable in
    a choppy/high-vol regime.
    """
    usable = [c for c in components if c.weight > 0]
    if not usable:
        return Prediction(0.0, "NEUTRAL", "Low", 0.0, [], vol_regime)

    total_weight = sum(c.weight for c in usable)
    composite = sum(c.score * c.weight for c in usable) / total_weight
    composite = max(-1.0, min(1.0, composite))

    nonzero = [c for c in usable if abs(c.score) > 1e-9]
    if nonzero:
        agree_sign = 1 if composite >= 0 else -1
        same_sign = sum(1 for c in nonzero if (1 if c.score > 0 else -1) == agree_sign)
        agreement = same_sign / len(nonzero)
    else:
        agreement = 0.0

    label = label_for(composite)

    if label == "NEUTRAL":
        confidence = "Low"
    elif agreement >= 0.75 and abs(composite) >= 0.4:
        confidence = "High"
    elif agreement >= 0.5 and abs(composite) >= 0.15:
        confidence = "Medium"
    else:
        confidence = "Low"

    if vol_regime == "high" and confidence == "High":
        confidence = "Medium"

    return Prediction(round(composite, 3), label, confidence, round(agreement, 2),
                      usable, vol_regime)
