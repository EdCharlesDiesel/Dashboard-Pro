"""Broker symbol → registry instrument mapping.

Extracted from ``mt4_import`` because it is not statement-specific: the live
MetaTrader 5 link (`src/services/mt5_link.py`) needs the identical mapping, and
two copies of "which broker suffix means XAU/USD" is exactly the kind of
duplication that drifts. ``mt4_import`` re-exports these names, so existing
callers (``pages/trade-journal.py``) keep working unchanged.

Brokers decorate the same instrument a dozen ways — ``EURUSD``, ``EURUSDi``,
``EURUSD.r``, ``EURUSDm``, ``GOLD``, ``XAUUSD.raw``. The rule is: strip to
letters, take the first six, look that up; then fall back to substring aliases
for the metals, which brokers name in words as often as in symbols.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.instruments.registry import INSTRUMENTS

# Display name (e.g. "EUR/USD") keyed by its compact form ("EURUSD").
_SYMBOL_KEYS: Dict[str, str] = {name.replace("/", ""): name for name in INSTRUMENTS}
_ALIASES = {"GOLD": "XAU/USD", "SILVER": "XAG/USD", "PLATINUM": "XPT/USD",
            # Energy: brokers name crude in words far more often than by the
            # XTI/XBR symbols. Without these a real oil trade maps to None and
            # the MT5 importer drops it as "unmapped" — silent data loss in the
            # journal rather than a visible error. Seen live as Exness "USOIL".
            "USOIL": "WTI/USD", "WTI": "WTI/USD", "XTIUSD": "WTI/USD",
            "CRUDE": "WTI/USD"}


def normalize_symbol(item: str) -> Optional[str]:
    """Map a broker instrument string ('EURUSD', 'XAUUSD.r', 'GOLD', 'EURUSDm')
    to an app instrument name, or None if it can't be matched automatically."""
    if not item:
        return None
    u = str(item).upper()
    core = re.sub(r"[^A-Z]", "", u)[:6]
    if core in _SYMBOL_KEYS:
        return _SYMBOL_KEYS[core]
    for alias, name in _ALIASES.items():
        if alias in u:
            return name
    return None


def build_symbol_map(items: List[str]) -> Dict[str, Optional[str]]:
    """Auto-map every distinct raw symbol seen in a statement / terminal."""
    return {it: normalize_symbol(it) for it in sorted(set(items))}
