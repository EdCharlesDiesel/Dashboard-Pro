"""Shared usage logging for the interactive tool pages (R:R Calculator,
Account Risk, Correlations, News Filter, Stop Structure).

These pages are calculators/views, not signal generators — they don't belong
in ``trade_setups`` (reserved for directional pair+bias reads) or in the
``app_state`` key/value store (reserved for durable single-value state like
the account balance). Instead every meaningful computation on these pages is
appended as one row to the ``tool_usage_log`` table via
:func:`log_tool_usage`, giving them the same "everything is in Postgres"
coverage the signal pages already have.

Same resilience contract as ``src/services/signal_store.py``: resolves the DB
target lazily (the sidebar/secrets target the pages' market data already
uses), never raises, and is a silent no-op when the DB isn't configured — a
logging failure must never break the tool it's logging.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("ForexDashboard")


def log_tool_usage(tool: str, payload: Dict[str, Any]) -> bool:
    """Insert one row for ``tool`` (e.g. ``'rr_calculator'``) with an
    arbitrary JSON-able ``payload`` of inputs/outputs. Returns whether it was
    saved; never raises."""
    try:
        from src.db.market_cache import _resolve_cfg
        cfg = _resolve_cfg()
    except Exception as exc:
        logger.warning("Tool usage logging skipped (%s): %s", tool, exc)
        return False
    if cfg is None:
        return False
    try:
        from src.db.cache import pooled_repository
        pooled_repository(cfg).log_tool_usage(tool, payload)
        return True
    except Exception as exc:
        logger.warning("Tool usage log failed (%s): %s", tool, exc)
        return False
