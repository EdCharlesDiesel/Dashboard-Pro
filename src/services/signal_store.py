"""Shared trade-signal persistence — save any page's signals to ``trade_setups``.

Generalizes the Market Overview auto-save so *every* page persists its signals
through a single call::

    from src.services.signal_store import persist_signals
    persist_signals("setup_ranker", signals)   # signals: list of normalized dicts

Each signal dict follows the loose shape :func:`src.core.signals.signal_to_setup_row`
consumes (``pair`` + ``bias`` required; ``entry``/``stop_loss``/``take_profit_*``/
``strength_score``/``conviction``/``thesis``/… optional). Analysis pages with no
price target still persist a directional read — the optional fields just stay NULL.

The service owns all the cross-cutting concerns so pages don't repeat them:

- **Dedupe** — a per-source :class:`NotifyCache` (JSON file) records which signals
  were already saved, keyed by ``dedupe_key`` (default: pair+bias+rounded-entry).
  Survives reruns and restarts; each unique signal is saved once.
- **DB target** — resolved via :func:`src.db.market_cache._resolve_cfg` (the same
  sidebar/secrets target the pages' market data already uses).
- **Source tag** — every row is written with ``source=<page>`` so signals are
  attributable and filterable per page.
- **Write** — pooled write straight to Postgres (no caching layer). The
  :func:`clear_read_caches` call is a retained no-op (reads are uncached now).
- **Graceful no-op** — if the DB isn't configured (or anything fails), nothing is
  saved and the dedupe ledger is left untouched (so a later configured run still
  saves). Persistence must never break a page.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Optional

from src.core.signals import signal_to_setup_row
from src.db.cache import clear_read_caches, pooled_repository
from src.db.market_cache import _resolve_cfg
from src.services.alert_service import NotifyCache
from src.services.session_service import SessionService

logger = logging.getLogger("ForexDashboard")


def default_dedupe_key(sig: Dict) -> str:
    """pair + bias + rounded entry — same dedupe Market Overview alerts use, so a
    fresh alert fires (and saves) again only when price has moved to a new level."""
    pair = sig.get("pair", "?")
    bias = sig.get("bias", "?")
    entry = sig.get("entry")
    try:
        price_key = f"{float(entry):.4f}" if entry else "na"
    except (TypeError, ValueError):
        price_key = "na"
    return f"{pair}_{bias}_{price_key}"


def persist_signals(
    source: str,
    signals: Iterable[Dict],
    *,
    session: Optional[str] = None,
    dedupe_key: Callable[[Dict], str] = default_dedupe_key,
    namespace: Optional[str] = None,
) -> int:
    """Persist a page's signals to ``trade_setups``. Returns the number saved.

    ``source`` tags the rows and names the dedupe ledger. ``signals`` are
    normalized signal dicts (see module docstring). ``session`` overrides the
    auto-detected trading session; ``dedupe_key`` and ``namespace`` let a page
    customize dedupe granularity. Never raises — failures are logged and return 0.
    """
    items: List[Dict] = [s for s in signals if s]
    if not items:
        return 0

    # Resolve the DB target FIRST. With no DB we return without consuming the
    # dedupe ledger, so the same signals still save once a DB is configured.
    try:
        cfg = _resolve_cfg()
    except Exception as exc:  # pragma: no cover - defensive; _resolve_cfg already guards
        logger.warning("Signal persistence skipped (%s): %s", source, exc)
        return 0
    if cfg is None:
        return 0

    cache = NotifyCache(namespace or f"signals_{source}")
    seen = cache.load()
    pending = [(s, dedupe_key(s)) for s in items]
    fresh = [(s, k) for s, k in pending if k not in seen]
    if not fresh:
        return 0

    try:
        sess = session if session is not None else SessionService.current().get("window", "")
        repo = pooled_repository(cfg)
        saved_keys: List[str] = []
        for sig, key in fresh:
            try:
                repo.save_setup(signal_to_setup_row(sig, sess), source=source)
                saved_keys.append(key)  # mark seen only after a successful write
            except Exception as exc:
                logger.warning("Signal save failed (%s/%s): %s", source, sig.get("pair"), exc)
        if saved_keys:
            # Atomic merge (not overwrite) so a concurrent run's saved keys aren't
            # lost — otherwise the same signal could be saved twice.
            cache.add(saved_keys)
            clear_read_caches()  # no-op kept for API compat; reads are uncached
            logger.info("Saved %d signal(s) from %s", len(saved_keys), source)
        return len(saved_keys)
    except Exception as exc:
        logger.warning("Signal persistence skipped (%s): %s", source, exc)
        return 0
