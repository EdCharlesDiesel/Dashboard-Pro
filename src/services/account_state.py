"""Tiny persistent store for the live account balance.

The Trade Journal reads the balance straight off the latest MT4 statement and
writes it here; the Setup Ranker (and any other page that sizes positions) reads
it back. A small JSON file rather than session state so the value survives a
restart and is visible to a page even if the Journal wasn't opened this session.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict

_PATH = os.path.join(os.getcwd(), "account_state.json")


def get() -> Dict:
    """Return {balance, source, updated_at} or {} if nothing has been stored."""
    try:
        if os.path.exists(_PATH):
            with open(_PATH) as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def get_balance(default: float = 10000.0) -> float:
    try:
        return float(get().get("balance", default))
    except (TypeError, ValueError):
        return default


def set_balance(balance: float, source: str = "manual") -> None:
    payload = {
        "balance": float(balance),
        "source": source,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(_PATH, "w") as fh:
            json.dump(payload, fh)
    except Exception:
        pass
