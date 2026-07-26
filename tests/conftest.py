"""Shared pytest fixtures for the src/ logic suite.

These build deterministic OHLC frames so indicator/scoring tests never touch
yfinance or a Streamlit runtime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── slow/network test gating ──────────────────────────────────────────────────
# Page smoke tests (AppTest) hit live yfinance and take 5–60s each, so they are
# skipped unless --runslow is passed. They stay out of the default fast suite and
# don't affect the coverage gate.
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run slow tests that hit the network (live yfinance page smoke tests)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: slow test that hits the network (live yfinance)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow (hits live yfinance)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ── live-DB isolation ─────────────────────────────────────────────────────────
# The fast suite must never reach a real Postgres — but every DB-optional code
# path (NotifyCache mirroring, tool_log, signal_store, market_cache) resolves
# its target from .streamlit/secrets.toml, and a developer machine may carry
# LIVE credentials there. Force the secrets-based resolution to "unconfigured"
# (empty password → _resolve_cfg() → None) for every test; tests that need a
# DBConfig construct one explicitly or monkeypatch their own, which overrides
# this autouse patch. Slow AppTest smoke runs (--runslow) are exempt — they
# exercise the real app on purpose.
@pytest.fixture(autouse=True)
def _no_live_db(request, monkeypatch):
    if "slow" in request.keywords:
        yield
        return
    from src.core import secrets as _secrets
    monkeypatch.setattr(_secrets, "db_config", lambda: {
        "host": "localhost", "port": 5432, "dbname": "trading",
        "user": "postgres", "password": "",
    })
    # Neutralize the worker's precomputed board for the same reason: a developer
    # machine may carry a real worker_board.json (from a live scan), and its
    # JSON fallback would otherwise shadow the live house-view compute that
    # bias_service tests assert on. Tests that exercise the board build one
    # explicitly and call the pure helpers directly, so this doesn't hide them.
    from src.services import precomputed as _pc
    monkeypatch.setattr(_pc, "read_board", lambda: None)
    yield


def _ohlc_from_close(close: "list[float] | np.ndarray") -> pd.DataFrame:
    """Wrap a close-price array in a minimal OHLCV frame.

    High/Low are nudged off Close by a small band so true-range based
    indicators (ATR/ADX) have non-degenerate inputs.
    """
    close = np.asarray(close, dtype=float)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(len(close), 1000.0),
        }
    )


@pytest.fixture
def ohlc_from_close():
    """Factory fixture: callable that turns a close array into an OHLCV frame."""
    return _ohlc_from_close


@pytest.fixture
def uptrend_frame():
    """A long, steadily rising series — enough bars for EMA200-based logic."""
    return _ohlc_from_close(np.linspace(100.0, 200.0, 260))


@pytest.fixture
def rising_zigzag():
    """Zigzag with both swing highs and swing lows trending up (BULLISH)."""
    hi, lo, base = [], [], 0.0
    for _ in range(5):
        hi += [base + 1, base + 3, base + 5, base + 3, base + 1]
        lo += [base + 0.5, base + 1.5, base + 2.5, base + 1.5, base + 0.5]
        base += 2
    return pd.DataFrame({"High": hi, "Low": lo, "Close": hi, "Open": hi,
                         "Volume": [1000.0] * len(hi)})


@pytest.fixture
def falling_zigzag():
    """Zigzag with both swing highs and swing lows trending down (BEARISH)."""
    hi, lo, base = [], [], 20.0
    for _ in range(5):
        hi += [base + 1, base + 3, base + 5, base + 3, base + 1]
        lo += [base + 0.5, base + 1.5, base + 2.5, base + 1.5, base + 0.5]
        base -= 2
    return pd.DataFrame({"High": hi, "Low": lo, "Close": hi, "Open": hi,
                         "Volume": [1000.0] * len(hi)})
