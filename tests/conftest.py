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
    config.addinivalue_line(
        "markers",
        "live_secrets: reads this machine's real secrets.toml; exempt from "
        "the _no_live_db stub and skipped where there is none"
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
    if "slow" in request.keywords or "live_secrets" in request.keywords:
        yield
        return
    from src.core import secrets as _secrets

    # Stub `_section`, not `db_config`. Replacing the resolver itself also
    # defeated the two tests that exist to verify the resolver - they ended up
    # asserting against this fixture and failed for a whole session
    # (2026-08-20 to 2026-08-22) looking like a real resolution bug.
    #
    # Emptying the [database] section gives the identical guarantee: no
    # secrets.toml -> env vars -> localhost:5432/trading with an empty password
    # -> _resolve_cfg() returns None, i.e. unconfigured. It also models the
    # container exactly, where .dockerignore keeps secrets.toml out of the image.
    _real_section = _secrets._section
    # Telegram is neutralised the same way, and for the same reason. Since the
    # scanner gained a Telegram channel (1.10.28) the confluence alert path
    # calls send_telegram_message for real, so the fast suite was making live
    # API calls - and on a machine with working credentials it delivered an
    # actual alert to the owner's phone during `pytest`.
    #
    # Stubbed at the *config*, not the function: stubbing
    # send_telegram_message itself would make tests/test_secrets_telegram.py
    # assert against this fixture instead of the real sender - exactly the
    # mistake that broke the two data_backbone tests. Emptying the config makes
    # the real function return "not configured" before it touches the network,
    # and a test that wants the sender simply patches telegram_config itself.
    monkeypatch.setattr(_secrets, "telegram_config",
                        lambda: {"bot_token": "", "chat_id": ""})
    monkeypatch.setattr(
        _secrets, "_section",
        lambda name: {} if name == "database" else _real_section(name))
    # Emptying [database] is only half the guarantee. `db_config()` falls back to
    # DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD and DATABASE_URL, and in the
    # containers those ARE set — so the comment above was exactly backwards: the
    # secrets stub does not "model the container", it is defeated there.
    #
    # Measured 2026-09-06: 18 tests failed inside the container that pass on the
    # host, with shapes like `assert {'a', 'b'} == set()` — a NotifyCache that
    # found real rows because its Postgres mirror was live. They read as
    # breakage; they were a real database leaking into tests that assume none.
    #
    # Removing the vars, not blanking them: an empty DB_HOST would still be a
    # "set" value for anything reading os.environ directly, whereas absent is
    # what a developer machine actually looks like.
    for _var in ("DATABASE_URL", "DB_HOST", "DB_PORT", "DB_NAME",
                 "DB_USER", "DB_PASSWORD"):
        monkeypatch.delenv(_var, raising=False)
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
