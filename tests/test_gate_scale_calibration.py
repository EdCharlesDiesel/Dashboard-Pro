"""One GateConfig must be correct across every price scale it whitelists.

The gate's limits were absolute point counts. `max_stop_points = 5000` is
4.61% of EUR/USD's price, 7.55% of silver's, and 0.113% of gold's - so the
same config passed silver and rejected every realistic gold trade. Because
one of the two whitelisted symbols worked, the config looked calibrated.

Live specs and ATR14 (D1, 20 bars) from the Exness terminal, 2026-09-06.
"""
from dataclasses import replace

import pytest

from src.execution.gate import (
    AccountState, GateConfig, MarketSnapshot, Signal, run_gate,
)

# symbol -> (bid, ask, point, digits, atr14, a realistic stop in price units)
INSTRUMENTS = {
    "XAUUSD": (4430.194, 4430.376, 0.001, 3, 104.37, 20.0),
    "XAGUSD": (66.201, 66.222, 0.001, 3, 2.5539, 1.00),
    "EURUSD": (1.16269, 1.16279, 0.00001, 5, 0.0042, 0.0050),
}


def _snap(sym):
    bid, ask, point, digits, atr, _ = INSTRUMENTS[sym]
    return MarketSnapshot(
        symbol=sym, bid=bid, ask=ask, point=point, digits=digits,
        tick_value=1.0, tick_size=point, volume_min=0.01, volume_step=0.01,
        volume_max=200.0, trade_allowed=True, stops_level_points=0.0, atr=atr)


def _acct():
    return AccountState(balance=10_000, equity=10_000, free_margin=9_000,
                        open_positions=0, enabled=True, dry_run=True)


def _cfg(sym):
    """Default config, but whitelisting the symbol under test.

    The shipped whitelist is metals-only. Without this, every EUR/USD case
    would carry a whitelist block, and a test asserting "no stop reason" would
    pass while the signal was in fact rejected - a guard green for the wrong
    reason.
    """
    return GateConfig(symbol_whitelist=(sym,))


@pytest.mark.parametrize("sym", sorted(INSTRUMENTS))
def test_a_realistic_trade_passes_on_every_scale(sym):
    """Entry at market, an ordinary stop, 2R target - must not be blocked
    for any reason involving stop size, spread or distance from market."""
    snap = _snap(sym)
    stop_dist = INSTRUMENTS[sym][5]
    sig = Signal(signal_id="s", symbol=sym, direction="BUY",
                 entry=snap.ask, stop=snap.ask - stop_dist,
                 tp1=snap.ask + 2 * stop_dist)
    res = run_gate(sig, snap, _acct(), _cfg(sym))
    scale_reasons = [r for r in res.reasons
                     if any(k in r for k in ("stop", "spread", "from market"))]
    assert not scale_reasons, f"{sym}: {scale_reasons}"


@pytest.mark.parametrize("sym", sorted(INSTRUMENTS))
def test_an_absurd_stop_is_blocked_on_every_scale(sym):
    """4x ATR is past the 3x ceiling regardless of instrument."""
    snap = _snap(sym)
    sig = Signal(signal_id="s", symbol=sym, direction="BUY",
                 entry=snap.ask, stop=snap.ask - 4 * snap.atr,
                 tp1=snap.ask + 8 * snap.atr)
    res = run_gate(sig, snap, _acct(), _cfg(sym))
    assert any("ATR" in r for r in res.reasons), f"{sym}: {res.reasons}"


def test_the_limits_carry_no_absolute_price_assumption():
    """No primary limit may be an absolute point count.

    This is the regression that would have caught the original bug: a point
    count cannot be correct on two instruments whose price differs 66x while
    sharing a `point`.
    """
    cfg = GateConfig()
    assert cfg.max_stop_atr_mult > 0
    assert cfg.max_entry_deviation_atr_mult > 0
    assert cfg.max_spread_frac_of_stop > 0
    # The absolute forms survive only as fallbacks, and must be loose enough
    # never to fire on a normal metals spread (gold's is 182 points).
    assert cfg.max_spread_points > 1_000


def test_gold_and_silver_are_treated_alike_despite_a_66x_price_gap():
    """The exact asymmetry that hid the bug."""
    results = {}
    for sym in ("XAUUSD", "XAGUSD"):
        snap = _snap(sym)
        stop_dist = INSTRUMENTS[sym][5]
        sig = Signal(signal_id="s", symbol=sym, direction="BUY",
                     entry=snap.ask, stop=snap.ask - stop_dist,
                     tp1=snap.ask + 2 * stop_dist)
        results[sym] = run_gate(sig, snap, _acct(), GateConfig()).ok
    assert results["XAUUSD"] == results["XAGUSD"], results
