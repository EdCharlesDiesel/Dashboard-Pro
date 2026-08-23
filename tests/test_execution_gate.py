"""The pre-trade gate — the last check between a malformed signal and a real order.

`src/execution/gate.py` is deliberately pure: no terminal, no database, no
network. That is what makes exhaustive testing possible, and it is why this
module stays inside the coverage gate while the queue and the executor beside it
are omitted.

**Every refusal test asserts the specific reason**, never bare `not ok`. The
first draft of this file did the latter and the baseline "clean" signal was
rejected as *synthetic* — round levels, integer R — which meant every refusal
test would have passed for that one reason regardless of the condition it named.
A test that cannot distinguish why it passed is not testing what it claims.
"""
from __future__ import annotations

import pytest

from src.execution.gate import (
    AccountState,
    GateConfig,
    MarketSnapshot,
    Signal,
    classify_order_type,
    compute_lots,
    run_gate,
)


def _snap(**over) -> MarketSnapshot:
    base = dict(symbol="XAUUSD", bid=2400.11, ask=2400.39, point=0.01, digits=2,
                tick_value=1.0, tick_size=0.01, volume_min=0.01,
                volume_step=0.01, volume_max=50.0, trade_allowed=True,
                stops_level_points=0.0, margin_per_lot=0.0)
    base.update(over)
    return MarketSnapshot(**base)


def _acct(**over) -> AccountState:
    base = dict(balance=10_000.0, equity=10_000.0, free_margin=9_000.0,
                open_positions=0, open_symbols=(), daily_loss_r=0.0,
                enabled=True, dry_run=False)
    base.update(over)
    return AccountState(**base)


def _sig(**over) -> Signal:
    # Deliberately *not* a round grid: measured-looking levels with a
    # non-integer R, so the synthetic-levels check does not fire and mask the
    # condition each test is actually about.
    base = dict(signal_id="s1", symbol="XAUUSD", direction="buy",
                entry=2400.37, stop=2391.84, tp1=2417.05)
    base.update(over)
    return Signal(**base)


def _why(res) -> str:
    return " | ".join(res.reasons).lower()


class TestTheBaselinePasses:
    def test_a_measured_signal_is_allowed(self):
        # Without this, every refusal below proves nothing.
        res = run_gate(_sig(), _snap(), _acct(), GateConfig())
        assert res.ok, f"baseline should pass, blocked by: {res.reasons}"


class TestTheGateRefuses:
    def test_the_kill_switch_stops_everything(self):
        res = run_gate(_sig(), _snap(), _acct(enabled=False), GateConfig())
        assert not res.ok and "kill switch" in _why(res)

    def test_a_symbol_off_the_whitelist(self):
        res = run_gate(_sig(symbol="EURUSD"), _snap(symbol="EURUSD"),
                       _acct(), GateConfig())
        assert not res.ok and "whitelist" in _why(res)

    def test_a_market_the_broker_has_closed(self):
        res = run_gate(_sig(), _snap(trade_allowed=False), _acct(), GateConfig())
        assert not res.ok and "not allowed" in _why(res)

    def test_a_buy_whose_stop_sits_above_entry(self):
        # An inverted signal, not a tight stop — it would size as negative risk.
        res = run_gate(_sig(direction="buy", entry=2400.37, stop=2409.12),
                       _snap(), _acct(), GateConfig())
        assert not res.ok and "stop at or above entry" in _why(res)

    def test_a_sell_whose_stop_sits_below_entry(self):
        res = run_gate(_sig(direction="sell", entry=2400.37, stop=2391.84,
                            tp1=2383.10), _snap(), _acct(), GateConfig())
        assert not res.ok and "stop at or below entry" in _why(res)

    def test_a_zero_width_stop(self):
        res = run_gate(_sig(stop=2400.37), _snap(), _acct(), GateConfig())
        assert not res.ok and "stop equals entry" in _why(res)

    def test_reward_below_the_configured_minimum(self):
        res = run_gate(_sig(entry=2400.37, stop=2391.84, tp1=2403.11),
                       _snap(), _acct(), GateConfig(min_rr_tp1=1.0))
        assert not res.ok and "r:r" in _why(res)

    def test_a_spread_wider_than_configured(self):
        res = run_gate(_sig(), _snap(bid=2400.11, ask=2410.11), _acct(),
                       GateConfig(max_spread_points=40.0))
        assert not res.ok and "spread" in _why(res)

    def test_too_many_positions_already_open(self):
        acct = _acct(open_positions=3, open_symbols=("XAGUSD",) * 3)
        res = run_gate(_sig(), _snap(), acct, GateConfig(max_concurrent_positions=3))
        assert not res.ok and "positions open" in _why(res)

    def test_a_second_position_in_the_same_symbol(self):
        acct = _acct(open_positions=1, open_symbols=("XAUUSD",))
        res = run_gate(_sig(), _snap(), acct, GateConfig(one_position_per_symbol=True))
        assert not res.ok and "already holding" in _why(res)

    def test_the_daily_loss_limit(self):
        res = run_gate(_sig(), _snap(), _acct(daily_loss_r=3.5),
                       GateConfig(max_daily_loss_r=3.0))
        assert not res.ok and "daily loss" in _why(res)

    def test_risk_above_the_cap(self):
        res = run_gate(_sig(risk_pct=5.0), _snap(), _acct(),
                       GateConfig(max_risk_pct=2.0))
        assert not res.ok and "above cap" in _why(res)

    def test_an_entry_stale_or_far_from_the_market(self):
        res = run_gate(_sig(entry=2000.13, stop=1991.44, tp1=2016.81),
                       _snap(), _acct(), GateConfig())
        assert not res.ok and "from market" in _why(res)

    def test_round_grid_levels_are_treated_as_placeholders(self):
        # The check that caught this file's own first draft: round numbers with
        # an integer R look generated, not measured off a chart.
        res = run_gate(_sig(entry=2400.0, stop=2390.0, tp1=2420.0),
                       _snap(), _acct(), GateConfig())
        assert not res.ok and "synthetic" in _why(res)


class TestSizing:
    def test_risk_scales_with_the_configured_percentage(self):
        half = compute_lots(_sig(), _snap(), _acct(), risk_pct=0.5)
        full = compute_lots(_sig(), _snap(), _acct(), risk_pct=1.0)
        assert half.ok and full.ok
        assert full.lots > half.lots

    def test_lots_land_on_the_brokers_volume_step(self):
        res = compute_lots(_sig(), _snap(volume_step=0.01), _acct(), risk_pct=1.0)
        assert res.ok
        steps = res.lots / 0.01
        assert steps == pytest.approx(round(steps), abs=1e-6)

    def test_sizing_never_exceeds_the_brokers_maximum(self):
        rich = _acct(balance=10_000_000.0, equity=10_000_000.0)
        res = compute_lots(_sig(), _snap(volume_max=0.05), rich, risk_pct=2.0)
        assert res.lots <= 0.05

    def test_an_account_too_small_for_the_minimum_lot_does_not_trade(self):
        # The dangerous case: rounding *up* to volume_min would silently risk
        # far more than the configured percentage.
        res = compute_lots(_sig(), _snap(volume_min=1.0),
                           _acct(balance=10.0, equity=10.0), risk_pct=0.5)
        assert not res.ok or res.lots == 0.0

    def test_a_zero_width_stop_does_not_divide_by_zero(self):
        res = compute_lots(_sig(stop=2400.37), _snap(), _acct(), risk_pct=1.0)
        assert not res.ok or res.lots == 0.0


class TestOrderType:
    def test_an_entry_at_the_market_classifies_as_a_known_type(self):
        kind = classify_order_type(_sig(entry=2400.39), _snap())
        assert isinstance(kind, str) and kind
