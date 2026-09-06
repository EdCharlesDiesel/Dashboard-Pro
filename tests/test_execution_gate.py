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

from dataclasses import replace

import pytest

from src.execution.gate import (
    AccountState,
    GateConfig,
    MarketSnapshot,
    Signal,
    classify_order_type,
    compute_lots,
    looks_synthetic,
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


def _gold_snap(bid=4430.194, ask=4430.376) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="XAUUSD", bid=bid, ask=ask, point=0.001, digits=3,
        tick_value=1.0, tick_size=0.001, volume_min=0.01, volume_step=0.01,
        volume_max=200.0, trade_allowed=True, stops_level_points=0.0)


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

    def test_a_spread_wider_than_the_backstop(self):
        # A wide stop keeps the spread-as-fraction-of-stop check from firing
        # (10.0 / 1000.0 = 1%, well under the 10% default), isolating the
        # absolute-points backstop that this test is actually about.
        res = run_gate(_sig(entry=2400.37, stop=1400.37, tp1=4400.37),
                       _snap(bid=2400.11, ask=2410.11), _acct(),
                       GateConfig(max_spread_points=40.0))
        assert not res.ok and "above 40" in _why(res)

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


class TestSpreadRelativeToTheStopItProtects:
    def test_gold_normal_spread_does_not_block_a_normal_stop(self):
        """182 points is gold's ordinary spread and 0.9% of a $20 stop."""
        snap = _gold_snap()
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4430.376, stop=4410.376, tp1=4470.376)
        res = run_gate(sig, snap, _acct(), GateConfig())
        assert not any("spread" in r for r in res.reasons), res.reasons

    def test_spread_blocks_when_it_eats_the_stop(self):
        """The same 182-point spread against a $1 stop is 18% of risk."""
        snap = _gold_snap()
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4430.376, stop=4429.376, tp1=4432.376)
        res = run_gate(sig, snap, _acct(), GateConfig())
        assert any("spread" in r for r in res.reasons), res.reasons


class TestMaximumStopMeasuredInATR:
    def test_normal_gold_stop_passes_when_atr_is_known(self):
        """$20 is 0.19 ATR on gold (ATR14 = $104). Nowhere near a 3x cap."""
        snap = replace(_gold_snap(), atr=104.37)
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4430.376, stop=4410.376, tp1=4470.376)
        res = run_gate(sig, snap, _acct(), GateConfig())
        assert not any("stop" in r and "maximum" in r for r in res.reasons), res.reasons

    def test_absurd_stop_still_blocked_in_atr_terms(self):
        """$400 is 3.8 ATR - past the 3x cap."""
        snap = replace(_gold_snap(), atr=104.37)
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4430.376, stop=4030.376, tp1=5230.376)
        res = run_gate(sig, snap, _acct(), GateConfig())
        assert any("ATR" in r for r in res.reasons), res.reasons

    def test_falls_back_to_points_when_atr_unknown(self):
        """atr=0 must not disable the check - it reverts to today's limit."""
        snap = _gold_snap()          # atr defaults to 0.0
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4430.376, stop=4410.376, tp1=4470.376)
        res = run_gate(sig, snap, _acct(), GateConfig())
        assert any("above maximum" in r for r in res.reasons), res.reasons


class TestEntryDeviationMeasuredInATR:
    def test_gold_pullback_entry_is_not_called_stale(self):
        snap = replace(_gold_snap(), atr=104.37)
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4420.376, stop=4400.376, tp1=4461.376)
        res = run_gate(sig, snap, _acct(), GateConfig())
        assert not any("from market" in r for r in res.reasons), res.reasons

    def test_the_stub_entry_is_still_blocked(self):
        """The motivating case from looks_synthetic's docstring: 1.10000 while
        EUR/USD trades at 1.16279 - 14.9 ATR away."""
        snap = MarketSnapshot(
            symbol="EURUSD", bid=1.16269, ask=1.16279, point=0.00001, digits=5,
            tick_value=1.0, tick_size=0.00001, volume_min=0.01, volume_step=0.01,
            volume_max=200.0, trade_allowed=True, stops_level_points=0.0,
            atr=0.0042)
        sig = Signal(signal_id="s", symbol="EURUSD", direction="BUY",
                     entry=1.10000, stop=1.09500, tp1=1.11000)
        # The default whitelist is metals-only, so EUR/USD would also collect a
        # whitelist block. Name it explicitly - otherwise this test could pass on
        # the wrong reason, which is how three guards in this repo already went
        # green while checking nothing.
        cfg = GateConfig(symbol_whitelist=("EURUSD",))
        res = run_gate(sig, snap, _acct(), cfg)
        assert any("from market" in r for r in res.reasons), res.reasons
        assert not any("whitelist" in r for r in res.reasons), res.reasons


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


class TestATR14FromRates:
    def test_atr14_from_rates_matches_a_hand_computed_value(self):
        """Three bars, no gaps: TR is just high-low, so ATR is their mean."""
        rates = [
            {"high": 10.0, "low": 8.0, "close": 9.0},
            {"high": 11.0, "low": 9.0, "close": 10.0},
            {"high": 12.0, "low": 10.0, "close": 11.0},
        ]
        from src.execution.mt5_executor import atr14_from_rates
        assert atr14_from_rates(rates) == pytest.approx(2.0)

    def test_atr14_counts_the_gap_not_just_the_range(self):
        """A bar that gaps up has a true range larger than its own high-low."""
        rates = [
            {"high": 10.0, "low": 9.0, "close": 9.5},
            {"high": 20.0, "low": 19.0, "close": 19.5},
        ]
        from src.execution.mt5_executor import atr14_from_rates
        assert atr14_from_rates(rates) == pytest.approx(10.5)

    def test_atr14_returns_zero_on_insufficient_data(self):
        from src.execution.mt5_executor import atr14_from_rates
        assert atr14_from_rates([]) == 0.0
        assert atr14_from_rates([{"high": 1.0, "low": 0.5, "close": 0.8}]) == 0.0


class TestExecutorConfigStaysScaleRelative:
    def test_the_executor_config_does_not_reimpose_the_old_point_limits(self, monkeypatch):
        """The runtime config must not undo the scale-relative limits.

        mt5_executor built its GateConfig with
        `max_spread_points=os.environ.get("EXECUTOR_MAX_SPREAD_PTS", "40")`.
        That silently restored the limit this plan removed - unit tests calling
        GateConfig() directly passed while the running executor still blocked
        every gold trade.
        """
        for var in ("EXECUTOR_MAX_SPREAD_PTS", "EXECUTOR_MAX_SPREAD_FRAC"):
            monkeypatch.delenv(var, raising=False)
        from src.execution.mt5_executor import build_gate_config
        cfg = build_gate_config()
        assert cfg.max_spread_points > 1_000, cfg.max_spread_points
        assert cfg.max_spread_frac_of_stop == pytest.approx(0.10)

    def test_the_executor_config_still_honours_an_explicit_override(self, monkeypatch):
        monkeypatch.setenv("EXECUTOR_MAX_SPREAD_FRAC", "0.05")
        from src.execution.mt5_executor import build_gate_config
        assert build_gate_config().max_spread_frac_of_stop == pytest.approx(0.05)


class TestTheStubDetectorIsScaleRelative:
    """Roundness must mean the same thing on a $1.16 pair and a $4,600 one.

    The grid was `point * 1000`, whose docstring assumed gold quotes at 2
    digits. This broker quotes it at 3, making the grid $1.00 instead of $10
    and rejecting the desk's own round entries (4482, 4649, 4626.5 all appear
    in trade_setups).
    """

    def _snap(self, symbol, bid, ask, point, digits):
        return MarketSnapshot(
            symbol=symbol, bid=bid, ask=ask, point=point, digits=digits,
            tick_value=1.0, tick_size=point, volume_min=0.01,
            volume_step=0.01, volume_max=200.0, trade_allowed=True,
            stops_level_points=0.0)

    def test_the_live_stub_is_still_caught(self):
        """pending_signals holds EURUSD 1.100000/1.095000/1.110000 right now.

        This is the whole reason the check exists. If this ever goes green
        the narrowing has gone too far.
        """
        snap = self._snap("EURUSD", 1.16269, 1.16279, 0.00001, 5)
        sig = Signal(signal_id="s", symbol="EURUSD", direction="BUY",
                     entry=1.100000, stop=1.095000, tp1=1.110000)
        assert looks_synthetic(sig, snap) is True

    def test_the_live_gold_signal_is_not_a_stub(self):
        """pending_signals also holds XAUUSD 4604.31/4589.77/4633.62."""
        snap = self._snap("XAUUSD", 4604.19, 4604.31, 0.001, 3)
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4604.31, stop=4589.77, tp1=4633.62)
        assert looks_synthetic(sig, snap) is False

    @pytest.mark.parametrize("entry", [4482.0, 4649.0, 4626.5])
    def test_real_round_gold_entries_are_not_stubs(self, entry):
        """Every one of these is a real entry_price from trade_setups."""
        snap = self._snap("XAUUSD", entry - 0.1, entry, 0.001, 3)
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=entry, stop=entry - 20.0, tp1=entry + 40.0)
        assert looks_synthetic(sig, snap) is False

    def test_round_levels_with_a_non_integer_r_are_not_stubs(self):
        """The docstring always said BOTH conditions were required."""
        snap = self._snap("XAUUSD", 4599.9, 4600.0, 0.001, 3)
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4600.0, stop=4580.0, tp1=4643.0)   # R = 3.15
        assert looks_synthetic(sig, snap) is False

    def test_a_generated_gold_setup_is_still_caught(self):
        """Round to the $5 grid AND exactly 3R - the narrowing must not
        reach this."""
        snap = self._snap("XAUUSD", 4599.9, 4600.0, 0.001, 3)
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4600.0, stop=4580.0, tp1=4660.0)   # R = 3.00
        assert looks_synthetic(sig, snap) is True

    def test_a_missing_tp1_cannot_be_a_stub(self):
        """With no tp1 there is no R to be suspicious about."""
        snap = self._snap("EURUSD", 1.16269, 1.16279, 0.00001, 5)
        sig = Signal(signal_id="s", symbol="EURUSD", direction="BUY",
                     entry=1.100000, stop=1.095000, tp1=None)
        assert looks_synthetic(sig, snap) is False

    def test_a_zero_entry_does_not_raise(self):
        """log10(0) is -inf; a malformed signal must not crash the gate."""
        snap = self._snap("EURUSD", 1.16269, 1.16279, 0.00001, 5)
        sig = Signal(signal_id="s", symbol="EURUSD", direction="BUY",
                     entry=0.0, stop=0.0, tp1=0.0)
        assert looks_synthetic(sig, snap) is False
