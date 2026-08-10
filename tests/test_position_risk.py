"""Unit tests for src/services/position_risk.py — pure maths, no I/O.

The three things that can silently produce a wrong verdict, each with its own
class: the short's reciprocal transform, the quote-currency conversion, and the
edge arithmetic that decides whether a position is worth holding at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.stochastic import GBMParams, calibrate_gbm
from src.services import position_risk as pr

RATES = {"USD": 1.0, "ZAR": 1 / 16.16229, "JPY": 1 / 158.494, "AUD": 0.70710}


def _params(sigma=0.10, mu=0.05):
    return GBMParams(mu=mu, sigma=sigma, nu=mu - 0.5 * sigma ** 2,
                     drag=0.5 * sigma ** 2, n_obs=500, periods_per_year=252)


def _pos(pair="EUR/USD", side="Long", lots=0.01, entry=1.10, spot=1.10,
         stop=1.09, target=1.12):
    return pr.Position(pair=pair, side=side, lots=lots, entry=entry, spot=spot,
                       stop=stop, target=target)


class TestUsdPnl:
    def test_usd_quoted_long_needs_no_conversion(self):
        # 0.01 lot = 1,000 units; a 0.01 move = $10.
        assert pr.usd_pnl("EUR/USD", "Long", 0.01, 1.10, 1.11, RATES) == pytest.approx(10.0)

    def test_short_profits_when_price_falls(self):
        assert pr.usd_pnl("EUR/USD", "Short", 0.01, 1.10, 1.09, RATES) == pytest.approx(10.0)
        assert pr.usd_pnl("EUR/USD", "Short", 0.01, 1.10, 1.11, RATES) == pytest.approx(-10.0)

    def test_zar_quoted_pnl_is_converted_from_rand(self):
        # 0.10 lot EUR/ZAR = 10,000 EUR; a 0.10 ZAR move = 1,000 ZAR, not $1,000.
        got = pr.usd_pnl("EUR/ZAR", "Long", 0.10, 18.60, 18.70, RATES)
        assert got == pytest.approx(1000.0 / 16.16229, rel=1e-9)

    def test_missing_rate_reports_unconverted_rather_than_zero(self):
        # Silently returning 0 would read as "this position cannot lose".
        got = pr.usd_pnl("EUR/ZAR", "Long", 0.10, 18.60, 18.70, {"USD": 1.0})
        assert got == pytest.approx(1000.0)

    def test_metal_uses_its_own_contract_size(self):
        # XAU is 100 oz per lot, not 100k: 0.10 lot = 10 oz, $1 move = $10.
        assert pr.usd_pnl("XAU/USD", "Long", 0.10, 4300.0, 4301.0, RATES) == pytest.approx(10.0)

    def test_degenerate_inputs_are_zero_not_exceptions(self):
        assert pr.usd_pnl("NOTAPAIR", "Long", 0.01, 1.0, 1.1, RATES) == 0.0
        assert pr.usd_pnl("EUR/USD", "Neutral", 0.01, 1.0, 1.1, RATES) == 0.0
        assert pr.usd_pnl("EUR/USD", "Long", 0.0, 1.0, 1.1, RATES) == 0.0


class TestShortReciprocalTransform:
    """The short path must be the long path mirrored — not an approximation."""

    def test_driftless_symmetric_barriers_are_a_coin_flip(self):
        # nu = 0 collapses the scale function to gambler's ruin, l / (u + l).
        # Equal log-distances therefore give exactly 0.5, both directions.
        sigma = 0.20
        p = _params(sigma=sigma, mu=0.5 * sigma ** 2)      # nu = 0
        k = 1.05
        long_view = pr.barrier_view("Long", 100.0, 100.0 / k, 100.0 * k, p)
        short_view = pr.barrier_view("Short", 100.0, 100.0 * k, 100.0 / k, p)
        assert long_view["p_take_profit"] == pytest.approx(0.5, abs=1e-6)
        assert short_view["p_take_profit"] == pytest.approx(0.5, abs=1e-6)

    def test_short_equals_long_with_the_log_drift_negated(self):
        # A short on S is a long on 1/S, whose log-drift is -nu. Pricing the
        # long with nu flipped must reproduce the short exactly.
        sigma, mu = 0.25, 0.12
        p = _params(sigma=sigma, mu=mu)
        short_view = pr.barrier_view("Short", 100.0, 105.0, 95.0, p)
        mirrored = _params(sigma=sigma, mu=-mu + sigma ** 2)
        # On the reciprocal the barriers swap: the short's stop at 105 becomes
        # the low barrier 1/105, and its target at 95 becomes the high one.
        long_view = pr.barrier_view("Long", 1 / 100.0, 1 / 105.0, 1 / 95.0, mirrored)
        assert short_view["p_take_profit"] == pytest.approx(long_view["p_take_profit"])

    def test_positive_drift_favours_the_long_side(self):
        p = _params(sigma=0.15, mu=0.30)                   # strongly positive nu
        long_view = pr.barrier_view("Long", 100.0, 95.0, 105.0, p)
        short_view = pr.barrier_view("Short", 100.0, 105.0, 95.0, p)
        assert long_view["p_take_profit"] > 0.5 > short_view["p_take_profit"]
        # ...and they are complementary: same barriers, opposite sides.
        assert long_view["p_take_profit"] + short_view["p_take_profit"] == pytest.approx(1.0)

    def test_reward_risk_is_reported_for_the_short_not_the_reciprocal(self):
        p = _params()
        view = pr.barrier_view("Short", 100.0, 110.0, 90.0, p)
        # Risking 10 to make 10 on the reciprocal is not exactly 1.0 in log
        # space, but it must be close — a wildly different number would mean the
        # transform leaked the reciprocal's scale into the report.
        assert view["reward_risk"] == pytest.approx(1.0, rel=0.25)


class TestAssess:
    def test_edge_is_probability_minus_required_win_rate(self):
        risk = pr.assess(_pos(), _params(), RATES, use_monte_carlo=False)
        expected = (risk.p_target - risk.breakeven) * 100
        assert risk.edge_pp == pytest.approx(expected)

    def test_expected_value_is_probability_weighted(self):
        risk = pr.assess(_pos(), _params(), RATES, use_monte_carlo=False)
        manual = risk.p_target * risk.win_usd + (1 - risk.p_target) * risk.loss_usd
        assert risk.ev_usd == pytest.approx(manual)

    def test_loss_is_negative_and_win_positive(self):
        risk = pr.assess(_pos(), _params(), RATES, use_monte_carlo=False)
        assert risk.win_usd > 0 > risk.loss_usd

    def test_probabilities_sum_to_one(self):
        risk = pr.assess(_pos(), _params(), RATES, use_monte_carlo=False)
        assert risk.p_target + risk.p_stop == pytest.approx(1.0)

    def test_zero_arithmetic_drift_gives_exactly_zero_edge(self):
        # mu = 0 makes the *price* a martingale, so P(target) lands precisely on
        # the win rate that reward:risk demands and EV is zero to machine
        # precision. Worth pinning: it is the reference point every real edge is
        # measured against, and it falls out of the scale function rather than
        # being coded anywhere.
        risk = pr.assess(_pos(stop=0.90, target=1.04, entry=1.0, spot=1.0),
                         _params(sigma=0.09, mu=0.0), RATES, use_monte_carlo=False)
        assert risk.p_target == pytest.approx(risk.breakeven, abs=1e-12)
        assert risk.edge_pp == pytest.approx(0.0, abs=1e-9)
        assert risk.ev_usd == pytest.approx(0.0, abs=1e-9)

    def test_high_hit_rate_can_still_be_negative_edge(self):
        # The GBP/ZAR shape: wins often, pays badly. This is the whole point of
        # the module — a 0.40 reward:risk needs a 71.4% hit rate to break even,
        # so a 68% winner is a loser.
        risk = pr.assess(_pos(stop=0.90, target=1.04, entry=1.0, spot=1.0),
                         _params(sigma=0.09, mu=-0.01), RATES, use_monte_carlo=False)
        assert risk.p_target > 0.60           # wins most of the time
        assert risk.ev_usd < 0                # and still loses money

    def test_unbracketed_position_is_not_priced(self):
        assert pr.assess(_pos(stop=None), _params(), RATES) is None
        assert pr.assess(_pos(target=None), _params(), RATES) is None

    def test_probability_is_the_closed_form_regardless_of_horizon(self):
        # Running the Monte Carlo must not move the verdict: it supplies the
        # unresolved diagnostic, never the probability being compared against
        # the breakeven rate.
        risk_mc = pr.assess(_pos(), _params(), RATES, n_sims=4000, use_monte_carlo=True)
        risk_cf = pr.assess(_pos(), _params(), RATES, use_monte_carlo=False)
        assert risk_mc.p_target == pytest.approx(risk_cf.p_target, abs=1e-12)
        assert risk_mc.edge_pp == pytest.approx(risk_cf.edge_pp, abs=1e-9)


class TestCone:
    def test_prices_ascend_with_the_quantiles(self):
        out = pr.cone(_pos(), _params(), RATES, n_paths=4000)
        assert out["prices"] == sorted(out["prices"])

    def test_long_pnl_ascends_with_price(self):
        out = pr.cone(_pos(side="Long"), _params(), RATES, n_paths=4000)
        assert out["pnl_usd"] == sorted(out["pnl_usd"])

    def test_short_pnl_is_reversed_so_bad_to_good_still_reads_left_to_right(self):
        # A short's worst outcome is the highest price. Without the flip the
        # 5% column would show the short's *best* case.
        out = pr.cone(_pos(side="Short", stop=1.12, target=1.09),
                      _params(), RATES, n_paths=4000)
        assert out["pnl_usd"] == sorted(out["pnl_usd"])
        assert out["pnl_usd"][0] < 0 < out["pnl_usd"][-1]

    def test_median_price_is_near_spot_for_a_short_horizon(self):
        out = pr.cone(_pos(), _params(sigma=0.10, mu=0.0), RATES,
                      horizon_days=1, n_paths=8000)
        assert out["prices"][2] == pytest.approx(1.10, rel=0.01)

    def test_is_reproducible_for_a_fixed_seed(self):
        a = pr.cone(_pos(), _params(), RATES, n_paths=2000, seed=11)
        b = pr.cone(_pos(), _params(), RATES, n_paths=2000, seed=11)
        assert a["prices"] == b["prices"]


class TestPositionsFromStore:
    def test_reads_the_stored_schema(self):
        rows = [{"pair": "USD/ZAR", "direction": "SHORT", "lot_size": 0.01,
                 "entry_price": 16.10832, "stop_loss": 16.4387,
                 "take_profit": 15.9388}]
        (position,) = pr.positions_from_store(rows)
        assert position.pair == "USD/ZAR" and position.side == "Short"
        assert position.lots == 0.01 and position.stop == pytest.approx(16.4387)
        assert position.bracketed

    def test_reads_the_live_mt5_schema(self):
        rows = [{"pair": "XAU/USD", "direction": "buy", "volume": 0.10,
                 "price_open": 4339.899, "price_current": 4354.645,
                 "sl": 4299.594, "tp": 4443.724}]
        (position,) = pr.positions_from_store(rows)
        assert position.lots == 0.10
        assert position.spot == pytest.approx(4354.645)
        assert position.target == pytest.approx(4443.724)

    def test_zero_stop_means_no_stop_not_a_stop_at_zero(self):
        # MT5 reports "no stop" as 0.0. Treating that as a real barrier would
        # price the position against a 100% loss.
        rows = [{"pair": "USD/ZAR", "direction": "SHORT", "lot_size": 0.01,
                 "entry_price": 16.1, "stop_loss": 0, "take_profit": 15.9}]
        (position,) = pr.positions_from_store(rows)
        assert position.stop is None and not position.bracketed

    def test_spot_falls_back_to_entry_when_no_live_price(self):
        rows = [{"pair": "EUR/USD", "direction": "LONG", "lot_size": 0.01,
                 "entry_price": 1.15}]
        (position,) = pr.positions_from_store(rows)
        assert position.spot == pytest.approx(1.15)

    def test_malformed_rows_are_skipped(self):
        rows = [{"pair": "", "direction": "LONG", "entry_price": 1.0},
                {"pair": "EUR/USD", "direction": "Neutral", "entry_price": 1.0},
                {"pair": "EUR/USD", "direction": "LONG"},
                {"pair": "EUR/USD", "direction": "LONG", "entry_price": 1.15}]
        assert len(pr.positions_from_store(rows)) == 1

    def test_empty_input(self):
        assert pr.positions_from_store([]) == []
        assert pr.positions_from_store(None) == []


class TestContractUnits:
    def test_forex_is_the_hundred_thousand_standard(self):
        assert pr.contract_units("EUR/USD", 0.01) == pytest.approx(1000.0)

    def test_metals_and_oil_have_their_own_sizes(self):
        assert pr.contract_units("XAU/USD", 0.10) == pytest.approx(10.0)
        assert pr.contract_units("XAG/USD", 0.10) == pytest.approx(500.0)


class TestAgainstRealCalibration:
    """One end-to-end pass on synthetic-but-realistic prices."""

    def test_calibrated_params_price_a_position(self):
        rng = np.random.default_rng(3)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(500) * 0.01))
        params = calibrate_gbm(prices)
        spot = float(prices[-1])
        position = _pos(pair="EUR/USD", side="Long", entry=spot, spot=spot,
                        stop=spot * 0.98, target=spot * 1.03)
        risk = pr.assess(position, params, RATES, use_monte_carlo=False)
        assert 0.0 < risk.p_target < 1.0
        assert risk.reward_risk == pytest.approx(1.5, rel=0.05)


class TestPositionFromIdea:
    """`Idea` -> `Position`, the join between the board and the scale function."""

    def _idea(self, **kw):
        from src.core.todays_trades import Idea
        base = dict(pair="EUR/USD", direction="Long", entry=1.10,
                    stop=1.09, target=1.12, lots=0.02)
        base.update(kw)
        idea = Idea(pair=base["pair"], direction=base["direction"])
        for field in ("entry", "stop", "target", "lots"):
            setattr(idea, field, base[field])
        return idea

    def test_maps_a_sized_idea(self):
        position = pr.position_from_idea(self._idea())
        assert position.pair == "EUR/USD" and position.side == "Long"
        assert position.lots == 0.02 and position.bracketed

    def test_spot_defaults_to_entry(self):
        assert pr.position_from_idea(self._idea()).spot == pytest.approx(1.10)

    def test_explicit_spot_wins(self):
        assert pr.position_from_idea(self._idea(), spot=1.11).spot == pytest.approx(1.11)

    def test_unsized_idea_is_unbracketed_not_priced_at_zero(self):
        # size_idea leaves stop/target None when it cannot size the instrument.
        position = pr.position_from_idea(self._idea(stop=None, target=None))
        assert position is not None and not position.bracketed

    def test_idea_without_entry_is_rejected(self):
        assert pr.position_from_idea(self._idea(entry=None)) is None


class TestScreen:
    """The filter the desk asked for, and the one it actually needs."""

    def _position(self, pair, side, stop, target, lots=0.01):
        return pr.Position(pair=pair, side=side, lots=lots, entry=1.0, spot=1.0,
                           stop=stop, target=target)

    def test_keeps_target_first_and_positive_edge(self):
        # Wide target, tight stop, positive drift: high P and a payoff to match.
        positions = [self._position("EUR/USD", "Long", 0.98, 1.02)]
        params = {"EUR/USD": _params(sigma=0.08, mu=0.10)}
        assert len(pr.screen(positions, params, RATES, n_sims=4000)) == 1

    def test_drops_high_probability_with_negative_edge(self):
        # The GBP/ZAR trap: wins ~68% of the time, needs 71.4%, so it is a loser
        # that a naive "P(target) > 50%" screen would rank near the top.
        positions = [self._position("EUR/USD", "Long", 0.90, 1.04)]
        params = {"EUR/USD": _params(sigma=0.09, mu=-0.01)}
        assert pr.screen(positions, params, RATES, n_sims=4000) == []

    def test_any_edge_mode_keeps_it(self):
        positions = [self._position("EUR/USD", "Long", 0.90, 1.04)]
        params = {"EUR/USD": _params(sigma=0.09, mu=-0.01)}
        kept = pr.screen(positions, params, RATES, n_sims=4000,
                         require_positive_edge=False)
        assert len(kept) == 1 and kept[0].edge_pp < 0

    def test_drops_below_the_probability_floor(self):
        # Far target, near stop: target rarely comes first, whatever it pays.
        positions = [self._position("EUR/USD", "Long", 0.995, 1.20)]
        params = {"EUR/USD": _params(sigma=0.10, mu=0.0)}
        assert pr.screen(positions, params, RATES, n_sims=4000) == []

    def test_ranks_by_edge_not_probability(self):
        positions = [self._position("EUR/USD", "Long", 0.97, 1.01),
                     self._position("GBP/USD", "Long", 0.98, 1.04)]
        params = {"EUR/USD": _params(sigma=0.08, mu=0.02),
                  "GBP/USD": _params(sigma=0.08, mu=0.35)}
        kept = pr.screen(positions, params, RATES, n_sims=4000,
                         min_p_target=0.0)
        assert [r.edge_pp for r in kept] == sorted(
            [r.edge_pp for r in kept], reverse=True)

    def test_unpriceable_and_unbracketed_are_skipped_quietly(self):
        positions = [self._position("EUR/USD", "Long", None, 1.02),
                     self._position("NOPE/XXX", "Long", 0.98, 1.02)]
        assert pr.screen(positions, {"EUR/USD": _params()}, RATES) == []

    def test_empty_input(self):
        assert pr.screen([], {}, RATES) == []


class TestUnresolvedPaths:
    """The conditional-vs-unconditional trap in the Monte Carlo figure.

    `breakeven_win_rate` is 1/(1+R:R) — the share of *resolved* trades that must
    win. Comparing it against the unconditional MC probability treats every path
    that touched neither barrier as a loss, so a perfectly good trade with wide
    barriers reads as negative edge purely because the horizon was short.
    """

    def _wide(self):
        # +-10%/+4% barriers against ~2.6% of 21-day vol: most paths resolve
        # neither, which is exactly when the two figures diverge.
        return _pos(entry=1.0, spot=1.0, stop=0.90, target=1.04)

    def test_unresolved_fraction_is_reported(self):
        risk = pr.assess(self._wide(), _params(sigma=0.09, mu=-0.01), RATES,
                         n_sims=4000)
        assert risk.p_unresolved > 0.5          # most paths are still open

    def test_neither_monte_carlo_figure_replaces_the_closed_form(self):
        # Both MC figures are wrong for this comparison, in opposite directions:
        # the unconditional one is dragged toward zero by unresolved paths, the
        # conditional one toward 1.0 because a short horizon reaches the near
        # barrier and never the far one. The verdict must use neither.
        from src.core.stochastic import barrier_probabilities
        params = _params(sigma=0.09, mu=-0.01)
        risk = pr.assess(self._wide(), params, RATES, n_sims=20_000)
        raw = barrier_probabilities(1.0, 1.04, 0.90, params.sigma, params.mu,
                                    horizon_years=21 / 252, n_sims=20_000)
        assert raw["mc_p_take_profit"] < 0.5          # unconditional: too low
        assert raw["mc_p_tp_given_resolved"] > 0.95   # conditional: too high
        assert risk.p_target == pytest.approx(raw["p_take_profit"], abs=1e-12)
        assert 0.6 < risk.p_target < 0.75             # the honest figure

    def test_tight_barriers_resolve_and_report_no_unresolved(self):
        risk = pr.assess(_pos(entry=1.0, spot=1.0, stop=0.995, target=1.005),
                         _params(sigma=0.30, mu=0.0), RATES, n_sims=4000)
        assert risk.p_unresolved == pytest.approx(0.0, abs=0.01)

    def test_closed_form_path_reports_zero_unresolved(self):
        risk = pr.assess(_pos(), _params(), RATES, use_monte_carlo=False)
        assert risk.p_unresolved == 0.0


class TestVerdict:
    """The prose the page shows. Both numbers, always — either alone misleads."""

    def _risk(self, p_target, breakeven, rr, edge=None, unresolved=0.0):
        return pr.PositionRisk(
            pair="EUR/USD", side="Long", p_target=p_target,
            p_stop=1 - p_target, reward_risk=rr, breakeven=breakeven,
            edge_pp=(p_target - breakeven) * 100 if edge is None else edge,
            win_usd=10.0, loss_usd=-5.0, ev_usd=1.0, p_unresolved=unresolved)

    def test_positive_edge_says_take(self):
        out = pr.verdict(self._risk(0.348, 0.333, 2.0))
        assert out["verdict"] == "TAKE" and out["reason"].startswith("TAKE")

    def test_negative_edge_says_skip(self):
        out = pr.verdict(self._risk(0.310, 0.333, 2.0))
        assert out["verdict"] == "SKIP" and out["reason"].startswith("SKIP")

    def test_reason_names_both_the_required_and_the_actual_rate(self):
        # "34.8% of the time" alone reads as a bad trade; "2:1" alone reads as a
        # good one. Only the pair is an argument.
        out = pr.verdict(self._risk(0.348, 0.333, 2.0))
        assert "33.3%" in out["reason"] and "34.8%" in out["reason"]
        assert "2.00:1" in out["reason"]

    def test_sub_fifty_percent_is_explained_not_flagged_as_bad(self):
        # A 2:1 target is stopped more often than not by construction. The page
        # must say so, or every good trade looks like a warning.
        out = pr.verdict(self._risk(0.348, 0.333, 2.0))
        assert "not a reason to skip" in out["reason"]
        assert out["verdict"] == "TAKE"

    def test_high_probability_low_payoff_still_skips(self):
        # The GBP/ZAR shape, in words.
        out = pr.verdict(self._risk(0.788, 0.759, 0.32, edge=-2.0))
        assert out["verdict"] == "SKIP"
        assert "gives only" in out["reason"]

    def test_no_sub_fifty_note_when_target_is_the_likelier_barrier(self):
        out = pr.verdict(self._risk(0.788, 0.759, 0.32))
        assert "Stop is hit first" not in out["reason"]

    def test_unresolved_share_is_surfaced_when_material(self):
        out = pr.verdict(self._risk(0.60, 0.50, 1.0, unresolved=0.60), horizon_days=21)
        assert "40% resolve inside 21d" in out["reason"]

    def test_no_unresolved_note_when_the_trade_resolves(self):
        out = pr.verdict(self._risk(0.60, 0.50, 1.0, unresolved=0.01))
        assert "resolve inside" not in out["reason"]
