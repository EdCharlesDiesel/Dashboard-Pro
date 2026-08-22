"""The event reaction map's pure core: surprise scoring, regime rotation, the
exposure board, release timing, and the board -> trade_setups mapping, for all
four events (NFP, CPI, PPI, FOMC).

No Streamlit runtime, no network, no DB — everything here is arithmetic over
hand-built inputs, which is the whole reason the maths lives in src/core/
rather than in the page.

``TestNFPRegression`` is the gate on the four-event generalisation: it pins the
exact numbers the NFP-only implementation produced before every function grew
a ``spec`` argument.
"""
from datetime import date

import pytest

from src.core.nfp_reaction import (
    EVENTS,
    Surprise,
    board_to_signals,
    chain_leaves,
    compute_surprise,
    release_datetime_sast,
    score_instruments,
    timing_frame,
)

BALANCED = "Balanced"
RATES_LED = "Rates-led (good news is bad news)"
GROWTH_SCARE = "Growth-scare (bad news is bad news)"

NFP, CPI, PPI, FOMC = (EVENTS["NFP"], EVENTS["CPI"],
                       EVENTS["PPI"], EVENTS["FOMC"])

# Component weights, by key, for whichever event is under test.
W = {c.key: c.weight for c in NFP.components}


def nfp_input(**kw) -> dict:
    """An on-consensus NFP entry, with any component overridden by keyword."""
    base = {"nfp": 150.0, "nfp_c": 150.0, "rev": 0.0,
            "ur": 4.2, "ur_c": 4.2, "ahe": 0.3, "ahe_c": 0.3}
    base.update(kw)
    return base


# ===========================================================================
# Surprise scoring
# ===========================================================================

class TestComputeSurprise:
    def test_a_print_matching_consensus_scores_zero(self):
        s = compute_surprise(NFP, nfp_input())
        assert s.composite == pytest.approx(0.0)
        assert s.direction == "neutral"
        assert s.label == "In line"

    def test_a_beat_is_hawkish_and_a_miss_is_dovish(self):
        beat = compute_surprise(NFP, nfp_input(nfp=275.0))
        miss = compute_surprise(NFP, nfp_input(nfp=25.0))
        assert beat.composite > 0 and beat.direction == "hawkish"
        assert miss.composite < 0 and miss.direction == "dovish"

    def test_unemployment_is_inverted(self):
        """A LOWER unemployment print than forecast is hawkish, not dovish."""
        lower = compute_surprise(NFP, nfp_input(ur=4.0))
        higher = compute_surprise(NFP, nfp_input(ur=4.4))
        assert lower.z["ur"] > 0 and lower.composite > 0
        assert higher.z["ur"] < 0 and higher.composite < 0

    def test_headline_only_entry_renormalises_the_weights(self):
        """Omitting U3 and AHE must not drag the composite toward zero.

        With only nfp + rev live, the composite is their weighted mean over
        w=0.42+0.10, not over the full 1.0 — otherwise a headline-only entry
        would understate every surprise by ~half.
        """
        s = compute_surprise(NFP, {"nfp": 215.0, "nfp_c": 150.0, "rev": 0.0})
        assert s.z["nfp"] == pytest.approx(1.0)
        assert s.composite == pytest.approx(W["nfp"] / (W["nfp"] + W["rev"]))

    def test_a_partial_entry_drops_only_the_missing_leg(self):
        s = compute_surprise(NFP, {"nfp": 215.0, "nfp_c": 150.0, "rev": 0.0,
                                   "ur": 4.2, "ur_c": 4.2})       # no AHE
        live = W["nfp"] + W["rev"] + W["ur"]
        assert s.composite == pytest.approx(W["nfp"] / live)

    @pytest.mark.parametrize("composite,label", [
        (0.0, "In line"), (0.34, "In line"),
        (0.35, "Mild"), (0.99, "Mild"),
        (1.0, "Significant"), (1.99, "Significant"),
        (2.0, "Outlier"), (-3.0, "Outlier"),
    ])
    def test_label_boundaries(self, composite, label):
        assert Surprise(z={}, composite=composite).label == label


# ===========================================================================
# The exposure board
# ===========================================================================

class TestScoreInstruments:
    def test_board_covers_every_exposure_and_sorts_by_magnitude(self):
        board = score_instruments(NFP, 1.5, BALANCED)
        assert len(board) == len(NFP.exposures)
        assert list(board["abs_score"]) == sorted(board["abs_score"], reverse=True)

    def test_conviction_collapses_when_the_channels_cancel(self):
        from src.core.nfp_reaction import REGIMES
        w = REGIMES[BALANCED]
        # a = w_rate * b_rate, b = w_growth * b_growth; pick betas so a == -b.
        b_growth = -(w["w_rate"] * 0.55) / w["w_growth"]
        board = score_instruments(NFP, 1.0, BALANCED,
                                  overrides={"XAUUSD": (0.55, b_growth)})
        row = board.loc[board["symbol"] == "XAUUSD"].iloc[0]
        assert row["conviction"] == pytest.approx(0.0, abs=1e-9)
        assert row["score"] == pytest.approx(0.0, abs=1e-9)

    def test_conviction_is_one_when_the_channels_agree(self):
        board = score_instruments(NFP, 1.0, BALANCED,
                                  overrides={"XAUUSD": (1.0, 1.0)})
        row = board.loc[board["symbol"] == "XAUUSD"].iloc[0]
        assert row["conviction"] == pytest.approx(1.0)

    def test_equities_flip_sign_across_the_regime_rotation(self):
        """The whole point of the regime weights: a strong print sells equities
        when rates are the binding constraint and buys them in a growth scare."""
        rates = score_instruments(NFP, 1.0, RATES_LED)
        growth = score_instruments(NFP, 1.0, GROWTH_SCARE)
        us500_rates = rates.loc[rates["symbol"] == "US500", "score"].iloc[0]
        us500_growth = growth.loc[growth["symbol"] == "US500", "score"].iloc[0]
        assert us500_rates < 0 < us500_growth

    def test_the_dollar_loses_conviction_in_a_growth_scare(self):
        """DXY's growth beta is negative — the haven bid fights the rate
        discount, so a growth-scare regime is exactly when it whipsaws."""
        rates = score_instruments(NFP, 1.0, RATES_LED)
        growth = score_instruments(NFP, 1.0, GROWTH_SCARE)
        c_rates = rates.loc[rates["symbol"] == "DXY", "conviction"].iloc[0]
        c_growth = growth.loc[growth["symbol"] == "DXY", "conviction"].iloc[0]
        assert c_growth < 0.35 < c_rates

    def test_an_in_line_print_scores_the_whole_board_flat(self):
        board = score_instruments(NFP, 0.0, BALANCED)
        assert (board["score"] == 0.0).all()
        assert set(board["direction"]) == {"flat"}

    def test_channels_sum_to_the_score(self):
        board = score_instruments(NFP, 1.3, BALANCED)
        total = board["rate_channel"] + board["growth_channel"]
        assert total.sub(board["score"]).abs().max() < 1e-9


# ===========================================================================
# Timing
# ===========================================================================

class TestTiming:
    def test_release_time_is_dst_aware(self):
        """08:30 New York is 15:30 SAST in winter and 14:30 in summer. Hard-coding
        either one puts the desk an hour out for half the year."""
        winter = release_datetime_sast(NFP, date(2026, 1, 2))
        summer = release_datetime_sast(NFP, date(2026, 7, 3))
        assert (winter.hour, winter.minute) == (15, 30)
        assert (summer.hour, summer.minute) == (14, 30)

    def test_timing_frame_is_ordered_and_labelled(self):
        df = timing_frame(NFP, date(2026, 9, 4))
        assert list(df.columns) == ["Phase", "SAST", "What is happening"]
        assert df["Phase"].iloc[0] == "Pre-release drain"
        assert len(df) == 6


# ===========================================================================
# Board -> trade_setups
# ===========================================================================

class TestBoardToSignals:
    RELEASE = date(2026, 9, 4)

    def _signals(self, composite, regime=BALANCED, **kw):
        board = score_instruments(NFP, composite, regime)
        return board_to_signals(NFP, board, self.RELEASE, regime, composite, **kw)

    def test_an_in_line_print_persists_nothing(self):
        """Below the gate the page is declining to forecast — the same policy
        forecast_dashboard applies to a neutral driver score."""
        assert self._signals(0.0) == []
        assert self._signals(0.4) == []

    def test_only_registry_resolvable_symbols_are_persisted(self):
        pairs = {s["pair"] for s in self._signals(2.0)}
        assert pairs <= {"XAU/USD", "XAG/USD", "EUR/USD", "GBP/USD",
                         "USD/JPY", "AUD/USD", "USD/ZAR", "WTI/USD"}
        # These have no tradable registry pair and must never reach trade_setups.
        assert not pairs & {"DXY", "US500", "NAS100", "US10Y", "BTCUSD"}

    def test_oil_is_on_the_board_and_persists(self):
        """WTIUSD -> WTI/USD is registry-resolvable, unlike the index/rate/
        crypto rows, and must actually reach the store like the other FX and
        metals exposures do."""
        pairs = {s["pair"] for s in self._signals(2.0)}
        assert "WTI/USD" in pairs

    def test_bias_follows_the_sign_of_the_score(self):
        board = score_instruments(NFP, 2.0, BALANCED)
        by_pair = {s["pair"]: s for s in self._signals(2.0)}
        for _, row in board.iterrows():
            sig = by_pair.get({"XAUUSD": "XAU/USD", "EURUSD": "EUR/USD",
                               "USDJPY": "USD/JPY"}.get(row["symbol"], ""))
            if sig is None:
                continue
            assert sig["bias"] == ("Bullish" if row["score"] > 0 else "Bearish")

    def test_the_release_date_is_the_dedupe_period(self):
        for sig in self._signals(2.0):
            assert sig["bar_time"] == self.RELEASE

    def test_low_conviction_rows_are_dropped(self):
        loose = self._signals(2.0, min_conviction=0.0)
        strict = self._signals(2.0, min_conviction=0.9)
        assert len(strict) < len(loose)
        assert all(s["conviction_ratio"] >= 0.9 for s in strict)

    def test_every_signal_carries_the_fields_the_store_needs(self):
        for sig in self._signals(2.0):
            assert sig["pair"] and sig["bias"] in ("Bullish", "Bearish")
            assert sig["conviction"] in ("High", "Medium", "Low")
            assert 0.0 <= sig["strength_score"] <= 10.0
            assert BALANCED in sig["thesis"]

    def test_a_dovish_print_mirrors_a_hawkish_one(self):
        hawk = {s["pair"]: s["bias"] for s in self._signals(2.0)}
        dove = {s["pair"]: s["bias"] for s in self._signals(-2.0)}
        assert hawk and hawk.keys() == dove.keys()
        assert all(hawk[p] != dove[p] for p in hawk)


# ===========================================================================
# The transmission chain
# ===========================================================================

class TestChainLeaves:
    def test_the_trunk_follows_the_surprise(self):
        s = compute_surprise(NFP, {"nfp": 275.0, "nfp_c": 150.0})
        chain, _ = chain_leaves(NFP, s,
                                score_instruments(NFP, s.composite, BALANCED))
        assert [name for name, _ in chain] == ["jobs", "spending",
                                               "inflation", "rates"]
        assert all(up for _, up in chain)

    def test_the_leaves_are_read_off_the_board_not_the_surprise(self):
        """A hawkish print with the trunk pointing up must still draw equities
        up in a growth-scare regime. Deriving the leaves from the surprise
        would reproduce the naive chain this page exists to correct."""
        s = compute_surprise(NFP, {"nfp": 275.0, "nfp_c": 150.0})
        _, rates = chain_leaves(NFP, s,
                                score_instruments(NFP, s.composite, RATES_LED))
        _, growth = chain_leaves(NFP, s,
                                 score_instruments(NFP, s.composite, GROWTH_SCARE))
        assert dict(rates)["indices"] is False
        assert dict(growth)["indices"] is True

    def test_all_three_leaves_are_present(self):
        s = compute_surprise(NFP, nfp_input())
        _, leaves = chain_leaves(NFP, s,
                                 score_instruments(NFP, s.composite, BALANCED))
        assert [name for name, _ in leaves] == ["indices", "gold", "usd"]


# ===========================================================================
# The four-event generalisation
# ===========================================================================



class TestEventSpecs:
    def test_all_four_events_are_registered(self):
        assert set(EVENTS) == {"NFP", "CPI", "PPI", "FOMC"}

    def test_every_source_tag_fits_the_column_and_is_unique(self):
        tags = [e.source_tag for e in EVENTS.values()]
        assert all(len(t) <= 20 for t in tags), tags
        assert len(set(tags)) == 4

    def test_every_calendar_key_resolves_against_the_shared_calendar(self):
        from src.core.event_calendar import next_release
        for spec in EVENTS.values():
            # None is a legitimate answer (seed list exhausted); a raise is not.
            next_release(spec.calendar_key, date(2026, 8, 20))

    def test_component_weights_are_positive(self):
        for spec in EVENTS.values():
            assert spec.components
            assert all(c.weight > 0 for c in spec.components)

    def test_only_nfp_runs_through_jobs_and_spending(self):
        assert [n for n, _ in NFP.chain] == ["jobs", "spending",
                                             "inflation", "rates"]
        for spec in (CPI, PPI, FOMC):
            assert "jobs" not in [n for n, _ in spec.chain]

    def test_the_price_events_carry_a_node_that_moves_against_the_surprise(self):
        """A hot CPI is hawkish AND bad for real incomes. A chain whose every
        node points the same way is the naive chain this page exists to fix."""
        assert any(sign < 0 for _, sign in CPI.chain)
        assert any(sign < 0 for _, sign in PPI.chain)

    def test_fomc_is_an_afternoon_event(self):
        assert FOMC.release_time_ny.hour == 14
        for spec in (NFP, CPI, PPI):
            assert (spec.release_time_ny.hour,
                    spec.release_time_ny.minute) == (8, 30)


class TestNFPRegression:
    """The gate for the whole refactor.

    Every number below was captured from the pre-refactor implementation on
    2026-08-20 for the input (275 vs 150, +15k revision, U3 4.0 vs 4.2, AHE
    0.5 vs 0.3) under the Balanced regime. If one of these moves, the
    generalisation changed NFP behaviour and the implementation is wrong.
    Never edit an expected value here to make a run pass.
    """

    INPUT = {"nfp": 275.0, "nfp_c": 150.0, "rev": 15.0,
             "ur": 4.0, "ur_c": 4.2, "ahe": 0.5, "ahe_c": 0.3}

    def test_the_composite_and_every_component_z_are_unmoved(self):
        s = compute_surprise(NFP, self.INPUT)
        assert s.composite == pytest.approx(1.6197052947, abs=1e-9)
        assert s.z["nfp"] == pytest.approx(1.9230769231, abs=1e-9)
        assert s.z["rev"] == pytest.approx(0.2500000000, abs=1e-9)
        assert s.z["ur"] == pytest.approx(1.4285714286, abs=1e-9)
        assert s.z["ahe"] == pytest.approx(1.8181818182, abs=1e-9)
        assert s.label == "Significant" and s.direction == "hawkish"

    @pytest.mark.parametrize("symbol,score,conviction", [
        ("USDJPY", +1.6035082418, 1.0000000000),
        ("XAUUSD", -1.3565031843, 1.0000000000),
        ("US10Y",  +1.3119612887, 1.0000000000),
        ("XAGUSD", -0.9799217033, 0.6470588235),
        ("EURUSD", -0.8746408591, 0.8307692308),
        ("DXY",    +0.8665423327, 0.6184971098),
        ("GBPUSD", -0.7734092782, 0.7431906615),
        ("USDZAR", +0.7248181194, 0.3849462366),
        ("AUDUSD", -0.6195372752, 0.4358974359),
        ("US500",  +0.2672513736, 0.1764705882),
        ("BTCUSD", -0.2470050574, 0.2013201320),
        ("NAS100", +0.1295764236, 0.0707964602),
    ])
    def test_the_nfp_board_is_unmoved(self, symbol, score, conviction):
        board = score_instruments(
            NFP, compute_surprise(NFP, self.INPUT).composite, BALANCED)
        row = board.loc[board["symbol"] == symbol].iloc[0]
        assert row["score"] == pytest.approx(score, abs=1e-9)
        assert row["conviction"] == pytest.approx(conviction, abs=1e-9)


class TestGenericSurprise:
    def test_an_omitted_component_renormalises_rather_than_scoring_zero(self):
        full = compute_surprise(CPI, {"core_mm": 0.4, "core_mm_c": 0.3,
                                      "head_mm": 0.3, "head_mm_c": 0.3,
                                      "core_yy": 3.0, "core_yy_c": 3.0,
                                      "head_yy": 2.9, "head_yy_c": 2.9})
        partial = compute_surprise(CPI, {"core_mm": 0.4, "core_mm_c": 0.3})
        assert partial.composite > full.composite > 0

    def test_an_inverted_component_flips_sign(self):
        lower = compute_surprise(NFP, {"nfp": 150.0, "nfp_c": 150.0,
                                       "ur": 4.0, "ur_c": 4.2})
        assert lower.z["ur"] > 0 and lower.composite > 0

    def test_a_delta_only_component_needs_no_consensus(self):
        sd = {c.key: c.sd for c in FOMC.components}["decision_bp"]
        s = compute_surprise(FOMC, {"decision_bp": 25.0})
        assert s.z["decision_bp"] == pytest.approx(25.0 / sd)
        assert s.composite > 0

    def test_a_component_with_no_consensus_supplied_is_dropped_not_zeroed(self):
        """A paired component whose consensus box is left empty must not be
        scored as 'came in exactly on forecast' - that is a fabricated
        observation, and it would drag every composite toward zero."""
        s = compute_surprise(CPI, {"core_mm": 0.5, "core_mm_c": 0.3,
                                   "head_mm": 0.4})          # no head_mm_c
        assert "head_mm" not in s.z

    def test_hot_cpi_and_strong_nfp_are_both_hawkish(self):
        cpi = compute_surprise(CPI, {"core_mm": 0.5, "core_mm_c": 0.3})
        nfp = compute_surprise(NFP, {"nfp": 275.0, "nfp_c": 150.0})
        assert cpi.direction == nfp.direction == "hawkish"

    def test_an_empty_entry_scores_zero_rather_than_raising(self):
        assert compute_surprise(FOMC, {}).composite == 0.0


class TestPerEventExposures:
    def test_a_hawkish_print_sells_gold_under_every_event(self):
        for spec in EVENTS.values():
            board = score_instruments(spec, 2.0, BALANCED)
            gold = board.loc[board["symbol"] == "XAUUSD", "score"].iloc[0]
            assert gold < 0, spec.key

    def test_hot_cpi_sells_equities_even_in_a_growth_scare(self):
        """The stagflation asymmetry: a hawkish NFP is a growth *positive*, a
        hawkish CPI is a growth negative, so they diverge exactly where the
        growth channel carries the most weight."""
        nfp = score_instruments(NFP, 2.0, GROWTH_SCARE)
        cpi = score_instruments(CPI, 2.0, GROWTH_SCARE)
        assert nfp.loc[nfp["symbol"] == "US500", "score"].iloc[0] > 0
        assert cpi.loc[cpi["symbol"] == "US500", "score"].iloc[0] < 0

    def test_every_event_covers_the_same_symbol_universe(self):
        universe = {e.symbol for e in NFP.exposures}
        assert len(universe) == 13
        for spec in EVENTS.values():
            assert {e.symbol for e in spec.exposures} == universe

    def test_fomc_moves_more_than_ppi_for_the_same_surprise(self):
        """Magnitude ordering is a claim the page makes; pin it."""
        fomc = score_instruments(FOMC, 1.0, BALANCED)
        ppi = score_instruments(PPI, 1.0, BALANCED)
        f = fomc.loc[fomc["symbol"] == "XAUUSD", "expected_move"].iloc[0]
        p = ppi.loc[ppi["symbol"] == "XAUUSD", "expected_move"].iloc[0]
        assert f > p


class TestPerEventTiming:
    def test_the_fomc_frame_names_the_presser(self):
        df = timing_frame(FOMC, date(2026, 9, 16))
        assert any("presser" in p.lower() for p in df["Phase"])

    def test_an_0830_frame_does_not(self):
        df = timing_frame(NFP, date(2026, 9, 4))
        assert not any("presser" in p.lower() for p in df["Phase"])

    def test_fomc_has_no_cash_open_phase(self):
        """09:30 New York is four and a half hours BEFORE an FOMC decision, so
        shifting the 08:30 frame would print a phase that already happened."""
        df = timing_frame(FOMC, date(2026, 9, 16))
        assert not any("cash open" in p.lower() for p in df["Phase"])

    def test_fomc_lands_in_the_sast_evening(self):
        t0 = release_datetime_sast(FOMC, date(2026, 9, 16))
        assert t0.hour >= 19

    def test_the_0830_events_land_in_the_sast_afternoon(self):
        for spec in (NFP, CPI, PPI):
            t0 = release_datetime_sast(spec, date(2026, 9, 4))
            assert 14 <= t0.hour <= 15

    def test_phases_are_ordered_and_labelled(self):
        for spec in EVENTS.values():
            df = timing_frame(spec, date(2026, 9, 16))
            assert list(df.columns) == ["Phase", "SAST", "What is happening"]
            assert len(df) >= 4


class TestPerEventSignals:
    def test_each_event_names_itself_in_the_thesis(self):
        for spec in EVENTS.values():
            sigs = board_to_signals(spec, score_instruments(spec, 2.0, BALANCED),
                                    date(2026, 9, 16), BALANCED, 2.0)
            assert sigs, spec.key
            assert all(spec.label in s["thesis"] for s in sigs)

    def test_the_registry_filter_still_holds_for_every_event(self):
        for spec in EVENTS.values():
            sigs = board_to_signals(spec, score_instruments(spec, 2.0, BALANCED),
                                    date(2026, 9, 16), BALANCED, 2.0)
            assert not {s["pair"] for s in sigs} & {"DXY", "US500", "BTCUSD"}

    def test_the_composite_gate_holds_for_every_event(self):
        for spec in EVENTS.values():
            assert board_to_signals(spec, score_instruments(spec, 0.0, BALANCED),
                                    date(2026, 9, 16), BALANCED, 0.0) == []


class TestPerEventChain:
    def test_the_trunk_follows_each_events_own_signs(self):
        s = compute_surprise(CPI, {"core_mm": 0.6, "core_mm_c": 0.3})
        chain, _ = chain_leaves(CPI, s,
                                score_instruments(CPI, s.composite, BALANCED))
        by_name = dict(chain)
        assert by_name["prices"] is True          # hawkish, sign +1
        assert by_name["real income"] is False    # hawkish, sign -1

    def test_the_leaves_are_read_off_the_board_for_every_event(self):
        for spec in EVENTS.values():
            s = compute_surprise(spec, {})
            _, leaves = chain_leaves(spec, s,
                                     score_instruments(spec, 2.0, BALANCED))
            assert [n for n, _ in leaves] == ["indices", "gold", "usd"]
