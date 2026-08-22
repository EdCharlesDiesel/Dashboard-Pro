"""Unit tests for src/services/source_scorecard.py."""
import json

import numpy as np
import pandas as pd
import pytest

from src.services.source_scorecard import (
    DEFAULT_HORIZON,
    HIT,
    LOSS,
    MARKED,
    MISS,
    OPEN,
    UNRESOLVED,
    WIN,
    build_scorecard,
    evaluate_row,
    profit_factor,
    quality_passed,
    row_horizon,
    sortino_ratio,
)


def _bars(closes, lows=None, highs=None, start="2026-01-05"):
    closes = list(closes)
    n = len(closes)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "Open": closes,
        "High": highs if highs is not None else [c + 0.0010 for c in closes],
        "Low": lows if lows is not None else [c - 0.0010 for c in closes],
        "Close": closes,
        "Volume": np.zeros(n),
    }, index=idx)


def _row(**over):
    base = {
        "source": "setup_ranker",
        "instrument": "EUR/USD",
        "ticker": "EURUSD=X",
        "direction": "Long",
        "logged_at": "2026-01-02",
        "is_open": True,
        "sl_pips": 20.0,
        "tp1_pips": 40.0,
        "checks_detail": json.dumps({"entry": 1.1000}),
        "r_multiple": None,
        "outcome": None,
    }
    base.update(over)
    return base


class TestRecordedOutcome:
    def test_closed_row_uses_recorded_r(self):
        res = evaluate_row(_row(is_open=False, r_multiple=2.5), bars=None)
        assert res.status == WIN and res.r == 2.5

    def test_closed_loss(self):
        res = evaluate_row(_row(is_open=False, r_multiple=-1.0), bars=None)
        assert res.status == LOSS and res.r == -1.0

    def test_closed_no_r_falls_back_to_outcome(self):
        res = evaluate_row(_row(is_open=False, outcome="WIN"), bars=None)
        assert res.status == WIN and res.r == 1.0


class TestTpSlReplay:
    def test_tp_hit_first_is_win_with_rr(self):
        # Long from 1.1000, SL 20 pips (1.0980), TP1 40 pips (1.1040).
        bars = _bars([1.1010, 1.1050], lows=[1.1000, 1.1030],
                     highs=[1.1020, 1.1055])
        res = evaluate_row(_row(), bars)
        assert res.status == WIN
        assert res.r == pytest.approx(2.0)  # 40/20 pips

    def test_stop_hit_first_is_loss(self):
        bars = _bars([1.0975, 1.1100], lows=[1.0970, 1.1050],
                     highs=[1.1005, 1.1100])
        res = evaluate_row(_row(), bars)
        assert res.status == LOSS and res.r == -1.0

    def test_same_bar_both_levels_counts_as_loss(self):
        # One bar spans both SL (1.0980) and TP (1.1040) → conservative loss.
        bars = _bars([1.1010], lows=[1.0970], highs=[1.1050])
        res = evaluate_row(_row(), bars)
        assert res.status == LOSS

    def test_short_direction_mirrors(self):
        # Short from 1.1000, SL 20 pips → 1.1020, TP 40 pips → 1.0960.
        bars = _bars([1.0955], lows=[1.0950], highs=[1.0990])
        res = evaluate_row(_row(direction="Short"), bars)
        assert res.status == WIN and res.r == pytest.approx(2.0)

    def test_neither_level_touched_is_open(self):
        bars = _bars([1.1005, 1.1010], lows=[1.0995, 1.1000],
                     highs=[1.1015, 1.1020])
        assert evaluate_row(_row(), bars).status == OPEN


class TestHorizonPaths:
    def test_no_target_marks_r_at_horizon(self):
        closes = [1.1000 + 0.0002 * i for i in range(DEFAULT_HORIZON + 2)]
        bars = _bars(closes, lows=[c - 0.0005 for c in closes],
                     highs=[c + 0.0005 for c in closes])
        res = evaluate_row(_row(tp1_pips=None), bars)
        assert res.status == MARKED
        assert res.r == pytest.approx(
            (closes[DEFAULT_HORIZON - 1] - 1.1000) / (20 * 0.0001))

    def test_no_stop_directional_hit(self):
        closes = [1.1050] * (DEFAULT_HORIZON + 1)
        bars = _bars(closes)
        res = evaluate_row(_row(sl_pips=None, tp1_pips=None), bars)
        assert res.status == HIT and res.hit is True

    def test_no_stop_directional_miss_for_short(self):
        closes = [1.1050] * (DEFAULT_HORIZON + 1)
        bars = _bars(closes)
        res = evaluate_row(_row(direction="Short", sl_pips=None, tp1_pips=None), bars)
        assert res.status == MISS and res.hit is False

    def test_horizon_not_reached_is_open(self):
        bars = _bars([1.1005] * 3)
        res = evaluate_row(_row(sl_pips=None, tp1_pips=None), bars)
        assert res.status == OPEN


class TestUnresolvable:
    def test_neutral_bias_is_unresolved(self):
        assert evaluate_row(_row(direction="Neutral"), _bars([1.1] * 5)).status == UNRESOLVED

    def test_missing_entry_is_unresolved(self):
        row = _row(checks_detail=json.dumps({}))
        assert evaluate_row(row, _bars([1.1] * 5)).status == UNRESOLVED

    def test_no_bars_yet_is_open(self):
        assert evaluate_row(_row(), None).status == OPEN

    def test_strong_sell_direction_is_evaluable(self):
        closes = [1.0950] * (DEFAULT_HORIZON + 1)
        res = evaluate_row(_row(direction="STRONG_SELL", sl_pips=None,
                                tp1_pips=None), _bars(closes))
        assert res.status == HIT


class TestBuildScorecard:
    def test_aggregates_and_ranks_by_expectancy(self):
        win_bars = _bars([1.1050], lows=[1.1030], highs=[1.1055])
        loss_bars = _bars([1.0970], lows=[1.0965], highs=[1.1005])
        rows = [
            _row(source="good_page"),                       # win (+2R)
            _row(source="good_page"),                       # win (+2R)
            _row(source="bad_page", ticker="GBPUSD=X",
                 instrument="GBP/USD"),                     # loss (−1R)
            _row(source="quiet_page", direction="Neutral"),  # unresolved
        ]
        bars_by_ticker = {"EURUSD=X": win_bars, "GBPUSD=X": loss_bars}
        df = build_scorecard(rows, bars_by_ticker)

        assert list(df.index[:2]) == ["good_page", "bad_page"]
        good = df.loc["good_page"]
        assert good["signals"] == 2 and good["wins"] == 2
        assert good["win_rate"] == 100.0
        assert good["expectancy_r"] == pytest.approx(2.0)
        bad = df.loc["bad_page"]
        assert bad["losses"] == 1 and bad["expectancy_r"] == pytest.approx(-1.0)
        assert df.loc["quiet_page", "unresolved"] == 1

    def test_recorded_outcomes_dominate(self):
        rows = [
            _row(source="mt4_import", is_open=False, r_multiple=1.5),
            _row(source="mt4_import", is_open=False, r_multiple=-1.0),
        ]
        df = build_scorecard(rows, {})
        row = df.loc["mt4_import"]
        assert row["resolved"] == 2
        assert row["win_rate"] == 50.0
        assert row["expectancy_r"] == pytest.approx(0.25)

    def test_empty_rows_empty_frame(self):
        assert build_scorecard([], {}).empty

    def test_missing_source_defaults_to_checklist(self):
        rows = [_row(source=None, is_open=False, r_multiple=1.0)]
        df = build_scorecard(rows, {})
        assert "checklist" in df.index

    def test_null_ticker_rows_are_tolerated(self):
        # A SQL NULL ticker arrives as float('nan') via pandas .to_dict().
        # NaN is TRUTHY, so it slips past naive `if row["ticker"]` guards —
        # this caused a TypeError in the Trade Journal when the ticker set was
        # sorted (float vs str). The scorecard itself must simply mark such a
        # row unresolved rather than raise. Real source: pages that take a
        # free-text ticker (smart_money) and fail to resolve one.
        rows = [
            _row(source="smart_money", ticker=float("nan")),
            _row(source="smart_money", ticker=None),
        ]
        df = build_scorecard(rows, {})            # no bars for either
        assert df.loc["smart_money", "signals"] == 2
        assert df.loc["smart_money", "resolved"] == 0   # nothing guessed
        assert df.loc["smart_money", "open"] == 2       # still-open, unpriceable

    def test_checks_detail_as_dict_like_psycopg2_jsonb(self):
        # psycopg2 auto-deserializes JSONB — checks_detail arrives as a dict,
        # not a JSON string. The entry must still be found.
        bars = _bars([1.1050], lows=[1.1030], highs=[1.1055])
        row = _row(checks_detail={"entry": 1.1000})
        df = build_scorecard([row], {"EURUSD=X": bars})
        assert df.loc["setup_ranker", "wins"] == 1


class TestPerRowHorizon:
    """A row's own horizon_days overrides the caller's default (added 2026-08-04)."""

    def _row(self, detail):
        return {"source": "s", "direction": "Long", "checks_detail": detail}

    def test_recorded_horizon_wins(self):
        assert row_horizon(self._row('{"horizon_days": 20}'), 10) == 20

    def test_dict_detail_works_too(self):
        # psycopg2 hands back JSONB already deserialized.
        assert row_horizon(self._row({"horizon_days": 5}), 10) == 5

    def test_missing_falls_back_to_default(self):
        assert row_horizon(self._row('{"entry": 1.1}'), 10) == 10

    def test_unparseable_detail_falls_back(self):
        assert row_horizon(self._row("not json"), 10) == 10

    def test_nonsense_horizon_falls_back(self):
        assert row_horizon(self._row('{"horizon_days": 0}'), 10) == 10
        assert row_horizon(self._row('{"horizon_days": "soon"}'), 10) == 10


class TestQualityPassedRead:
    def test_reads_recorded_flag(self):
        assert quality_passed({"checks_detail": '{"quality_passed": true}'}) is True
        assert quality_passed({"checks_detail": '{"quality_passed": false}'}) is False

    def test_unrecorded_is_none_not_false(self):
        # None means "this page never told us", which is different from a
        # recorded failure — conflating them would bias any segmentation.
        assert quality_passed({"checks_detail": "{}"}) is None


# ===========================================================================
# Distribution-shape statistics
# ===========================================================================
# Expectancy is a mean, and a mean hides how it was earned. These two say
# something a mean cannot. Both are borrowed from nautilus_trader's
# crates/analysis/src/statistics, adapted to a per-signal R series.

class TestProfitFactor:
    def test_gross_win_over_gross_loss(self):
        assert profit_factor([2.0, -1.0, 1.0, -1.0]) == pytest.approx(1.5)

    def test_all_winners_is_undefined_not_infinite(self):
        """No losses means the ratio has no denominator. Printing infinity
        would claim a certainty the sample does not support; the wins/losses
        columns already show why the cell is blank."""
        assert profit_factor([1.0, 2.0]) is None

    def test_all_losers_is_zero(self):
        assert profit_factor([-1.0, -2.0]) == pytest.approx(0.0)

    def test_empty_is_none(self):
        assert profit_factor([]) is None

    def test_break_even_is_one(self):
        assert profit_factor([1.0, -1.0]) == pytest.approx(1.0)

    def test_zero_r_rows_count_as_neither(self):
        """A marked-to-close row can land exactly flat. It is not a win and
        not a loss, so it must not move the ratio in either direction."""
        assert profit_factor([2.0, -1.0, 0.0]) == pytest.approx(2.0)

    def test_non_finite_values_are_ignored(self):
        assert profit_factor([2.0, -1.0, float("nan")]) == pytest.approx(2.0)


class TestSortinoRatio:
    def test_a_symmetric_series_scores_zero(self):
        assert sortino_ratio([1.0, -1.0, 1.0, -1.0]) == pytest.approx(0.0)

    def test_downside_deviation_uses_all_observations(self):
        """Sortino & Price divide the squared downside by the FULL count, not
        the count of losers. r = [2, -1, 2, -1]: mean 0.5,
        dd = sqrt((0 + 1 + 0 + 1) / 4) = 0.7071 -> 0.7071."""
        assert sortino_ratio([2.0, -1.0, 2.0, -1.0]) == pytest.approx(
            0.5 / (0.5 ** 0.5), rel=1e-6)

    def test_no_downside_is_undefined(self):
        assert sortino_ratio([1.0, 2.0, 3.0]) is None

    def test_fewer_than_two_observations_is_none(self):
        assert sortino_ratio([1.0]) is None
        assert sortino_ratio([]) is None

    def test_a_nonzero_mar_shifts_the_threshold(self):
        # Against a 1R hurdle, [1, 1] has no excess and no downside.
        assert sortino_ratio([1.0, 1.0], mar=1.0) is None

    def test_a_nonzero_mar_creates_downside(self):
        # Against a 1R hurdle, a +0.5R signal is a shortfall.
        assert sortino_ratio([1.5, 0.5], mar=1.0) == pytest.approx(0.0)

    def test_upside_outliers_do_not_penalise(self):
        """The whole reason to prefer Sortino over Sharpe here: replacing a
        win with a much bigger win must never lower the ratio."""
        base = sortino_ratio([1.0, -1.0, 1.0, -1.0, 1.0])
        spiky = sortino_ratio([1.0, -1.0, 1.0, -1.0, 9.0])
        assert base is not None and spiky is not None and spiky > base

    def test_deeper_losses_lower_the_ratio(self):
        shallow = sortino_ratio([2.0, -0.5, 2.0, -0.5])
        deep = sortino_ratio([2.0, -2.0, 2.0, -2.0])
        assert shallow is not None and deep is not None and shallow > deep

    def test_it_is_not_annualised(self):
        """A per-signal ratio, deliberately. If this ever grows a sqrt(252)
        it is asserting a time basis these irregularly-spaced R-multiples do
        not have — different sources are already marked on different horizons.
        [2, -1, 2, -1] annualised would be ~11.2, not ~0.71."""
        assert sortino_ratio([2.0, -1.0, 2.0, -1.0]) < 1.0

    def test_non_finite_values_are_ignored(self):
        clean = sortino_ratio([2.0, -1.0, 2.0, -1.0])
        dirty = sortino_ratio([2.0, -1.0, 2.0, -1.0, float("inf")])
        assert dirty == pytest.approx(clean)


class TestScorecardCarriesTheNewColumns:
    def test_columns_are_present_and_populated(self):
        rows = [
            _row(source="mixed", is_open=False, r_multiple=2.0),
            _row(source="mixed", is_open=False, r_multiple=-1.0),
            _row(source="mixed", is_open=False, r_multiple=1.0),
            _row(source="mixed", is_open=False, r_multiple=-1.0),
        ]
        df = build_scorecard(rows, {})
        assert df.loc["mixed", "profit_factor"] == pytest.approx(1.5)
        assert not pd.isna(df.loc["mixed", "sortino"])

    def test_a_source_with_no_losses_reports_neither(self):
        rows = [_row(source="clean", is_open=False, r_multiple=1.0),
                _row(source="clean", is_open=False, r_multiple=2.0)]
        df = build_scorecard(rows, {})
        assert pd.isna(df.loc["clean", "profit_factor"])
        assert pd.isna(df.loc["clean", "sortino"])

    def test_an_unresolved_source_reports_neither(self):
        df = build_scorecard([_row(source="quiet", direction="Neutral")], {})
        assert pd.isna(df.loc["quiet", "profit_factor"])
        assert pd.isna(df.loc["quiet", "sortino"])

    def test_the_ranking_metric_is_unchanged(self):
        """New columns, not a new sort — re-ranking the board was not asked
        for and is a separate decision."""
        rows = [_row(source="a", is_open=False, r_multiple=3.0),
                _row(source="b", is_open=False, r_multiple=0.5)]
        df = build_scorecard(rows, {})
        assert list(df.index) == ["a", "b"]

    def test_profit_factor_separates_two_sources_on_the_same_expectancy(self):
        """The reason these columns exist. Both sources average +0.25R; one
        grinds it out, the other is carried by a single outlier."""
        # Both series sum to +2.0R over 8 signals, so expectancy is identical
        # at +0.25R and only the shape differs.
        grind = [_row(source="grind", is_open=False, r_multiple=r)
                 for r in (1.0, -0.5, 1.0, -0.5, 1.0, -0.5, 1.0, -0.5)]
        spike = [_row(source="spike", is_open=False, r_multiple=r)
                 for r in (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 9.0)]
        df = build_scorecard(grind + spike, {})
        assert df.loc["grind", "expectancy_r"] == pytest.approx(
            df.loc["spike", "expectancy_r"])
        assert df.loc["grind", "profit_factor"] > df.loc["spike", "profit_factor"]
        assert df.loc["grind", "sortino"] > df.loc["spike", "sortino"]
