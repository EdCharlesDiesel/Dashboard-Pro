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

    def test_checks_detail_as_dict_like_psycopg2_jsonb(self):
        # psycopg2 auto-deserializes JSONB — checks_detail arrives as a dict,
        # not a JSON string. The entry must still be found.
        bars = _bars([1.1050], lows=[1.1030], highs=[1.1055])
        row = _row(checks_detail={"entry": 1.1000})
        df = build_scorecard([row], {"EURUSD=X": bars})
        assert df.loc["setup_ranker", "wins"] == 1
