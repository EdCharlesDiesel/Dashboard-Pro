"""Unit tests for the Setup Ranker's 'why this trade' rationale template."""
from src.pages_lib.setup_ranker import trade_rationale_template


def _result(**over):
    base = {
        "pair": "EUR/USD",
        "direction": "LONG",
        "grade": "A",
        "score": 7,
        "max_score": 8,
        "pct": 87.5,
        "close": 1.1000,
        "sl_pips": 25,
        "spread_pct": 1.2,
        "quality_score": 3,
        "quality_max": 3,
        "scores": {
            "Weekly EMA": 1, "Daily Trend": 1, "Daily MACD": 1,
            "Weekly RSI": 0,
            "ATR Volatile": 1, "4H Zone": 1, "Spread/ATR": 1,
        },
    }
    base.update(over)
    return base


_LV = {"sl_price": 1.0975, "tp_price": 1.1050, "tp_pips": 50.0}


class TestTradeRationaleTemplate:
    def test_contains_grade_and_score(self):
        text = trade_rationale_template(_result(), _LV, 2.0)
        assert "EUR/USD LONG ranks Grade A" in text
        assert "7/8 direction criteria met (88%)" in text

    def test_lists_passed_and_failed_direction_criteria(self):
        text = trade_rationale_template(_result(), _LV, 2.0)
        assert "In favour: " in text and "Weekly EMA" in text
        assert "Missing: Weekly RSI." in text
        # Quality-gate criteria stay in the gate sentence, not the
        # direction-evidence lists.
        assert "In favour: " in text
        assert "ATR Volatile" not in text.split("Tradeability")[0]

    def test_quality_gate_all_clear_vs_failing(self):
        clear = trade_rationale_template(_result(), _LV, 2.0)
        assert "Tradeability gate 3/3 — all clear." in clear

        failing = trade_rationale_template(
            _result(quality_score=2,
                    scores={**_result()["scores"], "Spread/ATR": 0}),
            _LV, 2.0)
        assert "fails Spread/ATR" in failing

    def test_plan_levels_and_spread(self):
        text = trade_rationale_template(_result(), _LV, 2.0)
        assert "entry 1.10000" in text
        assert "stop 1.09750 (25p)" in text
        assert "target 1.10500 at 2.0R" in text
        assert "Spread is 1.2% of ATR." in text

    def test_missing_levels_omits_plan_sentence(self):
        text = trade_rationale_template(
            _result(), {"sl_price": None, "tp_price": None, "tp_pips": None}, 2.0)
        assert "Plan:" not in text
        assert "ranks Grade A" in text

    def test_no_failed_criteria_omits_missing_sentence(self):
        scores = {k: 1 for k in _result()["scores"]}
        text = trade_rationale_template(_result(scores=scores, score=8), _LV, 2.0)
        assert "Missing:" not in text
