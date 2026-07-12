"""Unit tests for src/services/prediction_service.py — the Instrument
Predictor's composite-signal aggregator. Pure logic, synthetic components
only; no yfinance/Postgres/Streamlit."""
from __future__ import annotations

import pytest

from src.services.prediction_service import Component, aggregate, label_for


class TestLabelFor:
    @pytest.mark.parametrize("composite,expected", [
        (0.9, "STRONG_BUY"), (0.5, "STRONG_BUY"), (0.3, "BUY"), (0.15, "BUY"),
        (0.0, "NEUTRAL"), (-0.1, "NEUTRAL"), (-0.15, "SELL"), (-0.4, "SELL"),
        (-0.5, "STRONG_SELL"), (-0.9, "STRONG_SELL"),
    ])
    def test_bands(self, composite, expected):
        assert label_for(composite) == expected


class TestAggregate:
    def test_no_components_is_neutral_low(self):
        pred = aggregate([])
        assert pred.label == "NEUTRAL"
        assert pred.confidence == "Low"
        assert pred.composite == 0.0

    def test_all_unavailable_components_is_neutral(self):
        comps = [Component("a", 0.9, weight=0.0), Component("b", -0.9, weight=0.0)]
        pred = aggregate(comps)
        assert pred.label == "NEUTRAL"
        assert pred.components == []

    def test_unanimous_bullish_is_strong_buy_high_confidence(self):
        comps = [
            Component("setup_score", 0.8, weight=0.30),
            Component("trend_signal", 1.0, weight=0.25),
            Component("currency_strength", 0.6, weight=0.25),
            Component("cot_composite", 0.7, weight=0.20),
        ]
        pred = aggregate(comps)
        assert pred.label == "STRONG_BUY"
        assert pred.confidence == "High"
        assert pred.agreement == 1.0

    def test_unanimous_bearish_is_strong_sell(self):
        comps = [
            Component("setup_score", -0.8, weight=0.30),
            Component("trend_signal", -1.0, weight=0.25),
        ]
        pred = aggregate(comps)
        assert pred.label == "STRONG_SELL"
        assert pred.confidence == "High"

    def test_conflicting_components_lower_confidence(self):
        comps = [
            Component("setup_score", 0.6, weight=0.30),
            Component("trend_signal", -0.6, weight=0.25),
            Component("currency_strength", 0.5, weight=0.25),
            Component("cot_composite", -0.5, weight=0.20),
        ]
        pred = aggregate(comps)
        assert pred.agreement == pytest.approx(0.5)
        assert pred.confidence in ("Low", "Medium")

    def test_missing_component_renormalizes_not_dilutes(self):
        """Two components agreeing strongly should give the same composite
        whether or not a third, unavailable (weight=0) component is present —
        a missing input must not silently drag the score toward zero."""
        with_missing = aggregate([
            Component("setup_score", 0.8, weight=0.30),
            Component("trend_signal", 0.8, weight=0.25),
            Component("currency_strength", 0.0, weight=0.0),  # unavailable
        ])
        without_missing = aggregate([
            Component("setup_score", 0.8, weight=0.30),
            Component("trend_signal", 0.8, weight=0.25),
        ])
        assert with_missing.composite == pytest.approx(without_missing.composite)

    def test_high_vol_regime_caps_confidence_at_medium(self):
        comps = [
            Component("setup_score", 0.9, weight=0.30),
            Component("trend_signal", 0.9, weight=0.25),
            Component("currency_strength", 0.9, weight=0.25),
        ]
        calm = aggregate(comps, vol_regime="normal")
        choppy = aggregate(comps, vol_regime="high")
        assert calm.confidence == "High"
        assert choppy.confidence == "Medium"

    def test_composite_is_clipped_to_unit_range(self):
        comps = [Component("a", 5.0, weight=1.0)]  # pathological out-of-range input
        pred = aggregate(comps)
        assert -1.0 <= pred.composite <= 1.0

    def test_weighted_average_math(self):
        comps = [
            Component("a", 1.0, weight=0.75),
            Component("b", -1.0, weight=0.25),
        ]
        pred = aggregate(comps)
        assert pred.composite == pytest.approx(0.5)
