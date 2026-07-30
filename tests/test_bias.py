"""Unit tests for src/core/bias.py — the canonical bias engine."""
import numpy as np
import pandas as pd
import pytest

from src.core.bias import (
    BEARISH,
    BULLISH,
    NEUTRAL,
    HouseView,
    TF_WEIGHTS,
    dissenting_timeframes,
    house_view,
    is_conflict,
    normalize_direction,
    timeframe_agreement,
    timeframe_bias,
)


def _ohlc(n: int, slope: float = 0.5, start: float = 100.0,
          freq: str = "D") -> pd.DataFrame:
    close = np.arange(n, dtype=float) * slope + start
    idx = pd.date_range("2025-01-01", periods=n, freq=freq)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )


# ── timeframe_bias ──────────────────────────────────────────────────────────

class TestTimeframeBias:
    def test_uptrend_is_bullish(self):
        tf = timeframe_bias(_ohlc(120, slope=0.5), "Daily")
        assert tf is not None
        assert tf.direction == BULLISH
        assert tf.score == 1.0
        assert all(tf.votes.values())
        assert tf.ema_fast > tf.ema_slow
        assert tf.price > tf.ema_fast

    def test_downtrend_is_bearish(self):
        tf = timeframe_bias(_ohlc(120, slope=-0.5, start=200.0), "Daily")
        assert tf is not None
        assert tf.direction == BEARISH
        assert tf.score == -1.0
        assert not any(tf.votes.values())

    def test_mixed_votes_are_neutral(self):
        # Long uptrend, then a sharp drop: EMA20 still above EMA50 but price
        # punches below both → 1 bull vote vs 2 bear votes → NEUTRAL.
        df = _ohlc(120, slope=0.5)
        df.iloc[-1, df.columns.get_loc("Close")] = 100.0
        tf = timeframe_bias(df, "Daily")
        assert tf is not None
        assert tf.direction == NEUTRAL
        assert -1.0 < tf.score < 1.0

    def test_too_short_history_returns_none(self):
        assert timeframe_bias(_ohlc(30), "Weekly") is None

    def test_none_and_empty_frames_return_none(self):
        assert timeframe_bias(None, "Daily") is None
        assert timeframe_bias(pd.DataFrame(), "Daily") is None

    def test_asof_is_last_bar_timestamp(self):
        df = _ohlc(120)
        tf = timeframe_bias(df, "Daily")
        assert tf.asof == df.index[-1]

    def test_score_and_direction_scale_matches_master(self):
        # The rule must match score_setup's weekly criterion: EMA20 > EMA50
        # AND price above both = uptrend. Unanimity required.
        df = _ohlc(120, slope=0.5)
        tf = timeframe_bias(df, "Weekly")
        assert tf.direction == BULLISH
        assert set(tf.votes) == {"EMA20 > EMA50", "Price > EMA20", "Price > EMA50"}


# ── house_view ──────────────────────────────────────────────────────────────

class TestHouseView:
    def test_all_timeframes_agree_bullish(self):
        up = _ohlc(120, slope=0.5)
        hv = house_view("EUR/USD", weekly=up, daily=up, h4=up)
        assert hv.direction == BULLISH
        assert hv.score == 1.0
        assert hv.aligned
        assert len(hv.timeframes) == 3

    def test_all_timeframes_agree_bearish(self):
        dn = _ohlc(120, slope=-0.5, start=200.0)
        hv = house_view("EUR/USD", weekly=dn, daily=dn, h4=dn)
        assert hv.direction == BEARISH
        assert hv.aligned

    def test_weekly_daily_conflict_is_neutral(self):
        up = _ohlc(120, slope=0.5)
        dn = _ohlc(120, slope=-0.5, start=200.0)
        hv = house_view("EUR/USD", weekly=up, daily=dn, h4=None)
        # Weekly +1 (w=3), Daily −1 (w=2) → composite +0.2 → NEUTRAL.
        assert hv.direction == NEUTRAL
        assert not hv.aligned
        assert hv.score == pytest.approx((3.0 - 2.0) / 5.0)

    def test_missing_timeframe_renormalizes_not_drags(self):
        up = _ohlc(120, slope=0.5)
        hv = house_view("EUR/USD", weekly=up, daily=up, h4=None)
        # 4H unavailable must not pull the composite toward zero.
        assert hv.score == 1.0
        assert hv.direction == BULLISH
        assert hv.timeframe("4H") is None
        assert hv.timeframe("Weekly") is not None

    def test_no_data_at_all(self):
        hv = house_view("EUR/USD")
        assert hv.direction == NEUTRAL
        assert hv.score == 0.0
        assert not hv.aligned
        assert hv.timeframes == ()
        assert hv.asof is None

    def test_weights_favor_weekly(self):
        assert TF_WEIGHTS["Weekly"] > TF_WEIGHTS["Daily"] > TF_WEIGHTS["4H"]

    def test_asof_is_freshest_timeframe(self):
        up_d = _ohlc(120, slope=0.5, freq="D")
        up_h4 = _ohlc(120, slope=0.5, freq="4h")
        hv = house_view("EUR/USD", daily=up_d, h4=up_h4)
        assert hv.asof == max(up_d.index[-1], up_h4.index[-1])

    def test_neutral_timeframe_breaks_alignment(self):
        up = _ohlc(120, slope=0.5)
        mixed = _ohlc(120, slope=0.5)
        mixed.iloc[-1, mixed.columns.get_loc("Close")] = 100.0
        hv = house_view("EUR/USD", weekly=up, daily=up, h4=mixed)
        assert not hv.aligned


# ── canonical parameter drift guard ─────────────────────────────────────────

class TestCanonicalParams:
    def test_bias_emas_derive_from_config(self):
        from src.core.bias import EMA_FAST, EMA_SLOW
        from src.core.config import default_config as cfg
        assert (EMA_FAST, EMA_SLOW) == (cfg.ema_fast, cfg.ema_slow)

    def test_config_matches_score_setup_literals(self):
        # score_setup (src/core/signals.py) hard-codes ema(close, 20)/ema(close, 50)
        # for its weekly/daily trend criteria. If the canonical pair is ever
        # retuned, those literals must move with it — this guard makes the
        # drift loud instead of silent.
        from src.core.config import default_config as cfg
        assert cfg.ema_fast == 20
        assert cfg.ema_slow == 50
        assert cfg.ema_long == 200
        assert (cfg.macd_fast, cfg.macd_slow, cfg.macd_signal) == (12, 26, 9)


# ── vocabulary + conflict helpers ───────────────────────────────────────────

class TestTimeframeAgreement:
    """Which frame dissents — the thing a Grade-A score can't tell you.

    Modelled on the real EUR/AUD case: weekly bearish, 4H already flipped
    bullish on a counter-trend bounce, so the ranker graded SHORT 'A' while the
    board sat NEUTRAL. The breakdown must name the 4H as the dissenter.
    """
    def _split_view(self):
        # weekly down, daily flat-ish, 4H up  ->  composite lands NEUTRAL
        return house_view(
            "EUR/AUD",
            weekly=_ohlc(120, slope=-0.5, freq="W"),
            daily=_ohlc(120, slope=-0.5),
            h4=_ohlc(120, slope=+0.5, freq="4h"),
        )

    def test_names_the_dissenting_timeframe(self):
        view = self._split_view()
        rows = {r["timeframe"]: r for r in timeframe_agreement(view, "SHORT")}
        assert rows["Weekly"]["agrees"] is True and rows["Weekly"]["opposes"] is False
        assert rows["4H"]["opposes"] is True      # bullish 4H against a SHORT
        assert dissenting_timeframes(view, "SHORT") == ("4H",)

    def test_flips_with_the_trade_direction(self):
        view = self._split_view()
        assert dissenting_timeframes(view, "LONG") == ("Weekly", "Daily")

    def test_neutral_frame_neither_agrees_nor_opposes(self):
        # An absent opinion is not opposition.
        view = house_view("X", weekly=_ohlc(120, slope=0.0, freq="W"))
        row = timeframe_agreement(view, "LONG")[0]
        assert row["direction"] == NEUTRAL
        assert row["agrees"] is False and row["opposes"] is False

    def test_carries_weight_and_score(self):
        view = self._split_view()
        for row in timeframe_agreement(view, "SHORT"):
            assert row["weight"] == TF_WEIGHTS[row["timeframe"]]
            assert -1.0 <= row["score"] <= 1.0

    def test_only_available_timeframes_appear(self):
        view = house_view("X", weekly=_ohlc(120, slope=0.5, freq="W"))
        assert [r["timeframe"] for r in timeframe_agreement(view, "LONG")] == ["Weekly"]

    def test_empty_view_is_safe(self):
        view = house_view("X")           # no frames at all
        assert timeframe_agreement(view, "LONG") == ()
        assert dissenting_timeframes(view, "LONG") == ()

    def test_unknown_direction_opposes_nothing(self):
        view = self._split_view()
        assert dissenting_timeframes(view, "NEUTRAL") == ()
        assert dissenting_timeframes(view, None) == ()


class TestVocabulary:
    @pytest.mark.parametrize("word", ["Bullish", "BUY", "STRONG_BUY", "Long", "up"])
    def test_bull_words(self, word):
        assert normalize_direction(word) == BULLISH

    @pytest.mark.parametrize("word", ["Bearish", "SELL", "STRONG_SELL", "Short", "down"])
    def test_bear_words(self, word):
        assert normalize_direction(word) == BEARISH

    @pytest.mark.parametrize("word", [None, "", "Neutral", "MIXED", "whatever"])
    def test_everything_else_is_neutral(self, word):
        assert normalize_direction(word) == NEUTRAL

    def test_conflict_only_when_opposite(self):
        assert is_conflict(BULLISH, "Short")
        assert is_conflict(BEARISH, "BUY")
        assert not is_conflict(BULLISH, "Long")
        assert not is_conflict(NEUTRAL, "Long")
        assert not is_conflict(BULLISH, "Neutral")
        assert not is_conflict(BULLISH, None)
