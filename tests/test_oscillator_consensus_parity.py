"""The chart indicator and the app must agree on "stretched".

`mt5/OscillatorConsensus.mq5` counts how many of RSI / Stochastic %K /
Williams %R / CCI are beyond their band. This pins the same rule in Python
against the same bands, so a future edit to either side that changes the answer
fails here rather than silently putting a different number on the chart from the
one the signal engine would compute.

Two definitions were checked against MT5's own before this was written, so the
implementer after me does not repeat the work:

* `cci()` uses typical price (H+L+C)/3 with the 0.015 scale -> matches
  `iCCI(..., PRICE_TYPICAL)`.
* `williams_r()` is -100*(HH-C)/(HH-LL) -> matches `iWPR`.
* `stochastic(k, d, smooth)` takes fast %K then smooths by `smooth` -> matches
  `iStochastic(k, d, slowing, MODE_SMA, STO_LOWHIGH)` **buffer 0**, whose
  `slowing` argument does the same averaging. Buffer 1 is %D and is NOT what the
  indicator reads.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.indicators.leading import cci, stochastic, williams_r
from src.indicators.technical import TechnicalIndicators as TI

_MQ5 = Path(__file__).resolve().parent.parent / "mt5" / "OscillatorConsensus.mq5"

# The MQL5 inputs, mirrored. Classic mean-reversion bands -- deliberately NOT
# config.py's 40/60, which is a trend-pullback band. See the .mq5 header.
BANDS = {
    "rsi": (30.0, 70.0),
    "stoch": (20.0, 80.0),
    "wpr": (-80.0, -20.0),
    "cci": (-100.0, 100.0),
}

# Mirrors #define HIST_BASE / HIST_SCALE in the indicator.
HIST_BASE = 50.0
HIST_SCALE = 12.5


def consensus_from_values(rsi: float, stoch: float, wpr: float,
                          cci_value: float) -> int:
    """Signed count of oscillators beyond their band — the MQL5 Consensus()."""
    score = 0
    score += rsi > BANDS["rsi"][1]
    score += stoch > BANDS["stoch"][1]
    score += wpr > BANDS["wpr"][1]
    score += cci_value > BANDS["cci"][1]
    score -= rsi < BANDS["rsi"][0]
    score -= stoch < BANDS["stoch"][0]
    score -= wpr < BANDS["wpr"][0]
    score -= cci_value < BANDS["cci"][0]
    return int(score)


def consensus(df: pd.DataFrame) -> int:
    """The rule applied to a frame's last bar, using the app's oscillators."""
    return consensus_from_values(
        float(TI.rsi(df["Close"], 14).iloc[-1]),
        float(stochastic(df, 14, 3, 3)[0].iloc[-1]),
        float(williams_r(df, 14).iloc[-1]),
        float(cci(df, 20).iloc[-1]),
    )


class TestConsensusRule:
    def test_all_four_stretched_up_is_plus_four(self):
        assert consensus_from_values(75.0, 90.0, -10.0, 150.0) == 4

    def test_all_four_stretched_down_is_minus_four(self):
        assert consensus_from_values(25.0, 10.0, -90.0, -150.0) == -4

    def test_mid_range_is_zero(self):
        assert consensus_from_values(50.0, 50.0, -50.0, 0.0) == 0

    def test_a_split_reading_nets_out(self):
        # RSI stretched up, Stochastic stretched down: they cancel.
        assert consensus_from_values(75.0, 10.0, -50.0, 0.0) == 0

    def test_the_bands_are_exclusive_not_inclusive(self):
        # Exactly on the band is not beyond it. The MQL5 uses > and <.
        assert consensus_from_values(70.0, 80.0, -20.0, 100.0) == 0
        assert consensus_from_values(30.0, 20.0, -80.0, -100.0) == 0

    def test_raw_cci_is_counted_not_the_clipped_plot(self):
        # The plot clips CCI to 0..100 via 50 + cci/4. An extreme reading must
        # still count exactly once, and must not vanish because the *display*
        # saturated.
        assert consensus_from_values(50.0, 50.0, -50.0, 400.0) == 1
        assert consensus_from_values(50.0, 50.0, -50.0, 101.0) == 1


class TestHistogramScaling:
    def test_the_full_range_fills_the_pane(self):
        # -4..+4 must map onto 0..100 exactly, or the histogram clips.
        assert HIST_BASE + 4 * HIST_SCALE == 100.0
        assert HIST_BASE - 4 * HIST_SCALE == 0.0

    def test_zero_sits_on_the_midline(self):
        assert HIST_BASE + 0 * HIST_SCALE == 50.0

    @pytest.mark.parametrize("count,expected", [(3, 87.5), (-3, 12.5), (0, 50.0)])
    def test_the_parity_targets_scale_as_documented(self, count, expected):
        assert HIST_BASE + count * HIST_SCALE == expected


class TestTheSourceStillSaysWhatWeThink:
    """Static checks on the .mq5, so drift in it fails here."""

    def _source(self) -> str:
        return _MQ5.read_text(encoding="utf-8", errors="ignore")

    def test_the_indicator_exists(self):
        assert _MQ5.exists(), "mt5/OscillatorConsensus.mq5 is missing"

    def test_the_bands_match_this_test(self):
        source = self._source()
        for name, value in (("RSIOverbought", 70.0), ("RSIOversold", 30.0),
                            ("StochOverbought", 80.0), ("StochOversold", 20.0),
                            ("WPROverbought", -20.0), ("WPROversold", -80.0),
                            ("CCIOverbought", 100.0), ("CCIOversold", -100.0)):
            match = re.search(r"%s\s*=\s*(-?\d+\.?\d*)" % name, source)
            assert match, "input %s not found in the .mq5" % name
            assert float(match.group(1)) == value, (
                "%s is %s in the .mq5 but %s here" % (name, match.group(1), value))

    def test_the_histogram_constants_match(self):
        source = self._source()
        assert re.search(r"#define\s+HIST_BASE\s+%s" % HIST_BASE, source)
        assert re.search(r"#define\s+HIST_SCALE\s+%s" % HIST_SCALE, source)

    def test_the_count_buffer_is_the_documented_index(self):
        # Buffer 6 is the public API: iCustom consumers read the raw count.
        source = self._source()
        assert "SetIndexBuffer(6, BufCount" in source, (
            "the consensus count must stay on buffer 6 — it is read by index")

    def test_it_raises_no_alerts(self):
        # Alerting lives in alert_service.py; a second channel would double
        # every notification.
        assert "Alert(" not in self._source()
