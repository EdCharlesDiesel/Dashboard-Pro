"""Unit tests for src/services/precomputed.py (pure serialization + freshness).

The DB/JSON store paths are best-effort I/O (same pattern as account_state) and
are exercised via the live DB in integration; here we pin the pure contract the
worker and the UI both depend on: a HouseView round-trips through the board
byte-for-byte in the fields the strip renders, and staleness is judged
correctly so a dead worker never serves an ancient board.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.core.bias import BULLISH, NEUTRAL, house_view
from src.services import precomputed as pc


def _trending_frame(n=80, start=1.0, step=0.01):
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    close = pd.Series([start + i * step for i in range(n)], index=idx)
    return pd.DataFrame({"Open": close, "High": close, "Low": close,
                         "Close": close, "Volume": 0.0}, index=idx)


class TestSerializationRoundTrip:
    def test_house_view_round_trips(self):
        frame = _trending_frame()
        view = house_view("EUR/USD", weekly=frame, daily=frame, h4=frame)
        assert view.direction == BULLISH        # a clean uptrend

        restored = pc.deserialize_house_view(pc.serialize_house_view(view))

        assert restored.pair == view.pair
        assert restored.direction == view.direction
        assert restored.score == pytest.approx(view.score)
        assert restored.aligned == view.aligned
        # The strip reads .timeframe(name).direction and .asof — both survive.
        for name in ("Weekly", "Daily", "4H"):
            assert restored.timeframe(name).direction == view.timeframe(name).direction
        assert restored.asof == view.asof

    def test_empty_view_round_trips(self):
        view = house_view("EUR/USD")            # no frames → NEUTRAL, no TFs
        assert view.direction == NEUTRAL and view.timeframes == ()
        restored = pc.deserialize_house_view(pc.serialize_house_view(view))
        assert restored.direction == NEUTRAL
        assert restored.timeframes == ()
        assert restored.asof is None


class TestBoardEnvelope:
    def test_build_board_stamps_utc(self):
        board = pc.build_board({"EUR/USD": {"hv": {}, "setup": None}})
        assert "computed_at" in board
        assert set(board["pairs"]) == {"EUR/USD"}
        assert pc.board_age_seconds(board) < 5

    def test_freshness_gate(self):
        fresh = pc.build_board({})
        assert pc.board_is_fresh(fresh, max_age_s=900)

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        stale = {"computed_at": old_ts, "pairs": {}}
        assert not pc.board_is_fresh(stale, max_age_s=900)

    def test_none_and_malformed_are_not_fresh(self):
        assert pc.board_age_seconds(None) is None
        assert not pc.board_is_fresh(None)
        assert pc.board_age_seconds({"pairs": {}}) is None       # no stamp
        assert not pc.board_is_fresh({"computed_at": "garbage"})


class TestLookups:
    def _board(self):
        frame = _trending_frame()
        view = house_view("EUR/USD", weekly=frame, daily=frame, h4=frame)
        return pc.build_board({
            "EUR/USD": {"hv": pc.serialize_house_view(view),
                        "setup": {"direction": "LONG", "grade": "A", "pct": 85}},
        })

    def test_house_view_from_board(self):
        v = pc.house_view_from_board("EUR/USD", self._board())
        assert v is not None and v.direction == BULLISH

    def test_missing_pair_returns_none(self):
        assert pc.house_view_from_board("GBP/USD", self._board()) is None
        assert pc.house_view_from_board("EUR/USD", None) is None
        assert pc.setup_from_board("EUR/USD", None) is None

    def test_setup_from_board(self):
        s = pc.setup_from_board("EUR/USD", self._board())
        assert s["grade"] == "A" and s["direction"] == "LONG"

    def test_board_pairs(self):
        assert pc.board_pairs(self._board()) == ["EUR/USD"]
        assert pc.board_pairs(None) == []
