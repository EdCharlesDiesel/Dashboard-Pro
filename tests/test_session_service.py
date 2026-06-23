"""Unit tests for src/services/session_service.py — Kill Zone classifier."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.services.session_service import SessionService


def _utc(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 6, 23, hour, minute, tzinfo=timezone.utc)


class TestPrimeWindows:
    def test_london_kill_zone(self):
        s = SessionService.current(_utc(8))
        assert s["window"] == "London Kill Zone"
        assert s["prime"] is True
        assert s["icon"] == "🟢"

    def test_ny_kill_zone(self):
        s = SessionService.current(_utc(13))
        assert s["window"] == "NY Kill Zone"
        assert s["prime"] is True

    def test_london_close_not_prime(self):
        s = SessionService.current(_utc(16))
        assert s["window"] == "London Close"
        assert s["prime"] is False
        assert s["icon"] == "🟡"

    def test_tokyo_avoid(self):
        s = SessionService.current(_utc(1))
        assert s["window"] == "Tokyo Session"
        assert s["prime"] is False
        assert s["icon"] == "🔴"


class TestBoundaries:
    def test_london_open_edge_is_inclusive(self):
        assert SessionService.current(_utc(7, 0))["window"] == "London Kill Zone"

    def test_london_close_edge_is_exclusive(self):
        # 09:00 is no longer in the London KZ (upper bound exclusive).
        assert SessionService.current(_utc(9, 0))["window"] != "London Kill Zone"

    def test_minutes_count_toward_fractional_hour(self):
        # 08:59 still inside London KZ, 09:01 outside.
        assert SessionService.current(_utc(8, 59))["window"] == "London Kill Zone"
        assert SessionService.current(_utc(9, 1))["window"] != "London Kill Zone"


class TestDeadZone:
    def test_dead_zone_between_sessions(self):
        s = SessionService.current(_utc(5))
        assert s["window"] == "Dead Zone"
        assert s["prime"] is False
        assert s["icon"] == "⚫"

    def test_dead_zone_counts_down_to_next_prime(self):
        # 05:00 → next prime is London KZ at 07:00 → 120 minutes.
        s = SessionService.current(_utc(5))
        assert "120m until London KZ" in s["desc"]

    def test_late_evening_rolls_to_next_london(self):
        s = SessionService.current(_utc(20))
        assert s["window"] == "Dead Zone"
        assert "London KZ" in s["desc"]


def test_time_string_format():
    s = SessionService.current(_utc(8, 30))
    assert s["time"] == "08:30 UTC"


def test_defaults_to_now(monkeypatch):
    # current() with no arg should not raise and must return a valid window.
    s = SessionService.current()
    assert s["window"] in {
        "London Kill Zone", "NY Kill Zone", "London Close",
        "Tokyo Session", "Dead Zone",
    }
