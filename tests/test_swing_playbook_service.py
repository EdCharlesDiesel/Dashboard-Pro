"""save_thesis/load_thesis must be able to target a specific week (Week
Ahead's "next week" pre-flight), while Swing Playbook's existing no-argument
call sites keep defaulting to today's Monday, unchanged."""
from __future__ import annotations

from datetime import date

from src.services import swing_playbook_service as sp


class _FakeRepo:
    def __init__(self):
        self.save_calls = []
        self.load_calls = []

    def save_swing_thesis(self, instrument, week_start, bias, invalidation):
        self.save_calls.append((instrument, week_start, bias, invalidation))

    def load_swing_thesis(self, instrument, week_start):
        self.load_calls.append((instrument, week_start))
        return {"bias": "Long", "invalidation": "x"}


def test_save_thesis_uses_explicit_week_start_when_given(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(sp, "_repo", lambda: fake)

    explicit = date(2026, 8, 24)
    ok = sp.save_thesis("EUR/USD", "Bullish", "daily close < 1.16", week_start_override=explicit)

    assert ok is True
    assert fake.save_calls == [("EUR/USD", explicit, "Bullish", "daily close < 1.16")]


def test_save_thesis_defaults_to_current_week_when_omitted(monkeypatch):
    """Swing Playbook's existing call sites pass nothing — must be untouched."""
    fake = _FakeRepo()
    monkeypatch.setattr(sp, "_repo", lambda: fake)
    monkeypatch.setattr(sp, "week_start", lambda now=None: date(2026, 8, 17))

    ok = sp.save_thesis("EUR/USD", "Neutral", "y")

    assert ok is True
    assert fake.save_calls == [("EUR/USD", date(2026, 8, 17), "Neutral", "y")]


def test_load_thesis_uses_explicit_week_start_when_given(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(sp, "_repo", lambda: fake)

    explicit = date(2026, 8, 24)
    result = sp.load_thesis("EUR/USD", week_start_override=explicit)

    assert result == {"bias": "Long", "invalidation": "x"}
    assert fake.load_calls == [("EUR/USD", explicit)]


def test_load_thesis_defaults_to_current_week_when_omitted(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(sp, "_repo", lambda: fake)
    monkeypatch.setattr(sp, "week_start", lambda now=None: date(2026, 8, 17))

    sp.load_thesis("EUR/USD")

    assert fake.load_calls == [("EUR/USD", date(2026, 8, 17))]
