"""Unit tests for src/debug/env.py — the local debug-mode flag."""
from __future__ import annotations

from src.debug.env import is_dev


class TestIsDev:
    def test_unset_is_prod(self, monkeypatch):
        monkeypatch.delenv("DASHBOARD_ENV", raising=False)
        assert is_dev() is False

    def test_dev_is_dev(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_ENV", "dev")
        assert is_dev() is True

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_ENV", "DEV")
        assert is_dev() is True

    def test_other_values_are_prod(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_ENV", "staging")
        assert is_dev() is False
