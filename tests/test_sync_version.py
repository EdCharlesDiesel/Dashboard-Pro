"""Unit tests for deploy/sync_version.py.

`.env` is the only place the Docker containers can be handed a secret — it is
gitignored, and compose interpolates from it automatically. `write_env` used to
open it with "w" and write nothing but the header and APP_VERSION, so every
version bump silently deleted anything else in the file. On 2026-08-20 there
were six bumps in one working day; a TELEGRAM_BOT_TOKEN line would not have
survived the afternoon.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    """Import deploy/sync_version.py by path — `deploy/` is not a package."""
    path = os.path.join(_REPO, "deploy", "sync_version.py")
    spec = importlib.util.spec_from_file_location("sync_version_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sv():
    return _load()


class TestWriteEnvPreservesOtherKeys:
    def test_it_keeps_unrelated_keys(self, sv, tmp_path):
        p = tmp_path / ".env"
        p.write_text("APP_VERSION=1.0.0\nTELEGRAM_BOT_TOKEN=abc\n", encoding="utf-8")
        sv.write_env("1.2.3", str(p))
        text = p.read_text(encoding="utf-8")
        assert "TELEGRAM_BOT_TOKEN=abc" in text
        assert "APP_VERSION=1.2.3" in text
        assert "APP_VERSION=1.0.0" not in text

    def test_it_does_not_duplicate_app_version(self, sv, tmp_path):
        p = tmp_path / ".env"
        p.write_text("APP_VERSION=1.0.0\n", encoding="utf-8")
        sv.write_env("1.2.3", str(p))
        assert p.read_text(encoding="utf-8").count("APP_VERSION=") == 1

    def test_it_creates_the_file_when_absent(self, sv, tmp_path):
        p = tmp_path / ".env"
        sv.write_env("1.2.3", str(p))
        assert "APP_VERSION=1.2.3" in p.read_text(encoding="utf-8")

    def test_a_missing_app_version_line_is_added(self, sv, tmp_path):
        p = tmp_path / ".env"
        p.write_text("TELEGRAM_CHAT_ID=42\n", encoding="utf-8")
        sv.write_env("1.2.3", str(p))
        text = p.read_text(encoding="utf-8")
        assert "TELEGRAM_CHAT_ID=42" in text and "APP_VERSION=1.2.3" in text

    def test_repeated_bumps_are_stable(self, sv, tmp_path):
        # Two bumps must not accumulate headers, blank lines or duplicates.
        p = tmp_path / ".env"
        p.write_text("TELEGRAM_CHAT_ID=42\n", encoding="utf-8")
        sv.write_env("1.2.3", str(p))
        first = p.read_text(encoding="utf-8")
        sv.write_env("1.2.4", str(p))
        second = p.read_text(encoding="utf-8")
        assert second.count("APP_VERSION=") == 1
        assert "TELEGRAM_CHAT_ID=42" in second
        assert len(second.splitlines()) == len(first.splitlines())

    def test_env_version_still_reads_it_back(self, sv, tmp_path):
        # The round trip the deploy check depends on.
        p = tmp_path / ".env"
        p.write_text("TELEGRAM_BOT_TOKEN=abc\n", encoding="utf-8")
        sv.write_env("9.9.9", str(p))
        assert sv.env_version(str(p)) == "9.9.9"
