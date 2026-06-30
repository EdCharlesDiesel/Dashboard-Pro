"""Unit tests for src/services/account_state.py.

The balance is written to both Postgres (cached) and a local JSON fallback, and
reads prefer the DB. No live DB — the DB layer is monkeypatched; the JSON path
uses a tmp file.
"""
from __future__ import annotations

import json

from src.services import account_state


def test_set_balance_writes_json_and_db(monkeypatch, tmp_path):
    path = tmp_path / "account_state.json"
    monkeypatch.setattr(account_state, "_PATH", str(path))
    db = {}
    monkeypatch.setattr(account_state, "_db_write", lambda payload: db.update(payload))

    account_state.set_balance(12345.0, source="mt4")

    saved = json.loads(path.read_text())
    assert saved["balance"] == 12345.0
    assert saved["source"] == "mt4"
    assert db["balance"] == 12345.0  # DB write attempted with the same payload


def test_get_prefers_db_over_json(monkeypatch, tmp_path):
    path = tmp_path / "account_state.json"
    path.write_text(json.dumps({"balance": 42.0, "source": "manual"}))
    monkeypatch.setattr(account_state, "_PATH", str(path))
    monkeypatch.setattr(account_state, "_db_read", lambda: {"balance": 999.0, "source": "db"})

    assert account_state.get()["balance"] == 999.0
    assert account_state.get_balance() == 999.0


def test_get_falls_back_to_json_when_db_empty(monkeypatch, tmp_path):
    path = tmp_path / "account_state.json"
    path.write_text(json.dumps({"balance": 42.0, "source": "manual"}))
    monkeypatch.setattr(account_state, "_PATH", str(path))
    monkeypatch.setattr(account_state, "_db_read", lambda: None)

    assert account_state.get()["balance"] == 42.0


def test_get_balance_default_when_nothing_stored(monkeypatch, tmp_path):
    monkeypatch.setattr(account_state, "_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(account_state, "_db_read", lambda: None)

    assert account_state.get_balance(7777.0) == 7777.0
