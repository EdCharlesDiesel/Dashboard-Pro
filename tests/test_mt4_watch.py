"""Unit tests for the durable MT4 watch-config store (JSON-fallback path;
the DB layer is monkeypatched off, matching the no-live-db test fixture)."""
import json

from src.services import mt4_watch


def test_defaults_when_nothing_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(mt4_watch, "_PATH", str(tmp_path / "cfg.json"))
    cfg = mt4_watch.load()
    assert cfg == {"path": "", "auto_on": False, "offset": 0.0}


def test_round_trip_via_json(tmp_path, monkeypatch):
    monkeypatch.setattr(mt4_watch, "_PATH", str(tmp_path / "cfg.json"))
    mt4_watch.save(r"C:\reports\history.htm", True, 2.0)
    cfg = mt4_watch.load()
    assert cfg["path"] == r"C:\reports\history.htm"
    assert cfg["auto_on"] is True
    assert cfg["offset"] == 2.0


def test_partial_stored_dict_fills_defaults(tmp_path, monkeypatch):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"path": "/x/report.htm"}))  # no auto_on/offset
    monkeypatch.setattr(mt4_watch, "_PATH", str(p))
    cfg = mt4_watch.load()
    assert cfg["path"] == "/x/report.htm"
    assert cfg["auto_on"] is False and cfg["offset"] == 0.0


def test_save_coerces_types(tmp_path, monkeypatch):
    monkeypatch.setattr(mt4_watch, "_PATH", str(tmp_path / "cfg.json"))
    mt4_watch.save(None, 1, "3")  # falsy path, truthy int, string offset
    cfg = mt4_watch.load()
    assert cfg["path"] == "" and cfg["auto_on"] is True and cfg["offset"] == 3.0
