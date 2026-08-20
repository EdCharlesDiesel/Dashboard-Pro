"""Unit tests for the open-position capture path:

  * `mt4_import.parse_mt4_open_positions` / `to_open_position_rows`
  * `src/services/open_positions.py` (durable store)

The statement fixture is modelled on the real MT4 book that motivated the
feature — three GBP/AUD sells and three AUD/JPY buys with **no stop loss**,
plus one stopped EUR/AUD sell — so the parser is exercised against the exact
shape that has to work.

No network, no live DB: conftest's autouse fixture forces the DB target to
unconfigured, so the store falls back to its JSON layer, which is redirected
to tmp_path.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.services import mt4_import, open_positions
from src.services.exposure import check_stack, net_currency_exposure

# --- statement fixture -------------------------------------------------------
# Closed rows carry a parseable close timestamp in the 9th cell; open rows have
# a blank there and the current price after it. All <td>, no <th>, so pandas
# keeps every row instead of promoting one to a header.
_OPEN_ROWS = [
    (262764269, "2026.07.27 15:30:53", "sell", 0.20, "GBPAUDi",
     1.90271, 0.00000, 1.88913, 1.90672, 0.05, -55.95),
    (262764270, "2026.07.27 15:30:53", "sell", 0.20, "GBPAUDi",
     1.90270, 0.00000, 1.88913, 1.90672, 0.05, -56.09),
    (262764346, "2026.07.27 15:36:34", "buy", 0.20, "AUDJPYi",
     114.496, 0.000, 117.988, 114.224, 0.66, -33.23),
    (262786677, "2026.07.28 18:08:54", "sell", 0.15, "EURAUDi",
     1.63407, 1.64427, 1.61241, 1.63334, 0.00, 7.64),
]

_CLOSED_ROW = (262700001, "2026.07.20 09:00:00", "buy", 0.10, "EURUSDi",
               1.13000, 1.12500, 1.14000, "2026.07.21 11:00:00", 1.13500,
               0.0, 0.0, 0.0, 50.0)

# A pending order must never count as exposure — its type cell is not an exact
# buy/sell, which is what the parser anchors on.
_PENDING_ROW = (262799999, "2026.07.28 10:00:00", "buy limit", 0.50, "EURUSDi",
                1.10000, 1.09000, 1.12000, "", 1.13000, 0.0, 0.0, 0.0, 0.0)


def _tr(cells) -> str:
    return "<tr>" + "".join("<td>{0}</td>".format(c) for c in cells) + "</tr>"


@pytest.fixture
def statement_bytes() -> bytes:
    rows = "".join(
        _tr((t, ot, ty, sz, it, op, sl, tp, "", cur, 0.0, 0.0, sw, pr))
        for (t, ot, ty, sz, it, op, sl, tp, cur, sw, pr) in _OPEN_ROWS
    )
    rows += _tr(_CLOSED_ROW) + _tr(_PENDING_ROW)
    html = "<html><body><table>{0}</table></body></html>".format(rows)
    return html.encode("utf-8")


@pytest.fixture(autouse=True)
def local_store(tmp_path, monkeypatch):
    """Redirect the store's JSON fallback into tmp_path so tests never write
    into the repo, and never read a developer's real book."""
    monkeypatch.setattr(open_positions, "_PATH", str(tmp_path / "open_positions.json"))
    yield


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
class TestParseOpenPositions:
    def test_finds_only_the_open_rows(self, statement_bytes):
        got = mt4_import.parse_mt4_open_positions(statement_bytes)
        assert {p["ticket"] for p in got} == {r[0] for r in _OPEN_ROWS}

    def test_excludes_closed_trades(self, statement_bytes):
        tickets = {p["ticket"] for p in
                   mt4_import.parse_mt4_open_positions(statement_bytes)}
        assert _CLOSED_ROW[0] not in tickets

    def test_excludes_pending_orders(self, statement_bytes):
        """A 'buy limit' is not exposure until it fills."""
        tickets = {p["ticket"] for p in
                   mt4_import.parse_mt4_open_positions(statement_bytes)}
        assert _PENDING_ROW[0] not in tickets

    def test_is_the_mirror_of_the_closed_parser(self, statement_bytes):
        """No row may be claimed by both parsers, and the closed one must still
        find its trade — the two together partition the statement."""
        opened = {p["ticket"] for p in
                  mt4_import.parse_mt4_open_positions(statement_bytes)}
        closed = {t["ticket"] for t in mt4_import.parse_mt4_html(statement_bytes)}
        assert opened & closed == set()
        assert closed == {_CLOSED_ROW[0]}

    def test_reads_the_fields(self, statement_bytes):
        got = mt4_import.parse_mt4_open_positions(statement_bytes)
        eur = next(p for p in got if p["ticket"] == 262786677)
        assert eur["type"] == "sell"
        assert eur["size"] == pytest.approx(0.15)
        assert eur["item"] == "EURAUDi"
        assert eur["open_price"] == pytest.approx(1.63407)
        assert eur["sl"] == pytest.approx(1.64427)
        assert eur["open_time"].year == 2026

    def test_deduplicates_tickets(self, statement_bytes):
        doubled = statement_bytes.replace(
            b"</table>", b"</table><table>" +
            _tr((262786677, "2026.07.28 18:08:54", "sell", 0.15, "EURAUDi",
                 1.63407, 1.64427, 1.61241, "", 1.63334, 0.0, 0.0, 0.0, 7.64)
                ).encode() + b"</table>")
        got = mt4_import.parse_mt4_open_positions(doubled)
        assert len([p for p in got if p["ticket"] == 262786677]) == 1

    def test_raises_on_a_file_with_no_tables(self):
        with pytest.raises(ValueError):
            mt4_import.parse_mt4_open_positions(b"<html><body>nope</body></html>")


class TestToOpenPositionRows:
    @pytest.fixture
    def rows(self, statement_bytes):
        parsed = mt4_import.parse_mt4_open_positions(statement_bytes)
        smap = mt4_import.build_symbol_map([p["item"] for p in parsed])
        return mt4_import.to_open_position_rows(parsed, smap)

    def test_maps_broker_symbols_to_registry_pairs(self, rows):
        mapped, _ = rows
        assert {r["pair"] for r in mapped} == {"GBP/AUD", "AUD/JPY", "EUR/AUD"}

    def test_maps_buy_sell_to_long_short(self, rows):
        mapped, _ = rows
        eur = next(r for r in mapped if r["pair"] == "EUR/AUD")
        aud = next(r for r in mapped if r["pair"] == "AUD/JPY")
        assert eur["direction"] == "SHORT"
        assert aud["direction"] == "LONG"

    def test_flags_missing_stops(self, rows):
        """The unstopped positions are the ones the guard shouts loudest about."""
        mapped, _ = rows
        assert {r["pair"] for r in mapped if not r["has_stop"]} == {"GBP/AUD",
                                                                   "AUD/JPY"}
        eur = next(r for r in mapped if r["pair"] == "EUR/AUD")
        assert eur["has_stop"] is True
        assert eur["stop_loss"] == pytest.approx(1.64427)

    def test_zero_stop_is_not_a_stop(self, rows):
        mapped, _ = rows
        gbp = next(r for r in mapped if r["pair"] == "GBP/AUD")
        assert gbp["stop_loss"] is None

    def test_labels_carry_the_ticket(self, rows):
        mapped, _ = rows
        assert all(r["label"].startswith("MT4 #") for r in mapped)

    def test_unmappable_symbols_are_skipped_and_counted(self):
        parsed = [{"ticket": 1, "type": "buy", "size": 0.1, "item": "BTCUSD",
                   "open_price": 1.0, "sl": 0, "tp": 0, "open_time": None}]
        mapped, skipped = mt4_import.to_open_position_rows(
            parsed, mt4_import.build_symbol_map(["BTCUSD"]))
        assert mapped == []
        assert skipped == {"BTCUSD": 1}

    def test_missing_size_is_skipped(self):
        parsed = [{"ticket": 1, "type": "buy", "size": None, "item": "EURUSDi",
                   "open_price": 1.0, "sl": 0, "tp": 0, "open_time": None}]
        mapped, skipped = mt4_import.to_open_position_rows(
            parsed, mt4_import.build_symbol_map(["EURUSDi"]))
        assert mapped == [] and skipped == {"EURUSDi": 1}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class TestStore:
    def test_round_trips(self):
        n = open_positions.save([{"pair": "EUR/AUD", "direction": "SHORT",
                                  "lot_size": 0.15, "has_stop": True}])
        assert n == 1
        book = open_positions.load()
        assert book[0]["pair"] == "EUR/AUD"
        assert book[0]["lot_size"] == 0.15

    def test_empty_when_never_saved(self):
        assert open_positions.load() == []

    def test_save_replaces_rather_than_appends(self):
        """A statement is a full snapshot; merging would resurrect positions
        you have since closed, and a guard that warns about trades you don't
        hold gets ignored."""
        open_positions.save([{"pair": "EUR/AUD", "direction": "SHORT"},
                             {"pair": "GBP/AUD", "direction": "SHORT"}])
        open_positions.save([{"pair": "EUR/AUD", "direction": "SHORT"}])
        assert [r["pair"] for r in open_positions.load()] == ["EUR/AUD"]

    def test_drops_malformed_rows(self):
        n = open_positions.save([
            {"pair": "EUR/AUD", "direction": "SHORT"},
            {"pair": "", "direction": "SHORT"},        # no pair
            {"pair": "GBP/AUD"},                        # no direction
            "not-a-dict", None,
        ])
        assert n == 1

    def test_keeps_only_known_fields(self):
        open_positions.save([{"pair": "EUR/AUD", "direction": "SHORT",
                              "secret": "should not persist"}])
        assert "secret" not in open_positions.load()[0]

    def test_clear_empties_the_book(self):
        open_positions.save([{"pair": "EUR/AUD", "direction": "SHORT"}])
        open_positions.clear()
        assert open_positions.load() == []

    def test_saved_at_is_recorded(self):
        open_positions.save([{"pair": "EUR/AUD", "direction": "SHORT"}])
        assert open_positions.saved_at()

    def test_saved_at_none_when_unset(self):
        assert open_positions.saved_at() is None

    def test_unstopped_filters_the_book(self):
        open_positions.save([
            {"pair": "GBP/AUD", "direction": "SHORT", "has_stop": False},
            {"pair": "EUR/AUD", "direction": "SHORT", "has_stop": True},
        ])
        assert [r["pair"] for r in open_positions.unstopped()] == ["GBP/AUD"]

    def test_db_layer_is_preferred_over_json(self, monkeypatch):
        open_positions.save([{"pair": "EUR/AUD", "direction": "SHORT"}])
        monkeypatch.setattr(open_positions, "_db_read", lambda: {
            "positions": [{"pair": "USD/CAD", "direction": "LONG"}],
            "saved_at": "2026-07-28T00:00:00+00:00",
        })
        assert [r["pair"] for r in open_positions.load()] == ["USD/CAD"]

    def test_corrupt_json_is_survivable(self, tmp_path, monkeypatch):
        path = tmp_path / "broken.json"
        path.write_text("{not json")
        monkeypatch.setattr(open_positions, "_PATH", str(path))
        assert open_positions.load() == []

    def test_non_dict_json_is_survivable(self, tmp_path, monkeypatch):
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]))
        monkeypatch.setattr(open_positions, "_PATH", str(path))
        assert open_positions.load() == []


# ---------------------------------------------------------------------------
# End to end: statement bytes -> stored book -> guard fires
# ---------------------------------------------------------------------------
def test_statement_to_guard_end_to_end(statement_bytes):
    """The whole point, in one test: import the real book, then ask the ranker's
    question — 'should I take EUR/AUD short?' — and get the stack warning that
    was missing when it mattered."""
    parsed = mt4_import.parse_mt4_open_positions(statement_bytes)
    smap = mt4_import.build_symbol_map([p["item"] for p in parsed])
    rows, _ = mt4_import.to_open_position_rows(parsed, smap)
    open_positions.save(rows)

    book = open_positions.load()
    net = net_currency_exposure(book)
    # 2× GBP/AUD short (0.20 each) + AUD/JPY long (0.20) + EUR/AUD short (0.15)
    assert net["AUD"] == pytest.approx(0.75)

    warnings = check_stack("EUR/AUD", "SHORT", book)
    aud = next(w for w in warnings if w.currency == "AUD")
    assert aud.count == 5           # four held legs + the candidate
    assert "GBP/AUD SHORT" in aud.summary()

    assert len(open_positions.unstopped()) == 3


class TestBrokerPnlSurvivesTheStore:
    """The page runs in a Linux container with no MetaTrader, so it cannot
    convert a non-USD-quoted pair by quoting the market. The broker already
    did that arithmetic — its P/L is in account currency — so the store has to
    carry it, together with the price it was measured at.

    Without both, a EUR/AUD projection is reported in AUD as though it were
    USD: about 1.4x out, with no error anywhere.
    """

    def test_make_row_keeps_broker_pnl_and_current_price(self):
        from src.services.open_positions import make_row
        row = make_row(pair="EUR/AUD", direction="SHORT", lot_size=0.2,
                       entry_price=1.62915, profit=-50.38, price_current=1.6327)
        assert row["profit"] == -50.38
        assert row["price_current"] == 1.6327

    def test_clean_does_not_silently_drop_them(self):
        # _clean rebuilds each row as {k: row.get(k) for k in _FIELDS}, so a
        # field missing from _FIELDS vanishes without raising - the failure
        # mode this test exists to catch.
        from src.services.open_positions import _clean
        kept = _clean([{"pair": "EUR/AUD", "direction": "SHORT",
                        "profit": -50.38, "price_current": 1.6327}])[0]
        assert kept["profit"] == -50.38
        assert kept["price_current"] == 1.6327

    def test_absent_broker_fields_stay_none_rather_than_zero(self):
        # Zero is a real P/L. None means "not reported", and the projection
        # must be able to tell those apart.
        from src.services.open_positions import make_row
        row = make_row(pair="EUR/USD", direction="LONG", lot_size=0.1)
        assert row["profit"] is None and row["price_current"] is None


class TestAgeMinutes:
    """How old the stored book is — the single implementation.

    Freshness is read from the *book's* stamp, not the balance's:
    `account_state` writes `updated_at` as naive **local** time, so a container
    running UTC computes a negative age for a balance the host wrote minutes
    ago — a number from the future, which reads as permanently fresh.
    `saved_at()` is ISO UTC and is written by the same sync call.
    """

    def test_none_when_nothing_is_stored(self, monkeypatch):
        monkeypatch.setattr(open_positions, "saved_at", lambda: None)
        assert open_positions.age_minutes() is None

    def test_minutes_since_the_stored_stamp(self, monkeypatch):
        monkeypatch.setattr(open_positions, "saved_at",
                            lambda: "2026-08-20T17:00:00+00:00")
        now = datetime(2026, 8, 20, 17, 30, tzinfo=timezone.utc)
        assert open_positions.age_minutes(now) == pytest.approx(30.0)

    def test_a_naive_stamp_is_read_as_utc_not_local(self, monkeypatch):
        # The trap. Read as local time in a UTC container this is off by the
        # offset, and on this host that means -120 minutes.
        monkeypatch.setattr(open_positions, "saved_at", lambda: "2026-08-20 17:00:00")
        now = datetime(2026, 8, 20, 17, 30, tzinfo=timezone.utc)
        assert open_positions.age_minutes(now) == pytest.approx(30.0)

    def test_unparseable_stamp_is_none_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(open_positions, "saved_at", lambda: "not a date")
        assert open_positions.age_minutes() is None

    def test_a_future_stamp_clamps_to_zero(self, monkeypatch):
        # Host/container clock skew must read as "fresh", never as a negative
        # age printed on screen.
        monkeypatch.setattr(open_positions, "saved_at",
                            lambda: "2026-08-20T18:00:00+00:00")
        now = datetime(2026, 8, 20, 17, 30, tzinfo=timezone.utc)
        assert open_positions.age_minutes(now) == 0.0
