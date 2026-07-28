"""Unit tests for src/services/mt5_link.py — the live MetaTrader 5 link.

A fake `MetaTrader5` module is injected everywhere, so these run on Linux/CI
with the package absent — which is the whole point: the real wheel is
Windows-only and the deploy of record is Linux containers, so "works without
it" is a behaviour under test, not an accident.

Shapes mirror the real terminal read taken from a live TradersWay MT5 demo:
position type 0=buy / 1=sell, and **absent stops reported as `sl = 0.0`**
rather than null — the trap `make_row` exists to close.
"""
from __future__ import annotations

import pytest

from src.services import mt5_link, open_positions
from src.services.exposure import check_stack, net_currency_exposure


class FakePosition:
    def __init__(self, ticket, symbol, type_, volume, price_open,
                 sl=0.0, tp=0.0, time=1785000000):
        self.ticket = ticket
        self.symbol = symbol
        self.type = type_
        self.volume = volume
        self.price_open = price_open
        self.sl = sl
        self.tp = tp
        self.time = time


class FakeAccount:
    def __init__(self, login=1121412, server="TradersWay-Demo", currency="USD",
                 balance=999.53, equity=1001.11, margin=314.01,
                 margin_free=687.10, margin_level=318.81):
        self.login = login
        self.server = server
        self.currency = currency
        self.balance = balance
        self.equity = equity
        self.margin = margin
        self.margin_free = margin_free
        self.margin_level = margin_level


class FakeTerminalInfo:
    name = "Traders Way MetaTrader 5"
    build = 6069
    connected = True


class FakeMT5:
    """Minimal stand-in for the vendor module."""

    _DEFAULT = object()   # sentinel: account=None must mean "not logged in"

    def __init__(self, positions=None, account=_DEFAULT, init_ok=True,
                 terminal=FakeTerminalInfo(), error=(1, "Success")):
        self._positions = positions
        self._account = FakeAccount() if account is FakeMT5._DEFAULT else account
        self._init_ok = init_ok
        self._terminal = terminal
        self._error = error
        self.initialized = 0
        self.shutdowns = 0

    def initialize(self):
        self.initialized += 1
        return self._init_ok

    def shutdown(self):
        self.shutdowns += 1

    def last_error(self):
        return self._error

    def terminal_info(self):
        return self._terminal

    def account_info(self):
        return self._account

    def positions_get(self):
        return self._positions


# The live book read off the terminal: one EURAUD sell, two USDJPY buys,
# none of them carrying a stop.
LIVE = [
    FakePosition(28393836, "EURAUD", 1, 0.1, 1.63268, sl=0.0, tp=1.62031),
    FakePosition(28393928, "USDJPY", 0, 0.1, 163.743, sl=0.0, tp=164.221),
    FakePosition(28393929, "USDJPY", 0, 0.1, 163.742, sl=0.0, tp=0.0),
]


@pytest.fixture(autouse=True)
def local_store(tmp_path, monkeypatch):
    monkeypatch.setattr(open_positions, "_PATH",
                        str(tmp_path / "open_positions.json"))
    yield


@pytest.fixture
def fake():
    return FakeMT5(positions=list(LIVE))


# ---------------------------------------------------------------------------
# Availability / probe
# ---------------------------------------------------------------------------
class TestAvailability:
    def test_available_is_false_without_the_package(self, monkeypatch):
        monkeypatch.setattr(mt5_link, "_import_mt5", lambda: None)
        assert mt5_link.available() is False

    def test_probe_explains_a_missing_package(self, monkeypatch):
        monkeypatch.setattr(mt5_link, "_import_mt5", lambda: None)
        ok, msg = mt5_link.probe()
        assert ok is False
        assert "not installed" in msg

    def test_probe_reports_the_terminal_and_account(self, fake):
        ok, msg = mt5_link.probe(mt5=fake)
        assert ok is True
        assert "MetaTrader 5" in msg and "1121412" in msg

    def test_probe_handles_terminal_not_running(self):
        ok, msg = mt5_link.probe(mt5=FakeMT5(init_ok=False,
                                             error=(-10005, "IPC timeout")))
        assert ok is False
        assert "not reachable" in msg and "IPC timeout" in msg

    def test_probe_handles_not_logged_in(self):
        ok, msg = mt5_link.probe(mt5=FakeMT5(account=None))
        assert ok is False
        assert "not logged in" in msg

    def test_probe_always_shuts_down(self, fake):
        mt5_link.probe(mt5=fake)
        assert fake.shutdowns == 1


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------
class TestPositionsToRows:
    @pytest.fixture
    def rows(self):
        from src.services.broker_symbols import build_symbol_map
        smap = build_symbol_map([p.symbol for p in LIVE])
        return mt5_link.positions_to_rows(LIVE, smap)

    def test_maps_symbols_to_registry_pairs(self, rows):
        mapped, _ = rows
        assert [r["pair"] for r in mapped] == ["EUR/AUD", "USD/JPY", "USD/JPY"]

    def test_maps_position_type_to_direction(self, rows):
        mapped, _ = rows
        assert mapped[0]["direction"] == "SHORT"   # type 1 = sell
        assert mapped[1]["direction"] == "LONG"    # type 0 = buy

    def test_zero_stop_is_not_a_stop(self, rows):
        """MT5 reports an absent stop as 0.0, not null. Treating that as a
        stop would hide the single most dangerous thing about the book."""
        mapped, _ = rows
        assert all(r["has_stop"] is False for r in mapped)
        assert all(r["stop_loss"] is None for r in mapped)

    def test_zero_take_profit_is_dropped_but_real_one_kept(self, rows):
        mapped, _ = rows
        assert mapped[0]["take_profit"] == pytest.approx(1.62031)
        assert mapped[2]["take_profit"] is None

    def test_labels_are_namespaced_to_mt5(self, rows):
        mapped, _ = rows
        assert mapped[0]["label"] == "MT5 #28393836"

    def test_opened_at_is_iso_utc(self, rows):
        mapped, _ = rows
        assert mapped[0]["opened_at"].endswith("+00:00")

    def test_unmappable_symbols_are_skipped_and_counted(self):
        from src.services.broker_symbols import build_symbol_map
        odd = [FakePosition(1, "BTCUSD", 0, 0.1, 50000.0)]
        mapped, skipped = mt5_link.positions_to_rows(
            odd, build_symbol_map(["BTCUSD"]))
        assert mapped == [] and skipped == {"BTCUSD": 1}

    def test_broker_suffixes_still_map(self):
        from src.services.broker_symbols import build_symbol_map
        suffixed = [FakePosition(1, "EURAUD.r", 1, 0.1, 1.63)]
        mapped, _ = mt5_link.positions_to_rows(
            suffixed, build_symbol_map(["EURAUD.r"]))
        assert mapped[0]["pair"] == "EUR/AUD"

    def test_empty_input_is_empty_output(self):
        assert mt5_link.positions_to_rows([], {}) == ([], {})
        assert mt5_link.positions_to_rows(None, {}) == ([], {})


# ---------------------------------------------------------------------------
# read_terminal
# ---------------------------------------------------------------------------
class TestReadTerminal:
    def test_reads_positions_and_account(self, fake):
        res = mt5_link.read_terminal(mt5=fake)
        assert res["ok"] is True
        assert len(res["rows"]) == 3
        assert res["account"]["balance"] == pytest.approx(999.53)
        assert res["account"]["margin_level"] == pytest.approx(318.81)

    def test_no_package_is_a_message_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(mt5_link, "_import_mt5", lambda: None)
        res = mt5_link.read_terminal()
        assert res["ok"] is False and res["rows"] == []

    def test_terminal_down_is_a_message(self):
        res = mt5_link.read_terminal(mt5=FakeMT5(init_ok=False))
        assert res["ok"] is False and "not reachable" in res["message"]

    def test_flat_account_is_success_with_no_rows(self):
        """positions_get() returns None when nothing is open; that is an empty
        book, not a read failure."""
        res = mt5_link.read_terminal(mt5=FakeMT5(positions=None,
                                                 error=(1, "Success")))
        assert res["ok"] is True and res["rows"] == []

    def test_genuine_read_failure_is_distinguished_from_flat(self):
        res = mt5_link.read_terminal(
            mt5=FakeMT5(positions=None, error=(-10004, "connection lost")))
        assert res["ok"] is False
        assert "Could not read positions" in res["message"]
        assert res["account"] is not None      # account still reported

    def test_unexpected_exception_is_swallowed(self):
        class Boom(FakeMT5):
            def positions_get(self):
                raise RuntimeError("terminal exploded")

        res = mt5_link.read_terminal(mt5=Boom())
        assert res["ok"] is False and "terminal exploded" in res["message"]

    def test_always_shuts_down_even_on_failure(self):
        class Boom(FakeMT5):
            def positions_get(self):
                raise RuntimeError("nope")

        fake = Boom()
        mt5_link.read_terminal(mt5=fake)
        assert fake.shutdowns == 1

    def test_margin_level_zero_becomes_none(self):
        """MT5 reports 0.0 when nothing is open — that is 'not applicable',
        not '0% and about to be liquidated'."""
        res = mt5_link.read_terminal(
            mt5=FakeMT5(positions=[], account=FakeAccount(margin_level=0.0)))
        assert res["account"]["margin_level"] is None


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------
class TestSync:
    def test_persists_positions_and_account(self, fake, monkeypatch):
        monkeypatch.setattr(mt5_link, "_import_mt5", lambda: fake)
        res = mt5_link.sync(set_balance=False)
        assert res["saved"] == 3
        assert [r["pair"] for r in open_positions.load()] == \
            ["EUR/AUD", "USD/JPY", "USD/JPY"]
        assert open_positions.account_snapshot()["login"] == 1121412

    def test_sets_the_account_balance(self, fake, monkeypatch):
        seen = {}
        from src.services import account_state
        monkeypatch.setattr(account_state, "set_balance",
                            lambda bal, source="": seen.update(bal=bal, src=source))
        mt5_link.sync(mt5=fake)
        assert seen["bal"] == pytest.approx(999.53)
        assert "MT5" in seen["src"]

    def test_balance_failure_does_not_lose_the_book(self, fake, monkeypatch):
        from src.services import account_state

        def boom(bal, source=""):
            raise RuntimeError("state store down")

        monkeypatch.setattr(account_state, "set_balance", boom)
        res = mt5_link.sync(mt5=fake)
        assert res["saved"] == 3

    def test_failed_read_saves_nothing(self):
        res = mt5_link.sync(mt5=FakeMT5(init_ok=False))
        assert res["ok"] is False and res["saved"] == 0
        assert open_positions.load() == []

    def test_replaces_rather_than_appends(self, fake, monkeypatch):
        open_positions.save([{"pair": "GBP/AUD", "direction": "SHORT"}])
        mt5_link.sync(mt5=fake, set_balance=False)
        assert "GBP/AUD" not in [r["pair"] for r in open_positions.load()]


# ---------------------------------------------------------------------------
# Margin warning
# ---------------------------------------------------------------------------
class TestMarginWarning:
    def test_warns_below_the_threshold(self):
        msg = mt5_link.margin_warning(
            {"margin_level": 108.25, "margin_free": 114.59, "currency": "USD"})
        assert msg and "108%" in msg and "114.59" in msg

    def test_silent_when_healthy(self):
        assert mt5_link.margin_warning({"margin_level": 318.81}) is None

    def test_silent_without_an_account(self):
        assert mt5_link.margin_warning(None) is None

    def test_silent_when_not_applicable(self):
        assert mt5_link.margin_warning({"margin_level": None}) is None
        assert mt5_link.margin_warning({"margin_level": 0}) is None

    def test_threshold_is_configurable(self):
        assert mt5_link.margin_warning({"margin_level": 250.0}) is None
        assert mt5_link.margin_warning({"margin_level": 250.0}, danger=300.0)


# ---------------------------------------------------------------------------
# End to end: terminal -> store -> guard
# ---------------------------------------------------------------------------
def test_live_terminal_to_guard_end_to_end(fake, monkeypatch):
    """Sync the live book, then ask the ranker's question. Two USD/JPY longs
    mean a third USD/JPY long is a stack — on both legs."""
    mt5_link.sync(mt5=fake, set_balance=False)
    book = open_positions.load()

    net = net_currency_exposure(book)
    assert net["USD"] == pytest.approx(0.2)     # two USDJPY buys
    assert net["JPY"] == pytest.approx(-0.2)
    assert net["AUD"] == pytest.approx(0.1)     # EURAUD sell = long AUD

    warnings = {w.currency: w for w in check_stack("USD/JPY", "LONG", book)}
    assert warnings["USD"].count == 3
    assert warnings["JPY"].count == 3

    # And a genuinely independent idea stays clean.
    assert check_stack("GBP/ZAR", "LONG", book) == []
    assert len(open_positions.unstopped()) == 3
