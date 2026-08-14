"""Unit tests for src/db/trade_repository.py.

No live Postgres — a fake connection (injected via ``connect_factory``) records
every SQL string and parameter set, so we can assert exactly what is written and
prove the save path persists all data.
"""
from __future__ import annotations

import re

import pytest

from src.db.trade_repository import DBConfig, TradeRepository


# ── fakes ─────────────────────────────────────────────────────────────────────
class FakeCursor:
    def __init__(self, fetchall_rows=None, fetchone_row=None):
        self.executed = []          # [(sql, params), ...]
        self._fetchall = fetchall_rows if fetchall_rows is not None else []
        self._fetchone = fetchone_row

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._fetchall

    def fetchone(self):
        return self._fetchone

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.closed = False
        self.entered = False

    def cursor(self, *args, **kwargs):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        self.closed = True

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        return False


def _repo(cursor):
    conn = FakeConnection(cursor)
    repo = TradeRepository(DBConfig(), connect_factory=lambda: conn)
    return repo, conn


def _sample_setup_row():
    return {
        "logged_at": "2026-06-23 12:00:00", "instrument": "EUR/USD",
        "ticker": "EURUSD=X", "direction": "Long", "session": "London Kill Zone",
        "score": "17/18", "verdict": "GO", "atr14": 0.0012, "atr20": 0.0010,
        "sl_pips": 18.0, "tp1_pips": 36.0, "tp2_pips": 54.0, "lot_size": 0.55,
        "risk_amount": 100.0, "rr_tp1": 2.0, "rr_tp2": 3.0, "account_bal": 10000.0,
        "risk_pct": 1.0, "checks_passed": 17, "checks_total": 18,
        "checks_detail": '{"check_1": true}', "notes": "clean setup",
    }


# ── completeness audit ────────────────────────────────────────────────────────
def _create_columns():
    """Column names declared in CREATE_SQL (the source-of-truth schema)."""
    cols = []
    for line in TradeRepository.CREATE_SQL.splitlines():
        m = re.match(
            r"\s*(\w+)\s+(SERIAL|VARCHAR|FLOAT|INT|JSONB|TIMESTAMP|TEXT|BOOLEAN)",
            line,
        )
        if m:
            cols.append(m.group(1))
    return cols


class TestSaveSetupCompleteness:
    def test_insert_persists_every_schema_column(self):
        cur = FakeCursor()
        repo, conn = _repo(cur)
        row = _sample_setup_row()
        repo.save_setup(row)

        sql, params = cur.executed[0]
        assert params is row  # the row dict is passed straight through

        # Every CREATE_SQL column except the auto id must be written by save_setup.
        expected = [c for c in _create_columns() if c != "id"]
        for col in expected:
            assert col in sql, f"column {col} missing from INSERT"
            assert f"%({col})s" in sql, f"placeholder for {col} missing"
        # ...and the row supplies a value for each (logged_at is set explicitly).
        for col in expected:
            assert col in row, f"save_setup row is missing {col}"

    def test_commit_and_close_called(self):
        cur = FakeCursor()
        repo, conn = _repo(cur)
        repo.save_setup(_sample_setup_row())
        assert conn.committed is True
        assert conn.closed is True  # closing() ran

    def test_default_call_passes_row_unchanged_without_source(self):
        # The no-source path must stay byte-identical: row passed straight
        # through, and the source column never appears in the INSERT.
        cur = FakeCursor()
        repo, _ = _repo(cur)
        row = _sample_setup_row()
        repo.save_setup(row)
        sql, params = cur.executed[0]
        assert params is row
        assert "source" not in sql
        assert "%(source)s" not in sql

    def test_source_arg_adds_column_without_mutating_row(self):
        cur = FakeCursor()
        repo, _ = _repo(cur)
        row = _sample_setup_row()
        repo.save_setup(row, source="market_overview")
        sql, params = cur.executed[0]
        assert "source" in sql and "%(source)s" in sql
        assert params["source"] == "market_overview"
        assert params is not row            # caller's dict untouched
        assert "source" not in row          # not mutated
        # every original schema column is still present
        for col in (c for c in _create_columns() if c != "id"):
            assert f"%({col})s" in sql


# ── writes ────────────────────────────────────────────────────────────────────
class TestWrites:
    def test_close_trade_params_order(self):
        cur = FakeCursor()
        repo, _ = _repo(cur)
        repo.close_trade(7, 1.1000, 1.1050, 50.0, 2.5, "WIN")
        sql, params = cur.executed[0]
        assert "UPDATE trade_setups" in sql
        assert "is_open = FALSE" in sql
        assert params == (1.1000, 1.1050, 50.0, 2.5, "WIN", 7)

    def test_delete_setup(self):
        cur = FakeCursor()
        repo, conn = _repo(cur)
        repo.delete_setup(42)
        sql, params = cur.executed[0]
        assert "DELETE FROM trade_setups WHERE id" in sql
        assert params == (42,)
        assert conn.committed is True

    def test_import_mt4_rows_batches_and_counts(self, monkeypatch):
        captured = {}

        def fake_execute_batch(cur, sql, payload):
            captured["sql"] = sql
            captured["payload"] = payload

        monkeypatch.setattr(
            "src.db.trade_repository.psycopg2.extras.execute_batch",
            fake_execute_batch,
        )
        cur = FakeCursor()
        repo, conn = _repo(cur)
        rows = [
            {"logged_at": "t", "instrument": "EUR/USD", "ticker": "EURUSD=X",
             "direction": "Long", "session": "x", "lot_size": 0.1,
             "entry_price": 1.1, "close_price": 1.2, "outcome": "WIN",
             "pips_gained": 100, "r_multiple": 2.0, "sl_pips": 50,
             "notes": "MT4 #123", "profit": 55.0, "ignored": "dropped"},
        ]
        n = repo.import_mt4_rows(rows)
        assert n == 1
        # Only the mapped keys are forwarded (extra keys dropped).
        assert "ignored" not in captured["payload"][0]
        assert captured["payload"][0]["instrument"] == "EUR/USD"
        assert conn.committed is True

    def test_import_mt4_rows_empty_is_noop(self):
        cur = FakeCursor()
        repo, _ = _repo(cur)
        assert repo.import_mt4_rows([]) == 0
        assert cur.executed == []

    def test_mark_invalidated_params(self):
        cur = FakeCursor()
        repo, conn = _repo(cur)
        repo.mark_invalidated(7, 1.0970)
        sql, params = cur.executed[0]
        assert "UPDATE trade_setups" in sql
        assert "invalidated_at = NOW()" in sql
        assert "invalidated_at IS NULL" in sql
        # Never touches outcome/close_price/is_open — a badge only.
        assert "outcome" not in sql
        assert "is_open" not in sql
        assert params == (1.0970, 7)
        assert conn.committed is True


class TestImportedTickets:
    def test_parses_ticket_numbers_from_notes(self):
        # imported_tickets reads plain tuples (no RealDictCursor).
        cur = FakeCursor(fetchall_rows=[("MT4 #123 closed",), ("MT4 #456",), (None,),
                                        ("no ticket here",)])
        repo, _ = _repo(cur)
        assert repo.imported_tickets() == {123, 456}
        sql, params = cur.executed[0]
        # Source is bound, not interpolated, so a second broker feed can reuse
        # this without rewriting the query.
        assert "source = %s" in sql
        assert params == ("mt4_import",)

    def test_scopes_to_the_requested_source_and_tag(self):
        cur = FakeCursor(fetchall_rows=[("MT5 #900",), ("MT4 #123",)])
        repo, _ = _repo(cur)
        # The MT4 row must not leak into the MT5 dedupe pool: the two feeds
        # number tickets independently and could collide on the same integer.
        assert repo.imported_tickets(source="mt5_import", tag="MT5") == {900}
        _, params = cur.executed[0]
        assert params == ("mt5_import",)

    def test_swallows_errors(self):
        repo = TradeRepository(DBConfig(), connect_factory=lambda: (_ for _ in ()).throw(OSError()))
        assert repo.imported_tickets() == set()


# ── schema ────────────────────────────────────────────────────────────────────
class TestSchema:
    def test_init_schema_creates_table_and_outcome_columns(self):
        cur = FakeCursor()
        repo, conn = _repo(cur)
        ok, msg = repo.init_schema()
        assert ok is True
        assert msg == "Connected"
        statements = " ".join(sql for sql, _ in cur.executed)
        assert "CREATE TABLE IF NOT EXISTS trade_setups" in statements
        # One ALTER per outcome column.
        assert statements.count("ADD COLUMN IF NOT EXISTS") == len(repo.OUTCOME_COLUMNS)

    def test_init_schema_reports_failure(self):
        def boom():
            raise RuntimeError("connection refused")

        repo = TradeRepository(DBConfig(), connect_factory=boom)
        ok, msg = repo.init_schema()
        assert ok is False
        assert "connection refused" in msg

    def test_init_schema_creates_tool_usage_log_table(self):
        cur = FakeCursor()
        repo, _ = _repo(cur)
        repo.init_schema()
        statements = " ".join(sql for sql, _ in cur.executed)
        assert "CREATE TABLE IF NOT EXISTS tool_usage_log" in statements

    def test_init_schema_creates_swing_theses_table(self):
        cur = FakeCursor()
        repo, _ = _repo(cur)
        repo.init_schema()
        statements = " ".join(sql for sql, _ in cur.executed)
        assert "CREATE TABLE IF NOT EXISTS swing_theses" in statements


# ── swing playbook theses ─────────────────────────────────────────────────────
class TestSwingThesis:
    def test_save_swing_thesis_upserts_with_params(self):
        cur = FakeCursor()
        repo, conn = _repo(cur)
        repo.save_swing_thesis("EUR/USD", "2026-07-13", "long", "invalid below 1.05")
        sql, params = cur.executed[0]
        assert "INSERT INTO swing_theses" in sql
        assert "ON CONFLICT (instrument, week_start) DO UPDATE" in sql
        assert params == {
            "instrument": "EUR/USD", "week_start": "2026-07-13",
            "bias": "long", "invalidation": "invalid below 1.05",
        }
        assert conn.committed is True

    def test_load_swing_thesis_returns_dict(self):
        cur = FakeCursor(fetchone_row={"bias": "short", "invalidation": "invalid above 2400"})
        repo, _ = _repo(cur)
        result = repo.load_swing_thesis("XAU/USD", "2026-07-13")
        assert result == {"bias": "short", "invalidation": "invalid above 2400"}
        sql, params = cur.executed[0]
        assert "SELECT bias, invalidation FROM swing_theses" in sql
        assert params == ("XAU/USD", "2026-07-13")

    def test_load_swing_thesis_returns_none_when_absent(self):
        cur = FakeCursor(fetchone_row=None)
        repo, _ = _repo(cur)
        assert repo.load_swing_thesis("EUR/USD", "2026-07-13") is None


# ── tool usage log ──────────────────────────────────────────────────────────
class TestToolUsageLog:
    def test_log_tool_usage_inserts_json_payload(self, monkeypatch):
        captured = {}

        class FakeJson:
            def __init__(self, value):
                captured["value"] = value

        monkeypatch.setattr(
            "src.db.trade_repository.psycopg2.extras.Json", FakeJson
        )
        cur = FakeCursor()
        repo, conn = _repo(cur)
        repo.log_tool_usage("rr_calculator", {"entry": 1.1, "sl_pips": 20})
        sql, params = cur.executed[0]
        assert "INSERT INTO tool_usage_log" in sql
        assert params[0] == "rr_calculator"
        assert captured["value"] == {"entry": 1.1, "sl_pips": 20}
        assert conn.committed is True


# ── app_state key/value store ─────────────────────────────────────────────────
class TestAppState:
    def test_init_schema_creates_app_state_table(self):
        cur = FakeCursor()
        repo, _ = _repo(cur)
        repo.init_schema()
        statements = " ".join(sql for sql, _ in cur.executed)
        assert "CREATE TABLE IF NOT EXISTS app_state" in statements

    def test_get_state_returns_decoded_value(self):
        # psycopg2 hands back the JSONB already decoded to a Python object;
        # get_state reads column 0 of the row.
        cur = FakeCursor(fetchone_row=({"balance": 12345.0, "source": "mt4"},))
        repo, _ = _repo(cur)
        val = repo.get_state("account_balance")
        assert val == {"balance": 12345.0, "source": "mt4"}
        sql, params = cur.executed[0]
        assert "FROM app_state WHERE key" in sql
        assert params == ("account_balance",)

    def test_get_state_absent_returns_none(self):
        cur = FakeCursor(fetchone_row=None)
        repo, _ = _repo(cur)
        assert repo.get_state("missing") is None

    def test_set_state_upserts_json(self, monkeypatch):
        captured = {}

        class FakeJson:
            def __init__(self, value):
                captured["value"] = value

        monkeypatch.setattr(
            "src.db.trade_repository.psycopg2.extras.Json", FakeJson
        )
        cur = FakeCursor()
        repo, conn = _repo(cur)
        repo.set_state("account_balance", {"balance": 5000.0})
        sql, params = cur.executed[0]
        assert "INSERT INTO app_state" in sql
        assert "ON CONFLICT (key) DO UPDATE" in sql
        assert params[0] == "account_balance"
        assert captured["value"] == {"balance": 5000.0}
        assert conn.committed is True


# ── reads ─────────────────────────────────────────────────────────────────────
class TestReads:
    def test_load_setups_returns_dicts(self):
        rows = [{"id": 1, "instrument": "EUR/USD"}, {"id": 2, "instrument": "XAU/USD"}]
        cur = FakeCursor(fetchall_rows=rows)
        repo, _ = _repo(cur)
        out = repo.load_setups(limit=10)
        assert out == rows
        sql, params = cur.executed[0]
        assert "FROM trade_setups" in sql
        assert params == (10,)

    def test_load_open_filters_open_trades(self):
        cur = FakeCursor(fetchall_rows=[{"id": 1}])
        repo, _ = _repo(cur)
        repo.load_open()
        sql, _ = cur.executed[0]
        assert "is_open IS TRUE OR is_open IS NULL" in sql

    def test_daily_losses_blocked_when_over_limit(self):
        cur = FakeCursor(fetchone_row={"cnt": 3})
        repo, conn = _repo(cur)
        res = repo.daily_losses(max_losses=2)
        assert res == {"losses_today": 3, "limit": 2, "blocked": True}
        assert conn.closed is True  # leak fix: connection is now closed

    def test_daily_losses_not_blocked(self):
        cur = FakeCursor(fetchone_row={"cnt": 1})
        repo, _ = _repo(cur)
        assert repo.daily_losses(max_losses=2)["blocked"] is False

    def test_daily_losses_swallows_errors(self):
        repo = TradeRepository(DBConfig(), connect_factory=lambda: (_ for _ in ()).throw(OSError()))
        res = repo.daily_losses()
        assert res["losses_today"] == 0
        assert res["blocked"] is False


class TestPerformanceStats:
    def test_none_when_no_rows(self):
        cur = FakeCursor(fetchall_rows=[])
        repo, _ = _repo(cur)
        assert repo.performance_stats() is None

    def test_win_rate_and_expectancy(self):
        rows = [
            {"outcome": "WIN", "r_multiple": 2.0, "pips_gained": 40},
            {"outcome": "WIN", "r_multiple": 3.0, "pips_gained": 60},
            {"outcome": "LOSS", "r_multiple": -1.0, "pips_gained": -20},
            {"outcome": "BE", "r_multiple": 0.0, "pips_gained": 0},
        ]
        cur = FakeCursor(fetchall_rows=rows)
        repo, _ = _repo(cur)
        stats = repo.performance_stats()
        assert stats["total"] == 4
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["be"] == 1
        assert stats["win_rate"] == pytest.approx(50.0)
        assert stats["avg_win_r"] == pytest.approx(2.5)
        assert stats["avg_loss_r"] == pytest.approx(1.0)


class TestRealizedPnl:
    def test_uses_broker_profit_when_present(self):
        rows = [{"is_open": False, "outcome": "WIN", "profit": 55.0,
                 "instrument": "EUR/USD"}]
        cur = FakeCursor(fetchall_rows=rows)
        repo, _ = _repo(cur)
        res = repo.realized_pnl()
        assert res["pnl"] == pytest.approx(55.0)
        assert res["counted"] == 1
        assert res["skipped"] == 0

    def test_derives_from_pips_when_no_profit(self):
        # EUR/USD pip value is 10.0; 30 pips × 0.5 lot × 10 = 150.0
        rows = [{"is_open": False, "outcome": "WIN", "profit": None,
                 "pips_gained": 30.0, "lot_size": 0.5, "instrument": "EUR/USD"}]
        cur = FakeCursor(fetchall_rows=rows)
        repo, _ = _repo(cur)
        res = repo.realized_pnl()
        assert res["pnl"] == pytest.approx(150.0)
        assert res["counted"] == 1

    def test_skips_incomputable_rows(self):
        rows = [{"is_open": False, "outcome": "WIN", "profit": None,
                 "pips_gained": None, "lot_size": None, "instrument": "EUR/USD"}]
        cur = FakeCursor(fetchall_rows=rows)
        repo, _ = _repo(cur)
        res = repo.realized_pnl()
        assert res["counted"] == 0
        assert res["skipped"] == 1

    def test_ignores_open_trades(self):
        rows = [{"is_open": True, "outcome": None, "profit": 99.0,
                 "instrument": "EUR/USD"}]
        cur = FakeCursor(fetchall_rows=rows)
        repo, _ = _repo(cur)
        res = repo.realized_pnl()
        assert res["pnl"] == 0.0
        assert res["counted"] == 0


# ── DBConfig ──────────────────────────────────────────────────────────────────
class TestDBConfig:
    def test_from_mapping_defaults_and_overrides(self):
        cfg = DBConfig.from_mapping({"host": "db", "name": "trading", "port": "5433"})
        assert cfg.host == "db"
        assert cfg.dbname == "trading"
        assert cfg.port == 5433  # coerced to int

    def test_as_kwargs_roundtrip(self):
        cfg = DBConfig(host="h", port=1, dbname="d", user="u", password="p")
        assert cfg.as_kwargs() == {"host": "h", "port": 1, "dbname": "d",
                                   "user": "u", "password": "p"}
