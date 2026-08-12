"""The INSERT and the row builder must agree on which columns exist.

The bug this exists to prevent, in full: `signal_to_setup_row` was taught to
stamp `entry_price`, unit tests confirmed it did, the fix shipped — and 60
freshly swept signals still landed with NULL. `save_setup`'s INSERT column list
simply did not mention `entry_price`, so psycopg2 ignored that key in the params
dict and the value was dropped between builder and database without an error
anywhere.

A test on the row builder cannot catch this, and neither can a test on the SQL.
Only comparing the two can. These tests are deliberately about the *seam*.
"""
from __future__ import annotations

import re

import pytest

from src.core.signals import signal_to_setup_row
from src.db.trade_repository import TradeRepository


def _insert_sql() -> str:
    """The INSERT statement `save_setup` executes, without running it."""
    import inspect
    source = inspect.getsource(TradeRepository.save_setup)
    match = re.search(r'sql = """(.*?)"""', source, re.S)
    assert match, "save_setup no longer defines its INSERT as a triple-quoted sql"
    return match.group(1)


def _insert_columns(sql: str) -> list:
    match = re.search(r"INSERT INTO trade_setups\s*\((.*?)\)\s*VALUES", sql, re.S)
    assert match, "could not parse the INSERT column list"
    return [c.strip() for c in match.group(1).replace("\n", " ").split(",") if c.strip()]


def _insert_placeholders(sql: str) -> list:
    # Scan everything after VALUES rather than trying to match the bracket:
    # a non-greedy `\((.*?)\)` stops at the first ")" — which is inside
    # "%(logged_at)s", not the end of the list.
    assert "VALUES" in sql, "could not find the VALUES list"
    return re.findall(r"%\((\w+)\)s", sql[sql.index("VALUES"):])


def _builder_row() -> dict:
    return signal_to_setup_row(
        {"pair": "EUR/USD", "bias": "LONG", "entry": 1.15563, "atr": 0.0012,
         "stop_loss_pips": 18.0, "conviction": "HIGH"},
        session="London")


class TestInsertMatchesBuilder:
    def test_columns_and_placeholders_line_up(self):
        sql = _insert_sql()
        assert _insert_columns(sql) == _insert_placeholders(sql)

    def test_entry_price_is_inserted(self):
        # The specific regression: present in the builder, absent from the SQL.
        assert "entry_price" in _insert_columns(_insert_sql())

    def test_every_inserted_column_is_produced_by_the_builder(self):
        # A placeholder with no matching key raises at execute time; catching it
        # here beats catching it on a live sweep.
        row = _builder_row()
        missing = [c for c in _insert_placeholders(_insert_sql())
                   if c not in row and c != "source"]
        assert not missing, "INSERT references keys the builder never sets: %s" % missing

    def test_scoring_critical_fields_survive_the_seam(self):
        # Whatever else drifts, these four are what makes a row resolvable at
        # all: without them source_scorecard can never mark a win or a loss.
        columns = _insert_columns(_insert_sql())
        row = _builder_row()
        for field in ("entry_price", "sl_pips", "tp1_pips", "direction"):
            assert field in columns, "%s is not inserted" % field
            assert field in row, "%s is not built" % field

    def test_the_builder_actually_populates_entry_price(self):
        assert _builder_row()["entry_price"] == pytest.approx(1.15563)

    def test_source_is_appended_without_disturbing_the_tail(self):
        # save_setup bolts `source` on with a string replace against the exact
        # tail "checks_passed, checks_total, checks_detail, notes". Adding a
        # column near that tail would silently break the replace and drop the
        # source tag on every persisted signal.
        sql = _insert_sql()
        assert "checks_passed, checks_total, checks_detail, notes\n" in sql
        assert "%(checks_passed)s, %(checks_total)s, %(checks_detail)s, %(notes)s\n" in sql
