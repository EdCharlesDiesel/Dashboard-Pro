"""Round-trip reassembly and row mapping for the MT5 closed-trade importer.

Fixtures mirror deal shapes taken from a live Exness MT5 account so the
arithmetic is checked against trades that actually happened.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.services.mt5_trade_import import (
    broker_utc_offset,
    pair_deals,
    parse_close_level,
    trips_to_journal_rows,
)

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)

SMAP = {"XAUUSD": "XAU/USD", "EURZAR": "EUR/ZAR", "USDJPY": "USD/JPY",
        "WHOKNOWS": None}


def deal(pid, entry, dtype, price, *, vol=0.1, profit=0.0, hours_ago=1,
         symbol="XAUUSD", comment="", commission=0.0, swap=0.0):
    return {"ticket": 1000 + pid + entry, "position_id": pid, "symbol": symbol,
            "type": dtype, "entry": entry, "volume": vol, "price": price,
            "profit": profit, "commission": commission, "swap": swap,
            "comment": comment, "time": NOW - timedelta(hours=hours_ago)}


class TestParseCloseLevel:
    def test_reads_a_stop_out(self):
        assert parse_close_level("[sl 18.76307]") == ("sl", 18.76307)

    def test_reads_a_take_profit(self):
        assert parse_close_level("[tp 4387.20300]") == ("tp", 4387.203)

    @pytest.mark.parametrize("comment", ["", "manual close", "[sl ]", None])
    def test_no_level_is_not_an_error(self, comment):
        assert parse_close_level(comment) == (None, None)


class TestPairDeals:
    def test_reassembles_a_long_round_trip(self):
        trips, _held = pair_deals(
            [deal(1, 0, 0, 4339.899, hours_ago=5),
             deal(1, 1, 1, 4354.912, profit=150.13, hours_ago=2)],
            days=7, now=NOW)
        assert len(trips) == 1
        t = trips[0]
        assert t["direction"] == "Long"
        assert t["entry_price"] == pytest.approx(4339.899)
        assert t["close_price"] == pytest.approx(4354.912)
        assert t["profit"] == pytest.approx(150.13)

    def test_direction_comes_from_the_entry_not_the_exit(self):
        # Exit type is always the opposite of entry; reading it inverts trades.
        trips, _held = pair_deals(
            [deal(2, 0, 1, 18.668, hours_ago=5),
             deal(2, 1, 0, 18.763, profit=-58.74, hours_ago=2)],
            days=7, now=NOW)
        assert trips[0]["direction"] == "Short"

    def test_open_position_is_skipped(self):
        trips, held = pair_deals([deal(3, 0, 0, 4339.9)], days=7, now=NOW)
        assert trips == []
        assert held["still_open"] == 1

    def test_exit_without_its_entry_is_skipped(self):
        # Opened before the fetch window: no entry price, so not scoreable.
        trips, held = pair_deals([deal(4, 1, 1, 111.917, profit=-9.99)],
                                 days=7, now=NOW)
        assert trips == []
        assert held["unpaired_exits"] == 1

    def test_trip_closed_before_the_cutoff_is_excluded(self):
        trips, _held = pair_deals(
            [deal(5, 0, 0, 4339.9, hours_ago=24 * 30),
             deal(5, 1, 1, 4354.9, profit=10.0, hours_ago=24 * 20)],
            days=7, now=NOW)
        assert trips == []

    def test_long_hold_counts_when_it_closed_inside_the_window(self):
        # Entry far outside `days` — the wider lookback is the whole point.
        trips, _held = pair_deals(
            [deal(6, 0, 0, 4300.0, hours_ago=24 * 90),
             deal(6, 1, 1, 4400.0, profit=500.0, hours_ago=3)],
            days=7, now=NOW)
        assert len(trips) == 1
        assert trips[0]["entry_price"] == pytest.approx(4300.0)

    def test_partial_closes_aggregate_into_one_trip(self):
        trips, _held = pair_deals([
            deal(7, 0, 0, 4400.0, vol=0.2, hours_ago=5),
            deal(7, 1, 1, 4410.0, vol=0.1, profit=100.0, hours_ago=3),
            deal(7, 1, 1, 4420.0, vol=0.1, profit=200.0, hours_ago=2),
        ], days=7, now=NOW)
        assert len(trips) == 1
        t = trips[0]
        assert t["volume"] == pytest.approx(0.2)
        assert t["close_price"] == pytest.approx(4415.0)  # volume-weighted
        assert t["profit"] == pytest.approx(300.0)

    def test_still_open_remainder_is_held_back_not_stored_as_complete(self):
        # 0.2 opened, only 0.1 scaled out. Storing this would write a closed row
        # carrying half the trade's P/L, and dedupe on position id would then
        # block the correction forever.
        trips, held = pair_deals([
            deal(10, 0, 0, 4400.0, vol=0.2, hours_ago=5),
            deal(10, 1, 1, 4410.0, vol=0.1, profit=100.0, hours_ago=2),
        ], days=7, now=NOW)
        assert trips == []
        assert held["partially_closed"] == 1

    def test_held_position_is_emitted_once_it_closes_in_full(self):
        # The same position a week later, with the remainder closed at a loss:
        # now it settles as one row whose P/L is the true net, not the +100
        # that a premature import would have frozen in.
        trips, held = pair_deals([
            deal(10, 0, 0, 4400.0, vol=0.2, hours_ago=5),
            deal(10, 1, 1, 4410.0, vol=0.1, profit=100.0, hours_ago=3),
            deal(10, 1, 1, 4380.0, vol=0.1, profit=-500.0, hours_ago=2),
        ], days=7, now=NOW)
        assert len(trips) == 1
        assert trips[0]["volume"] == pytest.approx(0.2)
        assert trips[0]["profit"] == pytest.approx(-400.0)
        assert held["partially_closed"] == 0

    def test_exit_volume_exceeding_entry_is_held_back(self):
        # Entry legs fell outside the window, so the volume-weighted entry
        # price would be computed from an incomplete set.
        trips, held = pair_deals([
            deal(11, 0, 0, 4400.0, vol=0.1, hours_ago=5),
            deal(11, 1, 1, 4410.0, vol=0.2, profit=100.0, hours_ago=2),
        ], days=7, now=NOW)
        assert trips == []
        assert held["partially_closed"] == 1

    def test_volume_is_the_size_opened_across_scale_ins(self):
        trips, _held = pair_deals([
            deal(12, 0, 0, 4400.0, vol=0.1, hours_ago=6),
            deal(12, 0, 0, 4420.0, vol=0.1, hours_ago=5),
            deal(12, 1, 1, 4430.0, vol=0.2, profit=300.0, hours_ago=2),
        ], days=7, now=NOW)
        assert trips[0]["volume"] == pytest.approx(0.2)
        assert trips[0]["entry_price"] == pytest.approx(4410.0)  # weighted

    def test_close_by_deal_counts_as_an_exit(self):
        # entry == 3 only ever closes, so it pairs like a normal exit rather
        # than being dropped.
        trips, held = pair_deals([
            deal(13, 0, 0, 4400.0, vol=0.1, hours_ago=5),
            deal(13, 3, 1, 4410.0, vol=0.1, profit=100.0, hours_ago=2),
        ], days=7, now=NOW)
        assert len(trips) == 1
        assert trips[0]["profit"] == pytest.approx(100.0)
        assert held["reversals"] == 0

    def test_reversal_deal_is_held_and_counted_not_dropped(self):
        # entry == 2 spans a close and an opposing open; the split is not
        # recoverable from the deal, so booking it would invent a P/L.
        trips, held = pair_deals([
            deal(14, 0, 0, 4400.0, vol=0.1, hours_ago=5),
            deal(14, 2, 1, 4410.0, vol=0.3, profit=100.0, hours_ago=2),
        ], days=7, now=NOW)
        assert trips == []
        assert held["reversals"] == 1

    def test_commission_and_swap_land_in_profit(self):
        trips, _held = pair_deals(
            [deal(8, 0, 0, 4400.0, hours_ago=5),
             deal(8, 1, 1, 4410.0, profit=100.0, commission=-2.0,
                  swap=-1.5, hours_ago=2)],
            days=7, now=NOW)
        assert trips[0]["profit"] == pytest.approx(96.5)

    def test_stop_price_read_from_the_close_comment(self):
        trips, _held = pair_deals(
            [deal(9, 0, 0, 18.668, symbol="EURZAR", hours_ago=5),
             deal(9, 1, 1, 18.763, symbol="EURZAR", profit=-58.74,
                  comment="[sl 18.76307]", hours_ago=2)],
            days=7, now=NOW)
        assert trips[0]["stop_price"] == pytest.approx(18.76307)


class TestTripsToJournalRows:
    def _trip(self, **kw):
        base = {"position_id": 42, "symbol": "XAUUSD", "direction": "Long",
                "volume": 0.2, "entry_price": 4400.0, "close_price": 4410.0,
                "open_time": NOW - timedelta(hours=5), "close_time": NOW,
                "profit": 200.0, "stop_price": None}
        base.update(kw)
        return base

    def test_long_pips_use_the_registry_pip_size(self):
        rows, _ = trips_to_journal_rows([self._trip()], SMAP)
        # XAU/USD pip_size is 0.1, not 0.0001.
        assert rows[0]["pips_gained"] == pytest.approx(100.0)
        assert rows[0]["instrument"] == "XAU/USD"

    def test_short_pips_are_inverted(self):
        rows, _ = trips_to_journal_rows(
            [self._trip(direction="Short", entry_price=4410.0,
                        close_price=4400.0)], SMAP)
        assert rows[0]["pips_gained"] == pytest.approx(100.0)

    def test_losing_short_is_negative(self):
        rows, _ = trips_to_journal_rows(
            [self._trip(direction="Short", entry_price=4400.0,
                        close_price=4410.0, profit=-200.0)], SMAP)
        assert rows[0]["pips_gained"] == pytest.approx(-100.0)
        assert rows[0]["outcome"] == "LOSS"

    def test_stop_out_loss_is_minus_one_r(self):
        rows, _ = trips_to_journal_rows(
            [self._trip(close_price=4395.0, profit=-100.0, stop_price=4395.0)],
            SMAP)
        assert rows[0]["sl_pips"] == pytest.approx(50.0)
        assert rows[0]["r_multiple"] == pytest.approx(-1.0)

    def test_winner_closed_by_a_trailed_stop_keeps_a_null_r(self):
        # The stop was dragged into profit, so |entry - stop| is where the trade
        # ended, not what it risked — a derived +1.0 would be an artefact.
        rows, _ = trips_to_journal_rows(
            [self._trip(close_price=4410.0, profit=200.0, stop_price=4410.0)],
            SMAP)
        assert rows[0]["r_multiple"] is None
        assert rows[0]["sl_pips"] is None

    def test_no_stop_leaves_r_multiple_null_rather_than_guessing(self):
        rows, _ = trips_to_journal_rows([self._trip()], SMAP)
        assert rows[0]["sl_pips"] is None
        assert rows[0]["r_multiple"] is None

    def test_stop_equal_to_entry_does_not_divide_by_zero(self):
        rows, _ = trips_to_journal_rows(
            [self._trip(close_price=4390.0, profit=-1.0, stop_price=4400.0)],
            SMAP)
        assert rows[0]["r_multiple"] is None
        assert rows[0]["sl_pips"] is None

    @pytest.mark.parametrize("profit,expected",
                             [(200.0, "WIN"), (-200.0, "LOSS"), (0.0, "BE")])
    def test_outcome_follows_net_profit(self, profit, expected):
        rows, _ = trips_to_journal_rows([self._trip(profit=profit)], SMAP)
        assert rows[0]["outcome"] == expected

    def test_direction_is_canonical_title_case(self):
        rows, _ = trips_to_journal_rows([self._trip()], SMAP)
        assert rows[0]["direction"] == "Long"

    def test_notes_tag_carries_the_position_id_for_dedupe(self):
        rows, _ = trips_to_journal_rows([self._trip(position_id=3063847188)], SMAP)
        assert rows[0]["notes"] == "MT5 #3063847188"
        assert rows[0]["ticket"] == 3063847188

    def test_logged_at_is_the_open_time_and_naive(self):
        rows, _ = trips_to_journal_rows([self._trip()], SMAP)
        assert rows[0]["logged_at"].tzinfo is None
        assert rows[0]["logged_at"] == (NOW - timedelta(hours=5)).replace(tzinfo=None)

    def test_unmapped_symbol_is_counted_not_dropped_silently(self):
        rows, skipped = trips_to_journal_rows(
            [self._trip(symbol="WHOKNOWS")], SMAP)
        assert rows == []
        assert skipped == {"WHOKNOWS": 1}

    def test_missing_price_is_skipped(self):
        rows, skipped = trips_to_journal_rows(
            [self._trip(close_price=None)], SMAP)
        assert rows == [] and skipped == {"XAUUSD": 1}


class TestBrokerUtcOffset:
    """The env-var default that feeds `closed_deals(broker_utc_offset=...)`."""

    def test_absent_env_means_utc(self, monkeypatch):
        monkeypatch.delenv("MT5_BROKER_UTC_OFFSET", raising=False)
        assert broker_utc_offset() == 0.0

    def test_reads_a_whole_hour_offset(self, monkeypatch):
        monkeypatch.setenv("MT5_BROKER_UTC_OFFSET", "3")
        assert broker_utc_offset() == 3.0

    def test_negative_offsets_are_supported(self, monkeypatch):
        monkeypatch.setenv("MT5_BROKER_UTC_OFFSET", "-5")
        assert broker_utc_offset() == -5.0

    def test_garbage_falls_back_to_utc_rather_than_raising(self, monkeypatch):
        # A scheduled import must not die over a typo in the environment.
        monkeypatch.setenv("MT5_BROKER_UTC_OFFSET", "UTC+3")
        assert broker_utc_offset() == 0.0
