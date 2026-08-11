"""`entry_price` must be stamped at persist time, not recomputed at render time.

The defect: entry_price was NULL on all 724 stored signals, so the Source
Scorecard could score only the 12 whose idea happened to carry `entry` inside
checks_detail. A signal with no recorded entry cannot be judged against
subsequent bars at all — and levels recomputed later by `size_idea` anchor to
*today's* price, so the same signal shows different levels tomorrow and is never
measured on what it actually proposed.
"""
from __future__ import annotations

import pytest

from src.core.signals import signal_to_setup_row


def _idea(**kw):
    base = {"pair": "EUR/USD", "bias": "LONG", "atr": 0.0012,
            "stop_loss_pips": 18.0, "conviction": "HIGH"}
    base.update(kw)
    return base


class TestEntryPriceIsStamped:
    def test_entry_key_is_promoted_to_the_column(self):
        row = signal_to_setup_row(_idea(entry=1.15563), session='London')
        assert row["entry_price"] == pytest.approx(1.15563)

    def test_price_is_used_when_entry_is_absent(self):
        row = signal_to_setup_row(_idea(price=1.16), session='London')
        assert row["entry_price"] == pytest.approx(1.16)

    def test_close_is_the_last_resort(self):
        row = signal_to_setup_row(_idea(close=1.17), session='London')
        assert row["entry_price"] == pytest.approx(1.17)

    def test_entry_wins_over_price_and_close(self):
        row = signal_to_setup_row(_idea(entry=1.10, price=1.20, close=1.30), session='London')
        assert row["entry_price"] == pytest.approx(1.10)

    def test_absent_price_stays_none_rather_than_zero(self):
        # 0.0 would be a *valid-looking* entry and would silently produce
        # nonsense R-multiples in the scorecard.
        assert signal_to_setup_row(_idea(), session='London')["entry_price"] is None

    def test_nan_is_rejected_like_every_other_numeric(self):
        # NaN is invalid JSONB and previously destroyed whole rows silently.
        assert signal_to_setup_row(_idea(entry=float("nan")),
                                   session='London')["entry_price"] is None

    def test_zero_entry_falls_through_rather_than_being_taken_literally(self):
        # A page reporting 0.0 has no price, not a price of zero.
        row = signal_to_setup_row(_idea(entry=0.0, close=1.18), session='London')
        assert row["entry_price"] == pytest.approx(1.18)


class TestScorecardCanNowResolve:
    def test_a_stamped_row_exposes_an_entry_to_the_resolver(self):
        from src.services.signal_expiry import extract_entry_price
        row = signal_to_setup_row(_idea(entry=1.15563), session='London')
        assert extract_entry_price(dict(row)) == pytest.approx(1.15563)

    def test_an_unstamped_row_is_what_the_resolver_could_not_score(self):
        from src.services.signal_expiry import extract_entry_price
        row = signal_to_setup_row(_idea(), session='London')
        row["checks_detail"] = "{}"
        assert extract_entry_price(dict(row)) is None


class TestDirectionIsCanonical:
    """One spelling per direction in the column.

    LONG, Long and Short coexisted, so `count(DISTINCT direction)` reported
    three directions where there were two, and any `WHERE direction = 'Long'`
    silently missed the LONG rows.
    """

    def test_upper_case_is_normalised(self):
        row = signal_to_setup_row(_idea(bias="LONG"), session="London")
        assert row["direction"] == "Long"

    def test_broker_dialect_is_normalised(self):
        for dialect in ("buy", "BUY", "bullish", "BULL"):
            row = signal_to_setup_row(_idea(bias=dialect), session="London")
            assert row["direction"] == "Long", dialect
        for dialect in ("sell", "SELL", "bearish", "BEAR"):
            row = signal_to_setup_row(_idea(bias=dialect), session="London")
            assert row["direction"] == "Short", dialect

    def test_already_canonical_is_untouched(self):
        assert signal_to_setup_row(_idea(bias="Short"),
                                   session="London")["direction"] == "Short"

    def test_unrecognised_value_passes_through_rather_than_vanishing(self):
        # Losing a direction to None would drop the row from every count
        # silently; keeping it makes the oddity visible.
        assert signal_to_setup_row(_idea(bias="NEUTRAL"),
                                   session="London")["direction"] == "NEUTRAL"

    def test_empty_direction_is_none(self):
        assert signal_to_setup_row(_idea(bias=""), session="London")["direction"] is None

    def test_the_stored_value_is_what_consensus_reads(self):
        from src.core.todays_trades import _side
        row = signal_to_setup_row(_idea(bias="BUY"), session="London")
        assert _side(row["direction"]) == "Long"
