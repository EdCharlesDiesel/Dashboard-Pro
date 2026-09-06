"""Parsing the Treasury Fiscal Data feed.

`src/services/fiscal_data.py` is pure — no network, no database — which is why
it sits inside the coverage gate while its collector in `data_backbone` is
omitted.

Two properties carry most of the risk, and neither is visible by eye:

* **Cents survive.** The debt figure needs 16 significant digits and `float64`
  gives ~15.95, so parsing through `float` yields `40102964278586.1015625`. It
  still *formats* to the right cents today, which is what makes it dangerous.
* **Every field arrives as a string**, including nulls. Coercion is ours, and a
  suppressed row read as `0` would enter the series as a cliff rather than a gap.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.services.fiscal_data import (
    AMOUNT_FIELDS,
    DEBT_TO_PENNY_PATH,
    collect_points,
    FiscalPoint,
    build_url,
    next_page_number,
    parse_rows,
)


def _payload(*rows, links=None):
    return {"data": list(rows), "links": links or {"next": None}}


class TestBuildUrl:
    def test_the_path_lands_on_the_fiscal_service_host(self):
        url = build_url(DEBT_TO_PENNY_PATH)
        assert url.startswith("https://api.fiscaldata.treasury.gov/")
        assert DEBT_TO_PENNY_PATH in url

    def test_paging_and_sort_are_expressed_as_the_api_wants_them(self):
        url = build_url(DEBT_TO_PENNY_PATH, page_size=100, page_number=3)
        assert "page%5Bsize%5D=100" in url or "page[size]=100" in url
        assert "page%5Bnumber%5D=3" in url or "page[number]=3" in url
        assert "sort=-record_date" in url

    def test_no_api_key_is_ever_attached(self):
        # The endpoint is unauthenticated. A key appearing here would mean a
        # secret had been threaded somewhere it is not needed.
        url = build_url(DEBT_TO_PENNY_PATH).lower()
        assert "api_key" not in url and "token" not in url


class TestParseRows:
    def test_amounts_keep_their_cents(self):
        point = parse_rows(_payload({
            "record_date": "2026-09-03",
            "tot_pub_debt_out_amt": "40102964278586.10"}))[0]
        assert isinstance(point.value, Decimal)
        assert point.value == Decimal("40102964278586.10")

    def test_parsing_does_not_route_through_float(self):
        # The regression this file exists for: float(  ) on this input gives
        # ...586.1015625. Comparing against the exact Decimal catches it; a
        # round-to-2 comparison would not.
        point = parse_rows(_payload({
            "record_date": "2026-09-03",
            "tot_pub_debt_out_amt": "40102964278586.10"}))[0]
        assert point.value != Decimal(float("40102964278586.10"))

    def test_the_record_date_becomes_a_date(self):
        point = parse_rows(_payload({
            "record_date": "2026-09-03",
            "tot_pub_debt_out_amt": "1.00"}))[0]
        assert point.record_date == date(2026, 9, 3)

    @pytest.mark.parametrize("bad", ["null", "", None, "  "])
    def test_a_missing_amount_is_skipped_not_zeroed(self, bad):
        # Treasury sends "null" for suppressed rows. Zero is a real debt figure;
        # storing it would put a cliff in the series instead of a gap.
        assert parse_rows(_payload({
            "record_date": "2026-09-03", "tot_pub_debt_out_amt": bad})) == []

    def test_a_row_without_a_date_is_skipped(self):
        assert parse_rows(_payload({"tot_pub_debt_out_amt": "1.00"})) == []

    def test_every_requested_amount_field_becomes_its_own_point(self):
        rows = parse_rows(_payload({
            "record_date": "2026-09-03",
            "tot_pub_debt_out_amt": "40102964278586.10",
            "debt_held_public_amt": "32423635247198.78",
            "intragov_hold_amt": "7679329031387.32"}))
        by_id = {p.series_id: p.value for p in rows}
        assert by_id["tot_pub_debt_out_amt"] == Decimal("40102964278586.10")
        assert by_id["debt_held_public_amt"] == Decimal("32423635247198.78")
        assert by_id["intragov_hold_amt"] == Decimal("7679329031387.32")

    def test_the_components_sum_to_the_total(self):
        # A real invariant of this dataset, and a cheap check that the three
        # fields were not transposed during parsing.
        rows = parse_rows(_payload({
            "record_date": "2026-09-03",
            "tot_pub_debt_out_amt": "40102964278586.10",
            "debt_held_public_amt": "32423635247198.78",
            "intragov_hold_amt": "7679329031387.32"}))
        by_id = {p.series_id: p.value for p in rows}
        assert (by_id["debt_held_public_amt"] + by_id["intragov_hold_amt"]
                == by_id["tot_pub_debt_out_amt"])

    def test_an_empty_feed_is_empty_not_an_error(self):
        assert parse_rows({"data": []}) == []
        assert parse_rows({}) == []


class TestPagination:
    def test_a_next_link_yields_the_next_page_number(self):
        payload = {"links": {"next": "&page%5Bnumber%5D=2&page%5Bsize%5D=100"}}
        assert next_page_number(payload) == 2

    def test_the_last_page_yields_none(self):
        assert next_page_number({"links": {"next": None}}) is None

    def test_absent_links_yield_none(self):
        assert next_page_number({}) is None


class TestFiscalPoint:
    def test_it_carries_series_date_and_value(self):
        p = FiscalPoint(record_date=date(2026, 9, 3),
                        series_id="tot_pub_debt_out_amt",
                        value=Decimal("1.00"))
        assert (p.record_date, p.series_id, p.value) == (
            date(2026, 9, 3), "tot_pub_debt_out_amt", Decimal("1.00"))


class TestMalformedInput:
    """The feed is external and unversioned; garbage must produce a gap.

    These cover the coercion failure paths. A parser that raises on a bad row
    takes the whole collector down with it; one that returns a default invents
    a data point. Both are worse than skipping the row.
    """

    def test_a_non_numeric_amount_is_skipped(self):
        assert parse_rows(_payload({
            "record_date": "2026-09-03",
            "tot_pub_debt_out_amt": "not-a-number"})) == []

    def test_a_malformed_date_is_skipped(self):
        assert parse_rows(_payload({
            "record_date": "03/09/2026",
            "tot_pub_debt_out_amt": "1.00"})) == []

    def test_one_bad_field_does_not_discard_its_good_siblings(self):
        rows = parse_rows(_payload({
            "record_date": "2026-09-03",
            "tot_pub_debt_out_amt": "40102964278586.10",
            "debt_held_public_amt": "oops"}))
        assert [p.series_id for p in rows] == ["tot_pub_debt_out_amt"]

    def test_a_fields_filter_reaches_the_url(self):
        url = build_url(DEBT_TO_PENNY_PATH, fields=("record_date", "tot_pub_debt_out_amt"))
        assert "fields=record_date%2Ctot_pub_debt_out_amt" in url


class TestCollectPoints:
    """Pagination, with the transport injected so no test touches the network.

    The feed is 4,193 pages at size 2. Two behaviours matter and they pull in
    opposite directions: a first backfill must walk every page, and the daily
    job must stop after one — a daily job that follows `links.next` would pull
    the entire history every night.
    """

    def _pages(self, *pages):
        """A fake transport: returns each payload in turn, records the URLs."""
        calls = []

        def fetch(url):
            calls.append(url)
            return pages[min(len(calls) - 1, len(pages) - 1)]

        return fetch, calls

    def test_one_page_is_fetched_when_not_backfilling(self):
        fetch, calls = self._pages(
            {"data": [{"record_date": "2026-09-03", "tot_pub_debt_out_amt": "1.00"}],
             "links": {"next": "&page%5Bnumber%5D=2"}})

        points = collect_points(fetch, DEBT_TO_PENNY_PATH, backfill=False)

        assert len(calls) == 1, "the daily job must not follow links.next"
        assert len(points) == 1

    def test_backfill_follows_every_page_until_next_is_none(self):
        fetch, calls = self._pages(
            {"data": [{"record_date": "2026-09-03", "tot_pub_debt_out_amt": "1.00"}],
             "links": {"next": "&page%5Bnumber%5D=2"}},
            {"data": [{"record_date": "2026-09-02", "tot_pub_debt_out_amt": "2.00"}],
             "links": {"next": "&page%5Bnumber%5D=3"}},
            {"data": [{"record_date": "2026-09-01", "tot_pub_debt_out_amt": "3.00"}],
             "links": {"next": None}})

        points = collect_points(fetch, DEBT_TO_PENNY_PATH, backfill=True)

        assert len(calls) == 3
        assert [str(p.value) for p in points] == ["1.00", "2.00", "3.00"]

    def test_backfill_stops_at_the_page_cap(self):
        # 4,193 pages exist. An unbounded loop against a feed that always
        # returns a next link would run forever; the cap is the circuit breaker.
        fetch, calls = self._pages(
            {"data": [{"record_date": "2026-09-03", "tot_pub_debt_out_amt": "1.00"}],
             "links": {"next": "&page%5Bnumber%5D=2"}})

        collect_points(fetch, DEBT_TO_PENNY_PATH, backfill=True, max_pages=5)

        assert len(calls) == 5, "an always-next feed must not loop forever"

    def test_an_empty_page_ends_the_walk(self):
        fetch, calls = self._pages(
            {"data": [], "links": {"next": "&page%5Bnumber%5D=2"}})

        assert collect_points(fetch, DEBT_TO_PENNY_PATH, backfill=True) == []
        assert len(calls) == 1


@pytest.mark.slow
class TestAgainstTheLiveFeed:
    """The one test that touches the network. Gated behind --runslow.

    Unit tests prove the parser against fixtures I wrote, which means they
    cannot catch the feed changing shape underneath us — a renamed field, a
    different date format, amounts arriving as numbers instead of strings. This
    is the only check that would notice.
    """

    def _live(self):
        import json
        import urllib.request

        url = build_url(DEBT_TO_PENNY_PATH, page_size=5)
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.load(resp)

    def test_the_live_feed_still_parses(self):
        points = parse_rows(self._live())
        assert points, "the live feed produced no points - has it changed shape?"
        assert all(isinstance(p.value, Decimal) for p in points)

    def test_the_headline_series_is_present_and_positive(self):
        totals = [p for p in parse_rows(self._live())
                  if p.series_id == "tot_pub_debt_out_amt"]
        assert totals, "tot_pub_debt_out_amt missing from the live feed"
        assert all(p.value > 0 for p in totals)

    def test_the_latest_record_is_recent(self):
        # Treasury publishes daily. A record older than ~10 days means the feed
        # has stalled, and a stalled feed that still returns 200 is the failure
        # a health check would otherwise miss.
        from datetime import date, timedelta

        newest = max(p.record_date for p in parse_rows(self._live()))
        assert newest >= date.today() - timedelta(days=10), (
            f"newest record is {newest} - the feed looks stale")

    def test_parsed_values_match_the_api_strings_exactly(self):
        """The precision claim, end to end against the real numbers."""
        payload = self._live()
        points = {(p.record_date.isoformat(), p.series_id): p.value
                  for p in parse_rows(payload)}
        for row in payload["data"]:
            for field in AMOUNT_FIELDS:
                raw = row.get(field)
                if not raw or str(raw).lower() == "null":
                    continue
                assert str(points[(row["record_date"], field)]) == str(raw), (
                    f"{field} on {row['record_date']} did not round-trip exactly")
