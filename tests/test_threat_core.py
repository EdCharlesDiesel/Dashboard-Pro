"""Unit tests for src/core/threat_core.py — the Threat Board's pure scoring
logic and its SQLAlchemy persistence layer.

Network-dependent helpers (usdjpy_snapshot, jpy_cot_net_percentile,
upcoming_red_events, scan_intervention_headlines, current_regime) are
monkeypatched in the build_report test rather than exercised directly — same
policy as the rest of the suite: no live yfinance/requests/DB calls. The
persistence functions (ensure_tables/load_positions/journal/last_state) use a
fake SQLAlchemy-shaped connection, mirroring test_trade_repository.py's fake
psycopg2 connection.
"""
from __future__ import annotations

import pytest

from src.core import threat_core as tc


# ── pip / position math ────────────────────────────────────────────────────
class TestPipSize:
    def test_fx_pair(self):
        assert tc.pip_size("EURUSD") == 0.0001

    def test_jpy_pair(self):
        assert tc.pip_size("USDJPY") == 0.01

    def test_lowercase_input(self):
        assert tc.pip_size("usdjpy") == 0.01
        assert tc.pip_size("eurusd") == 0.0001


class TestPipValueUsd:
    def test_jpy_quote(self):
        pos = tc.Position("USDJPY", "long", 0.20, 150.0, 149.5)
        assert tc.pip_value_usd(pos, usdjpy=150.0) == pytest.approx(20000 * 0.01 / 150.0)

    def test_usd_quote(self):
        pos = tc.Position("EURUSD", "long", 1.0, 1.09, 1.08)
        assert tc.pip_value_usd(pos, usdjpy=150.0) == pytest.approx(10.0)

    def test_exotic_quote_falls_back_to_flat_rate(self):
        pos = tc.Position("EURGBP", "long", 1.0, 0.85, 0.84)
        assert tc.pip_value_usd(pos, usdjpy=150.0) == pytest.approx(10.0)


class TestStopRiskUsd:
    def test_usdjpy_position(self):
        pos = tc.Position("USDJPY", "long", 0.20, 150.00, 149.50)
        # 50 pips * (20000*0.01/150) pip value
        expected_pips = 50.0
        expected_pip_value = 20000 * 0.01 / 150.0
        assert tc.stop_risk_usd(pos, usdjpy=150.0) == pytest.approx(
            expected_pips * expected_pip_value
        )


class TestCurrencyExposure:
    def test_single_long_position(self):
        positions = [tc.Position("AUDJPY", "long", 1.0, 95.0, 94.5)]
        assert tc.currency_exposure(positions) == {"AUD": 1.0, "JPY": -1.0}

    def test_multiple_positions_net(self):
        positions = [
            tc.Position("AUDJPY", "long", 1.0, 95.0, 94.5),
            tc.Position("EURUSD", "short", 0.5, 1.09, 1.10),
        ]
        exp = tc.currency_exposure(positions)
        assert exp == {"AUD": 1.0, "JPY": -1.0, "EUR": -0.5, "USD": 0.5}

    def test_offsetting_positions_are_dropped(self):
        positions = [
            tc.Position("EURUSD", "long", 1.0, 1.09, 1.08),
            tc.Position("EURUSD", "short", 1.0, 1.09, 1.10),
        ]
        assert tc.currency_exposure(positions) == {}


class TestCorrelatedStopRisk:
    def test_same_direction_clusters(self):
        pos1 = tc.Position("USDJPY", "long", 1.0, 150.0, 149.5)
        pos2 = tc.Position("AUDJPY", "long", 1.0, 95.0, 94.5)
        worst_usd, worst_ccy = tc.correlated_stop_risk([pos1, pos2], usdjpy=150.0)
        assert worst_ccy == "JPY"
        assert worst_usd == pytest.approx(
            tc.stop_risk_usd(pos1, 150.0) + tc.stop_risk_usd(pos2, 150.0)
        )

    def test_opposite_direction_does_not_cluster(self):
        # Long USDJPY (short JPY) + short AUDJPY (long JPY) cancel on JPY, so
        # JPY never becomes the worst cluster despite both touching it.
        pos1 = tc.Position("USDJPY", "long", 1.0, 150.0, 149.5)
        pos2 = tc.Position("AUDJPY", "short", 1.0, 95.0, 95.5)
        worst_usd, worst_ccy = tc.correlated_stop_risk([pos1, pos2], usdjpy=150.0)
        assert worst_ccy != "JPY"

    def test_no_positions_returns_zero(self):
        assert tc.correlated_stop_risk([], usdjpy=150.0) == (0.0, "")


# ── component scores ────────────────────────────────────────────────────────
class TestScoreConcentration:
    def test_non_positive_equity_is_safe(self):
        assert tc.score_concentration(100.0, equity=0.0) == 0.0

    def test_below_amber_scales_linearly(self):
        assert tc.score_concentration(50.0, equity=1000.0) == pytest.approx(20.0)

    def test_between_amber_and_red(self):
        assert tc.score_concentration(150.0, equity=1000.0) == pytest.approx(55.0)

    def test_at_or_above_red_capped_at_100(self):
        assert tc.score_concentration(250.0, equity=1000.0) == pytest.approx(85.0)
        assert tc.score_concentration(5000.0, equity=1000.0) == pytest.approx(100.0)


class TestScoreIntervention:
    def test_flat_or_long_jpy_is_never_a_threat(self):
        assert tc.score_intervention(163.5, 3.0, jpy_net_lots=0.0) == 0.0
        assert tc.score_intervention(163.5, 3.0, jpy_net_lots=0.5) == 0.0

    def test_zone_bands_short_jpy(self):
        assert tc.score_intervention(163.5, 0.0, jpy_net_lots=-1.0) == pytest.approx(85.0)
        assert tc.score_intervention(162.5, 0.0, jpy_net_lots=-1.0) == pytest.approx(70.0)
        assert tc.score_intervention(161.5, 0.0, jpy_net_lots=-1.0) == pytest.approx(40.0)
        assert tc.score_intervention(160.0, 0.0, jpy_net_lots=-1.0) == pytest.approx(10.0)

    def test_roc_bonus_added_and_capped_at_100(self):
        assert tc.score_intervention(163.5, 2.0, jpy_net_lots=-1.0) == pytest.approx(95.0)
        assert tc.score_intervention(163.5, 3.0, jpy_net_lots=-1.0) == pytest.approx(100.0)

    def test_headline_bonus_capped_at_15(self):
        assert tc.score_intervention(
            160.0, 0.0, jpy_net_lots=-1.0, headline_hits=1
        ) == pytest.approx(18.0)
        assert tc.score_intervention(
            160.0, 0.0, jpy_net_lots=-1.0, headline_hits=3
        ) == pytest.approx(25.0)  # min(15, 24) == 15


class TestScoreSqueeze:
    def test_no_data_or_flat_is_zero(self):
        assert tc.score_squeeze(None, jpy_net_lots=-1.0) == 0.0
        assert tc.score_squeeze(8.0, jpy_net_lots=0.0) == 0.0

    def test_short_jpy_thresholds(self):
        assert tc.score_squeeze(8.0, jpy_net_lots=-1.0) == 80.0
        assert tc.score_squeeze(20.0, jpy_net_lots=-1.0) == 55.0
        assert tc.score_squeeze(50.0, jpy_net_lots=-1.0) == 20.0

    def test_long_jpy_thresholds(self):
        assert tc.score_squeeze(95.0, jpy_net_lots=1.0) == 80.0
        assert tc.score_squeeze(80.0, jpy_net_lots=1.0) == 55.0
        assert tc.score_squeeze(50.0, jpy_net_lots=1.0) == 20.0


class TestScoreCalendar:
    def test_unavailable_feed_does_not_fake_a_signal(self):
        assert tc.score_calendar(None) == 0.0

    def test_capped_at_100(self):
        assert tc.score_calendar([{}] * 2) == 50.0
        assert tc.score_calendar([{}] * 10) == 100.0


class TestScoreRegime:
    def test_unknown_regime_is_zero(self):
        assert tc.score_regime(None, {}) == 0.0

    def test_risk_off_conflict_long_risk_ccy(self):
        assert tc.score_regime("risk_off", {"AUD": 1.0}) == 75.0

    def test_risk_off_conflict_short_haven(self):
        assert tc.score_regime("risk_off", {"JPY": -1.0}) == 75.0

    def test_risk_off_no_conflict(self):
        assert tc.score_regime("risk_off", {"AUD": -1.0}) == 10.0

    def test_risk_on_conflict_short_risk_ccy(self):
        assert tc.score_regime("risk_on", {"AUD": -1.0}) == 60.0

    def test_risk_on_no_conflict_when_long_risk(self):
        assert tc.score_regime("risk_on", {"AUD": 1.0}) == 10.0


class TestBand:
    def test_boundaries(self):
        assert tc.band(0.0) == "green"
        assert tc.band(39.9) == "green"
        assert tc.band(40.0) == "amber"
        assert tc.band(69.9) == "amber"
        assert tc.band(70.0) == "red"
        assert tc.band(100.0) == "red"


# ── build_report aggregation (network calls monkeypatched) ─────────────────
class TestBuildReport:
    def test_aggregates_weighted_components(self, monkeypatch):
        monkeypatch.setattr(tc, "usdjpy_snapshot", lambda: (163.5, 3.0))
        monkeypatch.setattr(tc, "jpy_cot_net_percentile", lambda: 8.0)
        monkeypatch.setattr(tc, "upcoming_red_events", lambda held, days=7: None)
        monkeypatch.setattr(tc, "scan_intervention_headlines", lambda: [])
        monkeypatch.setattr(tc, "current_regime", lambda: None)

        positions = [tc.Position("USDJPY", "long", 1.0, 150.0, 149.0)]
        rep = tc.build_report(positions, equity=10_000.0)

        assert isinstance(rep, tc.ThreatReport)
        exposure = tc.currency_exposure(positions)
        worst_usd, worst_ccy = tc.correlated_stop_risk(positions, 163.5)
        expected = {
            "concentration": tc.score_concentration(worst_usd, 10_000.0),
            "intervention": tc.score_intervention(
                163.5, 3.0, exposure.get("JPY", 0.0),
                (tc.JPY_ZONE_LOW, tc.JPY_ZONE_HIGH), 0,
            ),
            "squeeze": tc.score_squeeze(8.0, exposure.get("JPY", 0.0)),
            "calendar": tc.score_calendar(None),
            "regime": tc.score_regime(None, exposure),
        }
        assert rep.components == expected
        total = sum(expected[k] * tc.WEIGHTS[k] for k in expected) / sum(tc.WEIGHTS.values())
        assert rep.score == pytest.approx(round(total, 1))
        assert rep.state == tc.band(total)
        assert rep.detail["usdjpy_last"] == pytest.approx(163.5)
        assert rep.detail["worst_cluster_ccy"] == worst_ccy


# ── persistence (fake SQLAlchemy-shaped connection) ─────────────────────────
class _FakeResult:
    def __init__(self, rows=None, one=None):
        self._rows = rows or []
        self._one = one

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._one


class _FakeConn:
    def __init__(self, fetchall_rows=None, fetchone_row=None):
        self.executed = []  # [(sql_text, params), ...]
        self.committed = False
        self._rows = fetchall_rows or []
        self._one = fetchone_row

    def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params))
        return _FakeResult(self._rows, self._one)

    def commit(self):
        self.committed = True


class TestEnsureTables:
    def test_creates_both_tables_and_commits(self):
        conn = _FakeConn()
        tc.ensure_tables(conn)
        sqls = [sql for sql, _ in conn.executed]
        assert any("CREATE TABLE IF NOT EXISTS threat_positions" in s for s in sqls)
        assert any("CREATE TABLE IF NOT EXISTS threat_journal" in s for s in sqls)
        assert conn.committed


class TestLoadPositions:
    def test_maps_rows_to_position_objects(self):
        rows = [
            ("USDJPY", "long", 1.0, 150.0, 149.0),
            ("EURUSD", "short", 0.5, 1.09, 1.10),
        ]
        conn = _FakeConn(fetchall_rows=rows)
        assert tc.load_positions(conn) == [
            tc.Position("USDJPY", "long", 1.0, 150.0, 149.0),
            tc.Position("EURUSD", "short", 0.5, 1.09, 1.10),
        ]

    def test_no_rows_returns_empty_list(self):
        conn = _FakeConn(fetchall_rows=[])
        assert tc.load_positions(conn) == []


class TestJournal:
    def test_inserts_all_components_and_commits(self):
        conn = _FakeConn()
        rep = tc.ThreatReport(
            components={
                "concentration": 10.0, "intervention": 20.0, "squeeze": 30.0,
                "calendar": 40.0, "regime": 50.0,
            },
            detail={"usdjpy_last": 150.0}, score=30.0, state="amber",
        )
        tc.journal(conn, rep)

        sql, params = conn.executed[0]
        assert "INSERT INTO threat_journal" in sql
        assert params == {
            "s": 30.0, "st": "amber",
            "c": 10.0, "i": 20.0, "q": 30.0, "cal": 40.0, "r": 50.0,
            "d": '{"usdjpy_last": 150.0}',
        }
        assert conn.committed


class TestLastState:
    def test_returns_state_when_a_row_exists(self):
        conn = _FakeConn(fetchone_row=("red",))
        assert tc.last_state(conn) == "red"

    def test_returns_none_when_journal_is_empty(self):
        conn = _FakeConn(fetchone_row=None)
        assert tc.last_state(conn) is None
