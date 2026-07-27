"""Unit tests for src/services/week_ahead.py and src/services/econ_calendar.py.

Both are pure/Streamlit-free, so the maths and the parsing are tested directly.
The cone's contract matters most: bands must widen with sqrt(time), bracket
spot, and degrade to EWMA (never crash) when GARCH can't fit.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.services import econ_calendar as cal
from src.services import week_ahead as wa


def _returns(n=400, sigma=0.008, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.Series(rng.standard_normal(n) * sigma, index=idx)


class TestCone:
    def test_bands_bracket_spot_and_nest(self):
        cone = wa.cone_from_returns(_returns(), spot=1.1000)
        assert cone is not None
        assert cone["lo95"] < cone["lo68"] < cone["spot"] < cone["hi68"] < cone["hi95"]

    def test_bands_widen_with_time(self):
        cone = wa.cone_from_returns(_returns(), spot=100.0)
        hi = cone["hi68_path"]
        assert len(hi) == cone["horizon"] == wa.WEEK_DAYS
        assert all(hi[i] < hi[i + 1] for i in range(len(hi) - 1))   # monotonic
        lo = cone["lo68_path"]
        assert all(lo[i] > lo[i + 1] for i in range(len(lo) - 1))

    def test_sqrt_time_scaling(self):
        """Terminal sigma should be ~sqrt(5)x the one-day sigma, not 5x."""
        cone = wa.cone_from_returns(_returns(sigma=0.01), spot=100.0)
        # Recover per-day cumulative sigma from the 68% path.
        cum = [np.log(h / cone["spot"]) for h in cone["hi68_path"]]
        ratio = cum[-1] / cum[0]
        assert 1.9 < ratio < 2.6            # sqrt(5) = 2.236

    def test_higher_vol_gives_wider_range(self):
        calm = wa.cone_from_returns(_returns(sigma=0.004), spot=100.0)
        wild = wa.cone_from_returns(_returns(sigma=0.02), spot=100.0)
        assert wild["range_pct"] > calm["range_pct"] * 2

    def test_ewma_fallback_when_garch_unavailable(self, monkeypatch):
        import src.core.quant_models as qm
        monkeypatch.setattr(qm, "fit_garch",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no arch")))
        cone = wa.cone_from_returns(_returns(), spot=1.1000)
        assert cone is not None and cone["method"] == "ewma"
        assert cone["hi68"] > cone["spot"]

    def test_insufficient_data_returns_none(self):
        assert wa.cone_from_returns(_returns(n=10), spot=1.0) is None

    def test_bad_spot_returns_none(self):
        assert wa.cone_from_returns(_returns(), spot=0.0) is None
        assert wa.cone_from_returns(_returns(), spot=None) is None


class TestClassifyRange:
    @pytest.mark.parametrize("pct,expected", [
        (0.6, "tight"), (1.4, "normal"), (2.5, "wide"), (5.0, "very wide"),
    ])
    def test_buckets(self, pct, expected):
        assert wa.classify_range(pct) == expected


class TestCurrenciesFor:
    def test_fx_pair_has_both_legs(self):
        assert wa.currencies_for("EUR/USD") == {"EUR", "USD"}

    def test_metals_and_oil_are_usd_only(self):
        # XAU/WTI aren't currencies — their news driver is the dollar.
        for pair in ("XAU/USD", "XAG/USD", "XPT/USD", "WTI/USD"):
            assert wa.currencies_for(pair) == {"USD"}

    def test_malformed_input(self):
        assert wa.currencies_for("") == set()
        assert wa.currencies_for("GOLD") == set()


class TestRiskFlags:
    def test_wide_range_flagged(self):
        cone = {"range_pct": 4.0, "regime": "normal"}
        flags = wa.risk_flags(cone, None, 0)
        assert any("range" in f.lower() for f in flags)

    def test_crowded_cot_flagged_both_sides(self):
        assert any("crowded long" in f for f in wa.risk_flags(None, 95.0, 0))
        assert any("crowded short" in f for f in wa.risk_flags(None, 3.0, 0))

    def test_event_heavy_week_flagged(self):
        assert any("high-impact" in f for f in wa.risk_flags(None, None, 5))

    def test_quiet_setup_has_no_flags(self):
        cone = {"range_pct": 0.8, "regime": "normal"}
        assert wa.risk_flags(cone, 50.0, 1) == []


# ── econ_calendar ────────────────────────────────────────────────────────────
_RAW = [
    {"date": "2026-08-03T08:30:00-04:00", "time": "8:30am", "country": "USD",
     "title": "Non-Farm Payrolls", "impact": "High", "forecast": "180K",
     "previous": "206K"},
    {"date": "2026-08-04T05:00:00-04:00", "time": "5:00am", "country": "EUR",
     "title": "ECB Press Conference", "impact": "High"},
    {"date": "2026-08-04T02:00:00-04:00", "time": "2:00am", "country": "GBP",
     "title": "Retail Sales", "impact": "Medium"},
    {"date": "", "title": "broken row", "country": "USD", "impact": "High"},
]


class TestEconCalendar:
    def test_normalise_drops_malformed(self):
        evs = cal.normalise(_RAW)
        assert len(evs) == 3
        assert all(isinstance(e["date"], date) for e in evs)

    def test_normalise_defaults_missing_fields(self):
        e = [x for x in cal.normalise(_RAW) if x["currency"] == "EUR"][0]
        assert e["forecast"] == "—" and e["previous"] == "—"

    def test_filter_by_impact_and_currency(self):
        evs = cal.normalise(_RAW)
        assert len(cal.filter_events(evs, None, ("High",))) == 2
        assert len(cal.filter_events(evs, {"USD"}, ("High",))) == 1
        assert len(cal.filter_events(evs, {"GBP"}, ("High",))) == 0
        assert len(cal.filter_events(evs, {"GBP"}, ("Medium",))) == 1

    def test_count_by_currency(self):
        counts = cal.count_by_currency(cal.normalise(_RAW))
        assert counts == {"USD": 1, "EUR": 1}

    def test_group_by_day_is_sorted(self):
        days = list(cal.group_by_day(cal.normalise(_RAW)))
        assert days == sorted(days)

    def test_fetch_week_rejects_bad_arg(self):
        with pytest.raises(ValueError):
            cal.fetch_week("lastweek")

    def test_fetch_week_returns_empty_on_failure(self, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "get",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
        assert cal.fetch_week("nextweek") == []

    def test_empty_inputs_are_safe(self):
        assert cal.normalise([]) == []
        assert cal.filter_events([], None) == []
        assert cal.group_by_day([]) == {}
