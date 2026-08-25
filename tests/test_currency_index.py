import pandas as pd
import pytest

from src.services.currency_index import currency_index_series, daily_currency_returns


@pytest.fixture
def two_pair_closes():
    """Two registry pairs sharing USD: EUR/USD and USD/JPY (tickers EURUSD=X,
    USDJPY=X). 4 trading days, hand-picked moves:

    Day0->1: EURUSD +1% (EUR up, USD down from this leg)
             USDJPY +2% (USD up, JPY down from this leg)
    Day1->2: EURUSD -2%
             USDJPY  0%   (flat)
    Day2->3: EURUSD  NaN (holiday gap on this pair only)
             USDJPY +1%
    """
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    eurusd = [1.1000, 1.1110, 1.08878, None]
    usdjpy = [150.00, 153.00, 153.00, 154.53]
    return pd.DataFrame({"EURUSD=X": eurusd, "USDJPY=X": usdjpy}, index=idx)


def test_daily_currency_returns_cross_sectional_average(two_pair_closes):
    daily = daily_currency_returns(two_pair_closes)

    # Day1: EUR contributes +1% (only pair), USD contributes avg(-1%, +2%) = +0.5%,
    # JPY contributes -2% (only pair).
    assert daily.loc[two_pair_closes.index[1], "EUR"] == pytest.approx(0.01, abs=1e-4)
    assert daily.loc[two_pair_closes.index[1], "USD"] == pytest.approx(0.005, abs=1e-4)
    assert daily.loc[two_pair_closes.index[1], "JPY"] == pytest.approx(-0.02, abs=1e-4)

    # Day2: EURUSD -2%, USDJPY flat 0%.
    assert daily.loc[two_pair_closes.index[2], "EUR"] == pytest.approx(-0.02, abs=1e-4)
    assert daily.loc[two_pair_closes.index[2], "USD"] == pytest.approx((0.02 + 0.0) / 2, abs=1e-4)
    assert daily.loc[two_pair_closes.index[2], "JPY"] == pytest.approx(0.0, abs=1e-4)

    # Day3: EURUSD is NaN (holiday gap) -- EUR has no data, USD/JPY still get
    # USDJPY's +1% move, unaffected by the missing EUR leg.
    assert pd.isna(daily.loc[two_pair_closes.index[3], "EUR"])
    assert daily.loc[two_pair_closes.index[3], "USD"] == pytest.approx(0.01, abs=1e-4)
    assert daily.loc[two_pair_closes.index[3], "JPY"] == pytest.approx(-0.01, abs=1e-4)


def test_currency_index_series_starts_at_base_and_compounds(two_pair_closes):
    idx = currency_index_series(two_pair_closes, base=100.0)

    # Day0 (no prior close to diff against) is flat at the base for every currency.
    assert idx.loc[two_pair_closes.index[0], "EUR"] == pytest.approx(100.0)
    assert idx.loc[two_pair_closes.index[0], "USD"] == pytest.approx(100.0)
    assert idx.loc[two_pair_closes.index[0], "JPY"] == pytest.approx(100.0)

    # EUR: 100 * 1.01 (day1) * 0.98 (day2) = 98.98; day3 has no EUR data -> flat.
    assert idx.loc[two_pair_closes.index[2], "EUR"] == pytest.approx(100 * 1.01 * 0.98, rel=1e-3)
    assert idx.loc[two_pair_closes.index[3], "EUR"] == pytest.approx(
        idx.loc[two_pair_closes.index[2], "EUR"], rel=1e-3
    )


def test_currency_index_a_strengthening_currency_trends_up():
    """USD strengthens against every pair it appears in for 5 straight days ->
    its index should be monotonically rising."""
    idx_dates = pd.date_range("2026-02-01", periods=6, freq="D")
    eurusd = [1.10 * (0.99 ** i) for i in range(6)]   # EUR/USD falling -> USD up
    usdcad = [1.35 * (1.01 ** i) for i in range(6)]   # USD/CAD rising -> USD up
    closes = pd.DataFrame({"EURUSD=X": eurusd, "USDCAD=X": usdcad}, index=idx_dates)

    idx = currency_index_series(closes)
    usd = idx["USD"].dropna()
    assert list(usd) == sorted(usd)
    assert usd.iloc[-1] > usd.iloc[0]
