"""Per-currency strength as a time series (not just today's snapshot).

Generalizes ``src.pages_lib.currency_strength._currency_returns``'s
basket-average method — for each registry FX pair, attribute its % return to
the base currency and the sign-flipped return to the quote currency, then
average across every pair a currency appears in — into a *daily* series so it
can be plotted as an index over time, e.g. for the Currency Strength Index
page's per-currency line chart.

Pure: no Streamlit, no network, no DB. Callers pass in a wide ``closes``
DataFrame (columns = registry FX tickers, index = date) already fetched
through the app's cached spine (e.g. ``currency_strength._fetch_pair_closes``).
"""
from __future__ import annotations

import pandas as pd

from src.instruments.registry import INSTRUMENTS


def daily_currency_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """currency -> daily % return (decimal), averaged across every registry FX
    pair containing it that has data for that day.

    A pair missing on a given day (holiday gap) is excluded from that day's
    average via ``skipna`` rather than nulling out the whole currency — the
    same tolerance ``_currency_returns`` already has for its snapshot.
    """
    contributions: dict[str, list[pd.Series]] = {}
    for pair in INSTRUMENTS.forex_pairs():
        ticker = INSTRUMENTS[pair]["ticker"]
        if ticker not in closes.columns:
            continue
        # fill_method=None explicitly. The default is version-dependent and the
        # two pandas in play disagree: 2.3.3 (the containers) pads, so a holiday
        # gap becomes a *fake 0% move*; 3.0.2 (the dev venv) leaves it NaN. The
        # docstring above promises the gap is "excluded via skipna" — under
        # padding it is not NaN, so skipna excludes nothing and a missing pair
        # contributes flat to that day's average. Production ran the padding
        # version, so the local suite passing was the misleading half.
        ret = closes[ticker].pct_change(fill_method=None)
        base, quote = pair.split("/")
        contributions.setdefault(base, []).append(ret)
        contributions.setdefault(quote, []).append(-ret)

    out = {
        ccy: pd.concat(series_list, axis=1).mean(axis=1, skipna=True)
        for ccy, series_list in contributions.items()
    }
    return pd.DataFrame(out)


def currency_index_series(closes: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Cumulative index per currency, starting at ``base``.

    A day with no data at all for a currency (every contributing pair missing)
    contributes a 0% move rather than propagating NaN through the whole
    remaining series via ``cumprod``.
    """
    daily = daily_currency_returns(closes)
    return base * (1.0 + daily.fillna(0.0)).cumprod()
