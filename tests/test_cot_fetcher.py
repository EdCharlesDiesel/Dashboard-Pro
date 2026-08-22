import pandas as pd

from src.services.cot_fetcher import normalize_datetime64


def test_normalize_datetime64_unifies_mismatched_resolutions():
    dates = pd.to_datetime(["2024-01-01", "2024-01-08"])
    s_us = pd.Series(dates).astype("datetime64[us]")
    s_s = pd.Series(dates).astype("datetime64[s]")
    assert s_us.dtype != s_s.dtype  # sanity: this is the real-world mismatch

    a = normalize_datetime64(s_us)
    b = normalize_datetime64(s_s)
    assert str(a.dtype) == str(b.dtype) == "datetime64[ns]"
    assert (a.values == b.values).all()


def test_normalize_datetime64_enables_merge_asof():
    left_dates = pd.to_datetime(["2024-01-01", "2024-01-08"]).astype("datetime64[us]")
    right_dates = pd.to_datetime(["2024-01-01", "2024-01-09"]).astype("datetime64[s]")

    left = pd.DataFrame({
        "date": normalize_datetime64(pd.Series(left_dates)),
        "net_spec": [1, 2],
    })
    right = pd.DataFrame({
        "date": normalize_datetime64(pd.Series(right_dates)),
        "price": [10.0, 11.0],
    })

    merged = pd.merge_asof(
        left.sort_values("date"), right.sort_values("date"),
        on="date", direction="backward",
    )
    assert len(merged) == 2
