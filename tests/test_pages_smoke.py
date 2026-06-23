"""Headless page smoke tests via Streamlit's AppTest.

These prove each page *runs* end-to-end without raising — they are NOT unit
tests: AppTest executes the real page, which fetches live data from yfinance
(5–60s per page). They are gated behind --runslow so the default suite stays
fast and deterministic:

    pytest --runslow tests/test_pages_smoke.py --no-cov

Notes:
- `st.page_link` raises KeyError('url_pathname') under AppTest (no multipage
  registry); the shared nav renderer swallows it, so pages still run.
- "No pairs scored" / empty results reflect real market conditions, not bugs —
  we only assert the page raised no *uncaught* exception.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

_ROOT = Path(__file__).resolve().parents[1]

# A representative slice: the root entry (framework checklist) plus framework and
# legacy pages that share src/core. Kept small so the gated run stays bounded.
_PAGES = [
    "app.py",
    "pages/setup-ranker.py",
    "pages/correlations.py",
    "pages/market-overview.py",
]


@pytest.mark.slow
@pytest.mark.parametrize("page", _PAGES)
def test_page_runs_without_exception(page):
    path = _ROOT / page
    assert path.exists(), f"page not found: {page}"
    at = AppTest.from_file(str(path), default_timeout=300)
    at.run()
    exceptions = [str(e.value) for e in at.exception]
    assert not exceptions, f"{page} raised: {exceptions}"
