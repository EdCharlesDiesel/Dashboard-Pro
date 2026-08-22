"""Headless page smoke tests via Streamlit's AppTest.

These prove each page *runs* end-to-end without raising — they are NOT unit
tests: AppTest executes the real page, which fetches live data from yfinance
(5–60s per page). They are gated behind --runslow so the default suite stays
fast and deterministic:

    pytest --runslow tests/test_pages_smoke.py --no-cov

All 56 pages are swept, discovered from pages/ rather than listed by hand.

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


def _discover_pages() -> list[str]:
    """Every page in the app, found rather than listed.

    This was a hand-written slice of 5 pages out of 56, which meant 51 pages had
    no end-to-end coverage at all and a newly added page joined them silently.
    Discovery is the point: a page cannot escape the sweep by not being added to
    a list somebody has to remember to edit.
    """
    pages = sorted(p.name for p in (_ROOT / "pages").glob("*.py")
                   if not p.name.startswith("_"))
    return ["app.py"] + [f"pages/{name}" for name in pages]


_PAGES = _discover_pages()


@pytest.mark.slow
@pytest.mark.parametrize("page", _PAGES)
def test_page_runs_without_exception(page):
    path = _ROOT / page
    assert path.exists(), f"page not found: {page}"
    at = AppTest.from_file(str(path), default_timeout=300)
    at.run()
    exceptions = [str(e.value) for e in at.exception]
    assert not exceptions, f"{page} raised: {exceptions}"
