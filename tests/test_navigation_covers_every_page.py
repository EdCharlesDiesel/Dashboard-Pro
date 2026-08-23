"""Every page must be reachable from the sidebar, and every link must resolve.

`NAV_SECTIONS` in `src/pages_lib/navigation.py` is hand-maintained. A page added
to `pages/` without an entry there still *works* — Streamlit serves it, the
AppTest sweep exercises it, `/its_name` loads in a browser — it is simply
invisible to anyone using the app. That is a silent failure: nothing errors and
nothing is red.

It happened twice: `platinum_tab.py` was added with no entry, and before that the
page smoke test kept its own hand-written list of 5 pages while 51 others went
uncovered. Both were lists a human had to remember to edit. This test is the
cheaper half of that lesson — the nav genuinely needs curation (order, sections,
labels), so it stays hand-written, but it may not silently drift.

The reverse direction matters too: a nav entry pointing at a deleted page throws
`StreamlitAPIException` on the sidebar of *every* page, not just that one.
"""
from __future__ import annotations

import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAV = os.path.join(_REPO, "src", "pages_lib", "navigation.py")
_PAGES_DIR = os.path.join(_REPO, "pages")

# Match the path argument itself rather than the whole NavEntry(...) call: labels
# legitimately contain parentheses ("MTF Matrix (House View)"), and a `[^)]*`
# scan stops dead on the first one -- which produced a false "missing" report
# the first time this was measured.
_PATH = re.compile(r'"(pages/[A-Za-z0-9._\-]+\.py)"')


def _linked() -> set:
    with open(_NAV, encoding="utf-8") as fh:
        return set(_PATH.findall(fh.read()))


def _on_disk() -> set:
    return {"pages/" + f for f in os.listdir(_PAGES_DIR)
            if f.endswith(".py") and not f.startswith("_")}


def test_every_page_is_reachable_from_the_sidebar():
    missing = sorted(_on_disk() - _linked())
    assert not missing, (
        f"page(s) with no sidebar entry: {missing} — add a NavEntry to "
        f"NAV_SECTIONS in src/pages_lib/navigation.py, or the page is invisible "
        f"to anyone using the app even though it loads fine at /<name>")


def test_no_sidebar_entry_points_at_a_missing_page():
    stale = sorted(_linked() - _on_disk())
    assert not stale, (
        f"sidebar entries pointing at files that do not exist: {stale} — "
        f"st.page_link raises on every page's sidebar, not just this one")


def test_nav_codes_are_unique():
    """The four-letter code is the terminal-style handle; two pages sharing one
    makes the sidebar ambiguous and the keyboard jump land on the wrong page."""
    with open(_NAV, encoding="utf-8") as fh:
        codes = re.findall(r'NavEntry\(\s*"([A-Z0-9]{2,5})"', fh.read())
    dupes = sorted({c for c in codes if codes.count(c) > 1})
    assert not dupes, f"duplicate nav codes: {dupes}"
