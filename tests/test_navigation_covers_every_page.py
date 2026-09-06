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
# `app.py` is named explicitly rather than allowing any root-level file: three
# scripts (run_backtest*.py) sit beside it and are not pages, so a permissive
# root pattern would demand nav entries for them and fail on a healthy repo.
_PATH = re.compile(r'"(app\.py|pages/[A-Za-z0-9._\-]+\.py)"')

#: The Streamlit entry point — a nav entry (CHCK) and a page, but not in pages/.
_ROOT_PAGE = "app.py"


def _linked() -> set:
    with open(_NAV, encoding="utf-8") as fh:
        return set(_PATH.findall(fh.read()))


def _on_disk() -> set:
    pages = {"pages/" + f for f in os.listdir(_PAGES_DIR)
             if f.endswith(".py") and not f.startswith("_")}
    if os.path.exists(os.path.join(_REPO, _ROOT_PAGE)):
        pages.add(_ROOT_PAGE)
    return pages


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


_SECTION = re.compile(r'\(\s*"([^"]+)",\s*\[(.*?)\]\s*\)', re.S)

#: Above this, a sidebar section stops being scannable and becomes a list to
#: read. RESEARCH LAB reached 31 -- half the app in one bucket -- because it was
#: named for when you *don't* use a page, so everything untimed accumulated there.
MAX_SECTION_ENTRIES = 20


def _sections() -> dict:
    with open(_NAV, encoding="utf-8") as fh:
        body = fh.read()
    body = body[body.index("NAV_SECTIONS"):]
    return {title: _PATH.findall(block) for title, block in _SECTION.findall(body)}


def test_no_section_is_too_long_to_scan():
    oversized = {t: len(p) for t, p in _sections().items()
                 if len(p) > MAX_SECTION_ENTRIES}
    assert not oversized, (
        f"section(s) past {MAX_SECTION_ENTRIES} entries: {oversized} - split by "
        f"the question the pages answer, the way the other sections are named")


def test_every_page_belongs_to_exactly_one_section():
    """A page listed twice appears twice in the sidebar and is ambiguous; the
    set-based guard above cannot see it, because a set collapses duplicates."""
    seen = [path for paths in _sections().values() for path in paths]
    dupes = sorted({p for p in seen if seen.count(p) > 1})
    assert not dupes, f"page(s) in more than one nav section: {dupes}"


def test_the_sections_account_for_every_entry():
    # Guards the reshuffle itself: an entry dropped between sections still
    # leaves valid Python and a page that loads at its URL - it simply vanishes
    # from the sidebar, which is exactly how the platinum tab went missing.
    in_sections = {p for paths in _sections().values() for p in paths}
    assert in_sections == _linked(), (
        f"entries outside any section: {sorted(_linked() - in_sections)}")


class TestSectionDefaults:
    """Which sections open by default.

    Not cosmetic: the module header already argues the answer — the daily path
    is "walked in order, <=8 touches" while Weekend and Research are "visited on
    demand". So the routine stays one glance away and the 39 research links go
    behind a click.
    """

    def test_the_daily_path_opens(self):
        from src.pages_lib.navigation import section_opens_by_default
        for title in ("🌅 MORNING BRIEF",
                      "📋 PRE-SESSION",
                      "⚡ SESSION — execution only"):
            assert section_opens_by_default(title), title

    def test_on_demand_sections_start_closed(self):
        from src.pages_lib.navigation import section_opens_by_default
        for title in ("📅 WEEKEND — weekly bias & COT",
                      "🔗 CROSS-ASSET & MACRO — what is moving what",
                      "🧮 QUANT & MODELS — what the maths says",
                      "📚 REFERENCE"):
            assert not section_opens_by_default(title), title

    def test_an_unknown_section_defaults_closed(self):
        # A new section must not force itself open on every page; opening is
        # opt-in, so adding one cannot quietly lengthen the sidebar again.
        from src.pages_lib.navigation import section_opens_by_default
        assert not section_opens_by_default("🆕 SOMETHING NEW")

    def test_matching_survives_the_emoji_and_dash(self):
        # The title carries an emoji prefix and an em-dash; matching must key on
        # the stable ASCII part, not the whole decorated string.
        from src.pages_lib.navigation import section_opens_by_default
        assert section_opens_by_default("PRE-SESSION")
        assert section_opens_by_default("📋 PRE-SESSION — build the shortlist")

    def test_every_real_section_gets_a_decision(self):
        from src.pages_lib.navigation import NAV_SECTIONS, section_opens_by_default
        opened = [t for t, _ in NAV_SECTIONS if section_opens_by_default(t)]
        assert len(opened) == 3, f"expected the 3 daily-path sections, got {opened}"


class TestTheRootEntryIsCovered:
    """`app.py` is a nav entry too, and was the one the guards could not see.

    It is the Streamlit root and the CHCK "Daily Checklist" link. Because
    `_linked()` and `_on_disk()` shared a pages/-only pattern, both sides
    excluded it consistently and every assertion still passed — so it could have
    been deleted, renamed, or pointed at nothing with no test objecting.
    """

    def test_the_root_page_is_in_the_on_disk_set(self):
        assert "app.py" in _on_disk()

    def test_the_root_page_is_linked_from_the_nav(self):
        assert "app.py" in _linked()

    def test_root_level_scripts_are_not_treated_as_pages(self):
        # run_backtest*.py live beside app.py and are scripts. Demanding nav
        # entries for them would fail on a healthy repo, and the usual response
        # to that is to weaken the guard.
        disk = _on_disk()
        assert not [p for p in disk if p.startswith("run_backtest")]
