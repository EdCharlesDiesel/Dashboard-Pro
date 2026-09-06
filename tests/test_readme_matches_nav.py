"""The README's page map must match the sidebar.

`navigation.py` says so itself — *"the single source of truth for both the
sidebar and README/System_Guide's walkthroughs — keep those in sync when
reordering"* — and the README agrees: *"if the two ever disagree, the code
wins"*. Nothing enforced it, and on 2026-09-06 the nav gained 12 pages and was
resectioned while the README still described a single Research Lab.

A stale page map fails quietly. Nothing errors; a third of the tools simply stop
existing as far as the reader is concerned.
"""
from __future__ import annotations

import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_README = os.path.join(_REPO, "docs", "README.md")


def _readme() -> str:
    with open(_README, encoding="utf-8") as fh:
        return fh.read()


def _linked_paths(text: str) -> set:
    """Page paths the README links, as `../pages/x.py` or `../app.py`."""
    return set(re.findall(r"\(\.\./((?:pages/)?[A-Za-z0-9._\-]+\.py)\)", text))


def test_every_sidebar_page_is_documented():
    from src.pages_lib.navigation import NAV_SECTIONS

    text = _readme()
    linked = _linked_paths(text)
    missing = sorted({e.path for _t, es in NAV_SECTIONS for e in es} - linked)
    assert not missing, (
        f"in the sidebar but absent from docs/README.md: {missing} — the README "
        f"is read as the map of the system, so an undocumented page is an "
        f"invisible one")


def test_every_section_heading_appears(): 
    from src.pages_lib.navigation import NAV_SECTIONS

    text = _readme()
    # Match on the ASCII core: headings carry emoji and an em-dash suffix that
    # the README renders slightly differently.
    missing = []
    for title, _entries in NAV_SECTIONS:
        # Keep the hyphen: stripping it turns "PRE-SESSION" into "PRESESSION"
        # and "CROSS-ASSET" into "CROSSASSET", so the matcher failed on headings
        # that were perfectly correct.
        core = re.sub(r"[^A-Z&\- ]", "", title).strip()
        core = re.split(r"\s{2,}", core)[0].strip()
        if core and core.lower() not in text.lower():
            missing.append(core)
    assert not missing, f"nav sections with no README heading: {missing}"


def test_the_readme_documents_no_page_that_does_not_exist():
    """A link to a deleted page is a dead end for whoever follows it."""
    dead = [p for p in _linked_paths(_readme())
            if not os.path.exists(os.path.join(_REPO, p))]
    assert not dead, f"README links to missing pages: {sorted(dead)}"
