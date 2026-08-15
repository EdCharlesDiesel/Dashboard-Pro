"""Guard: FRED is read through the macro spine, never fetched per-caller.

The sibling of `tests/test_ohlc_spine.py`, for macro data. A module that builds
its own FRED URL picks its own window, timeout and cache policy, and two pages
then disagree about the same series — before `src/services/fred_data.py`
existed, three modules each held their own client.

Scope is `pages/`, `src/pages_lib/` and `src/core/`: the presentation layers
plus the shared helpers they call. `src/data_backbone/data_access.py` is
deliberately **outside** that scope — it is the one HTTP client, the thing the
spine itself delegates to.

`MIGRATING` only ever shrinks; `test_migrating_list_only_names_files_that_still_offend`
deletes an entry's justification the moment its module stops offending.

Pure `ast` — no network, no Streamlit runtime.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent.parent
SEARCH_DIRS = ("pages", "src/pages_lib", "src/core")

# Substrings that only ever appear in a hand-rolled FRED client.
FRED_MARKERS = ("stlouisfed.org", "fredgraph")

# Shrinks to empty as callers migrate. Adding to this set is a deferral, not a fix.
# Seeded 2026-08-15 from what the guard actually found — `daily_cockpit_tab`
# was a fourth client nobody had counted by eye.
# `set()`, never a bare `{}` literal: removing the last entry leaves an empty
# *dict*, and `MIGRATING == set()` is then quietly False. The OHLC guard hit
# exactly this, and seeding this list reproduced it.
MIGRATING: set[str] = set()


def _py_files() -> Iterator[Path]:
    for rel in SEARCH_DIRS:
        base = REPO / rel
        if base.exists():
            yield from base.rglob("*.py")


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _builds_a_fred_url(source: str) -> bool:
    """True if the module embeds a FRED endpoint in a string literal."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(marker in node.value for marker in FRED_MARKERS):
                return True
    return False


def _offends(path: Path) -> bool:
    return _builds_a_fred_url(path.read_text(encoding="utf-8"))


def test_no_module_builds_its_own_fred_url():
    offenders = {
        _rel(p) for p in _py_files()
        if _rel(p) not in MIGRATING and _offends(p)
    }
    assert not offenders, (
        "These build a FRED URL instead of calling "
        "src.services.fred_data.fred_series: " + ", ".join(sorted(offenders))
    )


def test_migrating_list_only_names_files_that_still_offend():
    stale = {
        rel for rel in MIGRATING
        if not (REPO / rel).exists() or not _offends(REPO / rel)
    }
    assert not stale, (
        "Migrated but still exempted — delete these entries so the guard "
        "covers them: " + ", ".join(sorted(stale))
    )


def test_the_one_http_client_is_outside_the_guarded_scope():
    """`data_access` is *supposed* to hold the URL; it is what the spine calls.

    If it ever moves under a guarded directory this test fails loudly rather
    than the guard quietly starting to flag the legitimate client.
    """
    client = REPO / "src/data_backbone/data_access.py"
    assert client.exists(), "the spine's HTTP client moved; re-point this guard"
    assert _offends(client), "data_access no longer holds the FRED endpoint"
    assert not any(client == p for p in _py_files()), (
        "the one legitimate FRED client is inside the guarded scope"
    )


def test_migration_is_complete():
    """MIGRATING exists only to shrink. Empty means the rule is self-enforcing."""
    assert MIGRATING == set(), (
        "Still building their own FRED URL: " + ", ".join(sorted(MIGRATING))
    )
