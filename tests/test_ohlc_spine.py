"""Guard: pages fetch OHLC through the canonical spine, never privately.

``src/services/market_data.py`` fixes one window, resample rule and TTL per
timeframe (``daily_ohlc`` / ``weekly_ohlc`` / ``h4_ohlc`` / ``hourly_ohlc``). A
page that calls ``yf.download`` itself picks a private lookback, and an EMA over
a private lookback is exactly how two pages come to disagree about the same
instrument at the same moment — the drift this repo has hit before.

Widening through the spine's own argument stays legal (``daily_ohlc(t,
period="2y")``); reaching past it does not. Interval, resample rule and TTL are
fixed in ``market_data.py`` and nowhere else.

Two lists carry the exceptions, and the difference between them matters:

* ``LEGITIMATE_NON_SPINE`` — permanent and reasoned. These fetch series the
  spine does not model at all (index futures, bond yields, CFTC positioning),
  so there is no spine call to migrate them to.
* ``MIGRATING`` — temporary. Files that *should* use the spine and do not yet.
  It only ever shrinks; ``test_migrating_list_only_names_files_that_still_offend``
  fails if an entry is left behind after its page is fixed, so the list cannot
  quietly become a second permanent exemption.

Pure ``ast`` — no Streamlit runtime, no network. Mirrors the static-analysis
approach of ``tests/test_no_credential_inputs.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent.parent
SEARCH_DIRS = ("pages", "src/pages_lib")

# Permanent, reasoned exceptions.
#
# The bar is "the spine cannot serve this series", not "this is not a registry
# pair". Two entries were removed on 2026-08-15 after that distinction was
# tested: `disconnect_monitor_tab` migrated ^NDX and DX-Y.NYB through the spine
# without trouble, proving `cached_ohlc` is ticker-generic — so
# `overnight_drift_tab` (ES=F, ^VIX) and `bonds_gold_dxy_app` (^TNX, TIP) had
# no reason to be exempt and were migrated too. Only genuinely non-OHLC
# sources remain.
LEGITIMATE_NON_SPINE = frozenset({
    "src/pages_lib/commodity_cot_lib.py",   # CFTC COT positioning: not OHLC at all
})

# Empty: every page that could migrate has. Adding to this set is not a fix —
# it is a deferral, and `test_migrating_list_only_names_files_that_still_offend`
# will delete the entry for you as soon as the page stops offending.
#
# Annotated and built with `set()` rather than a bare `{}` literal: the last
# entry being removed left `MIGRATING = {}`, which is an empty *dict*, and the
# overlap check below died with a TypeError instead of passing.
MIGRATING: set[str] = set()

_DIRECT_FETCH_PREFIXES = (
    "yf.download", "yf.Ticker",
    "yfinance.download", "yfinance.Ticker",
)


def _py_files() -> Iterator[Path]:
    for rel in SEARCH_DIRS:
        yield from (REPO / rel).rglob("*.py")


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _fetches_directly(tree: ast.AST) -> bool:
    """True if the module calls yfinance's fetchers itself."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.startswith(_DIRECT_FETCH_PREFIXES):
            return True
    return False


def _offends(path: Path) -> bool:
    return _fetches_directly(ast.parse(path.read_text(encoding="utf-8")))


def test_no_page_fetches_ohlc_outside_the_spine():
    offenders = set()
    for path in _py_files():
        rel = _rel(path)
        if rel in LEGITIMATE_NON_SPINE or rel in MIGRATING:
            continue
        if _offends(path):
            offenders.add(rel)
    assert not offenders, (
        "These fetch OHLC privately instead of via src/services/market_data.py: "
        + ", ".join(sorted(offenders))
        + ". Use daily_ohlc/weekly_ohlc/h4_ohlc/hourly_ohlc (widening "
          "period/days is fine), or add a reasoned entry to LEGITIMATE_NON_SPINE."
    )


def test_migrating_list_only_names_files_that_still_offend():
    """A page that has been migrated must be removed from MIGRATING.

    Without this, a finished migration leaves its exemption behind and the
    guard silently stops covering a file it now could.
    """
    stale = set()
    for rel in MIGRATING:
        path = REPO / rel
        if not path.exists() or not _offends(path):
            stale.add(rel)
    assert not stale, (
        "Already migrated (or gone) but still exempted in MIGRATING — delete "
        "these entries so the guard covers them: " + ", ".join(sorted(stale))
    )


def test_permanent_exceptions_still_exist_and_still_fetch():
    """A permanent exception that no longer fetches is just dead config."""
    stale = set()
    for rel in LEGITIMATE_NON_SPINE:
        path = REPO / rel
        if not path.exists() or not _offends(path):
            stale.add(rel)
    assert not stale, (
        "Listed in LEGITIMATE_NON_SPINE but no longer fetches directly (or is "
        "gone) — drop the entry: " + ", ".join(sorted(stale))
    )


def test_exception_lists_do_not_overlap():
    assert not (LEGITIMATE_NON_SPINE & MIGRATING)


def test_migration_is_complete():
    """MIGRATING exists only to shrink. Empty means the rule is self-enforcing.

    Kept as its own test rather than folded into the guard above: this one
    failing means "someone deferred a page", which is a different conversation
    from "someone added a private fetch".
    """
    assert MIGRATING == set(), (
        "Still bypassing the spine: " + ", ".join(sorted(MIGRATING))
    )
