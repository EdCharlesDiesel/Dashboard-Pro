"""`pct_change()` must never rely on the default `fill_method`.

The default is version-dependent and the two pandas in play disagree:

    pandas 2.3.3 (the containers):  a gap is padded  -> a fake 0% return
    pandas 3.0.2 (the dev venv):    a gap stays NaN   -> unknown, as intended

So the same code computes different returns in development and production, and
the development answer is the one that looks right. `currency_index` shipped
that way: its docstring promised a holiday gap was "excluded via skipna", but on
the padding version the gap was `0.0`, `skipna` excluded nothing, and a missing
pair contributed flat to a cross-sectional average. CI caught it; the local suite
had passed.

**`.dropna()` is not a defence.** It removes the leading NaN, but padding turns
an *interior* gap into a real `0.0` that `dropna` then keeps — so a market
holiday enters correlations, realised vol and betas as a genuine zero-return
observation, biasing every one of them toward zero.

This guard does not mandate `None`. It mandates that the choice is *written
down*: a site that genuinely wants padding may say `fill_method="pad"`.
"""
from __future__ import annotations

import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOTS = ("src", "pages")

#: A call with no arguments at all — the version-dependent form.
_BARE = re.compile(r"\.pct_change\(\s*\)")

#: Lines that only *talk* about the call. `src/core/horizons.py` documents the
#: old Market Overview behaviour as ``Close.pct_change().iloc[-1]``; rewriting
#: that would falsify a historical note, and a first pass at this fix did
#: exactly that before it was caught.
_PROSE = re.compile(r"^\s*(#|>>>|\.\.\.)|``")


def _offenders() -> list:
    hits = []
    for root in _ROOTS:
        for dirpath, _dirs, files in os.walk(os.path.join(_REPO, root)):
            if "__pycache__" in dirpath:
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if _BARE.search(line) and not _PROSE.search(line):
                            rel = os.path.relpath(path, _REPO).replace("\\", "/")
                            hits.append(f"{rel}:{lineno}")
    return sorted(hits)


def test_no_call_relies_on_the_default_fill_method():
    bad = _offenders()
    assert not bad, (
        f"bare pct_change() at: {bad} — pass fill_method explicitly. The default "
        f"differs between pandas 2.3.3 (containers, pads a gap to 0%) and 3.0.2 "
        f"(dev venv, leaves it NaN), so these compute different returns in "
        f"development and production")


def test_the_guard_ignores_prose_about_the_call():
    """A historical note is not a call site.

    `horizons.py` documents the old behaviour in its module docstring. The first
    pass at this fix rewrote that line, turning an accurate record of what the
    code *used to do* into a false one.
    """
    assert _PROSE.search("reported ``Close.pct_change().iloc[-1]`` on whichever")
    assert _PROSE.search("    # rets = px.pct_change()")
    assert not _PROSE.search("    rets = px.pct_change()")


def test_the_guard_would_catch_a_real_regression():
    assert _BARE.search("rets = px.pct_change()")
    assert _BARE.search("rets = px.pct_change( )")
    assert not _BARE.search("rets = px.pct_change(fill_method=None)")
