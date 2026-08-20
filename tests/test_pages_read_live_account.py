"""Pages that size real positions must consult the live account store.

The 2026-08-20 audit found three pages sizing against hardcoded figures —
$10,000 on two of them and $935 on a third — while four other pages read the
live balance correctly. Nothing failed; the numbers were simply wrong, and one
of them had been wrong long enough that $935 was a balance from months ago.

Deliberately *not* included: pages/vwap-ema-gold.py, whose `initial_capital`
is a backtest starting capital and is meant to be hypothetical.
"""
from __future__ import annotations

import ast
import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pages whose numbers are meant to describe the real account.
LIVE_ACCOUNT_PAGES = [
    "pages/risk-suite.py",
    "pages/swing_playbook_tab.py",
    "pages/threat_board_tab.py",
    "src/pages_lib/setup_ranker.py",
    "src/pages_lib/fib_entry.py",
]

# Figures that are only ever placeholder account sizes.
PLACEHOLDERS = {10000.0, 935.0}


def _source(rel: str) -> str:
    with open(os.path.join(_REPO, rel), encoding="utf-8") as fh:
        return fh.read()


def _hardcoded_account_widgets(src: str) -> list:
    """`number_input(..., value=<placeholder literal>)` calls, found by parsing.

    Parsed, not grepped. The first version of this test matched line by line
    and so missed threat_board_tab.py entirely, where the call is split and
    `value=935.0` sits on the continuation line with no `number_input` on it —
    the test passed while the bug it was written for was still there.
    """
    offenders = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name != "number_input":
            continue
        for kw in node.keywords:
            if kw.arg == "value" and isinstance(kw.value, ast.Constant)                     and isinstance(kw.value.value, (int, float))                     and float(kw.value.value) in PLACEHOLDERS:
                label = node.args[0].value if node.args and isinstance(
                    node.args[0], ast.Constant) else "?"
                offenders.append(f"{label} = {kw.value.value}")
    return offenders


@pytest.mark.parametrize("rel", LIVE_ACCOUNT_PAGES)
def test_page_consults_the_live_account_store(rel):
    src = _source(rel)
    assert ("account_state" in src) or ("account_snapshot" in src), (
        f"{rel} never reads the live account - its figures are whatever is "
        f"typed in, which is how $935 survived for months")


@pytest.mark.parametrize("rel", LIVE_ACCOUNT_PAGES)
def test_no_placeholder_balance_is_a_widget_default(rel):
    """A hardcoded `value=` on the account field is the bug itself.

    A placeholder is fine as the *fallback* argument of `.get(...)` or
    `get_balance(...)` - that is a Call, not a literal - but not as a widget
    default, because that is the number the user sees and sizes against.
    """
    offenders = _hardcoded_account_widgets(_source(rel))
    assert not offenders, f"{rel} hardcodes an account figure: {offenders}"
