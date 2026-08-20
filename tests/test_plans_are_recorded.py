"""Every version bump leaves a plan behind.

`.claude/CLAUDE.md` has required plans in `docs/plans/` since long before
2026-08-20, and they still stopped being written after two uses — because
nothing failed when they were not. VERSION reached 1.10.28 while the newest
plan named 1.10.19, and one of those nine unexplained versions had already been
committed.

A rule with no failure mode is a preference. This is the failure mode.
"""
from __future__ import annotations

import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLANS = os.path.join(_REPO, "docs", "plans")
_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.md$")


def _plan_files() -> list:
    return [f for f in os.listdir(_PLANS) if f.endswith(".md")]


def _read(name: str) -> str:
    with open(os.path.join(_PLANS, name), encoding="utf-8") as fh:
        return fh.read()


def test_the_current_version_is_claimed_by_a_plan():
    """A plan must *claim* the current VERSION, not merely mention it.

    The first version of this test asked whether the version string appeared
    anywhere in any plan, and passed instantly - satisfied by a plan quoting
    1.10.28 as the version it was bumping *from*. Every plan names its
    predecessor, so that check could never fail. The claim is the assertion
    that matters: "so this plan takes **1.10.29**".
    """
    with open(os.path.join(_REPO, "VERSION"), encoding="utf-8") as fh:
        version = fh.read().strip()
    claim = re.compile(r"takes \*\*" + re.escape(version) + r"\*\*")
    hits = [f for f in _plan_files() if claim.search(_read(f))]
    assert hits, (
        f"VERSION is {version} but no plan in docs/plans/ claims it "
        f"(expected a line reading 'takes **{version}**') - the change that "
        f"bumped it has no written plan")


def test_plan_filenames_follow_the_convention():
    # YYYY-MM-DD-<slug>.md, so the directory sorts chronologically and a plan
    # can be found from a date without opening anything.
    bad = [f for f in _plan_files() if not _NAME.match(f)]
    assert not bad, f"misnamed plans: {bad}"


def test_the_plans_directory_is_not_empty():
    assert _plan_files(), "docs/plans/ is empty"
