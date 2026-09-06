# Nav Guard Covers the Root Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the navigation guards to cover `app.py`, the one nav entry
currently outside their scope, without dragging in the root-level scripts that
are not pages.

**Architecture:** Two small changes in `tests/test_navigation_covers_every_page.py`
— the path pattern and the on-disk set. No production code moves.

**Tech Stack:** Python 3.14, pytest.

**Spec:** The owner's request, 2026-09-06: *"yes close that too"*, after the
guards were found to parse 61 of 62 nav entries.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.56**.
- **Scripts are not pages.** `run_backtest.py`, `run_backtest_diag.py` and
  `run_backtest_edge.py` sit at the repo root and must never be required to
  carry a nav entry.
- The five existing nav guards keep passing.

---

## Context

The guards match `"pages/....py"`, so they see 61 of the 62 nav entries. The
missing one is **`app.py`** — the `CHCK` "Daily Checklist (18-point)" link under
REFERENCE, and the root of the Streamlit app.

Nothing is broken today: `_linked()` and `_on_disk()` use the same pattern, so
both sides exclude it consistently and every assertion still holds. But it means
the root link is the one entry that could be deleted, renamed or pointed at a
missing file with no test noticing — the exact failure the guards exist to catch,
in the one place they do not look.

**The reason it cannot simply be "all root `.py` files"**: three of them are
scripts, not pages. Requiring nav entries for `run_backtest*.py` would make the
guard fail on a healthy repo, and the usual fix for that is to weaken the guard.

---

## Task 1: Widen the pattern, precisely

**Files:** Modify `tests/test_navigation_covers_every_page.py`

- [ ] **Step 1: Write the failing tests** — `app.py` is inside the guard's
      on-disk set; it is linked; and `run_backtest.py` is **not** required.
- [ ] **Step 2: Run, watch them fail.**
- [ ] **Step 3: Implement** — match `app.py` or `pages/*.py` exactly, and add
      `app.py` to the on-disk set only when it exists.
- [ ] **Step 4: Green.**

---

## Verification

1. **All guards pass**, now over 62 entries rather than 61.
2. **Mutation:** removing the `CHCK` entry from the nav fails a guard; it did
   not before this change.
3. **`run_backtest*.py` is still not demanded** — asserted directly.
4. **Full suite**, known GARCH failures and no third.
5. Show the owner the diff. **Never commit.**
