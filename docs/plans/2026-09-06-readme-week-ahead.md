# README: Sync The Map, Add The Week-Ahead Playbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `docs/README.md`'s page map back in step with the sidebar, and
add a short playbook for actually trading the coming week with it.

**Architecture:** Regenerate the section tables *from* `navigation.py` rather
than retyping them, preserving every existing description. Add a guard so the
two cannot drift again.

**Tech Stack:** Python 3.14, pytest.

**Spec:** The owner, 2026-09-06 (Sunday): *"update the read me file on how I can
use this system going forward to trade this week coming"*.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.65**.
- **Not one existing description is rewritten.** They are the owner's words about
  what each page means; this change moves them, it does not edit them.
- **`navigation.py` stays the single source of truth** — the README already says
  "if the two ever disagree, the code wins".

---

## Context

`navigation.py` carries an instruction nothing enforced:

> *"This is the single source of truth for both the sidebar and README/System_Guide's
> walkthroughs — keep those in sync when reordering."*

Today's nav work broke that. Measured:

- **12 pages are in the sidebar and absent from the README** — including
  `Today's Trades`, `Threat Board` and everything added this session.
- The README still documents **one 🔬 Research Lab section**, which was split
  into five (Cross-Asset & Macro, Structure & Setups, Trend & Momentum, Quant &
  Models, Events & Seasonality) at 1.10.53.

A page map that silently lags is worse than none: it is read as authoritative
and quietly omits a third of the tools.

**Why regenerate rather than edit.** 62 entries retyped by hand is how one gets
dropped, and a dropped row in a *document* fails even more quietly than in code —
nothing errors, the page just stops existing as far as the reader is concerned.
The existing descriptions are extracted by page path and re-emitted verbatim.

---

## Task 1: A guard, first

**Files:** Create `tests/test_readme_matches_nav.py`

- [ ] **Step 1: Write the failing test** — every `NAV_SECTIONS` entry appears in
      `docs/README.md`, and every section heading does too.
- [ ] **Step 2: Run, watch it fail** naming the 12 missing pages.

---

## Task 2: Regenerate the map

**Files:** Modify `docs/README.md`

- [ ] **Step 1:** Extract the existing `path -> description` pairs.
- [ ] **Step 2:** Re-emit the tables in nav order, under the current section
      headings, keeping every existing description verbatim.
- [ ] **Step 3:** Write descriptions for the 12 new pages only.
- [ ] **Step 4:** Guard green.

---

## Task 3: The week-ahead playbook

**Files:** Modify `docs/README.md`

- [ ] **Step 1:** A "Trading the week ahead" section: what to do Sunday, what to
      do each morning, what is frozen at session open, what to review Friday.
- [ ] **Step 2:** State plainly what is **not** ready — the executor is disarmed
      and dry-run only, and alerts are informational.

---

## Verification

1. **Guard passes** — 62 of 62 entries and every section heading documented.
2. **No existing description changed** — asserted by diffing the extracted map
   before and after.
3. **Full suite**, known GARCH failures and no third.
4. Show the owner the diff. **Never commit.**
