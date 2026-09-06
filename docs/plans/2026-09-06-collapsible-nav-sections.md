# Collapsible Nav Sections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every sidebar section collapsible, with the daily path open by
default and the on-demand sections closed, so 62 links stop being one scroll.

**Architecture:** `render_sidebar_nav()` wraps each section in `st.expander`
instead of `st.caption`. Which sections open by default is decided by one pure
function, so it is testable and not buried in rendering.

**Tech Stack:** Python 3.14, Streamlit, pytest.

**Spec:** The owner's request, 2026-09-06: *"Fix the navigation so that sections
🌅 MORNING BRIEF and 📋 PRE-SESSION with the rest should be collapsable"*.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.55**.
- **`NAV_SECTIONS` keeps its `(title, entries)` shape.** The nav guards parse it
  with a regex that expects exactly that; changing to 3-tuples or a dataclass
  would silently stop those guards matching — and a guard that matches nothing
  passes.
- **No link changes.** Same 62 entries, same sections, same order.
- **The nav must render even when a section fails**, as it does today.

---

## Context

The sidebar now has 10 sections and 62 links after the RESEARCH LAB split. That
is better organised but no shorter — every section is always fully drawn, so
reaching REFERENCE at the bottom means scrolling past everything.

**Which sections open by default is not cosmetic**, and the file already argues
the answer in its own header comment: the daily path is *"walked in order, ≤8
touches"*, while Weekend and Research are *"visited on demand"*. So Morning
Brief, Pre-Session and Session open; the rest start closed. That keeps the
routine one glance away and pushes the 39 research links behind a click.

**Why a function rather than a flag per tuple.** Adding a third element to each
tuple would break `tests/test_navigation_covers_every_page.py`, whose section
regex ends at `]\s*\)`. It would still parse *something*, so the suite would stay
green while the guards quietly stopped covering the sections — the worst failure
mode available, and one this session has already hit twice.

---

## Task 1: Decide, then render

**Files:**
- Modify: `src/pages_lib/navigation.py`
- Test: `tests/test_navigation_covers_every_page.py`

**Interfaces:**
- Produces: `section_opens_by_default(title: str) -> bool`

- [ ] **Step 1: Write the failing tests** — Morning Brief, Pre-Session and
      Session open; Weekend, the five research sections and Reference start
      closed; the check survives the emoji and em-dash in each title; an unknown
      section defaults **closed** (a new section should not force itself open).
- [ ] **Step 2: Run, watch them fail.**
- [ ] **Step 3: Implement** — match on the stable ASCII part of the title, since
      the emoji prefix is decoration and may change.
- [ ] **Step 4: Render** each section as `st.expander(title, expanded=...)`,
      keeping the existing per-entry `try/except`.
- [ ] **Step 5: Green.**

---

## Verification

1. **Unit tests green**, including the unknown-section default.
2. **All six existing nav guards still pass** — the tuple shape is unchanged, so
   the section regex still matches; confirmed by the section-size guard still
   reporting real numbers rather than an empty dict.
3. **A page renders** under AppTest with no exception.
4. **Full suite**, known GARCH failures and no third.
5. **Deployed and read back** from the running container.
6. Show the owner the diff. **Never commit.**
