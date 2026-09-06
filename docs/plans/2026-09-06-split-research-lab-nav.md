# Split the Research Lab Nav Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the 31-entry RESEARCH LAB section into five sections grouped by
what a trader is actually trying to answer, so the sidebar can be scanned instead
of read.

**Architecture:** Data-only change to `NAV_SECTIONS` in
`src/pages_lib/navigation.py`. No entry is retyped: the existing tuples are read,
regrouped by code, and written back, so the set of links is provably unchanged.

**Tech Stack:** Python 3.14, Streamlit, pytest.

**Spec:** The owner's request, 2026-09-06: *"split the research lab section"*,
after RESEARCH LAB was found to hold 31 of 61 entries.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.53** — the number after the highest existing *claim*.
- **Not one link may change.** Same codes, labels, icons and paths; only the
  section each sits in, and the order within it.
- **No page may be dropped or duplicated**, which is the whole risk of
  reshuffling 31 hand-written entries.
- `tests/test_navigation_covers_every_page.py` must stay green.

---

## Context

The sidebar has 61 entries in 6 sections, and one of them holds half of them:

| Section | Entries |
|---|---|
| MORNING BRIEF | 4 |
| PRE-SESSION | 7 |
| SESSION | 4 |
| WEEKEND | 13 |
| **RESEARCH LAB** | **31** |
| REFERENCE | 3 |

The other sections are named for *when* you use them — morning, pre-session,
in-session, weekend. RESEARCH LAB is named for when you *don't*, so everything
that is not time-bound has accumulated there. It is a residue, not a category,
and at 31 items it is no longer navigable.

**The grouping is by question, not by technique**, so it matches how the rest of
the nav already reads:

| New section | Question it answers | Entries |
|---|---|---|
| CROSS-ASSET & MACRO | what is moving what? | DXAU CIDX XPTU IDXC DISC CMEF SMRT RSKR |
| STRUCTURE & SETUPS | where is the level? | STRC 4HCZ RIBN PIVB LQHT 20DB AMD VWAP |
| TREND & MOMENTUM | which way, how strong? | DTRN DMCD LEAD |
| QUANT & MODELS | what does the maths say? | QNTM STOC FCST PRED PRDX MTGL HOLD ONDR |
| EVENTS & SEASONALITY | what does the calendar say? | EVNT SURP EWKV SEAS |

8 + 8 + 3 + 8 + 4 = **31**. A three-entry section is not a problem: MORNING BRIEF
and SESSION both have four.

**Why this is done programmatically.** Retyping 31 `NavEntry(...)` lines is
precisely how one gets dropped, and a dropped entry does not error — the page
still loads at its URL and simply vanishes from the sidebar, which is the exact
failure that hid the platinum tab. The existing tuples are parsed out and
re-emitted, and the before/after set is compared.

---

## Task 1: Regroup, without retyping

**Files:** Modify `src/pages_lib/navigation.py`

- [ ] **Step 1:** Capture the current `(code, label, icon, path)` set.
- [ ] **Step 2:** Rewrite the RESEARCH LAB block as the five sections above,
      re-emitting each entry verbatim from the captured tuples.
- [ ] **Step 3:** Assert the before/after set is **identical** — same size, no
      additions, no losses.
- [ ] **Step 4:** Nav guard green; the module imports and `NAV_SECTIONS` parses.

---

## Task 2: Guard the invariant

**Files:** Modify `tests/test_navigation_covers_every_page.py`

- [ ] **Step 1: Write the failing test** — every nav entry belongs to exactly one
      section, and no section exceeds 20 entries (the threshold at which a
      sidebar section stops being scannable and starts being a list).
- [ ] **Step 2:** Green.

---

## Verification

1. **The entry set is byte-identical** before and after, compared as a set of
   tuples — not eyeballed.
2. **61 pages, 61 entries**, no duplicate paths or codes.
3. **The sidebar renders** under AppTest with no exception.
4. **Full suite**, the known GARCH failures and no third.
5. **Deployed and visible.**
6. Show the owner the diff. **Never commit.**
