# This Week's Trades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standing weekly view of what the system actually found — which pairs
fired, on which days, how consistently, and which ones changed their mind —
answering "what happened this week" rather than "what do I take right now".

**Architecture:** Pure aggregation in `src/core/weeks_trades.py` (no Streamlit,
no DB) so it is measured and testable; a thin `BloombergPage` subclass renders
it. The signal read, liveness and side-parsing all come from the existing
`src/core/todays_trades` — this plan adds a *view*, and duplicating consensus
would create a second source of truth for the same question.

**Tech Stack:** Python 3.14, Streamlit, Postgres, pytest.

**Spec:** The owner's request, 2026-09-06: *"another page that I can always use,
something like Today's Trade but This Week's Trades"*.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.52** — the number after the highest existing *claim*
  (1.10.51), not after `VERSION`. Deriving it from `VERSION` collided with two
  of the owner's plans on 2026-09-06.
- **No second consensus.** `consensus()`, `is_live()` and `signal_horizon_days()`
  are imported, never reimplemented.
- **The page owns no maths**, matching every other `pages_lib` module.
- **New page must be linked** — `tests/test_navigation_covers_every_page.py`
  fails otherwise, by design.

---

## Context

**"Today's Trades" is not actually about today.** Its query reads
`logged_at >= CURRENT_DATE - INTERVAL '45 days'`, deliberately: `consensus()`
holds each signal for its own declared horizon, and a one-day window made the
board reverse whenever yesterday's votes aged out — with nothing in the market
having changed. So the page answers *"what is live and takeable now"*, not
*"what fired today"*.

That matters, because the naive build of this page — the same thing with
`INTERVAL '7 days'` — would produce a slightly-staler copy of a board that
already exists, and a second answer to a question that already has one.

**The gap worth filling is different.** Nothing today shows the *flow*: how many
distinct days a pair kept appearing, whether the system held one side all week or
flipped, which sources drove it. That is a review question, asked at the weekend
or before a session, and it is exactly what the 45-day store can already answer
without a single new query shape.

**What "this week" means here:** Monday 00:00 UTC to now. Not a rolling 7 days —
a rolling window slides the boundary every day and makes "this week" mean
something different each time it is opened, which is useless for review.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/core/weeks_trades.py` | **new, measured** — week bounds, per-day grouping, per-pair activity, side flips. Pure. |
| `tests/test_weeks_trades.py` | **new** — the aggregation contract. |
| `src/pages_lib/weeks_trades_page.py` | **new, omitted** — rendering only. |
| `pages/weeks-trades.py` | **new** — entry point, matching `todays-trades.py`. |
| `src/pages_lib/navigation.py` | **modify** — one `NavEntry`. |

---

## Task 1: The weekly aggregation

**Files:**
- Create: `src/core/weeks_trades.py`
- Test: `tests/test_weeks_trades.py`

**Interfaces:**
- Consumes: `src.core.todays_trades._side` semantics (Long/Short), `is_live`
- Produces:
  - `week_start(now: datetime) -> datetime` — Monday 00:00 of `now`'s week
  - `in_week(row, now) -> bool`
  - `by_day(rows, now) -> dict[date, list[dict]]`
  - `PairWeek` dataclass: `pair`, `longs`, `shorts`, `days_seen`, `sources`, `first_seen`, `last_seen`, `flipped`
  - `pair_activity(rows, now) -> list[PairWeek]` — sorted by `days_seen` desc

- [ ] **Step 1: Write the failing tests** — Monday is its own week start; a
      Sunday-night row belongs to the *previous* week; rows outside the week are
      excluded; a pair seen on three days reports `days_seen == 3` and not three
      separate entries; a pair with both sides reports `flipped=True`; a pair with
      one side does not; ordering is by `days_seen` descending.
- [ ] **Step 2: Run, watch them fail** (`ModuleNotFoundError`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Green**, bump, record.

---

## Task 2: The page

**Files:**
- Create: `src/pages_lib/weeks_trades_page.py`, `pages/weeks-trades.py`
- Modify: `src/pages_lib/navigation.py`

- [ ] **Step 1:** Subclass `BloombergPage` with `configure` / `sidebar` / `body`,
      reusing `TodaysTradesPage`'s signal read rather than writing a second one.
- [ ] **Step 2:** Body: the week's date range, a per-pair activity table
      (days seen, long/short counts, sources, flipped), and a per-day breakdown.
- [ ] **Step 3:** Empty-week state that says *why* it is empty (no signals
      persisted, or the DB is unreachable) — never a blank page.
- [ ] **Step 4:** `NavEntry` under **PRE-SESSION**, beside Today's Trades.
- [ ] **Step 5:** Nav guard green; page renders under AppTest with a non-zero
      widget count.

---

## Verification

1. **Unit tests green**, including the Sunday-boundary case.
2. **Coverage ≥ 80%**; `src/core/weeks_trades.py` measured and carrying its own
   tests.
3. **The page renders** under AppTest — **no exception AND a non-zero widget
   count**, since a page that draws nothing also raises nothing.
4. **Nav guard green** — 61 pages, 61 entries.
5. **Full suite**, the known GARCH failures and no third.
6. **Deployed and visible** in the running container.
7. Show the owner the diff. **Never commit.**
