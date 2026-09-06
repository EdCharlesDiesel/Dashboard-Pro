# Pin fill_method On Every pct_change Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the last 21 bare `pct_change()` calls, so no return series
depends on a pandas default that differs between the dev venv and production.

**Architecture:** `fill_method=None` at every site, plus a guard test that fails
on a bare call.

**Tech Stack:** Python 3.14, pandas, pytest.

**Spec:** The owner's request, 2026-09-06: *"fix the other 21 pct_change calls"*,
after 1.10.61 fixed the one CI caught.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.62**.
- **Verify on both pandas versions** — venv 3.0.2 and container 2.3.3. A green
  local suite is exactly what hid the first one.
- **A site that genuinely wants padding keeps it, explicitly.** The goal is that
  the choice is written down, not that one answer is imposed.

---

## Context

`currency_index.py` was fixed at 1.10.61 after CI caught it. The same latent
fault sits in 21 more places, and none pins `fill_method`:

```
pandas 2.3.3 (containers):  [None, 0.01, -0.02, 0.0 ]   <- gap padded to flat
pandas 3.0.2 (dev venv):    [None, 0.01, -0.02, None]   <- gap left unknown
```

**`.dropna()` does not save them.** Most of these calls chain `.dropna()`, which
removes the leading NaN — but padding converts an *interior* gap into a real
`0.0`, and `dropna` then keeps it. So a market holiday enters correlations,
volatility estimates and regressions as a genuine zero-return observation.

That biases everything built on it toward zero: correlations shrink, realised
vol reads low, betas flatten. Silently, and only on the version production runs.

**Judgement per site, one answer in practice.** Every one of the 21 computes a
return series where a missing bar means *unknown*, not *unchanged*, so
`fill_method=None` is right for all of them. The one that looked like an
exception — `cot_..._harness.py:194`, an equity curve — already writes
`.fillna(0)` immediately after, so it states its own intent and is unaffected
either way.

---

## Task 1: Pin every call site

**Files:** the 21 listed in `2026-09-06-pct-change-fill-method.md`

- [ ] **Step 1:** Replace `pct_change()` with `pct_change(fill_method=None)`.
- [ ] **Step 2:** Full suite. **A failure here is information**: it means that
      site depended on padding, and it gets its own decision rather than a
      blanket edit.

---

## Task 2: Guard against the next one

**Files:** Create `tests/test_no_bare_pct_change.py`

- [ ] **Step 1: Write the failing test** — no `pct_change()` without an explicit
      `fill_method` anywhere in `src/` or `pages/`.
- [ ] **Step 2:** Green once Task 1 lands.

---

## Verification

1. **Zero bare calls remain**, asserted by the guard.
2. **Full suite passes locally (pandas 3.0.2)** and the affected tests pass **in
   the container (2.3.3)**.
3. **Any test that changes behaviour is reported**, not silently updated.
4. Show the owner the diff. **Never commit.**
