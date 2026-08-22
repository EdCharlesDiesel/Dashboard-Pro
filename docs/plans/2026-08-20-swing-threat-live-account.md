# Swing Playbook & Threat Board — read the live account

> **Reconstructed 2026-08-22** from the session transcript. Approved and
> executed on 2026-08-20 but never written to `docs/plans/` at the time — see
> `2026-08-20-plans-in-docs.md`. The text is the approved plan; checkboxes are
> left unticked because the record of which steps ran lives in the transcript.

**Goal:** Stop the last two pages that size against hardcoded account figures — Swing Playbook's `$10,000` and Threat Board's `$935` — and add a guard so a hardcoded figure cannot quietly come back.

**Architecture:** The same live/manual pattern already proven on the Setup Ranker and Risk Suite, reusing `account_state.get_balance()`, `open_positions.account_snapshot()` and `open_positions.age_minutes()`. No new abstraction: the only new code is a repo-level regression test.

**Tech Stack:** Python 3.14, Streamlit, pytest.

**Spec:** The owner's request, 2026-08-20: "fix both pages", following the audit of every `account_bal` reader.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.23**, so this plan takes **1.10.24**.
- Branch `DEV-04/Market-Overview`.
- **Surgical.** Only the account-figure input on each page changes. No sizing maths, no layout restructuring, and `risk_pct` is left alone on both — Swing Playbook divides its own by 100, a different convention from the shared key, and unifying that is a separate change.
- **TDD** applies to the guard test. The Streamlit wiring in `pages/` is outside coverage and is verified by running each page and reading the number — stated plainly rather than pretended otherwise.

---

## Context

The audit found four readers already correct (`setup_ranker.py:287`, `fib_entry.py:202`, `daily_trading/state.py:30`, `daily_trading/sidebar.py:75`) and two wrong:

| Page | Hardcoded | vs live $3,844.15 | Effect |
|---|---|---|---|
| `swing_playbook_tab.py:58` | `10_000.0` | **2.60x** | Sizes weekly plans too large, via `build_playbook(selected, account, risk_pct)` at line 94 |
| `threat_board_tab.py:64` | `935.0` | **0.24x** | Overstates cluster risk ~4.1x, via `build_report(positions, equity, …)` at line 79 |

Swing Playbook is the worse of the two: it never touches `account_bal` or session state at all, so unlike the Risk Suite it could not even inherit the right number by arriving from another page. Threat Board errs the safe way — it makes risk look bigger than it is — but a board that says a 10% cluster is 41% will talk you out of good trades.

**Two design points that are not obvious:**

1. **`setdefault` alone is not a fix.** Seeding `st.session_state.setdefault("threat_equity", live)` sets the value once and then never updates it — the page would open correct at 09:00 and be silently stale by 15:00, which is the same class of bug. The live branch must *assign* on every run, which is exactly why the Setup Ranker's checkbox pattern is shaped the way it is.

2. **Threat Board wants equity, not balance.** It asks for "Account equity" and reports `worst_cluster_pct_equity`. Balance and equity differ by floating P/L — **$3,844.15 balance vs $3,552.45 equity** — so substituting `get_balance()` there would be a subtler version of the same bug. Equity lives only in `open_positions.account_snapshot()`, whose docstring warns it is absent for MT4-statement books, so a missing equity falls back to manual rather than silently borrowing the balance.

---

## Task 1: A guard so this cannot silently return

**Files:** Create `tests/test_pages_read_live_account.py`

- [ ] **Step 1: Write the failing test.** Two assertions per page, over the pages whose numbers are meant to describe the real account (`risk-suite`, `swing_playbook_tab`, `threat_board_tab`, `setup_ranker`, `fib_entry`): each must consult `account_state` or `account_snapshot`, and none may pass a placeholder literal (`10000` / `935`) as a widget default.

  Deliberately **not** included: `pages/vwap-ema-gold.py`, whose `initial_capital` is a backtest starting capital and is meant to be hypothetical.

- [ ] **Step 2: Run it, watch it fail** — expect `swing_playbook_tab.py` and `threat_board_tab.py` failing both tests. The three already-correct pages must **pass**; if they fail, the test is measuring the wrong thing.

---

## Task 2: Swing Playbook

**Files:** Modify `pages/swing_playbook_tab.py` (imports; the `account` input at line 58)

- [ ] **Step 1: Import** `from src.services import account_state, open_positions`.
- [ ] **Step 2: Replace the bare input** with the established pattern, joining the shared `account_bal` key so the page stops being an island. Assign every run, never `setdefault`. Show a red banner past 15 minutes and a caption naming the live source.
- [ ] **Step 3:** Confirm `account` is bound on both branches — it feeds `build_playbook(...)` and `_log_key`.
- [ ] **Step 4:** `python -m py_compile pages/swing_playbook_tab.py`.

---

## Task 3: Threat Board

**Files:** Modify `pages/threat_board_tab.py` (imports; the `equity` input at line 63)

- [ ] **Step 1: Import** `from src.services import open_positions`. Note: **not** `account_state` — this page needs equity, which the balance store does not hold.
- [ ] **Step 2: Put the toggle above the existing 3-column row** so the columns keep their shape, and read **equity** from the snapshot. The manual branch passes **no `value=`**: `threat_equity` is a keyed widget, so Streamlit takes its value from session state, seeded once above with `setdefault` so the first render has a number without re-introducing a hardcoded widget default.
- [ ] **Step 3:** Confirm `equity` is bound on both branches — it feeds `tc.build_report(...)` at line 79.
- [ ] **Step 4:** `python -m py_compile pages/threat_board_tab.py`, then re-run Task 1's tests — all four parametrised cases must now pass.

---

## Verification

1. **The guard goes green**, having failed for both pages in Task 1.
2. **The figures each page will use:** ~3844 balance and ~3552 equity — **different numbers**; that difference is the point.
3. **Both pages render with live figures** — rebuild, then open each in a **fresh tab** (the ordering that hid the Risk Suite bug). No `$10,000`, no `$935`.
4. **The Threat Board's percentages actually moved** — note one `worst_cluster_pct_equity` before and after; expect roughly a 4x reduction.
5. **Full suite:** coverage ≥ 80%, the known pre-existing failures, no new one.
6. **Deploy:** 1.10.24, four containers in sync.
7. Show the owner the diff. **Never commit.**

## What actually happened

Both pages verified in fresh tabs: `Account size: $3,844.15 live · MT5 terminal` and `Account equity (USD) $3,647.24`, with `has_10000` and `has_935` both false.

**The guard caught its own hole first.** The initial version matched line by line and **passed Threat Board** — because its `value=935.0` sits on a continuation line with no `number_input` on it, so the test was green while the bug it existed for was still there. Rewritten to parse with `ast` and inspect the actual `value=` keyword, it then failed both pages correctly and passed all three already-correct pages. A `.get(..., 10000.0)` fallback stays legal: that is a Call, not a widget default.

**Verification step 4 could not be completed.** The board had 0 positions, because `threat_core.load_positions()` reads its own hand-typed `threat_positions` table and `grep` for `open_positions|mt5` in that module returns 0 — it had never seen the real book. That discovery became plan 1.10.25.
