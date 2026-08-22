# Threat sentry hook — same book, same equity, and never silent

> **Reconstructed 2026-08-22** from the session transcript. Approved and
> executed on 2026-08-20 but never written to `docs/plans/` at the time — see
> `2026-08-20-plans-in-docs.md`. The text is the approved plan; checkboxes are
> left unticked because the record of which steps ran lives in the transcript.

**Goal:** Make `threat_sentry_hook` agree with the Threat Board page — same positions, same equity — and make it refuse to judge a stale book while saying so out loud rather than going quiet.

**Architecture:** Three small changes to one 90-line module, reusing what the page already uses: `open_positions.load()` + `tc.positions_from_book()` for positions, `open_positions.account_snapshot()` for equity, `open_positions.age_minutes()` for freshness. Plus the module's first tests — it is inside `--cov=src` and currently has none.

**Tech Stack:** Python 3.14, pytest.

**Spec:** The owner's request, 2026-08-20 ("fix the sentry hook too"), with the stale-feed decision: skip the check and alert that the feed is dead.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.26**, so this plan takes **1.10.27**.
- Branch `DEV-04/Market-Overview`.
- **TDD.** `src/core/threat_sentry_hook.py` is measured by coverage and has **no test file**; every change here arrives with one.
- **Do not wire it into a running service.** The request was to fix it, not switch it on — and switching it on is blocked anyway: `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` appear nowhere in `docker-compose.yml` or `.env`, so `_send_telegram` would only print.
- **No test may send a Telegram message.** `_send_telegram` is stubbed in every test.

---

## Context

The hook is dormant — nothing imports it, `evening_sentry` is not scheduled in compose or the Startup items, and the Telegram vars are unset. It is still worth fixing now, because the moment it is switched on it would contradict the page it is supposed to be watching:

| | Page (as of 1.10.26) | Hook today |
|---|---|---|
| Positions | `open_positions.load()` — the MT5 book | `tc.load_positions(conn)` — the hand-typed table, now empty |
| Equity | live from `account_snapshot()` | caller-supplied; the smoke test defaults to `935` |
| Stale data | red banner past 15 min | no concept of staleness |

So today it would evaluate **zero positions**, return `None`, and never alert at all — a sentry that is silent by construction. With the table empty, its `if not positions: return None` is indistinguishable from "all clear".

Two further details:

- **`_format_alert` picks its "Top driver" as `max(components, key=...)`**, the highest raw score. Since 1.10.26 the authoritative field is `detail["state_driver"]`, the components that actually set the headline. On today's book both give "concentration", so this is about not letting the two drift apart later.
- **Unstopped positions are invisible to the alert.** `positions_from_book` returns them separately; a position with no stop is precisely what a threat sentry should shout about, and one such leg is open right now (EUR/ZAR, `sl: 0`).

---

## Task 1: Read the same book and the same equity

**Files:** Modify `src/core/threat_sentry_hook.py` · Create `tests/test_threat_sentry_hook.py`

**Interfaces:** `run_threat_check(engine, equity: float | None = None, zone=...)` — `equity` becomes optional and defaults to the stored snapshot. Existing callers passing an equity keep working unchanged.

- [ ] **Step 1: Failing tests** — it reads the MT5 book and not the hand-typed table; equity defaults to the stored snapshot; an explicit equity still wins; and **no equity anywhere returns `None` rather than guessing** (never fall back to a constant — that is the $935 bug).
- [ ] **Step 2: Run, watch them fail.**
- [ ] **Step 3: Implement** — swap `tc.load_positions(conn)` for `open_positions.load()` through `tc.positions_from_book()`, and default `equity` from `account_snapshot().get("equity")`.
- [ ] **Step 4: Green.**

---

## Task 2: Refuse to judge a stale book, and say so

**Interfaces:** module constant `STALE_AFTER_MIN = 15`, matching the pages.

- [ ] **Step 1: Failing tests** — a stale book is not evaluated; a stale book **alerts that the feed is dead**; the dead-feed alert does not repeat every run; a fresh book is evaluated normally; an unknown age is treated as fresh (that is the empty-book case, already handled).
- [ ] **Step 2: Run, watch them fail.**
- [ ] **Step 3: Implement.** Before evaluating, read `age_minutes()`; past the threshold, send a feed-dead alert and return `None`. Suppress repeats by journaling a `stale` state and comparing with `tc.last_state(conn)` — the same transition rule the threat alerts already use, so one dead feed produces one message rather than one every cycle.
- [ ] **Step 4: Green.**

---

## Task 3: Say what the alert should say

- [ ] **Step 1: Failing tests** — the driver comes from `state_driver`, not the top raw score, with a fallback for reports journaled before 1.10.26; unstopped positions are named in the alert.
- [ ] **Step 2: Implement**, putting the unstopped pairs into `detail` before journaling.
- [ ] **Step 3: Fix the `__main__` smoke test** — drop `EQUITY` defaulting to `935` and use the live snapshot like everything else.
- [ ] **Step 4: Green.**

---

## Verification

1. **Unit tests** — this module had none at all before today.
2. **No Telegram message can escape a test run:** every test reaching an alert path stubs `_send_telegram`; `TELEGRAM_BOT_TOKEN` is unset so even an unstubbed call would print.
3. **Hook and page agree on the same book**, run against real data.
4. **The hook is still dormant** — fixing it must not switch it on.
5. **Full suite:** coverage should rise, since a measured module gains its first tests.
6. **Deploy:** 1.10.27, four containers in sync.
7. Show the owner the diff. **Never commit.**

## What actually happened

14 tests written and passing for a module that had none. The rendered alert reads:

```
🔴 *Threat Board: green → red*  (30.0/100)
Worst correlated stop-out: $6,080 (170.3% eq, ZAR)
Driver: Concentration (100)
⚠️ NO STOP on: EUR/ZAR
```

One expectation changed under execution: the plan predicted 8 usable / 1 unstopped, and the run returned **9 / 0**. Checked against the terminal rather than assumed — ticket `3114720333`, the EUR/ZAR leg that had shown `sl: 0`, had since been given `sl: 18.9549`. The board flagged an unstopped position and it was subsequently stopped, so the count was correct rather than a regression.
