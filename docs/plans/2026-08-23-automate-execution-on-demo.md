# Automate execution — signal to order, on the demo account

**Goal:** Close the one gap between a triple-confluence signal and an order in
MT5: nothing writes to the queue. Wire the producer, apply the schema, and run
the executor **dry-run first** on the demo account, so every gate decision is
observable before a single order is sent.

**Architecture:** No new components. `queue.py`, `gate.py`, `mt5_executor.py`
and `schema.sql` already exist and now live in `src/execution/`. The missing
piece is one call: the scanner's confluence branch — the same branch that already
sends the Telegram alert — also calls `enqueue_signal()`. The executor polls
Postgres from the Windows box; neither side reaches the other over the network.

**Tech Stack:** Python 3.14, Postgres, MetaTrader5, APScheduler.

**Spec:** The owner's request, 2026-08-23: "I am looking at MT5 I need to
automate this", following the EUR/USD alert that the gate blocked on three
counts.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `origin/Production` reads **1.10.41**; 1.10.42 is taken by the refactor in
  progress, so this plan takes **1.10.43**.
- **Demo only.** `account_info` reports `is_demo: true` (Exness-MT5Trial9,
  "DemoProd", $7,141.40). This plan does not touch a live account, and moving to
  one is a separate decision with its own plan.
- **Dry-run is the default and stays the default.** `EXECUTOR_DRY_RUN=1` unless
  the owner explicitly clears it, per run.
- **The gate is not optional and not weakened.** If a signal is blocked, the fix
  is the signal or the config — never loosening the gate to let it through.
- **The executor is launched by hand.** No autostart, no service, no compose
  entry in this plan.

---

## Context

Everything needed exists; one call is missing.

| Piece | State |
|---|---|
| `src/execution/queue.py` | complete — `enqueue_signal`, `claim_batch`, `mark_*`, `expire_stale`, `executor_state` (kill switch), `log_event` |
| `src/execution/gate.py` | complete and tested (88% covered, 21 tests) |
| `src/execution/mt5_executor.py` | complete — poll, gate, size, send, `reconcile()` |
| `src/execution/schema.sql` | complete — `pending_signals`, `execution_log`, `executor_state`, `executed_trades` |
| **the producer** | **missing — `grep enqueue_signal` matches only `queue.py` itself** |

So today a confluence fires, a Telegram alert is sent, and nothing is queued.
That is the whole gap.

**Where the producer goes.** `background_scanner.py` already computes
confluences (`_find_confluences`, line 250) and dispatches alerts through a
`NotifyCache` that dedupes on `dedupe_key()`. The enqueue belongs in that same
branch — the signal that is worth waking someone's phone for is exactly the
signal worth queueing — and `queue.make_signal_id()` gives the idempotency key
so a rescan cannot double-queue.

**What the EUR/USD alert taught us.** Run through the gate it was blocked three
ways: symbol not whitelisted, entry 6,770 points from market, levels synthetic.
Two of those are properties of the *signal*, and the alert reached the phone
anyway because nothing gated it. Wiring the queue behind the gate means a
malformed signal is refused and *logged* rather than acted on — but it also
means the alert path and the execution path can disagree, so the alert should
carry the verdict (Task 5).

**The whitelist is metals-only.** `EXECUTOR_SYMBOLS` defaults to
`XAUUSD,XAGUSD`, while the confluence scanner covers FX. Until that is widened
deliberately, an FX confluence will queue and then be refused by the gate. That
is the correct default — it fails closed — but it must be a decision, not a
surprise.

---

## Task 1: Stand up the schema

**Files:** none modified — `src/execution/schema.sql` applied

- [ ] **Step 1:** Apply the schema to the local Postgres. It is `CREATE TABLE IF
      NOT EXISTS` throughout, so it is safe to re-run.
- [ ] **Step 2:** Confirm all four tables exist and are empty.
- [ ] **Step 3:** Confirm `executor_state` starts **disabled** — the kill switch
      must default to off, so applying the schema cannot arm anything.

---

## Task 2: The producer, behind a flag

**Files:** Modify `src/services/background_scanner.py` · Test
`tests/test_background_scanner.py`

**Interfaces:** enqueue only when `EXECUTOR_ENQUEUE=1`. Absent the flag the
scanner behaves exactly as it does today.

- [ ] **Step 1: Failing tests** — a confluence enqueues exactly one signal; the
      same confluence twice enqueues once (idempotent on `signal_id`); enqueue
      failure does **not** suppress the Telegram alert; and with the flag unset
      nothing is enqueued at all.
- [ ] **Step 2: Run, watch them fail.**
- [ ] **Step 3: Implement** in the existing confluence branch, inside its own
      `try/except` so a queue outage can never take down alerting.
- [ ] **Step 4: Green.**

---

## Task 3: Dry run — prove the decisions before any order

**Files:** none modified

- [ ] **Step 1:** Launch `python -m src.execution.mt5_executor` with
      `EXECUTOR_DRY_RUN=1` against the demo terminal.
- [ ] **Step 2:** Enqueue a **deliberately bad** signal (the EUR/USD one) and
      confirm the gate refuses it with the same three reasons, recorded in
      `execution_log` rather than acted on.
- [ ] **Step 3:** Enqueue a **well-formed metals** signal priced off the live
      quote and confirm the executor sizes it, logs the intended order, and
      **sends nothing** — `get_positions` still returns 0.
- [ ] **Step 4:** Confirm `expire_stale()` retires an old signal so a queue entry
      cannot be filled hours after it was meaningful.

---

## Task 4: One live order on demo, watched

**Files:** none modified

- [ ] **Step 1:** With the owner present, clear `EXECUTOR_DRY_RUN` and arm the
      kill switch for a single well-formed metals signal.
- [ ] **Step 2:** Confirm via `get_positions` that exactly one position opened,
      with the stop attached, at the size the gate computed.
- [ ] **Step 3:** Confirm `executed_trades` and `execution_log` agree with the
      terminal — ticket, price, volume.
- [ ] **Step 4:** Disarm. Leaving it armed is a separate decision.

---

## Task 5: Make the alert and the execution agree

**Files:** Modify `src/services/confluence_alert.py`

- [ ] **Step 1:** Carry the gate verdict into the Telegram body, so an alert that
      execution would refuse says so on the phone instead of reading as a live
      instruction to enter.

---

## Verification

Evidence before claims.

1. **Four tables exist**, `executor_state` disabled by default.
2. **Producer tests green**, including the idempotency and
   alert-survives-queue-failure cases.
3. **Dry run refuses the bad signal** with the three known reasons, logged.
4. **Dry run sends nothing** — `get_positions` returns 0 after a well-formed
   signal is processed.
5. **The single live demo order** matches the terminal on ticket, price and
   volume.
6. **Full suite**, coverage ≥ 80%, the known GARCH failures and no third.
7. Show the owner the diff. **Never commit.**

## What this plan deliberately does not do

- Touch a live-money account.
- Autostart the executor, or add it to compose or the Startup folder.
- Widen `EXECUTOR_SYMBOLS` beyond metals.
- Weaken any gate threshold to make a blocked signal pass.

Each is a separate decision, and each is the kind that is easier to make
deliberately than to discover after the fact.

## What actually happened (Tasks 1-3)

**Task 1 — schema applied.** Four tables plus a view; `executor_state` seeded
`enabled=false, dry_run=true` with `ON CONFLICT DO NOTHING`, so standing it up
armed nothing and re-running cannot re-arm it.

**Task 2 — producer wired**, behind `EXECUTOR_ENQUEUE=1`, in the scanner's
existing confluence branch and inside its own `try/except`. Five tests, and
**five mutations each failing exactly one of them**: ignoring the flag, letting a
queue error escape, enqueuing every confluence instead of only the fresh ones,
inverting `LONG -> buy`, and dropping the slash-stripping.

Three of those assertions did not exist in the first draft, and the first
mutation run exposed why:

- *Direction was never asserted.* Inverting `LONG -> sell` left all tests green —
  the account would have taken the opposite side of every signal, silently, with
  a valid-looking stop and target.
- *`fresh` vs `confluences` could not be distinguished* by a single-confluence
  test, because with one already-alerted signal the whole branch is skipped. Only
  a mixed batch discriminates.
- *The dedupe ledger was unguarded.* Without the inner `try/except` the error
  reaches the outer handler, which skips `filter_new()` — so the signal is never
  marked seen and the same alert fires every cycle, forever. The test only
  checked that the alert was sent, which was true either way.

**Task 3 — dry run on the demo account.** Both queued signals were claimed,
gated and refused, every event stamped `dry_run=t`, and `get_positions` returned
**0**.

```
BLOCKED EURUSD BUY: symbol not in whitelist; entry 6770pts from market
                    (max 300) - stale or stub; levels look synthetic
BLOCKED XAUUSD BUY: stop 14540pts above maximum 5000; spread 182pts above 40
```

**The "well-formed" gold signal was not well-formed**, and the gate said so
better than the plan did. Its stop was ~14,500 points against a 5,000 limit, and
the live spread is 182 points because **the market is closed** — it is the
weekend. Both refusals are correct.

**Two environment facts worth recording.** The executor runs on the host, so its
`DATABASE_URL` must point at **127.0.0.1:5433** — the Docker Postgres. Port 5432
is a *different*, native Windows server that does not even hold this database;
`db_config()` on the host falls back to it and fails with "no password supplied".
And the kill switch is checked before `claim_batch`, so with `enabled=false`
nothing is claimed and nothing is logged — which is why the first dry run looked
inert rather than broken.

**Task 4 is not done and cannot be today.** A live demo order needs a signal that
passes the gate, and no signal can pass while the market is closed and the spread
is 4x the limit. It waits for market hours, with the owner present.

The executor was left **disarmed** (`enabled=false, dry_run=true`).
