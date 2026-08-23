# Fit the fifteen dropped files into Dashboard-Pro's actual structure

**Goal:** Take the 15 files dropped flat into `src/`, remove the eight that
belong to a different application, and place the seven that are genuinely
Dashboard-Pro where the repo's own conventions put that kind of code — with
their imports repaired so they actually load.

**Architecture:** No new behaviour. Files move, imports are rewritten to match,
and the two pure-logic modules gain tests because they land inside the coverage
gate. One genuinely new package, `src/execution/`, because the queue/gate/
executor trio is a coherent subsystem the repo has no home for.

**Tech Stack:** Python 3.14, Streamlit, Postgres, APScheduler, pytest.

**Spec:** The owner's request, 2026-08-23: "New files added Clean and refactor
please."

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `origin/Production` reads **1.10.41**, so this plan takes **1.10.42**.
  It was first written against 1.10.35 and claimed 1.10.36; Production moved
  on while this was in progress, so 1.10.36 would have been a downgrade.
- **Nothing is deleted without the owner saying so.** Files that do not belong
  are reported, not removed on my judgement.
- **`mt5_executor.py` is not wired into anything.** Placing a file is not the
  same as switching it on — see the decision below.
- **The 80% coverage floor must hold.** Adding ~2,000 unmeasured lines under
  `src/` would drop the suite to roughly 67% and fail it.

---

## Context

Fifteen files were added flat into `src/`, a directory that otherwise contains
only packages (`core/`, `services/`, `db/`, `ui/`, `pages_lib/`, …). They are
two unrelated groups.

**Eight belong to a different application.** `src/README.md` documents an
*Angular Material* frontend for a project called **orion-dashboard**, talking to
a .NET API at `https://localhost:7001/api/trade-plans` and referencing an
ASP.NET `Program.cs` CORS policy:

```
README.md  app.module.ts  trade-plan.service.ts  count-direction.pipe.ts
close-plan-dialog.component.ts  trade-plan-dashboard.component.{ts,html,scss}
```

Dashboard-Pro is Streamlit. It has no Angular app, no TypeScript build, no .NET
API — `package.json`, `angular.json` and `tsconfig.json` do not exist here. These
files cannot run, cannot be imported, and cannot be tested in this repo.

**Seven are genuinely Dashboard-Pro, written against a layout that does not
exist.** Each carries a `Drop at:` line naming a directory this repo has never
had — `src/execution/`, `src/engine/`, `src/jobs/`, `src/ui/tabs/` — and their
imports already reference those paths, so they are broken where they sit:

```
src/mt5_executor.py:31  from src.execution.gate import (...)
src/mt5_executor.py:40  from src.execution.queue import (...)
src/platinum_tab.py:22  from src.engine.platinum import (...)
```

All seven compile under 3.14; none of them *import*. `platinum_tab.py`'s
`from src.ui.theme import BloombergTheme` is the one cross-reference that is
already correct.

**Coverage is the constraint that decides placement.** `pyproject.toml` scopes
coverage to `src` and omits Streamlit pages, rendering, Postgres, and network
fetchers — "none unit-testable without a running app, DB, or Yahoo". The new
files split cleanly along that same line, so the existing rule decides where
each one goes rather than a fresh judgement call.

---

## Task 1: Report the eight, move nothing

**Files:** none modified

- [ ] **Step 1:** List the eight Angular/.NET files for the owner with the
      evidence above, and **stop**. They are currently `git add`-ed, so they
      would enter history on the next commit.
- [ ] **Step 2:** On the owner's word, either `git rm --cached` them and move
      them to a scratch directory outside the repo, or leave them untouched.

---

## Task 2: Place the pure-logic modules, with tests

**Files:** Create `src/core/platinum.py`, `src/execution/gate.py`,
`src/execution/__init__.py` · Create their test modules

Both are pure — no DB, no MT5, no Streamlit — so both land **inside** the
coverage gate and need tests to keep the floor.

| From | To | Why |
|---|---|---|
| `src/gate.py` | `src/execution/gate.py` | pure sizing/veto logic |
| `src/platinum.py` | `src/core/platinum.py` | the repo puts pure engines in `core/` (`threat_core`, `quant_models`) — not the invented `src/engine/` |

- [ ] **Step 1: Tests first for `gate.py`.** Its own docstring calls it "the only
      thing standing between a malformed signal and your account" — a rejected
      signal, a malformed one, zero/negative sizing, and the daily-loss cutoff.
- [ ] **Step 2: Tests for `platinum.py`** — the dollar-factor residual is the
      claim the module exists to make; assert it on constructed series.
- [ ] **Step 3:** Move both, fix imports, green.

---

## Task 3: Place the I/O modules, and omit them for the documented reason

**Files:** Create `src/execution/queue.py`, `src/execution/mt5_executor.py`,
`src/data_backbone/platinum_jobs.py`, `src/pages_lib/platinum.py`,
`pages/platinum_tab.py` · Modify `pyproject.toml`

| From | To | Coverage |
|---|---|---|
| `src/queue.py` | `src/execution/queue.py` | omit — Postgres |
| `src/mt5_executor.py` | `src/execution/mt5_executor.py` | omit — MT5 terminal + DB |
| `src/platinum_jobs.py` | `src/data_backbone/platinum_jobs.py` | omit — DB writes; `worker.py` is where APScheduler already lives |
| `src/platinum_tab.py` | `src/pages_lib/platinum.py` + `pages/platinum_tab.py` | already omitted (`src/pages_lib/*`), covered by the 57-page AppTest sweep |

- [ ] **Step 1:** Move each, rewrite `src.execution.*` / `src.engine.*` imports.
- [ ] **Step 2:** Add the three new omits to `pyproject.toml` **with the same
      one-line rationale style** the existing entries use.
- [ ] **Step 3:** `schema_execution.sql` → `src/execution/schema.sql`, beside the
      code that owns it. (`backups/` is the only other `.sql` location and is
      generated dumps, not source.)
- [ ] **Step 4:** The new page joins the sweep automatically — `_discover_pages()`
      globs `pages/*.py`, so it is covered the moment it lands.

---

## Verification

1. **Every moved module imports** — `python -c "import src.execution.gate"` and
   so on for all seven. This is the check that fails today.
2. **Tests for the two pure modules**, written before the move.
3. **Coverage stays ≥ 80%** — the number is reported before and after, since
   this is the constraint most likely to break.
4. **The 57-page sweep still passes**, now 58 with the platinum page.
5. **Full suite**, with the known GARCH failures and no third.
6. **Nothing was deleted** — the eight foreign files are reported, and their
   disposition is the owner's call.
7. Show the owner the diff. **Never commit.**

---

## Decision the owner must make: `mt5_executor.py` places live orders unattended

This is the reason this plan stops short of wiring anything up.

`src/services/mt5_trade.py` already exists and is emphatic about its own safety
model: trading is reachable only when `TRADE_JOURNAL_ALLOW_TRADING` is set **and**
the caller passes `confirm=True` **and** the UI's two-step Review/Confirm flow
was completed. Its docstring calls itself "a deliberate reversal of a documented
design decision", because `mt5_link` states that "a page has no legitimate reason
to place an order".

`mt5_executor.py` is a different animal: a polling worker that claims signals
from a Postgres queue and sends orders with **no human in the loop** — gated only
by `EXECUTOR_DRY_RUN` and a kill switch. That would be the third order-placing
path in the repo and the first unattended one.

Placing the file is safe; nothing imports it and it only runs if launched
explicitly. **This plan does that and no more.** Whether an unattended executor
should exist at all, and under what gates, is a decision about real money and
belongs to the owner — not something to be inherited by moving a file into a
tidier directory.

## What actually happened

Executed at 1.10.36. All seven files moved; **all seven now import**, which none
of them did before — `mt5_executor` referenced `src.execution.gate`,
`platinum_tab` referenced `src.engine.platinum`, and neither package existed.

| From | To |
|---|---|
| `src/gate.py` | `src/execution/gate.py` |
| `src/queue.py` | `src/execution/queue.py` |
| `src/mt5_executor.py` | `src/execution/mt5_executor.py` |
| `src/schema_execution.sql` | `src/execution/schema.sql` |
| `src/platinum.py` | `src/core/platinum.py` |
| `src/platinum_jobs.py` | `src/data_backbone/platinum_jobs.py` |
| `src/platinum_tab.py` | `src/pages_lib/platinum.py` |

Coverage went **82.64% → 83.18%**, so the floor held and improved rather than
being defended by omissions: `gate.py` reaches 88% and `platinum.py` 63%, both
measured. Only the queue and the executor were omitted, for the same documented
reason as the existing DB and terminal modules. `src/data_backbone/*` and
`src/pages_lib/*` were already omitted, so the collector and the tab needed no
new entry.

**A docstring pointed at a real module by the wrong path.** `platinum.py`
recommended the Yang-Zhang estimator "already in `src/engine/stochastic.py`".
That estimator does exist — in `src/core/stochastic.py`, verified by reading it.
The pointer was corrected rather than deleted.

**Two test failures were mine, not the code's**, and both are the kind that
would have made the suite lie:

1. The "clean signal" baseline was rejected as **synthetic** — round levels with
   an integer R. Every refusal test would then have passed for that single
   reason regardless of the condition it named. Rewritten so each asserts its
   own rejection string, on a measured-looking baseline that genuinely passes.
2. `disconnect_state(3.0) == disconnect_state(-3.0)` was asserted on the
   assumption it was non-directional. It returns `STRETCHED RICH` / `STRETCHED
   CHEAP`: magnitude picks the tier, sign picks the side.

Two further assertions started as `assert out is not None`, which cannot fail
for the reason claimed. Replaced with the module's actual discrimination:
noise as an extra factor gives ΔR² 0.000 / p(F) 0.76, a genuine factor gives
ΔR² 0.620 / p(F) 0.000 and recovers its coefficient to ±0.05.

**Reported wrongly once during verification:** that the new modules were absent
from the coverage report. They were not — the run had been piped through
`tail -7`, so the table was never captured. Measured directly afterwards.

**Left alone, deliberately:** the eight Angular/.NET files, and any wiring of
`mt5_executor`. `pages/nfp_reaction.py` appeared during this work; it is the
owner's, not part of this refactor, and joins the page sweep automatically.

**Known gap:** `platinum.py`'s `rolling_factor_model` and the bootstrap in
`cross_correlation` are untested (the bulk of the 63%). The tab's headline
z-score comes from `rolling_factor_model`, so it is the next thing worth
covering.
