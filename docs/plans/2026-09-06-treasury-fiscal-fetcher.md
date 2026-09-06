# Treasury Fiscal Data Fetcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pull US federal debt from the Treasury Fiscal Data API on a schedule
and store it to the cent, so fiscal stress becomes a real time series rather
than a number scraped off an animated clock.

**Architecture:** The same split the platinum work landed on. Pure parsing and
URL construction live in `src/services/fiscal_data.py` — no network, no DB, so
they sit inside the coverage gate and are tested exhaustively. The HTTP call and
the database write live in `src/data_backbone/fiscal_jobs.py`, which
`pyproject.toml` already omits, and register with the existing APScheduler
instance in `worker.py`.

**Tech Stack:** Python 3.14, `requests`, Postgres (`NUMERIC`), APScheduler,
pytest.

**Spec:** The owner's message, 2026-09-06, rejecting the scrape and choosing the
upstream source:

> *"Go upstream instead — which for Dashboard-Pro is the right call: Treasury
> Fiscal Data API — Debt to the Penny, daily, JSON, no key… the clock's
> per-second ticking is linear interpolation, not live data… fine as an eyeball
> dashboard, useless as a time series — you'd be regressing against a straight
> line."*

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.50**; Task 3 takes **1.10.51**.
  It first claimed 1.10.46 and its Task 1 took 1.10.47 — both were **already
  claimed** by `2026-08-24-currency-strength-index-page.md`. `VERSION` read
  1.10.45 at the time, so deriving "the next number" from `VERSION` alone was
  wrong: plans had claimed as far ahead as 1.10.49 without `VERSION` moving.
  The free number is the one after the highest *claim*, not after `VERSION`.
- **No API key.** The Treasury endpoint is unauthenticated; nothing here reads
  `secrets.toml` or adds an env var.
- **Store to the cent.** Values are `NUMERIC`, never `double precision` — see
  Context.
- **No scraping.** usdebtclock.org is not fetched, parsed, or referenced in code.
- **The network is never touched by a default test run.** Live calls are gated
  behind `--runslow`, matching `tests/test_pages_smoke.py`.

---

## Context

**The clock is not a data source.** Every figure on usdebtclock.org is drawn by
JavaScript over transparent GIFs; the per-second ticking is linear interpolation
between releases, and several derived fields (unfunded liabilities, per-citizen
figures) are the site's own projection assumptions rather than published
statistics. Regressing against that is regressing against a straight line.

**The Treasury endpoint is real and open.** Verified live before writing this:

```
GET /services/api/fiscal_service/v2/accounting/od/debt_to_penny
    ?sort=-record_date&page[size]=2
-> 200, keys: data / meta / links
   record_date          2026-09-03
   tot_pub_debt_out_amt "40102964278586.10"   (a STRING)
   debt_held_public_amt "32423635247198.78"
   intragov_hold_amt    "7679329031387.32"
   links.last           page[number]=4193
```

Three facts that shape the design:

1. **Every numeric field arrives as a string.** Coercion is ours to do, and is
   the obvious place for a silent bug.
2. **`float64` cannot hold this to the cent.** `40102964278586.10` needs 16
   significant digits; float64 gives ~15.95. Stored as a double it becomes
   `40102964278586.1015625` — off by 0.0016. It still *formats* to the right
   cents today, which is exactly what makes it dangerous, and it will stop doing
   so as the figure grows. So the existing `fred_series.value double precision`
   is not a home for this, and a new table with `NUMERIC(24, 2)` is.
3. **The feed is paginated** — 4,193 pages at size 2. A first backfill must
   follow `links.next`; the daily job must not.

**FRED is already wired and stays the fallback.** `data_provider.fetch_fred_series()`
works and `fred_series` holds 95,912 rows. `GFDEBTN` is quarterly and
revision-tracked — good for history, too coarse for a daily series. Treasury is
the authoritative daily print; FRED remains the cross-check.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/services/fiscal_data.py` | **new, measured** — URL building, response parsing, `Decimal` coercion. No I/O. |
| `tests/test_fiscal_data.py` | **new** — the parsing contract, including the precision case. |
| `src/data_backbone/fiscal_jobs.py` | **new, omitted** — HTTP with pagination, upsert, scheduler entry point. |
| `src/data_backbone/fiscal_schema.sql` | **new** — `fiscal_series` with `NUMERIC(24, 2)`. |
| `src/data_backbone/worker.py` | **modify** — register the daily job. |

---

## Task 1: The pure parsing layer

**Files:**
- Create: `src/services/fiscal_data.py`
- Test: `tests/test_fiscal_data.py`

**Interfaces:**
- Produces:
  - `DEBT_TO_PENNY_PATH: str`
  - `build_url(path: str, *, page_size: int = 100, page_number: int = 1, sort: str = "-record_date", fields: tuple[str, ...] | None = None) -> str`
  - `parse_rows(payload: dict) -> list[FiscalPoint]`
  - `FiscalPoint` dataclass: `record_date: date`, `series_id: str`, `value: Decimal`
  - `next_page_number(payload: dict) -> int | None`

- [x] **Step 1: Write the failing tests**

```python
def test_amounts_keep_their_cents():
    # float64 gives ~15.95 significant digits; this needs 16. Parsing through
    # float loses 0.0016 - correct to the cent today, wrong as the number grows.
    payload = {"data": [{"record_date": "2026-09-03",
                         "tot_pub_debt_out_amt": "40102964278586.10"}]}
    point = parse_rows(payload)[0]
    assert point.value == Decimal("40102964278586.10")
    assert isinstance(point.value, Decimal)

def test_a_null_amount_is_skipped_not_zeroed():
    # Treasury sends "null" for suppressed rows. Zero is a real debt figure and
    # would silently enter the series as a cliff.
    payload = {"data": [{"record_date": "2026-09-03",
                         "tot_pub_debt_out_amt": "null"}]}
    assert parse_rows(payload) == []

def test_next_page_is_none_on_the_last_page():
    assert next_page_number({"links": {"next": None}}) is None
```

- [x] **Step 2: Run them, watch them fail** — `ModuleNotFoundError`.
- [x] **Step 3: Implement** with `Decimal(str)`, never `float()`.
- [x] **Step 4: Green**, then bump the version and record it.

---

## Task 2: Schema and the collector

**Files:**
- Create: `src/data_backbone/fiscal_schema.sql`, `src/data_backbone/fiscal_jobs.py`
- Modify: `pyproject.toml` (no new omit needed — `src/data_backbone/*` already covers it)

**Interfaces:**
- Consumes: `build_url`, `parse_rows`, `next_page_number`, `FiscalPoint`
- Produces: `collect_debt_to_penny(engine, *, backfill: bool = False) -> int`

- [x] **Step 1:** `fiscal_series (series_id TEXT, record_date DATE, value NUMERIC(24,2), PRIMARY KEY (series_id, record_date))` — `CREATE TABLE IF NOT EXISTS`, matching `schema.sql`'s idempotent style.
- [x] **Step 2: Write the failing test** with a stub transport — no network:
      a two-page response is followed when `backfill=True` and **not** followed
      when `backfill=False`; re-running upserts rather than duplicating.
- [x] **Step 3: Run, watch it fail.**
- [x] **Step 4: Implement** — `ON CONFLICT (series_id, record_date) DO UPDATE`,
      since Treasury revises.
- [x] **Step 5: Green.** Bump, record.

---

## Task 3: Schedule it, and prove it against the live feed once

**Files:**
- Modify: `src/data_backbone/worker.py`
- Test: `tests/test_fiscal_data.py` (a `--runslow` live case)

- [x] **Step 1:** Register `collect_debt_to_penny` on the existing scheduler,
      daily after the US close. Treasury publishes once a day; polling faster
      buys nothing.
- [x] **Step 2:** Add a `@pytest.mark.slow` test that hits the real endpoint and
      asserts a row lands with `value > 0` and a `record_date` within 10 days.
- [x] **Step 3:** Run it once with `--runslow`, then confirm the row in Postgres
      **equals the API string exactly** — the precision claim proved end to end,
      not just in a unit test.

---

## Verification

Evidence before claims.

1. **Unit tests green**, including the `Decimal` precision case and the
   null-not-zero case.
2. **Coverage ≥ 80%** — `fiscal_data.py` is measured and must carry its own
   tests; `fiscal_jobs.py` is omitted by the existing `src/data_backbone/*` rule.
3. **A default `pytest` run makes no network call** — the live case is skipped
   without `--runslow`.
4. **The stored value equals the API string to the cent**, read back from
   Postgres via `psql`, not via Python's float formatting.
5. **Re-running the collector does not duplicate rows** — count before and after.
6. **Full suite**, the known GARCH failures and no third.
7. Show the owner the diff. **Never commit.**

## Out of scope, deliberately

- Any page or chart. This plan produces a series; rendering it is a separate
  plan with its own version.
- BEA and BLS. Named in the spec as future sources; each is its own API with its
  own auth and shape.
- TIC foreign-holdings data — the spec's suggestion for real variance, and the
  natural follow-on once this pipeline is proven.


## Task 1 — done

`src/services/fiscal_data.py` + `tests/test_fiscal_data.py`. **22 tests, 100%
coverage** of the module. TDD followed: tests written first, run, failed with
`ModuleNotFoundError`, then implemented.

**Five mutations, each failing tests** — the guarantees are load-bearing, not
decorative:

| Mutation | Result |
|---|---|
| parse amounts through `float()` | 3 failed |
| treat a missing amount as `0` | 3 failed |
| accept undated rows | 1 failed |
| always report another page | 2 failed |
| attach an `api_key` to the URL | 1 failed |

**Proved against the live feed**, not only fixtures:

```
API string   : 40102964278586.10
parsed exact : 40102964278586.10
EXACT match  : True
via float    : 40102964278586.1   <- what we avoided
```

The float rendering drops a digit outright at repr, which is the visible half of
the problem; the invisible half is that it also stores `...586.1015625`.

**Line endings:** the file was written LF while every other source file here is
CRLF under `core.autocrlf=true`. Normalised, after a mutation run reported
"bytes restored exactly: False" and surfaced it.

## Task 2 — done (both tasks land on 1.10.50)

`src/data_backbone/fiscal_schema.sql` + `fiscal_jobs.py`, with the pagination
loop placed in `fiscal_data.py` rather than the collector — it is pure given an
injected transport, so putting it there keeps it measured. **26 tests, still
100%** on that module.

Verified in Postgres, not in Python:

```
exact to the cent      : true
components sum to total: true
column type            : numeric(24,2)
10 dates x 3 series across 2 write batches -> upserted, not duplicated
```

The sum invariant holding *in the database* is the one that proves the NUMERIC
column end to end: `debt_held_public_amt + intragov_hold_amt =
tot_pub_debt_out_amt` is exact arithmetic on stored values, and would not hold
if anything on the path had passed through a float.

**One real bug, caught by running it rather than reading it.** `ensure_schema`
split the DDL on `;` and handed Postgres the trailing comment block as a
statement — non-empty after `strip()`, but no SQL in it. Postgres rejects that
with "can't execute an empty query", so schema creation failed on a file that is
perfectly valid. Fixed by dropping fragments whose only content is comments.

**Network flakiness during verification.** DNS for the Treasury host failed
mid-task (`getaddrinfo failed`) after having worked minutes earlier — the same
outage that took out `cli.github.com` and the Obsidian MCP server. The injected
transport is exactly what made the storage path provable anyway: the live API
string was replayed through `fetch=`, so schema, upsert, precision and
idempotency were all confirmed without the network.


## Task 3 — done (1.10.51)

`schedule_jobs(sched)` extracted from `worker.main()` and the collector
registered at 23:30 UTC, after the existing 22:00 refresh.

**The extraction was the point, not tidiness.** `main()` ends in
`sched.start()`, which blocks, so job registration could never be asserted — a
collector that is written, imported, passing its own tests and simply *never
scheduled* does nothing at all and looks perfectly healthy. That is the same
failure as a page with no `NavEntry`: the code is fine, the wiring is missing,
and nothing errors. Six tests now hold it, written first and all failing on
`schedule_jobs` not existing.

One of them earns its place beyond registration: `backfill` must be `False` on
the cron entry. `True` would re-walk ~4,000 pages every night — working,
invisible in the logs, and a standing hammering of a free public API for data
that has not changed.

**Live verification, network permitting and it did:**

```
pytest --runslow tests/test_fiscal_data.py   -> 30 passed
collect_debt_to_penny(engine)                -> 30 points written
```

Read back through `psql`, deliberately not through Python — a float in the read
path would hide exactly the defect this design exists to prevent:

```
 debt_held_public_amt | 2026-09-03 | 32423635247198.78
 intragov_hold_amt    | 2026-09-03 |  7679329031387.32
 tot_pub_debt_out_amt | 2026-09-03 | 40102964278586.10
 total rows: 30 across 10 dates
```

30 rows across 10 dates after repeated runs — the upsert holds.

The four live tests are gated behind `--runslow`, so a default run still makes
no network call. They cover what fixtures cannot: the feed changing shape, going
stale while still returning 200, or starting to send amounts as JSON numbers
rather than strings.
