# Fix the COT merge_asof dtype crash and surprise_tab's unguarded DB connect

Version on creation of this plan: **1.10.39** (VERSION currently reads 1.10.38).

## Context

Two unrelated bugs surfaced by the latest full-page smoke test, both confirmed
by "yes fix both":

1. `pages/cot_signals.py` and `pages/cot_trade_signal_walk_forward_backtest_harness.py`
   each independently call `pd.merge_asof(series[...], price_df, on="date",
   direction="backward")` and both crash with `MergeError: incompatible merge
   keys [0] dtype('<M8[us]') and dtype('<M8[s]'), must be the same type`.
2. `pages/surprise_tab.py` crashes with `psycopg2.OperationalError:
   connection to server at "localhost" (::1), port 5432 failed: fe_sendauth:
   no password supplied` — it connects to Postgres unconditionally instead of
   checking whether a DB is actually configured first, unlike every other
   DB-touching page in the app.

## Root causes (measured, not guessed)

**(1) The dtype mismatch is real and reproducible.** Ran both fetchers
directly against live data:
- `get_instrument_series()` (`src/services/cot_fetcher.py:160`) does
  `df["date"] = pd.to_datetime(df[date_col])` on CFTC's date-only strings →
  pandas 2.x infers **`datetime64[us]`**.
- `_price_series()` (duplicated verbatim in both page files) builds
  `pd.DataFrame({"date": close.index, ...})` from `cached_ohlc`'s
  `DatetimeIndex` (a Postgres round-trip via `market_cache`) → comes back as
  **`datetime64[s]`**.

Confirmed live:
```
series date dtype: datetime64[us]
price_df date dtype: datetime64[s]
```
`pd.merge_asof`'s `on=` column requires identical dtype (unlike plain
`merge`), so any two datetime64 sources at different resolutions break it —
this is a pandas 2.x resolution-inference quirk, not a data problem in either
source.

**(2) The DB-connect crash is a real gap in surprise_tab.py specifically, not
a broken local DB story.** Locally, `src/core/secrets.py::db_config()`
currently resolves to `localhost:5432/postgres` with an **empty password**,
because this machine's `.streamlit/secrets.toml [database]` section holds
Railway's raw exported variable names (`PGHOST`, `PGPASSWORD`, uppercase
`DATABASE_URL`, …) rather than the app's canonical lowercase schema
(`host`/`port`/`user`/`password`/`url`) that `db_config()` parses. Confirmed
live:
```
db_config() -> {'host': 'localhost', 'port': 5432, 'dbname': 'trading',
                 'user': 'postgres', 'password': 'EMPTY'}
```
Production is unaffected — `.github/workflows/build.yml`'s deploy job writes
its own `.streamlit/secrets.toml` on the CI runner with the canonical
`[database] url = "${{ secrets.DATABASE_URL }}"`, sourced from the repo's
`production` GitHub Environment secret, which is not readable by any tool
(GitHub secrets are write-only by design) and isn't needed for this fix.

Every other DB-touching page tolerates this fine because they gate on
`st.session_state["db_ok"]` (set once per session by `auto_connect()`, which
itself checks `if not cfg.password: ... return False` and never attempts a
real connection) — see `pages/trade-journal.py`'s
`if not st.session_state.get("journal_db_ok"): st.info(...); st.stop()`.
`BloombergTheme.apply()` (`T.apply()`, called by every legacy page including
this one) already triggers `auto_connect()` via its connection banner, so
`st.session_state["db_ok"]` is reliably populated by the time
`surprise_tab.py` reaches its own connect call — it just never checks it.
**User decision: fix by adding the same graceful-degradation guard every
other page already uses, not by changing secrets.toml's shape or wiring in
real local DB credentials.**

## Global constraints

- Never commit. Show the diff.
- Every completed task bumps the patch via `python deploy/sync_version.py <next>`.
- Tests first (`test-driven-development`).
- `docker-compose.yml`'s four app-tier services (`app`, `worker`, `scanner`,
  `sweeper`) all bake source into their image at build time — rebuild and
  recreate all four after this change lands, never just edit source and
  assume it's live.

## Starting state (measured)

- `VERSION` → `1.10.38`.
- `pd.merge_asof` call sites: `pages/cot_signals.py:341` and
  `pages/cot_trade_signal_walk_forward_backtest_harness.py:340`, both
  independent copies of the same pattern (no shared helper today).
- `src/services/cot_fetcher.py` has no `datetime64`-resolution helper and no
  dedicated test file (`tests/test_cot_fetcher.py` does not exist).
- `pages/surprise_tab.py:433-437`'s `if __name__ == "__main__":` block calls
  `_sentry_store().engine.connect()` unconditionally, with no `db_ok` check —
  the only DB-touching page in the app missing this guard.
- No existing test exercises either page end-to-end.

---

## Task 1 — normalize datetime64 resolution before the COT/price merge_asof

This task takes **1.10.40**.

**Steps**
- [x] Added `tests/test_cot_fetcher.py` — confirmed `ImportError:
  normalize_datetime64` before implementing.
- [x] Implemented `normalize_datetime64` in `src/services/cot_fetcher.py`.
  Both new tests green (2/2).
- [x] Applied it at both call sites — `series["date"]`/`price_df["date"]`
  normalized via `.assign(...)` immediately before `pd.merge_asof` in both
  `pages/cot_signals.py` and
  `pages/cot_trade_signal_walk_forward_backtest_harness.py`.
- [x] Re-ran the live reproduction script after the fix: `before: [us] [s]` →
  `after: [ns] [ns]` → `merge_asof OK, rows: 52`.
- [x] Full suite: 1939 passed, 3 pre-existing/unrelated failures (2 GARCH
  arch-package tests, 1 plan-version test expected until this task's bump).
- [x] `python deploy/sync_version.py 1.10.40`.

## Task 2 — guard surprise_tab.py's DB connect on `db_ok`

This task takes **1.10.41**.

**Steps**
- [x] Added `tests/test_surprise_tab_db_guard.py` — confirmed it fails first
  with the exact reproduced `psycopg2.OperationalError: ... fe_sendauth: no
  password supplied`.
- [x] Added the guard in `pages/surprise_tab.py`'s `__main__` block, matching
  `trade-journal.py`'s idiom: after `T.apply()`, check
  `st.session_state.get("db_ok")`; if falsy, `st.info(...)` then `st.stop()`
  before ever calling `_sentry_store()`.
- [x] Re-ran `tests/test_surprise_tab_db_guard.py` — green (1/1).
- [x] `python deploy/sync_version.py 1.10.41`.
- [ ] Rebuild + recreate all four app-tier services (`app`, `worker`,
  `scanner`, `sweeper`) from the new image; spot-check both fixes live
  (AppTest per page, preferred over browser automation per this session's own
  experience).
