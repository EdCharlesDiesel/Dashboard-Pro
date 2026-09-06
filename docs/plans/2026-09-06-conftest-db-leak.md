# Conftest DB Leak Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the fast suite's "no live database" guarantee hold inside the
containers, where it was silently defeated.

**Architecture:** The autouse `_no_live_db` fixture empties the `[database]`
secrets section but not the `DB_*` environment fallback. Remove those vars too.

**Tech Stack:** Python 3.14, pytest.

**Spec:** The owner's request, 2026-09-06: *"fix the conftest DB leak in the
container tests"*.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.64**.
- **Stub at the config, never the function.** The fixture's existing comments
  record why twice; that stays true here.
- **`--runslow` and `live_secrets` stay exempt** — they exercise the real app on
  purpose.

---

## Context

`_no_live_db` empties the `[database]` secrets section and reasons:

> *"no secrets.toml → env vars → localhost:5432/trading with an empty password →
> `_resolve_cfg()` returns None, i.e. unconfigured. **It also models the
> container exactly**, where .dockerignore keeps secrets.toml out of the image."*

That last claim is backwards. `db_config()` falls back to `DB_HOST`, `DB_PORT`,
`DB_NAME`, `DB_USER`, `DB_PASSWORD` and `DATABASE_URL` — and the containers set
them. So the fallback the fixture relies on to end at "unconfigured" ends at a
working connection instead. The one environment the comment claims to model is
the one place the guarantee does not hold.

**Measured 2026-09-06:** 18 tests failed inside the container that pass on the
host, in shapes like `assert {'a', 'b'} == set()` — a `NotifyCache` returning
real rows because its Postgres mirror was live. They read as breakage. They were
a real database leaking into tests that assume none, and they had been
misattributed to environmental causes more than once in the same session.

**Remove the variables rather than blank them.** An empty `DB_HOST` is still a
*set* value to anything reading `os.environ` directly; absent is what a developer
machine actually looks like, which is the state the fixture is trying to
reproduce.

---

## Task 1: Remove the env fallback for the fast suite

**Files:** Modify `tests/conftest.py`

- [ ] **Step 1:** `monkeypatch.delenv` the six variables, `raising=False`.
- [ ] **Step 2:** Run the container suite — the 18 DB-leak failures must go.
- [ ] **Step 3:** Run the host suite — no regression.

---

## Verification

1. **Container failures drop from 33 to 19**, and every remaining one is a
   repo-hygiene test needing a file `.dockerignore` excludes.
2. **Zero DB-leak failures remain**, checked by file, not by count alone.
3. **Host suite unchanged**, known GARCH failures and no third.
4. Show the owner the diff. **Never commit.**

## What actually happened

33 → **19**, and the 19 are exactly the environmental set: `test_version` (10),
`test_no_secrets_tracked` (3), `test_no_credentials_in_compose` (3),
`test_plans_are_recorded`, `test_mt5_watchdog`, `test_data_backbone_config` — all
needing `.env`, `docker-compose.yml`, `.git`, `deploy/` or `docs/plans/`, none of
which belong in a production image.

The plan guard caught this plan being missing before the suite went green, which
is the third time today it has held the line on a version bump.
