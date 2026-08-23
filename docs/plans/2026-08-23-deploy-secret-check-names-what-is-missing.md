# The deploy guard must name what is missing

**Goal:** Make the deploy's secret check report every secret, so a failed run
says whether one value is missing or the environment is not wired at all — two
faults with opposite fixes, currently indistinguishable.

**Architecture:** One step in `.github/workflows/build.yml`. Presence booleans
only; no value and no length is ever printed.

**Tech Stack:** GitHub Actions, Python.

**Spec:** The owner's request, 2026-08-23: "fix this please", on a deploy failing
`DATABASE_URL secret is empty - refusing to deploy` for the second time.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.44**, so this plan takes **1.10.45**.
- **No secret value, or length, may ever be printed.** Presence only.
- **The check must still fail closed.** A missing required secret refuses the
  deploy exactly as before.

---

## Context

The check printed two booleans and exited on the first failure:

```
DATABASE_URL secret is empty - refusing to deploy
database.url present: False
telegram configured : False
```

That is not enough to act on. "DATABASE_URL is empty" reads as one missing
secret, and the obvious response is to go and re-add that one value — but the
same message appears when *no* secret reaches the job at all, where re-adding a
value fixes nothing because the environment itself is not connected.

The evidence for which was already on screen and easy to miss: `DATABASE_URL`
and `TELEGRAM_BOT_TOKEN` are unrelated, and **both** were empty. Two independent
secrets failing together is a wiring fault, not two coincidences. The check
should say that itself rather than leaving it to be inferred.

---

## Task 1: Report every secret, and name the fault

**Files:** Modify `.github/workflows/build.yml`

- [ ] **Step 1:** Check all nine, print a present/required table.
- [ ] **Step 2:** If **zero** are present, fail with the environment-wiring
      message — name, and the deployment-branch rule.
- [ ] **Step 3:** If some are present but a required one is missing, name it.
- [ ] **Step 4:** Presence booleans only.

---

## Verification

1. **Three simulated states**, run against the real step body extracted from the
   workflow: all-empty, one-missing, all-present.
2. **All-empty** exits non-zero with the environment message; **one-missing**
   names the secret; **all-present** exits 0.
3. **The workflow parses** and the release guards stay green.
4. Show the owner the diff. **Never commit.**

## What actually happened

Implemented and verified by extracting the step body straight from the YAML and
running it against three constructed configs, rather than trusting it by reading:

```
ALL EMPTY (environment not wired)   exit=1  0/9 present -> environment message
ONE MISSING (DATABASE_URL only)     exit=1  8/9 present -> names DATABASE_URL
ALL PRESENT                         exit=0  9/9 present
```

This does not fix the owner's failing deploy — the secrets still are not
reaching the job. It makes the next run say which of the two faults it is.
