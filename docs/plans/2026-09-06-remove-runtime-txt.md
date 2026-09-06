# Remove runtime.txt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete `runtime.txt` and the constraint it imposed, now that Streamlit
Community Cloud is no longer a deploy target.

**Architecture:** Delete one file, correct the one comment that cites it.

**Tech Stack:** Python 3.14.

**Spec:** The owner, 2026-09-06: *"runtime.txt is dead, remove it"*.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.63**.
- **Only what the file actually constrained changes.** The pandas pin stays —
  see below.

---

## Context

`runtime.txt` contained `3.12` and was read by exactly one consumer: Streamlit
Community Cloud. With that target retired it is dead weight, and worse than
dead: it was the last thing in the repo claiming a Python other than 3.14, and
`requirements.txt` cites it to justify a live constraint —

> *"runtime.txt still pins 3.12 for Streamlit Community Cloud, so every version
> here must also install on 3.12 — they all do."*

Left in place, that comment would keep a retired platform's requirement alive
in the mind of whoever reads it next.

Nothing else references it: no test, no workflow, no Dockerfile.

**What does not change: pandas stays at 2.3.3.** It would be easy to read this
as unblocking pandas 3.x, since the 3.12 constraint is gone. It does not. The
pin exists for a *separate*, still-live reason recorded in the same block —
Streamlit Cloud's container segfaulted on pandas 3.0.2 + numpy 2.4.4, and more
immediately, 2.3.3 is the version the containers run and the version the whole
suite is verified against. Two reasons were bundled in one comment; only one of
them just expired.

---

## Task 1: Delete it and correct the citation

**Files:** Delete `runtime.txt` · Modify `requirements.txt`

- [ ] **Step 1:** `git rm runtime.txt`.
- [ ] **Step 2:** Replace the citing comment with the reason that survives, so
      the pandas pin does not read as unmotivated.
- [ ] **Step 3:** Confirm nothing else references the file.

---

## Verification

1. **No reference to `runtime.txt`** outside `docs/plans/`.
2. **Full suite**, known GARCH failures and no third.
3. **The app still builds and runs** — it never read the file, so this should be
   a no-op, and that is worth confirming rather than assuming.
4. Show the owner the diff. **Never commit.**
