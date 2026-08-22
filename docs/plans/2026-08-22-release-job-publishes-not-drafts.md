# The release job must publish, not draft — and prove it did

**Goal:** Stop the release job producing a draft, and make it verify its own
outcome, so the Releases page can never again show a version that has no tag
behind it.

**Architecture:** Two lines of hardening on the job added in
`2026-08-22-release-on-production-merge.md` (1.10.34) — an explicit
`--draft=false`, and a step that asserts the end state — plus repairs to guard
tests that were passing for the wrong reason.

**Tech Stack:** GitHub Actions, `gh` CLI, pytest.

**Spec:** The 1.10.34 merge published a **draft**, observed by the owner and
confirmed against the remote.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.34**, so this plan takes **1.10.35**.
- Branch `DEV-04/release-job-publishes-not-drafts`.
- **Assertions must be mutation-tested.** A guard that cannot fail is worse than
  no guard, because it is believed.

---

## Context

The 1.10.34 merge ran green and the Releases page showed *Production V1.10.34* —
while `git ls-remote --tags origin` still returned only `ProductionV1.0.1`. The
release was a **draft**, and a draft creates no tag.

That is worse than the stale page the job set out to fix. The drift was still
there, now behind a green check, and the job had reported success for an outcome
it never checked.

**The guard tests did not catch it, and could not have.** Mutation testing found
three defects, each making a test pass for the wrong reason:

1. **The version check could never match.** It used `\b\d+\.\d+\.\d+\b`, but the
   tag is built as `ProductionV$VERSION`, so a hard-coded one reads
   `ProductionV1.10.34` — and `V1` has **no word boundary** between letter and
   digit. The assertion only ever matched a version quoted in a *comment*.
   Hard-coding the tag left all seven tests green.
2. **Comment stripping did nothing.** The helper filtered `yaml.dump()` output
   line by line, but `yaml.dump` escapes newlines inside `run:` strings, so no
   line ever started with `#`.
3. **A presence check matched the wrong occurrence.** `"isDraft"` appears twice
   (the `--json` field list and the `--jq` filter), so deleting it from `--json`
   left the test green.

**Two mutation runs were themselves meaningless** — the target string was absent,
so nothing changed and the pass was reported as if it meant something. A quoted
heredoc still collapsed `\\` to `\`, so the intended target never existed.
*"The test passed"* and *"the mutation ran"* are different claims.

---

## Task 1: Publish rather than draft, and verify it

**Files:** Modify `.github/workflows/build.yml`

- [ ] **Step 1:** Pass `--draft=false` explicitly to `gh release create`.
- [ ] **Step 2:** Add a *Confirm it actually published* step asserting `isDraft`
      is false **and** the tag is on the remote, failing loudly otherwise.

---

## Task 2: Repair the guards

**Files:** Modify `tests/test_release_workflow.py`

- [ ] **Step 1:** Drop the leading `\b` from the version pattern.
- [ ] **Step 2:** Strip shell comments from the **raw** strings, and route every
      assertion through that text — in both directions, since a flag present
      only in a comment must not satisfy a presence check either.
- [ ] **Step 3:** Assert `--json isDraft` rather than bare `isDraft`, plus
      `exit 1`, so a check that never fails cannot pass.
- [ ] **Step 4:** Add tests for the two new behaviours.

---

## Verification

1. **Six mutations, each failing exactly one assertion:** hard-coded version;
   `--draft=false` deleted with its comment left behind; `isDraft` stripped from
   `--json`; `needs: deploy` removed; `contents: write` downgraded; every
   `exit 1` neutered.
2. **The mutation runner asserts the mutation applied** before believing any
   result, and restores the workflow **byte-identical** — including CRLF.
3. **Full suite**, with the known GARCH failures and no third.
4. Show the owner the diff. **Never commit.**

## Note for the owner

The existing 1.10.34 draft must be published (or deleted) by hand: the fixed job
finds it via `gh release view` and takes the already-exists branch, so it will
skip rather than replace it.
