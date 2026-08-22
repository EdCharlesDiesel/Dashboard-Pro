# A GitHub Release on every Production merge

**Goal:** Make a merge to `Production` publish a GitHub Release matching
`VERSION`, so the Releases page stops reading `Production V1.0.1` from two
months ago while the app ships 1.10.33.

**Architecture:** One new job in the existing workflow, after the Railway deploy
succeeds. No new tooling: `gh` is pre-installed on GitHub runners, and `VERSION`
is already the single source of truth that `sync_version.py` propagates. The
release is *derived*, never hand-typed — the same rule the version bar already
follows.

**Tech Stack:** GitHub Actions, `gh` CLI.

**Spec:** The owner's request, 2026-08-22: "I need with every production merge
with railway the release needs to be updated as well."

## Global Constraints

- **Never commit.** The owner reviews and commits.
- `VERSION` reads **1.10.33**, so this plan takes **1.10.34**.
- **The release must follow the deploy, not race it.** A release advertising a
  version that failed to reach Railway is worse than a stale one.
- **A re-run must not fail.** Re-running a successful workflow, or merging
  without a version bump, must not turn the pipeline red.
- **No secrets in release notes.** Notes are generated from commit titles.

---

## Context

The repository has exactly **one** tag, `ProductionV1.0.1`, and one release
titled *Production V1.0.1* — while `VERSION` reads 1.10.33 and 30 deployments
have run. Nothing publishes a release today: `.github/workflows/build.yml` ends
at `railway up --detach`.

The naming convention already in use is `ProductionV<version>` for the tag and
`Production V<version>` for the title, so this follows it rather than switching
to a bare `v1.10.33` and leaving two conventions side by side.

**Why it goes after `deploy` and not beside it.** The `deploy` job is gated on
the `production` environment, which can require a reviewer — so a merge does not
necessarily deploy immediately. Chaining the release behind it means the
Releases page reflects what is actually running on Railway.

**The re-run problem.** `gh release create` fails if the tag exists. Two
situations hit that legitimately: re-running a green workflow, and merging a
change that did not bump `VERSION`. Neither is an error worth failing the
pipeline for, so the job checks first and skips with a notice.

---

## Task 1: The release job

**Files:** Modify `.github/workflows/build.yml`

- [ ] **Step 1:** Add a `release` job with `needs: deploy`, the same
      `if: github.ref == 'refs/heads/Production' && github.event_name != 'pull_request'`
      guard, and `permissions: contents: write` (the default token is read-only
      for contents, so the job cannot create a tag without this).
- [ ] **Step 2:** Read the version into the job's output:
      `echo "version=$(cat VERSION)" >> "$GITHUB_OUTPUT"`, and validate it is
      semver — an empty or malformed `VERSION` must not create a junk tag.
- [ ] **Step 3:** Skip cleanly when the release already exists
      (`gh release view "ProductionV$VERSION"`), logging a notice that says the
      version was not bumped.
- [ ] **Step 4:** Otherwise `gh release create "ProductionV$VERSION"
      --title "Production V$VERSION" --generate-notes --target Production`.

---

## Task 2: Guard it

**Files:** Create `tests/test_release_workflow.py`

The workflow cannot be unit-tested end to end, but the things that would
silently break it are all statically checkable — and a broken release job is
only discovered on a production merge, which is the worst time to find out.

- [ ] **Step 1: Failing tests** — the `release` job exists and is gated on
      `needs: deploy`; it declares `contents: write`; it is restricted to the
      `Production` branch and excludes pull requests; and the tag it builds is
      derived from `VERSION` rather than hard-coded (no literal version string
      anywhere in the job).
- [ ] **Step 2: Implement**, then green.

---

## Verification

1. **The workflow parses**, and the job graph reads `build → deploy → release`.
2. **The guard tests pass**, and fail if the `needs: deploy` chain is removed.
3. **The tag that would be produced** is printed from the real `VERSION`
   (`ProductionV1.10.34`) — computed, not typed.
4. **Dry-run the skip path** locally: given an existing tag, the logic takes the
   notice branch rather than the create branch.
5. Show the owner the diff. **Never commit.**

## Notes for the owner

- The first run after this merges will create `ProductionV<version>`; the old
  `ProductionV1.0.1` release is left untouched.
- Railway may deploy twice if its own GitHub integration is still enabled
  alongside this workflow's `railway up` — worth checking in the Railway
  dashboard.
- If you would rather have `v1.10.34`-style tags, that is a one-line change; the
  existing convention was kept deliberately.

## What actually happened

Implemented at 1.10.34. Job graph is now `build → deploy → release`; five guard
tests in `tests/test_release_workflow.py`.

**Task order was inverted deliberately.** The plan listed the job first and the
guard second, which would have meant writing tests against code that already
worked — not a TDD cycle. The tests were written first and **all five failed**
for the right reason (no `release` job), then passed once the job was added.

**The guard was mutation-tested rather than merely run.** Deleting
`needs: deploy` from the workflow made exactly one test fail
(`test_release_runs_only_after_a_successful_deploy`) and the other four pass,
which is what proves the assertion is load-bearing instead of incidentally true.
The file was restored and re-verified green.

**One real defect found in verification.** The first version regex was
`^[0-9]+\.[0-9]+\.[0-9]+$`, which rejects `1.10.34-rc1` — a version
`deploy/sync_version.py` *accepts*, since its `_SEMVER` allows a prerelease
suffix. A prerelease bump would therefore have passed the repo's own validator
and then failed the production pipeline. The job now uses the same grammar;
both were checked against seven inputs and agree on all of them.

The job's shell logic was simulated against the real `VERSION` rather than
assumed: it derives `ProductionV1.10.34` / *Production V1.10.34*, matching the
existing `ProductionV1.0.1` convention; the existing-tag branch prints a notice
and exits 0; and empty, `1.10`, `v1.10.34` and `1.10.34.5` all fail loudly.

`--target "$GITHUB_SHA"` tags the exact merged commit rather than the branch
name, and `fetch-depth: 0` is required for `--generate-notes` to diff against
the previous release.

**Not yet proven:** this cannot run until a real merge to Production. The first
such merge will create `ProductionV<version>`; if it fails, the deploy has
already happened by then — the release job is last on purpose.

## Follow-up: the first real run produced a DRAFT

The 1.10.34 merge ran green and the Releases page showed *Production V1.10.34*,
but the release was a **draft** — and a draft creates no tag, so the remote still
carried only `ProductionV1.0.1`. The drift this plan set out to end was still
there, now behind a green check.

Fixed in `2026-08-22-release-job-publishes-not-drafts.md` (1.10.35), which also
records three guard tests here that were passing for the wrong reason.
