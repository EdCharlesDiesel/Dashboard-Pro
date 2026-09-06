# Every Production Merge Releases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee that every merge into `Production` publishes a GitHub Release
matching `VERSION` — by stopping an unrelated Railway deploy failure from
silently skipping the release, and by failing a pull request that does not bump
`VERSION` in the first place.

**Architecture:** Two independent changes to `.github/workflows/build.yml`. The
`release` job stops chaining behind `deploy` and instead runs whenever `build`
succeeded, recording the deploy's actual result in the release notes so it can
never misrepresent itself. A new `version-bump` job fails a pull request whose
`VERSION` is not greater than the newest `ProductionV*` tag.

**Tech Stack:** GitHub Actions, `gh` CLI, bash, pytest.

**Spec:** The owner's instruction, 2026-09-06: *"Make sure everytime I merge to
production the version on github also increases same as the one you're treating
versioning in the system"*, plus their two decisions recorded under Context.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.68**.
- **`VERSION` stays the single source of truth.** No version may be typed into
  the workflow; the tag is always derived from the file.
- **A release must never claim more than it knows.** Decoupling from `deploy` is
  only acceptable because the notes state the deploy outcome explicitly.
- **`tests/test_release_workflow.py` is updated, not weakened.** One assertion
  there is deliberately reversed; it must be replaced with one that encodes the
  new rule and its reason, never deleted.
- **This plan cannot fix the deploy itself.** The Railway secrets live in the
  owner's GitHub environment — see "Notes the owner must act on".

---

## Context

The Releases page shows **one** release, `ProductionV1.0.1`, from 2026-06-14,
while `VERSION` reads 1.10.67. Measured 2026-09-06 against the GitHub API:

| | |
|---|---|
| Releases | 1 — `ProductionV1.0.1` |
| Tags | 2 — `ProductionV1.0.1`, `ProductionV1.10.35` |
| Production pushes since 2026-08-25 | **every one failed** |

`ProductionV1.10.35` is a tag with no release: the drift this job was built to
end, now in the opposite direction.

### The release job is correct and is already deployed

`origin/Production`'s `build.yml` contains the `release` job, identical to the
local copy — it derives the tag from `VERSION`, passes `--draft=false`, and
verifies its own outcome. Nothing about it is broken.

**It simply never runs.** From the failing run's job list:

```
Build (Python 3.14)            -> success
Deploy to Railway (production) -> failure
    ok    Write runtime secrets
    FAIL  Sanity-check the written config
    skipped  Install Railway CLI
    skipped  Deploy
Publish GitHub Release         -> skipped
```

The chain is `build → deploy → release`, so a deploy that cannot read its
secrets takes the release down with it. That is why three months of versions are
missing.

### The two decisions this plan implements

The owner was asked and chose, 2026-09-06:

1. **Release anyway, note the deploy status.** A merge that passes tests gets a
   release even when the deploy fails, and the notes say so.
2. **Fail a Production PR that does not increase `VERSION`.** Today the release
   step exits 0 with a notice when the tag already exists, so a merge without a
   bump is silently release-less.

Decision 1 **reverses a documented design choice.**
`test_release_runs_only_after_a_successful_deploy` currently asserts
`"deploy" in needs`, reasoning that *"a release advertising a version that never
reached Railway is worse than a stale Releases page."* That argument is
answered by the evidence above: the page is not stale, it is three months wrong,
and a tag already exists with no release. Recording the deploy result in the
notes keeps the honesty the original rule was protecting while removing the
single point of failure.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `.github/workflows/build.yml` | CI/CD pipeline | `release` job's `needs`/`if` + notes banner; new `version-bump` job |
| `tests/test_release_workflow.py` | guards the release job's shape | replace one test, add three |

---

## Task 1: A failed deploy must not skip the release

**Files:**
- Modify: `.github/workflows/build.yml` (`release` job, lines ~168–246)
- Test: `tests/test_release_workflow.py`

**Interfaces:**
- Produces: the `release` job now depends on `[build, deploy]` with an
  `always()` guard, and exposes the deploy result to its notes step.

- [ ] **Step 1: Replace the coupling test**

`tests/test_release_workflow.py` already provides everything needed — use these
exact helpers and add none:

- `_workflow() -> dict` — the parsed `build.yml`.
- `_release_job() -> dict` — the `release` job, with a useful assertion message
  if it is missing.
- `_executable_text(node) -> str` — every string under a node with shell comment
  lines removed. **Use this for any "does the body contain X" assertion.** Its
  docstring records why it strips comments on the raw strings rather than on
  `yaml.dump` output: `yaml.dump` escapes newlines, so line-by-line filtering of
  dumped YAML silently does nothing — a guard in this repo already passed for
  that reason.

Delete `test_release_runs_only_after_a_successful_deploy` and put this in its
place — same position in the file, so the diff reads as a replacement:

```python
def test_release_survives_a_failed_deploy():
    """A deploy failure must not erase a version from the Releases page.

    This reverses the original rule, deliberately. The release job used to
    chain behind `deploy`, on the argument that a release advertising a
    version that never reached Railway is worse than a stale Releases page.

    Measured 2026-09-06, that trade went the wrong way: every Production push
    since 2026-08-25 failed at the deploy's secrets sanity-check, so the
    Releases page sat three months behind at V1.0.1 while VERSION read
    1.10.67 - and ProductionV1.10.35 existed as a tag with no release. A page
    that is three months wrong is not 'stale', it is misleading.

    The honesty the old rule protected is kept by
    `test_the_notes_record_the_deploy_outcome` below: the release still says
    whether it reached Railway.
    """
    job = _release_job()
    needs = job.get("needs", [])
    assert "build" in needs, "release must still require a green build"
    guard = str(job.get("if", ""))
    assert "always()" in guard, (
        "release must run even when an earlier job failed, or a deploy "
        "problem silently skips it - see this test's docstring")
    assert "needs.build.result" in guard, (
        "always() alone would release on a RED build; the guard must still "
        "require build to have succeeded")


def test_the_notes_record_the_deploy_outcome():
    """Decoupling is only safe if the release cannot misrepresent itself."""
    body = _executable_text(_release_job())
    assert "needs.deploy.result" in body, (
        "the release notes must state whether the Railway deploy succeeded")
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_release_workflow.py -v --no-cov`
Expected: `test_release_survives_a_failed_deploy` FAILS on the `always()`
assertion, and `test_the_notes_record_the_deploy_outcome` FAILS because the
body has no `needs.deploy.result`.

- [ ] **Step 3: Rewire the job**

In `.github/workflows/build.yml`, replace the `release` job's `needs`/`if`
header:

```yaml
  release:
    name: Publish GitHub Release
    # Runs on a green build, whether or not the deploy succeeded.
    #
    # This job used to sit behind `deploy`, so that a release always meant
    # "this reached Railway". The cost of that guarantee was measured on
    # 2026-09-06: every Production push since 2026-08-25 failed at the
    # deploy's secrets sanity-check, so the Releases page stayed at V1.0.1
    # for three months while VERSION reached 1.10.67, and ProductionV1.10.35
    # ended up a tag with no release.
    #
    # `always()` is required because a skipped/failed `deploy` would
    # otherwise skip this job too; `needs.build.result == 'success'` is what
    # stops it releasing a red build, which `always()` alone would allow.
    needs: [build, deploy]
    if: |
      always()
      && needs.build.result == 'success'
      && github.ref == 'refs/heads/Production'
      && github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    permissions:
      contents: write
```

- [ ] **Step 4: Record the deploy outcome in the notes**

Add a step immediately after "Publish the release", so the banner is applied to
the release that was just created:

```yaml
      - name: Record the deploy outcome in the notes
        if: steps.publish.outcome == 'success'
        env:
          GH_TOKEN: ${{ github.token }}
          TAG: ${{ steps.v.outputs.tag }}
          DEPLOY_RESULT: ${{ needs.deploy.result }}
        # Two calls rather than one: `gh release create --generate-notes`
        # builds the changelog, and this reads it back and prepends a banner.
        # Passing --notes alongside --generate-notes has ambiguous precedence,
        # and a release whose notes silently lost the changelog would be worse
        # than the extra API call.
        run: |
          if [ "$DEPLOY_RESULT" = "success" ]; then
            BANNER="> Deployed to Railway."
          else
            BANNER="> **This version was NOT deployed to Railway** (deploy job: \`$DEPLOY_RESULT\`).
          > The code is tagged and released; the running deployment may be older."
          fi
          BODY="$(gh release view "$TAG" --json body --jq '.body')"
          printf '%s\n\n%s\n' "$BANNER" "$BODY" > /tmp/notes.md
          gh release edit "$TAG" --notes-file /tmp/notes.md
          echo "::notice::Release $TAG notes record deploy result: $DEPLOY_RESULT"
```

Give the existing "Publish the release" step `id: publish` so the `if` above can
read its outcome. It must keep its early `exit 0` when the tag already exists —
that path leaves `outcome == 'success'` and simply re-applies the banner, which
is harmless and correct on a re-run.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_release_workflow.py -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Validate the YAML actually parses**

A workflow that does not parse fails at GitHub, not locally, and the `if:` block
above uses a multi-line scalar that is easy to get wrong.

```bash
python -c "
import yaml
d = yaml.safe_load(open('.github/workflows/build.yml'))
rel = d['jobs']['release']
print('needs :', rel['needs'])
print('if    :', ' '.join(str(rel['if']).split()))
print('steps :', [s.get('name') for s in rel['steps']])
"
```

Expected: `needs` is `['build', 'deploy']`, the `if` contains `always()` and
`needs.build.result == 'success'`, and the step list includes both
"Publish the release" and "Record the deploy outcome in the notes".

---

## Task 2: A Production PR must increase VERSION

Without this, "every merge increases the version" is unenforced: the release
step exits 0 with a notice when the tag already exists, so a merge that forgot
the bump produces no release and no error.

**Files:**
- Modify: `.github/workflows/build.yml` (new job)
- Test: `tests/test_release_workflow.py`

**Interfaces:**
- Produces: a `version-bump` job, running on pull requests targeting
  `Production`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_production_pr_must_bump_the_version():
    """A merge that forgets the bump produces no release and no error.

    The release step exits 0 with a notice when the tag already exists, which
    is right for a re-run and wrong for a forgotten bump - the two are
    indistinguishable after the fact. This check moves the failure to the pull
    request, where it can still be fixed.
    """
    jobs = _workflow()["jobs"]
    assert "version-bump" in jobs, (
        "no job enforces that VERSION increases on a Production PR")
    guard = str(jobs["version-bump"].get("if", ""))
    assert "pull_request" in guard, (
        "the check must run on pull requests, not only after the merge - "
        "after the merge it is too late to fix")
    assert "Production" in guard, (
        "the check must be scoped to PRs targeting Production")


def test_the_bump_check_compares_against_the_published_tags():
    """Comparing against anything else re-introduces a hand-maintained copy."""
    body = _executable_text(_workflow()["jobs"]["version-bump"])
    assert "ProductionV" in body, (
        "the check must compare VERSION against the existing ProductionV tags")
    assert "VERSION" in body, "the check never reads the VERSION file"
```

Both assertions go through `_executable_text`, so a flag or name that survives
only in a comment cannot make them pass — the failure mode its docstring warns
about.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_release_workflow.py -k bump -v --no-cov`
Expected: FAIL with `no job enforces that VERSION increases`.

- [ ] **Step 3: Add the job**

Add after the `build` job:

```yaml
  version-bump:
    name: VERSION must increase
    # Pull requests into Production only. Merging without a bump is not an
    # error the pipeline can report afterwards: the release step cannot tell a
    # forgotten bump from a legitimate re-run, so both exit 0 quietly. Failing
    # here is the only point where it is still fixable.
    if: github.event_name == 'pull_request' && github.base_ref == 'Production'
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          # Tags are the comparison basis, so the shallow default will not do.
          fetch-depth: 0

      - name: Compare VERSION against the newest ProductionV tag
        run: |
          VERSION="$(tr -d '[:space:]' < VERSION)"
          LATEST="$(git tag -l 'ProductionV*' \
                    | sed 's/^ProductionV//' \
                    | sort -V | tail -1)"
          echo "VERSION file : $VERSION"
          echo "newest tag   : ${LATEST:-<none>}"

          if [ -z "$LATEST" ]; then
            echo "::notice::No ProductionV tag yet - nothing to compare against."
            exit 0
          fi

          python - "$VERSION" "$LATEST" <<'PY'
          import sys

          def parts(v):
              # Prerelease suffixes sort before their release, matching the
              # grammar deploy/sync_version.py accepts.
              core = v.split("-")[0].split("+")[0]
              return tuple(int(x) for x in core.split("."))

          new, old = sys.argv[1], sys.argv[2]
          if parts(new) > parts(old):
              print(f"ok: {new} > {old}")
              sys.exit(0)
          sys.exit(
              f"::error::VERSION is {new}, which does not increase on the "
              f"newest released tag {old}. Bump it with "
              f"`python deploy/sync_version.py <new version>` before merging, "
              f"or this merge publishes no release.")
          PY
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_release_workflow.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Prove the comparison logic itself**

The bash/YAML wrapper cannot be unit-tested, but the comparison can. Add:

```python
import subprocess
import sys


def _compare(new: str, old: str) -> int:
    """Run the same comparison the workflow runs."""
    script = (
        "import sys\n"
        "def parts(v):\n"
        "    core = v.split('-')[0].split('+')[0]\n"
        "    return tuple(int(x) for x in core.split('.'))\n"
        "new, old = sys.argv[1], sys.argv[2]\n"
        "sys.exit(0 if parts(new) > parts(old) else 1)\n"
    )
    return subprocess.run([sys.executable, "-c", script, new, old]).returncode


def test_the_version_comparison_is_numeric_not_lexical():
    """'1.10.67' vs '1.9.0' is the case a string compare gets wrong."""
    assert _compare("1.10.67", "1.9.0") == 0
    assert _compare("1.9.0", "1.10.67") == 1


def test_an_equal_version_is_rejected():
    assert _compare("1.10.67", "1.10.67") == 1
```

`1.10.67 > 1.9.0` is the case that matters: a lexical comparison calls
`"1.10.67" < "1.9.0"` and would reject every bump past `.9`.

- [ ] **Step 6: Run the tests and validate the YAML**

Run: `python -m pytest tests/test_release_workflow.py -v --no-cov`, then the
`yaml.safe_load` check from Task 1 Step 6, extended to assert `version-bump` is
present and its `if` names `pull_request` and `Production`.

---

## Verification

Evidence before claims.

1. **The full guard suite passes:**
   `python -m pytest tests/test_release_workflow.py -v --no-cov`.
2. **The workflow parses**, and `release.needs == ['build', 'deploy']` with an
   `always()` guard — Task 1 Step 6's output.
3. **The reversed test is replaced, not deleted** — `git diff` on
   `tests/test_release_workflow.py` shows
   `test_release_runs_only_after_a_successful_deploy` removed and
   `test_release_survives_a_failed_deploy` added in its place.
4. **The version comparison is numeric** —
   `test_the_version_comparison_is_numeric_not_lexical`.
5. **Full suite:** `python -m pytest -q --no-cov` — the 2 known GARCH failures
   in `tests/test_quant_models.py`, no third. Compare the *set*, never the count.
6. **Version:** `python deploy/sync_version.py 1.10.68`, then `--check`.
   `sync_version.py` rewrites `.env`, which holds live credentials — confirm all
   10 keys survive and the file is still CRLF, reporting key **names** only.
7. Show the owner the diff. **Never commit.**

## Notes the owner must act on

- **This plan does not fix the deploy.** The `Sanity-check the written config`
  step fails because the `production` environment's secrets are not reaching the
  job. Open the failing run's log — that step prints a present/absent table for
  all nine secrets and never prints a value. Two different faults produce
  different output there, and they need opposite fixes:
  - *every* secret absent → the environment is not reaching the job at all
    (check Settings → Environments → `production`: the name must match
    `environment: production` exactly, and its deployment-branch rule must allow
    `Production`);
  - one named secret absent → that single secret needs setting.
  Until that is fixed, releases will publish with the "**NOT deployed to
  Railway**" banner — correctly, because that is what is happening.
- **`ProductionV1.10.35` is an orphan tag** with no release. Once this lands,
  either publish a release for it or delete the tag, so tags and releases agree.
- **`VERSION` on `origin/Production` is 1.10.65**, while local is ahead. The
  first merge after this plan lands will be the first to exercise both new
  checks — expect the `version-bump` job to pass and the release to publish
  `ProductionV<local VERSION>`.
