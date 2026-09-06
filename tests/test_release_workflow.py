"""The Production merge must publish a GitHub Release matching ``VERSION``.

The Releases page read *Production V1.0.1* from two months earlier while the app
shipped 1.10.33 across 30 deployments, because nothing published a release —
the workflow ended at ``railway up``.

None of this can be exercised end to end from a test: it only runs on a real
merge to Production. But everything that would silently break it is statically
checkable, and a broken release job is otherwise discovered *during* a
production merge, which is the worst moment to find out.
"""
from __future__ import annotations

import os
import re

import pytest

yaml = pytest.importorskip("yaml", reason="pyyaml needed to parse the workflow")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKFLOW = os.path.join(_REPO, ".github", "workflows", "build.yml")


def _workflow() -> dict:
    with open(_WORKFLOW, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _executable_text(node, _parts=None) -> str:
    """Every string in the job with shell comment lines removed.

    Assertions must read the *logic*, not the prose around it, in both
    directions: a hard-coded version in a comment is documentation and must not
    fail the suite, while a flag that appears only in a comment must not let a
    check pass after the flag itself was deleted. Comments are stripped here,
    on the raw strings — ``yaml.dump`` escapes newlines, so filtering its output
    line by line silently does nothing.
    """
    parts = [] if _parts is None else _parts
    if isinstance(node, dict):
        for value in node.values():
            _executable_text(value, parts)
    elif isinstance(node, list):
        for value in node:
            _executable_text(value, parts)
    elif isinstance(node, str):
        parts.append("\n".join(line for line in node.splitlines()
                               if not line.lstrip().startswith("#")))
    return "\n".join(parts)


def _release_job() -> dict:
    jobs = _workflow()["jobs"]
    assert "release" in jobs, (
        "no `release` job in build.yml - a merge to Production deploys but "
        "never publishes a release, which is how the Releases page came to read "
        "V1.0.1 while VERSION read 1.10.33")
    return jobs["release"]


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


def test_release_can_write_tags():
    # The default GITHUB_TOKEN is read-only for contents, so without this the
    # job fails at `gh release create` with a 403 - on a production merge.
    perms = _release_job().get("permissions") or {}
    assert perms.get("contents") == "write", (
        f"release.permissions.contents is {perms.get('contents')!r}; creating a "
        f"tag needs `contents: write`")


def test_release_is_restricted_to_production():
    guard = str(_release_job().get("if", ""))
    assert "refs/heads/Production" in guard, "release job must be gated on Production"
    assert "pull_request" in guard, (
        "release job must exclude pull_request events, or opening a PR would "
        "publish a release")


def test_the_tag_is_derived_from_the_version_file():
    """A hand-typed version is the bug this whole scheme exists to prevent.

    VERSION is the single source of truth that sync_version.py propagates; the
    release must read it, never carry its own copy.
    """
    job = _release_job()
    body = _executable_text(job)
    assert "VERSION" in body, "the release job never reads the VERSION file"

    # `body` is already comment-stripped. That matters: the job's shell comments
    # legitimately cite past versions (1.10.34 drafted, ProductionV1.0.1 was the
    # stale release), and a literal in a comment is documentation — only one in
    # the logic is the drift this test exists to catch.
    # No leading \b: the tag is built as `ProductionV$VERSION`, so a hard-coded
    # one reads `ProductionV1.10.34` — and `V1` has no word boundary between the
    # letter and the digit. With \b this test matched only the version cited in
    # a comment and sailed straight past the real thing, which a mutation test
    # caught: hard-coding the tag left all seven assertions green.
    literal = re.search(r"\d+\.\d+\.\d+", body)
    assert not literal, (
        f"release job contains a hard-coded version {literal.group(0)!r} - "
        f"derive it from the VERSION file instead")


def test_the_release_is_published_not_drafted():
    """The 1.10.34 run created a DRAFT, and a draft creates no tag.

    So the Releases page showed *Production V1.10.34* while the remote still
    carried only `ProductionV1.0.1` — the precise drift this job exists to end,
    except now behind a green check, which is worse than the stale page it
    replaced.
    """
    body = _executable_text(_release_job())
    assert "--draft=false" in body, (
        "gh release create must pass --draft=false explicitly; a drafted "
        "release publishes no tag")


def test_the_job_verifies_its_own_outcome():
    """`gh release create` can exit 0 having produced nothing usable.

    The job must assert the end state — released, not drafted, tag actually on
    the remote — rather than trusting the exit code.
    """
    body = _executable_text(_release_job())
    # `--json isDraft`, not bare "isDraft": the string appears twice (the --json
    # field list and the --jq filter), so asserting the bare name let a mutation
    # that stripped it from --json leave this test green.
    assert "--json isDraft" in body, (
        "the job never asks the API whether the release ended up a draft")
    assert "ls-remote" in body, (
        "the job never confirms the tag reached the remote")
    assert "exit 1" in body, (
        "the job checks the end state but never fails on it, so a draft would "
        "still report success")


def test_a_rerun_does_not_fail_the_pipeline():
    """Re-running a green workflow, or merging without a version bump, both hit
    an existing tag. `gh release create` fails on that, and neither case is an
    error worth turning the pipeline red for."""
    body = _executable_text(_release_job())
    assert "release view" in body, (
        "the job must check whether the release already exists (gh release "
        "view) and skip, or a re-run fails on the existing tag")


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
