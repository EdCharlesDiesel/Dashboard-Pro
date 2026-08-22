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


def test_release_runs_only_after_a_successful_deploy():
    """A release advertising a version that never reached Railway is worse
    than a stale one, so the job chains behind `deploy` rather than beside it."""
    needs = _release_job().get("needs")
    needs = [needs] if isinstance(needs, str) else (needs or [])
    assert "deploy" in needs, (
        f"release.needs is {needs!r} - it must include `deploy`, or a release "
        f"can be published for a deploy that failed")


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
