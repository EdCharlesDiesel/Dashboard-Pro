"""No secrets file may be tracked by git.

`.gitignore` named one exact path, `.streamlit/secrets.toml`, so when
`secrets.production.toml` was added on 2026-08-22 it was not covered and
reached the git index carrying DATABASE_URL, two Postgres passwords, three API
keys, a Gmail app password and a bot token. It was caught before any commit -
git history has no undo, and the branch is already pushed to GitHub.

This checks the **index**, not the working tree: a file on disk is harmless, a
file staged for commit is one `git commit` from being permanent.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SECRETS = re.compile(r"secrets.*\.toml$")


def _tracked() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=_REPO,
                         capture_output=True, text=True, timeout=60)
    return out.stdout.splitlines()


def test_no_secrets_file_is_tracked():
    bad = [f for f in _tracked()
           if _SECRETS.search(f) and not f.endswith(".example")]
    assert not bad, (
        f"secrets file(s) tracked by git: {bad} - run "
        f"`git rm --cached <file>` before committing; the values would be "
        f"permanent in history")


def test_the_example_templates_are_still_tracked():
    # The widened ignore pattern must not hide the templates a new machine
    # needs to get started.
    assert any(f.endswith("secrets.toml.example") for f in _tracked())


def test_the_ignore_pattern_covers_a_new_variant():
    """A future secrets.staging.toml must be ignored without editing .gitignore.

    --no-index because git never reports a *tracked* file as ignored, which
    would make this pass for the wrong reason on exactly the file that started
    this.
    """
    probe = ".streamlit/secrets.staging.toml"
    rc = subprocess.run(["git", "check-ignore", "--no-index", "-q", probe],
                        cwd=_REPO, capture_output=True, timeout=60).returncode
    assert rc == 0, f"{probe} would not be ignored - widen .gitignore"
