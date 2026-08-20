"""Guards the .claude/ signal-kit layout.

Markdown and JSON have no import to break, so nothing else would notice a skill
whose frontmatter stopped parsing or an agent renamed by half a word. For files
that carry no executable code this test *is* the red-green cycle: it fails
before the file exists and passes after.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CLAUDE_DIR = Path(__file__).parent.parent / ".claude"

SKILLS = ["scan-shortlist", "confirm-bias", "size-risk",
          "run-checklist", "execute-and-log"]
AGENTS = ["setup-scorer", "risk-auditor", "trade-plan-writer", "trade-reviewer"]
REFERENCE_DOCS = ["1-scoring-criteria.md", "2-mt5-tooling.md",
                  "3-trade-plan-framework.md", "4-execution-handoff.md"]


def _frontmatter(path: Path) -> dict:
    """The frontmatter block as a dict. Hand-parsed: the repo has no YAML
    dependency and the frontmatter is flat `key: value` only."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path.name} has no frontmatter"
    out = {}
    for line in text.split("---\n")[1].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


@pytest.mark.parametrize("name", SKILLS)
def test_every_pipeline_skill_is_loadable(name):
    path = CLAUDE_DIR / "skills" / name / "SKILL.md"
    assert path.is_file(), f"missing {path}"
    fm = _frontmatter(path)
    assert fm.get("name") == name
    assert fm.get("description", "").startswith("Use when"), \
        "a description the model cannot match on is a skill that never fires"


@pytest.mark.parametrize("name", AGENTS)
def test_every_agent_is_loadable(name):
    path = CLAUDE_DIR / "agents" / f"{name}.md"
    assert path.is_file(), f"missing {path}"
    fm = _frontmatter(path)
    assert fm.get("name") == name
    assert fm.get("description")


@pytest.mark.parametrize("doc", REFERENCE_DOCS)
def test_reference_docs_exist(doc):
    assert (CLAUDE_DIR / "reference" / doc).is_file()


def test_the_existing_process_skills_survive():
    # The 12 process skills predate this plan; adding pipeline skills beside
    # them must not disturb them.
    for name in ("writing-plans", "test-driven-development",
                 "verification-before-completion",
                 "experienced-institutional-fx-trade"):
        assert (CLAUDE_DIR / "skills" / name / "SKILL.md").is_file()


def test_settings_json_is_valid_and_does_not_pre_authorise_trading():
    """The four gates exist so a trade is never one careless allow-rule away.

    An `allow` entry for a trade tool would silently remove the confirmation
    step that `mt5_trade` is built around, so this asserts on the whole allow
    list rather than trusting review to catch it.
    """
    settings = json.loads((CLAUDE_DIR / "settings.json").read_text(encoding="utf-8"))
    allow = settings.get("permissions", {}).get("allow", [])
    for tool in ("open_position", "close_position", "modify_position",
                 "place_pending_order", "cancel_pending_order"):
        assert not any(tool in rule for rule in allow), \
            f"{tool} must never be pre-authorised"


def test_audit_hook_is_shell_and_wired():
    hook = CLAUDE_DIR / "hooks" / "PostToolUse.sh"
    assert hook.is_file()
    assert hook.read_text(encoding="utf-8").startswith("#!/")
    settings = json.loads((CLAUDE_DIR / "settings.json").read_text(encoding="utf-8"))
    wired = json.dumps(settings.get("hooks", {}))
    assert "PostToolUse.sh" in wired, "a hook on disk that nothing calls is not a hook"


def test_run_session_command_exists():
    assert (CLAUDE_DIR / "commands" / "run-session.md").is_file()
