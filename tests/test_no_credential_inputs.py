"""Guard: no page may render database credentials through Streamlit.

Regression test for the leak fixed when the ``◆ POSTGRESQL`` credential blocks
were removed from the checklist sidebar and the Trade Journal. The subtle part:
``st.text_input(..., type="password", value=<real password>)`` only masks the
*rendering* — the prefilled value is still shipped to the browser as widget
state. Together with the plaintext host/port/dbname/user inputs beside it, any
page that rendered those handed a full Neon credential set to whoever could
reach the app. ``src/db/connection.py`` ``auto_connect()`` resolves the target
from ``secrets.toml`` / env and keeps it in server-side session state (which
Streamlit never sends to the client), so nothing needs to render it — ever.

Two rules, enforced by static analysis over the source tree:

1. No ``st.*`` call anywhere passes ``type="password"``.
2. No ``st.*`` call anywhere takes a DB-credential identifier as an argument —
   which covers inputs (``st.text_input(value=st.session_state.db_pass)``) *and*
   plain display (``st.markdown(f"host={db_host}")``), since both reach the
   browser.

Reading ``st.session_state.db_host`` to *use* it (e.g. passing it to a cached
query) stays legal — only handing it to Streamlit is banned. ``db_ok`` / ``db_msg``
are deliberately not credentials: the sidebar's connection badge still renders
them.

Pure ``ast`` — no Streamlit runtime, no DB, no network.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Identifiers that resolve to a piece of the DB connection target. `db_config`
# is here because `secrets.db_config()["password"]` is the same leak by another
# route. `db_ok`/`db_msg` are intentionally absent — they carry no secret.
BANNED_IDENTIFIERS = frozenset({
    "db_config",
    "db_host",
    "db_name",
    "db_pass",
    "db_password",
    "db_port",
    "db_user",
})

# `archive/` is dead experiments the project rule says never to touch, and the
# venv/tests aren't shipped UI.
EXCLUDED_DIRS = frozenset({".venv", "archive", "tests", "__pycache__", ".git"})


def _source_files() -> List[Path]:
    """Every shipped Python file that could render UI."""
    files = [REPO_ROOT / "app.py"]
    for directory in ("pages", "src"):
        files.extend(sorted((REPO_ROOT / directory).rglob("*.py")))
    return [
        f for f in files
        if f.is_file() and not EXCLUDED_DIRS.intersection(f.relative_to(REPO_ROOT).parts)
    ]


def _root_name(node: ast.AST) -> str | None:
    """Leftmost name of an attribute chain: ``st.sidebar.text_input`` -> ``st``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _streamlit_calls(tree: ast.AST) -> Iterator[ast.Call]:
    """Every ``st.…(…)`` call that *renders*, including ``st.sidebar.…``.

    ``st.session_state.get("db_pass", …)`` is excluded: session state is
    server-side, so reading a credential out of it is exactly the legal path the
    app uses. Only handing the value to a rendering call is a leak.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _root_name(node.func) != "st":
            continue
        if ast.unparse(node.func).startswith("st.session_state"):
            continue
        yield node


def _identifiers_in(node: ast.AST) -> Iterator[str]:
    """Names, attribute labels and string keys anywhere beneath ``node``
    (f-strings included — ``ast.walk`` descends into ``FormattedValue``).

    String constants count because ``st.session_state.get("db_host", …)`` is the
    same leak spelled with a subscript key instead of an attribute — that's the
    form the Trade Journal used, and an identifiers-only scan missed it.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            yield child.attr
        elif isinstance(child, ast.Name):
            yield child.id
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child.value


def _credential_aliases(tree: ast.AST) -> set[str]:
    """Local names holding a resolved DB config, e.g. ``_db = secrets.db_config()``.

    Without this, ``_db["host"]`` inside a widget reads as an innocent subscript
    and the whole credential set leaks through one level of indirection.
    """
    factories = {"db_config", "current_db_config"}
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        called = {
            _called_name(child)
            for child in ast.walk(node.value)
            if isinstance(child, ast.Call)
        }
        if not called.intersection(factories):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    aliases.add(name.id)
    return aliases


def _called_name(call: ast.Call) -> str | None:
    """``secrets.db_config()`` -> ``db_config``; ``db_config()`` -> ``db_config``."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _arguments_of(call: ast.Call) -> Iterator[ast.AST]:
    """The call's arguments only — never its ``func``, so ``st.session_state`` as
    the callee chain can't self-trigger."""
    yield from call.args
    for kw in call.keywords:
        yield kw.value


def _scan(path: Path) -> Tuple[List[str], List[str]]:
    """Return (password-typed widgets, credential-carrying calls) as messages."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(REPO_ROOT).as_posix()
    banned = BANNED_IDENTIFIERS | _credential_aliases(tree)
    masked: List[str] = []
    credentials: List[str] = []

    for call in _streamlit_calls(tree):
        widget = ast.unparse(call.func)

        for kw in call.keywords:
            if (
                kw.arg == "type"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "password"
            ):
                masked.append(f"{rel}:{call.lineno}: {widget}(type=\"password\")")

        for argument in _arguments_of(call):
            leaked = sorted(banned.intersection(_identifiers_in(argument)))
            if leaked:
                credentials.append(
                    f"{rel}:{call.lineno}: {widget}(...) receives {', '.join(leaked)}"
                )

    return masked, credentials


@pytest.fixture(scope="module")
def scanned() -> List[Tuple[Path, List[str], List[str]]]:
    return [(path, *_scan(path)) for path in _source_files()]


def test_source_files_were_actually_scanned() -> None:
    """Cheap self-check: a glob that silently matches nothing would make every
    assertion below vacuously true."""
    files = _source_files()
    assert len(files) > 40, f"only found {len(files)} source files — glob is broken"
    names = {f.name for f in files}
    assert "trade-journal.py" in names
    assert "sidebar.py" in names


def test_no_password_inputs(scanned) -> None:
    """No widget may be a password field.

    The app has no password to collect: DB credentials come from
    ``secrets.toml``/env, and the only auth is the reverse proxy's basic-auth
    (see DEPLOY-RAILWAY.md). If real user login is ever added, it goes through an
    identity provider (Streamlit's native OIDC), which never routes a password
    through a widget either — so this rule should hold permanently.
    """
    offenders = [msg for _, masked, _ in scanned for msg in masked]
    assert not offenders, (
        "password input(s) reintroduced — a prefilled type=\"password\" still "
        "sends its value to the browser:\n  " + "\n  ".join(offenders)
    )


def test_no_db_credentials_passed_to_streamlit(scanned) -> None:
    """DB credentials must never be handed to Streamlit, as input or as display."""
    offenders = [msg for _, _, creds in scanned for msg in creds]
    assert not offenders, (
        "DB credential(s) rendered — use auto_connect()/current_db_config() and "
        "keep them in server-side session state:\n  " + "\n  ".join(offenders)
    )


def test_detects_a_reintroduced_credential_widget(tmp_path: Path) -> None:
    """The scanner must actually catch the pattern it's guarding against —
    otherwise the two tests above pass because the check is broken, not because
    the tree is clean. Rebuilds the exact code that was removed."""
    offending = tmp_path / "regression.py"
    offending.write_text(
        "import streamlit as st\n"
        "with st.sidebar:\n"
        "    st.session_state.db_host = st.text_input('Host', value=st.session_state.db_host)\n"
        "    st.session_state.db_pass = st.text_input(\n"
        "        'Password', value=st.session_state.db_pass, type='password',\n"
        "    )\n"
        "    st.markdown(f\"target: {st.session_state.db_user}\")\n",
        encoding="utf-8",
    )
    # _scan reports paths relative to the repo root; point it at tmp_path instead.
    global REPO_ROOT
    original, REPO_ROOT = REPO_ROOT, tmp_path
    try:
        masked, credentials = _scan(offending)
    finally:
        REPO_ROOT = original

    assert len(masked) == 1, masked
    # db_host input, db_pass input, and the f-string markdown display.
    assert len(credentials) == 3, credentials
    assert any("db_user" in msg for msg in credentials), credentials


def test_detects_the_indirect_journal_pattern(tmp_path: Path) -> None:
    """The Trade Journal's spelling must be caught too.

    It leaked through two layers the first version of this scanner missed: a
    string session key (``st.session_state.get("db_host", …)``) instead of an
    attribute, and a local alias (``_db = secrets.db_config()`` then
    ``_db["host"]``) instead of a named credential. Both forms are the real
    removed code, so this case is what keeps the guard honest.
    """
    offending = tmp_path / "journal_style.py"
    offending.write_text(
        "import streamlit as st\n"
        "from src.core import secrets\n"
        "_db = secrets.db_config()\n"
        "_host = st.text_input('Host', value=st.session_state.get('db_host', _db['host']))\n"
        "_user = st.text_input('User', value=_db['user'])\n",
        encoding="utf-8",
    )
    global REPO_ROOT
    original, REPO_ROOT = REPO_ROOT, tmp_path
    try:
        masked, credentials = _scan(offending)
    finally:
        REPO_ROOT = original

    assert masked == []  # no type="password" in this snippet — rule 2 must carry it
    assert len(credentials) == 2, credentials
    # Line 4 leaks via the string session key; line 5 only via the `_db` alias.
    assert any("db_host" in msg for msg in credentials), credentials
    assert any(msg.endswith("receives _db") for msg in credentials), credentials


def test_safe_credential_use_is_not_flagged(tmp_path: Path) -> None:
    """Passing credentials to a non-Streamlit callable, and rendering the
    non-secret ``db_ok``/``db_msg`` gates, must both stay legal — otherwise this
    guard would force the sidebar's connection badge out of the app."""
    safe = tmp_path / "safe.py"
    safe.write_text(
        "import streamlit as st\n"
        "from src.db.cache import cached_realized_pnl\n"
        "pnl = cached_realized_pnl(\n"
        "    st.session_state.db_host, st.session_state.db_port,\n"
        "    st.session_state.db_name, st.session_state.db_user,\n"
        "    st.session_state.db_pass,\n"
        ")\n"
        "st.metric('Realized P&L', pnl)\n"
        "st.markdown(f\"badge: {st.session_state.db_ok} {st.session_state.db_msg}\")\n",
        encoding="utf-8",
    )
    global REPO_ROOT
    original, REPO_ROOT = REPO_ROOT, tmp_path
    try:
        masked, credentials = _scan(safe)
    finally:
        REPO_ROOT = original

    assert masked == []
    assert credentials == []
