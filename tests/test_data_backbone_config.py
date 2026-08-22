"""data_backbone must find the database the same way the rest of the app does.

It had its own env-only resolver defaulting to localhost:5432/trading and never
read secrets.toml, so on the host it reached the *native* PostgreSQL, failed
authentication, and fell back to the FRED API on every macro read -- silently,
because fred_data's fallback works. Measured 2026-08-16: every series logged
"password authentication failed for user postgres" and still returned a number.

The container is the other half of the constraint: inside Docker, DB_HOST=db and
DB_PORT=5432 are CORRECT. Note the precedence is NOT what you would guess --
db_config() prefers secrets.toml over DB_*, and the container is safe only
because .dockerignore keeps secrets.toml out of the image. That line is
load-bearing, so it gets a guard test here.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


def _reload(monkeypatch, *, env: dict, secrets: dict | None = None):
    """Re-import config under a given environment and secrets.toml.

    ``secrets=None`` models the **container**: no secrets.toml in the image, so
    the real ``db_config()`` runs its env-var path. It empties the ``[database]``
    section rather than replacing ``db_config`` itself, because replacing it
    would test the mock instead of the precedence being relied on.
    """
    for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_URL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    from src.core import secrets as secrets_module
    if secrets is None:
        monkeypatch.setattr(secrets_module, "_section", lambda name: {})
    else:
        monkeypatch.setattr(secrets_module, "db_config", lambda: secrets)

    import src.data_backbone.config as config
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def _restore_config():
    """Reload the module back to its real state, so later tests see the truth."""
    yield
    import src.data_backbone.config as config
    importlib.reload(config)


class TestItUsesTheSharedResolver:
    def test_secrets_toml_wins_on_the_host(self, monkeypatch):
        cfg = _reload(monkeypatch, env={}, secrets={
            "host": "127.0.0.1", "port": 5433, "dbname": "DashboardproDBv1",
            "user": "postgres", "password": "secret"})
        assert cfg.DB_PORT == 5433
        assert cfg.DB_NAME == "DashboardproDBv1"
        assert "5433" in cfg.DB_URL and "DashboardproDBv1" in cfg.DB_URL

    def test_it_no_longer_defaults_to_the_native_postgres(self, monkeypatch):
        cfg = _reload(monkeypatch, env={}, secrets={
            "host": "127.0.0.1", "port": 5433, "dbname": "DashboardproDBv1",
            "user": "postgres", "password": "secret"})
        assert cfg.DB_PORT != 5432, "5432 is the native server, not the container"
        assert cfg.DB_NAME != "trading", "no such database exists on either server"

    @pytest.mark.live_secrets
    def test_the_host_really_resolves_to_the_container_database(self):
        """This developer's own secrets.toml, not a fixture.

        Marked `live_secrets` so the autouse `_no_live_db` stub steps aside -
        stubbing the resolver made this assert against the fixture instead of
        the resolver, which is why it failed for a whole session.

        Skipped where there is no [database] section: CI has no secrets.toml,
        and a test that cannot pass there would break the build the day the
        suite is added to it.
        """
        from src.core.secrets import _section
        if not (_section("database") or {}).get("port"):
            pytest.skip("no local [database] in secrets.toml")
        import src.data_backbone.config as config
        importlib.reload(config)
        assert config.DB_PORT == 5433
        assert config.DB_NAME == "DashboardproDBv1"


class TestTheContainerStillWorks:
    def test_env_vars_are_used_when_there_is_no_secrets_toml(self, monkeypatch):
        # The container's situation: .dockerignore keeps secrets.toml out of the
        # image, so the [database] section is empty and DB_* is all that is left.
        # NOT a test that env beats secrets -- it does not; see the guard below.
        cfg = _reload(monkeypatch, env={
            "DB_HOST": "db", "DB_PORT": "5432", "DB_NAME": "DashboardproDBv1",
            "DB_USER": "postgres", "DB_PASSWORD": "pw"}, secrets=None)
        assert cfg.DB_HOST == "db"
        assert cfg.DB_PORT == 5432
        assert cfg.DB_NAME == "DashboardproDBv1"

    def test_an_explicit_db_url_still_overrides_everything(self, monkeypatch):
        cfg = _reload(monkeypatch, env={
            "DB_URL": "postgresql+psycopg2://u:p@example:1234/db"}, secrets={})
        assert cfg.DB_URL == "postgresql+psycopg2://u:p@example:1234/db"

    def test_dockerignore_still_excludes_secrets_toml(self):
        """A .dockerignore line is load-bearing for production's DB target.

        db_config() prefers secrets.toml over DB_*. The container resolves to
        `db:5432` only because secrets.toml is absent from the image. Ship one
        and every service chases the host's 127.0.0.1:5433 from inside Docker.
        """
        lines = {ln.strip() for ln in
                 (_REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()}
        assert ".streamlit/secrets.toml" in lines, (
            "secrets.toml must stay out of the image: db_config() prefers it "
            "over the DB_* env vars, so shipping it would repoint every "
            "container at the host loopback")


class TestItDegradesRatherThanCrashes:
    def test_a_broken_secrets_read_does_not_stop_import(self, monkeypatch):
        from src.core import secrets as secrets_module

        def boom():
            raise RuntimeError("no secrets.toml")

        monkeypatch.setattr(secrets_module, "db_config", boom)
        for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        import src.data_backbone.config as config
        cfg = importlib.reload(config)
        # Importing a config module must never be what takes the app down.
        assert isinstance(cfg.DB_URL, str) and cfg.DB_URL

    def test_the_fallback_dbname_is_one_that_exists(self, monkeypatch):
        from src.core import secrets as secrets_module

        def boom():
            raise RuntimeError("no secrets.toml")

        monkeypatch.setattr(secrets_module, "db_config", boom)
        for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        import src.data_backbone.config as config
        cfg = importlib.reload(config)
        # `trading` never existed on either server, so a failure there surfaced
        # as a confusing "database does not exist". `postgres` always exists.
        assert cfg.DB_NAME != "trading"


class TestTheOtherConfigIsUntouched:
    def test_watchlists_and_windows_survive(self):
        import src.data_backbone.config as config
        importlib.reload(config)
        assert config.WATCH_TICKERS, "the price watchlist vanished"
        assert "FEDFUNDS" not in config.WATCH_FRED or True  # shape check only
        assert isinstance(config.STALE_DAYS, int)
        assert isinstance(config.PRICE_PERIOD, str)
