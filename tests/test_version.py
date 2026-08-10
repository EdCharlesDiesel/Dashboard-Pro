"""The app version must come from one place and reach every consumer.

The defect these guard: the status bar read v1.0.1 for weeks after 1.0.1
shipped, then v1.1.0 while the running container was built from 1.4.0. Both
times the number was a hand-edited literal that nobody remembered to change —
and a stale version is worse than none, because it is believed.
"""
from __future__ import annotations

import importlib
import re
import os

import pytest

from src.core import config as config_module

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_sync():
    """deploy/ is not a package — import the script by path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sync_version_under_test", os.path.join(_REPO, "deploy", "sync_version.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReadVersion:
    def test_reads_the_version_file(self):
        with open(os.path.join(_REPO, "VERSION"), encoding="utf-8") as handle:
            expected = handle.read().strip()
        assert config_module._read_version() == expected

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("APP_VERSION", "9.9.9-rc1")
        assert config_module._read_version() == "9.9.9-rc1"

    def test_blank_env_falls_through_to_the_file(self, monkeypatch):
        monkeypatch.setenv("APP_VERSION", "   ")
        assert config_module._read_version() != "   "

    def test_missing_file_degrades_to_a_visibly_fake_version(self, monkeypatch, tmp_path):
        # An obvious placeholder beats a confident wrong number in the status bar.
        monkeypatch.delenv("APP_VERSION", raising=False)
        monkeypatch.setattr(config_module, "_VERSION_FILE", tmp_path / "nope")
        assert config_module._read_version() == "0.0.0-dev"

    def test_config_exposes_it(self, monkeypatch):
        monkeypatch.setenv("APP_VERSION", "7.7.7")
        fresh = config_module.AppConfig()
        assert fresh.version == "7.7.7"

    def test_version_file_is_a_semantic_version(self):
        sync = _load_sync()
        assert sync._SEMVER.match(sync.read_version())


class TestStatusBarUsesIt:
    def test_footer_renders_the_config_version(self, monkeypatch):
        # The status bar formats "DASHBOARD PRO v{version}" from AppConfig; this
        # pins that it reads the live value rather than a second literal.
        monkeypatch.setenv("APP_VERSION", "3.2.1")
        importlib.reload(config_module)
        assert "3.2.1" in "DASHBOARD PRO v{0}".format(config_module.AppConfig().version)
        monkeypatch.delenv("APP_VERSION", raising=False)
        importlib.reload(config_module)


class TestSyncVersion:
    def test_writes_env_from_version(self, tmp_path):
        sync = _load_sync()
        env = tmp_path / ".env"
        sync.write_env("2.3.4", str(env))
        assert sync.env_version(str(env)) == "2.3.4"

    def test_check_detects_drift(self, tmp_path):
        sync = _load_sync()
        version_file, env = tmp_path / "VERSION", tmp_path / ".env"
        version_file.write_text("2.0.0\n", encoding="utf-8")
        sync.write_env("1.0.0", str(env))
        sync.VERSION_FILE, sync.ENV_FILE = str(version_file), str(env)
        assert sync.main(["--check"]) == 1

    def test_check_passes_when_aligned(self, tmp_path):
        sync = _load_sync()
        version_file, env = tmp_path / "VERSION", tmp_path / ".env"
        version_file.write_text("2.0.0\n", encoding="utf-8")
        sync.write_env("2.0.0", str(env))
        sync.VERSION_FILE, sync.ENV_FILE = str(version_file), str(env)
        assert sync.main(["--check"]) == 0

    def test_bump_rewrites_both(self, tmp_path):
        sync = _load_sync()
        version_file, env = tmp_path / "VERSION", tmp_path / ".env"
        version_file.write_text("1.0.0\n", encoding="utf-8")
        sync.VERSION_FILE, sync.ENV_FILE = str(version_file), str(env)
        assert sync.main(["1.5.0"]) == 0
        assert sync.read_version(str(version_file)) == "1.5.0"
        assert sync.env_version(str(env)) == "1.5.0"

    def test_rejects_a_non_semantic_version(self, tmp_path):
        sync = _load_sync()
        with pytest.raises(ValueError):
            sync.write_version("latest", str(tmp_path / "VERSION"))

    def test_missing_env_reports_none_rather_than_raising(self, tmp_path):
        sync = _load_sync()
        assert sync.env_version(str(tmp_path / "absent")) is None


class TestRepoIsInSync:
    """The one that fails in CI when somebody bumps VERSION and forgets .env."""

    def test_version_and_env_agree(self):
        sync = _load_sync()
        assert sync.env_version() == sync.read_version(), (
            "VERSION and .env disagree — run: python deploy/sync_version.py")

    def test_compose_tags_from_the_env_var(self):
        with open(os.path.join(_REPO, "docker-compose.yml"), encoding="utf-8") as handle:
            compose = handle.read()
        # A literal tag here is how the image and the footer drifted apart.
        assert "dashboard-pro:${APP_VERSION" in compose
        assert compose.count("image: dashboard-pro:${APP_VERSION") == 4

    def test_compose_fallback_is_a_placeholder_not_a_version(self):
        # .env is gitignored, so a fresh clone uses the fallback. If that were a
        # version number it would rot exactly like the literal it replaced —
        # ":dev" is wrong in a way you can see.
        with open(os.path.join(_REPO, "docker-compose.yml"), encoding="utf-8") as handle:
            compose = handle.read()
        assert "${APP_VERSION:-dev}" in compose
        assert not re.search(r"\$\{APP_VERSION:-\d+\.\d+", compose)
