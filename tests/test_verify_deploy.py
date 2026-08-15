"""Content-drift guard: does the running image hold the code on disk?

`sync_version.py --check` compares *labels* (VERSION / .env / compose tag) and
passes happily while the running container is a day behind. These cover the
fingerprint that compares contents, and the status codes that keep "could not
check" distinct from "the code is stale".
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy"))

import verify_deploy as vd  # noqa: E402


class TestFingerprint:
    def test_identical_trees_agree(self):
        a = {"src/a.py": b"print(1)", "app.py": b"x = 2"}
        b = {"app.py": b"x = 2", "src/a.py": b"print(1)"}
        assert vd.fingerprint(a) == vd.fingerprint(b)

    def test_changed_content_changes_the_digest(self):
        base = {"src/a.py": b"print(1)"}
        assert vd.fingerprint(base) != vd.fingerprint({"src/a.py": b"print(2)"})

    def test_missing_file_changes_the_digest(self):
        both = {"src/a.py": b"x", "src/b.py": b"y"}
        assert vd.fingerprint(both) != vd.fingerprint({"src/a.py": b"x"})

    def test_added_file_changes_the_digest(self):
        # The real case: the container lacked src/services/mt5_trade_import.py.
        one = {"src/a.py": b"x"}
        assert vd.fingerprint(one) != vd.fingerprint(
            {"src/a.py": b"x", "src/new.py": b""})

    def test_rename_is_not_confused_with_a_content_change(self):
        # Paths are length-prefixed, so concatenation cannot alias: without
        # that, {"ab": "c"} and {"a": "bc"} would hash identically.
        assert vd.fingerprint({"ab.py": b"c"}) != vd.fingerprint({"a.py": b"bc"})

    def test_separator_style_does_not_matter(self):
        assert vd.fingerprint({"src\\a.py": b"x"}) == \
               vd.fingerprint({"src/a.py": b"x"})

    def test_empty_tree_is_stable(self):
        assert vd.fingerprint({}) == vd.fingerprint({})


class TestSourceSelection:
    @pytest.mark.parametrize("rel", [
        "src/a.py", "pages/x.py", "app.py", "src/deep/nested/mod.py"])
    def test_python_sources_count(self, rel):
        assert vd._is_source(rel) is True

    @pytest.mark.parametrize("rel", [
        "src/__pycache__/a.cpython-311.pyc",
        "src/__pycache__/a.py",          # a .py *inside* a cache dir
        "src/a.pyc",
        "src/notes.md",
        "src/.pytest_cache/x.py",
    ])
    def test_caches_and_non_python_are_excluded(self, rel):
        # Byte-compiled caches differ between a Windows host and a Linux image
        # for reasons that have nothing to do with the source, so counting them
        # would make the check cry wolf on every run.
        assert vd._is_source(rel) is False

    def test_windows_separators_are_normalised_when_excluding(self):
        assert vd._is_source("src\\__pycache__\\a.py") is False


class TestParsePs:
    def test_picks_app_containers_and_ignores_the_database(self):
        out = ("dashboard-pro-app-1\tdashboard-pro:1.8.0\n"
               "dashboard-pro-db-1\tpostgres:18-alpine\n"
               "dashboard-pro-sweeper-1\tdashboard-pro:1.8.0\n")
        assert vd.parse_ps(out) == ["dashboard-pro-app-1",
                                    "dashboard-pro-sweeper-1"]

    def test_matches_any_tag_including_a_stale_one(self):
        # The whole point: a container still on an old tag must be *found* so
        # it can be reported as drift, not filtered out as uninteresting.
        out = "old-one\tdashboard-pro:1.7.1\nnew-one\tdashboard-pro:1.8.0\n"
        assert vd.parse_ps(out) == ["new-one", "old-one"]

    def test_no_containers_is_empty_not_an_error(self):
        assert vd.parse_ps("") == []

    def test_malformed_lines_are_ignored(self):
        assert vd.parse_ps("garbage\nonly-one-field\n") == []


class TestStatusCodes:
    def test_codes_are_distinct(self):
        # "Could not check" must not be mistakable for "the code is stale":
        # one is a statement about the checker, the other about the code.
        assert len({vd.IN_SYNC, vd.DRIFT, vd.UNVERIFIABLE}) == 3
        assert vd.IN_SYNC == 0

    def test_every_status_has_a_verdict_line(self):
        for status in (vd.IN_SYNC, vd.DRIFT, vd.UNVERIFIABLE):
            assert vd._VERDICT[status]

    def test_no_containers_is_unverifiable_not_drift(self, monkeypatch):
        monkeypatch.setattr(vd, "read_host_tree", lambda *a, **k: {"app.py": b"x"})
        monkeypatch.setattr(vd, "app_containers", lambda: [])
        status, lines = vd.compare()
        assert status == vd.UNVERIFIABLE
        assert any("no running" in line for line in lines)

    def test_docker_failure_is_unverifiable_not_drift(self, monkeypatch):
        def boom():
            raise RuntimeError("Cannot connect to the Docker daemon")
        monkeypatch.setattr(vd, "read_host_tree", lambda *a, **k: {"app.py": b"x"})
        monkeypatch.setattr(vd, "app_containers", boom)
        status, lines = vd.compare()
        assert status == vd.UNVERIFIABLE
        assert any("Docker daemon" in line for line in lines)

    def test_matching_container_is_in_sync(self, monkeypatch):
        tree = {"app.py": b"x"}
        monkeypatch.setattr(vd, "read_host_tree", lambda *a, **k: dict(tree))
        monkeypatch.setattr(vd, "app_containers", lambda: ["c1"])
        monkeypatch.setattr(vd, "read_container_tree", lambda name: dict(tree))
        status, _ = vd.compare()
        assert status == vd.IN_SYNC

    def test_stale_container_reports_drift_and_names_the_files(self, monkeypatch):
        monkeypatch.setattr(vd, "read_host_tree",
                            lambda *a, **k: {"app.py": b"new",
                                             "src/added.py": b""})
        monkeypatch.setattr(vd, "app_containers", lambda: ["c1"])
        monkeypatch.setattr(vd, "read_container_tree",
                            lambda name: {"app.py": b"old"})
        status, lines = vd.compare()
        assert status == vd.DRIFT
        blob = "\n".join(lines)
        assert "DRIFT" in blob
        assert "src/added.py" in blob     # missing from container
        assert "app.py" in blob           # differs

    def test_one_stale_container_among_healthy_ones_still_reports_drift(
            self, monkeypatch):
        host = {"app.py": b"new"}
        monkeypatch.setattr(vd, "read_host_tree", lambda *a, **k: dict(host))
        monkeypatch.setattr(vd, "app_containers", lambda: ["good", "stale"])
        monkeypatch.setattr(
            vd, "read_container_tree",
            lambda name: dict(host) if name == "good" else {"app.py": b"old"})
        status, _ = vd.compare()
        assert status == vd.DRIFT

    def test_unreadable_container_does_not_masquerade_as_drift(self, monkeypatch):
        def reader(name):
            raise RuntimeError("tar: not found")
        monkeypatch.setattr(vd, "read_host_tree", lambda *a, **k: {"app.py": b"x"})
        monkeypatch.setattr(vd, "app_containers", lambda: ["c1"])
        monkeypatch.setattr(vd, "read_container_tree", reader)
        status, lines = vd.compare()
        assert status == vd.UNVERIFIABLE
        assert any("UNREADABLE" in line for line in lines)


class TestQuietMode:
    def test_quiet_stays_silent_when_in_sync(self, monkeypatch, capsys):
        monkeypatch.setattr(vd, "compare", lambda: (vd.IN_SYNC, ["noise"]))
        assert vd.main(["--quiet"]) == 0
        assert capsys.readouterr().out == ""

    def test_quiet_still_explains_a_failure(self, monkeypatch, capsys):
        # An unexplained non-zero exit is one people learn to ignore.
        monkeypatch.setattr(vd, "compare", lambda: (vd.DRIFT, ["app.py differs"]))
        assert vd.main(["--quiet"]) == 1
        out = capsys.readouterr().out
        assert "app.py differs" in out
        assert "OUT OF SYNC" in out

    def test_unverifiable_exits_two(self, monkeypatch, capsys):
        monkeypatch.setattr(vd, "compare", lambda: (vd.UNVERIFIABLE, ["docker down"]))
        assert vd.main(["--quiet"]) == 2
        assert "COULD NOT VERIFY" in capsys.readouterr().out
