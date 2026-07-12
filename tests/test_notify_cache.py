"""Unit tests for src/services/alert_service.NotifyCache — the file-backed
dedupe ledger. Focus on the concurrency-safety guarantees: atomic writes and
merge-not-clobber under concurrent read-modify-write. No SMTP or Streamlit."""
from __future__ import annotations

import json
import threading

import pytest

from src.services.alert_service import NotifyCache


@pytest.fixture
def cache(tmp_path, monkeypatch):
    # NotifyCache anchors its file at os.getcwd(); pin it to a tmp dir.
    monkeypatch.chdir(tmp_path)
    return NotifyCache("test_ns")


class TestBasics:
    def test_roundtrip_save_and_load(self, cache):
        cache.save(["a", "b"])
        assert cache.load() == {"a", "b"}

    def test_load_missing_file_is_empty_set(self, cache):
        assert cache.load() == set()

    def test_reset_removes_the_file(self, cache):
        cache.save(["a"])
        cache.reset()
        assert cache.load() == set()

    def test_filter_new_returns_and_records_unseen(self, cache):
        assert cache.filter_new(["a", "b", "a"]) == ["a", "b"]  # de-duped, ordered
        assert cache.filter_new(["a", "c"]) == ["c"]            # "a" already seen
        assert cache.load() == {"a", "b", "c"}


class TestAtomicWrite:
    def test_write_is_atomic_no_temp_left_behind(self, cache, tmp_path):
        cache.save(["a", "b"])
        # File is valid JSON and no stray temp files linger in the dir.
        with open(cache.path) as fh:
            assert set(json.load(fh)["keys"]) == {"a", "b"}
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".notify_")]
        assert leftovers == []

    def test_load_never_sees_partial_file(self, cache):
        # os.replace swaps the file in whole; a reader sees old or new, never half.
        cache.save(["a"])
        cache.save(["a", "b", "c"])
        assert cache.load() == {"a", "b", "c"}


class TestMergeSemantics:
    def test_add_unions_with_disk_not_overwrites(self, cache):
        cache.save(["a"])
        cache.add(["b", "c"])
        assert cache.load() == {"a", "b", "c"}

    def test_add_returns_merged_set(self, cache):
        cache.save(["a"])
        assert cache.add(["b"]) == {"a", "b"}


class TestDBMirror:
    """The DB layer is monkeypatched (instance methods) — no live Postgres.
    Verifies load() merges local+DB and writes mirror to both sides."""

    def test_load_merges_local_and_db(self, cache, monkeypatch):
        cache.save(["a"])
        monkeypatch.setattr(cache, "_db_read", lambda: {"b", "c"})
        assert cache.load() == {"a", "b", "c"}

    def test_load_falls_back_to_local_when_db_unset(self, cache, monkeypatch):
        cache.save(["a"])
        monkeypatch.setattr(cache, "_db_read", lambda: None)
        assert cache.load() == {"a"}

    def test_save_mirrors_to_db(self, cache, monkeypatch):
        mirrored = {}
        monkeypatch.setattr(cache, "_db_write", lambda keys: mirrored.update(keys=keys))
        cache.save(["a", "b"])
        assert mirrored["keys"] == {"a", "b"}

    def test_filter_new_mirrors_merged_set_to_db(self, cache, monkeypatch):
        mirrored = {}
        monkeypatch.setattr(cache, "_db_write", lambda keys: mirrored.update(keys=keys))
        cache.filter_new(["a", "b"])
        assert mirrored["keys"] == {"a", "b"}

    def test_reset_clears_db_too(self, cache, monkeypatch):
        cache.save(["a"])
        reset_called = []
        monkeypatch.setattr(cache, "_db_reset", lambda: reset_called.append(True))
        cache.reset()
        assert reset_called == [True]
        assert cache.load() == set()

    def test_db_helpers_degrade_gracefully_without_a_configured_db(self, cache):
        # No monkeypatching — _resolve_cfg() has no live DB target in tests, so
        # every DB helper must no-op/return None rather than raise.
        assert cache._db_read() is None
        cache._db_write({"a"})   # must not raise
        cache._db_reset()        # must not raise


class TestConcurrency:
    def test_concurrent_add_loses_no_keys(self, cache):
        """Many threads each add a distinct key; the lock + atomic write must
        preserve every one (a plain load→mutate→overwrite would drop most)."""
        keys = [f"k{i}" for i in range(50)]
        barrier = threading.Barrier(len(keys))

        def worker(k):
            barrier.wait()          # maximize overlap of the read-modify-write
            cache.add([k])

        threads = [threading.Thread(target=worker, args=(k,)) for k in keys]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cache.load() == set(keys)

    def test_shared_lock_across_instances_for_same_path(self, cache):
        # Two NotifyCache objects on the same namespace share one lock, so their
        # writes to the same file are still serialized.
        other = NotifyCache("test_ns")
        assert other._lock is cache._lock
