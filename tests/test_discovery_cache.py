from __future__ import annotations

import json
import os
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import amc_pipeline.discovery as disc
from amc_pipeline.config import PipelineConfig
from amc_pipeline.discovery import discover_audio_files, discovery_counts


@contextmanager
def patch_attrs(**overrides):
    # Timeout/poll/stale are module constants read at import, so tests that need
    # to shrink them (to avoid multi-hour waits) patch the globals directly.
    saved = {k: getattr(disc, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(disc, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(disc, k, v)


@contextmanager
def env(**overrides: str | None):
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build_tree(root: Path) -> None:
    calls = {
        "2022/callA": ("audio.wav", {"agent": "x", "nested": {"k": 1}}),
        "2022/callB": ("audio.wav", None),
        "2022/callC": ("recording.mp3", {"agent": "y"}),
        "2023/callD": ("audio.flac", {"agent": "z"}),
        "2023/callE": ("audio.opus", None),
        "2023/callF": ("audio.m4a", {"agent": "w"}),
    }
    for i, (rel_dir, (fname, sidecar)) in enumerate(calls.items()):
        d = root / rel_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_bytes(b"audio-bytes-" + bytes([i]) * (i + 1))
        if sidecar is not None:
            (d / "metadata.json").write_text(json.dumps(sidecar))


def records_by_id(recs):
    return {r.file_id: r for r in recs}


class DiscoveryCacheEquivalenceTest(unittest.TestCase):
    def _config(self, input_root: Path, output_root: Path, hash_mode: str, num_shards: int, shard_index):
        cfg = PipelineConfig(input_root=input_root, output_root=output_root)
        cfg.discovery.hash_mode = hash_mode
        cfg.discovery.num_shards = num_shards
        cfg.discovery.shard_index = shard_index
        return cfg

    def test_cache_records_match_walk_across_modes_and_shards(self):
        for hash_mode in ("path", "fast", "path_size_mtime", "metadata", "content"):
            for num_shards, shard_index in [(1, None), (4, 0), (4, 1), (4, 2), (4, 3)]:
                with self.subTest(hash_mode=hash_mode, num_shards=num_shards, shard_index=shard_index):
                    with TemporaryDirectory() as tmp:
                        tmp_path = Path(tmp)
                        root = tmp_path / "input"
                        build_tree(root)
                        out = tmp_path / "out"
                        cache = tmp_path / "cache"

                        # Walk path (cache explicitly disabled).
                        with env(AMC_DISCOVERY_CACHE="0", AMC_DISCOVERY_CACHE_DIR=None):
                            walk_recs = discover_audio_files(
                                self._config(root, out, hash_mode, num_shards, shard_index)
                            )
                            walk_counts = discovery_counts(
                                self._config(root, out, hash_mode, num_shards, shard_index)
                            )

                        # Cache path (explicit shared cache dir).
                        with env(AMC_DISCOVERY_CACHE="auto", AMC_DISCOVERY_CACHE_DIR=str(cache)):
                            cache_recs = discover_audio_files(
                                self._config(root, out, hash_mode, num_shards, shard_index)
                            )
                            cache_counts = discovery_counts(
                                self._config(root, out, hash_mode, num_shards, shard_index)
                            )
                            # Second read hits the already-built manifest.
                            cache_recs_2 = discover_audio_files(
                                self._config(root, out, hash_mode, num_shards, shard_index)
                            )

                        self.assertEqual(records_by_id(walk_recs), records_by_id(cache_recs))
                        self.assertEqual(records_by_id(walk_recs), records_by_id(cache_recs_2))
                        self.assertEqual(walk_counts, cache_counts)

    def test_full_shard_partition_is_complete_and_disjoint(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "input"
            build_tree(root)
            out = tmp_path / "out"
            cache = tmp_path / "cache"
            num_shards = 4

            with env(AMC_DISCOVERY_CACHE="0", AMC_DISCOVERY_CACHE_DIR=None):
                all_recs = discover_audio_files(self._config(root, out, "path", 1, None))
            all_ids = set(records_by_id(all_recs))

            union: set[str] = set()
            with env(AMC_DISCOVERY_CACHE="auto", AMC_DISCOVERY_CACHE_DIR=str(cache)):
                for shard_index in range(num_shards):
                    shard_recs = discover_audio_files(
                        self._config(root, out, "path", num_shards, shard_index)
                    )
                    shard_ids = set(records_by_id(shard_recs))
                    self.assertTrue(union.isdisjoint(shard_ids))
                    union |= shard_ids
            self.assertEqual(union, all_ids)

    def test_auto_cache_dir_only_for_sharded_layout(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "input"
            build_tree(root)
            run_root = tmp_path / "run"
            sharded_out = run_root / "outputs" / "shard-0"
            sharded_out.mkdir(parents=True, exist_ok=True)

            # Auto-enabled for RUN_ROOT/outputs/shard-N: a manifest appears under
            # RUN_ROOT/discovery-cache without any explicit env.
            with env(AMC_DISCOVERY_CACHE="auto", AMC_DISCOVERY_CACHE_DIR=None):
                discover_audio_files(self._config(root, sharded_out, "path", 4, 0))
            self.assertTrue((run_root / "discovery-cache").exists())

            # Non-sharded layout: no implicit cache directory is created.
            plain_out = tmp_path / "plain_output"
            plain_out.mkdir(parents=True, exist_ok=True)
            with env(AMC_DISCOVERY_CACHE="auto", AMC_DISCOVERY_CACHE_DIR=None):
                discover_audio_files(self._config(root, plain_out, "path", 4, 0))
            self.assertFalse((plain_out / "discovery-cache").exists())
            self.assertFalse((plain_out.parent / "discovery-cache").exists())

    def _only_key_dir(self, cache: Path) -> Path:
        # The real key dir depends on root.resolve() + the configured extension
        # set, so locate it dynamically rather than recomputing the hash here.
        dirs = [p for p in cache.iterdir() if p.is_dir()]
        self.assertEqual(len(dirs), 1, f"expected one cache key dir, got {dirs}")
        return dirs[0]

    def test_stale_building_lock_is_reclaimed(self):
        # A builder that died leaves a .building lock with a frozen (old) mtime.
        # The next worker must detect it as stale, reclaim it, rebuild the
        # manifest, and return correct records -- not block.
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "input"
            build_tree(root)
            out = tmp_path / "out"
            cache = tmp_path / "cache"

            # Build once so the real key dir exists, then simulate a crashed
            # builder: drop .done and plant a stale lock with an ancient mtime.
            with env(AMC_DISCOVERY_CACHE="auto", AMC_DISCOVERY_CACHE_DIR=str(cache)):
                discover_audio_files(self._config(root, out, "path", 4, 0))
            key_dir = self._only_key_dir(cache)
            (key_dir / ".done").unlink()
            lock = key_dir / ".building"
            lock.write_text("99999")  # pid of a dead builder
            old = time.time() - 10_000  # far older than default stale (120s)
            os.utime(lock, (old, old))

            with env(AMC_DISCOVERY_CACHE="auto", AMC_DISCOVERY_CACHE_DIR=str(cache)):
                recs_after = discover_audio_files(self._config(root, out, "path", 4, 0))
            with env(AMC_DISCOVERY_CACHE="0", AMC_DISCOVERY_CACHE_DIR=None):
                walk_recs = discover_audio_files(self._config(root, out, "path", 4, 0))

            self.assertEqual(records_by_id(walk_recs), records_by_id(recs_after))
            self.assertTrue((key_dir / ".done").exists())

    def test_fresh_building_lock_is_not_stolen_and_falls_back(self):
        # A lock that is being heartbeated (fresh mtime) must NOT be stolen.
        # With a short build timeout the waiter gives up and falls back to the
        # live walk rather than corrupting another builder's manifest.
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "input"
            build_tree(root)
            out = tmp_path / "out"
            cache = tmp_path / "cache"

            with env(AMC_DISCOVERY_CACHE="auto", AMC_DISCOVERY_CACHE_DIR=str(cache)):
                discover_audio_files(self._config(root, out, "path", 4, 0))
            key_dir = self._only_key_dir(cache)
            (key_dir / ".done").unlink()
            lock = key_dir / ".building"
            lock.write_text("12345")  # a "live" builder holds the lock (fresh mtime)

            with env(AMC_DISCOVERY_CACHE="auto", AMC_DISCOVERY_CACHE_DIR=str(cache)), patch_attrs(
                _CACHE_BUILD_TIMEOUT_SEC=1.0, _CACHE_POLL_SEC=0.2
            ):
                recs = discover_audio_files(self._config(root, out, "path", 4, 0))
            with env(AMC_DISCOVERY_CACHE="0", AMC_DISCOVERY_CACHE_DIR=None):
                walk_recs = discover_audio_files(self._config(root, out, "path", 4, 0))

            # Fell back to the live walk (correct records) and left the lock alone.
            self.assertEqual(records_by_id(walk_recs), records_by_id(recs))
            self.assertTrue(lock.exists())
            self.assertFalse((key_dir / ".done").exists())

    def test_heartbeat_only_refreshes_lock_while_progress_advances(self):
        # The anti-wedge guarantee: the build lock's mtime is refreshed ONLY when
        # the walk's `seen` counter moves. A frozen counter (wedged builder) must
        # let the lock age so another box can reclaim it.
        import threading

        with TemporaryDirectory() as tmp:
            lock = Path(tmp) / ".building"
            lock.write_text("123")
            ancient = time.time() - 1000
            os.utime(lock, (ancient, ancient))
            progress: dict = {"seen": 0}
            stop = threading.Event()
            with patch_attrs(_CACHE_HEARTBEAT_SEC=0.05):
                t = threading.Thread(
                    target=disc._heartbeat_lock, args=(lock, stop, progress), daemon=True
                )
                t.start()
                try:
                    time.sleep(0.3)  # several beats, but no progress
                    self.assertLess(lock.stat().st_mtime, time.time() - 100)  # still ancient
                    progress["seen"] = 10  # walk advances
                    time.sleep(0.3)
                    self.assertGreater(lock.stat().st_mtime, time.time() - 100)  # refreshed
                finally:
                    stop.set()
                    t.join(timeout=1)

    def test_build_publishes_progress_json(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "input"
            build_tree(root)
            out = tmp_path / "out"
            cache = tmp_path / "cache"
            with env(AMC_DISCOVERY_CACHE="auto", AMC_DISCOVERY_CACHE_DIR=str(cache)), patch_attrs(
                _CACHE_PROGRESS_EVERY=1
            ):
                discover_audio_files(self._config(root, out, "path", 4, 0))
            key_dir = self._only_key_dir(cache)
            self.assertTrue((key_dir / ".done").exists())
            prog = json.loads((key_dir / "progress.json").read_text())
            self.assertTrue(prog["done"])
            self.assertGreaterEqual(prog["seen"], 1)

    def test_meta_mismatch_falls_back_to_walk(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "input"
            build_tree(root)
            out = tmp_path / "out"
            cache = tmp_path / "cache"

            with env(AMC_DISCOVERY_CACHE="auto", AMC_DISCOVERY_CACHE_DIR=str(cache)):
                recs = discover_audio_files(self._config(root, out, "path", 4, 0))
                self.assertTrue(recs)
                # Corrupt the meta so it no longer matches; discovery must still
                # return correct records by falling back to the live walk.
                key_dir = next(p for p in cache.iterdir() if p.is_dir())
                (key_dir / "meta.json").write_text(json.dumps({"input_root": "/nope"}))
                recs_after = discover_audio_files(self._config(root, out, "path", 4, 0))

            with env(AMC_DISCOVERY_CACHE="0", AMC_DISCOVERY_CACHE_DIR=None):
                walk_recs = discover_audio_files(self._config(root, out, "path", 4, 0))

            self.assertEqual(records_by_id(walk_recs), records_by_id(recs_after))


if __name__ == "__main__":
    unittest.main()
