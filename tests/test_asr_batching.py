import unittest
from types import SimpleNamespace

from amc_pipeline.config import ASRModelConfig
from amc_pipeline.models import ASRResult
from amc_pipeline.transcription import (
    _DEFAULT_BATCH_LIMITS,
    DataParallelAdapter,
    _dynamic_batches,
    _prefetch_batches,
    _resolve_batch_limits,
)


def _seg(segment_id: str, duration_sec: float):
    return SimpleNamespace(segment_id=segment_id, duration_sec=duration_sec)


class _EchoChild:
    progress_enabled = True

    def __init__(self, name="granite"):
        self.name = name
        self.seen = []

    def preflight(self):
        return None

    def transcribe_batch(self, segments):
        self.seen = [s.segment_id for s in segments]
        return [ASRResult(s.segment_id, self.name, f"t::{s.segment_id}", 0.9, "en") for s in segments]

    def close(self):
        return None


class DynamicBatchingTests(unittest.TestCase):
    def test_partition_preserves_order_and_membership(self):
        segments = [_seg(f"s{i}", 1.0) for i in range(10)]
        batches = list(_dynamic_batches(segments, audio_sec_budget=None, max_count=3))
        flattened = [s for batch in batches for s in batch]
        self.assertEqual([s.segment_id for s in flattened], [s.segment_id for s in segments])
        self.assertEqual([len(b) for b in batches], [3, 3, 3, 1])

    def test_count_cap_closes_batch(self):
        segments = [_seg(f"s{i}", 0.1) for i in range(5)]
        batches = list(_dynamic_batches(segments, audio_sec_budget=1000.0, max_count=2))
        self.assertEqual([len(b) for b in batches], [2, 2, 1])

    def test_audio_budget_closes_batch(self):
        # 6s budget; segments of 4s -> one per batch (4+4 > 6)
        segments = [_seg(f"s{i}", 4.0) for i in range(3)]
        batches = list(_dynamic_batches(segments, audio_sec_budget=6.0, max_count=None))
        self.assertEqual([len(b) for b in batches], [1, 1, 1])

    def test_budget_packs_multiple_short_segments(self):
        # 10s budget; segments of 2s -> 5 per batch
        segments = [_seg(f"s{i}", 2.0) for i in range(12)]
        batches = list(_dynamic_batches(segments, audio_sec_budget=10.0, max_count=None))
        self.assertEqual([len(b) for b in batches], [5, 5, 2])

    def test_oversized_segment_still_forms_its_own_batch(self):
        segments = [_seg("big", 99.0), _seg("a", 1.0), _seg("b", 1.0)]
        batches = list(_dynamic_batches(segments, audio_sec_budget=5.0, max_count=None))
        # big (99) exceeds budget alone -> its own batch; a+b fit (2 <= 5)
        self.assertEqual([[s.segment_id for s in b] for b in batches], [["big"], ["a", "b"]])

    def test_empty_input_yields_nothing(self):
        self.assertEqual(list(_dynamic_batches([], audio_sec_budget=10.0, max_count=4)), [])

    def test_count_and_budget_both_apply_whichever_first(self):
        segments = [_seg(f"s{i}", 2.0) for i in range(6)]
        # budget 10s would allow 5, but count cap 3 hits first
        batches = list(_dynamic_batches(segments, audio_sec_budget=10.0, max_count=3))
        self.assertEqual([len(b) for b in batches], [3, 3])


class ResolveBatchLimitsTests(unittest.TestCase):
    def test_defaults_used_when_unset(self):
        cfg = ASRModelConfig(batch_size=0)
        cap, budget = _resolve_batch_limits(cfg, "cohere")
        self.assertEqual((cap, budget), _DEFAULT_BATCH_LIMITS["cohere"])

    def test_batch_size_is_count_cap_fallback(self):
        cfg = ASRModelConfig(batch_size=6)
        cap, budget = _resolve_batch_limits(cfg, "granite")
        self.assertEqual(cap, 6)
        self.assertEqual(budget, _DEFAULT_BATCH_LIMITS["granite"][1])

    def test_max_batch_size_overrides_batch_size(self):
        cfg = ASRModelConfig(batch_size=2, max_batch_size=16)
        cap, _ = _resolve_batch_limits(cfg, "qwen")
        self.assertEqual(cap, 16)

    def test_explicit_audio_budget_overrides_default(self):
        cfg = ASRModelConfig(batch_audio_sec_budget=42.0)
        _, budget = _resolve_batch_limits(cfg, "cohere")
        self.assertEqual(budget, 42.0)


class BatchingInvarianceTests(unittest.TestCase):
    """The transcript of a segment must not depend on how segments are batched.

    Mirrors the adapter loop: partition duration-sorted segments with
    _dynamic_batches, transcribe each batch with a deterministic per-segment
    function, and confirm the per-segment results match batch-of-1 exactly.
    """

    @staticmethod
    def _transcribe(batch):
        # Deterministic per-segment "transcript"; independent of batch composition.
        return {s.segment_id: f"text::{s.segment_id}::{s.duration_sec}" for s in batch}

    def _run(self, segments, budget, cap):
        ordered = sorted(segments, key=lambda s: (s.duration_sec, s.segment_id))
        out = {}
        for batch in _dynamic_batches(ordered, budget, cap):
            out.update(self._transcribe(batch))
        return out

    def test_results_identical_across_batching_configs(self):
        segments = [_seg(f"s{i}", (i % 7) + 0.5) for i in range(40)]
        baseline = self._run(segments, budget=None, cap=1)
        for budget, cap in [(None, 4), (10.0, None), (10.0, 4), (100.0, 8), (3.0, 2)]:
            self.assertEqual(self._run(segments, budget, cap), baseline)


class PrefetchTests(unittest.TestCase):
    def test_prefetch_preserves_order_and_loads_each_chunk(self):
        chunks = [[_seg(f"c{c}s{i}", 1.0) for i in range(2)] for c in range(5)]
        loaded = []

        def loader(chunk):
            return [s.segment_id for s in chunk]

        produced = list(_prefetch_batches(chunks, loader, enabled=True, depth=2))
        self.assertEqual(len(produced), len(chunks))
        for original, (chunk, payload, error) in zip(chunks, produced):
            self.assertIsNone(error)
            self.assertEqual(chunk, original)
            self.assertEqual(payload, [s.segment_id for s in original])
            loaded.append(payload)
        self.assertEqual(loaded, [[s.segment_id for s in c] for c in chunks])

    def test_prefetch_disabled_matches_enabled_ordering(self):
        chunks = [[_seg(f"c{c}", 1.0)] for c in range(4)]
        loader = lambda chunk: [s.segment_id for s in chunk]
        enabled = [(c, p) for c, p, _ in _prefetch_batches(chunks, loader, enabled=True)]
        disabled = [(c, p) for c, p, _ in _prefetch_batches(chunks, loader, enabled=False)]
        self.assertEqual(enabled, disabled)

    def test_per_chunk_load_error_is_surfaced_not_raised(self):
        chunks = [[_seg("ok", 1.0)], [_seg("boom", 1.0)], [_seg("ok2", 1.0)]]

        def loader(chunk):
            if chunk[0].segment_id == "boom":
                raise ValueError("load failed")
            return [s.segment_id for s in chunk]

        produced = list(_prefetch_batches(chunks, loader, enabled=True, depth=2))
        self.assertEqual(len(produced), 3)
        self.assertIsNone(produced[0][2])
        self.assertIsInstance(produced[1][2], ValueError)
        self.assertIsNone(produced[1][1])
        self.assertIsNone(produced[2][2])


class DataParallelAdapterTests(unittest.TestCase):
    def test_shards_disjointly_and_merges_in_input_order(self):
        children = [_EchoChild(), _EchoChild()]
        adapter = DataParallelAdapter("granite", children)
        segments = [_seg(f"s{i}", 1.0) for i in range(7)]

        results = adapter.transcribe_batch(segments)

        self.assertEqual([r.segment_id for r in results], [s.segment_id for s in segments])
        self.assertEqual([r.transcript for r in results], [f"t::{s.segment_id}" for s in segments])
        # Disjoint round-robin partition covering every segment exactly once.
        union = set(children[0].seen) | set(children[1].seen)
        self.assertEqual(union, {s.segment_id for s in segments})
        self.assertFalse(set(children[0].seen) & set(children[1].seen))

    def test_single_child_delegates_directly(self):
        child = _EchoChild()
        adapter = DataParallelAdapter("granite", [child])
        segments = [_seg("a", 1.0), _seg("b", 1.0)]
        results = adapter.transcribe_batch(segments)
        self.assertEqual([r.segment_id for r in results], ["a", "b"])

    def test_child_exception_is_propagated(self):
        class _Boom(_EchoChild):
            def transcribe_batch(self, segments):
                raise RuntimeError("replica failed")

        adapter = DataParallelAdapter("granite", [_EchoChild(), _Boom()])
        with self.assertRaises(RuntimeError):
            adapter.transcribe_batch([_seg(f"s{i}", 1.0) for i in range(4)])


if __name__ == "__main__":
    unittest.main()
