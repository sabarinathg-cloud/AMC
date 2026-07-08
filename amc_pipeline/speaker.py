"""Speaker embedding + global clustering for leakage-safe train/test splits.

This module powers two per-shard pipeline stages plus one run-once global step:

  * ``speaker_embed`` (per shard, GPU): embed each segment's mono 16 kHz WAV with
    ReDimNet, pool one robust centroid per ``(call_id, channel)`` speaker "side",
    and write ``<embed_root>/shard-<N>.npz`` to shared storage.
  * ``cluster-speakers`` (run once, one box): gather every shard's side centroids,
    calibrate a similarity threshold from guaranteed negatives (the two channels of
    the same call are different speakers), build a strict mutual-kNN graph, and run a
    constrained union-find that never merges opposite channels of the same call and
    guards every merge by prototype similarity, compactness, and a hard size cap. The
    result is a global ``clusters.parquet`` mapping ``(call_id, channel) ->
    speaker_cluster_id``.
  * ``speaker_assign`` (per shard): join the global cluster id back onto every segment
    of that shard and regenerate ``all_segments.parquet`` with a ``speaker_cluster_id``
    column.

The approach is ported from the proven prototype in
``training_2026_full_optimized_complete(1)(1).py`` (the "STRICT RECLUSTER" variant),
which was written specifically to avoid the over-merging that leaks speakers across
splits. Grouping the final dataset by ``speaker_cluster_id`` guarantees no speaker
appears in both the train and test sets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .models import SegmentRecord
from .progress import iter_progress


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SpeakerParams:
    """Tunables for embedding + clustering.

    Defaults are the proven values from the prototype's strict variant. They can be
    overridden via the pipeline config file (``speaker_*`` keys) or the
    ``cluster-speakers`` CLI flags.
    """

    # Embedding.
    target_sample_rate: int = 16000
    min_segment_sec: float = 0.80
    max_segment_sec: float = 3.0
    embed_batch_size: int = 48
    embed_load_workers: int = 8
    # Cap segments embedded per (call, channel) side. A robust centroid needs only a
    # handful of clean segments; capping keeps the 21M-segment embed cost bounded.
    # 0 = embed every segment of the side.
    max_segments_per_side: int = 30
    # ReDimNet checkpoint pulled from torch.hub ("IDRnD/ReDimNet").
    model_name: str = "M"
    train_type: str = "ft_mix"
    dataset: str = "vb2+vox2+cnc"

    # Clustering.
    edge_sim_threshold: float = 0.92  # manual floor; auto-calibration may raise it
    component_sim_margin: float = 0.015  # component thr = max(0.90, edge - margin)
    cluster_p10_sim_min: float = 0.72
    max_call_sides_per_cluster: int = 800
    knn_k: int = 50
    knn_batch_size: int = 4096
    compactness_sample_size: int = 1000

    @classmethod
    def from_config(cls, cfg: Any) -> "SpeakerParams":
        return cls(
            target_sample_rate=int(getattr(cfg, "audio").target_sample_rate),
            min_segment_sec=float(getattr(cfg, "speaker_min_segment_sec", cls.min_segment_sec)),
            max_segment_sec=float(getattr(cfg, "speaker_max_segment_sec", cls.max_segment_sec)),
            embed_batch_size=int(getattr(cfg, "speaker_embed_batch_size", cls.embed_batch_size)),
            embed_load_workers=int(getattr(cfg, "speaker_embed_load_workers", cls.embed_load_workers)),
            max_segments_per_side=int(getattr(cfg, "speaker_embed_max_segments", cls.max_segments_per_side)),
            model_name=str(getattr(cfg, "speaker_model_name", cls.model_name)),
            train_type=str(getattr(cfg, "speaker_train_type", cls.train_type)),
            dataset=str(getattr(cfg, "speaker_dataset", cls.dataset)),
            edge_sim_threshold=float(getattr(cfg, "speaker_edge_sim_threshold", cls.edge_sim_threshold)),
            component_sim_margin=float(getattr(cfg, "speaker_component_sim_margin", cls.component_sim_margin)),
            cluster_p10_sim_min=float(getattr(cfg, "speaker_cluster_p10_sim_min", cls.cluster_p10_sim_min)),
            max_call_sides_per_cluster=int(getattr(cfg, "speaker_max_call_sides", cls.max_call_sides_per_cluster)),
            knn_k=int(getattr(cfg, "speaker_knn_k", cls.knn_k)),
        )


# ---------------------------------------------------------------------------
# Linear-algebra helpers (ported verbatim from the prototype)
# ---------------------------------------------------------------------------


def _l2_normalize_matrix(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _l2_normalize_vector(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


def _weighted_centroid(E: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    E = _l2_normalize_matrix(E.astype("float32"))
    if weights is None:
        c = E.mean(axis=0)
    else:
        w = np.asarray(weights, dtype="float32")
        w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
        w = np.clip(w, 1e-3, None)
        w = w / (w.sum() + 1e-12)
        c = (E * w[:, None]).sum(axis=0)
    return _l2_normalize_vector(c).astype("float32")


def _robust_side_centroid(E: np.ndarray, weights: np.ndarray | None = None):
    """One trimmed, weighted centroid per side.

    Returns ``(centroid, mean_sim, min_sim, n_used)``. Drops the lowest-similarity
    ~20% of segments before recomputing so a few noisy/cross-talk clips do not pull
    the speaker profile off its true center.
    """
    E = _l2_normalize_matrix(E.astype("float32"))
    if weights is None:
        weights = np.ones(len(E), dtype="float32")
    else:
        weights = np.asarray(weights, dtype="float32")
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        weights = np.clip(weights, 1e-3, None)

    if len(E) == 1:
        c = E[0]
        sims = E @ c
        return c.astype("float32"), float(sims.mean()), float(sims.min()), 1

    c1 = _weighted_centroid(E, weights)
    sims1 = E @ c1
    if len(E) >= 5:
        keep = sims1 >= np.quantile(sims1, 0.20)
    elif len(E) >= 3:
        keep = sims1 >= np.quantile(sims1, 0.10)
    else:
        keep = np.ones(len(E), dtype=bool)

    E2 = E[keep]
    w2 = weights[keep]
    c2 = _weighted_centroid(E2, w2)
    sims2 = E @ c2
    return c2.astype("float32"), float(np.mean(sims2)), float(np.min(sims2)), int(len(E2))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class SpeakerEmbedder:
    """ReDimNet speaker encoder loaded once from torch.hub, reused across batches."""

    def __init__(self, params: SpeakerParams):
        import torch  # local import: heavy, only needed for the embed stage

        self._torch = torch
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.model = torch.hub.load(
            "IDRnD/ReDimNet",
            "ReDimNet",
            model_name=params.model_name,
            train_type=params.train_type,
            dataset=params.dataset,
            trust_repo=True,
        ).to(self.device)
        self.model.eval()

    def load_wav(self, path: Path):
        """Load a segment WAV as a 1-D float32 tensor windowed to [min, max] seconds."""
        import torchaudio

        torch = self._torch
        wav, sr = torchaudio.load(str(path))
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.params.target_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.params.target_sample_rate)
        wav = wav.squeeze(0).float()
        wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        max_len = int(self.params.max_segment_sec * self.params.target_sample_rate)
        min_len = int(self.params.min_segment_sec * self.params.target_sample_rate)
        n = wav.numel()
        if n == 0:
            wav = torch.zeros(min_len, dtype=torch.float32)
            n = min_len
        if n > max_len:
            start = max(0, (n - max_len) // 2)
            wav = wav[start : start + max_len]
        if wav.numel() < min_len:
            wav = torch.nn.functional.pad(wav, (0, min_len - wav.numel()))
        return wav

    def _forward(self, wavs: list) -> np.ndarray:
        torch = self._torch
        max_len = max(w.numel() for w in wavs)
        batch = torch.zeros((len(wavs), max_len), dtype=torch.float32)
        for i, w in enumerate(wavs):
            batch[i, : w.numel()] = w
        batch = batch.to(self.device, non_blocking=True)
        precision = torch.float16 if torch.cuda.is_available() else torch.float32
        with torch.inference_mode():
            with torch.autocast(device_type=self.device_type, dtype=precision, enabled=torch.cuda.is_available()):
                emb = self.model(batch)
        emb = emb.detach().cpu().float().numpy()
        if emb.ndim == 3 and emb.shape[1] == 1:
            emb = emb[:, 0, :]
        elif emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1)
        return _l2_normalize_matrix(emb.astype("float32"))

    def embed(self, wavs: list) -> np.ndarray:
        """Embed a list of waveform tensors, halving the batch on CUDA OOM."""
        if not wavs:
            return np.zeros((0, 0), dtype="float32")
        try:
            return self._forward(wavs)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(wavs) == 1:
                raise
            if self.device_type == "cuda":
                self._torch.cuda.empty_cache()
            mid = len(wavs) // 2
            return np.vstack([self.embed(wavs[:mid]), self.embed(wavs[mid:])])


def _select_side_segments(segments: list[SegmentRecord], cap: int) -> list[SegmentRecord]:
    """Keep the longest ``cap`` segments of a side (longer clips carry more speaker info)."""
    if cap <= 0 or len(segments) <= cap:
        return segments
    return sorted(segments, key=lambda s: float(getattr(s, "duration_sec", 0.0) or 0.0), reverse=True)[:cap]


def embed_call_channels(
    segments: Iterable[SegmentRecord],
    params: SpeakerParams,
    *,
    progress_enabled: bool = True,
) -> dict[str, np.ndarray | list]:
    """Compute one robust centroid per ``(call_id, channel)`` for a shard's segments.

    Returns a dict of parallel arrays: ``centroids`` (float32 [S, dim]), ``call_id``
    (str), ``channel`` (str), ``n_segments`` (int), ``n_used`` (int), ``mean_sim``,
    ``min_sim`` (float32).
    """
    sides: dict[tuple[str, str], list[SegmentRecord]] = {}
    for seg in segments:
        key = (str(seg.call_id), str(int(seg.channel)))
        sides.setdefault(key, []).append(seg)

    embedder = SpeakerEmbedder(params)
    batch = max(1, params.embed_batch_size)

    centroids: list[np.ndarray] = []
    call_ids: list[str] = []
    channels: list[str] = []
    n_segments: list[int] = []
    n_used: list[int] = []
    mean_sims: list[float] = []
    min_sims: list[float] = []

    items = list(sides.items())
    for (call_id, channel), segs in iter_progress(
        items, desc="Speaker embed", total=len(items), unit="side", enabled=progress_enabled
    ):
        chosen = _select_side_segments(segs, params.max_segments_per_side)
        wavs = []
        weights = []
        for seg in chosen:
            try:
                wavs.append(embedder.load_wav(Path(seg.segment_audio_path)))
                dur = float(getattr(seg, "duration_sec", 0.0) or 0.0)
                weights.append(float(np.sqrt(np.clip(dur, 0.5, 3.0))))
            except Exception:
                continue
        if not wavs:
            continue
        embs = []
        for start in range(0, len(wavs), batch):
            embs.append(embedder.embed(wavs[start : start + batch]))
        E = np.vstack(embs).astype("float32")
        w = np.asarray(weights, dtype="float32")
        centroid, mean_sim, min_sim, used = _robust_side_centroid(E, weights=w)
        centroids.append(centroid)
        call_ids.append(call_id)
        channels.append(channel)
        n_segments.append(len(segs))
        n_used.append(used)
        mean_sims.append(mean_sim)
        min_sims.append(min_sim)

    if centroids:
        matrix = _l2_normalize_matrix(np.vstack(centroids).astype("float32"))
    else:
        matrix = np.zeros((0, 0), dtype="float32")
    return {
        "centroids": matrix,
        "call_id": np.asarray(call_ids, dtype=np.str_),
        "channel": np.asarray(channels, dtype=np.str_),
        "n_segments": np.asarray(n_segments, dtype=np.int32),
        "n_used": np.asarray(n_used, dtype=np.int32),
        "mean_sim": np.asarray(mean_sims, dtype=np.float32),
        "min_sim": np.asarray(min_sims, dtype=np.float32),
    }


def write_shard_embeddings(path: Path, data: dict[str, Any]) -> Path:
    """Atomically write a shard's side centroids to an ``.npz`` on shared storage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as fh:
        np.savez(fh, **data)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Global clustering
# ---------------------------------------------------------------------------


def _load_all_side_centroids(embed_root: Path) -> dict[str, Any]:
    files = sorted(Path(embed_root).glob("shard-*.npz"))
    if not files:
        raise FileNotFoundError(f"No shard-*.npz embedding files found under {embed_root}")
    mats: list[np.ndarray] = []
    call_ids: list[str] = []
    channels: list[str] = []
    for f in files:
        with np.load(f, allow_pickle=False) as npz:
            cents = npz["centroids"]
            if cents.size == 0:
                continue
            mats.append(cents.astype("float32"))
            call_ids.extend([str(x) for x in npz["call_id"].tolist()])
            channels.extend([str(x) for x in npz["channel"].tolist()])
    if not mats:
        raise ValueError(f"All embedding shards under {embed_root} were empty")
    X = _l2_normalize_matrix(np.vstack(mats).astype("float32"))
    if not (len(call_ids) == len(channels) == X.shape[0]):
        raise ValueError(f"Centroid/metadata length mismatch: X={X.shape[0]} call={len(call_ids)} chan={len(channels)}")
    return {"X": X, "call_id": call_ids, "channel": channels, "files": [str(f) for f in files]}


def _calibrate_threshold(X: np.ndarray, call_ids: list[str], channels: list[str], params: SpeakerParams) -> tuple[float, float, int]:
    """Threshold from same-call opposite-channel negatives (guaranteed different speakers)."""
    by_call: dict[str, list[int]] = {}
    for i, c in enumerate(call_ids):
        by_call.setdefault(c, []).append(i)
    neg: list[float] = []
    for idxs in by_call.values():
        for a_i in range(len(idxs)):
            for b_i in range(a_i + 1, len(idxs)):
                a, b = idxs[a_i], idxs[b_i]
                if channels[a] != channels[b]:
                    neg.append(float(X[a] @ X[b]))
    if len(neg) >= 100:
        auto = float(np.quantile(np.asarray(neg, dtype="float32"), 0.999) + 0.035)
        auto = min(max(auto, 0.90), 0.96)
    else:
        auto = 0.92
    edge = max(params.edge_sim_threshold, auto)
    component = max(0.90, edge - params.component_sim_margin)
    return edge, component, len(neg)


def _knn(X: np.ndarray, k: int, batch: int) -> tuple[np.ndarray, np.ndarray]:
    n, dim = X.shape
    k = min(k, max(1, n - 1))
    all_sims = np.empty((n, k + 1), dtype=np.float32)
    all_inds = np.empty((n, k + 1), dtype=np.int64)
    try:
        import faiss  # type: ignore

        faiss.omp_set_num_threads(os.cpu_count() or 8)
        index = faiss.IndexFlatIP(dim)
        index.add(X)
        for start in range(0, n, batch):
            end = min(start + batch, n)
            sims, inds = index.search(X[start:end], k + 1)
            all_sims[start:end] = sims
            all_inds[start:end] = inds
    except Exception:
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(X)
        for start in range(0, n, batch):
            end = min(start + batch, n)
            distances, inds = nn.kneighbors(X[start:end], return_distance=True)
            all_sims[start:end] = (1.0 - distances).astype("float32")
            all_inds[start:end] = inds.astype("int64")
    return all_sims, all_inds


def _strict_cluster(
    X: np.ndarray,
    call_ids: list[str],
    channels: list[str],
    edge_thr: float,
    component_thr: float,
    params: SpeakerParams,
) -> tuple[np.ndarray, dict[str, int], int]:
    """Mutual-kNN + constrained union-find. Returns (labels, reject_counts, accepted)."""
    n = X.shape[0]
    all_sims, all_inds = _knn(X, params.knn_k, params.knn_batch_size)

    neighbor_sets = [set(int(x) for x in all_inds[i] if int(x) >= 0 and int(x) != i) for i in range(n)]
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for sim, j in zip(all_sims[i], all_inds[i]):
            j = int(j)
            sim = float(sim)
            if j < 0 or j == i or sim < edge_thr:
                continue
            if i not in neighbor_sets[j]:  # mutual kNN only
                continue
            if i < j:
                edges.append((sim, i, j))
    edges.sort(reverse=True, key=lambda x: x[0])

    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)
    component_sum = X.copy().astype("float32")
    component_members: list[list[int]] = [[i] for i in range(n)]
    component_calls: list[dict[str, set]] = [{str(call_ids[i]): {str(channels[i])}} for i in range(n)]
    rng = np.random.default_rng(42)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def proto(root: int) -> np.ndarray:
        return _l2_normalize_vector(component_sum[root])

    def sample_members(members_a: list[int], members_b: list[int]) -> np.ndarray:
        total = len(members_a) + len(members_b)
        max_sample = params.compactness_sample_size
        if total <= max_sample:
            return np.asarray(members_a + members_b, dtype=np.int64)
        half = max_sample // 2
        if len(members_a) > half:
            a_sample = rng.choice(np.asarray(members_a, dtype=np.int64), size=half, replace=False)
        else:
            a_sample = np.asarray(members_a, dtype=np.int64)
        remaining = max_sample - len(a_sample)
        if len(members_b) > remaining:
            b_sample = rng.choice(np.asarray(members_b, dtype=np.int64), size=remaining, replace=False)
        else:
            b_sample = np.asarray(members_b, dtype=np.int64)
        return np.concatenate([a_sample, b_sample])

    def can_merge(a: int, b: int) -> tuple[bool, str]:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False, "same_component"
        if int(size[ra] + size[rb]) > params.max_call_sides_per_cluster:
            return False, "max_size_cap"
        ca, cb = component_calls[ra], component_calls[rb]
        for call_id in set(ca.keys()) & set(cb.keys()):
            if ca[call_id] != cb[call_id]:
                return False, "same_call_opposite_channel"
        if float(proto(ra) @ proto(rb)) < component_thr:
            return False, "low_component_similarity"
        new_proto = _l2_normalize_vector(component_sum[ra] + component_sum[rb])
        sample_ids = sample_members(component_members[ra], component_members[rb])
        p10 = float(np.quantile(X[sample_ids] @ new_proto, 0.10))
        if p10 < params.cluster_p10_sim_min:
            return False, "low_cluster_compactness"
        return True, "ok"

    def merge(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        component_sum[ra] += component_sum[rb]
        component_members[ra].extend(component_members[rb])
        component_members[rb] = []
        for call_id, chans in component_calls[rb].items():
            if call_id not in component_calls[ra]:
                component_calls[ra][call_id] = set(chans)
            else:
                component_calls[ra][call_id] |= set(chans)
        component_calls[rb] = {}

    reject_counts: dict[str, int] = {}
    accepted = 0
    for sim, i, j in edges:
        ok, reason = can_merge(i, j)
        if ok:
            merge(i, j)
            accepted += 1
        else:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    # Stable, deterministic label ids ordered by descending cluster size.
    unique, counts = np.unique(roots, return_counts=True)
    order = unique[np.argsort(-counts, kind="stable")]
    root_to_label = {int(r): k for k, r in enumerate(order)}
    labels = np.array([root_to_label[int(r)] for r in roots], dtype=np.int64)
    return labels, reject_counts, accepted


def cluster_global(embed_root: Path, out_path: Path, params: SpeakerParams) -> dict[str, Any]:
    """Cluster all shards' side centroids and write ``clusters.parquet``.

    ``clusters.parquet`` columns: ``call_id`` (str), ``channel`` (str),
    ``speaker_cluster_id`` (str, e.g. ``spk_000123``).
    """
    loaded = _load_all_side_centroids(embed_root)
    X = loaded["X"]
    call_ids = loaded["call_id"]
    channels = loaded["channel"]
    n = X.shape[0]

    edge_thr, component_thr, n_neg = _calibrate_threshold(X, call_ids, channels, params)
    labels, reject_counts, accepted = _strict_cluster(X, call_ids, channels, edge_thr, component_thr, params)
    cluster_ids = [f"spk_{int(lbl):06d}" for lbl in labels]

    # QC: no cluster may hold two channels of the same call (the hard invariant that
    # makes speaker_cluster_id a safe train/test grouping key).
    collisions = 0
    per_cluster_call_channels: dict[str, dict[str, set]] = {}
    for call_id, channel, cid in zip(call_ids, channels, cluster_ids):
        d = per_cluster_call_channels.setdefault(cid, {})
        chans = d.setdefault(call_id, set())
        if chans and channel not in chans:
            collisions += 1
        chans.add(channel)
    if collisions:
        raise AssertionError(
            f"Speaker clustering produced {collisions} same-call/opposite-channel collisions; "
            "the constrained union-find should make this impossible."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_clusters_parquet(out_path, call_ids, channels, cluster_ids)

    n_clusters = len(set(cluster_ids))
    cluster_side_counts: dict[str, int] = {}
    for cid in cluster_ids:
        cluster_side_counts[cid] = cluster_side_counts.get(cid, 0) + 1
    size_values = np.array(list(cluster_side_counts.values()), dtype=np.int64) if cluster_side_counts else np.array([0])
    summary = {
        "stage": "cluster_speakers",
        "embed_root": str(embed_root),
        "out_path": str(out_path),
        "shards": len(loaded["files"]),
        "call_sides_total": int(n),
        "distinct_calls": int(len({c for c in call_ids})),
        "speaker_clusters": int(n_clusters),
        "edge_sim_threshold": edge_thr,
        "component_sim_threshold": component_thr,
        "same_call_negatives_used": int(n_neg),
        "accepted_edges": int(accepted),
        "reject_counts": reject_counts,
        "same_call_channel_collisions": int(collisions),
        "cluster_side_count": {
            "max": int(size_values.max()),
            "mean": float(size_values.mean()),
            "singletons": int((size_values == 1).sum()),
        },
    }
    return summary


def _write_clusters_parquet(out_path: Path, call_ids: list[str], channels: list[str], cluster_ids: list[str]) -> None:
    tmp = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "call_id": pa.array(call_ids, type=pa.string()),
                "channel": pa.array(channels, type=pa.string()),
                "speaker_cluster_id": pa.array(cluster_ids, type=pa.string()),
            }
        )
        pq.write_table(table, str(tmp), compression="zstd")
    except Exception:
        # pyarrow/zstd unavailable: fall back to a JSONL sibling so the mapping is still durable.
        tmp = out_path.with_suffix(".jsonl")
        with tmp.open("w") as fh:
            for call_id, channel, cid in zip(call_ids, channels, cluster_ids):
                fh.write(json.dumps({"call_id": call_id, "channel": channel, "speaker_cluster_id": cid}) + "\n")
        os.replace(tmp, out_path.with_suffix(".jsonl"))
        return
    os.replace(tmp, out_path)


def load_cluster_map(path: Path) -> dict[tuple[str, str], str]:
    """Load ``(call_id, channel) -> speaker_cluster_id`` from clusters.parquet (or .jsonl)."""
    path = Path(path)
    if not path.exists():
        jsonl = path.with_suffix(".jsonl")
        if jsonl.exists():
            path = jsonl
        else:
            raise FileNotFoundError(f"Speaker cluster map not found: {path}")
    mapping: dict[tuple[str, str], str] = {}
    if path.suffix == ".jsonl":
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                mapping[(str(row["call_id"]), str(row["channel"]))] = str(row["speaker_cluster_id"])
        return mapping
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=["call_id", "channel", "speaker_cluster_id"])
    call_ids = table.column("call_id").to_pylist()
    channels = table.column("channel").to_pylist()
    cluster_ids = table.column("speaker_cluster_id").to_pylist()
    for call_id, channel, cid in zip(call_ids, channels, cluster_ids):
        mapping[(str(call_id), str(channel))] = str(cid)
    return mapping
