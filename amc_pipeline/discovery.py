from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from .config import PipelineConfig
from .models import AudioFileRecord

# --- Discovery cache -------------------------------------------------------
# On a large shared filesystem (e.g. FSx Lustre) the dominant cost of a sharded
# run is metadata: every stage on every box calls discover_audio_files(), which
# walks the ENTIRE input tree (os.walk + stat + sidecar read) before filtering
# to its shard. With N boxes x ~10 stages that is N*10 full tree walks per run.
#
# The cache turns that into a single walk: the first process to need discovery
# for a given (input_root, exts, hash_mode, num_shards) builds a per-shard
# manifest on shared storage; everyone else reads their shard slice with zero
# filesystem walking. Records are reconstructed byte-identically (same file_id,
# shard assignment, and input_signature) so existing on-disk stage markers stay
# valid. If anything goes wrong (timeout, meta mismatch) callers fall back to
# the live walk, so the cache can only help or be neutral -- never wrong.
_CACHE_ENV = "AMC_DISCOVERY_CACHE"  # "0"/"off" disables; default auto
_CACHE_DIR_ENV = "AMC_DISCOVERY_CACHE_DIR"  # explicit override
_CACHE_BUILD_TIMEOUT_SEC = float(os.environ.get("AMC_DISCOVERY_CACHE_TIMEOUT", "7200"))
_CACHE_POLL_SEC = float(os.environ.get("AMC_DISCOVERY_CACHE_POLL", "5"))
# A live builder heartbeats its .building lock (bumps mtime) every
# _CACHE_HEARTBEAT_SEC; a lock whose mtime is older than _CACHE_STALE_SEC is
# treated as abandoned (builder crashed/killed) and reclaimed. Stale must be a
# comfortable multiple of heartbeat so a slow-but-alive builder is never stolen.
# Default 120s (vs the old 3600s): a killed builder used to hang the whole fleet
# for an hour; now another box rebuilds within ~2 min.
_CACHE_HEARTBEAT_SEC = float(os.environ.get("AMC_DISCOVERY_CACHE_HEARTBEAT", "30"))
_CACHE_STALE_SEC = float(os.environ.get("AMC_DISCOVERY_CACHE_STALE", "120"))
# How often (in files) the builder flushes shard handles and rewrites progress.json.
# Small enough that `tmp_lines` and progress.json are a live progress signal; large
# enough that flushing 1 line at a time never dominates the walk.
_CACHE_PROGRESS_EVERY = int(os.environ.get("AMC_DISCOVERY_CACHE_PROGRESS_EVERY", "500"))
# Per-shard directory scan. When set, a sharded worker enumerates ONLY the top-level
# call directories that hash to its shard (<root>/<call_id>/<audio>), instead of
# walking the whole tree and filtering. This makes each box touch only its ~1/N of
# the input, so no single process ever reads every sidecar -- the exact operation
# that wedged the shared-cache builder on an HSM-released (S3-cold) tree. It also
# makes the shared discovery cache unnecessary (each box is independent), so when
# this is on the cache is bypassed. Layout assumption: the immediate child dir name
# is the call_id (true for the audio.* sidecar layout; matches infer_call_id).
_SHARD_DIRSCAN_ENV = "AMC_SHARD_DIRSCAN"


def discover_audio_files(config: PipelineConfig) -> list[AudioFileRecord]:
    if config.input_root is None:
        raise ValueError("input_root is required for discovery")
    _validate_shard_config(config)
    root = config.input_root.resolve()
    exts = {e.lower() for e in config.supported_extensions}
    hash_mode = normalized_hash_mode(config.discovery.hash_mode)
    num_shards = int(config.discovery.num_shards or 1)
    shard_index = config.discovery.shard_index

    # Per-shard dir scan: each box enumerates only its own ~1/N call dirs (no whole-tree
    # walk, no shared cache, no single builder reading every sidecar). The records are
    # identical to the full-walk+filter path for the <root>/<call_id>/<audio> layout.
    if _shard_dirscan_enabled() and shard_index is not None and num_shards > 1:
        records: list[AudioFileRecord] = []
        for path in _iter_shard_audio_paths(root, exts, num_shards, shard_index):
            rel = path.relative_to(root)
            call_id = infer_call_id(path, rel)
            records.append(_record_for_path(path, rel, call_id, hash_mode))
        return records

    cache_root = _discovery_cache_dir(config)
    if cache_root is not None:
        cached = _records_via_cache(root, exts, hash_mode, num_shards, shard_index, cache_root)
        if cached is not None:
            return cached

    records = []
    for path in _iter_audio_paths(root, exts):
        rel = path.relative_to(root)
        call_id = infer_call_id(path, rel)
        if not _belongs_to_shard(call_id, num_shards, shard_index):
            continue
        records.append(_record_for_path(path, rel, call_id, hash_mode))
    return records


def _record_for_path(path: Path, rel: Path, call_id: str, hash_mode: str) -> AudioFileRecord:
    """Build one AudioFileRecord by touching the live filesystem."""
    content_hash = fingerprint_file(path, rel, hash_mode)
    return _assemble_record(
        rel=rel,
        source_path=path.resolve(),
        size_bytes=path.stat().st_size,
        content_hash=content_hash,
        hash_mode=hash_mode,
        sidecar=read_sidecar_metadata(path),
        call_id=call_id,
    )


def _assemble_record(
    *,
    rel: Path,
    source_path: Path,
    size_bytes: int,
    content_hash: str,
    hash_mode: str,
    sidecar: dict[str, Any],
    call_id: str,
) -> AudioFileRecord:
    """Single source of truth for record fields shared by walk and cache paths."""
    mode = normalized_hash_mode(hash_mode)
    year = rel.parts[0] if len(rel.parts) > 1 else "unknown"
    file_id = hashlib.sha256(f"{rel.as_posix()}:{content_hash}".encode("utf-8")).hexdigest()[:24]
    duplicate_group_id = content_hash[:24] if mode == "content" else file_id
    return AudioFileRecord(
        file_id=file_id,
        source_path=source_path,
        relative_path=rel,
        year=year,
        call_id=call_id,
        extension=rel.suffix.lower(),
        size_bytes=size_bytes,
        content_hash=content_hash,
        duplicate_group_id=duplicate_group_id,
        sidecar_metadata=sidecar,
    )


def _iter_audio_paths(root: Path, exts: set[str]) -> Iterator[Path]:
    seen_dirs: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real_dir = Path(dirpath).resolve()
        if real_dir in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(real_dir)
        dirnames[:] = sorted(dirnames)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.is_file() and path.suffix.lower() in exts:
                yield path


def _shard_dirscan_enabled() -> bool:
    return os.environ.get(_SHARD_DIRSCAN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


# How many children a directory may have and still be read as a list of GROUPS
# (year dirs) rather than a list of calls. The corpus is staged either as
# <root>/<call_id>/ (hundreds of thousands of children) or as <root>/<year>/<call_id>/
# (a handful), so the count separates the two without a stat per child.
_MAX_GROUP_DIRS = 64


def _shard_call_dirs(
    root: Path, num_shards: int, shard_index: int
) -> Iterator[Path]:
    """Yield the call dirs under `root` that hash to this shard.

    Handles both stagings of the corpus: call dirs directly under `root`, and call dirs
    one level down under year dirs (how a run's `input/` is wired -- `input/2025` is a
    symlink to the year tree, which is also what puts "2025" in the manifest's `year`).
    Only the CALL dir name is ever hashed, so either way this yields exactly the calls
    the full-walk path would assign to this shard.
    """
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        return
    # A handful of children means they are year dirs, so recurse to find the calls. The
    # check is on the count alone: with the flat layout, asking "is this a call dir?"
    # per child would stat all ~400k of them, the metadata churn this scan exists to
    # avoid.
    if 0 < len(entries) <= _MAX_GROUP_DIRS:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            yield from _shard_call_dirs(Path(entry.path), num_shards, shard_index)
        return
    for entry in entries:
        # Hash-filter on the directory NAME first (pure CPU), before touching the
        # filesystem. On Lustre, readdir returns no d_type, so entry.is_dir() would
        # stat() every one of the (here ~480k) root entries -- ~7ms each under fleet
        # MDT contention => ~tens of minutes PER BOX of pure metadata churn before any
        # real work. By selecting our ~1/N dirs by name hash first we only ever stat
        # this shard's slice.
        if _shard_of(entry.name, num_shards) == shard_index:
            yield Path(entry.path)


def _iter_shard_audio_paths(
    root: Path, exts: set[str], num_shards: int, shard_index: int
) -> Iterator[Path]:
    """Yield audio paths only under the call dirs that hash to this shard.

    The call dir name IS the call_id (the same value infer_call_id derives for an
    audio.* file, and the same key _shard_of uses), so this yields EXACTLY the files
    this shard would own under the full-walk+filter path while touching only this
    shard's ~1/N of the call dirs instead of the whole tree. No process reads sidecars
    outside its slice, which is what wedged the shared-cache builder on an
    HSM-released (S3-cold) tree.
    """
    for call_dir in _shard_call_dirs(root, num_shards, shard_index):
        try:
            # A non-directory that happens to hash here just raises and is skipped.
            names = sorted(e.name for e in os.scandir(call_dir))
        except OSError:
            continue
        for name in names:
            path = call_dir / name
            if path.suffix.lower() in exts and path.is_file():
                yield path


def infer_call_id(path: Path, relative_path: Path) -> str:
    if path.stem.lower() == "audio" and path.parent.name:
        return path.parent.name
    stable = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{stable}"


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalized_hash_mode(mode: str) -> str:
    return str(mode or "content").lower().replace("-", "_")


def fingerprint_file(path: Path, relative_path: Path, mode: str) -> str:
    mode = normalized_hash_mode(mode)
    if mode == "content":
        return hash_file(path)
    if mode in {"fast", "path"}:
        return hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()
    if mode in {"path_size_mtime", "metadata"}:
        stat = path.stat()
        payload = f"{relative_path.as_posix()}:{stat.st_size}:{int(stat.st_mtime_ns)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    raise ValueError(f"Unsupported discovery hash mode: {mode}")


def _fingerprint_from_cache(rel: Path, size: int, mtime_ns: int, content_hash: str | None, mode: str) -> str:
    """Recompute the discovery fingerprint from cached metadata (no I/O)."""
    mode = normalized_hash_mode(mode)
    if mode == "content":
        if content_hash is None:
            raise ValueError("content-mode cache entry missing content hash")
        return content_hash
    if mode in {"fast", "path"}:
        return hashlib.sha256(rel.as_posix().encode("utf-8")).hexdigest()
    if mode in {"path_size_mtime", "metadata"}:
        payload = f"{rel.as_posix()}:{size}:{int(mtime_ns)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    raise ValueError(f"Unsupported discovery hash mode: {mode}")


def _validate_shard_config(config: PipelineConfig) -> None:
    num_shards = int(config.discovery.num_shards or 1)
    shard_index = config.discovery.shard_index
    config.discovery.num_shards = num_shards
    if num_shards < 1:
        raise ValueError("discovery.num_shards must be >= 1")
    if shard_index is None:
        return
    shard_index = int(shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("discovery.shard_index must be between 0 and num_shards - 1")
    config.discovery.shard_index = shard_index


def _belongs_to_shard(call_id: str, num_shards: int, shard_index: int | None) -> bool:
    if shard_index is None or num_shards <= 1:
        return True
    return _shard_of(call_id, num_shards) == shard_index


def _shard_of(call_id: str, num_shards: int) -> int:
    value = int(hashlib.sha1(str(call_id).encode("utf-8")).hexdigest(), 16)
    return value % num_shards


def read_sidecar_metadata(audio_path: Path) -> dict[str, Any]:
    sidecar = audio_path.parent / "metadata.json"
    if not sidecar.exists():
        return {}
    try:
        return flatten_dict(json.loads(sidecar.read_text()))
    except Exception as exc:
        return {"metadata_error": repr(exc)}


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, new_key))
        else:
            out[new_key] = value
    return out


# --- Discovery cache internals --------------------------------------------


def _discovery_cache_dir(config: PipelineConfig) -> Path | None:
    """Resolve the shared cache root, or None to disable (fall back to walking)."""
    mode = os.environ.get(_CACHE_ENV, "auto").strip().lower()
    if mode in {"0", "off", "false", "no", "disable", "disabled"}:
        return None
    explicit = os.environ.get(_CACHE_DIR_ENV)
    if explicit:
        return Path(explicit)
    # Auto: only the sharded fleet layout RUN_ROOT/outputs/shard-<n> has a shared
    # parent (RUN_ROOT) common to every worker. Anything else (tests, ad-hoc
    # single runs) gets no implicit cache so behavior is unchanged.
    out = config.output_root
    if out is not None and out.parent.name == "outputs":
        return out.parent.parent / "discovery-cache"
    return None


def _cache_key(root: Path, exts: set[str], hash_mode: str, num_shards: int) -> str:
    payload = "|".join(
        [str(root), ",".join(sorted(exts)), normalized_hash_mode(hash_mode), str(int(num_shards))]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _records_via_cache(
    root: Path,
    exts: set[str],
    hash_mode: str,
    num_shards: int,
    shard_index: int | None,
    cache_root: Path,
) -> list[AudioFileRecord] | None:
    base = _ensure_cache(root, exts, hash_mode, num_shards, cache_root)
    if base is None:
        return None
    records: list[AudioFileRecord] = []
    for entry in _read_cache_entries(base, num_shards, shard_index):
        rel = Path(entry["rel"])
        # root / rel reproduces the unresolved path the live walk passed to
        # infer_call_id (same stem + immediate parent name), so call_id matches.
        call_id = infer_call_id(root / rel, rel)
        content_hash = _fingerprint_from_cache(
            rel, int(entry["size"]), int(entry.get("mtime_ns", 0)), entry.get("chash"), hash_mode
        )
        records.append(
            _assemble_record(
                rel=rel,
                source_path=Path(entry["src"]),
                size_bytes=int(entry["size"]),
                content_hash=content_hash,
                hash_mode=hash_mode,
                sidecar=entry.get("sidecar") or {},
                call_id=call_id,
            )
        )
    return records


def discovery_counts(config: PipelineConfig) -> tuple[int, str, int]:
    """Return (shard_file_count, signature, total_count) for the input subset.

    Mirrors the historical ops/input_signature.py contract exactly so existing
    stage markers stay valid, but uses the shared cache when available.
    """
    if config.input_root is None:
        raise ValueError("input_root is required for discovery")
    _validate_shard_config(config)
    root = config.input_root.resolve()
    exts = {e.lower() for e in config.supported_extensions}
    hash_mode = normalized_hash_mode(config.discovery.hash_mode)
    num_shards = int(config.discovery.num_shards or 1)
    shard_index = config.discovery.shard_index

    # Per-shard dir scan: count/signature this shard's slice only. `total` is reported
    # as the shard's own count (this box never sees other shards); the run-level
    # "input present?" guard only needs total > 0, which holds for any non-empty shard.
    if _shard_dirscan_enabled() and shard_index is not None and num_shards > 1:
        shard_pairs = sorted(
            (path.relative_to(root).as_posix(), path.stat().st_size)
            for path in _iter_shard_audio_paths(root, exts, num_shards, shard_index)
        )
        return len(shard_pairs), _signature_for(shard_pairs), len(shard_pairs)

    cache_root = _discovery_cache_dir(config)
    if cache_root is not None:
        base = _ensure_cache(root, exts, hash_mode, num_shards, cache_root)
        if base is not None:
            entries = list(_read_cache_entries(base, num_shards, shard_index))
            shard_pairs = sorted((str(e["rel"]), int(e["size"])) for e in entries)
            total = _read_cache_total(base)
            return len(shard_pairs), _signature_for(shard_pairs), total

    total = 0
    shard_pairs: list[tuple[str, int]] = []
    for path in _iter_audio_paths(root, exts):
        total += 1
        rel = path.relative_to(root)
        call_id = infer_call_id(path, rel)
        if not _belongs_to_shard(call_id, num_shards, shard_index):
            continue
        shard_pairs.append((rel.as_posix(), path.stat().st_size))
    shard_pairs.sort()
    return len(shard_pairs), _signature_for(shard_pairs), total


def _signature_for(shard_pairs: list[tuple[str, int]]) -> str:
    payload = "\n".join(f"{rel}:{size}" for rel, size in shard_pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _ensure_cache(
    root: Path, exts: set[str], hash_mode: str, num_shards: int, cache_root: Path
) -> Path | None:
    """Ensure a complete manifest exists; build it once under a cross-host lock.

    Returns the manifest base dir, or None if the cache could not be produced
    (so the caller walks the tree instead).
    """
    base = cache_root / _cache_key(root, exts, hash_mode, num_shards)
    done = base / ".done"
    deadline = time.time() + _CACHE_BUILD_TIMEOUT_SEC
    while True:
        if done.exists():
            if _meta_ok(base, root, hash_mode, num_shards):
                return base
            return None  # stale/mismatched manifest -> fall back to walk
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        lock = base / ".building"
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if _is_stale(lock):
                _safe_unlink(lock)
                continue
            if time.time() > deadline:
                return None
            time.sleep(_CACHE_POLL_SEC)
            continue
        except OSError:
            return None
        try:
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
        except OSError:
            _safe_unlink(lock)
            return None
        # Keep the lock's mtime fresh while we walk so a long (but alive) build is
        # never mistaken for stale; the moment this process dies the heartbeat
        # stops, the mtime freezes, and another box reclaims after _CACHE_STALE_SEC.
        progress = {"seen": 0, "started_at": time.time(), "done": False}
        stop_hb = threading.Event()
        hb = threading.Thread(target=_heartbeat_lock, args=(lock, stop_hb, progress), daemon=True)
        hb.start()
        try:
            _build_cache(base, root, exts, hash_mode, num_shards, progress)
        except Exception:
            # Leave no half-written .done; drop the lock so another worker retries.
            _safe_unlink(done)
            return None
        finally:
            progress["done"] = True
            stop_hb.set()
            hb.join(timeout=2)
            _safe_unlink(lock)
        # loop: .done now exists


def _build_cache(
    base: Path,
    root: Path,
    exts: set[str],
    hash_mode: str,
    num_shards: int,
    progress: dict[str, Any] | None = None,
) -> None:
    mode = normalized_hash_mode(hash_mode)
    tmp = base / ".tmp"
    if tmp.exists():
        for p in tmp.glob("*"):
            _safe_unlink(p)
    tmp.mkdir(parents=True, exist_ok=True)
    handles = [open(tmp / f"shard-{i}.jsonl", "w", encoding="utf-8") for i in range(num_shards)]
    total = 0
    try:
        for path in _iter_audio_paths(root, exts):
            rel = path.relative_to(root)
            call_id = infer_call_id(path, rel)
            st = path.stat()
            entry: dict[str, Any] = {
                "rel": rel.as_posix(),
                "src": str(path.resolve()),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "sidecar": read_sidecar_metadata(path),
            }
            if mode == "content":
                entry["chash"] = hash_file(path)
            handles[_shard_of(call_id, num_shards)].write(json.dumps(entry) + "\n")
            total += 1
            if progress is not None:
                # Advancing `seen` is what keeps the build lock fresh (see
                # _heartbeat_lock); flushing makes tmp_lines + progress.json a live
                # signal and bounds how much a crashed builder loses.
                progress["seen"] = total
                if total % _CACHE_PROGRESS_EVERY == 0:
                    for h in handles:
                        h.flush()
                    _write_progress(base, progress)
    finally:
        for h in handles:
            h.close()
    # Publish atomically: move shard files into place, then meta, then .done last.
    for i in range(num_shards):
        os.replace(tmp / f"shard-{i}.jsonl", base / f"shard-{i}.jsonl")
    meta = {
        "input_root": str(root),
        "hash_mode": mode,
        "num_shards": int(num_shards),
        "exts": sorted(exts),
        "total": int(total),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    _safe_rmdir(tmp)
    (base / ".done").write_text("ok")
    if progress is not None:
        progress["seen"] = total
        _write_progress(base, {**progress, "done": True})


def _read_cache_entries(base: Path, num_shards: int, shard_index: int | None) -> Iterator[dict[str, Any]]:
    if shard_index is None or num_shards <= 1:
        indices = range(num_shards)
    else:
        indices = range(int(shard_index), int(shard_index) + 1)
    for i in indices:
        shard_file = base / f"shard-{i}.jsonl"
        if not shard_file.exists():
            continue
        with shard_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _read_cache_total(base: Path) -> int:
    try:
        meta = json.loads((base / "meta.json").read_text())
        return int(meta.get("total", 0))
    except Exception:
        return 0


def _meta_ok(base: Path, root: Path, hash_mode: str, num_shards: int) -> bool:
    try:
        meta = json.loads((base / "meta.json").read_text())
    except Exception:
        return False
    return (
        meta.get("input_root") == str(root)
        and normalized_hash_mode(meta.get("hash_mode", "")) == normalized_hash_mode(hash_mode)
        and int(meta.get("num_shards", -1)) == int(num_shards)
    )


def _is_stale(lock: Path) -> bool:
    try:
        return (time.time() - lock.stat().st_mtime) > _CACHE_STALE_SEC
    except OSError:
        return True


def _heartbeat_lock(lock: Path, stop: threading.Event, progress: dict[str, Any]) -> None:
    """Refresh the build lock's mtime ONLY while the walk is making progress.

    Runs in a daemon thread for the build's lifetime. Each beat it compares the
    builder's `seen` counter to the previous beat:
      * advanced  -> the walk is alive and moving; bump mtime so the lock stays held.
      * unchanged -> the walk is wedged (e.g. blocked in the kernel restoring an
        HSM-released file) OR the process died. Either way we DON'T bump, so the
        mtime ages past _CACHE_STALE_SEC and another box reclaims and retries.
    This ties liveness to *progress*, not mere process existence -- the blind spot
    that let a kernel-wedged builder hold the lock forever while looking healthy.
    If the lock vanishes (reclaimed/removed) we stop.
    """
    last_seen = 0
    while not stop.wait(_CACHE_HEARTBEAT_SEC):
        seen = int(progress.get("seen", 0))
        if seen == last_seen:
            continue
        last_seen = seen
        try:
            os.utime(lock, None)
        except OSError:
            return


def _write_progress(base: Path, progress: dict[str, Any]) -> None:
    """Publish build progress for observability (print_run_status, ops probes)."""
    try:
        (base / "progress.json").write_text(
            json.dumps(
                {
                    "seen": int(progress.get("seen", 0)),
                    "started_at": progress.get("started_at"),
                    "updated_at": time.time(),
                    "done": bool(progress.get("done", False)),
                }
            )
        )
    except OSError:
        pass


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _safe_rmdir(path: Path) -> None:
    try:
        for p in path.glob("*"):
            _safe_unlink(p)
        path.rmdir()
    except OSError:
        pass
