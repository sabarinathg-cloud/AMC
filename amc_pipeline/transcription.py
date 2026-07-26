from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import gc
import json
import math
import os
import queue
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .config import ASRModelConfig, PipelineConfig
from .models import ASRResult, SegmentRecord
from .progress import iter_progress


QWEN_REQUIRED_TRANSFORMERS = "4.57.6"

# Per-model dynamic-batching defaults: (max segment count, audio-seconds budget).
# A batch is closed when either the count cap or the summed audio-seconds budget
# is reached, so batches stay GPU-friendly regardless of segment length.
_DEFAULT_BATCH_LIMITS: dict[str, tuple[int, float]] = {
    "qwen": (8, 240.0),
    "cohere": (4, 160.0),
    "granite": (4, 160.0),
    # Parakeet is 0.6B and pads only to the longest clip in the batch, so it
    # tolerates far larger batches than the seq2seq models at a fraction of VRAM.
    "parakeet": (128, 1024.0),
}
# How many upcoming batches to load on a background thread while the GPU runs.
_PREFETCH_DEPTH = 2

# Whisper segment packing (#5). Whisper is trained on 30s windows; feeding it the
# raw sub-second VAD clips one at a time is out-of-distribution (worse WER, and
# slow because every call pays the full encode/decode + VAD overhead). We instead
# concatenate consecutive same-call segments into ~`_WHISPER_PACK_WINDOW_SEC`
# super-clips separated by `_WHISPER_PACK_GAP_SEC` of silence, run one batched
# transcribe per pack, then map the timestamped output back to each segment.
_WHISPER_PACK_WINDOW_SEC = 28.0
_WHISPER_PACK_GAP_SEC = 0.6


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


class ASRAdapter:
    name = "base"
    progress_enabled = True

    def preflight(self) -> None:
        return None

    def transcribe_batch(self, segments: list[SegmentRecord]) -> list[ASRResult]:
        raise NotImplementedError

    def close(self) -> None:
        torch = getattr(self, "_torch", None)
        for attr in ["_model", "_processor", "_tokenizer", "_prompt"]:
            if hasattr(self, attr):
                setattr(self, attr, None)
        _release_model_memory(torch)


class FixtureASRAdapter(ASRAdapter):
    def __init__(self, name: str, transcript: str, confidence: float = 0.9, language: str = "en"):
        self.name = name
        self.transcript = transcript
        self.confidence = confidence
        self.language = language

    def transcribe_batch(self, segments: list[SegmentRecord]) -> list[ASRResult]:
        return [
            ASRResult(s.segment_id, self.name, self.transcript, self.confidence, self.language, self.confidence)
            for s in segments
        ]


class WhisperAdapter(ASRAdapter):
    name = "whisper"

    def __init__(
        self,
        model_path: str,
        batch_size: int = 0,
        device: str = "auto",
        pack_window_sec: float = _WHISPER_PACK_WINDOW_SEC,
        pack_gap_sec: float = _WHISPER_PACK_GAP_SEC,
    ):
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.device = device
        self.pack_window_sec = max(1.0, float(pack_window_sec))
        self.pack_gap_sec = max(0.0, float(pack_gap_sec))
        # Per-segment is the DEFAULT because it is WER-safe: each clip is already a
        # tight VAD segment, so Whisper transcribes it exactly. Segment packing (#5)
        # is ~3x faster but, on very short backchannel clips, Whisper hallucinates
        # filler into the inter-segment silence/padding ("Hello" -> "Hello okay okay
        # ... no"), which fails our best-WER bar. Opt into packing with
        # AMC_WHISPER_BATCHED=1 when throughput matters more than transcript fidelity.
        self.batched = os.environ.get("AMC_WHISPER_BATCHED", "0") == "1"
        # Full-call batched path (LOSSLESS speedup, ~4-5x): rebuild each call's
        # per-channel timeline from its segment clips at their REAL start offsets
        # (so the inter-segment silences are the true gaps), then run the batched
        # pipeline over the whole call. VAD splits the call back into ~30s windows
        # and decodes them in parallel on the GPU. Unlike the old _pack path this
        # preserves real timing, so VAD cuts exactly at segment boundaries and word
        # timestamps map back cleanly -- transcripts match the per-segment path.
        # Off by default until the parity check confirms WER on the fleet.
        self.fullfile = os.environ.get("AMC_WHISPER_FULLFILE_BATCH", "0") == "1"
        # beam_size=5 is the WER-safe default. Greedy (1) is roughly 2-3x faster
        # per clip; on tight VAD segments the beam rarely changes the 1-best, so
        # AMC_WHISPER_BEAM_SIZE=1 is the cheapest throughput lever -- gate it on
        # the parity benchmark (ops/asr_model_bench.py) before adopting it.
        self.beam_size = max(1, _env_int("AMC_WHISPER_BEAM_SIZE", 5))
        self._model = None
        self._internal_batch_size = 0

    def preflight(self) -> None:
        _require_path(self.model_path, self.name)
        _require_import("faster_whisper", self.name)

    def _load(self):
        if self._model is None:
            from faster_whisper import BatchedInferencePipeline, WhisperModel  # type: ignore

            torch = _optional_import("torch")
            _clear_cuda(torch)
            fw_device = _resolve_torch_device(self.device, torch, prefer_colon=False)
            # compute_type: float16 is the WER-safe GPU default. int8_float16 is a
            # near-lossless CTranslate2 quantization (<1% WER drift in practice) that
            # frees ~35% VRAM and is marginally faster -- opt in with
            # AMC_WHISPER_COMPUTE_TYPE=int8_float16 once parity is confirmed.
            default_compute_type = "float16" if fw_device == "cuda" else "int8"
            fw_compute_type = os.environ.get("AMC_WHISPER_COMPUTE_TYPE", "").strip() or default_compute_type
            default_internal_batch = 32 if fw_device == "cuda" else 4
            self._internal_batch_size = self.batch_size if self.batch_size and self.batch_size > 1 else default_internal_batch
            base = WhisperModel(
                str(self.model_path),
                device=fw_device,
                compute_type=fw_compute_type,
                local_files_only=True,
                cpu_threads=max(1, (os.cpu_count() or 1) // 2),
                num_workers=4,
            )
            self._model = BatchedInferencePipeline(model=base)
        return self._model

    def transcribe_batch(self, segments: list[SegmentRecord]) -> list[ASRResult]:
        model = self._load()
        if self.fullfile:
            np = _optional_import("numpy")
            if np is not None:
                return self._transcribe_calls_batched(model, np, segments)
        np = _optional_import("numpy") if self.batched else None
        if np is None:
            # Legacy / fallback: one transcribe per segment (still drops the
            # redundant VAD pass per #1 since each clip is already a VAD segment).
            return [
                self._transcribe_one(model, seg)
                for seg in iter_progress(
                    segments, desc="Whisper ASR", total=len(segments), unit="segment", enabled=self.progress_enabled
                )
            ]

        by_id: dict[str, ASRResult] = {}
        packs = list(self._pack_segments(segments))
        for pack in iter_progress(packs, desc="Whisper ASR", total=len(packs), unit="pack", enabled=self.progress_enabled):
            try:
                self._transcribe_pack(model, np, pack, by_id)
            except Exception:
                # A single bad pack must not drop a whole batch of segments; retry
                # those segments one at a time on the simple path.
                for seg, _start, _end in pack:
                    by_id[seg.segment_id] = self._transcribe_one(model, seg)
        return [
            by_id.get(s.segment_id, ASRResult(s.segment_id, self.name, "", 0.0, error="NOT_RUN"))
            for s in segments
        ]

    def _pack_segments(
        self, segments: list[SegmentRecord]
    ) -> Iterator[list[tuple[SegmentRecord, float, float]]]:
        """Group consecutive same-call segments into ~`pack_window_sec` clips.

        Segments are grouped by call and ordered by start time so the packed audio
        approximates the original call (good context => better WER). Each yielded
        pack is a list of ``(segment, start_sec_in_pack, end_sec_in_pack)`` using
        nominal durations; the exact end is recomputed from decoded samples when
        the pack audio is built.
        """
        by_call: dict[str, list[SegmentRecord]] = {}
        for seg in segments:
            by_call.setdefault(seg.call_id or seg.file_id or "", []).append(seg)
        for call_segs in by_call.values():
            call_segs.sort(key=lambda s: (float(getattr(s, "start_sec", 0.0) or 0.0), s.segment_id))
            pack: list[tuple[SegmentRecord, float, float]] = []
            cursor = 0.0
            for seg in call_segs:
                dur = max(0.0, float(getattr(seg, "duration_sec", 0.0) or 0.0))
                if pack and cursor + dur > self.pack_window_sec:
                    yield pack
                    pack = []
                    cursor = 0.0
                pack.append((seg, cursor, cursor + dur))
                cursor += dur + self.pack_gap_sec
            if pack:
                yield pack

    def _transcribe_pack(self, model, np, pack, by_id: dict[str, ASRResult]) -> None:
        sr = 16000
        gap = np.zeros(int(self.pack_gap_sec * sr), dtype=np.float32)
        parts: list[Any] = []
        windows: list[tuple[SegmentRecord, float, float]] = []
        for seg, _start, _end in pack:
            samples = np.asarray(_load_mono_samples(Path(seg.segment_audio_path)), dtype=np.float32)
            if samples.size == 0:
                samples = np.zeros(int(0.1 * sr), dtype=np.float32)
            start = sum(p.size for p in parts) / sr
            parts.append(samples)
            windows.append((seg, start, start + samples.size / sr))
            if gap.size:
                parts.append(gap)
        audio = np.concatenate(parts) if parts else np.zeros(int(0.1 * sr), dtype=np.float32)

        # Batched, independent per-segment decode: let the batched pipeline's VAD
        # re-split the pack back into one chunk per segment (our gap is well above
        # the tuned min-silence) and decode each chunk independently on the GPU. No
        # cross-segment context, so none of Whisper's long-audio drop/merge/repeat
        # behaviour -- transcripts match the solo per-segment path, at batch speed.
        out, info = model.transcribe(
            audio,
            language=None,
            beam_size=self.beam_size,
            batch_size=self._internal_batch_size,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 150,  # < pack_gap so it splits AT our gaps
                "speech_pad_ms": 200,            # don't clip leading/trailing phonemes
                "threshold": 0.3,                # keep quiet backchannels (avoid drops)
            },
            condition_on_previous_text=False,
            without_timestamps=False,
            # Kill Whisper's decode-loop hallucinations ("okay. okay. okay...") that
            # otherwise show up on short/ambiguous clips. Trigram block leaves real
            # double-words intact; the mild penalty discourages longer runs.
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
        )
        lang = getattr(info, "language", None)
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        dur_after_vad = getattr(info, "duration_after_vad", None)

        buckets: dict[str, list[tuple[float, str]]] = {seg.segment_id: [] for seg, _s, _e in pack}
        for ws in out:
            text = (getattr(ws, "text", "") or "").strip()
            if not text:
                continue
            ws_start = float(getattr(ws, "start", 0.0) or 0.0)
            ws_end = float(getattr(ws, "end", ws_start) or ws_start)
            best_seg, best_ov = None, 0.0
            for seg, w_start, w_end in windows:
                ov = min(ws_end, w_end) - max(ws_start, w_start)
                if ov > best_ov:
                    best_ov, best_seg = ov, seg
            if best_seg is None:
                mid = 0.5 * (ws_start + ws_end)
                best_seg = min(windows, key=lambda w: abs(0.5 * (w[1] + w[2]) - mid))[0]
            buckets[best_seg.segment_id].append((ws_start, text))

        for seg, _s, _e in pack:
            ordered = sorted(buckets.get(seg.segment_id, []), key=lambda x: x[0])
            transcript = " ".join(t for _, t in ordered).strip()
            if not transcript:
                # Safety net: the VAD dropped this (usually a very short/quiet clip).
                # Re-transcribe it solo so packing can never lose a segment.
                by_id[seg.segment_id] = self._transcribe_one(model, seg)
                continue
            by_id[seg.segment_id] = ASRResult(
                seg.segment_id,
                self.name,
                transcript,
                lang_prob,
                lang,
                lang_prob,
                raw={"duration_after_vad": dur_after_vad},
            )

    def _transcribe_one(self, model, segment: SegmentRecord) -> ASRResult:
        try:
            out, info = model.transcribe(
                str(segment.segment_audio_path),
                language=None,
                beam_size=self.beam_size,
                batch_size=self._internal_batch_size,
                # The clip is already a tight VAD segment, so a second Silero pass
                # is pure overhead and can clip leading/trailing phonemes (#1).
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
            )
            text = " ".join(x.text.strip() for x in out).strip()
            language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
            return ASRResult(
                segment.segment_id,
                self.name,
                text,
                language_probability,
                getattr(info, "language", None),
                language_probability,
                raw={"duration_after_vad": getattr(info, "duration_after_vad", None)},
            )
        except Exception as exc:
            return ASRResult(segment.segment_id, self.name, "", 0.0, error=repr(exc))

    def _transcribe_calls_batched(
        self, model, np, segments: list[SegmentRecord]
    ) -> list[ASRResult]:
        """Lossless full-call batched decode.

        Groups segments by (call/file, channel), reconstructs each channel's
        timeline by writing every segment's samples at its real ``start_sec``
        offset (gaps stay as true silence), runs the batched pipeline over the
        whole call with word timestamps, and maps each word back to the segment
        whose window it overlaps. Empty/dropped segments fall back to the solo
        per-segment decode so packing can never lose a segment.
        """
        sr = 16000
        by_id: dict[str, ASRResult] = {}

        groups: dict[tuple[str, int], list[SegmentRecord]] = {}
        for seg in segments:
            key = (seg.call_id or seg.file_id or "", int(getattr(seg, "channel", 0) or 0))
            groups.setdefault(key, []).append(seg)

        for call_segs in iter_progress(
            list(groups.values()),
            desc="Whisper ASR",
            total=len(groups),
            unit="call",
            enabled=self.progress_enabled,
        ):
            try:
                self._transcribe_one_call(model, np, sr, call_segs, by_id)
            except Exception:
                for seg in call_segs:
                    by_id[seg.segment_id] = self._transcribe_one(model, seg)

        return [
            by_id.get(s.segment_id, ASRResult(s.segment_id, self.name, "", 0.0, error="NOT_RUN"))
            for s in segments
        ]

    def _transcribe_one_call(self, model, np, sr: int, call_segs, by_id: dict[str, ASRResult]) -> None:
        call_segs = sorted(
            call_segs, key=lambda s: (float(getattr(s, "start_sec", 0.0) or 0.0), s.segment_id)
        )
        base = min(float(getattr(s, "start_sec", 0.0) or 0.0) for s in call_segs)

        windows: list[tuple[SegmentRecord, float, float]] = []
        loaded: list[tuple[float, Any]] = []
        for seg in call_segs:
            samples = np.asarray(_load_mono_samples(Path(seg.segment_audio_path)), dtype=np.float32)
            if samples.size == 0:
                # No audio to place; will be picked up by the empty-bucket fallback.
                windows.append((seg, -1.0, -1.0))
                continue
            start = max(0.0, float(getattr(seg, "start_sec", 0.0) or 0.0) - base)
            loaded.append((start, samples))
            windows.append((seg, start, start + samples.size / sr))

        if not loaded:
            for seg in call_segs:
                by_id[seg.segment_id] = self._transcribe_one(model, seg)
            return

        total = max(int(round((start + s.size / sr) * sr)) for start, s in loaded)
        audio = np.zeros(total, dtype=np.float32)
        for start, samples in loaded:
            off = int(round(start * sr))
            end = min(off + samples.size, total)
            audio[off:end] = samples[: end - off]

        out, info = model.transcribe(
            audio,
            language=None,
            beam_size=self.beam_size,
            batch_size=self._internal_batch_size,
            # Real silences separate the segments, so VAD re-splits the call into
            # its original speech regions; speech_pad keeps phonemes intact.
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 150,
                "speech_pad_ms": 200,
                "threshold": 0.3,
            },
            condition_on_previous_text=False,
            word_timestamps=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
        )
        lang = getattr(info, "language", None)
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        dur_after_vad = getattr(info, "duration_after_vad", None)

        real_windows = [(seg, s, e) for seg, s, e in windows if e >= 0.0]
        buckets: dict[str, list[tuple[float, str]]] = {seg.segment_id: [] for seg, _s, _e in real_windows}
        for out_seg in out:
            words = getattr(out_seg, "words", None)
            tokens = words if words else [out_seg]
            for tok in tokens:
                text = (getattr(tok, "word", None) or getattr(tok, "text", "") or "").strip()
                if not text:
                    continue
                t_start = float(getattr(tok, "start", 0.0) or 0.0)
                t_end = float(getattr(tok, "end", t_start) or t_start)
                best_seg, best_ov = None, 0.0
                for seg, w_start, w_end in real_windows:
                    ov = min(t_end, w_end) - max(t_start, w_start)
                    if ov > best_ov:
                        best_ov, best_seg = ov, seg
                if best_seg is None and real_windows:
                    mid = 0.5 * (t_start + t_end)
                    best_seg = min(real_windows, key=lambda w: abs(0.5 * (w[1] + w[2]) - mid))[0]
                if best_seg is not None:
                    buckets[best_seg.segment_id].append((t_start, text))

        for seg in call_segs:
            ordered = sorted(buckets.get(seg.segment_id, []), key=lambda x: x[0])
            transcript = " ".join(t for _, t in ordered).strip()
            if not transcript:
                by_id[seg.segment_id] = self._transcribe_one(model, seg)
                continue
            by_id[seg.segment_id] = ASRResult(
                seg.segment_id,
                self.name,
                transcript,
                lang_prob,
                lang,
                lang_prob,
                raw={"duration_after_vad": dur_after_vad},
            )


class ParakeetAdapter(ASRAdapter):
    """NVIDIA Parakeet TDT (FastConformer + Token-and-Duration Transducer).

    The Whisper replacement. Whisper's encoder always consumes a fixed 30s mel
    window, so on our ~4s VAD clips the large majority of its compute goes into
    padding silence. FastConformer consumes the true clip length and pads only to
    the longest clip in the batch, which is where the order-of-magnitude speedup
    on this workload comes from -- at equal or better WER.

    Runs on stock `transformers` (ParakeetForTDT landed in 5.9.0), so no NeMo
    dependency. That transformers floor is above the main venv's qwen-pinned
    4.57.6, so this adapter lives in its own venv exactly like cohere/align;
    see ops/setup_env.sh.
    """

    name = "parakeet"

    def __init__(self, cfg: ASRModelConfig):
        self.model_path = Path(cfg.path or "")
        self.batch_size = max(1, int(cfg.batch_size or 64))
        self.device = cfg.device
        self.dtype_name = cfg.dtype
        self.attn_implementation = cfg.attn_implementation
        self.prefetch = bool(cfg.prefetch)
        self.cap, self.audio_budget = _resolve_batch_limits(cfg, self.name)
        self._processor = None
        self._model = None
        self._torch = None

    def preflight(self) -> None:
        _require_path(self.model_path, self.name)
        _require_import("torch", self.name)
        _require_import("transformers", self.name)
        _require_parakeet_support()

    def _load(self):
        if self._model is None:
            _configure_quiet_transformers()
            import torch  # type: ignore

            # Must precede the transformers import: 5.x resolves FP8 MX dtypes at
            # import time and this venv is on the shared torch 2.5.1 pin, same as
            # the cohere adapter below.
            _ensure_torch_fp8_dtype_shim(torch)
            from transformers import AutoModelForTDT, AutoProcessor  # type: ignore

            _configure_torch_for_inference(torch)
            _clear_cuda(torch)
            self._torch = torch
            self._processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
            # bf16 is NVIDIA's reference inference precision for this checkpoint and
            # is what the published WER was measured at, so it is the default here
            # rather than an opt-in. Override with the model's `dtype` config field.
            default_dtype = torch.bfloat16 if _wants_cuda(self.device, torch) else torch.float32
            self._model = _load_model_with_attn(
                AutoModelForTDT,
                str(self.model_path),
                attn_implementation=self.attn_implementation,
                device_map=_resolve_device_map(self.device, torch),
                torch_dtype=_resolve_dtype(self.dtype_name, torch, default_dtype),
                local_files_only=True,
            )
            self._model.eval()
        return self._processor, self._model, self._torch

    def transcribe_batch(self, segments: list[SegmentRecord]) -> list[ASRResult]:
        self._load()
        by_id: dict[str, ASRResult] = {}
        # Parakeet v3 detects language itself, so unlike the seq2seq adapters there
        # is no per-language grouping. Sorting by duration keeps intra-batch padding
        # minimal, which is a direct speed win when the batch pads to the longest clip.
        ordered = sorted(segments, key=lambda s: (s.duration_sec, s.segment_id))
        chunks = list(_dynamic_batches(ordered, self.audio_budget, self.cap))

        def _load_audios(chunk: list[SegmentRecord]) -> list[list[float]]:
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
                return list(pool.map(lambda s: _load_cohere_audio(s.segment_audio_path), chunk))

        produced = _prefetch_batches(chunks, _load_audios, enabled=self.prefetch)
        for chunk, audios, load_error in iter_progress(
            produced, desc="Parakeet ASR", total=len(chunks), unit="batch", enabled=self.progress_enabled
        ):
            if load_error is not None:
                texts, errors = self._transcribe_segments_individually(chunk)
            else:
                texts, errors = self._transcribe_loaded(audios)
            for segment, text, error in zip(chunk, texts, errors):
                by_id[segment.segment_id] = ASRResult(
                    segment.segment_id,
                    self.name,
                    text,
                    0.0,
                    _language_code(segment.language),
                    error=error,
                )
        return [by_id.get(s.segment_id, ASRResult(s.segment_id, self.name, "", 0.0, error="NOT_RUN")) for s in segments]

    def _transcribe_segments_individually(self, chunk: list[SegmentRecord]) -> tuple[list[str], list[str | None]]:
        """Slow path used only when the prefetch loader raised for a whole batch."""
        texts: list[str] = []
        errors: list[str | None] = []
        for segment in chunk:
            try:
                audio = _load_cohere_audio(segment.segment_audio_path)
            except Exception as exc:
                texts.append("")
                errors.append(repr(exc))
                continue
            batch_texts, batch_errors = self._transcribe_loaded([audio])
            texts.append(batch_texts[0])
            errors.append(batch_errors[0])
        return texts, errors

    def _transcribe_loaded(self, audios: list[list[float]]) -> tuple[list[str], list[str | None]]:
        torch = self._torch
        assert torch is not None
        try:
            return self._generate(audios)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(audios) > 1:
                return _split_and_join(self._transcribe_loaded, audios)
            return [""], ["CUDA_OUT_OF_MEMORY"]
        except Exception as exc:
            if len(audios) > 1:
                return _split_and_join(self._transcribe_loaded, audios)
            return [""], [repr(exc)]

    def _generate(self, audios: list[list[float]]) -> tuple[list[str], list[str | None]]:
        processor, model, torch = self._load()
        np = _optional_import("numpy")
        arrays: Any = [np.asarray(a, dtype=np.float32) for a in audios] if np is not None else audios
        sample_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000) or 16000)
        inputs = processor(arrays, sampling_rate=sample_rate, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            generated = model.generate(**inputs)
        # TDT generate returns ParakeetRNNTGenerateOutput(sequences, durations), not a
        # bare id tensor. Handing the container to batch_decode iterates its KEYS, so
        # the tokenizer receives the strings "sequences"/"durations" and raises
        # "argument 'ids': 'str' object cannot be interpreted as an integer".
        predicted_ids = getattr(generated, "sequences", generated)
        decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        texts = [str(t).strip() for t in decoded]
        if len(texts) != len(audios):
            raise RuntimeError(f"parakeet returned {len(texts)} transcripts for {len(audios)} inputs")
        return texts, [None] * len(texts)


class QwenAdapter(ASRAdapter):
    name = "qwen"

    def __init__(self, cfg: ASRModelConfig):
        self.model_path = Path(cfg.path or "")
        self.batch_size = max(1, int(cfg.batch_size or 8))
        self.device = cfg.device
        self.cap, self.audio_budget = _resolve_batch_limits(cfg, self.name)
        self._model = None
        self._torch = None

    def preflight(self) -> None:
        _require_path(self.model_path, self.name)
        _require_qwen_model_config(self.model_path)
        _require_import("qwen_asr", self.name)
        _require_import("torch", self.name)
        _require_qwen_runtime_versions()

    def _load(self):
        if self._model is None:
            _configure_quiet_transformers()
            import torch  # type: ignore

            from qwen_asr import Qwen3ASRModel  # type: ignore

            _configure_torch_for_inference(torch)
            _clear_cuda(torch)
            qwen_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            qwen_device = _resolve_torch_device(self.device, torch, prefer_colon=True)
            self._model = Qwen3ASRModel.from_pretrained(
                str(self.model_path),
                dtype=qwen_dtype,
                device_map=qwen_device,
                max_inference_batch_size=self.cap,
                max_new_tokens=256,
            )
            _set_pad_token_to_eos(self._model)
            self._torch = torch
        return self._model, self._torch

    def transcribe_batch(self, segments: list[SegmentRecord]) -> list[ASRResult]:
        model, torch = self._load()
        by_id: dict[str, ASRResult] = {}
        for language, group in _segments_by_language(segments).items():
            ordered = sorted(group, key=lambda s: (s.duration_sec, s.segment_id))
            chunks = list(_dynamic_batches(ordered, self.audio_budget, self.cap))
            for chunk in iter_progress(chunks, desc=f"Qwen ASR {language}", total=len(chunks), unit="batch", enabled=self.progress_enabled):
                texts, errors = self._transcribe_paths(model, torch, [str(s.segment_audio_path) for s in chunk], _language_name(language))
                for segment, text, error in zip(chunk, texts, errors):
                    by_id[segment.segment_id] = ASRResult(
                        segment.segment_id,
                        self.name,
                        text,
                        0.0,
                        _language_code(language),
                        error=error,
                    )
        return [by_id.get(s.segment_id, ASRResult(s.segment_id, self.name, "", 0.0, error="NOT_RUN")) for s in segments]

    def _transcribe_paths(self, model, torch, paths: list[str], language_name: str) -> tuple[list[str], list[str | None]]:
        try:
            with torch.inference_mode():
                raw = model.transcribe(audio=paths, language=language_name)
            return _texts_and_errors(raw, len(paths))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._transcribe_paths_one_by_one(model, torch, paths, language_name)
        except Exception:
            return self._transcribe_paths_one_by_one(model, torch, paths, language_name)

    def _transcribe_paths_one_by_one(self, model, torch, paths: list[str], language_name: str) -> tuple[list[str], list[str | None]]:
        texts: list[str] = []
        errors: list[str | None] = []
        for path in paths:
            try:
                with torch.inference_mode():
                    raw = model.transcribe(audio=[path], language=language_name)
                batch_texts, batch_errors = _texts_and_errors(raw, 1)
                texts.append(batch_texts[0])
                errors.append(batch_errors[0])
            except Exception as exc:
                texts.append("")
                errors.append(repr(exc))
        return texts, errors


class CohereAdapter(ASRAdapter):
    name = "cohere"

    def __init__(self, cfg: ASRModelConfig):
        self.model_path = Path(cfg.path or "")
        self.batch_size = max(1, int(cfg.batch_size or 4))
        self.device = cfg.device
        self.dtype_name = cfg.dtype
        self.attn_implementation = cfg.attn_implementation
        self.prefetch = bool(cfg.prefetch)
        self.cap, self.audio_budget = _resolve_batch_limits(cfg, self.name)
        self._processor = None
        self._model = None
        self._torch = None
        self._device = None
        self._dtype = None

    def preflight(self) -> None:
        _require_path(self.model_path, self.name)
        _require_import("transformers", self.name)
        _require_import("torch", self.name)

    def transcribe_batch(self, segments: list[SegmentRecord]) -> list[ASRResult]:
        self._load()
        by_id: dict[str, ASRResult] = {}
        for language, group in _segments_by_language(segments).items():
            ordered = sorted(group, key=lambda s: (s.duration_sec, s.segment_id))
            chunks = list(_dynamic_batches(ordered, self.audio_budget, self.cap))
            lang_code = _language_code(language)

            def _load_audios(chunk: list[SegmentRecord]) -> list[list[float]]:
                with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
                    return list(pool.map(lambda s: _load_cohere_audio(s.segment_audio_path), chunk))

            produced = _prefetch_batches(chunks, _load_audios, enabled=self.prefetch)
            for chunk, audios, load_error in iter_progress(
                produced, desc=f"Cohere ASR {language}", total=len(chunks), unit="batch", enabled=self.progress_enabled
            ):
                if load_error is not None:
                    texts, errors = self._transcribe_paths([str(s.segment_audio_path) for s in chunk], lang_code)
                else:
                    texts, errors = self._transcribe_loaded(audios, lang_code)
                for segment, text, error in zip(chunk, texts, errors):
                    by_id[segment.segment_id] = ASRResult(
                        segment.segment_id,
                        self.name,
                        text,
                        0.0,
                        lang_code,
                        error=error,
                    )
        return [by_id.get(s.segment_id, ASRResult(s.segment_id, self.name, "", 0.0, error="NOT_RUN")) for s in segments]

    def _load(self):
        if self._model is None:
            _configure_quiet_transformers()
            import torch  # type: ignore

            _ensure_torch_fp8_dtype_shim(torch)
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration  # type: ignore

            _configure_torch_for_inference(torch)
            _clear_cuda(torch)
            self._torch = torch
            self._processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
            # Default stays float32 (bit-for-bit unchanged) until the parity gate
            # (ops/asr_parity_check.py) clears bf16; users opt in via dtype config.
            torch_dtype = _resolve_dtype(self.dtype_name, torch, torch.float32)
            self._model = _load_model_with_attn(
                CohereAsrForConditionalGeneration,
                str(self.model_path),
                attn_implementation=self.attn_implementation,
                device_map=_resolve_device_map(self.device, torch),
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
            self._model.eval()
            self._device = _model_device(self._model)
            self._dtype = _model_dtype(self._model)
        return self._processor, self._model, self._torch

    def _transcribe_loaded(self, audios: list[list[float]], language: str) -> tuple[list[str], list[str | None]]:
        torch = self._torch
        assert torch is not None
        try:
            return self._generate_from_audios(audios, language)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(audios) > 1:
                return _split_and_join(lambda part: self._transcribe_loaded(part, language), audios)
            return [""], ["CUDA_OUT_OF_MEMORY"]
        except Exception:
            if len(audios) > 1:
                return _split_and_join(lambda part: self._transcribe_loaded(part, language), audios)
            text, err = self._generate_single_audio(audios[0], language)
            return [text], [err]

    def _transcribe_paths(self, paths: list[str], language: str) -> tuple[list[str], list[str | None]]:
        """Path-based slow path used only when prefetch loading raised for a batch."""
        torch = self._torch
        assert torch is not None
        try:
            audios = [_load_cohere_audio(p) for p in paths]
        except Exception:
            return _split_and_join(lambda part: self._transcribe_paths(part, language), paths) if len(paths) > 1 else (
                [self._generate_single_path(paths[0], language)[0]],
                [self._generate_single_path(paths[0], language)[1]],
            )
        return self._transcribe_loaded(audios, language)

    def _generate_from_audios(self, audios: list[list[float]], language: str) -> tuple[list[str], list[str | None]]:
        processor, model, torch = self._load()
        # The Cohere feature extractor calls `.shape` on each raw waveform, so it
        # needs ndarray inputs -- plain Python lists raise
        # AttributeError("'list' object has no attribute 'shape'"). numpy ships
        # with transformers, so this is always importable in the cohere venv.
        import numpy as np  # type: ignore

        np_audios = [np.asarray(a, dtype=np.float32) for a in audios]
        inputs = processor(
            np_audios,
            sampling_rate=16000,
            return_tensors="pt",
            language=language,
            punctuation=False,
        )
        audio_chunk_index = _clone_for_decode(inputs.get("audio_chunk_index", None) if hasattr(inputs, "get") else None, torch)
        inputs = _move_inputs_to_device_dtype(inputs, self._device, self._dtype, torch)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=256)
        texts = _decode_cohere(processor, outputs, audio_chunk_index, language)
        return _pad_texts(texts, len(audios))

    def _generate_single_audio(self, audio: list[float], language: str) -> tuple[str, str | None]:
        try:
            texts, _ = self._generate_from_audios([audio], language)
            return (texts[0] if texts else ""), None
        except Exception as exc:
            return "", repr(exc)

    def _generate_single_path(self, path: str, language: str) -> tuple[str, str | None]:
        try:
            return self._generate_single_audio(_load_cohere_audio(path), language)
        except Exception as exc:
            return "", repr(exc)


class GraniteAdapter(ASRAdapter):
    name = "granite"

    def __init__(self, cfg: ASRModelConfig):
        self.model_path = Path(cfg.path or "")
        self.batch_size = max(1, int(cfg.batch_size or 4))
        self.device = cfg.device
        self.dtype_name = cfg.dtype
        self.attn_implementation = cfg.attn_implementation
        self.prefetch = bool(cfg.prefetch)
        self.cap, self.audio_budget = _resolve_batch_limits(cfg, self.name)
        self._processor = None
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None
        self._dtype = None
        self._device_str = "cpu"
        self._prompt = None

    def preflight(self) -> None:
        _require_path(self.model_path, self.name)
        _require_import("transformers", self.name)
        _require_import("torch", self.name)

    def transcribe_batch(self, segments: list[SegmentRecord]) -> list[ASRResult]:
        _, _, _, torch = self._load()
        by_id: dict[str, ASRResult] = {}
        ordered = sorted(segments, key=lambda s: (s.duration_sec, s.segment_id))
        chunks = list(_dynamic_batches(ordered, self.audio_budget, self.cap))

        def _load_wavs(chunk: list[SegmentRecord]):
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
                return list(pool.map(lambda s: _load_granite_tensor(Path(s.segment_audio_path), torch), chunk))

        produced = _prefetch_batches(chunks, _load_wavs, enabled=self.prefetch)
        for chunk, wavs, load_error in iter_progress(
            produced, desc="Granite ASR", total=len(chunks), unit="batch", enabled=self.progress_enabled
        ):
            if load_error is not None:
                texts, errors = self._transcribe_paths([str(s.segment_audio_path) for s in chunk])
            else:
                texts, errors = self._transcribe_loaded(wavs)
            for segment, text, error in zip(chunk, texts, errors):
                by_id[segment.segment_id] = ASRResult(
                    segment.segment_id,
                    self.name,
                    text,
                    0.0,
                    _language_code(segment.language),
                    error=error,
                )
        return [by_id.get(s.segment_id, ASRResult(s.segment_id, self.name, "", 0.0, error="NOT_RUN")) for s in segments]

    def _load(self):
        if self._model is None:
            _configure_quiet_transformers()
            import torch  # type: ignore
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # type: ignore

            _configure_torch_for_inference(torch)
            _clear_cuda(torch)
            self._torch = torch
            if _wants_cuda(self.device, torch):
                self._device_str = self.device if (self.device and self.device.startswith("cuda")) else "cuda"
            else:
                self._device_str = "cpu"
            default_dtype = torch.bfloat16 if self._device_str.startswith("cuda") else torch.float32
            granite_torch_dtype = _resolve_dtype(self.dtype_name, torch, default_dtype)
            self._processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=True)
            self._tokenizer = getattr(self._processor, "tokenizer", None)
            if self._tokenizer is not None and getattr(self._tokenizer, "pad_token_id", None) is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self._model = _load_model_with_attn(
                AutoModelForSpeechSeq2Seq,
                str(self.model_path),
                attn_implementation=self.attn_implementation,
                device_map=self._device_str if self._device_str.startswith("cuda") else None,
                torch_dtype=granite_torch_dtype,
                local_files_only=True,
                trust_remote_code=True,
            )
            self._model.eval()
            self._device = _model_device(self._model)
            self._dtype = _model_dtype(self._model)
            self._prompt = _granite_prompt(self._tokenizer)
        return self._processor, self._model, self._tokenizer, self._torch

    def _transcribe_loaded(self, wavs: list) -> tuple[list[str], list[str | None]]:
        torch = self._torch
        assert torch is not None
        try:
            return self._generate_from_wavs(wavs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(wavs) > 1:
                return _split_and_join(self._transcribe_loaded, wavs)
            return [""], ["CUDA_OUT_OF_MEMORY"]
        except Exception as exc:
            if len(wavs) > 1:
                return _split_and_join(self._transcribe_loaded, wavs)
            return [""], [repr(exc)]

    def _transcribe_paths(self, paths: list[str]) -> tuple[list[str], list[str | None]]:
        """Path-based slow path used only when prefetch loading raised for a batch."""
        torch = self._torch
        assert torch is not None
        try:
            wavs = [_load_granite_tensor(Path(p), torch) for p in paths]
        except Exception as exc:
            if len(paths) > 1:
                return _split_and_join(self._transcribe_paths, paths)
            return [""], [repr(exc)]
        return self._transcribe_loaded(wavs)

    def _generate_from_wavs(self, wavs: list) -> tuple[list[str], list[str | None]]:
        processor, model, tokenizer, torch = self._load()
        prompts = [self._prompt or _granite_prompt(tokenizer)] * len(wavs)
        inputs = processor(prompts, wavs, device=self._device_str, return_tensors="pt", padding=True)
        inputs = _move_inputs_to_device_dtype(inputs, self._device, self._dtype, torch)
        gen_kwargs = {"max_new_tokens": 400, "do_sample": False, "num_beams": 1}
        if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        with torch.inference_mode():
            output = model.generate(**inputs, **gen_kwargs)
        texts, errors = _decode_granite_generated_only(output, inputs, tokenizer, processor)
        if len(texts) != len(wavs):
            return texts[: len(wavs)] + [""] * max(0, len(wavs) - len(texts)), ["BAD_OUTPUT_LENGTH"] * len(wavs)
        return texts, errors


class DataParallelAdapter(ASRAdapter):
    """Run one model replica per visible GPU over disjoint segment shards.

    Off by default (config-gated). Only engaged when ``data_parallel`` is set and
    more than one CUDA device is visible. Each child adapter is pinned to a single
    ``cuda:N`` device, so all existing per-adapter logic (dynamic batching, prefetch,
    CUDA-OOM halving) is reused unchanged. Results are merged by segment id, so the
    final per-segment output is independent of how shards were assigned.
    """

    def __init__(self, name: str, children: list[ASRAdapter]):
        self.name = name
        self._children = children

    def preflight(self) -> None:
        for child in self._children:
            child.preflight()

    def transcribe_batch(self, segments: list[SegmentRecord]) -> list[ASRResult]:
        n = len(self._children)
        if n <= 1:
            return self._children[0].transcribe_batch(segments)
        shards = [segments[i::n] for i in range(n)]  # round-robin balances duration mix
        merged: dict[str, ASRResult] = {}
        lock = threading.Lock()
        errors: list[BaseException] = []

        def _work(child: ASRAdapter, shard: list[SegmentRecord]) -> None:
            if not shard:
                return
            try:
                child.progress_enabled = False
                results = child.transcribe_batch(shard)
            except BaseException as exc:  # noqa: BLE001 - surfaced after join
                with lock:
                    errors.append(exc)
                return
            with lock:
                for result in results:
                    merged[result.segment_id] = result

        threads = [threading.Thread(target=_work, args=(c, s), name=f"asr-dp-{i}") for i, (c, s) in enumerate(zip(self._children, shards))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return [merged.get(s.segment_id, ASRResult(s.segment_id, self.name, "", 0.0, error="NOT_RUN")) for s in segments]

    def close(self) -> None:
        for child in self._children:
            try:
                child.close()
            except Exception:
                pass


def _visible_gpu_count() -> int:
    torch = _optional_import("torch")
    if torch is None:
        return 0
    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _build_single_adapter(name: str, model_cfg: ASRModelConfig) -> ASRAdapter:
    if name == "whisper":
        return WhisperAdapter(model_cfg.path or "", model_cfg.batch_size, model_cfg.device)
    if name == "qwen":
        return QwenAdapter(model_cfg)
    if name == "cohere":
        return CohereAdapter(model_cfg)
    if name == "granite":
        return GraniteAdapter(model_cfg)
    if name == "parakeet":
        return ParakeetAdapter(model_cfg)
    raise ValueError(f"Unsupported ASR model: {name}")


def build_enabled_adapters(config: PipelineConfig) -> list[ASRAdapter]:
    adapters: list[ASRAdapter] = []
    for name, model_cfg in config.asr_models.items():
        if not model_cfg.enabled:
            continue
        # Optional, config-gated multi-GPU data parallelism for the seq2seq models.
        if getattr(model_cfg, "data_parallel", False) and name in ("qwen", "cohere", "granite", "parakeet"):
            gpu_count = _visible_gpu_count()
            if gpu_count > 1:
                children = [
                    _build_single_adapter(name, dataclasses.replace(model_cfg, device=f"cuda:{i}"))
                    for i in range(gpu_count)
                ]
                adapters.append(DataParallelAdapter(name, children))
                continue
        adapters.append(_build_single_adapter(name, model_cfg))
    return adapters


def preflight_adapters(adapters: Iterable[ASRAdapter]) -> None:
    for adapter in adapters:
        adapter.preflight()


def _require_path(path: Path, name: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{name} model path does not exist: {path}")


def _require_qwen_model_config(path: Path) -> None:
    config_path = path / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"qwen model config missing: {config_path}")
    try:
        payload = json.loads(config_path.read_text())
    except Exception as exc:
        raise RuntimeError(f"qwen model config is not readable JSON: {config_path}: {exc}") from exc
    if "thinker_config" not in payload:
        raise RuntimeError(
            "qwen model config.json is missing thinker_config. "
            "The local Qwen3-ASR model snapshot appears stale or incomplete; "
            "refresh the qwen3-asr-1.7b model directory, then rerun this stage."
        )


def _require_qwen_runtime_versions() -> None:
    transformers_version = _package_version("transformers")
    if transformers_version != QWEN_REQUIRED_TRANSFORMERS:
        raise RuntimeError(
            "qwen ASR must run with the qwen-asr supported runtime: "
            f"transformers=={QWEN_REQUIRED_TRANSFORMERS}. "
            f"Found transformers=={transformers_version}. "
            "Install the same Qwen runtime used by the training notebook with: "
            f"python3 -m pip install --force-reinstall 'qwen-asr==0.0.6' "
            f"'transformers=={QWEN_REQUIRED_TRANSFORMERS}' 'accelerate==1.12.0'"
        )


def _require_parakeet_support() -> None:
    """Fail fast when transformers predates ParakeetForTDT (added in 5.9.0).

    The main venv is pinned to 4.57.6 by qwen, so hitting this almost always means
    the asr_parakeet stage fell back to the main venv instead of its own.

    This resolves the concrete `ParakeetForTDT` class rather than checking
    `hasattr(transformers, ...)`. transformers exposes its public API lazily, so
    the attribute is present on the module long before anything confirms the
    class can actually be constructed -- a hasattr check passes on a venv whose
    torch is too old and defers the failure to mid-run model load.
    """
    import torch  # type: ignore
    import transformers  # type: ignore

    _ensure_torch_fp8_dtype_shim(torch)
    try:
        from transformers import ParakeetForTDT  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "parakeet requires transformers>=5.9.0 with a compatible torch; importing "
            f"ParakeetForTDT failed under transformers=={_package_version('transformers')}, "
            f"torch=={getattr(torch, '__version__', '?')} ({type(exc).__name__}: {exc}). "
            "The asr_parakeet stage must run in the dedicated parakeet venv -- see "
            "ops/setup_env.sh."
        ) from exc


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"required package is not installed: {distribution}") from exc


def _require_import(module: str, name: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(f"{name} model enabled but Python module '{module}' is not installed")


def _optional_import(module: str):
    if importlib.util.find_spec(module) is None:
        return None
    return importlib.import_module(module)


def _configure_quiet_transformers() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore")
    try:
        from transformers import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()
    except Exception:
        pass


def _ensure_torch_fp8_dtype_shim(torch) -> None:
    """Alias FP8/FP4 dtypes that transformers 5.x references at import but torch 2.5.1 lacks.

    transformers >= 5.x imports ``torch.float8_e8m0fnu`` (and other Microscaling MX dtypes)
    as module-level constants in ``integrations/finegrained_fp8.py``; that dtype was only
    added in torch 2.7. The cohere venv pins torch 2.5.1 (cu121) to match the shared
    cuDNN/NVIDIA stack, so importing ``CohereAsrForConditionalGeneration`` would raise
    ``AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'``.

    The Cohere ASR model loads in bf16/float32 and never uses MXFP8, so aliasing the missing
    dtype to an existing one lets the import succeed without affecting any real numerics. This
    is a no-op on torch >= 2.7 (the attributes already exist).
    """
    for name in ("float8_e8m0fnu", "float8_e4m3fn", "float8_e5m2", "float4_e2m1fn_x2"):
        if not hasattr(torch, name):
            try:
                setattr(torch, name, torch.float32)
            except Exception:
                pass


def _configure_torch_for_inference(torch) -> None:
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def _clear_cuda(torch) -> None:
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _release_model_memory(torch) -> None:
    gc.collect()
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def _wants_cuda(device: str, torch) -> bool:
    if device and device != "auto":
        return device.startswith("cuda")
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_torch_device(device: str, torch, prefer_colon: bool) -> str:
    if device and device != "auto":
        if device.startswith("cuda") and not prefer_colon:
            return "cuda"
        return device
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return "cuda:0" if prefer_colon else "cuda"
        except Exception:
            pass
    return "cpu"


def _set_pad_token_to_eos(model) -> None:
    try:
        if hasattr(model, "processor"):
            tok = model.processor.tokenizer
            tok.pad_token_id = tok.eos_token_id
        elif hasattr(model, "tokenizer"):
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    except Exception:
        pass


def _chunks(items: list[SegmentRecord] | list[str], batch_size: int):
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start : start + max(1, batch_size)]


def _resolve_batch_limits(cfg: ASRModelConfig, model_name: str) -> tuple[int, float | None]:
    """Return (max_count, audio_sec_budget) for dynamic batching.

    `max_batch_size` (if set) caps the count; otherwise `batch_size` is the cap.
    `batch_audio_sec_budget` (if set) caps summed audio seconds; otherwise the
    per-model default budget applies. Either limit closes a batch.
    """
    default_count, default_budget = _DEFAULT_BATCH_LIMITS.get(model_name, (4, None))
    if cfg.max_batch_size and int(cfg.max_batch_size) > 0:
        cap = int(cfg.max_batch_size)
    elif cfg.batch_size and int(cfg.batch_size) > 0:
        cap = int(cfg.batch_size)
    else:
        cap = default_count
    cap = max(1, cap)
    if cfg.batch_audio_sec_budget is not None and float(cfg.batch_audio_sec_budget) > 0:
        budget: float | None = float(cfg.batch_audio_sec_budget)
    else:
        budget = default_budget
    return cap, budget


def _dynamic_batches(
    segments: list[SegmentRecord],
    audio_sec_budget: float | None,
    max_count: int | None,
) -> Iterator[list[SegmentRecord]]:
    """Pack duration-sorted segments into batches bounded by count and audio seconds.

    The input is assumed sorted by duration. A batch is closed once adding the
    next segment would exceed the count cap or the audio-seconds budget; a single
    oversized segment still forms its own batch so nothing is dropped. Output is a
    partition of the input preserving order, so per-segment results are unaffected.
    """
    cap = max(1, int(max_count)) if max_count else None
    budget = float(audio_sec_budget) if (audio_sec_budget and audio_sec_budget > 0) else None
    batch: list[SegmentRecord] = []
    total = 0.0
    for seg in segments:
        dur = max(0.0, float(getattr(seg, "duration_sec", 0.0) or 0.0))
        if batch and (
            (cap is not None and len(batch) >= cap)
            or (budget is not None and total + dur > budget)
        ):
            yield batch
            batch = []
            total = 0.0
        batch.append(seg)
        total += dur
    if batch:
        yield batch


def _prefetch_batches(
    chunks: list[list[SegmentRecord]],
    loader: Callable[[list[SegmentRecord]], Any],
    enabled: bool = True,
    depth: int = _PREFETCH_DEPTH,
) -> Iterator[tuple[list[SegmentRecord], Any, Exception | None]]:
    """Yield (chunk, payload, error) overlapping CPU loading with GPU compute.

    A bounded background thread loads upcoming batches (queue depth `depth`) while
    the caller runs `generate` on the current one. Ordering is preserved exactly.
    Per-chunk load failures are surfaced as the `error` element so the caller can
    fall back to the path-based slow path without aborting the whole run.
    """
    if not enabled or len(chunks) <= 1:
        for chunk in chunks:
            try:
                yield chunk, loader(chunk), None
            except Exception as exc:  # noqa: BLE001 - reported to caller
                yield chunk, None, exc
        return

    result_queue: "queue.Queue[tuple[list[SegmentRecord], Any, Exception | None] | object]" = queue.Queue(maxsize=max(1, depth))
    sentinel = object()

    def _produce() -> None:
        for chunk in chunks:
            try:
                payload = loader(chunk)
                result_queue.put((chunk, payload, None))
            except Exception as exc:  # noqa: BLE001 - reported to caller
                result_queue.put((chunk, None, exc))
        result_queue.put(sentinel)

    worker = threading.Thread(target=_produce, name="asr-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            item = result_queue.get()
            if item is sentinel:
                break
            yield item  # type: ignore[misc]
    finally:
        worker.join()


def _resolve_dtype(name: str | None, torch, default):
    """Map a config dtype string to a torch dtype; "auto"/unknown -> default."""
    key = (name or "auto").strip().lower()
    if key in ("", "auto"):
        return default
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(key, default)


def _resolve_device_map(device: str, torch):
    """device_map for HF: pin to an explicit cuda:N when given, else "auto"/None."""
    if not _wants_cuda(device, torch):
        return None
    if device and device.startswith("cuda:"):
        return {"": device}
    return "auto"


def _load_model_with_attn(model_cls, model_path: str, attn_implementation: str | None = None, **kwargs):
    """Load a HF model, preferring SDPA attention (faster, numerically equivalent).

    SDPA does not change outputs versus eager attention, so this is safe under the
    no-quality-drop guardrail. If the chosen implementation is unsupported by the
    model/transformers version, fall back to the default attention silently.
    """
    chosen = attn_implementation or "sdpa"
    try:
        return model_cls.from_pretrained(model_path, attn_implementation=chosen, **kwargs)
    except Exception:
        return model_cls.from_pretrained(model_path, **kwargs)


def _segments_by_language(segments: list[SegmentRecord]) -> dict[str, list[SegmentRecord]]:
    grouped: dict[str, list[SegmentRecord]] = {}
    for segment in segments:
        grouped.setdefault(_language_code(segment.language), []).append(segment)
    return grouped


def _language_code(language: str | None) -> str:
    value = (language or "en").strip().lower()
    if value.startswith("spanish") or value.startswith("es"):
        return "es"
    return "en"


def _language_name(language: str | None) -> str:
    return "Spanish" if _language_code(language) == "es" else "English"


def _texts_and_errors(raw, expected_len: int) -> tuple[list[str], list[str | None]]:
    if not isinstance(raw, list):
        raw = [raw]
    texts = [_extract_text(item) for item in raw]
    if len(texts) != expected_len:
        texts = texts[:expected_len] + [""] * max(0, expected_len - len(texts))
        return texts, ["BAD_OUTPUT_LENGTH"] * expected_len
    return texts, [None] * expected_len


def _pad_texts(texts: list[str], expected_len: int) -> tuple[list[str], list[str | None]]:
    if len(texts) != expected_len:
        return texts[:expected_len] + [""] * max(0, expected_len - len(texts)), ["BAD_OUTPUT_LENGTH"] * expected_len
    return texts, [None] * expected_len


def _split_and_join(func, paths: list[str]) -> tuple[list[str], list[str | None]]:
    midpoint = len(paths) // 2
    left_texts, left_errors = func(paths[:midpoint])
    right_texts, right_errors = func(paths[midpoint:])
    return left_texts + right_texts, left_errors + right_errors


def _extract_text(item) -> str:
    if item is None:
        return ""
    text = getattr(item, "text", None)
    if text is not None:
        return str(text).strip()
    if isinstance(item, dict):
        for key in ["text", "transcript", "prediction"]:
            if item.get(key) is not None:
                return str(item[key]).strip()
    return str(item).strip()


def _load_mono_samples(path: Path) -> list[float]:
    from .audio import read_wav

    buf = read_wav(path)
    if buf.channels == 1:
        return [_clean_sample(row[0]) for row in buf.frames]
    return [_clean_sample(sum(row) / len(row)) for row in buf.frames]


def _load_cohere_audio(path: str | Path) -> list[float]:
    min_samples = 8000
    samples = _load_mono_samples(Path(path))
    if not samples:
        return [0.0] * min_samples
    if len(samples) < min_samples:
        samples = samples + [0.0] * (min_samples - len(samples))
    return samples


def _load_granite_tensor(path: Path, torch):
    samples = _load_mono_samples(path)
    if not samples:
        samples = [0.0] * 8000
    tensor = torch.tensor(samples, dtype=torch.float32).reshape(1, -1)
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _clean_sample(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def _model_device(model):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def _model_dtype(model):
    try:
        return model.dtype
    except Exception:
        return next(model.parameters()).dtype


def _clone_for_decode(value, torch):
    try:
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
    except Exception:
        pass
    return value


def _move_inputs_to_device_dtype(inputs, device, dtype, torch):
    if hasattr(inputs, "to"):
        try:
            moved = inputs.to(device, dtype=dtype)
            if moved is not None:
                return moved
            return inputs
        except TypeError:
            moved = inputs.to(device)
            return moved if moved is not None else inputs
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
    return inputs


def _decode_cohere(processor, outputs, audio_chunk_index, language: str) -> list[str]:
    try:
        decoded = processor.decode(outputs, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language=language)
    except TypeError:
        decoded = processor.decode(outputs, skip_special_tokens=True)
    if isinstance(decoded, str):
        return [decoded.strip()]
    if isinstance(decoded, list):
        return [str(x).strip() for x in decoded]
    return [str(decoded).strip()]


def _granite_prompt(tokenizer) -> str:
    user_prompt = "<|audio|>can you transcribe the speech into a written format?"
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True)
    return user_prompt


def _decode_granite_generated_only(model_outputs, model_inputs, tokenizer, processor) -> tuple[list[str], list[str | None]]:
    if "input_ids" not in model_inputs:
        raise RuntimeError("Granite model_inputs missing input_ids. Cannot safely remove prompt tokens.")
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    if model_outputs.shape[-1] <= num_input_tokens:
        return [""] * model_outputs.shape[0], ["NO_NEW_TOKENS_PROMPT_ONLY"] * model_outputs.shape[0]
    new_tokens = model_outputs[:, num_input_tokens:]
    decoder = tokenizer if tokenizer is not None else processor
    decoded = decoder.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
    texts = [_clean_text(x) for x in decoded]
    errors: list[str | None] = []
    for text in texts:
        low = text.lower()
        if low.startswith("user:") or "can you transcribe the speech" in low:
            errors.append("PROMPT_ECHO")
        else:
            errors.append(None)
    return texts, errors


def _clean_text(text: str) -> str:
    value = str(text)
    for bad in ["<|end_of_text|>", "<|endoftext|>", "<|end|>"]:
        value = value.replace(bad, "")
    return " ".join(value.strip().split())
