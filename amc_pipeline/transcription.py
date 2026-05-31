from __future__ import annotations

import importlib
import importlib.util
import gc
import inspect
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from .config import PipelineConfig
from .models import ASRResult, SegmentRecord
from .progress import iter_progress


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

    def __init__(self, model_path: str, batch_size: int = 0, device: str = "auto"):
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.device = device
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
            fw_compute_type = "float16" if fw_device == "cuda" else "int8"
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
        results: list[ASRResult] = []
        for segment in iter_progress(segments, desc="Whisper ASR", total=len(segments), unit="segment", enabled=self.progress_enabled):
            try:
                out, info = model.transcribe(
                    str(segment.segment_audio_path),
                    language=None,
                    beam_size=5,
                    batch_size=self._internal_batch_size,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                )
                text = " ".join(x.text.strip() for x in out).strip()
                language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
                results.append(
                    ASRResult(
                        segment.segment_id,
                        self.name,
                        text,
                        language_probability,
                        getattr(info, "language", None),
                        language_probability,
                        raw={"duration_after_vad": getattr(info, "duration_after_vad", None)},
                    )
                )
            except Exception as exc:
                results.append(ASRResult(segment.segment_id, self.name, "", 0.0, error=repr(exc)))
        return results


class QwenAdapter(ASRAdapter):
    name = "qwen"

    def __init__(self, model_path: str, batch_size: int = 8, device: str = "auto"):
        self.model_path = Path(model_path)
        self.batch_size = max(1, int(batch_size or 8))
        self.device = device
        self._model = None
        self._torch = None

    def preflight(self) -> None:
        _require_path(self.model_path, self.name)
        _require_import("qwen_asr", self.name)
        _require_import("torch", self.name)

    def _load(self):
        if self._model is None:
            _configure_quiet_transformers()
            import torch  # type: ignore

            _patch_transformers_check_model_inputs_for_qwen()
            from qwen_asr import Qwen3ASRModel  # type: ignore

            _configure_torch_for_inference(torch)
            _clear_cuda(torch)
            qwen_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            qwen_device = _resolve_torch_device(self.device, torch, prefer_colon=True)
            self._model = Qwen3ASRModel.from_pretrained(
                str(self.model_path),
                dtype=qwen_dtype,
                device_map=qwen_device,
                max_inference_batch_size=self.batch_size,
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
            chunks = list(_chunks(ordered, self.batch_size))
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

    def __init__(self, model_path: str, batch_size: int = 4, device: str = "auto"):
        self.model_path = Path(model_path)
        self.batch_size = max(1, int(batch_size or 4))
        self.device = device
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
            chunks = list(_chunks(group, self.batch_size))
            for chunk in iter_progress(chunks, desc=f"Cohere ASR {language}", total=len(chunks), unit="batch", enabled=self.progress_enabled):
                texts, errors = self._transcribe_paths([str(s.segment_audio_path) for s in chunk], _language_code(language))
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

    def _load(self):
        if self._model is None:
            _configure_quiet_transformers()
            import torch  # type: ignore
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration  # type: ignore

            _configure_torch_for_inference(torch)
            _clear_cuda(torch)
            self._torch = torch
            self._processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
            self._model = CohereAsrForConditionalGeneration.from_pretrained(
                str(self.model_path),
                device_map="auto" if _wants_cuda(self.device, torch) else None,
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            self._model.eval()
            self._device = _model_device(self._model)
            self._dtype = _model_dtype(self._model)
        return self._processor, self._model, self._torch

    def _transcribe_paths(self, paths: list[str], language: str) -> tuple[list[str], list[str | None]]:
        torch = self._torch
        assert torch is not None
        try:
            return self._transcribe_paths_no_fallback(paths, language)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(paths) > 1:
                return _split_and_join(lambda part: self._transcribe_paths(part, language), paths)
            return [""], ["CUDA_OUT_OF_MEMORY"]
        except Exception as exc:
            if len(paths) > 1:
                return _split_and_join(lambda part: self._transcribe_paths(part, language), paths)
            text, err = self._transcribe_single(paths[0], language)
            return [text], [err if err is not None else None]

    def _transcribe_paths_no_fallback(self, paths: list[str], language: str) -> tuple[list[str], list[str | None]]:
        processor, model, torch = self._load()
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            audios = list(pool.map(_load_cohere_audio, paths))
        inputs = processor(
            audios,
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
        return _pad_texts(texts, len(paths))

    def _transcribe_single(self, path: str, language: str) -> tuple[str, str | None]:
        processor, model, torch = self._load()
        try:
            audio = _load_cohere_audio(path)
            inputs = processor(
                audio,
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
            return (texts[0] if texts else ""), None
        except Exception as exc:
            return "", repr(exc)


class GraniteAdapter(ASRAdapter):
    name = "granite"

    def __init__(self, model_path: str, batch_size: int = 4, device: str = "auto"):
        self.model_path = Path(model_path)
        self.batch_size = max(1, int(batch_size or 4))
        self.device = device
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
        self._load()
        by_id: dict[str, ASRResult] = {}
        ordered = sorted(segments, key=lambda s: (s.duration_sec, s.segment_id))
        chunks = list(_chunks(ordered, self.batch_size))
        for chunk in iter_progress(chunks, desc="Granite ASR", total=len(chunks), unit="batch", enabled=self.progress_enabled):
            texts, errors = self._transcribe_paths([str(s.segment_audio_path) for s in chunk])
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
            self._device_str = "cuda" if _wants_cuda(self.device, torch) else "cpu"
            granite_torch_dtype = torch.bfloat16 if self._device_str == "cuda" else torch.float32
            self._processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=True)
            self._tokenizer = getattr(self._processor, "tokenizer", None)
            if self._tokenizer is not None and getattr(self._tokenizer, "pad_token_id", None) is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.model_path),
                device_map=self._device_str if self._device_str == "cuda" else None,
                torch_dtype=granite_torch_dtype,
                local_files_only=True,
                trust_remote_code=True,
            )
            self._model.eval()
            self._device = _model_device(self._model)
            self._dtype = _model_dtype(self._model)
            self._prompt = _granite_prompt(self._tokenizer)
        return self._processor, self._model, self._tokenizer, self._torch

    def _transcribe_paths(self, paths: list[str]) -> tuple[list[str], list[str | None]]:
        torch = self._torch
        assert torch is not None
        try:
            return self._transcribe_paths_no_fallback(paths)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(paths) > 1:
                return _split_and_join(self._transcribe_paths, paths)
            return [""], ["CUDA_OUT_OF_MEMORY"]
        except Exception as exc:
            if len(paths) > 1:
                return _split_and_join(self._transcribe_paths, paths)
            return [""], [repr(exc)]

    def _transcribe_paths_no_fallback(self, paths: list[str]) -> tuple[list[str], list[str | None]]:
        processor, model, tokenizer, torch = self._load()
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            wavs = list(pool.map(lambda p: _load_granite_tensor(Path(p), torch), paths))
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
        if len(texts) != len(paths):
            return texts[: len(paths)] + [""] * max(0, len(paths) - len(texts)), ["BAD_OUTPUT_LENGTH"] * len(paths)
        return texts, errors


def build_enabled_adapters(config: PipelineConfig) -> list[ASRAdapter]:
    adapters: list[ASRAdapter] = []
    for name, model_cfg in config.asr_models.items():
        if not model_cfg.enabled:
            continue
        if name == "whisper":
            adapters.append(WhisperAdapter(model_cfg.path or "", model_cfg.batch_size, model_cfg.device))
        elif name == "qwen":
            adapters.append(QwenAdapter(model_cfg.path or "", model_cfg.batch_size, model_cfg.device))
        elif name == "cohere":
            adapters.append(CohereAdapter(model_cfg.path or "", model_cfg.batch_size, model_cfg.device))
        elif name == "granite":
            adapters.append(GraniteAdapter(model_cfg.path or "", model_cfg.batch_size, model_cfg.device))
        else:
            raise ValueError(f"Unsupported ASR model: {name}")
    return adapters


def preflight_adapters(adapters: Iterable[ASRAdapter]) -> None:
    for adapter in adapters:
        adapter.preflight()


def _require_path(path: Path, name: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{name} model path does not exist: {path}")


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


def _patch_transformers_check_model_inputs_for_qwen() -> bool:
    try:
        from transformers.utils import generic  # type: ignore
    except Exception:
        return False
    original = getattr(generic, "check_model_inputs", None)
    if original is None or getattr(original, "_amc_qwen_compat", False):
        return False
    try:
        signature = inspect.signature(original)
        params = list(signature.parameters.values())
        needs_func = bool(params) and params[0].default is inspect.Signature.empty
    except Exception:
        needs_func = False
    if not needs_func:
        return False
    compat = _make_check_model_inputs_compat(original)
    setattr(generic, "check_model_inputs", compat)
    return True


def _make_check_model_inputs_compat(original):
    def compat(func=None, *args, **kwargs):
        if func is None:
            def decorator(real_func):
                return original(real_func, *args, **kwargs)

            return decorator
        return original(func, *args, **kwargs)

    compat._amc_qwen_compat = True  # type: ignore[attr-defined]
    return compat


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
