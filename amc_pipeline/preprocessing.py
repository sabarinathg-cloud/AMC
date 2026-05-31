from __future__ import annotations

from pathlib import Path

from .audio import decode_to_wav, extract_channel, read_wav, temp_wav_path, write_mono_segment
from .config import PipelineConfig
from .models import AudioFileRecord, SegmentRecord
from .segmentation import split_chunks_on_pauses, vad_intervals


def preprocess_file(record: AudioFileRecord, config: PipelineConfig) -> list[SegmentRecord]:
    config.ensure_output_dirs()
    working_path = record.source_path
    if record.source_path.suffix.lower() != ".wav":
        working_path = decode_to_wav(record.source_path, config.cache_dir / "preprocess" / record.file_id / "decoded.wav", config.audio.target_sample_rate)
    buffer = read_wav(working_path)
    segments: list[SegmentRecord] = []
    for channel in range(buffer.channels):
        samples = extract_channel(buffer, channel)
        intervals = vad_intervals(
            samples,
            buffer.sample_rate,
            backend=config.audio.vad_backend,
            threshold_ratio=config.audio.vad_threshold_ratio,
            window_ms=config.audio.vad_window_ms,
            merge_gap_sec=config.audio.merge_gap_sec,
            silero_repo_or_dir=config.audio.silero_repo_or_dir,
            merge=False,
        )
        if not intervals and samples:
            intervals = [(0.0, len(samples) / buffer.sample_rate)]
        chunks = split_chunks_on_pauses(
            intervals,
            len(samples) / buffer.sample_rate,
            target_sec=config.audio.target_segment_sec,
            max_sec=config.audio.max_segment_sec,
            min_sec=config.audio.min_segment_sec,
            min_split_gap_sec=config.audio.min_split_gap_sec,
            merge_gap_sec=config.audio.merge_gap_sec,
            max_recursion=config.audio.max_split_recursion,
        )
        for idx, (start_sec, end_sec) in enumerate(chunks):
            start_sample = max(0, int(round(start_sec * buffer.sample_rate)))
            end_sample = min(len(samples), int(round(end_sec * buffer.sample_rate)))
            if end_sample <= start_sample:
                continue
            segment_id = f"{record.year}_{record.call_id}_ch{channel}_{idx:05d}"
            out_path = config.output_root / record.year / "segments" / record.call_id / f"{segment_id}.wav"
            segment_samples = samples[start_sample:end_sample]
            write_mono_segment(out_path, segment_samples, buffer.sample_rate)
            segments.append(
                SegmentRecord(
                    segment_id=segment_id,
                    file_id=record.file_id,
                    call_id=record.call_id,
                    year=record.year,
                    channel=channel,
                    source_path=record.source_path,
                    segment_audio_path=out_path.resolve(),
                    start_sec=start_sample / buffer.sample_rate,
                    end_sec=end_sample / buffer.sample_rate,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    duration_sec=(end_sample - start_sample) / buffer.sample_rate,
                    duration_samples=end_sample - start_sample,
                    sample_rate=buffer.sample_rate,
                )
            )
    return segments
