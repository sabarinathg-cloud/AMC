#!/usr/bin/env python3
"""Extract named entities per VAD segment from a consensus manifest parquet.

Each row of the consensus parquet is one VAD segment. This script runs the
pipeline's existing NER/PII detectors (amc_pipeline/pii_detection.py) over the
chosen transcript column and emits, for every segment, the list of named
entities found in it.

Outputs (next to --output, which is a stem):
  <output>.segments.jsonl   one JSON object per segment: {segment_id, ..., entities: [...]}
  <output>.entities.parquet flat/long table, one row per (segment, entity)
  <output>.summary.json     counts by entity type + run config

Examples
--------
# Quick smoke test on 500 rows with spaCy (true general-purpose NER):
python ops/ner_per_segment.py \
    --input /mnt/amc-data/.../consensus_segments_with_speaker_clusters.parquet \
    --output /mnt/amc-data/.../ner/segment_entities \
    --detectors spacy --limit 500

# Full run with spaCy + GLiNER + piiranha + regex:
python ops/ner_per_segment.py \
    --input .../consensus_segments_with_speaker_clusters.parquet \
    --output .../ner/segment_entities \
    --detectors spacy,gliner,piiranha,regex
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Allow `python ops/ner_per_segment.py` from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amc_pipeline.config import PipelineConfig  # noqa: E402
from amc_pipeline.models import PIISpan  # noqa: E402
from amc_pipeline.pii_detection import (  # noqa: E402
    PIIDetector,
    build_enabled_detectors,
    preflight_detectors,
    resolve_overlaps,
)

# Columns carried through to the per-segment output (when present) so the
# entities can be tied back to the call / speaker without a rejoin.
_ID_COLUMNS = [
    "segment_id",
    "call_id",
    "channel",
    "channel_idx",
    "start_sec",
    "end_sec",
    "duration",
    "speaker_cluster_id",
    "speaker_cluster",
    "split_group_id",
]

# Preferred transcript columns, best first. The chosen one is the text we run
# NER over; we fall back down the list when a column is missing/empty.
_TEXT_FALLBACKS = [
    "consensus_text",
    "majority_text",
    "soft_consensus_text",
    "whisper_transcript",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Consensus segments parquet (one row per VAD segment).")
    parser.add_argument("--output", type=Path, required=True, help="Output path stem; .segments.jsonl / .entities.parquet / .summary.json are written.")
    parser.add_argument(
        "--detectors",
        default="spacy",
        help="Comma-separated detectors to enable: spacy,gliner,piiranha,regex,rule_name. Default 'spacy' (general NER).",
    )
    parser.add_argument("--text-column", default=None, help=f"Transcript column to analyse. Default: first present of {_TEXT_FALLBACKS}.")
    parser.add_argument("--batch-size", type=int, default=256, help="Texts per detector forward pass.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (smoke testing).")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Drop entities below this confidence.")
    parser.add_argument("--no-parquet", action="store_true", help="Skip the long entities.parquet (write JSONL + summary only).")
    return parser.parse_args()


def _build_detectors(detector_names: str) -> list[PIIDetector]:
    wanted = {x.strip() for x in detector_names.split(",") if x.strip()}
    cfg = PipelineConfig(output_root=Path("."))
    unknown = wanted - set(cfg.pii_models)
    if unknown:
        raise SystemExit(f"unknown detector(s): {sorted(unknown)}; choose from {sorted(cfg.pii_models)}")
    for name, model_cfg in cfg.pii_models.items():
        model_cfg.enabled = name in wanted
    detectors = build_enabled_detectors(cfg)
    if not detectors:
        raise SystemExit("no detectors enabled")
    preflight_detectors(detectors)
    return detectors


def _resolve_text_column(columns: list[str], requested: str | None) -> str:
    if requested is not None:
        if requested not in columns:
            raise SystemExit(f"--text-column '{requested}' not in parquet columns")
        return requested
    for candidate in _TEXT_FALLBACKS:
        if candidate in columns:
            return candidate
    raise SystemExit(f"no transcript column found; tried {_TEXT_FALLBACKS}")


def _merge_detector_spans(per_detector: list[list[PIISpan]], min_confidence: float) -> list[PIISpan]:
    """Combine spans from every detector for one segment into a single,
    overlap-resolved list ordered by character offset."""
    combined: list[PIISpan] = []
    for spans in per_detector:
        combined.extend(spans)
    merged = resolve_overlaps(combined)
    if min_confidence > 0.0:
        merged = [s for s in merged if s.confidence >= min_confidence]
    return merged


def _span_to_entity(span: PIISpan) -> dict:
    return {
        "entity_type": span.entity_type,
        "text": span.text,
        "start_char": span.start_char,
        "end_char": span.end_char,
        "confidence": round(float(span.confidence), 4),
        "source": span.source,
    }


def main() -> int:
    args = parse_args()
    try:
        import pandas as pd  # noqa: E402  (deferred so --help works without pandas)
    except ImportError:
        raise SystemExit("pandas is required: pip install pandas pyarrow")

    if not args.input.exists():
        raise SystemExit(f"input parquet not found: {args.input}")

    df = pd.read_parquet(args.input)
    if args.limit is not None:
        df = df.head(args.limit)
    columns = df.columns.tolist()
    text_column = _resolve_text_column(columns, args.text_column)
    id_columns = [c for c in _ID_COLUMNS if c in columns]
    detectors = _build_detectors(args.detectors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    segments_path = args.output.with_suffix(".segments.jsonl")
    entities_path = args.output.with_suffix(".entities.parquet")
    summary_path = args.output.with_suffix(".summary.json")

    # Build the list of texts (with per-segment fallback when the chosen column is empty).
    text_series = df[text_column].astype("string")
    for fallback in _TEXT_FALLBACKS:
        if fallback != text_column and fallback in df.columns:
            text_series = text_series.fillna(df[fallback].astype("string"))
    texts = [("" if t is pd.NA or t is None else str(t)) for t in text_series.tolist()]

    type_counter: Counter[str] = Counter()
    segments_with_entities = 0
    total_entities = 0
    entity_rows: list[dict] = []

    id_records = df[id_columns].to_dict("records") if id_columns else [{} for _ in range(len(df))]

    with segments_path.open("w", encoding="utf-8") as seg_handle:
        for start in range(0, len(texts), args.batch_size):
            chunk = texts[start : start + args.batch_size]
            per_detector_results = [detector.detect_batch(chunk) for detector in detectors]
            for offset in range(len(chunk)):
                merged = _merge_detector_spans([res[offset] for res in per_detector_results], args.min_confidence)
                entities = [_span_to_entity(s) for s in merged]
                record = dict(id_records[start + offset])
                record["text_column"] = text_column
                record["text"] = chunk[offset]
                record["entity_count"] = len(entities)
                record["entities"] = entities
                seg_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

                if entities:
                    segments_with_entities += 1
                total_entities += len(entities)
                for ent in entities:
                    type_counter[ent["entity_type"]] += 1
                    if not args.no_parquet:
                        flat = {k: record.get(k) for k in id_columns}
                        flat.update(ent)
                        entity_rows.append(flat)

    if not args.no_parquet:
        pd.DataFrame(entity_rows).to_parquet(entities_path, index=False)

    summary = {
        "input": str(args.input),
        "rows_processed": len(texts),
        "text_column": text_column,
        "detectors": [d.name for d in detectors],
        "min_confidence": args.min_confidence,
        "segments_with_entities": segments_with_entities,
        "total_entities": total_entities,
        "entities_by_type": dict(sorted(type_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "outputs": {
            "segments_jsonl": str(segments_path),
            "entities_parquet": None if args.no_parquet else str(entities_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
