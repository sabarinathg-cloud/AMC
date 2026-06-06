#!/usr/bin/env bash
# Merge per-shard pipeline outputs into a single dataset.
#
#   - rsync redacted call audio + segment clips from every $RUN_ROOT/outputs/shard-*
#     into $FINAL_OUT (preserving the <year>/... layout)
#   - assemble a PARQUET DATASET: one part file per shard at
#       $FINAL_OUT/manifests/parquet/part-shard-<N>.parquet
#     The parquet/ subdir holds ONLY parquet, so readers can target the directory directly
#     (no giant single-file concat):
#       pyarrow.dataset.dataset("$FINAL_OUT/manifests/parquet")
#       duckdb:  SELECT * FROM read_parquet('$FINAL_OUT/manifests/parquet/*.parquet')
#       pandas:  pd.read_parquet("$FINAL_OUT/manifests/parquet")
#       polars:  pl.scan_parquet("$FINAL_OUT/manifests/parquet/*.parquet")
#   - also concatenate manifests/all_segments.jsonl -> $FINAL_OUT/manifests/all_segments.jsonl
#     as a durable, stream-readable fallback (cheap to write, no parse).
#   - the heavy single-file CSV is OPT-IN (MERGE_CSV=1) -- at millions of rows a single CSV is
#     the bottleneck (multi-GB, slow parse, no column pruning). Prefer the parquet parts.
#
# Why a plain rsync/parts merge is safe:
#   * Sharding is by call-id hash, so each call lives on exactly one shard -> the per-shard
#     <year>/<call>/... and <year>/segments/<call>/... paths are DISJOINT across shards and
#     never collide when merged into one tree.
#   * Manifest paths (segment_audio_path_rel, ...) are relative to each shard's output_root and
#     mirror the same <year>/... layout, so they stay valid relative to $FINAL_OUT after merge.
#
# Idempotent: re-running rsyncs only changed files and rebuilds the merged manifests fresh.
#
# Usage (run on ANY one instance that can see the shared mount):
#   RUN_ROOT=/mnt/amc-data/amc-runs/smoke-20 \
#   FINAL_OUT=/mnt/amc-data/amc-redacted-final/smoke-20 \
#   bash ops/merge_shards.sh
#   # add MERGE_CSV=1 to also emit a single big all_segments.csv (slow at scale).
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required, e.g. /mnt/amc-data/amc-runs/smoke-20}"
FINAL_OUT="${FINAL_OUT:?FINAL_OUT is required, e.g. /mnt/amc-data/amc-redacted-final/smoke-20}"
MERGE_CSV="${MERGE_CSV:-0}"

# Use the main venv python (has pandas/pyarrow) for the JSONL->parquet fallback; degrade to
# python3 / skip conversion if neither is available.
VENV_ROOT="${AMC_VENV_ROOT:-/mnt/amc-data/venvs}"
MAIN_PY="$VENV_ROOT/main/bin/python"
[[ -x "$MAIN_PY" ]] || MAIN_PY="$(command -v python3 || true)"

shopt -s nullglob
shards=("$RUN_ROOT"/outputs/shard-*)
if [[ ${#shards[@]} -eq 0 ]]; then
  echo "ERROR: no shards found under $RUN_ROOT/outputs" >&2
  exit 1
fi

manifest_dir="$FINAL_OUT/manifests"
parquet_dir="$manifest_dir/parquet"
mkdir -p "$FINAL_OUT" "$manifest_dir" "$parquet_dir"
echo "Merging ${#shards[@]} shard(s): $RUN_ROOT/outputs -> $FINAL_OUT"

# 1) Audio: redacted calls + segment clips. Drop per-shard bookkeeping (.pii_pipeline) and the
#    per-shard manifests dirs (top-level and per-year) -- we rebuild merged manifests below.
for d in "${shards[@]}"; do
  echo "  rsync $(basename "$d")"
  rsync -a \
    --exclude '.pii_pipeline/' \
    --exclude 'manifests/' \
    "$d"/ "$FINAL_OUT"/
done

# 2) Parquet dataset: one part per shard. Prefer the part the pipeline already wrote; otherwise
#    convert that shard's JSONL once (per-shard row count is bounded, so this is memory-safe).
rm -f "$parquet_dir"/part-shard-*.parquet
nparts=0
for d in "${shards[@]}"; do
  idx="$(basename "$d")"; idx="${idx#shard-}"
  src_parq="$d/manifests/all_segments.parquet"
  src_jsonl="$d/manifests/all_segments.jsonl"
  dst="$parquet_dir/part-shard-$idx.parquet"
  if [[ -f "$src_parq" ]]; then
    cp -f "$src_parq" "$dst"
    nparts=$((nparts + 1))
  elif [[ -s "$src_jsonl" && -n "$MAIN_PY" ]]; then
    if "$MAIN_PY" - "$src_jsonl" "$dst" <<'PY'
import sys
import pandas as pd
src, dst = sys.argv[1], sys.argv[2]
# Match the pipeline's all-string Parquet schema so this fallback part unions cleanly with the
# parts the manifest stage wrote (the pipeline emits every column as a string).
df = pd.read_json(src, lines=True).astype("string")
try:
    df.to_parquet(dst, index=False, compression="zstd")
except Exception:
    df.to_parquet(dst, index=False)
PY
    then
      nparts=$((nparts + 1))
    else
      echo "  WARN: failed to convert $src_jsonl -> parquet" >&2
    fi
  fi
done

# 3) Durable, stream-readable JSONL concatenation (safety net; cheap to write, no parse).
merge_jsonl="$manifest_dir/all_segments.jsonl"
: > "$merge_jsonl"
for d in "${shards[@]}"; do
  j="$d/manifests/all_segments.jsonl"
  [[ -f "$j" ]] && cat "$j" >> "$merge_jsonl"
done

# 4) Optional single big CSV (opt-in; slow at scale). Header from first shard, bodies appended.
if [[ "$MERGE_CSV" == "1" ]]; then
  merge_csv="$manifest_dir/all_segments.csv"
  : > "$merge_csv"
  csv_header_written=0
  for d in "${shards[@]}"; do
    c="$d/manifests/all_segments.csv"
    [[ -f "$c" ]] || continue
    if [[ "$csv_header_written" -eq 0 ]]; then
      cat "$c" >> "$merge_csv"
      csv_header_written=1
    else
      tail -n +2 "$c" >> "$merge_csv"
    fi
  done
fi

# 5) Summary / sanity counts.
calls=$(find "$FINAL_OUT" -type f -path '*/audio.*' -not -path '*/segments/*' 2>/dev/null | wc -l | tr -d ' ')
clips=$(find "$FINAL_OUT" -type f -path '*/segments/*' 2>/dev/null | wc -l | tr -d ' ')
rows=$(wc -l < "$merge_jsonl" 2>/dev/null | tr -d ' ')

echo "Done."
echo "  redacted calls   : $calls"
echo "  segment clips    : $clips"
echo "  manifest rows    : $rows"
echo "  parquet parts    : $nparts  ($parquet_dir/part-shard-*.parquet)"
echo "  durable jsonl    : $merge_jsonl"
[[ "$MERGE_CSV" == "1" ]] && echo "  merged csv       : $manifest_dir/all_segments.csv (opt-in)"
echo
echo "Read the dataset as one table, e.g.:"
echo "  duckdb -c \"SELECT count(*) FROM read_parquet('$parquet_dir/*.parquet')\""
echo "  python -c \"import pandas as pd; print(pd.read_parquet('$parquet_dir').shape)\""
