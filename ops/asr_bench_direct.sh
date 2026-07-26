#!/usr/bin/env bash
set -euo pipefail

# Run the ASR panel benchmark (ops/asr_model_bench.py) on ONE fleet instance via SSM.
#
# Answers the question "can we replace Whisper without losing accuracy?" using our
# own audio: it scores each variant against the rest of the consensus panel
# (qwen/cohere/granite), which is an independent reference for both the incumbent
# and the candidate.
#
# Variants (VARIANTS, comma-separated; default runs all four):
#   baseline        score the whisper transcripts already in the manifests (no GPU work)
#   whisper_greedy  whisper re-run with beam_size=1              (Track A, free speedup)
#   whisper_int8    whisper re-run with int8_float16 + greedy    (Track A, free speedup)
#   parakeet        Parakeet TDT 0.6b v3                          (Track B, the replacement)
#
# Usage:
#   RUN_ROOT=/mnt/amc-data/amc-runs/2022-full bash ops/asr_bench_direct.sh
#   RUN_ROOT=... LIMIT=5000 VARIANTS=parakeet bash ops/asr_bench_direct.sh

export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_TAG="${PROJECT_TAG:-amc-ec2-fleet}"
REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
VENV_ROOT="${AMC_VENV_ROOT:-/mnt/amc-data/venvs}"
SHARED_ROOT="${AMC_SHARED_ROOT:-/mnt/amc-data}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/amc-data/pipeline/models}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required (an existing run dir on the shared mount)}"
LIMIT="${LIMIT:-2000}"
SEED="${SEED:-1234}"
VARIANTS="${VARIANTS:-baseline,whisper_greedy,parakeet}"
TIMEOUT="${TIMEOUT:-7200}"
OUT_DIR="$RUN_ROOT/reports/asr_bench"

case "$RUN_ROOT" in "$SHARED_ROOT"/*) : ;; *) echo "ERROR: RUN_ROOT must be under $SHARED_ROOT" >&2; exit 2 ;; esac

TARGET="$(aws ssm describe-instance-information \
  --region "$AWS_REGION" \
  --filters "Key=tag:Project,Values=$PROJECT_TAG" \
  --query 'InstanceInformationList[].[InstanceId,PingStatus]' \
  --output text | awk '$2=="Online"{print $1}' | sort | head -1)"
if [[ -z "$TARGET" || "$TARGET" == "None" ]]; then
  echo "No online SSM instance found for Project=$PROJECT_TAG (is the ASG still at 0?)" >&2
  exit 2
fi

echo "Benchmarking on instance: $TARGET"
echo "Run root:                 $RUN_ROOT"
echo "Variants:                 $VARIANTS   (limit=$LIMIT, seed=$SEED)"
echo "Reports:                  $OUT_DIR"

# SSM AWS-RunShellScript runs the payload under /bin/sh (dash on Ubuntu), which chokes
# on the bash body below -- so base64 it locally and execute under bash remotely.
REMOTE_SCRIPT="$(cat <<REMOTE
set -u
cd "$REPO_DIR"
mkdir -p "$OUT_DIR"

MAIN_PY="$VENV_ROOT/main/bin/python"
PARAKEET_PY="$VENV_ROOT/parakeet/bin/python"
[ -x "\$MAIN_PY" ] || MAIN_PY="\$(command -v python3.10 || command -v python3)"

run_variant() {
  name="\$1"; shift
  py="\$1"; shift
  log="$OUT_DIR/\$name.log"
  echo "=== \$name ==="
  if [ ! -x "\$py" ]; then
    echo "  SKIP: \$py missing"
    return 0
  fi
  # Same seed for every variant => identical sample => comparable numbers.
  "\$py" ops/asr_model_bench.py \
      --run-root "$RUN_ROOT" --limit "$LIMIT" --seed "$SEED" \
      --json-out "$OUT_DIR/\$name.json" "\$@" > "\$log" 2>&1
  rc=\$?
  if [ \$rc -ne 0 ]; then
    echo "  FAILED rc=\$rc; tail of \$log:"
    tail -20 "\$log"
  else
    sed -n '/=====/,\$p' "\$log"
  fi
  return 0
}

for v in \$(echo "$VARIANTS" | tr ',' ' '); do
  case "\$v" in
    baseline)
      run_variant baseline "\$MAIN_PY" --baseline-only
      ;;
    whisper_greedy)
      # Subshell + export: an env prefix on a shell FUNCTION does not reliably scope
      # to that call across shells, and these must not leak into the next variant.
      ( export AMC_WHISPER_BEAM_SIZE=1
        run_variant whisper_greedy "\$MAIN_PY" --model whisper --label "whisper beam=1" )
      ;;
    whisper_int8)
      ( export AMC_WHISPER_BEAM_SIZE=1 AMC_WHISPER_COMPUTE_TYPE=int8_float16
        run_variant whisper_int8 "\$MAIN_PY" --model whisper --label "whisper beam=1 int8" )
      ;;
    parakeet)
      run_variant parakeet "\$PARAKEET_PY" --model parakeet \
        --model-path "$MODEL_ROOT/parakeet-tdt-0.6b-v3" --label "parakeet-tdt-0.6b-v3"
      ;;
    *)
      echo "unknown variant: \$v"
      ;;
  esac
done
echo "ALL VARIANTS DONE"
REMOTE
)"

PAYLOAD="$(printf '%s' "$REMOTE_SCRIPT" | base64 | tr -d '\n')"
CID="$(aws ssm send-command \
  --region "$AWS_REGION" \
  --instance-ids "$TARGET" \
  --document-name AWS-RunShellScript \
  --timeout-seconds "$TIMEOUT" \
  --parameters "commands=[\"echo $PAYLOAD | base64 -d | bash\"]" \
  --query 'Command.CommandId' --output text)"

echo "CommandId=$CID"
echo "Polling (this runs the models, so expect several minutes)..."

while :; do
  sleep 30
  STATUS="$(aws ssm get-command-invocation --region "$AWS_REGION" \
    --command-id "$CID" --instance-id "$TARGET" \
    --query 'Status' --output text 2>/dev/null || echo Pending)"
  echo "  status=$STATUS"
  case "$STATUS" in
    Success|Failed|Cancelled|TimedOut) break ;;
  esac
done

aws ssm get-command-invocation --region "$AWS_REGION" \
  --command-id "$CID" --instance-id "$TARGET" \
  --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
