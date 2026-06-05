#!/usr/bin/env bash
# Build the AMC pipeline Python environments.
#
# WHY TWO ENVIRONMENTS
#   The pipeline stages need mutually-incompatible dependency sets, so a single env
#   cannot satisfy all of them:
#     * The ASR stages (qwen/cohere/granite) require `qwen-asr==0.0.6`, which HARD-PINS
#       `transformers==4.57.6` + `accelerate==1.12.0` and is documented to need a
#       "fresh, isolated environment".
#     * `whisperx` (the `align` stage) pulls pyannote-audio + a torch/transformers stack;
#       the current release (3.8.x) demands torch~=2.8 and numpy>=2.1, which conflicts
#       with our torch 2.5.1 + numpy<2 + ctranslate2/cuDNN9 stack. The torch-2.5.1-safe
#       line is whisperx==3.4.2 with transformers==4.56.2 and huggingface_hub==0.35.0 --
#       different versions than the ASR stages use.
#   So we build:
#     main  -> every stage except align   (transformers 4.57.6, hf 0.36.2, faster-whisper,
#              gliner, spacy, torch 2.5.1+cu121)
#     align -> the align stage only        (whisperx 3.4.2, pyannote 3.4.0,
#              transformers 4.56.2, hf 0.35.0, torch 2.5.1+cu121)
#
# WHERE
#   Both venvs are built ONCE on shared storage ($AMC_VENV_ROOT, default
#   /mnt/amc-data/venvs) and reused by every instance. New/recreated instances do not
#   reinstall multi-GB wheels -- they just import from the shared venv. A signature
#   (hash of the requirement files + this script) gates rebuilds: change a pin -> rebuild.
#
# CONCURRENCY
#   Many instances may call this at once (boot storm). A flock makes exactly one build;
#   the rest wait for the ready marker. Every caller then runs an import verification
#   against the shared venv so each instance confirms torch/CUDA works locally.
#
# USAGE
#   bash ops/setup_env.sh              # build if needed, then verify
#   bash ops/setup_env.sh --verify     # verify only (no build)
#   FORCE_REBUILD=1 bash ops/setup_env.sh
set -uo pipefail

REPO_DIR="${REPO_DIR:-/mnt/amc-data/AMC}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_ROOT="${AMC_VENV_ROOT:-/mnt/amc-data/venvs}"
TORCH_INDEX="${AMC_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
LOCK_WAIT="${AMC_ENV_LOCK_WAIT:-3600}"
SCHEMA_VERSION="v1"   # bump to force a rebuild independent of requirement file changes

MAIN_VENV="$VENV_ROOT/main"
ALIGN_VENV="$VENV_ROOT/align"

log() { echo "[$(date -Is)] setup_env: $*"; }
die() { log "ERROR: $*"; exit 1; }

cd "$REPO_DIR" || die "repo not found at $REPO_DIR"
[[ -f docker/requirements-gpu.txt ]] || die "docker/requirements-gpu.txt missing"
[[ -f docker/requirements-align.txt ]] || die "docker/requirements-align.txt missing"

# Signature: rebuild when any input pin set or this script's schema changes.
signature() {
  cat docker/requirements-gpu.txt docker/requirements-align.txt \
      docker/constraints-gpu.txt pyproject.toml 2>/dev/null \
    | sha256sum | awk -v s="$SCHEMA_VERSION" '{print s"-"$1}'
}
SIG="$(signature)"

venv_ready() {  # $1=venv dir
  [[ -x "$1/bin/python" && -f "$1/.ready" && "$(cat "$1/.ready" 2>/dev/null)" == "$SIG" ]]
}

build_main() {
  log "building MAIN venv at $MAIN_VENV"
  rm -rf "$MAIN_VENV"
  "$PYTHON_BIN" -m venv "$MAIN_VENV" || die "venv create failed (main)"
  local pip="$MAIN_VENV/bin/python -m pip"
  $pip install --upgrade pip wheel setuptools || die "pip upgrade (main)"
  $pip install --index-url "$TORCH_INDEX" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    || die "torch install (main)"
  $pip install --constraint docker/constraints-gpu.txt -r docker/requirements-gpu.txt -e ".[qwen]" \
    || die "requirements/editable install (main)"
  # cuDNN ABI must match the torch 2.5.1 build (9.1.0.70); a stray 9.2.x from a prior
  # cu130 install causes CUDNN_STATUS_NOT_INITIALIZED. huggingface_hub<1.0 keeps the
  # piiranha tokenizer load from 404-ing under transformers 4.57.6.
  $pip install --force-reinstall --no-deps nvidia-cudnn-cu12==9.1.0.70 || die "cudnn pin (main)"
  $pip install --force-reinstall --no-deps huggingface_hub==0.36.2 || die "hf pin (main)"
  $pip install "numpy<2" || die "numpy pin (main)"
  "$MAIN_VENV/bin/python" -m spacy download en_core_web_sm || log "WARN: spacy model download failed (retry later)"
  echo "$SIG" > "$MAIN_VENV/.ready"
  log "MAIN venv ready"
}

build_align() {
  log "building ALIGN venv at $ALIGN_VENV"
  rm -rf "$ALIGN_VENV"
  "$PYTHON_BIN" -m venv "$ALIGN_VENV" || die "venv create failed (align)"
  local pip="$ALIGN_VENV/bin/python -m pip"
  $pip install --upgrade pip wheel setuptools || die "pip upgrade (align)"
  $pip install --index-url "$TORCH_INDEX" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    || die "torch install (align)"
  # Install the editable package WITHOUT the qwen extra so the 4.57.6 pin does not leak in.
  $pip install -r docker/requirements-align.txt -e "." || die "requirements/editable install (align)"
  $pip install --force-reinstall --no-deps nvidia-cudnn-cu12==9.1.0.70 || die "cudnn pin (align)"
  $pip install "numpy<2" || die "numpy pin (align)"
  echo "$SIG" > "$ALIGN_VENV/.ready"
  log "ALIGN venv ready"
}

verify() {
  local rc=0
  log "verifying MAIN venv ($MAIN_VENV)"
  "$MAIN_VENV/bin/python" - <<'PY' || rc=1
import torch, torchvision, faster_whisper, transformers, qwen_asr, gliner, spacy, soundfile, huggingface_hub as h
import amc_pipeline  # noqa: F401
print("  MAIN ok: torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "tv", torchvision.__version__, "tf", transformers.__version__, "hf", h.__version__)
assert torch.cuda.is_available(), "CUDA not available in MAIN venv"
PY
  log "verifying ALIGN venv ($ALIGN_VENV)"
  "$ALIGN_VENV/bin/python" - <<'PY' || rc=1
import torch, whisperx, transformers, huggingface_hub as h
import amc_pipeline  # noqa: F401
print("  ALIGN ok: torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "whisperx", getattr(whisperx, "__version__", "?"), "tf", transformers.__version__, "hf", h.__version__)
assert torch.cuda.is_available(), "CUDA not available in ALIGN venv"
PY
  return "$rc"
}

if [[ "${1:-}" == "--verify" ]]; then
  venv_ready "$MAIN_VENV" || die "MAIN venv not ready/stale; run without --verify to build"
  venv_ready "$ALIGN_VENV" || die "ALIGN venv not ready/stale; run without --verify to build"
  verify || die "verification failed"
  log "verification passed"
  exit 0
fi

mkdir -p "$VENV_ROOT"
exec 9>"$VENV_ROOT/.build.lock"
log "acquiring build lock (wait up to ${LOCK_WAIT}s)"
flock -w "$LOCK_WAIT" 9 || die "could not acquire build lock within ${LOCK_WAIT}s"

if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
  log "FORCE_REBUILD=1: rebuilding both venvs"
  build_main
  build_align
else
  if venv_ready "$MAIN_VENV"; then log "MAIN venv up-to-date (sig $SIG); skipping"; else build_main; fi
  if venv_ready "$ALIGN_VENV"; then log "ALIGN venv up-to-date (sig $SIG); skipping"; else build_align; fi
fi

# Release the lock before verification so other instances can proceed to verify in parallel.
flock -u 9 || true

verify || die "post-build verification failed"
log "all environments ready and verified (sig $SIG)"
