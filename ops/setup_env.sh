#!/usr/bin/env bash
# Build the AMC pipeline Python environments.
#
# WHY THREE ENVIRONMENTS
#   The pipeline stages need mutually-incompatible dependency sets, so a single env
#   cannot satisfy all of them:
#     * The ASR stages (qwen/granite) require `qwen-asr==0.0.6`, which HARD-PINS
#       `transformers==4.57.6` + `accelerate==1.12.0` and is documented to need a
#       "fresh, isolated environment".
#     * `whisperx` (the `align` stage) pulls pyannote-audio + a torch/transformers stack;
#       the current release (3.8.x) demands torch~=2.8 and numpy>=2.1, which conflicts
#       with our torch 2.5.1 + numpy<2 + ctranslate2/cuDNN9 stack. The torch-2.5.1-safe
#       line is whisperx==3.4.2 with transformers==4.56.2 and huggingface_hub==0.35.0 --
#       different versions than the ASR stages use.
#     * Cohere ASR (`asr_cohere`) needs `CohereAsrForConditionalGeneration`, added in
#       transformers 5.4.0 -- a major-version jump that is incompatible with the 4.57.6
#       pinned by qwen-asr. So cohere cannot share the main venv either.
#     * Parakeet TDT (`asr_parakeet`, the Whisper replacement) needs `ParakeetForTDT`,
#       added in transformers 5.9.0 -- same conflict as cohere.
#   So we build:
#     main     -> every stage except align + asr_cohere + asr_parakeet
#                 (transformers 4.57.6, hf 0.36.2, faster-whisper, gliner, spacy,
#                 torch 2.5.1+cu121)
#     align    -> the align stage only      (whisperx 3.4.2, pyannote 3.4.0,
#                 transformers 4.56.2, hf 0.35.0, torch 2.5.1+cu121)
#     cohere   -> the asr_cohere stage only (transformers>=5.4.0, torch 2.5.1+cu121)
#     parakeet -> the asr_parakeet stage only (transformers>=5.9.0, torch 2.5.1+cu121).
#                 OPTIONAL/experimental: build and verify failures only warn, so an
#                 in-evaluation model can never wedge the production stages.
#                 Skip entirely with AMC_SKIP_PARAKEET=1.
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
COHERE_VENV="$VENV_ROOT/cohere"
PARAKEET_VENV="$VENV_ROOT/parakeet"

log() { echo "[$(date -Is)] setup_env: $*"; }
die() { log "ERROR: $*"; exit 1; }
warn() { log "WARN: $*"; }

cd "$REPO_DIR" || die "repo not found at $REPO_DIR"
[[ -f docker/requirements-gpu.txt ]] || die "docker/requirements-gpu.txt missing"
[[ -f docker/requirements-align.txt ]] || die "docker/requirements-align.txt missing"
[[ -f docker/requirements-cohere.txt ]] || die "docker/requirements-cohere.txt missing"

# The parakeet venv (Whisper replacement, still under evaluation) is treated as
# OPTIONAL: a failure to build or verify it must never block the three venvs the
# production pipeline depends on. Set AMC_SKIP_PARAKEET=1 to leave it out entirely.
PARAKEET_ENABLED=1
[[ "${AMC_SKIP_PARAKEET:-0}" == "1" ]] && PARAKEET_ENABLED=0
[[ -f docker/requirements-parakeet.txt ]] || PARAKEET_ENABLED=0

# Per-venv signatures: rebuild a venv only when ITS pin set (or this script's schema)
# changes, so editing the align requirements does not force a main rebuild and vice versa.
sig_of() { cat "$@" 2>/dev/null | sha256sum | awk -v s="$SCHEMA_VERSION" '{print s"-"$1}'; }
MAIN_SIG="$(sig_of docker/requirements-gpu.txt docker/constraints-gpu.txt pyproject.toml)"
ALIGN_SIG="$(sig_of docker/requirements-align.txt pyproject.toml)"
COHERE_SIG="$(sig_of docker/requirements-cohere.txt pyproject.toml)"
PARAKEET_SIG="$(sig_of docker/requirements-parakeet.txt pyproject.toml)"

venv_sig() {  # $1=venv dir -> expected signature
  case "$1" in
    "$ALIGN_VENV") printf '%s' "$ALIGN_SIG" ;;
    "$COHERE_VENV") printf '%s' "$COHERE_SIG" ;;
    "$PARAKEET_VENV") printf '%s' "$PARAKEET_SIG" ;;
    *) printf '%s' "$MAIN_SIG" ;;
  esac
}

venv_ready() {  # $1=venv dir
  [[ -x "$1/bin/python" && -f "$1/.ready" && "$(cat "$1/.ready" 2>/dev/null)" == "$(venv_sig "$1")" ]]
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
  echo "$MAIN_SIG" > "$MAIN_VENV/.ready"
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
  # NOTE: no numpy<2 pin here. whisperx 3.4.2 + pyannote + librosa need numpy>=2; only the
  # main venv (spaCy/thinc, ctranslate2, onnxruntime) requires the NumPy 1.x ABI.
  echo "$ALIGN_SIG" > "$ALIGN_VENV/.ready"
  log "ALIGN venv ready"
}

build_cohere() {
  log "building COHERE venv at $COHERE_VENV"
  rm -rf "$COHERE_VENV"
  "$PYTHON_BIN" -m venv "$COHERE_VENV" || die "venv create failed (cohere)"
  local pip="$COHERE_VENV/bin/python -m pip"
  $pip install --upgrade pip wheel setuptools || die "pip upgrade (cohere)"
  $pip install --index-url "$TORCH_INDEX" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    || die "torch install (cohere)"
  # Install the editable package WITHOUT the qwen extra so the 4.57.6 pin does not leak in;
  # requirements-cohere.txt pulls transformers>=5.4.0 (CohereAsrForConditionalGeneration).
  $pip install -r docker/requirements-cohere.txt -e "." || die "requirements/editable install (cohere)"
  # Match the cuDNN 9 ABI of the torch 2.5.1+cu121 wheel (same pin as main/align) so a stray
  # newer nvidia-cudnn pulled transitively cannot trip CUDNN_STATUS_NOT_INITIALIZED.
  $pip install --force-reinstall --no-deps nvidia-cudnn-cu12==9.1.0.70 || die "cudnn pin (cohere)"
  # NOTE: no numpy<2 pin here (no ctranslate2/spaCy/onnxruntime in this venv). transformers 5.x
  # + librosa want numpy>=2; let the resolver choose.
  echo "$COHERE_SIG" > "$COHERE_VENV/.ready"
  log "COHERE venv ready"
}

# Optional/experimental: never `die`, so a parakeet build failure cannot wedge a
# fleet whose production stages only need main/align/cohere.
build_parakeet() {
  log "building PARAKEET venv at $PARAKEET_VENV"
  rm -rf "$PARAKEET_VENV"
  "$PYTHON_BIN" -m venv "$PARAKEET_VENV" || { warn "venv create failed (parakeet)"; return 1; }
  local pip="$PARAKEET_VENV/bin/python -m pip"
  $pip install --upgrade pip wheel setuptools || { warn "pip upgrade (parakeet)"; return 1; }
  $pip install --index-url "$TORCH_INDEX" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    || { warn "torch install (parakeet)"; return 1; }
  # Install the editable package WITHOUT the qwen extra so the transformers 4.57.6
  # pin does not leak in; requirements-parakeet.txt needs >=5.9.0 for ParakeetForTDT.
  $pip install -r docker/requirements-parakeet.txt -e "." || { warn "requirements/editable install (parakeet)"; return 1; }
  # Match the cuDNN 9 ABI of the torch 2.5.1+cu121 wheel (same pin as the other venvs).
  $pip install --force-reinstall --no-deps nvidia-cudnn-cu12==9.1.0.70 || { warn "cudnn pin (parakeet)"; return 1; }
  echo "$PARAKEET_SIG" > "$PARAKEET_VENV/.ready"
  log "PARAKEET venv ready"
}

verify() {
  local rc=0
  log "verifying MAIN venv ($MAIN_VENV)"
  "$MAIN_VENV/bin/python" - <<'PY' || rc=1
import torch, torchvision, faster_whisper, transformers, qwen_asr, gliner, spacy, soundfile, huggingface_hub as h
import faiss, sklearn  # speaker clustering (cluster-speakers)  # noqa: F401
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
  log "verifying COHERE venv ($COHERE_VENV)"
  "$COHERE_VENV/bin/python" - <<'PY' || rc=1
import torch, transformers
# Mirror the runtime load path: alias FP8 MX dtypes that transformers 5.x references at
# import but torch 2.5.1 lacks (added in torch 2.7). Harmless for the bf16/fp32 Cohere model.
from amc_pipeline.transcription import _ensure_torch_fp8_dtype_shim
_ensure_torch_fp8_dtype_shim(torch)
from transformers import CohereAsrForConditionalGeneration  # noqa: F401  (the whole point)
import amc_pipeline  # noqa: F401
print("  COHERE ok: torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "tf", transformers.__version__)
assert torch.cuda.is_available(), "CUDA not available in COHERE venv"
PY
  if (( PARAKEET_ENABLED == 1 )) && [[ -x "$PARAKEET_VENV/bin/python" ]]; then
    log "verifying PARAKEET venv ($PARAKEET_VENV)"
    # Soft: a broken experimental venv must not fail the whole verification.
    "$PARAKEET_VENV/bin/python" - <<'PY' || warn "PARAKEET venv verification failed (asr_parakeet stage unavailable)"
import torch, transformers
# Same torch 2.5.1 / transformers 5.x FP8 dtype gap the cohere venv hits above.
from amc_pipeline.transcription import _ensure_torch_fp8_dtype_shim
_ensure_torch_fp8_dtype_shim(torch)
# Resolve the concrete model class, not just AutoModelForTDT: transformers exposes
# its API lazily, so the Auto alias imports cleanly on a venv where the real
# Parakeet modeling module cannot load.
from transformers import ParakeetForTDT  # noqa: F401  (the whole point; needs transformers>=5.9.0)
import amc_pipeline  # noqa: F401
print("  PARAKEET ok: torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "tf", transformers.__version__)
assert torch.cuda.is_available(), "CUDA not available in PARAKEET venv"
PY
  fi
  return "$rc"
}

if [[ "${1:-}" == "--verify" ]]; then
  venv_ready "$MAIN_VENV" || die "MAIN venv not ready/stale; run without --verify to build"
  venv_ready "$ALIGN_VENV" || die "ALIGN venv not ready/stale; run without --verify to build"
  venv_ready "$COHERE_VENV" || die "COHERE venv not ready/stale; run without --verify to build"
  verify || die "verification failed"
  log "verification passed"
  exit 0
fi

# ---- NFS-safe single-builder election --------------------------------------------------
# IMPORTANT: flock() does NOT provide mutual exclusion across NFS clients -- advisory locks
# are not reliably honored between hosts, so a fleet-wide `flock` lets every instance enter
# the "single builder" section at once and clobber the shared venv. We instead use an atomic
# `mkdir` lease (the same primitive resume_shard.sh uses for shard claims), refreshed by a
# background heartbeat, with stale-lease takeover so a dead builder cannot wedge the fleet.
mkdir -p "$VENV_ROOT"
LOCK_DIR="$VENV_ROOT/.build.lock.d"
LEASE="$LOCK_DIR/lease"
SELF="$(hostname):$$"
LEASE_TTL="${AMC_ENV_LOCK_TTL:-900}"   # treat builder as dead if its lease is older than this
HB_PID=""
write_lease() { printf '%s %s\n' "$(date +%s)" "$SELF" > "$LEASE" 2>/dev/null || true; }
start_hb() { ( while true; do write_lease; sleep 60; done ) & HB_PID=$!; }
release_lock() {
  [[ -n "$HB_PID" ]] && kill "$HB_PID" 2>/dev/null || true
  HB_PID=""
  rm -rf "$LOCK_DIR" 2>/dev/null || true
}
trap 'release_lock' EXIT INT TERM

parakeet_ready() { (( PARAKEET_ENABLED == 0 )) || venv_ready "$PARAKEET_VENV"; }
both_ready() { venv_ready "$MAIN_VENV" && venv_ready "$ALIGN_VENV" && venv_ready "$COHERE_VENV" && parakeet_ready; }

am_builder=0
deadline=$(( $(date +%s) + LOCK_WAIT ))
while true; do
  if [[ "${FORCE_REBUILD:-0}" != "1" ]] && both_ready; then
    log "all venvs already built (main $MAIN_SIG / align $ALIGN_SIG / cohere $COHERE_SIG); skipping build"
    break
  fi
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    am_builder=1; write_lease; start_hb
    log "acquired build lease as $SELF"
    break
  fi
  # Lock is held: wait for the builder, or take over if its lease has gone stale.
  ep="$(awk 'NR==1{print $1}' "$LEASE" 2>/dev/null)"; ep="${ep:-0}"
  age=$(( $(date +%s) - ep ))
  if (( age > LEASE_TTL )); then
    log "stale build lease (age ${age}s > ${LEASE_TTL}s); taking over"
    tomb="$LOCK_DIR.dead.$$.$RANDOM"
    mv "$LOCK_DIR" "$tomb" 2>/dev/null && rm -rf "$tomb" 2>/dev/null || true
    continue
  fi
  if (( $(date +%s) > deadline )); then die "waited ${LOCK_WAIT}s for the build lease; giving up"; fi
  log "another instance is building (lease age ${age}s); waiting..."
  sleep 15
done

if (( am_builder == 1 )); then
  if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
    log "FORCE_REBUILD=1: clearing markers and rebuilding all venvs"
    rm -f "$MAIN_VENV/.ready" "$ALIGN_VENV/.ready" "$COHERE_VENV/.ready" "$PARAKEET_VENV/.ready" 2>/dev/null || true
    build_main
    build_align
    build_cohere
    (( PARAKEET_ENABLED == 1 )) && { build_parakeet || warn "parakeet venv build failed; continuing"; }
  else
    if venv_ready "$MAIN_VENV"; then log "MAIN venv up-to-date (sig $MAIN_SIG); skipping"; else build_main; fi
    if venv_ready "$ALIGN_VENV"; then log "ALIGN venv up-to-date (sig $ALIGN_SIG); skipping"; else build_align; fi
    if venv_ready "$COHERE_VENV"; then log "COHERE venv up-to-date (sig $COHERE_SIG); skipping"; else build_cohere; fi
    if (( PARAKEET_ENABLED == 1 )); then
      if venv_ready "$PARAKEET_VENV"; then log "PARAKEET venv up-to-date (sig $PARAKEET_SIG); skipping"
      else build_parakeet || warn "parakeet venv build failed; continuing"; fi
    fi
  fi
  release_lock   # let waiters proceed to verify in parallel
fi

verify || die "post-build verification failed"
log "all environments ready and verified (main $MAIN_SIG / align $ALIGN_SIG / cohere $COHERE_SIG)"
