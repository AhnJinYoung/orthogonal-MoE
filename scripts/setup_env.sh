#!/usr/bin/env bash
# Create (once) a persisted virtual environment on the PVC and install all deps
# into it. Idempotent: on a fresh pod that already has the PVC mounted, this
# returns in seconds because the venv and pip cache already exist.
#
#   bash scripts/setup_env.sh                  # create venv + install (auto CUDA)
#   FORCE=1 bash scripts/setup_env.sh          # force reinstall
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_env.sh
#
# When a GPU is present the matching CUDA torch wheel is selected automatically
# from the driver's CUDA version; set TORCH_INDEX_URL to override. A CPU-only
# torch already installed on a GPU pod is detected and replaced (it silently
# runs large models on host RAM and OOM-kills the pod).
#
# After this, every new pod just runs:  source scripts/env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- GPU / CUDA detection -------------------------------------------------
gpu_present() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

# Echo the torch wheel index URL matching the driver's CUDA version, or nothing.
detect_torch_index_url() {
  local cuda_ver major minor num tag
  cuda_ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
  [ -z "${cuda_ver}" ] && return 0
  major="${cuda_ver%%.*}"
  minor="${cuda_ver#*.}"
  num=$(( major * 100 + minor ))
  if   [ "${num}" -ge 1208 ]; then tag="cu128"
  elif [ "${num}" -ge 1206 ]; then tag="cu126"
  elif [ "${num}" -ge 1204 ]; then tag="cu124"
  elif [ "${num}" -ge 1201 ]; then tag="cu121"
  elif [ "${num}" -ge 1108 ]; then tag="cu118"
  else return 0
  fi
  echo "https://download.pytorch.org/whl/${tag}"
}

# True (exit 0) when torch can actually use the GPU. Requires an active venv.
# This catches both a CPU-only torch (version.cuda is None) and a CUDA build
# compiled for a newer CUDA than the driver supports (cuda.is_available() False).
torch_cuda_works() { python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

# Load PVC paths (env.sh tolerates a missing venv).
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh" || true

PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="${ROOT_DIR}/requirements.txt"
MARKER="${ORTHOMOE_PVC}/.env_ready"

# Resolve a CUDA wheel index automatically when a GPU is present and the caller
# did not pin one. Done before hashing so switching CPU<->CUDA busts the marker.
if [ -z "${TORCH_INDEX_URL:-}" ] && gpu_present; then
  TORCH_INDEX_URL="$(detect_torch_index_url)"
  if [ -n "${TORCH_INDEX_URL}" ]; then
    echo "[setup] GPU detected; auto-selected torch wheel index ${TORCH_INDEX_URL}"
    echo "[setup]   override with TORCH_INDEX_URL=... if you need a different CUDA build"
  fi
fi

REQ_HASH="$( (cat "${REQ_FILE}"; echo "torch=${TORCH_INDEX_URL:-default}") | sha256sum | awk '{print $1}')"

# Create the venv on the PVC if missing.
if [ ! -f "${ORTHOMOE_VENV}/bin/activate" ]; then
  echo "[setup] creating venv at ${ORTHOMOE_VENV}"
  "${PYTHON_BIN}" -m venv "${ORTHOMOE_VENV}"
fi
# shellcheck disable=SC1091
source "${ORTHOMOE_VENV}/bin/activate"

# Fast path: deps already installed for this requirements hash. Skip only when
# torch is also usable on this pod -- a cached CPU-only torch on a GPU box must
# be rebuilt, not skipped.
if [ "${FORCE:-0}" != "1" ] && [ -f "${MARKER}" ] && [ "$(cat "${MARKER}")" = "${REQ_HASH}" ]; then
  if ! gpu_present || torch_cuda_works; then
    echo "[setup] environment already up to date (hash ${REQ_HASH:0:12}). Use FORCE=1 to reinstall."
    python -c "import torch, transformers, datasets; print('[setup] torch', torch.__version__, '| cuda', torch.version.cuda)" 2>/dev/null || true
    exit 0
  fi
  echo "[setup] cached torch cannot use this pod's GPU; rebuilding torch."
fi

python -m pip install --upgrade pip wheel setuptools

# Decide whether torch needs (re)installing. A CPU-only torch on a GPU box is a
# silent footgun, so replace it with a CUDA build when one is available.
need_torch_install=0
if ! python -c "import torch" 2>/dev/null; then
  need_torch_install=1
elif gpu_present && ! torch_cuda_works; then
  echo "[setup] installed torch cannot use the GPU (CPU-only or built for a newer CUDA"
  echo "[setup]   than the driver supports); reinstalling a driver-matched CUDA build."
  python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  need_torch_install=1
fi

if [ "${need_torch_install}" = "1" ]; then
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    echo "[setup] installing torch from ${TORCH_INDEX_URL}"
    python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio
  else
    if gpu_present; then
      echo "[setup] WARNING: a GPU is present but no CUDA wheel index could be resolved;"
      echo "[setup]          installing the default (CPU-only) torch. Set TORCH_INDEX_URL"
      echo "[setup]          (e.g. https://download.pytorch.org/whl/cu124) and rerun with FORCE=1."
    fi
    python -m pip install torch
  fi
fi

echo "[setup] installing requirements"
python -m pip install -r "${REQ_FILE}"

# Record the hash so subsequent pods skip reinstalling.
echo "${REQ_HASH}" > "${MARKER}"
echo "[setup] done. venv=${ORTHOMOE_VENV}"
python -c "import torch, transformers, datasets; print('[setup] torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available(), '| transformers', transformers.__version__)"
echo "[setup] next pod only needs:  source scripts/env.sh"
