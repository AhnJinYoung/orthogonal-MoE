#!/usr/bin/env bash
# Source this file (do NOT execute it) to point the whole toolchain at the
# persistent volume and activate the persisted virtual environment:
#
#   source scripts/env.sh
#
# Everything that is expensive to recreate when a pod dies -- the venv, the
# Hugging Face hub/dataset caches, the pip wheel cache, torch/triton caches and
# the device_map offload folder -- is kept under $ORTHOMOE_PVC so a fresh pod
# only needs to re-source this file (and never re-download/re-install).
#
# Override the PVC mount point with:  export ORTHOMOE_PVC=/your/pvc/path

# Resolve repo root regardless of where this is sourced from.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _ORTHOMOE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _ORTHOMOE_ENV_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
ORTHOMOE_ROOT="$(cd "${_ORTHOMOE_ENV_DIR}/.." && pwd)"
export ORTHOMOE_ROOT

# Pick the PVC: explicit override > /pvc (if writable) > repo-local .pvc fallback.
if [ -z "${ORTHOMOE_PVC:-}" ]; then
  if [ -d /pvc ] && [ -w /pvc ]; then
    ORTHOMOE_PVC="/pvc/orthomoe"
  else
    ORTHOMOE_PVC="${ORTHOMOE_ROOT}/.pvc"
  fi
fi
export ORTHOMOE_PVC
mkdir -p "${ORTHOMOE_PVC}" 2>/dev/null || true

# Persisted virtual environment location.
export ORTHOMOE_VENV="${ORTHOMOE_VENV:-${ORTHOMOE_PVC}/venv}"

# Caches: keep every heavyweight download on the PVC.
export HF_HOME="${HF_HOME:-${ORTHOMOE_PVC}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ORTHOMOE_PVC}/pip-cache}"
export TORCH_HOME="${TORCH_HOME:-${ORTHOMOE_PVC}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ORTHOMOE_PVC}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ORTHOMOE_PVC}/xdg-cache}"
# device_map="auto" disk offload target (matches resources.offload_folder default).
export ORTHOMOE_OFFLOAD="${ORTHOMOE_OFFLOAD:-${ORTHOMOE_PVC}/offload}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${PIP_CACHE_DIR}" \
         "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${ORTHOMOE_OFFLOAD}" 2>/dev/null || true

# Conservative thread caps so a shared pod is not oversubscribed (the Python
# resources module re-applies these from the config too).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Make the package importable.
export PYTHONPATH="${ORTHOMOE_ROOT}/src:${PYTHONPATH:-}"

# Activate the persisted venv if it exists. setup_env.sh creates it.
if [ -f "${ORTHOMOE_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ORTHOMOE_VENV}/bin/activate"
  export ORTHOMOE_ENV_ACTIVE=1
else
  echo "[env] venv not found at ${ORTHOMOE_VENV}. Run: bash scripts/setup_env.sh" >&2
  export ORTHOMOE_ENV_ACTIVE=0
fi

echo "[env] ORTHOMOE_PVC=${ORTHOMOE_PVC}"
echo "[env] HF_HOME=${HF_HOME}"
echo "[env] venv=${ORTHOMOE_VENV} (active=${ORTHOMOE_ENV_ACTIVE})"
