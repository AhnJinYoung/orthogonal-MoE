#!/usr/bin/env bash
# Create (once) a persisted virtual environment on the PVC and install all deps
# into it. Idempotent: on a fresh pod that already has the PVC mounted, this
# returns in seconds because the venv and pip cache already exist.
#
#   bash scripts/setup_env.sh                  # create venv + install
#   FORCE=1 bash scripts/setup_env.sh          # force reinstall
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_env.sh
#
# After this, every new pod just runs:  source scripts/env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load PVC paths (env.sh tolerates a missing venv).
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh" || true

PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="${ROOT_DIR}/requirements.txt"
MARKER="${ORTHOMOE_PVC}/.env_ready"
REQ_HASH="$( (cat "${REQ_FILE}"; echo "torch=${TORCH_INDEX_URL:-default}") | sha256sum | awk '{print $1}')"

# Create the venv on the PVC if missing.
if [ ! -f "${ORTHOMOE_VENV}/bin/activate" ]; then
  echo "[setup] creating venv at ${ORTHOMOE_VENV}"
  "${PYTHON_BIN}" -m venv "${ORTHOMOE_VENV}"
fi
# shellcheck disable=SC1091
source "${ORTHOMOE_VENV}/bin/activate"

# Fast path: deps already installed for this requirements hash.
if [ "${FORCE:-0}" != "1" ] && [ -f "${MARKER}" ] && [ "$(cat "${MARKER}")" = "${REQ_HASH}" ]; then
  echo "[setup] environment already up to date (hash ${REQ_HASH:0:12}). Use FORCE=1 to reinstall."
  python -c "import torch, transformers, datasets; print('[setup] torch', torch.__version__)" 2>/dev/null || true
  exit 0
fi

python -m pip install --upgrade pip wheel setuptools

# Install PyTorch first if it is not already importable. A CUDA-matched wheel
# index can be supplied via TORCH_INDEX_URL; otherwise the default PyPI wheel
# is used (override per your CUDA stack).
if ! python -c "import torch" 2>/dev/null; then
  echo "[setup] installing torch"
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio
  else
    python -m pip install torch
  fi
fi

echo "[setup] installing requirements"
python -m pip install -r "${REQ_FILE}"

# Record the hash so subsequent pods skip reinstalling.
echo "${REQ_HASH}" > "${MARKER}"
echo "[setup] done. venv=${ORTHOMOE_VENV}"
python -c "import torch, transformers, datasets; print('[setup] torch', torch.__version__, '| transformers', transformers.__version__)"
echo "[setup] next pod only needs:  source scripts/env.sh"
