#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh" || true
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
python -m orthomoe.train --config "${ROOT_DIR}/configs/smoke_tiny_moe.yaml" --output "${ROOT_DIR}/outputs/smoke_train"
