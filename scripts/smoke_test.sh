#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
python -m orthomoe.train --config "${ROOT_DIR}/configs/smoke_tiny_moe.yaml" --output "${ROOT_DIR}/outputs/smoke_train"
