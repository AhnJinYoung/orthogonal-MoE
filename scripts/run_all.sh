#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/default_gemma4_26b.yaml}
RUN_NAME=${2:-$(date +%Y%m%d_%H%M%S)}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

OUT_DIR=${OUT_DIR:-"${ROOT_DIR}/outputs/${RUN_NAME}"}
mkdir -p "${OUT_DIR}"

echo "Config: ${CONFIG}"
echo "Output: ${OUT_DIR}"

python -m orthomoe.benchmark \
  --config "${CONFIG}" \
  --output "${OUT_DIR}/benchmark"

python -m orthomoe.visualize \
  --results "${OUT_DIR}/benchmark/benchmark.jsonl" \
  --outdir "${OUT_DIR}/figures"

if [[ "${RUN_GENERATE:-1}" == "1" ]]; then
  python -m orthomoe.generate \
    --config "${CONFIG}" \
    --output "${OUT_DIR}/generations.jsonl" || true
fi

# Continued pretraining is intentionally opt-in because 26B/35B training is expensive.
# Usage:
#   RUN_TRAIN=1 bash scripts/run_all.sh configs/default_gemma4_26b.yaml
if [[ "${RUN_TRAIN:-0}" == "1" ]]; then
  TRAIN_CONFIG=${TRAIN_CONFIG:-${CONFIG}}
  python -m orthomoe.train \
    --config "${TRAIN_CONFIG}" \
    --output "${OUT_DIR}/train"
fi

echo "Done. Results are in ${OUT_DIR}"
