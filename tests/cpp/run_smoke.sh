#!/usr/bin/env bash
# Run smoke tests for all ported PTO A5 operator tests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common/sim_env.sh"

TESTS=(
  tadd
  tmax
  tmul
  tload
  syncall
  tmov_acc2vec
  textract_acc2vec
  tinsert_acc2vec
  tmov_acc2mat
)

FAILED=()
PASSED=()

for test in "${TESTS[@]}"; do
  echo "============================================================"
  echo "SMOKE: ${test}"
  echo "============================================================"
  if "${SCRIPT_DIR}/${test}/run.sh"; then
    PASSED+=("${test}")
  else
    FAILED+=("${test}")
  fi
done

echo "============================================================"
echo "Smoke summary: ${#PASSED[@]} passed, ${#FAILED[@]} failed"
if ((${#FAILED[@]} > 0)); then
  echo "Failed: ${FAILED[*]}"
  exit 1
fi
echo "All smoke tests passed."
