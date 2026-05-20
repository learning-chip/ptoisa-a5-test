#!/usr/bin/env bash
# Generic build + optional gen_data + run wrapper for a standalone test directory.
set -euo pipefail

if [[ -z "${PTO_TEST_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  TEST_DIR="${1:?test directory required}"
  TEST_DIR="$(cd "${TEST_DIR}" && pwd)"
else
  TEST_DIR="${PTO_TEST_DIR}"
fi

BUILD_DIR="${TEST_DIR}/build"
EXEC_NAME="${PTO_TEST_NAME:?PTO_TEST_NAME required}"

RUN_ALL=0
FULL_BUILD=0
CASE_FILTER=""
SKIP_GEN=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--full] [--all] [--case <name>] [--skip-gen]

  (default)  build smoke binary, generate golden data, run smoke cases
  --full     build with all regression kernel instantiations
  --all      run full regression (requires --full build)
  --case     run a single named case
  --skip-gen skip golden data generation
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL_BUILD=1; shift ;;
    --all) RUN_ALL=1; FULL_BUILD=1; shift ;;
    --case) CASE_FILTER="$2"; shift 2 ;;
    --skip-gen) SKIP_GEN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# shellcheck disable=SC1091
source "${TEST_DIR}/../common/sim_env.sh"

BUILD_ARGS=()
[[ "${FULL_BUILD}" -eq 1 ]] && BUILD_ARGS+=(--full)
"${TEST_DIR}/build.sh" "${BUILD_ARGS[@]}"

if [[ "${SKIP_GEN}" -eq 0 && -f "${TEST_DIR}/gen_data.py" ]]; then
  GEN_ARGS=(--output-dir "${BUILD_DIR}/cases")
  [[ "${FULL_BUILD}" -eq 1 ]] && GEN_ARGS+=(--all)
  python3 "${TEST_DIR}/gen_data.py" "${GEN_ARGS[@]}"
fi

mkdir -p "${BUILD_DIR}/camodel_log"
export CAMODEL_LOG_PATH="${BUILD_DIR}/camodel_log"

cd "${BUILD_DIR}"
RUN_ARGS=()
if [[ -n "${CASE_FILTER}" ]]; then
  RUN_ARGS+=(--case "${CASE_FILTER}")
elif [[ "${RUN_ALL}" -eq 1 ]]; then
  RUN_ARGS+=(--all)
fi

echo "==> Running ${EXEC_NAME} ${RUN_ARGS[*]:-<smoke>}"
./"${EXEC_NAME}" "${RUN_ARGS[@]}"
