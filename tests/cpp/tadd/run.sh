#!/usr/bin/env bash
# Build, generate golden data, and run tadd smoke test in simulator mode.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"

RUN_ALL=0
FULL_BUILD=0
CASE_FILTER=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [--full] [--all] [--case <name>]

  (default)  build smoke binary, generate smoke golden data, run smoke cases
  --full     build with all regression kernel instantiations
  --all      run full regression after smoke (requires --full build)
  --case     run one named case
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      FULL_BUILD=1
      shift
      ;;
    --all)
      RUN_ALL=1
      FULL_BUILD=1
      shift
      ;;
    --case)
      CASE_FILTER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "ERROR: ASCEND_HOME_PATH is not set. Source CANN set_env.sh first." >&2
  exit 1
fi

# Simulator runtime setup (mirrors tests/script/run_st.py sim mode)
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/runtime/lib64' | paste -sd: - || true)"
fi
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/runtime/lib64/stub:${LD_LIBRARY_PATH:-}"

SETENV="${ASCEND_HOME_PATH}/bin/setenv.bash"
if [[ -f "${SETENV}" ]]; then
  # shellcheck disable=SC1090
  source "${SETENV}"
fi

SIMULATOR_LIB="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
export LD_LIBRARY_PATH="${SIMULATOR_LIB}:${LD_LIBRARY_PATH}"

BUILD_ARGS=()
if [[ "${FULL_BUILD}" -eq 1 ]]; then
  BUILD_ARGS+=(--full)
fi
"${SCRIPT_DIR}/build.sh" "${BUILD_ARGS[@]}"

GEN_ARGS=(--output-dir "${BUILD_DIR}/cases")
if [[ "${FULL_BUILD}" -eq 1 ]]; then
  GEN_ARGS+=(--all)
fi
python3 "${SCRIPT_DIR}/gen_data.py" "${GEN_ARGS[@]}"

mkdir -p "${BUILD_DIR}/camodel_log"
export CAMODEL_LOG_PATH="${BUILD_DIR}/camodel_log"

cd "${BUILD_DIR}"
RUN_ARGS=()
if [[ -n "${CASE_FILTER}" ]]; then
  RUN_ARGS+=(--case "${CASE_FILTER}")
elif [[ "${RUN_ALL}" -eq 1 ]]; then
  RUN_ARGS+=(--all)
fi

echo "==> Running tadd ${RUN_ARGS[*]:-<smoke>}"
./tadd "${RUN_ARGS[@]}"
