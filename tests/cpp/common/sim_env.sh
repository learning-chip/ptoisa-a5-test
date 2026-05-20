#!/usr/bin/env bash
# Set up CANN simulator runtime environment (mirrors tests/script/run_st.py sim mode).
set -euo pipefail

SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "ERROR: ASCEND_HOME_PATH is not set. Source CANN set_env.sh first." >&2
  exit 1
fi

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
