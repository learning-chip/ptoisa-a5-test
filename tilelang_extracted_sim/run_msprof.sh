#!/usr/bin/env bash
# Run extracted TileLang PTO kernels under msprof A5 CPU simulator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
  # shellcheck source=/dev/null
  source "${ASCEND_HOME_PATH}/bin/setenv.bash"
fi

SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export LD_LIBRARY_PATH="${SIM_LIB}:${LD_LIBRARY_PATH:-}"
ulimit -n 65535

TIMEOUT="${MSPROF_TIMEOUT:-30}"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/msprof"
mkdir -p "${OUTPUT_DIR}"

echo "==> msprof op simulator (Ascend950PR_9599)"
echo "    ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "    timeout=${TIMEOUT} min"

msprof op simulator \
  --soc-version=Ascend950PR_9599 \
  --timeout="${TIMEOUT}" \
  --output="${OUTPUT_DIR}" \
  python3 "${SCRIPT_DIR}/run_kernel.py" "$@"
