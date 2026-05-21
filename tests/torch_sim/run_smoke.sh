#!/usr/bin/env bash
# Run torch_sim smoke tests under msprof op simulator (Ascend950PR_9599).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
  # shellcheck source=/dev/null
  source "${ASCEND_HOME_PATH}/bin/setenv.bash"
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

SIM_LIB="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/tools/simulator/Ascend950PR_9599/lib"
export LD_LIBRARY_PATH="${SIM_LIB}:${LD_LIBRARY_PATH:-}"
ulimit -n 65535

OPS=(tadd tmax tmul tload syncall tmov_acc2vec textract_acc2vec tinsert_acc2vec tmov_acc2mat)
mkdir -p msprof_res

for op in "${OPS[@]}"; do
  echo "==> msprof ${op}"
  LOG="$(mktemp)"
  msprof op simulator --soc-version=Ascend950PR_9599 \
    --output="msprof_res/${op}" \
    python3 "${op}/test_${op}.py" 2>&1 | tee "${LOG}"
  if grep -qE '::[^ ]+ FAILED' "${LOG}"; then
    echo "FAILED: ${op} smoke case(s) failed"
    rm -f "${LOG}"
    exit 1
  fi
  rm -f "${LOG}"
done

echo "All torch_sim msprof smoke modules completed."
