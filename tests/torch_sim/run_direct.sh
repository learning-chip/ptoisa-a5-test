#!/usr/bin/env bash
# Run torch_sim smoke tests directly (no msprof). Use on real NPU or when msprof is unavailable.
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

exec python3 run_smoke.py
