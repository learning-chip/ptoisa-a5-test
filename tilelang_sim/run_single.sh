#!/usr/bin/env bash
# Run one tilelang example under msprof A5 CPU simulator.
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
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export TILELANG_SIM_ROOT="${SCRIPT_DIR}"
ulimit -n 65535

MANIFEST_ID=""
REL_PATH=""
TIMEOUT_MIN=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --id) MANIFEST_ID="$2"; shift 2 ;;
    --rel-path) REL_PATH="$2"; shift 2 ;;
    --timeout) TIMEOUT_MIN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "$MANIFEST_ID" && -z "$REL_PATH" ]]; then
  echo "Usage: $0 --id <manifest_id> | --rel-path <rel_path> [--timeout MINUTES]"
  exit 2
fi

mkdir -p results/logs results/msprof

ARGS=(python3 -m common.runner)
if [[ -n "$MANIFEST_ID" ]]; then
  ARGS+=(--manifest-id "$MANIFEST_ID")
else
  ARGS+=(--rel-path "$REL_PATH")
fi

OUT_DIR="results/msprof/${MANIFEST_ID:-$(echo "$REL_PATH" | tr '/' '_')}"
msprof op simulator \
  --soc-version=Ascend950PR_9599 \
  --timeout="${TIMEOUT_MIN}" \
  --output="${OUT_DIR}" \
  "${ARGS[@]}"
