#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PTO_TEST_DIR="${SCRIPT_DIR}"
export PTO_TEST_NAME=tmul
ARGS=( )

source "${SCRIPT_DIR}/../common/run_test.sh" "${ARGS[@]}" "$@"
