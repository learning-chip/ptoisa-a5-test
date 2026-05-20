#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PTO_SCRIPT_DIR="${SCRIPT_DIR}"
export PTO_TEST_NAME=tinsert_acc2vec
export PTO_KERNEL_SOURCES="tinsert_acc2vec_kernel.cpp"
export PTO_KERNEL_ARCH=mix
source "${SCRIPT_DIR}/../common/bisheng_build.sh" "$@"
