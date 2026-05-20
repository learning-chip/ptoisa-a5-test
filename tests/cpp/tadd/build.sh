#!/usr/bin/env bash
# Standalone bisheng build for tadd (no CMake, no GTest).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PTO_ISA_ROOT="${PROJECT_ROOT}/third-party/pto-isa"
BUILD_DIR="${SCRIPT_DIR}/build"
SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "ERROR: ASCEND_HOME_PATH is not set. Source CANN set_env.sh first." >&2
  exit 1
fi

ASCEND_DRIVER_PATH="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"
BISHENG="${ASCEND_HOME_PATH}/bin/bisheng"
if [[ ! -x "${BISHENG}" ]]; then
  BISHENG="$(command -v bisheng || true)"
fi
if [[ -z "${BISHENG}" || ! -x "${BISHENG}" ]]; then
  echo "ERROR: bisheng compiler not found." >&2
  exit 1
fi

FULL_BUILD=0
for arg in "$@"; do
  if [[ "${arg}" == "--full" ]]; then
    FULL_BUILD=1
  fi
done

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

COMMON_INCLUDES=(
  "-I${PTO_ISA_ROOT}/include"
  "-I${ASCEND_HOME_PATH}/include"
  "-I${ASCEND_DRIVER_PATH}/kernel/inc"
  "-I${SCRIPT_DIR}"
)

COMMON_CXX_FLAGS=(
  -std=c++17
  -O2
  -Wno-macro-redefined
  -Wno-ignored-attributes
  -Wno-unknown-attributes
)

KERNEL_FLAGS=(
  "${COMMON_INCLUDES[@]}"
  "-I${ASCEND_HOME_PATH}/pkg_inc"
  "-I${ASCEND_HOME_PATH}/pkg_inc/profiling"
  "-I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime"
  "${COMMON_CXX_FLAGS[@]}"
  -fPIC
  -xcce
  -Xhost-start
  -Xhost-end
  -mllvm
  -cce-aicore-stack-size=0x8000
  -mllvm
  -cce-aicore-function-stack-size=0x8000
  -mllvm
  -cce-aicore-record-overflow=true
  -mllvm
  -cce-aicore-addr-transform
  -mllvm
  -cce-aicore-dcci-insert-for-scalar=false
  --cce-aicore-arch=dav-c310-vec
  -DREGISTER_BASE
)

if [[ "${FULL_BUILD}" -eq 1 ]]; then
  KERNEL_FLAGS+=(-DPTOISA_FULL_TEST)
fi

echo "==> Compiling kernel: tadd_kernel.cpp"
"${BISHENG}" "${KERNEL_FLAGS[@]}" \
  -c "${SCRIPT_DIR}/tadd_kernel.cpp" \
  -o tadd_kernel.o

echo "==> Linking kernel shared library: libtadd_kernel.so"
"${BISHENG}" -fPIC -shared --cce-fatobj-link \
  -Wl,-soname,libtadd_kernel.so \
  tadd_kernel.o \
  -o libtadd_kernel.so

HOST_FLAGS=(
  "${COMMON_INCLUDES[@]}"
  "${COMMON_CXX_FLAGS[@]}"
  -xc++
  -include stdint.h
  -include stddef.h
)

if [[ "${FULL_BUILD}" -eq 1 ]]; then
  HOST_FLAGS+=(-DPTOISA_FULL_TEST)
fi

echo "==> Compiling host: main.cpp"
"${BISHENG}" "${HOST_FLAGS[@]}" \
  -c "${SCRIPT_DIR}/main.cpp" \
  -o main.o

SIMULATOR_LIB="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
if [[ ! -d "${SIMULATOR_LIB}" ]]; then
  echo "ERROR: simulator lib not found: ${SIMULATOR_LIB}" >&2
  exit 1
fi

echo "==> Linking executable: tadd"
"${BISHENG}" main.o \
  -o tadd \
  -L. -ltadd_kernel \
  -L"${ASCEND_HOME_PATH}/lib64" \
  -L"${SIMULATOR_LIB}" \
  -lruntime_camodel \
  -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase -lpthread \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64:${SIMULATOR_LIB}:${BUILD_DIR}"

echo "Build complete: ${BUILD_DIR}/tadd"
