#!/usr/bin/env bash
# Shared bisheng build helper for PTO A5 standalone tests.
# Configure via environment variables before sourcing:
#   PTO_TEST_NAME       - executable name (required)
#   PTO_KERNEL_SOURCES  - space-separated kernel .cpp files (required)
#   PTO_KERNEL_ARCH     - vec | mix (default: vec)
#   PTO_KERNEL_LIB      - output shared library name (default: lib${PTO_TEST_NAME}_kernel.so)
#   PTO_SCRIPT_DIR      - test directory containing sources (auto-set by wrapper)
set -euo pipefail

: "${PTO_TEST_NAME:?PTO_TEST_NAME is required}"
: "${PTO_KERNEL_SOURCES:?PTO_KERNEL_SOURCES is required}"

PTO_SCRIPT_DIR="${PTO_SCRIPT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "${PTO_SCRIPT_DIR}/../../.." && pwd)"
PTO_ISA_ROOT="${PROJECT_ROOT}/third-party/pto-isa"
BUILD_DIR="${PTO_SCRIPT_DIR}/build"
SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"
PTO_KERNEL_ARCH="${PTO_KERNEL_ARCH:-vec}"
PTO_KERNEL_LIB="${PTO_KERNEL_LIB:-lib${PTO_TEST_NAME}_kernel.so}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "ERROR: ASCEND_HOME_PATH is not set. Source CANN set_env.sh first." >&2
  exit 1
fi

ASCEND_DRIVER_PATH="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"
BISHENG="${ASCEND_HOME_PATH}/bin/bisheng"
if [[ ! -x "${BISHENG}" ]]; then
  BISHENG="$(command -v bisheng || true)"
fi
if [[ ! -x "${BISHENG}" ]]; then
  echo "ERROR: bisheng compiler not found." >&2
  exit 1
fi

FULL_BUILD=0
for arg in "$@"; do
  if [[ "${arg}" == "--full" ]]; then
    FULL_BUILD=1
  fi
done

case "${PTO_KERNEL_ARCH}" in
  vec) AICORE_ARCH="dav-c310-vec" ;;
  mix) AICORE_ARCH="dav-c310" ;;
  *) echo "ERROR: unknown PTO_KERNEL_ARCH=${PTO_KERNEL_ARCH}" >&2; exit 1 ;;
esac

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

COMMON_INCLUDES=(
  "-I${PTO_ISA_ROOT}/include"
  "-I${ASCEND_HOME_PATH}/include"
  "-I${ASCEND_DRIVER_PATH}/kernel/inc"
  "-I${PTO_SCRIPT_DIR}"
  "-I${PROJECT_ROOT}/tests/cpp/common"
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
  -mllvm -cce-aicore-stack-size=0x8000
  -mllvm -cce-aicore-function-stack-size=0x8000
  -mllvm -cce-aicore-record-overflow=true
  -mllvm -cce-aicore-addr-transform
  -mllvm -cce-aicore-dcci-insert-for-scalar=false
  "--cce-aicore-arch=${AICORE_ARCH}"
  -DREGISTER_BASE
)

if [[ "${FULL_BUILD}" -eq 1 ]]; then
  KERNEL_FLAGS+=(-DPTOISA_FULL_TEST)
fi

KERNEL_OBJECTS=()
for src in ${PTO_KERNEL_SOURCES}; do
  obj="$(basename "${src}" .cpp).o"
  echo "==> Compiling kernel: ${src}"
  "${BISHENG}" "${KERNEL_FLAGS[@]}" \
    -c "${PTO_SCRIPT_DIR}/${src}" \
    -o "${obj}"
  KERNEL_OBJECTS+=("${obj}")
done

echo "==> Linking kernel shared library: ${PTO_KERNEL_LIB}"
"${BISHENG}" -fPIC -shared --cce-fatobj-link \
  -Wl,-soname,"${PTO_KERNEL_LIB}" \
  "${KERNEL_OBJECTS[@]}" \
  -o "${PTO_KERNEL_LIB}"

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
  -c "${PTO_SCRIPT_DIR}/main.cpp" \
  -o main.o

SIMULATOR_LIB="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
if [[ ! -d "${SIMULATOR_LIB}" ]]; then
  echo "ERROR: simulator lib not found: ${SIMULATOR_LIB}" >&2
  exit 1
fi

LIB_NAME="${PTO_KERNEL_LIB#lib}"
LIB_NAME="${LIB_NAME%.so}"

echo "==> Linking executable: ${PTO_TEST_NAME}"
"${BISHENG}" main.o \
  -o "${PTO_TEST_NAME}" \
  -L. "-l${LIB_NAME}" \
  -L"${ASCEND_HOME_PATH}/lib64" \
  -L"${SIMULATOR_LIB}" \
  -lruntime_camodel \
  -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase -lpthread \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64:${SIMULATOR_LIB}:${BUILD_DIR}"

echo "Build complete: ${BUILD_DIR}/${PTO_TEST_NAME}"
