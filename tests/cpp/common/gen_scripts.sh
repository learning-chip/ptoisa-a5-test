#!/usr/bin/env bash
# Generate build.sh and run.sh for all standalone tests.
set -euo pipefail

ROOT="/workdir/ptoisa-a5-test/tests/cpp"

write_build() {
  local name="$1" arch="$2" sources="$3"
  cat > "${ROOT}/${name}/build.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export PTO_SCRIPT_DIR="\${SCRIPT_DIR}"
export PTO_TEST_NAME=${name}
export PTO_KERNEL_SOURCES="${sources}"
export PTO_KERNEL_ARCH=${arch}
source "\${SCRIPT_DIR}/../common/bisheng_build.sh" "\$@"
EOF
  chmod +x "${ROOT}/${name}/build.sh"
}

write_run() {
  local name="$1" skip_gen="${2:-0}"
  cat > "${ROOT}/${name}/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export PTO_TEST_DIR="\${SCRIPT_DIR}"
export PTO_TEST_NAME=${name}
ARGS=( )
$([[ "${skip_gen}" == "1" ]] && echo 'ARGS+=(--skip-gen)')
source "\${SCRIPT_DIR}/../common/run_test.sh" "\${ARGS[@]}" "\$@"
EOF
  chmod +x "${ROOT}/${name}/run.sh"
}

write_build tadd vec "tadd_kernel.cpp"
write_run tadd

write_build tmax vec "tmax_kernel.cpp"
write_run tmax

write_build tmul vec "tmul_kernel.cpp"
write_run tmul

write_build tload vec "tload_kernel.cpp"
write_run tload 1

write_build syncall vec "syncall_hard_kernel.cpp"
write_run syncall 1

write_build tmov_acc2vec mix "tmov_acc2vec_kernel.cpp"
write_run tmov_acc2vec

write_build textract_acc2vec mix "textract_acc2vec_kernel.cpp"
write_run textract_acc2vec

write_build tinsert_acc2vec mix "tinsert_acc2vec_kernel.cpp"
write_run tinsert_acc2vec

write_build tmov_acc2mat mix "tmov_acc2mat_kernel.cpp"
write_run tmov_acc2mat

echo "Generated build.sh and run.sh for all tests"
