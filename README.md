# Standalone PTO-ISA A5 simulator tests

Minimal, out-of-source test harness for PTO-ISA NPU (A5) operator tests. Uses plain `bisheng` shell commands — no CMake, no Google Test.

## Layout

```
ptoisa-a5-test/
├── third-party/pto-isa/     # git submodule → https://gitcode.com/cann/pto-isa.git
└── tests/cpp/tadd/          # tadd smoke / regression test
    ├── build.sh             # raw bisheng compile + link
    ├── run.sh               # build + gen golden data + run
    ├── main.cpp             # plain C++ host driver
    ├── tadd_kernel.cpp      # AICore kernel
    ├── gen_data.py          # golden data generator
    └── test_common.h        # lightweight file I/O + result compare
```

## Prerequisites

1. CANN toolkit installed with `ASCEND_HOME_PATH` set (e.g. `/usr/local/Ascend/cann-9.0.0`)
2. `bisheng` compiler on `PATH` (bundled with CANN)
3. Python 3 + NumPy

```bash
source ${ASCEND_HOME_PATH}/bin/setenv.bash
```

## Clone with submodule

```bash
git clone --recurse-submodules <this-repo-url> ptoisa-a5-test
cd ptoisa-a5-test

# Or, if already cloned:
git submodule update --init --recursive
```

## Run smoke test (default)

Runs two fast cases (`float 64×64`, `int32 64×64`):

```bash
cd tests/cpp/tadd
./run.sh
```

Equivalent manual steps:

```bash
cd tests/cpp/tadd
./build.sh
python3 gen_data.py --output-dir build/cases
cd build
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH}"
export CAMODEL_LOG_PATH="$(pwd)/camodel_log"
mkdir -p camodel_log
./tadd
```

Expected output ends with:

```
[INFO]  Summary: 2 passed, 0 failed
```

## Full regression (optional)

Build all kernel instantiations and run all 14 cases from the upstream tadd suite:

```bash
cd tests/cpp/tadd
./run.sh --all
```

## Run a single case

```bash
cd tests/cpp/tadd
./run.sh --case case_float_64x64_64x64_64x64_64x64
```

## Design notes

| Upstream (`pto-isa` repo) | This project |
|---|---|
| Nested CMake + `pto_vec_st()` macro | Plain `build.sh` calling `bisheng` directly |
| Google Test (`TEST_F`) | Plain `main()` with `--case` / `--all` flags |
| 14 cases run by default | 2-case smoke test by default |
| In-tree under `tests/npu/a5/...` | Out-of-source; PTO headers via `third-party/pto-isa` submodule |

## Troubleshooting

- **`ASCEND_HOME_PATH is not set`**: Source CANN environment script first.
- **`simulator lib not found`**: Set `SOC_VERSION` (default `Ascend950PR_9599`) to match your CANN install.
- **Case not found in smoke build**: Rebuild with `./build.sh --full` for non-smoke cases.
