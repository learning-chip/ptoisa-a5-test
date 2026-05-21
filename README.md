# Standalone PTO-ISA A5 Simulator Tests

Minimal, out-of-source test harness for PTO-ISA NPU (A5) operator tests. Uses plain `bisheng` shell commands — no CMake, no Google Test.

## Layout

```
ptoisa-a5-test/
├── third-party/pto-isa/          # git submodule → https://gitcode.com/cann/pto-isa.git
├── tests/python_wrapper/         # NumPy + pybind + ctypes (no torch_npu) — see tests/python_wrapper/README.md
├── tests/torch_sim/              # torch_npu + msprof op simulator — see tests/torch_sim/README.md
├── tests/cpp/
│   ├── common/                   # shared headers and build helpers
│   │   ├── test_common.h         # file I/O + result compare
│   │   ├── test_runner.hpp       # plain C++ test runner (--case / --all)
│   │   ├── bisheng_build.sh      # parameterized bisheng compile/link
│   │   ├── sim_env.sh            # simulator LD_LIBRARY_PATH setup
│   │   └── run_test.sh           # build + gen_data + run wrapper
│   ├── run_smoke.sh              # run all smoke tests
│   ├── tadd/                     # elementwise add
│   ├── tmax/                     # elementwise max
│   ├── tmul/                     # elementwise mul
│   ├── tload/                    # ND tensor load
│   ├── syncall/                  # cross-core synchronization
│   ├── tmov_acc2vec/             # accumulator → vector tile move
│   ├── textract_acc2vec/         # extract from acc to vec
│   ├── tinsert_acc2vec/          # insert into vec tile
│   └── tmov_acc2mat/             # accumulator → matrix tile move
└── README.md
```

Each test directory contains:

| File | Purpose |
|------|---------|
| `*_kernel.cpp` | AICore kernel (from upstream, compiled with `bisheng -xcce`) |
| `main.cpp` | Host driver (plain `main`, no GTest) |
| `gen_data.py` | Golden data generator (smoke cases by default) |
| `build.sh` | Calls `common/bisheng_build.sh` |
| `run.sh` | Build, generate data, run in simulator |

## Prerequisites

1. CANN toolkit with `ASCEND_HOME_PATH` set
2. `bisheng` compiler (bundled with CANN)
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

## Run all smoke tests

**Python (recommended when no real NPU):**

```bash
cd tests/python_wrapper
pip install -r requirements.txt
source ${ASCEND_HOME_PATH}/bin/setenv.bash
python3 -m common.build --all
python3 run_smoke.py
```

Expected: `12 passed` (pytest-style output). Uses camodel via `pto_runtime` pybind — no `torch` / `torch_npu`. See [tests/python_wrapper/wrapper_design_doc.md](tests/python_wrapper/wrapper_design_doc.md).

**torch_npu + msprof (portable to real NPU):**

```bash
cd tests/torch_sim
pip install -r requirements.txt
source ${ASCEND_HOME_PATH}/bin/setenv.bash
python3 -m common.build --all
./run_smoke.sh
```

Expected: **12 passed** (same cases). No `-lruntime_camodel` link; `msprof op simulator` drives torch_npu. See [tests/torch_sim/torch_sim_design_doc.md](tests/torch_sim/torch_sim_design_doc.md).

**C++ (alternative):**

```bash
cd tests/cpp
./run_smoke.sh
```

Expected: 9 test binaries, each reporting `Summary: N passed, 0 failed`.

## Run a single test

```bash
cd tests/cpp/tadd
./run.sh
```

## Smoke test cases (default)

| Test | Smoke case(s) | Notes |
|------|---------------|-------|
| `tadd` | `case_float_64x64_*`, `case_int32_64x64_*` | 2 cases |
| `tmax` | `case_float_16x32_*`, `case_int32_16x32_*` | 2 cases |
| `tmul` | `case_float_16x32_*`, `case_int32_16x32_*` | 2 cases |
| `tload` | `case_float_GT_128_128_VT_128_128_BLK1` | golden in kernel |
| `syncall` | `case_hard_aiv_only_all_blocks` | inline golden |
| `tmov_acc2vec` | `case_nz2nd_3` | smallest ND move |
| `textract_acc2vec` | `case_nz2nd_3` | smallest extract |
| `tinsert_acc2vec` | `case_nz2nd_3` | smallest insert |
| `tmov_acc2mat` | `case_nz2nd_4` | smallest ND mat move |

## Full regression (optional)

```bash
cd tests/cpp/tadd
./run.sh --all    # requires: ./build.sh --full first (or run.sh --all does both)
```

`--all` rebuilds with all kernel template instantiations and runs the full upstream case list.

## Manual bisheng build (example: tadd)

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
source ${ASCEND_HOME_PATH}/bin/setenv.bash
cd tests/cpp/tadd
./build.sh

# Kernel (vec arch):
# bisheng -xcce --cce-aicore-arch=dav-c310-vec ... -c tadd_kernel.cpp -o tadd_kernel.o
# bisheng -fPIC -shared --cce-fatobj-link ... -o libtadd_kernel.so tadd_kernel.o

# Host:
# bisheng -xc++ ... -c main.cpp -o main.o
# bisheng main.o -o tadd -L. -ltadd_kernel -lruntime_camodel ...
```

Mix-arch tests (`tmov_acc2vec`, `textract_acc2vec`, etc.) use `--cce-aicore-arch=dav-c310` instead of `dav-c310-vec`.

## Design vs upstream `pto-isa`

| Upstream | This project |
|----------|--------------|
| Nested CMake + `pto_vec_st()` / `pto_mix_st()` | `build.sh` → `bisheng` directly |
| Google Test (`TEST_F`) | Plain `main()` + `test_runner.hpp` |
| 10–50+ cases per binary by default | 1–2 smoke cases by default |
| In-tree under `tests/npu/a5/...` | Out-of-source via `third-party/pto-isa` submodule |

## Troubleshooting

- **`ASCEND_HOME_PATH is not set`**: Source CANN `setenv.bash` first.
- **`simulator lib not found`**: Set `SOC_VERSION` (default `Ascend950PR_9599`) to match your CANN install.
- **Case not found in smoke build**: Rebuild with `./build.sh --full` for non-smoke cases.
- **Missing golden files**: Run `python3 gen_data.py --output-dir build/cases` in the test directory.
