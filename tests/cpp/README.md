# C++ — PTO A5 Simulator Smoke Tests

Reference C++ harness for nine PTO operators. Each test is a standalone binary: host `main()` drives ACL device APIs, launches bisheng-built AICore kernels, and compares against golden data. Simulation comes from linking **`-lruntime_camodel`**.

The same **12 smoke cases** are also available via [python_wrapper](../python_wrapper/) and [torch_sim](../torch_sim/). See the [root README](../../README.md) for a comparison of all three paths.

## Prerequisites

- CANN toolkit with `ASCEND_HOME_PATH` set
- `bisheng` compiler (bundled with CANN)
- Python 3 + NumPy (golden generation via `gen_data.py`)
- `third-party/pto-isa` submodule initialized

```bash
source ${ASCEND_HOME_PATH}/bin/setenv.bash
```

## Run all smoke tests

```bash
cd tests/cpp
./run_smoke.sh
```

Runs nine operators sequentially. Expected: each binary reports `Summary: N passed, 0 failed`.

## Run a single test

```bash
cd tests/cpp/tadd
./run.sh
```

## Layout

```
tests/cpp/
├── common/
│   ├── test_common.h         # file I/O + result compare
│   ├── test_runner.hpp       # plain C++ test runner (--case / --all)
│   ├── bisheng_build.sh      # parameterized bisheng compile/link
│   ├── sim_env.sh            # simulator LD_LIBRARY_PATH setup
│   └── run_test.sh           # build + gen_data + run wrapper
├── run_smoke.sh              # run all smoke tests
└── <op>/                     # tadd, tmax, tmul, tload, syncall, acc2 ops
    ├── *_kernel.cpp          # AICore kernel (bisheng -xcce)
    ├── main.cpp              # host driver
    ├── gen_data.py           # golden data generator
    ├── build.sh              # calls common/bisheng_build.sh
    └── run.sh                # build, generate data, run in simulator
```

## Smoke cases

| Test | Smoke case(s) | Count |
|------|---------------|-------|
| `tadd` | `case_float_64x64_*`, `case_int32_64x64_*` | 2 |
| `tmax`, `tmul` | `case_float_16x32_*`, `case_int32_16x32_*` | 2 each |
| `tload` | `case_float_GT_128_128_VT_128_128_BLK1` | 1 |
| `syncall` | `case_hard_aiv_only_all_blocks` | 1 |
| `tmov_acc2vec`, `textract_acc2vec`, `tinsert_acc2vec` | `case_nz2nd_3` | 1 each |
| `tmov_acc2mat` | `case_nz2nd_4` | 1 |

## Full regression (optional)

```bash
cd tests/cpp/tadd
./run.sh --all    # or: ./build.sh --full && ./run.sh --all
```

`--all` rebuilds with all kernel template instantiations and runs the full upstream case list (not just smoke).

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

Mix-arch tests (`tmov_acc2vec`, `textract_acc2vec`, `tinsert_acc2vec`, `tmov_acc2mat`) use `--cce-aicore-arch=dav-c310` instead of `dav-c310-vec`.

## Architecture

```
main.cpp (ACL: aclInit, malloc, memcpy, stream)
    → Launch* kernel templates
    → -lruntime_camodel (CPU cycle-accurate simulator)
```

Bisheng flags are centralized in `common/bisheng_build.sh`. Python wrapper and torch_sim builds mirror these kernel flags.

## Troubleshooting

- **`ASCEND_HOME_PATH is not set`**: Source CANN `setenv.bash` first.
- **`simulator lib not found`**: Set `SOC_VERSION` (default `Ascend950PR_9599`) to match your CANN install; `common/sim_env.sh` configures `LD_LIBRARY_PATH`.
- **Case not found in smoke build**: Rebuild with `./build.sh --full` for non-smoke cases.
- **Missing golden files**: Run `python3 gen_data.py --output-dir build/cases` in the test directory.

## See also

- [Root README](../../README.md) — three-path overview
- [python_wrapper/README.md](../python_wrapper/README.md) — NumPy + pybind port of the same cases
- [torch_sim/README.md](../torch_sim/README.md) — torch_npu + msprof port
