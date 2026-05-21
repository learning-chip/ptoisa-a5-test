# Python Wrapper — PTO A5 Simulator Smoke Tests

NumPy-first smoke tests for nine PTO operators, using the **CANN camodel** (`runtime_camodel`) instead of `torch_npu` or real NPU hardware.

The same **12 smoke cases** are also available via [cpp](../cpp/) and [torch_sim](../torch_sim/). See the [root README](../../README.md) for a comparison of all three paths.

See [wrapper_design_doc.md](wrapper_design_doc.md) for why this approach was chosen.

## Prerequisites

- CANN toolkit with `ASCEND_HOME_PATH` set
- `bisheng` compiler (bundled with CANN)
- Python 3 + dependencies:

```bash
pip install -r requirements.txt
```

- `third-party/pto-isa` submodule initialized

```bash
source ${ASCEND_HOME_PATH}/bin/setenv.bash
```

## Build

From this directory:

```bash
source ${ASCEND_HOME_PATH}/bin/setenv.bash
python3 -m common.build --all
```

Build one operator:

```bash
python3 -m common.build tadd
```

Artifacts:

| Output | Location |
|--------|----------|
| `pto_runtime.so` | `common/build/` |
| `lib<op>.so` | `<op>/build/` |

## Run smoke tests

```bash
source ${ASCEND_HOME_PATH}/bin/setenv.bash
python3 run_smoke.py
```

Each operator runs in its own process (same isolation model as separate cpp binaries). Expected: **12 passed**.

Run a single operator:

```bash
python3 tadd/test_tadd.py
```

## Layout

```
python_wrapper/
├── common/
│   ├── build.py           # bisheng (mirrors cpp bisheng_build.sh)
│   ├── sim_env.py         # LD_LIBRARY_PATH + re-exec for simulator libs
│   ├── pto_runtime.cpp    # pybind11 ACL bridge (-lruntime_camodel)
│   ├── pto_runtime.py     # Python helper
│   ├── numeric.py         # ResultCmp-style NumPy compare
│   ├── reporter.py        # pytest-style lines, no pytest
│   └── acc_golden_smoke.py
├── <op>/
│   ├── *_kernel.cpp       # copied from tests/cpp
│   ├── launch_api.cpp     # extern "C" smoke launchers
│   └── test_<op>.py       # prep + launch + verify in one file
├── run_smoke.py
└── wrapper_design_doc.md
```

## Smoke cases

| Test | Cases |
|------|-------|
| tadd | `case_float_64x64`, `case_int32_64x64` |
| tmax / tmul | `case_float_16x32`, `case_int32_16x32` |
| tload | `case_float_GT_128_128_VT_128_128_BLK1` |
| syncall | `case_hard_aiv_only_all_blocks` |
| tmov_acc2vec / textract / tinsert | `case_nz2nd_3` |
| tmov_acc2mat | `case_nz2nd_4` |

## Architecture

```
NumPy buffers
    → pto_runtime (pybind: aclInit, malloc, memcpy, stream)
    → ctypes → lib<op>.so (launch_api extern "C" → Launch* templates)
    → camodel simulator
```

Bisheng flags match `tests/cpp/common/bisheng_build.sh` (`dav-c310-vec` / `dav-c310`, `-DREGISTER_BASE`, `-lruntime_camodel`). **Do not** use jit_cpp `-DMEMORY_BASE` / `dav-2201` flags.

## See also

- [Root README](../../README.md) — three-path overview
- [cpp/README.md](../cpp/README.md) — C++ reference harness
- [torch_sim/README.md](../torch_sim/README.md) — torch_npu + msprof path
