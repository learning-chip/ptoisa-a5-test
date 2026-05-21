# torch_sim — PTO A5 Smoke Tests (torch_npu + msprof)

Same nine operators and **12 smoke cases** as `tests/cpp` and `tests/python_wrapper`, executed via **torch_npu** device tensors and **ctypes** kernel launches under **`msprof op simulator`** (no `-lruntime_camodel` link).

See [torch_sim_design_doc.md](torch_sim_design_doc.md) for why this is the portable real-NPU-ready path.

## Prerequisites

- CANN 9.0+ with `ASCEND_HOME_PATH` set
- `bisheng` compiler (bundled with CANN)
- Python 3 + dependencies:

```bash
pip install -r requirements.txt
```

- `third-party/pto-isa` submodule initialized (same as cpp tests)
- `torch_npu` 2.9.x matching your CANN version

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

Artifacts: `lib<op>.so` in `<op>/build/` (no `pto_runtime` pybind module).

## Run smoke tests (msprof simulator)

Primary runner — one msprof invocation per operator:

```bash
source ${ASCEND_HOME_PATH}/bin/setenv.bash
chmod +x run_smoke.sh
./run_smoke.sh
```

Or aggregate pytest-style output via `run_smoke.py` (wraps direct execution, useful after `./run_direct.sh` or for debugging):

```bash
python3 run_smoke.py
```

Expected: **12 passed** (same cases as `tests/cpp/run_smoke.sh`).

Run a single operator:

```bash
msprof op simulator --soc-version=Ascend950PR_9599 \
  --output=msprof_res/tadd python3 tadd/test_tadd.py
```

## Run without msprof (real NPU / future hardware)

```bash
chmod +x run_direct.sh
./run_direct.sh
```

Same test scripts; no camodel link required. Use this path when Ascend950 hardware is available.

## Layout

```
torch_sim/
├── common/
│   ├── build.py           # bisheng (no runtime_camodel)
│   ├── torch_runtime.py   # torch_npu init, tensors, stream ptr
│   ├── numeric.py
│   ├── reporter.py
│   ├── ctypes_utils.py
│   └── acc_golden_smoke.py
├── <op>/
│   ├── *_kernel.cpp
│   ├── launch_api.cpp
│   └── test_<op>.py
├── run_smoke.sh           # msprof per operator
├── run_smoke.py           # aggregate PASSED/FAILED
├── run_direct.sh          # no msprof
└── torch_sim_design_doc.md
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

## Three execution paths

| Path | Runtime | Link model | Best for |
|------|---------|------------|----------|
| `tests/cpp` | ACL C++ | `-lruntime_camodel` | Reference / bisheng flags |
| `tests/python_wrapper` | pybind ACL + NumPy | `-lruntime_camodel` | Lightweight Python, no torch |
| `tests/torch_sim` | torch_npu + ctypes | no camodel | Same code path as future real NPU |

## Architecture

```
torch.zeros(..., device=npu)
    → torch_npu (stream, data_ptr)
    → ctypes → lib<op>.so (launch_api extern "C")
    → msprof op simulator (Ascend950PR_9599)
```

Bisheng kernel flags match `tests/cpp/common/bisheng_build.sh`. Host link uses `-lstdc++` only — **not** `-lruntime_camodel`.
