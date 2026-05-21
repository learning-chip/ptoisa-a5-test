# Standalone PTO-ISA A5 Simulator Tests

Minimal, out-of-source test harness for nine PTO-ISA NPU (A5) operator smoke tests. No CMake, no Google Test. The same **12 smoke cases** can be run three ways — pick the path that matches your stack.

For the smoothest setup, use the [950 Dockerfile](https://github.com/learning-chip/agent_docker_npu/pull/8). No real NPU is required for the C++ or python_wrapper paths; torch_sim uses the `msprof op simulator`.

## Three execution paths

| Path | Directory | Runtime | How simulation works | Best for |
|------|-----------|---------|----------------------|----------|
| **C++** | [`tests/cpp`](tests/cpp/) | ACL C++ `main()` | Link `-lruntime_camodel` | Reference harness, bisheng flags |
| **Python wrapper** | [`tests/python_wrapper`](tests/python_wrapper/) | NumPy + pybind ACL + ctypes | `-lruntime_camodel` in `pto_runtime.so` | Lightweight Python, no torch |
| **torch_sim** | [`tests/torch_sim`](tests/torch_sim/) | torch_npu + ctypes | `msprof op simulator` (no camodel link) | Same stack as future real NPU |

All three compile the same A5 kernels (`dav-c310-vec` / `dav-c310`, `-DREGISTER_BASE`) from sources copied under each test tree, with headers from `third-party/pto-isa`.

## Operators and smoke cases

Nine operators, **12 smoke cases** total — identical across all paths:

| Test | Smoke case(s) | Count |
|------|---------------|-------|
| `tadd` | float / int32 64×64 | 2 |
| `tmax`, `tmul` | float / int32 16×32 | 2 each |
| `tload` | `case_float_GT_128_128_VT_128_128_BLK1` | 1 |
| `syncall` | `case_hard_aiv_only_all_blocks` | 1 |
| `tmov_acc2vec`, `textract_acc2vec`, `tinsert_acc2vec` | `case_nz2nd_3` | 1 each |
| `tmov_acc2mat` | `case_nz2nd_4` | 1 |

## Repository layout

```
ptoisa-a5-test/
├── third-party/pto-isa/       # git submodule → https://gitcode.com/cann/pto-isa.git
├── tests/cpp/                 # C++ reference → tests/cpp/README.md
├── tests/python_wrapper/      # NumPy + pybind → tests/python_wrapper/README.md
├── tests/torch_sim/           # torch_npu + msprof → tests/torch_sim/README.md
└── README.md
```

## Prerequisites (shared)

1. CANN toolkit with `ASCEND_HOME_PATH` set
2. `bisheng` compiler (bundled with CANN)
3. `third-party/pto-isa` submodule initialized

```bash
git clone --recurse-submodules <this-repo-url> ptoisa-a5-test
cd ptoisa-a5-test

# Or, if already cloned:
git submodule update --init --recursive

source ${ASCEND_HOME_PATH}/bin/setenv.bash
```

Additional deps per path:

| Path | Extra requirements |
|------|-------------------|
| `tests/cpp` | Python 3 + NumPy (golden generation only) |
| `tests/python_wrapper` | `pip install -r tests/python_wrapper/requirements.txt` |
| `tests/torch_sim` | `pip install -r tests/torch_sim/requirements.txt` (`torch`, `torch_npu`) |

## Quick start

Each path has its own README with build details, layout, and troubleshooting.

### C++ (reference)

```bash
cd tests/cpp
./run_smoke.sh
```

Expected: 9 binaries, each `Summary: N passed, 0 failed`. See [tests/cpp/README.md](tests/cpp/README.md).

### Python wrapper (NumPy, no torch)

```bash
cd tests/python_wrapper
pip install -r requirements.txt
python3 -m common.build --all
python3 run_smoke.py
```

Expected: **12 passed**. See [tests/python_wrapper/README.md](tests/python_wrapper/README.md) and [wrapper_design_doc.md](tests/python_wrapper/wrapper_design_doc.md).

### torch_sim (torch_npu + msprof)

```bash
cd tests/torch_sim
pip install -r requirements.txt
python3 -m common.build --all
./run_smoke.sh
```

Expected: **12 passed**. See [tests/torch_sim/README.md](tests/torch_sim/README.md) and [torch_sim_design_doc.md](tests/torch_sim/torch_sim_design_doc.md).

## Design vs upstream `pto-isa`

| Upstream | This project |
|----------|--------------|
| Nested CMake + `pto_vec_st()` / `pto_mix_st()` | Plain `bisheng` shell / Python build scripts |
| Google Test (`TEST_F`) | Smoke runners with numeric golden compare |
| 10–50+ cases per binary by default | 12 shared smoke cases by default |
| In-tree under `tests/npu/a5/...` | Out-of-source via `third-party/pto-isa` submodule |

## Further reading

- [tests/cpp/README.md](tests/cpp/README.md) — per-test layout, single-test runs, full regression, bisheng flags
- [tests/python_wrapper/wrapper_design_doc.md](tests/python_wrapper/wrapper_design_doc.md) — why pybind + camodel instead of torch_npu alone
- [tests/torch_sim/torch_sim_design_doc.md](tests/torch_sim/torch_sim_design_doc.md) — why msprof enables torch_npu without hardware
