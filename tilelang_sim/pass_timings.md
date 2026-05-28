# PASS example simulator timings (A5 CPU sim)

Generated: 2026-05-28T04:00:31.648986+00:00

Measured via `msprof op simulator --soc-version=Ascend950PR_9599` with `target=pto`, `platform=A5`.

| Metric | Meaning |
|--------|---------|
| **sim_model_run_ms** | PEM `Model RUN TIME` — full simulator session (model start→stop), includes kernel launch + sim overhead |
| **sim_core_duration_us** | Max AICore `duration_time` from msprof core-operator table (actual simulated compute, typically µs) |
| **wall_time_s** | Host end-to-end time (JIT compile + msprof startup + sim session) |

Regenerate: `./run_pass_timings.sh` (uses [configs/pass_shapes.json](configs/pass_shapes.json) for input shapes).

| Example | Input shape(s) | sim_model_run_ms | sim_core_duration_us | wall_time_s |
|---------|----------------|------------------|----------------------|-------------|
| `activation/tanh.py` | case1: M=64, N=64, block_M=64, block_N=64, elements=4096, dtype=float32; case2: M=128, N=128, block_M=64, block_N=64, elements=16384, dtype=float32 | 32259.200 | 1.25 | 46.6 |
| `developer_mode/gemm_developer.py` | M=128, N=256, K=128, block_M=128, block_N=256, K_L1=64, dtype=float16 | 19192.100 | 3.68 | 34.3 |
| `developer_mode/matmul_add_developer.py` | M=128, N=256, K=128, block_M=128, block_N=256, block_K=64, dtype=float16 | 31327.900 | 8.24 | 46.5 |
| `elementwise/elementwise_add.py` | M=128, N=256, block_M=128, block_N=256, elements=32768, dtype=float32 | 28772.600 | 6.14 | 44.5 |
| `elementwise/elementwise_add_pipeline.py` | M=128, N=256, block_M=128, block_N=256, elements=32768, dtype=float32 | 20168.400 | 3.91 | 35.3 |
| `gemm/example_gemm.py` | M=128, N=256, K=128, block_M=128, block_N=256, K_L1=64, dtype=float16 | 19434.300 | 3.61 | 35.3 |
| `gemm/example_gemm_infer_scope.py` | M=128, N=256, K=128, block_M=128, block_N=256, K_L1=64, dtype=float16 | 19128.900 | 3.68 | 35.3 |
| `gemm/example_gemm_persistent.py` | M=128, N=256, K=128, block_M=128, block_N=256, K_L1=64, dtype=float16 | 19242.600 | 3.61 | 35.2 |
| `gemm/example_gemm_pto_developer.py` | M=128, N=256, K=128, block_M=128, block_N=256, K_L1=64, dtype=float16 | 19089.600 | 3.68 | 35.3 |
| `gemm/example_gemm_tail_block_developer.py` | case1: M=126, N=80, K=159, block_M=32, block_N=32, K_L1=32, dtype=float16; case2: M=557, N=512, K=539, block_M=64, block_N=64, K_L1=64, dtype=float16; case3: M=512, N=611, K=512, block_M=128, block_N=128, K_L1=128, dtype=float16; case4: M=1142, N=1230, K=1079, block_M=128, block_N=256, K_L1=64, dtype=float16; (Hardcoded test_configs in script (manifest CLI args ignored)) | 673155.000 | 2.60 | 690.0 |
| `pipeline/matmul_add_pipeline.py` | M=128, N=256, K=128, block_M=128, block_N=256, block_K=64, dtype=float16 | 33344.600 | 9.92 | 48.6 |
| `quant_batch_matmul/example_quant_batch_matmul.py` | case1: Batch=8, M=64, N=64, K=64, block_M=128, block_N=256, block_K=64; case2: Batch=8, M=64, N=64, K=64, block_M=128, block_N=256, block_K=64; (Batch defaults to 8 (--b not overridden in manifest)) | 244095.000 | 5.32 | 260.7 |
| `quant_batch_matmul/example_quant_matmul.py` | case1: M=64, N=64, K=64, block_M=128, block_N=256, block_K=64; case2: M=64, N=64, K=64, block_M=128, block_N=256, block_K=64 | 63487.200 | 5.32 | 78.5 |

## Per-example logs

- `ex_0014`: [results/pass_timings/ex_0014.log](results/pass_timings/ex_0014.log)
- `ex_0036`: [results/pass_timings/ex_0036.log](results/pass_timings/ex_0036.log)
- `ex_0037`: [results/pass_timings/ex_0037.log](results/pass_timings/ex_0037.log)
- `ex_0040`: [results/pass_timings/ex_0040.log](results/pass_timings/ex_0040.log)
- `ex_0041`: [results/pass_timings/ex_0041.log](results/pass_timings/ex_0041.log)
- `ex_0051`: [results/pass_timings/ex_0051.log](results/pass_timings/ex_0051.log)
- `ex_0052`: [results/pass_timings/ex_0052.log](results/pass_timings/ex_0052.log)
- `ex_0055`: [results/pass_timings/ex_0055.log](results/pass_timings/ex_0055.log)
- `ex_0056`: [results/pass_timings/ex_0056.log](results/pass_timings/ex_0056.log)
- `ex_0057`: [results/pass_timings/ex_0057.log](results/pass_timings/ex_0057.log)
- `ex_0096`: [results/pass_timings/ex_0096.log](results/pass_timings/ex_0096.log)
- `ex_0106`: [results/pass_timings/ex_0106.log](results/pass_timings/ex_0106.log)
- `ex_0107`: [results/pass_timings/ex_0107.log](results/pass_timings/ex_0107.log)
