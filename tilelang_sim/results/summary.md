# TileLang-Ascend A5 CPU Sim Results

Generated: 2026-05-28T03:35:30.473355+00:00

## Executive summary

- **Total examples**: 127 (from `bench_test.sh` inventory)
- **Runnable under PTO+A5 sim**: 115
- **PASS** (no crash + numerical check): **13**
- **FAIL**: 102 (crash=86, compile=6, accuracy=5, timeout=5)
- **SKIP** (out of scope): 12

JIT settings enforced by `common/bootstrap.py`: `target="pto"`, `platform="A5"` (A5 libgen branch: `dav-c310`, `-DREGISTER_BASE`).

### PASS examples

- `activation/tanh.py`
- `developer_mode/gemm_developer.py`
- `developer_mode/matmul_add_developer.py`
- `elementwise/elementwise_add.py`
- `elementwise/elementwise_add_pipeline.py`
- `gemm/example_gemm.py`
- `gemm/example_gemm_infer_scope.py`
- `gemm/example_gemm_persistent.py`
- `gemm/example_gemm_pto_developer.py`
- `gemm/example_gemm_tail_block_developer.py`
- `pipeline/matmul_add_pipeline.py`
- `quant_batch_matmul/example_quant_batch_matmul.py`
- `quant_batch_matmul/example_quant_matmul.py`

## Summary counts

| Status | Count |
|--------|-------|
| FAIL_ACCURACY | 5 |
| FAIL_COMPILE | 6 |
| FAIL_CRASH | 86 |
| FAIL_TIMEOUT | 5 |
| PASS | 13 |
| SKIP | 12 |

## Per-example results

| Example | Status | Duration (s) | Notes |
|---------|--------|--------------|-------|
| `[bench_sfa] sparse_flash_attn_pa` | SKIP | 0.0 | bench_sfa performance harness, not correctness smoke |
| `[bench_sfa] sparse_flash_attn_pa_baseline` | SKIP | 0.0 | bench_sfa performance harness, not correctness smoke |
| `[bench_sfa] sparse_flash_attn_pa_developer` | SKIP | 0.0 | bench_sfa performance harness, not correctness smoke |
| `[bench_sfa] sparse_flash_attn_pa_no_cv_pipeline` | SKIP | 0.0 | bench_sfa performance harness, not correctness smoke |
| `aclgraph/rms_rope_aclgraph.py` | FAIL_CRASH | 7.8 | SystemExit: 2 |
| `activation/gelu_grad.py` | FAIL_CRASH | 157.3 | RuntimeError: operator():../third_party/op-plugin/op_plugin/ops/opapi/StructKernelNpuOpApi.cpp:3600 NPU function error:  |
| `activation/gelu_mul.py` | FAIL_COMPILE | 10.1 | RuntimeError: Compilation Failed! ['bisheng', '--cce-aicore-arch=dav-c310', '-DREGISTER_BASE', '-O2', '-std=gnu++17', '- |
| `activation/sigmoid.py` | FAIL_CRASH | 49.4 | RuntimeError: sigmoid:../third_party/op-plugin/op_plugin/ops/opapi/StructKernelNpuOpApi.cpp:2918 NPU function error: dev |
| `activation/sigmoidv2.py` | FAIL_CRASH | 15.9 | RuntimeError: sigmoid:../third_party/op-plugin/op_plugin/ops/opapi/StructKernelNpuOpApi.cpp:2918 NPU function error: dev |
| `activation/sigmoidv2_slice.py` | FAIL_CRASH | 17.2 | RuntimeError: sigmoid:../third_party/op-plugin/op_plugin/ops/opapi/StructKernelNpuOpApi.cpp:2918 NPU function error: dev |
| `activation/silu.py` | FAIL_CRASH | 49.5 | RuntimeError: sigmoid:../third_party/op-plugin/op_plugin/ops/opapi/StructKernelNpuOpApi.cpp:2918 NPU function error: dev |
| `activation/swi_glu.py` | FAIL_CRASH | 37.7 | RuntimeError: silu:../third_party/op-plugin/op_plugin/ops/opapi/StructKernelNpuOpApi.cpp:3006 NPU function error: device |
| `activation/swi_glu_grad.py` | FAIL_TIMEOUT | 1800.0 | msprof 30-minute timeout or simulator abort (status 139/134) |
| `activation/swi_glu_v2.py` | FAIL_TIMEOUT | 1800.0 | msprof 30-minute timeout or simulator abort (status 139/134) |
| `activation/tanh.py` | PASS | 40.1 |  |
| `autotune/example_gemm_autotune.py` | SKIP | 0.0 | autotune search exceeds sim time budget |
| `autotune/example_gemm_carver.py` | SKIP | 0.0 | autotune search exceeds sim time budget |
| `batch_gemm/batch_gemm.py` | FAIL_CRASH | 8.0 | tvm.error.DiagnosticError: Traceback (most recent call last): |
| `blocksparse_gemm/example_blocksparse_gemm.py` | FAIL_CRASH | 11.9 | RuntimeError: gt:../third_party/op-plugin/op_plugin/ops/opapi/GtKernelNpuOpApi.cpp:42 NPU function error: device error t |
| `causal_conv1d/causal_conv1d.py` | FAIL_CRASH | 8.3 | RuntimeError: operator():../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: device error type  |
| `causal_conv1d/causal_conv1d_pto.py` | FAIL_CRASH | 8.3 | RuntimeError: operator():../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: device error type  |
| `chunk_gated_delta_rule/chunk_gated_delta_rule.py` | FAIL_CRASH | 7.9 | RuntimeError: exit code 0 without success marker |
| `chunk_gated_delta_rule/expert_chunk_gated_delta_rule.py` | FAIL_CRASH | 7.8 | RuntimeError: exit code 0 without success marker |
| `convolution/example_convolution.py` | FAIL_CRASH | 60.1 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `convolution/example_convolution_autotune.py` | FAIL_CRASH | 174.8 | RuntimeError: Auto-tuning failed: No configuration successfully compiled and passed benchmarking/validation. |
| `cross_entropy_loss/example_cross_entro.py` | FAIL_TIMEOUT | 1800.0 | msprof 30-minute timeout or simulator abort (status 139/134) |
| `cumsum_gdn/example_cumsum.py` | FAIL_COMPILE | 10.3 | RuntimeError: Compilation Failed! ['bisheng', '--cce-aicore-arch=dav-c310', '-DREGISTER_BASE', '-O2', '-std=gnu++17', '- |
| `cumsum_kda/example_cumsum_kda.py` | FAIL_COMPILE | 10.3 | RuntimeError: Compilation Failed! ['bisheng', '--cce-aicore-arch=dav-c310', '-DREGISTER_BASE', '-O2', '-std=gnu++17', '- |
| `deepseek_v4/act_quant.py` | FAIL_CRASH | 120.0 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `deepseek_v4/hc_split_sinkhorn.py` | FAIL_CRASH | 77.7 | RuntimeError: sigmoid:../third_party/op-plugin/op_plugin/ops/opapi/StructKernelNpuOpApi.cpp:2918 NPU function error: dev |
| `deepseek_v4/int8_gemm.py` | FAIL_CRASH | 7.9 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `deepseek_v4/sparse_attention.py` | FAIL_CRASH | 8.1 | RuntimeError: random_op_api_:../third_party/op-plugin/op_plugin/ops/opapi/RandomKernelNpuOpApi.cpp:124 NPU function erro |
| `dequantize_gemm/example_dequant_gemm_fine_grained.py` | FAIL_CRASH | 421.5 | tvm.error.DiagnosticError: Traceback (most recent call last): |
| `dequantize_gemm/example_dequant_gemm_w4a8.py` | FAIL_CRASH | 7.8 | SystemExit: 2 |
| `developer_mode/flash_attn_bshd_developer.py` | FAIL_CRASH | 132.0 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `developer_mode/gelu_mul_developer.py` | FAIL_CRASH | 47.4 | RuntimeError: gelu_out:../third_party/op-plugin/op_plugin/ops/opapi/GeluOutKernelNpuOpApi.cpp:26 NPU function error: dev |
| `developer_mode/gemm_developer.py` | PASS | 26.9 |  |
| `developer_mode/matmul_add_developer.py` | PASS | 39.3 |  |
| `developer_mode/sparse_flash_attn_developer.py` | FAIL_CRASH | 13.2 | RuntimeError: randperm_op_api:../third_party/op-plugin/op_plugin/ops/opapi/RandpermKernelNpuOpApi.cpp:28 NPU function er |
| `developer_mode/sparse_flash_attn_developer_vid_reduce.py` | FAIL_CRASH | 13.2 | RuntimeError: randperm_op_api:../third_party/op-plugin/op_plugin/ops/opapi/RandpermKernelNpuOpApi.cpp:28 NPU function er |
| `elementwise/elementwise_add.py` | PASS | 36.6 |  |
| `elementwise/elementwise_add_pipeline.py` | PASS | 28.2 |  |
| `elementwise/setvalue_example.py` | FAIL_CRASH | 11.9 | RuntimeError: arange_out_op_api:../third_party/op-plugin/op_plugin/ops/opapi/ArangeKernelNpuOpApi.cpp:36 NPU function er |
| `flash_attention/fa_opt/flash_attn_bhsd_ascendc.py` | SKIP | 0.0 | uses torch_npu.npu_fusion_attention, not TileLang JIT |
| `flash_attention/fa_opt/flash_attn_bhsd_auto_pipeline_h16_d128.py` | FAIL_ACCURACY | 7.8 | AssertionError: dim must be 128 |
| `flash_attention/fa_opt/flash_attn_bhsd_auto_pipeline_h32_d512.py` | FAIL_CRASH | 46.3 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py` | FAIL_ACCURACY | 7.8 | AssertionError: dim must be 128 |
| `flash_attention/flash_attn_bhsd.py` | FAIL_CRASH | 180.0 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `flash_attention/flash_attn_bhsd_cc_sync.py` | FAIL_CRASH | 166.5 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `flash_attention/paged_flash_attn_bhsd.py` | FAIL_CRASH | 7.9 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `fused_sigmoid_gating_delta_rule/fused_sigmoid_gating_delta_rule_varlen.py` | FAIL_CRASH | 7.8 | RuntimeError: exit code 0 without success marker |
| `gemm/example_gemm.py` | PASS | 27.0 |  |
| `gemm/example_gemm_infer_scope.py` | PASS | 26.9 |  |
| `gemm/example_gemm_intrinsic.py` | SKIP | 0.0 | target=ascendc out of scope |
| `gemm/example_gemm_intrinsic_persistent.py` | SKIP | 0.0 | target=ascendc out of scope |
| `gemm/example_gemm_persistent.py` | PASS | 26.8 |  |
| `gemm/example_gemm_pto_developer.py` | PASS | 26.9 |  |
| `gemm/example_gemm_tail_block_developer.py` | PASS | 693.5 |  |
| `gemm/example_gemm_transpose_l1.py` | FAIL_COMPILE | 10.1 | RuntimeError: Compilation Failed! ['bisheng', '--cce-aicore-arch=dav-c310', '-DREGISTER_BASE', '-O2', '-std=gnu++17', '- |
| `gemm_aot/run_example_gemm_aot.sh` | SKIP | 0.0 | AOT ctypes workflow, not @tilelang.jit PTO |
| `gemm_splitk/example_tilelang_gemm_splitk.py` | FAIL_CRASH | 7.7 | SystemExit: 2 |
| `gemv/example_gemv_c.py` | FAIL_CRASH | 37.4 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `gemv/example_gemv_v.py` | FAIL_CRASH | 238.6 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `grouped_gemm/example_grouped_gemm_bwd.py` | FAIL_CRASH | 7.7 | SystemExit: 2 |
| `grouped_gemm/example_grouped_gemm_fwd.py` | FAIL_CRASH | 7.9 | SystemExit: 2 |
| `grouped_gemm/example_grouped_gemm_fwd_ptr.py` | FAIL_CRASH | 7.7 | SystemExit: 2 |
| `hadamard_transform/example_hadamard_transform.py` | FAIL_CRASH | 7.7 | SystemExit: 2 |
| `lightning_indexer/example_lightning_indexer.py` | FAIL_TIMEOUT | 1800.0 | msprof 30-minute timeout or simulator abort (status 139/134) |
| `lightning_indexer/example_lightning_indexer_dynamic_shape.py` | FAIL_CRASH | 8.1 | e const*) |
| `linear_attention_and_rnn/gdn/gdn_chunk_cumsum.py` | FAIL_CRASH | 279.7 | RuntimeError: cumsum:../third_party/op-plugin/op_plugin/ops/opapi/CumsumKernelNpuOpApi.cpp:69 NPU function error: device |
| `linear_attention_and_rnn/gdn/gdn_chunk_h.py` | FAIL_CRASH | 7.9 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/gdn/gdn_chunk_o.py` | FAIL_CRASH | 8.0 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/gdn/gdn_chunk_scaled_dot_kkt.py` | FAIL_CRASH | 8.0 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/gdn/gdn_solve_tril.py` | FAIL_CRASH | 8.0 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/gdn/gdn_wy_fast.py` | FAIL_CRASH | 7.9 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/gdn_full.py` | FAIL_CRASH | 8.0 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/linear_attention_causal.py` | FAIL_CRASH | 8.0 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/linear_attention_normalize.py` | FAIL_CRASH | 8.0 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_cumsum.py` | FAIL_TIMEOUT | 1800.0 | msprof 30-minute timeout or simulator abort (status 139/134) |
| `linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_h.py` | FAIL_CRASH | 8.6 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_o.py` | FAIL_CRASH | 8.6 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_scaled_dot_kkt.py` | FAIL_CRASH | 8.7 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/opt_gdn/opt_gdn_solve_tril.py` | FAIL_CRASH | 7.9 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/opt_gdn/opt_gdn_wy_fast.py` | FAIL_CRASH | 8.5 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `linear_attention_and_rnn/opt_gdn_full.py` | FAIL_CRASH | 7.9 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `moe_token_permute/moe_token_permute.py` | FAIL_CRASH | 7.9 | RuntimeError: random_op_api_:../third_party/op-plugin/op_plugin/ops/opapi/RandomKernelNpuOpApi.cpp:124 NPU function erro |
| `moe_token_permute/moe_token_permute_grad.py` | FAIL_CRASH | 7.9 | RuntimeError: random_op_api_:../third_party/op-plugin/op_plugin/ops/opapi/RandomKernelNpuOpApi.cpp:124 NPU function erro |
| `moe_token_permute/moe_token_unpermute.py` | FAIL_CRASH | 8.0 | RuntimeError: randperm_op_api:../third_party/op-plugin/op_plugin/ops/opapi/RandpermKernelNpuOpApi.cpp:28 NPU function er |
| `moe_token_permute/moe_token_unpermute_grad.py` | FAIL_CRASH | 7.9 | RuntimeError: randperm_op_api:../third_party/op-plugin/op_plugin/ops/opapi/RandpermKernelNpuOpApi.cpp:28 NPU function er |
| `moe_token_permute/moe_token_utils.py` | FAIL_CRASH | 7.9 | Kernel Output Match |
| `normalization/layer_norm.py` | FAIL_CRASH | 34.3 | RuntimeError: operator():../third_party/op-plugin/op_plugin/ops/opapi/LayerNormKernelNpuOpApi.cpp:98 NPU function error: |
| `normalization/rms_norm.py` | FAIL_CRASH | 54.6 | RuntimeError: pow:../third_party/op-plugin/op_plugin/ops/opapi/StructKernelNpuOpApi.cpp:2438 NPU function error: device  |
| `pad/example_broadcast.py` | FAIL_CRASH | 19.6 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `pad/example_broadcast_pipeline.py` | FAIL_CRASH | 22.1 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `pipeline/flash_attn_bshd_pipeline.py` | FAIL_CRASH | 135.2 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `pipeline/gemm_v0_pipeline.py` | FAIL_ACCURACY | 26.8 | AssertionError: Tensor-likes are not close! |
| `pipeline/matmul_add_pipeline.py` | PASS | 41.2 |  |
| `pipeline/sparse_flash_attn_gqa_pipeline.py` | FAIL_CRASH | 12.8 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `pipeline/sparse_flash_attn_gqa_pipeline_pto.py` | FAIL_CRASH | 8.2 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `pos_embedding/rms_norm.py` | FAIL_CRASH | 298.8 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `pos_embedding/rms_rope_fused.py` | FAIL_CRASH | 7.8 | SystemExit: 2 |
| `pos_embedding/rms_rope_fused_mask.py` | FAIL_CRASH | 7.8 | SystemExit: 2 |
| `pos_embedding/rope.py` | FAIL_CRASH | 7.8 | SystemExit: 2 |
| `pos_embedding/rope_mask.py` | FAIL_CRASH | 7.8 | SystemExit: 2 |
| `pos_embedding/rope_mask_bwd.py` | FAIL_CRASH | 7.9 | SystemExit: 2 |
| `print/elementwise_print.py` | FAIL_COMPILE | 10.0 | RuntimeError: Compilation Failed! ['bisheng', '--cce-aicore-arch=dav-c310', '-DREGISTER_BASE', '-O2', '-std=gnu++17', '- |
| `quant_batch_matmul/example_quant_batch_matmul.py` | PASS | 253.0 |  |
| `quant_batch_matmul/example_quant_matmul.py` | PASS | 71.2 |  |
| `random_1d/random_1d.py` | FAIL_CRASH | 7.9 | SystemExit: 2 |
| `reduce/example_col_reduce_max_slice_buffer.py` | FAIL_CRASH | 16.7 | RuntimeError: max:../third_party/op-plugin/op_plugin/ops/opapi/MaxKernelNpuOpApi.cpp:82 NPU function error: device error |
| `reduce/example_reduce_min.py` | FAIL_CRASH | 22.8 | RuntimeError: min:../third_party/op-plugin/op_plugin/ops/opapi/MinKernelNpuOpApi.cpp:86 NPU function error: device error |
| `reduce/example_reduce_min_pipeline.py` | SKIP | 0.0 | target=ascendc out of scope |
| `reduce/example_row_reduce_max_slice_buffer.py` | FAIL_CRASH | 17.3 | RuntimeError: max:../third_party/op-plugin/op_plugin/ops/opapi/MaxKernelNpuOpApi.cpp:82 NPU function error: device error |
| `seer_attention/block_sparse_attn.py` | FAIL_CRASH | 7.9 | RuntimeError: fill_:../third_party/op-plugin/op_plugin/ops/opapi/FillKernelNpuOpApi.cpp:26 NPU function error: device er |
| `simple_fusion/matmul_add.py` | FAIL_ACCURACY | 39.4 | AssertionError: Tensor-likes are not close! |
| `simple_fusion/matmul_add_infer_scope.py` | FAIL_ACCURACY | 39.6 | AssertionError: Tensor-likes are not close! |
| `softmax/example_online_softmax.py` | FAIL_CRASH | 65.3 | RuntimeError: operator():../third_party/op-plugin/op_plugin/ops/opapi/StructKernelNpuOpApi.cpp:193 NPU function error: d |
| `sort/example_merge_sort.py` | FAIL_COMPILE | 10.1 | RuntimeError: Compilation Failed! ['bisheng', '--cce-aicore-arch=dav-c310', '-DREGISTER_BASE', '-O2', '-std=gnu++17', '- |
| `sparse_flash_attention/example_sparse_flash_attn.py` | FAIL_CRASH | 12.5 | RuntimeError: randperm_op_api:../third_party/op-plugin/op_plugin/ops/opapi/RandpermKernelNpuOpApi.cpp:28 NPU function er |
| `sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py` | FAIL_CRASH | 14.4 | RuntimeError: randperm_op_api:../third_party/op-plugin/op_plugin/ops/opapi/RandpermKernelNpuOpApi.cpp:28 NPU function er |
| `sparse_flash_attention/example_sparse_flash_attn_gqa.py` | FAIL_CRASH | 12.7 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `sparse_flash_attention/example_sparse_flash_attn_gqa_pto.py` | FAIL_CRASH | 8.3 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `sparse_flash_attention/example_sparse_flash_attn_gqa_pto_developer.py` | FAIL_CRASH | 12.7 | RuntimeError: copy_d2d_baseformat_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:90 NPU function error: dev |
| `sparse_flash_attention/example_sparse_flash_attn_mask.py` | FAIL_CRASH | 14.6 | RuntimeError: randperm_op_api:../third_party/op-plugin/op_plugin/ops/opapi/RandpermKernelNpuOpApi.cpp:28 NPU function er |
| `sparse_flash_attention/example_sparse_flash_attn_mask_pa.py` | FAIL_CRASH | 14.4 | RuntimeError: randperm_op_api:../third_party/op-plugin/op_plugin/ops/opapi/RandpermKernelNpuOpApi.cpp:28 NPU function er |
| `topk_selector/example_topk_selector.py` | FAIL_CRASH | 316.2 | RuntimeError: topk:../third_party/op-plugin/op_plugin/ops/opapi/TopKKernelNpuOpApi.cpp:69 NPU function error: device err |
| `torch_tl_ascend/test_example.sh` | SKIP | 0.0 | PyTorch C extension integration, not standard JIT |
