#!/usr/bin/env python3
"""Discover tilelang-ascend examples (mirrors bench_test.sh) and emit manifest.json."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXAMPLES_ROOT = Path("/workdir/tilelang-ascend/examples")
TILELANG_SIM_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = TILELANG_SIM_ROOT / "configs" / "small_shapes.json"

EXTRA_TASKS = [
    (
        "sparse_flash_attention/bench_sfa",
        "python bench_sfa.py --file sparse_flash_attn_pa_baseline",
        "[bench_sfa] sparse_flash_attn_pa_baseline",
    ),
    (
        "sparse_flash_attention/bench_sfa",
        "python bench_sfa.py --file sparse_flash_attn_pa_developer",
        "[bench_sfa] sparse_flash_attn_pa_developer",
    ),
    (
        "sparse_flash_attention/bench_sfa",
        "python bench_sfa.py --file sparse_flash_attn_pa_no_cv_pipeline",
        "[bench_sfa] sparse_flash_attn_pa_no_cv_pipeline",
    ),
    (
        "sparse_flash_attention/bench_sfa",
        "python bench_sfa.py --file sparse_flash_attn_pa",
        "[bench_sfa] sparse_flash_attn_pa",
    ),
]

SKIP_EXACT = {
    "gemm/example_gemm_intrinsic.py": "target=ascendc out of scope",
    "gemm/example_gemm_intrinsic_persistent.py": "target=ascendc out of scope",
    "reduce/example_reduce_min_pipeline.py": "target=ascendc out of scope",
    "gemm_aot/run_example_gemm_aot.sh": "AOT ctypes workflow, not @tilelang.jit PTO",
    "torch_tl_ascend/test_example.sh": "PyTorch C extension integration, not standard JIT",
    "flash_attention/fa_opt/flash_attn_bhsd_ascendc.py": "uses torch_npu.npu_fusion_attention, not TileLang JIT",
    "flash_attention/fa_opt/run.py": "msprof benchmark orchestrator, not correctness smoke",
    "flash_attention/fa_opt/plot.py": "plotting utility, not kernel test",
    "autotune/example_gemm_autotune.py": "autotune search exceeds sim time budget",
    "autotune/example_gemm_carver.py": "autotune search exceeds sim time budget",
}

SKIP_PREFIXES = (
    "dispatch_combine/",
    "shmem/",
)

CATEGORY_BY_DIR = {
    "gemm": "gemm",
    "gemv": "gemv",
    "simple_fusion": "fusion",
    "pipeline": "fusion",
    "developer_mode": "fusion",
    "elementwise": "elementwise",
    "activation": "activation",
    "batch_gemm": "batch_gemm",
    "flash_attention": "flash_attention",
    "sparse_flash_attention": "flash_attention",
    "seer_attention": "flash_attention",
    "convolution": "convolution",
    "normalization": "normalization",
    "reduce": "reduce",
    "softmax": "softmax",
    "pad": "pad",
    "pos_embedding": "pos_embedding",
    "quant_batch_matmul": "quant",
    "dequantize_gemm": "dequantize_gemm",
    "topk_selector": "topk",
    "cumsum_gdn": "cumsum",
    "cumsum_kda": "cumsum",
    "chunk_gated_delta_rule": "chunk_gated_delta_rule",
    "fused_sigmoid_gating_delta_rule": "gdn",
    "linear_attention_and_rnn": "gdn",
    "grouped_gemm": "grouped_gemm",
    "blocksparse_gemm": "blocksparse_gemm",
    "hadamard_transform": "hadamard",
    "lightning_indexer": "lightning_indexer",
    "moe_token_permute": "moe",
    "random_1d": "random_1d",
    "causal_conv1d": "gdn",
    "deepseek_v4": "gemm",
    "aclgraph": "normalization",
    "print": "elementwise",
    "gemm_splitk": "gemm",
}


def _category_for(rel: str) -> str:
    top = rel.split("/", 1)[0]
    return CATEGORY_BY_DIR.get(top, "default")


def _timeout_for(category: str) -> int:
    if category in {"flash_attention", "gdn", "chunk_gated_delta_rule"}:
        return 1800
    return 600


def collect_test_scripts(examples_root: Path) -> list[str]:
    scripts: list[str] = []

    for dir_path in sorted(examples_root.iterdir()):
        if not dir_path.is_dir():
            continue
        rel_dir = dir_path.relative_to(examples_root).as_posix()
        if rel_dir in {"dispatch_combine", "shmem"}:
            continue

        if rel_dir == "gemm_aot":
            scripts.append("gemm_aot/run_example_gemm_aot.sh")
            continue
        if rel_dir == "torch_tl_ascend":
            scripts.append("torch_tl_ascend/test_example.sh")
            continue

        if rel_dir == "flash_attention":
            for py in sorted(dir_path.glob("*.py")):
                if py.name.endswith("_golden.py"):
                    continue
                scripts.append(f"flash_attention/{py.name}")
            continue

        for py in sorted(dir_path.glob("*.py")):
            if py.name in {"__init__.py", "sfa_golden.py"} or py.name.endswith("_golden.py"):
                continue
            scripts.append(py.relative_to(examples_root).as_posix())

        for sub in sorted(p for p in dir_path.iterdir() if p.is_dir() and p.name != "bench_sfa"):
            for py in sorted(sub.glob("*.py")):
                if py.name in {"__init__.py", "sfa_golden.py"} or py.name.endswith("_golden.py"):
                    continue
                scripts.append(py.relative_to(examples_root).as_posix())

        for sh in sorted(dir_path.glob("run_*.sh")) + sorted(dir_path.glob("test_*.sh")):
            scripts.append(sh.relative_to(examples_root).as_posix())
        for sub in sorted(p for p in dir_path.iterdir() if p.is_dir()):
            for sh in sorted(sub.glob("run_*.sh")) + sorted(sub.glob("test_*.sh")):
                scripts.append(sh.relative_to(examples_root).as_posix())

    for task_dir, _cmd, display in EXTRA_TASKS:
        scripts.append(f"CUSTOM::{display}::{task_dir}")

    fa_opt = examples_root / "flash_attention" / "fa_opt"
    if fa_opt.is_dir():
        for py in sorted(fa_opt.glob("flash_*.py")):
            scripts.append(f"flash_attention/fa_opt/{py.name}")

    return sorted(set(scripts))


def _extra_args(rel: str, category: str, config: dict) -> list[str]:
    wrappers = config.get("wrappers", {})
    if rel in wrappers:
        return []
    per = config.get("per_script", {})
    if rel in per:
        return per[rel]
    return config.get("category_defaults", {}).get(category, config.get("category_defaults", {}).get("default", []))


def _script_path(rel: str, config: dict) -> str:
    wrappers = config.get("wrappers", {})
    if rel in wrappers:
        return wrappers[rel]
    return rel


def build_manifest(examples_root: Path) -> list[dict]:
    config = json.loads(CONFIG_PATH.read_text())
    entries: list[dict] = []

    for idx, item in enumerate(collect_test_scripts(examples_root)):
        entry: dict = {
            "id": f"ex_{idx:04d}",
            "rel_path": item,
            "category": "custom",
            "skip_reason": None,
            "extra_args": [],
            "timeout_s": 600,
            "run_path": item,
            "kind": "python",
        }

        if item.startswith("CUSTOM::"):
            _, display, task_dir = item.split("::", 2)
            entry.update(
                {
                    "rel_path": display,
                    "category": "benchmark",
                    "skip_reason": "bench_sfa performance harness, not correctness smoke",
                    "kind": "custom",
                    "run_path": task_dir,
                }
            )
            entries.append(entry)
            continue

        if item.endswith(".sh"):
            entry["kind"] = "shell"

        skip_reason = SKIP_EXACT.get(item)
        if not skip_reason:
            for prefix in SKIP_PREFIXES:
                if item.startswith(prefix):
                    skip_reason = "multi-NPU / cluster setup required"
                    break

        category = _category_for(item)
        entry["category"] = category
        entry["timeout_s"] = _timeout_for(category)
        entry["skip_reason"] = skip_reason
        entry["extra_args"] = _extra_args(item, category, config)
        entry["run_path"] = _script_path(item, config)
        if entry["run_path"].startswith("wrappers/"):
            entry["run_path"] = entry["run_path"]

        entries.append(entry)

    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=TILELANG_SIM_ROOT / "manifest.json")
    args = parser.parse_args()

    manifest = build_manifest(EXAMPLES_ROOT)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    runnable = sum(1 for e in manifest if not e.get("skip_reason"))
    print(f"Wrote {len(manifest)} entries ({runnable} runnable) to {args.output}")


if __name__ == "__main__":
    main()
