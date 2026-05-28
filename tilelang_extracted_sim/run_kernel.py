#!/usr/bin/env python3
"""Launch extracted TileLang PTO kernels via ctypes (no TileLang at runtime)."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.build import build_kernel  # noqa: E402
from common.torch_runtime import empty_npu, init_torch_npu, stream_ptr, sync, vp, zeros_npu  # noqa: E402

_MANIFEST = ROOT / "manifest.json"
_LIBS: dict[str, ctypes.CDLL] = {}


def _load_manifest() -> dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def _load_lib(kernel_id: str) -> tuple[ctypes.CDLL, dict]:
    if kernel_id in _LIBS:
        manifest = _load_manifest()
        entry = next(k for k in manifest["kernels"] if k["id"] == kernel_id)
        return _LIBS[kernel_id], entry

    manifest = _load_manifest()
    entry = next(k for k in manifest["kernels"] if k["id"] == kernel_id)
    lib_path = build_kernel(kernel_id)
    lib = ctypes.CDLL(str(lib_path))
    symbol = entry.get("call_symbol", "call")
    fn = getattr(lib, symbol)
    n_tensors = entry.get("num_tensor_args") or 3
    argtypes = [ctypes.c_void_p] * n_tensors + [ctypes.c_void_p]
    fn.argtypes = argtypes
    fn.restype = None
    _LIBS[kernel_id] = lib
    return lib, entry


def _launch(kernel_id: str, tensors: list[torch.Tensor]) -> None:
    lib, entry = _load_lib(kernel_id)
    symbol = entry.get("call_symbol", "call")
    fn = getattr(lib, symbol)
    args = [vp(t) for t in tensors] + [stream_ptr()]
    fn(*args)
    sync()


def _make_elementwise_add(case: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
    m, n = case["M"], case["N"]
    a = torch.randn(m, n, dtype=torch.float32, device="npu:0")
    b = torch.randn(m, n, dtype=torch.float32, device="npu:0")
    c = zeros_npu((m, n), torch.float32)
    ref = a + b
    return [a, b, c], ref


def _make_matmul_add_dev(case: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
    m, n, k = case["M"], case["N"], case["K"]
    a = torch.randn(m, k, dtype=torch.float16, device="npu:0")
    b = torch.randn(k, n, dtype=torch.float16, device="npu:0")
    d = torch.randn(m, n, dtype=torch.float16, device="npu:0")
    c = empty_npu((m, n), torch.float16)
    workspace = empty_npu((1, m, n), torch.float16)
    ref = a @ b + d
    return [a, b, c, d, workspace], ref


def _make_gemm(case: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
    m, n, k = case["M"], case["N"], case["K"]
    a = torch.randn(m, k, dtype=torch.float16, device="npu:0")
    b = torch.randn(k, n, dtype=torch.float16, device="npu:0")
    c = empty_npu((m, n), torch.float16)
    ref = a @ b
    return [a, b, c], ref


def _make_matmul_add_pipeline(case: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
    m, n, k = case["M"], case["N"], case["K"]
    a = torch.randn(m, k, dtype=torch.float16, device="npu:0")
    b = torch.randn(k, n, dtype=torch.float16, device="npu:0")
    d = torch.randn(m, n, dtype=torch.float16, device="npu:0")
    c = empty_npu((m, n), torch.float16)
    workspace = empty_npu((m, n), torch.float16)
    ref = a @ b + d
    return [a, b, c, d, workspace], ref


def _make_tanh(case: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
    m, n = case["M"], case["N"]
    a = torch.randn(m, n, dtype=torch.float32, device="npu:0")
    b = zeros_npu((m, n), torch.float32)
    ref = torch.tanh(a)
    return [a, b], ref


def _dtype_map(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int32": torch.int32,
    }[name]


def _make_quant_matmul(case: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
    m, n, k = case["M"], case["N"], case["K"]
    in_dtype = _dtype_map(case["in_dtype"])
    out_dtype = _dtype_map(case["out_dtype"])
    scale_n = n if case["scale_size"] == "N" else 1
    a = torch.randint(-128, 127, (m, k), dtype=in_dtype, device="npu:0")
    b = torch.randint(-128, 127, (k, n), dtype=in_dtype, device="npu:0")
    scale = torch.randn(scale_n, dtype=torch.float32, device="npu:0")
    c = empty_npu((m, n), out_dtype)
    workspace = empty_npu((m, n), torch.int32)
    ref = (a.to(torch.int32) @ b.to(torch.int32)).to(torch.float32) * scale
    ref = ref.to(out_dtype)
    return [a, b, scale, c, workspace], ref


def _make_quant_batch_matmul(case: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
    batch = case.get("Batch", 8)
    m, n, k = case["M"], case["N"], case["K"]
    in_dtype = _dtype_map(case["in_dtype"])
    out_dtype = _dtype_map(case["out_dtype"])
    scale_n = n if case["scale_size"] == "N" else 1
    a = torch.randint(-128, 127, (batch, m, k), dtype=in_dtype, device="npu:0")
    b = torch.randint(-128, 127, (batch, k, n), dtype=in_dtype, device="npu:0")
    scale = torch.randn(scale_n, dtype=torch.float32, device="npu:0")
    c = empty_npu((batch, m, n), out_dtype)
    workspace = empty_npu((batch, m, n), torch.int32)
    ref = torch.einsum("bmk,bkn->bmn", a.to(torch.int32), b.to(torch.int32)).to(torch.float32) * scale
    ref = ref.to(out_dtype)
    return [a, b, scale, c, workspace], ref


def _output_tensor(tensors: list[torch.Tensor], entry: dict) -> torch.Tensor:
    out_idx = entry.get("out_idx", [2])
    idx = out_idx[0] if isinstance(out_idx, list) else out_idx
    return tensors[idx]


def _input_builder(kernel_id: str, entry: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
    case = entry["case"]
    example = entry["tilelang_example"]
    if "elementwise" in example:
        return _make_elementwise_add(case)
    if example.endswith("matmul_add_pipeline.py"):
        return _make_matmul_add_pipeline(case)
    if example.endswith("matmul_add_developer.py"):
        return _make_matmul_add_dev(case)
    if "tanh" in kernel_id or "tanh" in example:
        return _make_tanh(case)
    if "quant_batch_matmul" in example:
        return _make_quant_batch_matmul(case)
    if "quant_matmul" in example:
        return _make_quant_matmul(case)
    if "gemm" in example or "matmul" in example or "developer" in example:
        return _make_gemm(case)
    raise ValueError(f"no input builder for {kernel_id} ({example})")


def run_smoke(kernel_id: str, rtol: float = 1e-2, atol: float = 1e-2) -> None:
    init_torch_npu()
    _, entry = _load_lib(kernel_id)
    tensors, ref = _input_builder(kernel_id, entry)
    _launch(kernel_id, tensors)
    out = _output_tensor(tensors, entry)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=rtol, atol=atol)
    print(f"PASS {kernel_id} shape={tuple(out.shape)} dtype={out.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extracted TileLang PTO kernel smoke tests")
    parser.add_argument("--kernel", help="kernel id from manifest.json")
    parser.add_argument("--list", action="store_true", help="list kernel ids")
    parser.add_argument("--all", action="store_true", help="run smoke for all kernels")
    args = parser.parse_args()
    manifest = _load_manifest()
    if args.list:
        for entry in manifest["kernels"]:
            print(entry["id"])
        return
    if args.all:
        failed = []
        for entry in manifest["kernels"]:
            try:
                run_smoke(entry["id"])
            except Exception as exc:
                failed.append((entry["id"], str(exc)))
                print(f"FAIL {entry['id']}: {exc}", file=sys.stderr)
        if failed:
            raise SystemExit(1)
        return
    if not args.kernel:
        parser.print_help()
        raise SystemExit(1)
    run_smoke(args.kernel)


if __name__ == "__main__":
    main()
