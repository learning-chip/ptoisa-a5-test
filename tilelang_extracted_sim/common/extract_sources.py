#!/usr/bin/env python3
"""Extract PTO C++ sources from tilelang_sim PASS examples (compile-only, no NPU launch)."""

from __future__ import annotations

import json
import re
import runpy
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SIM_ROOT = ROOT.parent / "tilelang_sim"
TL_EXAMPLES = Path("/sources/tilelang-ascend/examples")
KERNEL_DIR = ROOT / "kernels"
MANIFEST_PATH = ROOT / "manifest.json"
PASS_SHAPES = SIM_ROOT / "configs" / "pass_shapes.json"

sys.path.insert(0, str(SIM_ROOT))
import common.bootstrap  # noqa: E402,F401

import tilelang  # noqa: E402
import torch  # noqa: E402
from tilelang.jit.kernel import JITKernel  # noqa: E402

@dataclass
class KernelRecord:
    id: str
    source_file: str
    lib_file: str
    tilelang_example: str
    case: dict[str, Any] = field(default_factory=dict)
    call_symbol: str = "call"
    block_dim: int | None = None
    num_tensor_args: int | None = None
    source_bytes: int = 0
    params: list[str] = field(default_factory=list)
    out_idx: Any = None


_CAPTURED: list[KernelRecord] = []
_ID_QUEUE: list[tuple[str, str, dict[str, Any]]] = []
_ORIG_INIT = JITKernel.__init__


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _parse_launch_info(source: str) -> dict[str, Any]:
    call_match = re.search(
        r'extern\s+"C"\s+void\s+(\w+)\s*\(([^)]*)\)',
        source,
    )
    launch_match = re.search(r"launch_kernel<<<\s*(\d+)\s*,", source)
    block_dim = int(launch_match.group(1)) if launch_match else None
    call_symbol = call_match.group(1) if call_match else "call"
    call_args = call_match.group(2) if call_match else ""
    tensor_args = [
        a.strip()
        for a in call_args.split(",")
        if "handle" in a.lower() and "stream" not in a.lower()
    ]
    return {
        "call_symbol": call_symbol,
        "block_dim": block_dim,
        "num_tensor_args": len(tensor_args),
    }


def _dummy_kernel_output(self: JITKernel, *args, **kwargs):
    out_idx = self.out_idx
    indices = out_idx if isinstance(out_idx, list) else ([out_idx] if out_idx is not None else [2])
    outputs = []
    for idx in indices:
        real_idx = idx if idx >= 0 else len(self.params) + idx
        param = self.params[real_idx]
        shape = tuple(int(x) for x in param.shape)
        outputs.append(torch.zeros(shape, dtype=param.dtype, device="npu:0"))
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def _hook_init(self, *args, **kwargs):
    _ORIG_INIT(self, *args, **kwargs)
    if kwargs.get("from_database", False):
        return
    if not _ID_QUEUE:
        return
    kernel_id, example_key, case = _ID_QUEUE.pop(0)
    source = self.get_kernel_source()
    launch = _parse_launch_info(source)
    source_file = f"{kernel_id}_a5.cpp"
    out_path = KERNEL_DIR / source_file
    out_path.write_text(source, encoding="utf-8")
    _CAPTURED.append(
        KernelRecord(
            id=kernel_id,
            source_file=source_file,
            lib_file=f"lib{kernel_id}_a5.so",
            tilelang_example=example_key,
            case=case,
            call_symbol=launch["call_symbol"],
            block_dim=launch["block_dim"],
            num_tensor_args=launch["num_tensor_args"],
            source_bytes=len(source),
            params=[str(p) for p in self.params],
            out_idx=self.out_idx,
        )
    )


def _install_runtime_patches() -> None:
    JITKernel.__init__ = _hook_init
    JITKernel.__call__ = _dummy_kernel_output
    torch.testing.assert_close = lambda *a, **k: None


def _example_path(example_key: str, cfg: dict[str, Any]) -> Path:
    run_path = cfg.get("run_path")
    if run_path:
        if run_path.startswith("wrappers/"):
            return SIM_ROOT / run_path
        return TL_EXAMPLES / run_path
    return TL_EXAMPLES / example_key


def _kernel_id(example_key: str, case: dict[str, Any], *, suffix: str = "") -> str:
    base = _slug(example_key.replace("/", "_").replace(".py", ""))
    if suffix:
        return f"{base}_{suffix}"
    m, n, k = case.get("M"), case.get("N"), case.get("K")
    if m is not None and n is not None and k is not None:
        return f"{base}_m{m}_n{n}_k{k}"
    if m is not None and n is not None:
        return f"{base}_m{m}_n{n}"
    return base


def _run_example(example_key: str, cfg: dict[str, Any], specs: list[tuple[str, dict[str, Any]]]) -> None:
    global _ID_QUEUE
    path = _example_path(example_key, cfg)
    cli = cfg.get("cli_args", [])
    _ID_QUEUE = [(kid, example_key, case) for kid, case in specs]
    sys.argv = [path.name, *cli]
    tilelang.cache.clear_cache()
    before = len(_CAPTURED)
    runpy.run_path(str(path), run_name="__main__")
    if len(_CAPTURED) - before != len(specs):
        raise RuntimeError(
            f"expected {len(specs)} captures for {example_key}, got {len(_CAPTURED) - before}"
        )


def _specs_for_example(example_key: str, cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    cases = cfg["cases"]
    if example_key == "activation/tanh.py":
        return [
            (f"tanh_m{c['M']}_n{c['N']}", c)
            for c in cases
        ]
    if example_key == "gemm/example_gemm_tail_block_developer.py":
        return [
            (f"gemm_tail_block_m{c['M']}_n{c['N']}_k{c['K']}", c)
            for c in cases
        ]
    if example_key == "quant_batch_matmul/example_quant_matmul.py":
        return [
            (
                f"quant_matmul_scale{c['scale_size']}_{c['out_dtype']}_m{c['M']}",
                c,
            )
            for c in cases
        ]
    if example_key == "quant_batch_matmul/example_quant_batch_matmul.py":
        batch = cases[0].get("Batch", 8)
        return [
            (
                f"quant_batch_matmul_b{batch}_scale{c['scale_size']}_{c['out_dtype']}_m{c['M']}",
                c,
            )
            for c in cases
        ]
    return [(_kernel_id(example_key, cases[0]), cases[0])]


def extract_all() -> list[KernelRecord]:
    shapes = json.loads(PASS_SHAPES.read_text(encoding="utf-8"))
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    _install_runtime_patches()
    errors: list[str] = []

    for example_key, cfg in shapes.items():
        try:
            specs = _specs_for_example(example_key, cfg)
            _run_example(example_key, cfg, specs)
        except Exception as exc:
            errors.append(f"{example_key}: {exc}\n{traceback.format_exc()}")

    manifest = {
        "generated_from": str(PASS_SHAPES),
        "tilelang_examples_root": str(TL_EXAMPLES),
        "target": "pto",
        "platform": "A5",
        "kernels": [asdict(r) for r in _CAPTURED],
        "errors": errors,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return _CAPTURED


def main() -> None:
    records = extract_all()
    print(f"Extracted {len(records)} kernels into {KERNEL_DIR}")
    for rec in records:
        print(f"  - {rec.id}: {rec.source_file} ({rec.source_bytes} bytes, block_dim={rec.block_dim})")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["errors"]:
        print(f"Errors ({len(manifest['errors'])}):", file=sys.stderr)
        for err in manifest["errors"]:
            print(err, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
