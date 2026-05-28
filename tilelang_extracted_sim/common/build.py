#!/usr/bin/env python3
"""Bisheng build helper for TileLang-extracted A5 PTO kernels (dav-c310, REGISTER_BASE)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "kernels"
BUILD_DIR = ROOT / "build"
MANIFEST_PATH = ROOT / "manifest.json"


def _manifest_kernels() -> dict[str, dict]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in data.get("kernels", [])}


def _tl_root() -> Path:
    env = os.environ.get("TL_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    for candidate in (
        Path("/sources/tilelang-ascend"),
        Path("/workdir/tilelang-ascend"),
    ):
        if candidate.is_dir():
            return candidate
    raise EnvironmentError("TileLang root not found; set TL_ROOT")


def _tilelang_template_path(tl_root: Path) -> Path:
    env = os.environ.get("TL_TEMPLATE_PATH")
    if env:
        return Path(env)
    src = tl_root / "src"
    if (src / "tl_templates" / "pto" / "common.h").is_file():
        return src
    raise EnvironmentError("TileLang template path not found under TL_ROOT/src")


def _ascend_home() -> Path:
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        raise EnvironmentError("ASCEND_HOME_PATH is not set. Source CANN setenv.bash first.")
    return Path(home)


def _bisheng() -> str:
    ascend = _ascend_home()
    candidate = ascend / "bin" / "bisheng"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("bisheng")
    if found:
        return found
    raise FileNotFoundError("bisheng compiler not found")


def _compile_flags() -> list[str]:
    tl_root = _tl_root()
    ascend = _ascend_home()
    templates = _tilelang_template_path(tl_root)
    driver = os.environ.get("ASCEND_DRIVER_PATH", "/usr/local/Ascend/driver")
    return [
        "bisheng",
        "--cce-aicore-arch=dav-c310",
        "-DREGISTER_BASE",
        "-O2",
        "-std=gnu++17",
        "-xcce",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        f"-I{tl_root}/3rdparty/pto-isa/include",
        f"-I{ascend}/include",
        f"-I{ascend}/include/experiment/msprof",
        f"-I{ascend}/include/experiment/runtime",
        "-I/usr/local/Ascend/driver/kernel/inc",
        f"-I{ascend}/pkg_inc",
        f"-I{ascend}/pkg_inc/runtime",
        f"-I{ascend}/pkg_inc/profiling",
        f"-I{templates}",
        f"-L{ascend}/lib64",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-lruntime",
        "-lstdc++",
        "-lascendcl",
        "-lm",
        "-ltiling_api",
        "-lplatform",
        "-lc_sec",
        "-ldl",
        "-fPIC",
        "--shared",
    ]


def _run(cmd: list[str], cwd: Path) -> None:
    print("==>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build_kernel(kernel_id: str, force: bool = False) -> Path:
    kernels = _manifest_kernels()
    if kernel_id not in kernels:
        raise ValueError(f"unknown kernel: {kernel_id}")
    spec = kernels[kernel_id]
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / spec["lib_file"]
    if out.is_file() and not force:
        return out

    src_path = KERNEL_DIR / spec["source_file"]
    if not src_path.is_file():
        raise FileNotFoundError(f"missing source: {src_path}; run common/extract_sources.py first")

    cmd = _compile_flags() + [str(src_path), "-o", str(out)]
    _run(cmd, BUILD_DIR)
    print(f"Built {out}")
    return out


def build_all(force: bool = False) -> dict[str, Path]:
    return {kid: build_kernel(kid, force=force) for kid in _manifest_kernels()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TileLang-extracted A5 PTO kernels")
    parser.add_argument("--kernel")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        if args.all:
            build_all(force=args.force)
        elif args.kernel:
            build_kernel(args.kernel, force=args.force)
        else:
            parser.print_help()
            raise SystemExit(1)
    except (EnvironmentError, FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
