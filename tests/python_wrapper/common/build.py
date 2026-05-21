#!/usr/bin/env python3
"""Bisheng build helper mirroring tests/cpp/common/bisheng_build.sh."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


WRAPPER_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WRAPPER_ROOT.parent.parent
PTO_ISA_ROOT = PROJECT_ROOT / "third-party" / "pto-isa"
COMMON_CPP = PROJECT_ROOT / "tests" / "cpp" / "common"


def _ascend_home() -> Path:
    home = os.environ.get("ASCEND_HOME_PATH")
    if not home:
        raise EnvironmentError("ASCEND_HOME_PATH is not set. Source CANN setenv.bash first.")
    return Path(home)


def _bisheng() -> str:
    ascend = _ascend_home()
    candidate = ascend / "bin" / "bisheng"
    if candidate.is_file():
        return str(candidate)
    import shutil

    found = shutil.which("bisheng")
    if found:
        return found
    raise FileNotFoundError("bisheng compiler not found")


def _aicore_arch(kernel_arch: str) -> str:
    if kernel_arch == "vec":
        return "dav-c310-vec"
    if kernel_arch == "mix":
        return "dav-c310"
    raise ValueError(f"unknown kernel arch: {kernel_arch}")


def _common_includes(test_dir: Path) -> list[str]:
    ascend = _ascend_home()
    driver = os.environ.get("ASCEND_DRIVER_PATH", "/usr/local/Ascend/driver")
    return [
        f"-I{PTO_ISA_ROOT}/include",
        f"-I{ascend}/include",
        f"-I{driver}/kernel/inc",
        f"-I{test_dir}",
        f"-I{COMMON_CPP}",
    ]


def _common_cxx_flags() -> list[str]:
    return [
        "-std=c++17",
        "-O2",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-Wno-unknown-attributes",
    ]


def _kernel_flags(test_dir: Path, kernel_arch: str) -> list[str]:
    ascend = _ascend_home()
    arch = _aicore_arch(kernel_arch)
    flags = (
        _common_includes(test_dir)
        + [
            f"-I{ascend}/pkg_inc",
            f"-I{ascend}/pkg_inc/profiling",
            f"-I{ascend}/pkg_inc/runtime/runtime",
        ]
        + _common_cxx_flags()
        + [
            "-fPIC",
            "-xcce",
            "-Xhost-start",
            "-Xhost-end",
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
            f"--cce-aicore-arch={arch}",
            "-DREGISTER_BASE",
        ]
    )
    return flags


def _host_flags(test_dir: Path) -> list[str]:
    return _common_includes(test_dir) + _common_cxx_flags() + [
        "-xc++",
        "-include",
        "stdint.h",
        "-include",
        "stddef.h",
        "-fPIC",
    ]


def _link_dirs(build_dir: Path) -> tuple[list[str], list[str]]:
    ascend = _ascend_home()
    soc = os.environ.get("SOC_VERSION", "Ascend950PR_9599")
    sim_lib = ascend / "tools" / "simulator" / soc / "lib"
    if not sim_lib.is_dir():
        raise FileNotFoundError(f"simulator lib not found: {sim_lib}")
    libs = [
        f"-L{ascend}/lib64",
        f"-L{sim_lib}",
        f"-L{build_dir}",
        "-lruntime_camodel",
        "-lstdc++",
        "-lascendcl",
        "-lm",
        "-ltiling_api",
        "-lplatform",
        "-lc_sec",
        "-ldl",
        "-lnnopbase",
        "-lpthread",
    ]
    rpath = f"-Wl,-rpath,{ascend}/lib64:{sim_lib}:{build_dir}"
    return libs, [rpath]


def _run(cmd: list[str], cwd: Path) -> None:
    print("==>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def compile_kernels(
    test_dir: Path,
    kernel_sources: list[str],
    kernel_arch: str,
    lib_name: str,
) -> Path:
    bisheng = _bisheng()
    build_dir = test_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    objects: list[str] = []
    for src in kernel_sources:
        src_path = test_dir / src
        obj = build_dir / f"{src_path.stem}.o"
        _run(
            [bisheng, *_kernel_flags(test_dir, kernel_arch), "-c", str(src_path), "-o", str(obj)],
            cwd=build_dir,
        )
        objects.append(str(obj))
    out = build_dir / lib_name
    _run(
        [bisheng, "-fPIC", "-shared", "--cce-fatobj-link", "-Wl,-soname," + lib_name, *objects, "-o", str(out)],
        cwd=build_dir,
    )
    return out


def compile_host_objects(test_dir: Path, host_sources: list[str]) -> list[Path]:
    bisheng = _bisheng()
    build_dir = test_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    objs: list[Path] = []
    for src in host_sources:
        src_path = test_dir / src
        obj = build_dir / f"{src_path.stem}.o"
        _run([bisheng, *_host_flags(test_dir), "-c", str(src_path), "-o", str(obj)], cwd=build_dir)
        objs.append(obj)
    return objs


def link_shared(
    test_dir: Path,
    host_objects: list[Path],
    kernel_lib: str,
    out_name: str,
) -> Path:
    bisheng = _bisheng()
    build_dir = test_dir / "build"
    lib_base = kernel_lib.replace("lib", "").replace(".so", "")
    libs, rpath = _link_dirs(build_dir)
    out = build_dir / out_name
    _run(
        [
            bisheng,
            *[str(o) for o in host_objects],
            "-shared",
            "-o",
            str(out),
            f"-l{lib_base}",
            *libs,
            *rpath,
        ],
        cwd=build_dir,
    )
    return out


def build_test_lib(
    test_name: str,
    kernel_sources: list[str],
    host_sources: list[str] | None = None,
    kernel_arch: str = "vec",
    kernel_lib: str | None = None,
    out_lib: str | None = None,
) -> Path:
    test_dir = WRAPPER_ROOT / test_name
    kernel_lib = kernel_lib or f"lib{test_name}_kernel.so"
    out_lib = out_lib or f"lib{test_name}.so"
    host_sources = host_sources or ["launch_api.cpp"]
    compile_kernels(test_dir, kernel_sources, kernel_arch, kernel_lib)
    host_objs = compile_host_objects(test_dir, host_sources)
    return link_shared(test_dir, host_objs, kernel_lib, out_lib)


def build_pto_runtime() -> Path:
    import pybind11
    import sysconfig

    test_dir = WRAPPER_ROOT / "common"
    build_dir = test_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    bisheng = _bisheng()
    src = test_dir / "pto_runtime.cpp"
    obj = build_dir / "pto_runtime.o"
    py_inc = pybind11.get_include()
    py_inc_user = pybind11.get_include(user=True)
    py_headers = sysconfig.get_path("include")
    _run(
        [
            bisheng,
            *_host_flags(test_dir),
            f"-I{py_headers}",
            f"-I{py_inc}",
            f"-I{py_inc_user}",
            "-c",
            str(src),
            "-o",
            str(obj),
        ],
        cwd=build_dir,
    )
    libs, rpath = _link_dirs(build_dir)
    out = build_dir / "pto_runtime.so"
    py_lib = sysconfig.get_config_var("LIBDIR") or ""
    py_ld = f"-L{py_lib}" if py_lib else ""
    py_ldflags = sysconfig.get_config_var("LDFLAGS") or ""
    py_libs = sysconfig.get_config_var("LIBS") or "-lpython3.11"
    link_cmd = [bisheng, str(obj), "-shared", "-o", str(out), py_ld, *py_ldflags.split(), py_libs, *libs, *rpath]
    _run(link_cmd, cwd=build_dir)
    return out


TESTS = {
    "tadd": {
        "kernels": ["tadd_kernel.cpp"],
        "arch": "vec",
    },
    "tmax": {"kernels": ["tmax_kernel.cpp"], "arch": "vec"},
    "tmul": {"kernels": ["tmul_kernel.cpp"], "arch": "vec"},
    "tload": {"kernels": ["tload_kernel.cpp"], "arch": "vec"},
    "syncall": {"kernels": ["syncall_hard_kernel.cpp"], "arch": "vec"},
    "tmov_acc2vec": {"kernels": ["tmov_acc2vec_kernel.cpp"], "arch": "mix"},
    "textract_acc2vec": {"kernels": ["textract_acc2vec_kernel.cpp"], "arch": "mix"},
    "tinsert_acc2vec": {"kernels": ["tinsert_acc2vec_kernel.cpp"], "arch": "mix"},
    "tmov_acc2mat": {"kernels": ["tmov_acc2mat_kernel.cpp"], "arch": "mix"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build python_wrapper artifacts")
    parser.add_argument("--all", action="store_true", help="Build pto_runtime and all test libs")
    parser.add_argument("--runtime", action="store_true", help="Build pto_runtime only")
    parser.add_argument("test", nargs="?", help="Build a single test lib")
    args = parser.parse_args()

    sys.path.insert(0, str(WRAPPER_ROOT))
    from common.sim_env import setup_simulator_env

    setup_simulator_env()

    if args.all or args.runtime:
        build_pto_runtime()
    if args.all:
        for name, spec in TESTS.items():
            build_test_lib(name, spec["kernels"], kernel_arch=spec["arch"])
    elif args.test:
        if args.test not in TESTS:
            parser.error(f"unknown test: {args.test}")
        spec = TESTS[args.test]
        build_test_lib(args.test, spec["kernels"], kernel_arch=spec["arch"])
    elif not args.runtime:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
