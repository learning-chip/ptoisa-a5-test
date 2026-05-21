#!/usr/bin/env python3
"""Self-contained tload smoke test (kernel-embedded golden)."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

_WRAPPER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WRAPPER))

from common.ctypes_utils import bind_launch, bind_launch_int_return
from common.numeric import assert_arrays_match
from common.runtime_holder import get_runtime

_LIB = ctypes.CDLL(str(Path(__file__).parent / "build" / "libtload.so"))
bind_launch_int_return(_LIB, "pto_get_input_golden_1", 2)
bind_launch(_LIB, "pto_launch_tload_1", 4)

M, N = 1024, 1024
IN_BYTES = M * N * 4
OUT_BYTES = M * N * 4
LOG_BYTES = 64 * 128 * 8


def case_float_GT_128_128_VT_128_128_BLK1():
    rt = get_runtime()
    h_in = rt.malloc_host(IN_BYTES)
    h_gold = rt.malloc_host(OUT_BYTES)
    actual_bytes = _LIB.pto_get_input_golden_1(h_in, h_gold)

    inp = np.empty((M, N), dtype=np.float32)
    ctypes.memmove(inp.ctypes.data, h_in, IN_BYTES)
    golden = np.empty(actual_bytes // 4, dtype=np.float32)
    ctypes.memmove(golden.ctypes.data, h_gold, actual_bytes)

    d_out = rt.malloc_device(OUT_BYTES)
    d_src = rt.malloc_device(IN_BYTES)
    d_log = rt.malloc_device(LOG_BYTES)
    rt.h2d(d_src, inp)
    rt.h2d(d_out, np.zeros((M, N), dtype=np.float32))
    rt.h2d(d_log, np.zeros(64 * 128, dtype=np.uint64))

    _LIB.pto_launch_tload_1(d_out, d_src, d_log, rt.stream)
    rt.sync()
    out = rt.d2h(d_out, golden.shape, np.float32)

    assert_arrays_match(golden, out, eps=0.0)


SMOKE_CASES = [
    ("case_float_GT_128_128_VT_128_128_BLK1", case_float_GT_128_128_VT_128_128_BLK1),
]

if __name__ == "__main__":
    from common.reporter import run_smoke_cases
    from common.runtime_holder import shutdown_runtime
    from common.sim_env import setup_simulator_env

    setup_simulator_env()
    try:
        run_smoke_cases(Path(__file__), SMOKE_CASES)
    finally:
        shutdown_runtime()

