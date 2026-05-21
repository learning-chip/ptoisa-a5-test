#!/usr/bin/env python3
"""Self-contained tmax smoke tests."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

_WRAPPER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WRAPPER))

from common.ctypes_utils import bind_launch
from common.numeric import assert_arrays_match
from common.runtime_holder import get_runtime

np.random.seed(19)
_LIB = ctypes.CDLL(str(Path(__file__).parent / "build" / "libtmax.so"))
bind_launch(_LIB, "pto_tmax_float_16x32", 4)
bind_launch(_LIB, "pto_tmax_int32_16x32", 4)


def _make_max_golden(dtype, dh, dw, s0h, s0w, s1h, s1w, vr, vc):
    in1 = np.random.randint(1, 10, size=(s0h, s0w)).astype(dtype)
    in2 = np.random.randint(1, 10, size=(s1h, s1w)).astype(dtype)
    golden = np.zeros((dh, dw), dtype=dtype)
    golden[:vr, :vc] = np.maximum(in1[:vr, :vc], in2[:vr, :vc])
    return in1, in2, golden


def case_float_16x32():
    in1, in2, golden = _make_max_golden(np.float32, 16, 32, 16, 64, 16, 32, 16, 32)
    rt = get_runtime()
    d_out = rt.malloc_device(golden.nbytes)
    d_s0 = rt.malloc_device(in1.nbytes)
    d_s1 = rt.malloc_device(in2.nbytes)
    rt.h2d(d_s0, in1)
    rt.h2d(d_s1, in2)
    _LIB.pto_tmax_float_16x32(d_out, d_s0, d_s1, rt.stream)
    rt.sync()
    out = rt.d2h(d_out, (16, 32), np.float32)
    assert_arrays_match(golden, out, eps=0.001)


def case_int32_16x32():
    in1, in2, golden = _make_max_golden(np.int32, 16, 32, 16, 64, 16, 32, 16, 32)
    rt = get_runtime()
    d_out = rt.malloc_device(golden.nbytes)
    d_s0 = rt.malloc_device(in1.nbytes)
    d_s1 = rt.malloc_device(in2.nbytes)
    rt.h2d(d_s0, in1)
    rt.h2d(d_s1, in2)
    _LIB.pto_tmax_int32_16x32(d_out, d_s0, d_s1, rt.stream)
    rt.sync()
    out = rt.d2h(d_out, (16, 32), np.int32)
    assert_arrays_match(golden, out, eps=0.0)


SMOKE_CASES = [
    ("case_float_16x32", case_float_16x32),
    ("case_int32_16x32", case_int32_16x32),
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

