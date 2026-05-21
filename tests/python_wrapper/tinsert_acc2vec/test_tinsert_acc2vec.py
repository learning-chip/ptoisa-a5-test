#!/usr/bin/env python3
"""Self-contained tinsert_acc2vec smoke test."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

_WRAPPER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WRAPPER))

from common.acc_golden_smoke import make_nz2nd_3_insert
from common.ctypes_utils import bind_launch
from common.numeric import assert_arrays_match
from common.runtime_holder import get_runtime

_LIB = ctypes.CDLL(str(Path(__file__).parent / "build" / "libtinsert_acc2vec.so"))
bind_launch(_LIB, "pto_launch_nz2nd_3", 5)


def case_nz2nd_3():
    x1, x2, dst_preload, golden = make_nz2nd_3_insert()
    rt = get_runtime()
    d_out = rt.malloc_device(golden.nbytes)
    d_x1 = rt.malloc_device(x1.nbytes)
    d_x2 = rt.malloc_device(x2.nbytes)
    d_dst = rt.malloc_device(dst_preload.nbytes)
    rt.h2d(d_x1, x1)
    rt.h2d(d_x2, x2)
    rt.h2d(d_dst, dst_preload)
    _LIB.pto_launch_nz2nd_3(d_out, d_x1, d_x2, d_dst, rt.stream)
    rt.sync()
    out = rt.d2h(d_out, golden.shape, np.uint32)
    assert_arrays_match(golden, out, eps=0.001)


SMOKE_CASES = [
    ("case_nz2nd_3", case_nz2nd_3),
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

