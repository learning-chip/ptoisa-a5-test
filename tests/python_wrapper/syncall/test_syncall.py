#!/usr/bin/env python3
"""Self-contained syncall smoke test."""

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

_LIB = ctypes.CDLL(str(Path(__file__).parent / "build" / "libsyncall.so"))
bind_launch(_LIB, "pto_launch_hard_syncall_18", 3)


def case_hard_aiv_only_all_blocks():
    block_count = 18
    stride = 8
    element_count = block_count * stride
    byte_size = element_count * np.dtype(np.int32).itemsize

    rt = get_runtime()
    d_out = rt.malloc_device(byte_size)
    d_flags = rt.malloc_device(byte_size)
    zeros = np.zeros(element_count, dtype=np.int32)
    rt.h2d(d_out, zeros)
    rt.h2d(d_flags, zeros)
    _LIB.pto_launch_hard_syncall_18(d_out, d_flags, rt.stream)
    rt.sync()
    raw = rt.d2h(d_out, (element_count,), np.int32)

    golden = np.ones(block_count, dtype=np.int32)
    sampled = np.array([raw[i * stride] for i in range(block_count)], dtype=np.int32)
    assert_arrays_match(golden, sampled, eps=0.0)


SMOKE_CASES = [
    ("case_hard_aiv_only_all_blocks", case_hard_aiv_only_all_blocks),
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

