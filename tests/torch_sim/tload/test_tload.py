#!/usr/bin/env python3
"""Self-contained tload smoke test (torch_npu + ctypes)."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from common.ctypes_utils import bind_launch, bind_launch_int_return
from common.numeric import assert_arrays_match
from common.torch_runtime import data_ptr, empty_npu, init_torch_npu, npu_tensor, stream_ptr, sync, to_numpy, zeros_npu

_LIB = ctypes.CDLL(str(Path(__file__).parent / "build" / "libtload.so"))
bind_launch_int_return(_LIB, "pto_get_input_golden_1", 2)
bind_launch(_LIB, "pto_launch_tload_1", 4)

M, N = 1024, 1024


def case_float_GT_128_128_VT_128_128_BLK1():
    init_torch_npu()
    in_bytes = M * N * 4
    h_in = (ctypes.c_byte * in_bytes)()
    h_gold = (ctypes.c_byte * in_bytes)()
    actual_bytes = _LIB.pto_get_input_golden_1(
        ctypes.cast(h_in, ctypes.c_void_p), ctypes.cast(h_gold, ctypes.c_void_p)
    )

    inp = np.frombuffer(h_in, dtype=np.float32, count=M * N).reshape(M, N).copy()
    golden = np.frombuffer(h_gold, dtype=np.float32, count=actual_bytes // 4).copy()

    src = npu_tensor(inp)
    out = zeros_npu((M, N), torch.float32)
    log = zeros_npu((64 * 128,), torch.int64)

    _LIB.pto_launch_tload_1(data_ptr(out), data_ptr(src), data_ptr(log), stream_ptr())
    sync()
    assert_arrays_match(golden, to_numpy(out).flatten()[: golden.size], eps=0.0)


SMOKE_CASES = [
    ("case_float_GT_128_128_VT_128_128_BLK1", case_float_GT_128_128_VT_128_128_BLK1),
]


if __name__ == "__main__":
    from common.reporter import run_smoke_cases

    results = run_smoke_cases(Path(__file__), SMOKE_CASES)
    raise SystemExit(1 if any(not r.passed for r in results) else 0)
