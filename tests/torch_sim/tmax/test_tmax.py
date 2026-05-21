#!/usr/bin/env python3
"""Self-contained tmax smoke tests (torch_npu + ctypes)."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from common.ctypes_utils import bind_launch
from common.numeric import assert_arrays_match
from common.torch_runtime import data_ptr, empty_npu, init_torch_npu, npu_tensor, stream_ptr, sync, to_numpy

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
    init_torch_npu()
    in1, in2, golden = _make_max_golden(np.float32, 16, 32, 16, 64, 16, 32, 16, 32)
    out = empty_npu((16, 32), torch.float32)
    s0 = npu_tensor(in1)
    s1 = npu_tensor(in2)
    _LIB.pto_tmax_float_16x32(data_ptr(out), data_ptr(s0), data_ptr(s1), stream_ptr())
    sync()
    assert_arrays_match(golden, to_numpy(out), eps=0.001)


def case_int32_16x32():
    init_torch_npu()
    in1, in2, golden = _make_max_golden(np.int32, 16, 32, 16, 64, 16, 32, 16, 32)
    out = empty_npu((16, 32), torch.int32)
    s0 = npu_tensor(in1)
    s1 = npu_tensor(in2)
    _LIB.pto_tmax_int32_16x32(data_ptr(out), data_ptr(s0), data_ptr(s1), stream_ptr())
    sync()
    assert_arrays_match(golden, to_numpy(out), eps=0.0)


SMOKE_CASES = [
    ("case_float_16x32", case_float_16x32),
    ("case_int32_16x32", case_int32_16x32),
]


if __name__ == "__main__":
    from common.reporter import run_smoke_cases

    results = run_smoke_cases(Path(__file__), SMOKE_CASES)
    raise SystemExit(1 if any(not r.passed for r in results) else 0)
