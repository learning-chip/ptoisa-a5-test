#!/usr/bin/env python3
"""Self-contained tadd smoke tests (torch_npu + ctypes)."""

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
from common.torch_runtime import data_ptr, empty_npu, npu_tensor, stream_ptr, sync, to_numpy

np.random.seed(19)
_LIB = ctypes.CDLL(str(Path(__file__).parent / "build" / "libtadd.so"))
bind_launch(_LIB, "pto_tadd_float_64x64", 4)
bind_launch(_LIB, "pto_tadd_int32_64x64", 4)


def _make_add_golden(dtype, dh, dw, s0h, s0w, s1h, s1w, vr, vc):
    in1 = np.random.randint(1, 10, size=(s0h, s0w)).astype(dtype)
    in2 = np.random.randint(1, 10, size=(s1h, s1w)).astype(dtype)
    golden = np.zeros((dh, dw), dtype=dtype)
    golden[:vr, :vc] = in1[:vr, :vc] + in2[:vr, :vc]
    return in1, in2, golden


def case_float_64x64():
    in1, in2, golden = _make_add_golden(np.float32, 64, 64, 64, 64, 64, 64, 64, 64)
    out = empty_npu(in1.shape, torch.float32)
    s0 = npu_tensor(in1)
    s1 = npu_tensor(in2)
    _LIB.pto_tadd_float_64x64(data_ptr(out), data_ptr(s0), data_ptr(s1), stream_ptr())
    sync()
    assert_arrays_match(golden, to_numpy(out), eps=0.001)


def case_int32_64x64():
    in1, in2, golden = _make_add_golden(np.int32, 64, 64, 64, 64, 64, 64, 64, 64)
    out = empty_npu(in1.shape, torch.int32)
    s0 = npu_tensor(in1)
    s1 = npu_tensor(in2)
    _LIB.pto_tadd_int32_64x64(data_ptr(out), data_ptr(s0), data_ptr(s1), stream_ptr())
    sync()
    assert_arrays_match(golden, to_numpy(out), eps=0.0)


SMOKE_CASES = [
    ("case_float_64x64", case_float_64x64),
    ("case_int32_64x64", case_int32_64x64),
]


if __name__ == "__main__":
    from common.reporter import run_smoke_cases
    from common.torch_runtime import init_torch_npu

    init_torch_npu()
    results = run_smoke_cases(Path(__file__), SMOKE_CASES)
    failed = sum(1 for r in results if not r.passed)
    raise SystemExit(1 if failed else 0)
