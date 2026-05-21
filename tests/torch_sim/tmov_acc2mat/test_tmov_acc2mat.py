#!/usr/bin/env python3
"""Self-contained tmov_acc2mat smoke test (torch_npu + ctypes)."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from common.acc_golden_smoke import make_nz2nd_3_matmul
from common.ctypes_utils import bind_launch
from common.numeric import assert_arrays_match
from common.torch_runtime import data_ptr, empty_npu, init_torch_npu, npu_tensor, stream_ptr, sync, to_numpy

_LIB = ctypes.CDLL(str(Path(__file__).parent / "build" / "libtmov_acc2mat.so"))
bind_launch(_LIB, "pto_launch_nz2nd_4", 4)
M, K, N = 6, 7, 8


def case_nz2nd_4():
    init_torch_npu()
    x1, x2, golden = make_nz2nd_3_matmul()
    out = empty_npu((M, N), torch.int32)
    d_x1 = npu_tensor(x1)
    d_x2 = npu_tensor(x2)
    _LIB.pto_launch_nz2nd_4(data_ptr(out), data_ptr(d_x1), data_ptr(d_x2), stream_ptr())
    sync()
    out_u32 = to_numpy(out).view(np.uint32)
    assert_arrays_match(golden, out_u32, eps=0.001)


SMOKE_CASES = [("case_nz2nd_4", case_nz2nd_4)]


if __name__ == "__main__":
    from common.reporter import run_smoke_cases

    results = run_smoke_cases(Path(__file__), SMOKE_CASES)
    raise SystemExit(1 if any(not r.passed for r in results) else 0)
