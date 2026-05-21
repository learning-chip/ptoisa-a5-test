#!/usr/bin/env python3
"""Self-contained tinsert_acc2vec smoke test (torch_npu + ctypes)."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from common.acc_golden_smoke import make_nz2nd_3_insert
from common.ctypes_utils import bind_launch
from common.numeric import assert_arrays_match
from common.torch_runtime import data_ptr, empty_npu, npu_tensor, stream_ptr, sync, to_numpy

_LIB = ctypes.CDLL(str(Path(__file__).parent / "build" / "libtinsert_acc2vec.so"))
bind_launch(_LIB, "pto_launch_nz2nd_3", 5)


def case_nz2nd_3():
    x1, x2, dst_preload, golden = make_nz2nd_3_insert()
    out = empty_npu(golden.shape, torch.int32)
    d_x1 = npu_tensor(x1)
    d_x2 = npu_tensor(x2)
    d_dst = npu_tensor(dst_preload.view(np.int32))
    _LIB.pto_launch_nz2nd_3(data_ptr(out), data_ptr(d_x1), data_ptr(d_x2), data_ptr(d_dst), stream_ptr())
    sync()
    assert_arrays_match(golden, to_numpy(out).view(np.uint32), eps=0.001)


SMOKE_CASES = [("case_nz2nd_3", case_nz2nd_3)]


if __name__ == "__main__":
    from common.reporter import run_smoke_cases
    from common.torch_runtime import init_torch_npu

    init_torch_npu()
    results = run_smoke_cases(Path(__file__), SMOKE_CASES)
    raise SystemExit(1 if any(not r.passed for r in results) else 0)
