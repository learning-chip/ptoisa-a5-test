#!/usr/bin/env python3
"""Self-contained syncall smoke test (torch_npu + ctypes)."""

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
from common.torch_runtime import data_ptr, init_torch_npu, stream_ptr, sync, to_numpy, zeros_npu

_LIB = ctypes.CDLL(str(Path(__file__).parent / "build" / "libsyncall.so"))
bind_launch(_LIB, "pto_launch_hard_syncall_18", 3)


def case_hard_aiv_only_all_blocks():
    init_torch_npu()
    block_count = 18
    stride = 8
    element_count = block_count * stride
    out = zeros_npu((element_count,), torch.int32)
    flags = zeros_npu((element_count,), torch.int32)
    _LIB.pto_launch_hard_syncall_18(data_ptr(out), data_ptr(flags), stream_ptr())
    sync()
    raw = to_numpy(out)
    golden = np.ones(block_count, dtype=np.int32)
    sampled = np.array([raw[i * stride] for i in range(block_count)], dtype=np.int32)
    assert_arrays_match(golden, sampled, eps=0.0)


SMOKE_CASES = [
    ("case_hard_aiv_only_all_blocks", case_hard_aiv_only_all_blocks),
]


if __name__ == "__main__":
    from common.reporter import run_smoke_cases

    results = run_smoke_cases(Path(__file__), SMOKE_CASES)
    raise SystemExit(1 if any(not r.passed for r in results) else 0)
