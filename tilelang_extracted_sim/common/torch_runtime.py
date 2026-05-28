"""torch_npu runtime helpers for extracted TileLang PTO kernels."""

from __future__ import annotations

import ctypes

import numpy as np
import torch
import torch_npu  # noqa: F401

_DEVICE = "npu:0"


def init_torch_npu(device: str = _DEVICE) -> None:
    global _DEVICE
    _DEVICE = device
    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(device)


def npu_tensor(np_arr) -> torch.Tensor:
    return torch.from_numpy(np_arr).to(_DEVICE)


def empty_npu(shape, dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=_DEVICE)


def zeros_npu(shape, dtype) -> torch.Tensor:
    np_dtype = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: torch.bfloat16,
        torch.int8: np.int8,
        torch.int32: np.int32,
    }.get(dtype)
    if np_dtype is None:
        raise TypeError(f"unsupported dtype for zeros_npu: {dtype}")
    return npu_tensor(np.zeros(shape, dtype=np_dtype))


def stream_ptr() -> int:
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def vp(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def sync() -> None:
    torch.npu.synchronize()
