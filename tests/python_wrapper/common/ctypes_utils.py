"""ctypes helpers for kernel launch libraries."""

from __future__ import annotations

import ctypes


def bind_launch(lib, name: str, n_ptr_args: int) -> None:
    """Bind extern launch symbol with void* arguments and optional stream."""
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p] * n_ptr_args
    fn.restype = None


def bind_launch_int_return(lib, name: str, n_ptr_args: int) -> None:
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p] * n_ptr_args
    fn.restype = ctypes.c_int
