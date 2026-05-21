"""Python helper around the pto_runtime pybind11 module."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

_WRAPPER_ROOT = Path(__file__).resolve().parent.parent
if str(_WRAPPER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WRAPPER_ROOT))

from common.sim_env import setup_simulator_env

setup_simulator_env()

_BUILD_DIR = Path(__file__).resolve().parent / "build"
sys.path.insert(0, str(_BUILD_DIR))
import pto_runtime as _rt  # noqa: E402


class PtoRuntime:
    def __init__(self, device_id: int = 0) -> None:
        _rt.init()
        _rt.set_device(device_id)
        self._device_id = device_id
        self._stream = _rt.create_stream()
        self._allocs: list[tuple[int, bool]] = []

    @property
    def stream(self) -> int:
        return self._stream

    def sync(self) -> None:
        _rt.sync_stream(self._stream)

    def malloc_host(self, nbytes: int) -> int:
        ptr = _rt.malloc_host(nbytes)
        self._allocs.append((ptr, True))
        return ptr

    def malloc_device(self, nbytes: int) -> int:
        ptr = _rt.malloc_device(nbytes)
        self._allocs.append((ptr, False))
        return ptr

    def free(self, ptr: int, host: bool) -> None:
        if host:
            _rt.free_host(ptr)
        else:
            _rt.free_device(ptr)
        self._allocs = [(p, h) for p, h in self._allocs if p != ptr]

    def _temp_host(self, nbytes: int) -> int:
        return _rt.malloc_host(nbytes)

    def h2d(self, dst: int, arr: np.ndarray) -> None:
        host = self._temp_host(arr.nbytes)
        ctypes.memmove(host, arr.ctypes.data, arr.nbytes)
        _rt.memcpy_h2d(dst, host, arr.nbytes)
        _rt.free_host(host)

    def d2h(self, src: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        host = self._temp_host(nbytes)
        _rt.memcpy_d2h(host, src, nbytes)
        out = np.empty(shape, dtype=dtype)
        ctypes.memmove(out.ctypes.data, host, nbytes)
        _rt.free_host(host)
        return out

    def release_all_device_memory(self) -> None:
        """Free tracked allocations between smoke cases (keep ACL session alive)."""
        for ptr, is_host in reversed(list(self._allocs)):
            self.free(ptr, host=is_host)

    def refresh_between_cases(self) -> None:
        """Reset stream/device state between kernels (mirrors per-case cpp main)."""
        _rt.sync_stream(self._stream)
        self.release_all_device_memory()
        _rt.destroy_stream(self._stream)
        _rt.reset_device(self._device_id)
        _rt.set_device(self._device_id)
        self._stream = _rt.create_stream()

    def close(self) -> None:
        self.release_all_device_memory()
        _rt.destroy_stream(self._stream)
        _rt.reset_device(self._device_id)
        _rt.finalize()

    def __enter__(self) -> PtoRuntime:
        return self

    def __exit__(self, *args) -> None:
        self.close()
