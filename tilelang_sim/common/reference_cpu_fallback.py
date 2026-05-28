"""Route common PyTorch reference ops through CPU when inputs are on NPU (msprof sim)."""

from __future__ import annotations

import torch


def _is_npu_device(device) -> bool:
    if device is None:
        try:
            return torch.get_default_device().type == "npu"
        except Exception:
            return False
    if isinstance(device, str):
        return device.startswith("npu")
    return getattr(device, "type", None) == "npu"


def _cpu_then_npu(factory, *args, **kwargs):
    device = kwargs.pop("device", None)
    target = device
    if target is None and torch.get_default_device().type == "npu":
        target = torch.get_default_device()
    if _is_npu_device(target):
        kwargs["device"] = "cpu"
        out = factory(*args, **kwargs)
        return out.to(target)
    if device is not None:
        kwargs["device"] = device
    return factory(*args, **kwargs)


def enable_reference_cpu_fallback() -> None:
    if getattr(enable_reference_cpu_fallback, "_done", False):
        return

    for name in ("randn", "rand", "empty", "zeros", "ones", "full"):
        orig = getattr(torch, name)

        def _make_wrapper(fn):
            def wrapped(*args, **kwargs):
                return _cpu_then_npu(fn, *args, **kwargs)

            return wrapped

        setattr(torch, name, _make_wrapper(orig))

    _tensor_npu = torch.Tensor.npu

    def npu_via_cpu(self, device=None, non_blocking=False):
        if self.device.type == "cpu":
            return _tensor_npu(self, device=device, non_blocking=non_blocking)
        return _tensor_npu(self.cpu(), device=device, non_blocking=non_blocking)

    torch.Tensor.npu = npu_via_cpu  # type: ignore[assignment]

    def _cpu_binary(orig):
        def wrapped(self, other):
            if self.device.type != "npu":
                return orig(self, other)
            other_cpu = other.cpu() if isinstance(other, torch.Tensor) else other
            return orig(self.cpu(), other_cpu).to(device=self.device)

        return wrapped

    for name in ("__add__", "__sub__", "__mul__", "__matmul__", "__truediv__"):
        if hasattr(torch.Tensor, name):
            orig = getattr(torch.Tensor, name)
            setattr(torch.Tensor, name, _cpu_binary(orig))

    _torch_matmul = torch.matmul

    def matmul_cpu(input, other, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.device.type == "npu":
            other_cpu = other.cpu() if isinstance(other, torch.Tensor) else other
            return _torch_matmul(input.cpu(), other_cpu, *args, **kwargs).to(input.device)
        return _torch_matmul(input, other, *args, **kwargs)

    torch.matmul = matmul_cpu  # type: ignore[assignment]

    _torch_tanh = torch.tanh

    def tanh_cpu(input, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.device.type == "npu":
            return _torch_tanh(input.cpu(), *args, **kwargs).to(input.device)
        return _torch_tanh(input, *args, **kwargs)

    torch.tanh = tanh_cpu  # type: ignore[assignment]

    _torch_softmax = torch.softmax

    def softmax_cpu(input, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.device.type == "npu":
            return _torch_softmax(input.cpu(), *args, **kwargs).to(input.device)
        return _torch_softmax(input, *args, **kwargs)

    torch.softmax = softmax_cpu  # type: ignore[assignment]

    _assert_close = torch.testing.assert_close

    def assert_close_cpu(actual, expected, *args, **kwargs):
        if isinstance(actual, torch.Tensor) and actual.device.type == "npu":
            actual = actual.cpu()
        if isinstance(expected, torch.Tensor) and expected.device.type == "npu":
            expected = expected.cpu()
        return _assert_close(actual, expected, *args, **kwargs)

    torch.testing.assert_close = assert_close_cpu  # type: ignore[assignment]

    enable_reference_cpu_fallback._done = True  # type: ignore[attr-defined]
