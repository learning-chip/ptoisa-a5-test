"""Process-wide ACL runtime shared across smoke cases in one runner invocation."""

from __future__ import annotations

from common.pto_runtime import PtoRuntime

_runtime: PtoRuntime | None = None


def get_runtime() -> PtoRuntime:
    global _runtime
    if _runtime is None:
        _runtime = PtoRuntime()
    return _runtime


def shutdown_runtime() -> None:
    global _runtime
    if _runtime is not None:
        _runtime.close()
        _runtime = None
