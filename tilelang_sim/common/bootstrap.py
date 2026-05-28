"""Force target=pto and platform=A5 before any tilelang example imports."""

from __future__ import annotations

from typing import Any

import tilelang
from tilelang.jit import jit as _original_jit

from common.reference_cpu_fallback import enable_reference_cpu_fallback
from common.torch_runtime import init_torch_npu


class SkipExampleError(Exception):
    """Raised when an example is out of scope (e.g. target=ascendc)."""


_PATCHED = False


def _forcing_jit(
    func=None,
    /,
    *,
    out_idx: Any = None,
    workspace_idx: Any = None,
    target: Any = "auto",
    target_host: Any = None,
    platform: str = "auto",
    execution_backend: Any = "cython",
    verbose: bool = False,
    pass_configs: Any = None,
    debug_root_path: Any = None,
):
    if target == "ascendc":
        raise SkipExampleError("target=ascendc is out of scope for A5 PTO sim testing")

    return _original_jit(
        func,
        out_idx=out_idx,
        workspace_idx=workspace_idx,
        target="pto",
        target_host=target_host,
        platform="A5",
        execution_backend=execution_backend,
        verbose=verbose,
        pass_configs=pass_configs,
        debug_root_path=debug_root_path,
    )


def apply_bootstrap() -> None:
    global _PATCHED
    if _PATCHED:
        return

    tilelang.jit = _forcing_jit  # type: ignore[assignment]
    tilelang.cache.clear_cache()
    enable_reference_cpu_fallback()
    init_torch_npu()
    _PATCHED = True


apply_bootstrap()
