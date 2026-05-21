"""Pytest-style test reporting without pytest."""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class CaseResult:
    nodeid: str
    passed: bool
    duration_s: float
    error: str | None = None


def run_smoke_cases(
    module_path: Path,
    cases: list[tuple[str, Callable[[], None]]],
) -> list[CaseResult]:
    """Run named callables and collect results."""
    from common.runtime_holder import get_runtime

    rel = module_path.as_posix()
    results: list[CaseResult] = []
    for name, fn in cases:
        nodeid = f"{rel}::{name}"
        t0 = time.perf_counter()
        try:
            fn()
            get_runtime().refresh_between_cases()
            results.append(
                CaseResult(nodeid=nodeid, passed=True, duration_s=time.perf_counter() - t0)
            )
            print(f"{nodeid} PASSED")
        except Exception as exc:
            get_runtime().refresh_between_cases()
            results.append(
                CaseResult(
                    nodeid=nodeid,
                    passed=False,
                    duration_s=time.perf_counter() - t0,
                    error="".join(traceback.format_exception(exc)),
                )
            )
            print(f"{nodeid} FAILED")
            print(traceback.format_exc())
    return results


def load_test_module(path: Path):
    """Import test_<op>.py as a module."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def print_summary(results: list[CaseResult], elapsed_s: float) -> int:
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print("=" * 60)
    print(f"======================== {passed} passed, {failed} failed in {elapsed_s:.2f}s ========================")
    return 0 if failed == 0 else 1
