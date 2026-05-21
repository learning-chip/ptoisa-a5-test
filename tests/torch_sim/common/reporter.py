"""Pytest-style test reporting without pytest."""

from __future__ import annotations

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
    from common.torch_runtime import sync

    rel = module_path.as_posix()
    results: list[CaseResult] = []
    for name, fn in cases:
        nodeid = f"{rel}::{name}"
        t0 = time.perf_counter()
        try:
            fn()
            sync()
            results.append(
                CaseResult(nodeid=nodeid, passed=True, duration_s=time.perf_counter() - t0)
            )
            print(f"{nodeid} PASSED")
        except Exception:
            sync()
            results.append(
                CaseResult(
                    nodeid=nodeid,
                    passed=False,
                    duration_s=time.perf_counter() - t0,
                    error=traceback.format_exc(),
                )
            )
            print(f"{nodeid} FAILED")
            print(traceback.format_exc())
    return results


def print_summary(results: list[CaseResult], elapsed_s: float) -> int:
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print("=" * 60)
    print(f"======================== {passed} passed, {failed} failed in {elapsed_s:.2f}s ========================")
    return 0 if failed == 0 else 1
