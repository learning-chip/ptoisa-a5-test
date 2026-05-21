#!/usr/bin/env python3
"""Run all python_wrapper smoke tests with pytest-style output."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

WRAPPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WRAPPER_ROOT))

from common.sim_env import setup_simulator_env

setup_simulator_env()

TEST_MODULES = [
    "tadd/test_tadd.py",
    "tmax/test_tmax.py",
    "tmul/test_tmul.py",
    "tload/test_tload.py",
    "syncall/test_syncall.py",
    "tmov_acc2vec/test_tmov_acc2vec.py",
    "textract_acc2vec/test_textract_acc2vec.py",
    "tinsert_acc2vec/test_tinsert_acc2vec.py",
    "tmov_acc2mat/test_tmov_acc2mat.py",
]

_RESULT_RE = re.compile(r"^(?P<nodeid>\S+::\S+) (PASSED|FAILED)$")


def main() -> int:
    t0 = time.perf_counter()
    passed = 0
    failed = 0
    for rel in TEST_MODULES:
        script = WRAPPER_ROOT / rel
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(WRAPPER_ROOT),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        for line in (proc.stdout or "").splitlines():
            m = _RESULT_RE.match(line.strip())
            if not m:
                continue
            if m.group(2) == "PASSED":
                passed += 1
            else:
                failed += 1
        if proc.returncode != 0 and not proc.stdout:
            failed += 1
            print(f"{rel} FAILED (exit {proc.returncode})")
    print("=" * 60)
    print(f"======================== {passed} passed, {failed} failed in {time.perf_counter() - t0:.2f}s ========================")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
