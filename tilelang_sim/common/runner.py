"""Execute one tilelang example and classify PASS/FAIL/SKIP."""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path

TILELANG_SIM_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = Path("/workdir/tilelang-ascend/examples")
MANIFEST_PATH = TILELANG_SIM_ROOT / "manifest.json"
RESULTS_DIR = TILELANG_SIM_ROOT / "results"

SUCCESS_PATTERNS = [
    re.compile(r"Kernel Output Match!", re.I),
    re.compile(r"Batch Kernel Output Match!", re.I),
    re.compile(r"Random Kernel Output Match!", re.I),
    re.compile(r"All rms_norm tests passed!", re.I),
    re.compile(r"Test Passed!", re.I),
    re.compile(r"Test passed!", re.I),
    re.compile(r"TEST PASSED!", re.I),
]

FAIL_COMPILE_PATTERNS = [
    re.compile(r"Compilation Failed", re.I),
    re.compile(r"Compile kernel failed", re.I),
    re.compile(r"RuntimeError: Compilation Failed", re.I),
]

FAIL_ACCURACY_PATTERNS = [
    re.compile(r"AssertionError", re.I),
    re.compile(r"assert_close", re.I),
    re.compile(r"Tensor.*not equal", re.I),
    re.compile(r"Accuracy check failed", re.I),
]


@dataclass
class RunResult:
    id: str
    rel_path: str
    status: str
    category: str
    duration_s: float
    notes: str = ""
    error_snippet: str = ""


def _load_manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text())


def _find_entry(manifest: list[dict], manifest_id: str | None, rel_path: str | None) -> dict:
    for entry in manifest:
        if manifest_id and entry["id"] == manifest_id:
            return entry
        if rel_path and entry["rel_path"] == rel_path:
            return entry
    raise SystemExit(f"Manifest entry not found: id={manifest_id} rel_path={rel_path}")


def _classify_output(text: str, exc: BaseException | None) -> tuple[str, str]:
    if exc is not None:
        tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
        from common.bootstrap import SkipExampleError

        if isinstance(exc, SkipExampleError):
            return "SKIP", tb
        msg = tb + "\n" + traceback.format_exc()
        if any(p.search(msg) for p in FAIL_ACCURACY_PATTERNS):
            return "FAIL_ACCURACY", msg[-2000:]
        if any(p.search(msg) for p in FAIL_COMPILE_PATTERNS):
            return "FAIL_COMPILE", msg[-2000:]
        return "FAIL_CRASH", msg[-2000:]

    if any(p.search(text) for p in SUCCESS_PATTERNS):
        return "PASS", ""

    if any(p.search(text) for p in FAIL_COMPILE_PATTERNS):
        return "FAIL_COMPILE", text[-2000:]

    if any(p.search(text) for p in FAIL_ACCURACY_PATTERNS):
        return "FAIL_ACCURACY", text[-2000:]

    return "FAIL_CRASH", text[-2000:]


def _resolve_run_target(entry: dict) -> tuple[Path, Path]:
    run_path = entry["run_path"]
    if run_path.startswith("wrappers/"):
        script = TILELANG_SIM_ROOT / run_path
        cwd = script.parent
        return script, cwd

    if entry.get("kind") == "shell":
        script = EXAMPLES_ROOT / run_path
        return script, script.parent

    script = EXAMPLES_ROOT / run_path
    return script, script.parent


def _build_argv(entry: dict, script: Path) -> list[str]:
    extra = entry.get("extra_args") or []
    if entry.get("kind") == "shell":
        return extra
    return [script.name, *extra]


def run_entry(entry: dict) -> RunResult:
    t0 = time.perf_counter()
    rel_path = entry["rel_path"]

    if entry.get("skip_reason"):
        result = RunResult(
            id=entry["id"],
            rel_path=rel_path,
            status="SKIP",
            category=entry.get("category", "unknown"),
            duration_s=0.0,
            notes=entry["skip_reason"],
        )
        write_log(result)
        return result

    script, cwd = _resolve_run_target(entry)
    if not script.exists():
        return RunResult(
            id=entry["id"],
            rel_path=rel_path,
            status="FAIL_CRASH",
            category=entry.get("category", "unknown"),
            duration_s=time.perf_counter() - t0,
            notes=f"script not found: {script}",
        )

    buf = io.StringIO()
    exc: BaseException | None = None

    if entry.get("kind") == "shell":
        argv = ["bash", script.name, *(_build_argv(entry, script))]
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
        text = proc.stdout + proc.stderr
        status = "PASS" if proc.returncode == 0 else "FAIL_CRASH"
        if status != "PASS":
            status, snippet = _classify_output(text, None)
            if status == "FAIL_CRASH" and proc.returncode != 0:
                snippet = text[-2000:]
        else:
            snippet = ""
        duration = time.perf_counter() - t0
        result = RunResult(entry["id"], rel_path, status, entry.get("category", "unknown"), duration, snippet)
        write_log(result, text)
        return result

    sys.path.insert(0, str(TILELANG_SIM_ROOT))
    sys.path.insert(0, str(cwd))
    old_argv = sys.argv[:]
    sys.argv = _build_argv(entry, script)

    try:
        import common.bootstrap  # noqa: F401

        with redirect_stdout(buf), redirect_stderr(buf):
            import runpy

            runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        text = buf.getvalue()
        if e.code not in (0, None):
            exc = e
        elif not any(p.search(text) for p in SUCCESS_PATTERNS):
            exc = RuntimeError(f"exit code {e.code} without success marker")
    except BaseException as e:
        exc = e
    finally:
        sys.argv = old_argv

    text = buf.getvalue()
    status, snippet = _classify_output(text, exc)
    duration = time.perf_counter() - t0
    notes = snippet
    result = RunResult(entry["id"], rel_path, status, entry.get("category", "unknown"), duration, notes)
    write_log(result, text)
    return result


def write_log(result: RunResult, text: str = "") -> None:
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{result.id}.log"
    body = [
        f"id: {result.id}",
        f"rel_path: {result.rel_path}",
        f"status: {result.status}",
        f"category: {result.category}",
        f"duration_s: {result.duration_s:.2f}",
        f"notes: {result.notes}",
        "",
        text,
    ]
    path.write_text("\n".join(body))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-id")
    parser.add_argument("--rel-path")
    parser.add_argument("--write-log", action="store_true", default=True)
    args = parser.parse_args()

    manifest = _load_manifest()
    entry = _find_entry(manifest, args.manifest_id, args.rel_path)
    result = run_entry(entry)
    print(json.dumps(asdict(result)))
    raise SystemExit(0 if result.status in {"PASS", "SKIP"} else 1)


if __name__ == "__main__":
    main()
