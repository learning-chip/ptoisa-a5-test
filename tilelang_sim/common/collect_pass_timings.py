#!/usr/bin/env python3
"""Re-run PASS examples under msprof and record simulator timing + input shapes."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest.json"
SUMMARY = ROOT / "results" / "summary.json"
SHAPES = ROOT / "configs" / "pass_shapes.json"
OUT_JSON = ROOT / "pass_timings.json"
OUT_MD = ROOT / "pass_timings.md"
OUT_DIR = ROOT / "results" / "pass_timings"

MODEL_RUN_RE = re.compile(r"Model RUN TIME:\s*([\d.]+)\s*ms")
CORE_LINE_RE = re.compile(r"^(core\S+)\s+([\d.]+)\s+([\d.]+)\s*$")


def _load_pass_entries() -> list[dict]:
    summary = json.loads(SUMMARY.read_text())
    manifest = {e["id"]: e for e in json.loads(MANIFEST.read_text())}
    passes = [r for r in summary["rows"] if r.get("status") == "PASS"]
    out = []
    for row in passes:
        entry = manifest[row["id"]]
        out.append({**row, **entry})
    return out


def _parse_msprof_output(text: str) -> dict:
    model_runs = [float(x) for x in MODEL_RUN_RE.findall(text)]
    cores: list[tuple[str, float, float]] = []
    in_core_table = False
    for line in text.splitlines():
        if "core_name" in line and "duration_time" in line:
            in_core_table = True
            continue
        if in_core_table:
            m = CORE_LINE_RE.match(line.strip())
            if m:
                cores.append((m.group(1), float(m.group(2)), float(m.group(3))))
            elif line.strip() and not line.startswith("core"):
                in_core_table = False

    sim_model_ms = model_runs[-1] if model_runs else None
    core_duration_us = max((c[1] for c in cores), default=None)
    core_running_us = max((c[2] for c in cores), default=None)
    return {
        "sim_model_run_ms": sim_model_ms,
        "sim_core_duration_us": core_duration_us,
        "sim_core_running_us": core_running_us,
        "sim_model_run_ms_all": model_runs,
        "sim_cores": [{"name": n, "duration_us": d, "running_us": r} for n, d, r in cores],
    }


def _shape_summary(shape_meta: dict) -> str:
    cases = shape_meta.get("cases") or []
    if not cases:
        return shape_meta.get("note", "")
    parts = []
    for i, c in enumerate(cases):
        keys = ("Batch", "M", "N", "K", "block_M", "block_N", "block_K", "K_L1", "elements", "dtype")
        desc = ", ".join(f"{k}={c[k]}" for k in keys if k in c)
        if len(cases) > 1:
            parts.append(f"case{i + 1}: {desc}")
        else:
            parts.append(desc)
    note = shape_meta.get("note")
    if note:
        parts.append(f"({note})")
    return "; ".join(parts)


def _run_one(entry: dict, shape_meta: dict) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"{entry['id']}.log"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(ROOT / "run_single.sh"), "--id", entry["id"], "--timeout", "30"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    wall_s = time.perf_counter() - t0
    text = proc.stdout + proc.stderr
    log_path.write_text(text)
    timing = _parse_msprof_output(text)
    runner_json = None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"status"' in line:
            try:
                runner_json = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    return {
        "id": entry["id"],
        "rel_path": entry["rel_path"],
        "run_path": entry.get("run_path", entry["rel_path"]),
        "cli_args": shape_meta.get("cli_args", entry.get("extra_args", [])),
        "input_shapes": shape_meta.get("cases", []),
        "shape_summary": _shape_summary(shape_meta),
        "soc_version": "Ascend950PR_9599",
        "target": "pto",
        "platform": "A5",
        "wall_time_s": round(wall_s, 2),
        "runner_duration_s": runner_json.get("duration_s") if runner_json else None,
        "status": runner_json.get("status") if runner_json else ("PASS" if proc.returncode == 0 else "FAIL"),
        **timing,
    }


def _write_markdown(rows: list[dict], generated_at: str) -> None:
    lines = [
        "# PASS example simulator timings (A5 CPU sim)",
        "",
        f"Generated: {generated_at}",
        "",
        "Measured via `msprof op simulator --soc-version=Ascend950PR_9599` with `target=pto`, `platform=A5`.",
        "",
        "- **sim_model_run_ms**: PEM model run time reported by the simulator (end of kernel session).",
        "- **sim_core_duration_us**: max AICore `duration_time` from msprof core operator table (when present).",
        "- **wall_time_s**: end-to-end host time including JIT compile + sim startup.",
        "",
        "| Example | Input shape(s) | sim_model_run_ms | sim_core_duration_us | wall_time_s |",
        "|---------|----------------|------------------|----------------------|-------------|",
    ]
    for r in rows:
        sim_ms = "" if r.get("sim_model_run_ms") is None else f"{r['sim_model_run_ms']:.3f}"
        core_us = "" if r.get("sim_core_duration_us") is None else f"{r['sim_core_duration_us']:.2f}"
        wall = f"{r['wall_time_s']:.1f}"
        shape = r.get("shape_summary", "").replace("|", "\\|")
        lines.append(f"| `{r['rel_path']}` | {shape} | {sim_ms} | {core_us} | {wall} |")
    lines.extend(["", "## Per-example logs", ""])
    for r in rows:
        lines.append(f"- `{r['id']}`: [results/pass_timings/{r['id']}.log](results/pass_timings/{r['id']}.log)")
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    shapes = json.loads(SHAPES.read_text())
    entries = _load_pass_entries()
    rows = []
    for entry in entries:
        meta = shapes.get(entry["rel_path"], {})
        print(f"Timing {entry['rel_path']} ...")
        rows.append(_run_one(entry, meta))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "soc_version": "Ascend950PR_9599",
        "simulator": "msprof op simulator",
        "target": "pto",
        "platform": "A5",
        "count": len(rows),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(rows, payload["generated_at"])
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
