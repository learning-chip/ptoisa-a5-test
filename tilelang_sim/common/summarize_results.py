#!/usr/bin/env python3
"""Aggregate per-example run results into summary.json and summary.md."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest.json"
RESULTS = ROOT / "results"
SUMMARY_JSON = RESULTS / "summary.json"
SUMMARY_MD = RESULTS / "summary.md"


def load_results() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text())
    rows: list[dict] = []
    for entry in manifest:
        row = {
            "id": entry["id"],
            "rel_path": entry["rel_path"],
            "category": entry.get("category"),
            "skip_reason": entry.get("skip_reason"),
            "status": None,
            "duration_s": None,
            "notes": "",
        }
        log_path = RESULTS / "logs" / f"{entry['id']}.log"
        if log_path.exists():
            text = log_path.read_text()
            for line in text.splitlines():
                if line.startswith("status:"):
                    row["status"] = line.split(":", 1)[1].strip()
                elif line.startswith("duration_s:"):
                    try:
                        row["duration_s"] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("notes:"):
                    row["notes"] = line.split(":", 1)[1].strip()
        if row["status"] is None and entry.get("skip_reason"):
            row["status"] = "SKIP"
            row["notes"] = entry["skip_reason"]
        rows.append(row)
    return rows


def write_summary(rows: list[dict]) -> None:
    counts = Counter(r["status"] or "NOT_RUN" for r in rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(rows),
        "counts": dict(counts),
        "rows": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# TileLang-Ascend A5 CPU Sim Results",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Executive summary",
        "",
        f"- **Total examples**: {len(rows)} (from `bench_test.sh` inventory)",
        f"- **Runnable under PTO+A5 sim**: {len(rows) - counts.get('SKIP', 0)}",
        f"- **PASS** (no crash + numerical check): **{counts.get('PASS', 0)}**",
        f"- **FAIL**: {counts.get('FAIL_CRASH',0) + counts.get('FAIL_COMPILE',0) + counts.get('FAIL_ACCURACY',0) + counts.get('FAIL_TIMEOUT',0)} "
        f"(crash={counts.get('FAIL_CRASH',0)}, compile={counts.get('FAIL_COMPILE',0)}, "
        f"accuracy={counts.get('FAIL_ACCURACY',0)}, timeout={counts.get('FAIL_TIMEOUT',0)})",
        f"- **SKIP** (out of scope): {counts.get('SKIP', 0)}",
        "",
        "JIT settings enforced by `common/bootstrap.py`: `target=\"pto\"`, `platform=\"A5\"` (A5 libgen branch: `dav-c310`, `-DREGISTER_BASE`).",
        "",
        "### PASS examples",
        "",
    ]
    passes = [r for r in rows if r.get("status") == "PASS"]
    if passes:
        for r in passes:
            lines.append(f"- `{r['rel_path']}`")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Summary counts", "", "| Status | Count |", "|--------|-------|"])
    for status in sorted(counts.keys()):
        lines.append(f"| {status} | {counts[status]} |")
    lines.extend(["", "## Per-example results", "", "| Example | Status | Duration (s) | Notes |", "|---------|--------|--------------|-------|"])
    for r in rows:
        dur = "" if r["duration_s"] is None else f"{r['duration_s']:.1f}"
        notes = (r.get("notes") or "").replace("|", "\\|").replace("\n", " ")[:120]
        lines.append(f"| `{r['rel_path']}` | {r['status'] or 'NOT_RUN'} | {dur} | {notes} |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_JSON} and {SUMMARY_MD}")


if __name__ == "__main__":
    write_summary(load_results())
