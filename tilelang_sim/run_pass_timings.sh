#!/usr/bin/env bash
# Record msprof simulator timings for all PASS tilelang examples.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m common.collect_pass_timings
