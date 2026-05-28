#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 common/extract_sources.py
