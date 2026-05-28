#!/usr/bin/env bash
# Run all runnable tilelang examples sequentially under msprof A5 CPU simulator.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PILOT=false
START_ID=""
LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pilot) PILOT=true; shift ;;
    --start-id) START_ID="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

python3 wrappers/gen_sparse_flash_small.py 2>/dev/null || true
python3 -m common.collect_examples

mkdir -p results/logs results/msprof

if $PILOT; then
  PILOT_PATHS=(
    "elementwise/elementwise_add.py"
    "gemm/example_gemm.py"
    "activation/tanh.py"
    "flash_attention/flash_attn_bhsd.py"
    "reduce/example_col_reduce_max_slice_buffer.py"
  )
  echo "Pilot mode: ${#PILOT_PATHS[@]} examples"
  for rel in "${PILOT_PATHS[@]}"; do
    echo "==> pilot ${rel}"
    ./run_single.sh --rel-path "$rel" --timeout 30 || true
  done
  python3 -m common.summarize_results
  exit 0
fi

TOTAL=$(python3 -c "import json; print(len(json.load(open('manifest.json'))))")
PROGRESS="results/progress.jsonl"
: > "$PROGRESS"

idx=0
passed=0
failed=0
skipped=0
not_run=0

while IFS= read -r line; do
  id=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$line")
  rel=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['rel_path'])" "$line")
  skip=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('skip_reason') or '')" "$line")
  timeout=$(python3 -c "import json,sys; print(max(1, int(json.loads(sys.argv[1]).get('timeout_s',600))//60))" "$line")

  idx=$((idx + 1))
  if [[ -n "$START_ID" && "$id" < "$START_ID" ]]; then
    continue
  fi
  if [[ -n "$LIMIT" ]]; then
    ran=$((passed + failed + skipped))
    if [[ $ran -ge $LIMIT ]]; then
      break
    fi
  fi

  echo "==> [$idx/$TOTAL] $id $rel"
  if [[ -n "$skip" ]]; then
    out=$(python3 -m common.runner --manifest-id "$id" 2>&1) || true
    echo "$out"
    echo "$out" >> "$PROGRESS"
    skipped=$((skipped + 1))
    continue
  fi

  out=""
  set +e
  out=$(./run_single.sh --id "$id" --timeout "$timeout" 2>&1)
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "$out" | tail -3
    passed=$((passed + 1))
    status="PASS"
  else
    echo "$out" | tail -8
    failed=$((failed + 1))
    status="FAIL"
  fi
  echo "{\"id\":\"$id\",\"rel_path\":\"$rel\",\"status\":\"$status\"}" >> "$PROGRESS"
done < <(python3 -c "import json; [print(json.dumps(e)) for e in json.load(open('manifest.json'))]")

python3 -m common.summarize_results
echo "Done: passed=$passed failed=$failed skipped=$skipped total=$TOTAL"
