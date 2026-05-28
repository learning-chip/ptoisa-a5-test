# Reproduction steps

From `/workdir/ptoisa-a5-test/tilelang_sim`:

```bash
# 1. Regenerate example manifest (127 entries)
python3 -m common.collect_examples

# 2. Quick validation (5 examples, ~3 min)
./run_all.sh --pilot

# 3. Full sweep (sequential, ~3 h on this host)
./run_all.sh
# log: results/full_sweep.log

# 4. Single example
./run_single.sh --rel-path gemm/example_gemm.py --timeout 30

# 5. Regenerate summary from logs
python3 -m common.summarize_results
```

## Environment notes

- TileLang-Ascend is pre-installed (`/sources/tilelang-ascend`, env in `~/.bashrc`).
- `run_single.sh` sources CANN `setenv.bash` and prepends `Ascend950PR_9599` simulator libs.
- Examples are executed with `common/bootstrap.py` forcing `target=pto`, `platform=A5`.

## Latest results (full sweep)

See [results/summary.md](results/summary.md) and [results/summary.json](results/summary.json).

PASS example simulator timings: [pass_timings.md](pass_timings.md).

| Status | Count |
|--------|------:|
| PASS | 13 |
| FAIL_CRASH | 86 |
| FAIL_COMPILE | 6 |
| FAIL_ACCURACY | 5 |
| FAIL_TIMEOUT | 5 |
| SKIP | 12 |
