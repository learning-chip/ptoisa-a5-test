"""Generate a small-shape sparse flash attention smoke script from the upstream example."""

from __future__ import annotations

from pathlib import Path

SRC = Path("/workdir/tilelang-ascend/examples/sparse_flash_attention/example_sparse_flash_attn.py")
OUT = Path(__file__).resolve().parent / "sparse_flash_attn_small.py"

REPLACEMENTS = [
    ("seq_len = 128", "seq_len = 64"),
    ("seq_len_kv = 32768", "seq_len_kv = 256"),
    ("topk=2048,", "topk=64,"),
    ("B, S, SKV, H, HKV, DQK, DV, topk = 1, 128, 32768, 128, 1, 576, 512, 2048", "B, S, SKV, H, HKV, DQK, DV, topk = 1, 64, 256, 16, 1, 576, 512, 64"),
    ("q_start_s_index = 4096 * 7", "q_start_s_index = 64"),
]

header = '''"""Small-shape sparse flash attention smoke (generated wrapper)."""
import common.bootstrap  # noqa: F401

'''
text = SRC.read_text()
for old, new in REPLACEMENTS:
    text = text.replace(old, new)

OUT.write_text(header + text)
