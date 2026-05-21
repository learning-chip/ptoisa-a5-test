"""Smoke golden helpers for acc2 mix-arch tests (ND path, no quant)."""

from __future__ import annotations

import numpy as np

np.random.seed(19)


def _matmul_golden_u32(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Match cpp gen_data: float32 matmul, compare as uint32 bits."""
    mat = np.matmul(x1.astype(np.float32), x2.astype(np.float32)).astype(np.float32)
    return mat.view(np.uint32).reshape(mat.shape)


def make_nz2nd_3_matmul() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tmov_acc2vec / tmov_acc2mat smoke: float16 x float16 -> uint32, M=6,K=7,N=8."""
    m, k, n = 6, 7, 8
    x1 = np.random.randint(1, 5, size=(m, k)).astype(np.float16)
    x2 = np.random.randint(1, 5, size=(k, n)).astype(np.float16)
    golden = _matmul_golden_u32(x1, x2)
    return x1, x2, golden


def make_nz2nd_3_insert() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """tinsert smoke: write mat into dst at index (2,0) (float32 bits as uint32)."""
    m, k, n = 6, 7, 8
    index_rows, index_cols = 2, 0
    dst_row, dst_col = 10, 16

    x1 = np.random.randint(1, 5, size=(m, k)).astype(np.float16)
    x2 = np.random.randint(1, 5, size=(k, n)).astype(np.float16)
    mat = np.matmul(x1.astype(np.float32), x2.astype(np.float32)).astype(np.float32)

    dst_f = np.zeros((dst_row, dst_col), dtype=np.float32)
    dstm = min(dst_row, index_rows + m)
    dstn = min(dst_col, index_cols + n)
    srcm = min(m, dst_row - index_rows)
    srcn = min(n, dst_col - index_cols)
    dst_f[index_rows:dstm, index_cols:dstn] = mat[0:srcm, 0:srcn]
    golden = dst_f.view(np.uint32)
    return x1, x2, golden, golden.copy()


def make_nz2nd_3_extract() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """textract smoke: extract mat region into dst-shaped golden (float32 bits as uint32)."""
    m, k, n = 6, 7, 8
    index_rows, index_cols = 2, 0
    dst_row, dst_col = 10, 16

    x1 = np.random.randint(1, 5, size=(m, k)).astype(np.float16)
    x2 = np.random.randint(1, 5, size=(k, n)).astype(np.float16)
    mat = np.matmul(x1.astype(np.float32), x2.astype(np.float32)).astype(np.float32)

    dst_f = np.zeros((dst_row, dst_col), dtype=np.float32)
    golden_f = np.zeros((dst_row, dst_col), dtype=np.float32)
    golden_f[: m - index_rows, : n - index_cols] = mat[index_rows:, index_cols:]
    return x1, x2, dst_f.view(np.uint32), golden_f.view(np.uint32)
