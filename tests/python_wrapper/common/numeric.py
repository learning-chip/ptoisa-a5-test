"""Numerical comparison matching tests/cpp/common/test_common.h ResultCmp."""

from __future__ import annotations

import numpy as np


def arrays_match(
    expected: np.ndarray,
    actual: np.ndarray,
    eps: float = 0.001,
    threshold: int | None = None,
    zero_count_threshold: int = 1000,
) -> tuple[bool, str]:
    """Return (ok, message) using cpp ResultCmp-style logic."""
    expected = np.asarray(expected).flatten()
    actual = np.asarray(actual).flatten()

    if expected.size != actual.size:
        return False, f"size mismatch: golden={expected.size}, actual={actual.size}"

    if threshold is None:
        threshold = int(expected.size * eps)

    exp_f = expected.astype(np.float64)
    act_f = actual.astype(np.float64)
    diff = np.abs(exp_f - act_f)
    rel = np.where(np.abs(exp_f) > 1e-12, diff / np.abs(exp_f), diff)

    zero_count = int(
        np.sum((np.abs(act_f) <= 1e-6) & (np.abs(exp_f) > 1e-6))
    )
    err_mask = (diff > eps) & (rel > eps)
    err_count = int(np.sum(err_mask))

    max_diff = float(np.max(diff)) if diff.size else 0.0
    ok = err_count <= threshold and zero_count <= zero_count_threshold
    msg = (
        f"max diff: {max_diff}, err count: {err_count}, err threshold: {threshold}, "
        f"zero count: {zero_count}"
    )
    return ok, msg


def assert_arrays_match(
    expected: np.ndarray,
    actual: np.ndarray,
    eps: float = 0.001,
    threshold: int | None = None,
) -> None:
    ok, msg = arrays_match(expected, actual, eps=eps, threshold=threshold)
    if not ok:
        raise AssertionError(msg)
