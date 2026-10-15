"""stats.value_counts: the histogram ChimeraX's MatrixValueStatistics defines, faster."""

import numpy as np
import pytest
from cxngff import stats

RNG = np.random.default_rng(5)


def reference(m, bins=10000):
    """The definition: bins centred on min and max, one count per voxel."""
    lo, hi = float(m.min()), float(m.max())
    first, last = (lo, hi) if hi > lo else (lo - 1, hi + 1)
    bsize = (last - first) / (bins - 1)
    start = first - 0.5 * bsize
    idx = np.clip(np.floor((m.astype(np.float64) - start) / bsize), 0, bins - 1).astype(int)
    return int(lo), int(hi), np.bincount(idx.ravel(), minlength=bins).astype(np.float64)


@pytest.mark.parametrize("dtype", ["u1", "u2", "i2", "i1"])
def test_exact_for_small_blocks(dtype):
    info = np.iinfo(dtype)
    m = RNG.integers(max(info.min, -500), min(info.max, 900), (7, 30, 40)).astype(dtype)
    lo, hi, counts = stats.value_counts(m)
    rlo, rhi, rcounts = reference(m)
    assert (lo, hi) == (rlo, rhi)
    np.testing.assert_array_equal(counts, rcounts)
    assert counts.sum() == m.size


def test_constant_block_gets_a_range_round_it():
    m = np.full((4, 5, 6), 7, "u2")
    lo, hi, counts = stats.value_counts(m)
    rlo, rhi, rcounts = reference(m)
    assert (lo, hi) == (7, 7)
    np.testing.assert_array_equal(counts, rcounts)


def test_large_blocks_are_sampled_and_scaled_back(monkeypatch):
    monkeypatch.setattr(stats, "SAMPLE_VOXELS", 20000)
    m = RNG.gamma(2.0, 30.0, (60, 70, 80)).clip(0, 65535).astype("u2")
    assert stats.sample_stride(m.shape) > 1
    lo, hi, counts = stats.value_counts(m)
    rlo, rhi, rcounts = reference(m)
    assert (lo, hi) == (rlo, rhi)  # extremes stay exact
    assert counts.sum() == pytest.approx(m.size, rel=1e-6)
    assert np.abs(counts - rcounts).sum() / m.size < 0.1  # same histogram within sampling noise


def test_can_count_only_small_integers():
    assert stats.can_count(np.zeros((2, 2, 2), "u2"))
    assert not stats.can_count(np.zeros((2, 2, 2), "f4"))
    assert not stats.can_count(np.zeros((2, 2, 2), "u4"))
    assert not stats.can_count(np.zeros((0, 2, 2), "u2"))
