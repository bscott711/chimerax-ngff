"""Value histogram of a time point, fast enough to compute for every frame.

This module must not import chimerax: the portable tests run it outside ChimeraX.

ChimeraX's MatrixValueStatistics walks every voxel twice in single-threaded C++ (about
1.5 s for one 882 MB channel at full resolution), on the user interface thread, each time
a new time point is shown with the Volume Viewer open. Here the minimum and maximum are
exact (numpy, no GIL) and the 10000-bin histogram is counted on a strided sample of at
most SAMPLE_VOXELS voxels, scaled back up, so it can be computed on a prefetch thread.
Blocks up to SAMPLE_VOXELS are counted exactly.
"""

import numpy as np

SAMPLE_VOXELS = 32 * 2**20
_SLAB = 2**21  # voxels counted per numpy call, so the GIL is handed back often


def can_count(m):
    """Whether value_counts handles this array's type (small integers)."""
    return m.dtype.kind in "iu" and m.dtype.itemsize <= 2 and m.size > 0


def sample_stride(shape):
    """Per-axis stride that brings a block under SAMPLE_VOXELS (1 for small blocks)."""
    s = 1
    while np.prod([-(-n // s) for n in shape], dtype=np.int64) > SAMPLE_VOXELS:
        s += 1
    return s


def bin_range(minimum, maximum, bins):
    """Centres of the first and last bin and the bin size (as ChimeraX's bin_range)."""
    if maximum > minimum:
        first, last = minimum, maximum
    else:
        first, last = minimum - 1, maximum + 1
    return first, last, (last - first) / (bins - 1)


def value_counts(m, bins=10000):
    """(minimum, maximum, counts) of an integer array, as MatrixValueStatistics defines them.

    counts has `bins` float64 entries over [minimum, maximum] with the first and last bin
    centred on the extremes. The counts come from a strided sample when the block is large.
    """
    minimum, maximum = int(m.min()), int(m.max())
    first, last, bsize = bin_range(minimum, maximum, bins)
    start = first - 0.5 * bsize
    values = np.arange(minimum, maximum + 1, dtype=np.float64)
    bin_of_value = np.clip(np.floor((values - start) / bsize), 0, bins - 1).astype(np.intp)

    stride = sample_stride(m.shape)
    sample = m if stride == 1 else m[::stride, ::stride, ::stride]
    by_value = np.zeros(maximum - minimum + 1, np.float64)
    flat = sample.reshape(-1) if sample.flags.c_contiguous else sample.ravel()
    for i in range(0, flat.size, _SLAB):
        part = flat[i:i + _SLAB].astype(np.intp)
        if minimum:
            part -= minimum
        by_value += np.bincount(part, minlength=by_value.size)
    if stride > 1:
        by_value *= m.size / sample.size
    counts = np.zeros(bins, np.float64)
    np.add.at(counts, bin_of_value, by_value)
    return minimum, maximum, counts
