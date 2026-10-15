"""ChimeraX GridData for OME-Zarr images, modelled on chimerax.map_data.ims.ims_grid."""

import math

import numpy as np
from chimerax.map_data import GridData

from .ngff import display_shift, eight_bit_ok, to_uint8
from .store import _BACKGROUND_POOL, _POOL


class NGFFGrid(GridData):
    """One (time point, channel, level) of an ngff.Image."""

    def __init__(self, image, level, t, c, name, rgba, shift=None):
        self.image = image
        self.level = level
        self.t = t
        self.c = c
        self.shift = shift  # None: values as stored; else shown as uint8 of value >> shift
        lev = image.levels[level]
        # volume_from_grid_data reuses an open grid with the same (path, grid_id),
        # so every (level, t, c) needs its own id.
        value_type = image.value_type if shift is None else np.dtype(np.uint8)
        GridData.__init__(self, image.level_size(level), value_type=value_type,
                          origin=image.level_origin(level), step=image.level_step(level),
                          name=name, path=image.root, file_type="ngff",
                          grid_id=f"{lev.path} t={t} c={c}", default_color=rgba)

    def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
        m = self.image.read_block(self.level, self.t, self.c, ijk_origin, ijk_size, ijk_step)
        return m if self.shift is None else to_uint8(m, self.shift, _POOL)

    def read_ahead(self, ijk_origin, ijk_size, ijk_step):
        """(matrix, any_data) for the prefetcher, read on its own threads.

        any_data is False when none of the chunks the block touches is in the store: the
        time point has not been written yet (a live acquisition), so the block of fill
        values must not be cached. Chunks missing among others are normal (all-zero chunks
        are not written) and give fill values, as they do when ChimeraX reads.
        """
        report = {}
        m = self.image.read_block(self.level, self.t, self.c, ijk_origin, ijk_size, ijk_step,
                                  background=True, report=report)
        if self.shift is not None:
            m = to_uint8(m, self.shift, _BACKGROUND_POOL)
        return m, report.get("missing", 0) < report.get("chunks", 1)


def rendering_options(image, projection="mip", gpu_colormap=True):
    """Initial rendering options for the volumes of an image (GridData.initial_rendering_options).

    projection "mip" shows a maximum-intensity projection (channels are then drawn one over
    the other with a max blend, the usual fluorescence overlay); "blend" is ChimeraX's own
    blended image. Both use 3-D textures (ChimeraX would otherwise pick its projection from
    the single plane it shows while opening, before show_full_region widens it).
    gpu_colormap colours on the GPU: the 16-bit data is uploaded as it is and mapped through
    a 1-D colour table, instead of colouring every plane on the CPU into an RGBA texture.
    """
    options = {"colormap_on_gpu": bool(gpu_colormap)}
    if min(image.level_size(0)) >= 8:
        options["projection_mode"] = "3d"
    if projection == "mip":
        options["maximum_intensity_projection"] = True
    elif gpu_colormap:
        options["blend_on_gpu"] = True
    return options


def image_grids(image, levels="auto", projection="mip", gpu_colormap=True, bits=16):
    """Grids for every (t, c) of an image, each wrapping its subsample levels.

    Every grid gets initial_image_thresholds, so ChimeraX never reads a hidden frame
    just to choose its levels (in scripts it would otherwise read all of them while
    opening). levels="auto" uses a linear ramp from the background to the maximum of time
    point 0 at the coarsest level; "omero" uses the channel's OMERO window where there is
    one. projection and gpu_colormap are described at rendering_options(). bits=8 shows
    unsigned 16-bit data as uint8 (value >> a per-channel shift): half the memory, half the
    bytes to the GPU, and image levels in those units; shifts are in the returned info.
    Returns (grids, warnings).
    """
    from chimerax.map_data.subsample import SubsampledGrid

    sublevels = image.subsample_levels()
    coarsest = sublevels[-1] if sublevels else (0, (1, 1, 1))
    thresholds = []
    for c, ch in enumerate(image.channels):
        if levels == "omero" and ch.window is not None:
            start, end = ch.window
            thresholds.append(((start, 0.0), (end, 1.0)))
        else:
            thresholds.append(auto_image_levels(image, coarsest[0], c))

    shifts = [None] * image.nc
    if bits == 8 and eight_bit_ok(image.levels[0].array.dtype):
        shifts = [display_shift(th[-1][0]) for th in thresholds]
        thresholds = [tuple((v / 2**sh, b) for v, b in th) for th, sh in zip(thresholds, shifts)]
    image.display_shifts = shifts

    options = rendering_options(image, projection, gpu_colormap)
    nt, nc = image.nt, image.nc
    grids = []
    for t in range(nt):
        for c in range(nc):
            ch = image.channels[c]
            name = ch.label or image.name
            g = NGFFGrid(image, 0, t, c, name, ch.rgba, shifts[c])
            if sublevels:
                g = SubsampledGrid(g)
                for k, cell in sublevels:
                    g.add_subsamples(NGFFGrid(image, k, t, c, name, ch.rgba, shifts[c]), cell)
            if nt > 1:
                g.series_index = t
            if nc > 1:
                g.channel = c
            g.initial_style = "image"
            g.initial_image_thresholds = thresholds[c]
            g.initial_rendering_options = options
            grids.append(g)
    return grids, list(image.warnings)


# Voxel budget for the statistics read when the coarsest level is large: ChimeraX's
# default image voxel limit (16 Mvoxels).
_STATS_VOXELS = 16 * 2**20


def auto_image_levels(image, k, c):
    """ChimeraX's initial image levels (Volume.initial_image_levels) for channel c.

    Computed from time point 0 of level k, read at the power-of-two step that keeps it
    under 16 Mvoxels: the 10th percentile (background) is black and the maximum is full
    brightness, joined by a straight line. ChimeraX's own rule bends the curve through the
    99th percentile at 0.8; a single linear ramp keeps brightness proportional to intensity.
    One deviation from ChimeraX: constant data (an empty store) is shown dark, where its
    rule puts the single value at full brightness and draws a solid white block.
    """
    from chimerax.map_data import MatrixValueStatistics

    size = image.level_size(k)
    step = 1
    while math.prod(-(-n // step) for n in size) > _STATS_VOXELS:
        step *= 2
    m = image.read_block(k, 0, c, (0, 0, 0), size, (step, step, step))
    ms = MatrixValueStatistics([m])
    vlow, vmax = ms.rank_data_value(0.10), ms.maximum
    if ms.minimum >= vmax:
        return ((vmax, 0.0), (vmax + 1.0, 1.0))
    if not vlow < vmax:
        vlow = ms.minimum
    return ((vlow, 0.0), (vmax, 1.0))


def show_full_region(volumes):
    """Show the whole volume where ChimeraX chose a single plane, if a subsample level serves it.

    ChimeraX shows only the middle z plane of maps over 256 Mvoxels, because reading
    them whole at a coarse step is slow. With multiscale levels it is cheap: the
    voxel-limit step is read from a coarser level.
    """
    from chimerax.map.volume import faces_per_axis, ijk_step_for_voxel_limit
    from chimerax.map_data.subsample import SubsampledGrid

    for v in volumes:
        d = v.data
        ijk_min, ijk_max, _ = v.region
        if not isinstance(d, SubsampledGrid) or d.size[2] == 1 or ijk_min[2] != ijk_max[2]:
            continue
        ro = v.rendering_options
        full_max = tuple(s - 1 for s in d.size)
        step = ijk_step_for_voxel_limit((0, 0, 0), full_max, (1, 1, 1),
                                        faces_per_axis(True, ro.image_mode),
                                        ro.limit_voxel_count, ro.voxel_limit)
        if tuple(d.choose_subsampling(step)[0]) == (1, 1, 1):
            continue  # would read full resolution at a coarse step: keep the plane
        v.new_region((0, 0, 0), full_max, step, adjust_step=False)
