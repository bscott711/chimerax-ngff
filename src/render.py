"""Hand a time point's data to OpenGL as it is, instead of copying it plane by plane.

ChimeraX builds the 3-D texture of an image by looping over its z planes, each fetched from
the data cache and copied into a new array (a single-threaded copy of 882 MB per channel at
full resolution, about 0.2 s). With the colour map on the GPU the planes are the raw values,
so the stacked array is just the block of data ChimeraX already holds: for the whole volume,
the very array the prefetcher read. install() lets NGFF volumes upload that block directly.
Everything else keeps ChimeraX's own code.
"""

_installed = False


def install():
    """Wrap Texture3dPlanes._texture_3d_data once; other data formats are unaffected."""
    global _installed
    if _installed:
        return
    from chimerax.map import image3d

    stock = image3d.Texture3dPlanes._texture_3d_data

    def texture_3d_data(self):
        block = _direct_block(self, image3d)
        return stock(self) if block is None else block

    image3d.Texture3dPlanes._texture_3d_data = texture_3d_data
    _installed = True


def _direct_block(planes, image3d):
    """The 3-D array the stock loop would build, or None when it must be left to ChimeraX."""
    ir = planes._image_render
    if isinstance(ir, image3d.BlendedImage) or not ir._rendering_options.colormap_on_gpu:
        return None
    if getattr(ir._data, "file_type", None) != "ngff":
        return None
    ijk_min, ijk_max, ijk_step = ir._texture_region
    z = 2
    kstep = ijk_step[z]
    k0 = kstep * (ijk_min[z] // kstep)
    k1 = kstep * (ijk_max[z] // kstep)
    planes_z = (k1 - k0 + kstep) // kstep
    origin = (ijk_min[0], ijk_min[1], k0)
    size = (ijk_max[0] - ijk_min[0] + 1, ijk_max[1] - ijk_min[1] + 1, k1 - k0 + 1)
    m = ir._data.matrix(origin, size, ijk_step)
    nx = (size[0] + ijk_step[0] - 1) // ijk_step[0]
    ny = (size[1] + ijk_step[1] - 1) // ijk_step[1]
    if m.ndim != 3 or tuple(m.shape) != (planes_z, ny, nx):
        return None
    return m
