"""open command entry point for OME-Zarr stores."""

import os

# Initial image levels: "auto" keeps ChimeraX's histogram-based levels, "omero" uses
# each channel's OMERO window as a linear ramp from start (dark) to end (full).
# This is an internal switch for comparing the two looks; the environment variable
# CHIMERAX_NGFF_LEVELS=auto|omero overrides it.
INITIAL_LEVELS = "auto"


def levels_mode():
    mode = os.environ.get("CHIMERAX_NGFF_LEVELS", "").strip().lower() or INITIAL_LEVELS
    return mode if mode in ("auto", "omero") else "auto"


def projection_mode():
    """"mip" (default) or "blend", from CHIMERAX_NGFF_PROJECTION."""
    mode = os.environ.get("CHIMERAX_NGFF_PROJECTION", "").strip().lower()
    return mode if mode in ("mip", "blend") else "mip"


def gpu_colormap():
    """Colour on the GPU unless CHIMERAX_NGFF_GPU_COLORMAP is 0/false/no/off."""
    value = os.environ.get("CHIMERAX_NGFF_GPU_COLORMAP", "").strip().lower()
    return value not in ("0", "false", "no", "off")


def display_bits():
    """8 (default) or 16 (values as stored), from CHIMERAX_NGFF_DISPLAY_BITS."""
    return 16 if os.environ.get("CHIMERAX_NGFF_DISPLAY_BITS", "").strip() == "16" else 8


def open_ngff(session, path, file_name, series=None, bits=None, homeview=None, align=None):
    """Open a local OME-Zarr store as volumes; returns (models, status message).

    Unsigned 16-bit data is shown as 8-bit (value >> a per-channel shift) unless bits=16 (or
    CHIMERAX_NGFF_DISPLAY_BITS=16): half the memory and half the bytes to the GPU, so twice
    as many time points stay resident. The camera starts at the lab's home view (the one napari
    opens with; see homeview.py) unless homeview=false or CHIMERAX_NGFF_HOME_VIEW=0. A channel alignment saved
    beside the store by chanalign (see alignment.py) moves the channels it lists, unless align=false or
    CHIMERAX_NGFF_ALIGN=0.
    """
    from chimerax.core.errors import UserError

    from .grid import image_grids, show_full_region
    from .ngff import NGFFError, open_image
    from .store import ZarrError, no_hugepage_stalls

    no_hugepage_stalls()
    from .render import install as install_direct_upload
    install_direct_upload()
    if bits not in (None, 8, 16):
        raise UserError(f"bits must be 8 or 16, not {bits}")
    path = path.rstrip("/") or "/"
    try:
        image = open_image(path, series=series)
        from . import alignment
        align_note = alignment.apply_to_image(image, path, alignment.enabled(align))
        grids, warnings = image_grids(image, levels=levels_mode(), projection=projection_mode(),
                                      gpu_colormap=gpu_colormap(), bits=bits or display_bits())
    except (NGFFError, ZarrError) as e:
        raise UserError(str(e)) from None

    name = (file_name or "").rstrip("/") or os.path.basename(path)
    for w in warnings:
        session.logger.warning(f"{name}: {w}")
    if align_note:
        session.logger.info(f"{name}: {align_note}")

    from chimerax.map.volume import Volume, open_grids
    with _deferred_volume_updates(session):
        models, _ = open_grids(session, grids, name)
        volumes = [v for m in models for v in m.all_models() if isinstance(v, Volume)]
        show_full_region(volumes)
    from . import homeview as home
    if home.enabled(homeview):
        home.schedule(session, models)
    from .prefetch import start
    start(session, models)
    return models, describe(image, name)


class _deferred_volume_updates:
    """Open as in interactive use, where volume drawings update when they are drawn.

    In scripts (.cxc files, --script) ChimeraX updates each volume's drawing as soon
    as its region is set, which reads the data of every hidden time point of a series
    while it opens. With the flag cleared only shown volumes are read: ChimeraX's
    volume update manager still updates them when the open command finishes.
    """

    def __init__(self, session):
        self.session = session
        self.saved = None

    def __enter__(self):
        flag = getattr(self.session, "in_script", None)
        if flag:
            try:
                self.session.in_script = type(flag)()  # a fresh, unset flag
                self.saved = flag
            except Exception:
                self.saved = None
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            self.session.in_script = self.saved
        return False


def describe(image, name):
    """One-line summary for the log: shape, levels, time points, channels, voxel size."""
    nx, ny, nz = image.level_size(0)
    sx, sy, sz = image.level_step(0)
    unit = image.unit or "(no unit)"
    parts = [f"Opened {name}"]
    if image.series is not None:
        parts[0] += f" series {image.series}"
    version = f"OME-NGFF {image.version}" if image.version else "zarr array"
    parts.append(version)
    parts.append(f"size {nx} x {ny} x {nz}")
    parts.append(f"{len(image.levels)} level{'s' if len(image.levels) != 1 else ''}")
    frames = f"{image.nt} time point{'s' if image.nt != 1 else ''}"
    if image.nt > 1 and image.time_step is not None and image.time_unit:
        frames += f" (time step {image.time_step:g} {image.time_unit})"
    parts.append(frames)
    labels = [ch.label for ch in image.channels if ch.label]
    chans = f"{image.nc} channel{'s' if image.nc != 1 else ''}"
    if labels:
        chans += " (" + ", ".join(labels) + ")"
    parts.append(chans)
    parts.append(f"voxel size {sx:g} x {sy:g} x {sz:g} {unit}")
    parts.append(str(image.levels[0].array.dtype.newbyteorder("=").name))
    if image.display_shifts and any(sh is not None for sh in image.display_shifts):
        shown = ", ".join(f"{ch.label or f'channel {c}'} / {2 ** sh}"
                          for c, (ch, sh) in enumerate(zip(image.channels, image.display_shifts)))
        parts.append(f"shown as 8-bit (values {shown})")
    return ", ".join(parts)
