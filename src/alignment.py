"""Channel alignment: the sidecar `<name>.align.json` written by channelaligner (bscott711/channelaligner, package chanalign).

Pure Python (no ChimeraX import) so the portable tests run it. The contract is the same one argus-desktop's
napari_open.py reads (argus-desktop/CONTRACT.md): the file sits next to `<name>.ome.zarr`, nowhere else.
`shift_um` is (z, y, x) micrometres ADDED to a channel's positions, which here moves the channel's volume origin. A file
for a differently shaped store, or of another version, is ignored.
"""

import json
import os

VERSION = 1
_UM_PER_UNIT = {"micrometer": 1.0, "micron": 1.0, "um": 1.0, "µm": 1.0, "nanometer": 1e-3, "nm": 1e-3,
                "millimeter": 1e3, "mm": 1e3}


def _stem(store_path):
    name = os.path.basename(os.path.realpath(store_path).rstrip("/"))
    for suffix in (".ome.zarr", ".zarr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def sidecar_file(store_path):
    real = os.path.realpath(store_path)
    return os.path.join(os.path.dirname(real), _stem(real) + ".align.json")


def read_alignment(store_path, size_xyz, step_xyz, unit, n_channels):
    """({channel: (x, y, z) offset in the axes' own units}, note) for a store; ({}, note) when there is none to use.

    size_xyz and step_xyz are level 0's voxel counts and sizes, in x, y, z order as ChimeraX uses.
    """
    path = sidecar_file(store_path)
    if not os.path.isfile(path):
        return {}, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}, f"{path}: unreadable alignment file; ignored"
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return {}, f"{path}: not an alignment file this version reads; ignored"
    shape, voxel = data.get("shape_zyx"), data.get("voxel_um")
    sx, sy, sz = step_xyz
    # The file's voxel_um is micrometres; compare in the store's own unit.
    per_unit = _UM_PER_UNIT.get((unit or "micrometer").lower())
    if per_unit is None:
        return {}, f"{path}: the store's unit {unit!r} is not a length unit chanalign knows; ignored"
    nx, ny, nz = size_xyz
    ok = (isinstance(shape, list) and tuple(shape) == (nz, ny, nx) and isinstance(voxel, list) and len(voxel) == 3
          and all(abs(a - b * per_unit) <= 1e-3 * abs(a) for a, b in zip(voxel, (sz, sy, sx))))
    if not ok:
        return {}, f"{path}: made for another store (shape {shape}, voxel {voxel}); ignored"
    offsets = {}
    for m in data.get("moving") or []:
        try:
            c, (dz, dy, dx) = int(m["channel"]), (float(v) for v in m["shift_um"])
        except (KeyError, TypeError, ValueError):
            return {}, f"{path}: unreadable shift; ignored"
        if 0 <= c < n_channels and (dx or dy or dz):
            offsets[c] = (dx / per_unit, dy / per_unit, dz / per_unit)
    return offsets, f"channel alignment from {path} ({data.get('method', '?')})"


def apply_to_image(image, store_path, enabled=True):
    """Set image.channel_offsets from the sidecar; returns a note for the log (or None)."""
    image.channel_offsets = {}
    if not enabled:
        return None
    offsets, note = read_alignment(store_path, image.level_size(0), image.level_step(0), image.unit, image.nc)
    image.channel_offsets = offsets
    if offsets:
        shown = ", ".join(f"{image.channels[c].label or f'channel {c}'} by ({x:+.3f}, {y:+.3f}, {z:+.3f})"
                          for c, (x, y, z) in sorted(offsets.items()))
        return f"{note}: shifted {shown} {image.unit or ''} (x, y, z)".replace("  ", " ")
    return note


def enabled(option=None):
    """The `align` open option, else CHIMERAX_NGFF_ALIGN (0/false/no/off disable it); default on."""
    if option is not None:
        return bool(option)
    return os.environ.get("CHIMERAX_NGFF_ALIGN", "").strip().lower() not in ("0", "false", "no", "off")
