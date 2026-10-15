# ChimeraX-NGFF

Open OME-Zarr (OME-NGFF) images in [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/).

The bundle is pure Python with **no pip dependencies**: it reads zarr with the numpy and
imagecodecs that ChimeraX already ships, so installing it cannot shadow or upgrade
ChimeraX's own packages. It needs ChimeraX 1.6 or later and is tested on ChimeraX 1.12
(Python 3.11, numpy 1.26.4, imagecodecs 2024.6.1).

## Install

From a wheel (GitHub release or `dist/` after a build):

```
chimerax --nogui --exit --cmd "toolshed install /path/to/chimerax_ngff-0.1.0-py3-none-any.whl"
```

or type `toolshed install /path/to/chimerax_ngff-0.1.0-py3-none-any.whl` in the ChimeraX
command line. Add `user false` to install for every user of a shared ChimeraX (needs
write access to its site-packages). Once the bundle is on the ChimeraX Toolshed,
`toolshed install ChimeraX-NGFF` works too.

From source: `chimerax --nogui --exit --cmd "devel install /path/to/chimerax-ngff user true"`.

## Use

```
open /data/cell.ome.zarr
open /data/cell.ome.zarr format ngff          # force this reader
open /data/plate.zarr series 1                # a series group of a bioformats2raw store
```

File ▸ Open Folder... (format "OME-Zarr (OME-NGFF) image") and dragging the folder onto
the ChimeraX window work as well. The format's name is **OME-Zarr (NGFF)** and its
nickname **ngff**; it is the default for `.zarr`, so `open x.zarr` picks it even when the
Toolshed's other "OME-Zarr" bundle is listed. A folder name ending in `/` has no suffix
for ChimeraX, so `open x.zarr/` needs `format ngff`.

What you get:

- **Time points** become a volume series (slider, `vseries` commands) and **channels**
  separate coloured volumes (a multi-channel series when there are both).
- **Colours** come from the OMERO metadata; without it channels are green, magenta,
  cyan, yellow, then gray (a single channel keeps ChimeraX's default colour).
- **Resolution levels**: ChimeraX reads a coarser level whenever it shows the data at a
  coarser step, so a 57 GB time series opens in a fraction of a second. Large images
  open showing the whole volume at the coarsest step that fits ChimeraX's voxel limit
  (ChimeraX would otherwise show only the middle plane); `volume #1 step 1` shows full
  resolution.
- **Units** are used as the store gives them (micrometres stay micrometres, as with
  ChimeraX's OME-TIFF reader); the log line reports the unit, size, levels, time points
  and channels.
- **Initial image levels** follow ChimeraX's own rule, computed once per channel from the
  first time point at the coarsest level and shared by all time points.

Settings (environment variables):

| variable | effect |
|---|---|
| `CHIMERAX_NGFF_CACHE_MB` | size of the decoded-chunk cache, default 1024; 0 or less disables it |
| `CHIMERAX_NGFF_LEVELS` | `auto` (default) or `omero`: use each channel's OMERO window as a linear ramp instead |

## Supported

| | supported | not supported (v0.1) |
|---|---|---|
| storage | local folders | http(s), s3 and other remote stores; zip stores |
| zarr v2 | `.zarray`, `.`/`/` separators, C/F order, null/blosc/zstd/zlib/gzip/lz4/bz2/lzma | filters (refused with the filter's name) |
| zarr v3 | `zarr.json`, default/v2 chunk keys, transpose, bytes (either endian), blosc, zstd, gzip, crc32c (not verified), sharding (index at start or end) | nested sharding, storage transformers, extension data types |
| OME-NGFF | 0.1 to 0.5, axes as dicts or strings or absent, scale and translation (per dataset and per multiscale), bioformats2raw layouts, plain zarr arrays | labels, HCS plates and wells, transforms stored in files |
| axes | t and c optional, 2 or 3 space axes, other axes of length 1 | other axes longer than 1 |
| data types | (u)int8/16/32, float32/64 as is; float16, (u)int64 as float32; bool as uint8 | complex |

Known limitation: ChimeraX sessions save NGFF volumes but cannot restore them yet
("Unknown map file format ngff").

## Development

```
# portable tests (no ChimeraX); zarr-python is only a test-time fixture writer
uv run --no-project --with pytest --with numpy --with imagecodecs --with 'zarr>=3' pytest tests/portable
uv run --no-project --with pytest --with numpy --with imagecodecs --with 'zarr<3' --with 'numcodecs<0.16' pytest tests/portable

# optional: bit-exact reads of the lab's real stores (paths in the test file)
NGFF_REAL_STORES=1 uv run --no-project --with pytest --with numpy --with imagecodecs --with 'zarr>=3' pytest tests/portable/test_real_stores.py

# inside ChimeraX, with the bundle installed
NGFF_SMOKE_FLAT=<2-channel store> NGFF_SMOKE_LAYOUT3=<layout-3 store> \
  chimerax --nogui --exit --script tests/chimerax/smoke.py
```

`tests/chimerax/` also has `timing.py`, `render.py` and `gui_open.py` (the last two need a
display). Lint with `ruff check src tests`.

## License

MIT; see `LICENSE`.
