# ChimeraX-NGFF

Open OME-Zarr (OME-NGFF) images in [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/).

The bundle is pure Python with **no pip dependencies**: it reads zarr with the numpy and
imagecodecs that ChimeraX already ships, so installing it cannot shadow or upgrade
ChimeraX's own packages. It needs ChimeraX 1.6 or later and is tested on ChimeraX 1.12
(Python 3.11, numpy 1.26.4, imagecodecs 2024.6.1).

## Install

From a wheel (GitHub release or `dist/` after a build):

```
chimerax --nogui --exit --cmd "toolshed install /path/to/chimerax_ngff-0.2.0-py3-none-any.whl"
```

or type `toolshed install /path/to/chimerax_ngff-0.2.0-py3-none-any.whl` in the ChimeraX
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
- **Initial image levels** are a straight line from the background (10th percentile,
  black) to the maximum (full brightness), computed once per channel from the first time
  point at the coarsest level and shared by all time points.
- **Rendering defaults**: a maximum-intensity projection, coloured on the GPU (the 16-bit
  data is uploaded as it is and mapped through a colour table, not coloured plane by
  plane on the CPU). `volume #1 maximumIntensityProjection false` gives ChimeraX's
  blended image back for one volume; `CHIMERAX_NGFF_PROJECTION=blend` does it for every
  store.

### Fast time points

ChimeraX reads, colours and uploads a time point the first time it is shown. For a
882 MB-per-channel time point that was 2 to 5 s per step. The bundle works ahead:

- **Prefetch.** From the moment a series opens, the time points around the one shown
  (ahead in the direction of travel first, wrapping round as the player does) are read on
  background threads, at the resolution and region being shown, into ChimeraX's data
  cache. About two dozen time points are kept ready while you look at one; once you start
  stepping or playing, the window grows to the memory budget. Time points not yet in the
  store (a live acquisition) are looked for again every few seconds.
- **Draw ahead.** When nothing has changed for 0.3 s, the next time points' 3-D textures
  are built one channel at a time, so stepping onto them only swaps a texture. Video
  memory is bounded: at most 48 GB of textures, and never below 24 GB left free for other
  programs on the GPU.
- **Direct upload.** The block the prefetcher read goes to OpenGL as it is, instead of
  being copied plane by plane into a staging array first.
- **Histogram.** The Volume Viewer histogram of each time point is counted on the
  prefetch thread (exact minimum and maximum; the 10000-bin counts come from a strided
  sample for blocks over 32 Mvoxels) instead of by ChimeraX's single-threaded walk over
  every voxel (1.5 s per full-resolution channel).
- **Memory.** Blocks are evicted farthest-from-the-shown-time first, not least recently
  used, because a block read ahead is the newest and the one needed last. numpy's request
  for transparent huge pages is switched off in ChimeraX (it stalled big allocations).

Measured on a 150-time-point, 2-channel store (419 x 1458 x 722 voxels, 1.76 GB per time
point) on an RTX PRO 6000, one new time point per step, full resolution:
v0.1 2.1 s; v0.2 0.04 s within the drawn-ahead window, 0.2 to 0.4 s beyond it (limited by
reading 1.7 GB per step from GPFS). At half resolution (`volume #1 step 2`) and coarser
the whole series fits in memory. See `docs/HANDOFF.md` for the numbers and how they were taken.

Settings (environment variables):

| variable | effect |
|---|---|
| `CHIMERAX_NGFF_CACHE_MB` | size of the decoded-chunk cache, default 1024; 0 or less disables it |
| `CHIMERAX_NGFF_LEVELS` | `auto` (default) or `omero`: use each channel's OMERO window as a linear ramp instead |
| `CHIMERAX_NGFF_PROJECTION` | `mip` (default) or `blend` (ChimeraX's blended image, blended on the GPU) |
| `CHIMERAX_NGFF_GPU_COLORMAP` | `1` (default) or `0`: colour on the CPU, as ChimeraX does by default |
| `CHIMERAX_NGFF_PREFETCH_GB` | memory for time points read ahead; default half of the memory available when a series opens, never leaving less than 48 GB available; `0` turns prefetch off |
| `CHIMERAX_NGFF_PREFETCH_THREADS` | decode threads for prefetch, default 16 |
| `CHIMERAX_NGFF_GPU_GB` | video memory for textures of time points not shown, default 48; `0` turns drawing ahead and the bound off |

## Supported

| | supported | not supported |
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

`tests/chimerax/` also has `prefetch_check.py` (no display), `timing.py`, `render.py`,
`gui_open.py` and `bench.py` (these need a display). `run_gui.sh` starts ChimeraX on a
private Xvfb with Mesa software GL (`NGFF_GUI_GL=software`, measures everything except the
GPU), on a GPU through VirtualGL (does not work with ChimeraX yet), or on the DCV desktop
with its NVIDIA GPU (`NGFF_GUI_GL=dcv`, the one to use for timings), always with a private
HOME so your own ChimeraX settings are untouched. Lint with `ruff check src tests`.

## License

MIT; see `LICENSE`.
