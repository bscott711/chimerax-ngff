# Handoff: chimerax-ngff

Started 2026-10-14 from a session in `~/projects/asset-forge`; work continues from this repo.
The approved plan is `docs/PLAN.md`. This file records the facts that plan rests on, so a
new session does not have to re-derive them.

## Why this repo exists

ChimeraX 1.12 on Argus (`~/software/chimerax/usr/bin/chimerax-daily`, Python 3.11, numpy 1.26.4)
could not open our OME-Zarr stores.

- The installed plugin was ChimeraX-OME-Zarr 0.7.0 (uermel), in the ChimeraX user site.
  It imports four zarr-2-only APIs (`zarr.attrs`, `zarr.core.Array`, `zarr.storage.FSStore`,
  `zarr.LRUStoreCache`); its unpinned dependencies resolved to zarr 3.1.6, so it failed on import.
- The same dependency install put numpy 2.2.6 in the user site, shadowing ChimeraX's numpy 1.26.4.
- Upstream's zarr-3 version (1.0.0-alpha.1) requires Python >= 3.14 and ChimeraX-Core >= 1.13.dev.
- Even on zarr 2, 0.7.0 cannot open our layout-3 stores: it zips `group.arrays()` against the
  multiscales datasets, and our root group holds no arrays (datasets are at `0/0`, `0/1`, `0/2`).
- 0.7.0 converts micrometres to angstroms (x 1e4). ChimeraX's own OME-TIFF reader uses
  PhysicalSize as given (`map_data/imagestack/ome_tiff.py`).

User decisions (2026-10-14): a new public repo and bundle (this one) rather than a fork; rolled
out to all Argus users through the shared ChimeraX install; never modify upstream uermel code.

## Done so far (2026-10-14)

- Moved the broken user site (`~/.local/share/ChimeraX/1.12/{lib,bin}`) to
  `~/archive/chimerax-user-site-2026-10-14/` (README inside says how to restore).
  ChimeraX then loads numpy 1.26.4 from its own site-packages.
- Phase A bundle v0.1 on branch `feat/bundle` (PR to main): `src/` reader and opener,
  portable tests (zarr 2 and 3 writers), opt-in real-store tests, `tests/chimerax/`
  smoke, timing, render and GUI scripts. Installed for testing in bscott's ChimeraX
  user site only (`devel install ... user true`); the system-wide install is pending.

## Found while building v0.1

- **Pyproject openers run in batch mode.** The classifier carries `batch:false` as a
  string, and `open_command/manager.py` tests `if batch or not check_path` without
  converting it, so `open()` gets a list of paths and the `name` option (often None).
  `src/__init__.py` handles both forms.
- **Scripts read every hidden frame on open.** In scripts (.cxc, `--script`) ChimeraX
  updates each volume's drawing as soon as its region is set, which computes levels and
  reads data for all 150-300 hidden volumes of a series (Cell_002: 9.3 s, 301 reads).
  `open.py` runs `open_grids` with a fresh `session.in_script` flag and every grid gets
  `initial_image_thresholds` (computed once per channel), so opening reads 3 regions
  (0.11 s). Displayed volumes are still drawn by ChimeraX's command-finished handler.
- **Large maps open as one plane.** ChimeraX shows only the middle z plane of maps over
  256 Mvoxels; with levels the bundle switches to the full volume at the voxel-limit
  step (step 4 = level 2 for our stores).
- **`open x.zarr/` without a format fails** in ChimeraX's suffix detection ("has no
  suffix"); `format ngff` works.
- **Sessions do not restore** NGFF volumes: `map/session.py` reopens data through the
  `map_data` format registry, which has no `ngff` entry.
- **The Cell_009 backfill store is all zeros**, as are its DSR TIFFs (`decon_stage/
  Decon/DSR_decon`, checked with tifffile) and its MIP stills. All 11 backfill stores
  under `/mmfs2/.../opym_backfill/outputs/.../microscopy/Cell_0*` (written 2026-10-14)
  decode to 0 at level 1 for their first, middle and last time points; zarr-python
  agrees for Cell_009. That is a pipeline problem, not a reader one.
  `20260924-SVO-memNG-mScar2xFYVE-FLM-Macropinocytosis/Cell_006` (flat, 2 channels,
  real signal) stands in for colour checks.
- The OMERO window written by opym (0-300) renders our data nearly black (values are
  about 0-100), so ChimeraX's automatic levels stay the default (`CHIMERAX_NGFF_LEVELS`).

## Facts about ChimeraX 1.12 that the design uses

- `chimerax.map.volume.open_grids(session, grids, name)` builds MapSeries / MapChannelsModel /
  MultiChannelSeries from each grid's `series_index` and `channel`. The Imaris reader
  (`map_data/ims/ims_grid.py`) is the model to follow.
- `chimerax.map_data.subsample.SubsampledGrid(level0)` + `add_subsamples(grid, cell_size)` gives
  multiscale reads. `SubsampledGrid.matrix` asks a coarser level for `ceil(size/cell)` voxels,
  which can be one more than the level holds (419 planes -> level 1 has 209, asked for 210), so
  reads must clamp and pad.
- `volume_from_grid_data` reuses an already opened grid with the same `(path, grid_id)`;
  every (t, c) grid needs a distinct `grid_id`.
- imagecodecs 2024.6.1 in ChimeraX has blosc, blosc2, zstd, lz4, zlib, gzip; no CRC32C.
  One real 64x256x256 uint16 blosc-lz4 chunk decodes in 13 ms.
- Data formats with `allow-directory = true` appear in File > Open Folder... (`open_command/dialog.py`).
- The toolshed's catalogue of uninstalled bundles registers uermel's "OME-Zarr" format for `.zarr`
  (version 0.5.3). With two formats on one suffix, `open x.zarr` is ambiguous (error in nogui, a
  prompt in the GUI) unless one format declares `default-for = [".zarr"]`.
- Bundle build: `chimerax --nogui --exit --cmd "devel build <dir>"` reads `pyproject.toml`
  (`chimerax/bundle_builder/bundle_builder_toml.py`, classifier keys in `classifiers.py`).
- `toolshed reload installed` refreshes ChimeraX's installed-bundle cache.

## Our stores (decon output in zarr)

All zarr v2, blosc-lz4, chunks (1,1,64,256,256), uint16, axes t,c,z,y,x, units micrometer,
scale 0.136 isotropic at level 0, 3 levels (x2 per level), OMERO channels with colors and
window 0-300.

- Live lane, bioformats2raw layout 3: root `multiscales` points at `0/0..0/2`; series `0/` is the
  volume, series `1/` is a z-MIP. Example (57 GB, shape (150,1,419,1458,686)):
  `/mmfs1/scratch/jacks.local/microscopy/20261014-SVO-AktPH_mSc-FLM/Cell_002/viewer/Cell_002_dsr.ome.zarr`
- Backfill, flat: root `multiscales` paths `0..2`. Example (shape (150,2,419,1458,539),
  colors 00FF00 / FF00FF):
  `/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/outputs/mmfs1/scratch/jacks.local/microscopy/Cell_009/viewer/Cell_009_dsr.ome.zarr`
- Raw acquisition stores (skewed, not deskewed): 4-D t,z,y,x at `p0`, placeholder scale
  `[1,1,1,1]`, real z step only in the `z` coordinate array. Example:
  `/mmfs1/scratch/jacks.local/microscopy/20261014-SVO-AktPH_mSc-FLM/Cell_002_mScarlet_561.ome.zarr`
- An independent reader for the processed stores exists:
  `~/projects/opym-viewer/src/opym_viewer/volumes.py` (`ZarrStore.read(t, c)`, level 0 only).

## Related places

- argus-desktop (`~/projects/argus-desktop`, local git): `bin/argus-open` refuses zarr for
  ChimeraX today; CONTRACT.md lines 43-46 say so.
- opym-dashboard: only edit the worktree `~/projects/opym-dashboard-open-in-viewer`
  (branch `feat/open-in-viewer`); production serves from `~/projects/opym-dashboard`.
- `bioimaging/backfill/viewer_export.py` writes a `.cxc` that opens the DSR TIFFs; Phase B of
  the plan points it at the zarr instead.
