# ChimeraX OME-Zarr: new `chimerax-ngff` bundle, Argus rollout, then the output-tree cleanup

## Context

ChimeraX 1.12 on Argus (`~/software/chimerax/usr/bin/chimerax-daily`, Python 3.11) cannot open our OME-Zarr stores. Evidence from this session:

- **The installed plugin is broken.** It is ChimeraX-OME-Zarr 0.7.0 (uermel), installed in `~/.local/share/ChimeraX/1.12/lib/python3.11/site-packages`. It imports four zarr-2-only APIs: `zarr.attrs`, `zarr.core.Array`, `zarr.storage.FSStore` and `zarr.LRUStoreCache`. Its unpinned dependencies resolved to zarr 3.1.6, so all four fail (reproduced with ChimeraX's own Python).
- **Those same dependencies put numpy 2.2.6 in the user site.** It shadows ChimeraX's numpy 1.26.4. Everything in that user site came from this one plugin install.
- **Upstream's fix needs a newer ChimeraX.** It is in upstream 1.0.0-alpha.1, which requires Python ≥3.14 and ChimeraX-Core ≥1.13.dev, so it can't be installed on 1.12.
- **0.7.0 can't read our layout either, even on zarr 2.** The plugin zips `group.arrays()` against `multiscales.datasets`. Our live-lane stores are bioformats2raw layout 3: the root multiscales point at `0/0`, `0/1`, `0/2`, and the root group holds no arrays.
- **0.7.0 also rescales units.** It converts µm to Å (×1e4). ChimeraX's own OME-TIFF reader (`map_data/imagestack/ome_tiff.py:135`) uses PhysicalSize as given, in µm.
- **Our decon output in zarr.** This is `<leaf>/viewer/<base>_dsr.ome.zarr`: zarr v2, blosc-lz4 chunks of (1,1,64,256,256), uint16, axes t,c,z,y,x in µm, scale 0.136 isotropic, 3 levels, and an OMERO block (labels, colors 00FF00/FF00FF, window 0–300).
  - Live-lane stores, for example `/mmfs1/.../20261014-SVO-AktPH_mSc-FLM/Cell_002/viewer/Cell_002_dsr.ome.zarr`: layout 3, root paths `0/0..0/2`, with series `1/` being a z-MIP. Shape (150,1,419,1458,686), 57 GB.
  - Backfill stores, for example `/mmfs2/.../outputs/mmfs1/scratch/jacks.local/microscopy/Cell_009/viewer/Cell_009_dsr.ome.zarr`: flat, paths `0..2`, shape (150,2,419,1458,539).
- **ChimeraX already has what a lean bundle needs.**
  - imagecodecs 2024.6.1 ships blosc, blosc2, zstd, lz4 and zlib. Decoding one real chunk with ChimeraX's own Python and numpy 1.26 took 13 ms.
  - `chimerax.map.volume.open_grids` builds MapSeries, MapChannelsModel and MultiChannelSeries from `series_index` and `channel`, and `chimerax.map_data.subsample.SubsampledGrid` handles multiscale levels. The Imaris reader (`map_data/ims/ims_grid.py`) uses exactly this pattern.
  - `allow_directory` formats appear in File ▸ Open Folder… (`open_command/dialog.py:75`).

**User decisions:**
- Build a new public repo, **bscott711/chimerax-ngff** (bundle **ChimeraX-NGFF**), rather than patching the fork.
- Roll it out to **all Argus users**.
- The cleanup means the **per-dataset output tree** opym writes. It comes after ChimeraX works.
- Don't touch upstream uermel code (standing rule).

**Intended outcome:** `open <store>.ome.zarr` works for every Argus user, as does File ▸ Open Folder…, drag-and-drop, and the dashboard/opym:// "Open in ChimeraX" button. It opens a time series with green/magenta channels at the right µm scale, without loading 57 GB.

---

## Phase A: `chimerax-ngff` bundle (priority)

Create the repo at `~/projects/chimerax-ngff` with `gh repo create bscott711/chimerax-ngff --public` and the MIT license. A scaffold commit goes to main. Later changes go on branches, through PRs I merge myself.

```
pyproject.toml          # [chimerax] bundle: ChimeraX-NGFF, category Volume Data, Core >= 1.6, py3-none-any, no pip deps
src/__init__.py         # BundleAPI.run_provider -> NGFFOpenerInfo
src/open.py             # open_ngff(session, path, series=None) -> open_grids(...), applies OMERO window, log message
src/store.py            # PURE (numpy + imagecodecs, no chimerax): zarr v2/v3 metadata, chunk keys, codecs, region reads
src/ngff.py             # PURE: OME-NGFF 0.1-0.5 + bioformats2raw parsing -> levels, axes, step/origin, channels
src/grid.py             # NGFFGrid(GridData) per (t, c, level); SubsampledGrid assembly
src/docs/user/ngff.html
tests/portable/         # pytest, outside ChimeraX
tests/chimerax/smoke.py # run with chimerax --nogui --exit --script
docs/HANDOFF.md         # this session's findings + plan, so work continues from this repo
```

**Format registration (pyproject):**
- Data format "OME-Zarr (NGFF)": nickname `ngff`, suffixes `.zarr`, `allow-directory = true`, category Volume Data.
- Open provider: `want-path = true`, `check-path = false`.
- The name and nickname deliberately differ from uermel's "OME-Zarr"/`ome-zarr`/`omezarr`, so both can exist on the toolshed.

**store.py (reader):**
- **zarr v2:** `.zarray` with dimension_separator `.` or `/`, order C or F, and compressor blosc/zstd/zlib/gzip/lz4/None. Any filter raises a clear error.
- **zarr v3:** `zarr.json` with chunk_key_encoding `default` (`c/…`) or `v2`, and the codecs `transpose`, `bytes` (endian), `blosc`, `zstd`, `gzip`, `crc32c` (stripped, not verified: ChimeraX's imagecodecs lacks CRC32C) and `sharding_indexed` (index at start or end).
- **Reads:** a missing chunk becomes `fill_value`. `read(region_zyx, step)` decodes only the chunks the region touches, on a module ThreadPoolExecutor (imagecodecs releases the GIL), as `opym-viewer/src/opym_viewer/volumes.py:77` does.
- **Cache:** a bounded LRU of decoded chunks, 1 GiB by default, set with `CHIMERAX_NGFF_CACHE_MB`. Plane-by-plane slicing then doesn't re-decode 64-plane chunks.

**ngff.py (metadata):**
- **Root:** use the root `multiscales` if present. Otherwise, with `bioformats2raw.layout`, use series `0`, or `series N`.
- **Axes:** accept dict or string axes (v0.3), and infer tczyx by ndim when axes are missing.
- **Optional axes:** t and c are optional, and other non-space axes raise an error.
- **Per-level transforms:** scale plus optional translation give step and origin per level, in **units as given, not converted**. This matches ChimeraX's OME-TIFF, and the unit goes in the open message.
- **Channels:** `omero.channels` gives label, color and window. Without OMERO, channel 0 is green and channel 1 magenta (lab rule, never red/green).

**grid.py / open.py:**
- **Grids:** one `NGFFGrid(GridData)` per (t, c). Its level 0 is wrapped in `SubsampledGrid`, which gets `add_subsamples(level_k, cell_size)` from integer scale ratios. A non-integer level is skipped with a warning.
- **Grid attributes:** set `series_index = t` when nt > 1, `channel = c` when nc > 1, `rgba` from the colour, the name from the label, and `initial_style = 'image'`.
- **Opening:** hand the grids to `chimerax.map.volume.open_grids(session, grids, name)`. ChimeraX then makes the MultiChannelSeries / MapSeries / MapChannelsModel itself, chooses the initial step under its voxel limit, and reads lazily.
- **Contrast:** apply the OMERO window as the initial image levels. Keep this only if the visual check says it beats ChimeraX's auto levels.
- **Open options:** `series` only. Remote stores (http/s3) and labels are out of scope for v0.1.

### Tests (Phase A)
- **Portable pytest** (`uv run --with pytest --with numpy --with imagecodecs --with 'zarr>=3' …`, plus a `zarr<3` run for v2 writers):
  - Round-trip stores written by zarr-python against numpy truth: v2 C/F order, both separators, each compressor, missing and edge chunks, strided and offset regions; v3 sharded and unsharded, with crc32c.
  - NGFF parsing fixtures: 0.3 string axes, 0.4, 0.5, bioformats2raw, our layout-3 root `0/0`, the flat backfill store, and the raw acquisition store (4-D tzyx, placeholder scale `[1,1,1,1]`).
- **Real stores, compared bit-exact against an independent reader.** Read T0 regions at all 3 levels from the Cell_009 flat store and the 20261014 Cell_002 layout-3 store. Compare with `opym_viewer.volumes.ZarrStore.read` (level 0) and with direct `imagecodecs` decoding (levels 1–2).
- **ChimeraX smoke test** (`chimerax --nogui --exit --script tests/chimerax/smoke.py <store>`), asserting:
  - Cell_009 opens as a MultiChannelSeries with 150 frames, 2 channels, xyz size (539,1458,419), step 0.136, and green and magenta.
  - Cell_002 opens as a MapSeries with 150 frames.
- **Visual check** (judged by image): render T0 of both stores offscreen (`--offscreen`) in image style, plus 3 orthogonal planes. Use the correctness criteria from the zarr DSR geometry memory: a flat coverslip line, a top-down footprint, and depth not the largest dimension. Show them to the user next to the dashboard MIP of the same cell.
- **Timing:** record time to first image and per-frame step on the 57 GB store.

### Install and rollout (all Argus users)
1. **Clear my broken user site.** Move `~/.local/share/ChimeraX/1.12/lib/python3.11/site-packages` to `~/archive/chimerax-user-site-2026-10-14/`. It is reversible, and everything in it came with ChimeraX-OME-Zarr 0.7.0. Then check that ChimeraX starts and reports numpy 1.26.4.
2. **Build the wheel:** `chimerax --nogui --exit --cmd "devel build ~/projects/chimerax-ngff"`.
3. **Install system-wide** into the shared ChimeraX with `… --cmd "toolshed install <wheel> user false"`. I own `~/software/chimerax` and its site-packages is world-readable. Verify as a fresh user with `HOME=$(mktemp -d) chimerax --nogui --exit --script smoke.py`.
4. **argus-desktop** (`~/projects/argus-desktop`, local git):
   - In `bin/argus-open`, let `.zarr`/`.ome.zarr` folders go to ChimeraX with a forced `format ngff`, and remove the refusal at lines 247–251.
   - Update CONTRACT.md lines 43–46 and tests/, commit, then `./deploy`.
5. **opym-dashboard:** only in the worktree `~/projects/opym-dashboard-open-in-viewer` (branch feat/open-in-viewer), never in `~/projects/opym-dashboard`.
   - Show "Open in ChimeraX" for zarr datasets, test on port 8429, commit and push.
   - Deploying to the live dashboard waits for your OK, because production serves straight from that checkout.
6. **Release:**
   - Tag v0.1.0 and publish a GitHub release with the wheel attached. README covers install, usage and the format nickname.
   - Toolshed: you upload the wheel at cxtoolshed.rbvi.ucsf.edu (it needs your account). I prepare the description text.

### Handoff, memory, notebook
- **Continuing from the new repo:** `docs/HANDOFF.md` in the new repo holds this session's evidence and the plan. Later sessions run from `~/projects/chimerax-ngff`.
- **Memory:** add `project_chimerax_ngff.md` in `~/.claude/projects/-home-SDSMT-LOCAL-bscott-projects/memory/` with a MEMORY.md pointer. Correct the Argus-desktop memory's "ChimeraX buttons are TIFF only" line.
- **Lab notebook artifact** (CLAUDE.md rule), published once Phase A works, is committed and pushed:
  - Before: the 4 import failures, the numpy shadowing, TIFF-only buttons.
  - Now: renders, the test matrix, timings.
  - Decision cards: new bundle vs fork, no dependencies, µm units, native `open_grids`/SubsampledGrid, system-wide install.
  - A timeline, with every number checked against its source.
  - Its URL goes in HANDOFF and memory, and Phase B updates the same URL.

---

## Phase B: per-dataset output tree cleanup (after Phase A ships)

**B1. Inventory (read-only).** Cover both roots: live leaves on /mmfs1 and the backfill mirror on /mmfs2. For every folder, record its writer (file:function), every reader, its total size across all datasets, and whether it is redundant now that ChimeraX reads the zarr. Known so far:

| folder | writer | readers / notes |
|---|---|---|
| `decon_stage/Decon/DSR_decon/` | backfill PetaKit (TIFFs, ~713 MB/cell) / live lane (`.live_status.json` only) | `bioimaging/backfill/pipeline.py` (read_live_status), opym-viewer TIFF fallback, `.cxc` |
| `decon_stage/Decon/{MIPs,Masks}`, `parameters.*` | PetaKit by-products | CORE seglab (Masks?) |
| `decon_qc/` | PSF/OTF diagnostics | `bioimaging/backfill/cli.py`, `opym/discovery.py` |
| `qc/rawproj/*.npz` (~1.2 GB/cell), `qc/*.json*` | `opym/stream/qcproj.py` | `CORE/packages/celldet/src/celldet/liveqc.py` |
| `mip_movies/` | `bioimaging/backfill/mip_movie.py` | dashboard |
| `viewer/<base>_dsr.ome.zarr`, `.cxc`, `viewer_export.json` | `opym/ome_zarr_writer.py`, `bioimaging/backfill/viewer_export.py` | opym-viewer, naparym-live, `opym/live_view.py` |

**B2. Propose a target layout to you before writing any code.** One table, with each folder marked keep, flatten, move or delete-after-use, the bytes saved, and the consumers that need updating. Change `.cxc` to `open <store> format ngff` (TIFF only as a fallback).

**B3. Implement on feature branches** in opym_local and bioimaging, plus any affected readers in CORE, opym-viewer, naparym-live and the dashboard.
- Readers accept both the old and new paths.
- A migration script for existing leaves defaults to a dry run.
- Deploy between acquisitions with your OK, then restart opym-serve, opym-receive and opym-backfill.
- PRs are merged by me.

---

## Verification (end to end)
1. The portable pytest suite passes, and the real-store reads are bit-exact against the independent reader.
2. The `--nogui` smoke test passes for both stores, both from my account and from a clean HOME (the fresh-user simulation).
3. Offscreen renders of T0 match the dashboard MIP orientation, and you judge them by eye.
4. In the GUI on DCV, using the Xvfb :97 recipe for automation and then your own check on :2, all of these work:
   - `open …/Cell_009_dsr.ome.zarr`
   - File ▸ Open Folder… with the OME-Zarr (NGFF) format
   - drag-and-drop of the folder
   - an opym:// "Open in ChimeraX" link
   - In each case, time playback works, both channels toggle, and the coloring is green/magenta.
5. After the user-site cleanup, ChimeraX reports numpy 1.26.4, and TIFF opens through argus-desktop still work.
