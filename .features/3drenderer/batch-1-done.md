# Batch 1 — Completed

Core 3D renderer infrastructure, primary-channel controls, toolbar, and overlay **data** path. UI for per-overlay LUT/opacity and several parity items remain for Batch 2.

Baseline commit: [`e9c6311b`](https://github.com/SchmollerLab/Cell_ACDC/commit/e9c6311b) · Fork HEAD: [`14edb05c`](https://github.com/SchmollerLab/Cell_ACDC/commit/14edb05c)

## Honored commits mapping

How upstream ElpadoCan commits map to batch 1 deliverables on this branch. Full table: [README — Commits to honor](README.md#commits-to-honor).

| Commit | Delivered feature(s) | Where in codebase |
|--------|----------------------|-------------------|
| `a3ad90e9` | Initial GPU 3D z-stack renderer | [`renderer3d.py`](../../cellacdc/renderer3d.py) — `VolumeRenderer3DWindow`, vispy canvas |
| `6e9efa5f` | CI test dependencies (vispy, PyOpenGL) | `requirements_test.txt`, CI workflows |
| `5fdb9237` | Module-level GUI skip for CI | [`tests/test_renderer3d.py`](../../tests/test_renderer3d.py) — restored in `14edb05c` |
| `537cd967` | Decouple 3D from z-projection dropdown | No `_ZPROJMODE_3D`; launch via dedicated button only |
| `1dd00e8e` | Launch 3D renderer button with `:3d.svg` | [`gui.py`](../../cellacdc/gui.py) — `launch3dRendererButton`, `launch3dRendererAction` |
| `391377c1` | Multiple overlay volumes | `_volume_nodes` dict with `overlay:N` keys (evolved from parallel list) |
| `e8aceb63` | Multi-channel volume nodes (one per channel) | Per-channel primary keys in `_volume_nodes`; fixed in `e9c6311b` |
| `e9c6311b` | Batch 1 completion on multi-channel model | Checklist items 2–11 below; colormap helpers in [`colors.py`](../../cellacdc/colors.py) |
| `14edb05c` | CI + renderer UI conventions | `pytest.skip` restore; `relative_step_size=current_step` in renderer |

**Superseded (refactor, patterns retained):** `0f64a810`, `3f0d6f2a`, `ca5f179d` — test fixes for GUI init/cleanup; current tests follow the same intent with updated structure.

**Preservation rules for later batches:** keep unified `_volume_nodes` (primary + `overlay:N`); keep `vispy_cmap_from_spec` / `pg_to_vispy_cmap` in `colors.py`; do not reintroduce `_ZPROJMODE_3D`.

## Checklist items delivered

| # | Feature | Implementation |
|---|---------|----------------|
| 2 | Auto + Full LUT buttons | `_add_lut_items()`, `_on_auto_clim()`, `_on_full_clim()` |
| 3 | Colormap from LUT slider | `_on_lut_changed()` → `set_cmap()` → `pg_to_vispy_cmap` |
| 4 | Gamma slider + numeric | `VolumeRendererControls._gamma_spin` (`sliderWithSpinBox`) |
| 5 | Step slider + numeric | `VolumeRendererControls._step_spin` |
| 6 | Opacity control (primary) | Form-row `sliderWithSpinBox` per channel — **not** right-side colorbar yet |
| 7–9 | Home toolbar + Save + `H` | `VolumeRendererToolbar`, wired in `_init_ui()` |
| 10–11 | Overlay segm + fluo volumes | `_get_overlay_zstacks()` → `update_overlay_volumes()` |
| — | Launch 3D renderer button | `launch3dRendererButton` / `launch3dRendererAction` (`1dd00e8e`, `537cd967`) |

## Primary channel LUT

[`renderer3d.py`](../../cellacdc/renderer3d.py) `_add_lut_items()` creates one [`baseHistogramLUTitem`](../../cellacdc/widgets.py) per channel with:

- Auto / Full buttons above each histogram
- `sigLookupTableChanged` → `set_clim` + `set_cmap`
- `include_rescale_lut_options=False` (main GUI uses full `myHistogramLUTitem` menu)

Partial vs main GUI (`gui.py` `imgGrad`):

- Uses `baseHistogramLUTitem`, not `myHistogramLUTitem`
- No "Clim:" axis labelling parity (Batch 2)
- No rescale-intensities menu (2D image / z-stack / time)

## Toolbar

[`VolumeRendererToolbar`](../../cellacdc/widgets.py):

- **Home view** — `:home.svg` icon, shortcut `H`, emits `sigHomeView` → `reset_view()`
- **Save** — `:file-save.svg`, shortcut `Ctrl+S`, emits `sigSave` → `save_screenshot()`

Wired in `VolumeRenderer3DWindow._init_ui()`.

## Rendering controls panel

`VolumeRendererControls` provides:

- Gamma, step, per-primary-channel opacity (form sliders)
- Rendering mode, interpolation, ISO threshold, attenuation, depiction mode, z-plane slider

Overlay volumes pick up gamma/step/interp at creation time in `_init_overlay_volume_node()`.

## Overlay data path

### GUI side — [`gui.py`](../../cellacdc/gui.py)

`_get_overlay_zstacks()` returns `list[tuple]` of `(data, opacity, cmap[, mode])`:

1. **Fluorescence overlays** — checked overlay channels, alpha scrollbar value, hardcoded `_FLUO_CMAPS[i]`
2. **Primary segmentation mask** — when draw mode includes "overlay segm. masks" and `labelsAlphaSlider > 0`; binary `(lab > 0)` volume
3. **Overlay label channels** — when overlay-labels button active; binary masks with `_LABEL_CMAPS[j]`

Launched and refreshed from:

- `_launch_3d_renderer()`
- `_update_3d_renderer_if_active()` (frame/position navigation)

Voxel sizes: `_get_current_voxel_sizes()` reads `PhysicalSizeZ/Y/X` from `posData`.

### Renderer side — [`renderer3d.py`](../../cellacdc/renderer3d.py)

- Unified `_volume_nodes` dict: primary keys = channel names; overlays = `overlay:0`, `overlay:1`, …
- `update_overlay_volumes()` replaces overlay nodes on each call
- `_normalize_overlay_volume()` min–max normalizes to [0, 1]
- `_init_overlay_volume_node()` applies `vispy_cmap_from_spec(cmap_spec)` and stored opacity

## Colormap conversion

| Path | Function | Used for |
|------|----------|----------|
| PG gradient → vispy | [`pg_to_vispy_cmap`](../../cellacdc/colors.py) | Primary LUT changes |
| Plain colour name | [`vispy_cmap_from_spec`](../../cellacdc/colors.py) | Overlay volumes |

## z-anisotropy (transform only)

`set_voxel_scale(dz, dy, dx)` stores physical µm sizes and applies vispy `STTransform` in `_apply_voxel_scale()`, accounting for GPU downsampling strides (`_last_strides`). No UI control; no `ndimage.zoom` resampling.

## Launch adapter

[`_GuiWinRenderer3DAdapter`](../../cellacdc/gui.py) implements `VolumeRendererAdapter`:

- `get_current_zstack()` → `_get_current_zstack()`
- `get_voxel_sizes()` → `_get_current_voxel_sizes()`

## CI

[`tests/test_renderer3d.py`](../../tests/test_renderer3d.py) uses a module-level `pytest.skip` for GUI/OpenGL environments. Run locally: `pytest tests/test_renderer3d.py -v`.

## Known gaps carried to later batches

- Right-side grayscale opacity colorbar (`imgGradRight` pattern)
- In-renderer overlay LUT + opacity sliders with live sync
- `myHistogramLUTitem` / "Clim:" labelling parity
- Overlay colours ignore main GUI `lutItem` gradients
- Cell ID isolation
- z-anisotropy numeric UI and zoom-based correction
