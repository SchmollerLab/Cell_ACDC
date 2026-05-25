# Batch 4 — z-Anisotropy UI + Resampling

Expose physical z-anisotropy to the user and optionally correct voxel data with `scipy.ndimage.zoom` instead of relying solely on vispy transform scaling.

## Checklist targets

| # | Feature | Status |
|---|---------|--------|
| 17 | z-anisotropy via `scipy.ndimage.zoom` | **Not started** |
| 18 | z-anisotropy numeric control | **Not started** |

## Goals

1. Add a user-facing z-anisotropy factor (numeric spin box / slider) in the 3D renderer controls.
2. Continue supporting metadata-driven physical sizes from `_get_current_voxel_sizes()`.
3. Optionally resample volume data with `scipy.ndimage.zoom` when anisotropy correction is enabled (vs. transform-only display today).
4. Document trade-offs: resampling quality/cost vs. GPU transform stretch.

---

## 1. Current state (transform only)

[`set_voxel_scale(dz, dy, dx)`](../../cellacdc/renderer3d.py) stores physical µm sizes and applies vispy `STTransform` in `_apply_voxel_scale()`, accounting for GPU downsampling strides (`_last_strides`).

- Voxel sizes flow: `_get_current_voxel_sizes()` → `_launch_3d_renderer()` / `_update_3d_renderer_if_active()` → `set_voxel_scale()`.
- No user override; no `ndimage.zoom` on the numpy volume before upload.

See also [batch-1-done.md — z-anisotropy (transform only)](batch-1-done.md#z-anisotropy-transform-only).

---

## 2. Numeric z-anisotropy control

### Tasks

- [ ] Add control in `VolumeRendererControls` (e.g. "Z anisotropy:" spin box, default from metadata ratio `dz/dy` or `dz/dx`).
- [ ] On change: update `STTransform` scale even when metadata is missing or wrong.
- [ ] Persist value for the session (optional: save in renderer settings JSON if one exists).
- [ ] Show units / help text (physical µm per pixel vs. unitless stretch factor — pick one convention and document).

### Files

| File | Changes |
|------|---------|
| [`cellacdc/renderer3d.py`](../../cellacdc/renderer3d.py) | UI control, `_apply_voxel_scale()` user override |
| [`cellacdc/gui.py`](../../cellacdc/gui.py) | Optional: pass default anisotropy from metadata |

---

## 3. `scipy.ndimage.zoom` resampling path

### Tasks

- [ ] Add toggle or mode: **transform only** (current) vs. **resample volume**.
- [ ] When resampling: compute zoom factors from desired anisotropy and apply `ndimage.zoom` to primary + overlay volumes before GPU upload.
- [ ] Update `_last_strides` / downsampling logic to match resampled shape.
- [ ] Benchmark memory and launch latency; consider lazy resample on anisotropy change only.
- [ ] Match interpolation order with vispy volume interpolation where possible.

### Files

| File | Changes |
|------|---------|
| [`cellacdc/renderer3d.py`](../../cellacdc/renderer3d.py) | Resample helper, hook in `update_volume()` / `update_overlay_volumes()` |
| [`cellacdc/gui.py`](../../cellacdc/gui.py) | Unlikely changes unless adapter passes resample preference |

---

## Design notes

| Approach | Pros | Cons |
|----------|------|------|
| Transform only (`STTransform`) | Fast, no extra memory, current behaviour | Stretched voxels; sampling artifacts on thick Z |
| `ndimage.zoom` resample | Cubic/isotropic voxels for rendering | CPU cost, memory spike, must re-run on data refresh |

Batch 4 should implement the numeric control first (transform path), then add optional resampling behind an explicit user toggle.

## Dependencies

- Batch 1 volume upload and stride logic (complete).
- Cell ID masking (batch 3) should apply **after** resampling if both are active, or resample masked volumes consistently.

## Acceptance criteria

- User can adjust z-anisotropy from the 3D window without editing metadata.
- Default reflects `PhysicalSizeZ/Y/X` when available.
- Optional resampling mode visibly corrects elongated Z voxels and updates on frame sync.
- Transform-only mode remains available as the fast default.
