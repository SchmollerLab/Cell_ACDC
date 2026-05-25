# Batch 2 — LUT Polish + Overlay UI

Bring primary LUT and overlay controls in the 3D window to parity with the main GUI, and wire live updates so slider changes apply immediately (not only on frame navigation).

## Checklist targets

| # | Feature |
|---|---------|
| 1 | "Clim:" colorbar slider matching main GUI |
| 6 | Opacity as right-side grayscale colorbar |
| 12 | Opacity sliders for overlaid fluorescence channels |
| 13 | LUT sliders for overlay channels |
| 14 | Segmentation mask opacity slider in 3D UI |

## Goals

1. Upgrade primary LUT toward main GUI behaviour.
2. Add right-side grayscale opacity colorbar (mirror `imgGradRight`).
3. Add in-renderer overlay LUT + opacity controls.
4. Add segmentation mask opacity control in the 3D window.
5. Live-sync overlay control changes to volume nodes.

---

## 1. Primary LUT parity

### Current state

[`renderer3d.py`](../../cellacdc/renderer3d.py) `_add_lut_items()` uses [`baseHistogramLUTitem`](../../cellacdc/widgets.py) with `include_rescale_lut_options=False`.

Main GUI uses [`myHistogramLUTitem`](../../cellacdc/widgets.py) on `guiWin.imgGrad` with full gradient menu, child LUT linkage, and settings restore.

### Tasks

- [ ] Evaluate switching primary LUT to `myHistogramLUTitem` (or extend `baseHistogramLUTitem` with Clim labelling only).
- [ ] Match axis label style — user request specifies **"Clim:"** as the colorbar slider label (main GUI uses channel name via `setAxisLabel`; confirm desired label text).
- [ ] Consider enabling rescale-intensities options if 3D should respect the same rescale policy as 2D (may need adapter hooks to re-fetch normalized data).
- [ ] Keep existing Auto / Full buttons and `_on_lut_changed` → `set_clim` / `set_cmap` wiring.

### Files

| File | Changes |
|------|---------|
| [`cellacdc/renderer3d.py`](../../cellacdc/renderer3d.py) | `_add_lut_items()`, `_on_lut_changed`, `_on_auto_clim`, `_on_full_clim` |
| [`cellacdc/widgets.py`](../../cellacdc/widgets.py) | LUT item class choice, axis label helper if needed |

### Reference — main GUI primary LUT

```4256:4310:cellacdc/gui.py
        self.imgGrad = widgets.myHistogramLUTitem(parent=self, name='image')
        ...
        self.imgGradRight = widgets.baseHistogramLUTitem(
            name='image', parent=self, gradientPosition='left'
        )
        ...
        self.imgGrad.setChildLutItem(self.imgGradRight)
```

---

## 2. Right-side grayscale opacity colorbar

### Current state

Primary opacity is a form-row [`sliderWithSpinBox`](../../cellacdc/widgets.py) in `VolumeRendererControls` (`Opacity ({channel}):`).

Main GUI uses vertical gradient LUT on the right (`imgGradRight`, `gradientPosition='left'`) linked via `setChildLutItem`.

### Tasks

- [ ] Add `baseHistogramLUTitem` with grayscale gradient to the right of the vispy canvas (same layout slot as LUT row, or dedicated column).
- [ ] Link to primary LUT via `setChildLutItem` if dual-slider behaviour should match 2D.
- [ ] Map gradient tick positions → `volume_node.opacity` (or equivalent vispy alpha control).
- [ ] Deprecate or hide redundant form-row opacity sliders once colorbar works (avoid duplicate controls).

### Files

| File | Changes |
|------|---------|
| [`cellacdc/renderer3d.py`](../../cellacdc/renderer3d.py) | `_init_ui()`, `_add_lut_items()` or new `_add_opacity_lut()`, `VolumeRendererControls` |
| [`cellacdc/widgets.py`](../../cellacdc/widgets.py) | Reuse `baseHistogramLUTitem` pattern from main GUI |

---

## 3. In-renderer overlay LUT sliders

### Current state

Overlays receive hardcoded cmap strings from [`_get_overlay_zstacks()`](../../cellacdc/gui.py):

```20191:20217:cellacdc/gui.py
        _FLUO_CMAPS = ['green', 'magenta', 'cyan', 'yellow', 'orange']
        ...
                cmap = _FLUO_CMAPS[i % len(_FLUO_CMAPS)]
                result.append((data, opacity, cmap))
```

Main GUI builds per-channel two-colour gradients in [`getOverlayItems()`](../../cellacdc/gui.py) via `initColormapOverlayLayerItem(initColor, lutItem)`.

### Tasks

- [ ] Extend overlay tuple schema to carry channel name + optional PG gradient state (or read from GUI `overlayLayersItems` at sync time).
- [ ] Add overlay LUT widgets in 3D window (one per active overlay fluo channel) — mirror two-colour gradient UI.
- [ ] On LUT change: `pg_to_vispy_cmap(lutItem.gradient.colorMap())` → update `overlay:N` node `cmap`.
- [ ] Fix 2D/3D colour mismatch by sourcing overlay colour from GUI `lutItem` instead of `_FLUO_CMAPS[i]`.

### Files

| File | Changes |
|------|---------|
| [`cellacdc/gui.py`](../../cellacdc/gui.py) | `_get_overlay_zstacks()` — include channel metadata / lut gradient |
| [`cellacdc/renderer3d.py`](../../cellacdc/renderer3d.py) | Overlay LUT layout, `update_overlay_volumes()`, node cmap updates |
| [`cellacdc/colors.py`](../../cellacdc/colors.py) | Already has `pg_to_vispy_cmap`; may need overlay-specific helpers |

---

## 4. In-renderer overlay opacity sliders

### Current state

Opacity values are read once from main GUI alpha scrollbars when `_get_overlay_zstacks()` runs. No controls inside the 3D window for overlays.

### Tasks

- [ ] Add per-overlay opacity control in 3D UI (form slider or mini colorbar per overlay channel).
- [ ] On change: update `volume_node.opacity` directly for matching `overlay:N` key.
- [ ] Optionally bidirectional sync with main GUI alpha scrollbars (nice-to-have; document one-way vs two-way).

### Segmentation mask opacity

Main GUI: [`labelsAlphaSlider`](../../cellacdc/widgets.py) on `myHistogramLUTitem` / `imgGrad`.

- [ ] Add dedicated "Segmentation opacity" control in 3D controls panel.
- [ ] When changed, rebuild or update segm overlay node opacity without full frame reload if possible.

---

## 5. Live sync

### Current state

[`_update_3d_renderer_if_active()`](../../cellacdc/gui.py) pushes fresh data on frame/position changes only. Overlay slider edits in the main GUI do not refresh the 3D window until navigation.

### Tasks

- [ ] Connect main GUI overlay alpha / labels alpha changes to `_update_3d_renderer_if_active()` or a lighter `update_overlay_volumes()` call.
- [ ] In-renderer overlay control changes should update nodes in place (preferred) rather than full `_get_overlay_zstacks()` rebuild when only opacity/cmap changed.
- [ ] Add `VolumeRenderer3DWindow.set_overlay_opacity(index, value)` and `set_overlay_cmap(index, cmap)` helpers for incremental updates.

### Files

| File | Changes |
|------|---------|
| [`cellacdc/gui.py`](../../cellacdc/gui.py) | Signal connections from overlay widgets; optional sync callbacks |
| [`cellacdc/renderer3d.py`](../../cellacdc/renderer3d.py) | Incremental overlay node update API |

---

## Suggested implementation order

1. Fix overlay colour source in `_get_overlay_zstacks()` (quick win for 2D/3D match).
2. Primary LUT labelling + optional `myHistogramLUTitem` upgrade.
3. Right-side primary opacity colorbar.
4. Overlay opacity sliders in 3D UI.
5. Overlay LUT sliders in 3D UI.
6. Live sync wiring (main GUI ↔ 3D renderer).

## Acceptance criteria

- Primary LUT looks and behaves like main GUI Clim slider (label, gradient, Auto/Full, cmap).
- Primary opacity adjustable via right-side grayscale colorbar.
- Each active overlay fluo channel has LUT + opacity in the 3D window.
- Segmentation mask opacity adjustable without leaving the 3D window.
- Changing overlay opacity or LUT in either GUI updates the 3D view immediately.

## Out of scope (later batches)

- Cell ID isolation → [batch-3-cell-id.md](batch-3-cell-id.md)
- z-anisotropy UI → [batch-4-z-anisotropy.md](batch-4-z-anisotropy.md)
