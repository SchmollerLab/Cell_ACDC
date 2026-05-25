# Batch 3 — Cell ID Isolation

Allow the 3D renderer to show a single labelled cell in isolation — either by selecting a cell ID from a control or by clicking a cell in the volume view.

## Checklist targets

| # | Feature | Status |
|---|---------|--------|
| 15 | Cell ID selector (show one cell) | **Not started** |
| 16 | Clickable Cell ID (show one cell) | **Not started** |

## Goals

1. Add a cell ID selector control in the 3D window (dropdown, spin box, or list synced with main GUI label set).
2. When a cell ID is selected, mask the primary and/or overlay segmentation volumes so only that label is visible.
3. Support click-to-pick: raycast or slice-based picking on the vispy canvas to set the active cell ID.
4. Provide a clear "show all cells" reset (e.g. ID 0 or dedicated button).

---

## 1. Cell ID selector

### Current state

Segmentation overlays arrive as binary `(lab > 0)` masks via [`_get_overlay_zstacks()`](../../cellacdc/gui.py). The renderer has no per-label filtering; all labelled pixels render together.

### Tasks

- [ ] Read label volume (not just binary mask) from adapter or a new `get_label_zstack()` hook.
- [ ] Add UI control for active cell ID (0 = all cells).
- [ ] Apply label mask in `update_volume()` / overlay node setup: `data * (labels == cell_id)` or equivalent GPU-friendly masking.
- [ ] Sync selector with main GUI cell selection if one exists (document one-way vs two-way).

### Files

| File | Changes |
|------|---------|
| [`cellacdc/gui.py`](../../cellacdc/gui.py) | Adapter hook for raw label data; optional sync with 2D cell selection |
| [`cellacdc/renderer3d.py`](../../cellacdc/renderer3d.py) | Cell ID control widget, masking logic on volume nodes |

---

## 2. Clickable cell ID (pick in 3D view)

### Current state

No picking or cell-ID interaction on the vispy canvas.

### Tasks

- [ ] Handle mouse click on canvas → map screen coords to volume index (vispy scene picking or manual unproject).
- [ ] Read label value at picked voxel; set active cell ID and refresh masked volumes.
- [ ] Visual feedback (cursor, status bar, or highlight) for picked ID.
- [ ] Ignore picks on background (label 0).

### Files

| File | Changes |
|------|---------|
| [`cellacdc/renderer3d.py`](../../cellacdc/renderer3d.py) | Canvas event handler, pick → cell ID, refresh |

---

## Dependencies

- Batch 1 overlay data path (complete) — segmentation volumes must remain available as labelled data, not only binary masks.
- Batch 2 overlay UI (optional) — in-renderer segm opacity may interact with per-cell masking; coordinate designs before implementing.

## Out of scope

- z-anisotropy controls → [batch-4-z-anisotropy.md](batch-4-z-anisotropy.md)
- Tracking / lineage-aware cell selection across time (future enhancement)

## Acceptance criteria

- User can enter or select a cell ID and see only that cell in the 3D view.
- User can click a cell in the 3D canvas to isolate it.
- Resetting to "all cells" restores the full segmentation view.
- Frame navigation preserves the selected cell ID until explicitly cleared.
