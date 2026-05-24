"""View adapter for draw-clear-region workflows."""

from __future__ import annotations


class DrawClearRegionMixin:
    """Qt-facing adapter around the scriptable draw-clear view-model."""

    """Headless draw-clear-region decision rules."""

    single_z_slice_projection = "single z-slice"

    def _z_range(self, size_z):
        z_projection = None
        single_z_range = None
        if self.isSegm3D:
            z_projection = self.zProjComboBox.currentText()
            if self.is_single_z_projection(z_projection):
                single_z_range = self.drawClearRegionToolbar.zRange(
                    self.z_lab(), size_z
                )
        return self.z_range_for_projection(
            is_segm_3d=self.isSegm3D,
            z_projection=z_projection,
            size_z=size_z,
            single_z_range=single_z_range,
        )

    def clear_objects_in_freehand_region(self):
        self.logger.info("Clearing objects inside freehand region...")
        self.storeUndoRedoStates(False, storeImage=False, storeOnlyZoom=True)

        pos_data = self.data[self.pos_i]
        z_range = self._z_range(pos_data.SizeZ)
        region_slice = self.freeRoiItem.slice(zRange=z_range)
        mask = self.freeRoiItem.mask()
        region_lab = pos_data.lab[(...,) + region_slice].copy()

        enclosed_only = (
            self.drawClearRegionToolbar.clearOnlyEnclosedObjsRadioButton.isChecked()
        )
        selection_result = self.view_model.label_edits.select_labels_in_region(
            region_lab,
            mask,
            enclosed_only=enclosed_only,
        )
        clear_ids = selection_result.selected_ids

        if not clear_ids:
            self.logger.warning(
                self.empty_selection_warning(enclosed_only=enclosed_only)
            )
            return

        self.deleteIDmiddleClick(clear_ids, False, False)
        self.update_cca_df_deletedIDs(pos_data, clear_ids)
        self.freeRoiItem.clear()
        self.updateAllImages()

    def empty_selection_warning(self, *, enclosed_only: bool) -> str:
        if enclosed_only:
            return "None of the objects in the freehand region are fully enclosed"
        return "None of the objects are touching the freehand region"

    def is_single_z_projection(self, z_projection: str) -> bool:
        return z_projection == self.single_z_slice_projection

    def toggle(self, checked):
        pos_data = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.drawClearRegionButton)
            self.connectLeftClickButtons()

        self.drawClearRegionToolbar.setVisible(checked)
        state = self.toolbar_state(
            checked=checked,
            is_segm_3d=self.isSegm3D,
            size_z=pos_data.SizeZ,
        )
        if not state.update_z_control:
            return
        if state.z_control_enabled:
            self.drawClearRegionToolbar.setZslicesControlEnabled(
                True, SizeZ=state.size_z
            )
            return
        self.drawClearRegionToolbar.setZslicesControlEnabled(False)

    def toolbar_state(
        self,
        *,
        checked: bool,
        is_segm_3d: bool,
        size_z: int,
    ) -> DrawClearRegionToolbarState:
        if not is_segm_3d:
            return DrawClearRegionToolbarState(update_z_control=True)
        if not checked:
            return DrawClearRegionToolbarState(update_z_control=False)
        return DrawClearRegionToolbarState(
            update_z_control=True,
            z_control_enabled=True,
            size_z=size_z,
        )

    def z_range_for_projection(
        self,
        *,
        is_segm_3d: bool,
        z_projection: str,
        size_z: int,
        single_z_range,
    ):
        if not is_segm_3d:
            return None
        if z_projection == self.single_z_slice_projection:
            return single_z_range
        return (0, size_z)
