"""View adapter for draw-clear-region workflows."""

from __future__ import annotations



from dataclasses import dataclass
class DrawClearRegionView:
    """Qt-facing adapter around the scriptable draw-clear view-model."""

    """Headless draw-clear-region decision rules."""

    single_z_slice_projection = 'single z-slice'

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

    def is_single_z_projection(self, z_projection: str) -> bool:
        return z_projection == self.single_z_slice_projection

    def empty_selection_warning(self, *, enclosed_only: bool) -> str:
        if enclosed_only:
            return (
                'None of the objects in the freehand region are fully enclosed'
            )
        return 'None of the objects are touching the freehand region'


    def __init__(self, host):
        self.host = host
    def toggle(self, checked):
        pos_data = self.host.data[self.host.pos_i]
        if checked:
            self.host.disconnectLeftClickButtons()
            self.host.uncheckLeftClickButtons(self.host.drawClearRegionButton)
            self.host.connectLeftClickButtons()

        self.host.drawClearRegionToolbar.setVisible(checked)
        state = self.toolbar_state(
            checked=checked,
            is_segm_3d=self.host.isSegm3D,
            size_z=pos_data.SizeZ,
        )
        if not state.update_z_control:
            return
        if state.z_control_enabled:
            self.host.drawClearRegionToolbar.setZslicesControlEnabled(
                True, SizeZ=state.size_z
            )
            return
        self.host.drawClearRegionToolbar.setZslicesControlEnabled(False)

    def clear_objects_in_freehand_region(self):
        self.host.logger.info('Clearing objects inside freehand region...')
        self.host.storeUndoRedoStates(
            False, storeImage=False, storeOnlyZoom=True
        )

        pos_data = self.host.data[self.host.pos_i]
        z_range = self._z_range(pos_data.SizeZ)
        region_slice = self.host.freeRoiItem.slice(zRange=z_range)
        mask = self.host.freeRoiItem.mask()
        region_lab = pos_data.lab[(...,) + region_slice].copy()

        enclosed_only = (
            self.host.drawClearRegionToolbar
            .clearOnlyEnclosedObjsRadioButton.isChecked()
        )
        selection_result = (
            self.host.view_model.label_edits.select_labels_in_region(
                region_lab,
                mask,
                enclosed_only=enclosed_only,
            )
        )
        clear_ids = selection_result.selected_ids

        if not clear_ids:
            self.host.logger.warning(
                self.empty_selection_warning(
                    enclosed_only=enclosed_only
                )
            )
            return

        self.host.deleteIDmiddleClick(clear_ids, False, False)
        self.host.update_cca_df_deletedIDs(pos_data, clear_ids)
        self.host.freeRoiItem.clear()
        self.host.updateAllImages()

    def _z_range(self, size_z):
        z_projection = None
        single_z_range = None
        if self.host.isSegm3D:
            z_projection = self.host.zProjComboBox.currentText()
            if self.is_single_z_projection(z_projection):
                single_z_range = self.host.drawClearRegionToolbar.zRange(
                    self.host.z_lab(), size_z
                )
        return self.z_range_for_projection(
            is_segm_3d=self.host.isSegm3D,
            z_projection=z_projection,
            size_z=size_z,
            single_z_range=single_z_range,
        )