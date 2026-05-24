"""View adapter for draw-clear-region workflows."""

from __future__ import annotations

from cellacdc.viewmodels.draw_clear_region_viewmodel import (
    DrawClearRegionViewModel,
)


class DrawClearRegionView:
    """Qt-facing adapter around the scriptable draw-clear view-model."""

    def __init__(self, host, view_model: DrawClearRegionViewModel):
        self.host = host
        self.view_model = view_model

    def toggle(self, checked):
        pos_data = self.host.data[self.host.pos_i]
        if checked:
            self.host.disconnectLeftClickButtons()
            self.host.uncheckLeftClickButtons(self.host.drawClearRegionButton)
            self.host.connectLeftClickButtons()

        self.host.drawClearRegionToolbar.setVisible(checked)
        state = self.view_model.toolbar_state(
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
                self.view_model.empty_selection_warning(
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
            if self.view_model.is_single_z_projection(z_projection):
                single_z_range = self.host.drawClearRegionToolbar.zRange(
                    self.host.z_lab(), size_z
                )
        return self.view_model.z_range_for_projection(
            is_segm_3d=self.host.isSegm3D,
            z_projection=z_projection,
            size_z=size_z,
            single_z_range=single_z_range,
        )
