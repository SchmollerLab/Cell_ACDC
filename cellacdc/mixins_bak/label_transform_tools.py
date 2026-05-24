"""View adapter for label transform tools."""

from __future__ import annotations

import skimage.measure


class LabelTransformToolsMixin:
    """Qt-facing adapter around label transform tool contracts."""

    """Headless decision rules for label transform tools."""

    def _set_temp_img_expand_label_contours(self, previous_coords, ax=0):
        self.contoursImage[previous_coords] = [0, 0, 0, 0]
        current_lab_2d_rp = skimage.measure.regionprops(self.currentLab2D)
        for obj in current_lab_2d_rp:
            if obj.label == self.expandingID:
                self.addObjContourToContoursImage(obj=obj, ax=ax, force=True)
                break

    def _set_temp_img_expand_label_segm_masks(self, previous_coords, ax=0):
        labels_image = self.getLabelsLayerImage(ax=ax)
        labels_image[previous_coords] = 0
        labels_image[previous_coords] = self.expandingID

        if ax == 0:
            self.labelsLayerImg1.setImage(self.labelsLayerImg1.image, autoLevels=False)
        else:
            self.labelsLayerRightImg.setImage(
                self.labelsLayerRightImg.image, autoLevels=False
            )

    def expand_label(self, dilation=True):
        pos_data = self.data[self.pos_i]
        if self.hoverLabelID == 0:
            self.isExpandingLabel = False
            return

        reinit_expanding_lab = self.should_reinitialize_expansion(
            expanding_id=self.expandingID,
            hover_label_id=self.hoverLabelID,
            dilation=dilation,
            is_dilation=self.isDilation,
        )
        label_id = self.hoverLabelID
        obj = pos_data.rp[pos_data.IDs.index(label_id)]

        if reinit_expanding_lab:
            self.storeUndoRedoStates(False)
            self.isExpandingLabel = True
            self.expandingID = label_id
            self.expandingLab = None
            self.expandFootprintSize = 1

        lab_2d = self.get_2Dlab(pos_data.lab)
        resize_result = self.view_model.label_edits.resize_label_object(
            lab_2d,
            self.currentLab2D,
            obj.coords,
            self.expandingID,
            self.expandFootprintSize,
            dilation=dilation,
            seed_labels=self.expandingLab,
        )
        self.expandingLab = resize_result.seed_labels
        self.isDilation = dilation
        previous_coords = resize_result.previous_coords
        expanded_obj_coords = resize_result.resized_coords

        self.set_2Dlab(lab_2d)
        self.currentLab2D = lab_2d
        self.update_rp()

        if self.labelsGrad.showLabelsImgAction.isChecked():
            self.img2.setImage(img=self.currentLab2D, autoLevels=False)

        self.set_temp_img_expand_label(previous_coords, expanded_obj_coords)

    def expand_label_callback(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.expandLabelToolButton)
            self.connectLeftClickButtons()
            self.expandFootprintSize = 1
            return

        self.clearHighlightedID()
        alpha = self.imgGrad.labelsAlphaSlider.value()
        self.labelsLayerImg1.setOpacity(alpha)
        self.labelsLayerRightImg.setOpacity(alpha)
        self.hoverLabelID = 0
        self.expandingID = 0
        self.updateAllImages()

    def move_delta(self, *, previous_pos, current_pos) -> tuple[int, int]:
        x_start, y_start = previous_pos
        x_current, y_current = current_pos
        return x_current - x_start, y_current - y_start

    def move_label(self, x_pos, y_pos):
        pos_data = self.data[self.pos_i]
        lab_2d = self.get_2Dlab(pos_data.lab)
        y_size, x_size = lab_2d.shape
        x_data, y_data = int(x_pos), int(y_pos)
        if not self.point_in_shape(
            x=x_data,
            y=y_data,
            shape=(y_size, x_size),
        ):
            return

        self.clearObjContour(ID=self.movingID, ax=0)
        delta_x, delta_y = self.move_delta(
            previous_pos=self.prevMovePos,
            current_pos=(x_data, y_data),
        )
        move_result = self.view_model.label_edits.move_label_object(
            pos_data.lab,
            self.movingObjCoords,
            self.movingID,
            delta_y=delta_y,
            delta_x=delta_x,
            shape=(y_size, x_size),
        )
        self.movingObjCoords = move_result.moved_coords
        self.currentLab2D = self.get_2Dlab(pos_data.lab)
        if self.labelsGrad.showLabelsImgAction.isChecked():
            self.img2.setImage(self.currentLab2D, autoLevels=False)

        self.set_temp_img1_move_label()
        self.prevMovePos = (x_data, y_data)

    def move_label_button_toggled(self, checked):
        if not self.should_clear_move_state(checked=checked):
            return
        self.hoverLabelID = 0
        self.highlightedID = 0
        self.highLightIDLayerImg1.clear()
        self.highLightIDLayerRightImage.clear()
        self.setHighlightID(False)

    def point_in_shape(self, *, x: int, y: int, shape) -> bool:
        y_size, x_size = shape
        return x >= 0 and y >= 0 and x < x_size and y < y_size

    def reset_expand_label(self):
        self.expandingID = self.reset_expand_label_id()

    def reset_expand_label_id(self) -> int:
        return -1

    def set_temp_img1_move_label(self, ax=0):
        if ax == 0:
            how = self.drawIDsContComboBox.currentText()
        else:
            how = self.getAnnotateHowRightImage()

        if how.find("contours") != -1:
            current_lab_2d_rp = skimage.measure.regionprops(self.currentLab2D)
            for obj in current_lab_2d_rp:
                if obj.label == self.movingID:
                    self.addObjContourToContoursImage(obj=obj, ax=ax)
                    break
        elif how.find("overlay segm. masks") != -1:
            if ax == 0:
                self.labelsLayerImg1.setImage(self.currentLab2D, autoLevels=False)
                self.highLightIDLayerImg1.image[:] = 0
                mask = self.currentLab2D == self.movingID
                self.highLightIDLayerImg1.image[mask] = self.movingID
                highlighted_image = self.highLightIDLayerImg1.image
                self.highLightIDLayerImg1.setImage(highlighted_image)
            else:
                self.labelsLayerRightImg.setImage(self.currentLab2D, autoLevels=False)
                self.highLightIDLayerRightImage.image[:] = 0
                mask = self.currentLab2D == self.movingID
                self.highLightIDLayerRightImage.image[mask] = self.movingID
                highlighted_image = self.highLightIDLayerRightImage.image
                self.highLightIDLayerRightImage.setImage(highlighted_image)

    def set_temp_img_expand_label(
        self,
        previous_coords,
        expanded_obj_coords,
        ax=0,
    ):
        if ax == 0:
            self.drawIDsContComboBox.currentText()
        else:
            self.getAnnotateHowRightImage()

        self._set_temp_img_expand_label_contours(previous_coords, ax=ax)

    def should_clear_move_state(self, *, checked: bool) -> bool:
        return not checked

    def should_reinitialize_expansion(
        self,
        *,
        expanding_id: int,
        hover_label_id: int,
        dilation: bool,
        is_dilation: bool,
    ) -> bool:
        return expanding_id != hover_label_id or dilation != is_dilation

    def should_start_moving_label(self, label_id: int) -> bool:
        return label_id != 0

    def start_moving_label(self, x_pos, y_pos):
        pos_data = self.data[self.pos_i]
        x_data, y_data = int(x_pos), int(y_pos)
        lab_2d = self.get_2Dlab(pos_data.lab)
        label_id = lab_2d[y_data, x_data]
        if not self.should_start_moving_label(label_id):
            self.isMovingLabel = False
            return

        self.isMovingLabel = True
        self.searchedIDitemRight.setData([], [])
        self.searchedIDitemLeft.setData([], [])
        self.movingID = label_id
        self.prevMovePos = (x_data, y_data)
        moving_obj = pos_data.rp[pos_data.IDs.index(label_id)]
        self.movingObjCoords = moving_obj.coords.copy()
        yy, xx = moving_obj.coords[:, -2], moving_obj.coords[:, -1]
        self.currentLab2D[yy, xx] = 0
