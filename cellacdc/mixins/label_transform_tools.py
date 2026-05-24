"""View adapter for label transform tools."""

from __future__ import annotations

import skimage.measure



class LabelTransformToolsView:
    """Qt-facing adapter around label transform tool contracts."""

    """Headless decision rules for label transform tools."""

    def reset_expand_label_id(self) -> int:
        return -1

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

    def point_in_shape(self, *, x: int, y: int, shape) -> bool:
        y_size, x_size = shape
        return x >= 0 and y >= 0 and x < x_size and y < y_size

    def move_delta(self, *, previous_pos, current_pos) -> tuple[int, int]:
        x_start, y_start = previous_pos
        x_current, y_current = current_pos
        return x_current - x_start, y_current - y_start

    def should_clear_move_state(self, *, checked: bool) -> bool:
        return not checked


    def __init__(self, host):
        self.host = host
    def reset_expand_label(self):
        self.host.expandingID = self.reset_expand_label_id()

    def expand_label_callback(self, checked):
        if checked:
            self.host.disconnectLeftClickButtons()
            self.host.uncheckLeftClickButtons(self.host.expandLabelToolButton)
            self.host.connectLeftClickButtons()
            self.host.expandFootprintSize = 1
            return

        self.host.clearHighlightedID()
        alpha = self.host.imgGrad.labelsAlphaSlider.value()
        self.host.labelsLayerImg1.setOpacity(alpha)
        self.host.labelsLayerRightImg.setOpacity(alpha)
        self.host.hoverLabelID = 0
        self.host.expandingID = 0
        self.host.updateAllImages()

    def expand_label(self, dilation=True):
        pos_data = self.host.data[self.host.pos_i]
        if self.host.hoverLabelID == 0:
            self.host.isExpandingLabel = False
            return

        reinit_expanding_lab = (
            self.should_reinitialize_expansion(
                expanding_id=self.host.expandingID,
                hover_label_id=self.host.hoverLabelID,
                dilation=dilation,
                is_dilation=self.host.isDilation,
            )
        )
        label_id = self.host.hoverLabelID
        obj = pos_data.rp[pos_data.IDs.index(label_id)]

        if reinit_expanding_lab:
            self.host.storeUndoRedoStates(False)
            self.host.isExpandingLabel = True
            self.host.expandingID = label_id
            self.host.expandingLab = None
            self.host.expandFootprintSize = 1

        lab_2d = self.host.get_2Dlab(pos_data.lab)
        resize_result = self.host.view_model.label_edits.resize_label_object(
            lab_2d,
            self.host.currentLab2D,
            obj.coords,
            self.host.expandingID,
            self.host.expandFootprintSize,
            dilation=dilation,
            seed_labels=self.host.expandingLab,
        )
        self.host.expandingLab = resize_result.seed_labels
        self.host.isDilation = dilation
        previous_coords = resize_result.previous_coords
        expanded_obj_coords = resize_result.resized_coords

        self.host.set_2Dlab(lab_2d)
        self.host.currentLab2D = lab_2d
        self.host.update_rp()

        if self.host.labelsGrad.showLabelsImgAction.isChecked():
            self.host.img2.setImage(img=self.host.currentLab2D, autoLevels=False)

        self.set_temp_img_expand_label(previous_coords, expanded_obj_coords)

    def start_moving_label(self, x_pos, y_pos):
        pos_data = self.host.data[self.host.pos_i]
        x_data, y_data = int(x_pos), int(y_pos)
        lab_2d = self.host.get_2Dlab(pos_data.lab)
        label_id = lab_2d[y_data, x_data]
        if not self.should_start_moving_label(label_id):
            self.host.isMovingLabel = False
            return

        self.host.isMovingLabel = True
        self.host.searchedIDitemRight.setData([], [])
        self.host.searchedIDitemLeft.setData([], [])
        self.host.movingID = label_id
        self.host.prevMovePos = (x_data, y_data)
        moving_obj = pos_data.rp[pos_data.IDs.index(label_id)]
        self.host.movingObjCoords = moving_obj.coords.copy()
        yy, xx = moving_obj.coords[:, -2], moving_obj.coords[:, -1]
        self.host.currentLab2D[yy, xx] = 0

    def move_label(self, x_pos, y_pos):
        pos_data = self.host.data[self.host.pos_i]
        lab_2d = self.host.get_2Dlab(pos_data.lab)
        y_size, x_size = lab_2d.shape
        x_data, y_data = int(x_pos), int(y_pos)
        if not self.point_in_shape(
            x=x_data,
            y=y_data,
            shape=(y_size, x_size),
        ):
            return

        self.host.clearObjContour(ID=self.host.movingID, ax=0)
        delta_x, delta_y = self.move_delta(
            previous_pos=self.host.prevMovePos,
            current_pos=(x_data, y_data),
        )
        move_result = self.host.view_model.label_edits.move_label_object(
            pos_data.lab,
            self.host.movingObjCoords,
            self.host.movingID,
            delta_y=delta_y,
            delta_x=delta_x,
            shape=(y_size, x_size),
        )
        self.host.movingObjCoords = move_result.moved_coords
        self.host.currentLab2D = self.host.get_2Dlab(pos_data.lab)
        if self.host.labelsGrad.showLabelsImgAction.isChecked():
            self.host.img2.setImage(self.host.currentLab2D, autoLevels=False)

        self.set_temp_img1_move_label()
        self.host.prevMovePos = (x_data, y_data)

    def move_label_button_toggled(self, checked):
        if not self.should_clear_move_state(checked=checked):
            return
        self.host.hoverLabelID = 0
        self.host.highlightedID = 0
        self.host.highLightIDLayerImg1.clear()
        self.host.highLightIDLayerRightImage.clear()
        self.host.setHighlightID(False)

    def _set_temp_img_expand_label_segm_masks(self, previous_coords, ax=0):
        labels_image = self.host.getLabelsLayerImage(ax=ax)
        labels_image[previous_coords] = 0
        labels_image[previous_coords] = self.host.expandingID

        if ax == 0:
            self.host.labelsLayerImg1.setImage(
                self.host.labelsLayerImg1.image, autoLevels=False
            )
        else:
            self.host.labelsLayerRightImg.setImage(
                self.host.labelsLayerRightImg.image, autoLevels=False
            )

    def _set_temp_img_expand_label_contours(self, previous_coords, ax=0):
        self.host.contoursImage[previous_coords] = [0, 0, 0, 0]
        current_lab_2d_rp = skimage.measure.regionprops(self.host.currentLab2D)
        for obj in current_lab_2d_rp:
            if obj.label == self.host.expandingID:
                self.host.addObjContourToContoursImage(
                    obj=obj, ax=ax, force=True
                )
                break

    def set_temp_img_expand_label(
        self,
        previous_coords,
        expanded_obj_coords,
        ax=0,
    ):
        if ax == 0:
            how = self.host.drawIDsContComboBox.currentText()
        else:
            how = self.host.getAnnotateHowRightImage()

        self._set_temp_img_expand_label_contours(previous_coords, ax=ax)

    def set_temp_img1_move_label(self, ax=0):
        if ax == 0:
            how = self.host.drawIDsContComboBox.currentText()
        else:
            how = self.host.getAnnotateHowRightImage()

        if how.find('contours') != -1:
            current_lab_2d_rp = skimage.measure.regionprops(
                self.host.currentLab2D
            )
            for obj in current_lab_2d_rp:
                if obj.label == self.host.movingID:
                    self.host.addObjContourToContoursImage(obj=obj, ax=ax)
                    break
        elif how.find('overlay segm. masks') != -1:
            if ax == 0:
                self.host.labelsLayerImg1.setImage(
                    self.host.currentLab2D, autoLevels=False
                )
                self.host.highLightIDLayerImg1.image[:] = 0
                mask = self.host.currentLab2D == self.host.movingID
                self.host.highLightIDLayerImg1.image[mask] = self.host.movingID
                highlighted_image = self.host.highLightIDLayerImg1.image
                self.host.highLightIDLayerImg1.setImage(highlighted_image)
            else:
                self.host.labelsLayerRightImg.setImage(
                    self.host.currentLab2D, autoLevels=False
                )
                self.host.highLightIDLayerRightImage.image[:] = 0
                mask = self.host.currentLab2D == self.host.movingID
                self.host.highLightIDLayerRightImage.image[mask] = (
                    self.host.movingID
                )
                highlighted_image = self.host.highLightIDLayerRightImage.image
                self.host.highLightIDLayerRightImage.setImage(highlighted_image)