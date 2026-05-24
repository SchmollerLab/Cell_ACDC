"""Qt view adapter for brush and eraser tools."""

from __future__ import annotations

import cv2
import numpy as np
import skimage.measure
from qtpy.QtWidgets import QCheckBox

from cellacdc import html_utils, settings_csv_path, widgets
from cellacdc.viewmodels.brush_tools_viewmodel import BrushToolsViewModel


class BrushToolsView:
    """Qt-facing adapter around brush and eraser tool workflows."""

    LEGACY_METHODS = (
        'instructHowDeleteID',
        'checkWarnDeletedIDwithEraser',
        'brushAutoFillToggled',
        'brushAutoHideToggled',
        'fillHolesID',
        'brushReleased',
        'brushSize_cb',
        'autoIDtoggled',
        'Brush_cb',
        'showEditIDwidgets',
        'resetCursors',
        'updateEraserCursor',
        'setDiskMask',
        'getDiskMask',
        'applyEraserMask',
        'changeBrushID',
        'applyBrushMask',
        'setBrushID',
        'initFloodMaskImage',
        'getMagicWandFloodTolerance',
        'initTempLayerBrush',
        '_setTempImageBrushContour',
        'setTempBrushMaskFromWand',
        'setTempImg1Brush',
        'getLabelsLayerImage',
        'clearObjFromMask',
        'setTempImg1Eraser',
        'Eraser_cb',
    )

    def __init__(self, host, view_model: BrushToolsViewModel):
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'view_model', view_model)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host', 'view_model'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def bind_legacy_methods(self):
        for name in self.LEGACY_METHODS:
            setattr(self.host, name, getattr(self, name))

    def instructHowDeleteID(self):
        if 'showInfoDeleteObject' not in self.df_settings.index:
            self.df_settings.at['showInfoDeleteObject', 'value'] = (
                self.view_model.default_delete_object_info_value()
            )

        showInfoDeleteObject = self.view_model.should_show_delete_object_info(
            self.df_settings.at['showInfoDeleteObject', 'value']
        )
        if not showInfoDeleteObject:
            return

        actionText = self.middleClickText()
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'You have deleted an object using the eraser tool.<br><br>'
            'Did you know that you can use the "Delete object" action<br>'
            'to <b>delete an object with a single click</b>?<br><br>'
            f'To do so, use the following action: <code>{actionText}</code><br><br>'
            'Note: You can also set a custom shortcut by going to the menu<br>'
            '<code>Settings --&gt; Customise keyboard shortcuts...</code>.'
        )
        doNotShowAgainCheckbox = QCheckBox('Do not show again')
        msg.information(
            self.host, 'Delete objects with single click', txt,
            widgets=doNotShowAgainCheckbox
        )

        showInfoDeleteObjectValue = self.view_model.delete_object_info_value(
            doNotShowAgainCheckbox.isChecked()
        )
        self.df_settings.at['showInfoDeleteObject', 'value'] = (
            showInfoDeleteObjectValue
        )
        self.df_settings.to_csv(settings_csv_path)

    def checkWarnDeletedIDwithEraser(self):
        posData = self.data[self.pos_i]

        for ID in self.erasedIDs:
            if ID == 0:
                continue
            if ID in posData.IDs_idxs:
                continue

            self.instructHowDeleteID()

            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Delete ID with eraser')
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df('Delete ID with eraser')

            return True

        return False

    def brushAutoFillToggled(self, checked):
        val = self.view_model.checked_setting_value(checked)
        self.df_settings.at['brushAutoFill', 'value'] = val
        self.df_settings.to_csv(self.settings_csv_path)

    def brushAutoHideToggled(self, checked):
        val = self.view_model.checked_setting_value(checked)
        self.df_settings.at['brushAutoHide', 'value'] = val
        self.df_settings.to_csv(self.settings_csv_path)

    # @exec_time
    def fillHolesID(self, ID, sender='brush'):
        posData = self.data[self.pos_i]
        if sender == 'brush':
            if not self.view_model.should_fill_holes(
                sender,
                auto_fill_checked=self.brushAutoFillCheckbox.isChecked(),
            ):
                return False

            lab2D = self.get_2Dlab(posData.lab)
            result = self.host.view_model.label_edits.fill_label_holes(
                lab2D,
                ID,
            )
            self.set_2Dlab(result.labels)
            return True
        return False

    def brushReleased(self):
        posData = self.data[self.pos_i]
        self.fillHolesID(posData.brushID, sender='brush')

        # Update data (rp, etc)
        self.update_rp(update_IDs=self.isNewID,)

        # Repeat tracking
        if self.autoIDcheckbox.isChecked():
            self.trackManuallyAddedObject(posData.brushID, self.isNewID)
        else:
            self.update_rp(update_IDs=posData.brushID not in posData.IDs_idxs)

        # Update images
        if self.isNewID:
            editTxt = 'Add new ID with brush tool'
            if self.isSnapshot:
                self.fixCcaDfAfterEdit(editTxt)
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df(editTxt)
        else:
            self.updateAllImages()

        self.isNewID = False

    def brushSize_cb(self, value):
        self.ax2_EraserCircle.setSize(value*2)
        self.ax1_BrushCircle.setSize(value*2)
        self.ax2_BrushCircle.setSize(value*2)
        self.ax1_EraserCircle.setSize(value*2)
        self.ax2_EraserX.setSize(value)
        self.ax1_EraserX.setSize(value)
        self.setDiskMask()

    def autoIDtoggled(self, checked):
        self.editIDspinboxAction.setDisabled(checked)
        self.editIDLabelAction.setDisabled(checked)
        if not checked and self.editIDspinbox.value() == 0:
            newID = self.setBrushID(return_val=True)
            self.editIDspinbox.setValue(newID)

    def Brush_cb(self, checked):
        if checked:
            self.typingEditID = False
            self.setDiskMask()
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )
            self.updateBrushCursor(self.xHoverImg, self.yHoverImg)
            self.setBrushID()

            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            c = self.defaultToolBarButtonColor
            self.eraserButton.setStyleSheet(f'background-color: {c}')
            self.connectLeftClickButtons()
            self.image_controls_view.setFocusGraphics()
        else:
            self.ax1_lostObjScatterItem.setVisible(True)
            self.ax2_lostObjScatterItem.setVisible(True)
            self.ax1_lostTrackedScatterItem.setVisible(True)
            self.ax2_lostTrackedScatterItem.setVisible(True)

            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
            self.resetCursors()

        self.showEditIDwidgets(checked)
        self.mode_controls_view.enableSizeSpinbox(checked)

    def showEditIDwidgets(self, visible):
        self.editIDLabelAction.setVisible(visible)
        self.editIDspinboxAction.setVisible(visible)
        self.autoIDcheckboxAction.setVisible(visible)
        showToolbar = (
            self.view_model.brush_toolbar_visible(
                visible,
                brush_size_visible=self.brushSizeAction.isVisible(),
                auto_fill_visible=self.brushAutoFillAction.isVisible(),
                auto_hide_visible=self.brushAutoHideAction.isVisible(),
            )
        )
        self.brushEraserToolBar.setVisible(showToolbar)

    def resetCursors(self):
        self.ax1_cursor.setData([], [])
        self.ax2_cursor.setData([], [])
        while self.app.overrideCursor() is not None:
            self.app.restoreOverrideCursor()

    def updateEraserCursor(self, x, y, xyLocked=None, isHoverImg1=True):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.currentLab2D
        Y, X = _img.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        size = self.brushSizeSpinbox.value()*2
        self.setHoverToolSymbolData(
            [x], [y], self.activeEraserCircleCursors(isHoverImg1),
            size=size
        )
        self.setHoverToolSymbolData(
            [x], [y], self.activeEraserXCursors(isHoverImg1),
            size=int(size/2)
        )

        isMouseDrag = (
            self.isMouseDragImg1 or self.isMouseDragImg2
        )
        if isMouseDrag:
            return

        if xyLocked is not None:
            xdata, ydata = xyLocked

        self.setHoverToolSymbolColor(
            xdata, ydata, self.eraserCirclePen,
            self.activeEraserCircleCursors(isHoverImg1),
            self.eraserButton, hoverRGB=None
        )

    def setDiskMask(self):
        brushSize = self.brushSizeSpinbox.value()
        self.diskMask = self.view_model.disk_mask(brushSize)

    def getDiskMask(self, xdata, ydata):
        brushSize = self.brushSizeSpinbox.value()
        return self.view_model.disk_mask_bounds(
            self.currentLab2D.shape[-2:],
            brushSize,
            xdata,
            ydata,
            self.diskMask,
        )

    # @exec_time
    def applyEraserMask(self, mask):
        posData = self.data[self.pos_i]
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if isZslice:
                posData.lab[self.z_lab(), mask] = 0
            else:
                posData.lab[:, mask] = 0
        else:
            posData.lab[mask] = 0

    def changeBrushID(self):
        """Function called when pressing or releasing shift
        """
        if not self.isSegm3D:
            # Changing brush ID with shift is only for 3D segm
            return

        if not self.brushButton.isChecked():
            # Brush if not active
            return

        if not self.isMouseDragImg2 and not self.isMouseDragImg1:
            # Mouse is not brushing at the moment
            return

        posData = self.data[self.pos_i]
        forceNewObj = not self.isNewID

        if forceNewObj:
            # Shift is down --> force new object with brush
            # e.g., 24 --> 28:
            # 24 is hovering ID that we store as self.prevBrushID
            # 24 object becomes 28 that is the new posData.brushID
            self.isNewID = True
            self.changedID = posData.brushID
            self.restoreBrushID = posData.brushID
            # Set a new ID
            self.setBrushID()
        else:
            # Shift released or hovering on ID in z+-1
            # --> restore brush ID from before shift was pressed or from
            # when we started brushing from outside an object
            # but we hovered on ID in z+-1 while dragging.
            # We change the entire 28 object to 24 so before changing the
            # brush ID back to 24 we builg the mask with 28 to change it to 24
            self.isNewID = False
            self.changedID = posData.brushID
            # Restore ID
            posData.brushID = self.restoreBrushID

        brushID = posData.brushID
        brushIDmask = self.get_2Dlab(posData.lab) == self.changedID
        self.applyBrushMask(brushIDmask, brushID)
        if self.isMouseDragImg1:
            self.brushColor = self.lut[posData.brushID]/255
            self.setTempImg1Brush(True, brushIDmask, posData.brushID)

    def applyBrushMask(self, mask, ID, toLocalSlice=None):
        posData = self.data[self.pos_i]
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if isZslice:
                if toLocalSlice is not None:
                    toLocalSlice = (self.z_lab(), *toLocalSlice)
                    posData.lab[toLocalSlice][mask] = ID
                else:
                    posData.lab[self.z_lab()][mask] = ID
            else:
                if toLocalSlice is not None:
                    for z in range(len(posData.lab)):
                        _slice = (z, *toLocalSlice)
                        posData.lab[_slice][mask] = ID
                else:
                    posData.lab[:, mask] = ID
        else:
            if toLocalSlice is not None:
                posData.lab[toLocalSlice][mask] = ID
            else:
                posData.lab[mask] = ID

    def setBrushID(self, useCurrentLab=True, return_val=False):
        # Make sure that the brushed ID is always a new one based on
        # already visited frames
        posData = self.data[self.pos_i]
        wl_init = posData.whitelist and posData.whitelist.whitelistIDs
        id_groups = []
        if useCurrentLab:
            IDs_tot = set(posData.IDs)
            if wl_init:
                try:
                    IDs_tot.update(posData.whitelist.originalLabsIDs[posData.frame_i])
                except:
                    pass
                try:
                    if posData.whitelist.whitelistIDs[posData.frame_i]:
                        IDs_tot.update(posData.whitelist.whitelistIDs[posData.frame_i])
                except:
                    pass
            id_groups.append(IDs_tot)
        for frame_i, storedData in enumerate(posData.allData_li):
            if frame_i == posData.frame_i:
                continue
            lab = storedData['labels']
            if lab is not None:
                rp = storedData['regionprops']
                IDs_tot = {obj.label for obj in rp}
                if wl_init:
                    if self.whitelistCheckOriginalLabels(warning=False, frame_i=frame_i):
                        IDs_tot.update(posData.whitelist.originalLabsIDs[frame_i])
                    if posData.whitelist.whitelistIDs[frame_i]:
                        IDs_tot.update(posData.whitelist.whitelistIDs[frame_i])
                id_groups.append(IDs_tot)
            else:
                break

        posData.brushID = (
            self.host.view_model.label_edits.next_available_label_id(
                id_groups,
                manual_edit_info=posData.editID_info,
            )
        )
        if return_val:
            return posData.brushID

    def initFloodMaskImage(self):
        posData = self.data[self.pos_i]
        self.flood_img = posData.img_data[posData.frame_i]
        if not self.isSegm3D and posData.SizeZ > 1:
            self.flood_img = self.get_2Dimg_from_3D(self.flood_img)
            return

    def getMagicWandFloodTolerance(self):
        tol_perc = self.wandControlsToolbar.toleranceSpinbox.value()
        posData = self.data[self.pos_i]
        _min, _max = posData.img_data_min_max
        return self.view_model.magic_wand_flood_tolerance(tol_perc, _min, _max)

    def initTempLayerBrush(self, ID, ax=0):
        if ax == 0:
            how = self.drawIDsContComboBox.currentText()
        else:
            how = self.getAnnotateHowRightImage()

        self.hideItemsHoverBrush(ID=ID, force=True)
        Y, X = self.img1.image.shape[:2]
        tempImage = np.zeros((Y, X), dtype=np.uint32)
        if how.find('contours') != -1:
            tempImage[self.currentLab2D==ID] = ID
            self.brushImage = tempImage.copy()
            self.brushContourImage = np.zeros((Y, X, 4), dtype=np.uint8)
            color = self.imgGrad.contoursColorButton.color()
            self.brushContoursRgba = color.getRgb()
            opacity = 1.0
        else:
            opacity = self.imgGrad.labelsAlphaSlider.value()
            color = self.lut[ID]
            lut = np.zeros((2, 4), dtype=np.uint8)
            lut[1,-1] = 255
            lut[1,:-1] = color
            self.tempLayerImg1.setLookupTable(lut)
        self.tempLayerImg1.setOpacity(opacity)
        self.tempLayerImg1.setImage(tempImage, force_set_linked=True)

    def _setTempImageBrushContour(self):
        pass

    def setTempBrushMaskFromWand(self, flood_mask, init=False):
        if not np.any(flood_mask):
            return

        posData = self.data[self.pos_i]
        mask = np.logical_or(
            flood_mask,
            posData.lab==posData.brushID
        )
        if mask.ndim == 3:
            z_slice = self.zSliceScrollBar.sliderPosition()
            mask = mask[z_slice]

        self.setTempImg1Brush(init, mask, posData.brushID)

    # @exec_time
    def setTempImg1Brush(self, init: bool, mask, ID, toLocalSlice=None, ax=0):
        if init:
            self.initTempLayerBrush(ID, ax=ax)

        if self.annotContourCheckbox.isChecked():
            brushImage = self.brushImage
        else:
            brushImage = self.tempLayerImg1.image

        if toLocalSlice is None:
            brushImage[mask] = ID
        else:
            brushImage[toLocalSlice][mask] = ID

        if self.annotContourCheckbox.isChecked():
            try:
                obj = skimage.measure.regionprops(brushImage)[0]
            except IndexError:
                return
            objContour = [self.getObjContours(obj)]
            self.brushContourImage[:] = 0
            img = self.brushContourImage
            color = self.brushContoursRgba
            cv2.drawContours(img, objContour, -1, color, 1)
            self.tempLayerImg1.setImage(img, force_set_linked=True)
        else:
            self.tempLayerImg1.setImage(brushImage, force_set_linked=True)

    def getLabelsLayerImage(self, ax=0):
        if ax == 0:
            return self.labelsLayerImg1.image
        else:
            return self.labelsLayerRightImg.image

    def clearObjFromMask(self, image, mask, toLocalSlice=None):
        if mask is None:
            return image

        if toLocalSlice is None:
            image[mask] = 0
        else:
            image[toLocalSlice][mask] = 0

        return image

    # @exec_time
    def setTempImg1Eraser(self, mask, init=False, toLocalSlice=None, ax=0):
        if init:
            self.erasedLab = np.zeros_like(self.currentLab2D)

        if ax == 0:
            how = self.drawIDsContComboBox.currentText()
        else:
            how = self.getAnnotateHowRightImage()

        if ax == 1 and not self.labelsGrad.showRightImgAction.isChecked():
            return

        if how.find('contours') != -1:
            self.clearObjFromMask(
                self.contoursImage, mask, toLocalSlice=toLocalSlice
            )
            erasedRp = skimage.measure.regionprops(self.erasedLab)
            for obj in erasedRp:
                self.addObjContourToContoursImage(obj=obj, ax=ax)
        elif how.find('overlay segm. masks') != -1:
            labelsImage = self.getLabelsLayerImage(ax=ax)
            self.clearObjFromMask(labelsImage, mask, toLocalSlice=toLocalSlice)
            if ax == 0:
                self.labelsLayerImg1.setImage(
                    self.labelsLayerImg1.image, autoLevels=False
                )
            else:
                self.labelsLayerRightImg.setImage(
                    self.labelsLayerRightImg.image, autoLevels=False
                )

    def Eraser_cb(self, checked):
        if checked:
            self.setDiskMask()
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
            self.updateEraserCursor(self.xHoverImg, self.yHoverImg)
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            c = self.defaultToolBarButtonColor
            self.brushButton.setStyleSheet(f'background-color: {c}')
            self.connectLeftClickButtons()
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )
            self.resetCursors()
            self.updateAllImages()

        self.showEditIDwidgets(checked)
        self.mode_controls_view.enableSizeSpinbox(checked)
