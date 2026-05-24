"""Qt view adapter for brush and eraser tools."""

from __future__ import annotations

import cv2
import numpy as np
import skimage.measure
from qtpy.QtWidgets import QCheckBox

from cellacdc import html_utils, settings_csv_path, widgets

from .geometry import Geometry
from .tool_activation import ToolActivation

class BrushTools(Geometry, ToolActivation):
    """Extracted from guiWin."""

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
            self.setFocusGraphics()
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
        self.enableSizeSpinbox(checked)

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
        self.enableSizeSpinbox(checked)

    def _setTempImageBrushContour(self):
        pass

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

    def autoIDtoggled(self, checked):
        self.editIDspinboxAction.setDisabled(checked)
        self.editIDLabelAction.setDisabled(checked)
        if not checked and self.editIDspinbox.value() == 0:
            newID = self.setBrushID(return_val=True)
            self.editIDspinbox.setValue(newID)

    def brushAutoFillToggled(self, checked):
        val = 'Yes' if checked else 'No'
        self.df_settings.at['brushAutoFill', 'value'] = val
        self.df_settings.to_csv(self.settings_csv_path)

    def brushAutoHideToggled(self, checked):
        val = 'Yes' if checked else 'No'
        self.df_settings.at['brushAutoHide', 'value'] = val
        self.df_settings.to_csv(self.settings_csv_path)

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

    def clearObjFromMask(self, image, mask, toLocalSlice=None):
        if mask is None:
            return image

        if toLocalSlice is None:
            image[mask] = 0
        else:
            image[toLocalSlice][mask] = 0
        
        return image

    def fillHolesID(self, ID, sender='brush'):
        posData = self.data[self.pos_i]
        if sender == 'brush':
            if not self.brushAutoFillCheckbox.isChecked():
                return False
            
            lab2D = self.get_2Dlab(posData.lab)
            mask = lab2D == ID
            filledMask = scipy.ndimage.binary_fill_holes(mask)
            lab2D[filledMask] = ID

            self.set_2Dlab(lab2D)
            return True
        return False

    def getDiskMask(self, xdata, ydata):
        Y, X = self.currentLab2D.shape[-2:]

        brushSize = self.brushSizeSpinbox.value()
        yBottom, xLeft = ydata-brushSize, xdata-brushSize
        yTop, xRight = ydata+brushSize+1, xdata+brushSize+1

        if xLeft<0:
            if yBottom<0:
                # Disk mask out of bounds top-left
                diskMask = self.diskMask.copy()
                diskMask = diskMask[-yBottom:, -xLeft:]
                yBottom = 0
            elif yTop>Y:
                # Disk mask out of bounds bottom-left
                diskMask = self.diskMask.copy()
                diskMask = diskMask[0:Y-yBottom, -xLeft:]
                yTop = Y
            else:
                # Disk mask out of bounds on the left
                diskMask = self.diskMask.copy()
                diskMask = diskMask[:, -xLeft:]
            xLeft = 0

        elif xRight>X:
            if yBottom<0:
                # Disk mask out of bounds top-right
                diskMask = self.diskMask.copy()
                diskMask = diskMask[-yBottom:, 0:X-xLeft]
                yBottom = 0
            elif yTop>Y:
                # Disk mask out of bounds bottom-right
                diskMask = self.diskMask.copy()
                diskMask = diskMask[0:Y-yBottom, 0:X-xLeft]
                yTop = Y
            else:
                # Disk mask out of bounds on the right
                diskMask = self.diskMask.copy()
                diskMask = diskMask[:, 0:X-xLeft]
            xRight = X

        elif yBottom<0:
            # Disk mask out of bounds on top
            diskMask = self.diskMask.copy()
            diskMask = diskMask[-yBottom:]
            yBottom = 0

        elif yTop>Y:
            # Disk mask out of bounds on bottom
            diskMask = self.diskMask.copy()
            diskMask = diskMask[0:Y-yBottom]
            yTop = Y

        else:
            # Disk mask fully inside the image
            diskMask = self.diskMask

        return yBottom, xLeft, yTop, xRight, diskMask

    def getLabelsLayerImage(self, ax=0):
        if ax == 0:
            return self.labelsLayerImg1.image
        else:
            return self.labelsLayerRightImg.image

    def getMagicWandFloodTolerance(self):
        tol_perc = self.wandControlsToolbar.toleranceSpinbox.value()
        if tol_perc == 0:
            return
        
        posData = self.data[self.pos_i]
        _min, _max = posData.img_data_min_max
        tol_fraction = tol_perc/100
        tol = (_max - _min) * tol_fraction
        
        return tol

    def initFloodMaskImage(self):
        posData = self.data[self.pos_i]
        self.flood_img = posData.img_data[posData.frame_i]
        if not self.isSegm3D and posData.SizeZ > 1:
            self.flood_img = self.get_2Dimg_from_3D(self.flood_img)
            return

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

    def instructHowDeleteID(self):
        if 'showInfoDeleteObject' not in self.df_settings.index:
            self.df_settings.at['showInfoDeleteObject', 'value'] = 'Yes'
        
        showInfoDeleteObject = (
            self.df_settings.at['showInfoDeleteObject', 'value'] == 'Yes'
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
            self, 'Delete objects with single click', txt,
            widgets=doNotShowAgainCheckbox
        )
        
        showInfoDeleteObjectValue = (
            'No' if doNotShowAgainCheckbox.isChecked() else 'Yes'
        )
        self.df_settings.at['showInfoDeleteObject', 'value'] = (
            showInfoDeleteObjectValue
        )
        self.df_settings.to_csv(settings_csv_path)

    def resetCursors(self):
        self.ax1_cursor.setData([], [])
        self.ax2_cursor.setData([], [])
        while self.app.overrideCursor() is not None:
            self.app.restoreOverrideCursor()

    def setBrushID(self, useCurrentLab=True, return_val=False):
        # Make sure that the brushed ID is always a new one based on
        # already visited frames
        posData = self.data[self.pos_i]
        wl_init = posData.whitelist and posData.whitelist.whitelistIDs
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
            newID = max(IDs_tot, default=0)
        else:
            newID = 0
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
                _max = max(IDs_tot, default=0)
                if _max > newID:
                    newID = _max
            else:
                break

        for y, x, manual_ID in posData.editID_info:
            if manual_ID > newID:
                newID = manual_ID
        posData.brushID = newID+1
        if return_val:
            return posData.brushID

    def setDiskMask(self):
        brushSize = self.brushSizeSpinbox.value()
        # diam = brushSize*2
        # center = (brushSize, brushSize)
        # diskShape = (diam+1, diam+1)
        # diskMask = np.zeros(diskShape, bool)
        # rr, cc = skimage.draw.disk(center, brushSize+1, shape=diskShape)
        # diskMask[rr, cc] = True
        self.diskMask = skimage.morphology.disk(brushSize, dtype=bool)

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
            # objContour = core.get_obj_contours(
            #     obj_image=(brushImage>0).astype(np.uint8), local=True
            # )
            self.brushContourImage[:] = 0
            img = self.brushContourImage
            color = self.brushContoursRgba
            cv2.drawContours(img, objContour, -1, color, 1)
            self.tempLayerImg1.setImage(img, force_set_linked=True)
        else:
            self.tempLayerImg1.setImage(brushImage, force_set_linked=True)

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

    def showEditIDwidgets(self, visible):
        self.editIDLabelAction.setVisible(visible)
        self.editIDspinboxAction.setVisible(visible)
        self.autoIDcheckboxAction.setVisible(visible)
        showToolbar = (
            visible
            or self.brushSizeAction.isVisible()
            or self.brushAutoFillAction.isVisible()
            or self.brushAutoHideAction.isVisible()
        )
        self.brushEraserToolBar.setVisible(showToolbar)

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
