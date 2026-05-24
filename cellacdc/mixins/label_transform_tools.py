"""View adapter for label transform tools."""

from __future__ import annotations

import skimage.measure

from .brush_tools import BrushTools
from .label_editing import LabelEditing


class LabelTransformTools(BrushTools, LabelEditing):
    """Extracted from guiWin."""

    def expandLabel(self, dilation=True):
        posData = self.data[self.pos_i]
        if self.hoverLabelID == 0:
            self.isExpandingLabel = False
            return

        # Re-initialize label to expand when we hover on a different ID
        # or we change direction
        reinitExpandingLab = (
            self.expandingID != self.hoverLabelID or dilation != self.isDilation
        )

        ID = self.hoverLabelID

        obj = posData.rp[posData.IDs.index(ID)]

        if reinitExpandingLab:
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            # hoverLabelID different from previously expanded ID --> reinit
            self.isExpandingLabel = True
            self.expandingID = ID
            self.expandingLab = np.zeros_like(self.currentLab2D)
            self.expandingLab[obj.coords[:, -2], obj.coords[:, -1]] = ID
            self.expandFootprintSize = 1

        prevCoords = (obj.coords[:, -2], obj.coords[:, -1])
        self.currentLab2D[obj.coords[:, -2], obj.coords[:, -1]] = 0
        lab_2D = self.get_2Dlab(posData.lab)
        lab_2D[obj.coords[:, -2], obj.coords[:, -1]] = 0

        footprint = skimage.morphology.disk(self.expandFootprintSize)
        if dilation:
            expandedLab = skimage.morphology.dilation(self.expandingLab, footprint)
            self.isDilation = True
        else:
            expandedLab = skimage.morphology.erosion(self.expandingLab, footprint)
            self.isDilation = False

        # Prevent expanding into neighbouring labels
        expandedLab[self.currentLab2D > 0] = 0

        # Get coords of the dilated/eroded object
        expandedObj = skimage.measure.regionprops(expandedLab)[0]
        expandedObjCoords = (expandedObj.coords[:, -2], expandedObj.coords[:, -1])

        # Add the dilated/erored object
        self.currentLab2D[expandedObjCoords] = self.expandingID
        lab_2D[expandedObjCoords] = self.expandingID

        self.set_2Dlab(lab_2D)
        self.currentLab2D = lab_2D

        self.update_rp()

        if self.labelsGrad.showLabelsImgAction.isChecked():
            self.img2.setImage(img=self.currentLab2D, autoLevels=False)

        self.setTempImgExpandLabel(prevCoords, expandedObjCoords)

    def expandLabelCallback(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            self.connectLeftClickButtons()
            self.expandFootprintSize = 1
        else:
            self.clearHighlightedID()
            alpha = self.imgGrad.labelsAlphaSlider.value()
            self.labelsLayerImg1.setOpacity(alpha)
            self.labelsLayerRightImg.setOpacity(alpha)
            self.hoverLabelID = 0
            self.expandingID = 0
            self.updateAllImages()

    def _setTempImgExpandLabelContours(self, prevCoords, ax=0):
        self.contoursImage[prevCoords] = [0, 0, 0, 0]
        currentLab2Drp = skimage.measure.regionprops(self.currentLab2D)
        for obj in currentLab2Drp:
            if obj.label == self.expandingID:
                # self.clearObjContour(obj=obj, ax=ax)
                self.addObjContourToContoursImage(obj=obj, ax=ax, force=True)
                break

    def _setTempImgExpandLabelSegmMasks(self, prevCoords, ax=0):
        # Remove previous overlaid mask
        labelsImage = self.getLabelsLayerImage(ax=ax)
        labelsImage[prevCoords] = 0

        # Overlay new moved mask
        labelsImage[prevCoords] = self.expandingID

        if ax == 0:
            self.labelsLayerImg1.setImage(self.labelsLayerImg1.image, autoLevels=False)
        else:
            self.labelsLayerRightImg.setImage(
                self.labelsLayerRightImg.image, autoLevels=False
            )

    def resetExpandLabel(self):
        self.expandingID = -1

    def startMovingLabel(self, xPos, yPos):
        posData = self.data[self.pos_i]
        xdata, ydata = int(xPos), int(yPos)
        lab_2D = self.get_2Dlab(posData.lab)
        ID = lab_2D[ydata, xdata]
        if ID == 0:
            self.isMovingLabel = False
            return

        posData = self.data[self.pos_i]
        self.isMovingLabel = True

        self.searchedIDitemRight.setData([], [])
        self.searchedIDitemLeft.setData([], [])
        self.movingID = ID
        self.prevMovePos = (xdata, ydata)
        movingObj = posData.rp[posData.IDs.index(ID)]
        self.movingObjCoords = movingObj.coords.copy()
        yy, xx = movingObj.coords[:, -2], movingObj.coords[:, -1]
        self.currentLab2D[yy, xx] = 0

    def moveLabel(self, xPos, yPos):
        posData = self.data[self.pos_i]
        lab_2D = self.get_2Dlab(posData.lab)
        Y, X = lab_2D.shape
        xdata, ydata = int(xPos), int(yPos)
        if xdata < 0 or ydata < 0 or xdata >= X or ydata >= Y:
            return

        self.clearObjContour(ID=self.movingID, ax=0)

        xStart, yStart = self.prevMovePos
        deltaX = xdata - xStart
        deltaY = ydata - yStart

        yy, xx = self.movingObjCoords[:, -2], self.movingObjCoords[:, -1]

        if self.isSegm3D:
            zz = self.movingObjCoords[:, 0]
            posData.lab[zz, yy, xx] = 0
        else:
            posData.lab[yy, xx] = 0

        self.movingObjCoords[:, -2] = self.movingObjCoords[:, -2] + deltaY
        self.movingObjCoords[:, -1] = self.movingObjCoords[:, -1] + deltaX

        yy, xx = self.movingObjCoords[:, -2], self.movingObjCoords[:, -1]

        yy[yy < 0] = 0
        xx[xx < 0] = 0
        yy[yy >= Y] = Y - 1
        xx[xx >= X] = X - 1

        if self.isSegm3D:
            zz = self.movingObjCoords[:, 0]
            posData.lab[zz, yy, xx] = self.movingID
        else:
            posData.lab[yy, xx] = self.movingID

        self.currentLab2D = self.get_2Dlab(posData.lab)
        if self.labelsGrad.showLabelsImgAction.isChecked():
            self.img2.setImage(self.currentLab2D, autoLevels=False)

        self.setTempImg1MoveLabel()

        self.prevMovePos = (xdata, ydata)

    def setTempImgExpandLabel(self, prevCoords, expandedObjCoords, ax=0):
        if ax == 0:
            how = self.drawIDsContComboBox.currentText()
        else:
            how = self.getAnnotateHowRightImage()

        self._setTempImgExpandLabelContours(prevCoords, ax=ax)

    def setTempImg1MoveLabel(self, ax=0):
        if ax == 0:
            how = self.drawIDsContComboBox.currentText()
        else:
            how = self.getAnnotateHowRightImage()

        if how.find("contours") != -1:
            currentLab2Drp = skimage.measure.regionprops(self.currentLab2D)
            for obj in currentLab2Drp:
                if obj.label == self.movingID:
                    self.addObjContourToContoursImage(obj=obj, ax=ax)
                    break
        elif how.find("overlay segm. masks") != -1:
            if ax == 0:
                self.labelsLayerImg1.setImage(self.currentLab2D, autoLevels=False)
                self.highLightIDLayerImg1.image[:] = 0
                mask = self.currentLab2D == self.movingID
                self.highLightIDLayerImg1.image[mask] = self.movingID
                highlightedImage = self.highLightIDLayerImg1.image
                self.highLightIDLayerImg1.setImage(highlightedImage)
            else:
                self.labelsLayerRightImg.setImage(self.currentLab2D, autoLevels=False)
                self.highLightIDLayerRightImage.image[:] = 0
                mask = self.currentLab2D == self.movingID
                self.highLightIDLayerRightImage.image[mask] = self.movingID
                highlightedImage = self.highLightIDLayerRightImage.image
                self.highLightIDLayerRightImage.setImage(highlightedImage)

    def moveLabelButtonToggled(self, checked):
        if not checked:
            self.hoverLabelID = 0
            self.highlightedID = 0
            self.highLightIDLayerImg1.clear()
            self.highLightIDLayerRightImage.clear()
            self.setHighlightID(False)
