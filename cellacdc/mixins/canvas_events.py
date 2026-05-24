"""Qt view adapter for canvas mouse events."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import skimage.segmentation

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QGuiApplication, QMouseEvent
from qtpy.QtWidgets import QAction, QMessageBox

from cellacdc import apps, exception_handler

from .canvas_context_menu import CanvasContextMenu
from .canvas_selection import CanvasSelection
from .label_editing import LabelEditing


class CanvasEvents(CanvasContextMenu, CanvasSelection, LabelEditing):
    """Extracted from guiWin."""

    def gui_mousePressEventImg1(self, event: QMouseEvent):
        self.typingEditID = False
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        isMod = alt
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        isCcaMode = mode == "Cell cycle analysis"
        isCustomAnnotMode = mode == "Custom annotations"
        left_click = event.button() == Qt.MouseButton.LeftButton and not isMod
        middle_click = self.isMiddleClick(event, modifiers)
        right_click = event.button() == Qt.MouseButton.RightButton
        isPanImageClick = self.isPanImageClick(event, modifiers)
        brushON = self.brushButton.isChecked()
        curvToolON = self.curvToolButton.isChecked()
        histON = self.setIsHistoryKnownButton.isChecked()
        eraserON = self.eraserButton.isChecked()
        rulerON = self.rulerButton.isChecked()
        wandON = self.wandToolButton.isChecked() and not isPanImageClick
        polyLineRoiON = self.addDelPolyLineRoiButton.isChecked()
        labelRoiON = self.labelRoiButton.isChecked()
        keepObjON = self.keepIDsButton.isChecked()
        whitelistIDsON = self.whitelistIDsButton.isChecked()
        separateON = self.separateBudButton.isChecked()
        addPointsByClickingButton = self.buttonAddPointsByClickingActive()
        manualBackgroundON = self.manualBackgroundButton.isChecked()
        magicPromptsON = self.magicPromptsToolButton.isChecked()
        pointsLayerON = self.togglePointsLayerAction.isChecked()
        copyContourON = (
            self.copyLostObjButton.isChecked()
            and self.ax1_lostObjScatterItem.hoverLostID > 0
        )
        findNextMotherButtonON = self.findNextMotherButton.isChecked()
        unknownLineageButtonON = self.unknownLineageButton.isChecked()
        drawClearRegionON = self.drawClearRegionButton.isChecked()
        zoomRectON = self.zoomRectButton.isChecked()

        # Check if right-click on segment of polyline roi to add segment
        segments = self.gui_getHoveredSegmentsPolyLineRoi()
        if len(segments) == 1 and right_click:
            seg = segments[0]
            seg.roi.segmentClicked(seg, event)
            return

        # Check if right-click on handle of polyline roi to remove it
        handles = self.gui_getHoveredHandlesPolyLineRoi()
        if len(handles) == 1 and right_click:
            handle = handles[0]
            handle.roi.removeHandle(handle)
            return

        # Check if click on ROI
        isClickOnDelRoi = self.gui_clickedDelRoi(event, left_click, right_click)
        if isClickOnDelRoi:
            return

        dragImgLeft = (
            left_click
            and not brushON
            and not histON
            and not curvToolON
            and not eraserON
            and not rulerON
            and not wandON
            and not polyLineRoiON
            and not labelRoiON
            and not middle_click
            and not keepObjON
            and not separateON
            and not manualBackgroundON
            and not drawClearRegionON
            and addPointsByClickingButton is None
            and not whitelistIDsON
            and not zoomRectON
        )
        if isPanImageClick:
            dragImgLeft = True

        is_right_click_custom_ON = any(
            [b.isChecked() for b in self.customAnnotDict.keys()]
        )

        canAnnotateDivision = (
            not self.assignBudMothButton.isChecked()
            and not self.setIsHistoryKnownButton.isChecked()
            and not self.curvToolButton.isChecked()
            and not is_right_click_custom_ON
            and not labelRoiON
            and not separateON
        )

        # In timelapse mode division can be annotated if isCcaMode and right-click
        # while in snapshot mode with Ctrl+right-click
        isAnnotateDivision = (right_click and isCcaMode and canAnnotateDivision) or (
            right_click and ctrl and self.isSnapshot
        )

        isCustomAnnot = (
            (right_click or dragImgLeft)
            and (isCustomAnnotMode or self.isSnapshot)
            and self.customAnnotButton is not None
        )

        is_right_click_action_ON = any(
            [b.isChecked() for b in self.checkableQButtonsGroup.buttons()]
        )

        isOnlyRightClick = (
            right_click
            and canAnnotateDivision
            and not isAnnotateDivision
            and not isMod
            and not is_right_click_action_ON
            and not is_right_click_custom_ON
            and not copyContourON
            and not findNextMotherButtonON
            and not unknownLineageButtonON
            and not middle_click
        )

        if isOnlyRightClick:
            # Start timer or check if it is a double-right-click
            if self.countRightClicks == 0:
                self.isDoubleRightClick = False
                self.countRightClicks = 1
                self.doubleRightClickTimeElapsed = False
                screenPos = event.screenPos()
                self._img1_click_xy = (screenPos.x(), screenPos.y())
                QTimer.singleShot(400, self.doubleRightClickTimerCallBack)
                return
            elif self.countRightClicks == 1 and not self.doubleRightClickTimeElapsed:
                self.isDoubleRightClick = True
                self.countRightClicks = 0
                self.editIDbutton.setChecked(True)

        # Left click actions
        canCurv = (
            curvToolON
            and not self.assignBudMothButton.isChecked()
            and not brushON
            and not dragImgLeft
            and not eraserON
            and not polyLineRoiON
            and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON
            and not drawClearRegionON
            and not magicPromptsON
            and not zoomRectON
        )
        canBrush = (
            brushON
            and not curvToolON
            and not rulerON
            and not dragImgLeft
            and not eraserON
            and not wandON
            and not labelRoiON
            and not manualBackgroundON
            and addPointsByClickingButton is None
            and not drawClearRegionON
            and not magicPromptsON
            and not zoomRectON
        )
        canErase = (
            eraserON
            and not curvToolON
            and not rulerON
            and not dragImgLeft
            and not brushON
            and not wandON
            and not polyLineRoiON
            and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON
            and not drawClearRegionON
            and not magicPromptsON
            and not zoomRectON
        )
        canRuler = (
            rulerON
            and not curvToolON
            and not brushON
            and not dragImgLeft
            and not brushON
            and not wandON
            and not polyLineRoiON
            and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON
            and not drawClearRegionON
            and not magicPromptsON
            and not zoomRectON
        )
        canWand = (
            wandON
            and not curvToolON
            and not brushON
            and not dragImgLeft
            and not brushON
            and not rulerON
            and not polyLineRoiON
            and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON
            and not drawClearRegionON
            and not magicPromptsON
            and not zoomRectON
        )
        canPolyLine = (
            polyLineRoiON
            and not wandON
            and not curvToolON
            and not brushON
            and not dragImgLeft
            and not brushON
            and not rulerON
            and not labelRoiON
            and not manualBackgroundON
            and addPointsByClickingButton is None
            and not drawClearRegionON
            and not magicPromptsON
            and not zoomRectON
        )
        canLabelRoi = (
            labelRoiON
            and not wandON
            and not curvToolON
            and not brushON
            and not dragImgLeft
            and not brushON
            and not rulerON
            and not polyLineRoiON
            and not keepObjON
            and addPointsByClickingButton is None
            and not manualBackgroundON
            and not drawClearRegionON
            and not whitelistIDsON
            and not magicPromptsON
            and not zoomRectON
        )
        canKeep = (
            keepObjON
            and not wandON
            and not curvToolON
            and not brushON
            and not dragImgLeft
            and not brushON
            and not rulerON
            and not polyLineRoiON
            and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON
            and not drawClearRegionON
            and not whitelistIDsON
            and not magicPromptsON
            and not zoomRectON
        )
        canWhitelistIDs = (
            whitelistIDsON
            and not wandON
            and not curvToolON
            and not brushON
            and not dragImgLeft
            and not brushON
            and not rulerON
            and not polyLineRoiON
            and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON
            and not drawClearRegionON
            and not keepObjON
            and not magicPromptsON
            and not zoomRectON
        )
        canAddPoint = (
            (pointsLayerON or magicPromptsON)
            and addPointsByClickingButton is not None
            and not wandON
            and not curvToolON
            and not brushON
            and not dragImgLeft
            and not brushON
            and not rulerON
            and not polyLineRoiON
            and not labelRoiON
            and not keepObjON
            and not manualBackgroundON
            and not drawClearRegionON
            and not zoomRectON
        )
        canAddManualBackgroundObj = (
            manualBackgroundON
            and not wandON
            and not curvToolON
            and not brushON
            and not dragImgLeft
            and not brushON
            and not rulerON
            and not polyLineRoiON
            and not labelRoiON
            and addPointsByClickingButton is None
            and not keepObjON
            and not drawClearRegionON
            and not magicPromptsON
            and not whitelistIDsON
            and not zoomRectON
        )
        canDrawClearRegion = (
            drawClearRegionON
            and not wandON
            and not curvToolON
            and not brushON
            and not dragImgLeft
            and not brushON
            and not rulerON
            and not labelRoiON
            and not manualBackgroundON
            and addPointsByClickingButton is None
            and not polyLineRoiON
            and not magicPromptsON
            and not whitelistIDsON
            and not zoomRectON
        )
        canZoomRect = (
            zoomRectON
            and not curvToolON
            and not brushON
            and not dragImgLeft
            and not brushON
            and not rulerON
            and not polyLineRoiON
            and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON
            and not drawClearRegionON
            and not wandON
            and not whitelistIDsON
            and not magicPromptsON
        )

        # Enable dragging of the image window or the scalebar
        if dragImgLeft and not isCustomAnnot:
            x, y = event.pos().x(), event.pos().y()
            if hasattr(self, "scaleBar"):
                if self.scaleBar.isHighlighted():
                    self.scaleBar.mousePressed(x, y)
                    return
            if hasattr(self, "timestamp"):
                if self.timestamp.isHighlighted():
                    self.timestamp.mousePressed(x, y)
                    return
            pg.ImageItem.mousePressEvent(self.img1, event)
            event.ignore()
            return

        isAllowedActionViewer = canAddPoint or canRuler

        if mode == "Viewer" and not isAllowedActionViewer:
            self.startBlinkingModeCB()
            event.ignore()
            return

        # Allow right-click or middle-click actions on both images
        eventOnImg2 = (
            (
                right_click or (middle_click and not canAddPoint)
                # or (left_click and separateON)
            )
            and (mode == "Segmentation and Tracking" or self.isSnapshot)
            and not isAnnotateDivision
            and not manualBackgroundON
        )
        if eventOnImg2:
            event.isImg1Sender = True
            self.gui_mousePressEventImg2(event)

        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        Y, X = self.get_2Dlab(posData.lab).shape
        if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
        else:
            return

        # Paint new IDs with brush and left click on the left image
        if left_click and canBrush:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            lab_2D = self.get_2Dlab(posData.lab)
            Y, X = lab_2D.shape

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False, storeOnlyZoom=True)

            ID = self.getHoverID(xdata, ydata)

            if ID > 0:
                posData.brushID = ID
                self.isNewID = False
            else:
                # Update brush ID. Take care of disappearing cells to remember
                # to not use their IDs anymore in the future
                self.isNewID = True
                self.setBrushID()
                self.updateLookuptable(lenNewLut=posData.brushID + 1)

            self.brushColor = self.lut[posData.brushID] / 255

            self.yPressAx2, self.xPressAx2 = y, x

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)
            diskSlice = (slice(ymin, ymax), slice(xmin, xmax))

            self.isMouseDragImg1 = True

            # Draw new objects
            localLab = lab_2D[diskSlice]
            mask = diskMask.copy()
            if not self.isPowerBrush() and not ctrl:
                mask[localLab != 0] = False

            self.applyBrushMask(mask, posData.brushID, toLocalSlice=diskSlice)

            self.setImageImg2(updateLookuptable=False)

            how = self.drawIDsContComboBox.currentText()
            lab2D = self.get_2Dlab(posData.lab)
            self.globalBrushMask = np.zeros(lab2D.shape, dtype=bool)
            brushMask = localLab == posData.brushID
            brushMask = np.logical_and(brushMask, diskMask)
            self.setTempImg1Brush(
                True, brushMask, posData.brushID, toLocalSlice=diskSlice
            )

            self.lastHoverID = -1

        elif left_click and canErase:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            lab_2D = self.get_2Dlab(posData.lab)
            Y, X = lab_2D.shape

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False, storeOnlyZoom=True)

            self.yPressAx2, self.xPressAx2 = y, x
            # Keep a list of erased IDs got erased
            self.erasedIDs = set()

            if self.xyOnCtrlPressedFirstTime is not None:
                self.erasedID = self.getHoverID(*self.xyOnCtrlPressedFirstTime)
            else:
                self.erasedID = self.getHoverID(xdata, ydata)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            # Build eraser mask
            mask = np.zeros(lab_2D.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True

            # If user double-pressed 'b' then erase over ALL labels
            color = self.eraserButton.palette().button().color().name()
            eraseOnlyOneID = (
                color != self.doublePressKeyButtonColor and self.erasedID != 0
            )

            self.eraseOnlyOneID = eraseOnlyOneID

            if eraseOnlyOneID:
                mask[lab_2D != self.erasedID] = False

            self.setTempImg1Eraser(mask, init=True)
            self.applyEraserMask(mask)

            self.erasedIDs.update(lab_2D[mask])

            for erasedID in self.erasedIDs:
                if erasedID == 0:
                    continue
                self.erasedLab[lab_2D == erasedID] = erasedID

            self.isMouseDragImg1 = True

        elif canAddPoint:
            action = addPointsByClickingButton.action
            self.storeUndoAddPoint(action)
            x, y = event.pos().x(), event.pos().y()
            hoveredPoints = action.scatterItem.pointsAt(event.pos())
            if len(hoveredPoints) > 0:
                removed_ids = self.removeClickedPoints(action, hoveredPoints)
                if not magicPromptsON:
                    removed_id = min(removed_ids)
                    addPointsByClickingButton.pointIdSpinbox.setValue(removed_id)
                    addPointsByClickingButton.pointIdSpinbox.removedId = removed_id
                else:
                    self.restorePrevPointIdRightClick(addPointsByClickingButton)
                self.drawPointsLayers(computePointsLayers=False)
            else:
                point_id = self.getAddedPointId(
                    magicPromptsON,
                    addPointsByClickingButton,
                    right_click,
                    left_click,
                    middle_click,
                )
                if point_id is None:
                    return

                self.addClickedPoint(action, x, y, point_id)
                self.drawPointsLayers(computePointsLayers=False)

                point_id = self.getClickedPointNewId(
                    action,
                    point_id,
                    addPointsByClickingButton.pointIdSpinbox,
                    isMagicPrompts=magicPromptsON,
                )
                addPointsByClickingButton.pointIdSpinbox.setValue(
                    point_id, setLinkedWidget=False
                )

        elif left_click and canDrawClearRegion:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.freeRoiItem.addPoint(xdata, ydata)

            self.isMouseDragImg1 = True

        elif left_click and canRuler or canPolyLine:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            closePolyLine = len(self.startPointPolyLineItem.pointsAt(event.pos())) > 0
            if not self.tempSegmentON or canPolyLine:
                # Keep adding anchor points for polyline
                self.ax1_rulerAnchorsItem.setData([xdata], [ydata])
                self.tempSegmentON = True
            else:
                modifiers = QGuiApplication.keyboardModifiers()
                ctrl = modifiers == Qt.ControlModifier
                self.tempSegmentON = False
                xxRA, yyRA = self.ax1_rulerAnchorsItem.getData()
                x0, y0 = xxRA[0], yyRA[0]
                if ctrl:
                    x1, y1 = transformation.snap_xy_to_closest_angle(
                        x0, y0, xdata, ydata
                    )
                else:
                    x1, y1 = xdata, ydata
                lengthText = self.getRulerLengthText()
                self.ax1_rulerPlotItem.setData(
                    [x0, x1], [y0, y1], lengthText=lengthText
                )
                self.ax1_rulerAnchorsItem.setData([x0, x1], [y0, y1])

            xxPolyLine = self.startPointPolyLineItem.getData()[0]
            if canPolyLine and len(xxPolyLine) == 0:
                # Create and add roi item
                self.createDelPolyLineRoi()
                # Add start point of polyline roi
                self.startPointPolyLineItem.setData([xdata], [ydata])
                self.polyLineRoi.points.append((xdata, ydata))
            elif canPolyLine:
                # Add points to polyline roi and eventually close it
                if not closePolyLine:
                    self.polyLineRoi.points.append((xdata, ydata))
                self.addPointsPolyLineRoi(closed=closePolyLine)
                if closePolyLine:
                    # Close polyline ROI
                    if len(self.polyLineRoi.getLocalHandlePositions()) == 2:
                        self.polyLineRoi = self.replacePolyLineRoiWithLineRoi(
                            self.polyLineRoi
                        )
                    self.tempSegmentON = False
                    self.ax1_rulerAnchorsItem.setData([], [])
                    self.ax1_rulerPlotItem.setData([], [])
                    self.startPointPolyLineItem.setData([], [])
                    self.addRoiToDelRoiInfo(self.polyLineRoi)
                    # Call roi moving on closing ROI
                    self.delROImoving(self.polyLineRoi)
                    self.delROImovingFinished(self.polyLineRoi)

        elif left_click and canKeep:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(self.get_2Dlab(posData.lab), y, x)
                keepID_win = apps.QLineEditDialog(
                    title="Clicked on background",
                    msg="You clicked on the background.\n"
                    "Enter ID that you want to keep",
                    parent=self,
                    allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True,
                )
                keepID_win.exec_()
                if keepID_win.cancel:
                    return
                else:
                    ID = keepID_win.EntryID

            if ID in self.keptObjectsIDs:
                self.keptObjectsIDs.remove(ID)
                self.clearHighlightedText()
            else:
                self.keptObjectsIDs.append(ID)
                self.highlightLabelID(ID)

            self.updateTempLayerKeepIDs()

        elif left_click and canWhitelistIDs:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]

            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(self.get_2Dlab(posData.lab), y, x)
                keepID_win = apps.QLineEditDialog(
                    title="Clicked on background",
                    msg="You clicked on the background.\n"
                    "Enter ID that you want to select",
                    parent=self,
                    allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True,
                )
                keepID_win.exec_()
                if keepID_win.cancel:
                    return
                else:
                    ID = keepID_win.EntryID

            posData = self.data[self.pos_i]

            if not posData.whitelist:
                wl_init = False
                if not hasattr(self, "tempWhitelistIDs"):
                    self.tempWhitelistIDs = (
                        set()
                    )  # not updated, only use in this context
                    current_whitelist = self.tempWhitelistIDs
                else:
                    current_whitelist = self.tempWhitelistIDs
            else:
                wl_init = True
                current_whitelist = posData.whitelist.get(posData.frame_i)

            if ID in current_whitelist:
                current_whitelist.remove(ID)
                self.removeHighlightLabelID(IDs=[ID])
            else:
                current_whitelist.add(ID)
                self.highlightLabelID(ID)

            self.whitelistIDsToolbar.whitelistLineEdit.setText(current_whitelist)

            if wl_init:
                posData.whitelist[posData.frame_i] = current_whitelist
            else:
                self.tempWhitelistIDs = current_whitelist

            self.whitelistUpdateTempLayer()

        elif right_click and copyContourON:
            hoverLostID = self.ax1_lostObjScatterItem.hoverLostID
            self.copyLostObjectMask(hoverLostID)
            self.update_rp()
            self.updateAllImages()
            self.store_data()

        elif right_click and canCurv:
            # Draw manually assisted auto contour
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = self.get_2Dlab(posData.lab).shape

            self.autoCont_x0 = xdata
            self.autoCont_y0 = ydata
            self.xxA_autoCont, self.yyA_autoCont = [], []
            self.curvAnchors.addPoints([x], [y])
            img = self.getDisplayedImg1()
            self.autoContObjMask = np.zeros(img.shape, np.uint8)
            self.isRightClickDragImg1 = True

        elif left_click and canCurv:
            # Draw manual spline
            x, y = event.pos().x(), event.pos().y()
            Y, X = self.get_2Dlab(posData.lab).shape

            # Check if user clicked on starting anchor again --> close spline
            closeSpline = False
            clickedAnchors = self.curvAnchors.pointsAt(event.pos())
            xxA, yyA = self.curvAnchors.getData()
            if len(xxA) > 0:
                if len(xxA) == 1:
                    self.splineHoverON = True
                x0, y0 = xxA[0], yyA[0]
                if len(clickedAnchors) > 0:
                    xA_clicked, yA_clicked = clickedAnchors[0].pos()
                    if x0 == xA_clicked and y0 == yA_clicked:
                        x = x0
                        y = y0
                        closeSpline = True

            # Add anchors
            self.curvAnchors.addPoints([x], [y])
            try:
                xx, yy = self.curvHoverPlotItem.getData()
                self.curvPlotItem.setData(xx, yy)
            except Exception as e:
                # traceback.print_exc()
                pass

            if closeSpline:
                self.splineHoverON = False
                self.curvToolSplineToObj()
                self.update_rp()
                if self.autoIDcheckbox.isChecked():
                    self.trackManuallyAddedObject(posData.brushID, True)
                if self.isSnapshot:
                    self.fixCcaDfAfterEdit("Add new ID with curvature tool")
                    self.updateAllImages()
                else:
                    self.warnEditingWithCca_df("Add new ID with curvature tool")
                self.clearCurvItems()
                self.curvTool_cb(True)

        elif left_click and canWand:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = self.get_2Dlab(posData.lab).shape
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

            self.isNewID = False
            posData.brushID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if posData.brushID == 0:
                self.setBrushID()
                self.updateLookuptable(lenNewLut=posData.brushID + 1)
                self.isNewID = True
            self.brushColor = self.img2.lut[posData.brushID] / 255

            # NOTE: flood is on mousedrag or release
            tol = self.getMagicWandFloodTolerance()
            self.initFloodMaskImage()
            if self.isSegm3D:
                z_slice = self.zSliceScrollBar.sliderPosition()
                seed = (z_slice, ydata, xdata)
            else:
                seed = (ydata, xdata)

            flood_mask = skimage.segmentation.flood(self.flood_img, seed, tolerance=tol)

            drawUnderMask = np.logical_or(
                posData.lab == 0, posData.lab == posData.brushID
            )
            self.flood_mask = np.logical_and(flood_mask, drawUnderMask)

            if self.wandControlsToolbar.autoFillHolesCheckbox.isChecked():
                self.flood_mask = core.binary_fill_holes(self.flood_mask)

            if self.wandControlsToolbar.useConvexHullCheckbox.isChecked():
                self.flood_mask = core.convex_hull_mask(self.flood_mask)

            self.setTempBrushMaskFromWand(self.flood_mask, init=True)
            self.isMouseDragImg1 = True

        elif right_click and self.manualTrackingButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            manualTrackID = self.manualTrackingToolbar.spinboxID.value()
            clickedID = self.getClickedID(
                xdata, ydata, text=f"that you want to assign to {manualTrackID}"
            )
            if clickedID is None:
                return

            if clickedID == manualTrackID:
                self.manualTrackingToolbar.showWarning(
                    f"The clicked object already has ID = {manualTrackID}"
                )
                return

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

            posData = self.data[self.pos_i]
            currentIDs = posData.IDs.copy()
            if manualTrackID in currentIDs:
                tempID = max(currentIDs) + 1
                posData.lab[posData.lab == clickedID] = tempID
                posData.lab[posData.lab == manualTrackID] = clickedID
                posData.lab[posData.lab == tempID] = manualTrackID
                self.manualTrackingToolbar.showWarning(
                    f"The ID {manualTrackID} already exists --> "
                    f"ID {manualTrackID} has been swapped with {clickedID}"
                )
            else:
                posData.lab[posData.lab == clickedID] = manualTrackID
                self.manualTrackingToolbar.showInfo(
                    f"ID {clickedID} changed to {manualTrackID}."
                )

            self.update_rp()
            self.updateAllImages()

        elif right_click and manualBackgroundON:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)

            delID = posData.manualBackgroundLab[ydata, xdata]
            if delID == 0:
                return

            self.clearManualBackgroundObject(delID)
            textItem = self.manualBackgroundTextItems.pop(delID)
            self.ax1.removeItem(textItem)
            self.setManualBackgroundImage()

        elif left_click and canAddManualBackgroundObj:
            x, y = event.pos().x(), event.pos().y()

            self.addManualBackgroundObject(x, y)
            self.setManualBackgroundImage()
            self.setManualBackgrounNextID()

        # Label ROI mouse press
        elif (left_click or right_click) and canLabelRoi:
            if right_click:
                # Force model initialization on mouse release
                self.labelRoiModel = None

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)

            if self.labelRoiIsRectRadioButton.isChecked():
                self.labelRoiItem.setPos((xdata, ydata))
            elif self.labelRoiIsFreeHandRadioButton.isChecked():
                self.freeRoiItem.addPoint(xdata, ydata)

            self.isMouseDragImg1 = True

        # Annotate cell cycle division
        elif isAnnotateDivision:
            if posData.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(self.get_2Dlab(posData.lab), y, x)
                divID_prompt = apps.QLineEditDialog(
                    title="Clicked on background",
                    msg="You clicked on the background.\n"
                    "Enter ID that you want to annotate as divided",
                    parent=self,
                    allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True,
                )
                divID_prompt.exec_()
                if divID_prompt.cancel:
                    return
                else:
                    ID = divID_prompt.EntryID
                    obj_idx = posData.IDs.index(ID)
                    y, x = posData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            if not self.isSnapshot:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                # Annotate or undo division
                self.manualCellCycleAnnotation(ID)
            else:
                self.undoBudMothAssignment(ID)

        # Assign bud to mother (mouse down on bud)
        elif right_click and self.assignBudMothButton.isChecked():
            if self.clickedOnBud:
                # NOTE: self.clickedOnBud is set to False when assigning a mother
                # is successfull in mouse release event
                # We still have to click on a mother
                return

            if posData.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(self.get_2Dlab(posData.lab), y, x)
                budID_prompt = apps.QLineEditDialog(
                    title="Clicked on background",
                    msg="You clicked on the background.\n"
                    "Enter ID of a bud you want to correct mother assignment",
                    parent=self,
                    allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True,
                )
                budID_prompt.exec_()
                if budID_prompt.cancel:
                    return
                else:
                    ID = budID_prompt.EntryID

            obj_idx = posData.IDs.index(ID)
            y, x = posData.rp[obj_idx].centroid
            xdata, ydata = int(x), int(y)

            relationship = posData.cca_df.at[ID, "relationship"]
            is_history_known = posData.cca_df.at[ID, "is_history_known"]
            self.clickedOnHistoryKnown = is_history_known
            # We allow assiging a cell in G1 as bud only on first frame
            # OR if the history is unknown
            if relationship != "bud" and posData.frame_i > 0 and is_history_known:
                txt = (
                    f"You clicked on ID {ID} which is NOT a bud.\n"
                    "To assign a bud to a cell start by clicking on a bud "
                    "and release on a cell in G1"
                )
                msg = QMessageBox()
                msg.critical(self, "Not a bud", txt, msg.Ok)
                return

            self.clickedOnBud = True
            self.xClickBud, self.yClickBud = xdata, ydata

        # Annotate (or undo) that cell has unknown history
        elif right_click and self.setIsHistoryKnownButton.isChecked():
            if posData.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(self.get_2Dlab(posData.lab), y, x)
                unknownID_prompt = apps.QLineEditDialog(
                    title="Clicked on background",
                    msg="You clicked on the background.\n"
                    "Enter ID that you want to annotate as "
                    '"history UNKNOWN/KNOWN"',
                    parent=self,
                    allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True,
                )
                unknownID_prompt.exec_()
                if unknownID_prompt.cancel:
                    return
                else:
                    ID = unknownID_prompt.EntryID
                    obj_idx = posData.IDs.index(ID)
                    y, x = posData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            self.annotateIsHistoryKnown(ID)
            if not self.setIsHistoryKnownButton.findChild(QAction).isChecked():
                self.setIsHistoryKnownButton.setChecked(False)

        elif isCustomAnnot:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(self.get_2Dlab(posData.lab), y, x)
                clickedBkgrDialog = apps.QLineEditDialog(
                    title="Clicked on background",
                    msg="You clicked on the background.\n"
                    "Enter ID that you want to annotate as divided",
                    parent=self,
                    allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True,
                )
                clickedBkgrDialog.exec_()
                if clickedBkgrDialog.cancel:
                    return
                else:
                    ID = clickedBkgrDialog.EntryID
                    obj_idx = posData.IDs.index(ID)
                    y, x = posData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            button = self.doCustomAnnotation(ID)
            if button is None:
                return

            keepActive = self.customAnnotDict[button]["state"]["keepActive"]
            if not keepActive:
                button.setChecked(False)

        elif right_click and findNextMotherButtonON:
            if posData.frame_i == 0:
                return

            self.find_mother_action(posData, event, ydata, xdata)

        elif right_click and unknownLineageButtonON:
            if posData.frame_i == 0:
                return

            self.annotate_unknown_lineage_action(posData, event, ydata, xdata)

        elif (left_click or right_click) and canZoomRect:
            if left_click:
                x, y = event.pos().x(), event.pos().y()
                xdata, ydata = int(x), int(y)

                self.zoomRectItem.setPos((xdata, ydata))

                self.isMouseDragImg1 = True
            else:
                try:
                    xRange, yRange = self.zoomRectItem.getLastRange()
                    self.ax1.setRange(xRange=xRange, yRange=yRange, padding=0)
                except Exception as err:
                    QTimer.singleShot(100, self.autoRange)
