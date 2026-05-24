"""Qt view adapter for canvas drawing interactions."""

from __future__ import annotations

import numpy as np
import skimage.segmentation

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QAction, QMessageBox

from cellacdc import apps, exception_handler, html_utils, widgets

from .canvas_selection import CanvasSelection
from .label_editing import LabelEditing

class CanvasDrawing(CanvasSelection, LabelEditing):
    """Extracted from guiWin."""

    def gui_addCreatedAxesItems(self):
        self.ax1.addItem(self.ax1_contoursImageItem)
        self.ax1.addItem(self.ax1_lostObjImageItem)
        self.ax1.addItem(self.ax1_lostTrackedObjImageItem)
        self.ax1.addItem(self.ax1_oldMothBudLinesItem)
        self.ax1.addItem(self.ax1_newMothBudLinesItem)
        self.ax1.addItem(self.ax1_lostObjScatterItem)
        self.ax1.addItem(self.ax1_lostTrackedScatterItem)
        self.ax1.addItem(self.ccaFailedScatterItem)
        self.ax1.addItem(self.yellowContourScatterItem)

        self.ax2.addItem(self.ax2_contoursImageItem)
        self.ax2.addItem(self.ax2_lostObjImageItem)
        self.ax2.addItem(self.ax2_lostTrackedObjImageItem)
        self.ax2.addItem(self.ax2_oldMothBudLinesItem)
        self.ax2.addItem(self.ax2_newMothBudLinesItem)
        self.ax2.addItem(self.ax2_lostObjScatterItem)

        self.textAnnot[0].addToPlotItem(self.ax1)
        self.textAnnot[1].addToPlotItem(self.ax2)
        
        self.ax1.addItem(self.exportMaskImageItem)
        self.ax1.exportMaskImageItem = self.exportMaskImageItem

    def gui_mouseDragEventImg1(self, event):
        x, y = event.pos().x(), event.pos().y()
        
        if hasattr(self, 'scaleBar'):
            if self.scaleBarDialog is not None:
                self.scaleBarDialog.locCombobox.setCurrentText('Custom')
            if self.scaleBar.isHighlighted() and self.scaleBar.clicked:
                self.scaleBar.setLocationProperty('custom')
                self.scaleBar.move(x, y)
                return
        
        if hasattr(self, 'timestamp'):
            if self.timestampDialog is not None:
                self.timestampDialog.locCombobox.setCurrentText('Custom')
            if self.timestamp.isHighlighted() and self.timestamp.clicked:
                self.timestamp.setLocationProperty('custom')
                self.timestamp.move(x, y)
                return
        
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return
        
        posData = self.data[self.pos_i]
        Y, X = self.get_2Dlab(posData.lab).shape
        xdata, ydata = int(x), int(y)
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            return
        
        if self.isRightClickDragImg1 and self.curvToolButton.isChecked():
            self.drawAutoContour(y, x)

        # Brush dragging mouse --> keep brushing
        elif self.isMouseDragImg1 and self.brushButton.isChecked():
            lab_2D = self.get_2Dlab(posData.lab)

            # t1 = time.perf_counter()

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            # t2 = time.perf_counter()

            diskSlice = (slice(ymin, ymax), slice(xmin, xmax))

            # Build brush mask
            mask = np.zeros(lab_2D.shape, bool)
            mask[diskSlice][diskMask] = True
            mask[rrPoly, ccPoly] = True
            
            modifiers = QGuiApplication.keyboardModifiers()
            ctrl = modifiers == Qt.ControlModifier

            # t3 = time.perf_counter()
            if not self.isPowerBrush() and not ctrl:
                mask[lab_2D!=0] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.ax2_BrushCirclePen,
                    (self.ax2_BrushCircle, self.ax1_BrushCircle),
                    self.brushButton, brush=self.ax2_BrushCircleBrush
                )

            # t4 = time.perf_counter()

            # Apply brush mask
            self.applyBrushMask(mask, posData.brushID)

            self.setImageImg2(updateLookuptable=False)

            # t5 = time.perf_counter()

            lab2D = self.get_2Dlab(posData.lab)
            brushMask = np.logical_and(
                lab2D[diskSlice] == posData.brushID, diskMask
            )
            self.setTempImg1Brush(
                False, brushMask, posData.brushID, 
                toLocalSlice=diskSlice
            )

            # t6 = time.perf_counter()

            # printl(
            #     'Brush exec times =\n'
            #     f'  * {(t1-t0)*1000 = :.4f} ms\n'
            #     f'  * {(t2-t1)*1000 = :.4f} ms\n'
            #     f'  * {(t3-t2)*1000 = :.4f} ms\n'
            #     f'  * {(t4-t3)*1000 = :.4f} ms\n'
            #     f'  * {(t5-t4)*1000 = :.4f} ms\n'
            #     f'  * {(t6-t5)*1000 = :.4f} ms\n'
            #     f'  * {(t6-t0)*1000 = :.4f} ms'
            # )

        # Eraser dragging mouse --> keep erasing
        elif self.isMouseDragImg1 and self.eraserButton.isChecked():
            posData = self.data[self.pos_i]
            lab_2D = self.get_2Dlab(posData.lab)
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            diskSlice = (slice(ymin, ymax), slice(xmin, xmax))

            # Build eraser mask
            mask = np.zeros(lab_2D.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            if self.eraseOnlyOneID:
                mask[lab_2D!=self.erasedID] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.eraserCirclePen,
                    (self.ax2_EraserCircle, self.ax1_EraserCircle),
                    self.eraserButton, hoverRGB=self.img2.lut[self.erasedID],
                    ID=self.erasedID
                )

            self.erasedIDs.update(lab_2D[mask])
            self.applyEraserMask(mask)

            self.setImageImg2()
            
            for erasedID in self.erasedIDs:
                if erasedID == 0:
                    continue
                self.erasedLab[lab_2D==erasedID] = erasedID
                self.erasedLab[mask] = 0

            eraserMask = mask[diskSlice]
            self.setTempImg1Eraser(eraserMask, toLocalSlice=diskSlice)
            self.setTempImg1Eraser(eraserMask, toLocalSlice=diskSlice, ax=1)

        # Move label dragging mouse --> keep moving
        elif self.isMovingLabel and self.moveLabelToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            self.moveLabel(x, y)

        # Wand dragging mouse --> keep doing the magic
        elif self.isMouseDragImg1 and self.wandToolButton.isChecked():
            tol = self.getMagicWandFloodTolerance()
            if self.isSegm3D:
                z_slice = self.zSliceScrollBar.sliderPosition()
                seed = (z_slice, ydata, xdata)
            else:
                seed = (ydata, xdata)
                
            flood_mask = skimage.segmentation.flood(
                self.flood_img, seed, tolerance=tol
            )
            drawUnderMask = np.logical_or(
                posData.lab==0, posData.lab==posData.brushID
            )
            flood_mask = np.logical_and(flood_mask, drawUnderMask)

            self.flood_mask[flood_mask] = True

            if self.wandControlsToolbar.autoFillHolesCheckbox.isChecked():
                self.flood_mask = core.binary_fill_holes(self.flood_mask)
            
            if self.wandControlsToolbar.useConvexHullCheckbox.isChecked():
                self.flood_mask = core.convex_hull_mask(self.flood_mask)

            self.setTempBrushMaskFromWand(self.flood_mask)
        
        # Label ROI dragging mouse --> draw ROI
        elif self.isMouseDragImg1 and self.labelRoiButton.isChecked():
            if self.labelRoiIsRectRadioButton.isChecked():
                x0, y0 = self.labelRoiItem.pos()
                w, h = (xdata-x0), (ydata-y0)
                self.labelRoiItem.setSize((w, h))
            elif self.labelRoiIsFreeHandRadioButton.isChecked():
                self.freeRoiItem.addPoint(xdata, ydata)
        
        # Draw freehand clear region --> draw region
        elif self.isMouseDragImg1 and self.drawClearRegionButton.isChecked():
            self.freeRoiItem.addPoint(xdata, ydata)
        
        # Label ROI dragging mouse --> draw ROI
        elif self.isMouseDragImg1 and self.zoomRectButton.isChecked():
            x0, y0 = self.zoomRectItem.pos()
            w, h = (xdata-x0), (ydata-y0)
            self.zoomRectItem.setSize((w, h))

    def gui_mouseDragEventImg2(self, event):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        Y, X = self.get_2Dlab(posData.lab).shape
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            return

        # Eraser dragging mouse --> keep erasing
        if self.isMouseDragImg2 and self.eraserButton.isChecked():
            posData = self.data[self.pos_i]
            lab_2D = self.get_2Dlab(posData.lab)
            Y, X = lab_2D.shape
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            brushSize = self.brushSizeSpinbox.value()
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            # Build eraser mask
            mask = np.zeros(lab_2D.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            if self.eraseOnlyOneID:
                mask[lab_2D!=self.erasedID] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.eraserCirclePen,
                    (self.ax2_EraserCircle, self.ax1_EraserCircle),
                    self.eraserButton, hoverRGB=self.img2.lut[self.erasedID],
                    ID=self.erasedID
                )

            self.erasedIDs.update(lab_2D[mask])

            self.applyEraserMask(mask)
            self.setImageImg2(updateLookuptable=False)

        # Brush paint dragging mouse --> keep painting
        if self.isMouseDragImg2 and self.brushButton.isChecked():
            posData = self.data[self.pos_i]
            lab_2D = self.get_2Dlab(posData.lab)
            Y, X = lab_2D.shape
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            # Build brush mask
            mask = np.zeros(lab_2D.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            # If user double-pressed 'b' then draw over the labels
            color = self.brushButton.palette().button().color().name()
            if color != self.doublePressKeyButtonColor:
                mask[lab_2D!=0] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.ax2_BrushCirclePen,
                    (self.ax2_BrushCircle, self.ax1_BrushCircle),
                    self.eraserButton, brush=self.ax2_BrushCircleBrush
                )

            # Apply brush mask
            self.applyBrushMask(mask, self.ax2BrushID)

            self.setImageImg2()

        # Move label dragging mouse --> keep moving
        elif self.isMovingLabel and self.moveLabelToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            self.moveLabel(x, y)

    def gui_mouseReleaseEventImg1(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        right_click = event.button() == Qt.MouseButton.RightButton and not alt
        
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return
        
        Y, X = self.get_2Dlab(posData.lab).shape
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            self.isMouseDragImg2 = False
            self.updateAllImages()
            return
        
        if hasattr(self, 'scaleBar'):
            if self.scaleBar.isHighlighted() and self.scaleBar.clicked:
                self.scaleBar.clicked = False
                return
        
        if hasattr(self, 'timestamp'):
            if self.timestamp.isHighlighted() and self.timestamp.clicked:
                self.timestamp.clicked = False
                return
        
        sendRightClickImg2 = (
            (mode=='Segmentation and Tracking' or self.isSnapshot)
            and right_click
        )
        if sendRightClickImg2:
            # Allow right-click actions on both images
            self.gui_mouseReleaseEventImg2(event)

        # Right-click curvature tool mouse release
        if self.isRightClickDragImg1 and self.curvToolButton.isChecked():
            self.isRightClickDragImg1 = False
            try:
                self.curvToolSplineToObj(isRightClick=True)
                self.update_rp()
                if self.autoIDcheckbox.isChecked():
                    self.trackManuallyAddedObject(posData.brushID, True)
                if self.isSnapshot:
                    self.fixCcaDfAfterEdit('Add new ID with curvature tool')
                    self.updateAllImages()
                else:
                    self.warnEditingWithCca_df('Add new ID with curvature tool')
                self.clearCurvItems()
                self.curvTool_cb(True)
            except ValueError:
                self.clearCurvItems()
                self.curvTool_cb(True)
                pass

        # Eraser mouse release --> update IDs and contours
        elif self.isMouseDragImg1 and self.eraserButton.isChecked():
            self.isMouseDragImg1 = False

            self.clearTempBrushImage()
        
            # Update data (rp, etc)
            self.update_rp()

            doUpdateImages = self.checkWarnDeletedIDwithEraser()
            
            if doUpdateImages:
                self.updateAllImages()

        # Brush button mouse release
        elif self.isMouseDragImg1 and self.brushButton.isChecked():
            self.isMouseDragImg1 = False

            self.clearTempBrushImage()
            
            self.brushReleased()

        # Wand tool release, add new object
        elif self.isMouseDragImg1 and self.wandToolButton.isChecked():
            self.isMouseDragImg1 = False

            self.clearTempBrushImage()

            posData = self.data[self.pos_i]
            posData.lab[self.flood_mask] = posData.brushID
            
            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.trackManuallyAddedObject(posData.brushID, self.isNewID)

            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Add new ID with magic-wand')
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df('Add new ID with magic-wand')
        
        # Label ROI mouse release --> label the ROI with labelRoiWorker
        elif self.isMouseDragImg1 and self.labelRoiButton.isChecked():
            self.labelRoiRunning = True
            self.app.setOverrideCursor(Qt.WaitCursor)
            self.isMouseDragImg1 = False

            if self.labelRoiIsFreeHandRadioButton.isChecked():
                self.freeRoiItem.closeCurve()
            
            proceed = self.labelRoiCheckStartStopFrame()
            if not proceed:
                self.labelRoiCancelled()
                return

            roiImg, self.labelRoiSlice = self.getLabelRoiImage()

            if roiImg.size == 0:
                self.labelRoiCancelled()
                return

            if self.labelRoiModel is None:
                cancel = self.initLabelRoiModel()
                if cancel:
                    self.labelRoiCancelled()
                    return
            
            # Restore state of button because it was maybe unchecked by 
            # using other tools that are allowed --> see "elif" case in 
            # labelRoi_cb
            self.labelRoiButton.blockSignals(True)
            self.labelRoiButton.setChecked(True)
            self.labelRoiToolbar.setVisible(True)
            self.labelRoiButton.blockSignals(False)
            
            roiSecondChannel = None
            if self.secondChannelName is not None:
                secondChannelData = self.getSecondChannelData()
                roiSecondChannel = secondChannelData[self.labelRoiSlice]
            
            isTimelapse = self.labelRoiTrangeCheckbox.isChecked()
            if isTimelapse:
                start_n = self.labelRoiStartFrameNoSpinbox.value()
                stop_n = self.labelRoiStopFrameNoSpinbox.value()
                self.progressWin = apps.QDialogWorkerProgress(
                    title='ROI segmentation', parent=self,
                    pbarDesc=f'Segmenting frames n. {start_n} to {stop_n}...'
                )
                self.progressWin.show(self.app)
                self.progressWin.mainPbar.setMaximum(stop_n-start_n)                

            
            self.app.restoreOverrideCursor() 
            labelRoiWorker = self.labelRoiActiveWorkers[-1]
            labelRoiWorker.start(
                roiImg, posData, 
                roiSecondChannel=roiSecondChannel, 
                isTimelapse=isTimelapse
            )            
            self.app.setOverrideCursor(Qt.WaitCursor)
            self.logger.info(
                f'Magic labeller started on image ROI = {self.labelRoiSlice}...'
            )
            self.titleLabel.setText('Magic labeller is doing its magic...')
            self.setDisabled(True)

        # Move label mouse released, update move
        elif self.isMovingLabel and self.moveLabelToolButton.isChecked():
            self.isMovingLabel = False

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True, assign_unique_new_IDs=False)

            if not self.moveLabelToolButton.findChild(QAction).isChecked():
                self.moveLabelToolButton.setChecked(False)
            else:
                self.updateAllImages()

        # Assign mother to bud
        elif self.assignBudMothButton.isChecked() and self.clickedOnBud:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == self.get_2Dlab(posData.lab)[self.yClickBud, self.xClickBud]:
                return

            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab), y, x
                )
                mothID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as mother cell',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                mothID_prompt.exec_()
                if mothID_prompt.cancel:
                    return
                else:
                    ID = mothID_prompt.EntryID
                    obj_idx = posData.IDs.index(ID)
                    y, x = posData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            if self.isSnapshot:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)

            relationship = posData.cca_df.at[ID, 'relationship']
            ccs = posData.cca_df.at[ID, 'cell_cycle_stage']
            is_history_known = posData.cca_df.at[ID, 'is_history_known']
            # We allow assiging a cell in G1 as mother only on first frame
            # OR if the history is unknown
            if relationship == 'bud' and posData.frame_i > 0 and is_history_known:
                self.assignBudMothButton.setChecked(False)
                txt = html_utils.paragraph(
                    f'You clicked on <b>ID {ID}</b> which is a <b>BUD</b>.<br><br>'
                    'To assign a bud <b>start by clicking on the bud</b> '
                    'and release on a cell in G1'
                )
                msg = widgets.myMessageBox()
                msg.critical(
                    self, 'Released on a bud', txt
                )
                self.assignBudMothButton.setChecked(True)
                return

            elif posData.frame_i == 0:
                # Check that clicked bud actually is smaller that mother
                # otherwise warn the user that he might have clicked first
                # on a mother
                budID = self.get_2Dlab(posData.lab)[self.yClickBud, self.xClickBud]
                new_mothID = self.get_2Dlab(posData.lab)[ydata, xdata]
                bud_obj_idx = posData.IDs.index(budID)
                new_moth_obj_idx = posData.IDs.index(new_mothID)
                rp_budID = posData.rp[bud_obj_idx]
                rp_new_mothID = posData.rp[new_moth_obj_idx]
                if rp_budID.area >= rp_new_mothID.area:
                    self.assignBudMothButton.setChecked(False)
                    msg = widgets.myMessageBox()
                    txt = (
                        f'You clicked FIRST on ID {budID} and then on {new_mothID}.<br>'
                        f'For me this means that you want ID {budID} to be the '
                        f'BUD of ID {new_mothID}.<br>'
                        f'However <b>ID {budID} is bigger than {new_mothID}</b> '
                        f'so maybe you should have clicked FIRST on {new_mothID}?<br><br>'
                        'What do you want me to do?'
                    )
                    txt = html_utils.paragraph(txt)
                    swapButton, keepButton = msg.warning(
                        self, 'Which one is bud?', txt,
                        buttonsTexts=(
                            f'Assign ID {new_mothID} as the bud of ID {budID}',
                            f'Keep ID {budID} as the bud of  ID {new_mothID}'
                        )
                    )
                    if msg.clickedButton == swapButton:
                        (xdata, ydata,
                        self.xClickBud, self.yClickBud) = (
                            self.xClickBud, self.yClickBud,
                            xdata, ydata
                        )
                    self.assignBudMothButton.setChecked(True)

            elif is_history_known and not self.clickedOnHistoryKnown:
                self.assignBudMothButton.setChecked(False)
                budID = self.get_2Dlab(posData.lab)[ydata, xdata]
                # Allow assigning an unknown cell ONLY to another unknown cell
                txt = (
                    f'You started by clicking on ID {budID} which has '
                    'UNKNOWN history, but you then clicked/released on '
                    f'ID {ID} which has KNOWN history.\n\n'
                    'Only two cells with UNKNOWN history can be assigned as '
                    'relative of each other.'
                )
                msg = QMessageBox()
                msg.critical(
                    self, 'Released on a cell with KNOWN history', txt, msg.Ok
                )
                self.assignBudMothButton.setChecked(True)
                return

            self.clickedOnHistoryKnown = is_history_known
            self.xClickMoth, self.yClickMoth = xdata, ydata
            
            if ccs != 'G1' and posData.frame_i > 0:
                self.assignBudMothButton.setChecked(False)
                self.onMotherNotInG1(ID)
                self.assignBudMothButton.setChecked(True)
            else:
                self.annotateBudToDifferentMother()

            if not self.assignBudMothButton.findChild(QAction).isChecked():
                self.assignBudMothButton.setChecked(False)

            self.clickedOnBud = False
            self.BudMothTempLine.setData([], [])
        
        # Draw clear region mouse release
        elif self.isMouseDragImg1 and self.drawClearRegionButton.isChecked():
            self.isMouseDragImg1 = False
            self.freeRoiItem.closeCurve()
            self.clearObjsFreehandRegion()
        
        # Zoom rect mouse release 
        elif self.isMouseDragImg1 and self.zoomRectButton.isChecked():
            self.isMouseDragImg1 = False
            self.zoomRectDone()
