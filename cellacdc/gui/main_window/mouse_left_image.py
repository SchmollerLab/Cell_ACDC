"""Mouse press, drag and release of the left image (img1) of the main GUI,
where the editing tools act. Mixins of guiWin
(gui/main_window/window.py) only: the measurements viewer never composes
them.
"""
from functools import partial
import numpy as np
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation
from qtpy.QtCore import Qt, QPoint, QTimer
from qtpy.QtGui import QGuiApplication, QMouseEvent
from qtpy.QtWidgets import QAction, QMessageBox
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import exception_handler
from cellacdc import html_utils
from cellacdc.shared.dialogs import generic as dialogs_generic
from cellacdc import core
from cellacdc.myutils import image_utils
from cellacdc.shared.widgets import message_box as widgets_message_box
from cellacdc import qutils
from cellacdc import transformation
from cellacdc.config import STANDARD_MOUSE_BUTTONS


class LeftImageMouseGuiElements:
    """Press, drag and release handlers of the left image (img1), where the
    editing tools act, plus the clicked-ID lookup, the double
    right-click timer and the image LUT context menu.
    """

    @exception_handler
    def gui_mouseDragEventImg1(self, event):
        x, y = event.pos().x(), event.pos().y()
        
        whitelistIDsON = self.whitelistIDsButton.isChecked()
        drawWhitelistRegionON = (
            whitelistIDsON and self.whitelistIDsToolbar.isRoiToggled()
        )
        
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

        if self.isBrushStrokeDragImg1:
            # Painting brush strokes prompts (clipped at image borders)
            self.brushStrokesPaintAt(x, y)
            return

        if self.isBoxPromptDragImg1:
            # Drawing a box prompt (clipped at image borders)
            self.boxPromptsDragTo(x, y)
            return

        posData = self.data[self.pos_i]
        Y, X = self.get_2Dlab(posData.lab, force_z=False).shape
        xdata, ydata = int(x), int(y)
        if not image_utils.is_in_bounds(xdata, ydata, X, Y):
            return

        if self.isRightClickDragImg1 and self.curvToolButton.isChecked():
            self.drawAutoContour(y, x)

        # Brush dragging mouse --> keep brushing
        elif self.isMouseDragImg1 and self.brushButton.isChecked():
            lab_2D = self.get_2Dlab(posData.lab, force_z=False)

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

            lab2D = self.get_2Dlab(posData.lab, force_z=False)
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
            lab_2D = self.get_2Dlab(posData.lab, force_z=False)
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
        elif self.isMouseDragImg1 and self.clearFreehandRoiButton.isChecked():
            self.freeRoiItem.addPoint(xdata, ydata)
        
        # Draw freehand merge region --> draw region
        elif self.isMouseDragImg1 and self.mergeIDsButton.isChecked():
            self.freeRoiItem.addPoint(xdata, ydata)
        
        # Draw freehand whitelist IDs region --> draw region
        elif self.isMouseDragImg1 and drawWhitelistRegionON:
            self.freeRoiItem.addPoint(xdata, ydata)
        
        # Label ROI dragging mouse --> draw ROI
        elif self.isMouseDragImg1 and self.zoomRectButton.isChecked():
            x0, y0 = self.zoomRectItem.pos()
            w, h = (xdata-x0), (ydata-y0)
            self.zoomRectItem.setSize((w, h))

    def gui_imgGradShowContextMenu(self, x, y):
        if hasattr(self, 'scaleBar'):
            if self.scaleBar.isHighlighted():
                self.scaleBar.showContextMenu(x, y)
                return
        
        if hasattr(self, 'timestamp'):
            if self.timestamp.isHighlighted():
                self.timestamp.showContextMenu(x, y)
                return
            
        self.imgGrad.gradient.menu.popup(QPoint(int(x), int(y)))

    @exception_handler
    def gui_mouseReleaseEventImg1(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        right_click = event.button() == Qt.MouseButton.RightButton and not alt

        if self.isBrushStrokeDragImg1:
            # End of brush strokes prompts painting
            self.brushStrokesEndDrag()
            return

        if self.isBoxPromptDragImg1:
            # End of box prompt drawing
            self.boxPromptsEndDrag(event)
            return

        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        whitelistIDsON = self.whitelistIDsButton.isChecked()
        drawWhitelistRegionON = (
            whitelistIDsON and self.whitelistIDsToolbar.isRoiToggled()
        )

        Y, X = self.get_2Dlab(posData.lab, force_z=False).shape
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        if not image_utils.is_in_bounds(xdata, ydata, X, Y):
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
                mask, returnID = self.curvToolSplineToObj(isRightClick=True)
                if mask is not None:
                    self.update_rp() # how can I optimize this? I think not possible tbh
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

            erasedIDs = [ID for ID in self.erasedIDs if ID != 0]
        
            # Update data (rp, etc)
            self.update_rp(
                use_curr_view=True,
                specific_IDs=erasedIDs or None,
            ) # only visible stuff can be deleted

            doUpdateImages = self.checkWarnDeletedIDwithEraser()
            
            if not doUpdateImages:
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
            # only visible stuff can be added, plus doesnt draw over eixisting
            self.update_rp(use_curr_view=True, specific_IDs=posData.brushID)

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
                self.progressWin = dialogs_generic.QDialogWorkerProgress(
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
            self.update_rp(use_curr_view=True) # only visible stuff can be moved

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
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                mothID_prompt = dialogs_generic.QLineEditDialog(
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
                    centroid = posData.rp.get_centroid(ID)
                    ydata, xdata = self.getObjCentroid(centroid)
                    ydata, xdata = int(ydata), int(xdata)

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
                msg = widgets_message_box.myMessageBox()
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
                bug_obj = posData.rp.get_obj_from_ID(budID)
                new_mother_obj = posData.rp.get_obj_from_ID(new_mothID)
                if bug_obj.area >= new_mother_obj.area:
                    self.assignBudMothButton.setChecked(False)
                    msg = widgets_message_box.myMessageBox()
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
        elif self.isMouseDragImg1 and self.clearFreehandRoiButton.isChecked():
            self.isMouseDragImg1 = False
            self.freeRoiItem.closeCurve()
            self.clearObjsFreehandRegion()
        
        # Zoom rect mouse release 
        elif self.isMouseDragImg1 and self.zoomRectButton.isChecked():
            self.isMouseDragImg1 = False
            self.zoomRectDone()
        
        # Draw merge region mouse release
        elif self.isMouseDragImg1 and self.mergeIDsButton.isChecked():
            self.isMouseDragImg1 = False
            self.mergeObjsFreehandRegion(event)
        
        # Draw freehand whitelist IDs region mouse release
        elif self.isMouseDragImg1 and drawWhitelistRegionON:
            self.isMouseDragImg1 = False
            self.whiteListIDsFreehandRegion(event)

    @exception_handler
    def gui_mousePressEventImg1(self, event: QMouseEvent):
        self.typingEditID = False
        if getattr(self, 'magicPromptsDrawingBox', None) is not None:
            # Another mouse button pressed while drawing a box prompt -->
            # cancel the box (before any other action of that button, e.g.,
            # middle-click deleting an object)
            self.boxPromptsCancelDrag()
            return

        if getattr(self, 'magicPromptsBoxItems', None):
            if self.isMovingMagicPromptsBox():
                # Another button pressed while a box prompt is moved or
                # resized --> leave it to pyqtgraph (img1 must not grab the
                # mouse, otherwise the move is never finished)
                event.ignore()
                return

        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        isMod = alt
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        isCcaMode = mode == 'Cell cycle analysis'
        isCustomAnnotMode = mode == 'Custom annotations'
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
            and self.ax1_lostObjScatterItem.hoverLostID>0
        )
        findNextMotherButtonON = (
            self.findNextMotherButton.isChecked()
            and mode == 'Normal division: Lineage tree'
        )
        unknownLineageButtonON = (
            self.unknownLineageButton.isChecked()
            and mode == 'Normal division: Lineage tree'
        )
        drawClearRegionON = self.clearFreehandRoiButton.isChecked()
        zoomRectON = self.zoomRectButton.isChecked()
        drawMergeRegionON = self.mergeIDsButton.isChecked()
        drawWhitelistRegionON = (
            whitelistIDsON and self.whitelistIDsToolbar.isRoiToggled()
        )
        addToWhitelistByClick = (
            whitelistIDsON and not self.whitelistIDsToolbar.isRoiToggled()
        )

        activeRois = []
        if self.labelRoiButton.isChecked() and self.labelRoiIsRectRadioButton.isChecked():
            activeRois.append(getattr(self, 'labelRoiItem', None))
        if self.zoomRectButton.isChecked():
            activeRois.append(getattr(self, 'zoomRectItem', None))

        activeRois = [
            roi for roi in activeRois if self.gui_isActiveInterceptRoi(roi)]
        
        # check that no other button was clicked
        
        click_shortcut_action = ''
        if event.button() not in STANDARD_MOUSE_BUTTONS:
            for name, button in self.mouseBindings.items():
                if button == event.button():
                    click_shortcut_action = name
                    self.mousePressEvent(event)
                    break
                    

        if any(self.gui_isHoveringRoiHandle(roi) for roi in activeRois):
            event.ignore()
            return

        if left_click and any(
            self.gui_isMouseInsideRoi(roi, event) for roi in activeRois
        ):
            event.ignore()
            return

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
        isClickOnDelRoi = self.gui_handleClickOnDelRoi(
            event, left_click, right_click
        )
        if isClickOnDelRoi:
            return

        # Ctrl+right-click in snapshot mode annotates divisions
        isSnapshotDivisionClick = right_click and ctrl and self.isSnapshot
        isSelectCellsClick = (
            (left_click or right_click) and not middle_click and not alt
            and not isSnapshotDivisionClick
            and self.isBrushStrokesSelectCellsActive()
        )
        if isSelectCellsClick:
            # Select (or unselect) cells to add them to the brush strokes.
            # Do not ignore the event (no pan nor context menu)
            self.brushStrokesSelectCellClick(event)
            return

        isBrushStrokesClick = (
            (left_click or right_click) and not middle_click and not alt
            and not isSnapshotDivisionClick
            and self.isBrushStrokesPromptActive()
        )
        if isBrushStrokesClick:
            # Paint brush strokes prompts (left = foreground,
            # right = background, Shift = erase). Do not ignore the event
            # so that img1 grabs the mouse (no pan nor context menu)
            self.brushStrokesMousePress(event, left_click, modifiers)
            return

        isBoxPromptClick = (
            (left_click or right_click) and not middle_click and not alt
            and not isSnapshotDivisionClick
            and self.isBoxPromptActive()
        )
        if isBoxPromptClick:
            # Draw box prompts (left = new object, right = object of the
            # "Right-click ID"). Left-presses on the border or handles of a
            # box are left to the box (move, resize) and right-clicks
            # without dragging open the menu of the box (remove)
            self.boxPromptsMousePress(event, right_click)
            return

        isBoxPromptOtherClick = (
            (middle_click or (right_click and alt))
            and self.isBoxPromptActive()
        )
        if isBoxPromptOtherClick:
            # With box prompts, points (also the negative ones) are added
            # only with the "Points" prompt type, and middle-click does not
            # delete objects (like with points)
            return

        dragImgLeft = (
            left_click and not brushON and not histON
            and not curvToolON and not eraserON and not rulerON
            and not wandON and not polyLineRoiON and not labelRoiON
            and not middle_click and not keepObjON and not separateON
            and not manualBackgroundON and not drawClearRegionON
            and addPointsByClickingButton is None and not whitelistIDsON
            and not zoomRectON and not drawMergeRegionON
            and not drawWhitelistRegionON
        )
        if isPanImageClick:
            dragImgLeft = True
            if self.hoveredMagicPromptsBox() is not None:
                # The hovered box prompt claimed the left drags (pyqtgraph
                # hover) but ignores Alt+drag --> re-send the hover with the
                # pressed button, so that the view gets the pan
                try:
                    self.ax1.scene().sendHoverEvents(event)
                except Exception as err:
                    pass

        is_right_click_custom_ON = any([
            b.isChecked() for b in self.customAnnotDict.keys()
        ])

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
        isAnnotateDivision = (
            (right_click and isCcaMode and canAnnotateDivision)
            or (right_click and ctrl and self.isSnapshot and canAnnotateDivision)
        )

        isCustomAnnot = (
            (right_click or dragImgLeft)
            and (isCustomAnnotMode or self.isSnapshot)
            and self.customAnnotButton is not None
        )

        is_right_click_action_ON = any([
            b.isChecked() for b in self.checkableQButtonsGroup.buttons()
        ])

        isOnlyRightClick = (
            right_click and canAnnotateDivision and not isAnnotateDivision
            and not isMod and not is_right_click_action_ON
            and not is_right_click_custom_ON and not copyContourON 
            and not findNextMotherButtonON and not unknownLineageButtonON
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
            elif (
                self.countRightClicks == 1 
                and not self.doubleRightClickTimeElapsed
            ):            
                self.isDoubleRightClick = True
                self.countRightClicks = 0
                self.editIDbutton.setChecked(True)

        # Left click actions
        canCurv = (
            curvToolON and not self.assignBudMothButton.isChecked()
            and not brushON and not dragImgLeft and not eraserON
            and not polyLineRoiON and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON and not drawClearRegionON
            and not magicPromptsON and not zoomRectON
            and not drawMergeRegionON and not drawWhitelistRegionON
        )
        canBrush = (
            brushON and not curvToolON and not rulerON
            and not dragImgLeft and not eraserON and not wandON
            and not labelRoiON and not manualBackgroundON
            and addPointsByClickingButton is None and not drawClearRegionON
            and not magicPromptsON and not zoomRectON
            and not drawMergeRegionON and not drawWhitelistRegionON
        )
        canErase = (
            eraserON and not curvToolON and not rulerON
            and not dragImgLeft and not brushON and not wandON
            and not polyLineRoiON and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON and not drawClearRegionON
            and not magicPromptsON and not zoomRectON
            and not drawMergeRegionON and not drawWhitelistRegionON
        )
        canRuler = (
            rulerON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not wandON
            and not polyLineRoiON and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON and not drawClearRegionON
            and not magicPromptsON and not zoomRectON
            and not drawMergeRegionON and not drawWhitelistRegionON
        )
        canWand = (
            wandON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON
            and not polyLineRoiON and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON and not drawClearRegionON
            and not magicPromptsON and not zoomRectON
            and not drawMergeRegionON and not drawWhitelistRegionON
        )
        canPolyLine = (
            polyLineRoiON and not wandON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON
            and not labelRoiON and not manualBackgroundON
            and addPointsByClickingButton is None
            and not drawClearRegionON and not magicPromptsON
            and not zoomRectON and not drawMergeRegionON
            and not drawWhitelistRegionON
        )
        canLabelRoi = (
            labelRoiON and not wandON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON
            and not polyLineRoiON and not keepObjON
            and addPointsByClickingButton is None
            and not manualBackgroundON and not drawClearRegionON
            and not whitelistIDsON and not magicPromptsON
            and not zoomRectON and not drawMergeRegionON
            and not drawWhitelistRegionON
        )
        canKeep = (
            keepObjON and not wandON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON
            and not polyLineRoiON and not labelRoiON 
            and addPointsByClickingButton is None
            and not manualBackgroundON and not drawClearRegionON
            and not whitelistIDsON and not magicPromptsON
            and not zoomRectON and not drawMergeRegionON
            and not drawWhitelistRegionON
        )
        canWhitelistIDs = (
            addToWhitelistByClick and not wandON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON
            and not polyLineRoiON and not labelRoiON 
            and addPointsByClickingButton is None
            and not manualBackgroundON and not drawClearRegionON
            and not keepObjON and not magicPromptsON
            and not zoomRectON and not drawMergeRegionON
            and not drawWhitelistRegionON
        )
        canDrawWhitelistIDsRoi = (
            drawWhitelistRegionON and not wandON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON
            and not polyLineRoiON and not labelRoiON 
            and addPointsByClickingButton is None
            and not manualBackgroundON and not drawClearRegionON
            and not keepObjON and not magicPromptsON
            and not zoomRectON and not drawMergeRegionON
            and not addToWhitelistByClick
        )
        canAddPoint = (
            (pointsLayerON or magicPromptsON)
            and addPointsByClickingButton is not None and not wandON 
            and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON
            and not polyLineRoiON and not labelRoiON  and not keepObjON
            and not manualBackgroundON and not drawClearRegionON
            and not zoomRectON and not drawMergeRegionON
            and not drawWhitelistRegionON
        )
        canAddManualBackgroundObj = (
            manualBackgroundON and not wandON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON
            and not polyLineRoiON and not labelRoiON 
            and addPointsByClickingButton is None
            and not keepObjON and not drawClearRegionON
            and not magicPromptsON and not whitelistIDsON
            and not zoomRectON and not drawMergeRegionON
            and not drawWhitelistRegionON
        )
        canDrawClearRegion = (
            drawClearRegionON and not wandON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON
            and not labelRoiON and not manualBackgroundON
            and addPointsByClickingButton is None
            and not polyLineRoiON and not magicPromptsON
            and not whitelistIDsON and not zoomRectON
            and not drawMergeRegionON and not drawWhitelistRegionON
        )
        canDrawMergeRegion = (
            drawMergeRegionON and not wandON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON
            and not labelRoiON and not manualBackgroundON
            and addPointsByClickingButton is None
            and not polyLineRoiON and not magicPromptsON
            and not whitelistIDsON and not zoomRectON
            and not drawClearRegionON and not drawWhitelistRegionON
        )
        
        canZoomRect = (
            zoomRectON and not curvToolON and not brushON
            and not dragImgLeft and not brushON and not rulerON
            and not polyLineRoiON and not labelRoiON
            and addPointsByClickingButton is None
            and not manualBackgroundON and not drawClearRegionON
            and not wandON and not whitelistIDsON and not magicPromptsON
            and not drawMergeRegionON and not drawWhitelistRegionON
        )
        
        # Enable dragging of the image window or the scalebar
        if dragImgLeft and not isCustomAnnot:
            x, y = event.pos().x(), event.pos().y()
            if hasattr(self, 'scaleBar'):
                if self.scaleBar.isHighlighted():
                    self.scaleBar.mousePressed(x, y)
                    return
            if hasattr(self, 'timestamp'):
                if self.timestamp.isHighlighted():
                    self.timestamp.mousePressed(x, y)
                    return
            pg.ImageItem.mousePressEvent(self.img1, event)
            event.ignore()
            return

        isAllowedActionViewer = (
            canAddPoint 
            or canRuler
            or click_shortcut_action == "Previous"
            or click_shortcut_action == "Next"
        )
        if mode == 'Viewer' and not isAllowedActionViewer:
            self.startBlinkingModeCB()
            event.ignore()
            return
        
        # Allow right-click or middle-click actions on both images
        eventOnImg2 = (
            (
                right_click or (middle_click and not canAddPoint) 
                # or (left_click and separateON)
            )
            and (mode=='Segmentation and Tracking' or self.isSnapshot)
            and not isAnnotateDivision and not manualBackgroundON
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
            lab_2D = self.get_2Dlab(posData.lab, force_z=False)
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
                self.updateLookuptable(lenNewLut=posData.brushID+1)

            self.brushColor = self.lut[posData.brushID]/255

            self.yPressAx2, self.xPressAx2 = y, x

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)
            diskSlice = (slice(ymin, ymax), slice(xmin, xmax))

            self.isMouseDragImg1 = True

            # Draw new objects
            localLab = lab_2D[diskSlice]
            mask = diskMask.copy()
            if not self.isPowerBrush() and not ctrl:
                mask[localLab!=0] = False

            self.applyBrushMask(mask, posData.brushID, toLocalSlice=diskSlice)

            self.setImageImg2(updateLookuptable=False)

            lab2D = self.get_2Dlab(posData.lab, force_z=False)
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
            lab_2D = self.get_2Dlab(posData.lab, force_z=False)
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
                color != self.doublePressKeyButtonColor
                and self.erasedID != 0
            )

            self.eraseOnlyOneID = eraseOnlyOneID

            if eraseOnlyOneID:
                mask[lab_2D!=self.erasedID] = False

            self.setTempImg1Eraser(mask, init=True)
            self.applyEraserMask(mask)

            self.erasedIDs.update(lab_2D[mask])  

            for erasedID in self.erasedIDs:
                if erasedID == 0:
                    continue
                self.erasedLab[lab_2D==erasedID] = erasedID
            
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
                    addPointsByClickingButton.pointIdSpinbox.removedId = (
                        removed_id
                    )
                else:
                    self.restorePrevPointIdRightClick(addPointsByClickingButton)
                self.drawPointsLayers(computePointsLayers=False)
            else:
                point_id = self.getAddedPointId(
                    magicPromptsON, addPointsByClickingButton, 
                    right_click, left_click, middle_click
                )
                if point_id is None:
                    return
                
                self.addClickedPoint(action, x, y, point_id)
                self.drawPointsLayers(computePointsLayers=False)
                
                point_id = self.getClickedPointNewId(
                    action, point_id, 
                    addPointsByClickingButton.pointIdSpinbox,
                    isMagicPrompts=magicPromptsON
                )
                addPointsByClickingButton.pointIdSpinbox.setValue(
                    point_id, setLinkedWidget=False
                )
        
        elif left_click and canDrawClearRegion:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.freeRoiItem.addPoint(xdata, ydata)
            
            self.isMouseDragImg1 = True
        
        elif (left_click or right_click) and canDrawMergeRegion:
            self.freeRoiItem.clear()

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.freeRoiItem.addPoint(xdata, ydata)
            
            self.isMouseDragImg1 = True
        
        elif left_click and canRuler or canPolyLine:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            closePolyLine = (
                len(self.startPointPolyLineItem.pointsAt(event.pos())) > 0
            )
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
                    self.applyDelROI(self.polyLineRoi)
                    QTimer.singleShot(
                        300, partial(self.updateDelROIinFutureFrames, self.polyLineRoi)
                    )

                    # reset polyline roi
                    self.polyLineRoi = None
        
        elif left_click and canKeep:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                keepID_win = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                        'Enter ID that you want to keep',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
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
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                keepID_win = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                        'Enter ID that you want to select',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                keepID_win.exec_()
                if keepID_win.cancel:
                    return
                else:
                    ID = keepID_win.EntryID
            
            posData = self.data[self.pos_i]
            
            current_whitelist = (
                self.whitelistIDsToolbar.whitelistLineEdit.values()
            )

            if ID in current_whitelist:
                current_whitelist.remove(ID)
                self.removeHighlightLabelID(IDs=[ID])
            else:
                current_whitelist.add(ID)
                self.highlightLabelID(ID)
            
            self.whitelistIDsToolbar.whitelistLineEdit.setText(
                current_whitelist
            )

            self.whitelistUpdateTempLayer()

        elif (right_click or left_click) and canDrawWhitelistIDsRoi:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.freeRoiItem.addPoint(xdata, ydata)
            
            self.isMouseDragImg1 = True
        
        elif right_click and copyContourON:
            hoverLostID = self.ax1_lostObjScatterItem.hoverLostID
            self.copyLostObjectMask(hoverLostID)
            self.update_rp(use_curr_view=True) # only visible
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
            if len(xxA)>0:
                if len(xxA) == 1:
                    self.splineHoverON = True
                x0, y0 = xxA[0], yyA[0]
                if len(clickedAnchors)>0:
                    xA_clicked, yA_clicked = clickedAnchors[0].pos()
                    if x0==xA_clicked and y0==yA_clicked:
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
                self.update_rp()  # dont think I can optimize this
                if self.autoIDcheckbox.isChecked():
                    self.trackManuallyAddedObject(posData.brushID, True)
                if self.isSnapshot:
                    self.fixCcaDfAfterEdit('Add new ID with curvature tool')
                    self.updateAllImages()
                else:
                    self.warnEditingWithCca_df('Add new ID with curvature tool')
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
                self.updateLookuptable(
                    lenNewLut=posData.brushID+1
                )
                self.isNewID = True
            self.brushColor = self.img2.lut[posData.brushID]/255

            # NOTE: flood is on mousedrag or release
            tol = self.getMagicWandFloodTolerance()
            self.initFloodMaskImage()
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
                xdata, ydata, text=f'that you want to assign to {manualTrackID}'
            )
            if clickedID is None:
                return

            if clickedID == manualTrackID:
                self.manualTrackingToolbar.showWarning(
                    f'The clicked object already has ID = {manualTrackID}'
                )
                return

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

            posData = self.data[self.pos_i]
            currentIDs = posData.IDs.copy()
            if manualTrackID in currentIDs:
                clicked_obj = posData.rp.get_obj_from_ID(clickedID)
                manual_track_obj = posData.rp.get_obj_from_ID(manualTrackID)
                posData.lab[clicked_obj.slice][clicked_obj.image] = manualTrackID
                posData.lab[manual_track_obj.slice][manual_track_obj.image] = (
                    clickedID)
                self.manualTrackingToolbar.showWarning(
                    f'The ID {manualTrackID} already exists --> '
                    f'ID {manualTrackID} has been swapped with {clickedID}'
                )
                assignments = {clickedID: manualTrackID,
                               manualTrackID: clickedID}
            else:
                clicked_obj = posData.rp.get_obj_from_ID(clickedID)
                posData.lab[clicked_obj.slice][clicked_obj.image] = manualTrackID
                self.manualTrackingToolbar.showInfo(
                    f'ID {clickedID} changed to {manualTrackID}.'
                )
                assignments = {clickedID: manualTrackID}
            
            # only ID change, so use assignments
            # not 3D ready yet? Otherwise I must set assignments to None
            self.update_rp(assignments=assignments)
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
            if getattr(self, 'manualBackgroundObj', None) is None:
                blinker = qutils.QControlBlink(
                    self.manualBackgroundToolbar.infoLabel, 
                    qparent=self
                )
                blinker.start()
                return

            x, y = event.pos().x(), event.pos().y()
            
            self.addManualBackgroundObject(x, y)            
            self.setManualBackgroundImage()
            self.setManualBackgrounNextID()

        # Label ROI mouse press
        elif (left_click or right_click) and canLabelRoi:
            if right_click:
                # Force model initialization on mouse release
                previousLabelRoiModel = self.labelRoiModel
                self.labelRoiModel = None
                # Otherwise released when replaced in `self.models`
                if not any(previousLabelRoiModel is m for m in self.models):
                    self.releaseSegmModel(
                        getattr(previousLabelRoiModel, 'model_name', None), 
                        previousLabelRoiModel
                    )
            
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
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                divID_prompt = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as divided',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                divID_prompt.exec_()
                if divID_prompt.cancel:
                    return
                else:
                    ID = divID_prompt.EntryID
                    centroid = posData.rp.get_centroid(ID)
                    ydata, xdata = self.getObjCentroid(centroid)
                    ydata, xdata = int(ydata), int(xdata)

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
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                budID_prompt = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID of a bud you want to correct mother assignment',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                budID_prompt.exec_()
                if budID_prompt.cancel:
                    return
                else:
                    ID = budID_prompt.EntryID

            obj_idx = posData.IDs.index(ID)
            centroid = posData.rp.get_centroid(ID)
            ydata, xdata = self.getObjCentroid(centroid)
            ydata, xdata = int(ydata), int(xdata)

            relationship = posData.cca_df.at[ID, 'relationship']
            is_history_known = posData.cca_df.at[ID, 'is_history_known']
            self.clickedOnHistoryKnown = is_history_known
            # We allow assiging a cell in G1 as bud only on first frame
            # OR if the history is unknown
            if relationship != 'bud' and posData.frame_i > 0 and is_history_known:
                txt = (f'You clicked on ID {ID} which is NOT a bud.\n'
                       'To assign a bud to a cell start by clicking on a bud '
                       'and release on a cell in G1')
                msg = QMessageBox()
                msg.critical(
                    self, 'Not a bud', txt, msg.Ok
                )
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
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                unknownID_prompt = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as '
                         '"history UNKNOWN/KNOWN"',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                unknownID_prompt.exec_()
                if unknownID_prompt.cancel:
                    return
                else:
                    ID = unknownID_prompt.EntryID
                    centroid = posData.rp.get_centroid(ID)
                    ydata, xdata = self.getObjCentroid(centroid)
                    ydata, xdata = int(ydata), int(xdata)

            self.annotateIsHistoryKnown(ID)
            if not self.setIsHistoryKnownButton.findChild(QAction).isChecked():
                self.setIsHistoryKnownButton.setChecked(False)

        elif isCustomAnnot:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                clickedBkgrDialog = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as divided',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                clickedBkgrDialog.exec_()
                if clickedBkgrDialog.cancel:
                    return
                else:
                    ID = clickedBkgrDialog.EntryID
                    centroid = posData.rp.get_centroid(ID)
                    ydata, xdata = self.getObjCentroid(centroid)
                    ydata, xdata = int(ydata), int(xdata)

            button = self.doCustomAnnotation(ID)
            if button is None:
                return
            
            keepActive = self.customAnnotDict[button]['state']['keepActive']
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
                    self.ax1.setRange(
                        xRange=xRange, 
                        yRange=yRange, 
                        padding=0
                    )
                except Exception as err:
                    QTimer.singleShot(100, self.autoRange)

    def getClickedID(self, xdata, ydata, text='', action='clicked'):
        posData = self.data[self.pos_i]
        ID = self.get_2Dlab(posData.lab)[ydata, xdata]
        if ID == 0:
            msg = (
                f'You {action} on the background.\n'
                f'Enter here the ID {text}'
            )
            nearest_ID = core.nearest_nonzero_2D(
                self.get_2Dlab(posData.lab,force_z=False), ydata, xdata
            )
            clickedBkgrID = dialogs_generic.QLineEditDialog(
                title=f'{action.capitalize()} on background',
                msg=msg, parent=self, allowedValues=posData.IDs,
                defaultTxt=str(nearest_ID),
                isInteger=True
            )
            clickedBkgrID.exec_()
            if clickedBkgrID.cancel:
                return
            else:
                ID = clickedBkgrID.EntryID
        return ID

    def doubleRightClickTimerCallBack(self):
        if self.isDoubleRightClick:
            self.doubleRightClickTimeElapsed = False
            return
        self.doubleRightClickTimeElapsed = True
        self.countRightClicks = 0
        
        # Time to double right click on img1 expired --> single right-click
        self.gui_imgGradShowContextMenu(*self._img1_click_xy)        
