"""Qt view adapter for canvas selection interactions."""

from __future__ import annotations

import time

import pyqtgraph as pg
import scipy.ndimage
import skimage.morphology

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QAction, QGraphicsSceneMouseEvent

from cellacdc import apps, exception_handler
from cellacdc.viewmodels.canvas_selection_viewmodel import (
    CanvasSelectionViewModel,
)


class CanvasSelectionView:
    """Qt-facing adapter for canvas selection workflows."""

    LEGACY_METHODS = (
        'gui_mousePressEventImg2',
        'gui_mouseReleaseEventImg2',
    )

    def __init__(self, host, view_model: CanvasSelectionViewModel):
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

    @exception_handler
    def gui_mousePressEventImg2(self, event: QGraphicsSceneMouseEvent):
        if self._dispatch_tool_event_if_enabled(event, phase='press', image='img2'):
            return
        modifiers = QGuiApplication.keyboardModifiers()
        alt = modifiers == Qt.AltModifier
        shift = modifiers == Qt.ShiftModifier
        shift_regardless = bool(modifiers & Qt.ShiftModifier)
        isMod = alt
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        left_click = event.button() == Qt.MouseButton.LeftButton and not alt
        middle_click = self.isMiddleClick(event, modifiers)
        right_click = event.button() == Qt.MouseButton.RightButton and not alt
        isPanImageClick = self.isPanImageClick(event, modifiers)
        eraserON = self.eraserButton.isChecked()
        brushON = self.brushButton.isChecked()
        separateON = self.separateBudButton.isChecked()
        self.typingEditID = False

        # Drag image if neither brush or eraser are On pressed
        dragImg = self.view_model.should_drag_image(
            left_click=left_click,
            eraser_on=eraserON,
            brush_on=brushON,
            middle_click=middle_click,
            pan_click=isPanImageClick,
        )

        # Enable dragging of the image window like pyqtgraph original code
        if dragImg:
            pg.ImageItem.mousePressEvent(self.img2, event)
            event.ignore()
            return

        if self.view_model.should_blink_viewer_mode(
            mode=mode,
            middle_click=middle_click,
        ):
            self.mode_controls_view.startBlinkingModeCB()
            event.ignore()
            return

        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        Y, X = self.get_2Dlab(posData.lab).shape
        if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
        else:
            return

        # Check if right click on ROI
        isClickOnDelRoi = self.canvas_context_menu_view.clicked_deleted_roi(
            event,
            left_click,
            right_click,
        )
        if isClickOnDelRoi:
            return

        # show gradient widget menu if none of the right-click actions are ON
        # and event is not coming from image 1
        is_right_click_action_ON = any([
            b.isChecked() for b in self.checkableQButtonsGroup.buttons()
        ])
        is_right_click_custom_ON = any([
            b.isChecked() for b in self.customAnnotDict.keys()
        ])
        is_event_from_img1 = False
        if hasattr(event, 'isImg1Sender'):
            is_event_from_img1 = event.isImg1Sender

        is_only_right_click = (
            right_click and not is_right_click_action_ON and not middle_click
        )

        showLabelsGradMenu = self.view_model.should_show_labels_menu(
            right_click=right_click,
            right_action_on=is_right_click_action_ON,
            middle_click=middle_click,
            event_from_img1=is_event_from_img1,
        )

        if showLabelsGradMenu:
            self.labelsGrad.showMenu(event)
            event.ignore()
            return

        editInViewerMode = self.view_model.should_blink_viewer_mode(
            mode=mode,
            middle_click=middle_click,
            right_action_on=is_right_click_action_ON,
            custom_action_on=is_right_click_custom_ON,
            right_click=right_click,
        )

        if editInViewerMode:
            self.mode_controls_view.startBlinkingModeCB()
            event.ignore()
            return

        # Left-click is used for brush, eraser, separate bud, curvature tool
        # and magic labeller
        # Brush and eraser are mutually exclusive but we want to keep the eraser
        # or brush ON and disable them temporarily to allow left-click with
        # separate ON
        canDelete = self.view_model.can_delete(
            mode=mode,
            is_snapshot=self.isSnapshot,
        )

        # Delete ID (set to 0)
        if middle_click and canDelete:
            t0 = time.perf_counter()
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            delID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if delID == 0:
                nearest_ID = self.view_model.nearest_nonzero_2d(
                    self.get_2Dlab(posData.lab), y, x
                )
                delID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.<br>'
                        'Enter here ID(s) that you want to delete<br><br>'
                        'You can enter multiple IDs separated by comma',
                    parent=self.host,
                    allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    allowList=True,
                    isInteger=True
                )
                delID_prompt.exec_()
                if delID_prompt.cancel:
                    return
                delIDs = delID_prompt.EntryID
            else:
                delIDs = [delID]

            # Ask to propagate change to all future visited frames
            key = 'Delete ID'
            askAction = self.askHowFutureFramesActions[key]
            doNotShow = not askAction.isChecked()
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                delIDs, key, doNotShow,
                posData.UndoFutFrames_DelID, posData.applyFutFrames_DelID
            )

            if UndoFutFrames is None:
                return

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)
            posData.doNotShowAgain_DelID = doNotShowAgain
            posData.UndoFutFrames_DelID = UndoFutFrames
            posData.applyFutFrames_DelID = applyFutFrames
            includeUnvisited = posData.includeUnvisitedInfo['Delete ID']

            delID_mask = self.deleteIDmiddleClick(
                delIDs, applyFutFrames, includeUnvisited, shift=shift_regardless
            )
            if delID_mask.ndim == 3:
                delID_mask = delID_mask[self.z_lab()]

            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Delete ID')
            else:
                self.warnEditingWithCca_df('Delete ID', update_images=False)

            self.setImageImg2()
            delROIsIDs = self.setAllTextAnnotations()
            self.setAllContoursImages(delROIsIDs=delROIsIDs, compute=False)

            how = self.drawIDsContComboBox.currentText()
            if how.find('overlay segm. masks') != -1:
                self.labelsLayerImg1.image[delID_mask] = 0
                self.labelsLayerImg1.setImage(self.labelsLayerImg1.image)

            how_ax2 = self.getAnnotateHowRightImage()
            if how_ax2.find('overlay segm. masks') != -1:
                self.labelsLayerRightImg.image[delID_mask] = 0
                self.labelsLayerRightImg.setImage(self.labelsLayerRightImg.image)

            self.highlightLostNew()

        # Separate bud or objects with same ID
        elif right_click and separateON:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.view_model.nearest_nonzero_2d(
                    self.get_2Dlab(posData.lab), y, x)
                sepID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to split',
                    parent=self.host, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                sepID_prompt.exec_()
                if sepID_prompt.cancel:
                    return
                else:
                    ID = sepID_prompt.EntryID
                y, x = posData.rp[posData.IDs_idxs[ID]].centroid[-2:]
                xdata, ydata = int(x), int(y)

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            max_ID = max(posData.IDs, default=1)

            if self.isSegm3D and not shift:
                z = self.zSliceScrollBar.sliderPosition()
                posData.lab, splittedIDs = (
                    self.view_model.separate_with_label(
                        posData.lab, posData.rp, [ID], max_ID,
                        click_coords_list=[(z, ydata, xdata)]
                    )
                )
                success = True
                # self.set_2Dlab(lab2D)
            elif not shift:
                result = self.view_model.split_along_convexity_defects(
                    ID, self.get_2Dlab(posData.lab), max_ID
                )
                lab2D, success, splittedIDs = result
                self.set_2Dlab(lab2D)
            else:
                success = False

            # If automatic bud separation was not successfull call manual one
            if not success:
                posData.disableAutoActivateViewerWindow = True
                img = self.getDisplayedImg1()
                col = 'manual_separate_draw_mode'
                drawMode = self.df_settings.at[col, 'value']
                manualSep = apps.manualSeparateGui(
                    self.get_2Dlab(posData.lab), ID, img,
                    fontSize=self.fontSize,
                    IDcolor=self.lut[ID],
                    parent=self.host,
                    drawMode=drawMode
                )
                manualSep.setState(self.lastManualSeparateState)
                manualSep.show()
                manualSep.centerWindow()
                manualSep.show(block=True)
                if manualSep.cancel:
                    posData.disableAutoActivateViewerWindow = False
                    if not self.separateBudButton.findChild(QAction).isChecked():
                        self.separateBudButton.setChecked(False)
                    return
                self.lastManualSeparateState = manualSep.state()
                lab2D = self.get_2Dlab(posData.lab)
                lab2D[manualSep.lab!=0] = manualSep.lab[manualSep.lab!=0]
                self.set_2Dlab(lab2D)
                splittedIDs = [obj.label for obj in manualSep.rp]
                posData.disableAutoActivateViewerWindow = False
                self.canvas_tool_view.store_manual_separate_draw_mode(
                    self.df_settings,
                    self.settings_csv_path,
                    manualSep.drawMode,
                )

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.trackSubsetIDs(splittedIDs)

            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Separate IDs')
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df('Separate IDs')

            self.store_data()

            if not self.separateBudButton.findChild(QAction).isChecked():
                self.separateBudButton.setChecked(False)

        # Fill holes
        elif right_click and self.fillHolesToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.view_model.nearest_nonzero_2d(
                    self.get_2Dlab(posData.lab), y, x
                )
                clickedBkgrID = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here the ID that you want to '
                         'fill the holes of',
                    parent=self.host, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                clickedBkgrID.exec_()
                if clickedBkgrID.cancel:
                    return
                else:
                    ID = clickedBkgrID.EntryID

            if ID in posData.lab:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                obj_idx = posData.IDs.index(ID)
                obj = posData.rp[obj_idx]
                objMask = self.getObjImage(obj.image, obj.bbox)
                localFill = scipy.ndimage.binary_fill_holes(objMask)
                posData.lab[self.getObjSlice(obj.slice)][localFill] = ID

                self.update_rp()
                self.updateAllImages()

                if not self.fillHolesToolButton.findChild(QAction).isChecked():
                    self.fillHolesToolButton.setChecked(False)

        # Hull contour
        elif right_click and self.hullContToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.view_model.nearest_nonzero_2d(
                    self.get_2Dlab(posData.lab), y, x
                )
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here the ID that you want to '
                         'replace with Hull contour',
                    parent=self.host, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                mergeID_prompt.exec_()
                if mergeID_prompt.cancel:
                    return
                else:
                    ID = mergeID_prompt.EntryID

            if ID in posData.lab:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                obj_idx = posData.IDs.index(ID)
                obj = posData.rp[obj_idx]
                objMask = self.getObjImage(obj.image, obj.bbox)
                localHull = skimage.morphology.convex_hull_image(objMask)
                posData.lab[self.getObjSlice(obj.slice)][localHull] = ID

                self.update_rp()
                self.updateAllImages()

                if not self.hullContToolButton.findChild(QAction).isChecked():
                    self.hullContToolButton.setChecked(False)

        # Move label
        elif right_click and self.moveLabelToolButton.isChecked():
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

            x, y = event.pos().x(), event.pos().y()
            self.label_transform_tools_view.start_moving_label(x, y)

        # Fill holes
        elif right_click and self.fillHolesToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.view_model.nearest_nonzero_2d(
                    self.get_2Dlab(posData.lab), y, x
                )
                clickedBkgrID = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here the ID that you want to '
                         'fill the holes of',
                    parent=self.host, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                clickedBkgrID.exec_()
                if clickedBkgrID.cancel:
                    return
                else:
                    ID = clickedBkgrID.EntryID

        # Merge IDs
        elif right_click and self.mergeIDsButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.view_model.nearest_nonzero_2d(
                    self.get_2Dlab(posData.lab), y, x
                )
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here first ID that you want to merge',
                    parent=self.host, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                mergeID_prompt.exec_()
                if mergeID_prompt.cancel:
                    self.mergeObjsTempLine.setData([], [])
                    return
                else:
                    ID = mergeID_prompt.EntryID

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            self.firstID = ID

            obj_idx = posData.IDs_idxs[ID]
            obj = posData.rp[obj_idx]
            yc, xc = self.getObjCentroid(obj.centroid)
            self.clickObjYc, self.clickObjXc = int(yc), int(xc)

        # Edit ID
        elif right_click and self.editIDbutton.isChecked():
            if self._dispatch_tool_event_if_enabled(event, phase='press', image='img2'):
                return
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.view_model.nearest_nonzero_2d(
                    self.get_2Dlab(posData.lab), y, x
                )
                editID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                        'Enter here ID that you want to replace with a new one',
                    parent=self.host, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                editID_prompt.show(block=True)

                if editID_prompt.cancel:
                    return
                else:
                    ID = editID_prompt.EntryID

            obj_idx = posData.IDs_idxs[ID]
            y, x = posData.rp[obj_idx].centroid[-2:]
            xdata, ydata = int(x), int(y)

            posData.disableAutoActivateViewerWindow = True
            currentIDs = posData.IDs.copy()
            self.setAllIDs(onlyVisited=True)
            addPropagateCheckbox = (
                not self.isSnapshot
                and posData.frame_i == self.navigateScrollBar.maximum() - 1
                and posData.frame_i < posData.SizeT - 1
            )
            editID = apps.EditIDDialog(
                ID, posData.IDs,
                doNotShowAgain=self.doNotAskAgainExistingID,
                parent=self.host,
                entryID=self.getNearestLostObjID(y, x),
                nextUniqueID=self.setBrushID(return_val=True),
                allIDs=posData.allIDs,
                addPropagateCheckbox=addPropagateCheckbox
            )
            editID.show(block=True)
            if editID.cancel:
                posData.disableAutoActivateViewerWindow = False
                if not self.editIDbutton.findChild(QAction).isChecked():
                    self.editIDbutton.setChecked(False)
                return

            if editID.assignNewID:
                self.assignNewIDfromClickedID(ID, event)
                return

            if not self.doNotAskAgainExistingID:
                self.editIDmergeIDs = editID.mergeWithExistingID
            self.doNotAskAgainExistingID = editID.doNotAskAgainExistingID

            self.applyEditID(
                ID, currentIDs, editID.how, x, y,
                shift=shift,
                doPropagateUnvisited=editID.doPropagateFutureFrames
            )

        elif (right_click or left_click) and self.keepIDsButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.view_model.nearest_nonzero_2d(
                    self.get_2Dlab(posData.lab), y, x
                )
                keepID_win = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                        'Enter ID that you want to keep',
                    parent=self.host, allowedValues=posData.IDs,
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

        # Annotate cell as removed from the analysis
        elif right_click and self.binCellButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.view_model.nearest_nonzero_2d(
                    self.get_2Dlab(posData.lab), y, x
                )
                binID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to remove from the analysis',
                    parent=self.host, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                binID_prompt.exec_()
                if binID_prompt.cancel:
                    return
                else:
                    ID = binID_prompt.EntryID

            # Ask to propagate change to all future visited frames
            key = 'Exclude cell from analysis'
            askAction = self.askHowFutureFramesActions[key]
            doNotShow = not askAction.isChecked()
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                ID, key, doNotShow,
                posData.UndoFutFrames_BinID,
                posData.applyFutFrames_BinID
            )

            if UndoFutFrames is None:
                # User cancelled the process
                return

            posData.doNotShowAgain_BinID = doNotShowAgain
            posData.UndoFutFrames_BinID = UndoFutFrames
            posData.applyFutFrames_BinID = applyFutFrames

            self.current_frame_i = posData.frame_i

            # Apply Exclude cell from analysis to future frames if requested
            if applyFutFrames:
                # Store current data before going to future frames
                self.store_data()
                for i in range(posData.frame_i+1, endFrame_i+1):
                    posData.frame_i = i
                    self.get_data()
                    if ID in posData.binnedIDs:
                        posData.binnedIDs.remove(ID)
                    else:
                        posData.binnedIDs.add(ID)
                    self.update_rp_metadata(draw=False)
                    self.store_data(autosave=i==endFrame_i)

                self.app.restoreOverrideCursor()

            # Back to current frame
            if applyFutFrames:
                posData.frame_i = self.current_frame_i
                self.get_data()

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            if ID in posData.binnedIDs:
                posData.binnedIDs.remove(ID)
            else:
                posData.binnedIDs.add(ID)

            self.annotate_rip_and_bin_IDs(updateLabel=True)

            # Gray out ore restore binned ID
            self.updateLookuptable()

            if not self.binCellButton.findChild(QAction).isChecked():
                self.binCellButton.setChecked(False)

        # Annotate cell as dead
        elif right_click and self.ripCellButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = self.view_model.nearest_nonzero_2d(
                    self.get_2Dlab(posData.lab), y, x
                )
                ripID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as dead',
                    parent=self.host, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                ripID_prompt.exec_()
                if ripID_prompt.cancel:
                    return
                else:
                    ID = ripID_prompt.EntryID

            # Ask to propagate change to all future visited frames
            key = 'Annotate cell as dead'
            askAction = self.askHowFutureFramesActions[key]
            doNotShow = not askAction.isChecked()
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                ID, key, doNotShow,
                posData.UndoFutFrames_RipID,
                posData.applyFutFrames_RipID
            )

            if UndoFutFrames is None:
                return

            posData.doNotShowAgain_RipID = doNotShowAgain
            posData.UndoFutFrames_RipID = UndoFutFrames
            posData.applyFutFrames_RipID = applyFutFrames

            self.current_frame_i = posData.frame_i

            # Apply Edit ID to future frames if requested
            if applyFutFrames:
                # Store current data before going to future frames
                self.store_data()
                for i in range(posData.frame_i+1, endFrame_i+1):
                    posData.frame_i = i
                    self.get_data()
                    if ID in posData.ripIDs:
                        posData.ripIDs.remove(ID)
                    else:
                        posData.ripIDs.add(ID)
                    self.update_rp_metadata(draw=False)
                    self.store_data(autosave=i==endFrame_i)
                self.app.restoreOverrideCursor()

            # Back to current frame
            if applyFutFrames:
                posData.frame_i = self.current_frame_i
                self.get_data()

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            if ID in posData.ripIDs:
                posData.ripIDs.remove(ID)
            else:
                posData.ripIDs.add(ID)

            self.annotate_rip_and_bin_IDs(updateLabel=True)

            # Gray out dead ID
            self.updateLookuptable()
            self.store_data()

            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Annotate ID as dead')
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df('Annotate ID as dead')

            if not self.ripCellButton.findChild(QAction).isChecked():
                self.ripCellButton.setChecked(False)

    @exception_handler
    def gui_mouseReleaseEventImg2(self, event):
        if self._dispatch_tool_event_if_enabled(event, phase='release', image='img2'):
            return
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())

        Y, X = self.get_2Dlab(posData.lab).shape
        try:
            x, y = event.pos().x(), event.pos().y()
        except Exception as e:
            return

        xdata, ydata = int(x), int(y)
        in_bounds = self.view_model.is_in_bounds(xdata, ydata, X, Y)
        if self.view_model.is_viewer_mode(mode):
            return

        should_process = self.view_model.should_process_release(
            mode=mode,
            in_bounds=in_bounds,
        )
        if not should_process:
            self.isMouseDragImg2 = False
            self.updateAllImages()
            return

        # Move label mouse released, update move
        if self.isMovingLabel and self.moveLabelToolButton.isChecked():
            self.isMovingLabel = False

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True, assign_unique_new_IDs=False)

            self.updateAllImages()

            if not self.moveLabelToolButton.findChild(QAction).isChecked():
                self.moveLabelToolButton.setChecked(False)

        # Merge IDs
        elif self.mergeIDsButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            lab2D = self.get_2Dlab(posData.lab)
            ID = lab2D[ydata, xdata]
            if ID == 0:
                nearest_ID = self.view_model.nearest_nonzero_2d(
                    lab2D, y, x
                )
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to merge with ID '
                         f'{self.firstID}',
                    parent=self.host, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                mergeID_prompt.exec_()
                if mergeID_prompt.cancel:
                    return
                else:
                    ID = mergeID_prompt.EntryID
                    obj_idx = posData.IDs_idxs[ID]
                    obj = posData.rp[obj_idx]
                    y2, x2 = self.getObjCentroid(obj.centroid)
                    self.mergeObjsTempLine.addPoint(x2, y2)

            xx, yy = self.mergeObjsTempLine.getData()
            IDs_to_merge = lab2D[yy.astype(int), xx.astype(int)]
            for ID in IDs_to_merge:
                if ID == 0:
                    continue
                posData.lab[posData.lab==ID] = self.firstID

            self.mergeObjsTempLine.setData([], [])
            self.clickObjYc, self.clickObjXc = None, None

            # Update data (rp, etc)
            self.update_rp()

            ask_back_prop = True

            if posData.frame_i == 0:
                ask_back_prop = False
                prev_IDs = []
            else:
                prev_IDs = posData.allData_li[posData.frame_i-1]['IDs']

            if  all(ID not in prev_IDs for ID in IDs_to_merge):
                ask_back_prop = False

            if not self.isFrameCcaAnnotated() and ask_back_prop:
                proceed = self.askPropagateChangePast(f'Merge IDs {IDs_to_merge}')
                if proceed:
                    self.propagateMergeObjsPast(IDs_to_merge)
                    self.whitelistPropagateIDs(only_future_frames=False, update_lab=True) # in the update_rp() call, this should also be done

            # Repeat tracking
            self.tracking(
                enforce=True, assign_unique_new_IDs=False,
                separateByLabel=False
            )

            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Merge IDs')
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df('Merge IDs')

            if not self.mergeIDsButton.findChild(QAction).isChecked():
                self.mergeIDsButton.setChecked(False)
            self.store_data()
