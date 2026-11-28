"""Mouse events of the main GUI except the left image: shared click helpers
and the right axis (img2, rightImageItem). The left image handlers are
in gui/main_window/mouse_left_image.py. Mixins of guiWin
(gui/main_window/window.py) only: the measurements viewer never composes
them.
"""
import time
import numpy as np
import scipy.optimize
import scipy.interpolate
import scipy.ndimage
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QAction, QGraphicsSceneMouseEvent
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import exception_handler
from cellacdc.shared.dialogs import generic as dialogs_generic
from ..dialogs import editing as gui_dialogs_editing
from cellacdc import core
from cellacdc.myutils import image_utils
from cellacdc import printl
from cellacdc import is_mac
from cellacdc import core_split_IDs


class MouseEventsGuiElements:
    """Mouse events except those of the left image: click helpers shared by
    both images (pan, middle and macOS right clicks), the release hook
    called by eventFilter and the press/drag/release handlers of the
    right axis (labels image img2 and rightImageItem) with its LUT
    context menu.
    """

    def onMouseRelease(self):
        if self.mergeIDsButton.isChecked() and self.xHoverImg is None:
            # Mouse released outside of images --> clear free roi item
            self.freeRoiItem.clear()

    def isPanImageClick(self, mouseEvent, modifiers):
        left_click = mouseEvent.button() == Qt.MouseButton.LeftButton
        return modifiers == Qt.AltModifier and left_click

    def middleClickText(self):
        if self.delObjAction is None and is_mac:
            return 'Command + Left Click'
        
        if self.delObjAction is None:
            return 'Middle Click'
        
        delObjKeySequence, delObjQtButton = self.delObjAction
        
        if delObjQtButton == Qt.MouseButton.LeftButton:
            buttonName = 'Left click'
        elif delObjQtButton == Qt.MouseButton.RightButton:
            buttonName = 'Right click'
        else:
            buttonName = 'Middle click'
        
        if delObjKeySequence is None:
            return buttonName
        
        return f'{delObjKeySequence.toString()} + {buttonName}'

    def isDefaultMiddleClick(self, mouseEvent, modifiers):
        if is_mac:
            middle_click = (
                mouseEvent.button() == Qt.MouseButton.LeftButton
                and modifiers == Qt.ControlModifier
                and not self.brushButton.isChecked()
            )
        else:
            middle_click = mouseEvent.button() == Qt.MouseButton.MiddleButton
        return middle_click

    def isMiddleClick(self, mouseEvent, modifiers):
        if self.delObjAction is None:
            return self.isDefaultMiddleClick(mouseEvent, modifiers)
               
        delObjKeySequence, delObjQtButton = self.delObjAction
        if delObjKeySequence is None:
            # Setting only middle click on mac is allowed, however the 
            # delObjKeySequence is None and the tool button is never checked
            isDelObjectActive = True
        else:
            isDelObjectActive = self.delObjToolAction.isChecked()
        
        mouseEventButton = self.changeRightClickToLeftOnMac(mouseEvent)
        
        middle_click = (
            mouseEventButton == delObjQtButton and isDelObjectActive
        )
        
        return middle_click

    @exception_handler
    def gui_mousePressEventImg2(self, event: QGraphicsSceneMouseEvent):
        if getattr(self, 'magicPromptsBoxItems', None):
            if self.isMovingMagicPromptsBox():
                # Another button pressed while a box prompt is moved or
                # resized --> leave it to pyqtgraph (img2 must not grab the
                # mouse, otherwise the move is never finished)
                event.ignore()
                return

        modifiers = QGuiApplication.keyboardModifiers()
        alt = modifiers == Qt.AltModifier
        shift = modifiers == Qt.ShiftModifier
        ctrl = modifiers == Qt.ControlModifier
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
        zProjHow = self.zProjComboBox.currentText()
        isZslice = zProjHow == 'single z-slice'
        self.typingEditID = False
        # Drag image if neither brush or eraser are On pressed
        dragImg = (
            left_click and not eraserON and not
            brushON and not middle_click
        )
        if isPanImageClick:
            dragImg = True

        # Enable dragging of the image window like pyqtgraph original code
        if dragImg:
            pg.ImageItem.mousePressEvent(self.img2, event)
            event.ignore()
            return

        if mode == 'Viewer' and middle_click:
            self.startBlinkingModeCB()
            event.ignore()
            return

        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        Y, X = self.get_2Dlab(posData.lab, force_z=False).shape
        if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
            ID = self.get_2Dlab(posData.lab, force_z=False)[ydata, xdata]
        else:
            return

        # Check if right click on ROI
        isClickOnDelRoi = self.gui_handleClickOnDelRoi(event, left_click, right_click)
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
        
        showLabelsGradMenu = (
            is_only_right_click and not is_event_from_img1
        )
        
        if showLabelsGradMenu:
            self.labelsGrad.showMenu(event)
            event.ignore()
            return

        editInViewerMode = (
            (is_right_click_action_ON or is_right_click_custom_ON)
            and (right_click or middle_click) and mode=='Viewer'
        )

        if editInViewerMode:
            self.startBlinkingModeCB()
            event.ignore()
            return
        
        # Left-click is used for brush, eraser, separate bud, curvature tool
        # and magic labeller
        # Brush and eraser are mutually exclusive but we want to keep the eraser
        # or brush ON and disable them temporarily to allow left-click with
        # separate ON
        canDelete = mode == 'Segmentation and Tracking' or self.isSnapshot

        # Delete ID (set to 0)
        if middle_click and canDelete:
            t0 = time.perf_counter()
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            delID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if delID == 0:
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                delID_prompt = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.<br>'
                        'Enter here ID(s) that you want to delete<br><br>'
                        'You can enter multiple IDs separated by comma',
                    parent=self, 
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

            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            
            delID_mask = self.deleteIDmiddleClick(
                delIDs, applyFutFrames, includeUnvisited, 
                modifiers=modifiers
            )
            if delID_mask.ndim == 3:
                delID_mask = delID_mask[self.z_lab()]

            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Delete ID')
            else:
                self.warnEditingWithCca_df('Delete ID', update_images=False)
            
            self.setImageImg2()
            self.setAllTextAnnotations()
            self.setAllContoursImages(compute=False)
            self.setOverlaySegmMasks()            
            self.highlightLostNew()
            
        # Separate bud or objects with same ID
        elif right_click and separateON:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                    )
                sepID_prompt = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to split',
                    parent=self, allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                sepID_prompt.exec_()
                if sepID_prompt.cancel:
                    return
                else:
                    ID = sepID_prompt.EntryID
                
                centroid = posData.rp.get_centroid(ID)
                y, x = self.getObjCentroid(centroid)
                xdata, ydata = int(x), int(y)

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            max_ID = max(posData.IDs, default=1)

            separation_mode = self.separateBudToolbar.separationMode()
            if self.splitIDwith3DActionModifiers(modifiers, ID):
                splittedIDs = [ID, posData.brushID]
                printl(splittedIDs)
                success = True
            elif (
                    self.isSegm3D and not ctrl and not (shift and isZslice)
                    and separation_mode == '3D Convexity defects'
                ):
                posData.lab, success, splittedIDs = (
                    core_split_IDs.split_along_convexity_defects_3D(
                        ID, posData.lab, max_ID, rp=posData.rp,
                        voxel_size=(
                            posData.PhysicalSizeZ,
                            posData.PhysicalSizeY,
                            posData.PhysicalSizeX,
                        ),
                        **self.separateBudToolbar.splitConv3DKwargs(),
                    )
                )
            elif (
                self.isSegm3D and not ctrl and not (shift and isZslice)
                and separation_mode == 'Slice by slice'
            ):
                result = core_split_IDs.split_along_convexity_defects_slice_by_slice(
                    ID, posData.lab, max_ID,
                    rp=posData.rp,
                    **self.separateBudToolbar.splitConv2DKwargs(),
                )
                posData.lab, success, splittedIDs = result
            elif self.isSegm3D and not ctrl and (shift and isZslice):
                rp_2D = self.get2DRP()
                lab_2D = self.get_2Dlab(force_z=True)
                result = core_split_IDs.split_along_convexity_defects(
                    ID, lab_2D, max_ID, rp=rp_2D,
                    **self.separateBudToolbar.splitConv2DKwargs(),
                )
                lab_2D, success, splittedIDs = result
                # assign largest resulting ID the original ID
                obj = rp_2D.get_obj_from_ID(ID)
                IDs, counts = np.unique(lab_2D[obj.slice][obj.image], return_counts=True)
                max_size_idx = np.argmax(counts)
                largest_ID = IDs[max_size_idx]
                original_values = lab_2D[obj.slice][obj.image].copy()
                lab_2D[obj.slice][obj.image] = np.where(
                    original_values == largest_ID,
                     ID,
                     np.where(original_values == ID, largest_ID, original_values),
                     )
                self.set_2Dlab(lab_2D)
                
            elif not ctrl and not self.isSegm3D:
                result = core_split_IDs.split_along_convexity_defects(
                    ID, self.get_2Dlab(posData.lab), max_ID, rp=posData.rp,
                    **self.separateBudToolbar.splitConv2DKwargs(),
                )
                lab2D, success, splittedIDs = result
                self.set_2Dlab(lab2D)
            else:  # ctrl+right-click triggers manual separation directly
                success = False
                
            if (
                success 
                and self.separateBudToolbar.goToSepSliceChecked()
                and self.isSegm3D
                ):
                
                ogObject = posData.rp.get_obj_from_ID(ID)
                bbox = ogObject.bbox
                min_z, min_y, min_x, max_z, max_y, max_x = bbox
                splittedIDs_exc_ID = [i for i in splittedIDs if i != ID]
                for z in range(min_z, max_z + 1):
                    lab2D = self.get_2Dlab(force_z=True, slice_i=z)
                    # check if any of the splitted IDs are present in the current 2D slice
                    if any(i in lab2D for i in splittedIDs_exc_ID):
                        # set z to the current slice index
                        self.zSliceScrollBar.setSliderPosition(z)
                        self.update_z_slice(z)
                        break
            
            # If automatic bud separation was not successfull call manual one
            if not success:
                posData.disableAutoActivateViewerWindow = True
                img = self.getDisplayedImg1()
                lastManualSeparateState = self.getLastManualSeparateState()
                if shift and isZslice and self.isSegm3D:
                    lab = self.get_2Dlab(force_z=True)
                    img = self.getDisplayedImg1()
                    rp = self.get2DRP()
                elif self.isSegm3D:
                    lab = posData.lab
                    img = self.getDisplayedZstack()
                    rp = posData.rp
                else:
                    lab = posData.lab
                    img = self.getDisplayedImg1()
                    rp = posData.rp

                if self.isSegm3D:
                    start_slice = self.zSliceScrollBar.sliderPosition()
                else:
                    start_slice = None

                draw_mode = (
                    'threepoints_arc'
                    if lastManualSeparateState['is_three_points_active']
                    else 'freehand'
                )
                labels_lut = self.getLabelsImageLut()
                labels_alpha = self.imgGrad.labelsAlphaSlider.value()
                manualSep = getattr(self, 'manualSeparateGui', None)
                img_lut_df = self.imgGrad.saveState(self.df_settings)
                if manualSep is None:
                    manualSep = gui_dialogs_editing.manualSeparateGui(
                        lab, ID, img,
                        fontSize=self.fontSize,
                        IDcolor=self.lut[ID],
                        parent=self,
                        drawMode=draw_mode,
                        start_slice=start_slice,
                        mouseBindings=self.mouseBindings,
                        labelsLut=labels_lut,
                        labelsAlpha=labels_alpha,
                        rp=rp,
                        img_lut_df=img_lut_df,
                    )
                    self.manualSeparateGui = manualSep
                else:
                    manualSep.setSessionData(
                        lab, ID, img,
                        fontSize=self.fontSize,
                        IDcolor=self.lut[ID],
                        drawMode=draw_mode,
                        start_slice=start_slice,
                        mouseBindings=self.mouseBindings,
                        labelsLut=labels_lut,
                        labelsAlpha=labels_alpha,
                        rp=rp,
                        img_lut_df=img_lut_df,
                    )

                manualSep.setState(lastManualSeparateState)

                manualSep.centerWindow()
                manualSep.show(block=True)
                if manualSep.cancel:
                    posData.disableAutoActivateViewerWindow = False
                    if not self.separateBudButton.findChild(QAction).isChecked():
                        self.separateBudButton.setChecked(False)
                    return
                self.setLastManualSeparateState(manualSep.state())
                if self.isSegm3D and manualSep.lab.ndim == 3:
                    changed_mask = manualSep.lab != 0
                    posData.lab[changed_mask] = manualSep.lab[changed_mask]
                    splittedIDs = list(np.unique(manualSep.lab[changed_mask]))
                elif self.isSegm3D and manualSep.lab.ndim == 2:
                    changed_mask = manualSep.lab != 0
                    lab_2D = self.get_2Dlab(force_z=True)
                    lab_2D[changed_mask] = manualSep.lab[changed_mask]
                    self.set_2Dlab(lab_2D)
                    splittedIDs = list(np.unique(manualSep.lab[changed_mask]))
                else:
                    lab2D = self.get_2Dlab(posData.lab)
                    lab2D[manualSep.lab!=0] = manualSep.lab[manualSep.lab!=0]
                    self.set_2Dlab(lab2D)
                    splittedIDs = [obj.label for obj in manualSep.rp]
                posData.disableAutoActivateViewerWindow = False

            # Update data (rp, etc)
            bbox = self.update_rp_get_bbox(use_bbox=True, specific_IDs=ID) # use old ID to get bbox
            specific_IDs = list(splittedIDs) + [ID]
            self.update_rp(specific_IDs=specific_IDs, preloaded_bbox=bbox)

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
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                clickedBkgrID = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here the ID that you want to '
                         'fill the holes of',
                    parent=self, allowedValues=posData.IDs,
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
                if (shift and isZslice) and self.isSegm3D:
                    rp2D = self.rpCurr2D()
                    obj = rp2D.get_obj_from_ID(ID)
                else: # shift hold or 2D from the getgo
                    obj = posData.rp.get_obj_from_ID(ID)

                localFill = scipy.ndimage.binary_fill_holes(obj.image)

                if (shift and isZslice) and self.isSegm3D:
                    curr_z = self.zSliceScrollBar.sliderPosition()
                    posData.lab[curr_z][obj.slice][localFill] = ID
                else:
                    posData.lab[obj.slice][localFill] = ID

                # here it is impossible that hole filling overwrites an ID which
                # otuches border
                
                self.update_rp(use_bbox=True, specific_IDs=ID)
                self.updateAllImages()

                if not self.fillHolesToolButton.findChild(QAction).isChecked():
                    self.fillHolesToolButton.setChecked(False)
        
        # Hull contour
        elif right_click and self.hullContToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                mergeID_prompt = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here the ID that you want to '
                         'replace with Hull contour',
                    parent=self, allowedValues=posData.IDs,
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
                if (shift and isZslice) and self.isSegm3D:
                    rp2D = self.rpCurr2D()
                    obj = rp2D.get_obj_from_ID(ID)
                else:
                    obj = posData.rp.get_obj_from_ID(ID)

                localHull = skimage.morphology.convex_hull_image(obj.image)
                if (shift and isZslice) and self.isSegm3D:
                    curr_z = self.zSliceScrollBar.sliderPosition()
                    hull_lab = posData.lab[curr_z][obj.slice]
                else:
                    hull_lab = posData.lab[obj.slice]

                IDs_overwritten = np.unique(hull_lab[localHull])
                IDs_overwritten = IDs_overwritten[IDs_overwritten != 0]
                hull_lab[localHull] = ID

                self.update_rp(use_bbox=True, specific_IDs=IDs_overwritten)
                self.updateAllImages()

                if not self.hullContToolButton.findChild(QAction).isChecked():
                    self.hullContToolButton.setChecked(False)

        # Move label
        elif right_click and self.moveLabelToolButton.isChecked():
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

            x, y = event.pos().x(), event.pos().y()
            self.startMovingLabel(x, y)

        # Edit ID
        elif right_click and self.editIDbutton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                editID_prompt = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                        'Enter here ID that you want to replace with a new one',
                    parent=self, 
                    allowedValues=posData.IDs,
                    defaultTxt=str(nearest_ID),
                    isInteger=True
                )
                editID_prompt.show(block=True)

                if editID_prompt.cancel:
                    return
                else:
                    ID = editID_prompt.EntryID
            
            centroid = posData.rp.get_centroid(ID, exact=True)
            y, x = self.getObjCentroid(centroid)
            xdata, ydata = int(x), int(y)

            posData.disableAutoActivateViewerWindow = True
            currentIDs = posData.IDs.copy()
            self.setAllIDs(onlyVisited=True)
            addPropagateCheckbox = (
                not self.isSnapshot
                and posData.frame_i == self.navigateScrollBar.maximum() - 1
                and posData.frame_i < posData.SizeT - 1
            )
            editID = gui_dialogs_editing.EditIDDialog(
                ID, posData.IDs, 
                doNotShowAgain=self.doNotAskAgainExistingID,
                parent=self, 
                entryID=self.getNearestLostObjID(y, x), 
                nextUniqueID=self.setBrushID(return_val=True), 
                allIDs=posData.allIDs,
                addPropagateCheckbox=addPropagateCheckbox,
                isSegm3D=self.isSegm3D,
                SizeZ=posData.SizeZ,
                currentZslice=self.zSliceScrollBar.sliderPosition(),
                zSliceStartValue=self.editIDToolbar.getStartValue()
            )
            if self.isSegm3D:
                editID.setHowToPropagateZslices(
                    *self.editIDToolbar.getHowToPropagateZslices()
                )

            editID.show(block=True)
            if editID.cancel:
                posData.disableAutoActivateViewerWindow = False
                if not self.editIDbutton.findChild(QAction).isChecked():
                    self.editIDbutton.setChecked(False)
                return

            if editID.assignNewID:
                self.assignNewIDfromClickedID(
                    ID, event,
                    howToPropagateZslices=editID.howToPropagateZslices
                )
                return
            
            merging_IDs = editID.mergeWithExistingID
            if not self.doNotAskAgainExistingID:    
                self.editIDmergeIDs = editID.mergeWithExistingID
            self.doNotAskAgainExistingID = editID.doNotAskAgainExistingID
            
            self.applyEditID(
                ID, currentIDs, editID.how, x, y, 
                doPropagateUnvisited=editID.doPropagateFutureFrames,
                merging_IDs=merging_IDs,
                howToPropagateZslices=editID.howToPropagateZslices
            )
        
        elif (right_click or left_click) and self.keepIDsButton.isChecked():
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

        # Annotate cell as removed from the analysis
        elif right_click and self.binCellButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            if ID == 0:
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                binID_prompt = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to remove from the analysis',
                    parent=self, allowedValues=posData.IDs,
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
                nearest_ID = core.nearest_nonzero_2D(
                    self.get_2Dlab(posData.lab,force_z=False), y, x
                )
                ripID_prompt = dialogs_generic.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as dead',
                    parent=self, allowedValues=posData.IDs,
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

            # Apply annotate cell as dead to future frames if requested
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

    def gui_rightImageShowContextMenu(self, event):
        try:
            # Convert QPointF to QPoint
            self.imgGradRight.gradient.menu.popup(event.screenPos().toPoint())
        except AttributeError:
            self.imgGradRight.gradient.menu.popup(event.screenPos())

    @exception_handler
    def gui_mouseDragEventImg2(self, event):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        Y, X = self.get_2Dlab(posData.lab, force_z=False).shape
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        if not image_utils.is_in_bounds(xdata, ydata, X, Y):
            return

        # Eraser dragging mouse --> keep erasing
        if self.isMouseDragImg2 and self.eraserButton.isChecked():
            posData = self.data[self.pos_i]
            lab_2D = self.get_2Dlab(posData.lab, force_z=False)
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
            lab_2D = self.get_2Dlab(posData.lab, force_z=False)
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

    @exception_handler
    def gui_mouseReleaseEventImg2(self, event):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        Y, X = self.get_2Dlab(posData.lab, force_z=False).shape
        try:
            x, y = event.pos().x(), event.pos().y()
        except Exception as e:
            return
            
        xdata, ydata = int(x), int(y)
        if not image_utils.is_in_bounds(xdata, ydata, X, Y):
            self.isMouseDragImg2 = False
            self.updateAllImages()
            return

        # Move label mouse released, update move
        if self.isMovingLabel and self.moveLabelToolButton.isChecked():
            self.isMovingLabel = False

            # Update data (rp, etc)
            self.update_rp() # IDK can I do optimization here?

            # Repeat tracking
            self.tracking(enforce=True, assign_unique_new_IDs=False)

            self.updateAllImages()

            if not self.moveLabelToolButton.findChild(QAction).isChecked():
                self.moveLabelToolButton.setChecked(False)

    @exception_handler
    def gui_mousePressRightImage(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        isMod = alt
        right_click = event.button() == Qt.MouseButton.RightButton and not isMod
        is_right_click_action_ON = any([
            b.isChecked() for b in self.checkableQButtonsGroup.buttons()
        ])
        self.typingEditID = False
        showLabelsGradMenu = (
            right_click and not is_right_click_action_ON
            and not self.isBrushStrokesPromptActive()
            and not self.isBoxPromptActive()
        )
        if showLabelsGradMenu:
            self.gui_rightImageShowContextMenu(event)
            event.ignore()
        else: 
            self.gui_mousePressEventImg1(event)

    @exception_handler
    def gui_mouseDragRightImage(self, event):
        self.gui_mouseDragEventImg1(event)

    @exception_handler
    def gui_mouseReleaseRightImage(self, event):
        self.gui_mouseReleaseEventImg1(event)

    def changeRightClickToLeftOnMac(self, mouseEvent):
        button = mouseEvent.button()
        if not is_mac:
            return button
        
        delObjKeySequence, delObjQtButton = self.delObjAction
        if delObjKeySequence is None:
            return button
        
        if not delObjKeySequence.toString() == 'Control':
            return button
        
        if button != Qt.MouseButton.RightButton:
            return button
        
        if delObjQtButton == Qt.MouseButton.LeftButton:
            # On mac, pressing "Control" and clicking with left button changes 
            # it to a right click button --> here, left click is required for 
            # delete object --> force return of left click
            return Qt.MouseButton.LeftButton
        
        return button
