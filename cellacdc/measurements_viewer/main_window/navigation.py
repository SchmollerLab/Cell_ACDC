"""Frame, z-slice and Cell ID navigation of the measurements viewer
(ViewerNavigationGuiElements). Defines ARROW_KEYS, also read by
keyPressEvent in measurements_viewer/main_window/window.py. Spec:
measurements_viewer/__init__.py.
"""
import numpy as np
import skimage.measure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QAbstractSlider
from cellacdc import core, qutils


# Left/Right: frames (or z-slices), Up/Down: Cell IDs (`navigateWithArrowKey`)
ARROW_KEYS = (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down)


class ViewerNavigationGuiElements:
    """Frames, z-slices, planes and Cell ID navigation: arrow keys,
    Next/Prev, scroll bars and spin boxes, frame range, zoom to and
    follow the cell, ID search, plus the hooks that the shared GUI
    mixins call (setFrame, updateScrollbars, update_z_slice,
    updateZproj, switchViewedPlane).
    """

    def isArrowNavigationKeyPress(self, obj, event):
        """Whether the key press `event` of `obj` (seen by `eventFilter`) is
        an arrow key navigating frames/z-slices or cells: no modifiers, sent
        to an enabled widget of this window (disabled during loads, see
        `runInThread`) while no popup (e.g., a menu) is open, and neither to
        the menu bar (keyboard navigation of the menus) nor to a widget that
        uses the arrow keys itself (text, spin boxes, drop-down menus,
        lists)."""
        if event.key() not in ARROW_KEYS:
            return False
        # Arrow keys always have the keypad modifier on macOS
        modifiers = event.modifiers()
        if (modifiers != Qt.KeyboardModifier.NoModifier
                and modifiers != Qt.KeyboardModifier.KeypadModifier):
            return False
        if not self.data:
            return False
        if not obj.isWidgetType() or obj.window() is not self:
            return False
        # Application event filters see the key presses that Qt then drops
        # for disabled widgets (`QWidget::event`)
        if not self.isEnabled() or not obj.isEnabled():
            return False
        # Menus: after Alt (Windows/Linux) the menu bar has the focus, and an
        # open menu passes Left/Right on to its menu bar to switch menus while
        # the focus widget is still the one focused before
        if obj.inherits('QMenuBar') or obj.inherits('QMenu'):
            return False
        if QApplication.activePopupWidget() is not None:
            return False
        return not qutils.focus_widget_uses_arrow_keys()

    def navigateWithArrowKey(self, key):
        """Left/Right: previous/next frame (z-slice when the z-slice
        navigation is active), Up/Down: next/previous Cell ID."""
        if key in (Qt.Key_Left, Qt.Key_Right):
            step = 1 if key == Qt.Key_Right else -1
            if self.isZsliceNavigationActive():
                z = self.zSliceScrollBar.sliderPosition() + step
                self.zSliceScrollBar.setSliderPosition(z)
            else:
                self.setFrame(self.frame_i + step)
        else:
            self.stepID(1 if key == Qt.Key_Up else -1)

    def isZsliceNavigationActive(self):
        """Next/Previous step z-slices instead of frames (z-stacks only)."""
        return (
            bool(self.data) and self.posData().SizeZ > 1
            and self.zSliceCheckbox.isChecked()
        )

    def nextActionTriggered(self):
        if self.isZsliceNavigationActive():
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )
        else:
            self.navigateScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )

    def prevActionTriggered(self):
        if self.isZsliceNavigationActive():
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepSub
            )
        else:
            self.navigateScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepSub
            )

    def onSearchId(self, ID):
        if not self.isDataLoaded:
            self.logger.warning(
                'Data not loaded yet. Key pressing events are not connected.'
            )
            return

        self.get_data()
        self.findID(ID=ID)

    def initFramesNavigation(self):
        posData = self.posData()
        SizeT = posData.SizeT
        # A smaller maximum clamps the current value and would emit
        # `valueChanged` (-> `setFrame`). The values are set below.
        self.navigateScrollBar.blockSignals(True)
        self.navSpinBox.blockSignals(True)
        self.navigateScrollBar.setMaximum(SizeT)
        self.navigateScrollBar.setAbsoluteMaximum(SizeT)
        self.navSpinBox.setMaximum(SizeT)
        self.navigateScrollBar.blockSignals(False)
        self.navSpinBox.blockSignals(False)
        self.navSizeLabel.setText(f'/{SizeT}')
        self.navigateScrollBar.setDisabled(SizeT == 1)
        self.navSpinBox.setDisabled(SizeT == 1)
        self.resetNavigateFramesScrollbar(frame_i=self.frame_i)
        self.navSpinBox.setValueNoEmit(self.frame_i+1)

    def initZsliceNavigation(self):
        posData = self.posData()
        isZstack = posData.SizeZ > 1
        self.imgGrad.rescaleAcrossZstackAction.setDisabled(not isZstack)
        for widget in self.zSliceNavigationWidgets():
            widget.setVisible(isZstack)

        # Reset the plane also for 2D data: a zy/zx plane left from a
        # previously loaded z-stack would otherwise stay active (hidden),
        # disabling custom annotations and projections.
        self.switchPlaneCombobox.blockSignals(True)
        self.switchPlaneCombobox.setPlane('xy')
        self.switchPlaneCombobox.blockSignals(False)
        self.zProjComboBox.setDisabled(False)
        # Disabled (and turned off) on zy/zx planes: it stays off here
        self.overlayButton.setDisabled(False)
        self.overlayButtonPrevState = None

        if not isZstack:
            return

        # 2D segmentation masks cannot be displayed on planes other than xy
        self.switchPlaneCombobox.setDisabled(not self.isSegm3D)
        # A smaller maximum clamps the current z-slice and would emit
        # `valueChanged` (-> `update_z_slice`). The values are set below.
        self.zSliceScrollBar.blockSignals(True)
        self.zSliceSpinbox.blockSignals(True)
        self.setZsliceNavigationRange(posData.SizeZ)
        self.zSliceScrollBar.blockSignals(False)
        self.zSliceSpinbox.blockSignals(False)

        z = posData.SizeZ // 2
        zProjHow = 'single z-slice'
        segmInfo_df = getattr(posData, 'segmInfo_df', None)
        idx = (getattr(posData, 'filename', None), self.frame_i)
        try:
            z = int(segmInfo_df.at[idx, 'z_slice_used_gui'])
            zProjHow = segmInfo_df.at[idx, 'which_z_proj_gui']
        except Exception as err:
            pass
        
        self.zProjComboBox.blockSignals(True)
        self.zProjComboBox.setCurrentText(zProjHow)
        self.zProjComboBox.blockSignals(False)
        self.setZSliceWidgetsEnabled(zProjHow)
        
        self.zSliceScrollBar.blockSignals(True)
        self.zSliceScrollBar.setSliderPosition(z)
        self.zSliceScrollBar.blockSignals(False)
        self.zSliceSpinbox.setValueNoEmit(z+1)

    def zSliceNavigationWidgets(self):
        return (
            self.zSliceCheckbox, self.zSliceSpinbox, self.SizeZlabel, 
            self.zSliceScrollBar, self.zProjComboBox, self.switchPlaneCombobox
        )

    def setZSliceWidgetsEnabled(self, how):
        enabled = how == 'single z-slice'
        self.zSliceScrollBar.setEnabled(enabled)
        self.zSliceSpinbox.setEnabled(enabled)
        self.zSliceCheckbox.setEnabled(enabled)

    def initIDsNavigation(self, ID=None):
        self.IDs = self.getAllIDs()
        numIDs = len(self.IDs)
        self.numIDsLabel.setText(f'({numIDs} cells)')
        hasIDs = numIDs > 0
        self.idScrollBar.setEnabled(hasIDs)
        self.idSpinBox.setEnabled(hasIDs)
        if not hasIDs:
            self.selectedID = None
            return
        
        if ID not in self.IDs:
            # Select first ID present in current frame (if any)
            frameIDs = set(np.unique(self.getLabels()))
            ID = next((_ID for _ID in self.IDs if _ID in frameIDs), self.IDs[0])
        
        self.selectedID = ID
        self.idScrollBar.blockSignals(True)
        self.idScrollBar.setMaximum(numIDs-1)
        self.idScrollBar.setValue(self.IDs.index(ID))
        self.idScrollBar.blockSignals(False)
        self.idSpinBox.setMaximum(max(self.IDs))
        self.idSpinBox.setValueNoEmit(ID)
        self.ensureTimepointPredictions()

    def posComboboxIndexChanged(self, pos_i):
        self.loadPosition(pos_i, ID=self.selectedID)

    def navigateSpinboxValueChanged(self, value):
        self.navigateScrollBar.setSliderPosition(value)

    def framesScrollBarValueChanged(self, frame_n):
        self.setFrame(frame_n-1)

    def setFrame(self, frame_i):
        posData = self.posData()
        first_i, last_i = self.framesRange()
        frame_i = int(np.clip(frame_i, first_i, last_i))
        self.navSpinBox.setValueNoEmit(frame_i+1)
        if self.navigateScrollBar.sliderPosition() != frame_i+1:
            self.resetNavigateFramesScrollbar(frame_i=frame_i)
        
        if frame_i == self.frame_i:
            return
        
        self.frame_i = frame_i
        posData.frame_i = frame_i
        self.updateImages()
        self.zoomToSelectedID()
        self.updateCurrentFrameLine()
        # As in `gui.guiWin.next_frame`: frame of the open 3D viewer
        self.update3DViewer()

    def refreshUnlinkedSlideshow(self):
        """Show a new z-slice/projection/plane also in a slideshow not
        linked to the frames (`updateImages` -> `updateViewerWindow`
        refreshes only a linked one). Not on position changes: its frame
        might not exist there."""
        win = self.slideshowWin
        if win is None:
            return
        if win.linkWindow and win.linkWindowCheckbox.isChecked():
            return
        if win.frame_i >= self.posData().SizeT:
            return
        win.update_img()

    def update_z_slice(self, z):
        self.zSliceSpinbox.setValueNoEmit(z+1)
        self.updateImages()
        self.refreshUnlinkedSlideshow()

        self.update3DViewerViewRegion()

    def updateZproj(self, how):
        self.setZSliceWidgetsEnabled(how)
        self.updateImages()
        self.refreshUnlinkedSlideshow()

        self.update3DViewerViewRegion()

    def switchViewedPlane(self, previousPlane, currentPlane):
        posData = self.posData()
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes == 'z':
            self.zProjComboBox.setDisabled(False)
            depthSize = posData.SizeZ
        else:
            self.zProjComboBox.blockSignals(True)
            self.zProjComboBox.setCurrentText('single z-slice')
            self.zProjComboBox.blockSignals(False)
            self.setZSliceWidgetsEnabled('single z-slice')
            self.zProjComboBox.setDisabled(True)
            SizeY, SizeX = posData.img_data.shape[-2:]
            depthSize = SizeY if depthAxes == 'y' else SizeX
        
        self.zSliceScrollBar.blockSignals(True)
        self.setZsliceNavigationRange(depthSize)
        self.zSliceScrollBar.setSliderPosition(depthSize // 2)
        self.zSliceScrollBar.blockSignals(False)
        self.zSliceSpinbox.setValueNoEmit(depthSize // 2 + 1)

        self.initContoursImage()
        self.setOverlayEnabledForPlane(depthAxes)
        self.updateImages()
        self.refreshUnlinkedSlideshow()
        self.ax1.autoRange()
        self.zoomToSelectedID()
        # The z-slice widgets changed without signals above --> send the
        # new plane and slice also if the view range did not change
        self.update3DViewerViewRegion()

    def setOverlayEnabledForPlane(self, depthAxes):
        """As `gui.guiWin.switchViewedPlane`: overlay channels are drawn
        only on the xy plane (`getOlImg` would index the z-stack with the
        y/x index of the scrollbar). On zy/zx turn the overlay off and
        disable its button, back on xy restore the previous state."""
        if depthAxes != 'z':
            if self.overlayButtonPrevState is None:
                # Coming from xy (not from the other of zy/zx)
                self.overlayButtonPrevState = self.overlayButton.isChecked()
            if self.overlayButton.isChecked():
                self.overlayButton.setChecked(False)
            self.overlayButton.setDisabled(True)
            return
        if self.overlayButtonPrevState is None:
            return
        wasChecked = self.overlayButtonPrevState
        self.overlayButtonPrevState = None
        self.overlayButton.setDisabled(False)
        if wasChecked:
            self.overlayButton.setChecked(True)

    def idScrollBarValueChanged(self, idx):
        if not self.IDs:
            return
        ID = self.IDs[idx]
        self.idSpinBox.setValueNoEmit(ID)
        self.setSelectedID(ID)
        
        if self.rescalePlotCheckbox.isChecked():
            self.autoscaleMeasurementsPlot()

    def idSpinBoxEditingFinished(self):
        ID = self.idSpinBox.value()
        if ID == self.selectedID:
            return
        
        if ID not in self.IDs:
            self.statusBar().showMessage(f'Cell ID {ID} does not exist.', 3000)
            self.idSpinBox.setValueNoEmit(self.selectedID)
            return
        
        self.idScrollBar.setValue(self.IDs.index(ID))

    def setSelectedID(self, ID):
        if ID != self.selectedID:
            self.logger.info(f'Selected Cell ID {ID}')
        self.selectedID = ID
        comparisonData = self.comparisonPlotData()
        self.ensureTimepointPredictions()
        isFrameChanged = self.updateFramesRange()
        self.zoomToSelectedID(goToZslice=True)
        self.updateImages()
        if isFrameChanged:
            # Moved into the cell's frames ("Only frames with selected cell")
            self.update3DViewer()
        if (comparisonData is not None
                and comparisonData.equals(self.comparisonPlotData())):
            # The comparison scatter shows all cells: replotting it is slow
            # on large tables and would reset the zoom
            return
        self.plotMeasurements()

    def framesRange(self):
        """Get the (first, last) frame index the user can navigate to."""
        SizeT = self.posData().SizeT
        if not self.restrictFramesCheckbox.isChecked():
            return 0, SizeT-1
        
        if self.selectedID is None:
            return 0, SizeT-1
        
        frames = self.getIDframes(self.selectedID)
        frames = frames[(frames >= 0) & (frames < SizeT)]
        if len(frames) == 0:
            return 0, SizeT-1
        
        return int(frames[0]), int(frames[-1])

    def updateFramesRange(self):
        """Restrict the frames navigation widgets to `framesRange` and move 
        the current frame into the range if it is outside. 
        
        Returns True if the current frame changed (images and the 3D viewer,
        `update3DViewer`, not updated).
        """
        posData = self.posData()
        first_i, last_i = self.framesRange()
        
        self.navigateScrollBar.blockSignals(True)
        self.navigateScrollBar.setMinimum(first_i+1)
        self.navigateScrollBar.setMaximum(last_i+1)
        self.navigateScrollBar.blockSignals(False)
        self.navSpinBox.blockSignals(True)
        self.navSpinBox.setMinimum(first_i+1)
        self.navSpinBox.setMaximum(last_i+1)
        self.navSpinBox.blockSignals(False)
        
        isRangeZero = first_i == last_i
        self.navigateScrollBar.setDisabled(isRangeZero)
        self.navSpinBox.setDisabled(isRangeZero)
        
        frame_i = int(np.clip(self.frame_i, first_i, last_i))
        isFrameChanged = frame_i != self.frame_i
        self.frame_i = frame_i
        posData.frame_i = frame_i
        self.navSpinBox.setValueNoEmit(frame_i+1)
        self.resetNavigateFramesScrollbar(frame_i=frame_i)
        if isFrameChanged:
            self.updateCurrentFrameLine()
        return isFrameChanged

    def restrictFramesToggled(self, checked):
        self.logger.info(
            f'Only frames with selected cell {"on" if checked else "off"}'
        )
        self._set_setting_value('restrictFrames', checked)
        if not self.data:
            return
        isFrameChanged = self.updateFramesRange()
        if not isFrameChanged:
            return
        self.updateImages()
        if self.followIDcheckbox.isChecked():
            self.zoomToSelectedID()
        self.update3DViewer()

    def followIDtoggled(self, checked):
        self.logger.info(f'Follow cell {"on" if checked else "off"}')
        if checked:
            self.zoomToSelectedID()
            
        self._set_setting_value('followID', checked)

    def goToZsliceID(self, ID):
        """Move the z-slice scrollbar to the slice of `ID` (along the 
        current depth axis) if the ID is not visible in the current slice.
        """
        posData = self.posData()
        if not self.isSegm3D or posData.SizeZ == 1:
            return
        
        if self.zProjComboBox.currentText() != 'single z-slice':
            return
        
        lab = self.getLabels()
        lab2D = self.get_2Dlab(lab, force_z=False)
        if np.any(lab2D == ID):
            return
        
        objMask = (lab == ID).astype(np.uint8)
        if not np.any(objMask):
            return
        
        obj = skimage.measure.regionprops(objMask)[0]
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes == 'z':
            current_z = self.zSliceScrollBar.sliderPosition()
            z = core.nearest_nonzero_z_idx_from_z_centroid(
                obj, current_z=current_z
            )
        elif depthAxes == 'y':
            z = round(obj.centroid[1])
        else:
            z = round(obj.centroid[2])
        
        self.zSliceScrollBar.setSliderPosition(int(z))

    def zoomToSelectedID(self, goToZslice=False, force=False):
        ID = self.selectedID
        if ID is None:
            return
        
        if not self.followIDcheckbox.isChecked() and not force:
            return
        
        if goToZslice or self.followIDcheckbox.isChecked():
            self.goToZsliceID(ID)
        
        lab = self.getLabels()
        lab2D = self.get_2Dlab(lab, force_z=False)
        objMask = None
        if np.any(lab2D == ID):
            objMask = lab2D == ID
        elif self.switchPlaneCombobox.depthAxes() == 'z':
            # ID not visible on current slice --> zoom on its projection
            objMask = lab == ID
            if objMask.ndim == 3:
                objMask = objMask.any(axis=0)
        
        if objMask is None or not np.any(objMask):
            return
        
        rows, cols = np.nonzero(objMask)
        obj_bbox = (rows.min(), cols.min(), rows.max()+1, cols.max()+1)
        self.setViewRangeToObjBbox(obj_bbox, padding=10)

    def stepID(self, step):
        if not self.IDs:
            return
        idx = int(np.clip(
            self.idScrollBar.value() + step, 0, len(self.IDs)-1
        ))
        self.idScrollBar.setValue(idx)
        # The spinbox arrows already changed its value by +-1
        self.idSpinBox.setValueNoEmit(self.currID())

    def currID(self):
        if not self.IDs:
            return None
        idx = self.idScrollBar.value()
        return self.IDs[idx]

    def updateScrollbars(self):
        """Sync the navigation widgets after `posData.frame_i` was changed 
        externally (e.g., by `SearchIDGuiElements.findID`)"""
        frame_i = self.frame_i
        self.navSpinBox.setValueNoEmit(frame_i+1)
        self.resetNavigateFramesScrollbar(frame_i=frame_i)
        self._lab = None
        self._labKey = None
        self.updateCurrentFrameLine()
        self.plotMeasurements()
        self.updateIDnotPresentLabel()
        self.update3DViewer()

    def highlightSearchedID(self, ID, force=False, greyOthers=True):
        self.highlightedID = ID
        self.highLightIDLayerImg1.clear()
        self.searchedIDitemLeft.clear()
        if ID == 0:
            return
        self.highlightSelectedID()

    def goToObjectID(self, ID):
        if ID not in self.IDs:
            self.statusBar().showMessage(
                f'Cell ID {ID} does not exist in the acdc_output table.', 
                3000
            )
            return
        
        idx = self.IDs.index(ID)
        if self.idScrollBar.value() == idx:
            self.setSelectedID(ID)
        else:
            self.idScrollBar.setValue(idx)

    def goToLostObjectID(self, ID, color=None):
        self.goToObjectID(ID)

    def goToAcceptedLostObjectID(self, ID):
        self.goToObjectID(ID)

    def autoRange(self):
        self.ax1.autoRange()
        if self.followIDcheckbox.isChecked():
            # Keep the view on the selected cell (e.g., after resizing the 
            # bottom layout)
            self.zoomToSelectedID()
