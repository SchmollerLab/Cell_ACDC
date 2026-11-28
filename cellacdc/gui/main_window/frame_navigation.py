"""Frame, position and z-slice navigation of the main GUI (the navigation
widgets shared with the measurements viewer are in
gui/mixins/navigation.py). Mixins of guiWin (gui/main_window/window.py)
only: the measurements viewer never composes them.
"""
from functools import partial
import numpy as np
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QAbstractSlider
from cellacdc import exception_handler
from cellacdc import myutils
from cellacdc import printl
from cellacdc import qutils
from . import constants as gui_main_constants


class MainNavigationGuiElements:
    """Frame, position and z-slice navigation of the main GUI: next and
    previous frame and position, scroll bars and spin boxes, viewed
    plane switching and z-projection. The navigation widgets shared with
    the measurements viewer are in gui/mixins/navigation.py.
    """

    def goToFrameNumber(self, frame_n):
        posData = self.data[self.pos_i]
        prev_frame_i = posData.frame_i
        posData.frame_i = frame_n - 1
        proceed_cca, _ = self.get_data()
        if not self.restorePrevFrameIfNotProceed(proceed_cca, prev_frame_i):
            return

        self.updateAllImages()
        self.updateScrollbars()

    def zProjLockViewToggled(self, checked):
        self.updateZproj(self.zProjComboBox.currentText())

    def goToObjectID(self, ID):
        posData = self.data[self.pos_i]
        obj = posData.rp.get_obj_from_ID(ID)
        self.goToZsliceSearchedID(obj)
        
        self.highlightSearchedID(ID)
        propsQGBox = self.guiTabControl.propsQGBox
        propsQGBox.idSB.setValue(ID)
        is_obj_visible = self.isObjVisibleViewRange(obj.bbox)
        if not is_obj_visible:
            centroid = posData.rp.get_centroid(obj.label)
            yc, xc = self.getObjCentroid(centroid)
            pos = (int(xc), int(yc))
            # set center of view to the lost object's position
            self.ax1.setCenter(pos)        

    def enableZstackWidgets(self, enabled):
        if enabled:
            myutils.setRetainSizePolicy(self.zSliceScrollBar)
            myutils.setRetainSizePolicy(self.zProjComboBox)
            myutils.setRetainSizePolicy(self.zSliceOverlay_SB)
            myutils.setRetainSizePolicy(self.zProjOverlay_CB)
            myutils.setRetainSizePolicy(self.overlay_z_label)
            self.zSliceScrollBar.setDisabled(False)
            self.zProjComboBox.show()
            if self.data[self.pos_i].SizeT > 1:
                self.zProjLockViewButton.show()
            self.zSliceScrollBar.show()
            self.zSliceCheckbox.show()
            self.zSliceSpinbox.show()
            self.switchPlaneCombobox.show()
            self.switchPlaneCombobox.setDisabled(False)
            self.SizeZlabel.show()
            self.launch3dViewerButton.show()
            self.launch3dViewerButton.setDisabled(False)
            self.launch3dViewerAction.setDisabled(False)
        else:
            myutils.setRetainSizePolicy(self.zSliceScrollBar, retain=False)
            myutils.setRetainSizePolicy(self.zProjComboBox, retain=False)
            myutils.setRetainSizePolicy(self.zSliceOverlay_SB, retain=False)
            myutils.setRetainSizePolicy(self.zProjOverlay_CB, retain=False)
            myutils.setRetainSizePolicy(self.overlay_z_label, retain=False)
            self.zSliceScrollBar.setDisabled(True)
            self.zProjComboBox.hide()
            self.zProjComboBox.hide()
            self.zSliceScrollBar.hide()
            self.zSliceCheckbox.hide()
            self.zSliceSpinbox.hide()
            self.SizeZlabel.hide()
            self.switchPlaneCombobox.hide()
            self.switchPlaneCombobox.setDisabled(True)
            self.launch3dViewerButton.hide()
            self.launch3dViewerButton.setDisabled(True)
            self.launch3dViewerAction.setDisabled(True)
        
        self.showZsliceIndexAction.setVisible(enabled)
        self.imgGrad.rescaleAcrossZstackAction.setDisabled(not enabled)
        for ch, overlayItems in self.overlayLayersItems.items():
            lutItem = overlayItems[1]
            lutItem.rescaleAcrossZstackAction.setDisabled(not enabled)

    def setFrameNavigationDisabled(self, disable: bool, why: str, warn=True):
        """Disables the frame navigation buttons and scrollbar.
        This is used when the user is not allowed to navigate through frames
        Call again to unlock it again. Also sets tooltips to inform the user
        
        Parameters
        ----------
        disable : bool
            if the navigation should be disabled
        why : str
            the reason for disabeling the navigation.
        """
        
        if disable:
            self.whyNavigateDisabled.add(why)
        else:
            try:
                self.whyNavigateDisabled.remove(why)
            except KeyError:
                pass
        
        if len(self.whyNavigateDisabled) == 0:
            disable = False
        else:
            disable = True

        # Apply the disable/enable state
        self.prevAction.setDisabled(disable)
        self.nextAction.setDisabled(disable)
        self.navigateScrollBar.setDisabled(disable)
        
        # Set appropriate tooltip
        if not disable:
            self.navigateScrollBar.setToolTip(
                'NOTE: The maximum frame number that can be visualized with this '
                'scrollbar\n'
                'is the last visited frame with the selected mode\n'
                '(see "Mode" selector on the top-right).\n\n'
                'If the scrollbar does not move it means that you never visited\n'
                'any frame with current mode.\n\n'
                'Note that the "Viewer" mode allows you to scroll ALL frames.'
            )
            return
        
        txt = f'Frame navigation disabled: {self.whyNavigateDisabled}'
        if warn:
            self.logger.info(txt)
        self.navigateScrollBar.setToolTip(txt)

    def isNavigateActionOnNextFrame(self):
        posData = self.data[self.pos_i]
        if posData.SizeT == 1:
            return False
        
        ax1_coords = self.getMouseDataCoordsRightImage()
        if ax1_coords is None:
            return False
        
        if not self.labelsGrad.showNextFrameAction.isEnabled():
            return False
        
        if not self.labelsGrad.showNextFrameAction.isChecked():
            return
        
        # Mouse is on right image and next frame action is checked
        return True 

    def rightImageFramesScrollbarValueChanged(self, value):
        img = self.nextFrameImage(current_frame_i=value-2)
        self.img1.linkedImageItem.frame_i = value
        self.img1.linkedImageItem.setImage(img)

    def nextActionTriggered(self):
        if self.isNavigateActionOnNextFrame():
            self.rightImageFramesScrollbar.setValue(
                self.rightImageFramesScrollbar.value()+1 
            )
            return
        
        stepAddAction = QAbstractSlider.SliderAction.SliderSingleStepAdd
        if self.zKeptDown or self.zSliceCheckbox.isChecked():
            self.zSliceScrollBar.triggerAction(stepAddAction)
        else:
            self.navigateScrollBar.triggerAction(stepAddAction)

    def prevActionTriggered(self):
        if self.isNavigateActionOnNextFrame():
            self.rightImageFramesScrollbar.setValue(
                self.rightImageFramesScrollbar.value()-1 
            )
            return
        
        stepSubAction = QAbstractSlider.SliderAction.SliderSingleStepSub
        if self.zKeptDown or self.zSliceCheckbox.isChecked():
            self.zSliceScrollBar.triggerAction(stepSubAction)
        else:
            self.navigateScrollBar.triggerAction(stepSubAction)

    def resetNavigateScrollbar(self):
        try:
            self.navigateScrollBar.blockSignals(True)
            self.navigateScrollBar.actionTriggered.disconnect()
            self.navigateScrollBar.sliderReleased.disconnect()
            self.navigateScrollBar.sliderMoved.disconnect()
            # self.navigateScrollBar.valueChanged.disconnect()
            self.navigateScrollBar.setSliderPosition(self.navSpinBox.value())
        except Exception as e:
            if "disconnect()" not in str(e):
                printl(e)
            pass

        self.navigateScrollBar.blockSignals(False)
        self.navigateScrollBar.actionTriggered.connect(
            self.framesScrollBarActionTriggered
        )
        self.navigateScrollBar.sliderReleased.connect(self.framesScrollBarReleased)
        self.navigateScrollBar.sliderMoved.connect(self.framesScrollBarMoved)

    @exception_handler
    def next_cb(self):
        if self.isSnapshot:
            self.next_pos()
        else:
            self.next_frame()
        if self.curvToolButton.isChecked():
            self.curvTool_cb(True)
        
        self.updatePropsWidget('')

    @exception_handler
    def prev_cb(self):
        if self.isSnapshot:
            self.prev_pos()
        else:
            self.prev_frame()
        if self.curvToolButton.isChecked():
            self.curvTool_cb(True)
        
        self.updatePropsWidget('')

    def updateScrollbars(self):
        self.updateItemsMousePos()
        self.updateFramePosLabel()
        posData = self.data[self.pos_i]
        navPos = self.pos_i+1 if self.isSnapshot else posData.frame_i+1
        self.navigateScrollBar.setSliderPosition(navPos)
        if posData.SizeZ > 1:
            self.updateZsliceScrollbar(posData.frame_i)
            idx = (posData.filename, posData.frame_i)
            self.zSliceScrollBar.setMaximum(posData.SizeZ-1)
            self.zSliceSpinbox.setMaximum(posData.SizeZ)
            self.SizeZlabel.setText(f'/{posData.SizeZ}')

    def next_pos(self):
        self.store_data(autosave=False)
        self.exportCurrentView(isOnNext=True)
        prev_pos_i = self.pos_i
        if self.pos_i < self.num_pos-1:
            self.pos_i += 1
            self.updateSegmDataAutoSaveWorker()
        else:
            self.logger.info('You reached last position.')
            self.pos_i = 0
        self.updatePos()

    def updatePos(self):
        self.clearUndoQueue()
        # IDs of other positions are unrelated --> clear the cells selected
        # to be added to the brush strokes
        self.brushStrokesSelectCellsToolbar.resetSelection()
        self.setStatusBarLabel()
        self.checkManageVersions()
        self.removeAlldelROIsCurrentFrame()
        self.resetManualBackgroundItems()
        proceed_cca, never_visited = self.get_data(debug=False)
        self.pointsLayerLoadedDfsToData()
        self.flushDirtyPointsLayersAutosave()
        self.initContoursImage()
        self.initTextAnnot()
        self.postProcessing()
        self.updateScrollbars()
        self.updatePreprocessPreview()
        self.updateCombineChannelsPreview()
        self.updateAllImages()            
        self.computeSegm()
        self.zoomOut()
        self.restartZoomAutoPilot()
        self.initManualBackgroundObject()
        self.updateObjectCounts()
        self.updateItemsMousePos()

    def prev_pos(self):
        self.store_data(debug=False, autosave=False)
        prev_pos_i = self.pos_i
        if self.pos_i > 0:
            self.pos_i -= 1
            self.updateSegmDataAutoSaveWorker()
        else:
            self.logger.info('You reached first position.')
            self.pos_i = self.num_pos-1
        self.updatePos()

    def checkIfFutureFrameManualAnnotPastFrames(self):
        if not self.manualAnnotPastButton.isChecked():
            return True
        
        posData = self.data[self.pos_i]
        frame_to_restore = self.manualAnnotState.get('frame_i_to_restore')
        if posData.frame_i <= frame_to_restore:
            return True
        
        warn_txt = (
            'WARNING: Cannot navigate to future frames while in '
            'manual annotation mode.'
        )
        self.logger.info(warn_txt)
        self.statusBarLabel.setText(f'<p style="color:red;">{warn_txt}</p>')
        
        return False

    def next_frame(self, warn=True):        
        if self.editCcaWidget is not None:
            self.editCcaWidget.show()
            self.editCcaWidget.raise_()
            self.editCcaWidget.activateWindow()
            self.logger.info(
                '[WARNING]: dialogue to initialize cell cycle annotations '
                'is open. Cannot navigate frames'
            )
            return

        proceed = self.checkIfFutureFrameManualAnnotPastFrames()
        if not proceed:
            return
        
        proceed = self.askInitCcaFirstFrame()
        if not proceed:
            return
        
        proceed = self.askInitLinTreeFirstFrame()
        if not proceed:
            return
        
        mode = str(self.modeComboBox.currentText())
        posData = self.data[self.pos_i]
        
        if posData.frame_i >= posData.SizeT-1:
            # Store data for current frame
            self.exportCurrentView(isOnNext=True)
            if mode != 'Viewer':
                self.store_data(debug=False)
            msg = 'You reached the last segmented frame!'
            self.logger.info(msg)
            self.titleLabel.setText(msg, color=self.titleColor)
            return

        proceed = self.warnLostObjects()
        if not proceed:
            self.resetNavigateScrollbar()
            return     

        # Store data for current frame
        if mode != 'Viewer':
            self.store_data(debug=False)

        self.askLineageTreeChanges()
        posData.frame_i += 1
        self.removeAlldelROIsCurrentFrame()
        proceed_cca, never_visited = self.get_data()
        if not proceed_cca:
            posData.frame_i -= 1
            self.get_data()
            self.logger.info(
                'No data for current frame. '
            )
            return
        
        if mode == 'Segmentation and Tracking' or self.isSnapshot:
            self.addExistingDelROIs()
            self.applyAllDelROI()
        
        self.updatePreprocessPreview()
        self.updateCombineChannelsPreview()
        self.postProcessing()
        self.tracking(storeUndo=True, wl_update=False) 
        notEnoughG1Cells, proceed = self.attempt_auto_cca()
        if notEnoughG1Cells or not proceed:
            posData.frame_i -= 1
            self.get_data()
            self.setAllTextAnnotations()
            self.logger.info(
                'Not enough G1 cells to compute cell cycle annotations.'
            )
            return
        
        self.resetExpandLabel()
        self.updateAllImages()
        self.updateHighlightedAxis()
        self.updateViewerWindow()
        self.updateLastVisitedFrame(last_visited_frame_i=posData.frame_i-1)
        self.setNavigateScrollBarMaximum()
        self.updateScrollbars()
        self.computeSegm()
        self.initGhostObject()
        self.drawManualTrackingGhost(self.xHoverImg, self.yHoverImg)
        self.whitelistPropagateIDs()
        self.zoomToCells()
        self.updateItemsMousePos()
        self.updateObjectCounts()
        self.update3DViewer()
        
        self.exportCurrentView(isOnNext=True)
        self.apply_tools_on_new_frame()

    def apply_tools_on_new_frame(self):
        mode = str(self.modeComboBox.currentText())
        if mode != 'Segmentation and Tracking':
            return
        posData = self.data[self.pos_i]
        if not (posData.last_tracked_i <= posData.frame_i) or posData.frame_i == self.lastFrameRanOnFirstVisitTools:
            return
        
        self.lastFrameRanOnFirstVisitTools = posData.frame_i
        for name, checkbox in self.applyToolNewFrameActions.items():
            if not checkbox.isChecked():
                continue
            
            tool_button = self.applyToolNewFrameButtons[name]
            try:
                if tool_button.start_event_loop:
                    self.wait_worker_loop = QEventLoop()
                if hasattr(tool_button, 'click'):
                    tool_button.click()
                elif hasattr(tool_button, 'trigger'):
                    tool_button.trigger()
                else:
                    printl(
                        f"Warning: {name} has no click or trigger method"
                    )

                if hasattr(self, "wait_worker_loop") and self.wait_worker_loop is not None and tool_button.start_event_loop:
                    self.wait_worker_loop.exec_()
                    

            except Exception as e:
                self.logger.info(f"Error applying tool {name}: {e}")

    def setNavigateScrollBarMaximum(self, last_cca_frame_i=None):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Segmentation and Tracking':
            if posData.last_tracked_i is not None:
                if posData.frame_i > posData.last_tracked_i:
                    self.navigateScrollBar.setMaximum(posData.frame_i+1)
                    self.navSpinBox.setMaximum(posData.frame_i+1)
                else:
                    self.navigateScrollBar.setMaximum(posData.last_tracked_i+1)
                    self.navSpinBox.setMaximum(posData.last_tracked_i+1)
            else:
                self.navigateScrollBar.setMaximum(posData.frame_i+1)
                self.navSpinBox.setMaximum(posData.frame_i+1)
            
            self.updateLastCheckedFrameWidgets(self.navSpinBox.maximum()-1)
        elif mode == 'Cell cycle analysis':
            if last_cca_frame_i is not None:
                self.navigateScrollBar.setMaximum(last_cca_frame_i+1)
                self.navSpinBox.setMaximum(last_cca_frame_i+1)
            elif posData.frame_i > self.last_cca_frame_i:
                self.navigateScrollBar.setMaximum(posData.frame_i+1)
                self.navSpinBox.setMaximum(posData.frame_i+1)
            else:
                self.navigateScrollBar.setMaximum(self.last_cca_frame_i+1)
                self.navSpinBox.setMaximum(self.last_cca_frame_i+1)
            
            self.lastTrackedFrameLabel.setText(
                f'Last cc annot. frame n. = {self.navSpinBox.maximum()}'
            )
        elif mode == 'Normal division: Lineage tree':
            if self.lineage_tree is None:
                self.navigateScrollBar.setMaximum(posData.frame_i+1)
                self.navSpinBox.setMaximum(posData.frame_i+1)
            else:
                if self.lineage_tree.frames_for_dfs:
                    i = max(self.lineage_tree.frames_for_dfs)
                else:
                    i = 0
                # The current frame is not annotated yet after a reset of
                # its annotations (`resetLin_tree_future`)
                i = max(i, posData.frame_i)
                self.navigateScrollBar.setMaximum(i+1)
                self.navSpinBox.setMaximum(i+1)

    # @exec_time
    def prev_frame(self):
        posData = self.data[self.pos_i]    
        if posData.frame_i <= 0:
            msg = 'You reached the first frame!'
            self.logger.info(msg)
            self.titleLabel.setText(msg, color=self.titleColor)
            return
        
        # Store data for current frame
        mode = str(self.modeComboBox.currentText())
        if mode != 'Viewer':
            self.store_data(debug=False)
            
        self.removeAlldelROIsCurrentFrame()           
        self.askLineageTreeChanges()
        posData.frame_i -= 1
        _, never_visited = self.get_data()
        
        if mode == 'Segmentation and Tracking' or self.isSnapshot:
            self.addExistingDelROIs()
        
        self.resetExpandLabel()
        self.updatePreprocessPreview()
        self.updateCombineChannelsPreview()
        self.postProcessing()
        self.tracking()
        self.whitelistPropagateIDs(update_lab=True)
        self.updateAllImages()
        self.updateScrollbars()
        self.updateHighlightedAxis()
        self.zoomToCells()
        self.initGhostObject()
        self.drawManualTrackingGhost(self.xHoverImg, self.yHoverImg)
        self.updateViewerWindow()
        self.updateItemsMousePos()
        self.updateObjectCounts()
        self.update3DViewer()

    def connectScrollbars(self):
        self.t_label.show()
        self.navigateScrollBar.show()
        self.navigateScrollBar.setDisabled(False)

        if self.data[0].SizeZ > 1:
            self.enableZstackWidgets(True)
            self.setZsliceNavigationRange(self.data[0].SizeZ)
            qutils.tryDisconnectSignal(self.zSliceScrollBar, 'actionTriggered')
            qutils.tryDisconnectSignal(self.zSliceScrollBar, 'sliderReleased' )
            qutils.tryDisconnectSignal(self.zProjComboBox, 'currentTextChanged')
            qutils.tryDisconnectSignal(self.zProjComboBox, 'activated')
            qutils.tryDisconnectSignal(
                self.switchPlaneCombobox, 'sigPlaneChanged'
            )
            qutils.tryDisconnectSignal(self.launch3dViewerButton, 'toggled')
            qutils.tryDisconnectSignal(self.zProjLockViewButton, 'toggled')
            self.zSliceScrollBar.actionTriggered.connect(
                self.zSliceScrollBarActionTriggered
            )
            self.zSliceScrollBar.sliderReleased.connect(
                self.zSliceScrollBarReleased
            )
            self.zProjComboBox.currentTextChanged.connect(self.updateZproj)
            self.zProjComboBox.activated.connect(self.clearComboBoxFocus)
            self.switchPlaneCombobox.sigPlaneChanged.connect(
                self.switchViewedPlane
            )
            self.zProjLockViewButton.toggled.connect(self.zProjLockViewToggled)
            self.launch3dViewerButton.toggled.connect(
                self.launch3dViewerToggled
            )

        posData = self.data[self.pos_i]
        if posData.SizeT == 1:
            self.t_label.setText('Position n.')
            self.navigateScrollBar.setMinimum(1)
            self.navigateScrollBar.setMaximum(len(self.data))
            self.navigateScrollBar.setAbsoluteMaximum(len(self.data))
            self.navSpinBox.setMaximum(len(self.data))
            self.navigateScrollBar.connectEvents({
                'sliderMoved': self.PosScrollBarMoved,
                'sliderReleased': self.PosScrollBarReleased,
                'actionTriggered': self.PosScrollBarAction
            })
        else:
            self.navigateScrollBar.setMinimum(1)
            self.navigateScrollBar.setAbsoluteMaximum(posData.SizeT)
            self.rightImageFramesScrollbar.setMinimum(1)
            self.rightImageFramesScrollbar.setMaximum(posData.SizeT)
            if posData.last_tracked_i is not None:
                self.navigateScrollBar.setMaximum(posData.last_tracked_i+1)
                self.navSpinBox.setMaximum(posData.last_tracked_i+1)
            self.t_label.setText('Frame n. ')
            self.navigateScrollBar.connectEvents({
                'sliderMoved': self.framesScrollBarMoved,
                'sliderReleased': self.framesScrollBarReleased,
                'actionTriggered': self.framesScrollBarActionTriggered
            })
            self.rightImageFramesScrollbar.connectValueChanged(
                self.rightImageFramesScrollbarValueChanged
            )

    def isZsliceScrollBarActive(self):
        """False with 2D images: z-stack data loaded before in the same GUI
        leaves the z-slice scrollbar connected (`connectScrollbars`), and the
        keys stepping it (Up/Down, Home/End, ...) would reach `update_z_slice`,
        which needs `segmInfo_df` (None without z-stacks)."""
        return self.data[self.pos_i].SizeZ > 1

    def zSliceScrollBarActionTriggered(self, action):
        if not self.isZsliceScrollBarActive():
            return
        singleMove = (
            action == gui_main_constants.SliderSingleStepAdd
            or action == gui_main_constants.SliderSingleStepSub
            or action == gui_main_constants.SliderPageStepAdd
            or action == gui_main_constants.SliderPageStepSub
        )
        if singleMove:
            self.update_z_slice(self.zSliceScrollBar.sliderPosition())
        elif action == gui_main_constants.SliderMove:
            if self.zSliceScrollBarStartedMoving and self.isSegm3D:
                self.clearAx1Items(onlyHideText=True)
                self.clearAx2Items(onlyHideText=True)
            posData = self.data[self.pos_i]
            idx = (posData.filename, posData.frame_i)
            z = self.zSliceScrollBar.sliderPosition()
            if self.switchPlaneCombobox.depthAxes() == 'z': 
                posData.segmInfo_df.at[idx, 'z_slice_used_gui'] = z
            self.zSliceSpinbox.setValueNoEmit(z+1)
            img = self._getImageupdateAllImages(None)
            self.img1.setCurrentZsliceIndex(z)
            self.img1.setImage(
                img, next_frame_image=self.nextFrameImage(),
                scrollbar_value=posData.frame_i+2
            )
            try:
                self.setOverlayImages()
            except Exception as err:
                pass
            
            if self.labelsGrad.showLabelsImgAction.isChecked():
                self.img2.setImage(posData.lab, z=z, autoLevels=False)
            self.updateViewerWindow()
            self.setTextAnnotZsliceScrolling()
            self.setGraphicalAnnotZsliceScrolling()
            self.setOverlayLabelsItems()
            self.drawPointsLayers(computePointsLayers=False)
            self.setBrushStrokesLayerImage()
            self.drawMagicPromptsBoxes()
            self.zSliceScrollBarStartedMoving = False
            self.highlightSearchedID(self.highlightedID, force=True)

    def zSliceScrollBarReleased(self):
        if not self.isZsliceScrollBarActive():
            return
        self.clearTempBrushImage()
        self.zSliceScrollBarStartedMoving = True
        self.update_z_slice(self.zSliceScrollBar.sliderPosition())

    def setSwitchViewedPlaneDisabled(self, disabled):
        posData = self.data[self.pos_i]
        if posData.SizeZ == 1:
            return

        self.switchPlaneCombobox.setDisabled(disabled)
        if disabled:
            self.switchPlaneCombobox.setCurrentIndex(0)

    def _setViewRangeSwitchPlane(self, previousPlane):
        posData = self.data[self.pos_i]
        SizeZ = posData.SizeZ
        SizeY, SizeX = self.img1.image.shape[:2]
        currentPlane = self.switchPlaneCombobox.plane()
        if previousPlane == 'xy':
            if currentPlane == 'zy':
                self.ax1.setRange(xRange=self.yRangePrev)
                unusedRange = np.clip(self.xRangePrev, 0, SizeX)
            elif currentPlane == 'zx':
                self.ax1.setRange(xRange=self.xRangePrev)
                unusedRange = np.clip(self.yRangePrev, 0, SizeY)
        elif previousPlane == 'zy':
            if currentPlane == 'xy':
                self.ax1.setRange(yRange=self.xRangePrev)
                unusedRange = np.clip(self.yRangePrev, 0, SizeZ)
            elif currentPlane == 'zx':
                self.ax1.setRange(yRange=self.yRangePrev)
                unusedRange = np.clip(self.xRangePrev, 0, SizeY)
        elif previousPlane == 'zx':
            if currentPlane == 'xy':
                self.ax1.setRange(xRange=self.xRangePrev)
                unusedRange = np.clip(self.yRangePrev, 0, SizeZ)
            elif currentPlane == 'zy':
                self.ax1.setRange(yRange=self.yRangePrev)
                unusedRange = np.clip(self.xRangePrev, 0, SizeX)
        
        sliceValue = round((unusedRange[0] + unusedRange[1])/2)
        self.zSliceScrollBar.setSliderPosition(sliceValue)
        self.update_z_slice(self.zSliceScrollBar.sliderPosition())

    def setViewRangeSwitchPlane(self, previousPlane):
        self.autoRange()
        QTimer.singleShot(
            100, partial(self._setViewRangeSwitchPlane, previousPlane)
        )

    # def gui_preinitAnnotations(self):
    #     og_frame = self.data[self.pos_i].frame_i
    #     print('Pre-initializing annotations for all frames and positions...')
    #     for posData in self.data:
    #         og_frame = self.data[self.pos_i].frame_i
    #         pbar = tqdm(desc='Pre-initializing annotations', total=posData.SizeT, ncols=100)
    #         for frame_i in range(posData.SizeT):
    #             lab = posData.allData_li[frame_i]['labels']
    #             if lab is None:
    #                 lab = posData.segm_data[frame_i]
    #             rp = posData.allData_li[frame_i]['regionprops']
    #             if rp is None:
    #                 rp =  self._acdcRegionProps(lab)
    #                 posData.allData_li[frame_i]['regionprops'] = rp
                    
    #             posData.rp = rp
    #             posData.lab = lab
    #             self.setLostNewOldPrevIDs()
    #             posData.frame_i = frame_i
    #             for ax in self.textAnnot.values():
    #                 ax.setAnnotations(
    #                     posData=posData,
    #                     isVisibleCheckFunc=self._trueFunction,
    #                     highlightedID=self.highlightedID, 
    #                     annotateLost=True,
    #                     getObjCentroidFunc=self.getObjCentroid,
    #                     rp_func=self.get2DRP,
    #                     rp3D=posData.rp,
    #                     updateAllTextAnnotations=True
    #                 )
    #                 ax.update()
    #             pbar.update(1)
    #         posData.frame_i = og_frame
    #         posData.rp = posData.allData_li[og_frame]['regionprops']
    #         posData.lab = posData.allData_li[og_frame]['labels']
    #         pbar.close()
        
    # def _trueFunction(self, *args, **kwargs):
    #     return True
    
    def switchViewedPlane(self, previousPlane, currentPlane):
        posData = self.data[self.pos_i]
        self.xRangePrev, self.yRangePrev = self.ax1.viewRange()
        self.zSlicePrev = self.zSliceScrollBar.sliderPosition()
        
        self.zProjComboBox.setCurrentText('single z-slice')
        depthAxes = self.switchPlaneCombobox.depthAxes()
        self.onEscape()
        if depthAxes != 'z':
            # Disable projections on plane that is not xy
            self.zProjComboBox.setCurrentText('single z-slice')
            self.zProjComboBox.setDisabled(True)
            
            # Clear annotations
            self.clearAllItems()
            self.setHighlightID(False)
            
            # Disable annotations on a plane that is not yz
            self.setDoNotAnnotate(True)
            self.setDisabledAnnotCheckBoxes(0, True)
            self.setDisabledAnnotCheckBoxes(1, True)
            self.setEnabledAnnotCheckBoxesLeftZdepthAxes()
            self.overlayButtonPrevState = self.overlayButton.isChecked()
            self.overlayButton.setChecked(False)
            self.overlayButton.setDisabled(True)
        else:
            self.zProjComboBox.setDisabled(False)
            self.setDisabledAnnotCheckBoxes(0, False)
            self.setDisabledAnnotCheckBoxes(1, False)
            self.overlayButton.setDisabled(False)
            if self.overlayButtonPrevState:
                self.overlayButton.setChecked(self.overlayButtonPrevState)
            self.updateZsliceScrollbar(posData.frame_i)
        
        SizeY, SizeX = posData.img_data[posData.frame_i].shape[-2:]
        
        if depthAxes != 'z' and self.isSnapshot:
            # Disable editing when the plane is not xy
            self.disableEditingViewPlaneNotXY()
        elif self.isSnapshot:
            # Re-enable editing in snapshot mode when the plane is xy
            self.setEnabledSnapshotMode()
        
        if depthAxes == 'z':
            maxSliceNum = posData.SizeZ
        elif depthAxes == 'y':
            maxSliceNum = SizeY
        else:
            maxSliceNum = SizeX
        
        maxSliceText = f'/{maxSliceNum}'
        self.SizeZlabel.setText(maxSliceText)
        self.zSliceCheckbox.setText(f'{depthAxes}-slice')
        self.zSliceScrollBar.setMaximum(maxSliceNum-1)
        self.zSliceSpinbox.setMaximum(maxSliceNum)
        
        self.initContoursImage()
        self.updateAllImages()
        QTimer.singleShot(
            200, partial(self.setViewRangeSwitchPlane, previousPlane)
        )

    def update_z_slice(self, z):
        posData = self.data[self.pos_i]
        if self.switchPlaneCombobox.depthAxes() == 'z': 
            if self.zProjLockViewButton.isChecked():
                idx = [
                    (posData.filename, frame_i) 
                    for frame_i in range(posData.SizeT)
                ]
            else:
                idx = [
                    (posData.filename, frame_i) 
                    for frame_i in range(posData.frame_i, posData.SizeT)
                ]
            posData.segmInfo_df.loc[idx, 'z_slice_used_gui'] = z
                
        self.updatePreprocessPreview()
        self.updateCombineChannelsPreview()
        self.highlightedID = self.getHighlightedID()
        self.updateAllImages(
            computePointsLayers=False, 
            computeContours=False,
            updateLookuptable=True
        )
        self.updateItemsMousePos()
        if self.isSegm3D:
            self.updateObjectCounts()
            self.editIDToolbar.setCurrentZslice(z, posData.SizeZ)

        self.update3DViewerViewRegion()

    def updateZproj(self, how):
        for p, posData in enumerate(self.data[self.pos_i:]):
            if self.zProjLockViewButton.isChecked():
                idx = [
                    (posData.filename, frame_i) 
                    for frame_i in range(posData.SizeT)
                ]
            else:
                idx = [(posData.filename, posData.frame_i)]
            posData.segmInfo_df.loc[idx, 'which_z_proj_gui'] = how
            posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)

        self.setZSliceWidgetsEnabled(how)

        if how == 'single z-slice':
            self.update_z_slice(self.zSliceScrollBar.sliderPosition())
        else:
            self.updateAllImages()

        self.update3DViewerViewRegion()

    def setZSliceWidgetsEnabled(self, how):
        posData = self.data[self.pos_i]
        if how == 'single z-slice':
            self.zSliceScrollBar.setDisabled(False)
            self.zSliceSpinbox.setDisabled(False)
            self.zSliceCheckbox.setDisabled(False)
            self.setZprojDisabled(False)
        else:
            self.zSliceScrollBar.setDisabled(True)
            self.zSliceSpinbox.setDisabled(True)
            self.zSliceCheckbox.setDisabled(True)
            self.setZprojDisabled(self.isSegm3D)

    def setZprojDisabled(self, disabled, storePrevState=False):
        self.combineChannelsAction.setDisabled(disabled)
        for action in self.editToolBar.actions():
            button = self.editToolBar.widgetForAction(action)
            if button == self.eraserButton:
                continue
            
            if button in self.toolsActiveInProj3Dsegm:
                continue

            if action in self.functionsNotTested3D:
                continue
            
            if button in self.functionsNotTested3D:
                continue
            
            try:
                tooltip = button.toolTip()
                prefix = 'WARNING: Disabled due to projection mode\n\n'
                if disabled:
                    if not tooltip.startswith(prefix):
                        button.setToolTip(prefix + tooltip)
                else:
                    if tooltip.startswith(prefix):
                        button.setToolTip(tooltip[len(prefix):])
            except:
                pass

            action.setDisabled(disabled)

            if disabled:
                try:
                    button.setChecked(False)
                except Exception as err:
                    pass

    def navigateSpinboxValueChanged(self, value):
        if self.isSnapshot:        
            self.navigateScrollBar.setSliderPosition(value)
            self.navigateScrollBarStartedMoving = True
            self.PosScrollBarMoved(value)
        else:
            self.navigateScrollBar.setValueNoSignal(value)

    def navigateSpinboxEditingFinished(self):
        if self.isSnapshot:
            self.PosScrollBarReleased()
        else:
            self.framesScrollBarReleased()

    def PosScrollBarAction(self, action):
        if action == gui_main_constants.SliderSingleStepAdd:
            self.next_cb()
        elif action == gui_main_constants.SliderSingleStepSub:
            self.prev_cb()
        elif action == gui_main_constants.SliderPageStepAdd:
            self.PosScrollBarReleased()
        elif action == gui_main_constants.SliderPageStepSub:
            self.PosScrollBarReleased()

    def PosScrollBarMoved(self, pos_n):
        if self.navigateScrollBarStartedMoving:
            self.store_data() 
            
        self.pos_i = pos_n-1
        self.updateFramePosLabel()
        proceed_cca, never_visited = self.get_data()
        self.updateAllImages()
        self.setStatusBarLabel()
        self.navigateScrollBarStartedMoving = False

    def PosScrollBarReleased(self):
        if (
                self.pos_i == self.navigateScrollBar.sliderPosition()-1
                and self.navigateScrollBarStartedMoving
            ):
            # Slider released without changing value --> do nothing
            return
        
        self.navigateScrollBarStartedMoving = True
        self.pos_i = self.navigateScrollBar.sliderPosition()-1
        self.updateFramePosLabel()
        self.updatePos()

    def framesScrollBarActionTriggered(self, action):
        if action == gui_main_constants.SliderSingleStepAdd:
            # Clicking on dialogs triggered by next_cb might trigger
            # pressEvent of navigateQScrollBar, avoid that
            self.navigateScrollBar.disableCustomPressEvent()
            self.next_cb()
            QTimer.singleShot(100, self.navigateScrollBar.enableCustomPressEvent)
        elif action == gui_main_constants.SliderSingleStepSub:
            self.prev_cb()
        elif action == gui_main_constants.SliderPageStepAdd:
            self.framesScrollBarReleased(do_store_data=True)
        elif action == gui_main_constants.SliderPageStepSub:
            self.framesScrollBarReleased(do_store_data=True)

    def restorePrevFrameIfNotProceed(self, proceed_cca, prev_frame_i):
        """Go back to `prev_frame_i` if `get_data` could not load the
        requested frame (`proceed_cca` is False, e.g., frame never checked
        in "Segmentation and Tracking" mode while in "Cell cycle analysis"
        mode). In that case `get_data` returns early and the data of the
        previous frame would otherwise be displayed and stored as the data
        of the requested frame.

        Returns
        -------
        bool
            True if the requested frame was loaded, False if we went back
            to `prev_frame_i`.
        """
        if proceed_cca:
            return True

        posData = self.data[self.pos_i]
        posData.frame_i = prev_frame_i
        self.get_data()
        self.resetNavigateFramesScrollbar()
        self.updateFramePosLabel()
        return False

    def framesScrollBarMoved(self, frame_n):
        if self.navigateScrollBarStartedMoving:
            mode = str(self.modeComboBox.currentText())
            if mode != 'Viewer':
                self.store_data(debug=False)
                
        posData = self.data[self.pos_i]
        prev_frame_i = posData.frame_i
        posData.frame_i = frame_n-1
        proceed_cca, _ = self.get_data()
        if not self.restorePrevFrameIfNotProceed(proceed_cca, prev_frame_i):
            self.navigateScrollBarStartedMoving = False
            return

        self.updateLostNewCurrentIDs()
        self.updateAllImages()
        self.updateFramePosLabel()
        self.updateViewerWindow()
        self.updateTimestampFrame()
        self.updateHighlightedAxis()
        self.navigateScrollBarStartedMoving = False

    def framesScrollBarReleased(self, do_store_data=False):
        posData = self.data[self.pos_i]
        if (
                posData.frame_i == self.navigateScrollBar.sliderPosition()-1
                and self.navigateScrollBarStartedMoving
            ):
            # Slider released without changing value --> do nothing
            return
        
        mode = str(self.modeComboBox.currentText())
        if mode != 'Viewer' and do_store_data:
            self.store_data(debug=False)
            
        self.navigateScrollBarStartedMoving = True
        prev_frame_i = posData.frame_i
        posData.frame_i = self.navigateScrollBar.sliderPosition()-1
        self.updateFramePosLabel()
        proceed_cca, never_visited = self.get_data()
        if not self.restorePrevFrameIfNotProceed(proceed_cca, prev_frame_i):
            return

        if do_store_data:
            self.updateLostNewCurrentIDs()
        self.updateLostNewCurrentIDs()
        self.updateAllImages()
        self.update3DViewer()
        if do_store_data: 
            self.updateFramePosLabel()
            self.updateViewerWindow()
            self.updateHighlightedAxis()

    def setTextAnnotZsliceScrolling(self):
        pass

    def setGraphicalAnnotZsliceScrolling(self):
        posData = self.data[self.pos_i]
        if self.isSegm3D:
            self.currentLab2D = posData.lab[self.z_lab()]
            self.setOverlaySegmMasks()
            self.doCustomAnnotation(0)
            self.update_rp_metadata()
        else:
            self.currentLab2D = posData.lab
            self.setOverlaySegmMasks()
        self.updateContoursImage(0)
        self.updateContoursImage(1)

    def updateZsliceScrollbar(self, frame_i):
        posData = self.data[self.pos_i]
        if self.switchPlaneCombobox.depthAxes() != 'z':
            return
            
        idx = (posData.filename, frame_i)
        try:
            z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
        except ValueError as e:
            z = posData.segmInfo_df.loc[idx, 'z_slice_used_gui'].iloc[0] 
        try:
            zProjHow = posData.segmInfo_df.at[idx, 'which_z_proj_gui']
        except ValueError as e:
            zProjHow = posData.segmInfo_df.loc[idx, 'which_z_proj_gui'].iloc[0] 
        
        self.zProjComboBox.setCurrentText(zProjHow)
        self.setZSliceWidgetsEnabled(zProjHow)
        
        reconnect = False
        try:
            self.zSliceScrollBar.actionTriggered.disconnect()
            self.zSliceScrollBar.sliderReleased.disconnect()
            reconnect = True
        except TypeError:
            pass
        self.zSliceScrollBar.setSliderPosition(z)
        if reconnect:
            self.zSliceScrollBar.actionTriggered.connect(
                self.zSliceScrollBarActionTriggered
            )
            self.zSliceScrollBar.sliderReleased.connect(
                self.zSliceScrollBarReleased
            )
        self.zSliceSpinbox.setValueNoEmit(z+1)

    def updateFramePosLabel(self):
        if self.isSnapshot:
            posData = self.data[self.pos_i]
            self.navSpinBox.setValueNoEmit(self.pos_i+1)
        else:
            posData = self.data[0]
            self.navSpinBox.setValueNoEmit(posData.frame_i+1)

    def isFirstTimeOnNextFrame(self):
        posData = self.data[self.pos_i]
        posData.last_tracked_i = self.navigateScrollBar.maximum()-1
        return posData.frame_i > posData.last_tracked_i
