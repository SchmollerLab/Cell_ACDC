"""Qt view adapter for mode and toolbar state controls."""

from __future__ import annotations

from qtpy.QtCore import QTimer

from cellacdc import disableWindow

from .tool_activation import ToolActivation

class ModeControls(ToolActivation):
    """Extracted from guiWin."""

    def blinkModeComboBox(self):
        if self.flag:
            self.modeComboBox.setStyleSheet('background-color: orange')
        else:
            self.modeComboBox.setStyleSheet('background-color: none')
        self.flag = not self.flag

    def changeMode(self, text):
        self.reconnectUndoRedo()
        self.updateModeMenuAction()
        self.clearCustomAnnot()
        posData = self.data[self.pos_i]
        mode = text
        prevMode = self.modeComboBox.previousText()
        self.annotateToolbar.setVisible(False)
        if prevMode != 'Viewer':
            self.store_data(autosave=True)
            
        self.copyLostObjButton.setChecked(False)
        self.stopCcaIntegrityCheckerWorker()
        self.setAutoSaveSegmentationEnabled(False)
        self.setAutoSaveAnnotationsEnabled(False)
        if prevMode == 'Normal division: Lineage tree':
            self.askLineageTreeChanges()
            self.lineage_tree = None
            self.editLin_TreeBar.setVisible(False)
            self.uncheckAllButtonsFromButtonGroup(self.editLin_TreeGroup)

        elif prevMode == 'Cell cycle analysis':
            self.setEnabledCcaToolbar(enabled=False)

        if mode == 'Segmentation and Tracking':
            self.setAutoSaveSegmentationEnabled(True)
            self.setSwitchViewedPlaneDisabled(True)
            self.trackingMenu.setDisabled(False)
            self.modeToolBar.setVisible(True)
            self.lastTrackedFrameLabel.setText('')
            self.initSegmTrackMode()
            self.setEnabledEditToolbarButton(enabled=True)
            self.addExistingDelROIs()
            self.isFirstTimeOnNextFrame()
            self.setEnabledCcaToolbar(enabled=False)
            self.clearComputedContours()
            self.realTimeTrackingToggle.setDisabled(False)
            self.realTimeTrackingToggle.label.setDisabled(False)
            if posData.cca_df is not None:
                self.store_cca_df()
            self.restorePrevAnnotOptions()
            self.whitelistViewOGIDs(False)
        elif mode == 'Cell cycle analysis':
            self.setAutoSaveAnnotationsEnabled(True)
            self.setSwitchViewedPlaneDisabled(True)
            self.startCcaIntegrityCheckerWorker()
            proceed = self.initCca()
            if proceed:
                self.applyDelROIs()
            self.modeToolBar.setVisible(True)
            self.realTimeTrackingToggle.setDisabled(True)
            self.realTimeTrackingToggle.label.setDisabled(True)
            self.computeAllContours()
            # RAWR!!!!!
            # self.computeAllObjToObjCostPairs()
            if proceed:
                self.setEnabledEditToolbarButton(enabled=False)
                if self.isSnapshot:
                    self.editToolBar.setVisible(True)
                self.setEnabledCcaToolbar(enabled=True)
                self.removeAlldelROIsCurrentFrame()
                self.setAnnotOptionsCcaMode()
                self.clearGhost()
        elif mode == 'Viewer':
            self.autoSaveTimer.stop()
            self.setSwitchViewedPlaneDisabled(False)
            self.modeToolBar.setVisible(True)
            self.realTimeTrackingToggle.setDisabled(True)
            self.realTimeTrackingToggle.label.setDisabled(True)
            self.setEnabledEditToolbarButton(enabled=False)
            self.setEnabledCcaToolbar(enabled=False)
            self.removeAlldelROIsCurrentFrame()
            self.setStatusBarLabel()
            self.navigateScrollBar.setMaximum(posData.SizeT)
            self.navSpinBox.setMaximum(posData.SizeT)
            self.clearGhost()
            self.computeAllContours()
        elif mode == 'Custom annotations':
            self.setAutoSaveAnnotationsEnabled(True)
            self.setSwitchViewedPlaneDisabled(True)
            self.modeToolBar.setVisible(True)
            self.realTimeTrackingToggle.setDisabled(True)
            self.realTimeTrackingToggle.label.setDisabled(True)
            self.setEnabledEditToolbarButton(enabled=False)
            self.setEnabledCcaToolbar(enabled=False)
            self.removeAlldelROIsCurrentFrame()
            self.annotateToolbar.setVisible(True)
            self.clearGhost()
            self.doCustomAnnotation(0)
            self.computeAllContours()
        elif mode == 'Snapshot':
            self.setAutoSaveAnnotationsEnabled(True)
            self.setSwitchViewedPlaneDisabled(False)
            self.reconnectUndoRedo()
            self.setEnabledSnapshotMode()
            self.doCustomAnnotation(0)
            self.clearComputedContours()
        elif mode == 'Normal division: Lineage tree': # Mode activation for lineage tree
            # self.startLinTreeIntegrityCheckerWorker() # need to replace (postponed)
            proceed = self.initLinTree()
            self.setEnabledCcaToolbar(enabled=False)
            self.setNavigateScrollBarMaximum()
            if proceed:
                self.applyDelROIs()
            self.modeToolBar.setVisible(True)
            self.realTimeTrackingToggle.setDisabled(True)
            self.realTimeTrackingToggle.label.setDisabled(True)
            if proceed:
                self.setAutoSaveAnnotationsEnabled(True)
                self.setEnabledEditToolbarButton(enabled=False)
                if self.isSnapshot:
                    self.editToolBar.setVisible(True)
                self.removeAlldelROIsCurrentFrame()
                self.setAnnotOptionsLin_treeMode()
                self.clearGhost()
                self.editLin_TreeBar.setVisible(True)
        
        self.disableNonFunctionalButtons()

    def changeModeFromMenu(self, action):
        self.modeComboBox.setCurrentText(action.text())

    def clearComboBoxFocus(self, mode):
        # Remove focus from modeComboBox to avoid the key_up changes its value
        self.sender().clearFocus()
        try:
            self.timer.stop()
            self.modeComboBox.setStyleSheet('background-color: none')
        except Exception as e:
            pass

    def disableEditingViewPlaneNotXY(self):
        posData = self.data[self.pos_i]
        self.manuallyEditCcaAction.setDisabled(True)
        for action in self.segmActions:
            action.setDisabled(True)
        if posData.SizeT == 1:
            self.segmVideoMenu.setDisabled(True)
        self.postProcessSegmAction.setDisabled(True)
        self.autoSegmAction.setDisabled(True)
        self.ccaToolBar.setVisible(False)
        self.editToolBar.setVisible(False)
        for action in self.ccaToolBar.actions():
            button = self.editToolBar.widgetForAction(action)
            if button is not None:
                button.setDisabled(True)
            action.setVisible(False)
        for action in self.editToolBar.actions():
            button = self.editToolBar.widgetForAction(action)
            action.setVisible(False)
            if button is not None:
                button.setDisabled(True)

    def enableSizeSpinbox(self, enabled):
        self.brushSizeLabelAction.setVisible(enabled)
        self.brushSizeAction.setVisible(enabled)
        self.brushAutoFillAction.setVisible(enabled)
        self.brushAutoHideAction.setVisible(enabled)
        self.brushEraserToolBar.setVisible(enabled)        
        self.disableNonFunctionalButtons()

    def nonViewerEditMenuOpened(self):
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            self.startBlinkingModeCB()

    def reconnectUndoRedo(self):
        try:
            self.undoAction.triggered.disconnect()
            self.redoAction.triggered.disconnect()
        except Exception as e:
            pass
        mode = self.modeComboBox.currentText()
        if mode == 'Segmentation and Tracking' or mode == 'Snapshot':
            self.undoAction.triggered.connect(self.undo)
            self.redoAction.triggered.connect(self.redo)
        elif mode == 'Cell cycle analysis':
            self.undoAction.triggered.connect(self.UndoCca)
        elif mode == 'Custom annotations':
            self.undoAction.triggered.connect(self.undoCustomAnnotation)
        else:
            self.undoAction.setDisabled(True)
            self.redoAction.setDisabled(True)

    def restorePrevAnnotOptions(self):
        if self.prevAnnotOptions is None:
            return
        self.restoreAnnotOptions_ax1(options=self.prevAnnotOptions)
        self.setDrawAnnotComboboxText()
        self.prevAnnotOptions = None

    def setEnabledCcaToolbar(self, enabled=False):
        self.manuallyEditCcaAction.setDisabled(False)
        self.viewCcaTableAction.setDisabled(False)
        self.ccaToolBar.setVisible(enabled)
        for action in self.ccaToolBar.actions():
            button = self.ccaToolBar.widgetForAction(action)
            action.setVisible(enabled)
            button.setEnabled(enabled)

    def setEnabledEditToolbarButton(self, enabled=False):
        for action in self.segmActions:
            action.setEnabled(enabled)

        for action in self.segmActionsVideo:
            action.setEnabled(enabled)

        self.relabelSequentialAction.setEnabled(enabled)
        self.repeatTrackingMenuAction.setEnabled(enabled)
        self.repeatTrackingVideoAction.setEnabled(enabled)
        self.postProcessSegmAction.setEnabled(enabled)
        self.autoSegmAction.setEnabled(enabled)
        self.editToolBar.setVisible(enabled)
        mode = self.modeComboBox.currentText()
        ccaON = mode == 'Cell cycle analysis'
        for action in self.editToolBar.actions():
            button = self.editToolBar.widgetForAction(action)
            # Keep binCellButton active in cca mode
            if button==self.binCellButton and not enabled and ccaON:
                action.setVisible(True)
                button.setEnabled(True)
            else:
                action.setVisible(enabled)
                button.setEnabled(enabled)
        if not enabled:
            self.setUncheckedAllButtons()

    def setEnabledFileToolbar(self, enabled):
        for action in self.fileToolBar.actions():
            button = self.fileToolBar.widgetForAction(action)
            if action == self.openFolderAction or action == self.newAction:
                continue
            if action == self.manageVersionsAction:
                continue
            if action == self.openFileAction:
                continue
            action.setEnabled(enabled)
            button.setEnabled(enabled)

    def setEnabledSnapshotMode(self):
        posData = self.data[self.pos_i]
        self.manuallyEditCcaAction.setDisabled(False)
        self.viewCcaTableAction.setDisabled(False)
        for action in self.segmActions:
            action.setDisabled(False)

        self.segmVideoMenu.setDisabled(True)
        self.trackingMenu.setDisabled(True)
        self.modeToolBar.setVisible(False)
        
        self.relabelSequentialAction.setDisabled(False)
        self.postProcessSegmAction.setDisabled(False)
        self.autoSegmAction.setDisabled(False)
        self.ccaToolBar.setVisible(True)
        self.editToolBar.setVisible(True)
        self.reinitLastSegmFrameAction.setVisible(False)
        for action in self.ccaToolBar.actions():
            button = self.ccaToolBar.widgetForAction(action)
            if button == self.assignBudMothButton:
                button.setDisabled(False)
                action.setVisible(True)
            elif action == self.reInitCcaAction:
                action.setVisible(True)
            elif action == self.assignBudMothAutoAction and posData.SizeT==1:
                action.setVisible(True)
        for action in self.editToolBar.actions():
            button = self.editToolBar.widgetForAction(action)
            action.setVisible(True)
            button.setEnabled(True)
        self.realTimeTrackingToggle.setDisabled(True)
        self.realTimeTrackingToggle.label.setDisabled(True)
        self.repeatTrackingAction.setVisible(False)
        self.manualTrackingAction.setVisible(False)
        button = self.editToolBar.widgetForAction(self.repeatTrackingAction)
        button.setDisabled(True)
        button = self.editToolBar.widgetForAction(self.manualTrackingAction)
        button.setDisabled(True)
        self.disableNonFunctionalButtons()
        self.reinitLastSegmFrameAction.setVisible(False)

    def setFramesSnapshotMode(self):
        self.measurementsMenu.setDisabled(False)
        self.setPermanentGreedyCmapPreferences()
        if self.isSnapshot:
            self.realTimeTrackingToggle.setDisabled(True)
            self.realTimeTrackingToggle.label.setDisabled(True)
            try:
                self.drawIDsContComboBox.currentIndexChanged.disconnect()
            except Exception as e:
                pass
            
            self.imgGrad.rescaleAcrossTimeAction.setDisabled(True)
            self.repeatTrackingAction.setDisabled(True)
            self.manualTrackingAction.setDisabled(True)
            self.logger.info('Setting GUI mode to "Snapshots"...')
            self.modeComboBox.clear()
            self.modeComboBox.addItems(['Snapshot'])
            self.modeComboBox.setDisabled(True)
            self.modeMenu.menuAction().setVisible(False)
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
            self.drawIDsContComboBox.setCurrentIndex(1)
            self.modeToolBar.setVisible(False)
            self.skipToNewIdAction.setVisible(False)
            self.skipToNewIdAction.setDisabled(True)
            self.modeComboBox.setCurrentText('Snapshot')
            self.annotateToolbar.setVisible(True)
            self.labelsGrad.showNextFrameAction.setDisabled(True)
            self.drawIDsContComboBox.currentIndexChanged.connect(
                self.drawIDsContComboBox_cb
            )
            self.showTreeInfoCheckbox.hide()
            self.rightImageFramesScrollbar.setVisible(False)
            self.rightImageFramesScrollbar.setDisabled(True)
            if not self.isSegm3D:
                self.manualBackgroundAction.setVisible(True)
                self.manualBackgroundAction.setDisabled(False)
            else:
                self.manualBackgroundAction.setVisible(False)
                self.manualBackgroundAction.setDisabled(True)
            self.manualAnnotPastButton.setDisabled(True)
            self.manualAnnotPastButton.action.setDisabled(True)
            self.manualAnnotPastButton.setVisible(False)
            self.manualAnnotPastButton.action.setVisible(False)
            self.copyLostObjButton.setDisabled(True)
            self.copyLostObjButton.action.setDisabled(True)
            self.copyLostObjButton.setVisible(False)
            self.copyLostObjButton.action.setVisible(False)
            self.segForLostIDsAction.setVisible(False)
            self.segForLostIDsAction.setDisabled(True)
            self.delNewObjAction.setVisible(False)
            self.delNewObjAction.setDisabled(True)
        else:
            self.imgGrad.rescaleAcrossTimeAction.setDisabled(False)
            self.annotateToolbar.setVisible(False)
            self.realTimeTrackingToggle.setDisabled(False)
            self.repeatTrackingAction.setDisabled(False)
            self.manualTrackingAction.setDisabled(False)
            self.modeComboBox.setDisabled(False)
            self.modeMenu.menuAction().setVisible(True)
            self.skipToNewIdAction.setVisible(True)
            self.skipToNewIdAction.setDisabled(False)
            try:
                self.modeComboBox.activated.disconnect()
                self.modeComboBox.sigTextChanged.disconnect()
                self.drawIDsContComboBox.currentIndexChanged.disconnect()
            except Exception as e:
                pass
                # traceback.print_exc()
            self.modeComboBox.clear()
            self.modeComboBox.addItems(self.modeItems)
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
            self.modeComboBox.sigTextChanged.connect(self.changeMode)
            self.modeComboBox.activated.connect(self.clearComboBoxFocus)
            self.drawIDsContComboBox.currentIndexChanged.connect(
                                                    self.drawIDsContComboBox_cb)
            self.modeComboBox.setCurrentText('Viewer')
            self.showTreeInfoCheckbox.show()
            self.manualBackgroundAction.setVisible(False)
            self.manualBackgroundAction.setDisabled(True)
            self.labelsGrad.showNextFrameAction.setDisabled(False)  
            self.manualAnnotPastButton.setDisabled(False)
            self.manualAnnotPastButton.action.setDisabled(False)
            self.manualAnnotPastButton.setVisible(True)
            self.manualAnnotPastButton.action.setVisible(True)
            self.copyLostObjButton.setDisabled(False)
            self.copyLostObjButton.action.setDisabled(False)
            self.copyLostObjButton.setVisible(True)
            self.copyLostObjButton.action.setVisible(True)
            self.segForLostIDsAction.setVisible(True)
            self.segForLostIDsAction.setDisabled(False)
            self.delNewObjAction.setVisible(True)
            self.delNewObjAction.setDisabled(False)
        
        for ch, overlayItems in self.overlayLayersItems.items():
            lutItem = overlayItems[1]
            lutItem.rescaleAcrossTimeAction.setDisabled(self.isSnapshot)      

    def startBlinkingModeCB(self):
        try:
            self.timer.stop()
            self.stopBlinkTimer.stop()
        except Exception as e:
            pass
        if self.rulerButton.isChecked():
            return
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.blinkModeComboBox)
        self.timer.start(200)
        self.stopBlinkTimer = QTimer(self)
        self.stopBlinkTimer.timeout.connect(self.stopBlinkingCB)
        self.stopBlinkTimer.start(2000)

    def stopBlinkingCB(self):
        self.timer.stop()
        self.modeComboBox.setStyleSheet('background-color: none')

    def uncheckAllButtonsFromButtonGroup(self, buttonGroup):
        for button in buttonGroup.buttons():
            if not button.isCheckable():
                continue
            
            if not button.isChecked():
                continue
            
            button.setChecked(False)

    def updateModeMenuAction(self):
        self.modeActionGroup.triggered.disconnect()
        for action in self.modeActionGroup.actions():
            if action.text() != self.modeComboBox.currentText():
                continue
            action.setChecked(True)
            break
        self.modeActionGroup.triggered.connect(self.changeModeFromMenu)
