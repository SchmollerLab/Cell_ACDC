"""Mode switching of the main GUI and the state of the tool buttons. Mixins
of guiWin (gui/main_window/window.py) only: the measurements viewer
never composes them.
"""
import re
from tqdm import tqdm
from qtpy.QtCore import QTimer
from cellacdc import disableWindow
from cellacdc import html_utils
from cellacdc import widgets
from cellacdc import _warnings
from . import constants as gui_main_constants


class ModesGuiElements:
    """Mode switching (changeMode, snapshot and timelapse modes,
    segmentation and tracking init), enabling of toolbars and buttons,
    (dis)connection of the left-click tool buttons, annotation options
    per mode and the keep-tool-active toggles.
    """

    def keepToolActiveActionToggled(self, checked, toolName=None):
        if toolName is None:
            parentToolButton = self.sender().parent()
            toolName = re.findall(r'Name: (.*)', parentToolButton.toolTip())[0]

        if checked:
            self.df_settings.at[toolName, 'value'] = 'keepActive'
        else:
            self.df_settings = self.df_settings.drop(
                index=toolName, errors='ignore'
            )
        self.df_settings.to_csv(self.settings_csv_path)

    def applyToolNewFrameActionToggled(self, checked, toolName=None):
        if toolName is None:
            parentToolButton = self.sender().parent()
            toolName = re.findall(r'Name: (.*)', parentToolButton.toolTip())[0]
        toolName = toolName.strip()
        button = self.applyToolNewFrameButtons[toolName]
        toolName = toolName.replace(' ', '_')
        settingName = f'{toolName}_applyNewFrame'
        if checked:
            self.df_settings.at[settingName, 'value'] = 'applyNewFrame'
            button.setStyleSheet(f'background-color: {gui_main_constants.GREEN_HEX}')
        else:
            self.df_settings = self.df_settings.drop(
                index=settingName, errors='ignore'
            )
            button.setStyleSheet('background-color: none')
        self.df_settings.to_csv(self.settings_csv_path)

    def keepAllToolsActiveActionToggled(self, checked):
        for action in self.keepToolActiveActions.values():
            action.setChecked(checked)

        data_loaded = True
        if not hasattr(self, 'data'):
            data_loaded = False
            try:
                self.labelRoiTrangeCheckbox.disconnect()
            except TypeError:
                pass
        self.labelRoiTrangeCheckbox.setChecked(checked) # why this is not wrapped in a QAction?

        if data_loaded:
            self.labelRoiTrangeCheckbox.toggled.connect(
                self.labelRoiTrangeCheckboxToggled
            )

    def uncheckQButton(self, button):
        # Manual exclusive where we allow to uncheck all buttons
        for b in self.checkableQButtonsGroup.buttons():
            if b != button:
                b.setChecked(False)

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

    def enableSizeSpinbox(self, enabled):
        self.brushSizeLabelAction.setVisible(enabled)
        self.brushSizeAction.setVisible(enabled)
        self.brushAutoFillAction.setVisible(enabled)
        self.brushAutoHideAction.setVisible(enabled)
        self.brushEraserToolBar.setVisible(enabled)        
        self.setNonFunctionalButtonsDisabled()

    def clearComboBoxFocus(self, mode):
        # Remove focus from modeComboBox to avoid the key_up changes its value
        self.sender().clearFocus()
        try:
            self.timer.stop()
            self.modeComboBox.setStyleSheet('background-color: none')
        except Exception as e:
            pass

    def updateModeMenuAction(self):
        self.modeActionGroup.triggered.disconnect()
        for action in self.modeActionGroup.actions():
            if action.text() != self.modeComboBox.currentText():
                continue
            action.setChecked(True)
            break
        self.modeActionGroup.triggered.connect(self.changeModeFromMenu)

    def changeModeFromMenu(self, action):
        self.modeComboBox.setCurrentText(action.text())

    def restorePrevAnnotOptions(self):
        if self.prevAnnotOptions is None:
            return
        
        self.restoreAnnotOptions(0, options=self.prevAnnotOptions)
        self.prevAnnotOptions = None

    def uncheckAllButtonsFromButtonGroup(self, buttonGroup):
        for button in buttonGroup.buttons():
            if not button.isCheckable():
                continue
            
            if not button.isChecked():
                continue
            
            button.setChecked(False)

    @disableWindow
    def changeMode(self, text):
        self.reconnectUndoRedo()
        self.updateModeMenuAction()
        self.clearCustomAnnot()
        posData = self.data[self.pos_i]
        mode = text
        prevMode = self.modeComboBox.previousText()
        self.annotateToolbar.setVisible(False)
        if mode not in ('Segmentation and Tracking', 'Snapshot'):
            # Its segmentation edits (and their annotations warning) do
            # not belong in the other modes
            self.closePostProcessSegmWin()
        if prevMode != 'Viewer' and not self.isCancellingModeInit:
            self.store_data(autosave=True)
            
        self.copyLostObjButton.setChecked(False)
        self.stopCcaIntegrityCheckerWorker()
        self.setAutoSaveSegmentationEnabled(False)
        self.setAutoSaveAnnotationsEnabled(False)
        if prevMode == 'Normal division: Lineage tree':
            # The combo box already shows the new mode: check the pending
            # lineage tree edits of the mode being left (Cancel keeps them
            # without propagating, it does not cancel the mode switch)
            try:
                self.askLineageTreeChanges(isLeavingMode=True)
            finally:
                self.lineage_tree = None
                self.clearLineageTreeBackup()
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
            self.modeToolBar.setVisible(True)
            self.realTimeTrackingToggle.setDisabled(True)
            self.realTimeTrackingToggle.label.setDisabled(True)
            # RAWR!!!!!
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
        elif mode == 'Snapshot':
            self.setAutoSaveAnnotationsEnabled(True)
            self.setSwitchViewedPlaneDisabled(False)
            self.reconnectUndoRedo()
            self.setEnabledSnapshotMode()
            self.doCustomAnnotation(0)
        elif mode == 'Normal division: Lineage tree': # Mode activation for lineage tree
            # self.startLinTreeIntegrityCheckerWorker() # need to replace (postponed)
            proceed = self.initLinTree()
            self.setEnabledCcaToolbar(enabled=False)
            self.setNavigateScrollBarMaximum()
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
        
        self.setNonFunctionalButtonsDisabled()

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
        self.setNonFunctionalButtonsDisabled()
        self.reinitLastSegmFrameAction.setVisible(False)

    def disconnectLeftClickButtons(self):
        for button in self.leftClickButtons:
            try:
                button.toggled.disconnect()
            except Exception as e:
                # Not all the LeftClickButtons have toggled connected
                pass

    def uncheckLeftClickButtons(self, sender):
        for button in self.leftClickButtons:
            if button != sender:
                button.setChecked(False)

        isBrushStrokesOrBoxPromptON = (
            self.magicPromptsToolButton.isChecked()
            and self.magicPromptsToolbar.promptType() in ('brush_stroke', 'box')
        )
        isOtherTool = (
            sender is not None
            and sender is not self.magicPromptsToolButton
            and sender not in self.promptPointsLayerClickButtons()
        )
        if isBrushStrokesOrBoxPromptON and isOtherTool:
            # Activating another left-click tool (e.g., brush) deactivates
            # painting brush strokes or drawing box prompts (W is not a
            # left-click button)
            self.magicPromptsToolButton.setChecked(False)

        if button != self.labelRoiButton:
            # self.labelRoiButton is disconnected so we manually call uncheck
            self.labelRoi_cb(False)
        self.secondLevelToolbar.setVisible(True)
        for toolbar in self.controlToolBars:
            try:
                toolbar.keepVisibleWhenActive
                if toolbar.isVisible():
                    self.secondLevelToolbar.setVisible(False)
                    continue
            except:
                pass
            toolbar.setVisible(False) 
        
        self.enableSizeSpinbox(False)
        if sender is not None:
            self.keepIDsButton.setChecked(False)

    def connectLeftClickButtonsPointsLayersToolbar(self):
        for toolbar in self.pointsLayersToolbars:
            for action in toolbar.actions()[1:]:
                if not hasattr(action, 'layerTypeIdx'):
                    continue
                if action.layerTypeIdx != 4:
                    continue
                action.button.toggled.connect(
                    self.addPointsByClickingButtonToggled
                )

    def connectLeftClickButtons(self):
        self.brushButton.toggled.connect(self.Brush_cb)
        self.curvToolButton.toggled.connect(self.curvTool_cb)
        self.rulerButton.toggled.connect(self.ruler_cb)
        self.eraserButton.toggled.connect(self.Eraser_cb)
        self.wandToolButton.toggled.connect(self.wand_cb)
        self.labelRoiButton.toggled.connect(self.labelRoi_cb)
        # NOTE: `magicPromptsToolButton` is not in `leftClickButtons`, hence
        # it is never disconnected --> do not connect it here again (it
        # would call `magicPrompts_cb` multiple times per toggle)
        self.clearFreehandRoiButton.toggled.connect(self.clearFreehandRoi_cb)
        self.expandLabelToolButton.toggled.connect(self.expandLabelCallback)
        self.addDelPolyLineRoiButton.toggled.connect(self.addDelPolyLineRoi_cb)
        self.manualBackgroundButton.toggled.connect(self.manualBackground_cb)
        self.whitelistIDsButton.toggled.connect(self.whitelistIDs_cb)
        self.zoomRectButton.toggled.connect(self.zoomRectActionToggled)
        self.mergeIDsButton.toggled.connect(self.mergeIDs_cb)
        self.connectLeftClickButtonsPointsLayersToolbar()

    def setUncheckedAllButtons(self, buttonsToNotUncheck=None):
        self.clickedOnBud = False
        if buttonsToNotUncheck is None:
            buttonsToNotUncheck = set()
            
        try:
            self.BudMothTempLine.setData([], [])
        except Exception as e:
            pass
        for button in self.checkableButtons:
            if button in buttonsToNotUncheck:
                continue
            button.setChecked(False)
        
        if self.countObjsButton not in buttonsToNotUncheck:
            self.countObjsButton.setChecked(False)
        self.splineHoverON = False
        self.tempSegmentON = False
        self.isRightClickDragImg1 = False
        self.clearCurvItems(removeItems=False)

    def activateAnnotations(self):
        if self.annotContourCheckbox(0).isChecked():
            return
        
        if self.annotOverlaySegmMaskCheckbox(0).isChecked():
            return
        
        self.annotOverlaySegmMaskCheckbox(0).setChecked(True)

    def setNonFunctionalButtonsDisabled(self):
        if self.isSegm3D:
            disabledTooltip = (
                'Disabled for 3D. If this sounds useful, please contact us '
                'so we can implement it!'
            )

            def addDisabledTooltip(widget):
                if widget is None:
                    return
                currentTooltip = widget.toolTip()
                if disabledTooltip in currentTooltip:
                    return
                newTooltip = (
                    f'{disabledTooltip}\n\n{currentTooltip}'
                    if currentTooltip else disabledTooltip
                )
                widget.setToolTip(newTooltip)

            for item in self.functionsNotTested3D:
                if hasattr(item, 'action'):
                    toolButton = item
                    action = toolButton.action
                    toolButton.setDisabled(True)
                elif hasattr(item, 'toolbar'):
                    toolbar = item.toolbar
                    action = item
                    toolButton = toolbar.widgetForAction(action)
                    toolButton.setDisabled(True)
                else:
                    action = item
                    toolButton = None

                action.setDisabled(True)
                addDisabledTooltip(action)
                addDisabledTooltip(toolButton)
        else:
            def restoreTooltip(widget):
                if widget is None:
                    return
                try:
                    widget.setToolTip(widget._tooltip)
                except Exception as err:
                    pass
                
            for item in self.functionsNotTested3D:
                if hasattr(item, 'action'):
                    toolButton = item
                    action = toolButton.action
                    toolButton.setDisabled(False)
                elif hasattr(item, 'toolbar'):
                    toolbar = item.toolbar
                    action = item
                    toolButton = toolbar.widgetForAction(action)
                    toolButton.setDisabled(False)
                else:
                    action = item
                    toolButton = None

                action.setDisabled(False)
                restoreTooltip(action)
                restoreTooltip(toolButton)

    def setFramesSnapshotMode(self):
        self.measurementsMenu.setDisabled(False)
        self.setPermanentGreedyCmapPreferences()
        if self.isSnapshot:
            self.realTimeTrackingToggle.setDisabled(True)
            self.realTimeTrackingToggle.label.setDisabled(True)
            
            self.imgGrad.rescaleAcrossTimeAction.setDisabled(True)
            self.repeatTrackingAction.setDisabled(True)
            self.manualTrackingAction.setDisabled(True)
            self.logger.info('Setting GUI mode to "Snapshots"...')
            self.modeComboBox.clear()
            self.modeComboBox.addItems(['Snapshot'])
            self.modeComboBox.setDisabled(True)
            self.modeMenu.menuAction().setVisible(False)
            self.modeToolBar.setVisible(False)
            self.skipToNewIdAction.setVisible(False)
            self.skipToNewIdAction.setDisabled(True)
            self.modeComboBox.setCurrentText('Snapshot')
            self.annotateToolbar.setVisible(True)
            self.labelsGrad.showNextFrameAction.setDisabled(True)
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
            self.manualBackgroundButton.setVisible(True)
            self.manualBackgroundButton.setDisabled(False)
            self.manualBackgroundButton.setCheckable(True)
            self.editObjectTrackSettingsAction
            self.showFrameIndexAction.setVisible(False)
            self.editObjectTrackSettingsAction.setVisible(False)
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
            except Exception as e:
                pass
                # traceback.print_exc()
            self.modeComboBox.clear()
            self.modeComboBox.addItems(self.modeItems)
            self.modeComboBox.sigTextChanged.connect(self.changeMode)
            self.modeComboBox.activated.connect(self.clearComboBoxFocus)
            self.modeComboBox.setCurrentText('Viewer')
            
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
            self.manualBackgroundButton.setVisible(False)
            self.manualBackgroundButton.setDisabled(True)
            self.manualBackgroundButton.setCheckable(False)
            self.showFrameIndexAction.setVisible(True)
            self.editObjectTrackSettingsAction.setVisible(True)
            
        for ch, overlayItems in self.overlayLayersItems.items():
            lutItem = overlayItems[1]
            lutItem.rescaleAcrossTimeAction.setDisabled(self.isSnapshot)      

    def initSegmTrackMode(self):
        posData = self.data[self.pos_i]
        last_tracked_i = self.get_last_tracked_i()
        
        if posData.frame_i > last_tracked_i:
            # Prompt user to go to last tracked frame
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                f'The last visited frame in "Segmentation and Tracking mode" '
                f'is frame {last_tracked_i+1}.\n\n'
                f'We recommend to resume from that frame.<br><br>'
                'How do you want to proceed?'
            )
            goToButton, stayButton = msg.warning(
                self, 'Go to last visited frame?', txt,
                buttonsTexts=(
                    f'Resume from frame {last_tracked_i+1} (RECOMMENDED)',
                    f'Stay on current frame {posData.frame_i+1}'
                )
            )
            if msg.clickedButton == goToButton:
                posData.frame_i = last_tracked_i
                self.lastFrameRanOnFirstVisitTools = posData.frame_i
                self.get_data()
                self.updateAllImages()
                self.updateScrollbars()
            else:
                last_tracked_i = posData.frame_i
                current_frame_i = posData.frame_i
                self.lastFrameRanOnFirstVisitTools = posData.frame_i
                self.logger.info(
                    f'Storing data up until frame n. {current_frame_i+1}...'
                )
                pbar = tqdm(total=current_frame_i+1, ncols=100)
                for i in range(current_frame_i):
                    posData.frame_i = i
                    self.get_data()
                    self.store_data(autosave=i==current_frame_i-1)
                    pbar.update()
                pbar.close()

                posData.frame_i = current_frame_i
                self.get_data()
        
        self.highlightLostNew()
        self.updateLastCheckedFrameWidgets(last_tracked_i)

        self.isFirstTimeOnNextFrame()
        self.initRealTimeTracker()

    def updateLastCheckedFrameWidgets(self, last_tracked_i):
        self.navigateScrollBar.setMaximum(last_tracked_i+1)
        self.navSpinBox.setMaximum(last_tracked_i+1)
        self.lastTrackedFrameLabel.setText(
            f'Last checked frame n. = {last_tracked_i+1}'
        )

    def cancelModeInit(self, defaultMode='Viewer'):
        """Switch to `defaultMode` after the initialization of a mode (e.g.,
        "Cell cycle analysis") was cancelled.

        Data is not stored on this mode change because the cancelled mode
        was never active. The mode change that started the initialization
        already stored the data according to the actual previous mode.
        Storing here would mark the current frame as checked, even if it
        was only visited in "Viewer" mode.
        """
        self.isCancellingModeInit = True
        try:
            self.modeComboBox.setCurrentText(defaultMode)
        finally:
            self.isCancellingModeInit = False

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

    def blinkModeComboBox(self):
        if self.flag:
            self.modeComboBox.setStyleSheet('background-color: orange')
        else:
            self.modeComboBox.setStyleSheet('background-color: none')
        self.flag = not self.flag

    def stopBlinkingCB(self):
        self.timer.stop()
        self.modeComboBox.setStyleSheet('background-color: none')

    # def annotGenNumTreeToggled(self, checked):
    #     self.textAnnot[0].setGenNumTreeAnnotationsEnabled(checked)
    
    # def annotLabelIDtreeToggled(self, checked):
    #     self.textAnnot[0].setLabelTreeAnnotationsEnabled(checked)

    def setDoNotAnnotate(self, checked):
        self.annotDoNotAnntoateCheckbox(0).setChecked(checked)
        self.annotDoNotAnntoateCheckbox(1).setChecked(checked)

    def setEnabledAnnotCheckBoxesLeftZdepthAxes(self):
        if not self.isSegm3D:
            return

        self.annotIDsCheckbox(0).setDisabled(False)
        self.annotContourCheckbox(0).setDisabled(False)
        self.annotIDsCheckbox(0).setChecked(True)
        self.annotContourCheckbox(0).setChecked(True)

    def setDisabledAnnotCheckBoxes(self, ax: int, disabled: bool, debug=False):
        self.annotOptionsCheckboxes[ax]['container'].setDisabled(disabled)

    def checkHandleTooManyNewItems(self):
        posData = self.data[self.pos_i]
        num_objects = len(posData.rp)
        if num_objects < 1500:
            return True

        out = _warnings.warnTooManyNewItems(self, num_objects, self)
        cancel, deactivateAnnot = out
        if cancel:
            return False
        
        if deactivateAnnot:
            self.annotIDsCheckbox(0).setChecked(False)
            self.annotIDsCheckbox(1).setChecked(False)
            return True

    def storeAnnotOptionsBeforeModeSwitch(self):
        """Store the current annotation options so that they can be restored
        when leaving the mode (see `restorePrevAnnotOptions`), and uncheck
        "Do not annotate" so the mode-specific options can be set.
        """
        # Only the left image options are set by the modes and restored by
        # `restorePrevAnnotOptions`. The right image options are changed
        # too only when they are linked to the left ones (i.e., right image
        # controls are not independent), otherwise leave them untouched.
        doNotAnnotLeftCheckbox = self.annotDoNotAnntoateCheckbox(0)
        doNotAnnotRightCheckbox = self.annotDoNotAnntoateCheckbox(1)
        isRightLinked = doNotAnnotLeftCheckbox.isLinkedTo(
            doNotAnnotRightCheckbox
        )

        self.prevAnnotOptions = {0: self.getCurrentAnnotOptions(0)}

        # The options are now stored in `prevAnnotOptions`. Clear the
        # options to restore, otherwise unchecking "Do not annotate"
        # (also when it happens because a mutually exclusive option is
        # checked) would restore them while the mode-specific options
        # are being set (see `onDoNotAnnotateChecked`)
        self.annotOptionsToRestore[0] = {}
        if isRightLinked:
            self.annotOptionsToRestore[1] = {}

        # If linked, this unchecks the right image checkbox too
        doNotAnnotLeftCheckbox.setChecked(False)

        # Keep the double space bar state only if the (independent) right
        # image is still not annotated, so that the next double space bar
        # press restores its options
        self.doubleSpaceBarState = (
            self.doubleSpaceBarState and doNotAnnotRightCheckbox.isChecked()
        )
