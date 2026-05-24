"""Qt view adapter for action and shortcut workflows."""

from __future__ import annotations

import os
import re

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import QAction, QActionGroup, QToolButton

from cellacdc import apps, is_mac, settings_folderpath, widgets

shortcut_filepath = os.path.join(settings_folderpath, "shortcuts.ini")

from .image_display import ImageDisplay


class Actions(ImageDisplay):
    """Extracted from guiWin."""

    def editShortcuts_cb(self):
        if is_mac:
            delObjKeySequenceText = "Ctrl"
            delObjButtonText = "Left click"
        else:
            delObjKeySequenceText = ""
            delObjButtonText = "Middle click"

        if self.delObjAction is not None:
            delObjKeySequence, delObjQtButton = self.delObjAction
            if delObjKeySequence is None:
                delObjKeySequenceText = ""
            else:
                delObjKeySequenceText = delObjKeySequence.toString()
            delObjKeySequenceText = delObjKeySequenceText.encode(
                "ascii", "ignore"
            ).decode("utf-8")
            delObjButtonText = (
                "Left click"
                if delObjQtButton == Qt.MouseButton.LeftButton
                else "Middle click"
            )

        win = apps.ShortcutEditorDialog(
            self.widgetsWithShortcut,
            delObjectKey=delObjKeySequenceText,
            delObjectButton=delObjButtonText,
            zoomOutKeyValue=self.zoomOutKeyValue,
            parent=self,
        )
        win.exec_()
        if win.cancel:
            return

        self.delObjAction = win.delObjAction
        self.zoomOutKeyValue = win.zoomOutKeyValue
        self.setShortcuts(win.customShortcuts)

    def gui_connectActions(self):
        # Connect File actions
        if self.debug:
            self.createEmptyDataAction.triggered.connect(self._createEmptyData)
        self.segmNdimIndicator.clicked.connect(self.segmNdimIndicatorClicked)
        self.newWindowAction.triggered.connect(self.openNewWindow)
        self.newAction.triggered.connect(self.newFile)
        self.openFolderAction.triggered.connect(self.openFolder)
        self.openFileAction.triggered.connect(self.openFile)
        self.manageVersionsAction.triggered.connect(self.manageVersions)
        self.saveAction.triggered.connect(self.saveData)
        self.saveAsAction.triggered.connect(self.saveAsData)
        self.exportToVideoAction.triggered.connect(self.exportToVideoTriggered)
        self.exportToImageAction.triggered.connect(self.exportToImageTriggered)
        self.quickSaveAction.triggered.connect(self.quickSave)
        self.viewPreprocDataToggle.toggled.connect(self.viewPreprocDataToggled)
        self.viewCombineChannelDataToggle.toggled.connect(
            self.viewCombineChannelDataToggled
        )
        self.autoSaveToggle.toggled.connect(self.autoSaveToggled)
        self.autoSaveAnnotToggle.toggled.connect(self.autoSaveAnnotToggled)
        self.autoSaveIntervalDialog.sigValueChanged.connect(
            self.autoSaveIntervalValueChanged
        )
        self.autoSaveIntervalEditButton.clicked.connect(self.autoSaveIntervalEdit)
        self.ccaIntegrCheckerToggle.toggled.connect(self.ccaIntegrCheckerToggled)
        self.annotLostObjsToggle.toggled.connect(self.annotLostObjsToggled)
        self.highLowResAction.clicked.connect(self.highLowResToggled)
        self.showInExplorerAction.triggered.connect(self.showInExplorer_cb)
        self.exitAction.triggered.connect(self.close)
        self.undoAction.triggered.connect(self.undo)
        self.redoAction.triggered.connect(self.redo)
        self.nextAction.triggered.connect(self.nextActionTriggered)
        self.prevAction.triggered.connect(self.prevActionTriggered)

        self.invertBwAction.toggled.connect(self.invertBw)
        self.toggleColorSchemeAction.triggered.connect(self.onToggleColorScheme)
        self.pxModeAction.clicked.connect(self.pxModeActionToggled)
        self.editShortcutsAction.triggered.connect(self.editShortcuts_cb)
        self.editAutoSaveIntervalAction.triggered.connect(
            self.autoSaveIntervalEditButton.click
        )
        self.showMirroredCursorAction.toggled.connect(self.showMirroredCursorToggled)

        # Connect Help actions
        self.tipsAction.triggered.connect(self.showTipsAndTricks)
        self.UserManualAction.triggered.connect(utils.browse_docs)
        self.openLogFileAction.triggered.connect(self.openLogFile)
        self.showLogFilesAction.triggered.connect(self.showLogFiles)
        self.aboutAction.triggered.connect(self.showAbout)
        # Connect Open Recent to dynamically populate it
        # self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)
        self.checkableQButtonsGroup.buttonClicked.connect(self.uncheckQButton)

        self.showPropsDockButton.sigClicked.connect(self.showPropsDockWidget)

        self.loadCustomAnnotationsAction.triggered.connect(self.loadCustomAnnotations)
        self.addCustomAnnotationAction.triggered.connect(self.addCustomAnnotation)
        self.viewAllCustomAnnotAction.toggled.connect(self.viewAllCustomAnnot)
        self.addCustomModelVideoAction.triggered.connect(
            self.showInstructionsCustomModel
        )
        self.addCustomModelFrameAction.triggered.connect(
            self.showInstructionsCustomModel
        )
        self.addCustomModelFrameAction.callback = self.segmFrameCallback
        self.addCustomModelVideoAction.callback = self.segmVideoCallback

        self.addCustomPromptModelAction.triggered.connect(
            self.showInstructionsCustomPromptModel
        )
        self.segmWithPromptableModelAction.triggered.connect(
            self.segmWithPromptableModelActionTriggered
        )

    def gui_connectEditActions(self):
        self.showInExplorerAction.setEnabled(True)
        self.setEnabledFileToolbar(True)
        self.loadFluoAction.setEnabled(True)
        self.isEditActionsConnected = True

        self.preprocessImageAction.triggered.connect(self.preprocessAction.trigger)
        self.combineChannelsAction.triggered.connect(
            self.combineChannelsActionTriggered
        )

        self.overlayButton.toggled.connect(self.overlay_cb)
        self.countObjsButton.toggled.connect(self.countObjectsCb)
        self.togglePointsLayerAction.toggled.connect(self.pointsLayerToggled)
        self.overlayLabelsButton.toggled.connect(self.overlayLabels_cb)
        self.overlayButton.sigRightClick.connect(self.showOverlayContextMenu)
        self.labelRoiButton.sigRightClick.connect(self.showLabelRoiContextMenu)
        self.overlayLabelsButton.sigRightClick.connect(
            self.showOverlayLabelsContextMenu
        )
        self.rulerButton.toggled.connect(self.ruler_cb)
        self.loadFluoAction.triggered.connect(self.loadFluo_cb)
        self.loadPosAction.triggered.connect(self.loadPosTriggered)
        # self.reloadAction.triggered.connect(self.reload_cb)
        self.findIdAction.triggered.connect(self.findID)
        self.zoomRectButton.toggled.connect(self.zoomRectActionToggled)
        self.autoPilotButton.toggled.connect(self.autoPilotToggled)
        self.skipToNewIdAction.triggered.connect(self.skipForwardToNewID)
        self.slideshowButton.toggled.connect(self.launchSlideshow)

        self.copyLostObjButton.toggled.connect(self.copyLostObjContour_cb)
        self.manualAnnotPastButton.toggled.connect(self.manualAnnotPast_cb)

        self.segmSingleFrameMenu.triggered.connect(self.segmFrameCallback)
        self.segmVideoMenu.triggered.connect(self.segmVideoCallback)

        self.postProcessSegmAction.toggled.connect(self.postProcessSegm)
        self.autoSegmAction.toggled.connect(self.autoSegm_cb)
        self.realTimeTrackingToggle.clicked.connect(self.realTimeTrackingClicked)
        self.repeatTrackingAction.triggered.connect(self.repeatTracking)
        self.manualTrackingButton.toggled.connect(self.manualTracking_cb)
        self.manualBackgroundButton.toggled.connect(self.manualBackground_cb)
        self.repeatTrackingMenuAction.triggered.connect(self.repeatTracking)
        self.repeatTrackingVideoAction.triggered.connect(self.repeatTrackingVideo)
        for rtTrackerAction in self.trackingAlgosGroup.actions():
            rtTrackerAction.toggled.connect(self.rtTrackerActionToggled)
        self.editRtTrackerParamsAction.triggered.connect(self.initRealTimeTracker)
        self.delObjsOutSegmMaskAction.triggered.connect(
            self.delObjsOutSegmMaskActionTriggered
        )
        self.mergeIDsButton.toggled.connect(self.mergeObjs_cb)
        self.brushButton.toggled.connect(self.Brush_cb)
        self.eraserButton.toggled.connect(self.Eraser_cb)
        self.curvToolButton.toggled.connect(self.curvTool_cb)
        self.wandToolButton.toggled.connect(self.wand_cb)
        self.labelRoiButton.toggled.connect(self.labelRoi_cb)
        self.magicPromptsToolButton.toggled.connect(self.magicPrompts_cb)
        self.drawClearRegionButton.toggled.connect(self.drawClearRegion_cb)
        self.reInitCcaAction.triggered.connect(self.reInitCca)
        self.moveLabelToolButton.toggled.connect(self.moveLabelButtonToggled)
        self.editCcaToolAction.triggered.connect(
            self.manualEditCcaToolbarActionTriggered
        )
        self.assignBudMothAutoAction.triggered.connect(self.autoAssignBud_YeastMate)
        self.keepIDsButton.toggled.connect(self.keepIDs_cb)

        self.whitelistIDsButton.toggled.connect(self.whitelistIDs_cb)

        self.whitelistIDsToolbar.sigWhitelistChanged.connect(self.whitelistIDsChanged)

        self.whitelistIDsToolbar.sigWhitelistAccepted.connect(self.whitelistIDsAccepted)

        self.whitelistIDsToolbar.sigViewOGIDs.connect(self.whitelistViewOGIDs)

        self.whitelistIDsToolbar.sigAddNewIDs.connect(self.whitelistAddNewIDsToggled)

        self.whitelistIDsToolbar.sigLoadOGLabs.connect(self.whitelistLoadOGLabs_cb)

        self.whitelistIDsToolbar.sigTrackOGagainstPreviousFrame.connect(
            self.whitelistTrackOGagainstPreviousFrame_cb
        )

        self.expandLabelToolButton.toggled.connect(self.expandLabelCallback)

        self.reinitLastSegmFrameAction.triggered.connect(self.reInitLastSegmFrame)

        self.defaultRescaleIntensActionGroup.triggered.connect(
            self.defaultRescaleIntensLutActionToggled
        )

        # self.repeatAutoCcaAction.triggered.connect(self.repeatAutoCca)
        self.manuallyEditCcaAction.triggered.connect(self.manualEditCca)
        self.addScaleBarAction.toggled.connect(self.addScaleBar)
        self.addTimestampAction.toggled.connect(self.addTimestamp)
        self.saveLabColormapAction.triggered.connect(self.saveLabelsColormap)

        self.enableSmartTrackAction.toggled.connect(self.enableSmartTrack)
        # Brush/Eraser size action
        self.brushSizeSpinbox.valueChanged.connect(self.brushSize_cb)
        self.autoIDcheckbox.toggled.connect(self.autoIDtoggled)
        # Mode
        self.modeActionGroup.triggered.connect(self.changeModeFromMenu)
        self.modeComboBox.sigTextChanged.connect(self.changeMode)
        self.modeComboBox.activated.connect(self.clearComboBoxFocus)
        self.equalizeHistPushButton.toggled.connect(self.equalizeHist)

        self.editOverlayColorAction.triggered.connect(self.toggleOverlayColorButton)
        self.editTextIDsColorAction.triggered.connect(self.toggleTextIDsColorButton)
        self.overlayColorButton.sigColorChanging.connect(self.changeOverlayColor)
        self.overlayColorButton.sigColorChanged.connect(self.saveOverlayColor)
        self.textIDsColorButton.sigColorChanging.connect(self.updateTextAnnotColor)
        self.textIDsColorButton.sigColorChanged.connect(self.saveTextIDsColors)

        self.setMeasurementsAction.triggered.connect(self.showSetMeasurements)
        self.addCustomMetricAction.triggered.connect(self.addCustomMetric)
        self.addCombineMetricAction.triggered.connect(self.addCombineMetric)

        self.labelsGrad.colorButton.sigColorChanging.connect(self.updateBkgrColor)
        self.labelsGrad.colorButton.sigColorChanged.connect(self.saveBkgrColor)
        self.labelsGrad.sigGradientChangeFinished.connect(self.updateLabelsCmap)
        self.labelsGrad.sigGradientChanged.connect(self.ticksCmapMoved)
        self.labelsGrad.textColorButton.sigColorChanging.connect(
            self.updateTextLabelsColor
        )
        self.labelsGrad.textColorButton.sigColorChanged.connect(
            self.saveTextLabelsColor
        )
        # self.addFontSizeActions(
        #     self.labelsGrad.fontSizeMenu, self.setFontSizeActionChecked
        # )

        self.labelsGrad.shuffleCmapAction.triggered.connect(self.shuffle_cmap)
        self.labelsGrad.greedyShuffleCmapAction.triggered.connect(
            self.greedyShuffleCmap
        )
        self.labelsGrad.permanentGreedyCmapAction.toggled.connect(
            self.permanentGreedyCmapToggled
        )
        self.shuffleCmapAction.triggered.connect(self.shuffle_cmap)
        self.greedyShuffleCmapAction.triggered.connect(self.greedyShuffleCmap)
        self.labelsGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)
        self.labelsGrad.sigShowLabelsImgToggled.connect(self.showLabelImageItem)
        self.labelsGrad.sigShowRightImgToggled.connect(self.showRightImageItem)
        self.labelsGrad.sigShowNextFrameToggled.connect(self.showNextFrameImageItem)

        self.labelsGrad.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )

        # self.addFontSizeActions(
        #     self.imgGrad.fontSizeMenu, self.setFontSizeActionChecked
        # )
        self.imgGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)
        self.imgGrad.textColorButton.disconnect()
        self.imgGrad.textColorButton.clicked.connect(
            self.editTextIDsColorAction.trigger
        )
        self.imgGrad.labelsAlphaSlider.valueChanged.connect(self.updateLabelsAlpha)
        self.imgGrad.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )

        # Drawing mode
        self.drawIDsContComboBox.currentIndexChanged.connect(
            self.drawIDsContComboBox_cb
        )
        self.drawIDsContComboBox.activated.connect(self.clearComboBoxFocus)

        self.annotateRightHowCombobox.currentIndexChanged.connect(
            self.annotateRightHowCombobox_cb
        )
        self.annotateRightHowCombobox.activated.connect(self.clearComboBoxFocus)

        self.showTreeInfoCheckbox.toggled.connect(self.setAnnotInfoMode)

        # Left
        self.annotIDsCheckbox.clicked.connect(self.annotOptionClicked)
        self.annotCcaInfoCheckbox.clicked.connect(self.annotOptionClicked)
        self.annotContourCheckbox.clicked.connect(self.annotOptionClicked)
        self.annotSegmMasksCheckbox.clicked.connect(self.annotOptionClicked)
        self.drawMothBudLinesCheckbox.clicked.connect(self.annotOptionClicked)
        self.drawNothingCheckbox.clicked.connect(self.annotOptionClicked)
        self.annotNumZslicesCheckbox.clicked.connect(self.annotOptionClicked)

        # Right
        self.annotIDsCheckboxRight.clicked.connect(self.annotOptionClickedRight)
        self.annotCcaInfoCheckboxRight.clicked.connect(self.annotOptionClickedRight)
        self.annotContourCheckboxRight.clicked.connect(self.annotOptionClickedRight)
        self.annotSegmMasksCheckboxRight.clicked.connect(self.annotOptionClickedRight)
        self.drawMothBudLinesCheckboxRight.clicked.connect(self.annotOptionClickedRight)
        self.drawNothingCheckboxRight.clicked.connect(self.annotOptionClickedRight)
        self.annotNumZslicesCheckboxRight.clicked.connect(self.annotOptionClickedRight)

        self.segmentToolAction.triggered.connect(self.segmentToolActionTriggered)

        self.addDelRoiAction.triggered.connect(self.addDelROI)
        self.addDelPolyLineRoiButton.toggled.connect(self.addDelPolyLineRoi_cb)
        self.delBorderObjAction.triggered.connect(self.delBorderObj)
        self.delNewObjAction.triggered.connect(self.delNewObj)

        self.brushAutoFillCheckbox.toggled.connect(self.brushAutoFillToggled)
        self.brushAutoHideCheckbox.toggled.connect(self.brushAutoHideToggled)

        self.imgGrad.sigAddScaleBar.connect(self.addScaleBarAction.setChecked)
        self.imgGrad.sigAddTimestamp.connect(self.addTimestampAction.setChecked)
        self.imgGrad.gradient.sigGradientChangeFinished.connect(
            self.imgGradLUTfinished_cb
        )

        # self.normalizeQActionGroup.triggered.connect(
        #     self.normaliseIntensitiesActionTriggered
        # )
        self.imgPropertiesAction.triggered.connect(self.editImgProperties)

        self.relabelSequentialAction.triggered.connect(self.relabelSequentialCallback)

        self.zoomToObjsAction.triggered.connect(self.zoomToObjsActionCallback)
        self.zoomOutAction.triggered.connect(self.zoomOut)
        self.preprocessAction.triggered.connect(self.preprocessActionTriggered)
        self.combineChannelsAction.triggered.connect(
            self.combineChannelsActionTriggered
        )

        self.viewCcaTableAction.triggered.connect(self.viewCcaTable)

        self.guiTabControl.propsQGBox.idSB.valueChanged.connect(
            self.propsWidgetIDvalueChanged
        )
        self.guiTabControl.highlightCheckbox.toggled.connect(
            self.highlightIDonHoverCheckBoxToggled
        )
        self.guiTabControl.highlightSearchedCheckbox.toggled.connect(
            self.highlightSearchedIDcheckBoxToggled
        )
        intensMeasurQGBox = self.guiTabControl.intensMeasurQGBox
        intensMeasurQGBox.additionalMeasCombobox.currentTextChanged.connect(
            self.updatePropsWidget
        )
        intensMeasurQGBox.channelCombobox.currentTextChanged.connect(
            self.updatePropsWidget
        )

        propsQGBox = self.guiTabControl.propsQGBox
        propsQGBox.additionalPropsCombobox.currentTextChanged.connect(
            self.updatePropsWidget
        )

    def gui_createActions(self):
        # File actions
        self.segmNdimIndicator = widgets.ToolButtonTextIcon(text="")
        self.segmNdimIndicator.setCheckable(True)
        self.segmNdimIndicator.setChecked(True)
        # self.segmNdimIndicator.setDisabled(True)

        if self.debug:
            self.createEmptyDataAction = QAction(self)
            self.createEmptyDataAction.setText("DEBUG: Create empty data")

        self.newWindowAction = QAction("New Window", self)

        self.newAction = QAction(self)
        self.newAction.setText("&New Segmentation File...")
        self.newAction.setIcon(QIcon(":file-new.svg"))
        self.openFolderAction = QAction(
            QIcon(":folder-open.svg"), "&Load Folder...", self
        )
        self.openFileAction = QAction(
            QIcon(":image.svg"), "&Open Image/Video File...", self
        )
        self.manageVersionsAction = QAction(
            QIcon(":manage_versions.svg"), "Load Older Versions...", self
        )
        self.manageVersionsAction.setDisabled(True)
        self.saveAction = QAction(QIcon(":file-save.svg"), "Save", self)
        self.saveAsAction = QAction("Save as...", self)
        self.exportToVideoAction = QAction("&Video...", self)
        self.exportToImageAction = QAction("&Image...", self)
        self.quickSaveAction = QAction("Save Only Segmentation Masks", self)
        self.loadFluoAction = QAction("Load Fluorescence Images...", self)
        self.loadPosAction = QAction("Load Different Position...", self)
        # self.reloadAction = QAction(
        #     QIcon(":reload.svg"), "Reload segmentation file", self
        # )
        self.nextAction = QAction("Next", self)
        self.prevAction = QAction("Previous", self)
        self.showInExplorerAction = QAction(
            QIcon(":drawer.svg"), f"&{self.openFolderText}", self
        )
        self.exitAction = QAction("&Exit", self)
        self.undoAction = QAction(QIcon(":undo.svg"), "Undo", self)
        self.redoAction = QAction(QIcon(":redo.svg"), "Redo", self)
        # String-based key sequences
        self.newWindowAction.setShortcut("Ctrl+Shift+N")
        self.newAction.setShortcut("Ctrl+N")
        self.openFolderAction.setShortcut("Ctrl+O")
        self.loadPosAction.setShortcut("Shift+P")
        self.saveAsAction.setShortcut("Ctrl+Shift+S")
        self.exportToVideoAction.setShortcut("Ctrl+Shift+V")
        self.exportToImageAction.setShortcut("Ctrl+Shift+I")
        self.saveAction.setShortcut("Ctrl+Alt+S")
        self.quickSaveAction.setShortcut("Ctrl+S")
        self.undoAction.setShortcut("Ctrl+Z")
        self.redoAction.setShortcut("Ctrl+Y")
        self.nextAction.setShortcut(Qt.Key_Right)
        self.prevAction.setShortcut(Qt.Key_Left)
        self.addAction(self.nextAction)
        self.addAction(self.prevAction)
        # Help tips
        newTip = "Create a new segmentation file"
        self.newAction.setStatusTip(newTip)
        self.newAction.setWhatsThis("Create a new empty segmentation file")

        self.autoPilotButton = QAction(self)
        self.autoPilotButton.setIcon(QIcon(":auto-pilot.svg"))
        self.autoPilotButton.setCheckable(True)
        self.autoPilotButton.setShortcut("Ctrl+Shift+A")

        self.findIdAction = QAction(self)
        self.findIdAction.setIcon(QIcon(":find.svg"))
        self.findIdAction.setShortcut("Ctrl+F")

        self.zoomRectButton = QToolButton(self)
        self.zoomRectButton.setIcon(QIcon(":zoom_rect.svg"))
        self.zoomRectButton.setCheckable(True)
        self.zoomRectButton.setShortcut("Shift+Z")
        self.LeftClickButtons.append(self.zoomRectButton)
        self.checkableButtons.append(self.zoomRectButton)
        self.checkableQButtonsGroup.addButton(self.zoomRectButton)
        self.widgetsWithShortcut["Zoom to rectangular area"] = self.zoomRectButton

        self.skipToNewIdAction = QAction(self)
        self.skipToNewIdAction.setIcon(QIcon(":skip_forward_new_ID.svg"))
        self.skipToNewIdAction.setShortcut(widgets.KeySequenceFromText(Qt.Key_PageUp))

        self.skipToNewIdAction.setDisabled(True)

        # Edit actions
        models = utils.get_list_of_models()
        models = [*models, "local_seg"]  # Add local_seg for SegForLostIDsAction
        self.segmActions = []
        self.modelNames = []
        self.acdcSegment_li = []
        self.models = []
        for model_name in models:
            action = QAction(f"{model_name}...")
            self.segmActions.append(action)
            self.modelNames.append(model_name)
            self.models.append(None)
            self.acdcSegment_li.append(None)
            action.setDisabled(True)

        self.addCustomModelFrameAction = QAction("Add custom model...", self)
        self.addCustomModelVideoAction = QAction("Add custom model...", self)

        self.segmWithPromptableModelAction = QAction("Select promptable model...", self)
        self.addCustomPromptModelAction = QAction(
            "Add custom promptable model...", self
        )

        self.segmActionsVideo = []
        for model_name in models:
            action = QAction(f"{model_name}...")
            self.segmActionsVideo.append(action)
            action.setDisabled(True)

        self.postProcessSegmAction = QAction("Segmentation post-processing...", self)
        self.postProcessSegmAction.setDisabled(True)
        self.postProcessSegmAction.setCheckable(True)

        self.EditSegForLostIDsSetSettings = QAction(
            "Edit settings for Segmenting lost IDs...", self
        )
        self.EditSegForLostIDsSetSettings.triggered.connect(
            self.SegForLostIDsSetSettings
        )

        self.repeatTrackingAction = QAction(
            QIcon(":repeat-tracking.svg"), "Repeat tracking", self
        )
        self.repeatTrackingAction.setShortcut("Shift+T")
        self.widgetsWithShortcut["Repeat Tracking"] = self.repeatTrackingAction

        self.editRtTrackerParamsAction = QAction(
            "Edit real-time tracker parameters...", self
        )

        self.repeatTrackingMenuAction = QAction(
            "Track current frame with real-time tracker...", self
        )
        self.repeatTrackingMenuAction.setDisabled(True)
        self.repeatTrackingMenuAction.setShortcut("Shift+T")

        self.repeatTrackingVideoAction = QAction(
            "Select a tracker and track multiple frames...", self
        )
        self.repeatTrackingVideoAction.setDisabled(True)
        self.repeatTrackingVideoAction.setShortcut("Alt+Shift+T")

        self.trackingAlgosGroup = QActionGroup(self)
        self.trackWithAcdcAction = QAction("Cell-ACDC", self)
        self.trackWithAcdcAction.setCheckable(True)
        self.trackingAlgosGroup.addAction(self.trackWithAcdcAction)

        self.trackWithYeazAction = QAction("YeaZ", self)
        self.trackWithYeazAction.setCheckable(True)
        self.trackingAlgosGroup.addAction(self.trackWithYeazAction)

        rt_trackers = utils.get_list_of_real_time_trackers()
        for rt_tracker in rt_trackers:
            rtTrackerAction = QAction(rt_tracker, self)
            rtTrackerAction.setCheckable(True)
            self.trackingAlgosGroup.addAction(rtTrackerAction)

        self.trackWithAcdcAction.setChecked(True)
        aliases = utils.aliases_real_time_trackers()

        if "tracking_algorithm" in self.df_settings.index:
            trackingAlgo = self.df_settings.at["tracking_algorithm", "value"]
            if trackingAlgo in aliases:
                trackingAlgo = aliases[trackingAlgo]
            if trackingAlgo == "Cell-ACDC":
                self.trackWithAcdcAction.setChecked(True)
            elif trackingAlgo == "YeaZ":
                self.trackWithYeazAction.setChecked(True)
            else:
                for rtTrackerAction in self.trackingAlgosGroup.actions():
                    if rtTrackerAction.text() == trackingAlgo:
                        rtTrackerAction.setChecked(True)
                        break

        self.setMeasurementsAction = QAction("Set measurements...")
        self.addCustomMetricAction = QAction("Add custom measurement...")
        self.addCombineMetricAction = QAction("Add combined measurement...")

        # Standard key sequence
        # self.copyAction.setShortcut(QKeySequence.StandardKey.Copy)
        # self.pasteAction.setShortcut(QKeySequence.StandardKey.Paste)
        # self.cutAction.setShortcut(QKeySequence.StandardKey.Cut)
        # Help actions
        self.tipsAction = QAction("Tips and tricks...", self)
        self.UserManualAction = QAction("User Documentation...", self)
        self.openLogFileAction = QAction("Open log file...", self)
        self.showLogFilesAction = QAction("Show log files...", self)
        self.aboutAction = QAction("About Cell-ACDC", self)
        # self.aboutAction = QAction("&About...", self)

        # Assign mother to bud button
        self.assignBudMothAutoAction = QAction(self)
        self.assignBudMothAutoAction.setIcon(QIcon(":autoAssign.svg"))
        self.assignBudMothAutoAction.setVisible(False)

        self.editCcaToolAction = QAction(self)
        self.editCcaToolAction.setIcon(QIcon(":edit_cca.svg"))
        # self.editCcaToolAction.setDisabled(True)
        self.editCcaToolAction.setVisible(False)

        self.reInitCcaAction = QAction(self)
        self.reInitCcaAction.setIcon(QIcon(":reinitCca.svg"))
        self.reInitCcaAction.setVisible(False)

        self.toggleColorSchemeAction = QAction("Switch to light theme")
        self.gui_updateSwitchColorSchemeActionText()

        self.pxModeAction = widgets.CheckableAction("Fixed size text annotations")
        self.pxModeAction.setChecked(True)
        pxModeTooltip = (
            "When the text annotations are with fixed size they scale relative "
            "to the object when zooming in/out (fixed size in pixels).\n"
            "This is typically faster to render, but it makes annotations "
            "smaller/larger when zooming in/out, respectively.\n\n"
            "Try activating it to speed up the annotation of many objects "
            "in high resolution mode.\n\n"
            "After activating it, you might need to increase the font size "
            "from the menu on the top menubar `Edit --> Font size`."
        )
        self.pxModeAction.setToolTip(pxModeTooltip)

        self.highLowResAction = widgets.CheckableAction(
            "High resolution text annotations"
        )
        highLowResTooltip = (
            "Resolution of the text annotations. High resolution results "
            "in slower update of the annotations.\n"
            "Not recommended with a number of segmented objects > 500.\n\n"
        )
        self.highLowResAction.setToolTip(highLowResTooltip)

        self.editAutoSaveIntervalAction = QAction(
            "Change autosave interval (minutes or frames)...", self
        )

        self.editShortcutsAction = QAction("Customize keyboard shortcuts...", self)
        self.editShortcutsAction.setShortcut("Ctrl+K")

        self.showMirroredCursorAction = QAction("Show mirrored cursor on images", self)
        self.showMirroredCursorAction.setCheckable(True)
        if "showMirroredCursor" in self.df_settings.index:
            checked = self.df_settings.at["showMirroredCursor", "value"] == "Yes"
            self.showMirroredCursorAction.setChecked(checked)
        else:
            self.showMirroredCursorAction.setChecked(True)
        self.showMirroredCursorAction.setShortcut("Ctrl+M")

        self.editTextIDsColorAction = QAction("Text annotation color...", self)
        self.editTextIDsColorAction.setDisabled(True)

        self.editOverlayColorAction = QAction("Overlay color...", self)
        self.editOverlayColorAction.setDisabled(True)

        self.manuallyEditCcaAction = QAction("Edit cell cycle annotations...", self)
        self.manuallyEditCcaAction.setShortcut("Ctrl+Shift+P")
        self.manuallyEditCcaAction.setDisabled(True)

        self.viewCcaTableAction = QAction("View cell cycle annotations...", self)
        self.viewCcaTableAction.setDisabled(True)
        self.viewCcaTableAction.setShortcut("Ctrl+P")

        self.addScaleBarAction = QAction("Add scale bar", self)
        self.addScaleBarAction.setCheckable(True)

        self.addTimestampAction = QAction("Add timestamp", self)
        self.addTimestampAction.setCheckable(True)

        self.invertBwAction = QAction("Invert black/white", self)
        self.invertBwAction.setCheckable(True)
        checked = self.df_settings.at["is_bw_inverted", "value"] == "Yes"
        self.invertBwAction.setChecked(checked)

        self.shuffleCmapAction = QAction("Randomly shuffle colormap", self)
        self.shuffleCmapAction.setShortcut("Shift+S")

        self.greedyShuffleCmapAction = QAction("Greedily shuffle colormap", self)
        self.greedyShuffleCmapAction.setShortcut("Alt+Shift+S")

        self.saveLabColormapAction = QAction("Save labels colormap...", self)

        self.normalizeRawAction = QAction("Do not normalize. Display raw image", self)
        self.normalizeToFloatAction = QAction(
            "Convert to floating point format with values [0, 1]", self
        )
        # self.normalizeToUbyteAction = QAction(
        #     'Rescale to 8-bit unsigned integer format with values [0, 255]', self)
        self.normalizeRescale0to1Action = QAction("Rescale to [0, 1]", self)
        self.normalizeByMaxAction = QAction("Normalize by max value", self)
        self.normalizeRawAction.setCheckable(True)
        self.normalizeToFloatAction.setCheckable(True)
        # self.normalizeToUbyteAction.setCheckable(True)
        self.normalizeRescale0to1Action.setCheckable(True)
        self.normalizeByMaxAction.setCheckable(True)
        self.normalizeQActionGroup = QActionGroup(self)
        self.normalizeQActionGroup.addAction(self.normalizeRawAction)
        self.normalizeQActionGroup.addAction(self.normalizeToFloatAction)
        # self.normalizeQActionGroup.addAction(self.normalizeToUbyteAction)
        self.normalizeQActionGroup.addAction(self.normalizeRescale0to1Action)
        self.normalizeQActionGroup.addAction(self.normalizeByMaxAction)

        self.preprocessAction = QAction("Pre-processing...", self)
        self.preprocessAction.setShortcut("Alt+Shift+P")

        self.combineChannelsAction = QAction(
            "Combine and manipulate channels and/or segmentation files...", self
        )
        self.combineChannelsAction.setShortcut("Alt+Shift+C")

        self.zoomToObjsAction = QAction("Zoom to objects  (Shortcut: H key)", self)
        self.zoomOutAction = QAction("Zoom out  (Shortcut: double press H key)", self)

        self.relabelSequentialAction = QAction("Relabel IDs sequentially...", self)
        self.relabelSequentialAction.setShortcut("Ctrl+L")
        self.relabelSequentialAction.setDisabled(True)

        self.setLastUserNormAction()

        self.autoSegmAction = QAction("Enable automatic segmentation", self)
        self.autoSegmAction.setCheckable(True)
        self.autoSegmAction.setDisabled(True)

        self.enableSmartTrackAction = QAction(
            "Smart handling of enabling/disabling tracking", self
        )
        self.enableSmartTrackAction.setCheckable(True)
        self.enableSmartTrackAction.setChecked(True)

        self.enableAutoZoomToCellsAction = QAction(
            'Automatic zoom to all cells when pressing "Next/Previous"', self
        )
        self.enableAutoZoomToCellsAction.setCheckable(True)

        self.imgPropertiesAction = QAction("Properties...", self)
        self.imgPropertiesAction.setDisabled(True)

        self.addDelRoiAction = QAction(self)
        self.addDelRoiAction.roiType = "rect"
        self.addDelRoiAction.setIcon(QIcon(":addDelRoi.svg"))

        self.addDelPolyLineRoiButton = QToolButton(self)
        self.addDelPolyLineRoiButton.setCheckable(True)
        self.addDelPolyLineRoiButton.setIcon(QIcon(":addDelPolyLineRoi.svg"))

        self.checkableButtons.append(self.addDelPolyLineRoiButton)
        self.LeftClickButtons.append(self.addDelPolyLineRoiButton)

        self.delBorderObjAction = QAction(self)
        self.delBorderObjAction.setIcon(QIcon(":delBorderObj.svg"))

        self.delNewObjAction = QAction(self)
        self.delNewObjAction.setIcon(QIcon(":delNewObj.svg"))

        self.loadCustomAnnotationsAction = QAction(self)
        self.loadCustomAnnotationsAction.setIcon(QIcon(":load_annotation.svg"))
        self.loadCustomAnnotationsAction.setToolTip(
            "Load previously used custom annotations"
        )

        self.addCustomAnnotationAction = QAction(self)
        self.addCustomAnnotationAction.setIcon(QIcon(":addCustomAnnotation.svg"))
        self.addCustomAnnotationAction.setToolTip("Add custom annotation")
        # self.functionsNotTested3D.append(self.addCustomAnnotationAction)

        self.viewAllCustomAnnotAction = QAction(self)
        self.viewAllCustomAnnotAction.setCheckable(True)
        self.viewAllCustomAnnotAction.setIcon(QIcon(":eye.svg"))
        self.viewAllCustomAnnotAction.setToolTip("Show all custom annotations")

    def gui_updateSwitchColorSchemeActionText(self):
        if self._colorScheme == "dark":
            txt = "Switch to light theme"
        else:
            txt = "Switch to dark theme"
        self.toggleColorSchemeAction.setText(txt)

    def initShortcuts(self):
        from . import config

        cp = config.ConfigParser()
        if os.path.exists(shortcut_filepath):
            cp.read(shortcut_filepath)

        if "keyboard.shortcuts" not in cp:
            cp["keyboard.shortcuts"] = {}

        if cp.has_option("keyboard.shortcuts", "Zoom out"):
            zoomOutKeyValueStr = cp["keyboard.shortcuts"]["Zoom out"]
            try:
                self.zoomOutKeyValue = int(zoomOutKeyValueStr)
            except Exception as err:
                self.logger.warning(
                    f"{zoomOutKeyValueStr} is not a valid key "
                    'zooming out action. Restoring default key "H".'
                )

        if "delete_object.action" not in cp:
            self.delObjAction = None
        else:
            delObjKeySequenceText = cp["delete_object.action"]["Key sequence"]
            delObjButtonText = cp["delete_object.action"]["Mouse button"]
            delObjQtButton = (
                Qt.MouseButton.LeftButton
                if delObjButtonText == "Left click"
                else Qt.MouseButton.MiddleButton
            )
            if not delObjKeySequenceText:
                delObjKeySequence = None
            else:
                delObjKeySequence = widgets.KeySequenceFromText(delObjKeySequenceText)
            self.delObjToolAction.setChecked(True)
            self.delObjAction = delObjKeySequence, delObjQtButton

        shortcuts = {}
        for name, widget in self.widgetsWithShortcut.items():
            if name not in cp.options("keyboard.shortcuts"):
                if hasattr(widget, "keyPressShortcut"):
                    key = widget.keyPressShortcut
                    shortcut = widgets.KeySequenceFromText(key)
                else:
                    shortcut = widget.shortcut()
                shortcut_text = shortcut.toString()
                cp["keyboard.shortcuts"][name] = shortcut_text
            else:
                shortcut_text = cp["keyboard.shortcuts"][name]
                shortcut = widgets.KeySequenceFromText(shortcut_text)

            shortcuts[name] = (shortcut_text, shortcut)
        self.setShortcuts(shortcuts, save=False)
        with open(shortcut_filepath, "w") as ini:
            cp.write(ini)

    def setShortcuts(self, shortcuts: dict, save=True):
        for name, (text, shortcut) in shortcuts.items():
            widget = self.widgetsWithShortcut[name]
            if shortcut is None:
                shortcut = QKeySequence()
            if hasattr(widget, "keyPressShortcut"):
                widget.keyPressShortcut = shortcut
            else:
                widget.setShortcut(shortcut)
            s = widget.toolTip()
            toolTip = re.sub(r'Shortcut: "(.*)"', f'Shortcut: "{text}"', s)
            widget.setToolTip(toolTip)

        if not save:
            return

        from . import config

        cp = config.ConfigParser()
        if os.path.exists(shortcut_filepath):
            cp.read(shortcut_filepath)

        if "keyboard.shortcuts" not in cp:
            cp["keyboard.shortcuts"] = {}

        for name, (text, shortcut) in shortcuts.items():
            cp["keyboard.shortcuts"][name] = text

        cp["keyboard.shortcuts"]["Zoom out"] = str(self.zoomOutKeyValue)

        if self.delObjAction is None:
            with open(shortcut_filepath, "w") as ini:
                cp.write(ini)
            return

        delObjKeySequence, delObjQtButton = self.delObjAction
        try:
            if delObjKeySequence is None:
                delObjKeySequenceText = ""
            else:
                delObjKeySequenceText = delObjKeySequence.toString()

            delObjKeySequenceText = delObjKeySequenceText.encode(
                "ascii", "ignore"
            ).decode("utf-8")
            delObjButtonText = (
                "Left click"
                if delObjQtButton == Qt.MouseButton.LeftButton
                else "Middle click"
            )
            cp["delete_object.action"] = {
                "Key sequence": delObjKeySequenceText,
                "Mouse button": delObjButtonText,
            }
        except Exception as err:
            self.logger.warning(
                f"{delObjKeySequence} is not a valid keys sequence for "
                "deleting objects. Setting default action"
            )
            self.delObjAction = None
            cp.remove_section("delete_object.action")

        with open(shortcut_filepath, "w") as ini:
            cp.write(ini)
