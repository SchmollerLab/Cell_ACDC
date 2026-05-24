"""Qt view adapter for action and shortcut workflows."""

from __future__ import annotations

import os
import re

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import QAction, QActionGroup, QToolButton

from cellacdc import apps, is_mac, myutils, settings_folderpath, widgets

shortcut_filepath = os.path.join(settings_folderpath, "shortcuts.ini")

from .image_display import ImageDisplay


class Actions(ImageDisplay):
    """Extracted from guiWin."""

    def _connect_method_if_present(self, signal, method_name):
        method = getattr(self, method_name, None)
        if method is not None:
            signal.connect(method)

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
            self._connect_method_if_present(
                self.createEmptyDataAction.triggered, "_createEmptyData"
            )
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
        if hasattr(self, "viewCombineChannelDataToggle"):
            self._connect_method_if_present(
                self.viewCombineChannelDataToggle.toggled,
                "viewCombineChannelDataToggled",
            )
        self.autoSaveToggle.toggled.connect(self.autoSaveToggled)
        self.autoSaveAnnotToggle.toggled.connect(self.autoSaveAnnotToggled)
        self.autoSaveIntervalDialog.sigValueChanged.connect(
            self.autoSaveIntervalValueChanged
        )
        self.autoSaveIntervalEditButton.clicked.connect(self.autoSaveIntervalEdit)
        if hasattr(self, "ccaIntegrCheckerToggle"):
            self._connect_method_if_present(
                self.ccaIntegrCheckerToggle.toggled, "ccaIntegrCheckerToggled"
            )
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
        self.UserManualAction.triggered.connect(myutils.browse_docs)
        self.openLogFileAction.triggered.connect(self.openLogFile)
        self.showLogFilesAction.triggered.connect(self.showLogFiles)
        self.aboutAction.triggered.connect(self.showAbout)
        # Connect Open Recent to dynamically populate it
        # self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)
        self.checkableQButtonsGroup.buttonClicked.connect(self.uncheckQButton)

        self._connect_method_if_present(
            self.showPropsDockButton.sigClicked, "showPropsDockWidget"
        )

        self._connect_method_if_present(
            self.loadCustomAnnotationsAction.triggered, "loadCustomAnnotations"
        )
        self._connect_method_if_present(
            self.addCustomAnnotationAction.triggered, "addCustomAnnotation"
        )
        self._connect_method_if_present(
            self.viewAllCustomAnnotAction.toggled, "viewAllCustomAnnot"
        )
        self._connect_method_if_present(
            self.addCustomModelVideoAction.triggered, "showInstructionsCustomModel"
        )
        self._connect_method_if_present(
            self.addCustomModelFrameAction.triggered, "showInstructionsCustomModel"
        )
        if hasattr(self, "segmFrameCallback"):
            self.addCustomModelFrameAction.callback = self.segmFrameCallback
        if hasattr(self, "segmVideoCallback"):
            self.addCustomModelVideoAction.callback = self.segmVideoCallback

        self._connect_method_if_present(
            self.addCustomPromptModelAction.triggered,
            "showInstructionsCustomPromptModel",
        )
        self._connect_method_if_present(
            self.segmWithPromptableModelAction.triggered,
            "segmWithPromptableModelActionTriggered",
        )

    def gui_connectEditActions(self):
        self.showInExplorerAction.setEnabled(True)
        self.setEnabledFileToolbar(True)
        self.loadFluoAction.setEnabled(True)
        self.isEditActionsConnected = True

        if hasattr(self, "preprocessImageAction") and hasattr(self, "preprocessAction"):
            self.preprocessImageAction.triggered.connect(self.preprocessAction.trigger)
        self._connect_method_if_present(
            self.combineChannelsAction.triggered, "combineChannelsActionTriggered"
        )

        self._connect_method_if_present(self.overlayButton.toggled, "overlay_cb")
        self._connect_method_if_present(self.countObjsButton.toggled, "countObjectsCb")
        self._connect_method_if_present(
            self.togglePointsLayerAction.toggled, "pointsLayerToggled"
        )
        self._connect_method_if_present(
            self.overlayLabelsButton.toggled, "overlayLabels_cb"
        )
        self._connect_method_if_present(
            self.overlayButton.sigRightClick, "showOverlayContextMenu"
        )
        self._connect_method_if_present(
            self.labelRoiButton.sigRightClick, "showLabelRoiContextMenu"
        )
        self._connect_method_if_present(
            self.overlayLabelsButton.sigRightClick, "showOverlayLabelsContextMenu"
        )
        self._connect_method_if_present(self.rulerButton.toggled, "ruler_cb")
        self._connect_method_if_present(self.loadFluoAction.triggered, "loadFluo_cb")
        self._connect_method_if_present(self.loadPosAction.triggered, "loadPosTriggered")
        self._connect_method_if_present(self.findIdAction.triggered, "findID")
        self._connect_method_if_present(
            self.zoomRectButton.toggled, "zoomRectActionToggled"
        )
        self._connect_method_if_present(
            self.autoPilotButton.toggled, "autoPilotToggled"
        )
        self._connect_method_if_present(
            self.skipToNewIdAction.triggered, "skipForwardToNewID"
        )
        self._connect_method_if_present(
            self.slideshowButton.toggled, "launchSlideshow"
        )

        self._connect_method_if_present(
            self.copyLostObjButton.toggled, "copyLostObjContour_cb"
        )
        self._connect_method_if_present(
            self.manualAnnotPastButton.toggled, "manualAnnotPast_cb"
        )

        self._connect_method_if_present(
            self.segmSingleFrameMenu.triggered, "segmFrameCallback"
        )
        self._connect_method_if_present(
            self.segmVideoMenu.triggered, "segmVideoCallback"
        )

        self._connect_method_if_present(
            self.postProcessSegmAction.toggled, "postProcessSegm"
        )
        self._connect_method_if_present(self.autoSegmAction.toggled, "autoSegm_cb")
        self._connect_method_if_present(
            self.realTimeTrackingToggle.clicked, "realTimeTrackingClicked"
        )
        self._connect_method_if_present(
            self.repeatTrackingAction.triggered, "repeatTracking"
        )
        self._connect_method_if_present(
            self.manualTrackingButton.toggled, "manualTracking_cb"
        )
        self._connect_method_if_present(
            self.manualBackgroundButton.toggled, "manualBackground_cb"
        )
        self._connect_method_if_present(
            self.repeatTrackingMenuAction.triggered, "repeatTracking"
        )
        self._connect_method_if_present(
            self.repeatTrackingVideoAction.triggered, "repeatTrackingVideo"
        )
        for rtTrackerAction in self.trackingAlgosGroup.actions():
            self._connect_method_if_present(
                rtTrackerAction.toggled, "rtTrackerActionToggled"
            )
        self._connect_method_if_present(
            self.editRtTrackerParamsAction.triggered, "initRealTimeTracker"
        )
        self._connect_method_if_present(
            self.delObjsOutSegmMaskAction.triggered, "delObjsOutSegmMaskActionTriggered"
        )
        self._connect_method_if_present(self.mergeIDsButton.toggled, "mergeObjs_cb")
        self._connect_method_if_present(self.brushButton.toggled, "Brush_cb")
        self._connect_method_if_present(self.eraserButton.toggled, "Eraser_cb")
        self._connect_method_if_present(self.curvToolButton.toggled, "curvTool_cb")
        self._connect_method_if_present(self.wandToolButton.toggled, "wand_cb")
        self._connect_method_if_present(self.labelRoiButton.toggled, "labelRoi_cb")
        self._connect_method_if_present(
            self.magicPromptsToolButton.toggled, "magicPrompts_cb"
        )
        self._connect_method_if_present(
            self.drawClearRegionButton.toggled, "drawClearRegion_cb"
        )
        self._connect_method_if_present(self.reInitCcaAction.triggered, "reInitCca")
        self._connect_method_if_present(
            self.moveLabelToolButton.toggled, "moveLabelButtonToggled"
        )
        self._connect_method_if_present(
            self.editCcaToolAction.triggered, "manualEditCcaToolbarActionTriggered"
        )
        self._connect_method_if_present(
            self.assignBudMothAutoAction.triggered, "autoAssignBud_YeastMate"
        )
        self._connect_method_if_present(self.keepIDsButton.toggled, "keepIDs_cb")

        self._connect_method_if_present(
            self.whitelistIDsButton.toggled, "whitelistIDs_cb"
        )

        if hasattr(self, "whitelistIDsToolbar"):
            self._connect_method_if_present(
                self.whitelistIDsToolbar.sigWhitelistChanged, "whitelistIDsChanged"
            )
            self._connect_method_if_present(
                self.whitelistIDsToolbar.sigWhitelistAccepted, "whitelistIDsAccepted"
            )
            self._connect_method_if_present(
                self.whitelistIDsToolbar.sigViewOGIDs, "whitelistViewOGIDs"
            )
            self._connect_method_if_present(
                self.whitelistIDsToolbar.sigAddNewIDs, "whitelistAddNewIDsToggled"
            )
            self._connect_method_if_present(
                self.whitelistIDsToolbar.sigLoadOGLabs, "whitelistLoadOGLabs_cb"
            )
            self._connect_method_if_present(
                self.whitelistIDsToolbar.sigTrackOGagainstPreviousFrame,
                "whitelistTrackOGagainstPreviousFrame_cb",
            )

        self._connect_method_if_present(
            self.expandLabelToolButton.toggled, "expandLabelCallback"
        )

        self._connect_method_if_present(
            self.reinitLastSegmFrameAction.triggered, "reInitLastSegmFrame"
        )

        self._connect_method_if_present(
            self.defaultRescaleIntensActionGroup.triggered,
            "defaultRescaleIntensLutActionToggled",
        )

        self._connect_method_if_present(
            self.manuallyEditCcaAction.triggered, "manualEditCca"
        )
        self._connect_method_if_present(self.addScaleBarAction.toggled, "addScaleBar")
        self._connect_method_if_present(
            self.addTimestampAction.toggled, "addTimestamp"
        )
        self._connect_method_if_present(
            self.saveLabColormapAction.triggered, "saveLabelsColormap"
        )

        self._connect_method_if_present(
            self.enableSmartTrackAction.toggled, "enableSmartTrack"
        )
        self._connect_method_if_present(
            self.brushSizeSpinbox.valueChanged, "brushSize_cb"
        )
        self._connect_method_if_present(self.autoIDcheckbox.toggled, "autoIDtoggled")
        self._connect_method_if_present(
            self.modeActionGroup.triggered, "changeModeFromMenu"
        )
        self._connect_method_if_present(
            self.modeComboBox.sigTextChanged, "changeMode"
        )
        self._connect_method_if_present(
            self.modeComboBox.activated, "clearComboBoxFocus"
        )
        self._connect_method_if_present(
            self.equalizeHistPushButton.toggled, "equalizeHist"
        )

        self._connect_method_if_present(
            self.editOverlayColorAction.triggered, "toggleOverlayColorButton"
        )
        self._connect_method_if_present(
            self.editTextIDsColorAction.triggered, "toggleTextIDsColorButton"
        )
        self._connect_method_if_present(
            self.overlayColorButton.sigColorChanging, "changeOverlayColor"
        )
        self._connect_method_if_present(
            self.overlayColorButton.sigColorChanged, "saveOverlayColor"
        )
        self._connect_method_if_present(
            self.textIDsColorButton.sigColorChanging, "updateTextAnnotColor"
        )
        self._connect_method_if_present(
            self.textIDsColorButton.sigColorChanged, "saveTextIDsColors"
        )

        self._connect_method_if_present(
            self.setMeasurementsAction.triggered, "showSetMeasurements"
        )
        self._connect_method_if_present(
            self.addCustomMetricAction.triggered, "addCustomMetric"
        )
        self._connect_method_if_present(
            self.addCombineMetricAction.triggered, "addCombineMetric"
        )

        self._connect_method_if_present(
            self.labelsGrad.colorButton.sigColorChanging, "updateBkgrColor"
        )
        self._connect_method_if_present(
            self.labelsGrad.colorButton.sigColorChanged, "saveBkgrColor"
        )
        self._connect_method_if_present(
            self.labelsGrad.sigGradientChangeFinished, "updateLabelsCmap"
        )
        self._connect_method_if_present(
            self.labelsGrad.sigGradientChanged, "ticksCmapMoved"
        )
        self._connect_method_if_present(
            self.labelsGrad.textColorButton.sigColorChanging, "updateTextLabelsColor"
        )
        self._connect_method_if_present(
            self.labelsGrad.textColorButton.sigColorChanged, "saveTextLabelsColor"
        )

        self._connect_method_if_present(
            self.labelsGrad.shuffleCmapAction.triggered, "shuffle_cmap"
        )
        self._connect_method_if_present(
            self.labelsGrad.greedyShuffleCmapAction.triggered, "greedyShuffleCmap"
        )
        self._connect_method_if_present(
            self.labelsGrad.permanentGreedyCmapAction.toggled,
            "permanentGreedyCmapToggled",
        )
        self._connect_method_if_present(
            self.shuffleCmapAction.triggered, "shuffle_cmap"
        )
        self._connect_method_if_present(
            self.greedyShuffleCmapAction.triggered, "greedyShuffleCmap"
        )
        self._connect_method_if_present(
            self.labelsGrad.invertBwAction.toggled, "setCheckedInvertBW"
        )
        self._connect_method_if_present(
            self.labelsGrad.sigShowLabelsImgToggled, "showLabelImageItem"
        )
        self._connect_method_if_present(
            self.labelsGrad.sigShowRightImgToggled, "showRightImageItem"
        )
        self._connect_method_if_present(
            self.labelsGrad.sigShowNextFrameToggled, "showNextFrameImageItem"
        )

        self._connect_method_if_present(
            self.labelsGrad.defaultSettingsAction.triggered, "restoreDefaultSettings"
        )

        self._connect_method_if_present(
            self.imgGrad.invertBwAction.toggled, "setCheckedInvertBW"
        )
        self.imgGrad.textColorButton.disconnect()
        if hasattr(self, "editTextIDsColorAction"):
            self.imgGrad.textColorButton.clicked.connect(
                self.editTextIDsColorAction.trigger
            )
        self._connect_method_if_present(
            self.imgGrad.labelsAlphaSlider.valueChanged, "updateLabelsAlpha"
        )
        self._connect_method_if_present(
            self.imgGrad.defaultSettingsAction.triggered, "restoreDefaultSettings"
        )

        self._connect_method_if_present(
            self.drawIDsContComboBox.currentIndexChanged, "drawIDsContComboBox_cb"
        )
        self._connect_method_if_present(
            self.drawIDsContComboBox.activated, "clearComboBoxFocus"
        )

        self._connect_method_if_present(
            self.annotateRightHowCombobox.currentIndexChanged,
            "annotateRightHowCombobox_cb",
        )
        self._connect_method_if_present(
            self.annotateRightHowCombobox.activated, "clearComboBoxFocus"
        )

        self._connect_method_if_present(
            self.showTreeInfoCheckbox.toggled, "setAnnotInfoMode"
        )

        self._connect_method_if_present(
            self.annotIDsCheckbox.clicked, "annotOptionClicked"
        )
        self._connect_method_if_present(
            self.annotCcaInfoCheckbox.clicked, "annotOptionClicked"
        )
        self._connect_method_if_present(
            self.annotContourCheckbox.clicked, "annotOptionClicked"
        )
        self._connect_method_if_present(
            self.annotSegmMasksCheckbox.clicked, "annotOptionClicked"
        )
        self._connect_method_if_present(
            self.drawMothBudLinesCheckbox.clicked, "annotOptionClicked"
        )
        self._connect_method_if_present(
            self.drawNothingCheckbox.clicked, "annotOptionClicked"
        )
        self._connect_method_if_present(
            self.annotNumZslicesCheckbox.clicked, "annotOptionClicked"
        )

        self._connect_method_if_present(
            self.annotIDsCheckboxRight.clicked, "annotOptionClickedRight"
        )
        self._connect_method_if_present(
            self.annotCcaInfoCheckboxRight.clicked, "annotOptionClickedRight"
        )
        self._connect_method_if_present(
            self.annotContourCheckboxRight.clicked, "annotOptionClickedRight"
        )
        self._connect_method_if_present(
            self.annotSegmMasksCheckboxRight.clicked, "annotOptionClickedRight"
        )
        self._connect_method_if_present(
            self.drawMothBudLinesCheckboxRight.clicked, "annotOptionClickedRight"
        )
        self._connect_method_if_present(
            self.drawNothingCheckboxRight.clicked, "annotOptionClickedRight"
        )
        self._connect_method_if_present(
            self.annotNumZslicesCheckboxRight.clicked, "annotOptionClickedRight"
        )

        self._connect_method_if_present(
            self.segmentToolAction.triggered, "segmentToolActionTriggered"
        )

        self._connect_method_if_present(self.addDelRoiAction.triggered, "addDelROI")
        self._connect_method_if_present(
            self.addDelPolyLineRoiButton.toggled, "addDelPolyLineRoi_cb"
        )
        self._connect_method_if_present(
            self.delBorderObjAction.triggered, "delBorderObj"
        )
        self._connect_method_if_present(self.delNewObjAction.triggered, "delNewObj")

        self._connect_method_if_present(
            self.brushAutoFillCheckbox.toggled, "brushAutoFillToggled"
        )
        self._connect_method_if_present(
            self.brushAutoHideCheckbox.toggled, "brushAutoHideToggled"
        )

        self.imgGrad.sigAddScaleBar.connect(self.addScaleBarAction.setChecked)
        self.imgGrad.sigAddTimestamp.connect(self.addTimestampAction.setChecked)
        self._connect_method_if_present(
            self.imgGrad.gradient.sigGradientChangeFinished, "imgGradLUTfinished_cb"
        )

        self._connect_method_if_present(
            self.imgPropertiesAction.triggered, "editImgProperties"
        )

        self._connect_method_if_present(
            self.relabelSequentialAction.triggered, "relabelSequentialCallback"
        )

        self._connect_method_if_present(
            self.zoomToObjsAction.triggered, "zoomToObjsActionCallback"
        )
        self._connect_method_if_present(self.zoomOutAction.triggered, "zoomOut")
        self._connect_method_if_present(
            self.preprocessAction.triggered, "preprocessActionTriggered"
        )
        self._connect_method_if_present(
            self.combineChannelsAction.triggered, "combineChannelsActionTriggered"
        )

        self._connect_method_if_present(
            self.viewCcaTableAction.triggered, "viewCcaTable"
        )

        self._connect_method_if_present(
            self.guiTabControl.propsQGBox.idSB.valueChanged, "propsWidgetIDvalueChanged"
        )
        self._connect_method_if_present(
            self.guiTabControl.highlightCheckbox.toggled,
            "highlightIDonHoverCheckBoxToggled",
        )
        self._connect_method_if_present(
            self.guiTabControl.highlightSearchedCheckbox.toggled,
            "highlightSearchedIDcheckBoxToggled",
        )
        intensMeasurQGBox = self.guiTabControl.intensMeasurQGBox
        self._connect_method_if_present(
            intensMeasurQGBox.additionalMeasCombobox.currentTextChanged,
            "updatePropsWidget",
        )
        self._connect_method_if_present(
            intensMeasurQGBox.channelCombobox.currentTextChanged, "updatePropsWidget"
        )

        propsQGBox = self.guiTabControl.propsQGBox
        self._connect_method_if_present(
            propsQGBox.additionalPropsCombobox.currentTextChanged, "updatePropsWidget"
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
        models = myutils.get_list_of_models()
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
        self._connect_method_if_present(
            self.EditSegForLostIDsSetSettings.triggered, "SegForLostIDsSetSettings"
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

        rt_trackers = myutils.get_list_of_real_time_trackers()
        for rt_tracker in rt_trackers:
            rtTrackerAction = QAction(rt_tracker, self)
            rtTrackerAction.setCheckable(True)
            self.trackingAlgosGroup.addAction(rtTrackerAction)

        self.trackWithAcdcAction.setChecked(True)
        aliases = myutils.aliases_real_time_trackers()

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
        from cellacdc import config

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

        from cellacdc import config

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
