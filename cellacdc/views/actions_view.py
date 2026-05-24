"""Qt view adapter for action and shortcut workflows."""

from __future__ import annotations

import os
import re

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import QAction, QActionGroup, QToolButton

from cellacdc import apps, is_mac, settings_folderpath, widgets
from cellacdc.viewmodels.actions_viewmodel import ActionsViewModel

shortcut_filepath = os.path.join(settings_folderpath, 'shortcuts.ini')


class ActionsView:
    """Qt-facing adapter around action construction and shortcut editing."""

    LEGACY_METHODS = (
        'gui_createActions',
        'gui_updateSwitchColorSchemeActionText',
        'gui_connectActions',
        'initShortcuts',
        'setShortcuts',
        'editShortcuts_cb',
        'gui_connectEditActions',
    )

    def __init__(self, host, view_model: ActionsViewModel):
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

    def gui_createActions(self):
        # File actions
        self.segmNdimIndicator = widgets.ToolButtonTextIcon(text='')
        self.segmNdimIndicator.setCheckable(True)
        self.segmNdimIndicator.setChecked(True)
        # self.segmNdimIndicator.setDisabled(True)

        if self.debug:
            self.createEmptyDataAction = QAction(self.host)
            self.createEmptyDataAction.setText("DEBUG: Create empty data")

        self.newWindowAction = QAction("New Window", self.host)

        self.newAction = QAction(self.host)
        self.newAction.setText("&New Segmentation File...")
        self.newAction.setIcon(QIcon(":file-new.svg"))
        self.openFolderAction = QAction(
            QIcon(":folder-open.svg"), "&Load Folder...", self.host
        )
        self.openFileAction = QAction(
            QIcon(":image.svg"),"&Open Image/Video File...", self.host
        )
        self.manageVersionsAction = QAction(
            QIcon(":manage_versions.svg"), "Load Older Versions...", self.host
        )
        self.manageVersionsAction.setDisabled(True)
        self.saveAction = QAction(QIcon(":file-save.svg"), "Save", self.host)
        self.saveAsAction = QAction("Save as...", self.host)
        self.exportToVideoAction = QAction("&Video...", self.host)
        self.exportToImageAction = QAction("&Image...", self.host)
        self.quickSaveAction = QAction("Save Only Segmentation Masks", self.host)
        self.loadFluoAction = QAction("Load Fluorescence Images...", self.host)
        self.loadPosAction = QAction("Load Different Position...", self.host)
        # self.reloadAction = QAction(
        #     QIcon(":reload.svg"), "Reload segmentation file", self
        # )
        self.nextAction = QAction('Next', self.host)
        self.prevAction = QAction('Previous', self.host)
        self.showInExplorerAction = QAction(
            QIcon(":drawer.svg"), f"&{self.openFolderText}", self.host
        )
        self.exitAction = QAction("&Exit", self.host)
        self.undoAction = QAction(QIcon(":undo.svg"), "Undo", self.host)
        self.redoAction = QAction(QIcon(":redo.svg"), "Redo", self.host)
        # String-based key sequences
        self.newWindowAction.setShortcut('Ctrl+Shift+N')
        self.newAction.setShortcut('Ctrl+N')
        self.openFolderAction.setShortcut('Ctrl+O')
        self.loadPosAction.setShortcut('Shift+P')
        self.saveAsAction.setShortcut('Ctrl+Shift+S')
        self.exportToVideoAction.setShortcut('Ctrl+Shift+V')
        self.exportToImageAction.setShortcut('Ctrl+Shift+I')
        self.saveAction.setShortcut('Ctrl+Alt+S')
        self.quickSaveAction.setShortcut('Ctrl+S')
        self.undoAction.setShortcut('Ctrl+Z')
        self.redoAction.setShortcut('Ctrl+Y')
        self.nextAction.setShortcut(Qt.Key_Right)
        self.prevAction.setShortcut(Qt.Key_Left)
        self.addAction(self.nextAction)
        self.addAction(self.prevAction)
        # Help tips
        newTip = "Create a new segmentation file"
        self.newAction.setStatusTip(newTip)
        self.newAction.setWhatsThis("Create a new empty segmentation file")

        self.autoPilotButton = QAction(self.host)
        self.autoPilotButton.setIcon(QIcon(":auto-pilot.svg"))
        self.autoPilotButton.setCheckable(True)
        self.autoPilotButton.setShortcut('Ctrl+Shift+A')

        self.findIdAction = QAction(self.host)
        self.findIdAction.setIcon(QIcon(":find.svg"))
        self.findIdAction.setShortcut('Ctrl+F')

        self.zoomRectButton = QToolButton(self.host)
        self.zoomRectButton.setIcon(QIcon(":zoom_rect.svg"))
        self.zoomRectButton.setCheckable(True)
        self.zoomRectButton.setShortcut('Shift+Z')
        self.LeftClickButtons.append(self.zoomRectButton)
        self.checkableButtons.append(self.zoomRectButton)
        self.checkableQButtonsGroup.addButton(self.zoomRectButton)
        self.widgetsWithShortcut['Zoom to rectangular area'] = (
            self.zoomRectButton
        )

        self.skipToNewIdAction = QAction(self.host)
        self.skipToNewIdAction.setIcon(QIcon(":skip_forward_new_ID.svg"))
        self.skipToNewIdAction.setShortcut(
            widgets.KeySequenceFromText(Qt.Key_PageUp)
        )

        self.skipToNewIdAction.setDisabled(True)

        # Edit actions
        models = self.view_model.model_registry.segmentation_models(
            include_local_seg=True
        )
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

        self.addCustomModelFrameAction = QAction('Add custom model...', self.host)
        self.addCustomModelVideoAction = QAction('Add custom model...', self.host)

        self.segmWithPromptableModelAction = QAction(
            'Select promptable model...', self.host
        )
        self.addCustomPromptModelAction = QAction(
            'Add custom promptable model...', self.host
        )

        self.segmActionsVideo = []
        for model_name in models:
            action = QAction(f"{model_name}...")
            self.segmActionsVideo.append(action)
            action.setDisabled(True)

        self.postProcessSegmAction = QAction(
            "Segmentation post-processing...", self.host
        )
        self.postProcessSegmAction.setDisabled(True)
        self.postProcessSegmAction.setCheckable(True)

        self.EditSegForLostIDsSetSettings = QAction(
            "Edit settings for Segmenting lost IDs...", self.host
        )
        self.EditSegForLostIDsSetSettings.triggered.connect(
            self.seg_for_lost_ids_view.SegForLostIDsSetSettings
        )

        self.repeatTrackingAction = QAction(
            QIcon(":repeat-tracking.svg"), "Repeat tracking", self.host
        )
        self.repeatTrackingAction.setShortcut('Shift+T')
        self.widgetsWithShortcut['Repeat Tracking'] = self.repeatTrackingAction


        self.editRtTrackerParamsAction = QAction(
            'Edit real-time tracker parameters...', self.host
        )

        self.repeatTrackingMenuAction = QAction(
            'Track current frame with real-time tracker...', self.host
        )
        self.repeatTrackingMenuAction.setDisabled(True)
        self.repeatTrackingMenuAction.setShortcut('Shift+T')

        self.repeatTrackingVideoAction = QAction(
            'Select a tracker and track multiple frames...', self.host
        )
        self.repeatTrackingVideoAction.setDisabled(True)
        self.repeatTrackingVideoAction.setShortcut('Alt+Shift+T')

        self.trackingAlgosGroup = QActionGroup(self.host)
        self.trackWithAcdcAction = QAction('Cell-ACDC', self.host)
        self.trackWithAcdcAction.setCheckable(True)
        self.trackingAlgosGroup.addAction(self.trackWithAcdcAction)

        self.trackWithYeazAction = QAction('YeaZ', self.host)
        self.trackWithYeazAction.setCheckable(True)
        self.trackingAlgosGroup.addAction(self.trackWithYeazAction)

        rt_trackers = self.view_model.model_registry.real_time_trackers()
        for rt_tracker in rt_trackers:
            rtTrackerAction = QAction(rt_tracker, self.host)
            rtTrackerAction.setCheckable(True)
            self.trackingAlgosGroup.addAction(rtTrackerAction)

        self.trackWithAcdcAction.setChecked(True)
        aliases = self.view_model.model_registry.real_time_tracker_aliases()

        if 'tracking_algorithm' in self.df_settings.index:
            trackingAlgo = self.df_settings.at['tracking_algorithm', 'value']
            if trackingAlgo in aliases:
                trackingAlgo = aliases[trackingAlgo]
            if trackingAlgo == 'Cell-ACDC':
                self.trackWithAcdcAction.setChecked(True)
            elif trackingAlgo == 'YeaZ':
                self.trackWithYeazAction.setChecked(True)
            else:
                for rtTrackerAction in self.trackingAlgosGroup.actions():
                    if rtTrackerAction.text() == trackingAlgo:
                        rtTrackerAction.setChecked(True)
                        break

        self.setMeasurementsAction = QAction('Set measurements...')
        self.addCustomMetricAction = QAction('Add custom measurement...')
        self.addCombineMetricAction = QAction('Add combined measurement...')

        # Standard key sequence
        # self.copyAction.setShortcut(QKeySequence.StandardKey.Copy)
        # self.pasteAction.setShortcut(QKeySequence.StandardKey.Paste)
        # self.cutAction.setShortcut(QKeySequence.StandardKey.Cut)
        # Help actions
        self.tipsAction = QAction("Tips and tricks...", self.host)
        self.UserManualAction = QAction("User Documentation...", self.host)
        self.openLogFileAction = QAction("Open log file...", self.host)
        self.showLogFilesAction = QAction("Show log files...", self.host)
        self.aboutAction = QAction("About Cell-ACDC", self.host)
        # self.aboutAction = QAction("&About...", self.host)

        # Assign mother to bud button
        self.assignBudMothAutoAction = QAction(self.host)
        self.assignBudMothAutoAction.setIcon(QIcon(":autoAssign.svg"))
        self.assignBudMothAutoAction.setVisible(False)

        self.editCcaToolAction = QAction(self.host)
        self.editCcaToolAction.setIcon(QIcon(":edit_cca.svg"))
        # self.editCcaToolAction.setDisabled(True)
        self.editCcaToolAction.setVisible(False)

        self.reInitCcaAction = QAction(self.host)
        self.reInitCcaAction.setIcon(QIcon(":reinitCca.svg"))
        self.reInitCcaAction.setVisible(False)

        self.toggleColorSchemeAction = QAction(
            'Switch to light theme'
        )
        self.gui_updateSwitchColorSchemeActionText()

        self.pxModeAction = widgets.CheckableAction(
            'Fixed size text annotations'
        )
        self.pxModeAction.setChecked(True)
        pxModeTooltip = (
            'When the text annotations are with fixed size they scale relative '
            'to the object when zooming in/out (fixed size in pixels).\n'
            'This is typically faster to render, but it makes annotations '
            'smaller/larger when zooming in/out, respectively.\n\n'
            'Try activating it to speed up the annotation of many objects '
            'in high resolution mode.\n\n'
            'After activating it, you might need to increase the font size '
            'from the menu on the top menubar `Edit --> Font size`.'
        )
        self.pxModeAction.setToolTip(pxModeTooltip)

        self.highLowResAction = widgets.CheckableAction(
            'High resolution text annotations'
        )
        highLowResTooltip = (
            'Resolution of the text annotations. High resolution results '
            'in slower update of the annotations.\n'
            'Not recommended with a number of segmented objects > 500.\n\n'
        )
        self.highLowResAction.setToolTip(highLowResTooltip)

        self.editAutoSaveIntervalAction = QAction(
            'Change autosave interval (minutes or frames)...', self.host
        )

        self.editShortcutsAction = QAction(
            'Customize keyboard shortcuts...', self.host
        )
        self.editShortcutsAction.setShortcut('Ctrl+K')

        self.showMirroredCursorAction = QAction(
            'Show mirrored cursor on images', self.host
        )
        self.showMirroredCursorAction.setCheckable(True)
        if 'showMirroredCursor' in self.df_settings.index:
            checked = self.df_settings.at['showMirroredCursor', 'value'] == 'Yes'
            self.showMirroredCursorAction.setChecked(checked)
        else:
            self.showMirroredCursorAction.setChecked(True)
        self.showMirroredCursorAction.setShortcut('Ctrl+M')

        self.editTextIDsColorAction = QAction('Text annotation color...', self.host)
        self.editTextIDsColorAction.setDisabled(True)

        self.editOverlayColorAction = QAction('Overlay color...', self.host)
        self.editOverlayColorAction.setDisabled(True)

        self.manuallyEditCcaAction = QAction(
            'Edit cell cycle annotations...', self.host
        )
        self.manuallyEditCcaAction.setShortcut('Ctrl+Shift+P')
        self.manuallyEditCcaAction.setDisabled(True)

        self.viewCcaTableAction = QAction(
            'View cell cycle annotations...', self.host
        )
        self.viewCcaTableAction.setDisabled(True)
        self.viewCcaTableAction.setShortcut('Ctrl+P')


        self.addScaleBarAction = QAction('Add scale bar', self.host)
        self.addScaleBarAction.setCheckable(True)

        self.addTimestampAction = QAction('Add timestamp', self.host)
        self.addTimestampAction.setCheckable(True)

        self.invertBwAction = QAction('Invert black/white', self.host)
        self.invertBwAction.setCheckable(True)
        checked = self.df_settings.at['is_bw_inverted', 'value'] == 'Yes'
        self.invertBwAction.setChecked(checked)

        self.shuffleCmapAction =  QAction('Randomly shuffle colormap', self.host)
        self.shuffleCmapAction.setShortcut('Shift+S')

        self.greedyShuffleCmapAction =  QAction(
            'Greedily shuffle colormap', self.host
        )
        self.greedyShuffleCmapAction.setShortcut('Alt+Shift+S')

        self.saveLabColormapAction = QAction(
            'Save labels colormap...', self.host
        )

        self.normalizeRawAction = QAction(
            'Do not normalize. Display raw image', self.host)
        self.normalizeToFloatAction = QAction(
            'Convert to floating point format with values [0, 1]', self.host)
        # self.normalizeToUbyteAction = QAction(
        #     'Rescale to 8-bit unsigned integer format with values [0, 255]', self.host)
        self.normalizeRescale0to1Action = QAction(
            'Rescale to [0, 1]', self.host)
        self.normalizeByMaxAction = QAction(
            'Normalize by max value', self.host)
        self.normalizeRawAction.setCheckable(True)
        self.normalizeToFloatAction.setCheckable(True)
        # self.normalizeToUbyteAction.setCheckable(True)
        self.normalizeRescale0to1Action.setCheckable(True)
        self.normalizeByMaxAction.setCheckable(True)
        self.normalizeQActionGroup = QActionGroup(self.host)
        self.normalizeQActionGroup.addAction(self.normalizeRawAction)
        self.normalizeQActionGroup.addAction(self.normalizeToFloatAction)
        # self.normalizeQActionGroup.addAction(self.normalizeToUbyteAction)
        self.normalizeQActionGroup.addAction(self.normalizeRescale0to1Action)
        self.normalizeQActionGroup.addAction(self.normalizeByMaxAction)

        self.preprocessAction = QAction(
            'Pre-processing...', self.host
        )
        self.preprocessAction.setShortcut('Alt+Shift+P')

        self.combineChannelsAction = QAction(
            'Combine and manipulate channels and/or segmentation files...', self.host
        )
        self.combineChannelsAction.setShortcut('Alt+Shift+C')

        self.zoomToObjsAction = QAction(
            'Zoom to objects  (Shortcut: H key)', self.host
        )
        self.zoomOutAction = QAction(
            'Zoom out  (Shortcut: double press H key)', self.host
        )

        self.relabelSequentialAction = QAction(
            'Relabel IDs sequentially...', self.host
        )
        self.relabelSequentialAction.setShortcut('Ctrl+L')
        self.relabelSequentialAction.setDisabled(True)

        self.setLastUserNormAction()

        self.autoSegmAction = QAction(
            'Enable automatic segmentation', self.host)
        self.autoSegmAction.setCheckable(True)
        self.autoSegmAction.setDisabled(True)

        self.enableSmartTrackAction = QAction(
            'Smart handling of enabling/disabling tracking', self.host)
        self.enableSmartTrackAction.setCheckable(True)
        self.enableSmartTrackAction.setChecked(True)

        self.enableAutoZoomToCellsAction = QAction(
            'Automatic zoom to all cells when pressing "Next/Previous"', self.host)
        self.enableAutoZoomToCellsAction.setCheckable(True)

        self.imgPropertiesAction = QAction('Properties...', self.host)
        self.imgPropertiesAction.setDisabled(True)

        self.addDelRoiAction = QAction(self.host)
        self.addDelRoiAction.roiType = 'rect'
        self.addDelRoiAction.setIcon(QIcon(":addDelRoi.svg"))

        self.addDelPolyLineRoiButton = QToolButton(self.host)
        self.addDelPolyLineRoiButton.setCheckable(True)
        self.addDelPolyLineRoiButton.setIcon(QIcon(":addDelPolyLineRoi.svg"))

        self.checkableButtons.append(self.addDelPolyLineRoiButton)
        self.LeftClickButtons.append(self.addDelPolyLineRoiButton)

        self.delBorderObjAction = QAction(self.host)
        self.delBorderObjAction.setIcon(QIcon(":delBorderObj.svg"))

        self.delNewObjAction = QAction(self.host)
        self.delNewObjAction.setIcon(QIcon(":delNewObj.svg"))

        self.loadCustomAnnotationsAction = QAction(self.host)
        self.loadCustomAnnotationsAction.setIcon(QIcon(":load_annotation.svg"))
        self.loadCustomAnnotationsAction.setToolTip(
            'Load previously used custom annotations'
        )

        self.addCustomAnnotationAction = QAction(self.host)
        self.addCustomAnnotationAction.setIcon(QIcon(":addCustomAnnotation.svg"))
        self.addCustomAnnotationAction.setToolTip('Add custom annotation')
        # self.functionsNotTested3D.append(self.addCustomAnnotationAction)

        self.viewAllCustomAnnotAction = QAction(self.host)
        self.viewAllCustomAnnotAction.setCheckable(True)
        self.viewAllCustomAnnotAction.setIcon(QIcon(":eye.svg"))
        self.viewAllCustomAnnotAction.setToolTip('Show all custom annotations')
        # self.functionsNotTested3D.append(self.viewAllCustomAnnotAction)

        # self.imgGradLabelsAlphaUpAction = QAction(self.host)
        # self.imgGradLabelsAlphaUpAction.setVisible(False)
        # self.imgGradLabelsAlphaUpAction.setShortcut('Ctrl+Up')

    def gui_updateSwitchColorSchemeActionText(self):
        if self._colorScheme == 'dark':
            txt = 'Switch to light theme'
        else:
            txt = 'Switch to dark theme'
        self.toggleColorSchemeAction.setText(txt)

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
        self.exportToVideoAction.triggered.connect(
            self.exporting_view.exportToVideoTriggered
        )
        self.exportToImageAction.triggered.connect(
            self.exporting_view.exportToImageTriggered
        )
        self.quickSaveAction.triggered.connect(self.quickSave)
        self.viewPreprocDataToggle.toggled.connect(
            self.preprocessing_view.viewPreprocDataToggled
        )
        self.viewCombineChannelDataToggle.toggled.connect(
            self.viewCombineChannelDataToggled
        )
        self.autoSaveToggle.toggled.connect(self.autoSaveToggled)
        self.autoSaveAnnotToggle.toggled.connect(self.autoSaveAnnotToggled)
        self.autoSaveIntervalDialog.sigValueChanged.connect(
            self.autoSaveIntervalValueChanged
        )
        self.autoSaveIntervalEditButton.clicked.connect(
            self.autoSaveIntervalEdit
        )
        self.ccaIntegrCheckerToggle.toggled.connect(
            self.ccaIntegrCheckerToggled
        )
        self.annotLostObjsToggle.toggled.connect(self.annotLostObjsToggled)
        self.highLowResAction.clicked.connect(self.highLowResToggled)
        self.showInExplorerAction.triggered.connect(self.showInExplorer_cb)
        self.exitAction.triggered.connect(self.close)
        self.undoAction.triggered.connect(self.undo_redo_view.undo)
        self.redoAction.triggered.connect(self.undo_redo_view.redo)
        self.nextAction.triggered.connect(self.nextActionTriggered)
        self.prevAction.triggered.connect(self.prevActionTriggered)

        self.invertBwAction.toggled.connect(self.invertBw)
        self.toggleColorSchemeAction.triggered.connect(self.onToggleColorScheme)
        self.pxModeAction.clicked.connect(self.pxModeActionToggled)
        self.editShortcutsAction.triggered.connect(self.editShortcuts_cb)
        self.editAutoSaveIntervalAction.triggered.connect(
            self.autoSaveIntervalEditButton.click
        )
        self.showMirroredCursorAction.toggled.connect(
            self.showMirroredCursorToggled
        )

        # Connect Help actions
        self.tipsAction.triggered.connect(self.showTipsAndTricks)
        self.UserManualAction.triggered.connect(
            self.view_model.app_shell.browse_docs
        )
        self.openLogFileAction.triggered.connect(self.openLogFile)
        self.showLogFilesAction.triggered.connect(self.showLogFiles)
        self.aboutAction.triggered.connect(self.showAbout)
        # Connect Open Recent to dynamically populate it
        # self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)
        self.checkableQButtonsGroup.buttonClicked.connect(self.uncheckQButton)

        self.showPropsDockButton.sigClicked.connect(self.showPropsDockWidget)

        self.loadCustomAnnotationsAction.triggered.connect(
            self.custom_annotations_view.loadCustomAnnotations
        )
        self.addCustomAnnotationAction.triggered.connect(
            self.custom_annotations_view.addCustomAnnotation
        )
        self.viewAllCustomAnnotAction.toggled.connect(
            self.custom_annotations_view.viewAllCustomAnnot
        )
        self.addCustomModelVideoAction.triggered.connect(
            self.showInstructionsCustomModel
        )
        self.addCustomModelFrameAction.triggered.connect(
            self.showInstructionsCustomModel
        )
        self.addCustomModelFrameAction.callback = self.segmFrameCallback
        self.addCustomModelVideoAction.callback = self.segmVideoCallback

        self.addCustomPromptModelAction.triggered.connect(
            self.magic_prompts_view.showInstructionsCustomPromptModel
        )
        self.segmWithPromptableModelAction.triggered.connect(
            self.magic_prompts_view.segmWithPromptableModelActionTriggered
        )

    def initShortcuts(self):
        from cellacdc import config
        cp = config.ConfigParser()
        if os.path.exists(shortcut_filepath):
            cp.read(shortcut_filepath)

        shortcuts_section = self.view_model.keyboard_shortcuts_section
        delete_section = self.view_model.delete_object_section
        if shortcuts_section not in cp:
            cp[shortcuts_section] = {}

        if cp.has_option(shortcuts_section, 'Zoom out'):
            zoomOutKeyValueStr = cp[shortcuts_section]['Zoom out']
            try:
                self.zoomOutKeyValue = int(zoomOutKeyValueStr)
            except Exception as err:
                self.logger.warning(
                    f'{zoomOutKeyValueStr} is not a valid key '
                    'zooming out action. Restoring default key "H".'
                )

        if delete_section not in cp:
            self.delObjAction = None
        else:
            delObjKeySequenceText = (
                cp[delete_section][self.view_model.delete_key_option]
            )
            delObjButtonText = (
                cp[delete_section][self.view_model.delete_button_option]
            )
            delObjQtButton = (
                Qt.MouseButton.LeftButton
                if self.view_model.delete_object_button_is_left_click(
                    delObjButtonText
                )
                else Qt.MouseButton.MiddleButton
            )
            if not delObjKeySequenceText:
                delObjKeySequence = None
            else:
                delObjKeySequence = widgets.KeySequenceFromText(
                    delObjKeySequenceText
                )
            self.delObjToolAction.setChecked(True)
            self.delObjAction = delObjKeySequence, delObjQtButton

        shortcuts = {}
        for name, widget in self.widgetsWithShortcut.items():
            if name not in cp.options(shortcuts_section):
                if hasattr(widget, 'keyPressShortcut'):
                    key = widget.keyPressShortcut
                    shortcut = widgets.KeySequenceFromText(key)
                else:
                    shortcut = widget.shortcut()
                shortcut_text = shortcut.toString()
                cp[shortcuts_section][name] = shortcut_text
            else:
                shortcut_text = cp[shortcuts_section][name]
                shortcut = widgets.KeySequenceFromText(shortcut_text)

            shortcuts[name] = (shortcut_text, shortcut)
        self.setShortcuts(shortcuts, save=False)
        with open(shortcut_filepath, 'w') as ini:
            cp.write(ini)

    def setShortcuts(self, shortcuts: dict, save=True):
        for name, (text, shortcut) in shortcuts.items():
            widget = self.widgetsWithShortcut[name]
            if shortcut is None:
                shortcut = QKeySequence()
            if hasattr(widget, 'keyPressShortcut'):
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

        shortcuts_section = self.view_model.keyboard_shortcuts_section
        delete_section = self.view_model.delete_object_section
        if shortcuts_section not in cp:
            cp[shortcuts_section] = {}

        for name, (text, shortcut) in shortcuts.items():
            cp[shortcuts_section][name] = text

        cp[shortcuts_section]['Zoom out'] = str(self.zoomOutKeyValue)

        if self.delObjAction is None:
            with open(shortcut_filepath, 'w') as ini:
                cp.write(ini)
            return

        delObjKeySequence, delObjQtButton = self.delObjAction
        try:
            if delObjKeySequence is None:
                delObjKeySequenceText = ''
            else:
                delObjKeySequenceText = delObjKeySequence.toString()

            delObjKeySequenceText = (
                self.view_model.sanitize_key_sequence_text(
                    delObjKeySequenceText
                )
            )
            delObjButtonText = self.view_model.delete_object_button_text(
                is_left_click=(
                    delObjQtButton == Qt.MouseButton.LeftButton
                )
            )
            cp[delete_section] = {
                self.view_model.delete_key_option: delObjKeySequenceText,
                self.view_model.delete_button_option: delObjButtonText
            }
        except Exception as err:
            self.logger.warning(
                f'{delObjKeySequence} is not a valid keys sequence for '
                'deleting objects. Setting default action'
            )
            self.delObjAction = None
            cp.remove_section(delete_section)

        with open(shortcut_filepath, 'w') as ini:
            cp.write(ini)

    def editShortcuts_cb(self):
        delObjKeySequenceText, delObjButtonText = (
            self.view_model.default_delete_object_texts(is_mac=is_mac)
        )

        if self.delObjAction is not None:
            delObjKeySequence, delObjQtButton = self.delObjAction
            if delObjKeySequence is None:
                delObjKeySequenceText = ''
            else:
                delObjKeySequenceText = delObjKeySequence.toString()
            delObjKeySequenceText = (
                self.view_model.sanitize_key_sequence_text(
                    delObjKeySequenceText
                )
            )
            delObjButtonText = self.view_model.delete_object_button_text(
                is_left_click=(
                    delObjQtButton == Qt.MouseButton.LeftButton
                )
            )

        win = apps.ShortcutEditorDialog(
            self.widgetsWithShortcut,
            delObjectKey=delObjKeySequenceText,
            delObjectButton=delObjButtonText,
            zoomOutKeyValue=self.zoomOutKeyValue,
            parent=self.host
        )
        win.exec_()
        if win.cancel:
            return

        self.delObjAction = win.delObjAction
        self.zoomOutKeyValue = win.zoomOutKeyValue
        self.setShortcuts(win.customShortcuts)

    def gui_connectEditActions(self):
        self.showInExplorerAction.setEnabled(True)
        self.mode_controls_view.setEnabledFileToolbar(True)
        self.loadFluoAction.setEnabled(True)
        self.isEditActionsConnected = True

        self.preprocessImageAction.triggered.connect(
            self.preprocessAction.trigger
        )
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
        self.findIdAction.triggered.connect(self.object_search_view.findID)
        self.zoomRectButton.toggled.connect(self.zoomRectActionToggled)
        self.autoPilotButton.toggled.connect(self.autoPilotToggled)
        self.skipToNewIdAction.triggered.connect(
            self.object_search_view.skipForwardToNewID
        )
        self.slideshowButton.toggled.connect(self.launchSlideshow)

        self.copyLostObjButton.toggled.connect(self.copyLostObjContour_cb)
        self.manualAnnotPastButton.toggled.connect(
            self.manualAnnotPast_cb
        )

        self.segmSingleFrameMenu.triggered.connect(self.segmFrameCallback)
        self.segmVideoMenu.triggered.connect(self.segmVideoCallback)

        self.postProcessSegmAction.toggled.connect(self.postProcessSegm)
        self.autoSegmAction.toggled.connect(self.autoSegm_cb)
        self.realTimeTrackingToggle.clicked.connect(self.realTimeTrackingClicked)
        self.repeatTrackingAction.triggered.connect(self.repeatTracking)
        self.manualTrackingButton.toggled.connect(self.manualTracking_cb)
        self.manualBackgroundButton.toggled.connect(self.manualBackground_cb)
        self.repeatTrackingMenuAction.triggered.connect(self.repeatTracking)
        self.repeatTrackingVideoAction.triggered.connect(
            self.repeatTrackingVideo
        )
        for rtTrackerAction in self.trackingAlgosGroup.actions():
            rtTrackerAction.toggled.connect(self.rtTrackerActionToggled)
        self.editRtTrackerParamsAction.triggered.connect(
            self.initRealTimeTracker
        )
        self.delObjsOutSegmMaskAction.triggered.connect(
            self.object_cleanup_view
            .delete_objects_outside_mask_action_triggered
        )
        self.mergeIDsButton.toggled.connect(self.mergeObjs_cb)
        self.brushButton.toggled.connect(self.Brush_cb)
        self.eraserButton.toggled.connect(self.Eraser_cb)
        self.curvToolButton.toggled.connect(
            self.curvature_tools_view.curvTool_cb
        )
        self.wandToolButton.toggled.connect(self.wand_cb)
        self.labelRoiButton.toggled.connect(self.labelRoi_cb)
        self.magicPromptsToolButton.toggled.connect(self.magicPrompts_cb)
        self.drawClearRegionButton.toggled.connect(
            self.draw_clear_region_view.toggle
        )
        self.reInitCcaAction.triggered.connect(self.reInitCca)
        self.moveLabelToolButton.toggled.connect(
            self.label_transform_tools_view.move_label_button_toggled
        )
        self.editCcaToolAction.triggered.connect(
            self.manualEditCcaToolbarActionTriggered
        )
        self.assignBudMothAutoAction.triggered.connect(
            self.autoAssignBud_YeastMate
        )
        self.keepIDsButton.toggled.connect(self.keepIDs_cb)

        self.whitelistIDsButton.toggled.connect(self.whitelistIDs_cb)

        self.whitelistIDsToolbar.sigWhitelistChanged.connect(
            self.whitelistIDsChanged
        )

        self.whitelistIDsToolbar.sigWhitelistAccepted.connect(
            self.whitelistIDsAccepted
        )

        self.whitelistIDsToolbar.sigViewOGIDs.connect(self.whitelistViewOGIDs)

        self.whitelistIDsToolbar.sigAddNewIDs.connect(self.whitelistAddNewIDsToggled)

        self.whitelistIDsToolbar.sigLoadOGLabs.connect(self.whitelistLoadOGLabs_cb)

        self.whitelistIDsToolbar.sigTrackOGagainstPreviousFrame.connect(
            self.whitelistTrackOGagainstPreviousFrame_cb
        )

        self.expandLabelToolButton.toggled.connect(
            self.label_transform_tools_view.expand_label_callback
        )

        self.reinitLastSegmFrameAction.triggered.connect(
            self.reInitLastSegmFrame
        )


        self.defaultRescaleIntensActionGroup.triggered.connect(
            self.defaultRescaleIntensLutActionToggled
        )

        # self.repeatAutoCcaAction.triggered.connect(self.repeatAutoCca)
        self.manuallyEditCcaAction.triggered.connect(self.manualEditCca)
        self.addScaleBarAction.toggled.connect(
            self.display_decorations_view.add_scale_bar
        )
        self.addTimestampAction.toggled.connect(
            self.display_decorations_view.add_timestamp
        )
        self.saveLabColormapAction.triggered.connect(self.saveLabelsColormap)

        self.enableSmartTrackAction.toggled.connect(self.enableSmartTrack)
        # Brush/Eraser size action
        self.brushSizeSpinbox.valueChanged.connect(self.brushSize_cb)
        self.autoIDcheckbox.toggled.connect(self.autoIDtoggled)
        # Mode
        self.modeActionGroup.triggered.connect(
            self.mode_controls_view.changeModeFromMenu
        )
        self.modeComboBox.sigTextChanged.connect(
            self.mode_controls_view.changeMode
        )
        self.modeComboBox.activated.connect(
            self.mode_controls_view.clearComboBoxFocus
        )
        self.equalizeHistPushButton.toggled.connect(self.equalizeHist)

        self.editOverlayColorAction.triggered.connect(self.toggleOverlayColorButton)
        self.editTextIDsColorAction.triggered.connect(self.toggleTextIDsColorButton)
        self.overlayColorButton.sigColorChanging.connect(self.changeOverlayColor)
        self.overlayColorButton.sigColorChanged.connect(self.saveOverlayColor)
        self.textIDsColorButton.sigColorChanging.connect(self.updateTextAnnotColor)
        self.textIDsColorButton.sigColorChanged.connect(self.saveTextIDsColors)

        self.setMeasurementsAction.triggered.connect(
            self.measurements_view.show_set_measurements
        )
        self.addCustomMetricAction.triggered.connect(
            self.measurements_view.add_custom_metric
        )
        self.addCombineMetricAction.triggered.connect(
            self.measurements_view.add_combine_metric
        )

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
        self.imgGrad.labelsAlphaSlider.valueChanged.connect(
            self.updateLabelsAlpha
        )
        self.imgGrad.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )

        # Drawing mode
        self.drawIDsContComboBox.currentIndexChanged.connect(
            self.drawIDsContComboBox_cb
        )
        self.drawIDsContComboBox.activated.connect(
            self.mode_controls_view.clearComboBoxFocus
        )

        self.annotateRightHowCombobox.currentIndexChanged.connect(
            self.annotateRightHowCombobox_cb
        )
        self.annotateRightHowCombobox.activated.connect(
            self.mode_controls_view.clearComboBoxFocus
        )

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
        self.annotIDsCheckboxRight.clicked.connect(
            self.annotOptionClickedRight)
        self.annotCcaInfoCheckboxRight.clicked.connect(
            self.annotOptionClickedRight)
        self.annotContourCheckboxRight.clicked.connect(
            self.annotOptionClickedRight)
        self.annotSegmMasksCheckboxRight.clicked.connect(
            self.annotOptionClickedRight)
        self.drawMothBudLinesCheckboxRight.clicked.connect(
            self.annotOptionClickedRight)
        self.drawNothingCheckboxRight.clicked.connect(
            self.annotOptionClickedRight)
        self.annotNumZslicesCheckboxRight.clicked.connect(
            self.annotOptionClickedRight
        )

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

        self.relabelSequentialAction.triggered.connect(
            self.relabelSequentialCallback
        )

        self.zoomToObjsAction.triggered.connect(self.zoomToObjsActionCallback)
        self.zoomOutAction.triggered.connect(self.zoomOut)
        self.preprocessAction.triggered.connect(
            self.preprocessing_view.preprocessActionTriggered
        )
        self.combineChannelsAction.triggered.connect(self.combineChannelsActionTriggered)

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
