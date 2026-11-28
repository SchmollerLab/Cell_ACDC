"""Menu bar, actions and their connections, status bar, terminal widget,
tooltips and channel name actions of the main GUI. Also the
resetViewRange decorator. Mixins of guiWin (gui/main_window/window.py)
only: the measurements viewer never composes them.
"""
import re
from functools import partial
from functools import wraps
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAction, QLabel, QMenu, QToolButton, QActionGroup, QDockWidget
)
from cellacdc import html_utils
from cellacdc.myutils import filesystem, segm_models
from cellacdc.shared.widgets import (
    basic_widgets as widgets_basic_widgets,
    line_edits as widgets_line_edits,
    search_widgets as widgets_search_widgets,
    toolbars as widgets_toolbars,
)
from ..widgets import (
    edit_toolbars as gui_widgets_edit_toolbars,
    main_window_items as gui_widgets_main_window_items,
)
from cellacdc import rst_utils


def resetViewRange(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        self.storeViewRange()
        if func.__code__.co_argcount==1 and func.__defaults__ is None:
            result = func(self)
        elif func.__code__.co_argcount>1 and func.__defaults__ is None:
            result = func(self, *args)
        else:
            result = func(self, *args, **kwargs)
        QTimer.singleShot(200, self.resetRange)
        return result
    return inner_function


class MenusActionsGuiElements:
    """Menu bar, every QAction of the main GUI (gui_createActions) and its
    connections (gui_connectActions, gui_connectEditActions), status
    bar, terminal widget, tooltips read from the docs and channel name
    actions.
    """

    def setTooltips(self):
        tooltips = rst_utils.get_tooltips_from_docs()

        for key, tooltip in tooltips.items():
            setShortcut = getattr(self, key).shortcut().toString()
            if 'Shortcut: ' in tooltip:
                tooltip = tooltip.replace('Shortcut: ', '\nShortcut: ')
            elif setShortcut != "":
                tooltip = re.sub(
                    r'Shortcut: \"(.*)\"', 
                    f"Shortcut: \"{setShortcut}\"", 
                    tooltip
                )
            else:
                tooltip = re.sub(
                    r'Shortcut: \"(.*)\"', 
                    f"Shortcut: \"No shortcut\"", 
                    tooltip
                )
            
            getattr(self, key).setToolTip(tooltip)
            getattr(self, key)._tooltip = tooltip
            
            # add custom addition to the search input tooltip
            if key == "searchAction":
                self.searchWidget.setToolTip(tooltip)

    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        self.initMenuBar = menuBar
        menuBar.setNativeMenuBar(False)
        # File menu
        fileMenu = QMenu("&File", self)
        self.fileMenu = fileMenu
        menuBar.addMenu(fileMenu)
        if self.debug:
            fileMenu.addAction(self.createEmptyDataAction)
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.newWindowAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.openFolderAction)
        fileMenu.addAction(self.openFileAction)
        # Open Recent submenu
        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        fileMenu.addSeparator()
        fileMenu.addAction(self.manageVersionsAction)
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.saveAsAction)
        fileMenu.addAction(self.quickSaveAction)
        fileMenu.addSeparator()
        
        self.exportMenu = fileMenu.addMenu('Export')
        self.exportMenu.addAction(self.exportToVideoAction)
        self.exportMenu.addAction(self.exportToImageAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.loadFluoAction)
        fileMenu.addAction(self.loadPosAction)
        # Separator
        self.fileMenu.lastSeparator = fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        
        # Edit menu
        editMenu = menuBar.addMenu("&Edit")
        editMenu.addSeparator()

        editMenu.addAction(self.editShortcutsAction)
        editMenu.addAction(self.editTextIDsColorAction)
        editMenu.addAction(self.editOverlayColorAction)
        editMenu.addAction(self.manuallyEditCcaAction)
        editMenu.addAction(self.enableSmartTrackAction)
        editMenu.addAction(self.enableAutoZoomToCellsAction)

        # View menu
        self.viewMenu = menuBar.addMenu("&View")
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.viewCcaTableAction)

        # Image menu
        ImageMenu = menuBar.addMenu("&Image")
        ImageMenu.addSeparator()
        ImageMenu.addAction(self.imgPropertiesAction)
        self.defaultRescaleIntensLutMenu = ImageMenu.addMenu(
            "Default method to rescale intensities (LUT)"
        )

        self.defaultRescaleIntensActionGroup = QActionGroup(
            self.defaultRescaleIntensLutMenu
        )
        howTexts = (
            'Rescale each 2D image', 
            'Rescale across z-stack',
            'Rescale across time frames',
            'Do no rescale, display raw image'
        )
        try:
            self.defaultRescaleIntensHow = (
                self.df_settings.at['default_rescale_intens_how', 'value']
            )
        except Exception as err:
            self.defaultRescaleIntensHow = howTexts[0]
            
        for howText in howTexts:
            action = QAction(howText, self.defaultRescaleIntensLutMenu)
            action.setCheckable(True)
            if howText == self.defaultRescaleIntensHow:
                action.setChecked(True)
                
            self.defaultRescaleIntensActionGroup.addAction(action)
            self.defaultRescaleIntensLutMenu.addAction(action)
        
        ImageMenu.addAction(self.addScaleBarAction)
        ImageMenu.addAction(self.addTimestampAction)
        
        self.rescaleIntensMenu = ImageMenu.addMenu('Rescale intensities (LUT)')
        
        ImageMenu.addAction(self.preprocessAction)
        ImageMenu.addAction(self.combineChannelsAction)
        ImageMenu.addAction(self.saveLabColormapAction)
        ImageMenu.addAction(self.shuffleCmapAction)
        ImageMenu.addAction(self.greedyShuffleCmapAction)
        ImageMenu.addAction(self.zoomToObjsAction)
        ImageMenu.addAction(self.zoomOutAction)
        self.ImageMenu = ImageMenu

        # Segment menu
        SegmMenu = menuBar.addMenu("&Segment")
        self.segmentMenu = SegmMenu
        SegmMenu.addSeparator()
        self.segmSingleFrameMenu = SegmMenu.addMenu('Segment displayed frame')
        for action in self.segmActions:
            self.segmSingleFrameMenu.addAction(action)

        self.segmSingleFrameMenu.addSeparator()
        self.segmSingleFrameMenu.addAction(self.addCustomModelFrameAction)

        self.segmVideoMenu = SegmMenu.addMenu('Segment multiple frames')
        for action in self.segmActionsVideo:
            self.segmVideoMenu.addAction(action)

        self.segmVideoMenu.addSeparator()
        self.segmVideoMenu.addAction(self.addCustomModelVideoAction)
        
        self.segmWithPromptableModelMenu = SegmMenu.addMenu(
            'Segment with promptable model'
        )
        
        self.segmWithPromptableModelMenu.addAction(
            self.segmWithPromptableModelAction
        )
        
        self.segmWithPromptableModelMenu.addSeparator()
        self.segmWithPromptableModelMenu.addAction(
            self.addCustomPromptModelAction
        )

        SegmMenu.addAction(self.editSegForLostIDsSetSettings)
        SegmMenu.addAction(self.postProcessSegmAction)
        SegmMenu.addAction(self.autoSegmAction)
        SegmMenu.addAction(self.relabelSequentialAction)
        SegmMenu.aboutToShow.connect(self.nonViewerEditMenuOpened)

        # Tracking menu
        trackingMenu = menuBar.addMenu("&Tracking")
        self.trackingMenu = trackingMenu
        trackingMenu.addSeparator()
        selectTrackAlgoMenu = trackingMenu.addMenu(
            'Select real-time tracking algorithm'
        )
        for rtTrackerAction in self.trackingAlgosGroup.actions():
            selectTrackAlgoMenu.addAction(rtTrackerAction)

        trackingMenu.addAction(self.editRtTrackerParamsAction)
        trackingMenu.addAction(self.repeatTrackingVideoAction)

        trackingMenu.addAction(self.repeatTrackingMenuAction)
        trackingMenu.aboutToShow.connect(self.nonViewerEditMenuOpened)
        
        if self.mainWin is not None:
            trackingMenu.addAction(
                self.mainWin.applyTrackingFromTableAction
            )
            trackingMenu.addAction(
                self.mainWin.applyTrackingFromTrackMateXMLAction
            )

        # Measurements menu
        measurementsMenu = menuBar.addMenu("&Measurements")
        self.measurementsMenu = measurementsMenu
        measurementsMenu.addSeparator()
        measurementsMenu.addAction(self.setMeasurementsAction)
        measurementsMenu.addAction(self.addCustomMetricAction)
        measurementsMenu.addAction(self.addCombineMetricAction)
        measurementsMenu.setDisabled(True)

        # Settings menu
        self.settingsMenu = QMenu("Settings", self)
        self.settingsMenu.setToolTipsVisible(True)
        menuBar.addMenu(self.settingsMenu)
        self.settingsMenu.addAction(self.invertBwAction)
        self.settingsMenu.addAction(self.toggleColorSchemeAction)
        self.settingsMenu.addSeparator()
        editObjectTrackSettingsAction = self.settingsMenu.addAction(
            'Edit object track settings...'
        )
        editObjectTrackSettingsAction.triggered.connect(
            self.setAnnotateObjTrackSettings 
        )
        self.settingsMenu.addAction(self.highLowResAction)
        self.settingsMenu.addAction(self.scalingLabelsAction)
        self.settingsMenu.addAction(self.editShortcutsAction)
        self.settingsMenu.addAction(self.showMirroredCursorAction)
        self.settingsMenu.addSeparator()
        self.settingsMenu.addAction(self.editAutoSaveIntervalAction)
        self.settingsMenu.addSeparator()

        # Mode menu (actions added when self.modeComboBox is created)
        self.modeMenu = menuBar.addMenu('Mode')
        self.modeMenu.menuAction().setVisible(False)

        # Help menu
        helpMenu = menuBar.addMenu("&Help")
        helpMenu.addAction(self.openLogFileAction)
        helpMenu.addAction(self.showLogFilesAction)
        helpMenu.addAction(self.tipsAction)
        helpMenu.addAction(self.UserManualAction)
        self.helpMenu = helpMenu

        self.searchWidget = widgets_search_widgets.ButtonSearchWidget(self)
        self.searchWidget.sigTriggerBlink.connect(self.onSearchTriggerBlink)
        menuBar.setCornerWidget(self.searchWidget)
        self.focusOnSearchAction = QAction("Search for ID or functionality", self)
        self.focusOnSearchAction.setShortcut('Ctrl+F')
        self.focusOnSearchAction.triggered.connect(
            partial(self.focusOnSearch, blinking=True)
        )
        helpMenu.addAction(self.focusOnSearchAction)
        self.searchWidget.sigSearchId.connect(
            self.onSearchId
        )
        self.searchWidget.setMinimumHeight(
            self.searchWidget.sizeHint().height()
        )

        helpMenu.addSeparator()
        helpMenu.addAction(self.aboutAction)

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Permanent widget. Its text is updated at every mouse move -->
        # fixed size (growing only) to avoid re-layouting the main window
        self.wcLabel = widgets_basic_widgets.AutoGrowFixedSizeLabel('')
        self.statusbar.addPermanentWidget(self.wcLabel)

        # self.toggleTerminalButton = widgets.ToggleTerminalButton()
        # self.statusbar.addWidget(self.toggleTerminalButton)
        # self.toggleTerminalButton.sigClicked.connect(
        #     self.gui_terminalButtonClicked
        # )

        self.statusBarLabel = QLabel('')
        self.statusbar.addWidget(self.statusBarLabel)

    def gui_createTerminalWidget(self):
        self.terminal = gui_widgets_main_window_items.QLog(logger=self.logger)
        self.terminal.connect()
        self.terminalDock = QDockWidget('Log', self)

        self.terminalDock.setWidget(self.terminal)
        self.terminalDock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.terminalDock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.terminalDock)
        # self.terminalDock.widget().layout().setContentsMargins(10,0,10,0)
        self.terminalDock.setVisible(False)

    @resetViewRange
    def gui_terminalButtonClicked(self, terminalVisible):
        self.terminalDock.setVisible(terminalVisible)

    def gui_createActions(self):
        # File actions
        self.segmNdimIndicator = gui_widgets_edit_toolbars.ToolButtonTextIcon(text='')
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
            QIcon(":image.svg"),"&Open Image/Video File...", self
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
        
        self.nextAction = QAction('Next', self)
        self.defaultMouseShortcuts['Next'] = Qt.MouseButton.XButton2
        self.widgetsWithShortcut['Next'] = self.nextAction
        self.widgetsPersistentShortcut['Next'] = Qt.Key_Right

        self.prevAction = QAction('Previous', self)
        self.defaultMouseShortcuts['Previous'] = Qt.MouseButton.XButton1
        self.widgetsWithShortcut['Previous'] = self.prevAction
        self.widgetsPersistentShortcut['Previous'] = Qt.Key_Left
        
        self.showInExplorerAction = QAction(
            QIcon(":drawer.svg"), f"&{self.openFolderText}", self
        )
        self.exitAction = QAction("&Exit", self)
        self.undoAction = QAction(QIcon(":undo.svg"), "Undo", self)
        self.redoAction = QAction(QIcon(":redo.svg"), "Redo", self)
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

        self.addAction(self.nextAction)
        self.addAction(self.prevAction)
        # Help tips
        newTip = "Create a new segmentation file"
        self.newAction.setStatusTip(newTip)
        self.newAction.setWhatsThis("Create a new empty segmentation file")

        self.autoPilotButton = QAction(self)
        self.autoPilotButton.setIcon(QIcon(":auto-pilot.svg"))
        self.autoPilotButton.setCheckable(True)
        self.autoPilotButton.setShortcut('Ctrl+Shift+A')

        self.exportCurrentViewButton = QToolButton(self)
        self.exportCurrentViewButton.setIcon(QIcon(":folder-move.svg"))
        self.exportCurrentViewButton.setCheckable(True)
        self.exportCurrentViewButton.setShortcut('Ctrl+Shift+E')
        
        self.searchAction = QAction(self)
        self.searchAction.setIcon(QIcon(":find.svg"))
        
        self.zoomRectButton = QToolButton(self)
        self.zoomRectButton.setIcon(QIcon(":zoom_rect.svg"))
        self.zoomRectButton.setCheckable(True)
        self.zoomRectButton.setShortcut('Shift+Z')
        self.leftClickButtons.append(self.zoomRectButton)
        self.checkableButtons.append(self.zoomRectButton)
        self.checkableQButtonsGroup.addButton(self.zoomRectButton)
        self.widgetsWithShortcut['Zoom to rectangular area'] = (
            self.zoomRectButton
        )
        
        self.skipToNewIdAction = QAction(self)
        self.skipToNewIdAction.setIcon(QIcon(":skip_forward_new_ID.svg"))
        self.skipToNewIdAction.setShortcut(
            widgets_line_edits.KeySequenceFromText(Qt.Key_PageUp)
        )

        self.skipToNewIdAction.setDisabled(True)

        # Edit actions
        models = segm_models.get_list_of_models()
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

        self.addCustomModelFrameAction = QAction('Add custom model...', self)
        self.addCustomModelVideoAction = QAction('Add custom model...', self)
        
        self.segmWithPromptableModelAction = QAction(
            'Select promptable model...', self
        )
        self.addCustomPromptModelAction = QAction(
            'Add custom promptable model...', self
        )

        self.segmActionsVideo = []
        for model_name in models:
            action = QAction(f"{model_name}...")
            self.segmActionsVideo.append(action)
            action.setDisabled(True)

        self.postProcessSegmAction = QAction(
            "Segmentation post-processing...", self
        )
        self.postProcessSegmAction.setDisabled(True)
        self.postProcessSegmAction.setCheckable(True)

        self.editSegForLostIDsSetSettings = QAction(
            "Edit settings for Segmenting lost IDs...", self
        )
        self.editSegForLostIDsSetSettings.triggered.connect(
            self.SegForLostIDsSetSettings
        )

        self.repeatTrackingAction = QAction(
            QIcon(":repeat-tracking.svg"), "Repeat tracking", self
        )
        self.repeatTrackingAction.setShortcut('Shift+T')
        self.widgetsWithShortcut['Repeat Tracking'] = self.repeatTrackingAction

        self.editRtTrackerParamsAction = QAction(
            'Edit real-time tracker parameters...', self
        )
        
        self.repeatTrackingMenuAction = QAction(
            'Track current frame with real-time tracker...', self
        )
        self.repeatTrackingMenuAction.setDisabled(True)
        self.repeatTrackingMenuAction.setShortcut('Ctrl+T')

        self.repeatTrackingVideoAction = QAction(
            'Select a tracker and track multiple frames...', self
        )
        self.repeatTrackingVideoAction.setDisabled(True)
        self.repeatTrackingVideoAction.setShortcut('Alt+Shift+T')

        self.trackingAlgosGroup = QActionGroup(self)
        self.trackWithAcdcAction = QAction('Cell-ACDC', self)
        self.trackWithAcdcAction.setCheckable(True)
        self.trackingAlgosGroup.addAction(self.trackWithAcdcAction)

        self.trackWithYeazAction = QAction('YeaZ', self)
        self.trackWithYeazAction.setCheckable(True)
        self.trackingAlgosGroup.addAction(self.trackWithYeazAction)

        rt_trackers = segm_models.get_list_of_real_time_trackers()
        for rt_tracker in rt_trackers:
            rtTrackerAction = QAction(rt_tracker, self)
            rtTrackerAction.setCheckable(True)
            self.trackingAlgosGroup.addAction(rtTrackerAction)

        self.trackWithAcdcAction.setChecked(True)
        aliases = segm_models.aliases_real_time_trackers()

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

        self.toggleColorSchemeAction = QAction(
            'Switch to light theme'
        )
        self.gui_updateSwitchColorSchemeActionText()
        
        self.scalingLabelsAction = widgets_toolbars.CheckableAction(
            'Fix text annotations to screen size'
        )
        scalingLabelsTooltip = html_utils.paragraph(
            'When the text annotations are fixed to screen size, '
            'they do not scale relative '
            'to the objects when zooming in/out.'
        )
            
        self.scalingLabelsAction.setToolTip(scalingLabelsTooltip)
        checked = self._get_setting_value('scalingLabels', False, cast=bool)
        self.scalingLabelsAction.setChecked(checked)
        
        self.highLowResAction = widgets_toolbars.CheckableAction(
            'High resolution text annotations'
        )
        highLowResTooltip = (
            'Resolution of the text annotations. High resolution results '
            'in slower update of the annotations.\n'
            'Not recommended with a number of segmented objects > 500.\n\n'
        )
        self.highLowResAction.setToolTip(highLowResTooltip)
        checked = self._get_setting_value('highLowRes', True, cast=bool)
        self.highLowResAction.setChecked(checked)
        
        self.editAutoSaveIntervalAction = QAction(
            'Change autosave interval (minutes or frames)...', self
        )
        
        self.editShortcutsAction = QAction(
            'Customize keyboard shortcuts...', self
        )
        self.editShortcutsAction.setShortcut('Ctrl+K')
        
        self.showMirroredCursorAction = QAction(
            'Show mirrored cursor on images', self
        )
        self.showMirroredCursorAction.setCheckable(True)
        if 'showMirroredCursor' in self.df_settings.index:
            checked = self.df_settings.at['showMirroredCursor', 'value'] == 'Yes'
            self.showMirroredCursorAction.setChecked(checked)
        else:
            self.showMirroredCursorAction.setChecked(True)
        self.showMirroredCursorAction.setShortcut('Ctrl+M')

        self.editTextIDsColorAction = QAction('Text annotation color...', self)
        self.editTextIDsColorAction.setDisabled(True)

        self.editOverlayColorAction = QAction('Overlay color...', self)
        self.editOverlayColorAction.setDisabled(True)

        self.manuallyEditCcaAction = QAction(
            'Edit cell cycle annotations...', self
        )
        self.manuallyEditCcaAction.setShortcut('Ctrl+Shift+P')
        self.manuallyEditCcaAction.setDisabled(True)

        self.viewCcaTableAction = QAction(
            'View cell cycle annotations...', self
        )
        self.viewCcaTableAction.setDisabled(True)
        self.viewCcaTableAction.setShortcut('Ctrl+P')
        
        
        self.addScaleBarAction = QAction('Add scale bar', self)
        self.addScaleBarAction.setCheckable(True)
        
        self.addTimestampAction = QAction('Add timestamp', self)
        self.addTimestampAction.setCheckable(True)

        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        checked = self.df_settings.at['is_bw_inverted', 'value'] == 'Yes'
        self.invertBwAction.setChecked(checked)

        self.shuffleCmapAction =  QAction('Randomly shuffle colormap', self)
        self.shuffleCmapAction.setShortcut('Shift+S')

        self.greedyShuffleCmapAction =  QAction(
            'Greedily shuffle colormap', self
        )
        self.greedyShuffleCmapAction.setShortcut('Alt+Shift+S')

        self.saveLabColormapAction = QAction(
            'Save labels colormap...', self
        )

        self.normalizeRawAction = QAction(
            'Do not normalize. Display raw image', self)
        self.normalizeToFloatAction = QAction(
            'Convert to floating point format with values [0, 1]', self)
        # self.normalizeToUbyteAction = QAction(
        #     'Rescale to 8-bit unsigned integer format with values [0, 255]', self)
        self.normalizeRescale0to1Action = QAction(
            'Rescale to [0, 1]', self)
        self.normalizeByMaxAction = QAction(
            'Normalize by max value', self)
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

        self.preprocessAction = QAction(
            'Pre-processing...', self
        )
        self.preprocessAction.setShortcut('Alt+Shift+P')

        self.combineChannelsAction = QAction(
            'Combine and manipulate channels and/or segmentation files...', self
        )
        self.combineChannelsAction.setShortcut('Alt+Shift+C')
        
        self.zoomToObjsAction = QAction(
            'Zoom to objects  (Shortcut: H key)', self
        )
        self.zoomOutAction = QAction(
            'Zoom out  (Shortcut: double press H key)', self
        )

        self.relabelSequentialAction = QAction(
            'Relabel IDs sequentially...', self
        )
        self.relabelSequentialAction.setShortcut('Ctrl+L')
        self.relabelSequentialAction.setDisabled(True)

        self.setLastUserNormAction()

        self.autoSegmAction = QAction(
            'Enable automatic segmentation', self)
        self.autoSegmAction.setCheckable(True)
        self.autoSegmAction.setDisabled(True)

        self.enableSmartTrackAction = QAction(
            'Smart handling of enabling/disabling tracking', self)
        self.enableSmartTrackAction.setCheckable(True)
        self.enableSmartTrackAction.setChecked(True)

        self.enableAutoZoomToCellsAction = QAction(
            'Automatic zoom to all cells when pressing "Next/Previous"', self)
        self.enableAutoZoomToCellsAction.setCheckable(True)

        self.imgPropertiesAction = QAction('Properties...', self)
        self.imgPropertiesAction.setDisabled(True)

        self.addDelRoiAction = QAction(self)
        self.addDelRoiAction.roiType = 'rect'
        self.addDelRoiAction.setIcon(QIcon(":addDelRoi.svg"))
        
        self.addDelPolyLineRoiButton = QToolButton(self)
        self.addDelPolyLineRoiButton.setCheckable(True)
        self.addDelPolyLineRoiButton.setIcon(QIcon(":addDelPolyLineRoi.svg"))
        
        self.checkableButtons.append(self.addDelPolyLineRoiButton)
        self.leftClickButtons.append(self.addDelPolyLineRoiButton)
       
        self.delBorderObjAction = QAction(self)
        self.delBorderObjAction.setIcon(QIcon(":delBorderObj.svg"))
        
        self.delNewObjAction = QAction(self)
        self.delNewObjAction.setIcon(QIcon(":delNewObj.svg"))

        self.gui_createCustomAnnotationsActions()

        # self.imgGradLabelsAlphaUpAction = QAction(self)
        # self.imgGradLabelsAlphaUpAction.setVisible(False)
        # self.imgGradLabelsAlphaUpAction.setShortcut('Ctrl+Up')
        
        self.MaxProjToggleAction = QAction(self)
        self.MaxProjToggleAction.triggered.connect(
            self.maxProjToggleActionTriggered
        )
        # hide action
        self.MaxProjToggleAction.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.addAction(self.MaxProjToggleAction)
        # self.MaxProjToggleAction.setVisible(False)
        self.MaxProjToggleAction.setShortcut('Alt+X')
        self.widgetsWithShortcut['Toggle max z projection'] = self.MaxProjToggleAction

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
        self.viewPreprocDataToggle.toggled.connect(
            self.viewPreprocDataToggled
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
        self.scalingLabelsAction.clicked.connect(self.scalingLabelsToggled)
        self.showInExplorerAction.triggered.connect(self.showInExplorer_cb)
        self.exitAction.triggered.connect(self.close)
        self.undoAction.triggered.connect(self.undo)
        self.redoAction.triggered.connect(self.redo)
        self.nextAction.triggered.connect(self.nextActionTriggered)
        self.prevAction.triggered.connect(self.prevActionTriggered)

        self.invertBwAction.toggled.connect(self.invertBw)
        self.toggleColorSchemeAction.triggered.connect(self.onToggleColorScheme)
        self.editShortcutsAction.triggered.connect(self.editShortcuts_cb)
        self.editAutoSaveIntervalAction.triggered.connect(
            self.autoSaveIntervalEditButton.click
        )
        self.showMirroredCursorAction.toggled.connect(
            self.showMirroredCursorToggled
        )

        # Connect Help actions
        self.tipsAction.triggered.connect(self.showTipsAndTricks)
        self.UserManualAction.triggered.connect(filesystem.browse_docs)
        self.openLogFileAction.triggered.connect(self.openLogFile)
        self.showLogFilesAction.triggered.connect(self.showLogFiles)
        self.aboutAction.triggered.connect(self.showAbout)
        # Connect Open Recent to dynamically populate it
        # self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)
        self.checkableQButtonsGroup.buttonClicked.connect(self.uncheckQButton)

        self.showPropsDockButton.sigClicked.connect(self.showPropsDockWidget)

        self.gui_connectCustomAnnotationsActions()
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
        self.exportCurrentViewButton.toggled.connect(
            self.exportCurrentViewDataToggled
        )
        self.searchAction.triggered.connect(
            partial(self.focusOnSearch, blinking=True)
        )
        self.zoomRectButton.toggled.connect(self.zoomRectActionToggled)
        self.autoPilotButton.toggled.connect(self.autoPilotToggled)
        self.skipToNewIdAction.triggered.connect(self.skipForwardToNewID)        
        self.slideshowButton.toggled.connect(self.launchSlideshow)
        self.launch3dViewerAction.toggled.connect(
            self.launch3dViewerButton.setChecked
        )

        
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
            self.delObjsOutSegmMaskActionTriggered
        )
        self.brushButton.toggled.connect(self.Brush_cb)
        self.eraserButton.toggled.connect(self.Eraser_cb)
        self.curvToolButton.toggled.connect(self.curvTool_cb)
        self.wandToolButton.toggled.connect(self.wand_cb)
        self.labelRoiButton.toggled.connect(self.labelRoi_cb)
        self.magicPromptsToolButton.toggled.connect(self.magicPrompts_cb)
        self.clearFreehandRoiButton.toggled.connect(self.clearFreehandRoi_cb)
        self.reInitCcaAction.triggered.connect(self.reInitCca)
        self.moveLabelToolButton.toggled.connect(self.moveLabelButtonToggled)
        self.editIDbutton.toggled.connect(self.editID_cb)
        self.editCcaToolAction.triggered.connect(
            self.manualEditCcaToolbarActionTriggered
        )
        self.assignBudMothAutoAction.triggered.connect(
            self.autoAssignBud_YeastMate
        )
        self.keepIDsButton.toggled.connect(self.keepIDs_cb)
        self.mergeIDsButton.toggled.connect(self.mergeIDs_cb)

        self.whitelistIDsButton.toggled.connect(self.whitelistIDs_cb)

        self.whitelistIDsToolbar.sigWhitelistChanged.connect(
            self.whitelistIDsChanged
        )

        self.mergeIDsToolbar.sigAccept.connect(
            self.acceptMergeMultipleIDs
        )

        self.whitelistIDsToolbar.sigWhitelistAccepted.connect(
            self.whitelistIDsAccepted
        )

        self.whitelistIDsToolbar.sigViewOGIDs.connect(self.whitelistViewOGIDs)

        self.whitelistIDsToolbar.sigAddNewIDs.connect(
            self.whitelistAddNewIDsToggled
        )

        self.whitelistIDsToolbar.sigLoadOGLabs.connect(
            self.whitelistLoadOGLabs_cb
        )

        self.whitelistIDsToolbar.sigTrackOGagainstPreviousFrame.connect(
            self.whitelistTrackOGagainstPreviousFrame_cb
        )

        self.expandLabelToolButton.toggled.connect(self.expandLabelCallback)

        self.reinitLastSegmFrameAction.triggered.connect(
            self.reInitLastSegmFrame
        )

        
        self.defaultRescaleIntensActionGroup.triggered.connect(
            self.defaultRescaleIntensLutActionToggled
        )
        
        # self.repeatAutoCcaAction.triggered.connect(self.repeatAutoCca)
        self.manuallyEditCcaAction.triggered.connect(self.manualEditCca)
        self.addScaleBarAction.toggled.connect(self.addScaleBar)
        self.addTimestampAction.toggled.connect(self.addTimestamp)
        self.saveLabColormapAction.triggered.connect(self.saveLabelsColormap)
        
        self.annotateSingleMotherBudPairButton.toggled.connect(
            self.annotateSingleMotherBudPair_cb
        )

        self.enableSmartTrackAction.toggled.connect(self.enableSmartTrack)
        # Brush/Eraser size action
        self.brushSizeSpinbox.valueChanged.connect(self.brushSize_cb)
        self.interpZButton.toggled.connect(self.interpZModeToggled)
        self.interpZConfirmAction.triggered.connect(
            self.interpZConfirmTriggered
        )
        self.autoIDcheckbox.toggled.connect(self.autoIDtoggled)
        # Mode
        self.modeActionGroup.triggered.connect(self.changeModeFromMenu)
        self.modeComboBox.sigTextChanged.connect(self.changeMode)
        self.modeComboBox.activated.connect(self.clearComboBoxFocus)
        self.equalizeHistPushButton.toggled.connect(self.equalizeHist)
        
        self.editOverlayColorAction.triggered.connect(
            self.toggleOverlayColorButton
        )
        self.editTextIDsColorAction.triggered.connect(
            self.toggleTextIDsColorButton
        )
        self.overlayColorButton.sigColorChanging.connect(self.changeOverlayColor)
        self.overlayColorButton.sigColorChanged.connect(self.saveOverlayColor)
        # Cancel: back to the saved color and LUT of the channel
        self.overlayColorButton.colorDialog.rejected.connect(
            partial(self.overlayColorDialogRejected, self.overlayColorButton)
        )
        self.textIDsColorButton.sigColorChanging.connect(
            self.updateTextAnnotColor
        )
        self.textIDsColorButton.sigColorChanged.connect(self.saveTextIDsColors)

        self.setMeasurementsAction.triggered.connect(self.showSetMeasurements)
        self.addCustomMetricAction.triggered.connect(self.addCustomMetric)
        self.addCombineMetricAction.triggered.connect(self.addCombineMetric)

        self.labelsGrad.colorButton.sigColorChanging.connect(
            self.updateBkgrColor
        )
        self.labelsGrad.colorButton.sigColorChanged.connect(self.saveBkgrColor)
        self.labelsGrad.sigGradientChangeFinished.connect(self.updateLabelsCmap)
        self.labelsGrad.sigGradientChanged.connect(self.ticksCmapMoved)
        self.labelsGrad.textColorButton.sigColorChanging.connect(
            self.updateTextLabelsColor
        )
        self.labelsGrad.textColorButton.sigColorChanged.connect(
            self.saveTextLabelsColor
        )
        self.labelsGrad.setFontSize(self.fontSizeSpinBox.value())
        self.labelsGrad.sigFontSizeChanged.connect(
            self.fontSizeChangedContextMenu
        )

        self.labelsGrad.sigShuffleCmap.connect(self.shuffle_cmap)
        self.labelsGrad.sigGreeedyShuffleCmap.connect(
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
        self.labelsGrad.sigShowNextFrameToggled.connect(
            self.showNextFrameImageItem
        )
        
        self.labelsGrad.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )

        self.imgGrad.setFontSize(self.fontSizeSpinBox.value())
        self.imgGrad.sigFontSizeChanged.connect(
            self.fontSizeChangedContextMenu
        )

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
        self.imgPropertiesAction.triggered.connect(self.editImgProperties)

        self.relabelSequentialAction.triggered.connect(
            self.relabelSequentialCallback
        )

        self.zoomToObjsAction.triggered.connect(self.zoomToObjsActionCallback)
        self.zoomOutAction.triggered.connect(self.zoomOut)
        self.preprocessAction.triggered.connect(self.preprocessActionTriggered)
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

    def nonViewerEditMenuOpened(self):
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            self.startBlinkingModeCB()

    def createUserChannelNameAction(self):
        self.userChNameAction = QAction(self)
        self.userChNameAction.setCheckable(True)
        self.userChNameAction.setText(self.user_ch_name)

    def createChannelNamesActions(self):
        # LUT histogram channel name context menu actions
        self.chNamesQActionGroup = QActionGroup(self) 
        self.chNamesQActionGroup.addAction(self.userChNameAction)
        posData = self.data[self.pos_i]
        for action in self.fluoDataChNameActions:
            self.chNamesQActionGroup.addAction(action)       
            action.setChecked(False)
        
        self.userChNameAction.setChecked(True)

        for action in self.overlayContextMenu.actions():
            action.setChecked(False)
