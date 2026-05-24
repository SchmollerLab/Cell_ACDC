"""View adapter for the main menu."""

from __future__ import annotations

from qtpy.QtWidgets import QAction, QActionGroup, QMenu


class MainMenu:
    """Extracted from guiWin."""

    def gui_createMenuBar(self):
        menuBar = self.menuBar()
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

        SegmMenu.addAction(self.EditSegForLostIDsSetSettings)
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
        menuBar.addMenu(self.settingsMenu)
        self.settingsMenu.addAction(self.invertBwAction)
        self.settingsMenu.addAction(self.toggleColorSchemeAction)
        self.settingsMenu.addSeparator()
        self.settingsMenu.addAction(self.pxModeAction)
        self.settingsMenu.addAction(self.highLowResAction)
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
        helpMenu.addSeparator()
        helpMenu.addAction(self.aboutAction)
        self.helpMenu = helpMenu
