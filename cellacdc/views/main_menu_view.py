"""View adapter for the main menu."""

from __future__ import annotations

from qtpy.QtWidgets import QAction, QActionGroup, QMenu



class MainMenuView:
    """Qt-facing adapter around the main-menu view-model."""

    """Headless main-menu decision rules."""

    default_rescale_intensity_options = (
        'Rescale each 2D image',
        'Rescale across z-stack',
        'Rescale across time frames',
        'Do no rescale, display raw image',
    )

    def default_rescale_intensity_how(self, settings):
        try:
            return settings.at['default_rescale_intens_how', 'value']
        except Exception:
            return self.default_rescale_intensity_options[0]


    def __init__(self, host):
        self.host = host
    def create_menu_bar(self):
        menu_bar = self.host.menuBar()
        menu_bar.setNativeMenuBar(False)

        self._add_file_menu(menu_bar)
        self._add_edit_menu(menu_bar)
        self._add_view_menu(menu_bar)
        self._add_image_menu(menu_bar)
        self._add_segment_menu(menu_bar)
        self._add_tracking_menu(menu_bar)
        self._add_measurements_menu(menu_bar)
        self._add_settings_menu(menu_bar)
        self._add_mode_menu(menu_bar)
        self._add_help_menu(menu_bar)

    def _add_file_menu(self, menu_bar):
        file_menu = QMenu("&File", self.host)
        self.host.fileMenu = file_menu
        menu_bar.addMenu(file_menu)
        if self.host.debug:
            file_menu.addAction(self.host.createEmptyDataAction)
        file_menu.addAction(self.host.newAction)
        file_menu.addAction(self.host.newWindowAction)
        file_menu.addSeparator()
        file_menu.addAction(self.host.openFolderAction)
        file_menu.addAction(self.host.openFileAction)
        self.host.openRecentMenu = file_menu.addMenu("Open Recent")
        file_menu.addSeparator()
        file_menu.addAction(self.host.manageVersionsAction)
        file_menu.addAction(self.host.saveAction)
        file_menu.addAction(self.host.saveAsAction)
        file_menu.addAction(self.host.quickSaveAction)
        file_menu.addSeparator()

        self.host.exportMenu = file_menu.addMenu('Export')
        self.host.exportMenu.addAction(self.host.exportToVideoAction)
        self.host.exportMenu.addAction(self.host.exportToImageAction)
        file_menu.addSeparator()
        file_menu.addAction(self.host.loadFluoAction)
        file_menu.addAction(self.host.loadPosAction)
        self.host.fileMenu.lastSeparator = file_menu.addSeparator()
        file_menu.addAction(self.host.exitAction)

    def _add_edit_menu(self, menu_bar):
        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addSeparator()
        edit_menu.addAction(self.host.editShortcutsAction)
        edit_menu.addAction(self.host.editTextIDsColorAction)
        edit_menu.addAction(self.host.editOverlayColorAction)
        edit_menu.addAction(self.host.manuallyEditCcaAction)
        edit_menu.addAction(self.host.enableSmartTrackAction)
        edit_menu.addAction(self.host.enableAutoZoomToCellsAction)

    def _add_view_menu(self, menu_bar):
        self.host.viewMenu = menu_bar.addMenu("&View")
        self.host.viewMenu.addSeparator()
        self.host.viewMenu.addAction(self.host.viewCcaTableAction)

    def _add_image_menu(self, menu_bar):
        image_menu = menu_bar.addMenu("&Image")
        image_menu.addSeparator()
        image_menu.addAction(self.host.imgPropertiesAction)
        self.host.defaultRescaleIntensLutMenu = image_menu.addMenu(
            "Default method to rescale intensities (LUT)"
        )
        self._add_default_rescale_intensity_menu()

        image_menu.addAction(self.host.addScaleBarAction)
        image_menu.addAction(self.host.addTimestampAction)
        self.host.rescaleIntensMenu = image_menu.addMenu(
            'Rescale intensities (LUT)'
        )
        image_menu.addAction(self.host.preprocessAction)
        image_menu.addAction(self.host.combineChannelsAction)
        image_menu.addAction(self.host.saveLabColormapAction)
        image_menu.addAction(self.host.shuffleCmapAction)
        image_menu.addAction(self.host.greedyShuffleCmapAction)
        image_menu.addAction(self.host.zoomToObjsAction)
        image_menu.addAction(self.host.zoomOutAction)

    def _add_default_rescale_intensity_menu(self):
        self.host.defaultRescaleIntensActionGroup = QActionGroup(
            self.host.defaultRescaleIntensLutMenu
        )
        self.host.defaultRescaleIntensHow = (
            self.default_rescale_intensity_how(
                self.host.df_settings
            )
        )
        for how_text in self.default_rescale_intensity_options():
            action = QAction(
                how_text, self.host.defaultRescaleIntensLutMenu
            )
            action.setCheckable(True)
            if how_text == self.host.defaultRescaleIntensHow:
                action.setChecked(True)

            self.host.defaultRescaleIntensActionGroup.addAction(action)
            self.host.defaultRescaleIntensLutMenu.addAction(action)

    def _add_segment_menu(self, menu_bar):
        segment_menu = menu_bar.addMenu("&Segment")
        self.host.segmentMenu = segment_menu
        segment_menu.addSeparator()
        self.host.segmSingleFrameMenu = segment_menu.addMenu(
            'Segment displayed frame'
        )
        for action in self.host.segmActions:
            self.host.segmSingleFrameMenu.addAction(action)

        self.host.segmSingleFrameMenu.addSeparator()
        self.host.segmSingleFrameMenu.addAction(
            self.host.addCustomModelFrameAction
        )

        self.host.segmVideoMenu = segment_menu.addMenu(
            'Segment multiple frames'
        )
        for action in self.host.segmActionsVideo:
            self.host.segmVideoMenu.addAction(action)

        self.host.segmVideoMenu.addSeparator()
        self.host.segmVideoMenu.addAction(
            self.host.addCustomModelVideoAction
        )

        self.host.segmWithPromptableModelMenu = segment_menu.addMenu(
            'Segment with promptable model'
        )
        self.host.segmWithPromptableModelMenu.addAction(
            self.host.segmWithPromptableModelAction
        )
        self.host.segmWithPromptableModelMenu.addSeparator()
        self.host.segmWithPromptableModelMenu.addAction(
            self.host.addCustomPromptModelAction
        )

        segment_menu.addAction(self.host.EditSegForLostIDsSetSettings)
        segment_menu.addAction(self.host.postProcessSegmAction)
        segment_menu.addAction(self.host.autoSegmAction)
        segment_menu.addAction(self.host.relabelSequentialAction)
        segment_menu.aboutToShow.connect(
            self.host.mode_controls_view.nonViewerEditMenuOpened
        )

    def _add_tracking_menu(self, menu_bar):
        tracking_menu = menu_bar.addMenu("&Tracking")
        self.host.trackingMenu = tracking_menu
        tracking_menu.addSeparator()
        select_track_algo_menu = tracking_menu.addMenu(
            'Select real-time tracking algorithm'
        )
        for action in self.host.trackingAlgosGroup.actions():
            select_track_algo_menu.addAction(action)

        tracking_menu.addAction(self.host.editRtTrackerParamsAction)
        tracking_menu.addAction(self.host.repeatTrackingVideoAction)
        tracking_menu.addAction(self.host.repeatTrackingMenuAction)
        tracking_menu.aboutToShow.connect(
            self.host.mode_controls_view.nonViewerEditMenuOpened
        )

        if self.host.mainWin is not None:
            tracking_menu.addAction(
                self.host.mainWin.applyTrackingFromTableAction
            )
            tracking_menu.addAction(
                self.host.mainWin.applyTrackingFromTrackMateXMLAction
            )

    def _add_measurements_menu(self, menu_bar):
        measurements_menu = menu_bar.addMenu("&Measurements")
        self.host.measurementsMenu = measurements_menu
        measurements_menu.addSeparator()
        measurements_menu.addAction(self.host.setMeasurementsAction)
        measurements_menu.addAction(self.host.addCustomMetricAction)
        measurements_menu.addAction(self.host.addCombineMetricAction)
        measurements_menu.setDisabled(True)

    def _add_settings_menu(self, menu_bar):
        self.host.settingsMenu = QMenu("Settings", self.host)
        menu_bar.addMenu(self.host.settingsMenu)
        self.host.settingsMenu.addAction(self.host.invertBwAction)
        self.host.settingsMenu.addAction(self.host.toggleColorSchemeAction)
        self.host.settingsMenu.addSeparator()
        self.host.settingsMenu.addAction(self.host.pxModeAction)
        self.host.settingsMenu.addAction(self.host.highLowResAction)
        self.host.settingsMenu.addAction(self.host.editShortcutsAction)
        self.host.settingsMenu.addAction(self.host.showMirroredCursorAction)
        self.host.settingsMenu.addSeparator()
        self.host.settingsMenu.addAction(self.host.editAutoSaveIntervalAction)
        self.host.settingsMenu.addSeparator()

    def _add_mode_menu(self, menu_bar):
        self.host.modeMenu = menu_bar.addMenu('Mode')
        self.host.modeMenu.menuAction().setVisible(False)

    def _add_help_menu(self, menu_bar):
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self.host.openLogFileAction)
        help_menu.addAction(self.host.showLogFilesAction)
        help_menu.addAction(self.host.tipsAction)
        help_menu.addAction(self.host.UserManualAction)
        help_menu.addSeparator()
        help_menu.addAction(self.host.aboutAction)
        self.host.helpMenu = help_menu