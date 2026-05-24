"""View adapter for the main menu."""

from __future__ import annotations

from qtpy.QtWidgets import QAction, QActionGroup, QMenu


class MainMenuMixin:
    """Qt-facing adapter around the main-menu view-model."""

    """Headless main-menu decision rules."""

    default_rescale_intensity_options = ()

    def _add_default_rescale_intensity_menu(self):
        self.defaultRescaleIntensActionGroup = QActionGroup(
            self.defaultRescaleIntensLutMenu
        )
        self.defaultRescaleIntensHow = self.default_rescale_intensity_how(
            self.df_settings
        )
        for how_text in self.default_rescale_intensity_options():
            action = QAction(how_text, self.defaultRescaleIntensLutMenu)
            action.setCheckable(True)
            if how_text == self.defaultRescaleIntensHow:
                action.setChecked(True)

            self.defaultRescaleIntensActionGroup.addAction(action)
            self.defaultRescaleIntensLutMenu.addAction(action)

    def _add_edit_menu(self, menu_bar):
        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addSeparator()
        edit_menu.addAction(self.editShortcutsAction)
        edit_menu.addAction(self.editTextIDsColorAction)
        edit_menu.addAction(self.editOverlayColorAction)
        edit_menu.addAction(self.manuallyEditCcaAction)
        edit_menu.addAction(self.enableSmartTrackAction)
        edit_menu.addAction(self.enableAutoZoomToCellsAction)

    def _add_file_menu(self, menu_bar):
        file_menu = QMenu("&File", self)
        self.fileMenu = file_menu
        menu_bar.addMenu(file_menu)
        if self.debug:
            file_menu.addAction(self.createEmptyDataAction)
        file_menu.addAction(self.newAction)
        file_menu.addAction(self.newWindowAction)
        file_menu.addSeparator()
        file_menu.addAction(self.openFolderAction)
        file_menu.addAction(self.openFileAction)
        self.openRecentMenu = file_menu.addMenu("Open Recent")
        file_menu.addSeparator()
        file_menu.addAction(self.manageVersionsAction)
        file_menu.addAction(self.saveAction)
        file_menu.addAction(self.saveAsAction)
        file_menu.addAction(self.quickSaveAction)
        file_menu.addSeparator()

        self.exportMenu = file_menu.addMenu("Export")
        self.exportMenu.addAction(self.exportToVideoAction)
        self.exportMenu.addAction(self.exportToImageAction)
        file_menu.addSeparator()
        file_menu.addAction(self.loadFluoAction)
        file_menu.addAction(self.loadPosAction)
        self.fileMenu.lastSeparator = file_menu.addSeparator()
        file_menu.addAction(self.exitAction)

    def _add_help_menu(self, menu_bar):
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self.openLogFileAction)
        help_menu.addAction(self.showLogFilesAction)
        help_menu.addAction(self.tipsAction)
        help_menu.addAction(self.UserManualAction)
        help_menu.addSeparator()
        help_menu.addAction(self.aboutAction)
        self.helpMenu = help_menu

    def _add_image_menu(self, menu_bar):
        image_menu = menu_bar.addMenu("&Image")
        image_menu.addSeparator()
        image_menu.addAction(self.imgPropertiesAction)
        self.defaultRescaleIntensLutMenu = image_menu.addMenu(
            "Default method to rescale intensities (LUT)"
        )
        self._add_default_rescale_intensity_menu()

        image_menu.addAction(self.addScaleBarAction)
        image_menu.addAction(self.addTimestampAction)
        self.rescaleIntensMenu = image_menu.addMenu("Rescale intensities (LUT)")
        image_menu.addAction(self.preprocessAction)
        image_menu.addAction(self.combineChannelsAction)
        image_menu.addAction(self.saveLabColormapAction)
        image_menu.addAction(self.shuffleCmapAction)
        image_menu.addAction(self.greedyShuffleCmapAction)
        image_menu.addAction(self.zoomToObjsAction)
        image_menu.addAction(self.zoomOutAction)

    def _add_measurements_menu(self, menu_bar):
        measurements_menu = menu_bar.addMenu("&Measurements")
        self.measurementsMenu = measurements_menu
        measurements_menu.addSeparator()
        measurements_menu.addAction(self.setMeasurementsAction)
        measurements_menu.addAction(self.addCustomMetricAction)
        measurements_menu.addAction(self.addCombineMetricAction)
        measurements_menu.setDisabled(True)

    def _add_mode_menu(self, menu_bar):
        self.modeMenu = menu_bar.addMenu("Mode")
        self.modeMenu.menuAction().setVisible(False)

    def _add_segment_menu(self, menu_bar):
        segment_menu = menu_bar.addMenu("&Segment")
        self.segmentMenu = segment_menu
        segment_menu.addSeparator()
        self.segmSingleFrameMenu = segment_menu.addMenu("Segment displayed frame")
        for action in self.segmActions:
            self.segmSingleFrameMenu.addAction(action)

        self.segmSingleFrameMenu.addSeparator()
        self.segmSingleFrameMenu.addAction(self.addCustomModelFrameAction)

        self.segmVideoMenu = segment_menu.addMenu("Segment multiple frames")
        for action in self.segmActionsVideo:
            self.segmVideoMenu.addAction(action)

        self.segmVideoMenu.addSeparator()
        self.segmVideoMenu.addAction(self.addCustomModelVideoAction)

        self.segmWithPromptableModelMenu = segment_menu.addMenu(
            "Segment with promptable model"
        )
        self.segmWithPromptableModelMenu.addAction(self.segmWithPromptableModelAction)
        self.segmWithPromptableModelMenu.addSeparator()
        self.segmWithPromptableModelMenu.addAction(self.addCustomPromptModelAction)

        segment_menu.addAction(self.EditSegForLostIDsSetSettings)
        segment_menu.addAction(self.postProcessSegmAction)
        segment_menu.addAction(self.autoSegmAction)
        segment_menu.addAction(self.relabelSequentialAction)
        segment_menu.aboutToShow.connect(
            self.mode_controls_view.nonViewerEditMenuOpened
        )

    def _add_settings_menu(self, menu_bar):
        self.settingsMenu = QMenu("Settings", self)
        menu_bar.addMenu(self.settingsMenu)
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

    def _add_tracking_menu(self, menu_bar):
        tracking_menu = menu_bar.addMenu("&Tracking")
        self.trackingMenu = tracking_menu
        tracking_menu.addSeparator()
        select_track_algo_menu = tracking_menu.addMenu(
            "Select real-time tracking algorithm"
        )
        for action in self.trackingAlgosGroup.actions():
            select_track_algo_menu.addAction(action)

        tracking_menu.addAction(self.editRtTrackerParamsAction)
        tracking_menu.addAction(self.repeatTrackingVideoAction)
        tracking_menu.addAction(self.repeatTrackingMenuAction)
        tracking_menu.aboutToShow.connect(
            self.mode_controls_view.nonViewerEditMenuOpened
        )

        if self.mainWin is not None:
            tracking_menu.addAction(self.mainWin.applyTrackingFromTableAction)
            tracking_menu.addAction(self.mainWin.applyTrackingFromTrackMateXMLAction)

    def _add_view_menu(self, menu_bar):
        self.viewMenu = menu_bar.addMenu("&View")
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.viewCcaTableAction)

    def create_menu_bar(self):
        menu_bar = self.menuBar()
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

    def default_rescale_intensity_how(self, settings):
        try:
            return settings.at["default_rescale_intens_how", "value"]
        except Exception:
            return self.default_rescale_intensity_options[0]
