"""Qt view adapter for the application shell."""

from __future__ import annotations

import os
import re
from datetime import timedelta

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QWidget

from cellacdc import (
    _warnings,
    base_cca_dict,
    cca_df_colnames,
    html_utils,
    settings_csv_path,
    widgets,
)

from .actions import Actions
from .session import Session


class AppShell(Actions, Session):
    """Extracted from guiWin."""

    def about(self):
        pass

    def cleanUpOnError(self):
        self.onEscape()
        caller = "Cell-ACDC"
        if self.module.startswith("spotmax"):
            caller = "spotMAX"
        txt = f"WARNING: {caller} is in error state. Please, restart."
        _hl = "*" * 100
        self.titleLabel.setText(txt, color="r")
        self.logger.info(f"{_hl}\n{txt}\n{_hl}")

    def copyContent(self):
        pass

    def cutContent(self):
        pass

    def determineSlideshowWinPos(self):
        screens = self.app.screens()
        self.numScreens = len(screens)
        winScreen = self.screen()

        # Center main window and determine location of slideshow window
        # depending on number of screens available
        if self.numScreens > 1:
            for screen in screens:
                if screen != winScreen:
                    winScreen = screen
                    break

        winScreenGeom = winScreen.geometry()
        winScreenCenter = winScreenGeom.center()
        winScreenCenterX = winScreenCenter.x()
        winScreenCenterY = winScreenCenter.y()
        winScreenLeft = winScreenGeom.left()
        winScreenTop = winScreenGeom.top()
        self.slideshowWinLeft = winScreenCenterX - int(850 / 2)
        self.slideshowWinTop = winScreenCenterY - int(800 / 2)

    def initGlobalAttr(self):
        self.setOverlayColors()

        self.initImgCmap()

        # Colormap
        self.setLut()

        self.fluoDataChNameActions = []

        self.splineHoverON = False
        self.tempSegmentON = False
        self.xyOnCtrlPressedFirstTime = None
        self.typingEditID = False
        self.prevAnnotOptions = None
        self.ghostObject = None
        self.autoContourHoverON = False
        self.navigateScrollBarStartedMoving = True
        self.zSliceScrollBarStartedMoving = True
        self.labelRoiRunning = False
        self.isRangeReset = True
        self.lastManualSeparateState = None
        self.editIDmergeIDs = True
        self.doNotAskAgainExistingID = False
        self.doubleRightClickTimeElapsed = False
        self.isRealTimeTrackerInitialized = False
        self.isWarningCcaIntegrity = False
        self.isDoubleRightClick = False
        self.isExportingVideo = False
        self.pointsLayersNeverToggled = True
        self.highlightedIDopts = None
        self.timestampStartTimedelta = timedelta(seconds=0)
        self.keptObjectsIDs = widgets.KeptObjectIDsList(
            self.keptIDsLineEdit, self.keepIDsConfirmAction
        )
        self._ZprojWidgersEnabledState = None
        self.imgValueFormatter = "d"
        self.rawValueFormatter = "d"
        self.lastHoverID = -1
        self.annotOptionsToRestore = None
        self.annotOptionsToRestoreRight = None
        self.rescaleIntensChannelHowMapper = {
            self.user_ch_name: "Rescale each 2D image"
        }
        self.timestampDialog = None
        self.scaleBarDialog = None
        self.countObjsWindow = None
        self.initLabelRoiModelDialog = None

        # Second channel used by cellpose
        self.secondChannelName = None

        self.ax1_viewRange = None
        self.measurementsWin = None

        self.model_kwargs = None
        self.segmModelName = None
        self.labelRoiModel = None
        self.autoSegmDoNotAskAgain = False
        self.labelRoiGarbageWorkers = []
        self.labelRoiActiveWorkers = []

        self.clickedOnBud = False
        self.postProcessSegmWin = None

        self.UserEnforced_DisabledTracking = False
        self.UserEnforced_Tracking = False

        self.ax1BrushHoverID = 0

        self.disabled_cca_warnings = set()

        self.last_pos_i = -1
        self.last_frame_i = -1

        # Plots items
        self.isMouseDragImg2 = False
        self.isMouseDragImg1 = False
        self.isMovingLabel = False
        self.isRightClickDragImg1 = False
        self.clickObjYc, self.clickObjXc = None, None

        self.cca_df_colnames = cca_df_colnames
        self.cca_df_dtypes = [str, int, int, str, int, int, bool, bool, int]
        self.cca_df_default_values = list(base_cca_dict.values())
        self.cca_df_int_cols = [
            col for col in cca_df_colnames if type(base_cca_dict[col]) == int
        ]
        self.lin_tree_df_bool_col = [
            col for col in cca_df_colnames if isinstance(base_cca_dict[col], bool)
        ]

        self.lin_tree_col_checks = [
            "generation_num",
        ]

        # self.lin_tree_df_colnames = set(base_cca_df.keys()) | set(lineage_tree_cols)
        # # self.lin_tree_df_dtypes = [ #dk if i need this, for now ignored
        # #     str, int, int, str, int, int, bool, bool, int
        # # ]
        # self.lin_tree_df_default_values = list(base_cca_df.values()) + lineage_tree_cols_std_val
        self.lin_tree_df_int_cols = [
            "generation_num",
            "relative_ID",
            "emerg_frame_i",
            "division_frame_i",
            "corrected_on_frame_i",
        ]
        self.lin_tree_df_bool_col = [
            "is_history_known",
        ]

        self.lin_tree_col_checks = [
            "generation_num",
        ]

        self.lin_tree_df_colnames = (
            self.lin_tree_df_int_cols
            + self.lin_tree_df_bool_col
            + self.lin_tree_col_checks
        )
        self.SegForLostIDsSettings = {}

    def initProfileModels(self):
        self.logger.info("Initiliazing profilers...")

        from ._profile.spline_to_obj import model

        self.splineToObjModel = model.Model()

        self.splineToObjModel.fit()

    def onToggleColorScheme(self):
        if self.toggleColorSchemeAction.text().find("light") != -1:
            self._colorScheme = "light"
            setDarkModeToggleChecked = False
        else:
            self._colorScheme = "dark"
            setDarkModeToggleChecked = True
        self.gui_updateSwitchColorSchemeActionText()
        _warnings.warnRestartCellACDCcolorModeToggled(
            self._colorScheme, app_name=self._appName, parent=self
        )
        load.rename_qrc_resources_file(self._colorScheme)
        self.statusBarLabel.setText(
            html_utils.paragraph(
                f"<i>Restart {self._appName} for the change to take effect</i>",
                font_color="red",
            )
        )
        self.df_settings.at["colorScheme", "value"] = self._colorScheme
        self.df_settings.to_csv(settings_csv_path)

    def openLogFile(self):
        self.logger.info(f'Opening log file "{self.log_path}"...')
        utils.showInExplorer(self.log_path)

    def openNewWindow(self):
        self.logger.info("Opening a new window...")
        if self.launcherSlot is not None:
            self.launcherSlot()
            return

        winClass = self.__class__
        win = winClass(
            self.app, parent=self, mainWin=self.mainWin, version=self._version
        )
        win.run()
        self.newWindows.append(win)

    def pasteContent(self):
        pass

    def setDisabled(
        self, disabled: bool, keepDisabled: bool = None, force: bool = False
    ):
        if force:
            if disabled:
                super().setDisabled(disabled)
                return
            else:
                self.keepDisabled = False
                super().setDisabled(disabled)
                return

        if keepDisabled is not None:
            self.keepDisabled = keepDisabled

        if self.keepDisabled:
            if disabled:
                super().setDisabled(disabled)
                return
            else:
                return
        else:
            super().setDisabled(disabled)

    def setTooltips(self):
        tooltips = load.get_tooltips_from_docs()

        for key, tooltip in tooltips.items():
            setShortcut = getattr(self, key).shortcut().toString()
            if "Shortcut: " in tooltip:
                tooltip = tooltip.replace("Shortcut: ", "\nShortcut: ")
            elif setShortcut != "":
                tooltip = re.sub(
                    r"Shortcut: \"(.*)\"", f'Shortcut: "{setShortcut}"', tooltip
                )
            else:
                tooltip = re.sub(
                    r"Shortcut: \"(.*)\"", f'Shortcut: "No shortcut"', tooltip
                )

            getattr(self, key).setToolTip(tooltip)
            getattr(self, key)._tooltip = tooltip

    def setWindowIcon(self, icon=None):
        if icon is None:
            icon = QIcon(":icon.ico")
        super().setWindowIcon(icon)

    def setWindowTitle(self, title=None):
        if title is None:
            title = f"Cell-ACDC v{self._acdc_version} - GUI"
        super().setWindowTitle(title)

    def showAbout(self):
        from cellacdc.help import about

        self.aboutWin = about.QDialogAbout(parent=self)
        self.aboutWin.show()

    def showInExplorer_cb(self):
        posData = self.data[self.pos_i]
        path = posData.images_path
        utils.showInExplorer(path)

    def showLogFiles(self):
        log_files_path = os.path.dirname(self.log_path)
        self.logger.info(f'Opening log files folder "{log_files_path}"...')
        utils.showInExplorer(log_files_path)

    def showTipsAndTricks(self):
        from cellacdc.help import welcome

        self.welcomeWin = welcome.welcomeWin()
        self.welcomeWin.showAndSetSize()
        self.welcomeWin.showPage(self.welcomeWin.quickStartItem)
