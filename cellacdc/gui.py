import sys
import os

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import QMainWindow, QButtonGroup, QWidget

from . import myutils, autopilot, favourite_func_metrics_csv_path, settings_folderpath
from .myutils import setupLogger
from .gui_decorators import get_data_exception_handler, resetViewRange

custom_annot_path = os.path.join(settings_folderpath, 'custom_annotations.json')
shortcut_filepath = os.path.join(settings_folderpath, 'shortcuts.ini')
from .mixins import (
    WhitelistGui,
    DataLoading,
    CanvasRightImage,
    CanvasHover,
    LineageInteractions,
    CustomAnnotations,
    MagicPrompts,
    ObjectSearch,
    ObjectCleanup,
    SegForLostIds,
    Exporting,
    CombineWorker,
    CurvatureTools,
    DrawClearRegion,
    LabelTransformTools,
    DeletedRois,
    Saving,
    MainToolbar,
    QuickSettings,
    MainMenu,
    Measurements,
)

np.seterr(invalid='ignore')

if os.name == 'nt':
    try:
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass


class guiWin(QMainWindow,
             WhitelistGui,
             DataLoading,
             CanvasRightImage,
             CanvasHover,
             LineageInteractions,
             CustomAnnotations,
             MagicPrompts,
             ObjectSearch,
             ObjectCleanup,
             SegForLostIds,
             Exporting,
             CombineWorker,
             CurvatureTools,
             DrawClearRegion,
             LabelTransformTools,
             DeletedRois,
             Saving,
             MainToolbar,
             QuickSettings,
             MainMenu,
             Measurements):
    """Main Window."""

    sigClosed = Signal(object)
    sigExportFrame = Signal()

    def __init__(
            self, app, parent=None, buttonToRestore=None,
            mainWin=None, version=None, launcherSlot=None
        ):
        """Initializer."""

        super().__init__(parent)

        self._version = version

        from .trackers.YeaZ import tracking as tracking_yeaz
        self.tracking_yeaz = tracking_yeaz

        from .config import parser_args
        self.debug = parser_args['debug']

        self.buttonToRestore = buttonToRestore
        self.launcherSlot = launcherSlot
        self.mainWin = mainWin
        self.app = app
        self.closeGUI = False
        self._acdc_version = myutils.read_version()

        self.setAcceptDrops(True)
        self._appName = 'Cell-ACDC'

        self.lineage_tree = None
        self.already_synced_lin_tree = set()
        self.right_click_ID = None
        self.original_df_lin_tree = None
        self.original_df_lin_tree_i = None

    def run(self, module='acdc_gui', logs_path=None):
        self.setWindowIcon()
        self.setWindowTitle()

        self.is_win = sys.platform.startswith("win")
        if self.is_win:
            self.openFolderText = 'Show in Explorer...'
        else:
            self.openFolderText = 'Reveal in Finder...'

        self.is_error_state = False
        logger, logs_path, log_path, log_filename = setupLogger(
            module=module, logs_path=logs_path, caller=self._appName
        )
        if self._version is not None:
            logger.info(f'Initializing GUI v{self._version}')
        else:
            logger.info(f'Initializing GUI...')

        self.module = module
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path

        self.initProfileModels()
        self.loadLastSessionSettings()

        self.newWindows = []
        self.progressWin = None
        self.slideshowWin = None
        self.ccaTableWin = None
        self.exportToImageWindow = None
        self.customAnnotButton = None
        self.ccaCheckerRunning = False
        self.isDataLoaded = False
        self.highlightedID = 0
        self.hoverLabelID = 0
        self.expandingID = -1
        self.count = 0
        self.isDilation = True
        self.flag = True
        self.currentPropsID = 0
        self.isSegm3D = False
        self.newSegmEndName = ''
        self.closeGUI = False
        self.warnKeyPressedMsg = None
        self.img1ChannelGradients = {}
        self.AutoPilotProfile = autopilot.AutoPilotProfile()
        self.storeStateWorker = None
        self.AutoPilot = None
        self.widgetsWithShortcut = {}
        self.invertBwAlreadyCalledOnce = False
        self.zoomOutKeyValue = Qt.Key_H
        self.preprocWorker = None
        self.preprocessDialog = None
        self.viewOriginalLabels = True
        self.keepDisabled = False
        self.whitelistAddNewIDsFrame = None
        self.whitelistOriginalIDs = None
        self.whyNavigateDisabled = set()
        self.autoSaveTimer = QTimer()
        self.dirtyPointsLayerTableEndNames = set()

        self._setup_vars_combine()
        if 'autoSaveIntevalValue' not in self.df_settings.index:
            autoSaveIntevalValue = 2
            autoSaveIntervalUnit = 'minutes'
        else:
            autoSaveIntevalValue = float(
                self.df_settings.at['autoSaveIntevalValue', 'value']
            )
            autoSaveIntervalUnit = str(
                self.df_settings.at['autoSaveIntervalUnit', 'value']
            )

        self.autoSaveIntevalValueUnit = (
            autoSaveIntevalValue, autoSaveIntervalUnit
        )
        self.logger.info(
            'Autosave interval set to: '
            f'{autoSaveIntevalValue} {autoSaveIntervalUnit}'
        )

        self.checkableButtons = []
        self.LeftClickButtons = []
        self.toolsActiveInProj3Dsegm = set()
        self.customAnnotDict = {}

        self.functionsNotTested3D = []

        self.isSnapshot = False
        self.debugFlag = False
        self.pos_i = 0
        self.save_until_frame_i = 0
        self.countKeyPress = 0
        self.countRightClicks = 0
        self.xHoverImg, self.yHoverImg = None, None

        self.lastFrameRanOnFirstVisitTools = 0

        self.checkableQButtonsGroup = QButtonGroup(self)
        self.checkableQButtonsGroup.setExclusive(False)

        self.lazyLoader = None

        self.gui_createCursors()
        self.gui_createActions()
        self.gui_createMenuBar()

        self.gui_createToolBars()
        self.gui_createControlsToolbar()
        self.gui_createShowPropsButton()
        self.gui_createRegionPropsDockWidget()
        self.gui_createQuickSettingsWidgets()
        self.setTooltips()
        self.gui_populateToolSettingsMenu()

        self.autoSaveGarbageWorkers = []
        self.autoSaveActiveWorkers = []

        self.gui_connectActions()
        self.gui_createStatusBar()

        self.gui_createGraphicsPlots()
        self.gui_addGraphicsItems()

        self.gui_createImg1Widgets()
        self.gui_createLabWidgets()
        self.gui_createBottomWidgetsToBottomLayout()

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = self.gui_createMainLayout()
        self.mainLayout = mainLayout

        mainContainer.setLayout(mainLayout)

        self.isEditActionsConnected = False

        self.readRecentPaths()

        self.initShortcuts()
        self.show()
        QTimer.singleShot(100, self.resizeRangeWelcomeText)

        self.logger.info('GUI ready.')
