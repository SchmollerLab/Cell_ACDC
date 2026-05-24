import gc
import sys
import os
import shutil
import re
import traceback
import time
from copy import deepcopy
from datetime import datetime, timedelta
import inspect
import logging
import uuid
from collections import defaultdict, Counter
import zipfile
from functools import partial
from natsort import natsorted
from typing import Literal, Iterable, Dict, Any, List, Union, Set

import time
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib
import scipy.interpolate
import scipy.ndimage
import skimage
import skimage.io
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation

from functools import wraps
from skimage.color import gray2rgb, gray2rgba, label2rgb

from qtpy.QtCore import (
    Qt, QPoint, QTextStream, QSize, QRect, QRectF,
    QEventLoop, QTimer, QEvent, Signal,
    QThread, QMutex, QWaitCondition, QSettings, PYQT6
)
from qtpy.QtGui import (
    QIcon, QCursor, QGuiApplication, QColor,
    QFont, QMouseEvent
)
from qtpy.QtWidgets import (
    QAction, QLabel, QPushButton, QHBoxLayout, QSizePolicy,
    QMainWindow, QMenu, QToolBar, QGroupBox, QGridLayout,
    QScrollBar, QCheckBox, QToolButton, QSpinBox, QButtonGroup, QActionGroup, QFileDialog, QAbstractSlider, QMessageBox, QWidget, QGridLayout, 
    QDockWidget, QGraphicsProxyWidget, QVBoxLayout, QRadioButton, 
    QSpacerItem, QScrollArea, QFormLayout, QGraphicsSceneMouseEvent 
)

import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Custom modules
from . import (
    base_cca_dict,
)
from . import graphLayoutBkgrColor, darkBkgrColor
from . import cca_df_colnames
from . import load, prompts, apps, workers, html_utils
from . import core, myutils, dataPrep, widgets
from . import _warnings
from . import measurements, printl
from . import colors, annotate
from . import user_manual_url
from . import settings_folderpath, settings_csv_path
from . import favourite_func_metrics_csv_path
from . import qutils, autopilot, QtScoped
from . import data_structure_docs_url
from . import exporters
from . import io
from . import whitelist
from . import cli
from .trackers.CellACDC import CellACDC_tracker
from .myutils import exec_time, setupLogger, ArgSpec
from . import debugutils
from .plot import imshow
from . import gui_utils


np.seterr(invalid='ignore')

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

_font = QFont()
_font.setPixelSize(11)

font_13px = QFont()
font_13px.setPixelSize(13)

SliderSingleStepAdd = QtScoped.SliderSingleStepAdd()
SliderSingleStepSub = QtScoped.SliderSingleStepSub()
SliderPageStepAdd = QtScoped.SliderPageStepAdd()
SliderPageStepSub = QtScoped.SliderPageStepSub()
SliderMove = QtScoped.SliderMove()

from .viewmodels import MainGuiViewModel


class guiWin(QMainWindow):
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
        self.view_model = MainGuiViewModel()
        self.window_events_view = WindowEventsView(self)
        self.window_events_view.bind_legacy_methods()
        self.tracking_view = TrackingView(self)
        self.tracking_view.bind_legacy_methods()
        self.image_display_view = ImageDisplayView(self)
        self.image_display_view.bind_legacy_methods()
        self.data_loading_view = DataLoadingView(self)
        self.data_loading_view.bind_legacy_methods()
        self.cell_cycle_view = CellCycleView(self)
        self.cell_cycle_view.bind_legacy_methods()
        self.graphics_view = GraphicsView(self)
        self.graphics_view.bind_legacy_methods()
        self.actions_view = ActionsView(self)
        self.actions_view.bind_legacy_methods()
        self.app_shell_view = AppShellView(self)
        self.app_shell_view.bind_legacy_methods()
        self.annotation_display_view = AnnotationDisplayView(self)
        self.annotation_display_view.bind_legacy_methods()
        self.session_view = SessionView(self)
        self.session_view.bind_legacy_methods()
        self.frame_navigation_view = FrameNavigationView(self)
        self.frame_navigation_view.bind_legacy_methods()
        self.canvas_drawing_view = CanvasDrawingView(self)
        self.canvas_drawing_view.bind_legacy_methods()
        self.canvas_events_view = CanvasEventsView(self)
        self.canvas_events_view.bind_legacy_methods()
        self.canvas_selection_view = CanvasSelectionView(self)
        self.canvas_selection_view.bind_legacy_methods()
        self.canvas_context_menu_view = CanvasContextMenuView(self)
        self.canvas_right_image_view = CanvasRightImageView(self)
        self.canvas_hover_view = CanvasHoverView(self)
        self.canvas_hover_view.bind_legacy_methods()
        self.label_roi_view = LabelRoiView(self)
        self.label_roi_view.bind_legacy_methods()
        self.label_editing_view = LabelEditingView(self)
        self.label_editing_view.bind_legacy_methods()
        self.lineage_interactions_view = LineageInteractionsView(self)
        self.lineage_interactions_view.bind_legacy_methods()
        self.custom_annotations_view = CustomAnnotationsView(self)
        self.undo_redo_view = UndoRedoView(self)
        self.undo_redo_view.bind_legacy_methods()
        self.worker_view = WorkerView(self)
        self.worker_view.bind_legacy_methods()
        self.brush_tools_view = BrushToolsView(self)
        self.brush_tools_view.bind_legacy_methods()
        self.deleted_rois_view = DeletedRoisView(self)
        self.deleted_rois_view.bind_legacy_methods()
        self.draw_clear_region_view = DrawClearRegionView(self)
        self.display_decorations_view = DisplayDecorationsView(self)
        self.object_cleanup_view = ObjectCleanupView(self)
        self.object_properties_view = ObjectPropertiesView(self)
        self.object_properties_view.bind_legacy_methods()
        self.object_search_view = ObjectSearchView(self)
        self.curvature_tools_view = CurvatureToolsView(
            self,
            self.view_model.curvature,
        )
        self.seg_for_lost_ids_view = SegForLostIdsView(self)
        self.segmentation_view = SegmentationView(self)
        self.segmentation_view.bind_legacy_methods()
        self.saving_view = SavingView(self)
        self.saving_view.bind_legacy_methods()
        self.mode_controls_view = ModeControlsView(self)
        self.image_controls_view = ImageControlsView(self)
        self.preprocessing_view = PreprocessingView(self)
        self.magic_prompts_view = MagicPromptsView(self)
        self.exporting_view = ExportingView(self)
        self.main_toolbar_view = MainToolbarView(self)
        self.main_menu_view = MainMenuView(self)
        self.label_transform_tools_view = LabelTransformToolsView(self)
        self.measurements_view = MeasurementsView(self)
        self.quick_settings_view = QuickSettingsView(self)
        self.status_hover_view = StatusHoverView(self)
        self.points_layers_view = PointsLayersView(self)
        self.points_layers_view.bind_legacy_methods()
        self.tool_activation_view = ToolActivationView(self)
        self.tool_activation_view.bind_legacy_methods()
        self.layout_controls_view = LayoutControlsView(self)
        self.layout_controls_view.bind_legacy_methods()
        self.combine_view = CombineView(self)
        self.combine_view.bind_legacy_methods()
        self.whitelist_view = WhitelistView(self)
        self.whitelist_view.bind_legacy_methods()
        self.canvas_tool_view = CanvasToolView(self)
        self._acdc_version = self.view_model.app_shell.read_version()

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

        # Keep a list of functions that are not functional in 3D, yet
        self.functionsNotTested3D = []

        self.isSnapshot = False
        self.debugFlag = False
        self.pos_i = 0
        self.save_until_frame_i = 0
        self.countKeyPress = 0
        self.countRightClicks = 0
        self.xHoverImg, self.yHoverImg = None, None
        
        # Keep track on what frames the on first visit tools already ran
        self.lastFrameRanOnFirstVisitTools = 0
        
        # Buttons added to QButtonGroup will be mutually exclusive
        self.checkableQButtonsGroup = QButtonGroup(self)
        self.checkableQButtonsGroup.setExclusive(False)

        self.lazyLoader = None

        self.gui_createCursors()
        self.gui_createActions()
        self.main_menu_view.create_menu_bar()

        self.main_toolbar_view.gui_createToolBars()
        self.gui_createControlsToolbar()
        self.quick_settings_view.create_show_props_button()
        self.gui_createRegionPropsDockWidget()
        self.quick_settings_view.create_widgets()
        self.setTooltips()
        self.gui_populateToolSettingsMenu()

        self.autoSaveGarbageWorkers = []
        self.autoSaveActiveWorkers = []

        self.gui_connectActions()
        self.gui_createStatusBar()
        # self.gui_createTerminalWidget()

        self.image_controls_view.gui_createGraphicsPlots()
        self.gui_addGraphicsItems()

        self.image_controls_view.gui_createImg1Widgets()
        self.image_controls_view.gui_createLabWidgets()
        self.image_controls_view.gui_createBottomWidgetsToBottomLayout()

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
        # self.installEventFilter(self)

        self.logger.info('GUI ready.')


from .views.combine_view import CombineView
from .views.whitelist_view import WhitelistView
from .views.app_shell_view import AppShellView
from .views.actions_view import ActionsView
from .views.brush_tools_view import BrushToolsView
from .views.annotation_display_view import AnnotationDisplayView
from .views.canvas_context_menu_view import CanvasContextMenuView
from .views.canvas_drawing_view import CanvasDrawingView
from .views.canvas_events_view import CanvasEventsView
from .views.canvas_selection_view import CanvasSelectionView
from .views.canvas_right_image_view import CanvasRightImageView
from .views.canvas_tool_view import CanvasToolView
from .views.canvas_hover_view import CanvasHoverView
from .views.cell_cycle_view import CellCycleView
from .views.custom_annotations_view import CustomAnnotationsView
from .views.curvature_tools_view import CurvatureToolsView
from .views.display_decorations_view import DisplayDecorationsView
from .views.status_hover_view import StatusHoverView
from .views.deleted_rois_view import DeletedRoisView
from .views.data_loading_view import DataLoadingView
from .views.draw_clear_region_view import DrawClearRegionView
from .views.exporting_view import ExportingView
from .views.graphics_view import GraphicsView
from .views.frame_navigation_view import FrameNavigationView
from .views.image_controls_view import ImageControlsView
from .views.image_display_view import ImageDisplayView
from .views.label_editing_view import LabelEditingView
from .views.label_roi_view import LabelRoiView
from .views.label_transform_tools_view import LabelTransformToolsView
from .views.lineage_interactions_view import LineageInteractionsView
from .views.magic_prompts_view import MagicPromptsView
from .views.measurements_view import MeasurementsView
from .views.layout_controls_view import LayoutControlsView
from .views.main_menu_view import MainMenuView
from .views.mode_controls_view import ModeControlsView
from .views.object_search_view import ObjectSearchView
from .views.object_properties_view import ObjectPropertiesView
from .views.object_cleanup_view import ObjectCleanupView
from .views.points_layers_view import PointsLayersView
from .views.preprocessing_view import PreprocessingView
from .views.quick_settings_view import QuickSettingsView
from .views.saving_view import SavingView
from .views.seg_for_lost_ids_view import SegForLostIdsView
from .views.segmentation_view import SegmentationView
from .views.session_view import SessionView
from .views.tool_activation_view import ToolActivationView
from .views.main_toolbar_view import MainToolbarView
from .views.tracking_view import TrackingView
from .views.undo_redo_view import UndoRedoView
from .views.window_events_view import WindowEventsView
from .views.worker_view import WorkerView
