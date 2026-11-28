"""Class guiWin, the main GUI window for annotating and correcting
segmentation, tracking and cell cycle data.

The class body holds the base classes (QMainWindow, then the window-only
<Topic>GuiElements mixins of gui/main_window/, then the mixins shared
with the measurements viewer from gui/mixins/), the Signals, the
overriding class attributes (KEY_EVENT_KEYS), __init__, the Qt-named
overrides and the methods that call super(). Every other method lives in
the mixin of its topic. The module also runs the import-time setup of
the main GUI: the pyqtgraph row-major image axis order, the pandas
PerformanceWarning filter, np.seterr and the Windows AppUserModelID.
"""
import gc
import sys
import os
import shutil
import re
import textwrap
import traceback
from contextlib import contextmanager
import time
from copy import deepcopy
from datetime import datetime, timedelta
import inspect
import logging
import uuid
import json
from collections import defaultdict, Counter, deque
import psutil
import zipfile
from functools import partial
from tqdm import tqdm
from natsort import natsorted
from typing import Literal, Iterable, Dict, Any, List, Union, Tuple, Set
from itertools import combinations

import time
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib
import scipy.optimize
import scipy.interpolate
import scipy.ndimage
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation

from functools import wraps
from skimage.color import gray2rgb, gray2rgba, label2rgb

from qtpy.QtCore import (
    Qt, QPoint, QPointF, QTextStream, QSize, QRect, QRectF,
    QEventLoop, QTimer, QEvent, QObject, Signal,
    QThread, QMutex, QWaitCondition, QSettings, PYQT6,
)
from qtpy.QtGui import (
    QIcon, QKeySequence, QCursor, QGuiApplication, QPen, QPixmap, QColor,
    QFont, QKeyEvent, QMouseEvent, QPainter
)
from qtpy.QtWidgets import (
    QAction, QLabel, QPushButton, QHBoxLayout, QSizePolicy,
    QMainWindow, QMenu, QToolBar, QGroupBox, QGridLayout,
    QScrollBar, QCheckBox, QToolButton, QSpinBox, QButtonGroup, QActionGroup, QFileDialog, QAbstractSlider, QMessageBox, QWidget, QGridLayout, 
    QDockWidget, QGraphicsProxyWidget, QVBoxLayout, QRadioButton, 
    QSpacerItem, QScrollArea, QFormLayout, QGraphicsSceneMouseEvent, QApplication,
    QAbstractButton
)

import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Custom modules
from cellacdc import _window_stacking
from cellacdc import exception_handler, disableWindow
from cellacdc import base_cca_dict, lineage_tree_cols, lineage_tree_cols_std_val
from cellacdc import graphLayoutBkgrColor, darkBkgrColor
from cellacdc import cca_df_colnames
from cellacdc import load, prompts, html_utils
from cellacdc.shared.dialogs import generic as dialogs_generic
from cellacdc import core
from cellacdc.myutils import app_info
from cellacdc.data_prep import main_window as dataPrep
from cellacdc.shared.widgets import message_box as widgets_message_box
from cellacdc import _warnings, issues_url
from cellacdc import measurements, printl
from cellacdc import colors, annotate
from cellacdc import user_manual_url
from cellacdc import (settings_folderpath, settings_csv_path, 
               seg_for_lost_IDs_settings_path)
from cellacdc import favourite_func_metrics_csv_path
from cellacdc import shortcut_filepath
from cellacdc import qutils, autopilot, QtScoped
from cellacdc import _palettes
from cellacdc import transformation
from cellacdc import measure
from cellacdc import cca_functions
from cellacdc import exporters
from cellacdc import preprocess
from cellacdc import io
from . import whitelist_ids as gui_main_whitelist_ids
from cellacdc import cli
from cellacdc import is_mac
from cellacdc import urls
from cellacdc import path as acdc_path
from cellacdc.cca_functions import _calc_rot_vol
from cellacdc.help import welcome, about
from cellacdc.trackers.CellACDC_normal_division.CellACDC_normal_division_tracker import (
    normal_division_lineage_tree, is_lineage_tree_annotated,
    init_lineage_tree_cols_of_added_IDs, has_lineage_tree_values,
    lineage_tree_columns, lineage_tree_annotated_frames,
    LINEAGE_TREE_SHARED_CCA_COLS
)#, reorg_sister_cells_for_export)
from cellacdc import debugutils
from cellacdc import exec_time
from cellacdc.plot import imshow
from . import image_processing as gui_main_image_processing
from ..mixins import navigation as gui_navigation
from ..mixins import annotations as gui_annotations
from ..mixins import overlay as gui_overlay
from ..mixins import (
    object_info as gui_object_info,
    viewer_tools as gui_viewer_tools,
    window_core as gui_window_core,
)
from ..mixins import custom_annotations as gui_custom_annotations
from cellacdc.config import STANDARD_MOUSE_BUTTONS
from cellacdc import rst_utils
from cellacdc import core_split_IDs
from cellacdc.promptable_models import utils as prompt_utils
from cellacdc.promptable_models import saved_models
from . import constants as gui_main_constants
from .window_setup import WindowGuiElements
from .layout import LayoutGuiElements
from .menus_actions import MenusActionsGuiElements
from .toolbars import ToolbarsGuiElements
from .hover_cursors import CursorsGuiElements
from .hover_cursors import HoverEventsGuiElements
from .mouse_events import MouseEventsGuiElements
from .mouse_left_image import LeftImageMouseGuiElements
from .keyboard import KeyboardGuiElements
from .modes import ModesGuiElements
from .frame_navigation import MainNavigationGuiElements
from .frame_data import FrameDataGuiElements
from .frame_data import UndoRedoGuiElements
from .image_display import ImageDisplayGuiElements
from .image_display import TitleStatusGuiElements
from .drawing_tools import DrawingToolsGuiElements
from .drawing_tools import ManualBackgroundGuiElements
from .object_editing import ObjectEditingGuiElements
from .object_editing import KeepIDsGuiElements
from .rois import DeleteRoisGuiElements
from .rois import LabelRoiGuiElements
from .segmentation import SegmentationGuiElements
from .segmentation import SegForLostIDsGuiElements
from .magic_prompts import MagicPromptsGuiElements
from .prompt_annotations import BoxPromptsGuiElements
from .prompt_annotations import BrushStrokesPromptsGuiElements
from .tracking import TrackingGuiElements
from .tracking import LostObjectsGuiElements
from .tracking import ManualTrackingGuiElements
from .cca_annotation import CcaManualAnnotationGuiElements
from .cca_data import CcaDataGuiElements
from .cca_data import CcaIntegrityCheckerGuiElements
from .cca_data import AnnotationsAfterEditsGuiElements
from .lineage_tree import LineageTreeGuiElements
from .points_layers import PointsLayersGuiElements
from .loading import LoadDataGuiElements
from .loading import OpenFileGuiElements
from .saving import SaveDataGuiElements
from .saving import AutoSaveGuiElements
from .saving import SetMeasurementsGuiElements
from .exports import ExportsGuiElements
from .image_processing import PreprocessingGuiElements
np.seterr(invalid='ignore')

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass
      
class guiWin(
        QMainWindow,
        WindowGuiElements,
        LayoutGuiElements,
        MenusActionsGuiElements,
        ToolbarsGuiElements,
        CursorsGuiElements,
        HoverEventsGuiElements,
        MouseEventsGuiElements,
        LeftImageMouseGuiElements,
        KeyboardGuiElements,
        ModesGuiElements,
        MainNavigationGuiElements,
        FrameDataGuiElements,
        UndoRedoGuiElements,
        ImageDisplayGuiElements,
        TitleStatusGuiElements,
        DrawingToolsGuiElements,
        ManualBackgroundGuiElements,
        ObjectEditingGuiElements,
        KeepIDsGuiElements,
        DeleteRoisGuiElements,
        LabelRoiGuiElements,
        SegmentationGuiElements,
        SegForLostIDsGuiElements,
        MagicPromptsGuiElements,
        BoxPromptsGuiElements,
        BrushStrokesPromptsGuiElements,
        TrackingGuiElements,
        LostObjectsGuiElements,
        ManualTrackingGuiElements,
        CcaManualAnnotationGuiElements,
        CcaDataGuiElements,
        CcaIntegrityCheckerGuiElements,
        AnnotationsAfterEditsGuiElements,
        LineageTreeGuiElements,
        PointsLayersGuiElements,
        LoadDataGuiElements,
        OpenFileGuiElements,
        SaveDataGuiElements,
        AutoSaveGuiElements,
        SetMeasurementsGuiElements,
        ExportsGuiElements,
        PreprocessingGuiElements,
        gui_main_whitelist_ids.WhitelistGUIElements,
        gui_main_image_processing.CombineGuiElements,
        gui_main_image_processing.CombineGUIWorker,
        gui_navigation.FramesNavigationGuiElements,
        gui_navigation.ZsliceNavigationGuiElements,
        gui_navigation.ZoomToObjGuiElements,
        gui_navigation.RecentPathsGuiElements,
        gui_navigation.LutGuiElements,
        gui_annotations.AnnotationsGuiElements,
        gui_custom_annotations.CustomAnnotationsGuiElements,
        gui_viewer_tools.ZoomGuiElements,
        gui_viewer_tools.RulerGuiElements,
        gui_viewer_tools.ScaleBarTimestampGuiElements,
        gui_viewer_tools.ColorSchemeGuiElements,
        gui_viewer_tools.LabelsColormapGuiElements,
        gui_viewer_tools.ImagePropertiesGuiElements,
        gui_object_info.CcaTableGuiElements,
        gui_object_info.CountObjectsGuiElements,
        gui_window_core.WorkersGuiElements,
        gui_object_info.SearchIDGuiElements,
        gui_object_info.HoverInfoGuiElements,
        gui_window_core.ShortcutsGuiElements,
        gui_viewer_tools.SlideshowGuiElements,
        gui_viewer_tools.Volume3DViewerGuiElements,
        gui_object_info.RegionPropsGuiElements,
        gui_viewer_tools.BottomLayoutGuiElements,
        gui_overlay.OverlayChannelsGuiElements,
        gui_overlay.OverlayLabelsGuiElements,
        gui_overlay.ChannelImageHelpersGuiElements,
        gui_window_core.DefaultSettingsGuiElements,
        gui_window_core.GUICoreElements,
    ):
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

        from cellacdc.trackers.YeaZ import tracking as tracking_yeaz
        self.tracking_yeaz = tracking_yeaz

        from cellacdc.config import parser_args
        self.debug = parser_args['debug']
        self._debug = self.debug

        self.buttonToRestore = buttonToRestore
        self.launcherSlot = launcherSlot
        self.mainWin = mainWin
        self.app = app
        self.closeGUI = False
        self._acdc_version = app_info.read_version()

        self.setAcceptDrops(True)
        self._appName = 'Cell-ACDC'

        self.lineage_tree = None
        self.already_synced_lin_tree = set()
        self.right_click_ID = None
        self.original_df_lin_tree = None
        self.original_df_lin_tree_i = None

        self._waitOverrideActive = False
        self._busyTimer = QTimer(self)
        self._busyTimer.setInterval(100)
        self._busyTimer.timeout.connect(self.gui_refreshToolCursor)
    
    def setWindowIcon(self, icon=None):
        if icon is None:
            icon = QIcon(":icon.ico")
        super().setWindowIcon(icon)
    
    def setWindowTitle(self, title=None):
        if title is None:
            title = f'Cell-ACDC v{self._acdc_version} - GUI'
        super().setWindowTitle(title)
    
    def setDisabled(self,
                    disabled: bool,
                    keepDisabled: bool = None,
                    force: bool = False):
        if force:
            if not disabled:
                self.keepDisabled = False
            apply = True
        else:
            if keepDisabled is not None:
                self.keepDisabled = keepDisabled
            # while keepDisabled is set, only disabling is allowed
            apply = disabled or not self.keepDisabled

        if apply:
            super().setDisabled(disabled)

        # starts/stops the busy timer and sets/restores the cursor
        self.gui_refreshToolCursor()

    def dragEnterEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(file_path):
            exp_path = file_path
            basename = os.path.basename(file_path)
            if basename.find('Position_')!=-1 or basename=='Images':
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.acceptProposedAction()

    def dropEvent(self, event):
        event.setDropAction(Qt.CopyAction)
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.logger.info(f'Dragged and dropped path "{file_path}"')
        basename = os.path.basename(file_path)
        if os.path.isdir(file_path):
            exp_path = file_path
            self.openFolder(exp_path=exp_path)
        else:
            self.openFile(file_path=file_path)

    def changeEvent(self, event):
        try:
            self.delObjToolAction.setChecked(False)
        except Exception as err:
            return
    
    def leaveEvent(self, event):
        if self.slideshowWin is not None:
            posData = self.data[self.pos_i]
            mainWinGeometry = self.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinRight = mainWinLeft+mainWinWidth
            mainWinBottom = mainWinTop+mainWinHeight

            slideshowWinGeometry = self.slideshowWin.geometry()
            slideshowWinLeft = slideshowWinGeometry.left()
            slideshowWinTop = slideshowWinGeometry.top()
            slideshowWinWidth = slideshowWinGeometry.width()
            slideshowWinHeight = slideshowWinGeometry.height()

            # Determine if overlap
            overlap = (
                (slideshowWinTop < mainWinBottom) and
                (slideshowWinLeft < mainWinRight)
            )

            autoActivate = (
                self.isDataLoaded and not
                overlap and not
                posData.disableAutoActivateViewerWindow
            )

            if autoActivate:
                self.slideshowWin.setFocus()
                self.slideshowWin.activateWindow()

    def enterEvent(self, event):
        event.accept()
        if self.slideshowWin is not None:
            posData = self.data[self.pos_i]
            mainWinGeometry = self.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinRight = mainWinLeft+mainWinWidth
            mainWinBottom = mainWinTop+mainWinHeight

            slideshowWinGeometry = self.slideshowWin.geometry()
            slideshowWinLeft = slideshowWinGeometry.left()
            slideshowWinTop = slideshowWinGeometry.top()
            slideshowWinWidth = slideshowWinGeometry.width()
            slideshowWinHeight = slideshowWinGeometry.height()

            # Determine if overlap
            overlap = (
                (slideshowWinTop < mainWinBottom) and
                (slideshowWinLeft < mainWinRight)
            )

            autoActivate = (
                self.isDataLoaded and not
                overlap and not
                posData.disableAutoActivateViewerWindow
            )

            if autoActivate:
                # self.setFocus()
                self.activateWindow()
    
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            pos = self.resizeBottomLayoutLine.mapFromGlobal(event.globalPos())
            if pos.y()>=0:
                self.gui_raiseBottomLayoutContextMenu(event)
        if event.button() not in STANDARD_MOUSE_BUTTONS:
            if not self.isDataLoaded:
                self.logger.warning(
                    'Data not loaded yet. Key pressing events are not connected.'
                )
                return
            for name, button in self.mouseBindings.items():
                if not button == event.button():
                    continue
                action = self.widgetsWithShortcut[name]
                if hasattr(action, 'click'):
                    action.click()
                elif hasattr(action, 'trigger'):
                    action.trigger()
                return
        return super().mousePressEvent(event)

    def eventFilter(self, obj, ev):
        if not getattr(self, 'isDataLoaded', False):
            return super().eventFilter(obj, ev)

        evType = ev.type()
        if evType == QtScoped.QEventTypeAttribute('MouseButtonRelease'):
            self.onMouseRelease()

        if evType not in (QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress):
            # Installed on the app (all events, e.g., ~40 per mouse move)
            # --> return early for the events that are not handled below
            return super().eventFilter(obj, ev)

        if self.getFocusedInputWidget() is not None:
            return super().eventFilter(obj, ev)

        # The filter also sees the clicks that Qt ignores for a disabled
        # scroll bar (e.g., the GUI busy, or the lost cells prompt open)
        if (
                ev.type() == QEvent.MouseButtonPress
                and obj == self.navigateScrollBar
                and self.navigateScrollBar.isEnabled()
            ):
            self.navigateScrollBar.handleMousePressEventFromAppFilter(ev)

        # handle presistant shortcuts (for multiple shportcuts). The filter
        # is installed on the app: leave the key presses of windows not owned
        # by the GUI (e.g., the arrow keys of the measurements viewer) alone.
        # Not while the GUI (busy, or disabled by the measurements viewer's
        # `runInThread`) or the key's target is disabled: the filter also
        # sees the key presses that Qt ignores for disabled widgets, the
        # actions do not follow the GUI's enabled state, and a dialog that
        # they open would nest its event loop in the viewer's wait. Nor
        # while the action itself is disabled (`triggerPersistentShortcut`)
        if (
            ev.type() == QEvent.Type.KeyPress
            and self.isEnabled()
            and qutils.is_key_event_target_enabled(obj)
            and qutils.is_key_event_for_window(
                self, obj, owned_windows=self.ownedWindowsOutsideParentChain()
            )
            and self.triggerPersistentShortcut(ev.key())
        ):
            return True

        return super().eventFilter(obj, ev)

    @exception_handler
    def keyPressEvent(self, ev):
        ctrl = ev.modifiers() == Qt.ControlModifier
        if ctrl and ev.key() == Qt.Key_D:
            self.resizeLeaveSpaceTerminalBelow()
            return

        if ev.key() == Qt.Key_Q and self.debug:
            try:
                from cellacdc import _q_debug
                _q_debug.q_debug(self, ev)
            except Exception as err:
                printl(traceback.format_exc())
                printl('[ERROR]: Error with "_q_debug" module. See Traceback above.')
                pass

        modifiers = ev.modifiers()
        isAltModifier = modifiers == Qt.AltModifier
        isCtrlModifier = modifiers == Qt.ControlModifier
        isShiftModifier = modifiers == Qt.ShiftModifier

        if not self.isDataLoaded:
            is_passable_keycombin = (
                ev.key() == Qt.Key_Control
                or ev.key() == Qt.Key_Alt
                or ev.key() == Qt.Key_Shift
                or ev.key() == Qt.Key_Escape
                or (ev.key() == Qt.Key_F and isCtrlModifier)
            )
            if not is_passable_keycombin:
                self.logger.warning(
                    'Data not loaded yet. Key pressing events are not connected.'
                )
                return

        applied = self.applyEditIDKeyModifiers(modifiers)
        if applied:
            return

        applied = self.applySplitIDsKeyModifiers(modifiers)
        if applied:
            return

        if ev.key() == Qt.Key_Control:
            if not ctrl:
                self.wasCtrlPressedFirstTime = True
                self.onCtrlPressedFirstTime()
        
        if ev.key() == Qt.Key_PageDown:
            self.onKeyPageDown()
        
        if ev.key() == Qt.Key_PageUp:
            self.onKeyPageUp()
        
        if ev.key() == Qt.Key_Home:
            self.onKeyHome()
        
        if ev.key() == Qt.Key_End:
            self.onKeyEnd()

        self.triggerPersistentShortcut(ev.key())

        self.checkSetDelObjActionActive(ev)
        
        self.isZmodifier = (
            ev.key()== Qt.Key_Z and not isAltModifier
            and not isCtrlModifier and not isShiftModifier
        )
        if isShiftModifier:
            if self.brushButton.isChecked():
                # Force default brush symbol with shift down
                self.setHoverToolSymbolColor(
                    1, 1, self.ax2_BrushCirclePen,
                    (self.ax2_BrushCircle, self.ax1_BrushCircle),
                    self.brushButton, brush=self.ax2_BrushCircleBrush,
                    ID=0
                )
            if self.isSegm3D:
                self.changeBrushID()
                
        if (
            ev.key() == Qt.Key_Shift
            and isShiftModifier
            and not ev.isAutoRepeat()
            and self.mergeIDsToolbar.onlyCurrentZsliceCheckbox.isVisible()
            ):
            self.mergeIDsToolbar_onlyCurrentZsliceCheckbox_og_state = (   
                self.mergeIDsToolbar.onlyCurrentZsliceCheckbox.isChecked()
            )
            self.mergeIDsToolbar.onlyCurrentZsliceCheckbox.setChecked(True)

        if (
            ev.key() == Qt.Key_Shift
            and isShiftModifier
            and not ev.isAutoRepeat()
            and self.separateBudToolbar.sepModeSelector.isVisible()
            ):
            self.separateBudToolbar_sepModeSelector_og_state = (
                self.separateBudToolbar.separationMode()
            )
            self.separateBudToolbar.setSeparationMode(
                'Slice by slice'
            )
        
        isAnyModifier = isAltModifier or isCtrlModifier or isShiftModifier
        if not isAnyModifier and self.overlayButton.isChecked():
            isButtonClicked = self.checkOverlayToolbuttonClicked(ev)
            if isButtonClicked:
                return            
             
        isBrushActive = (
            self.brushButton.isChecked() or self.eraserButton.isChecked()
        )
        isManualTrackingActive = self.manualTrackingButton.isChecked()
        isManualBackgroundActive = self.manualBackgroundButton.isChecked()
        isTypingIDFunctionChecked = False
        if self.brushButton.isChecked() and not self.autoIDcheckbox.isChecked():
            success = self.keyPressCheckSetSpinboxValue(ev, self.editIDspinbox)
            isTypingIDFunctionChecked = True
        
        if isManualTrackingActive:
            isTypingIDFunctionChecked = self.keyPressCheckSetSpinboxValue(
                ev, self.manualTrackingToolbar.spinboxID
            )
        
        elif isManualBackgroundActive:
            isTypingIDFunctionChecked = self.keyPressCheckSetSpinboxValue(
                ev, self.manualBackgroundToolbar.spinboxID
            )
        
        addPointsByClickingButton = self.buttonAddPointsByClickingActive()
        if (
                addPointsByClickingButton is not None 
                and addPointsByClickingButton.toolbar.isVisible()
            ):
            isTypingIDFunctionChecked = self.keyPressCheckSetSpinboxValue(
                ev, addPointsByClickingButton.rightClickIDSpinbox
            )
        
        isBrushKey, isEraserKey = self.checkTriggerKeyPressShortcuts(ev)
        isExpandLabelActive = self.expandLabelToolButton.isChecked()
        isWandActive = self.wandToolButton.isChecked()
        isLabelRoiCircActive = (
            self.labelRoiButton.isChecked() 
            and self.labelRoiIsCircularRadioButton.isChecked()
        )
        isOverlaySegmLeft = self.isOverlaySegmMaskChecked(0)
        if ev.key()==Qt.Key_Up and not isCtrlModifier:
            self.keyUpCallback(
                isBrushActive, isWandActive, isExpandLabelActive, 
                isLabelRoiCircActive
            )
        elif ev.key()==Qt.Key_Down and not isCtrlModifier:
            self.keyDownCallback(
                isBrushActive, isWandActive, isExpandLabelActive, 
                isLabelRoiCircActive
            )
        elif ev.key() == Qt.Key_Enter or ev.key() == Qt.Key_Return:
            if ev.isAutoRepeat() or self.isEnterConfirmingTypedValue():
                # Enter confirming the value typed in a spin box or line
                # edit, or held down (e.g., the repeats after confirming the
                # frame number, since the spin boxes with `disableKeyPress`
                # clear their focus and do not pass the first Enter on)
                # --> do not apply the active tool (e.g., "Keep IDs" with
                # the IDs selected so far)
                pass
            elif self.interpZButton.isChecked():
                self.interpZConfirmAction.trigger()
            elif isTypingIDFunctionChecked:
                self.typingEditID = False
            elif self.keepIDsButton.isChecked():
                # Enter in the IDs line edit is handled (and accepted) by the
                # line edit. Check that the action is enabled because Qt5
                # triggers disabled actions too
                if self.keepIDsConfirmAction.isEnabled():
                    self.keepIDsConfirmAction.trigger()
            elif self.isBrushStrokesSelectCellsActive():
                # Enter in the IDs line edit is handled by the line edit
                self.brushStrokesSelectCellsToolbar.emitSigAddToTrainingData()
        elif ev.key() == Qt.Key_Escape:
            isEscapeHandled = self.brushStrokesSelectCellsEscape()
            if not isEscapeHandled:
                self.onEscape(
                    isTypingIDFunctionChecked=isTypingIDFunctionChecked
                )
        elif isAltModifier:
            isCursorSizeAll = self.app.overrideCursor() == Qt.SizeAllCursor
            # Alt is pressed while cursor is on images --> set SizeAllCursor
            if self.xHoverImg is not None and not isCursorSizeAll:
                self.app.setOverrideCursor(Qt.SizeAllCursor)
        elif isCtrlModifier and isOverlaySegmLeft:
            if ev.key() == Qt.Key_Up:
                val = self.imgGrad.labelsAlphaSlider.value()
                delta = 5/self.imgGrad.labelsAlphaSlider.maximum()
                val = val+delta
                self.imgGrad.labelsAlphaSlider.setValue(val, emitSignal=True)
            elif ev.key() == Qt.Key_Down:
                val = self.imgGrad.labelsAlphaSlider.value()
                delta = 5/self.imgGrad.labelsAlphaSlider.maximum()
                val = val-delta
                self.imgGrad.labelsAlphaSlider.setValue(val, emitSignal=True)
        elif ev.key() == self.zoomOutKeyValue:
            self.zoomToCells(enforce=True)
            if self.countKeyPress == 0:
                self.isKeyDoublePress = False
                self.countKeyPress = 1
                self.doubleKeyTimeElapsed = False
                self.Button = None
                QTimer.singleShot(400, self.doubleKeyTimerCallBack)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                self.ax1.autoRange()
                self.isKeyDoublePress = True
                self.countKeyPress = 0
        elif ev.key() == Qt.Key_Space:
            if self.countKeyPress == 0:
                # Single press --> wait that it's not double press
                self.isKeyDoublePress = False
                self.countKeyPress = 1
                self.doubleKeyTimeElapsed = False
                QTimer.singleShot(300, self.doubleKeySpacebarTimerCallback)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                self.isKeyDoublePress = True
                # Double press --> toggle draw nothing
                self.onDoubleSpaceBar()
                self.countKeyPress = 0
        elif isBrushKey or isEraserKey:
            if isBrushKey:
                self.Button = self.brushButton
            else:
                self.Button = self.eraserButton

            isButtonAvailable = (
                self.Button.isVisible()
                or self.editToolBar.isWidgetHiddenByOverflow(self.Button)
            )
            if not isButtonAvailable:
                return

            if self.countKeyPress == 0:
                # If first time clicking B activate brush and start timer
                # to catch double press of B
                if not self.Button.isChecked():
                    self.uncheck = False
                    self.Button.setChecked(True)
                else:
                    self.uncheck = True
                self.countKeyPress = 1
                self.isKeyDoublePress = False
                self.doubleKeyTimeElapsed = False

                QTimer.singleShot(400, self.doubleKeyTimerCallBack)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                self.isKeyDoublePress = True
                color = self.Button.palette().button().color().name()
                if color == self.doublePressKeyButtonColor:
                    c = self.defaultToolBarButtonColor
                else:
                    c = self.doublePressKeyButtonColor
                self.Button.setStyleSheet(f'background-color: {c}')
                self.countKeyPress = 0
                if self.xHoverImg is not None:
                    xdata, ydata = int(self.xHoverImg), int(self.yHoverImg)
                    if isBrushKey:
                        self.setHoverToolSymbolColor(
                            xdata, ydata, self.ax2_BrushCirclePen,
                            (self.ax2_BrushCircle, self.ax1_BrushCircle),
                            self.brushButton, brush=self.ax2_BrushCircleBrush
                        )
                    elif isEraserKey:
                        self.setHoverToolSymbolColor(
                            xdata, ydata, self.eraserCirclePen,
                            (self.ax2_EraserCircle, self.ax1_EraserCircle),
                            self.eraserButton
                        )
    
    def keyReleaseEvent(self, ev):
        is_modifier = False
        if self.app.overrideCursor() == Qt.SizeAllCursor:
            self.app.restoreOverrideCursor()
        if ev.key() == Qt.Key_Control:
            self.onCtrlReleased()
            is_modifier = True
        elif ev.key() == Qt.Key_Shift:
            self.onShiftReleased()
            is_modifier = True
        elif ev.key() == Qt.Key_Alt:
            is_modifier = True
        
        self.resetEditIDOptions()
        self.resetSplitIDsOptions()
            
        canRepeat = (
            ev.key() == Qt.Key_Left
            or ev.key() == Qt.Key_Right
            or ev.key() == Qt.Key_Up
            or ev.key() == Qt.Key_Down
            or ev.key() == Qt.Key_Control
            or ev.key() == Qt.Key_Backspace
            or self.checkSetDelObjActionActive(ev, checked=False)
            or is_modifier
            or self.delObjToolAction.isChecked()
        )

        if (
            ev.key() == Qt.Key_Shift
            and not ev.isAutoRepeat()
            and getattr(
                self,
                'separateBudToolbar_sepModeSelector_og_state',
                None,
            ) is not None
        ):
            self.separateBudToolbar.setSeparationMode(
                self.separateBudToolbar_sepModeSelector_og_state
            )
            self.separateBudToolbar_sepModeSelector_og_state = None
            
        if (
            ev.key() == Qt.Key_Shift
            and not ev.isAutoRepeat()
            and getattr(
                self,
                'mergeIDsToolbar_onlyCurrentZsliceCheckbox_og_state',
                None,
            ) is not None
        ):
            self.mergeIDsToolbar.onlyCurrentZsliceCheckbox.setChecked(
                self.mergeIDsToolbar_onlyCurrentZsliceCheckbox_og_state
            )
            self.mergeIDsToolbar_onlyCurrentZsliceCheckbox_og_state = None
        
        if canRepeat and ev.isAutoRepeat():
            return
        
        if ev.isAutoRepeat() and not ev.key() == Qt.Key_Z:
            if self.warnKeyPressedMsg is not None:
                return
            self.warnKeyPressedMsg = widgets_message_box.myMessageBox(
                showCentered=False, wrapText=False
            )
            txt = html_utils.paragraph(f"""
            Please, <b>do not keep the key "{ev.text().upper()}" 
            pressed.</b><br><br>
            It confuses me :)<br><br>
            Thanks!
            """)
            self.warnKeyPressedMsg.warning(self, 'Release the key, please', txt)
            self.warnKeyPressedMsg = None
        elif ev.isAutoRepeat() and ev.key() == Qt.Key_Z and self.isZmodifier:
            self.zKeptDown = True
        elif ev.key() == Qt.Key_Z and self.isZmodifier:
            posData = self.data[self.pos_i]
            self.isZmodifier = False
            if not self.zKeptDown and posData.SizeZ > 1:
                self.zSliceCheckbox.setChecked(not self.zSliceCheckbox.isChecked())
            self.zKeptDown = False

    # Keys handled in `keyPressEvent` and `keyReleaseEvent` (not shortcuts
    # of actions or buttons)
    KEY_EVENT_KEYS = {
        'Left': 'Go to previous frame',
        'Right': 'Go to next frame',
        'Up': 'Up arrow (navigation and tool size)',
        'Down': 'Down arrow (navigation and tool size)',
        'PgUp': 'Page up (navigation)',
        'PgDown': 'Page down (navigation)',
        'Home': 'Next z-slice',
        'End': 'Previous z-slice',
        'Space': 'Hide annotations (double press)',
        'Esc': 'Escape (cancel the current action)',
        'Return': 'Confirm (Enter)',
        'Enter': 'Confirm (Enter)',
        'Z': 'Toggle z-slice/projection (Z key)',
        # Opacity of the overlaid segmentation (`isOverlaySegmLeft`)
        'Ctrl+Up': 'Increase overlay segmentation opacity',
        'Ctrl+Down': 'Decrease overlay segmentation opacity',
    }
    # Digits toggle the overlay channels (`checkOverlayToolbuttonClicked`,
    # 0 = base layer)
    KEY_EVENT_KEYS.update({
        str(n): f'Toggle overlay channel {n} (overlay active)'
        for n in range(10)
    })
    
    def closeEvent(self, event):
        self.setDisabled(False)
        cancel = self.checkAskSavePointsLayers()
        if cancel:
            event.ignore()
            return
    
        self.onEscape()
        self.saveWindowGeometry()
        self.saveAnnotationOptions()
        
        if self.newWindows:
            cancel = self.askCloseAllWindows()
            if cancel:
                event.ignore()
                return

            for window in self.newWindows:
                window.close()

        if self.slideshowWin is not None:
            self.slideshowWin.close()
        if self.ccaTableWin is not None:
            self.ccaTableWin.close()
        
        proceed = self.askSaveOnClosing(event)
        if not proceed:
            event.ignore()
            return

        self.autoSaveClose()
        
        if self.autoSaveActiveWorkers:
            progressWin = dialogs_generic.QDialogWorkerProgress(
                title='Closing autosaving worker', parent=self,
                pbarDesc='Closing autosaving worker...'
            )
            progressWin.show(self.app)
            progressWin.mainPbar.setMaximum(0)
            self.waitCloseAutoSaveWorkerLoop = qutils.QWhileLoop(
                self._waitCloseAutoSaveWorker, period=250
            )
            self.waitCloseAutoSaveWorkerLoop.exec_()
            progressWin.workerFinished = True
            progressWin.close()
        
        self.stopPreprocWorker()
        self.stopCombineWorker()
        self.stopCcaIntegrityCheckerWorker(wait=True)
        self.waitCcaCheckerOrphanThreads()
        self.stopModelExecutions()
        
        # Close the inifinte loop of the thread
        if self.lazyLoader is not None:
            self.lazyLoader.exit = True
            self.lazyLoaderWaitCond.wakeAll()
            self.waitReadH5cond.wakeAll()
        
        if self.storeStateWorker is not None:
            # Close storeStateWorker
            self.storeStateWorker._stop()
            while self.storeStateWorker.isFinished:
                time.sleep(0.05)
        
        # Block main thread while separate threads closes
        time.sleep(0.1)

        self.clearMemory()

        self.logger.info('Closing GUI logger...')
        self.logger.close()
        
        if self.lazyLoader is None:
            self.sigClosed.emit(self)
        
        gc.collect()

    def showEvent(self, event):
        if self.mainWin is not None:
            if not self.mainWin.isMinimized():
                return
            self.mainWin.showAllWindows()
        # self.setFocus()
        self.activateWindow()
        self.searchWidget.set_height_based_on(self.initMenuBar)
    
    def super_show(self):
        super().show()

    def show(self):
        self.setFont(gui_main_constants._font)
        QMainWindow.show(self)

        self.setWindowState(Qt.WindowNoState)
        self.setWindowState(Qt.WindowActive)
        self.raise_()

        self.readSettings()
        self.storeDefaultAndCustomColors()

        self.h = self.navSpinBox.size().height()
        fontSizeFactor = None
        heightFactor = None
        if 'bottom_sliders_zoom_perc' in self.df_settings.index:
            val = int(self.df_settings.at['bottom_sliders_zoom_perc', 'value'])
            if val != 100:
                fontSizeFactor = val/100
                heightFactor = val/100

        self.defaultWidgetHeightBottomLayout = self.h
        self.checkBoxesHeight = 14
        self.fontPixelSize = 11
        self.defaultBottomLayoutHeight = self.img1BottomGroupbox.height()
        
        self.bottomLayout.setStretch(0, 0)
        self.bottomLayout.addSpacing(self.quickSettingsGroupbox.width())
        self.resizeSlidersArea(
            fontSizeFactor=fontSizeFactor, heightFactor=heightFactor
        )
        self.bottomScrollArea.hide()

        self.gui_initImg1BottomWidgets()
        self.img1BottomGroupbox.hide()

        w = self.showPropsDockButton.width()
        h = self.showPropsDockButton.height()

        self.showPropsDockButton.setMaximumWidth(15)
        self.showPropsDockButton.setMaximumHeight(120)
        
        for toolbar in self.controlToolBars:
            toolbar.setMinimumHeight(
                self.secondLevelToolbar.sizeHint().height()
            )

        self.graphLayout.setFocus()

    def resizeEvent(self, event):
        if hasattr(self, 'ax1'):
            self.ax1.autoRange()
