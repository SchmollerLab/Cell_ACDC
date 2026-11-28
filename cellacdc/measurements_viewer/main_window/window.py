"""Class MeasurementsViewerWin, the measurements viewer window, and run(),
its standalone entry point.

The class body holds the base classes (QMainWindow, then the
Viewer<Topic>GuiElements mixins of measurements_viewer/main_window/,
then MeasurementsViewerUtils, then the mixins shared with the main GUI
from gui/mixins/), the class attributes, __init__, the Qt-named
overrides and the overrides that call super(). The specification of the
viewer is the docstring of measurements_viewer/__init__.py.
"""
import os
import sys
import copy
import html
import json
import keyword
import re
import logging
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd

import skimage.measure

from qtpy.QtCore import (
    Qt, Signal, QTimer, QPoint, QEvent, QRect, QSaveFile, QIODevice,
    QThread, QMutex, QWaitCondition, QObject, QEventLoop, QCoreApplication
)
from qtpy.QtGui import (
    QFont, QIcon, QGuiApplication, QPixmap, QPainter, QCursor, QKeySequence,
    QColor, QMouseEvent
)
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QLabel, QGridLayout, QVBoxLayout, QHBoxLayout,
    QGroupBox, QSplitter, QLineEdit,
    QCheckBox, QComboBox, QAbstractItemView, QFileDialog, QAction,
    QApplication, QToolButton, QPushButton, QActionGroup, QAbstractSlider,
    QWidgetAction, QMenu, QColorDialog, QStyleOptionViewItem, QTreeView,
    QAbstractButton, QSizePolicy
)

import pyqtgraph as pg

from cellacdc import core, load, colors, transformation, qutils
from cellacdc import urls
from cellacdc import cellacdc_path
from cellacdc import (
    graphLayoutBkgrColor, darkBkgrColor, settings_csv_path, settings_folderpath
)
from cellacdc import cca_df_colnames, base_cca_dict
from cellacdc import regionprops as acdc_regionprops
from cellacdc.gui.mixins import (
    navigation as gui_navigation, annotations as gui_annotations
)
from cellacdc.gui.mixins import (
    object_info as gui_object_info,
    viewer_tools as gui_viewer_tools,
    window_core as gui_window_core,
)
from cellacdc.gui.mixins import overlay as gui_overlay
from cellacdc.config import STANDARD_MOUSE_BUTTONS
from cellacdc import html_utils
from cellacdc import timepoint_predictors
from .. import custom_annotation_predictors
from .. import phase_translator
from cellacdc import measurement_processing, background_values
from cellacdc.gui.mixins import custom_annotations as gui_custom_annotations
from cellacdc.gui.mixins.custom_annotations import (
    custom_annot_state_to_save, custom_annot_symbol
)
from ..custom_annotation_predictors import CustomAnnotationPredictor
from cellacdc.timepoint_predictors import (
    TimepointIdentifier, TIMEPOINT_PREFIX, parse_flags
)
from cellacdc import disableWindow
from ..viewer_utils import MeasurementsViewerUtils, get_logger
from .. import widgets as viewer_widgets
from . import navigation as viewer_navigation
from .ui_setup import ViewerSetupGuiElements
from .configuration import ViewerConfigurationGuiElements
from .data_io import ViewerDataIOGuiElements
from .navigation import ViewerNavigationGuiElements
from .image_panel import ViewerImagePanelGuiElements
from .plotting import ViewerPlottingGuiElements
from .columns import ViewerColumnsGuiElements
from .processing import ViewerProcessingGuiElements
from .phases import ViewerPhasesGuiElements
from .timepoints import ViewerTimepointsGuiElements
from .timepoints import ViewerPredictorEvaluationGuiElements
from .custom_annotations import ViewerCustomAnnotationsGuiElements


pg.setConfigOption('imageAxisOrder', 'row-major')

class MeasurementsViewerWin(
        QMainWindow,
        ViewerSetupGuiElements,
        ViewerConfigurationGuiElements,
        ViewerDataIOGuiElements,
        ViewerNavigationGuiElements,
        ViewerImagePanelGuiElements,
        ViewerPlottingGuiElements,
        ViewerColumnsGuiElements,
        ViewerProcessingGuiElements,
        ViewerPhasesGuiElements,
        ViewerTimepointsGuiElements,
        ViewerPredictorEvaluationGuiElements,
        ViewerCustomAnnotationsGuiElements,
        MeasurementsViewerUtils,
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
        gui_window_core.DefaultSettingsGuiElements,
        gui_overlay.OverlayChannelsGuiElements,
        gui_overlay.OverlayLabelsGuiElements,
        gui_overlay.ChannelImageHelpersGuiElements,
        gui_window_core.GUICoreElements,
    ):
    """Window to view images, navigate cells and plot their measurements.

    Parameters
    ----------
    data : list of `load.loadData`, optional
        Loaded positions (see `load_positions`). Each `posData` must have 
        `img_data` and `segm_data` with the time axis first. If None, the 
        window starts empty and data can be loaded from `File` menu.
    pos_i : int, optional
        Index of the position to display at startup. Default is 0.
    frame_i : int, optional
        Index of the frame to display at startup. Default is `posData.frame_i` 
        if present, otherwise 0.
    ID : int, optional
        Cell ID to select at startup. Default is the first ID.
    isSegm3D : bool, optional
        Whether the segmentation masks are 3D. If None, it is inferred from 
        the shape of `posData.segm_data`.
    invertBw : bool, optional
        If True, invert black/white (light background) as with the action 
        `Invert black/white` of the main GUI. Default is the value saved 
        in the settings.
    """
    sigClosed = Signal(object)
    # Columns of the Navigate grid (`gui_createNavigationWidgets`): label,
    # spin box, size label ('/N'), slider, then the extras (z-projection
    # combobox, ...) up to `NAV_NUM_COLS`
    NAV_LABEL_COL = 0
    NAV_SPINBOX_COL = 1
    NAV_SIZE_COL = 2
    NAV_SLIDER_COL = 3
    NAV_EXTRA_COL = 4
    NAV_NUM_COLS = 7
    # The bottom layout is different from guiWin's --> separate setting
    bottomLayoutHeightSettingKey = 'measurements_viewer_bottomLayoutHeight'
    # No delete-object tool --> hide its row in the shortcut editor
    showDeleteObjectShortcut = False

    def __init__(
            self, data=None, pos_i=0, frame_i=None, ID=None, isSegm3D=None,
            invertBw=False, parent=None
        ):
        super().__init__(parent)
        self.data = []
        self.pos_i = pos_i
        self.isSegm3D = isSegm3D
        self.isBwInverted = invertBw
        
        self.IDs = []
        self.selectedID = None
        self._IDframesCache = {}
        self.acdc_df = None
        self._lab = None
        self._labKey = None
        self.isDataLoaded = False
        self.user_ch_name = None
        self.configurationFilepath = None
        self.savedConfiguration = None
        self._startupConfigurationPrompted = False
        self._geometryRestored = False
        self.measurementProcessingThread = None
        self.measurementProcessingWorker = None
        # True while a load (`loadConfiguration`/`loadImagesPaths`, with
        # their dialogs) runs: other loads and closing are refused
        self._isLoadingData = False
        # True while the "Save changes before closing?" prompt is open:
        # loading is refused (`loadingInProgress`)
        self._isClosingPrompt = False
        self.rescaleIntensChannelHowMapper = {}
        self.timepointIdentifiers = []
        # Phase rules in descending priority; custom annotations with the
        # "Overrides phases" option are checked before them
        # (`phaseOverrideAnnotations`)
        self.phaseTranslatorRules = []
        # True while custom annotations are added/removed in bulk
        # (`bulkSettingCustomAnnotations`): no phase re-evaluation
        self._isBulkSettingCustomAnnotations = False
        # User-chosen phase colors {phase name: '#rrggbb'}
        self.phaseColors = {}
        self.timepointActions = {}
        self.timepointAutoActions = {}
        self.timepointLines = []
        self.timepointAutoPredictions = defaultdict(set)
        self.configurationInputs = []
        # inputOptionKey of the loaded (or last saved) input option, which
        # `configuration()` updates in place
        self.loadedInputKey = None
        self.measurementProcessingSteps = []
        self.combinedMeasurementSteps = []
        # Per-cell constants [{'name': str, 'equation': str}]
        # self.cellConstants = []
        self.logger = get_logger()
        self.loadSettings()
        # Re-evaluate phases of cells whose time points or phase-relevant
        # custom annotations change or whose phases are missing (saved in
        # the configuration)
        self.autoApplyPhases = False
        # Ignore frames after the last analyzed frame of each position in
        # all calculations (saved in the configuration; see
        # `analysisLastFrame`)
        self.maskUnanalyzedFrames = True
        # {`position_key`: last frame index} of the positions whose recorded
        # measurement step outputs were calculated only up to that frame
        # (NaN after it, read back as 0 from tables with cell cycle
        # columns); saved in the configuration
        self.measurementStepsLastFrames = {}
        # {id(posData): (posData, set of Cell_IDs)} awaiting phase auto-apply
        self._pendingPhaseIDs = {}
        self._phaseAutoApplyScheduled = False
        # {id(posData): (posData, set of phase names)} to clear because the
        # annotations overriding them are gone and nothing else can evaluate
        # phases (`schedulePhaseOverrideCells`)
        self._pendingPhaseClears = {}
        self._phaseClearScheduled = False
        self.initAnnotationsAttributes()
        self.initViewerToolsAttributes()
        self.initCustomAnnotationsAttributes()
        if sys.platform.startswith('win'):
            self.openFolderText = 'Show in Explorer...'
        else:
            self.openFolderText = 'Reveal in Finder...'
        
        self.setWindowTitle('Cell-ACDC - Measurements viewer')
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        
        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()
        self.gui_connectActions()
        self.gui_createGraphicsPlots()
        self.gui_createNavigationWidgets()
        self.gui_createAnnotationsWidgets()
        self.gui_createMeasurementsPanel()
        self.gui_createLayout()
        self.gui_createStatusBar()
        self.gui_connectSignals()
        self.gui_connectViewerToolsSignals()
        self.initShortcuts()
        self.readRecentPaths()
        self.postInitLoadSetSettings()

        self.setData(data, pos_i=pos_i, frame_i=frame_i, ID=ID)
        # Arrow keys navigate whatever widget has the focus (`eventFilter`)
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        # Installed on the application: the arrow keys reach `keyPressEvent`
        # only if the focus widget ignores them, while buttons and check
        # boxes use them to move the focus (e.g., the first toolbar button,
        # focused when the window is activated again after a dialog)
        if (
            event.type() == QEvent.Type.KeyPress
            and self.isArrowNavigationKeyPress(obj, event)
        ):
            self.navigateWithArrowKey(event.key())
            return True
        return super().eventFilter(obj, event)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._startupConfigurationPrompted:
            self._startupConfigurationPrompted = True
            QTimer.singleShot(0, self.loadConfiguration)
        # Restore the geometry saved on close only once: Qt also sends
        # (spontaneous) show events when the window is restored from minimized
        if not self._geometryRestored:
            self._geometryRestored = True
            self.readSettings(setting_suffix='_measurements_viewer')

    def setShortcuts(self, shortcuts: dict, save=True):
        super().setShortcuts(shortcuts, save=save)
        # A tool key edited in the shortcuts editor (or read from the
        # shortcuts.ini shared with the main GUI) can collide with a time
        # point or annotation key: check them again (the colliding one is
        # left unset with a warning, and set again once its key is free)
        self.updateTimepointAndAnnotationShortcuts()
    
    def resetDefaultAnnotationOptions(self, *args):
        super().resetDefaultAnnotationOptions(*args)
        self.syncTextAnnotOptions()
        if self.isDataLoaded:
            self.updateImages()

    def addAlphaScrollbar(self, channelName, imageItem):
        """Add the alpha slider of an overlay channel
        (`OverlayChannelsGuiElements.addAlphaScrollbar`) and move it to the
        columns of the Navigate grid: label from the label column, slider
        in the slider column."""
        row = self.alphaScrollbarRow
        alphaScrollBar = super().addAlphaScrollbar(channelName, imageItem)
        # Same height as the other sliders of the column
        alphaScrollBar.setFixedHeight(self.idScrollBar.maximumHeight())
        label = alphaScrollBar.label
        for widget in (label, alphaScrollBar):
            self.bottomLeftLayout.removeWidget(widget)
        self.bottomLeftLayout.addWidget(
            label, row, self.NAV_LABEL_COL,
            1, self.NAV_SLIDER_COL - self.NAV_LABEL_COL
        )
        self.bottomLeftLayout.addWidget(
            alphaScrollBar, row, self.NAV_SLIDER_COL
        )
        return alphaScrollBar

    def overlay_cb(self, checked):
        result = super().overlay_cb(checked)
        self.setStatusBarLabel(log=False)
        return result

    def overlayChannelToggled(self, checked):
        # Slot of the overlay context menu actions: `sender()` (read by the
        # mixin) is still the toggled action here
        super().overlayChannelToggled(checked)
        self.setStatusBarLabel(log=False)

    def onToggleColorScheme(self):
        # The mixin writes the restart notice into `statusBarLabel` (kept
        # until data is loaded again in the main GUI): show it as a
        # temporary message instead (the warning dialog also says it)
        super().onToggleColorScheme()
        self.setStatusBarLabel(log=False)
        self.statusBar().showMessage(
            f'Restart {self._appName} for the change to take effect.', 15000
        )
    
    def onMotherDaughterLine(self, checked, checkbox, ax=0):
        if checked:
            # Previous frame data required by `drawAllLineageTreeLines`,
            # which the mixin slot calls (frames without cca annotations)
            self.initLineageTreeFrameData()
        super().onMotherDaughterLine(checked, checkbox, ax=ax)

    def updatePropsWidget(self, ID, fromHover=False):
        # Writing the Object ID from code must not select the cell
        wasUpdating = self._isUpdatingPropsDock
        self._isUpdatingPropsDock = True
        try:
            super().updatePropsWidget(ID, fromHover=fromHover)
        finally:
            self._isUpdatingPropsDock = wasUpdating

    def propsWidgetIDvalueChanged(self, ID):
        """The Object ID of the "Cell-ACDC objects" dock changed: a user edit
        selects that cell (as clicking it), code updates (hover, sync with
        the selected cell) only show its properties."""
        if self._isUpdatingPropsDock or not self.isDataLoaded:
            return
        ID = int(ID)
        if ID != self.selectedID and ID in self.IDs:
            self.logger.info(f'Selecting Cell ID {ID} from the objects dock')
            # Selecting updates the images, which sync the dock
            self.goToObjectID(ID)
            return
        super().propsWidgetIDvalueChanged(ID)
    
    def launchSlideshow(self):
        super().launchSlideshow()
        win = self.slideshowWin
        if win is None:
            return
        # The slideshow shows the viewer's z-slice, projection and plane
        # (`imageViewer.update_img` asks `getImage`). Its own z-slider
        # would only write the main GUI's `segmInfo_df` (not used here)
        win.zSliceScrollBar.setDisabled(True)
        win.zSliceScrollBar.setVisible(False)
        win.z_label.setVisible(False)
    
    # --- Custom annotations -------------------------------------------------
    # Overrides of the `gui_custom_annotations.CustomAnnotationsGuiElements` 
    # hooks. The annotations are stored as 0/1 columns of the acdc_df of each 
    # position (the frame tables of `posData.allData_li` are temporary copies).
    def addCustomAnnotation(self):
        if not self.data:
            self.statusBar().showMessage(
                'Load data to add custom annotations.', 5000
            )
            return
        return super().addCustomAnnotation()
    
    def loadCustomAnnotations(self):
        if not self.data:
            self.statusBar().showMessage(
                'Load data to load custom annotations.', 5000
            )
            return
        with self.deferPredictorPredictions():
            return super().loadCustomAnnotations()

    def reinitCustomAnnot(self):
        with self.bulkSettingCustomAnnotations():
            super().reinitCustomAnnot()

    def addCustomAnnnotScatterPlot(self, symbolColor, symbol, toolButton):
        super().addCustomAnnnotScatterPlot(symbolColor, symbol, toolButton)
        # Overlay layers are re-added at z-value 0 on every load: keep the
        # symbols above them (annotation items 0.5-0.57) and below the
        # text IDs (1)
        scatterPlotItem = self.customAnnotDict[toolButton]['scatterPlotItem']
        scatterPlotItem.setZValue(0.6)

    # Transforms of the plot options that change the number of points or
    # the meaning of the x axis (`disableUnsupportedPlotTransforms`)
    UNSUPPORTED_PLOT_TRANSFORMS = (
        'fftCheck', 'derivativeCheck', 'phasemapCheck'
    )

    # Options of the plot options' "Downsample" menu that plot fewer points
    # than the data (`disableUnsupportedPlotDownsampling`); the first two
    # turn them on, the others only matter while "Downsample" is checked
    UNSUPPORTED_PLOT_DOWNSAMPLING = (
        'downsampleCheck', 'clipToViewCheck', 'downsampleSpin',
        'autoDownsampleCheck', 'peakRadio', 'meanRadio', 'subsampleRadio'
    )
    
    def invertBw(self, checked, update=True):
        # `ColorSchemeGuiElements.invertBw` inverts the text annotations of 
        # both images of guiWin while the viewer has only the left image
        textAnnot = self.textAnnot
        hasLeftTextAnnot = 0 in textAnnot
        if not hasLeftTextAnnot:
            del self.textAnnot
        else:
            textAnnot[1] = viewer_widgets._NoOpItem()
        try:
            super().invertBw(checked, update=update and self.isDataLoaded)
        finally:
            self.textAnnot = textAnnot
            textAnnot.pop(1, None)
        
        self.isBwInverted = checked
        background = graphLayoutBkgrColor if checked else darkBkgrColor
        self.measurementsPlot.setBackground(background)
        self.phaseBandPlot.setBackground(background)

    def annotateAllObjectTracks(self, settings=None):
        super().annotateAllObjectTracks(settings=settings)
        self.raiseOnDemandAx1Items()

    def addTimestamp(self, checked):
        super().addTimestamp(checked)
        self.raiseOnDemandAx1Items()

    def addScaleBar(self, checked):
        # No rescale: the bar is placed in the current view (the user's
        # zoom), and `ScaleBar.addToAxis` ignores its bounds
        super().addScaleBar(checked)
        self.raiseOnDemandAx1Items()
    
    def mousePressEvent(self, event):
        # Custom mouse bindings (e.g., "Next"/"Previous" on mouse side 
        # buttons) as in guiWin
        for name, button in self.mouseBindings.items():
            if event.button() != button:
                continue
            widget = self.widgetsWithShortcut.get(name)
            if widget is None:
                continue
            if hasattr(widget, 'click'):
                widget.click()
            else:
                widget.trigger()
            # Handled: a forwarded scene event must not propagate to the
            # window again (triggering the action twice)
            event.accept()
            return
        if not isinstance(event, QMouseEvent):
            # Scene events forwarded by `gui_mousePressEventImg1`
            event.ignore()
            return
        super().mousePressEvent(event)
    
    def keyPressEvent(self, event):
        if not self.data:
            super().keyPressEvent(event)
            return
        key = event.key()
        modifiers = event.modifiers()
        isNoModifier = modifiers == Qt.NoModifier
        if self.overlayButton.isChecked() and isNoModifier:
            isButtonClicked = self.checkOverlayToolbuttonClicked(event)
            if isButtonClicked:
                return
        
        if key == self.zoomOutKeyValue and isNoModifier:
            doubleKeyPress = False
            if self.countKeyPress == 0:
                self.countKeyPress = 1
                self.doubleKeyTimeElapsed = False
                QTimer.singleShot(400, self.resetCountKeyPress)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                doubleKeyPress = True
                self.countKeyPress = 0
                
            if self.measurementsPlot.underMouse():
                if not doubleKeyPress:
                    self.autoscaleMeasurementsPlot()
                else:
                    posData = self.posData()
                    max_frame_index = posData.SizeT
                    self.measurementsPlot.setXRange(0, max_frame_index)
                return
            
            if not self.img1.isUnderMouse():
                return
            # Over the image: press --> zoom to objects,
            # double press --> zoom out
            if not doubleKeyPress:
                self.zoomToCells(enforce=True)
            else:
                self.ax1.autoRange()
                
        elif key == Qt.Key_Space and isNoModifier:
            # Same as guiWin: double press --> toggle "Do not annotate"
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
        elif key == Qt.Key_Escape:
            if self.zoomRectButton.isChecked():
                self.zoomRectCancelled()
            elif self.rulerButton.isChecked() and self.tempSegmentON:
                self.tempSegmentON = False
                self.ax1_rulerPlotItem.setData([], [])
                self.ax1_rulerAnchorsItem.setData([], [])
            elif self.customAnnotButton is not None:
                self.setUncheckedAllCustomAnnotButtons()
            else:
                self.uncheckLeftClickButtons(None)
        elif key == Qt.Key_Z:
            if self.posData().SizeZ > 1:
                self.zSliceCheckbox.setChecked(
                    not self.zSliceCheckbox.isChecked()
                )
        elif key in viewer_navigation.ARROW_KEYS:
            # Also reached with modifiers or when a widget that uses the
            # arrow keys ignored them (`eventFilter` handles the others)
            self.navigateWithArrowKey(key)
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        if self._isLoadingData:
            # The load continues after its dialogs/worker thread return
            text = 'Wait for the data being loaded before closing the viewer.'
            self.logger.warning(text)
            self.statusBar().showMessage(text, 8000)
            event.ignore()
            return
        if self.measurementProcessingThread is not None:
            self.measurementProcessingWarning(
                'Measurement processing running',
                'Wait for measurement processing to finish before closing the viewer.'
            )
            event.ignore()
            return
        if self.isDataLoaded:
            # The prompt does not block the window: no load may replace the
            # data it saves (`loadingInProgress`) while it is open
            self._isClosingPrompt = True
            try:
                proceed = self.askSaveBeforeClosing()
            finally:
                self._isClosingPrompt = False
            if not proceed:
                event.ignore()
                return
            self.saveAnnotationOptions()
        if self.slideshowWin is not None:
            self.slideshowWin.close()
        if self.ccaTableWin is not None:
            self.ccaTableWin.close()
        if self.countObjsWindow is not None:
            self.countObjsWindow.close()
        volume_renderer = getattr(self, '_volume_renderer', None)
        if volume_renderer is not None:
            self._volume_renderer = None
            volume_renderer.force_close()
            # Free the window and the data it references (e.g., the
            # images, not copied), as `_launch3dViewer` does on closing
            volume_renderer.deleteLater()
        self.saveWindowGeometry(setting_suffix='_measurements_viewer')
        self.logger.info('Measurements viewer closed')
        super().closeEvent(event)
        self.sigClosed.emit(self)
        
        
def run(folderpath=None):
    """Launch the measurements viewer as a standalone application, optionally 
    loading `folderpath` (experiment, Position or Images folder)."""
    from cellacdc._run import _setup_app
    app, _ = _setup_app()
    win = MeasurementsViewerWin()
    if folderpath:
        # The folder is loaded directly: the startup configuration prompt
        # would open on top of the folder loading dialogs.
        win._startupConfigurationPrompted = True
    win.show()
    if folderpath:
        win.openFolder(folderpath=folderpath)
    app.exec_()


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else None)
