"""Standalone window to navigate frames, z-slices and cell IDs of Cell-ACDC 
data while plotting the time-course of `acdc_output` table columns (measurements)
of the selected cell, or two measurements against each other for all cells and
frames of the current position. Hover over a point for measurement details;
click a point to navigate to its frame and zoom to its cell.
In comparison mode, check two columns to select X and Y. X/Y markers replace
the color swatches; comparison and time-course selections are independent.
Column names are read-only, and the fixed-width selection indicator displays
the color or axis instead of a separate checkbox tick. Clicking anywhere on a
column (indicator or name) toggles its selection.
Measurement selectors use the equation editor's tree organization: numerical
and categorical inputs, channel-specific cell/background/custom measurements,
size measurements, region properties, and other numerical inputs. Plotting
lists only numerical and boolean columns. Search keeps matching leaves and
their ancestors visible; group headings cannot be selected for plotting or
processing.

The window loads data on its own (menu `File --> Load Folder...`, 
`Open Image File...` or `Open Recent`) and it shares only GUI elements with 
the main GUI (`gui.guiWin`) through the mixins in:

- `gui_navigation`: frames/z-slices navigation, zoom to object, recent paths 
  and LUT.
- `gui_annotations`: display of contours, segm. masks, IDs, cell cycle and 
  lineage info, mother-daughter lines and object tracks.
- `gui_viewer_tools`: the other 'Viewer' mode tools (zoom, ruler, scale bar, 
  timestamp, colormaps, search ID, hover info, shortcuts, slideshow, 
  3D viewer, objects properties, cell cycle table and objects count).
- `gui_overlay`: overlay of other channels and of other segmentation files.

Run it with `python -m cellacdc.measurements_viewer [folder_path]`.

At startup, load a saved configuration or skip to start normally. The File
menu can load, save or save a copy of a configuration; closing the window
also offers to save. Configurations are JSON files in the settings folder
and retain Images folders, image/overlay channels, segmentation, and both
plot modes' selections (including X/Y order and measurement normalization).

Use ``Process measurements...`` to apply a recipe to selected measurements of the
current position. Smooth measurement uses a centered moving average independently
for each cell; Dummy leaves values unchanged. Accepting prompts for an output
name and adds ``<measurement>_<name>`` columns without changing the originals.
Results stay in memory across position switches; Reload table discards them.
``Combine measurements...`` opens the combined-measurement equation editor.
Equations are evaluated on the current position's raw table, including
categorical inputs, and add columns without overwriting existing measurements.
Combined columns stay in memory and are not saved to the global equation
configuration. Numerical and boolean results are available for plotting;
string/category results are retained in the exported table.
``Export measurements...`` saves the table, including processed columns, to a
separate CSV. Processing is applied to raw measurements before plot normalization.

The ``Time points`` toolbar has ``Add time point identifier...`` to define a
named time point (e.g., "Bud emergence") with a color, a keyboard shortcut
(defaults Q, W, E, R, ...) and an optional per-cell predictor (an equation of
the table columns or ``predictor(df)`` from a Python file, see
``timepoint_predictors``). Each identifier gets its own toolbar action: hover
over the time-course plot and press its shortcut to add a time point for the
selected cell at the hovered frame (clicking the action adds it at the
current frame). Time point lines can be dragged like the current frame line;
right-click a line to remove it, or right-click an identifier action to edit,
re-evaluate its predictor, clear or remove it. Automatic predictions require
the identifier's ``autoApply`` flag and run once per cell, identifier and
position path during the viewer session. Failed predictions can be retried.
The toolbar's ``Re-evaluate predictors on all cells`` and ``... on displayed
cell`` re-run all predictors; cells with manual edits (added, moved or removed
time points) trigger a warning to overwrite or skip them. ``Per-cell
constants...`` defines constants evaluated once per cell (presets such as
"Maximum value" or "Was at beginning", or custom equations) that predictors
can use like columns; their values are stored with the time points of each
position. Identifier and constant definitions are stored in configurations
and exported time points are added as ``timepoint_<name>`` boolean columns.
"""
import os
import sys
import copy
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd

import skimage.measure

from qtpy.QtCore import (
    Qt, Signal, QTimer, QPoint, QEvent, QRect, QSaveFile, QIODevice,
    QThread, QMutex, QWaitCondition
)
from qtpy.QtGui import (
    QFont, QIcon, QGuiApplication, QPixmap, QPainter, QCursor, QKeySequence,
    QColor
)
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QLabel, QGridLayout, QVBoxLayout, QHBoxLayout,
    QGroupBox, QSplitter, QListView, QLineEdit,
    QCheckBox, QComboBox, QAbstractItemView, QFileDialog, QAction,
    QApplication, QToolButton, QPushButton, QActionGroup, QAbstractSlider,
    QWidgetAction, QMenu, QColorDialog, QStyleOptionViewItem, QTreeView,
    QAbstractButton
)

import pyqtgraph as pg

from . import widgets, core, load, myutils, colors, transformation
from . import (
    graphLayoutBkgrColor, darkBkgrColor, settings_csv_path, settings_folderpath
)
from . import cca_df_colnames, base_cca_dict
from . import regionprops as acdc_regionprops
from . import gui_navigation, gui_annotations, gui_viewer_tools, gui_overlay
from .config import STANDARD_MOUSE_BUTTONS
from . import gui_core
from . import apps
from . import html_utils
from . import printl
from . import workers
from . import timepoint_predictors
from .timepoint_predictors import TimepointIdentifier

RESCALE_INTENS_HOW_TEXTS = (
    'Rescale each 2D image',
    'Rescale across z-stack',
    'Rescale across time frames',
    'Do no rescale, display raw image'
)

pg.setConfigOption('imageAxisOrder', 'row-major')

_font = QFont()
_font.setPixelSize(11)

DEFAULT_MEASUREMENTS_COLUMNS = ('cell_area_pxl', 'cell_vol_vox', 'cell_area_um2')
INDEX_COLUMNS = ('frame_i', 'Cell_ID', 'time_seconds')
CCA_DF_INT_COLS = [
    col for col in cca_df_colnames if type(base_cca_dict[col]) == int
]
MEASUREMENTS_CONFIGURATIONS_PATH = os.path.join(
    settings_folderpath, 'measurements_viewer_configurations'
)

class _TimepointLine(pg.InfiniteLine):
    """Draggable line of a time point of the selected cell."""
    sigRemoveRequested = Signal(object)

    def __init__(self, identifier, ID, frame_i):
        color = identifier.color
        super().__init__(
            pos=frame_i+1, angle=90, movable=True,
            pen=pg.mkPen(color=color, width=2, style=Qt.DashLine),
            hoverPen=pg.mkPen(color=color, width=4),
            labelOpts={
                'position': 0.95, 'color': color, 'movable': True,
                'fill': (0, 0, 0, 120)
            }
        )
        self.uid = identifier.uid
        self.ID = ID
        self.frame_i = frame_i
        self.setToolTip(
            f'{identifier.name} (Cell ID {ID}, frame n. {frame_i+1})\n\n'
            'Drag to move. Right-click to remove.'
        )
        self.setZValue(9)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton and not self.moving:
            ev.accept()
            self.sigRemoveRequested.emit(self)
            return
        super().mouseClickEvent(ev)

class _MeasurementColumnDelegate(widgets._ButtonSearchHighlightDelegate):
    indicatorWidth = 28

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if not index.flags() & Qt.ItemIsUserCheckable:
            return
        option.features &= ~(
            QStyleOptionViewItem.HasCheckIndicator
            | QStyleOptionViewItem.HasDecoration
        )
        option.icon = QIcon()
        option.rect.adjust(self.indicatorWidth, 0, 0, 0)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if not index.flags() & Qt.ItemIsUserCheckable:
            return
        rect = QRect(
            option.rect.left()+2, option.rect.center().y()-8, 24, 16
        )
        icon = index.data(Qt.DecorationRole)
        painter.save()
        if icon is not None and not icon.isNull():
            icon.paint(painter, rect, Qt.AlignCenter)
        else:
            painter.setPen(option.palette.mid().color())
            painter.drawRect(rect.adjusted(4, 2, -4, -2))
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if not index.flags() & Qt.ItemIsUserCheckable:
            return False
        if event.type() in (
                QEvent.MouseButtonPress, QEvent.MouseButtonDblClick,
                QEvent.MouseButtonRelease):
            # The whole row (indicator and name) toggles the selection
            if event.button() != Qt.LeftButton:
                return False
            if not option.rect.contains(event.pos()):
                return False
            if event.type() != QEvent.MouseButtonRelease:
                return True
        elif event.type() == QEvent.KeyPress:
            if event.key() not in (Qt.Key_Space, Qt.Key_Select):
                return False
        else:
            return False
        state = (
            Qt.Unchecked
            if Qt.CheckState(index.data(Qt.CheckStateRole)) == Qt.Checked
            else Qt.Checked
        )
        return model.setData(index, state, Qt.CheckStateRole)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setWidth(size.width()+self.indicatorWidth)
        size.setHeight(max(size.height(), 20))
        return size

def get_images_paths(folderpath):
    """Get the list of `Images` folders from an experiment, Position or 
    Images folder path."""
    folderpath = os.path.normpath(folderpath)
    if os.path.isfile(folderpath):
        folderpath = os.path.dirname(folderpath)
    
    if os.path.basename(folderpath) == 'Images':
        return [folderpath]
    
    images_path = os.path.join(folderpath, 'Images')
    if os.path.isdir(images_path):
        return [images_path]
    
    pos_foldernames = myutils.get_pos_foldernames(folderpath)
    images_paths = [
        os.path.join(folderpath, pos, 'Images') for pos in pos_foldernames
    ]
    return [path for path in images_paths if os.path.isdir(path)]

def get_channels_and_segm_endnames(images_path):
    """Return the channel names and the segmentation files end names 
    (e.g., 'segm' or 'segm_<name>') available in `images_path`."""
    basename, chNames = myutils.getBasenameAndChNames(images_path)
    segm_files = load.get_segm_files(images_path)
    segm_endnames = load.get_endnames(basename, segm_files)
    segm_endnames = [
        endname.lstrip('_') for endname in segm_endnames 
        if not endname.endswith('_raw') and 'raw_postproc' not in endname
    ] or [endname.lstrip('_') for endname in segm_endnames]
    return chNames, sorted(set(segm_endnames))

def get_channel_from_filepath(file_path):
    """Return the channel name of an image file in an Images folder or None 
    if it cannot be determined."""
    images_path = os.path.dirname(file_path)
    filename, _ = os.path.splitext(os.path.basename(file_path))
    try:
        basename, chNames = myutils.getBasenameAndChNames(images_path)
    except Exception as err:
        return
    
    matching = [ch for ch in chNames if filename.endswith(ch)]
    if not matching:
        return
    return max(matching, key=len)

def load_position(images_path, channel, segm_endname='segm'):
    """Load image, segmentation, acdc_output table and metadata of one 
    Position. 
    
    Image and segmentation data always have the time axis first.
    """
    img_path = load.get_filename_from_channel(images_path, channel)
    if not img_path:
        raise FileNotFoundError(
            f'Could not find a file for channel "{channel}" '
            f'in "{images_path}".'
        )
    posData = load.loadData(img_path, channel)
    posData.getBasenameAndChNames()
    posData.buildPaths()
    posData.loadImgData()
    posData.loadOtherFiles(
        load_segm_data=True,
        load_acdc_df=True,
        load_metadata=True,
        loadSegmInfo=True,
        end_filename_segm=segm_endname
    )
    if not posData.segmFound:
        raise FileNotFoundError(
            f'Segmentation file ending with "{segm_endname}" not found '
            f'in "{images_path}".'
        )
    
    if posData.SizeZ > 1 and posData.img_data.ndim < 3:
        posData.SizeZ = 1
    
    if posData.SizeT == 1:
        posData.img_data = posData.img_data[np.newaxis]
        posData.segm_data = posData.segm_data[np.newaxis]
    
    posData.SizeT = min(len(posData.img_data), len(posData.segm_data))
    return posData

def load_positions(images_paths, channel, segm_endname='segm', logger=print):
    data = []
    for images_path in images_paths:
        try:
            data.append(load_position(images_path, channel, segm_endname))
        except Exception as err:
            logger(f'[WARNING]: Skipping "{images_path}". {err}')
    return data

class _NoOpItem:
    """Placeholder for main GUI items that do not exist in the viewer 
    (e.g., the right image LUT). Every method call is a no-op."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class _NoAutoPilotProfile:
    """The viewer does not record auto-pilot profiles like the main GUI"""
    def storeSelectedChannel(self, *args, **kwargs):
        pass
    
    def storeClickMessageBox(self, *args, **kwargs):
        pass
    
    def storeLoadedFluoChannels(self, *args, **kwargs):
        pass

class MeasurementsViewerWin(
        QMainWindow,
        gui_navigation.FramesNavigationGuiElements,
        gui_navigation.ZsliceNavigationGuiElements,
        gui_navigation.ZoomToObjGuiElements,
        gui_navigation.RecentPathsGuiElements,
        gui_navigation.LutGuiElements,
        gui_annotations.AnnotationsGuiElements,
        gui_viewer_tools.ZoomGuiElements,
        gui_viewer_tools.RulerGuiElements,
        gui_viewer_tools.ScaleBarTimestampGuiElements,
        gui_viewer_tools.ColorSchemeGuiElements,
        gui_viewer_tools.LabelsColormapGuiElements,
        gui_viewer_tools.ImagePropertiesGuiElements,
        gui_viewer_tools.CcaTableGuiElements,
        gui_viewer_tools.CountObjectsGuiElements,
        gui_viewer_tools.WorkersGuiElements,
        gui_viewer_tools.SearchIDGuiElements,
        gui_viewer_tools.HoverInfoGuiElements,
        gui_viewer_tools.ShortcutsGuiElements,
        gui_viewer_tools.SlideshowGuiElements,
        gui_viewer_tools.Volume3DViewerGuiElements,
        gui_viewer_tools.RegionPropsGuiElements,
        gui_viewer_tools.BottomLayoutGuiElements,
        gui_viewer_tools.DefaultSettingsGuiElements,
        gui_overlay.OverlayChannelsGuiElements,
        gui_overlay.OverlayLabelsGuiElements,
        gui_overlay.ChannelImageHelpersGuiElements,
        gui_core.GUICoreElements,
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
    # The bottom layout is different from guiWin's --> separate setting
    bottomLayoutHeightSettingKey = 'measurements_viewer_bottomLayoutHeight'

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
        self._startupConfigurationPrompted = False
        self.measurementProcessingThread = None
        self.measurementProcessingWorker = None
        self.rescaleIntensChannelHowMapper = {}
        self.timepointIdentifiers = []
        self.timepointActions = {}
        self.timepointAutoActions = {}
        self.timepointLines = []
        self.timepointAutoPredictions = defaultdict(set)
        # Per-cell constants [{'name': str, 'equation': str}]
        self.cellConstants = []
        self.logger = logging.getLogger('cellacdc-measurements-viewer')
        self.loadSettings()
        self.initAnnotationsAttributes()
        self.initViewerToolsAttributes()
        if sys.platform.startswith('win'):
            self.openFolderText = 'Show in Explorer...'
        else:
            self.openFolderText = 'Reveal in Finder...'
        
        self.setWindowTitle('Cell-ACDC - Measurements viewer')
        
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

    def showEvent(self, event):
        super().showEvent(event)
        if not self._startupConfigurationPrompted:
            self._startupConfigurationPrompted = True
            QTimer.singleShot(0, self.loadConfiguration)

    def selectConfigurationFilepath(self):
        from .apps import QTreeDialog

        paths = {}
        if os.path.isdir(MEASUREMENTS_CONFIGURATIONS_PATH):
            for filename in sorted(os.listdir(MEASUREMENTS_CONFIGURATIONS_PATH)):
                filepath = os.path.join(MEASUREMENTS_CONFIGURATIONS_PATH, filename)
                if filename.lower().endswith('.json') and os.path.isfile(filepath):
                    paths[filename] = filepath

        def browse():
            return QFileDialog.getOpenFileName(
                self, 'Load measurements viewer configuration',
                MEASUREMENTS_CONFIGURATIONS_PATH, 'JSON (*.json)'
            )[0]

        browseButton =  widgets.browseFileButton('Browse configuration...', self)
        skipButton = widgets.SkipPushButton('Skip/New configuration', self)
        dialog = QTreeDialog(
            [
                (name, datetime.fromtimestamp(os.path.getmtime(path)).strftime(
                    '%Y/%m/%d %H:%M'
                ))
                for name, path in paths.items()
            ],
            headerLabels=['Name', 'Date Modified'],
            title='Load measurements viewer configuration',
            infoText='Select a configuration to load, or skip loading:',
            parent=self,
            path_to_browse=MEASUREMENTS_CONFIGURATIONS_PATH if paths else None,
            additional_buttons=(browseButton,skipButton),
        )
        dialog.treeWidget.setCurrentItem(dialog.treeWidget.topLevelItem(0))
        browsed = []

        def browseSelected():
            filepath = browse()
            if filepath:
                browsed.append(filepath)
                dialog.close()

        browseButton.clicked.connect(browseSelected)
        skipButton.clicked.connect(lambda: setattr(dialog, 'cancel', True) or dialog.close())
        dialog.exec_()
        if browsed:
            return browsed[0]
        if dialog.cancel:
            return None
        return paths[dialog.selectedText]

    def configuration(self):
        posData = self.posData()
        segm_filename = os.path.splitext(
            os.path.basename(posData.segm_npz_path)
        )[0]
        segm_endname = segm_filename[len(posData.basename):].lstrip('_')
        overlay_channels = (
            sorted(self.checkedOverlayChannels)
            if self.overlayButton.isChecked() else []
        )
        visible_channels = [
            channel for channel, button in self.allOverlayToolbuttons.items()
            if button.isChecked()
            and (channel == self.user_ch_name or channel in overlay_channels)
        ]
        return {
            'version': 1,
            'images_paths': [
                os.path.abspath(pos.images_path) for pos in self.data
            ],
            'channel': self.user_ch_name,
            'segm_endname': segm_endname,
            'overlay_channels': overlay_channels,
            'visible_channels': visible_channels,
            'measurements_mode': self.measurementsModeCombobox.currentIndex(),
            'time_course_columns': (
                self.checkedColumns() if self._measurementsMode == 0
                else sorted(self._timeCourseColumns)
            ),
            'measurements_axes': self.measurementsAxes.copy(),
            'normalize_measurements': self.normalizeCheckbox.isChecked(),
            'timepoint_identifiers': [
                identifier.to_dict() for identifier in self.timepointIdentifiers
            ],
            'cell_constants': [dict(constant) for constant in self.cellConstants],
        }

    @staticmethod
    def validateConfiguration(configuration):
        if not isinstance(configuration, dict) or configuration.get('version') != 1:
            raise ValueError('Unsupported measurements viewer configuration.')
        for key in (
                'images_paths', 'overlay_channels', 'visible_channels',
                'time_course_columns'):
            value = configuration.get(key)
            if not isinstance(value, list) or any(
                    not isinstance(item, str) or not item for item in value):
                raise ValueError(f'Invalid configuration field: {key}.')
        if not configuration['images_paths']:
            raise ValueError('The configuration contains no Images folders.')
        for key in ('channel', 'segm_endname'):
            if not isinstance(configuration.get(key), str) or not configuration[key]:
                raise ValueError(f'Invalid configuration field: {key}.')
        axes = configuration.get('measurements_axes')
        if (not isinstance(axes, dict) or set(axes) != {'X', 'Y'}
                or any(not isinstance(value, str) for value in axes.values())
                or (axes['X'] and axes['X'] == axes['Y'])):
            raise ValueError('Invalid comparison axes in configuration.')
        if (type(configuration.get('measurements_mode')) is not int
                or configuration['measurements_mode'] not in (0, 1)
                or type(configuration.get('normalize_measurements')) is not bool):
            raise ValueError('Invalid plotting options in configuration.')
        available = {
            configuration['channel'], *configuration['overlay_channels']
        }
        if (not configuration['visible_channels']
                or not set(configuration['visible_channels']) <= available):
            raise ValueError('Invalid visible image channels in configuration.')
        # Optional for configurations saved before time point identifiers
        identifiers = configuration.get('timepoint_identifiers', [])
        if not isinstance(identifiers, list):
            raise ValueError('Invalid time point identifiers in configuration.')
        for identifier in identifiers:
            timepoint_predictors.validate_identifier_dict(identifier)
        names = [identifier['name'] for identifier in identifiers]
        if len(set(names)) != len(names):
            raise ValueError('Duplicated time point identifier names.')
        timepoint_predictors.validate_constants(
            configuration.get('cell_constants', [])
        )

    def configurationError(self, operation, error):
        self.logger.error('Could not %s configuration: %s', operation, error)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Configuration failed',
            f'Could not {operation} configuration:\n\n{error}'
        )

    def saveConfiguration(self, checked=False, save_as=False):
        if not self.isDataLoaded:
            self.configurationError('save', 'Load image data before saving.')
            return False
        filepath = self.configurationFilepath
        if save_as or filepath is None:
            try:
                os.makedirs(MEASUREMENTS_CONFIGURATIONS_PATH, exist_ok=True)
            except OSError as err:
                self.configurationError('save', err)
                return False
            current_files = sorted(os.listdir(MEASUREMENTS_CONFIGURATIONS_PATH))
            helpText = html_utils.paragraph(
                '''Select a filename for the measurements viewer configuration.
                Which allows you to later reload it, preserving displayed measurements.'''
                )
            date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            msg = apps.filenameDialog(
                ext='json', title='Save measurements viewer configuration as',
                parent=self, helpText=helpText,existingNames=current_files,
                allowEmpty=False, addDoNotSaveButton=True, defaultEntry=date
            )
            msg.exec_()
            if msg.cancel:
                return False
            if msg.doNotSave:
                return True
            filename = msg.filename
            if not filename:
                return False
            filepath = os.path.join(MEASUREMENTS_CONFIGURATIONS_PATH, filename)
            if not filepath.lower().endswith('.json'):
                filepath += '.json'
                if os.path.exists(filepath):
                    msg = widgets.myMessageBox(wrapText=False)
                    no, yes = msg.question(
                        self, 'Overwrite configuration?',
                        f'The file "{filepath}" already exists. Overwrite it?',
                        buttonsTexts=('Cancel', 'Overwrite')
                    )
                    if msg.clickedButton != yes:
                        return False
        try:
            configuration = self.configuration()
            self.validateConfiguration(configuration)
            payload = json.dumps(configuration, indent=2).encode('utf-8')
            output = QSaveFile(filepath)
            if not output.open(QIODevice.WriteOnly):
                raise OSError(output.errorString())
            if output.write(payload) != len(payload):
                output.cancelWriting()
                raise OSError(output.errorString())
            if not output.commit():
                raise OSError(output.errorString())
        except (OSError, ValueError, TypeError, AttributeError) as err:
            self.configurationError('save', err)
            return False
        self.configurationFilepath = filepath
        self.statusBar().showMessage(f'Configuration saved to "{filepath}".', 5000)
        return True

    def saveConfigurationAs(self, checked=False):
        return self.saveConfiguration(save_as=True)

    def loadConfiguration(self, checked=False, filepath=None):
        if filepath is None:
            filepath = self.selectConfigurationFilepath()
        if not filepath:
            return False
        try:
            with open(filepath, encoding='utf-8') as file:
                configuration = json.load(file)
            self.validateConfiguration(configuration)
            for path in configuration['images_paths']:
                if not os.path.isdir(path):
                    raise FileNotFoundError(f'Images folder not found: "{path}".')
                channels, segmentations = get_channels_and_segm_endnames(path)
                required = {
                    configuration['channel'], *configuration['overlay_channels']
                }
                missing = required.difference(channels)
                if missing:
                    raise ValueError(f'Missing channels in "{path}": {sorted(missing)}.')
                if configuration['segm_endname'] not in segmentations:
                    raise ValueError(f'Segmentation not found in "{path}".')
            self.setCursor(Qt.WaitCursor)
            try:
                data = [
                    load_position(
                        path, configuration['channel'],
                        configuration['segm_endname']
                    )
                    for path in configuration['images_paths']
                ]
            finally:
                self.unsetCursor()
        except (OSError, ValueError, TypeError, KeyError) as err:
            self.configurationError('load', err)
            return False

        self.isSegm3D = None
        self.setCellConstants(configuration.get('cell_constants', []))
        self.setTimepointIdentifiers([
            TimepointIdentifier.from_dict(identifier)
            for identifier in configuration.get('timepoint_identifiers', [])
        ])
        self.setData(data)
        self.configurationFilepath = None
        overlays = configuration['overlay_channels']
        if overlays:
            if not self.loadOverlayData(overlays):
                self.configurationError('load', 'Could not load overlay channels.')
                return False
            self.setCheckedOverlayContextMenusActions(overlays)
            self.overlayButton.setChecked(True)
        for channel, button in self.allOverlayToolbuttons.items():
            button.setChecked(channel in configuration['visible_channels'])
        self.setOverlayItemsOpacities()

        columns = {item.text() for item in self.columnsModel.measurementItems()}
        selected = set(configuration['time_course_columns'])
        axes = configuration['measurements_axes']
        missing = (selected | set(axes.values())).difference(columns, {''})
        self.measurementsModeCombobox.blockSignals(True)
        self.measurementsModeCombobox.setCurrentIndex(configuration['measurements_mode'])
        self.measurementsModeCombobox.blockSignals(False)
        # Do not replace the restored time-course selection with the old checks.
        self._measurementsMode = 1
        self._timeCourseColumns = selected & columns
        self.measurementsAxes = {
            axis: value if value in columns else '' for axis, value in axes.items()
        }
        self.measurementsModeChanged()
        self.normalizeCheckbox.setChecked(configuration['normalize_measurements'])
        self.configurationFilepath = filepath
        if missing:
            self.configurationError(
                'restore all plot selections',
                f'Columns no longer available: {", ".join(sorted(missing))}.'
            )
        return True
    
    def postInitLoadSetSettings(self):
        followID = self._get_setting_value('followID', default=True, cast=bool)
        self.followIDcheckbox.setChecked(followID)
        
        restrictFrames = self._get_setting_value('restrictFrames', default=True, cast=bool)
        self.restrictFramesCheckbox.setChecked(restrictFrames)
        
        rescalePlot = self._get_setting_value('rescalePlot', default=True, cast=bool)
        self.rescalePlotCheckbox.setChecked(rescalePlot)


    def annotatedAxesIndices(self):
        # The viewer has only the left image (no right image like in guiWin)
        return (0,)
    
    def setDoNotAnnotate(self, checked):
        # Same as `gui.guiWin.setDoNotAnnotate` but only for the left image
        self.annotDoNotAnntoateCheckbox(0).setChecked(checked)
    
    @property
    def frame_i(self):
        # The current frame is stored in the current position data as in 
        # guiWin (e.g., `SearchIDGuiElements.findID` sets `posData.frame_i`)
        if not self.data:
            return 0
        return self.data[self.pos_i].frame_i
    
    @frame_i.setter
    def frame_i(self, value):
        if not self.data:
            return
        self.data[self.pos_i].frame_i = value
    
    def initViewerToolsAttributes(self):
        """Initialize the attributes required by the mixins in 
        `gui_viewer_tools` and `gui_overlay`"""
        self._appName = 'Cell-ACDC'
        try:
            self._version = myutils.read_version()
        except Exception as err:
            self._version = None
        self.app = QApplication.instance()
        self.mainWin = None
        self.isSnapshot = False
        self.isTwoImageLayout = False
        
        self.exportMaskImage = None
        self.exportToImageWindow = None
        self.timestampStartTimedelta = timedelta(seconds=0)
        
        self.scaleBarDialog = None
        self.timestampDialog = None
        self.slideshowWin = None
        self.ccaTableWin = None
        self.countObjsWindow = None
        self.progressWin = None
        
        self.tempSegmentON = False
        self.isMouseDragImg1 = False
        self.invertBwAlreadyCalledOnce = False
        self.currentPropsID = -1
        self.xHoverImg, self.yHoverImg = None, None
        self.countKeyPress = 0
        self.doubleKeyTimeElapsed = False
        self.isKeyDoublePress = False
        self.doubleSpaceBarState = False
        
        self.mouseBindings = {}
        self.defaultMouseShortcuts = {
            'Next': Qt.MouseButton.XButton2, 
            'Previous': Qt.MouseButton.XButton1
        }
        self.zoomOutKeyValue = Qt.Key_H
        self.delObjAction = None
        self.delObjToolAction = QAction(self)
        self.delObjToolAction.setCheckable(True)
        self.delObjToolAction.setVisible(False)
        self.widgetsWithShortcut = {}
        self.leftClickButtons = []
        
        self.fluoDataChNameActions = []
        self.retainSizeLutItems = False
        self.AutoPilotProfile = _NoAutoPilotProfile()
        self.checkedOverlayChannels = set()
        self.overlayLabelsItems = {}
        self.drawModeOverlayLabelsChannels = {}
        self.existingSegmEndNames = []
        self.ch_names = []
        
        # Main GUI widgets that do not exist in the viewer
        self.equalizeHistPushButton = QPushButton(self)
        self.equalizeHistPushButton.hide()
        self.imgGradRight = _NoOpItem()
    
    def initAnnotationsAttributes(self):
        """Initialize the attributes required by 
        `gui_annotations.AnnotationsGuiElements`"""
        self.annotOptionsCheckboxes = defaultdict(dict)
        self.annotOptionsToRestore = defaultdict(dict)
        self.textAnnot = {}
        self.tracksPlotItems = {}
        self.overlayLayersItems = {}
        self.highlightedID = 0
        self.isExportingVideo = False
        self.lineage_tree = None
        self.cca_df_colnames = cca_df_colnames
        self.cca_df_int_cols = CCA_DF_INT_COLS
        self._acdcRegionProps = acdc_regionprops.acdcRegionprops
        
        # Annotations in the viewer are always displayed as in 'Viewer' mode
        self.modeComboBox = QComboBox(self)
        self.modeComboBox.addItem('Viewer')
        self.modeComboBox.hide()
        
        savedFontSize = '12'
        if 'fontSize' in self.df_settings.index:
            savedFontSize = str(self.df_settings.at['fontSize', 'value'])
        if savedFontSize.find('pt') != -1:
            savedFontSize = savedFontSize[:-2]
        try:
            self.fontSize = int(float(savedFontSize))
        except Exception as err:
            self.fontSize = 12
    
    def loadSettings(self):
        self.settings_csv_path = settings_csv_path
        if os.path.exists(settings_csv_path):
            self.df_settings = pd.read_csv(
                settings_csv_path, index_col='setting'
            )
        else:
            self.df_settings = pd.DataFrame(
                {'setting': [], 'value': []}, dtype=object
            ).set_index('setting')
        
        if 'is_bw_inverted' not in self.df_settings.index:
            self.df_settings.at['is_bw_inverted', 'value'] = 'No'
        
        try:
            self.defaultRescaleIntensHow = (
                self.df_settings.at['default_rescale_intens_how', 'value']
            )
        except Exception as err:
            self.defaultRescaleIntensHow = RESCALE_INTENS_HOW_TEXTS[0]
    
    def gui_createActions(self):
        self.openFolderAction = QAction(
            QIcon(":folder-open.svg"), "&Load Folder...", self
        )
        self.openFolderAction.setShortcut('Ctrl+O')
        self.openFolderAction.setToolTip(
            'Load an experiment, Position or Images folder'
        )
        self.openFileAction = QAction(
            QIcon(":image.svg"), "&Open Image File...", self
        )
        self.openFileAction.setToolTip(
            'Open an image file from the Images folder of a Position'
        )
        self.showInExplorerAction = QAction(
            QIcon(":drawer.svg"), f"&{self.openFolderText}", self
        )
        self.reloadAction = QAction(
            QIcon(":reload.svg"), "Reload acdc_output Table", self
        )
        self.reloadAction.setShortcut('F5')
        self.reloadAction.setToolTip(
            'Reload the acdc_output table from the file '
            '(e.g., after saving in the main GUI)'
        )
        self.exitAction = QAction("&Exit", self)
        self.loadConfigurationAction = QAction('Load configuration...', self)
        self.saveConfigurationAction = QAction('Save configuration', self)
        self.saveConfigurationAction.setShortcut('Ctrl+S')
        self.saveConfigurationAsAction = QAction('Save configuration as...', self)
        self.saveConfigurationAsAction.setShortcut('Ctrl+Shift+S')
        self.addTimepointIdentifierAction = QAction(
            QIcon(":add.svg"), 'Add time point identifier...', self
        )
        self.addTimepointIdentifierAction.setToolTip(
            'Add time point identifier\n\n'
            'Define a named time point (e.g., "Bud emergence") with a color, '
            'a keyboard shortcut and an optional per-cell predictor.\n'
            'Hover over the measurements plot and press the shortcut to add '
            'a time point at the hovered frame.'
        )
        self.reevaluateAllTimepointsAction = QAction(
            QIcon(":reload.svg"), 'Re-evaluate predictors on all cells', self
        )
        self.reevaluateAllTimepointsAction.setToolTip(
            'Re-evaluate predictors on all cells\n\n'
            'Run the predictors of all time point identifiers on every cell '
            'of the current position, replacing their time points.\n'
            'You will be warned if manual edits would be overwritten.'
        )
        self.reevaluateCellTimepointsAction = QAction(
            QIcon(":magic_wand.svg"), 
            'Re-evaluate predictors on displayed cell', self
        )
        self.reevaluateCellTimepointsAction.setToolTip(
            'Re-evaluate predictors on displayed cell\n\n'
            'Run the predictors of all time point identifiers only on the '
            'displayed cell, replacing its time points.\n'
            'You will be warned if manual edits would be overwritten.'
        )
        self.editCellConstantsAction = QAction(
            QIcon(":calc.svg"), 'Per-cell constants...', self
        )
        self.editCellConstantsAction.setToolTip(
            'Per-cell constants\n\n'
            'Define constants computed once per cell (e.g., "Maximum value" '
            'or "Was at beginning")\n'
            'that can be used like columns in the time point predictors.'
        )
        
        # Annotations actions (same as in the main GUI)
        self.scalingLabelsAction = widgets.CheckableAction(
            'Fix text annotations to screen size'
        )
        self.scalingLabelsAction.setToolTip(
            'When the text annotations are fixed to screen size, they do not '
            'scale relative to the objects when zooming in/out.'
        )
        self.scalingLabelsAction.setChecked(
            self._get_setting_value('scalingLabels', False, cast=bool)
        )
        self.highLowResAction = widgets.CheckableAction(
            'High resolution text annotations'
        )
        self.highLowResAction.setToolTip(
            'Resolution of the text annotations. High resolution results '
            'in slower update of the annotations.\n'
            'Not recommended with a number of segmented objects > 500.'
        )
        self.highLowResAction.setChecked(
            self._get_setting_value('highLowRes', True, cast=bool)
        )
        self.editTextIDsColorAction = QAction(
            'Edit text annotations color...', self
        )
        self.annotObjTrackSettingsAction = QAction(
            'Object tracks settings...', self
        )
        self.resetAnnotOptionsAction = QAction(
            'Reset default annotation options', self
        )
        self.annotationsActions = (
            self.editTextIDsColorAction, self.annotObjTrackSettingsAction,
            self.resetAnnotOptionsAction
        )
        self.gui_createViewerToolsActions()
    
    def gui_createViewerToolsActions(self):
        """Create the actions of the 'Viewer' mode tools as in `gui.guiWin`"""
        if 'colorScheme' in self.df_settings.index:
            self._colorScheme = self.df_settings.at['colorScheme', 'value']
        else:
            self._colorScheme = 'light'
        self.toggleColorSchemeAction = QAction(
            'Switch to light theme', self
        )
        self.gui_updateSwitchColorSchemeActionText()
        
        self.editShortcutsAction = QAction(
            'Customize keyboard shortcuts...', self
        )
        self.editShortcutsAction.setShortcut('Ctrl+K')
        
        self.editOverlayColorAction = QAction('Overlay color...', self)
        self.editOverlayColorAction.setDisabled(True)
        
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
        checked = (
            self._get_setting_value('is_bw_inverted', 'No', cast=str) == 'Yes'
            or self.isBwInverted
        )
        self.invertBwAction.setChecked(checked)
        
        self.shuffleCmapAction = QAction('Randomly shuffle colormap', self)
        self.shuffleCmapAction.setShortcut('Shift+S')
        
        self.greedyShuffleCmapAction = QAction(
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
        self.normalizeRescale0to1Action = QAction(
            'Rescale to [0, 1]', self)
        self.normalizeByMaxAction = QAction(
            'Normalize by max value', self)
        self.normalizeQActionGroup = QActionGroup(self)
        for action in (
                self.normalizeRawAction, self.normalizeToFloatAction,
                self.normalizeRescale0to1Action, self.normalizeByMaxAction
            ):
            action.setCheckable(True)
            self.normalizeQActionGroup.addAction(action)
        
        self.zoomToObjsAction = QAction(
            'Zoom to objects  (Shortcut: H key)', self
        )
        self.zoomOutAction = QAction(
            'Zoom out  (Shortcut: double press H key)', self
        )
        self.enableAutoZoomToCellsAction = QAction(
            'Automatic zoom to all cells when pressing "Next/Previous"', self
        )
        self.enableAutoZoomToCellsAction.setCheckable(True)
        
        self.imgPropertiesAction = QAction('Properties...', self)
        self.imgPropertiesAction.setDisabled(True)
        
        self.launch3dViewerAction = QAction(
            'Launch 3D viewer for the currently visualised volume', self
        )
        self.launch3dViewerAction.setIcon(QIcon(":3d.svg"))
        self.launch3dViewerAction.setCheckable(True)
        self.launch3dViewerAction.setDisabled(True)
        
        self.searchAction = QAction(self)
        self.searchAction.setIcon(QIcon(":find.svg"))
        self.searchAction.setToolTip('Search for ID or functionality')
        
        self.focusOnSearchAction = QAction(
            "Search for ID or functionality", self
        )
        self.focusOnSearchAction.setShortcut('Ctrl+F')
        
        # Frames (or z-slices if "z" is active) navigation as in guiWin
        self.nextAction = QAction('Next', self)
        self.widgetsWithShortcut['Next'] = self.nextAction
        self.prevAction = QAction('Previous', self)
        self.widgetsWithShortcut['Previous'] = self.prevAction
        self.addAction(self.nextAction)
        self.addAction(self.prevAction)
        
        self.MaxProjToggleAction = QAction(self)
        self.MaxProjToggleAction.triggered.connect(
            self.maxProjToggleActionTriggered
        )
        self.MaxProjToggleAction.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self.addAction(self.MaxProjToggleAction)
        self.MaxProjToggleAction.setShortcut('Alt+X')
        self.widgetsWithShortcut['Toggle max z projection'] = (
            self.MaxProjToggleAction
        )
        
        self.viewerToolsActions = (
            self.viewCcaTableAction, self.imgPropertiesAction, 
            self.addScaleBarAction, self.addTimestampAction, 
            self.shuffleCmapAction, self.greedyShuffleCmapAction,
            self.saveLabColormapAction, self.zoomToObjsAction, 
            self.zoomOutAction, self.searchAction, self.focusOnSearchAction,
            self.launch3dViewerAction, *self.normalizeQActionGroup.actions()
        )
    
    def gui_createMenuBar(self):
        fileMenu = self.menuBar().addMenu('&File')
        self.fileMenu = fileMenu
        fileMenu.addAction(self.openFolderAction)
        fileMenu.addAction(self.openFileAction)
        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        fileMenu.addSeparator()
        fileMenu.addAction(self.showInExplorerAction)
        fileMenu.addAction(self.reloadAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.loadConfigurationAction)
        fileMenu.addAction(self.saveConfigurationAction)
        fileMenu.addAction(self.saveConfigurationAsAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        
        editMenu = self.menuBar().addMenu('&Edit')
        editMenu.addAction(self.editShortcutsAction)
        editMenu.addAction(self.editOverlayColorAction)
        editMenu.addAction(self.enableAutoZoomToCellsAction)
        
        viewMenu = self.menuBar().addMenu('&View')
        viewMenu.addAction(self.viewCcaTableAction)
        viewMenu.addSeparator()
        viewMenu.addAction(self.toggleColorSchemeAction)
        
        ImageMenu = self.menuBar().addMenu('&Image')
        ImageMenu.addAction(self.imgPropertiesAction)
        normalizeIntensitiesMenu = ImageMenu.addMenu('Normalize intensities')
        normalizeIntensitiesMenu.addAction(self.normalizeRawAction)
        normalizeIntensitiesMenu.addAction(self.normalizeToFloatAction)
        normalizeIntensitiesMenu.addAction(self.normalizeRescale0to1Action)
        normalizeIntensitiesMenu.addAction(self.normalizeByMaxAction)
        self.normalizeToFloatAction.setChecked(True)
        if 'how_normIntensities' in self.df_settings.index:
            how = self.df_settings.at['how_normIntensities', 'value']
            for action in self.normalizeQActionGroup.actions():
                if action.text() == how:
                    action.setChecked(True)
                    break
        self.rescaleIntensMenu = ImageMenu.addMenu('Rescale intensities (LUT)')
        ImageMenu.addAction(self.invertBwAction)
        ImageMenu.addAction(self.addScaleBarAction)
        ImageMenu.addAction(self.addTimestampAction)
        ImageMenu.addSeparator()
        ImageMenu.addAction(self.shuffleCmapAction)
        ImageMenu.addAction(self.greedyShuffleCmapAction)
        ImageMenu.addAction(self.saveLabColormapAction)
        ImageMenu.addSeparator()
        ImageMenu.addAction(self.zoomToObjsAction)
        ImageMenu.addAction(self.zoomOutAction)
        ImageMenu.addAction(self.launch3dViewerAction)
        
        annotMenu = self.menuBar().addMenu('&Annotations')
        self.annotationsMenu = annotMenu
        annotMenu.addAction(self.scalingLabelsAction)
        annotMenu.addAction(self.highLowResAction)
        annotMenu.addAction(self.editTextIDsColorAction)
        annotMenu.addAction(self.annotObjTrackSettingsAction)
        annotMenu.addSeparator()
        annotMenu.addAction(self.resetAnnotOptionsAction)
        
        self.searchWidget = widgets.ButtonSearchWidget(
            self, loadingType=None
        )
        self.menuBar().setCornerWidget(self.searchWidget)
        self.searchWidget.setMinimumHeight(
            self.searchWidget.sizeHint().height()
        )
    
    def gui_createToolBars(self):
        fileToolBar = self.addToolBar("File")
        fileToolBar.setMovable(False)
        fileToolBar.addAction(self.openFolderAction)
        fileToolBar.addAction(self.openFileAction)
        fileToolBar.addAction(self.showInExplorerAction)
        fileToolBar.addAction(self.reloadAction)
        self.fileToolBar = fileToolBar
        
        # Navigation toolbar with the 'Viewer' mode tools as in guiWin
        navigateToolBar = widgets.ToolBar("Navigation", self)
        navigateToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.addToolBar(navigateToolBar)
        self.navigateToolBar = navigateToolBar
        
        self.timepointsToolBar = widgets.ToolBar("Time points", self)
        self.timepointsToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.timepointsToolBar.addAction(self.addTimepointIdentifierAction)
        self.timepointsToolBar.addAction(self.reevaluateAllTimepointsAction)
        self.timepointsToolBar.addAction(self.reevaluateCellTimepointsAction)
        self.timepointsToolBar.addAction(self.editCellConstantsAction)
        self.timepointsToolBar.addSeparator()
        self.updateReevaluateTimepointsActions()
        self.addToolBar(self.timepointsToolBar)
        
        navigateToolBar.addAction(self.searchAction)
        
        self.zoomRectButton = QToolButton(self)
        self.zoomRectButton.setIcon(QIcon(":zoom_rect.svg"))
        self.zoomRectButton.setCheckable(True)
        self.zoomRectButton.setShortcut('Shift+Z')
        self.zoomRectButton.setToolTip(
            'Zoom to rectangular area\n\n'
            'Draw a rectangle with left-click to zoom into the area.\n'
            'Right-click to zoom back to the previous range.\n\n'
            'Shortcut: "Shift+Z"'
        )
        navigateToolBar.addWidget(self.zoomRectButton)
        self.leftClickButtons.append(self.zoomRectButton)
        self.widgetsWithShortcut['Zoom to rectangular area'] = (
            self.zoomRectButton
        )
        
        self.slideshowButton = QToolButton(self)
        self.slideshowButton.setIcon(QIcon(":eye-plus.svg"))
        self.slideshowButton.setCheckable(True)
        self.slideshowButton.setShortcut('Ctrl+W')
        self.slideshowButton.setToolTip(
            'Open a new window to browse frames or channels\n\n'
            'Shortcut: "Ctrl+W"'
        )
        navigateToolBar.addWidget(self.slideshowButton)
        
        navigateToolBar.addAction(self.launch3dViewerAction)
        
        self.overlayButton = widgets.rightClickToolButton(parent=self)
        self.overlayButton.setIcon(QIcon(":overlay.svg"))
        self.overlayButton.setCheckable(True)
        self.overlayButton.setToolTip(
            'Overlay channels\' images.\n\n'
            'Right-click on the button to overlay additional channels.\n\n'
            'Use the "Alpha" sliders to adjust the opacity.'
        )
        self.overlayButtonAction = navigateToolBar.addWidget(
            self.overlayButton
        )
        
        self.countObjsButton = QToolButton(self)
        self.countObjsButton.setIcon(QIcon(":count_objects.svg"))
        self.countObjsButton.setCheckable(True)
        self.countObjsButton.setShortcut('Ctrl+Shift+C')
        self.countObjsButton.setToolTip(
            'Count objects in the current view\n\n'
            'Shortcut: "Ctrl+Shift+C"'
        )
        navigateToolBar.addWidget(self.countObjsButton)
        
        self.overlayLabelsButton = widgets.rightClickToolButton(parent=self)
        self.overlayLabelsButton.setIcon(QIcon(":overlay_labels.svg"))
        self.overlayLabelsButton.setCheckable(True)
        self.overlayLabelsButton.setToolTip(
            'Add contours/masks of additional segmentation files.\n\n'
            'Right-click on the button to select which segmentation files '
            'to overlay.'
        )
        self.overlayLabelsButtonAction = navigateToolBar.addWidget(
            self.overlayLabelsButton
        )
        
        self.rulerButton = QToolButton(self)
        self.rulerButton.setIcon(QIcon(":ruler.svg"))
        self.rulerButton.setCheckable(True)
        self.rulerButton.setToolTip(
            'Draw a straight line and show its length. '
            'Length is displayed on the bottom-right corner.\n\n'
            'Hold "Ctrl" to snap the line to the closest 15° angle.'
        )
        navigateToolBar.addWidget(self.rulerButton)
        self.leftClickButtons.append(self.rulerButton)
        
        self.overlayColorButton = pg.ColorButton(self, color=(230,230,230))
        self.overlayColorButton.setDisabled(True)
        self.overlayColorButton.hide()
        
        self.overlayToolbar = widgets.OverlayToolbar(parent=self)
        self.addToolBar(Qt.TopToolBarArea, self.overlayToolbar)
        self.overlayToolbar.setVisible(False)
        
        self.setEnabledFileToolbar(False)
    
    def gui_connectActions(self):
        self.openFolderAction.triggered.connect(self.openFolder)
        self.openFileAction.triggered.connect(self.openFile)
        self.showInExplorerAction.triggered.connect(self.showInExplorer_cb)
        self.reloadAction.triggered.connect(self.reloadButtonClicked)
        self.exitAction.triggered.connect(self.close)
        self.loadConfigurationAction.triggered.connect(self.loadConfiguration)
        self.saveConfigurationAction.triggered.connect(self.saveConfiguration)
        self.saveConfigurationAsAction.triggered.connect(self.saveConfigurationAs)
        self.addTimepointIdentifierAction.triggered.connect(
            self.addTimepointIdentifier
        )
        self.reevaluateAllTimepointsAction.triggered.connect(
            self.reevaluateAllTimepoints
        )
        self.reevaluateCellTimepointsAction.triggered.connect(
            self.reevaluateDisplayedCellTimepoints
        )
        self.editCellConstantsAction.triggered.connect(self.editCellConstants)
        
        self.scalingLabelsAction.toggled.connect(self.scalingLabelsToggled)
        self.highLowResAction.toggled.connect(self.highLowResToggled)
        self.editTextIDsColorAction.triggered.connect(
            self.toggleTextIDsColorButton
        )
        self.annotObjTrackSettingsAction.triggered.connect(
            self.setAnnotateObjTrackSettings
        )
        self.resetAnnotOptionsAction.triggered.connect(
            self.resetDefaultAnnotationOptions
        )
        
        self.toggleColorSchemeAction.triggered.connect(
            self.onToggleColorScheme
        )
        self.editShortcutsAction.triggered.connect(self.editShortcuts_cb)
        self.editOverlayColorAction.triggered.connect(
            self.toggleOverlayColorButton
        )
        self.viewCcaTableAction.triggered.connect(self.viewCcaTable)
        self.imgPropertiesAction.triggered.connect(self.editImgProperties)
        self.invertBwAction.toggled.connect(self.invertBw)
        self.addScaleBarAction.toggled.connect(self.addScaleBar)
        self.addTimestampAction.toggled.connect(self.addTimestamp)
        self.shuffleCmapAction.triggered.connect(self.shuffle_cmap)
        self.greedyShuffleCmapAction.triggered.connect(self.greedyShuffleCmap)
        self.saveLabColormapAction.triggered.connect(self.saveLabelsColormap)
        self.normalizeQActionGroup.triggered.connect(
            self.normaliseIntensitiesActionTriggered
        )
        self.zoomToObjsAction.triggered.connect(self.zoomToObjsActionCallback)
        self.zoomOutAction.triggered.connect(self.zoomOut)
        self.launch3dViewerAction.toggled.connect(
            self.launch3dViewerActionToggled
        )
        self.searchAction.triggered.connect(
            partial(self.focusOnSearch, blinking=True)
        )
        self.focusOnSearchAction.triggered.connect(self.focusOnSearch)
        self.addAction(self.focusOnSearchAction)
        self.searchWidget.sigTriggerBlink.connect(self.onSearchTriggerBlink)
        self.searchWidget.sigSearchId.connect(self.onSearchId)
        self.nextAction.triggered.connect(self.nextActionTriggered)
        self.prevAction.triggered.connect(self.prevActionTriggered)
        
        self.zoomRectButton.toggled.connect(self.zoomRectActionToggled)
        self.slideshowButton.toggled.connect(self.launchSlideshow)
        self.overlayButton.toggled.connect(self.overlay_cb)
        self.overlayButton.sigRightClick.connect(self.showOverlayContextMenu)
        self.countObjsButton.toggled.connect(self.countObjectsCb)
        self.overlayLabelsButton.toggled.connect(self.overlayLabels_cb)
        self.overlayLabelsButton.sigRightClick.connect(
            self.showOverlayLabelsContextMenu
        )
        self.rulerButton.toggled.connect(self.ruler_cb)
        self.overlayColorButton.sigColorChanging.connect(
            self.changeOverlayColor
        )
        self.overlayColorButton.sigColorChanged.connect(self.saveOverlayColor)
        self.overlayToolbar.sigSetTranspacency.connect(
            self.setOverlayTransparency
        )
        self.overlayToolbar.sigSetSingleChannel.connect(
            self.setOverlaySingleChannel
        )
    
    def setEnabledFileToolbar(self, enabled):
        for action in getattr(self, 'annotationsActions', ()):
            action.setEnabled(enabled)
        for action in getattr(self, 'viewerToolsActions', ()):
            action.setEnabled(enabled)
        self.navigateToolBar.setEnabled(enabled)
        alwaysEnabledActions = (self.openFolderAction, self.openFileAction)
        for action in self.fileToolBar.actions():
            if action in alwaysEnabledActions:
                continue
            action.setEnabled(enabled)
        if enabled and self.isDataLoaded:
            self.imgPropertiesAction.setEnabled(True)
            self.viewCcaTableAction.setEnabled(self.hasCcaAnnotations())
        else:
            self.imgPropertiesAction.setEnabled(False)
            self.viewCcaTableAction.setEnabled(False)
        
        # As in guiWin the 3D viewer is available only for z-stacks
        isZstack = (
            enabled and self.isDataLoaded and self.posData().SizeZ > 1
        )
        for widget in (
                getattr(self, 'launch3dViewerButton', None), 
                self.launch3dViewerAction
            ):
            if widget is not None:
                widget.setDisabled(not isZstack)
                widget.setVisible(isZstack)
        self.updateFunctionalitySearchRecords()

    def updateFunctionalitySearchRecords(self):
        """Expose only enabled functionality available in this viewer."""
        records = {}
        recentPathActions = tuple(self.openRecentMenu.actions())

        def add_record(name, target):
            if any(target is action for action in recentPathActions):
                return
            name = name.replace('&', '').strip().rstrip('.')
            if not name or name == 'Search for ID or functionality':
                return
            records[name.casefold()] = {
                'display': name,
                'search_name': name,
                'target': target,
            }

        def add_menu_actions(menu):
            if menu is self.openRecentMenu or menu.title() == 'Open Recent':
                return
            for action in menu.actions():
                if action.isSeparator():
                    continue
                submenu = action.menu()
                if submenu is not None:
                    add_menu_actions(submenu)
                elif action.isEnabled():
                    add_record(action.text(), action)

        for action in self.menuBar().actions():
            menu = action.menu()
            if menu is not None:
                add_menu_actions(menu)
            elif action.isEnabled():
                add_record(action.text(), action)

        for name, target in self.widgetsWithShortcut.items():
            targets = target if isinstance(target, (list, tuple)) else (target,)
            if all(member.isEnabled() for member in targets):
                add_record(name, target)

        for button in (
                getattr(self, name, None)
                for name in (
                    'zoomRectButton', 'slideshowButton', 'overlayButton',
                    'countObjsButton', 'overlayLabelsButton', 'rulerButton',
                    'normalizeCheckbox', 'refreshButton',
                )
            ):
            if (
                    button is not None
                    and button.isEnabled()
                    and not button.isHidden()
                ):
                name = button.text().strip()
                if not name:
                    tooltip = button.toolTip()
                    name = (
                        tooltip.splitlines()[0].strip()
                        if tooltip else ''
                    )
                add_record(name, button)

        self.searchWidget.setSearchRecords(records)
    
    def hasCcaAnnotations(self):
        return any(
            getattr(posData, 'cca_df', None) is not None 
            or (
                getattr(posData, 'acdc_df', None) is not None
                and 'cell_cycle_stage' in posData.acdc_df.columns
            )
            for posData in self.data
        )
    
    def nextActionTriggered(self):
        if self.zSliceCheckbox.isChecked():
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )
        else:
            self.navigateScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )
    
    def prevActionTriggered(self):
        if self.zSliceCheckbox.isChecked():
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepSub
            )
        else:
            self.navigateScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepSub
            )
    
    def launch3dViewerActionToggled(self, checked):
        self.launch3dViewerButton.setChecked(checked)
    
    def normaliseIntensitiesActionTriggered(self, action):
        how = action.text()
        self.df_settings.at['how_normIntensities', 'value'] = how
        self.df_settings.to_csv(self.settings_csv_path)
        if not self.isDataLoaded:
            return
        self.updateAllImages()
        self.updateImageValueFormatter()
    
    def getHardcodedShortcuts(self):
        hard_shortcuts = {}
        actions = (
            self.openFolderAction, self.reloadAction, 
            self.focusOnSearchAction, self.editShortcutsAction, 
            self.viewCcaTableAction, 
            self.shuffleCmapAction, self.greedyShuffleCmapAction,
            self.slideshowButton, self.countObjsButton
        )
        for widget in actions:
            shortcut = widget.shortcut()
            if shortcut.isEmpty():
                continue
            text = widget.text() or widget.toolTip().split('\n')[0]
            hard_shortcuts[text] = shortcut.toString()
        return hard_shortcuts
    
    def showInExplorer_cb(self):
        posData = self.posData()
        myutils.showInExplorer(posData.images_path)
    
    def openRecentFile(self, path):
        print(f'Opening recent folder: {path}')
        self.addToRecentPaths(path)
        self.openFolder(folderpath=path)
    
    def openFolder(self, checked=False, folderpath=None):
        if not folderpath:
            folderpath = QFileDialog.getExistingDirectory(
                self, 'Select experiment, Position or Images folder', 
                self.getMostRecentPath()
            )
        if not folderpath:
            return
        
        self.addToRecentPaths(folderpath)
        self.addPathToOpenRecentMenu(folderpath)
        
        images_paths = get_images_paths(folderpath)
        if not images_paths:
            self.warnLoading(
                f'No Position or Images folders found in "{folderpath}".'
            )
            return
        
        self.loadImagesPaths(images_paths)
    
    def openFile(self, checked=False, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 'Select image file from an Images folder', 
                self.getMostRecentPath(), 
                'Images (*.tif *.tiff *.npz *.npy *.h5);;All Files (*)'
            )
        if not file_path:
            return
        
        images_path = os.path.dirname(file_path)
        if os.path.basename(images_path) != 'Images':
            self.warnLoading(
                f'The file "{file_path}" is not in an "Images" folder.\n\n'
                'Create the Cell-ACDC data structure first (e.g., by opening '
                'the file in the main GUI).'
            )
            return
        
        self.addToRecentPaths(images_path)
        self.addPathToOpenRecentMenu(images_path)
        channel = get_channel_from_filepath(file_path)
        self.loadImagesPaths([images_path], channel=channel)
    
    def loadImagesPaths(self, images_paths, channel=None):
        try:
            chNames, segm_endnames = get_channels_and_segm_endnames(
                images_paths[0]
            )
        except Exception as err:
            self.warnLoading(str(err))
            return
        
        if not segm_endnames:
            self.warnLoading(
                f'No segmentation files found in "{images_paths[0]}".'
            )
            return
        
        if channel is None:
            channel = self.selectItem(
                'Select channel', 'Select channel to display:', chNames
            )
        if channel is None:
            return
        
        segm_endname = self.selectItem(
            'Select segmentation file', 'Select segmentation file to load:', 
            segm_endnames
        )
        if segm_endname is None:
            return
        
        self.setCursor(Qt.WaitCursor)
        try:
            data = load_positions(
                images_paths, channel, segm_endname=segm_endname
            )
        finally:
            self.unsetCursor()
        
        if not data:
            self.warnLoading(
                'None of the selected Positions could be loaded '
                '(see terminal for details).'
            )
            return
        
        self.isSegm3D = None
        self.setData(data)
    
    def selectItem(self, title, text, items):
        if len(items) == 1:
            return items[0]
        
        win = widgets.QDialogListbox(
            title, text, items, multiSelection=False, parent=self
        )
        win.exec_()
        if win.cancel or not win.selectedItemsText:
            return
        return win.selectedItemsText[0]
    
    def warnLoading(self, text):
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Loading data failed', text)
    
    def setData(self, data, pos_i=0, frame_i=None, ID=None):
        self.data = list(data) if data else []
        hasData = len(self.data) > 0
        self.centralWidget().setEnabled(hasData)
        self.setEnabledFileToolbar(hasData)
        
        self.posCombobox.blockSignals(True)
        self.posCombobox.clear()
        for posData in self.data:
            self.posCombobox.addItem(
                getattr(posData, 'pos_foldername', str(posData))
            )
        self.posCombobox.blockSignals(False)
        
        if not hasData:
            isMultiPos = False
            self.posLabel.setVisible(isMultiPos)
            self.posCombobox.setVisible(isMultiPos)
            return
        
        relPath = getattr(self.data[0], 'relPath', '')
        self.setWindowTitle(f'Cell-ACDC - Measurements viewer - {relPath}')
        self.pos_i = pos_i
        self.setPosDataDefaults()
        self.initLut()
        self.initViewerTools()
        self.loadPosition(pos_i, frame_i=frame_i, ID=ID)
        self.initViewerToolsAfterLoading()
        self.setEnabledFileToolbar(hasData)
        self.resizeBottomLayout()
    
    def resizeBottomLayout(self):
        # As in `gui.guiWin.resizeGui`: fit the bottom layout to its 
        # content and then restore the height saved by the user
        self.bottomScrollArea._resizeVertical()
        self.setBottomLayoutHeight()
    
    def setPosDataDefaults(self):
        """Add the metadata attributes that `load.loadData` always sets, but 
        that minimal posData objects passed to `setData` might lack"""
        defaults = {
            'PhysicalSizeX': 1.0, 'PhysicalSizeY': 1.0, 'PhysicalSizeZ': 1.0,
            'TimeIncrement': 1.0, 'basename': '', 'loadedFluoChannels': set(),
        }
        for posData in self.data:
            for name, value in defaults.items():
                if not hasattr(posData, name):
                    setattr(posData, name, copy.copy(value))
            for name in ('fluo_data_dict', 'fluo_bkgrData_dict'):
                if not hasattr(posData, name):
                    setattr(posData, name, {})
    
    def resetViewerToolsWidgets(self):
        """Uncheck the tools and remove the overlay items of previously 
        loaded data"""
        for button in (
                self.overlayButton, self.overlayLabelsButton, 
                self.rulerButton, self.zoomRectButton, self.countObjsButton,
                self.slideshowButton
            ):
            if button.isChecked():
                button.setChecked(False)
        if self.launch3dViewerAction.isChecked():
            self.launch3dViewerAction.setChecked(False)
        self.overlayToolbar.setVisible(False)
        for widget in self.overlayZsliceWidgets():
            widget.hide()
        
        self.removeOverlayItems()
        # Re-added by `gui_createOverlayItems`
        self.ax1.removeItem(self.rgbaImg1)
        for items in self.overlayLayersItems.values():
            imageItem, lutItem, alphaSB = items[:3]
            self.ax1.removeItem(imageItem)
            for widget in (alphaSB, alphaSB.label):
                self.bottomLeftLayout.removeWidget(widget)
                widget.deleteLater()
        alphaScrollbar = getattr(self.img1, 'alphaScrollbar', None)
        if alphaScrollbar is not None:
            for widget in (alphaScrollbar, alphaScrollbar.label):
                self.bottomLeftLayout.removeWidget(widget)
                widget.deleteLater()
            self.img1.alphaScrollbar = None
        self.alphaScrollbarRow = self.firstAlphaScrollbarRow
        
        for items in self.overlayLabelsItems.values():
            imageItem, contoursItem, gradItem = items
            self.ax1.removeItem(imageItem)
            self.ax1.removeItem(contoursItem)
            self.mainLayout.removeWidget(gradItem)
            gradItem.deleteLater()
        if hasattr(self, 'overlayLabelsContoursImages'):
            del self.overlayLabelsContoursImages
        
        self.overlayLayersItems = {}
        self.overlayLabelsItems = {}
        self.drawModeOverlayLabelsChannels = {}
        self.fluoDataChNameActions = []
        self.checkedOverlayChannels = set()
        self.overlayButton.setStyleSheet('')
        self.lutItemsLayout.addItem(self.imgGrad, row=0, col=0)
        
        if hasattr(self, 'zoomRectItem'):
            self.ax1.removeItem(self.zoomRectItem)
        self.tempSegmentON = False
        self.isMouseDragImg1 = False
        self.ax1_rulerPlotItem.setData([], [])
        self.ax1_rulerAnchorsItem.setData([], [])
        self.clearCursors()
    
    def initViewerTools(self):
        """Initialize the overlay of other channels and segmentation files 
        as in `gui.guiWin` after loading data"""
        self.resetViewerToolsWidgets()
        
        posData = self.posData()
        self.isSnapshot = posData.SizeT == 1
        try:
            self.ch_names, _ = get_channels_and_segm_endnames(
                posData.images_path
            )
        except Exception as err:
            self.ch_names = [self.user_ch_name]
        if self.user_ch_name not in self.ch_names:
            self.ch_names = [self.user_ch_name, *self.ch_names]
        
        existingSegmEndNames = set()
        for _posData in self.data:
            try:
                segm_files = load.get_segm_files(_posData.images_path)
                existingSegmEndNames.update(
                    load.get_endnames(_posData.basename, segm_files)
                )
            except Exception as err:
                # posData without files on disk --> no overlay labels
                pass
        self.existingSegmEndNames = existingSegmEndNames
        
        for _posData in self.data:
            _posData.ol_data_dict = {}
            _posData.ol_data = None
            _posData.ol_labels_data = None
            for ch in self.ch_names:
                _posData.lutLevels.setdefault(ch, {})
        
        self.setOverlayColors()
        self.createOverlayContextMenu()
        self.userChNameAction = QAction(self)
        self.userChNameAction.setCheckable(True)
        self.userChNameAction.setText(self.user_ch_name)
        self.gui_createOverlayColors()
        self.gui_createOverlayItems()
        
        self.createOverlayLabelsContextMenu(existingSegmEndNames)
        self.createOverlayLabelsItems(existingSegmEndNames)
        self.gui_addOverlayLayerItems()
        
        self.chNamesQActionGroup = QActionGroup(self)
        self.chNamesQActionGroup.addAction(self.userChNameAction)
        self.userChNameAction.setChecked(True)
        self.addActionsLutItemContextMenu(self.imgGrad)
        # Scrollbar for opacity of img1 (when overlaying)
        self.img1.alphaScrollbar = self.addAlphaScrollbar(
            self.user_ch_name, self.img1
        )
        
        self.guiTabControl.addChannels([self.user_ch_name])
        self.showPropsDockButton.setDisabled(False)
        self.setPermanentGreedyCmapPreferences()
    
    def initViewerToolsAfterLoading(self):
        self.updateImageValueFormatter()
        self.initPixelSizePropsDockWidget()
        if self.invertBwAction.isChecked() and not self.invertBwAlreadyCalledOnce:
            self.invertBw(True)
        self.gui_createZoomRectItem()
    
    def gui_createGraphicsPlots(self):
        self.graphLayout = pg.GraphicsLayoutWidget()
        if self.invertBwAction.isChecked():
            self.graphLayout.setBackground(graphLayoutBkgrColor)
            self.titleColor = 'black'
        else:
            self.graphLayout.setBackground(darkBkgrColor)
            self.titleColor = 'white'
        
        # Layout with the LUT items of the image and overlaid channels
        self.lutItemsLayout = self.graphLayout.addLayout(row=0, col=0)
        
        self.ax1 = widgets.MainPlotItem(ax_number=0)
        self.ax1.invertY(True)
        self.ax1.setAspectLocked(True)
        self.ax1.hideAxis('bottom')
        self.ax1.hideAxis('left')
        self.graphLayout.addItem(self.ax1, row=0, col=1)
        
        self.imgGrad = widgets.myHistogramLUTitem(parent=self, name='image')
        self.imgGrad.restoreState(self.df_settings)
        # Main GUI-only features
        imgGradMenu = self.imgGrad.gradient.menu
        imgGradMenu.removeAction(self.imgGrad.fontSizeMenu.menuAction())
        self.lutItemsLayout.addItem(self.imgGrad, row=0, col=0)
        for action in self.imgGrad.rescaleActionGroup.actions():
            if action.text() == self.defaultRescaleIntensHow:
                action.setChecked(True)
            self.rescaleIntensMenu.addAction(action)
        
        # Labels colormap used for the segm. masks overlay (as in guiWin)
        self.labelsGrad = widgets.labelsGradientWidget(parent=self)
        try:
            self.labelsGrad.restoreState(self.df_settings)
        except Exception as err:
            self.logger.info(
                'Failed to restore previously used colormap. '
                'Using default colormap "viridis"'
            )
            self.labelsGrad.item.loadPreset('viridis')
        self.labelsGrad.showRightImgAction.setChecked(False)
        self.hideRightImageLabelsGradActions()
        
        self.img1 = widgets.ParentImageItem()
        self.imgGrad.setImageItem(self.img1)
        self.img1.lutItem = self.imgGrad
        self.imgGrad.sigRescaleIntes.connect(self.rescaleIntensitiesLut)
        self.imgGrad.gradient.sigGradientChangeFinished.connect(
            self.imgGradLUTfinished_cb
        )
        self.ax1.addBaseImageItem(self.img1)
        
        # RGBA image for true transparency mode (added in 
        # `gui_createOverlayItems`)
        self.rgbaImg1 = pg.ImageItem()
        
        # Overwrite axes viewbox context menu
        self.ax1.vb.menu = self.imgGrad.gradient.menu
        
        self.gui_createAnnotationsItems()
        
        # Ruler plotItem and scatterItem
        rulerPen = pg.mkPen(color='r', style=Qt.DashLine, width=2)
        self.ax1_rulerPlotItem = widgets.RulerPlotItem(pen=rulerPen)
        self.ax1_rulerAnchorsItem = pg.ScatterPlotItem(
            symbol='o', size=9,
            brush=pg.mkBrush((255,0,0,50)),
            pen=pg.mkPen((255,0,0), width=2), tip=None
        )
        self.ax1.addItem(self.ax1_rulerPlotItem)
        self.ax1.addItem(self.ax1_rulerPlotItem.labelItem)
        self.ax1.addItem(self.ax1_rulerAnchorsItem)
        
        # Mirrored cursor from the slideshow window
        self.ax1_cursor = pg.ScatterPlotItem(
            symbol='d', pxMode=True, 
            pen=pg.mkPen('w', width=1),
            brush=pg.mkBrush((255, 255, 255, 70)), 
            size=16, tip=None
        )
        self.ax1.addItem(self.ax1_cursor)
        
        self.img1.hoverEvent = self.gui_hoverEventImg1
        self.img1.mousePressEvent = self.gui_mousePressEventImg1
        self.img1.mouseMoveEvent = self.gui_mouseDragEventImg1
        self.img1.mouseReleaseEvent = self.gui_mouseReleaseEventImg1
        self.img1.mouseClickEvent = self.gui_mouseClickEventImg1
        self.ax1.sigRangeChanged.connect(self.viewRangeChanged)
        
        self.labelsGrad.sigGradientChangeFinished.connect(self.updateLabelsCmap)
        self.labelsGrad.sigShuffleCmap.connect(self.shuffle_cmap)
        self.labelsGrad.sigGreeedyShuffleCmap.connect(self.greedyShuffleCmap)
        self.labelsGrad.permanentGreedyCmapAction.toggled.connect(
            self.permanentGreedyCmapToggled
        )
        self.labelsGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)
        self.labelsGrad.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )
        self.imgGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)
        self.imgGrad.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )
        self.imgGrad.sigAddScaleBar.connect(self.addScaleBarAction.setChecked)
        self.imgGrad.sigAddTimestamp.connect(self.addTimestampAction.setChecked)
    
    def hideRightImageLabelsGradActions(self):
        # The viewer has no right image (segm. image, duplicated left image, 
        # or next frame), hence the actions that only affect it are hidden
        lg = self.labelsGrad
        rightImageButtons = (lg.colorButton, lg.textColorButton)
        for action in lg.menu.actions():
            if not isinstance(action, QWidgetAction):
                continue
            widget = action.defaultWidget()
            if widget is None:
                continue
            if any(widget.isAncestorOf(b) for b in rightImageButtons):
                action.setVisible(False)
        
        for action in (
                lg.showLabelsImgAction, lg.showRightImgAction, 
                lg.showNextFrameAction, lg.fontSizeMenu.menuAction()
            ):
            action.setVisible(False)
    
    def gui_createAnnotationsItems(self):
        """Create the annotations items as in `gui.guiWin` (left image)"""
        # Text annotations color
        self.textIDsColorButton = pg.ColorButton(self)
        self.textIDsColorButton.hide()
        if 'textIDsColor' in self.df_settings.index:
            rgbString = self.df_settings.at['textIDsColor', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.gui_createTextAnnotColors(r, g, b, custom=True)
            self.textIDsColorButton.setColor((r, g, b))
        else:
            self.gui_createTextAnnotColors(0, 0, 0, custom=False)
        
        self.gui_createContourPens()
        self.gui_createMothBudLinePens()
        
        # Overlay segm. masks item
        self.labelsLayerImg1 = widgets.BaseLabelsImageItem()
        self.ax1.addItem(self.labelsLayerImg1)
        self.setLut()
        self.initLabelsImageItems()
        
        self.ax1_contoursImageItem = pg.ImageItem()
        self.ax1_oldMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.oldMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax1_newMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.newMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax1_movementAgainstPrevLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.oldMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax1_movementAgainstPrevScatterItem = (
            self.gui_getLostObjScatterItem()
        )
        # Required to initialize the tracked lost objects text color
        self.gui_getTrackedLostObjScatterItem()
        
        self.ax1.addItem(self.ax1_contoursImageItem)
        self.ax1.addItem(self.ax1_oldMothBudLinesItem)
        self.ax1.addItem(self.ax1_newMothBudLinesItem)
        self.ax1.addItem(self.ax1_movementAgainstPrevLinesItem)
        self.ax1.addItem(self.ax1_movementAgainstPrevScatterItem)
        
        # Selected ID highlight items (same as searched ID in guiWin)
        self.highLightIDLayerImg1 = pg.ImageItem()
        self.ax1.addItem(self.highLightIDLayerImg1)
        self.searchedIDitemLeft = pg.ScatterPlotItem()
        self.searchedIDitemLeft.setData(
            [], [], symbol='s', pxMode=False, size=1,
            brush=pg.mkBrush(color=(255,0,0,150)),
            pen=pg.mkPen(width=3, color='r'), tip=None
        )
        self.ax1.addItem(self.searchedIDitemLeft)
        
        self.textIDsColorButton.sigColorChanging.connect(
            self.updateTextAnnotColor
        )
        self.textIDsColorButton.sigColorChanged.connect(self.saveTextIDsColors)
        self.imgGrad.textColorButton.disconnect()
        self.imgGrad.textColorButton.clicked.connect(
            self.editTextIDsColorAction.trigger
        )
        self.imgGrad.labelsAlphaSlider.valueChanged.connect(
            self.updateLabelsAlpha
        )
    
    def initLabelsImageItems(self):
        lut = self.getLabelsImageLut()
        self.labelsLayerImg1.setLevels([0, len(lut)])
        self.labelsLayerImg1.setLookupTable(lut)
        alpha = self.imgGrad.labelsAlphaSlider.value()
        self.labelsLayerImg1.setOpacity(alpha)
    
    def updateLabelsAlpha(self, value):
        self.df_settings.at['overlaySegmMasksAlpha', 'value'] = value
        self.df_settings.to_csv(self.settings_csv_path)
        self.labelsLayerImg1.setOpacity(value)
    
    def gui_createAxesTextAnnotItems(self, allIDs):
        if 0 in self.textAnnot:
            self.textAnnot[0].removeFromPlotItem(self.ax1)
        self.gui_createTextAnnotItems(allIDs)
        self.gui_setTextAnnotColors()
        self.textAnnot[0].addToPlotItem(self.ax1)
        self.initTextAnnot()
        self.syncTextAnnotOptions()
    
    def syncTextAnnotOptions(self):
        """Set the text annotations flags from the annotation checkboxes"""
        for ax in self.annotatedAxesIndices():
            if ax not in self.textAnnot:
                continue
            textAnnot = self.textAnnot[ax]
            textAnnot.setLabelAnnot(self.annotIDsCheckbox(ax).isChecked())
            textAnnot.setCcaAnnot(self.isCellCycleInfoChecked(ax))
            textAnnot.setLineageAnnot(
                self.annotLineageInfoCheckbox(ax).isChecked()
            )
            textAnnot.setScaling(self.scalingLabelsAction.isChecked())
    
    def restoreViewerAnnotationOptions(self):
        """Restore the saved annotation options without triggering the 
        checkboxes callbacks (data might not be loaded yet)."""
        checkboxes = [
            checkbox for checkbox in self.annotOptionsCheckboxes[0].values()
            if hasattr(checkbox, 'setChecked')
        ]
        for checkbox in checkboxes:
            checkbox.blockSignals(True)
        try:
            self.restoreSavedAnnotationOptions()
        finally:
            for checkbox in checkboxes:
                checkbox.blockSignals(False)
        self.syncTextAnnotOptions()
    
    def resetDefaultAnnotationOptions(self, *args):
        super().resetDefaultAnnotationOptions(*args)
        self.syncTextAnnotOptions()
        if self.isDataLoaded:
            self.updateImages()
    
    def resetFocus(self):
        self.setFocus()
    
    def setTitleText(self, *args, **kwargs):
        # The viewer does not have the title label of the main GUI
        pass
    
    def setLookingGoodTitle(self):
        pass
    
    def get_labels_array(
            self, from_store=False, frame_i=None, return_existing=False,
            return_copy=True
        ):
        posData = self.posData()
        if frame_i is None:
            frame_i = posData.frame_i
        lab = posData.segm_data[frame_i]
        if return_copy:
            lab = lab.copy()
        if return_existing:
            return lab, False
        return lab
    
    def changeFontSize(self):
        fontSize = self.fontSizeSpinBox.value()
        if fontSize == self.fontSize:
            return
        
        self.fontSize = fontSize
        self._set_setting_value('fontSize', self.fontSize)
        if not self.isDataLoaded:
            return
        
        for ax in self.annotatedAxesIndices():
            self.textAnnot[ax].changeFontSize(self.fontSize)
        self.updateImages()
    
    def changeTextResolution(self):
        if not self.isDataLoaded:
            return
        
        mode = 'high' if self.highLowResAction.isChecked() else 'low'
        posData = self.posData()
        img_shape = self.img1.image.shape[:2]
        scalingMode = self.scalingLabelsAction.isChecked()
        for ax in self.annotatedAxesIndices():
            self.textAnnot[ax].changeResolution(
                mode, posData.allIDs, self.ax1, img_shape, 
                scalingMode=scalingMode
            )
        self.updateImages()
    
    def highLowResToggled(self, checked=True):
        self._set_setting_value('highLowRes', self.highLowResAction.isChecked())
        self.changeTextResolution()
    
    def scalingLabelsToggled(self, checked=True):
        self._set_setting_value(
            'scalingLabels', self.scalingLabelsAction.isChecked()
        )
        if not self.isDataLoaded:
            return
        for annot in self.textAnnot.values():
            annot.setScaling(self.scalingLabelsAction.isChecked())
    
    def imgGradLUTfinished_cb(self):
        self.df_settings = self.imgGrad.saveState(self.df_settings)
        self.df_settings.to_csv(self.settings_csv_path)
    
    def initLut(self):
        """Initialize LUT levels as in `gui.guiWin` after loading data."""
        self.isDataLoaded = True
        posData = self.posData()
        self.user_ch_name = posData.user_ch_name
        self.rescaleIntensChannelHowMapper.setdefault(
            self.user_ch_name, 'Rescale each 2D image'
        )
        for _posData in self.data:
            if not hasattr(_posData, 'lutLevels'):
                _posData.lutLevels = {self.user_ch_name: {}}
        self.imgGrad.setAxisLabel(self.user_ch_name)
        self.img1.preComputedMinMaxValues(self.data)
        self.imgGrad.rescaleAcrossZstackAction.setDisabled(
            posData.SizeZ == 1
        )
        self.initImgGradRescaleIntensitiesHowPreference()
        self.rescaleIntensitiesLut(setImage=False)
    
    def gui_createNavigationWidgets(self):
        self.gui_createFramesNavigationWidgets()
        self.t_label.setText('Frame n.')
        self.gui_createZsliceNavigationWidgets()
        # z-projection lock is a main GUI-only feature
        self.zProjLockViewButton.hide()
        
        self.zSliceOverlay_SB = widgets.ScrollBar(Qt.Horizontal)
        self.overlay_z_label = QLabel('Overlay z-slice  ')
        self.overlay_z_label.setFont(_font)
        self.overlay_z_label.setDisabled(True)
        self.zProjOverlay_CB = widgets.ComboBox()
        self.zProjOverlay_CB.setFont(_font)
        self.zProjOverlay_CB.addItems([
            'single z-slice', 'max z-projection', 'mean z-projection',
            'median z-proj.', 'same as above'
        ])
        self.zProjOverlay_CB.setCurrentIndex(4)
        self.zSliceOverlay_SB.setDisabled(True)
        for widget in self.overlayZsliceWidgets():
            widget.hide()
        
        self.idScrollBar = widgets.ScrollBar(Qt.Horizontal)
        self.idScrollBar.setMinimum(0)
        self.idLabel = QLabel('Cell ID')
        self.idLabel.setFont(_font)
        self.idSpinBox = widgets.SpinBox(disableKeyPress=True)
        self.idSpinBox.setMinimum(0)
        
        idSpinBoxHight = int(self.idSpinBox.sizeHint().height() * 0.8)
        self.idScrollBar.setFixedHeight(idSpinBoxHight)
        self.numIDsLabel = QLabel('')
        self.idNotPresentLabel = QLabel('')
        self.idNotPresentLabel.setFont(_font)
        self.followIDcheckbox = QCheckBox('Follow cell')
        self.followIDcheckbox.setToolTip(
            'If active, the view zooms on the selected cell also when '
            'changing frame or z-slice'
        )
        self.restrictFramesCheckbox = QCheckBox(
            'Only frames with selected cell'
        )
        self.restrictFramesCheckbox.setToolTip(
            'If active, the frames slider is restricted to the range of '
            'frames where the selected cell ID is present'
        )
        self.rescalePlotCheckbox = QCheckBox('Rescale plot')
        self.rescalePlotCheckbox.setToolTip(
            'If active, the plot is rescaled when changing Cell ID'
        )
        
        self.posLabel = QLabel('Position')
        self.posCombobox = QComboBox()
        
        navLayout = QGridLayout()
        row = 0
        posLayout = QHBoxLayout()
        posLayout.addWidget(self.posLabel)
        posLayout.addWidget(self.posCombobox)
        posLayout.addStretch(1)
        navLayout.addLayout(posLayout, row, 0, 1, 4)
        
        row += 1
        navLayout.addLayout(
            self.gui_createFramesNavigationLabelLayout(), row, 0, 
            alignment=Qt.AlignRight
        )
        navLayout.addWidget(self.navigateScrollBar, row, 1, 1, 2)
        
        row += 1
        navLayout.addLayout(
            self.gui_createZsliceCheckboxLayout(), row, 0, 
            alignment=Qt.AlignRight
        )
        navLayout.addWidget(self.zSliceScrollBar, row, 1, 1, 2)
        navLayout.addWidget(self.zProjComboBox, row, 3)
        navLayout.addWidget(self.switchPlaneCombobox, row, 4)
        navLayout.addWidget(self.launch3dViewerButton, row, 5)
        
        row += 1
        navLayout.addWidget(
            self.overlay_z_label, row, 0, alignment=Qt.AlignRight
        )
        navLayout.addWidget(self.zSliceOverlay_SB, row, 1, 1, 2)
        navLayout.addWidget(self.zProjOverlay_CB, row, 3)
        
        row += 1
        idLayout = QHBoxLayout()
        idLayout.addWidget(self.idLabel)
        idLayout.addWidget(self.idSpinBox)
        idLayout.addWidget(self.numIDsLabel)
        navLayout.addLayout(idLayout, row, 0, alignment=Qt.AlignRight)
        navLayout.addWidget(self.idScrollBar, row, 1, 1, 2)
        
        row += 1
        idOptionsLayout = QHBoxLayout()
        idOptionsLayout.addWidget(self.followIDcheckbox)
        idOptionsLayout.addWidget(self.restrictFramesCheckbox)
        idOptionsLayout.addWidget(self.idNotPresentLabel)
        idOptionsLayout.addWidget(self.rescalePlotCheckbox)
        idOptionsLayout.addStretch(1)
        navLayout.addLayout(idOptionsLayout, row, 0, 1, 6)
        
        for scrollbar in self.navigationScrollbars():
            scrollbar.setMinimumWidth(200)
        
        navLayout.setColumnStretch(0, 0)
        navLayout.setColumnStretch(1, 3)
        navLayout.setColumnStretch(2, 0)
        
        # Overlay channels alpha scrollbars are added from this row on 
        # (see `OverlayChannelsGuiElements.addAlphaScrollbar`)
        self.bottomLeftLayout = navLayout
        self.firstAlphaScrollbarRow = row + 1
        self.alphaScrollbarRow = row + 1
        self.h = self.navSpinBox.sizeHint().height()
        
        self.navigationGroupbox = QGroupBox('Navigate')
        self.navigationGroupbox.setLayout(navLayout)
    
    def overlayZsliceWidgets(self):
        return (self.overlay_z_label, self.zSliceOverlay_SB, self.zProjOverlay_CB)
    
    def navigationScrollbars(self):
        return (self.navigateScrollBar, self.zSliceScrollBar, self.idScrollBar)
    
    def gui_createAnnotationsWidgets(self):
        """Create the annotation options checkboxes and the quick settings 
        of `gui.guiWin` relevant for the display of the annotations."""
        self.gui_setupAnnotationOptionsLayout(0)
        
        self.annotLostObjsToggle = widgets.Toggle()
        annotLostObjsToggleTooltip = (
            'Toggle annotation of lost objects mode ON/OFF'
        )
        self.annotLostObjsToggle.setChecked(True)
        self.annotLostObjsToggle.setToolTip(annotLostObjsToggleTooltip)
        annotLostObjsLabel = QLabel('Annot. lost objects')
        annotLostObjsLabel.setToolTip(annotLostObjsToggleTooltip)
        self.annotLostObjsToggle.toggled.connect(self.annotLostObjsToggled)
        
        self.showAllContoursToggle = widgets.Toggle()
        showAllContoursTooltip = (
            'If active, all contours will be displayed, including inner '
            'contours (e.g. holes and sub-objects)'
        )
        self.showAllContoursToggle.setToolTip(showAllContoursTooltip)
        showAllContourLabel = QLabel('Show all contours')
        showAllContourLabel.setToolTip(showAllContoursTooltip)
        self.showAllContoursToggle.toggled.connect(
            self.showAllContoursToggled
        )
        
        self.fontSizeSpinBox = widgets.SpinBox()
        self.fontSizeSpinBox.setMinimum(1)
        self.fontSizeSpinBox.setMaximum(99)
        self.fontSizeSpinBox.setValue(self.fontSize)
        self.fontSizeSpinBox.editingFinished.connect(self.changeFontSize) 
        self.fontSizeSpinBox.sigUpClicked.connect(self.changeFontSize)
        self.fontSizeSpinBox.sigDownClicked.connect(self.changeFontSize)
        
        quickSettingsLayout = QHBoxLayout()
        quickSettingsLayout.addWidget(annotLostObjsLabel)
        quickSettingsLayout.addWidget(self.annotLostObjsToggle)
        quickSettingsLayout.addSpacing(15)
        quickSettingsLayout.addWidget(showAllContourLabel)
        quickSettingsLayout.addWidget(self.showAllContoursToggle)
        quickSettingsLayout.addSpacing(15)
        quickSettingsLayout.addWidget(QLabel('Font size'))
        quickSettingsLayout.addWidget(self.fontSizeSpinBox)
        quickSettingsLayout.addStretch(1)
        
        annotLayout = QVBoxLayout()
        annotLayout.addWidget(self.annotOptionsCheckboxes[0]['container'])
        annotLayout.addLayout(quickSettingsLayout)
        
        self.annotationsGroupbox = QGroupBox('Annotations')
        self.annotationsGroupbox.setLayout(annotLayout)
        
        self.restoreViewerAnnotationOptions()
    
    def gui_createMeasurementsPanel(self):
        self.measurementsPlot = pg.PlotWidget()
        if self.invertBwAction.isChecked():
            self.measurementsPlot.setBackground(graphLayoutBkgrColor)
        else:
            self.measurementsPlot.setBackground(darkBkgrColor)
        self.measurementsPlot.setLabel('bottom', 'Frame n.')
        self.measurementsPlot.showGrid(x=True, y=True, alpha=0.3)
        self.measurementsLegend = self.measurementsPlot.addLegend()
        
        self.currentFrameLine = pg.InfiniteLine(
            angle=90, movable=True, 
            pen=pg.mkPen(color=(255, 0, 0), width=2),
            hoverPen=pg.mkPen(color=(255, 100, 100), width=3)
        )
        self.currentFrameLine.setToolTip(
            'Current frame. Drag the line to change frame.'
        )
        self.currentFrameLine.setZValue(10)
        self.measurementsPlot.addItem(self.currentFrameLine, ignoreBounds=True)
        
        self.columnsSearchLineEdit = QLineEdit()
        self.columnsSearchLineEdit.setPlaceholderText('Search column...')
        self.columnsSearchLineEdit.setClearButtonEnabled(True)
        
        self.columnsModel = widgets.MeasurementTreeModel(self)
        self.columnsSearchProxy = widgets.MeasurementTreeSearchProxy(self)
        self.columnsSearchProxy.setSourceModel(self.columnsModel)
        self.columnsSearchProxy.setDynamicSortFilter(True)
        self.columnsSearchProxy.sort(0, Qt.AscendingOrder)

        self.columnsListWidget = QTreeView()
        self.columnsListWidget.setHeaderHidden(True)
        self.columnsListWidget.setIndentation(6)
        self.columnsListWidget.setUniformRowHeights(True)
        self.columnsListWidget.setModel(self.columnsSearchProxy)
        self.columnsHighlightDelegate = _MeasurementColumnDelegate(
            self.columnsListWidget
        )
        self.columnsListWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.columnsListWidget.setItemDelegate(
            self.columnsHighlightDelegate
        )
        self.columnsListWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.columnsListWidget.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.columnsListWidget.setToolTip(
            'Click a column to plot/unplot it. Right-click on a column to '
            'change its color.'
        )
        # Custom plotting colors set by the user {column: QColor}
        self.measurementsColors = {}

        self.measurementsModeCombobox = QComboBox()
        self.measurementsModeCombobox.addItems(
            ['Time course', 'Measurement vs Measurement']
        )
        self.measurementsModeCombobox.setToolTip(
            'Time course: selected cell. Measurement vs Measurement: all cells and '
            'all frames of the current position. Hover for details; click '
            'a point to zoom to its cell.'
        )
        self.measurementsAxes = {'X': '', 'Y': ''}
        self._measurementsMode = 0
        self._timeCourseColumns = set()
        
        self.normalizeCheckbox = QCheckBox('Normalize measurements (0-1)')
        self.normalizeCheckbox.setToolTip(
            'Rescale each measurement to the range 0-1 to compare measurements with '
            'different scales'
        )
        self.refreshButton = widgets.reloadPushButton(' Reload table ')
        self.refreshButton.setToolTip(
            'Reload the acdc_output table from the file '
            '(e.g., after saving in the main GUI)'
        )
        self.processMeasurementsButton = widgets.setPushButton('Process measurements...')
        self.processMeasurementsButton.clicked.connect(self.processMeasurements)
        self.combineMeasurementsButton = widgets.calcPushButton(
            'Combine measurements...'
        )
        self.combineMeasurementsButton.setToolTip(
            'Add columns calculated from the current position. '
            'Use Export measurements to save the results.'
        )
        self.combineMeasurementsButton.clicked.connect(self.combineMeasurements)
        self.exportMeasurementsButton = widgets.savePushButton(
            'Export measurements...'
        )
        self.exportMeasurementsButton.clicked.connect(self.exportMeasurements)
        
        timeCourseLayout = QVBoxLayout()
        timeCourseLayout.setContentsMargins(0, 0, 0, 0)
        timeCourseLayout.addWidget(QLabel('Columns to plot:'))
        timeCourseLayout.addWidget(self.columnsSearchLineEdit)
        timeCourseLayout.addWidget(self.columnsListWidget)
        self.timeCourseColumnsWidget = QWidget()
        self.timeCourseColumnsWidget.setLayout(timeCourseLayout)

        columnsLayout = QVBoxLayout()
        columnsLayout.addWidget(self.measurementsModeCombobox)
        columnsLayout.addWidget(self.timeCourseColumnsWidget)
        columnsLayout.addWidget(self.normalizeCheckbox)
        columnsLayout.addWidget(self.processMeasurementsButton)
        columnsLayout.addWidget(self.combineMeasurementsButton)
        columnsLayout.addWidget(self.exportMeasurementsButton)
        columnsLayout.addWidget(self.refreshButton)
        columnsWidget = QWidget()
        columnsWidget.setLayout(columnsLayout)
        
        self.measurementsSplitter = QSplitter(Qt.Horizontal)
        self.measurementsSplitter.addWidget(self.measurementsPlot)
        self.measurementsSplitter.addWidget(columnsWidget)
        self.measurementsSplitter.setStretchFactor(0, 3)
        self.measurementsSplitter.setStretchFactor(1, 1)
        
        self.measurementsGroupbox = QGroupBox('Measurements')
        measurementsGroupboxLayout = QVBoxLayout()
        measurementsGroupboxLayout.addWidget(self.measurementsSplitter)
        self.measurementsGroupbox.setLayout(measurementsGroupboxLayout)
    
    def gui_createLayout(self):
        self.gui_createRegionPropsDockWidget()
        self.gui_createShowPropsButton()
        
        # Column 0 is used by the overlay labels gradient items 
        # (see `OverlayLabelsGuiElements.createOverlayLabelsItems`)
        self.mainLayout = QGridLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.addLayout(self.leftSideDocksLayout, 0, 1, 3, 1)
        self.mainLayout.addWidget(self.graphLayout, 0, 2)
        # Labels colormap gradient editor on the right of the image (as in 
        # guiWin)
        self.mainLayout.addWidget(self.labelsGrad, 0, 3)
        
        # Resizable bottom layout (as in guiWin)
        self.resizeBottomLayoutLine = widgets.VerticalResizeHline()
        self.mainLayout.addWidget(self.resizeBottomLayoutLine, 1, 2)
        self.resizeBottomLayoutLine.dragged.connect(
            self.resizeBottomLayoutLineDragged
        )
        self.resizeBottomLayoutLine.clicked.connect(
            self.resizeBottomLayoutLineClicked
        )
        self.resizeBottomLayoutLine.released.connect(
            self.resizeBottomLayoutLineReleased
        )
        
        self.img1BottomGroupbox = QWidget()
        bottomLayout = QVBoxLayout()
        bottomLayout.setContentsMargins(0, 0, 0, 0)
        bottomLayout.addWidget(self.annotationsGroupbox)
        bottomLayout.addWidget(self.navigationGroupbox)
        bottomLayout.addStretch(1)
        self.img1BottomGroupbox.setLayout(bottomLayout)
        
        self.bottomScrollArea = widgets.ScrollArea(resizeVerticalOnShow=True)
        self.bottomScrollArea.setWidgetResizable(True)
        self.bottomScrollArea.setWidget(self.img1BottomGroupbox)
        self.mainLayout.addWidget(self.bottomScrollArea, 2, 2)
        self.mainLayout.setRowStretch(2, 0)
        self.bottomScrollArea._resizeVertical()
        self.mainLayout.setRowStretch(0, 1)
        self.mainLayout.setColumnStretch(2, 1)
        imagePanel = QWidget()
        imagePanel.setLayout(self.mainLayout)
        
        self.mainSplitter = QSplitter(Qt.Horizontal)
        self.mainSplitter.addWidget(imagePanel)
        self.mainSplitter.addWidget(self.measurementsGroupbox)
        self.mainSplitter.setStretchFactor(0, 1)
        self.mainSplitter.setStretchFactor(1, 1)
        
        self.setCentralWidget(self.mainSplitter)
        self.resize(1400, 800)
    
    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        self.statusBarLabel = QLabel('')
        self.statusbar.addWidget(self.statusBarLabel)
        # Permanent widget with mouse hover info (as in guiWin)
        self.wcLabel = QLabel('')
        self.statusbar.addPermanentWidget(self.wcLabel)
    
    def gui_connectViewerToolsSignals(self):
        self.launch3dViewerButton.toggled.connect(self.launch3dViewerToggled)
        
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
        self.showPropsDockButton.sigClicked.connect(self.showPropsDockWidget)
    
    def gui_connectSignals(self):
        self.posCombobox.currentIndexChanged.connect(
            self.posComboboxIndexChanged
        )
        
        self.navigateScrollBar.connectEvents({
            'valueChanged': self.framesScrollBarValueChanged
        })
        self.navSpinBox.connectValueChanged(self.navigateSpinboxValueChanged)
        
        self.zSliceScrollBar.valueChanged.connect(self.update_z_slice)
        self.zProjComboBox.currentTextChanged.connect(self.updateZproj)
        self.switchPlaneCombobox.sigPlaneChanged.connect(
            self.switchViewedPlane
        )
        
        self.idScrollBar.valueChanged.connect(self.idScrollBarValueChanged)
        self.idSpinBox.editingFinished.connect(self.idSpinBoxEditingFinished)
        self.idSpinBox.sigUpClicked.connect(self.idSpinBoxEditingFinished)
        self.idSpinBox.sigDownClicked.connect(self.idSpinBoxEditingFinished)
        self.followIDcheckbox.toggled.connect(self.followIDtoggled)
        self.restrictFramesCheckbox.toggled.connect(self.restrictFramesToggled)
        self.rescalePlotCheckbox.toggled.connect(self.rescalePlotToggled)
        
        self.currentFrameLine.sigPositionChangeFinished.connect(
            self.currentFrameLineMoved
        )
        self.columnsSearchLineEdit.textChanged.connect(self.filterColumns)
        self.columnsModel.itemChanged.connect(self.measurementColumnChanged)
        self.columnsListWidget.customContextMenuRequested.connect(
            self.showColumnsContextMenu
        )
        self.normalizeCheckbox.toggled.connect(self.plotMeasurements)
        self.measurementsModeCombobox.currentIndexChanged.connect(
            self.measurementsModeChanged
        )
        self.refreshButton.clicked.connect(self.reloadButtonClicked)
    
    def posData(self):
        return self.data[self.pos_i]

    def onSearchId(self, ID):
        if not self.isDataLoaded:
            self.logger.warning(
                'Data not loaded yet. Key pressing events are not connected.'
            )
            return

        self.get_data()
        self.findID(ID=ID)
    
    def loadPosition(self, pos_i, frame_i=None, ID=None):
        self.pos_i = pos_i
        posData = self.posData()
        self._lab = None
        self._labKey = None
        
        if self.isSegm3D is None:
            self.isSegm3D = posData.segm_data.ndim == 4
        
        if frame_i is None:
            frame_i = max(getattr(posData, 'frame_i', 0), 0)
        self.frame_i = min(frame_i, posData.SizeT-1)
        posData.frame_i = self.frame_i
        
        self.posCombobox.blockSignals(True)
        self.posCombobox.setCurrentIndex(pos_i)
        self.posCombobox.blockSignals(False)
        isMultiPos = len(self.data) > 1
        self.posLabel.setVisible(isMultiPos)
        self.posCombobox.setVisible(isMultiPos)
        
        self.initFramesNavigation()
        self.initZsliceNavigation()
        allIDs = self.initPosAnnotData(posData)
        self._IDframesCache = {}
        self.refreshAcdcDf(plot=False)
        self.ensureTimepointPredictions(posData)
        self.initIDsNavigation(ID=ID)
        self.updateFramesRange()
        
        self.get_data()
        self.initContoursImage()
        self.gui_createAxesTextAnnotItems(allIDs)
        self.updateImages()
        self.ax1.autoRange()
        self.zoomToSelectedID()
        self.plotMeasurements()
    
    def initPosAnnotData(self, posData):
        """Initialize the per-frame stored data (`allData_li`) and the 
        attributes of `posData` required by the annotations as in 
        `gui.guiWin` and return the set of all IDs."""
        if getattr(posData, '_isViewerAnnotDataInit', False):
            return posData.allIDs
        
        posData.allData_li = [None]*posData.SizeT
        allIDs, posData = core.count_objects_and_init_rps(
            posData, self.logger.info
        )
        for i in range(posData.SizeT):
            if posData.allData_li[i] is None:
                posData.allData_li[i] = myutils.get_empty_stored_data_dict()
        posData.allIDs = set(allIDs)
        
        try:
            posData.loadTrackedLostCentroids()
        except Exception as err:
            pass
        tracked_lost_centroids = getattr(posData, 'tracked_lost_centroids', None)
        if not isinstance(tracked_lost_centroids, dict):
            posData.tracked_lost_centroids = {}
        for i in range(posData.SizeT):
            posData.tracked_lost_centroids.setdefault(i, set())
        
        posData.lost_IDs = []
        posData.new_IDs = []
        posData.old_IDs = []
        posData.trackedLostIDs = set()
        posData.whitelist = getattr(posData, 'whitelist', None)
        posData.cca_df = None
        SizeY, SizeX = posData.segm_data.shape[-2:]
        if not hasattr(posData, 'SizeY'):
            posData.SizeY = SizeY
        if not hasattr(posData, 'SizeX'):
            posData.SizeX = SizeX
        posData._isViewerAnnotDataInit = True
        return posData.allIDs
    
    def getFrameAcdcDf(self, frame_i):
        """Get the acdc_df of `frame_i` (index is Cell_ID) as stored in 
        `posData.allData_li` by `gui.guiWin`"""
        if self.acdc_df is None:
            return
        
        frames = self.acdc_df.index.get_level_values(0)
        if frame_i not in frames:
            return
        
        df = self.acdc_df.loc[frame_i].copy()
        if 'cell_cycle_stage' in df.columns:
            cca_cols = df.columns.intersection(self.cca_df_colnames)
            cca_df = df[cca_cols].dropna()
            if cca_df.empty:
                df = df.drop(columns=self.cca_df_colnames, errors='ignore')
            else:
                df = df.loc[cca_df.index]
                cols = df.columns.intersection(self.cca_df_int_cols)
                try:
                    df[cols] = df[cols].astype('Int64')
                except Exception as err:
                    pass
        return df
    
    def get_data(self):
        """Load data of the current frame into `posData` as in 
        `gui.guiWin.get_data` (read-only)"""
        posData = self.posData()
        frame_i = posData.frame_i
        storedData = posData.allData_li[frame_i]
        posData.lab = self.getLabels(frame_i)
        rp = storedData['regionprops']
        if rp is None:
            rp = self._acdcRegionProps(posData.lab, precache_centroids=False)
            storedData['regionprops'] = rp
        storedData['labels'] = posData.lab
        posData.rp = rp
        posData.IDs = rp.IDs
        
        df = self.getFrameAcdcDf(frame_i)
        storedData['acdc_df'] = df
        cca_df = None
        if df is not None and 'cell_cycle_stage' in df.columns:
            cca_cols = df.columns.intersection(self.cca_df_colnames)
            cca_df = df[cca_cols].copy().dropna()
        posData.cca_df = cca_df
    
    def initFramesNavigation(self):
        posData = self.posData()
        SizeT = posData.SizeT
        self.navigateScrollBar.setMaximum(SizeT)
        self.navigateScrollBar.setAbsoluteMaximum(SizeT)
        self.navSpinBox.setMaximum(SizeT)
        self.navSizeLabel.setText(f'/{SizeT}')
        self.navigateScrollBar.setDisabled(SizeT == 1)
        self.navSpinBox.setDisabled(SizeT == 1)
        self.resetNavigateFramesScrollbar(frame_i=self.frame_i)
        self.navSpinBox.setValueNoEmit(self.frame_i+1)
    
    def initZsliceNavigation(self):
        posData = self.posData()
        isZstack = posData.SizeZ > 1
        self.imgGrad.rescaleAcrossZstackAction.setDisabled(not isZstack)
        for widget in self.zSliceNavigationWidgets():
            widget.setVisible(isZstack)
        
        if not isZstack:
            return
        
        self.switchPlaneCombobox.blockSignals(True)
        self.switchPlaneCombobox.setPlane('xy')
        self.switchPlaneCombobox.blockSignals(False)
        # 2D segmentation masks cannot be displayed on planes other than xy
        self.switchPlaneCombobox.setDisabled(not self.isSegm3D)
        self.zProjComboBox.setDisabled(False)
        self.setZsliceNavigationRange(posData.SizeZ)
        
        z = posData.SizeZ // 2
        zProjHow = 'single z-slice'
        segmInfo_df = getattr(posData, 'segmInfo_df', None)
        idx = (getattr(posData, 'filename', None), self.frame_i)
        try:
            z = int(segmInfo_df.at[idx, 'z_slice_used_gui'])
            zProjHow = segmInfo_df.at[idx, 'which_z_proj_gui']
        except Exception as err:
            pass
        
        self.zProjComboBox.blockSignals(True)
        self.zProjComboBox.setCurrentText(zProjHow)
        self.zProjComboBox.blockSignals(False)
        self.setZSliceWidgetsEnabled(zProjHow)
        
        self.zSliceScrollBar.blockSignals(True)
        self.zSliceScrollBar.setSliderPosition(z)
        self.zSliceScrollBar.blockSignals(False)
        self.zSliceSpinbox.setValueNoEmit(z+1)
    
    def zSliceNavigationWidgets(self):
        return (
            self.zSliceCheckbox, self.zSliceSpinbox, self.SizeZlabel, 
            self.zSliceScrollBar, self.zProjComboBox, self.switchPlaneCombobox
        )
    
    def setZSliceWidgetsEnabled(self, how):
        enabled = how == 'single z-slice'
        self.zSliceScrollBar.setEnabled(enabled)
        self.zSliceSpinbox.setEnabled(enabled)
        self.zSliceCheckbox.setEnabled(enabled)
    
    def getAcdcDf(self, posData=None):
        """Get the acdc_df of the position indexed by (frame_i, Cell_ID)."""
        if posData is None:
            posData = self.posData()
        
        acdc_df = getattr(posData, 'acdc_df', None)
        if acdc_df is None or acdc_df.empty:
            return
        
        return self._standardizeAcdcDf(acdc_df).sort_index()
    
    def _standardizeAcdcDf(self, acdc_df):
        if list(acdc_df.index.names) == ['frame_i', 'Cell_ID']:
            return acdc_df
        acdc_df = acdc_df.reset_index()
        acdc_df = acdc_df.drop(columns='index', errors='ignore')
        return acdc_df.set_index(['frame_i', 'Cell_ID'])
    
    def reloadAcdcDfFile(self, posData=None):
        if posData is None:
            posData = self.posData()
        
        acdc_df_path = getattr(posData, 'acdc_output_csv_path', None)
        if not acdc_df_path or not os.path.exists(acdc_df_path):
            return
        
        posData.loadAcdcDf(acdc_df_path)
    
    def reloadButtonClicked(self, checked=False):
        self.reloadAcdcDfFile()
        self.refreshAcdcDf()

    def measurementProcessingWarning(self, title, text):
        self.logger.warning('%s: %s', title, text)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, title, text)

    def combineMeasurements(self, checked=False):
        if self.measurementProcessingThread is not None:
            self.measurementProcessingWarning(
                'Measurement processing running',
                'Wait for measurement processing to finish before combining measurements.'
            )
            return
        if self.acdc_df is None or self.acdc_df.empty:
            self.measurementProcessingWarning('No measurements', 'Load measurements first.')
            return
        posData = self.posData()
        source = self.acdc_df
        dialog = apps.CombineMetricsEquationDialog(
            posData.chNames, posData.SizeZ > 1, self.isSegm3D,
            parent=self, closeOnOk=False, acdc_df=source, saveToConfig=False
        )
        dialog.setToolTip(
            'Results are added to the current position in memory. '
            'Use Export measurements to save them.'
        )
        dialog.sigOk.connect(
            partial(self.applyCombinedMeasurements, posData=posData, source=source)
        )
        dialog.exec_()
        dialog.deleteLater()

    def applyCombinedMeasurements(self, dialog, *, posData, source):
        dialog.cancel = True
        if (
            self.measurementProcessingThread is not None
            or not any(pos is posData for pos in self.data)
            or self.posData() is not posData
            or self.acdc_df is not source
        ):
            self.measurementProcessingWarning(
                'Measurements changed',
                'The source table changed. Reopen Combine measurements and try again.'
            )
            return
        equations, _ = dialog.getEquationsDict()
        collisions = set(equations).intersection(source.columns)
        if collisions:
            self.measurementProcessingWarning(
                'Invalid measurement name',
                'Choose a name that does not overwrite existing columns: '
                + ', '.join(sorted(collisions))
            )
            return
        try:
            result = source.copy()
            for name, expression in equations.items():
                values = source.eval(expression)
                if not isinstance(values, pd.Series):
                    raise ValueError('The equation must return one value per row.')
                result[name] = values
        except Exception as err:
            self.logger.exception('Combining measurements failed')
            self.measurementProcessingWarning('Combining measurements failed', str(err))
            return
        posData.acdc_df = result
        self.acdc_df = result
        self.updateColumnsList()
        self.plotMeasurements()
        dialog.cancel = False
        dialog.close()
        self.statusBar().showMessage(
            'Combined measurements added. Use Export measurements to save them.',
            5000
        )

    def processMeasurements(self, checked=False):
        if self.measurementProcessingThread is not None:
            return
        if self.acdc_df is None or self.acdc_df.empty:
            self.measurementProcessingWarning('No measurements', 'Load measurements first.')
            return
        columns = [
            col for col in self.acdc_df.select_dtypes(include='number').columns
            if col not in INDEX_COLUMNS
        ]
        if not columns:
            self.measurementProcessingWarning(
                'No measurements', 'The table has no numeric measurements to process.'
            )
            return
        dialog = apps.MeasurementProcessingRecipeDialog(
            columns, parent=self, acdc_df=self.acdc_df,
            allChNames=self.posData().chNames
        )
        dialog.exec_()
        if dialog.cancel:
            return       

        self.startMeasurementsProcessing(
            dialog.selectedChannels, dialog.selectedRecipe, dialog.saveNames
        )

    def startMeasurementsProcessing(self, columns, recipe, saveNames):
        posData = self.posData()
        self._measurementProcessingTarget = (
            posData, self.acdc_df, getattr(posData, 'acdc_df', None)
        )
        self.measurementProcessingThread = QThread(self)
        self.measurementProcessingWorker = workers.MeasurementProcessingWorkerGUI(
            QMutex(), QWaitCondition()
        )
        worker = self.measurementProcessingWorker
        worker.setupJob(
            df=self.acdc_df.copy(),
            cols=columns,
            recipe=recipe,
            new_colnames=saveNames
        )
        worker.moveToThread(self.measurementProcessingThread)
        self.measurementProcessingThread.started.connect(worker.runJob)
        worker.sigDone.connect(self.measurementProcessingDone)
        worker.signals.critical.connect(self.measurementProcessingFailed)
        worker.signals.finished.connect(self.measurementProcessingThread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        self.measurementProcessingThread.finished.connect(self.measurementProcessingFinished)
        self.processMeasurementsButton.setEnabled(False)
        self.combineMeasurementsButton.setEnabled(False)
        self.refreshButton.setEnabled(False)
        self.measurementProcessingThread.start()

    def measurementProcessingDone(self, processed):
        posData, source, original = self._measurementProcessingTarget
        if (not any(pos is posData for pos in self.data)
                or getattr(posData, 'acdc_df', None) is not original
                or (self.posData() is posData
                    and (self.acdc_df is None or not self.acdc_df.sort_index().equals(
                        source.sort_index()
                    )))):
            self.measurementProcessingWarning(
                'Measurements changed',
                'The source table changed during processing. Results were not applied.'
            )
            return

        posData.acdc_df = processed
        if self.posData() is posData:
            self.acdc_df = processed
            self.updateColumnsList()
            self.plotMeasurements()

    def measurementProcessingFailed(self, out):
        worker, error = out
        self.measurementProcessingWarning('Measurement processing failed', str(error))

    def measurementProcessingFinished(self):
        self.measurementProcessingThread.deleteLater()
        self.measurementProcessingThread = None
        self.measurementProcessingWorker = None
        self.processMeasurementsButton.setEnabled(True)
        self.combineMeasurementsButton.setEnabled(True)
        self.refreshButton.setEnabled(True)

    def exportMeasurements(self, checked=False):
        if self.acdc_df is None or self.acdc_df.empty:
            self.measurementProcessingWarning('No measurements', 'Load measurements first.')
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, 'Export measurements', 'processed_measurements.csv', 'CSV (*.csv)'
        )
        if not filepath:
            return
        try:
            export_df = self.acdc_df.copy()
            for column, values in self.timepointsColumns(export_df).items():
                export_df[column] = values
            export_df.to_csv(filepath)
        except (OSError, ValueError) as error:
            self.measurementProcessingWarning('Export failed', str(error))
            return
        self.statusBarLabel.setText(f'Measurements exported to {filepath}')
    
    def refreshAcdcDf(self, plot=True):
        self.acdc_df = self.getAcdcDf()
        self._IDframesCache = {}
        self.initLineageTree()
        self.updateColumnsList()
        if not plot:
            return
        self.initIDsNavigation(ID=self.selectedID)
        self.updateFramesRange()
        self.updateImages()
        self.plotMeasurements()
    
    def onMotherDaughterLine(self, checked, checkbox, ax=0):
        super().onMotherDaughterLine(checked, checkbox, ax=ax)
        if checked and self.initLineageTreeFrameData():
            self.drawAllLineageTreeLines()
    
    def initLineageTree(self):
        """Stand-in for the `normal_division_lineage_tree` of `gui.guiWin` 
        (only `frames_for_dfs` is required by 
        `AnnotationsGuiElements.drawAllLineageTreeLines`). Lineage tree lines 
        are drawn from the saved `parent_ID_tree` column of the acdc_df."""
        self.lineage_tree = None
        if self.acdc_df is None:
            return
        
        if 'parent_ID_tree' not in self.acdc_df.columns:
            return
        
        frames = self.acdc_df.index.get_level_values(0).unique()
        self.lineage_tree = SimpleNamespace(frames_for_dfs=set(frames))
    
    def _getLineageTreeFrameDf(self, frame_i, rp, prev_IDs=None):
        df = self.getFrameAcdcDf(frame_i)
        if df is None:
            return
        
        if 'emerg_frame_i' not in df.columns:
            df['emerg_frame_i'] = -1
        
        # Parents that cannot be drawn are unknown (-1) as in `gui.guiWin`
        parent_IDs = df['parent_ID_tree'].fillna(-1).astype(int)
        isDrawable = df.index.isin(rp.IDs)
        if prev_IDs is not None:
            isDrawable = isDrawable & parent_IDs.isin(prev_IDs).to_numpy()
        df['parent_ID_tree'] = parent_IDs.where(isDrawable, -1)
        return df
    
    def initLineageTreeFrameData(self):
        """Store the acdc_df of current and previous frame required by 
        `AnnotationsGuiElements.drawAllLineageTreeLines` in 
        `posData.allData_li`. Returns False if lineage tree lines cannot be 
        drawn."""
        if self.lineage_tree is None:
            return False
        
        posData = self.posData()
        frame_i = posData.frame_i
        if frame_i == 0:
            return True
        
        prevData = posData.allData_li[frame_i-1]
        prev_rp = prevData['regionprops']
        if prev_rp is None:
            prev_lab = self.getLabels(frame_i-1)
            prev_rp = self._acdcRegionProps(prev_lab, precache_centroids=False)
            prevData['regionprops'] = prev_rp
        
        prev_df = self._getLineageTreeFrameDf(frame_i-1, prev_rp)
        df = self._getLineageTreeFrameDf(
            frame_i, posData.rp, prev_IDs=prev_rp.IDs
        )
        if prev_df is None or df is None:
            return False
        
        prevData['acdc_df'] = prev_df
        posData.allData_li[frame_i]['acdc_df'] = df
        return True
    
    def updateColumnsList(self):
        checkedColumns = set(self.checkedColumns())
        isFirstInit = self.columnsModel.rowCount() == 0
        
        self.columnsModel.blockSignals(True)
        self.columnsModel.clear()
        if self.acdc_df is None:
            self.columnsModel.blockSignals(False)
            self.updateSignalAxes([])
            return
        
        numericColumns = [
            col for col in self.acdc_df.select_dtypes(
                include=['number', 'bool']
            ).columns
            if col not in INDEX_COLUMNS
        ]
        if isFirstInit and self.measurementsModeCombobox.currentIndex() == 0:
            for col in DEFAULT_MEASUREMENTS_COLUMNS:
                if col in numericColumns:
                    checkedColumns = {col}
                    break
            else:
                checkedColumns = set(numericColumns[:1])
        
        channels = getattr(self.posData(), 'chNames', ()) if self.data else ()
        self.columnsModel.populate(
            self.acdc_df[numericColumns], channels, checkable=True
        )
        for item in self.columnsModel.measurementItems():
            col = item.text()
            if col in checkedColumns:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
        self.columnsModel.blockSignals(False)
        self.columnsListWidget.expandAll()
        self.updateSignalAxes(numericColumns)
        self.filterColumns(self.columnsSearchLineEdit.text())

    def updateSignalAxes(self, columns):
        for axis, col in self.measurementsAxes.items():
            if col not in columns:
                self.measurementsAxes[axis] = ''
        self._timeCourseColumns.intersection_update(columns)
        self.updateColumnsIcons()
    
    def checkedColumns(self):
        columns = []
        for item in self.columnsModel.measurementItems():
            if item.checkState() == Qt.Checked:
                columns.append(item.text())
        return columns
    
    def filterColumns(self, text):
        self.columnsSearchProxy.setFilterText(text)
        self.columnsHighlightDelegate.setFilterText(text)
        if text.strip():
            self.columnsListWidget.expandAll()
        self.columnsListWidget.viewport().update()
    
    def getAllIDs(self):
        if self.acdc_df is not None:
            IDs = self.acdc_df.index.get_level_values('Cell_ID').unique()
            return sorted([int(ID) for ID in IDs if ID > 0])
        
        posData = self.posData()
        IDs = set()
        for frame_i in range(posData.SizeT):
            IDs.update(np.unique(self.getLabels(frame_i)))
        IDs.discard(0)
        return sorted([int(ID) for ID in IDs])
    
    def initIDsNavigation(self, ID=None):
        self.IDs = self.getAllIDs()
        numIDs = len(self.IDs)
        self.numIDsLabel.setText(f'({numIDs} cells)')
        hasIDs = numIDs > 0
        self.idScrollBar.setEnabled(hasIDs)
        self.idSpinBox.setEnabled(hasIDs)
        if not hasIDs:
            self.selectedID = None
            return
        
        if ID not in self.IDs:
            # Select first ID present in current frame (if any)
            frameIDs = set(np.unique(self.getLabels()))
            ID = next((_ID for _ID in self.IDs if _ID in frameIDs), self.IDs[0])
        
        self.selectedID = ID
        self.idScrollBar.blockSignals(True)
        self.idScrollBar.setMaximum(numIDs-1)
        self.idScrollBar.setValue(self.IDs.index(ID))
        self.idScrollBar.blockSignals(False)
        self.idSpinBox.setMaximum(max(self.IDs))
        self.idSpinBox.setValueNoEmit(ID)
    
    def getLabels(self, frame_i=None):
        """Get labels of `frame_i` (default current frame)."""
        posData = self.posData()
        if frame_i is None:
            frame_i = self.frame_i
        
        key = (self.pos_i, frame_i)
        if key == self._labKey:
            return self._lab
        
        lab = posData.segm_data[frame_i]
        
        if frame_i == self.frame_i:
            self._lab, self._labKey = lab, key
        return lab
    
    def getImage(self, frame_i=None, raw=False):
        posData = self.posData()
        if frame_i is None:
            frame_i = self.frame_i
        img = posData.img_data[frame_i]
        if posData.SizeZ > 1:
            img = self.get_2Dimg_from_3D(img)
        return img
    
    def get_2Dimg_from_3D(self, imgData, isLayer0=True, frame_i=None):
        img = self.get_2Dimg_from_switched_plane(imgData)
        if img is not None:
            return img
        
        zProjHow = self.zProjComboBox.currentText()
        z = self.zSliceScrollBar.sliderPosition()
        if not isLayer0:
            zProjHowOverlay = self.zProjOverlay_CB.currentText()
            if zProjHowOverlay != 'same as above':
                zProjHow = zProjHowOverlay
                z = self.zSliceOverlay_SB.sliderPosition()
        return self.zProjectImage(imgData, zProjHow, z)
    
    def getHighlightedIDfromPropsDock(self):
        if not self.propsDockWidget.isVisible():
            return 0
        guiTabControl = self.guiTabControl
        doHighlight = (
            guiTabControl.highlightCheckbox.isChecked()
            or guiTabControl.highlightSearchedCheckbox.isChecked()
        )
        if not doHighlight:
            return 0
        return guiTabControl.propsQGBox.idSB.value()
    
    def updateImages(self):
        """Update image and annotations as `gui.guiWin.updateAllImages` in 
        'Viewer' mode"""
        if not self.data:
            return
        
        self.get_data()
        self.highlightedID = (
            self.getHighlightedIDfromPropsDock() or self.selectedID or 0
        )
        self.clearAx1Items()
        
        self.rescaleIntensitiesLut(setImage=False)
        self.setImageImg1()
        
        posData = self.posData()
        self.currentLab2D = self.get_2Dlab(posData.lab, force_z=False)
        self.setOverlayImages()
        self.setOverlayLabelsItems()
        self.setOverlaySegmMasks()
        self.updateViewerWindow()
        
        self.setAllTextAnnotations()
        self.setAllContoursImages(compute=False)
        self.drawAllMothBudLines()
        if self.initLineageTreeFrameData():
            self.drawAllLineageTreeLines()
        
        if self.ccaTableWin is not None:
            zoomIDs = self.getZoomIDs()
            self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)
        
        self.highlightSelectedID()
        self.updateTimestampFrame()
        self.annotateAllObjectTracks()
        posData.visited = True
        self.updateObjectCounts()
        self.updateIDnotPresentLabel()
    
    def clearAx1Items(self):
        self.labelsLayerImg1.clear()
        self.labelsLayerImg1.setOpacity(self.imgGrad.labelsAlphaSlider.value())
        self.highLightIDLayerImg1.clear()
        self.searchedIDitemLeft.clear()
        self.ax1_contoursImageItem.clear()
        if 0 in self.textAnnot:
            self.textAnnot[0].clear()
        self.ax1_newMothBudLinesItem.setData([], [])
        self.ax1_oldMothBudLinesItem.setData([], [])
        self.ax1_movementAgainstPrevLinesItem.setData([], [])
        self.ax1_movementAgainstPrevScatterItem.setData([], [])
        self.clearObjTracks()
    
    def highlightSelectedID(self):
        """Highlight the selected ID as the searched ID in `gui.guiWin`"""
        ID = self.highlightedID
        if not ID:
            return
        
        posData = self.posData()
        obj = posData.rp.get_obj_from_ID(ID, warn=False)
        if obj is None:
            return
        
        if not self.isObjInCurrSlice(obj.bbox):
            return
        
        display_rp = self.get2DRP()
        obj = display_rp.get_obj_from_ID(ID, warn=False)
        if obj is None:
            return
        contours = self.getObjContours(
            obj,
            all_external=True,
            include_internal=self.showAllContoursToggle.isChecked()
        )
        for cont in contours:
            self.searchedIDitemLeft.addPoints(cont[:,0]+0.5, cont[:,1]+0.5)
    
    def updateAllImages(self, *args, **kwargs):
        self.updateImages()
    
    def setImageImg1(self):
        posData = self.posData()
        self.img1.setCurrentPosIndex(self.pos_i)
        self.img1.setCurrentFrameIndex(posData.frame_i)
        if posData.SizeZ > 1:
            zProjHow = self.zProjComboBox.currentText()
            if self.switchPlaneCombobox.depthAxes() != 'z':
                # Pre-computed min/max values refer to xy planes only
                z = None
            elif zProjHow == 'single z-slice':
                z = self.zSliceScrollBar.sliderPosition()
            else:
                z = zProjHow
            
            self.img1.setCurrentZsliceIndex(z)
        
        self.img1.setImage(self.getImage())
    
    def updateIDnotPresentLabel(self):
        if self.selectedID is None:
            self.idNotPresentLabel.setText('')
            return
        
        isPresent = np.any(self.getLabels() == self.selectedID)
        if isPresent:
            self.idNotPresentLabel.setText('')
        else:
            self.idNotPresentLabel.setText(
                f'<i>Cell ID {self.selectedID} is not present in frame '
                f'n. {self.frame_i+1}</i>'
            )
    
    def posComboboxIndexChanged(self, pos_i):
        self.loadPosition(pos_i, ID=self.selectedID)
    
    def navigateSpinboxValueChanged(self, value):
        self.navigateScrollBar.setSliderPosition(value)
    
    def framesScrollBarValueChanged(self, frame_n):
        self.setFrame(frame_n-1)
    
    def setFrame(self, frame_i):
        posData = self.posData()
        first_i, last_i = self.framesRange()
        frame_i = int(np.clip(frame_i, first_i, last_i))
        self.navSpinBox.setValueNoEmit(frame_i+1)
        if self.navigateScrollBar.sliderPosition() != frame_i+1:
            self.resetNavigateFramesScrollbar(frame_i=frame_i)
        
        if frame_i == self.frame_i:
            return
        
        self.frame_i = frame_i
        posData.frame_i = frame_i
        self.updateImages()
        self.zoomToSelectedID()
        self.updateCurrentFrameLine()
    
    def update_z_slice(self, z):
        self.zSliceSpinbox.setValueNoEmit(z+1)
        self.updateImages()
    
    def updateZproj(self, how):
        self.setZSliceWidgetsEnabled(how)
        self.updateImages()
    
    def switchViewedPlane(self, previousPlane, currentPlane):
        posData = self.posData()
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes == 'z':
            self.zProjComboBox.setDisabled(False)
            depthSize = posData.SizeZ
        else:
            self.zProjComboBox.blockSignals(True)
            self.zProjComboBox.setCurrentText('single z-slice')
            self.zProjComboBox.blockSignals(False)
            self.setZSliceWidgetsEnabled('single z-slice')
            self.zProjComboBox.setDisabled(True)
            SizeY, SizeX = posData.img_data.shape[-2:]
            depthSize = SizeY if depthAxes == 'y' else SizeX
        
        self.zSliceScrollBar.blockSignals(True)
        self.setZsliceNavigationRange(depthSize)
        self.zSliceScrollBar.setSliderPosition(depthSize // 2)
        self.zSliceScrollBar.blockSignals(False)
        self.zSliceSpinbox.setValueNoEmit(depthSize // 2 + 1)
        
        self.initContoursImage()
        self.updateImages()
        self.ax1.autoRange()
        self.zoomToSelectedID()
    
    def idScrollBarValueChanged(self, idx):
        if not self.IDs:
            return
        ID = self.IDs[idx]
        self.idSpinBox.setValueNoEmit(ID)
        self.setSelectedID(ID)
        
        if self.rescalePlotCheckbox.isChecked():
            self.measurementsPlot.autoRange()
            
    def rescalePlotToggled(self, checked):
        if checked:
            self.measurementsPlot.autoRange()
    
    def idSpinBoxEditingFinished(self):
        ID = self.idSpinBox.value()
        if ID == self.selectedID:
            return
        
        if ID not in self.IDs:
            self.statusBar().showMessage(f'Cell ID {ID} does not exist.', 3000)
            self.idSpinBox.setValueNoEmit(self.selectedID)
            return
        
        self.idScrollBar.setValue(self.IDs.index(ID))
    
    def setSelectedID(self, ID):
        self.selectedID = ID
        self.updateFramesRange()
        self.zoomToSelectedID(goToZslice=True)
        self.updateImages()
        self.plotMeasurements()
    
    def getIDframes(self, ID):
        """Get the sorted frame indices where `ID` is present."""
        key = (self.pos_i, ID)
        cached = self._IDframesCache.get(key)
        if cached is not None:
            return cached
        
        frames = None
        if self.acdc_df is not None:
            try:
                df_ID = self.acdc_df.xs(ID, level='Cell_ID')
                frames = df_ID.index.get_level_values(0).unique().to_numpy()
            except KeyError:
                frames = None
        
        if frames is None or len(frames) == 0:
            segm_data = self.posData().segm_data
            frames = np.array([
                frame_i for frame_i in range(len(segm_data))
                if np.any(segm_data[frame_i] == ID)
            ], dtype=int)
        
        frames = np.sort(frames.astype(int))
        self._IDframesCache[key] = frames
        return frames
    
    def framesRange(self):
        """Get the (first, last) frame index the user can navigate to."""
        SizeT = self.posData().SizeT
        if not self.restrictFramesCheckbox.isChecked():
            return 0, SizeT-1
        
        if self.selectedID is None:
            return 0, SizeT-1
        
        frames = self.getIDframes(self.selectedID)
        frames = frames[(frames >= 0) & (frames < SizeT)]
        if len(frames) == 0:
            return 0, SizeT-1
        
        return int(frames[0]), int(frames[-1])
    
    def updateFramesRange(self):
        """Restrict the frames navigation widgets to `framesRange` and move 
        the current frame into the range if it is outside. 
        
        Returns True if the current frame changed (images not updated).
        """
        posData = self.posData()
        first_i, last_i = self.framesRange()
        
        self.navigateScrollBar.blockSignals(True)
        self.navigateScrollBar.setMinimum(first_i+1)
        self.navigateScrollBar.setMaximum(last_i+1)
        self.navigateScrollBar.blockSignals(False)
        self.navSpinBox.blockSignals(True)
        self.navSpinBox.setMinimum(first_i+1)
        self.navSpinBox.setMaximum(last_i+1)
        self.navSpinBox.blockSignals(False)
        
        isRangeZero = first_i == last_i
        self.navigateScrollBar.setDisabled(isRangeZero)
        self.navSpinBox.setDisabled(isRangeZero)
        
        frame_i = int(np.clip(self.frame_i, first_i, last_i))
        isFrameChanged = frame_i != self.frame_i
        self.frame_i = frame_i
        posData.frame_i = frame_i
        self.navSpinBox.setValueNoEmit(frame_i+1)
        self.resetNavigateFramesScrollbar(frame_i=frame_i)
        if isFrameChanged:
            self.updateCurrentFrameLine()
        return isFrameChanged
    
    def restrictFramesToggled(self, checked):
        if not self.data:
            return
        isFrameChanged = self.updateFramesRange()
        if not isFrameChanged:
            return
        self.updateImages()
        if self.followIDcheckbox.isChecked():
            self.zoomToSelectedID()
            
        self._set_setting_value('restrictFrames', checked)
    
    def followIDtoggled(self, checked):
        if checked:
            self.zoomToSelectedID()
            
        self._set_setting_value('followID', checked)
    
    def goToZsliceID(self, ID):
        """Move the z-slice scrollbar to the slice of `ID` (along the 
        current depth axis) if the ID is not visible in the current slice.
        """
        posData = self.posData()
        if not self.isSegm3D or posData.SizeZ == 1:
            return
        
        if self.zProjComboBox.currentText() != 'single z-slice':
            return
        
        lab = self.getLabels()
        lab2D = self.get_2Dlab(lab, force_z=False)
        if np.any(lab2D == ID):
            return
        
        objMask = (lab == ID).astype(np.uint8)
        if not np.any(objMask):
            return
        
        obj = skimage.measure.regionprops(objMask)[0]
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes == 'z':
            current_z = self.zSliceScrollBar.sliderPosition()
            z = core.nearest_nonzero_z_idx_from_z_centroid(
                obj, current_z=current_z
            )
        elif depthAxes == 'y':
            z = round(obj.centroid[1])
        else:
            z = round(obj.centroid[2])
        
        self.zSliceScrollBar.setSliderPosition(int(z))
    
    def zoomToSelectedID(self, goToZslice=False, force=False):
        ID = self.selectedID
        if ID is None:
            return
        
        if not self.followIDcheckbox.isChecked() and not force:
            return
        
        if goToZslice or self.followIDcheckbox.isChecked():
            self.goToZsliceID(ID)
        
        lab = self.getLabels()
        lab2D = self.get_2Dlab(lab, force_z=False)
        objMask = None
        if np.any(lab2D == ID):
            objMask = lab2D == ID
        elif self.switchPlaneCombobox.depthAxes() == 'z':
            # ID not visible on current slice --> zoom on its projection
            objMask = lab == ID
            if objMask.ndim == 3:
                objMask = objMask.any(axis=0)
        
        if objMask is None or not np.any(objMask):
            return
        
        rows, cols = np.nonzero(objMask)
        obj_bbox = (rows.min(), cols.min(), rows.max()+1, cols.max()+1)
        self.setViewRangeToObjBbox(obj_bbox, padding=10)
    
    def updateCurrentFrameLine(self):
        self.currentFrameLine.blockSignals(True)
        self.currentFrameLine.setValue(self.frame_i+1)
        self.currentFrameLine.blockSignals(False)
    
    def currentFrameLineMoved(self, line):
        frame_n = int(round(line.value()))
        self.setFrame(frame_n-1)
        self.updateCurrentFrameLine()
    
    def timepointIdentifier(self, uid):
        for identifier in self.timepointIdentifiers:
            if identifier.uid == uid:
                return identifier
    
    def positionTimepoints(self, posData=None):
        """Time points of a position as {uid: {Cell_ID: sorted [frame_i]}}"""
        if posData is None:
            posData = self.posData()
        timepoints = getattr(posData, 'timepoints', None)
        if timepoints is None:
            timepoints = defaultdict(dict)
            posData.timepoints = timepoints
            posData.timepointsPredicted = set()
            # Cells with manual edits {uid: set of Cell_IDs}
            posData.timepointsEdited = defaultdict(set)
            # Per-cell constants values {name: {Cell_ID: value}}
            posData.cellConstants = {}
            posData.cellConstantsSource = None
        return timepoints
    
    def positionCellConstants(self, posData=None, acdc_df=None):
        """Values of the per-cell constants of a position 
        {name: {Cell_ID: value}} (or a Series indexed by (frame_i, Cell_ID) 
        for per-frame constants), recomputed if the constants or the 
        measurements table changed."""
        if posData is None:
            posData = self.posData()
        self.positionTimepoints(posData)
        if not self.cellConstants:
            posData.cellConstants = {}
            posData.cellConstantsSource = None
            return posData.cellConstants
        if acdc_df is None:
            acdc_df = (
                self.acdc_df if posData is self.posData() 
                else self.getAcdcDf(posData)
            )
        constants = tuple(
            tuple(timepoint_predictors.normalize_constant(constant).items())
            for constant in self.cellConstants
        )
        source = posData.cellConstantsSource
        if source is not None and source[0] is acdc_df and source[1] == constants:
            return posData.cellConstants
        try:
            values = timepoint_predictors.compute_cell_constants(
                acdc_df, self.cellConstants, SizeT=posData.SizeT
            )
        except Exception as err:
            self.logger.exception('Computing per-cell constants failed')
            self.measurementProcessingWarning(
                'Per-cell constants failed', 
                f'Computing the per-cell constants failed:\n\n{err}'
            )
            values = {}
        posData.cellConstants = values
        # Keep a reference to the table to detect changes by identity
        posData.cellConstantsSource = (acdc_df, constants)
        return values
    
    def acdcDfWithConstants(self, acdc_df=None, posData=None):
        """Copy of `acdc_df` (default current table) with the per-cell 
        constants added as columns."""
        if not self.data:
            return acdc_df if acdc_df is not None else self.acdc_df
        if posData is None:
            posData = self.posData()
        if acdc_df is None and posData is self.posData():
            acdc_df = self.acdc_df
        if acdc_df is None:
            return None
        values = self.positionCellConstants(posData, acdc_df=acdc_df)
        return timepoint_predictors.add_constants_columns(acdc_df, values)
    
    def setCellConstants(self, constants):
        timepoint_predictors.validate_constants(constants)
        self.cellConstants = [
            timepoint_predictors.normalize_constant(constant)
            for constant in constants
        ]
        for posData in self.data or []:
            if getattr(posData, 'timepoints', None) is not None:
                posData.cellConstants = {}
                posData.cellConstantsSource = None
    
    def askReevaluateAfterConstantsChanged(self):
        txt = html_utils.paragraph(
            'The per-cell constants changed.<br><br>'
            'Do you want to <b>re-evaluate the time point predictors</b> on '
            'all cells now?'
        )
        msg = widgets.myMessageBox(wrapText=False)
        _, yesButton = msg.question(
            self, 'Re-evaluate predictors?', txt, 
            buttonsTexts=('No', 'Yes, re-evaluate')
        )
        return msg.clickedButton == yesButton
    
    def editCellConstants(self, checked=False):
        posData = self.posData() if self.data else None
        dialog = apps.CellConstantsDialog(
            constants=self.cellConstants, acdc_df=self.acdc_df,
            SizeT=getattr(posData, 'SizeT', None), parent=self
        )
        dialog.exec_()
        if dialog.cancel or dialog.constants is None:
            return False
        newConstants = [
            timepoint_predictors.normalize_constant(constant) 
            for constant in dialog.constants
        ]
        if newConstants == self.cellConstants:
            return True
        self.setCellConstants(newConstants)
        if posData is not None:
            values = self.positionCellConstants(posData)
            self.statusBar().showMessage(
                f'Computed {len(values)} per-cell constant(s).', 5000
            )
        if posData is not None and self.predictorTimepointUids():
            # Other positions are re-predicted when they are loaded
            for otherPosData in self.data:
                if otherPosData is posData:
                    continue
                if getattr(otherPosData, 'timepoints', None) is not None:
                    otherPosData.timepointsPredicted.difference_update(
                        self.predictorTimepointUids()
                    )
                    positionPath = os.path.normcase(
                        os.path.abspath(otherPosData.pos_path)
                    )
                    for uid in self.predictorTimepointUids():
                        self.timepointAutoPredictions.pop((positionPath, uid), None)
            if self.askReevaluateAfterConstantsChanged():
                self.reevaluateAllTimepoints()
        return True
    
    def editedTimepointCells(self, posData=None):
        self.positionTimepoints(posData)
        if posData is None:
            posData = self.posData()
        return posData.timepointsEdited
    
    def markTimepointEdited(self, uid, ID):
        self.editedTimepointCells()[uid].add(ID)
    
    def reservedShortcuts(self, excludeUid=None):
        """Key sequences already used in the viewer {sequence: used by}"""
        reserved = {}
        for name in ('Left', 'Right', 'Up', 'Down', 'Space', 'Z', 'Esc'):
            reserved[name] = 'Viewer navigation'
        for n in range(1, 10):
            reserved[str(n)] = 'Overlay channels toggle'
        zoomOutKey = QKeySequence(self.zoomOutKeyValue).toString()
        reserved[zoomOutKey] = 'Zoom out'
        timepointActions = {
            action: uid for uid, action in self.timepointActions.items()
        }
        widgets_ = self.findChildren(QAction) + self.findChildren(QAbstractButton)
        for widget in widgets_:
            if timepointActions.get(widget, -1) == excludeUid:
                continue
            shortcut = widget.shortcut().toString()
            if not shortcut:
                continue
            text = widget.text() or widget.toolTip().split('\n')[0]
            reserved[shortcut] = text.replace('&', '') or shortcut
        for name, widget in self.widgetsWithShortcut.items():
            shortcut = getattr(widget, 'keyPressShortcut', None)
            if isinstance(shortcut, QKeySequence) and not shortcut.isEmpty():
                reserved[shortcut.toString()] = name
        return reserved
    
    def defaultTimepointIdentifier(self):
        names = {identifier.name for identifier in self.timepointIdentifiers}
        n = len(self.timepointIdentifiers) + 1
        while f'Time point {n}' in names:
            n += 1
        palette = timepoint_predictors.DEFAULT_COLORS
        usedColors = {
            tuple(identifier.color) for identifier in self.timepointIdentifiers
        }
        color = next(
            (c for c in palette if c not in usedColors), 
            palette[len(self.timepointIdentifiers) % len(palette)]
        )
        reserved = self.reservedShortcuts()
        shortcut = next(
            (key for key in timepoint_predictors.DEFAULT_SHORTCUTS 
             if key not in reserved), ''
        )
        return TimepointIdentifier(
            name=f'Time point {n}', color=color, shortcut=shortcut
        )
    
    def execTimepointIdentifierDialog(self, identifier, title):
        posData = self.posData() if self.data else None
        names = {
            other.name for other in self.timepointIdentifiers
            if other.uid != identifier.uid
        }
        dialog = apps.TimepointIdentifierDialog(
            identifier.name, identifier.color, identifier.shortcut,
            predictor=identifier.predictor, existingNames=names,
            reservedShortcuts=self.reservedShortcuts(excludeUid=identifier.uid),
            acdc_df=self.acdcDfWithConstants(), 
            SizeT=getattr(posData, 'SizeT', None),
            allChNames=getattr(posData, 'chNames', []),
            isZstack=getattr(posData, 'SizeZ', 1) > 1,
            isSegm3D=bool(self.isSegm3D), title=title, parent=self
        )
        dialog.autoApplyCheckBox.setChecked(identifier.autoApply)
        dialog.exec_()
        if dialog.cancel:
            return
        return dialog.identifier
    
    def addTimepointIdentifier(self, checked=False):
        identifier = self.defaultTimepointIdentifier()
        values = self.execTimepointIdentifierDialog(
            identifier, 'Add time point identifier'
        )
        if values is None:
            return
        identifier = TimepointIdentifier(**values)
        self.addTimepointIdentifierItem(identifier)
        self.runTimepointPredictor(identifier.uid)
        self.updateTimepointLines()
        return identifier
    
    def addTimepointIdentifierItem(self, identifier):
        self.timepointIdentifiers.append(identifier)
        action = QAction(self)
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        action.triggered.connect(
            partial(self.timepointActionTriggered, identifier.uid)
        )
        self.timepointActions[identifier.uid] = action
        self.timepointsToolBar.addAction(action)
        
        # Shift+shortcut re-evaluates the predictor for the selected cell.
        autoAction = QAction(self)
        autoAction.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        autoAction.triggered.connect(
            lambda checked=False: self.runTimepointPredictor(
                identifier.uid, force=True, autoAction_cb=True, replot=True
            )
        )
        self.addAction(autoAction)
        self.timepointAutoActions[identifier.uid] = autoAction
        
        self.updateTimepointAction(identifier)
        self.updateReevaluateTimepointsActions()
        button = self.timepointsToolBar.widgetForAction(action)
        if button is not None:
            button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            button.customContextMenuRequested.connect(
                partial(self.showTimepointActionContextMenu, identifier.uid)
            )
        return action
    
    def updateTimepointAction(self, identifier):
        action = self.timepointActions[identifier.uid]
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(pg.mkPen(color=identifier.color, width=4))
        painter.drawLine(16, 2, 16, 30)
        painter.setBrush(pg.mkBrush(identifier.color))
        painter.drawEllipse(9, 9, 14, 14)
        painter.end()
        action.setIcon(QIcon(pixmap))
        action.setText(identifier.name)
        action.setShortcut(QKeySequence(identifier.shortcut))
        shortcutText = (
            f'Shortcut: "{identifier.shortcut}", Auto: "Shift+{identifier.shortcut}"' 
            if identifier.shortcut else 'No shortcut'
        )
        predictorText = timepoint_predictors.predictor_description(
            identifier.predictor
        )
        action.setToolTip(
            f'Time point: {identifier.name}\n\n'
            'Hover over the measurements plot and press the shortcut to add '
            'it at the hovered frame for the selected cell.\n'
            'Click to add it at the current frame.\n'
            'Right-click for more options.\n\n'
            f'{shortcutText}\nPredictor: {predictorText}'
        )
        
        # update AutoAction
        autoAction = self.timepointAutoActions[identifier.uid]
        autoAction.setShortcut(QKeySequence(f'Shift+{identifier.shortcut}'))

    
    def showTimepointActionContextMenu(self, uid, pos):
        identifier = self.timepointIdentifier(uid)
        button = self.timepointsToolBar.widgetForAction(
            self.timepointActions[uid]
        )
        menu = QMenu(button)
        menu.addAction('Edit...', partial(self.editTimepointIdentifier, uid))
        hasPredictor = (
            identifier.predictor['type'] != timepoint_predictors.PREDICTOR_NONE
        )
        predictAllAction = menu.addAction(
            'Re-evaluate predictor on all cells', 
            partial(self.reevaluateTimepoints, [uid])
        )
        predictCellAction = menu.addAction(
            'Re-evaluate predictor on displayed cell', 
            partial(self.reevaluateDisplayedCellTimepoints, uids=[uid])
        )
        predictAllAction.setEnabled(hasPredictor and bool(self.data))
        predictCellAction.setEnabled(
            hasPredictor and bool(self.data) and self.selectedID is not None
        )
        menu.addAction(
            'Clear time points of current position', 
            partial(self.clearTimepoints, uid)
        )
        menu.addSeparator()
        menu.addAction('Remove', partial(self.removeTimepointIdentifier, uid))
        menu.exec_(button.mapToGlobal(pos))
    
    def editTimepointIdentifier(self, uid):
        identifier = self.timepointIdentifier(uid)
        values = self.execTimepointIdentifierDialog(
            identifier, f'Edit time point identifier "{identifier.name}"'
        )
        if values is None:
            return
        predictorChanged = values['predictor'] != identifier.predictor
        identifier.name = values['name']
        identifier.color = values['color']
        identifier.shortcut = values['shortcut']
        identifier.predictor = values['predictor']
        identifier.autoApply = values['autoApply']
        self.updateTimepointAction(identifier)
        self.updateReevaluateTimepointsActions()
        if predictorChanged:
            for key in list(self.timepointAutoPredictions):
                if key[1] == uid:
                    del self.timepointAutoPredictions[key]
            for posData in self.data:
                if posData is not self.posData():
                    getattr(posData, 'timepointsPredicted', set()).discard(uid)
            self.reevaluateTimepoints([uid])
        self.updateTimepointLines()
    
    def removeTimepointIdentifier(self, uid, ask=True):
        identifier = self.timepointIdentifier(uid)
        if ask:
            msg = widgets.myMessageBox(wrapText=False)
            _, yes = msg.question(
                self, 'Remove time point identifier?',
                f'Remove "{identifier.name}" and its time points of all '
                'positions?', buttonsTexts=('Cancel', 'Remove')
            )
            if msg.clickedButton != yes:
                return
        action = self.timepointActions.pop(uid)
        self.timepointsToolBar.removeAction(action)
        action.deleteLater()
        self.timepointIdentifiers.remove(identifier)
        for key in list(self.timepointAutoPredictions):
            if key[1] == uid:
                del self.timepointAutoPredictions[key]
        for posData in self.data:
            getattr(posData, 'timepoints', {}).pop(uid, None)
            getattr(posData, 'timepointsPredicted', set()).discard(uid)
            getattr(posData, 'timepointsEdited', {}).pop(uid, None)
            
        autoAction = self.timepointAutoActions.pop(uid)
        self.removeAction(autoAction)
        autoAction.deleteLater()
        
        self.updateReevaluateTimepointsActions()
        self.updateTimepointLines()
    
    def clearTimepoints(self, uid):
        if not self.data:
            return
        self.positionTimepoints()[uid] = {}
        self.editedTimepointCells()[uid] = set()
        self.updateTimepointLines()
    
    def setTimepointIdentifiers(self, identifiers):
        for identifier in list(self.timepointIdentifiers):
            self.removeTimepointIdentifier(identifier.uid, ask=False)
        for identifier in identifiers:
            self.addTimepointIdentifierItem(identifier)
    
    def runTimepointPredictor(
            self, uid, posData=None, replot=False, IDs=None, skipIDs=(),
            force=False, autoAction_cb=False, checked=False
        ):
        """Predict the time points of `uid` for the cells of `posData` 
        (default current position), replacing the existing ones.
        
        If `IDs` is not None, only these cells are predicted. Cells in 
        `skipIDs` keep their current time points. Unless `force` is True,
        require `autoApply` and skip cells already predicted in this session
        for the same position path and identifier. Returns False if the
        prediction failed."""
        identifier = self.timepointIdentifier(uid)
        if not self.data or identifier is None:
            return False
        if not force and not identifier.autoApply:
            return True
        
        if posData is None:
            posData = self.posData()
        timepoints = self.positionTimepoints(posData)
        if identifier.predictor['type'] == timepoint_predictors.PREDICTOR_NONE:
            return True
        acdc_df = (
            self.acdc_df if posData is self.posData() 
            else self.getAcdcDf(posData)
        )
        predictionKey = (os.path.normcase(os.path.abspath(posData.pos_path)), uid)
        alreadyPredicted = self.timepointAutoPredictions[predictionKey]
        skipIDs = set(skipIDs)
        if autoAction_cb:
            IDs = [self.currID()]
        if IDs is not None:
            IDs = set(IDs) - skipIDs
        if not force:
            if IDs is None:
                IDs = (
                    set(acdc_df.index.get_level_values('Cell_ID'))
                    if acdc_df is not None else set()
                )
            IDs -= alreadyPredicted | skipIDs
        if IDs is not None and not IDs:
            return True
        acdc_df = self.acdcDfWithConstants(acdc_df, posData)
        try:
            # All cells are passed since predictors can use other cells
            predicted = timepoint_predictors.predict_timepoints(
                acdc_df, identifier.predictor, SizeT=posData.SizeT, IDs=IDs
            )
        except Exception as err:
            self.logger.exception('Time point prediction failed')
            self.measurementProcessingWarning(
                'Time point prediction failed', 
                f'Predicting "{identifier.name}" failed:\n\n{err}'
            )
            return False
        
        cellTimepoints = timepoints[uid]
        if IDs is None:
            replacedIDs = set(cellTimepoints).union(predicted) - skipIDs
        else:
            replacedIDs = set(IDs) - skipIDs
        edited = posData.timepointsEdited[uid]
        for ID in replacedIDs:
            cellTimepoints.pop(ID, None)
            edited.discard(ID)
            if ID in predicted:
                cellTimepoints[ID] = predicted[ID]

        completedIDs = (
            set(acdc_df.index.get_level_values('Cell_ID'))
            if acdc_df is not None else set()
        )
        if IDs is not None:
            completedIDs.intersection_update(IDs)
        alreadyPredicted.update(completedIDs - skipIDs)
        posData.timepointsPredicted.add(uid)
        
        numCells = sum(1 for ID in replacedIDs if ID in predicted)
        numTimepoints = sum(len(predicted[ID]) for ID in replacedIDs if ID in predicted)
        self.statusBar().showMessage(
            f'Predicted {numTimepoints} "{identifier.name}" time point(s) '
            f'for {numCells} cell(s).', 5000
        )
        if replot:
            self.updateTimepointLines()
        return True
    
    def predictorTimepointUids(self, uids=None):
        if uids is None:
            uids = [identifier.uid for identifier in self.timepointIdentifiers]
        return [
            uid for uid in uids 
            if self.timepointIdentifier(uid).predictor['type'] 
            != timepoint_predictors.PREDICTOR_NONE
        ]
    
    def updateReevaluateTimepointsActions(self):
        hasPredictors = bool(self.predictorTimepointUids())
        self.reevaluateAllTimepointsAction.setEnabled(hasPredictors)
        self.reevaluateCellTimepointsAction.setEnabled(hasPredictors)
    
    def askOverwriteManualTimepoints(self, editedCells, scope):
        """Warn that re-evaluating overwrites manual edits.
        
        Parameters
        ----------
        editedCells : dict
            {identifier name: sorted list of edited Cell_IDs}
        scope : str
            Description of the re-evaluated cells.

        Returns
        -------
        str
            'overwrite', 'skip' (keep the manually edited cells) or 'cancel'.
        """
        details = '\n'.join(
            f'{name}: Cell ID {", ".join(str(ID) for ID in IDs)}'
            for name, IDs in editedCells.items()
        )
        txt = html_utils.paragraph(
            f'Time points of {scope} have been <b>edited manually</b>.<br><br>'
            'Re-evaluating the predictors will <b>overwrite</b> these edits.'
            '<br><br>Do you want to overwrite the manual edits or skip the '
            'edited cells?'
        )
        msg = widgets.myMessageBox(wrapText=False)
        _, skipButton, overwriteButton = msg.warning(
            self, 'Overwrite manual edits?', txt, detailsText=details,
            buttonsTexts=('Cancel', 'Skip edited cells', 'Overwrite')
        )
        if msg.clickedButton == overwriteButton:
            return 'overwrite'
        if msg.clickedButton == skipButton:
            return 'skip'
        return 'cancel'
    
    def reevaluateTimepoints(self, uids=None, IDs=None, posData=None):
        """Re-run the predictors of `uids` (default all identifiers with 
        a predictor) on cells `IDs` (default all cells) of `posData`, 
        warning if manual edits would be overwritten. Returns False if 
        cancelled."""
        if not self.data:
            return False
        if posData is None:
            posData = self.posData()
        uids = self.predictorTimepointUids(uids)
        if not uids:
            self.statusBar().showMessage(
                'No time point identifier with a predictor.', 5000
            )
            return False
        edited = self.editedTimepointCells(posData)
        editedCells = {}
        for uid in uids:
            IDs_edited = edited[uid] if IDs is None else edited[uid] & set(IDs)
            if IDs_edited:
                name = self.timepointIdentifier(uid).name
                editedCells[name] = sorted(IDs_edited)
        skip = False
        if editedCells:
            if IDs is not None and len(IDs) == 1:
                scope = f'the displayed cell (ID {next(iter(IDs))})'
            else:
                numCells = len(set().union(*editedCells.values()))
                scope = f'{numCells} cell(s)'
            if posData is not self.posData():
                posName = getattr(posData, 'pos_foldername', 'another position')
                scope = f'{scope} of position "{posName}"'
            answer = self.askOverwriteManualTimepoints(editedCells, scope)
            if answer == 'cancel':
                return False
            skip = answer == 'skip'
        for uid in uids:
            skipIDs = edited[uid] if skip else ()
            self.runTimepointPredictor(
                uid, posData=posData, IDs=IDs, skipIDs=set(skipIDs), force=True
            )
        self.updateTimepointLines()
        return True
    
    def reevaluateAllTimepoints(self, checked=False):
        return self.reevaluateTimepoints()
    
    def reevaluateDisplayedCellTimepoints(self, checked=False, uids=None):
        if self.selectedID is None:
            self.statusBar().showMessage('Select a cell first.', 5000)
            return False
        return self.reevaluateTimepoints(uids=uids, IDs=[self.selectedID])
    
    def ensureTimepointPredictions(self, posData=None):
        if posData is None:
            posData = self.posData()
        self.positionTimepoints(posData)
        self.positionCellConstants(posData)
        uids = [
            identifier.uid for identifier in self.timepointIdentifiers
            if identifier.autoApply
        ]
        for uid in self.predictorTimepointUids(uids):
            if (uid not in posData.timepointsPredicted
                    and posData.timepointsEdited[uid]):
                # Predictor changed while this position had manual edits
                self.reevaluateTimepoints([uid], posData=posData)
            else:
                self.runTimepointPredictor(uid, posData=posData)
    
    def clipFrame(self, frame_i):
        return int(min(max(frame_i, 0), self.posData().SizeT-1))
    
    def hoveredMeasurementsPlotFrame(self):
        """Frame index under the mouse cursor in the measurements plot or 
        None if the cursor is not over the plot."""
        plot = self.measurementsPlot
        if not plot.isVisible():
            return
        pos = plot.mapFromGlobal(QCursor.pos())
        if not plot.rect().contains(pos):
            return
        viewPos = plot.getViewBox().mapSceneToView(plot.mapToScene(pos))
        return self.clipFrame(int(round(viewPos.x())) - 1)
    
    def timepointActionTriggered(self, uid, checked=False):
        identifier = self.timepointIdentifier(uid)
        frame_i = self.hoveredMeasurementsPlotFrame()
        if frame_i is None:
            toolbar = self.timepointsToolBar
            toolbarPos = toolbar.mapFromGlobal(QCursor.pos())
            if not toolbar.isVisible() or not toolbar.rect().contains(toolbarPos):
                self.statusBar().showMessage(
                    f'Hover over the measurements plot to add a '
                    f'"{identifier.name}" time point with its shortcut.', 5000
                )
                return
            frame_i = self.frame_i if self.data else None
        self.addTimepoint(uid, frame_i)
    
    def addTimepoint(self, uid, frame_i, ID=None):
        identifier = self.timepointIdentifier(uid)
        if not self.data or frame_i is None:
            self.statusBar().showMessage('Load data to add time points.', 5000)
            return False
        if self.measurementsModeCombobox.currentIndex() == 1:
            self.statusBar().showMessage(
                'Switch to "Time course" mode to add time points.', 5000
            )
            return False
        if ID is None:
            ID = self.selectedID
        if ID is None:
            self.statusBar().showMessage('Select a cell to add time points.', 5000)
            return False
        frame_i = self.clipFrame(frame_i)
        cellTimepoints = self.positionTimepoints()[uid].setdefault(ID, [])
        if frame_i in cellTimepoints:
            self.statusBar().showMessage(
                f'"{identifier.name}" is already at frame n. {frame_i+1} '
                f'for Cell ID {ID}.', 5000
            )
            return False
        cellTimepoints.append(frame_i)
        cellTimepoints.sort()
        self.markTimepointEdited(uid, ID)
        self.updateTimepointLines()
        self.statusBar().showMessage(
            f'Added "{identifier.name}" at frame n. {frame_i+1} for '
            f'Cell ID {ID}.', 5000
        )
        return True
    
    def removeTimepoint(self, uid, ID, frame_i):
        cellTimepoints = self.positionTimepoints()[uid].get(ID, [])
        if frame_i in cellTimepoints:
            cellTimepoints.remove(frame_i)
        if not cellTimepoints:
            self.positionTimepoints()[uid].pop(ID, None)
    
    def timepointLineRemoveRequested(self, line):
        self.removeTimepoint(line.uid, line.ID, line.frame_i)
        self.markTimepointEdited(line.uid, line.ID)
        self.updateTimepointLines()
    
    def timepointLineMoved(self, line):
        new_frame_i = self.clipFrame(int(round(line.value())) - 1)
        if new_frame_i != line.frame_i:
            self.removeTimepoint(line.uid, line.ID, line.frame_i)
            cellTimepoints = self.positionTimepoints()[line.uid].setdefault(
                line.ID, []
            )
            if new_frame_i not in cellTimepoints:
                cellTimepoints.append(new_frame_i)
                cellTimepoints.sort()
            self.markTimepointEdited(line.uid, line.ID)
        # Rebuild the lines after the drag event has been fully handled
        QTimer.singleShot(0, self.updateTimepointLines)
    
    def updateTimepointLines(self):
        for line in self.timepointLines:
            self.measurementsPlot.removeItem(line)
        self.timepointLines = []
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        if not self.data or isComparison or self.selectedID is None:
            return
        timepoints = self.positionTimepoints()
        for identifier in self.timepointIdentifiers:
            frames = timepoints.get(identifier.uid, {}).get(self.selectedID, [])
            for frame_i in frames:
                line = _TimepointLine(identifier, self.selectedID, frame_i)
                line.sigPositionChangeFinished.connect(self.timepointLineMoved)
                line.sigRemoveRequested.connect(
                    self.timepointLineRemoveRequested
                )
                self.measurementsPlot.addItem(line, ignoreBounds=True)
                self.timepointLines.append(line)
    
    def timepointsColumns(self, acdc_df):
        """Boolean columns (one per identifier) flagging the time points of 
        the current position in `acdc_df`."""
        columns = {}
        timepoints = self.positionTimepoints()
        for identifier in self.timepointIdentifiers:
            pairs = {
                (frame_i, ID) 
                for ID, frames in timepoints.get(identifier.uid, {}).items() 
                for frame_i in frames
            }
            columns[f'timepoint_{identifier.name}'] = acdc_df.index.isin(
                list(pairs)
            ) if pairs else np.zeros(len(acdc_df), dtype=bool)
        return columns
    
    def getMeasurementsDf(self):
        """Get the subset of acdc_df for the selected ID (index is frame_i)"""
        if self.acdc_df is None or self.selectedID is None:
            return
        
        try:
            df_ID = self.acdc_df.xs(self.selectedID, level='Cell_ID')
        except KeyError:
            return
        
        return df_ID.sort_index()
    
    def plotMeasurements(self, *args):
        for item in self.measurementsPlot.listDataItems():
            self.measurementsPlot.removeItem(item)
        self.measurementsLegend.clear()
        self.updateCurrentFrameLine()
        self.updateColumnsIcons()

        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        self.currentFrameLine.setVisible(not isComparison)
        self.updateTimepointLines()
        if not self.data:
            self.measurementsPlot.setTitle('No data loaded')
            return
        if isComparison:
            self.plotMeasurementsComparison()
            return

        self.measurementsPlot.setLabel('bottom', 'Frame n.')
        isNormalized = self.normalizeCheckbox.isChecked()
        self.measurementsPlot.setLabel(
            'left', 'Normalized value' if isNormalized else 'Value'
        )
        
        posData = self.posData()
        self.measurementsPlot.setXRange(1, max(posData.SizeT, 2), padding=0.02)
        
        df_ID = self.getMeasurementsDf()
        columns = self.checkedColumns()
        if self.selectedID is None:
            self.measurementsPlot.setTitle('')
        else:
            self.measurementsPlot.setTitle(f'Cell ID {self.selectedID}')
        
        if df_ID is None or not columns:
            return
        
        frames_n = df_ID.index.get_level_values(0).to_numpy() + 1
        for c, col in enumerate(columns):
            if col not in df_ID.columns:
                continue
            values = pd.to_numeric(df_ID[col], errors='coerce').to_numpy()
            rawValues = values.astype(float)
            values = rawValues
            if isNormalized:
                values = self._normalize(values)
            color = self.getSignalColor(col, c, len(columns))
            pointData = [
                {
                    'frame_i': int(frame_n-1),
                    'Cell_ID': self.selectedID,
                    'values': {col: (raw, value)},
                    'normalized': isNormalized,
                }
                for frame_n, raw, value in zip(frames_n, rawValues, values)
            ]
            item = self.measurementsPlot.plot(
                frames_n, values, name=col,
                pen=pg.mkPen(color=color, width=2),
                symbol='o', symbolSize=5, symbolBrush=color, symbolPen=None,
                data=pointData
            )
            self.connectSignalPoints(item)

    def measurementsModeChanged(self, *args):
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        if self._measurementsMode == 0:
            self._timeCourseColumns = set(self.checkedColumns())
        self._measurementsMode = self.measurementsModeCombobox.currentIndex()
        columns = (
            set(self.measurementsAxes.values()) if isComparison
            else self._timeCourseColumns
        )
        self.columnsModel.blockSignals(True)
        for item in self.columnsModel.measurementItems():
            item.setCheckState(
                Qt.Checked if item.text() in columns else Qt.Unchecked
            )
        self.columnsModel.blockSignals(False)
        self.columnsListWidget.setToolTip(
            'Click two columns to plot: first X, then Y. '
            'Click a selected column again to free its axis.'
            if isComparison else
            'Click a column to plot/unplot it. Right-click on a column to '
            'change its color.'
        )
        self.plotMeasurements()

    def measurementColumnChanged(self, item):
        if not item.isCheckable():
            return
        if self.measurementsModeCombobox.currentIndex() == 1:
            col = item.text()
            if item.checkState() != Qt.Checked:
                for axis, measurement in self.measurementsAxes.items():
                    if measurement == col:
                        self.measurementsAxes[axis] = ''
            elif col not in self.measurementsAxes.values():
                axis = next(
                    (axis for axis, measurement in self.measurementsAxes.items()
                     if not measurement), None
                )
                if axis is None:
                    self.columnsModel.blockSignals(True)
                    item.setCheckState(Qt.Unchecked)
                    self.columnsModel.blockSignals(False)
                    self.statusBar().showMessage(
                        'Two measurements are already selected. Uncheck X or Y '
                        'before selecting another measurement.', 5000
                    )
                else:
                    self.measurementsAxes[axis] = col
        self.plotMeasurements()

    def plotMeasurementsComparison(self):
        xCol = self.measurementsAxes['X']
        yCol = self.measurementsAxes['Y']
        isNormalized = self.normalizeCheckbox.isChecked()
        suffix = ' (normalized)' if isNormalized else ''
        self.measurementsPlot.setLabel('bottom', f'{xCol}{suffix}')
        self.measurementsPlot.setLabel('left', f'{yCol}{suffix}')
        self.measurementsPlot.setTitle('All cells, all frames (current position)')
        self.measurementsPlot.enableAutoRange()
        if self.acdc_df is None:
            self.measurementsPlot.setTitle('No measurements available')
            return
        if not xCol or not yCol:
            self.measurementsPlot.setTitle('Select X and Y measurements from the column list')
            return

        df = self.acdc_df
        frames = df.index.get_level_values('frame_i').to_numpy()
        IDs = df.index.get_level_values('Cell_ID').to_numpy()
        inRange = (frames >= 0) & (frames < self.posData().SizeT) & (IDs > 0)
        df = df.loc[inRange]
        frames, IDs = frames[inRange], IDs[inRange]
        rawX = pd.to_numeric(df[xCol], errors='coerce').to_numpy(dtype=float)
        rawY = pd.to_numeric(df[yCol], errors='coerce').to_numpy(dtype=float)
        finite = np.isfinite(rawX) & np.isfinite(rawY)
        frames, IDs = frames[finite], IDs[finite]
        rawX, rawY = rawX[finite], rawY[finite]
        if not len(rawX):
            self.measurementsPlot.setTitle('No finite paired measurements')
            return
        x = self._normalize(rawX) if isNormalized else rawX
        y = self._normalize(rawY) if isNormalized else rawY
        pointData = [
            {
                'frame_i': int(frame_i),
                'Cell_ID': int(ID),
                'values': {xCol: (rx, vx), yCol: (ry, vy)},
                'normalized': isNormalized,
            }
            for frame_i, ID, rx, ry, vx, vy
            in zip(frames, IDs, rawX, rawY, x, y)
        ]
        item = self.measurementsPlot.plot(
            x, y, pen=None, symbol='o', symbolSize=7,
            symbolBrush=self.getSignalColor(yCol), symbolPen=None,
            data=pointData
        )
        self.connectSignalPoints(item)

    def connectSignalPoints(self, item):
        # PlotDataItem does not forward hover options to its scatter item.
        item.scatter.setData(
            x=item.xData, y=item.yData, data=item.opts['data'],
            hoverable=True, tip=self.measurementPointTip,
            hoverSize=11, hoverBrush=pg.mkBrush(255, 100, 100)
        )
        item.scatter.sigClicked.connect(self.measurementPointClicked)

    @staticmethod
    def measurementPointTip(x, y, data):
        lines = [
            f'Cell ID: {data["Cell_ID"]}',
            f'Frame: {data["frame_i"]+1}',
        ]
        for col, (raw, value) in data['values'].items():
            lines.append(f'{col}: {raw:.6g}')
            if data['normalized']:
                lines.append(f'{col} (normalized): {value:.6g}')
        return '\n'.join(lines)

    def measurementPointClicked(self, item, points, event):
        if event.button() != Qt.LeftButton or not len(points):
            return
        data = points[0].data()
        ID, frame_i = data['Cell_ID'], data['frame_i']
        if ID not in self.IDs or not 0 <= frame_i < self.posData().SizeT:
            self.statusBar().showMessage(
                f'Cell ID {ID} at frame {frame_i+1} is not available.', 5000
            )
            return
        # Navigating replots the measurements, deleting the clicked scatter item
        # while pyqtgraph is still dispatching the click to it --> defer
        QTimer.singleShot(
            100, partial(self.navigateToSignalPoint, ID, frame_i)
        )

    def navigateToSignalPoint(self, ID, frame_i):
        self.idScrollBar.setValue(self.IDs.index(ID))
        self.setFrame(frame_i)
        self.zoomToSelectedID(goToZslice=True, force=True)
    
    def getSignalColor(self, col, c=None, numColumns=None):
        """Get the plotting color of column `col`: the custom color if set 
        by the user, otherwise the default color based on the index `c` of 
        the column among the plotted ones."""
        color = self.measurementsColors.get(col)
        if color is not None:
            return color
        
        columns = self.checkedColumns()
        if c is None:
            c = columns.index(col) if col in columns else len(columns)
        if numColumns is None:
            numColumns = len(columns)
        return pg.intColor(c, hues=max(numColumns, 9))
    
    def updateColumnsIcons(self):
        """Show color swatches or comparison-axis markers beside columns."""
        columns = self.checkedColumns()
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        self.columnsModel.blockSignals(True)
        for item in self.columnsModel.measurementItems():
            col = item.text()
            item.setCheckable(True)
            if isComparison:
                axes = '/'.join(
                    axis for axis, measurement in self.measurementsAxes.items()
                    if measurement == col
                )
                item.setData(axes, Qt.AccessibleDescriptionRole)
                if not axes:
                    item.setIcon(QIcon())
                    continue
                pixmap = QPixmap(24, 16)
                pixmap.fill(Qt.transparent)
                painter = QPainter(pixmap)
                painter.setPen(self.columnsListWidget.palette().text().color())
                font = self.columnsListWidget.font()
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(pixmap.rect(), Qt.AlignCenter, axes)
                painter.end()
                item.setIcon(QIcon(pixmap))
                continue
            item.setData('', Qt.AccessibleDescriptionRole)
            if col not in columns:
                item.setIcon(QIcon())
                continue
            pixmap = QPixmap(12, 12)
            pixmap.fill(self.getSignalColor(col))
            item.setIcon(QIcon(pixmap))
        self.columnsModel.blockSignals(False)
        # Signals were blocked to avoid itemChanged, so repaint explicitly
        self.columnsListWidget.viewport().update()
    
    def showColumnsContextMenu(self, pos):
        proxy_index = self.columnsListWidget.indexAt(pos)
        if not proxy_index.isValid():
            return
        source_index = self.columnsSearchProxy.mapToSource(proxy_index)
        item = self.columnsModel.itemFromIndex(source_index)
        if not item.isCheckable():
            return
        
        col = item.text()
        menu = QMenu(self.columnsListWidget)

        plotAction = QAction('Plot', menu)
        plotAction.setCheckable(True)
        plotAction.setChecked(item.checkState() == Qt.Checked)
        plotAction.toggled.connect(
            lambda checked, item=item: item.setCheckState(
                Qt.Checked if checked else Qt.Unchecked
            )
        )
        menu.addAction(plotAction)
        if self.measurementsModeCombobox.currentIndex() == 1:
            menu.popup(self.columnsListWidget.viewport().mapToGlobal(pos))
            return
        menu.addSeparator()
        
        changeColorAction = QAction('Change color...', menu)
        changeColorAction.triggered.connect(
            lambda checked=False, col=col: self.changeSignalColor(col)
        )
        menu.addAction(changeColorAction)
        
        resetColorAction = QAction('Reset to default color', menu)
        resetColorAction.setEnabled(col in self.measurementsColors)
        resetColorAction.triggered.connect(
            lambda checked=False, col=col: self.resetSignalColor(col)
        )
        menu.addAction(resetColorAction)
        
        menu.popup(self.columnsListWidget.viewport().mapToGlobal(pos))
    
    def changeSignalColor(self, col):
        initialColor = pg.mkColor(self.getSignalColor(col))
        color = QColorDialog.getColor(
            initialColor, self, f'Select color of "{col}"'
        )
        if not color.isValid():
            return
        
        self.measurementsColors[col] = color
        self.plotMeasurements()
    
    def resetSignalColor(self, col):
        self.measurementsColors.pop(col, None)
        self.plotMeasurements()
    
    @staticmethod
    def _normalize(values):
        finite = np.isfinite(values)
        normalized = np.full(values.shape, np.nan, dtype=float)
        if not finite.any():
            return normalized
        _min, _max = values[finite].min(), values[finite].max()
        if _max == _min:
            normalized[finite] = 0
        else:
            normalized[finite] = (values[finite] - _min) / (_max - _min)
        return normalized
    
    def stepID(self, step):
        if not self.IDs:
            return
        idx = int(np.clip(
            self.idScrollBar.value() + step, 0, len(self.IDs)-1
        ))
        self.idScrollBar.setValue(idx)
        
    def currID(self):
        if not self.IDs:
            return None
        idx = self.idScrollBar.value()
        return self.IDs[idx]
    
    def updateScrollbars(self):
        """Sync the navigation widgets after `posData.frame_i` was changed 
        externally (e.g., by `SearchIDGuiElements.findID`)"""
        frame_i = self.frame_i
        self.navSpinBox.setValueNoEmit(frame_i+1)
        self.resetNavigateFramesScrollbar(frame_i=frame_i)
        self._lab = None
        self._labKey = None
        self.updateCurrentFrameLine()
        self.plotMeasurements()
        self.updateIDnotPresentLabel()
    
    def invertBw(self, checked, update=True):
        # `ColorSchemeGuiElements.invertBw` inverts the text annotations of 
        # both images of guiWin while the viewer has only the left image
        textAnnot = self.textAnnot
        hasLeftTextAnnot = 0 in textAnnot
        if not hasLeftTextAnnot:
            del self.textAnnot
        else:
            textAnnot[1] = _NoOpItem()
        try:
            super().invertBw(checked, update=update and self.isDataLoaded)
        finally:
            self.textAnnot = textAnnot
            textAnnot.pop(1, None)
        
        self.isBwInverted = checked
        if checked:
            self.measurementsPlot.setBackground(graphLayoutBkgrColor)
        else:
            self.measurementsPlot.setBackground(darkBkgrColor)
    
    def highlightSearchedID(self, ID, force=False, greyOthers=True):
        self.highlightedID = ID
        self.highLightIDLayerImg1.clear()
        self.searchedIDitemLeft.clear()
        if ID == 0:
            return
        self.highlightSelectedID()
    
    def goToObjectID(self, ID):
        if ID not in self.IDs:
            self.statusBar().showMessage(
                f'Cell ID {ID} does not exist in the acdc_output table.', 
                3000
            )
            return
        
        idx = self.IDs.index(ID)
        if self.idScrollBar.value() == idx:
            self.setSelectedID(ID)
        else:
            self.idScrollBar.setValue(idx)
    
    def goToLostObjectID(self, ID, color=None):
        self.goToObjectID(ID)
    
    def goToAcceptedLostObjectID(self, ID):
        self.goToObjectID(ID)
    
    def initLookupTableLab(self):
        self.initLabelsImageItems()
    
    def updateLookuptable(self, *args, **kwargs):
        self.initLabelsImageItems()
    
    def update_rp(self, *args, **kwargs):
        self.get_data()
    
    def clearCursors(self):
        self.ax1_cursor.setData([], [])
    
    def connectLeftClickButtons(self):
        # The viewer does not have editing tools that need to be reconnected
        pass
    
    def disconnectLeftClickButtons(self):
        pass
    
    def uncheckLeftClickButtons(self, sender):
        for button in self.leftClickButtons:
            if button == sender:
                continue
            if button.isChecked():
                button.setChecked(False)
    
    def drawPointsLayers(self, **kwargs):
        # Points layers are a main GUI-only feature
        pass
    
    def addPointsLayersToVolumeViewer(self, *args, **kwargs):
        pass
    
    def resetCursor(self):
        while self.app.overrideCursor() is not None:
            self.app.restoreOverrideCursor()
    
    def autoRange(self):
        self.ax1.autoRange()
        if self.followIDcheckbox.isChecked():
            # Keep the view on the selected cell (e.g., after resizing the 
            # bottom layout)
            self.zoomToSelectedID()
    
    def clearComboBoxFocus(self, *args):
        self.setFocus()
    
    def isScaleBarOrTimestampHighlighted(self):
        for name in ('scaleBar', 'timestamp'):
            item = getattr(self, name, None)
            if item is not None and item.isHighlighted():
                return item
    
    def resetCountKeyPress(self):
        self.doubleKeyTimeElapsed = True
        self.countKeyPress = 0
    
    def isHoveringRoiHandle(self, roi):
        # Same as `gui.guiWin.gui_isHoveringRoiHandle`
        if roi is None:
            return False
        try:
            handles = roi.getHandles()
        except Exception:
            return False

        for handle in handles:
            try:
                if handle.currentPen == handle.hoverPen:
                    return True
            except Exception:
                continue
        return False
    
    def gui_hoverEventImg1(self, event):
        if not self.isDataLoaded:
            return
        
        if event.isExit():
            self.xHoverImg, self.yHoverImg = None, None
            self.wcLabel.setText('')
            self.resetCursor()
            if self.slideshowWin is not None:
                self.slideshowWin.setMirroredCursorPos(None, None)
            return
        
        x, y = event.pos()
        self.xHoverImg, self.yHoverImg = x, y
        
        if self.slideshowWin is not None:
            self.slideshowWin.setMirroredCursorPos(x, y)
        
        if self.rulerButton.isChecked() and self.tempSegmentON:
            self.drawTempRulerLine(event)
        
        Y, X = self.currentLab2D.shape
        xdata, ydata = int(x), int(y)
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            self.wcLabel.setText('')
            return
        
        ID = self.currentLab2D[ydata, xdata]
        self.updatePropsWidget(ID, fromHover=True)
        activeToolButton = self.getActiveToolButton()
        txt = self.hoverValuesFormatted(xdata, ydata, activeToolButton, True)
        self.checkHighlightScaleBar(x, y, activeToolButton)
        self.checkHighlightTimestamp(x, y, activeToolButton)
        self.wcLabel.setText(txt)
    
    def gui_mousePressEventImg1(self, event):
        if not self.isDataLoaded:
            pg.ImageItem.mousePressEvent(self.img1, event)
            return
        
        if event.button() not in STANDARD_MOUSE_BUTTONS:
            self.mousePressEvent(event)
            return
        
        left_click = event.button() == Qt.MouseButton.LeftButton
        right_click = event.button() == Qt.MouseButton.RightButton
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        
        isZoomRectActive = self.zoomRectButton.isChecked()
        if isZoomRectActive and self.isHoveringRoiHandle(self.zoomRectItem):
            event.ignore()
            return
        
        if left_click and self.rulerButton.isChecked():
            if not self.tempSegmentON:
                self.ax1_rulerAnchorsItem.setData([xdata], [ydata])
                self.tempSegmentON = True
            else:
                modifiers = QGuiApplication.keyboardModifiers()
                ctrl = modifiers == Qt.ControlModifier
                self.tempSegmentON = False
                xxRA, yyRA = self.ax1_rulerAnchorsItem.getData()
                x0, y0 = xxRA[0], yyRA[0]
                if ctrl:
                    x1, y1 = transformation.snap_xy_to_closest_angle(
                        x0, y0, xdata, ydata
                    )
                else:
                    x1, y1 = xdata, ydata
                lengthText = self.getRulerLengthText()
                self.ax1_rulerPlotItem.setData(
                    [x0, x1], [y0, y1], lengthText=lengthText
                )
                self.ax1_rulerAnchorsItem.setData([x0, x1], [y0, y1])
            return
        
        if isZoomRectActive and (left_click or right_click):
            if left_click:
                self.zoomRectItem.setPos((xdata, ydata))
                self.isMouseDragImg1 = True
            else:
                try:
                    xRange, yRange = self.zoomRectItem.getLastRange()
                    self.ax1.setRange(xRange=xRange, yRange=yRange, padding=0)
                except Exception as err:
                    QTimer.singleShot(100, self.autoRange)
            return
        
        highlightedItem = self.isScaleBarOrTimestampHighlighted()
        if left_click and highlightedItem is not None:
            highlightedItem.mousePressed(x, y)
            return
        
        if right_click:
            screenPos = event.screenPos()
            self.imgGradShowContextMenu(screenPos.x(), screenPos.y())
            return
        
        pg.ImageItem.mousePressEvent(self.img1, event)
        event.ignore()
    
    def imgGradShowContextMenu(self, x, y):
        # Same as `gui.guiWin.gui_imgGradShowContextMenu`
        highlightedItem = self.isScaleBarOrTimestampHighlighted()
        if highlightedItem is not None:
            highlightedItem.showContextMenu(x, y)
            return
        self.imgGrad.gradient.menu.popup(QPoint(int(x), int(y)))
    
    def gui_mouseDragEventImg1(self, event):
        if not self.isDataLoaded:
            pg.ImageItem.mouseMoveEvent(self.img1, event)
            return
        
        x, y = event.pos().x(), event.pos().y()
        
        if hasattr(self, 'scaleBar'):
            if self.scaleBarDialog is not None:
                self.scaleBarDialog.locCombobox.setCurrentText('Custom')
            if self.scaleBar.isHighlighted() and self.scaleBar.clicked:
                self.scaleBar.setLocationProperty('custom')
                self.scaleBar.move(x, y)
                return
        
        if hasattr(self, 'timestamp'):
            if self.timestampDialog is not None:
                self.timestampDialog.locCombobox.setCurrentText('Custom')
            if self.timestamp.isHighlighted() and self.timestamp.clicked:
                self.timestamp.setLocationProperty('custom')
                self.timestamp.move(x, y)
                return
        
        if self.isMouseDragImg1 and self.zoomRectButton.isChecked():
            xdata, ydata = int(x), int(y)
            x0, y0 = self.zoomRectItem.pos()
            w, h = (xdata-x0), (ydata-y0)
            self.zoomRectItem.setSize((w, h))
            return
        
        pg.ImageItem.mouseMoveEvent(self.img1, event)
    
    def gui_mouseReleaseEventImg1(self, event):
        if not self.isDataLoaded:
            pg.ImageItem.mouseReleaseEvent(self.img1, event)
            return
        
        if hasattr(self, 'scaleBar'):
            if self.scaleBar.isHighlighted() and self.scaleBar.clicked:
                self.scaleBar.clicked = False
                return
        
        if hasattr(self, 'timestamp'):
            if self.timestamp.isHighlighted() and self.timestamp.clicked:
                self.timestamp.clicked = False
                return
        
        if self.isMouseDragImg1 and self.zoomRectButton.isChecked():
            self.isMouseDragImg1 = False
            self.zoomRectDone()
            return
        
        pg.ImageItem.mouseReleaseEvent(self.img1, event)
    
    def gui_mouseClickEventImg1(self, event):
        """Select the clicked cell (pyqtgraph sends click events only if 
        the mouse was not dragged, hence panning does not select)."""
        if not self.isDataLoaded:
            return
        
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        isToolActive = (
            self.rulerButton.isChecked() 
            or self.zoomRectButton.isChecked()
            or self.isScaleBarOrTimestampHighlighted() is not None
        )
        if isToolActive:
            return
        
        x, y = event.pos().x(), event.pos().y()
        lab2D = self.currentLab2D
        Y, X = lab2D.shape[-2:]
        xdata, ydata = int(x), int(y)
        if not (0 <= xdata < X and 0 <= ydata < Y):
            return
        
        ID = int(lab2D[ydata, xdata])
        if ID == 0:
            return
        
        event.accept()
        self.goToObjectID(ID)
    
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
                    self.measurementsPlot.autoRange()
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
            else:
                self.uncheckLeftClickButtons(None)
        elif key == Qt.Key_Z:
            self.zSliceCheckbox.setChecked(not self.zSliceCheckbox.isChecked())
        elif key in (Qt.Key_Left, Qt.Key_Right):
            step = 1 if key == Qt.Key_Right else -1
            isZstack = self.posData().SizeZ > 1
            if isZstack and self.zSliceCheckbox.isChecked():
                z = self.zSliceScrollBar.sliderPosition() + step
                self.zSliceScrollBar.setSliderPosition(z)
            else:
                self.setFrame(self.frame_i + step)
        elif key in (Qt.Key_Up, Qt.Key_Down):
            self.stepID(1 if key == Qt.Key_Up else -1)
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        if self.measurementProcessingThread is not None:
            self.measurementProcessingWarning(
                'Signal processing running',
                'Wait for measurement processing to finish before closing the viewer.'
            )
            event.ignore()
            return
        if self.isDataLoaded:
            if not self.saveConfiguration():
                event.ignore()
                return
        if self.isDataLoaded:
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
        super().closeEvent(event)
        self.sigClosed.emit(self)

def run(folderpath=None):
    """Launch the measurements viewer as a standalone application, optionally 
    loading `folderpath` (experiment, Position or Images folder)."""
    from ._run import _setup_app
    app, _ = _setup_app()
    win = MeasurementsViewerWin()
    win.show()
    if folderpath:
        win.openFolder(folderpath=folderpath)
    app.exec_()

if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else None)
