"""Builds the measurements viewer window (ViewerSetupGuiElements): actions,
menus, toolbars, image/plot/navigation/annotation widgets, the
measurements panel, layout, status bar, signal connections, hard-coded
shortcuts and Help actions. Holds the setup-only constants (quick start
text, phase translator and mask tool tips, Alt+P shortcut, label font).
Spec: measurements_viewer/__init__.py.
"""
from collections import defaultdict
from datetime import timedelta
from functools import partial
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QIcon
from qtpy.QtWidgets import (
    QWidget, QLabel, QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox,
    QSplitter, QLineEdit, QCheckBox, QComboBox, QAbstractItemView, QAction,
    QApplication, QToolButton, QPushButton, QActionGroup, QWidgetAction,
    QTreeView, QSizePolicy
)
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import colors
from cellacdc.myutils import app_info, filesystem
from cellacdc.shared.widgets import (
    basic_widgets as widgets_basic_widgets,
    combo_boxes as widgets_combo_boxes,
    lut_items as widgets_lut_items,
    message_box as widgets_message_box,
    numeric_inputs as widgets_numeric_inputs,
    plot_items as widgets_plot_items,
    push_buttons as widgets_push_buttons,
    search_widgets as widgets_search_widgets,
    toolbars as widgets_toolbars,
)
from cellacdc import urls
from cellacdc import graphLayoutBkgrColor, darkBkgrColor
from cellacdc import cca_df_colnames, base_cca_dict
from cellacdc import regionprops as acdc_regionprops
from cellacdc import html_utils
from .. import widgets as viewer_widgets
from . import plotting as viewer_plotting


_font = QFont()
_font.setPixelSize(11)

CCA_DF_INT_COLS = [
    col for col in cca_df_colnames if type(base_cca_dict[col]) == int
]

APPLY_PHASES_CELL_SHORTCUT = 'Alt+P'

MASK_UNANALYZED_TOOLTIP = (
    'Ignore frames after the last analyzed frame\n\n'
    'When checked, the calculations of each position use only the frames up to\n'
    'its last analyzed frame in the main GUI: the last cell cycle annotated\n'
    'frame, or the last tracked frame (last frame of the acdc_output table) when\n'
    'there are no cell cycle annotations. Time point and custom annotation\n'
    'predictors, phases, measurement processing, combined measurements and\n'
    'conditional formatting ignore the later frames (the "SizeT" of equations is\n'
    'the number of analyzed frames) and leave their outputs empty there. Images\n'
    'and plots still show all frames (ignored frames are shaded in the plot).\n\n'
    'Results calculated before changing this option are not recalculated\n'
    'automatically (except phases evaluated automatically): re-evaluate\n'
    'the predictors and the phases, or reload the configuration to\n'
    'recalculate the measurement steps. Saved in the configuration.'
)


PHASE_TRANSLATOR_TOOLTIP = (
    'Time point to phase translator\n\n'
    'Turn the time points of each cell into phases (e.g., "G1", "S/G2/M"):\n'
    'the frames between two consecutive time points get the same phase name.\n\n'
    'How to use it:\n'
    '1. Define time point identifiers ("Add time point identifier...")\n'
    '   and mark them on the cells (hover over the plot and press\n'
    '   their shortcut, or let their predictors find them).\n'
    '2. Open the translator and add a rule: check the time points that '
    'separate\n'
    '   the phases, put them in their order and name the phases before, '
    'between\n'
    '   and after them (optionally add a formula and required custom '
    'annotations).\n'
    '3. Click "Apply phase evaluation" on the displayed cell or on all cells\n'
    '   (or let the phases be evaluated automatically). Later, the "Evaluate\n'
    '   phases on displayed cell" button next to this one (shortcut "'
    + APPLY_PHASES_CELL_SHORTCUT + '")\n'
    '   re-evaluates the displayed cell without opening the translator.\n'
    '4. The phases are drawn as a colored band below the time course\n'
    '   plot and color the Measurement vs Measurement points. "Export\n'
    '   measurements..." saves the "phase" and "phase_<name>" columns.\n\n'
    'Rules are tried from top to bottom; the first one that matches a cell\n'
    'supplies its phases. Custom annotations with the "Overrides phases"\n'
    'option come first: an annotated cell gets the annotation name as\n'
    'phase on all its frames. Save the configuration to keep the rules.\n'
    'Details: Help --> Measurements viewer user guide.'
)

MEASUREMENTS_VIEWER_QUICK_START = (
    'The measurements viewer plots the measurements (columns of the '
    '<code>acdc_output</code> table) of the selected cell over time.'
    '<br><br><b>How to use it:</b>'
    + html_utils.to_list([
        'Load data with <code>File --> Load Folder...</code> (or load a '
        'saved configuration at startup).',
        'Select the cell with the <b>Cell ID</b> controls at the bottom or '
        'by clicking on it in the image; the frame scrollbar moves the '
        'current frame (red line on the plot).',
        'Click a column in <b>Columns to plot</b> to plot or remove its '
        'time course; right-click it to change its color or add '
        'conditional formatting. <b>Measurement vs Measurement</b> plots '
        'two columns against each other for all cells.',
        'Mark time points (<b>Time points</b> toolbar) and custom '
        'annotations, and turn time points into phases with the '
        '<b>Time point to phase translator</b>.',
        '<b>Process measurements...</b> and <b>Combine measurements...</b> '
        'add new columns; <b>Export measurements...</b> saves them. '
        '<code>File &gt; Save tables</code> (<code>Ctrl+S</code>) saves the '
        'acdc_output tables. Save the configuration (<code>File</code> menu, '
        '<code>Ctrl+Shift+S</code>) to reuse the setup on other data.',
    ], ordered=True)
    + 'Press <code>Ctrl+F</code> to search for a Cell ID or a '
    'functionality.'
)


class ViewerSetupGuiElements:
    """Builds the measurements viewer window: the attributes that the
    shared GUI mixins expect, actions, menus, toolbars, the image, plot,
    navigation and annotation widgets, the measurements panel, layout,
    status bar and signal connections, plus the hard-coded shortcuts and
    the Help actions. Spec: measurements_viewer/__init__.py.
    """

    def postInitLoadSetSettings(self):
        followID = self._get_setting_value('followID', default=True, cast=bool)
        self.followIDcheckbox.setChecked(followID)
        
        restrictFrames = self._get_setting_value('restrictFrames', default=True, cast=bool)
        self.restrictFramesCheckbox.setChecked(restrictFrames)
        
        rescalePlot = self._get_setting_value('rescalePlot', default=True, cast=bool)
        self.rescalePlotCheckbox.setChecked(rescalePlot)

    def initViewerToolsAttributes(self):
        """Initialize the attributes required by the mixins in 
        `gui_viewer_tools` and `gui_overlay`"""
        self._appName = 'Cell-ACDC'
        try:
            self._version = app_info.read_version()
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
        # Overlay state on the xy plane while a zy/zx plane turned it off,
        # None on the xy plane (`setOverlayEnabledForPlane`)
        self.overlayButtonPrevState = None
        self.currentPropsID = -1
        # True while the code (hover, sync with the selected cell) writes
        # the Object ID of the "Cell-ACDC objects" dock: only user edits
        # of the ID select a cell (see `propsWidgetIDvalueChanged`)
        self._isUpdatingPropsDock = False
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
        self.AutoPilotProfile = viewer_widgets._NoAutoPilotProfile()
        self.checkedOverlayChannels = set()
        self.overlayLabelsItems = {}
        self.drawModeOverlayLabelsChannels = {}
        self.existingSegmEndNames = []
        self.ch_names = []
        
        # Main GUI widgets that do not exist in the viewer
        self.equalizeHistPushButton = QPushButton(self)
        self.equalizeHistPushButton.hide()
        self.imgGradRight = viewer_widgets._NoOpItem()

    def initCustomAnnotationsAttributes(self):
        """Initialize the attributes required by 
        `gui_custom_annotations.CustomAnnotationsGuiElements`"""
        self.customAnnotDict = {}
        self.customAnnotButton = None
        self.savedCustomAnnot = {}
        # {button: (toggle action, Shift+shortcut re-evaluate action)}
        self.customAnnotShortcutActions = {}
        # Cells already predicted {(normcase pos_path, name): set of IDs}
        self.customAnnotAutoPredictions = defaultdict(set)
        # {name: positions whose table had the column before it was added}
        self.customAnnotPreexistingColumns = {}

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
        self.loadConfigurationAction = QAction(
            QIcon(":file-open.svg"), 'Load configuration...', self
        )
        self.loadConfigurationAction.setToolTip(
            'Load a measurements viewer configuration\n\n'
            'Restores inputs, measurement steps, plots, time point\n'
            'identifiers, custom annotations and phase rules.'
        )
        # Ctrl+S saves the tables (as in the main GUI), the configuration has
        # its own icon so that the two save buttons are told apart
        self.saveTablesAction = QAction(
            QIcon(":file-save.svg"), 'Save tables', self
        )
        self.saveTablesAction.setShortcut('Ctrl+S')
        self.saveTablesAction.setToolTip(
            'Save tables (Ctrl+S)\n\n'
            'Overwrite the acdc_output table of every Position with unsaved\n'
            'changes (time points, annotations, phases, measurement steps). The\n'
            'previous file is backed up in the Position\'s "recovery" folder.'
        )
        self.saveConfigurationAction = QAction(
            QIcon(":cog.svg"), 'Save configuration', self
        )
        self.saveConfigurationAction.setShortcut('Ctrl+Shift+S')
        self.saveConfigurationAction.setToolTip(
            'Save the current configuration (Ctrl+Shift+S)'
        )
        self.saveConfigurationAsAction = QAction(
            QIcon(":edit-copy.svg"), 'Save configuration as...', self
        )
        self.saveConfigurationAsAction.setToolTip(
            'Save the current configuration to a new file'
        )
        self.addTimepointIdentifierAction = QAction(
            QIcon(":addTimepoint.svg"), 'Add time point identifier...', self
        )
        self.addTimepointIdentifierAction.setToolTip(
            'Add time point identifier\n\n'
            'Define a named time point (e.g., "Bud '
            'emergence") with a color, a keyboard\n'
            'shortcut and an optional per-cell predictor. Hover over the measurements\n'
            'plot and press the shortcut to add a time point at the hovered frame.'
        )
        self.reevaluateAllTimepointsAction = QAction(
            QIcon(":reevaluateAllTimepoint.svg"),
            'Re-evaluate predictors on all cells', self
        )
        self.reevaluateAllTimepointsAction.setToolTip(
            'Re-evaluate predictors on all cells\n\n'
            'Run the predictors of all time point identifiers on every\n'
            'cell of the current position, replacing their time points.\n'
            'You will be warned if manual edits would be overwritten.'
        )
        self.reevaluateCellTimepointsAction = QAction(
            QIcon(":reevaluateTimepoint.svg"),
            'Re-evaluate predictors on displayed cell', self
        )
        self.reevaluateCellTimepointsAction.setToolTip(
            'Re-evaluate predictors on displayed cell\n\n'
            'Run the predictors of all time point identifiers only\n'
            'on the displayed cell, replacing its time points. You\n'
            'will be warned if manual edits would be overwritten.'
        )
        self.phaseTranslatorAction = QAction(
            QIcon(":phase_translator.svg"),
            'Time point to phase translator...', self
        )
        self.phaseTranslatorAction.setToolTip(PHASE_TRANSLATOR_TOOLTIP)
        self.phaseTranslatorAction.setStatusTip(
            'Translate the time points of each cell into phases (colored '
            'band below the plot, phase columns in the exported table)'
        )
        self.applyPhasesCellAction = QAction(
            QIcon(":phase_translator_apply.svg"),
            'Evaluate phases on displayed cell', self
        )
        self.applyPhasesCellAction.setShortcut(APPLY_PHASES_CELL_SHORTCUT)
        self.applyPhasesCellAction.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self.applyPhasesCellAction.setToolTip(
            'Evaluate phases on displayed cell\n\n'
            'Apply the rules of the time point to phase translator (and the '
            'custom\n'
            'annotations that override phases) to the displayed cell only, '
            'replacing its\n'
            'phases (like "Apply phase evaluation" on the displayed cell in '
            'the\n'
            'translator). Define the rules first with "Time point to phase '
            'translator...".\n\n'
            'Shortcut: "' + APPLY_PHASES_CELL_SHORTCUT + '"'
        )
        self.applyPhasesCellAction.setStatusTip(
            'Evaluate the phases of the displayed cell with the phase '
            'translator rules'
        )
        self.addAction(self.applyPhasesCellAction)
        # Editable in the shortcuts editor (Edit > Edit keyboard shortcuts)
        self.widgetsWithShortcut['Evaluate phases on displayed cell'] = (
            self.applyPhasesCellAction
        )
        self.maskUnanalyzedFramesAction = QAction(
            'Ignore frames after the last analyzed frame', self
        )
        self.maskUnanalyzedFramesAction.setCheckable(True)
        self.maskUnanalyzedFramesAction.setChecked(self.maskUnanalyzedFrames)
        self.maskUnanalyzedFramesAction.setToolTip(MASK_UNANALYZED_TOOLTIP)
        self.maskUnanalyzedFramesAction.setStatusTip(
            'Calculations use only the frames analyzed in the main GUI '
            '(last cell cycle annotated, else last tracked frame)'
        )

        # Help menu
        self.userGuideAction = QAction(
            QIcon(':browser.svg'), 'Measurements viewer user guide...', self
        )
        self.userGuideAction.setToolTip(
            'Open the measurements viewer documentation in the web browser'
        )
        self.quickStartAction = QAction(
            QIcon(':info.svg'), 'Quick start...', self
        )
        self.quickStartAction.setToolTip(
            'Show a short how-to of the measurements viewer'
        )
        self.acdcDocsAction = QAction('Cell-ACDC documentation...', self)
        # self.editCellConstantsAction = QAction(
        #     QIcon(":calc.svg"), 'Per-cell constants...', self
        # )
        # self.editCellConstantsAction.setToolTip(
        #     'Per-cell constants\n\n'
        #     'Define constants computed once per cell (e.g., "Maximum value" '
        #     'or "Was at beginning")\n'
        #     'that can be used like columns in the time point predictors.'
        # )
        
        self.gui_createCustomAnnotationsActions()
        # The shared mixin uses the main GUI's eye icon
        self.viewAllCustomAnnotAction.setIcon(
            QIcon(":viewCustomAnnotation.svg")
        )
        self.reevaluateAllCustomAnnotationsAction = QAction(
            QIcon(":reevaluateAllCustomAnnotation.svg"),
            'Re-evaluate annotation predictors on all cells', self
        )
        self.reevaluateAllCustomAnnotationsAction.setToolTip(
            'Re-evaluate annotation predictors on all cells\n\n'
            'Run the predictors of all custom annotations on every cell\n'
            'of the current position, replacing their annotations. You\n'
            'will be warned if manual edits would be overwritten.'
        )
        self.reevaluateCellCustomAnnotationsAction = QAction(
            QIcon(":reevaluateCustomAnnotation.svg"),
            'Re-evaluate annotation predictors on displayed cell', self
        )
        self.reevaluateCellCustomAnnotationsAction.setToolTip(
            'Re-evaluate annotation predictors on displayed cell\n\n'
            'Run the predictors of all custom annotations only on\n'
            'the displayed cell, replacing its annotations. You\n'
            'will be warned if manual edits would be overwritten.'
        )
        self.addCustomAnnotationAction.setToolTip(
            'Add custom annotation\n\n'
            'Define an annotation (saved as a 0/1 column) with a symbol, a keyboard\n'
            'shortcut and an optional predictor. Select a cell and press the\n'
            'shortcut to annotate/un-annotate it across its entire video history\n'
            '(Shift+shortcut re-evaluates its predictor on the selected cell).\n'
            'Alternatively, activate the toolbar button and right-click on a cell.'
        )
        
        # Annotations actions (same as in the main GUI)
        self.scalingLabelsAction = widgets_toolbars.CheckableAction(
            'Fix text annotations to screen size'
        )
        self.scalingLabelsAction.setToolTip(
            'When the text annotations are fixed to screen size, they\n'
            'do not scale relative to the objects when zooming in/out.'
        )
        self.scalingLabelsAction.setChecked(
            self._get_setting_value('scalingLabels', False, cast=bool)
        )
        self.highLowResAction = widgets_toolbars.CheckableAction(
            'High resolution text annotations'
        )
        self.highLowResAction.setToolTip(
            'Resolution of the text annotations. High '
            'resolution results in slower update\n'
            'of the annotations. Not recommended with '
            'a number of segmented objects > 500.'
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
        """Create the actions of the 'Viewer' mode tools as in `guiWin`"""
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
        # Required by ZoomGuiElements.zoomToCells, but not shown in the
        # viewer (it never zooms automatically on Next/Previous)
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
            self.launch3dViewerAction, self.MaxProjToggleAction,
            *self.normalizeQActionGroup.actions()
        )

    def gui_createMenuBar(self):
        fileMenu = self.menuBar().addMenu('&File')
        self.fileMenu = fileMenu
        fileMenu.addAction(self.openFolderAction)
        fileMenu.addAction(self.openFileAction)
        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        fileMenu.addAction(self.saveTablesAction)
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
        # enableAutoZoomToCellsAction is not in the menu: the viewer always
        # zooms with enforce=True (the action exists only for the mixin)

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
        annotMenu.setToolTipsVisible(True)
        self.annotationsMenu = annotMenu
        annotMenu.addAction(self.scalingLabelsAction)
        annotMenu.addAction(self.highLowResAction)
        annotMenu.addAction(self.editTextIDsColorAction)
        annotMenu.addAction(self.annotObjTrackSettingsAction)
        annotMenu.addSeparator()
        annotMenu.addAction(self.resetAnnotOptionsAction)
        annotMenu.addSeparator()
        annotMenu.addAction(self.phaseTranslatorAction)
        annotMenu.addAction(self.applyPhasesCellAction)
        annotMenu.addSeparator()
        annotMenu.addAction(self.maskUnanalyzedFramesAction)
        
        helpMenu = self.menuBar().addMenu('&Help')
        self.helpMenu = helpMenu
        helpMenu.addAction(self.userGuideAction)
        helpMenu.addAction(self.quickStartAction)
        helpMenu.addAction(self.acdcDocsAction)
        helpMenu.addSeparator()
        helpMenu.addAction(self.focusOnSearchAction)
        
        self.searchWidget = widgets_search_widgets.ButtonSearchWidget(
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
        # Saving counterpart of loading (only the tables can be edited)
        fileToolBar.addAction(self.saveTablesAction)
        fileToolBar.addAction(self.openFileAction)
        fileToolBar.addAction(self.showInExplorerAction)
        fileToolBar.addAction(self.reloadAction)
        fileToolBar.addSeparator()
        fileToolBar.addAction(self.loadConfigurationAction)
        fileToolBar.addAction(self.saveConfigurationAction)
        fileToolBar.addAction(self.saveConfigurationAsAction)
        self.fileToolBar = fileToolBar
        
        # Navigation toolbar with the 'Viewer' mode tools as in guiWin
        navigateToolBar = widgets_toolbars.ToolBar("Navigation", self)
        navigateToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.addToolBar(navigateToolBar)
        self.navigateToolBar = navigateToolBar
        
        self.timepointsToolBar = widgets_toolbars.ToolBar("Time points", self)
        self.timepointsToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.timepointsToolBar.addAction(self.addTimepointIdentifierAction)
        self.timepointsToolBar.addAction(self.reevaluateAllTimepointsAction)
        self.timepointsToolBar.addAction(self.reevaluateCellTimepointsAction)
        # self.timepointsToolBar.addAction(self.editCellConstantsAction)
        self.timepointsToolBar.addSeparator()
        self.updateReevaluateTimepointsActions()
        self.addToolBar(self.timepointsToolBar)
        
        self.gui_createAnnotateToolbar(area=Qt.TopToolBarArea, visible=True)
        self.annotateToolbar.addAction(
            self.reevaluateAllCustomAnnotationsAction
        )
        self.annotateToolbar.addAction(
            self.reevaluateCellCustomAnnotationsAction
        )
        self.annotateToolbar.addSeparator()
        self.updateReevaluateCustomAnnotationActions()

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
        
        self.overlayButton = widgets_toolbars.rightClickToolButton(parent=self)
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
        
        self.overlayLabelsButton = widgets_toolbars.rightClickToolButton(parent=self)
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
            'Draw a straight line and show its length.\n'
            'Length is displayed on the bottom-right corner.\n\n'
            'Hold "Ctrl" to snap the line to the closest 15° angle.'
        )
        navigateToolBar.addWidget(self.rulerButton)
        self.leftClickButtons.append(self.rulerButton)
        
        self.overlayColorButton = pg.ColorButton(self, color=(230,230,230))
        self.overlayColorButton.setDisabled(True)
        self.overlayColorButton.hide()
        
        self.overlayToolbar = widgets_toolbars.OverlayToolbar(parent=self)
        self.addToolBar(Qt.TopToolBarArea, self.overlayToolbar)
        self.overlayToolbar.setVisible(False)
        # Overlay modes as the user left them (applied by `overlay_cb`)
        self.restoreOverlayModes()

        self.phaseTranslatorToolbar = widgets_toolbars.ToolBar("Phases", self)
        self.phaseTranslatorToolbar.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.phaseTranslatorToolbar.addAction(self.phaseTranslatorAction)
        self.phaseTranslatorToolbar.addAction(self.applyPhasesCellAction)
        self.addToolBar(Qt.TopToolBarArea, self.phaseTranslatorToolbar)
        
        self.setEnabledFileToolbar(False)

    def gui_connectActions(self):
        self.openFolderAction.triggered.connect(self.openFolder)
        self.openFileAction.triggered.connect(self.openFile)
        self.showInExplorerAction.triggered.connect(self.showInExplorer_cb)
        self.reloadAction.triggered.connect(self.reloadButtonClicked)
        self.phaseTranslatorAction.triggered.connect(
            self.configurePhaseTranslator
        )
        self.applyPhasesCellAction.triggered.connect(
            self.applyPhasesOnDisplayedCell
        )
        self.maskUnanalyzedFramesAction.toggled.connect(
            self.maskUnanalyzedFramesToggled
        )
        self.userGuideAction.triggered.connect(self.openUserGuide)
        self.quickStartAction.triggered.connect(self.showQuickStart)
        self.acdcDocsAction.triggered.connect(self.openAcdcDocs)
        self.exitAction.triggered.connect(self.close)
        self.loadConfigurationAction.triggered.connect(self.loadConfiguration)
        self.saveTablesAction.triggered.connect(self.saveTables)
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
        self.reevaluateAllCustomAnnotationsAction.triggered.connect(
            self.reevaluateAllCustomAnnotations
        )
        self.reevaluateCellCustomAnnotationsAction.triggered.connect(
            self.reevaluateDisplayedCellCustomAnnotations
        )
        # self.editCellConstantsAction.triggered.connect(self.editCellConstants)
        self.gui_connectCustomAnnotationsActions()
        
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
        self.invertBwAction.toggled.connect(self.recolorSignalsForBackground)
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
        # Blink like the toolbar button: the focus alone is easy to miss
        self.focusOnSearchAction.triggered.connect(
            partial(self.focusOnSearch, blinking=True)
        )
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
        # Cancel: back to the saved color and LUT of the channel
        self.overlayColorButton.colorDialog.rejected.connect(
            partial(self.overlayColorDialogRejected, self.overlayColorButton)
        )
        self.overlayToolbar.sigSetTranspacency.connect(
            self.setOverlayTransparency
        )
        self.overlayToolbar.sigSetSingleChannel.connect(
            self.setOverlaySingleChannel
        )
        self.overlayToolbar.sigModesClicked.connect(self.saveOverlayModes)

    def setEnabledFileToolbar(self, enabled):
        for action in getattr(self, 'annotationsActions', ()):
            action.setEnabled(enabled)
        for action in getattr(self, 'viewerToolsActions', ()):
            action.setEnabled(enabled)
        self.navigateToolBar.setEnabled(enabled)
        alwaysEnabledActions = (
            self.openFolderAction, self.openFileAction,
            self.loadConfigurationAction
        )
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

    def getHardcodedShortcuts(self):
        """Shortcuts displayed (not editable) in the shortcuts editor, as
        {key sequence: (name, widget)}, the same format as
        `guiWin.getHardcodedShortcuts`.

        Also lists the keys of the time point identifiers and custom
        annotations (edited in their own dialogs), so that the editor flags
        a viewer tool key that collides with one of them."""
        hard_shortcuts = {}
        actions = (
            self.openFolderAction, self.reloadAction, self.saveTablesAction,
            self.saveConfigurationAction, self.saveConfigurationAsAction,
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
            name = text.replace('&', '').strip()
            if name.endswith('...'):
                name = name[:-3]
            hard_shortcuts[shortcut.toString()] = (name, widget)
        items = [
            (f'Time point "{identifier.name}"',
             self.timepointActions.get(identifier.uid),
             self.timepointAutoActions.get(identifier.uid))
            for identifier in self.timepointIdentifiers
        ]
        items.extend(
            (f'Annotation "{getattr(button, "name", "")}"', *actions)
            for button, actions in self.customAnnotShortcutActions.items()
        )
        for name, action, autoAction in items:
            for widget, widgetName in (
                    (action, name), (autoAction, f'{name} (re-evaluate)')):
                if widget is None or widget.shortcut().isEmpty():
                    continue
                hard_shortcuts[widget.shortcut().toString()] = (
                    widgetName, widget
                )
        return hard_shortcuts

    def showInExplorer_cb(self):
        posData = self.posData()
        filesystem.showInExplorer(posData.images_path)

    def openUserGuide(self, checked=False):
        """Open the measurements viewer documentation in the web browser."""
        self.logger.info(
            f'Opening the measurements viewer user guide '
            f'({urls.measurements_viewer_docs_url})...'
        )
        filesystem.browse_url(urls.measurements_viewer_docs_url)

    def openAcdcDocs(self, checked=False):
        self.logger.info(
            f'Opening the Cell-ACDC documentation ({urls.docs_homepage})...'
        )
        filesystem.browse_url(urls.docs_homepage)

    def showQuickStart(self, checked=False):
        """Short how-to (Help menu and info button of the measurements
        panel) with a button that opens the user guide."""
        self.logger.info('Showing the measurements viewer quick start')
        msg = widgets_message_box.myMessageBox()
        msg.information(
            self, 'Measurements viewer: quick start',
            html_utils.paragraph(MEASUREMENTS_VIEWER_QUICK_START),
            url_to_open=urls.measurements_viewer_docs_url,
            open_url_button_text='Open user guide'
        )

    def gui_createGraphicsPlots(self):
        self.graphLayout = pg.GraphicsLayoutWidget()
        # pyqtgraph ignores high-frequency mouse moves --> make sure the
        # last one is delivered (otherwise the hover position, e.g., of
        # the mirrored cursors, is stale when the mouse stops)
        self.mouseMoveTrailingEdgeFilter = widgets_plot_items.MouseMoveTrailingEdgeFilter(
            self.graphLayout
        )
        if self.invertBwAction.isChecked():
            self.graphLayout.setBackground(graphLayoutBkgrColor)
            self.titleColor = 'black'
        else:
            self.graphLayout.setBackground(darkBkgrColor)
            self.titleColor = 'white'
        
        # Layout with the LUT items of the image and overlaid channels
        self.lutItemsLayout = self.graphLayout.addLayout(row=0, col=0)
        
        self.ax1 = widgets_plot_items.MainPlotItem(ax_number=0)
        self.ax1.invertY(True)
        self.ax1.setAspectLocked(True)
        self.ax1.hideAxis('bottom')
        self.ax1.hideAxis('left')
        self.graphLayout.addItem(self.ax1, row=0, col=1)
        
        self.imgGrad = widgets_lut_items.myHistogramLUTitem(parent=self, name='image')
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
        self.labelsGrad = widgets_lut_items.labelsGradientWidget(parent=self)
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
        
        self.img1 = widgets_plot_items.ParentImageItem()
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
        self.ax1_rulerPlotItem = widgets_plot_items.RulerPlotItem(pen=rulerPen)
        self.ax1_rulerAnchorsItem = pg.ScatterPlotItem(
            symbol='o', size=9,
            brush=pg.mkBrush((255,0,0,50)),
            pen=pg.mkPen((255,0,0), width=2), tip=None
        )
        self.ax1.addItem(self.ax1_rulerPlotItem)
        self.ax1.addItem(self.ax1_rulerPlotItem.labelItem)
        self.ax1.addItem(self.ax1_rulerAnchorsItem)
        
        # Mirrored cursor from the slideshow window. Moved at every mouse
        # move --> lightweight item (much faster than a pg.ScatterPlotItem)
        self.ax1_cursor = widgets_plot_items.MirroredCursorItem(
            symbol='d',
            size=16,
            pen=pg.mkPen('w', width=1),
            brush=pg.mkBrush((255, 255, 255, 70)), 
        )
        self.ax1.addItem(self.ax1_cursor, ignoreBounds=True)

        # Overlay channel layers (z-value 0) are added later, when data is
        # loaded, and annotation layers are at 0.5-0.6: keep the ruler and
        # the mirrored cursor above both, below the text IDs (z-value 1).
        # Items created on demand get the same z-value in
        # `raiseOnDemandAx1Items`.
        topItems = (
            self.ax1_rulerPlotItem, self.ax1_rulerPlotItem.labelItem,
            self.ax1_rulerAnchorsItem, self.ax1_cursor,
        )
        for item in topItems:
            item.setZValue(0.9)

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
        """Create the annotations items as in `guiWin` (left image)"""
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
        self.labelsLayerImg1 = widgets_plot_items.BaseLabelsImageItem()
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

        # Overlay channel layers are added later (when data is loaded) with
        # the default z-value 0, which would draw them above these items.
        # Keep annotations above every image layer and below the text IDs
        # (z-value 1, see `gui_annotations`). Custom annotation symbols are
        # at 0.6 (see `addCustomAnnnotScatterPlot`), the ruler and the
        # mirrored slideshow cursor at 0.9 (see `gui_createGraphicsPlots`).
        annotationItems = (
            self.labelsLayerImg1, self.ax1_contoursImageItem,
            self.ax1_oldMothBudLinesItem, self.ax1_newMothBudLinesItem,
            self.ax1_movementAgainstPrevLinesItem,
            self.ax1_movementAgainstPrevScatterItem,
            self.highLightIDLayerImg1, self.searchedIDitemLeft,
        )
        for i, item in enumerate(annotationItems):
            item.setZValue(0.5 + i * 0.01)

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

    def gui_createNavigationWidgets(self):
        self.gui_createFramesNavigationWidgets()
        self.t_label.setText('Frame n.')
        self.gui_createZsliceNavigationWidgets()
        # z-projection lock is a main GUI-only feature
        self.zProjLockViewButton.hide()
        
        self.zSliceOverlay_SB = widgets_numeric_inputs.ScrollBar(Qt.Horizontal)
        self.overlay_z_label = QLabel('Overlay z-slice  ')
        self.overlay_z_label.setFont(_font)
        self.overlay_z_label.setDisabled(True)
        self.zProjOverlay_CB = widgets_combo_boxes.ComboBox()
        self.zProjOverlay_CB.setFont(_font)
        self.zProjOverlay_CB.addItems([
            'single z-slice', 'max z-projection', 'mean z-projection',
            'median z-proj.', 'same as above'
        ])
        self.zProjOverlay_CB.setCurrentIndex(4)
        self.zSliceOverlay_SB.setDisabled(True)
        self.zSliceOverlay_SB.setMinimumWidth(200)
        for widget in self.overlayZsliceWidgets():
            widget.hide()

        self.idScrollBar = widgets_numeric_inputs.ScrollBar(Qt.Horizontal)
        self.idScrollBar.setMinimum(0)
        self.idLabel = QLabel('Cell ID')
        self.idLabel.setFont(_font)
        self.idSpinBox = widgets_numeric_inputs.SpinBox(disableKeyPress=True)
        self.idSpinBox.setMinimum(0)
        
        idSpinBoxHight = int(self.idSpinBox.sizeHint().height() * 0.8)
        self.idScrollBar.setFixedHeight(idSpinBoxHight)
        self.numIDsLabel = QLabel('')
        self.idNotPresentLabel = QLabel('')
        self.idNotPresentLabel.setFont(_font)
        self.followIDcheckbox = QCheckBox('Follow cell')
        self.followIDcheckbox.setToolTip(
            'If active, the view zooms on the selected\n'
            'cell also when changing frame or z-slice'
        )
        self.restrictFramesCheckbox = QCheckBox(
            'Only frames with selected cell'
        )
        self.restrictFramesCheckbox.setToolTip(
            'If active, the frames slider is restricted to the\n'
            'range of frames where the selected cell ID is present'
        )
        self.rescalePlotCheckbox = QCheckBox('Rescale plot')
        self.rescalePlotCheckbox.setToolTip(
            'If active, the plot is rescaled when changing Cell ID'
        )
        
        self.normalizeCheckbox = QCheckBox('Normalize measurements (0-1)')
        self.normalizeCheckbox.setToolTip(
            'Rescale each measurement to the range 0-1 to\n'
            'compare measurements with different scales'
        )
        self.showPhasesCheckbox = QCheckBox('Show phases')
        self.showPhasesCheckbox.setToolTip(
            'Show the evaluated phases (phase column) as a band below the\n'
            'time course and as point colors in Measurement vs Measurement.\n'
            'Set colors in the time point to phase translator.'
        )
        self.showPhasesCheckbox.setChecked(self._get_setting_value(
            'measurements_viewer_show_phases', True, cast=bool
        ))
        self.showPhasesCheckbox.setEnabled(False)
 
        
        self.posLabel = QLabel('Position')
        self.posCombobox = QComboBox()
        # Same font as the other labels of the label column and same look as
        # the other sliders of the slider column
        self.posLabel.setFont(_font)
        self.zSliceCheckbox.setFont(_font)
        self.zSliceOverlay_SB.setFixedHeight(idSpinBoxHight)

        # One left-aligned grid column each for the labels, spin boxes,
        # size labels and sliders (`NAV_*_COL`), so that they line up across
        # rows. The main GUI builds its own navigation layout
        # (gui/main_window/layout.py).
        navLayout = QGridLayout()
        labelCol, spinCol = self.NAV_LABEL_COL, self.NAV_SPINBOX_COL
        sizeCol, sliderCol = self.NAV_SIZE_COL, self.NAV_SLIDER_COL
        extraCol = self.NAV_EXTRA_COL
        row = 0
        navLayout.addWidget(self.posLabel, row, labelCol)
        # Spans to the slider column so that its width does not widen the
        # spin box column
        navLayout.addWidget(
            self.posCombobox, row, spinCol, 1, sliderCol - spinCol + 1,
            alignment=Qt.AlignLeft
        )

        row += 1
        navLayout.addWidget(self.t_label, row, labelCol)
        navLayout.addWidget(self.navSpinBox, row, spinCol)
        navLayout.addWidget(self.navSizeLabel, row, sizeCol)
        navLayout.addWidget(self.navigateScrollBar, row, sliderCol)
        # As the main GUI's last tracked frame label (`updateLastAnalyzedFrameLabel`)
        self.lastAnalyzedFrameLabel = QLabel('')
        self.lastAnalyzedFrameLabel.setFont(_font)
        self.lastAnalyzedFrameLabel.setToolTip(MASK_UNANALYZED_TOOLTIP)
        navLayout.addWidget(
            self.lastAnalyzedFrameLabel, row, extraCol,
            1, self.NAV_NUM_COLS - extraCol
        )

        row += 1
        navLayout.addWidget(self.zSliceCheckbox, row, labelCol)
        navLayout.addWidget(self.zSliceSpinbox, row, spinCol)
        navLayout.addWidget(self.SizeZlabel, row, sizeCol)
        navLayout.addWidget(self.zSliceScrollBar, row, sliderCol)
        navLayout.addWidget(self.zProjComboBox, row, extraCol)
        navLayout.addWidget(self.switchPlaneCombobox, row, extraCol + 1)
        navLayout.addWidget(self.launch3dViewerButton, row, extraCol + 2)

        row += 1
        # No spin box: the label (its text changes with the z-slice, see
        # `gui_overlay`) spans to the slider column
        navLayout.addWidget(
            self.overlay_z_label, row, labelCol, 1, sliderCol - labelCol
        )
        navLayout.addWidget(self.zSliceOverlay_SB, row, sliderCol)
        navLayout.addWidget(self.zProjOverlay_CB, row, extraCol)

        row += 1
        navLayout.addWidget(self.idLabel, row, labelCol)
        navLayout.addWidget(self.idSpinBox, row, spinCol)
        navLayout.addWidget(self.numIDsLabel, row, sizeCol)
        navLayout.addWidget(self.idScrollBar, row, sliderCol)

        # Below the grid, so that the overlay alpha sliders (grid rows added
        # by `addAlphaScrollbar`) follow the other sliders
        idOptionsLayout = QHBoxLayout()
        idOptionsLayout.addWidget(self.followIDcheckbox)
        idOptionsLayout.addWidget(self.restrictFramesCheckbox)
        idOptionsLayout.addWidget(self.rescalePlotCheckbox)

        idOptionsLayout.addWidget(self.normalizeCheckbox)
        idOptionsLayout.addWidget(self.showPhasesCheckbox)
        
        idOptionsLayout.addWidget(self.idNotPresentLabel)
        idOptionsLayout.addStretch(1)

        for scrollbar in self.navigationScrollbars():
            scrollbar.setMinimumWidth(200)

        # Only the sliders stretch
        for col in range(self.NAV_NUM_COLS):
            navLayout.setColumnStretch(col, int(col == sliderCol))

        # Overlay channels alpha scrollbars are added from this row on
        # (see `addAlphaScrollbar`)
        self.bottomLeftLayout = navLayout
        self.firstAlphaScrollbarRow = row + 1
        self.alphaScrollbarRow = row + 1
        self.h = self.navSpinBox.sizeHint().height()

        navGroupLayout = QVBoxLayout()
        navGroupLayout.addLayout(navLayout)
        navGroupLayout.addLayout(idOptionsLayout)
        self.navigationGroupbox = QGroupBox('Navigate')
        self.navigationGroupbox.setLayout(navGroupLayout)

    def overlayZsliceWidgets(self):
        return (self.overlay_z_label, self.zSliceOverlay_SB, self.zProjOverlay_CB)

    def navigationScrollbars(self):
        return (self.navigateScrollBar, self.zSliceScrollBar, self.idScrollBar)

    def gui_createAnnotationsWidgets(self):
        """Create the annotation options checkboxes and the quick settings 
        of `guiWin` relevant for the display of the annotations."""
        self.gui_setupAnnotationOptionsLayout(0)
        
        self.annotLostObjsToggle = widgets_push_buttons.Toggle()
        annotLostObjsToggleTooltip = (
            'Toggle annotation of lost objects mode ON/OFF'
        )
        self.annotLostObjsToggle.setChecked(True)
        self.annotLostObjsToggle.setToolTip(annotLostObjsToggleTooltip)
        annotLostObjsLabel = QLabel('Annot. lost objects')
        annotLostObjsLabel.setToolTip(annotLostObjsToggleTooltip)
        self.annotLostObjsToggle.toggled.connect(self.annotLostObjsToggled)
        
        self.showAllContoursToggle = widgets_push_buttons.Toggle()
        showAllContoursTooltip = (
            'If active, all contours will be displayed, including\n'
            'inner contours (e.g. holes and sub-objects)'
        )
        self.showAllContoursToggle.setToolTip(showAllContoursTooltip)
        showAllContourLabel = QLabel('Show all contours')
        showAllContourLabel.setToolTip(showAllContoursTooltip)
        self.showAllContoursToggle.toggled.connect(
            self.showAllContoursToggled
        )
        
        self.fontSizeSpinBox = widgets_numeric_inputs.SpinBox()
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
        # Extra y axes of the time course on the right (`timeCourseYAxes`)
        self.measurementsPlot = pg.PlotWidget(plotItem=viewer_widgets._TimeCoursePlotItem())
        self.disableUnsupportedPlotTransforms()
        self.disableUnsupportedPlotDownsampling()
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
        # Frames ignored by the calculations (`updateUnanalyzedRegions`),
        # under the curves of every y axis
        self.unanalyzedRegion = self.createUnanalyzedRegion()
        self.measurementsPlot.getPlotItem().addBackgroundItem(
            self.unanalyzedRegion
        )
        # Hovered points of each scatter of the plot, for the joined tip
        # of all y axes (`measurementPointsHovered`)
        self._hoveredMeasurementPoints = {}

        # Solid band of the selected cell's phases below the time course,
        # x-linked to the measurements plot
        self.phaseBandPlot = pg.PlotWidget()
        self.phaseBandPlot.setBackground(
            graphLayoutBkgrColor if self.invertBwAction.isChecked()
            else darkBkgrColor
        )
        self.phaseBandPlot.setFixedHeight(32)
        self.phaseBandPlot.setMouseEnabled(x=False, y=False)
        self.phaseBandPlot.setMenuEnabled(False)
        self.phaseBandPlot.hideButtons()
        self.phaseBandPlot.hideAxis('bottom')
        self.phaseBandPlot.getAxis('left').setStyle(showValues=False)
        self.phaseBandPlot.getAxis('left').setTicks([[]])
        self.phaseBandPlot.getAxis('left').setLabel('Phase')
        self.phaseBandPlot.setYRange(0, 1, padding=0)
        self.phaseBandPlot.setXLink(self.measurementsPlot)
        self.phaseBandFrameLine = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(color=(255, 0, 0), width=2)
        )
        self.phaseBandFrameLine.setZValue(10)
        self.phaseBandPlot.addItem(self.phaseBandFrameLine, ignoreBounds=True)
        self.phaseBandUnanalyzedRegion = self.createUnanalyzedRegion()
        self.phaseBandPlot.addItem(
            self.phaseBandUnanalyzedRegion, ignoreBounds=True
        )
        self.phaseBandItems = []
        self.phaseBandPlot.setVisible(False)
        # Keep the x positions aligned while the main left axis auto-sizes
        # and extra y axes are shown, hidden or resized on the right
        self.measurementsPlot.getAxis('left').geometryChanged.connect(
            self.syncPhaseBandAxisWidth
        )
        self.measurementsPlot.getPlotItem().sigExtraYAxesChanged.connect(
            self.syncPhaseBandAxisWidth
        )
        measurementsPlotLayout = QVBoxLayout()
        measurementsPlotLayout.setContentsMargins(0, 0, 0, 0)
        measurementsPlotLayout.setSpacing(0)
        measurementsPlotLayout.addWidget(self.measurementsPlot)
        measurementsPlotLayout.addWidget(self.phaseBandPlot)
        measurementsPlotContainer = QWidget()
        measurementsPlotContainer.setLayout(measurementsPlotLayout)
        
        self.columnsSearchLineEdit = QLineEdit()
        self.columnsSearchLineEdit.setPlaceholderText('Search column...')
        self.columnsSearchLineEdit.setClearButtonEnabled(True)
        
        self.columnsModel = viewer_widgets.MeasurementTreeModel(self)
        self.columnsSearchProxy = widgets_search_widgets.MeasurementTreeSearchProxy(self)
        self.columnsSearchProxy.setSourceModel(self.columnsModel)
        self.columnsSearchProxy.setDynamicSortFilter(True)
        self.columnsSearchProxy.sort(0, Qt.AscendingOrder)

        self.columnsListWidget = QTreeView()
        self.columnsListWidget.setHeaderHidden(True)
        self.columnsListWidget.setIndentation(6)
        self.columnsListWidget.setUniformRowHeights(True)
        self.columnsListWidget.setModel(self.columnsSearchProxy)
        self.columnsHighlightDelegate = viewer_widgets._MeasurementColumnDelegate(
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
        self.columnsListWidget.setToolTip(viewer_plotting.TIME_COURSE_COLUMNS_TOOLTIP)
        # Custom plotting colors set by the user {column: QColor}
        self.measurementsColors = {}
        # Time course y axis number of the columns not on the default axis
        # 1, also while they are not plotted {column: int >= 2}
        self.timeCourseYAxes = {}
        # Index in SIGNAL_COLORS of the default color of each column that
        # was plotted, so that plotting another column does not recolor
        # the plotted ones {column: int}
        self._defaultSignalColorIndices = {}
        # Ordered rules {column: [{'condition': str, 'color': '#rrggbb'}]}
        self.conditionalFormatting = {}

        self.measurementsModeCombobox = QComboBox()
        self.measurementsModeCombobox.addItems(
            ['Time course', 'Measurement vs Measurement']
        )
        self.measurementsModeCombobox.setToolTip(
            'Time course: selected cell. Measurement '
            'vs Measurement: all cells and all frames\n'
            'of the current position. Hover for details; '
            'click a point to zoom to its cell.'
        )
        # Short how-to with a link to the user guide (also Help menu)
        self.measurementsHelpButton = widgets_push_buttons.infoPushButton()
        self.measurementsHelpButton.setToolTip(
            'How to use the measurements viewer (quick start and link to '
            'the user guide)'
        )
        self.measurementsHelpButton.clicked.connect(self.showQuickStart)
        self.measurementsAxes = {'X': '', 'Y': ''}
        self._measurementsMode = 0
        self._timeCourseColumns = set()
        
        self.refreshButton = widgets_push_buttons.reloadPushButton(' Reload table ')
        self.refreshButton.setToolTip(
            'Reload the acdc_output table from the file '
            '(e.g., after saving in the main GUI)'
        )
        self.processMeasurementsButton = widgets_push_buttons.setPushButton('Process measurements...')
        self.processMeasurementsButton.setToolTip(
            'Apply a processing recipe (e.g., smoothing, min-max normalization or\n'
            'ramp detection) to selected measurements of the current position.\n\n'
            'Each processed measurement is added as '
            'a new column (the measurement name plus\n'
            'the output suffix chosen in the dialog; '
            'the original columns are not changed),\n'
            'which can then be plotted and used by '
            'predictors. The recipe is recorded in\n'
            'the configuration and can be recalculated '
            'when the configuration is loaded.\n\n'
            'Use "Export measurements..." to save the new columns.'
        )
        self.processMeasurementsButton.clicked.connect(self.processMeasurements)
        self.combineMeasurementsButton = widgets_push_buttons.calcPushButton(
            'Combine measurements...'
        )
        self.combineMeasurementsButton.setToolTip(
            'Add columns calculated from the current position.\n'
            'Use Export measurements to save the results.'
        )
        self.combineMeasurementsButton.clicked.connect(self.combineMeasurements)
        self.exportMeasurementsButton = widgets_push_buttons.savePushButton(
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
        modeLayout = QHBoxLayout()
        modeLayout.setContentsMargins(0, 0, 0, 0)
        modeLayout.addWidget(self.measurementsModeCombobox, 1)
        modeLayout.addWidget(self.measurementsHelpButton)
        columnsLayout.addLayout(modeLayout)
        columnsLayout.addWidget(self.timeCourseColumnsWidget)
        columnsLayout.addWidget(self.processMeasurementsButton)
        columnsLayout.addWidget(self.combineMeasurementsButton)
        columnsLayout.addWidget(self.exportMeasurementsButton)
        columnsLayout.addWidget(self.refreshButton)
        columnsWidget = QWidget()
        columnsWidget.setLayout(columnsLayout)
        
        self.measurementsSplitter = QSplitter(Qt.Horizontal)
        self.measurementsSplitter.addWidget(measurementsPlotContainer)
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
        self.resizeBottomLayoutLine = widgets_basic_widgets.VerticalResizeHline()
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
        
        self.bottomScrollArea = widgets_basic_widgets.ScrollArea(resizeVerticalOnShow=True)
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
        # What is loaded (`setStatusBarLabel`). Plain text (folder and file
        # names are not HTML), clipped instead of widening the window (its
        # tool tip lists everything); temporary messages (`showMessage`)
        # hide it while they are shown
        self.statusBarLabel = QLabel('')
        self.statusBarLabel.setTextFormat(Qt.TextFormat.PlainText)
        self.statusBarLabel.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.statusbar.addWidget(self.statusBarLabel, 1)
        # Permanent widget with mouse hover info (as in guiWin). Its text is
        # updated at every mouse move --> fixed size (growing only) to avoid
        # re-layouting the main window
        self.wcLabel = widgets_basic_widgets.AutoGrowFixedSizeLabel('')
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
        # Arrows step to the next/previous existing ID (IDs can have gaps)
        self.idSpinBox.sigUpClicked.connect(partial(self.stepID, 1))
        self.idSpinBox.sigDownClicked.connect(partial(self.stepID, -1))
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
        self.normalizeCheckbox.toggled.connect(self.normalizeToggled)
        self.normalizeCheckbox.toggled.connect(self.plotMeasurements)
        self.showPhasesCheckbox.toggled.connect(self.showPhasesToggled)
        self.measurementsModeCombobox.currentIndexChanged.connect(
            self.measurementsModeChanged
        )
        self.refreshButton.clicked.connect(self.reloadButtonClicked)
