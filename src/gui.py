# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPyTop HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Yeast ACDC GUI for correcting Segmentation and Tracking errors"""
print('Importing modules...')
import sys
import os
import re
import traceback
import time
import datetime
import uuid
from functools import partial
from tqdm import tqdm
import threading, time

import cv2
import math
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.interpolate
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation
from skimage.color import gray2rgb, gray2rgba

from PyQt5.QtCore import (
    Qt, QFile, QTextStream, QSize, QRect, QRectF, QEventLoop, QTimer, QEvent
)
from PyQt5.QtGui import QIcon, QKeySequence, QCursor, QKeyEvent
from PyQt5.QtWidgets import (
    QAction, QApplication, QLabel, QPushButton,
    QMainWindow, QMenu, QToolBar, QGroupBox,
    QScrollBar, QCheckBox, QToolButton, QSpinBox,
    QComboBox, QDial, QButtonGroup, QActionGroup,
    QShortcut, QFileDialog, QDoubleSpinBox
)

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

# NOTE: Enable icons
import qrc_resources

# Custom modules
import load, prompts, apps, core, myutils
from myutils import download_model
from QtDarkMode import breeze_resources

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.yeastacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

print('Initializing...')

# Interpret image data as row-major instead of col-major
pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
np.random.seed(1568)

def qt_debug_trace():
    from PyQt5.QtCore import pyqtRemoveInputHook
    pyqtRemoveInputHook()
    import pdb; pdb.set_trace()

def exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            args[0].logger.exception(e)
    return inner_function

class guiWin(QMainWindow):
    """Main Window."""

    def __init__(self, app, parent=None, buttonToRestore=None, mainWin=None):
        """Initializer."""
        super().__init__(parent)
        self.loadLastSessionSettings()

        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin

        self.app = app
        self.num_screens = len(app.screens())

        # Center main window and determine location of slideshow window
        # depending on number of screens available
        if self.num_screens > 1:
            screen1 = app.screens()[0]
            screen2 = app.screens()[1]
            screen2Center = screen2.size().width()/2
            screen2Left = screen1.size().width()
            self.slideshowWinLeft = int(screen2Left+screen2Center-850/2)
            self.slideshowWinTop = int(screen1.size().height()/2 - 800/2)
        else:
            screen1 = app.screens()[0]
            self.slideshowWinLeft = int(screen1.size().width()-850)
            self.slideshowWinTop = int(screen1.size().height()/2 - 800/2)

        self.slideshowWin = None
        self.segmModel = None
        self.ccaTableWin = None
        self.data_loaded = False
        self.flag = True
        self.countBlinks = 0
        self.setWindowTitle("Yeast ACDC - GUI")
        self.setWindowIcon(QIcon(":assign-motherbud.svg"))
        self.setAcceptDrops(True)

        self.checkableButtons = []
        self.LeftClickButtons = []

        self.isSnapshot = False
        self.pos_i = 0
        self.countKeyPress = 0
        self.xHoverImg, self.yHoverImg = None, None

        # Buttons added to QButtonGroup will be mutually exclusive
        self.checkableQButtonsGroup = QButtonGroup(self)
        self.checkableQButtonsGroup.setExclusive(False)

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()

        self.gui_connectActions()
        self.gui_createStatusBar()

        self.gui_createGraphicsPlots()
        self.gui_addGraphicsItems()

        self.gui_createImg1Widgets()

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 2)
        mainLayout.addLayout(self.img1_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

        self.isEditActionsConnected = False

    def loadLastSessionSettings(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        temp_path = os.path.join(src_path, 'temp')
        csv_path = os.path.join(temp_path, 'settings.csv')
        self.settings_csv_path = csv_path
        if os.path.exists(csv_path):
            self.df_settings = pd.read_csv(csv_path, index_col='setting')
            if 'is_bw_inverted' not in self.df_settings.index:
                self.df_settings.at['is_bw_inverted', 'value'] = 'False'
            if 'fontSize' not in self.df_settings.index:
                self.df_settings.at['fontSize', 'value'] = '10pt'
            if 'overlayColor' not in self.df_settings.index:
                self.df_settings.at['overlayColor', 'value'] = '255-255-0'
            if 'how_normIntensities' not in self.df_settings.index:
                raw = 'Do not normalize. Display raw image'
                self.df_settings.at['how_normIntensities', 'value'] = raw
        else:
            idx = ['is_bw_inverted', 'fontSize', 'overlayColor', 'how_normIntensities']
            values = ['False', '10pt', '255-255-0', 'raw']
            self.df_settings = pd.DataFrame({'setting': idx,
                                             'value': values}
                                           ).set_index('setting')

    def dragEnterEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(file_path):
            exp_path = file_path
            if basename.find('Position_')!=-1 or basename=='Images':
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def dropEvent(self, event):
        event.setDropAction(Qt.CopyAction)
        file_path = event.mimeData().urls()[0].toLocalFile()
        basename = os.path.basename(file_path)
        if os.path.isdir(file_path):
            exp_path = file_path
            self.openFolder(exp_path=exp_path)
        else:
            self.openFile(file_path=file_path)


    def leaveEvent(self, event):
        if self.slideshowWin is not None:
            PosData = self.data[self.pos_i]
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
                self.data_loaded and not
                overlap and not
                PosData.disableAutoActivateViewerWindow
            )

            if autoActivate:
                self.slideshowWin.setFocus(True)
                self.slideshowWin.activateWindow()

    def enterEvent(self, event):
        if self.slideshowWin is not None:
            PosData = self.data[self.pos_i]
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
                self.data_loaded and not
                overlap and not
                PosData.disableAutoActivateViewerWindow
            )

            if autoActivate:
                self.setFocus(True)
                self.activateWindow()

    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        # fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.openFileAction)
        # Open Recent submenu
        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.loadFluoAction)
        # Separator
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        # Edit menu
        editMenu = menuBar.addMenu("&Edit")
        editMenu.addSeparator()
        # Font size
        self.fontSize = self.df_settings.at['fontSize', 'value']
        self.fontSizeMenu = editMenu.addMenu("Font size")
        fontActionGroup = QActionGroup(self)
        fs = int(re.findall('(\d+)pt', self.fontSize)[0])
        for i in range(2,25):
            action = QAction(self)
            action.setText(f'{i}')
            action.setCheckable(True)
            if i == fs:
                action.setChecked(True)
                self.fontSizeAction = action
            fontActionGroup.addAction(action)
            action = self.fontSizeMenu.addAction(action)
        editMenu.addAction(self.editTextIDsColorAction)
        editMenu.addAction(self.editOverlayColorAction)
        editMenu.addAction(self.manuallyEditCcaAction)
        editMenu.addAction(self.enableSmartTrackAction)
        editMenu.addAction(self.enableAutoZoomToCellsAction)


        # Image menu
        ImageMenu = menuBar.addMenu("&Image")
        ImageMenu.addSeparator()
        ImageMenu.addAction(self.imgPropertiesAction)
        filtersMenu = ImageMenu.addMenu("Filters")
        filtersMenu.addAction(self.gaussBlurAction)
        filtersMenu.addAction(self.edgeDetectorAction)
        filtersMenu.addAction(self.entropyFilterAction)
        normalizeIntensitiesMenu = ImageMenu.addMenu("Normalize intensities")
        normalizeIntensitiesMenu.addAction(self.normalizeRawAction)
        normalizeIntensitiesMenu.addAction(self.normalizeToFloatAction)
        # normalizeIntensitiesMenu.addAction(self.normalizeToUbyteAction)
        normalizeIntensitiesMenu.addAction(self.normalizeRescale0to1Action)
        normalizeIntensitiesMenu.addAction(self.normalizeByMaxAction)
        ImageMenu.addAction(self.invertBwAction)

        # Segment menu
        SegmMenu = menuBar.addMenu("&Segment")
        SegmMenu.addSeparator()
        SegmMenu.addAction(self.SegmActionYeaZ)
        SegmMenu.addAction(self.SegmActionCellpose)
        SegmMenu.addAction(self.SegmActionRW)
        SegmMenu.addAction(self.autoSegmAction)

        # Help menu
        helpMenu = menuBar.addMenu(QIcon(":help-content.svg"), "Help")
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addAction(self.aboutAction)

    def gui_createToolBars(self):
        toolbarSize = 34

        # File toolbar
        fileToolBar = self.addToolBar("File")
        # fileToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        fileToolBar.setMovable(False)
        # fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openAction)
        fileToolBar.addAction(self.saveAction)
        fileToolBar.addAction(self.showInExplorerAction)
        fileToolBar.addAction(self.reloadAction)
        fileToolBar.addAction(self.undoAction)
        fileToolBar.addAction(self.redoAction)
        self.fileToolBar = fileToolBar
        self.setEnabledFileToolbar(False)

        self.undoAction.setEnabled(False)
        self.redoAction.setEnabled(False)

        # Navigation toolbar
        navigateToolBar = QToolBar("Navigation", self)
        # navigateToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(navigateToolBar)
        navigateToolBar.addAction(self.prevAction)
        navigateToolBar.addAction(self.nextAction)

        self.slideshowButton = QToolButton(self)
        self.slideshowButton.setIcon(QIcon(":eye-plus.svg"))
        self.slideshowButton.setCheckable(True)
        self.slideshowButton.setShortcut('Ctrl+W')
        self.slideshowButton.setToolTip('Open slideshow (Ctrl+W)')
        navigateToolBar.addWidget(self.slideshowButton)

        self.overlayButton = QToolButton(self)
        self.overlayButton.setIcon(QIcon(":overlay.svg"))
        self.overlayButton.setCheckable(True)
        self.overlayButton.setToolTip('Overlay fluorescent image\n'
        'NOTE: Button has a green background if you successfully loaded fluorescent data\n\n'
        'If you need to overlay a different channel load (or load it again)\n'
        'from "File --> Load fluorescent images" menu.')
        navigateToolBar.addWidget(self.overlayButton)
        # self.checkableButtons.append(self.overlayButton)
        # self.checkableQButtonsGroup.addButton(self.overlayButton)

        self.rulerButton = QToolButton(self)
        self.rulerButton.setIcon(QIcon(":ruler.svg"))
        self.rulerButton.setCheckable(True)
        self.rulerButton.setToolTip(
            'Measure straight line. '
            'Length is displayed on the bottom-right corner.'
        )
        navigateToolBar.addWidget(self.rulerButton)
        self.checkableButtons.append(self.rulerButton)
        self.LeftClickButtons.append(self.rulerButton)

        # fluorescent image color widget
        colorsToolBar = QToolBar("Colors", self)

        self.overlayColorButton = pg.ColorButton(self, color=(230,230,230))
        # self.overlayColorButton.mousePressEvent = self.mousePressColorButton
        self.overlayColorButton.setDisabled(True)
        colorsToolBar.addWidget(self.overlayColorButton)

        self.textIDsColorButton = pg.ColorButton(self)
        colorsToolBar.addWidget(self.textIDsColorButton)

        self.addToolBar(colorsToolBar)
        colorsToolBar.setVisible(False)

        self.navigateToolBar = navigateToolBar

        # cca toolbar
        ccaToolBar = QToolBar("Cell cycle annotations", self)
        self.addToolBar(ccaToolBar)

        # Assign mother to bud button
        self.assignBudMothButton = QToolButton(self)
        self.assignBudMothButton.setIcon(QIcon(":assign-motherbud.svg"))
        self.assignBudMothButton.setCheckable(True)
        self.assignBudMothButton.setShortcut('a')
        self.assignBudMothButton.setVisible(False)
        self.assignBudMothButton.setToolTip(
            'Toggle "Assign bud to mother cell" mode ON/OFF\n\n'
            'ACTION: press with right button on bud and release on mother '
            '(right-click drag-and-drop)\n\n'
            'SHORTCUT: "A" key'
        )
        ccaToolBar.addWidget(self.assignBudMothButton)
        self.checkableButtons.append(self.assignBudMothButton)
        self.checkableQButtonsGroup.addButton(self.assignBudMothButton)

        # Set is_history_known button
        self.setIsHistoryKnownButton = QToolButton(self)
        self.setIsHistoryKnownButton.setIcon(QIcon(":history.svg"))
        self.setIsHistoryKnownButton.setCheckable(True)
        self.setIsHistoryKnownButton.setShortcut('u')
        self.setIsHistoryKnownButton.setVisible(False)
        self.setIsHistoryKnownButton.setToolTip(
            'Toggle "Annotate that a cell has an unknown history" mode ON/OFF\n\n'
            'EXAMPLE: useful for cells appearing from outside of the field of view\n\n'
            'ACTION: left-click on cell\n\n'
            'SHORTCUT: "U" key'
        )
        ccaToolBar.addWidget(self.setIsHistoryKnownButton)
        self.checkableButtons.append(self.setIsHistoryKnownButton)
        self.checkableQButtonsGroup.addButton(self.setIsHistoryKnownButton)

        ccaToolBar.addAction(self.reInitCcaAction)
        ccaToolBar.setVisible(False)
        self.ccaToolBar = ccaToolBar

        # Edit toolbar
        editToolBar = QToolBar("Edit", self)
        self.addToolBar(editToolBar)

        self.brushButton = QToolButton(self)
        self.brushButton.setIcon(QIcon(":brush.svg"))
        self.brushButton.setCheckable(True)
        self.brushButton.setToolTip(
            'Edit segmentation labels with a circular brush.\n'
            'Increase brush size with UP/DOWN arrows on the keyboard.\n\n'
            'Default behaviour:\n\n'
            '   - Painting on the background will create a new label.\n'
            '   - Edit an existing label by starting to paint on the label\n'
            '     (brush cursor changes color when hovering an existing label).\n'
            '   - Painting is always UNDER existing labels, unless you press\n'
            '     "b" key twice quickly. If double-press is successfull, '
            '     then brush button is red and brush cursor always white.\n\n'
            'SHORTCUT: "B" key')
        editToolBar.addWidget(self.brushButton)
        self.checkableButtons.append(self.brushButton)
        self.LeftClickButtons.append(self.brushButton)

        self.eraserButton = QToolButton(self)
        self.eraserButton.setIcon(QIcon(":eraser.png"))
        self.eraserButton.setCheckable(True)
        self.eraserButton.setToolTip(
            'Erase segmentation labels with a circular eraser.\n'
            'Increase eraser size with UP/DOWN arrows on the keyboard.\n\n'
            'Default behaviour:\n\n'
            '   - Starting to erase from the background (cursor is a red circle)\n '
            '     will erase any labels you hover above.\n'
            '   - Starting to erase from a specific label will erase only that label\n'
            '     (cursor is a circle with the color of the label).'
            '   - To enforce erasing all labels no matter where you start from '
            '     double-press "X" key. If double-press is successfull, '
            '     then eraser button is red and eraser cursor always red.\n\n'
            'SHORTCUT: "X" key')
        editToolBar.addWidget(self.eraserButton)
        self.checkableButtons.append(self.eraserButton)
        self.LeftClickButtons.append(self.eraserButton)

        self.curvToolButton = QToolButton(self)
        self.curvToolButton.setIcon(QIcon(":curvature-tool.svg"))
        self.curvToolButton.setCheckable(True)
        self.curvToolButton.setShortcut('c')
        self.curvToolButton.setToolTip(
            'Toggle "Curvature tool" ON/OFF\n\n'
            'ACTION: left-clicks for manual spline anchors,\n'
            'right button for drawing auto-contour\n\n'
            'SHORTCUT: "C" key')
        editToolBar.addWidget(self.curvToolButton)
        self.LeftClickButtons.append(self.curvToolButton)
        # self.checkableButtons.append(self.curvToolButton)

        self.hullContToolButton = QToolButton(self)
        self.hullContToolButton.setIcon(QIcon(":hull.svg"))
        self.hullContToolButton.setCheckable(True)
        self.hullContToolButton.setShortcut('f')
        self.hullContToolButton.setToolTip(
            'Toggle "Hull contour tool" ON/OFF\n\n'
            'ACTION: right-click on a cell to replace it with its hull contour.\n'
            'Use it to fill cracks and holes.\n\n'
            'SHORTCUT: "F" key')
        editToolBar.addWidget(self.hullContToolButton)
        self.checkableButtons.append(self.hullContToolButton)

        self.editID_Button = QToolButton(self)
        self.editID_Button.setIcon(QIcon(":edit-id.svg"))
        self.editID_Button.setCheckable(True)
        self.editID_Button.setShortcut('n')
        self.editID_Button.setToolTip(
            'Toggle "Edit ID" mode ON/OFF\n\n'
            'EXAMPLE: manually change ID of a cell\n\n'
            'ACTION: right-click on cell\n\n'
            'SHORTCUT: "N" key')
        editToolBar.addWidget(self.editID_Button)
        self.checkableButtons.append(self.editID_Button)
        self.checkableQButtonsGroup.addButton(self.editID_Button)

        self.separateBudButton = QToolButton(self)
        self.separateBudButton.setIcon(QIcon(":separate-bud.svg"))
        self.separateBudButton.setCheckable(True)
        self.separateBudButton.setShortcut('s')
        self.separateBudButton.setToolTip(
            'Toggle "Automatic/manual separation" mode ON/OFF\n\n'
            'EXAMPLE: separate mother-bud fused together\n\n'
            'ACTION: right-click for automatic and left-click for manual\n\n'
            'SHORTCUT: "S" key'
        )
        editToolBar.addWidget(self.separateBudButton)
        self.checkableButtons.append(self.separateBudButton)
        self.checkableQButtonsGroup.addButton(self.separateBudButton)

        self.mergeIDsButton = QToolButton(self)
        self.mergeIDsButton.setIcon(QIcon(":merge-IDs.svg"))
        self.mergeIDsButton.setCheckable(True)
        self.mergeIDsButton.setShortcut('m')
        self.mergeIDsButton.setToolTip(
            'Toggle "Merge IDs" mode ON/OFF\n\n'
            'EXAMPLE: merge/fuse two cells together\n\n'
            'ACTION: right-click\n\n'
            'SHORTCUT: "M" key'
        )
        editToolBar.addWidget(self.mergeIDsButton)
        self.checkableButtons.append(self.mergeIDsButton)
        self.checkableQButtonsGroup.addButton(self.mergeIDsButton)

        self.binCellButton = QToolButton(self)
        self.binCellButton.setIcon(QIcon(":bin.svg"))
        self.binCellButton.setCheckable(True)
        self.binCellButton.setToolTip(
            'Toggle "Annotate cell as removed from analysis" mode ON/OFF\n\n'
            'EXAMPLE: annotate that a cell is removed from downstream analysis.\n'
            '"is_cell_excluded" set to True in acdc_output.csv table\n\n'
            'ACTION: right-click\n\n'
            'SHORTCUT: "R" key'
        )
        self.binCellButton.setShortcut("r")
        editToolBar.addWidget(self.binCellButton)
        self.checkableButtons.append(self.binCellButton)
        self.checkableQButtonsGroup.addButton(self.binCellButton)

        self.ripCellButton = QToolButton(self)
        self.ripCellButton.setIcon(QIcon(":rip.svg"))
        self.ripCellButton.setCheckable(True)
        self.ripCellButton.setToolTip(
            'Toggle "Annotate cell as dead" mode ON/OFF\n\n'
            'EXAMPLE: annotate that a cell is dead.\n'
            '"is_cell_dead" set to True in acdc_output.csv table\n\n'
            'ACTION: right-click\n\n'
            'SHORTCUT: "D" key'
        )
        self.ripCellButton.setShortcut("d")
        editToolBar.addWidget(self.ripCellButton)
        self.checkableButtons.append(self.ripCellButton)
        self.checkableQButtonsGroup.addButton(self.ripCellButton)

        editToolBar.addAction(self.addDelRoiAction)

        editToolBar.addAction(self.repeatTrackingAction)

        # Widgets toolbar
        widgetsToolBar = QToolBar("Widgets", self)
        self.addToolBar(widgetsToolBar)

        self.disableTrackingCheckBox = QCheckBox("Disable tracking")
        self.disableTrackingCheckBox.setLayoutDirection(Qt.RightToLeft)
        self.disableTrackingAction = widgetsToolBar.addWidget(
                                            self.disableTrackingCheckBox)
        self.disableTrackingAction.setVisible(False)

        self.brushSizeSpinbox = QSpinBox()
        self.brushSizeSpinbox.setValue(4)
        brushSizeLabel = QLabel('   Size: ')
        brushSizeLabel.setBuddy(self.brushSizeSpinbox)
        self.brushSizeLabelAction = widgetsToolBar.addWidget(brushSizeLabel)
        self.brushSizeAction = widgetsToolBar.addWidget(self.brushSizeSpinbox)
        self.brushSizeLabelAction.setVisible(False)
        self.brushSizeAction.setVisible(False)

        widgetsToolBar.setVisible(False)
        self.widgetsToolBar = widgetsToolBar

        # Edit toolbar
        modeToolBar = QToolBar("Mode", self)
        self.addToolBar(modeToolBar)

        self.modeComboBox = QComboBox()
        self.modeComboBox.addItems(['Segmentation and Tracking',
                                    'Cell cycle analysis',
                                    'Viewer'])
        self.modeComboBoxLabel = QLabel('    Mode: ')
        self.modeComboBoxLabel.setBuddy(self.modeComboBox)
        modeToolBar.addWidget(self.modeComboBoxLabel)
        modeToolBar.addWidget(self.modeComboBox)
        modeToolBar.setVisible(False)


        self.modeToolBar = modeToolBar
        self.editToolBar = editToolBar
        self.editToolBar.setVisible(False)
        self.navigateToolBar.setVisible(False)

        # toolbarSize = 58
        # fileToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # navigateToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # ccaToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # editToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # widgetsToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        # modeToolBar.setIconSize(QSize(toolbarSize, toolbarSize))

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Temporary message
        self.statusbar.showMessage("Ready", 3000)
        # Permanent widget
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)

    def gui_createActions(self):
        # File actions
        self.newAction = QAction(self)
        self.newAction.setText("&New")
        self.newAction.setIcon(QIcon(":file-new.svg"))
        self.openAction = QAction(QIcon(":folder-open.svg"), "&Open folder...", self)
        self.openFileAction = QAction(QIcon(":image.svg"),"&Open image/video file...", self)
        self.saveAction = QAction(QIcon(":file-save.svg"),
                                  "&Save (Ctrl+S)", self)
        self.loadFluoAction = QAction("Load fluorescent images", self)
        self.reloadAction = QAction(QIcon(":reload.svg"),
                                          "Reload segmentation file", self)
        self.showInExplorerAction = QAction(QIcon(":drawer.svg"),
                                    "&Show in Explorer/Finder", self)
        self.exitAction = QAction("&Exit", self)
        self.undoAction = QAction(QIcon(":undo.svg"), "Undo (Ctrl+Z)", self)
        self.redoAction = QAction(QIcon(":redo.svg"), "Redo (Ctrl+Y)", self)
        # String-based key sequences
        self.newAction.setShortcut("Ctrl+N")
        self.openAction.setShortcut("Ctrl+O")
        self.saveAction.setShortcut("Ctrl+S")
        self.undoAction.setShortcut("Ctrl+Z")
        self.redoAction.setShortcut("Ctrl+Y")
        # Help tips
        newTip = "Create a new file"
        self.newAction.setStatusTip(newTip)
        self.newAction.setToolTip(newTip)
        self.newAction.setWhatsThis("Create a new and empty text file")
        # Edit actions
        self.SegmActionYeaZ = QAction("YeaZ...", self)
        self.SegmActionCellpose = QAction("Cellpose...", self)
        self.SegmActionRW = QAction("Random walker...", self)
        self.SegmActionYeaZ.setDisabled(True)
        self.SegmActionCellpose.setDisabled(True)
        self.SegmActionRW.setDisabled(True)

        self.prevAction = QAction(QIcon(":arrow-left.svg"),
                                        "Previous frame", self)
        self.nextAction = QAction(QIcon(":arrow-right.svg"),
                                  "Next Frame", self)
        self.prevAction.setShortcut("left")
        self.nextAction.setShortcut("right")

        self.repeatTrackingAction = QAction(
            QIcon(":repeat-tracking.svg"), "Repeat tracking", self
        )
        # Standard key sequence
        # self.copyAction.setShortcut(QKeySequence.Copy)
        # self.pasteAction.setShortcut(QKeySequence.Paste)
        # self.cutAction.setShortcut(QKeySequence.Cut)
        # Help actions
        self.helpContentAction = QAction("Help Content...", self)
        self.aboutAction = QAction("&About...", self)


        self.reInitCcaAction = QAction(self)
        self.reInitCcaAction.setIcon(QIcon(":reinitCca.svg"))
        self.reInitCcaAction.setVisible(False)
        self.reInitCcaAction.setToolTip(
            'Reinitialize cell cycle annotations table from this frame onward.\n'
            'NOTE: This will erase all the already annotated future frames information\n'
            '(from the current session not the saved information)'
        )

        self.editTextIDsColorAction = QAction('Edit text on IDs color...', self)
        self.editTextIDsColorAction.setDisabled(True)

        self.editOverlayColorAction = QAction('Edit overlay color...', self)
        self.editOverlayColorAction.setDisabled(True)

        self.manuallyEditCcaAction = QAction(
                            'Manually modify cell cycle annotations...', self)
        self.manuallyEditCcaAction.setDisabled(True)

        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        checked = self.df_settings.at['is_bw_inverted', 'value'] == 'True'
        self.invertBwAction.setChecked(checked)

        self.normalizeRawAction = QAction(
            'Do not normalize. Display raw image', self)
        self.normalizeToFloatAction = QAction(
            'Convert to floating point format with values [0, 1]', self)
        # self.normalizeToUbyteAction = QAction(
        #     'Rescale to 8-bit unsigned integer format with values [0, 255]', self)
        self.normalizeRescale0to1Action = QAction(
            'Rescale to [0, 1]', self)
        self.normalizeByMaxAction = QAction(
            'Normalize by max value', self)
        self.normalizeRawAction.setCheckable(True)
        self.normalizeToFloatAction.setCheckable(True)
        # self.normalizeToUbyteAction.setCheckable(True)
        self.normalizeRescale0to1Action.setCheckable(True)
        self.normalizeByMaxAction.setCheckable(True)
        self.normalizeQActionGroup = QActionGroup(self)
        self.normalizeQActionGroup.addAction(self.normalizeRawAction)
        self.normalizeQActionGroup.addAction(self.normalizeToFloatAction)
        # self.normalizeQActionGroup.addAction(self.normalizeToUbyteAction)
        self.normalizeQActionGroup.addAction(self.normalizeRescale0to1Action)
        self.normalizeQActionGroup.addAction(self.normalizeByMaxAction)

        self.setLastUserNormAction()

        self.autoSegmAction = QAction(
            'Enable automatic segmentation', self)
        self.autoSegmAction.setCheckable(True)
        self.autoSegmAction.setDisabled(True)

        self.enableSmartTrackAction = QAction(
            'Smart handling of enabling/disabling tracking', self)
        self.enableSmartTrackAction.setCheckable(True)
        self.enableSmartTrackAction.setChecked(True)

        self.enableAutoZoomToCellsAction = QAction(
            'Automatic zoom to all cells when pressing "Next/Previous"', self)
        self.enableAutoZoomToCellsAction.setCheckable(True)

        self.gaussBlurAction = QAction('Gaussian blur...', self)
        self.gaussBlurAction.setCheckable(True)

        self.imgPropertiesAction = QAction('Properties...', self)
        self.imgPropertiesAction.setDisabled(True)

        self.edgeDetectorAction = QAction('Edge detection...', self)
        self.edgeDetectorAction.setCheckable(True)

        self.entropyFilterAction = QAction('Object detection...', self)
        self.entropyFilterAction.setCheckable(True)

        self.addDelRoiAction = QAction(self)
        self.addDelRoiAction.setIcon(QIcon(":addDelRoi.svg"))
        self.addDelRoiAction.setToolTip(
            'Add resizable rectangle. Every ID touched by the rectangle will be '
            'automaticaly deleted.\n '
            'Moving the rectangle will restore deleted IDs if they are not '
            'touched by it anymore.\n'
            'To delete rectangle right-click on it --> remove.')

    def gui_connectActions(self):
        # Connect File actions
        self.newAction.triggered.connect(self.newFile)
        self.openAction.triggered.connect(self.openFolder)
        self.openFileAction.triggered.connect(self.openFile)
        self.saveAction.triggered.connect(self.saveFile)
        self.showInExplorerAction.triggered.connect(self.showInExplorer)
        self.exitAction.triggered.connect(self.close)

        self.undoAction.triggered.connect(self.undo)
        self.redoAction.triggered.connect(self.redo)

        # Connect Help actions
        self.helpContentAction.triggered.connect(self.helpContent)
        self.aboutAction.triggered.connect(self.about)
        # Connect Open Recent to dynamically populate it
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)

        self.checkableQButtonsGroup.buttonClicked.connect(self.uncheckQButton)

    def gui_connectEditActions(self):
        self.showInExplorerAction.setEnabled(True)
        self.navigateToolBar.setVisible(True)
        self.setEnabledFileToolbar(True)
        self.loadFluoAction.setEnabled(True)
        self.isEditActionsConnected = True
        self.fontSizeMenu.triggered.connect(self.changeFontSize)
        self.prevAction.triggered.connect(self.prev_cb)
        self.nextAction.triggered.connect(self.next_cb)
        self.overlayButton.toggled.connect(self.overlay_cb)
        self.rulerButton.toggled.connect(self.ruler_cb)
        self.loadFluoAction.triggered.connect(self.loadFluo_cb)
        self.reloadAction.triggered.connect(self.reload_cb)
        self.slideshowButton.toggled.connect(self.launchSlideshow)
        self.SegmActionYeaZ.triggered.connect(self.repeatSegmYeaZ)
        self.SegmActionCellpose.triggered.connect(self.repeatSegmCellpose)
        self.SegmActionRW.triggered.connect(self.randomWalkerSegm)
        self.autoSegmAction.toggled.connect(self.autoSegm_cb)
        self.disableTrackingCheckBox.clicked.connect(self.disableTracking)
        self.repeatTrackingAction.triggered.connect(self.repeatTracking)
        self.brushButton.toggled.connect(self.Brush_cb)
        self.eraserButton.toggled.connect(self.Eraser_cb)
        self.curvToolButton.toggled.connect(self.curvTool_cb)
        self.reInitCcaAction.triggered.connect(self.reInitCca)
        # self.repeatAutoCcaAction.triggered.connect(self.repeatAutoCca)
        self.manuallyEditCcaAction.triggered.connect(self.manualEditCca)
        self.invertBwAction.toggled.connect(self.invertBw)
        self.enableSmartTrackAction.toggled.connect(self.enableSmartTrack)
        # Brush/Eraser size action
        self.brushSizeSpinbox.valueChanged.connect(self.brushSize_cb)
        # Mode
        self.modeComboBox.currentIndexChanged.connect(self.changeMode)
        self.modeComboBox.activated.connect(self.clearComboBoxFocus)
        self.equalizeHistPushButton.clicked.connect(self.equalizeHist)
        self.editOverlayColorAction.triggered.connect(self.toggleOverlayColorButton)
        self.editTextIDsColorAction.triggered.connect(self.toggleTextIDsColorButton)
        self.overlayColorButton.sigColorChanging.connect(self.updateOlColors)
        self.textIDsColorButton.sigColorChanging.connect(self.updateTextIDsColors)
        self.textIDsColorButton.sigColorChanged.connect(self.saveTextIDsColors)
        self.alphaScrollBar.valueChanged.connect(self.updateOverlay)
        # Drawing mode
        self.drawIDsContComboBox.currentIndexChanged.connect(
                                                self.drawIDsContComboBox_cb)
        self.drawIDsContComboBox.activated.connect(self.clearComboBoxFocus)
        self.gaussBlurAction.toggled.connect(self.gaussBlur)
        self.edgeDetectorAction.toggled.connect(self.edgeDetection)
        self.entropyFilterAction.toggled.connect(self.entropyFilter)
        self.addDelRoiAction.triggered.connect(self.addDelROI)
        self.hist.sigLookupTableChanged.connect(self.histLUT_cb)
        self.normalizeQActionGroup.triggered.connect(self.saveNormAction)
        self.imgPropertiesAction.triggered.connect(self.editImgProperties)

    def gui_createImg1Widgets(self):
        self.img1_Widglayout = QtGui.QGridLayout()

        # Toggle contours/ID comboboxf
        row = 0
        self.drawIDsContComboBoxSegmItems = ['Draw IDs and contours',
                                             'Draw only cell cycle info',
                                             'Draw cell cycle info and contours',
                                             'Draw only mother-bud lines',
                                             'Draw only IDs',
                                             'Draw only contours',
                                             'Draw nothing']
        self.drawIDsContComboBoxCcaItems = ['Draw only cell cycle info',
                                            'Draw cell cycle info and contours',
                                            'Draw only mother-bud lines',
                                            'Draw IDs and contours',
                                            'Draw only IDs',
                                            'Draw only contours',
                                            'Draw nothing']
        self.drawIDsContComboBox = QComboBox()
        self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
        # Always adjust combobox width to largest item
        self.drawIDsContComboBox.setSizeAdjustPolicy(
                    self.drawIDsContComboBox.AdjustToContents)

        self.img1_Widglayout.addWidget(
                self.drawIDsContComboBox, row, 1, 1, 10,
                alignment=Qt.AlignCenter)

        # Frames scrollbar
        row += 1
        self.framesScrollBar = QScrollBar(Qt.Horizontal)
        self.framesScrollBar.setDisabled(True)
        self.framesScrollBar.setMinimum(1)
        self.framesScrollBar.setMaximum(1)
        self.framesScrollBar.setToolTip(
            'NOTE: The maximum frame number that can be visualized with this '
            'scrollbar\n'
            'is the last visited frame with the selected mode\n'
            '(see "Mode" selector on the top-right).\n\n'
            'If the scrollbar does not move it means that you never visited\n'
            'any frame with current mode.\n\n'
            'Note that the "Viewer" mode allows you to scroll ALL frames.'
        )
        t_label = QLabel('frame n.  ')
        _font = QtGui.QFont()
        _font.setPointSize(10)
        t_label.setFont(_font)
        self.t_label = t_label
        self.img1_Widglayout.addWidget(
                t_label, row, 0, alignment=Qt.AlignRight)
        self.img1_Widglayout.addWidget(
                self.framesScrollBar, row, 1, 1, 10)
        self.t_label.hide()
        self.framesScrollBar.hide()

        # z-slice scrollbar
        row += 1
        self.zSliceScrollBar = QScrollBar(Qt.Horizontal)
        _z_label = QLabel('z-slice  ')
        _font = QtGui.QFont()
        _font.setPointSize(10)
        _z_label.setFont(_font)
        self.z_label = _z_label
        self.img1_Widglayout.addWidget(
                _z_label, row, 0, alignment=Qt.AlignRight)
        self.img1_Widglayout.addWidget(
                self.zSliceScrollBar, row, 1, 1, 10)

        self.zProjComboBox = QComboBox()
        self.zProjComboBox.addItems(['single z-slice',
                                     'max z-projection',
                                     'mean z-projection',
                                     'median z-proj.'])

        self.img1_Widglayout.addWidget(self.zProjComboBox, row, 11, 1, 1)

        self.enableZstackWidgets(False)

        # Fluorescent overlay alpha
        row += 2
        alphaScrollBar_label = QLabel('Overlay alpha  ')
        alphaScrollBar_label.setFont(_font)
        alphaScrollBar = QScrollBar(Qt.Horizontal)
        alphaScrollBar.setMinimum(0)
        alphaScrollBar.setMaximum(40)
        alphaScrollBar.setValue(20)
        alphaScrollBar.setToolTip(
            'Control the alpha value of the overlay.\n'
            'alpha=0 results in NO overlay,\n'
            'alpha=1 results in only fluorescent data visible'
        )
        alphaScrollBar.setDisabled(True)
        self.alphaScrollBar = alphaScrollBar
        self.img1_Widglayout.addWidget(
            alphaScrollBar_label, row, 0, alignment=Qt.AlignRight
        )
        self.img1_Widglayout.addWidget(
            alphaScrollBar, row, 1, 1, 10
        )
        self.alphaScrollBar_label = alphaScrollBar_label
        self.alphaScrollBar.hide()
        self.alphaScrollBar_label.hide()

        # Left, top, right, bottom
        self.img1_Widglayout.setContentsMargins(80, 0, 0, 0)

    def gui_createGraphicsPlots(self):
        self.graphLayout = pg.GraphicsLayoutWidget()

        # Left plot
        self.ax1 = pg.PlotItem()
        self.ax1.invertY(True)
        self.ax1.setAspectLocked(True)
        self.ax1.hideAxis('bottom')
        self.ax1.hideAxis('left')
        self.graphLayout.addItem(self.ax1, row=1, col=1)

        # Right plot
        self.ax2 = pg.PlotItem()
        self.ax2.setAspectLocked(True)
        self.ax2.invertY(True)
        self.ax2.hideAxis('bottom')
        self.ax2.hideAxis('left')
        self.graphLayout.addItem(self.ax2, row=1, col=2)

    def gui_addGraphicsItems(self):
        # Auto image adjustment button
        proxy = QtGui.QGraphicsProxyWidget()
        equalizeHistPushButton = QPushButton("Auto")
        equalizeHistPushButton.setStyleSheet(
               'QPushButton {background-color: #282828; color: #F0F0F0;}')
        proxy.setWidget(equalizeHistPushButton)
        self.graphLayout.addItem(proxy, row=0, col=0)
        self.equalizeHistPushButton = equalizeHistPushButton

        # Left image histogram
        self.hist = pg.HistogramLUTItem()
        # Disable histogram default context Menu event
        self.hist.vb.raiseContextMenu = lambda x: None
        self.graphLayout.addItem(self.hist, row=1, col=0)

        # Title
        self.titleLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.titleLabel.setText(
            'File --> Open or Open recent to start the process')
        self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=2)

        # # Current frame text
        # self.frameLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        # self.frameLabel.setText(' ')
        # self.graphLayout.addItem(self.frameLabel, row=2, col=1, colspan=2)

    def gui_addPlotItems(self):
        if 'textIDsColor' in self.df_settings.index:
            rgbString = self.df_settings.at['textIDsColor', 'value']
            try:
                r, g, b = re.findall('(\d+), (\d+), (\d+)', rgbString)[0]
                r, g, b = int(r), int(g), int(b)
            except TypeError:
                r, g, b = 255, 255, 255
            self.ax1_oldIDcolor = (r, g, b)
            self.ax1_S_oldCellColor = (int(r*0.9), int(r*0.9), int(r*0.9))
            self.ax1_G1cellColor = (int(r*0.8), int(r*0.8), int(r*0.8), 178)
            self.textIDsColorButton.setColor((r, g, b))
        else:
            self.ax1_oldIDcolor = (255, 255, 255) # white
            self.ax1_S_oldCellColor = (229, 229, 229)
            self.ax1_G1cellColor = (204, 204, 204, 178)

        # Blank image
        self.blank = np.zeros((256,256), np.uint8)

        # Left image
        self.img1 = pg.ImageItem(self.blank)
        self.ax1.addItem(self.img1)
        for action in self.hist.gradient.menu.actions():
            try:
                action.name
                action.triggered.disconnect()
                action.triggered.connect(self.gradientContextMenuClicked)
            except AttributeError:
                pass

        # Right image
        self.img2 = pg.ImageItem(self.blank)
        self.ax2.addItem(self.img2)

        # Brush circle img1
        self.ax1_BrushCircle = pg.ScatterPlotItem()
        self.ax1_BrushCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=pg.mkBrush((255,255,255,50)),
                                 pen=pg.mkPen(width=2))
        self.ax1.addItem(self.ax1_BrushCircle)

        # Eraser circle img2
        self.ax2_EraserCircle = pg.ScatterPlotItem()
        self.eraserCirclePen = pg.mkPen(width=1.5, color='r')
        self.ax2_EraserCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=None,
                                 pen=self.eraserCirclePen)
        self.ax2.addItem(self.ax2_EraserCircle)
        self.ax2_EraserX = pg.ScatterPlotItem()
        self.ax2_EraserX.setData([], [], symbol='x', pxMode=False, size=3,
                                      brush=pg.mkBrush(color=(255,0,0,50)),
                                      pen=pg.mkPen(width=1.5, color='r'))
        self.ax2.addItem(self.ax2_EraserX)

        # Eraser circle img1
        self.ax1_EraserCircle = pg.ScatterPlotItem()
        self.ax1_EraserCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=None,
                                 pen=self.eraserCirclePen)
        self.ax1.addItem(self.ax1_EraserCircle)
        self.ax1_EraserX = pg.ScatterPlotItem()
        self.ax1_EraserX.setData([], [], symbol='x', pxMode=False, size=3,
                                      brush=pg.mkBrush(color=(255,0,0,50)),
                                      pen=pg.mkPen(width=1, color='r'))
        self.ax1.addItem(self.ax1_EraserX)

        # Brush circle img2
        self.ax2_BrushCirclePen = pg.mkPen(width=2)
        self.ax2_BrushCircleBrush = pg.mkBrush((255,255,255,50))
        self.ax2_BrushCircle = pg.ScatterPlotItem()
        self.ax2_BrushCircle.setData([], [], symbol='o', pxMode=False,
                                 brush=self.ax2_BrushCircleBrush,
                                 pen=self.ax2_BrushCirclePen)
        self.ax2.addItem(self.ax2_BrushCircle)

        # Random walker markers colors
        self.RWbkgrColor = (255,255,0)
        self.RWforegrColor = (124,5,161)


        # Experimental: brush cursors
        self.eraserCursor = QCursor(QIcon(":eraser.png").pixmap(30, 30))
        brushCursorPixmap = QIcon(":brush-cursor.png").pixmap(32, 32)
        self.brushCursor = QCursor(brushCursorPixmap, 16, 16)

        # Annotated metadata markers (ScatterPlotItem)
        self.ax2_binnedIDs_ScatterPlot = pg.ScatterPlotItem()
        self.ax2_binnedIDs_ScatterPlot.setData(
                                 [], [], symbol='t', pxMode=False,
                                 brush=pg.mkBrush((255,0,0,50)), size=15,
                                 pen=pg.mkPen(width=3, color='r'))
        self.ax2.addItem(self.ax2_binnedIDs_ScatterPlot)
        self.ax2_ripIDs_ScatterPlot = pg.ScatterPlotItem()
        self.ax2_ripIDs_ScatterPlot.setData(
                                 [], [], symbol='x', pxMode=False,
                                 brush=pg.mkBrush((255,0,0,50)), size=15,
                                 pen=pg.mkPen(width=2, color='r'))
        self.ax2.addItem(self.ax2_ripIDs_ScatterPlot)

        self.ax1_binnedIDs_ScatterPlot = pg.ScatterPlotItem()
        self.ax1_binnedIDs_ScatterPlot.setData(
                                 [], [], symbol='t', pxMode=False,
                                 brush=pg.mkBrush((255,0,0,50)), size=15,
                                 pen=pg.mkPen(width=3, color='r'))
        self.ax1.addItem(self.ax1_binnedIDs_ScatterPlot)
        self.ax1_ripIDs_ScatterPlot = pg.ScatterPlotItem()
        self.ax1_ripIDs_ScatterPlot.setData(
                                 [], [], symbol='x', pxMode=False,
                                 brush=pg.mkBrush((255,0,0,50)), size=15,
                                 pen=pg.mkPen(width=2, color='r'))
        self.ax1.addItem(self.ax1_ripIDs_ScatterPlot)

        # Ruler plotItem and scatterItem
        rulerPen = pg.mkPen(color='r', style=Qt.DashLine, width=2)
        self.ax1_rulerPlotItem = pg.PlotDataItem(pen=rulerPen)
        self.ax1_rulerAnchorsItem = pg.ScatterPlotItem(
            symbol='o', size=9,
            brush=pg.mkBrush((255,0,0,50)),
            pen=pg.mkPen((255,0,0), width=2)
        )
        self.ax1.addItem(self.ax1_rulerPlotItem)
        self.ax1.addItem(self.ax1_rulerAnchorsItem)

    def gui_createGraphicsItems(self):
        # Contour pens
        self.oldIDs_cpen = pg.mkPen(color=(200, 0, 0, 255*0.5), width=2)
        self.newIDs_cpen = pg.mkPen(color='r', width=3)
        self.tempNewIDs_cpen = pg.mkPen(color='g', width=3)
        self.lostIDs_cpen = pg.mkPen(color=(245, 184, 0, 100), width=4)

        # Lost ID question mark text color
        self.lostIDs_qMcolor = (245, 184, 0)

        # New bud-mother line pen
        self.NewBudMoth_Pen = pg.mkPen(color='r', width=3, style=Qt.DashLine)

        # Old bud-mother line pen
        self.OldBudMoth_Pen = pg.mkPen(color=(255,165,0), width=2,
                                       style=Qt.DashLine)

        # Temporary line item connecting bud to new mother
        self.BudMothTempLine = pg.PlotDataItem(pen=self.NewBudMoth_Pen)
        self.ax1.addItem(self.BudMothTempLine)

        # Red/green border rect item
        self.GreenLinePen = pg.mkPen(color='g', width=2)
        self.RedLinePen = pg.mkPen(color='r', width=2)
        self.ax1BorderLine = pg.PlotDataItem()
        self.ax1.addItem(self.ax1BorderLine)
        self.ax2BorderLine = pg.PlotDataItem(pen=pg.mkPen(color='r', width=2))
        self.ax2.addItem(self.ax2BorderLine)

        # Create enough PlotDataItems and LabelItems to draw contours and IDs.
        maxID = max([PosData.segm_data.max() for PosData in self.data])
        numItems = maxID+10
        self.ax1_ContoursCurves = []
        self.ax2_ContoursCurves = []
        self.ax1_BudMothLines = []
        self.ax1_LabelItemsIDs = []
        self.ax2_LabelItemsIDs = []
        for i in range(numItems):
            # Contours on ax1
            ContCurve = pg.PlotDataItem()
            self.ax1_ContoursCurves.append(ContCurve)
            self.ax1.addItem(ContCurve)

            # Bud mother line on ax1
            BudMothLine = pg.PlotDataItem()
            self.ax1_BudMothLines.append(BudMothLine)
            self.ax1.addItem(BudMothLine)

            # LabelItems on ax1
            ax1_IDlabel = pg.LabelItem()
            self.ax1_LabelItemsIDs.append(ax1_IDlabel)
            self.ax1.addItem(ax1_IDlabel)

            # LabelItems on ax2
            ax2_IDlabel = pg.LabelItem()
            self.ax2_LabelItemsIDs.append(ax2_IDlabel)
            self.ax2.addItem(ax2_IDlabel)

            # Contours on ax2
            ContCurve = pg.PlotDataItem()
            self.ax2_ContoursCurves.append(ContCurve)
            self.ax2.addItem(ContCurve)

    def gui_connectGraphicsEvents(self):
        self.img1.hoverEvent = self.gui_hoverEventImg1
        self.img2.hoverEvent = self.gui_hoverEventImg2
        self.img1.mousePressEvent = self.gui_mousePressEventImg1
        self.img1.mouseMoveEvent = self.gui_mouseDragEventImg1
        self.img1.mouseReleaseEvent = self.gui_mouseReleaseEventImg1
        self.img2.mousePressEvent = self.gui_mousePressEventImg2
        self.img2.mouseMoveEvent = self.gui_mouseDragEventImg2
        self.img2.mouseReleaseEvent = self.gui_mouseReleaseEventImg2
        self.hist.vb.contextMenuEvent = self.gui_raiseContextMenuLUT

    def gui_mousePressEventImg2(self, event):
        PosData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        left_click = event.button() == Qt.MouseButton.LeftButton
        mid_click = event.button() == Qt.MouseButton.MidButton
        right_click = event.button() == Qt.MouseButton.RightButton
        eraserON = self.eraserButton.isChecked()
        brushON = self.brushButton.isChecked()
        separateON = self.separateBudButton.isChecked()

        # Drag image if neither brush or eraser are On pressed
        dragImg = (
            left_click and not eraserON and not
            brushON and not separateON
        )

        # Enable dragging of the image window like pyqtgraph original code
        if dragImg:
            pg.ImageItem.mousePressEvent(self.img2, event)

        x, y = event.pos().x(), event.pos().y()
        ID = PosData.lab[int(y), int(x)]
        dragImgMiddle = mid_click

        if dragImgMiddle:
            pg.ImageItem.mousePressEvent(self.img2, event)

        if mode == 'Viewer':
            return

        # Check if right click on ROI
        delROIs = (PosData.allData_li[PosData.frame_i]
                                     ['delROIs_info']
                                     ['rois'].copy())
        for r, roi in enumerate(delROIs):
            x0, y0 = [int(c) for c in roi.pos()]
            w, h = [int(c) for c in roi.size()]
            x1, y1 = x0+w, y0+h
            clickedOnROI = (
                x>=x0 and x<=x1 and y>=y0 and y<=y1
            )
            raiseContextMenuRoi = right_click and clickedOnROI
            dragRoi = left_click and clickedOnROI
            if raiseContextMenuRoi:
                self.roi_to_del = roi
                self.roiContextMenu = QMenu(self)
                separator = QAction(self)
                separator.setSeparator(True)
                self.roiContextMenu.addAction(separator)
                action = QAction('Remove ROI')
                action.triggered.connect(self.removeROI)
                self.roiContextMenu.addAction(action)
                self.roiContextMenu.exec_(event.screenPos())
                return
            elif dragRoi:
                event.ignore()
                return


        # Left-click is used for brush, eraser, separate bud and curvature tool
        # Brush and eraser are mutually exclusive but we want to keep the eraser
        # or brush ON and disable them temporarily to allow left-click with
        # separate ON
        canErase = eraserON and not separateON and not dragImg
        canBrush = brushON and not separateON and not dragImg
        canDelete = mode == 'Segmentation and Tracking' or self.isSnapshot

        # Erase with brush and left click on the right image
        # NOTE: contours, IDs and rp will be updated
        # on gui_mouseReleaseEventImg2
        if left_click and canErase:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = PosData.lab.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                self.yPressAx2, self.xPressAx2 = y, x
                # Keep a global mask to compute which IDs got erased
                self.erasedIDs = []
                self.erasedID = PosData.lab[ydata, xdata]

                ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

                # Build eraser mask
                mask = np.zeros(PosData.lab.shape, bool)
                mask[ymin:ymax, xmin:xmax][diskMask] = True

                # If user double-pressed 'b' then erase over ALL labels
                color = self.eraserButton.palette().button().color().name()
                eraseOnlyOneID = (
                    color != self.doublePressKeyButtonColor
                    and self.erasedID != 0
                )
                if eraseOnlyOneID:
                    mask[PosData.lab!=self.erasedID] = False

                self.eraseOnlyOneID = eraseOnlyOneID

                self.erasedIDs.extend(PosData.lab[mask])
                PosData.lab[mask] = 0
                self.img2.updateImage()

                self.isMouseDragImg2 = True

        # Paint with brush and left click on the right image
        # NOTE: contours, IDs and rp will be updated
        # on gui_mouseReleaseEventImg2
        elif left_click and canBrush:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = PosData.lab.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                self.yPressAx2, self.xPressAx2 = y, x

                ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

                ID = PosData.lab[ydata, xdata]

                # If user double-pressed 'b' then draw over the labels
                color = self.brushButton.palette().button().color().name()
                drawUnder = color != self.doublePressKeyButtonColor

                if ID > 0 and drawUnder:
                    self.ax2BrushID = ID
                    PosData.isNewID = False
                else:
                    self.setBrushID()
                    self.ax2BrushID = PosData.brushID
                    PosData.isNewID = True

                self.isMouseDragImg2 = True

                # Draw new objects
                localLab = PosData.lab[ymin:ymax, xmin:xmax]
                mask = diskMask.copy()
                if drawUnder:
                    mask[localLab!=0] = False

                PosData.lab[ymin:ymax, xmin:xmax][diskMask] = self.ax2BrushID
                self.setImageImg2()

        # Delete entire ID (set to 0)
        elif mid_click and canDelete:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            delID = PosData.lab[ydata, xdata]
            if delID == 0:
                delID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to delete',
                    parent=self
                )
                delID_prompt.exec_()
                if delID_prompt.cancel:
                    return
                else:
                    delID = delID_prompt.EntryID

            # Ask to propagate change to all future visited frames
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                                    delID, 'Delete ID',
                                    PosData.doNotShowAgain_DelID,
                                    PosData.UndoFutFrames_DelID,
                                    PosData.applyFutFrames_DelID)

            if UndoFutFrames is None:
                return

            PosData.doNotShowAgain_DelID = doNotShowAgain
            PosData.UndoFutFrames_DelID = UndoFutFrames
            PosData.applyFutFrames_DelID = applyFutFrames

            self.current_frame_i = PosData.frame_i

            # Apply Delete ID to future frames if requested
            if applyFutFrames:
                self.app.setOverrideCursor(Qt.WaitCursor)
                # Store current data before going to future frames
                self.store_data()
                for i in range(PosData.frame_i+1, endFrame_i+1):
                    lab = PosData.allData_li[i]['labels']
                    if lab is None:
                        break

                    lab[lab==delID] = 0

                    # Store change
                    PosData.allData_li[i]['labels'] = lab.copy()
                    # Get the rest of the stored metadata based on the new lab
                    PosData.frame_i = i
                    self.get_data()
                    self.store_data()

                self.app.restoreOverrideCursor()

            # Back to current frame
            if applyFutFrames:
                PosData.frame_i = self.current_frame_i
                self.get_data()

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)
            PosData.lab[PosData.lab==delID] = 0

            # Update data (rp, etc)
            self.update_rp()

            self.warnEditingWithCca_df('Delete ID')

            self.setImageImg2()

            # Remove contour and LabelItem of deleted ID
            self.ax1_ContoursCurves[delID-1].setData([], [])
            self.ax1_LabelItemsIDs[delID-1].setText('')
            self.ax2_LabelItemsIDs[delID-1].setText('')

            self.checkIDs_LostNew()
            self.highlightLostNew()
            self.checkIDsMultiContour()

        # Separate bud
        elif (right_click or left_click) and self.separateBudButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = PosData.lab[ydata, xdata]
            if ID == 0:
                sepID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to split',
                    parent=self
                )
                sepID_prompt.exec_()
                if sepID_prompt.cancel:
                    return
                else:
                    ID = sepID_prompt.EntryID



            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            max_ID = PosData.lab.max()

            if right_click:
                PosData.lab, success = self.auto_separate_bud_ID(
                                             ID, PosData.lab, PosData.rp,
                                             max_ID, enforce=True)
            else:
                success = False

            # If automatic bud separation was not successfull call manual one
            if not success:
                PosData.disableAutoActivateViewerWindow = True
                img = self.getDisplayedCellsImg()
                manualSep = apps.manualSeparateGui(
                                PosData.lab, ID, img,
                                fontSize=self.fontSize,
                                IDcolor=self.img2.lut[ID],
                                parent=self)
                manualSep.show()
                manualSep.centerWindow()
                loop = QEventLoop(self)
                manualSep.loop = loop
                loop.exec_()
                if manualSep.cancel:
                    PosData.disableAutoActivateViewerWindow = False
                    self.separateBudButton.setChecked(False)
                    return
                PosData.lab[manualSep.lab!=0] = manualSep.lab[manualSep.lab!=0]
                PosData.disableAutoActivateViewerWindow = False

            # Update data (rp, etc)
            prev_IDs = [obj.label for obj in PosData.rp]
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True)

            # Update all images
            self.updateALLimg()
            self.warnEditingWithCca_df('Separate IDs')
            self.store_data()

            # Uncheck separate bud button
            self.separateBudButton.setChecked(False)

        # Replace with Hull Contour
        elif right_click and self.hullContToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = PosData.lab[ydata, xdata]
            if ID == 0:
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here the ID that you want to '
                         'replace with Hull contour',
                    parent=self
                )
                mergeID_prompt.exec_()
                if mergeID_prompt.cancel:
                    return
                else:
                    ID = mergeID_prompt.EntryID

            if ID in PosData.lab:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                obj_idx = PosData.IDs.index(ID)
                obj = PosData.rp[obj_idx]
                localHull = skimage.morphology.convex_hull_image(obj.image)
                PosData.lab[obj.slice][localHull] = ID

                self.update_rp()
                self.updateALLimg()

                self.hullContToolButton.setChecked(False)

        # Merge IDs
        elif right_click and self.mergeIDsButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = PosData.lab[ydata, xdata]
            if ID == 0:
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here first ID that you want to merge',
                    parent=self
                )
                mergeID_prompt.exec_()
                if mergeID_prompt.cancel:
                    return
                else:
                    ID = mergeID_prompt.EntryID

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            self.firstID = ID

        # Edit ID
        elif right_click and self.editID_Button.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = PosData.lab[ydata, xdata]
            if ID == 0:
                editID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter here ID that you want to replace with a new one',
                    parent=self
                )
                editID_prompt.exec_()
                if editID_prompt.cancel:
                    return
                else:
                    ID = editID_prompt.EntryID
                    obj_idx = PosData.IDs.index(ID)
                    y, x = PosData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            PosData.disableAutoActivateViewerWindow = True
            prev_IDs = [obj.label for obj in PosData.rp]
            editID = apps.editID_QWidget(ID, prev_IDs, parent=self)
            editID.exec_()
            if editID.cancel:
                PosData.disableAutoActivateViewerWindow = False
                self.editID_Button.setChecked(False)
                return

            # Ask to propagate change to all future visited frames
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                                    ID, 'Edit ID',
                                    PosData.doNotShowAgain_EditID,
                                    PosData.UndoFutFrames_EditID,
                                    PosData.applyFutFrames_EditID,
                                    applyTrackingB=True)

            if UndoFutFrames is None:
                return

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            for old_ID, new_ID in editID.how:
                if new_ID in prev_IDs:
                    tempID = PosData.lab.max() + 1
                    PosData.lab[PosData.lab == old_ID] = tempID
                    PosData.lab[PosData.lab == new_ID] = old_ID
                    PosData.lab[PosData.lab == tempID] = new_ID

                    # Clear labels IDs of the swapped IDs
                    self.ax2_LabelItemsIDs[old_ID-1].setText('')
                    self.ax1_LabelItemsIDs[old_ID-1].setText('')
                    self.ax2_LabelItemsIDs[new_ID-1].setText('')
                    self.ax1_LabelItemsIDs[new_ID-1].setText('')

                    self.addNewItems()

                    old_ID_idx = prev_IDs.index(old_ID)
                    new_ID_idx = prev_IDs.index(new_ID)
                    PosData.rp[old_ID_idx].label = new_ID
                    PosData.rp[new_ID_idx].label = old_ID
                    self.drawID_and_Contour(
                        PosData.rp[old_ID_idx],
                        drawContours=True
                    )
                    self.drawID_and_Contour(
                        PosData.rp[new_ID_idx],
                        drawContours=False
                    )

                    # Append information for replicating the edit in tracking
                    # List of tuples (y, x, replacing ID)
                    obj = PosData.rp[old_ID_idx]
                    y, x = obj.centroid
                    y, x = int(y), int(x)
                    PosData.editID_info.append((y, x, new_ID))
                    obj = PosData.rp[new_ID_idx]
                    y, x = obj.centroid
                    y, x = int(y), int(x)
                    PosData.editID_info.append((y, x, old_ID))
                else:
                    PosData.lab[PosData.lab == old_ID] = new_ID
                    # Clear labels IDs of the swapped IDs
                    self.ax2_LabelItemsIDs[old_ID-1].setText('')
                    self.ax1_LabelItemsIDs[old_ID-1].setText('')

                    self.addNewItems()

                    old_ID_idx = prev_IDs.index(old_ID)
                    PosData.rp[old_ID_idx].label = new_ID
                    self.drawID_and_Contour(
                        PosData.rp[old_ID_idx],
                        drawContours=True
                    )
                    # Append information for replicating the edit in tracking
                    # List of tuples (y, x, replacing ID)
                    obj = PosData.rp[old_ID_idx]
                    y, x = obj.centroid
                    y, x = int(y), int(x)
                    PosData.editID_info.append((y, x, new_ID))

            # Update rps
            self.update_rp()

            # Since we manually changed an ID we don't want to repeat tracking
            self.checkIDs_LostNew()
            self.highlightLostNew()
            self.checkIDsMultiContour()

            # Update colors for the edited IDs
            self.updateLookuptable()

            self.warnEditingWithCca_df('Edit ID')

            self.setImageImg2()
            self.editID_Button.setChecked(False)

            PosData.disableAutoActivateViewerWindow = True


            # Perform desired action on future frames
            PosData.doNotShowAgain_EditID = doNotShowAgain
            PosData.UndoFutFrames_EditID = UndoFutFrames
            PosData.applyFutFrames_EditID = applyFutFrames

            self.current_frame_i = PosData.frame_i

            if applyFutFrames:
                self.app.setOverrideCursor(Qt.WaitCursor)
                # Store data for current frame
                self.store_data()
                if endFrame_i is None:
                    self.app.restoreOverrideCursor()
                    return
                for i in range(PosData.frame_i+1, endFrame_i+1):
                    PosData.frame_i = i
                    self.get_data()
                    if self.onlyTracking:
                        self.tracking(enforce=True)
                    else:
                        for old_ID, new_ID in editID.how:
                            if new_ID in prev_IDs:
                                tempID = PosData.lab.max() + 1
                                PosData.lab[PosData.lab == old_ID] = tempID
                                PosData.lab[PosData.lab == new_ID] = old_ID
                                PosData.lab[PosData.lab == tempID] = new_ID
                            else:
                                PosData.lab[PosData.lab == old_ID] = new_ID
                        self.update_rp(draw=False)
                    self.store_data()

                # Back to current frame
                PosData.frame_i = self.current_frame_i
                self.get_data()
                self.app.restoreOverrideCursor()

        # Annotate cell as removed from the analysis
        elif right_click and self.binCellButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = PosData.lab[ydata, xdata]
            if ID == 0:
                binID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to remove from the analysis',
                    parent=self
                )
                binID_prompt.exec_()
                if binID_prompt.cancel:
                    return
                else:
                    ID = binID_prompt.EntryID

            # Ask to propagate change to all future visited frames
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                                    ID, 'Exclude cell from analysis',
                                    PosData.doNotShowAgain_BinID,
                                    PosData.UndoFutFrames_BinID,
                                    PosData.applyFutFrames_BinID)

            if UndoFutFrames is None:
                return

            PosData.doNotShowAgain_BinID = doNotShowAgain
            PosData.UndoFutFrames_BinID = UndoFutFrames
            PosData.applyFutFrames_BinID = applyFutFrames

            self.current_frame_i = PosData.frame_i

            # Apply Edit ID to future frames if requested
            if applyFutFrames:
                self.app.setOverrideCursor(Qt.WaitCursor)
                # Store current data before going to future frames
                self.store_data()
                for i in range(PosData.frame_i+1, endFrame_i+1):
                    PosData.frame_i = i
                    self.get_data()
                    if ID in PosData.binnedIDs:
                        PosData.binnedIDs.remove(ID)
                    else:
                        PosData.binnedIDs.add(ID)
                    self.update_rp_metadata(draw=False)
                    self.store_data()

                self.app.restoreOverrideCursor()

            # Back to current frame
            if applyFutFrames:
                PosData.frame_i = self.current_frame_i
                self.get_data()

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            if ID in PosData.binnedIDs:
                PosData.binnedIDs.remove(ID)
            else:
                PosData.binnedIDs.add(ID)

            self.update_rp_metadata()

            # Gray out ore restore binned ID
            self.updateLookuptable()

            self.binCellButton.setChecked(False)

        # Annotate cell as dead
        elif right_click and self.ripCellButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = PosData.lab[ydata, xdata]
            if ID == 0:
                ripID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as dead',
                    parent=self
                )
                ripID_prompt.exec_()
                if ripID_prompt.cancel:
                    return
                else:
                    ID = ripID_prompt.EntryID

            # Ask to propagate change to all future visited frames
            (UndoFutFrames, applyFutFrames, endFrame_i,
            doNotShowAgain) = self.propagateChange(
                                    ID, 'Annotate cell as dead',
                                    PosData.doNotShowAgain_RipID,
                                    PosData.UndoFutFrames_RipID,
                                    PosData.applyFutFrames_RipID)

            if UndoFutFrames is None:
                return

            PosData.doNotShowAgain_RipID = doNotShowAgain
            PosData.UndoFutFrames_RipID = UndoFutFrames
            PosData.applyFutFrames_RipID = applyFutFrames

            self.current_frame_i = PosData.frame_i

            # Apply Edit ID to future frames if requested
            if applyFutFrames:
                self.app.setOverrideCursor(Qt.WaitCursor)
                # Store current data before going to future frames
                self.store_data()
                for i in range(PosData.frame_i+1, endFrame_i+1):
                    PosData.frame_i = i
                    self.get_data()
                    if ID in PosData.ripIDs:
                        PosData.ripIDs.remove(ID)
                    else:
                        PosData.ripIDs.add(ID)
                    self.update_rp_metadata(draw=False)
                    self.store_data()
                self.app.restoreOverrideCursor()

            # Back to current frame
            if applyFutFrames:
                PosData.frame_i = self.current_frame_i
                self.get_data()

            # Store undo state before modifying stuff
            self.storeUndoRedoStates(UndoFutFrames)

            if ID in PosData.ripIDs:
                PosData.ripIDs.remove(ID)
            else:
                PosData.ripIDs.add(ID)

            self.update_rp_metadata()

            # Gray out dead ID
            self.updateLookuptable()
            self.store_data()

            self.warnEditingWithCca_df('Annotate ID as dead')

            self.ripCellButton.setChecked(False)

    def gui_mouseDragEventImg1(self, event):
        PosData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        PosData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return


        if self.isRightClickDragImg1 and self.curvToolButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.drawAutoContour(y, x)

        # Brush dragging mouse --> keep painting
        elif self.isMouseDragImg1 and self.brushButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = PosData.lab.shape
            brushSize = self.brushSizeSpinbox.value()
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

                rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

                # Build brush mask
                mask = np.zeros(PosData.lab.shape, bool)
                mask[ymin:ymax, xmin:xmax][diskMask] = True
                mask[rrPoly, ccPoly] = True

                # If user double-pressed 'b' then draw over the labels
                color = self.brushButton.palette().button().color().name()
                if color != self.doublePressKeyButtonColor:
                    mask[PosData.lab!=0] = False
                    self.setHoverToolSymbolColor(
                        xdata, ydata, self.ax2_BrushCirclePen,
                        (self.ax2_BrushCircle, self.ax1_BrushCircle),
                        self.brushButton, brush=self.ax2_BrushCircleBrush
                    )

                # Apply brush mask
                PosData.lab[mask] = PosData.brushID
                self.setImageImg2()
                self.setTempImg1Brush(ymin, ymax, xmin, xmax, mask)

        # Eraser dragging mouse --> keep erasing
        elif self.isMouseDragImg1 and self.eraserButton.isChecked():
            PosData = self.data[self.pos_i]
            Y, X = PosData.lab.shape
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            brushSize = self.brushSizeSpinbox.value()

            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            # Build eraser mask
            mask = np.zeros(PosData.lab.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            if self.eraseOnlyOneID:
                mask[PosData.lab!=self.erasedID] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.eraserCirclePen,
                    (self.ax2_EraserCircle, self.ax1_EraserCircle),
                    self.eraserButton, hoverRGB=self.img2.lut[self.erasedID],
                    ID=self.erasedID
                )


            self.erasedIDs.extend(PosData.lab[mask])
            PosData.lab[mask] = 0

            self.setImageImg2()

            self.erasesedLab = np.zeros_like(PosData.lab)
            for erasedID in np.unique(self.erasedIDs):
                if erasedID == 0:
                    continue
                self.erasesedLab[PosData.lab==erasedID] = erasedID
            erasedRp = skimage.measure.regionprops(self.erasesedLab)
            for obj in erasedRp:
                idx = obj.label-1
                curveID = self.ax1_ContoursCurves[idx]
                cont = self.getObjContours(obj)
                curveID.setData(cont[:,0], cont[:,1], pen=self.oldIDs_cpen)

    def gui_hoverEventImg1(self, event):
        PosData = self.data[self.pos_i]
        # Update x, y, value label bottom right
        if not event.isExit():
            self.xHoverImg, self.yHoverImg = event.pos()
        else:
            self.xHoverImg, self.yHoverImg = None, None

        drawRulerLine = (
            self.rulerButton.isChecked() and self.rulerHoverON
            and not event.isExit()
        )
        if drawRulerLine:
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            xxRA, yyRA = self.ax1_rulerAnchorsItem.getData()
            self.ax1_rulerPlotItem.setData([xxRA[0], xdata], [yyRA[0], ydata])

        if not event.isExit():
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.img1.image
            Y, X = _img.shape[:2]
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                maxVal = _img.max()
                ID = PosData.lab[ydata, xdata]
                maxID = PosData.lab.max()
                if _img.ndim > 2:
                    val = [v for v in val]
                    value = f'{val}'
                else:
                    value = f'{val:.2f}'
                txt = (
                    f'x={x:.2f}, y={y:.2f}, value={value}, '
                    f'max={maxVal:.2f}, ID={ID}, max_ID={maxID}'
                )
                xx, yy = self.ax1_rulerPlotItem.getData()
                if xx is not None:
                    lenPxl = math.sqrt((xx[0]-xx[1])**2 + (yy[0]-yy[1])**2)
                    pxlToUm = self.data[self.pos_i].PhysicalSizeX
                    lenTxt = (
                        f'length={lenPxl:.2f} pxl ({lenPxl*pxlToUm:.2f} um)'
                    )
                    txt = f'{txt}, {lenTxt}'
                self.wcLabel.setText(txt)
            else:
                self.clickedOnBud = False
                self.BudMothTempLine.setData([], [])
                self.wcLabel.setText(f'')

        # Draw eraser circle
        drawCircle = self.eraserButton.isChecked() and not event.isExit()
        if not event.isExit() and drawCircle:
            x, y = event.pos()
            self.updateEraserCursor(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )

        # Draw Brush circle
        drawCircle = self.brushButton.isChecked() and not event.isExit()
        if not event.isExit() and drawCircle:
            x, y = event.pos()
            self.updateBrushCursor(x, y)
            self.hideItemsHoverBrush(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )

        drawMothBudLine = (
            self.assignBudMothButton.isChecked() and self.clickedOnBud
            and not event.isExit()
        )
        if drawMothBudLine:
            x, y = event.pos()
            y2, x2 = y, x
            xdata, ydata = int(x), int(y)
            y1, x1 = self.yClickBud, self.xClickBud
            ID = PosData.lab[ydata, xdata]
            if ID == 0:
                self.BudMothTempLine.setData([x1, x2], [y1, y2])
            else:
                obj_idx = PosData.IDs.index(ID)
                obj = PosData.rp[obj_idx]
                y2, x2 = obj.centroid
                self.BudMothTempLine.setData([x1, x2], [y1, y2])

        # Temporarily draw spline curve
        # see https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
        drawSpline = (
            self.curvToolButton.isChecked() and self.splineHoverON
            and not event.isExit()
        )
        if drawSpline:
            x, y = event.pos()
            xx, yy = self.curvAnchors.getData()
            hoverAnchors = self.curvAnchors.pointsAt(event.pos())
            per=False
            # If we are hovering the starting point we generate
            # a closed spline
            if len(xx) >= 2:
                if len(hoverAnchors)>0:
                    xA_hover, yA_hover = hoverAnchors[0].pos()
                    if xx[0]==xA_hover and yy[0]==yA_hover:
                        per=True
                if per:
                    # Append start coords and close spline
                    xx = np.r_[xx, xx[0]]
                    yy = np.r_[yy, yy[0]]
                    xi, yi = self.getSpline(xx, yy, per=per)
                    # self.curvPlotItem.setData([], [])
                else:
                    # Append mouse coords
                    xx = np.r_[xx, x]
                    yy = np.r_[yy, y]
                    xi, yi = self.getSpline(xx, yy, per=per)
                self.curvHoverPlotItem.setData(xi, yi)

    def gui_hoverEventImg2(self, event):
        PosData = self.data[self.pos_i]
        if not event.isExit():
            self.xHoverImg, self.yHoverImg = event.pos()
        else:
            self.xHoverImg, self.yHoverImg = None, None
        # Update x, y, value label bottom right
        if not event.isExit():
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.img2.image
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                self.wcLabel.setText(
                    f'(x={x:.2f}, y={y:.2f}, value={val:.0f}, max={_img.max()})'
                )
            else:
                if self.eraserButton.isChecked() or self.brushButton.isChecked():
                    self.gui_mouseReleaseEventImg2(event)
                self.wcLabel.setText(f'')

        # Draw eraser circle
        drawCircle = self.eraserButton.isChecked() and not event.isExit()
        if drawCircle and not event.isExit():
            x, y = event.pos()
            self.updateEraserCursor(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )

        # Draw Brush circle
        drawCircle = self.brushButton.isChecked() and not event.isExit()
        if drawCircle and not event.isExit():
            x, y = event.pos()
            self.updateBrushCursor(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )

    def gui_raiseContextMenuLUT(self, event):
        PosData = self.data[self.pos_i]
        PosData.lutmenu = QMenu(self)
        PosData.lutmenu.addAction(self.userChNameAction)
        for action in PosData.fluoDataChNameActions:
            PosData.lutmenu.addAction(action)
        PosData.lutmenu.exec(event.screenPos())

    def gui_mouseDragEventImg2(self, event):
        # Eraser dragging mouse --> keep erasing
        if self.isMouseDragImg2 and self.eraserButton.isChecked():
            PosData = self.data[self.pos_i]
            Y, X = PosData.lab.shape
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            brushSize = self.brushSizeSpinbox.value()
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            # Build eraser mask
            mask = np.zeros(PosData.lab.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            if self.eraseOnlyOneID:
                mask[PosData.lab!=self.erasedID] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.eraserCirclePen,
                    (self.ax2_EraserCircle, self.ax1_EraserCircle),
                    self.eraserButton, hoverRGB=self.img2.lut[self.erasedID],
                    ID=self.erasedID
                )

            self.erasedIDs.extend(PosData.lab[mask])

            PosData.lab[mask] = 0
            self.setImageImg2()

        # Brush paint dragging mouse --> keep painting
        if self.isMouseDragImg2 and self.brushButton.isChecked():
            PosData = self.data[self.pos_i]
            Y, X = PosData.lab.shape
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            brushSize = self.brushSizeSpinbox.value()
            rrPoly, ccPoly = self.getPolygonBrush((y, x), Y, X)

            ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

            # Build brush mask
            mask = np.zeros(PosData.lab.shape, bool)
            mask[ymin:ymax, xmin:xmax][diskMask] = True
            mask[rrPoly, ccPoly] = True

            # If user double-pressed 'b' then draw over the labels
            color = self.brushButton.palette().button().color().name()
            if color != self.doublePressKeyButtonColor:
                mask[PosData.lab!=0] = False
                self.setHoverToolSymbolColor(
                    xdata, ydata, self.ax2_BrushCirclePen,
                    (self.ax2_BrushCircle, self.ax1_BrushCircle),
                    self.eraserButton, brush=self.ax2_BrushCircleBrush
                )

            # Apply brush mask
            PosData.lab[mask] = self.ax2BrushID
            self.setImageImg2()

    def gui_mouseReleaseEventImg2(self, event):
        PosData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        # Eraser mouse release --> update IDs and contours
        if self.isMouseDragImg2 and self.eraserButton.isChecked():
            self.isMouseDragImg2 = False
            erasedIDs = np.unique(self.erasedIDs)

            # Update data (rp, etc)
            self.update_rp()
            self.updateALLimg()

            for ID in erasedIDs:
                if ID not in PosData.lab:
                    self.warnEditingWithCca_df('Delete ID with eraser')
                    break

        # Brush mouse release --> update IDs and contours
        elif self.isMouseDragImg2 and self.brushButton.isChecked():
            self.isMouseDragImg2 = False

            self.update_rp()
            if PosData.isNewID:
                self.tracking(enforce=True)

            self.updateALLimg()
            self.warnEditingWithCca_df('Add new ID with brush tool')

        # Merge IDs
        elif self.mergeIDsButton.isChecked():
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = PosData.lab[ydata, xdata]
            if ID == 0:
                mergeID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to merge with ID '
                         f'{self.firstID}',
                    parent=self
                )
                mergeID_prompt.exec_()
                if mergeID_prompt.cancel:
                    return
                else:
                    ID = mergeID_prompt.EntryID

            PosData.lab[PosData.lab==ID] = self.firstID

            # Mask to keep track of which ID needs redrawing of the contours
            mergedID_mask = PosData.lab==self.firstID

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True)

            self.updateALLimg()
            self.mergeIDsButton.setChecked(False)
            self.store_data()
            self.warnEditingWithCca_df('Merge IDs')

    def gui_mouseReleaseEventImg1(self, event):
        PosData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return

        if mode=='Segmentation and Tracking' or self.isSnapshot:
            # Allow right-click actions on both images
            self.gui_mouseReleaseEventImg2(event)

        if self.isRightClickDragImg1 and self.curvToolButton.isChecked():
            self.isRightClickDragImg1 = False
            try:
                self.splineToObj()
                self.update_rp()
                self.tracking(enforce=True)
                self.updateALLimg()
                self.warnEditingWithCca_df('Add new ID with curvature tool')
                self.clearCurvItems()
                self.curvTool_cb(True)
            except ValueError:
                self.clearCurvItems()
                self.curvTool_cb(True)
                pass

        # Eraser mouse release --> update IDs and contours
        elif self.isMouseDragImg1 and self.eraserButton.isChecked():
            self.isMouseDragImg1 = False
            erasedIDs = np.unique(self.erasedIDs)

            # Update data (rp, etc)
            self.update_rp()
            self.updateALLimg()

            for ID in erasedIDs:
                if ID not in PosData.lab:
                    self.warnEditingWithCca_df('Delete ID with eraser')
                    break

        elif self.isMouseDragImg1 and self.brushButton.isChecked():
            self.isMouseDragImg1 = False

            # Update data (rp, etc)
            self.update_rp()

            # Repeat tracking
            self.tracking(enforce=True)

            # Update colors to include a new color for the new ID
            self.updateALLimg()
            self.warnEditingWithCca_df('Add new ID with brush tool')

        # Assign mother to bud
        elif self.assignBudMothButton.isChecked() and self.clickedOnBud:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = PosData.lab[ydata, xdata]
            if ID == PosData.lab[self.yClickBud, self.xClickBud]:
                return

            if ID == 0:
                mothID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as mother cell',
                    parent=self
                )
                mothID_prompt.exec_()
                if mothID_prompt.cancel:
                    return
                else:
                    ID = mothID_prompt.EntryID
                    obj_idx = PosData.IDs.index(ID)
                    y, x = PosData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            relationship = PosData.cca_df.at[ID, 'relationship']
            ccs = PosData.cca_df.at[ID, 'cell_cycle_stage']
            is_history_known = PosData.cca_df.at[ID, 'is_history_known']
            # We allow assiging a cell in G1 as mother only on first frame
            # OR if the history is unknown
            if relationship == 'bud' and PosData.frame_i > 0 and is_history_known:
                txt = (f'You clicked on ID {ID} which is a BUD.\n'
                       'To assign a bud to a cell start by clicking on a bud '
                       'and release on a cell in G1')
                msg = QtGui.QMessageBox()
                msg.critical(
                    self, 'Released on a bud', txt, msg.Ok
                )
                return

            elif ccs != 'G1' and PosData.frame_i > 0:
                txt = (f'You clicked on a cell (ID={ID}) which is NOT in G1.\n'
                       'To assign a bud to a cell start by clicking on a bud '
                       'and release on a cell in G1')
                msg = QtGui.QMessageBox()
                msg.critical(
                    self, 'Released on a cell NOT in G1', txt, msg.Ok
                )
                return

            elif PosData.frame_i == 0:
                # Check that clicked bud actually is smaller that mother
                # otherwise warn the user that he might have clicked first
                # on a mother
                budID = PosData.lab[self.yClickBud, self.xClickBud]
                new_mothID = PosData.lab[ydata, xdata]
                bud_obj_idx = PosData.IDs.index(budID)
                new_moth_obj_idx = PosData.IDs.index(new_mothID)
                rp_budID = PosData.rp[bud_obj_idx]
                rp_new_mothID = PosData.rp[new_moth_obj_idx]
                if rp_budID.area >= rp_new_mothID.area:
                    msg = QtGui.QMessageBox()
                    msg.setIcon(msg.Warning)
                    msg.setText(
                        f'You clicked FIRST on ID {budID} and then on {new_mothID}.\n'
                        f'For me this means that you want ID {budID} to be the '
                        f'BUD of ID {new_mothID}.\n'
                        f'However ID {budID} is bigger than {new_mothID} '
                        f'so maybe you shoul have clicked FIRST on {new_mothID}?\n\n'
                        'What do you want me to do?'
                    )
                    swapButton = QPushButton(
                            f'Assign ID {new_mothID} as the bud of ID {budID}'
                    )
                    keepButton = QPushButton(
                            f'Keep ID {budID} as the bud of  ID {new_mothID}'
                    )
                    msg.addButton(swapButton, msg.YesRole)
                    msg.addButton(keepButton, msg.NoRole)
                    msg.exec_()
                    if msg.clickedButton() == swapButton:
                        (xdata, ydata,
                        self.xClickBud, self.yClickBud) = (
                            self.xClickBud, self.yClickBud,
                            xdata, ydata
                        )

            elif is_history_known and not self.clickedOnHistoryKnown:
                budID = PosData.lab[ydata, xdata]
                # Allow assigning an unknown cell ONLY to another unknown cell
                txt = (
                    f'You started by clicking on ID {budID} which has '
                    'UNKNOWN history, but you then clicked/released on '
                    f'ID {ID} which has KNOWN history.\n\n'
                    'Only two cells with UNKNOWN history can be assigned as '
                    'relative of each other.')
                msg = QtGui.QMessageBox()
                msg.critical(
                    self, 'Released on a cell with KNOWN history', txt, msg.Ok
                )
                return

            self.clickedOnHistoryKnown = is_history_known
            self.xClickMoth, self.yClickMoth = xdata, ydata
            self.assignBudMoth()
            self.assignBudMothButton.setChecked(False)
            self.clickedOnBud = False
            self.BudMothTempLine.setData([], [])

    def gui_mousePressEventImg1(self, event):
        PosData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        is_cca_on = mode == 'Cell cycle analysis' or self.isSnapshot
        right_click = event.button() == Qt.MouseButton.RightButton
        left_click = event.button() == Qt.MouseButton.LeftButton
        mid_click = event.button() == Qt.MouseButton.MidButton
        brushON = self.brushButton.isChecked()
        curvToolON = self.curvToolButton.isChecked()
        histON = self.setIsHistoryKnownButton.isChecked()
        eraserON = self.eraserButton.isChecked()
        rulerON = self.rulerButton.isChecked()

        dragImgLeft = (
            left_click and not brushON and not histON
            and not curvToolON and not eraserON and not rulerON
        )

        dragImgMiddle = mid_click

        # Right click in snapshot mode is for spline tool
        canAnnotateDivision = (
             not self.assignBudMothButton.isChecked()
             and not (self.isSnapshot and self.curvToolButton.isChecked())
        )

        canCurv = (curvToolON and not self.assignBudMothButton.isChecked()
                   and not brushON and not dragImgLeft and not eraserON)
        canBrush = (brushON and not curvToolON and not rulerON
                    and not dragImgLeft and not eraserON)
        canErase = (eraserON and not curvToolON and not rulerON
                    and not dragImgLeft and not brushON)
        canRuler = (rulerON and not curvToolON and not brushON
                    and not dragImgLeft and not brushON)

        # Enable dragging of the image window like pyqtgraph original code
        if dragImgLeft:
            pg.ImageItem.mousePressEvent(self.img1, event)
            event.ignore()
            return

        if dragImgMiddle:
            pg.ImageItem.mousePressEvent(self.img1, event)
            event.ignore()
            return

        if mode == 'Viewer':
            return

        # Allow right-click actions on both images
        eventOnImg2 = (
            (right_click)
            and (mode=='Segmentation and Tracking' or self.isSnapshot)
        )
        if eventOnImg2:
            self.gui_mousePressEventImg2(event)

        # Paint new IDs with brush and left click on the left image
        if left_click and canBrush:
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = PosData.lab.shape
            brushSize = self.brushSizeSpinbox.value()
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                ID = PosData.lab[ydata, xdata]

                # If user double-pressed 'b' then draw over the labels
                color = self.brushButton.palette().button().color().name()
                drawUnder = color != self.doublePressKeyButtonColor

                if ID > 0 and drawUnder:
                    PosData.brushID = PosData.lab[ydata, xdata]
                else:
                    # Update brush ID. Take care of disappearing cells to remember
                    # to not use their IDs anymore in the future
                    self.setBrushID()

                self.yPressAx2, self.xPressAx2 = y, x

                ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

                self.isMouseDragImg1 = True

                # Draw new objects
                localLab = PosData.lab[ymin:ymax, xmin:xmax]
                mask = diskMask.copy()
                if drawUnder:
                    mask[localLab!=0] = False

                PosData.lab[ymin:ymax, xmin:xmax][mask] = PosData.brushID
                self.setImageImg2()
                self.brushColor = self.img2.lut[PosData.brushID]/255

                rgb_shape = (PosData.lab.shape[0], PosData.lab.shape[1], 3)
                self.whiteRGB = np.array([1.0,1.0,1.0])

                img = self.img1.image.copy()
                img = img/img.max()
                if self.overlayButton.isChecked():
                    self.imgRGB = img/img.max()
                else:
                    self.imgRGB = gray2rgb(img)

                self.setTempImg1Brush(ymin, ymax, xmin, xmax, mask)

        elif left_click and canErase:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = PosData.lab.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)
                self.yPressAx2, self.xPressAx2 = y, x
                # Keep a list of erased IDs got erased
                self.erasedIDs = []
                self.erasedID = PosData.lab[ydata, xdata]

                ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)

                # Build eraser mask
                mask = np.zeros(PosData.lab.shape, bool)
                mask[ymin:ymax, xmin:xmax][diskMask] = True

                # If user double-pressed 'b' then erase over ALL labels
                color = self.eraserButton.palette().button().color().name()
                eraseOnlyOneID = (
                    color != self.doublePressKeyButtonColor
                    and self.erasedID != 0
                )

                self.eraseOnlyOneID = eraseOnlyOneID

                if eraseOnlyOneID:
                    mask[PosData.lab!=self.erasedID] = False


                self.erasedIDs.extend(PosData.lab[mask])

                PosData.lab[mask] = 0

                self.erasesedLab = np.zeros_like(PosData.lab)
                for erasedID in np.unique(self.erasedIDs):
                    if erasedID == 0:
                        continue
                    self.erasesedLab[PosData.lab==erasedID] = erasedID
                erasedRp = skimage.measure.regionprops(self.erasesedLab)
                for obj in erasedRp:
                    idx = obj.label-1
                    curveID = self.ax1_ContoursCurves[idx]
                    cont = self.getObjContours(obj)
                    curveID.setData(cont[:,0], cont[:,1], pen=self.oldIDs_cpen)

                self.img2.updateImage()
                self.isMouseDragImg1 = True

        elif left_click and canRuler:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            if not self.rulerHoverON:
                self.ax1_rulerAnchorsItem.setData([xdata], [ydata])
                self.rulerHoverON = True
            else:
                self.rulerHoverON = False
                xxRA, yyRA = self.ax1_rulerAnchorsItem.getData()
                self.ax1_rulerPlotItem.setData([xxRA[0], x], [yyRA[0], y])
                self.ax1_rulerAnchorsItem.setData([xxRA[0], x], [yyRA[0], y])

        elif right_click and canCurv:
            # Draw manually assisted auto contour
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            Y, X = PosData.lab.shape
            clickOnImg = x >= 0 and x < X and y >= 0 and y < Y
            if not clickOnImg:
                return

            self.autoCont_x0 = xdata
            self.autoCont_y0 = ydata
            self.xxA_autoCont, self.yyA_autoCont = [], []
            self.curvAnchors.addPoints([x], [y])
            img = self.getDisplayedCellsImg()
            self.autoContObjMask = np.zeros(img.shape, np.uint8)
            self.isRightClickDragImg1 = True

        elif left_click and canCurv:
            # Draw manual spline
            x, y = event.pos().x(), event.pos().y()
            Y, X = PosData.lab.shape
            clickOnImg = x >= 0 and x < X and y >= 0 and y < Y
            if not clickOnImg:
                return

            # Check if user clicked on starting anchor again --> close spline
            closeSpline = False
            clickedAnchors = self.curvAnchors.pointsAt(event.pos())
            xxA, yyA = self.curvAnchors.getData()
            if len(xxA)>0:
                if len(xxA) == 1:
                    self.splineHoverON = True
                x0, y0 = xxA[0], yyA[0]
                if len(clickedAnchors)>0:
                    xA_clicked, yA_clicked = clickedAnchors[0].pos()
                    if x0==xA_clicked and y0==yA_clicked:
                        x = x0
                        y = y0
                        closeSpline = True

            # Add anchors
            self.curvAnchors.addPoints([x], [y])
            try:
                xx, yy = self.curvHoverPlotItem.getData()
                self.curvPlotItem.setData(xx, yy)
            except Exception as e:
                # traceback.print_exc()
                pass

            if closeSpline:
                self.splineHoverON = False
                self.splineToObj()
                self.update_rp()
                self.tracking(enforce=True)
                self.updateALLimg()
                self.warnEditingWithCca_df('Add new ID with curvature tool')
                self.clearCurvItems()
                self.curvTool_cb(True)

        # Annotate cell cycle division
        elif right_click and is_cca_on and canAnnotateDivision:
            if PosData.frame_i <= 0 and not self.isSnapshot:
                return

            if PosData.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = PosData.lab[ydata, xdata]
            if ID == 0:
                divID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as divided',
                    parent=self
                )
                divID_prompt.exec_()
                if divID_prompt.cancel:
                    return
                else:
                    ID = divID_prompt.EntryID
                    obj_idx = PosData.IDs.index(ID)
                    y, x = PosData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            if not self.isSnapshot:
                # Annotate or undo division
                self.manualCellCycleAnnotation(ID)
            else:
                self.undoBudMothAssignment(ID)

        # Assign bud to mother (mouse down on bud)
        elif right_click and self.assignBudMothButton.isChecked():
            if self.clickedOnBud:
                # NOTE: self.clickedOnBud is set to False when assigning a mother
                # is successfull in mouse release event
                # We still have to click on a mother
                return

            if PosData.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = PosData.lab[ydata, xdata]
            if ID == 0:
                budID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID of a bud you want to correct mother assignment',
                    parent=self
                )
                budID_prompt.exec_()
                if budID_prompt.cancel:
                    return
                else:
                    ID = budID_prompt.EntryID

            obj_idx = PosData.IDs.index(ID)
            y, x = PosData.rp[obj_idx].centroid
            xdata, ydata = int(x), int(y)

            relationship = PosData.cca_df.at[ID, 'relationship']
            is_history_known = PosData.cca_df.at[ID, 'is_history_known']
            self.clickedOnHistoryKnown = is_history_known
            # We allow assiging a cell in G1 as bud only on first frame
            # OR if the history is unknown
            if relationship != 'bud' and PosData.frame_i > 0 and is_history_known:
                txt = (f'You clicked on ID {ID} which is NOT a bud.\n'
                       'To assign a bud to a cell start by clicking on a bud '
                       'and release on a cell in G1')
                msg = QtGui.QMessageBox()
                msg.critical(
                    self, 'Not a bud', txt, msg.Ok
                )
                return

            self.clickedOnBud = True
            self.xClickBud, self.yClickBud = xdata, ydata

        # Annotate (or undo) that cell has unknown history
        elif left_click and self.setIsHistoryKnownButton.isChecked():
            if PosData.cca_df is None:
                return

            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            ID = PosData.lab[ydata, xdata]
            if ID == 0:
                unknownID_prompt = apps.QLineEditDialog(
                    title='Clicked on background',
                    msg='You clicked on the background.\n'
                         'Enter ID that you want to annotate as '
                         '"history UNKNOWN/KNOWN"',
                    parent=self
                )
                unknownID_prompt.exec_()
                if unknownID_prompt.cancel:
                    return
                else:
                    ID = unknownID_prompt.EntryID
                    obj_idx = PosData.IDs.index(ID)
                    y, x = PosData.rp[obj_idx].centroid
                    xdata, ydata = int(x), int(y)

            self.annotateIsHistoryKnown(ID)
            self.setIsHistoryKnownButton.setChecked(False)

        # Allow mid-click actions on both images
        elif mid_click:
            self.gui_mousePressEventImg2(event)

    def getDistantGray(self, desiredGray, bkgrGray):
        isDesiredSimilarToBkgr = (
            abs(desiredGray-bkgrGray) < 0.3
        )
        if isDesiredSimilarToBkgr:
            return 1-desiredGray
        else:
            return desiredGray

    def RGBtoGray(self, R, G, B):
        # see https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
        C_linear = (0.2126*R + 0.7152*G + 0.0722*B)/255
        if C_linear <= 0.0031309:
            gray = 12.92*C_linear
        else:
            gray = 1.055*(C_linear)**(1/2.4) - 0.055
        return gray

    def ruler_cb(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            self.connectLeftClickButtons()
        else:
            self.rulerHoverON = False
            self.ax1_rulerPlotItem.setData([], [])
            self.ax1_rulerAnchorsItem.setData([], [])

    def getOptimalLabelItemColor(self, LabelItemID, desiredGray):
        img = self.img1.image
        img = img/img.max()
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        x, y = LabelItemID.pos()
        w, h, x, y = int(w), int(h), int(x), int(y)
        colors = img[y:y+h, x:x+w]
        bkgrGray = colors.mean(axis=(0,1))
        if isinstance(bkgrGray, np.ndarray):
            bkgrGray = self.RGBtoGray(*bkgrGray)
        optimalGray = self.getDistantGray(desiredGray, bkgrGray)
        return optimalGray

    def editImgProperties(self, checked=True):
        PosData = self.data[self.pos_i]
        PosData.askInputMetadata(
                                ask_TimeIncrement=True,
                                ask_PhysicalSizes=True,
                                save=True
        )

    def setHoverToolSymbolData(self, xx, yy, ScatterItems, size=None):
        for item in ScatterItems:
            if size is None:
                item.setData(xx, yy)
            else:
                item.setData(xx, yy, size=size)

    def setHoverToolSymbolColor(self, xdata, ydata, pen, ScatterItems, button,
                                brush=None, hoverRGB=None, ID=None):
        PosData = self.data[self.pos_i]
        hoverID = PosData.lab[ydata, xdata] if ID is None else ID
        color = button.palette().button().color().name()
        drawAbove = color == self.doublePressKeyButtonColor
        if hoverID == 0 or drawAbove:
            for item in ScatterItems:
                item.setPen(pen)
                item.setBrush(brush)
        else:
            rgb = self.img2.lut[hoverID] if hoverRGB is None else hoverRGB
            rgbPen = np.clip(rgb*1.2, 0, 255)
            for item in ScatterItems:
                item.setPen(*rgbPen, width=2)
                item.setBrush(*rgb, 100)

    def getCheckNormAction(self):
        normalize = False
        how = ''
        for action in self.normalizeQActionGroup.actions():
            if action.isChecked():
                how = action.text()
                normalize = True
                break
        return action, normalize, how

    def normalizeIntensities(self, img):
        action, normalize, how = self.getCheckNormAction()
        if not normalize:
            return img
        if how == 'Do not normalize. Display raw image':
            return img
        elif how == 'Convert to floating point format with values [0, 1]':
            img = skimage.img_as_float(img)
            return img
        # elif how == 'Rescale to 8-bit unsigned integer format with values [0, 255]':
        #     img = skimage.img_as_float(img)
        #     img = (img*255).astype(np.uint8)
        #     return img
        elif how == 'Rescale to [0, 1]':
            img = skimage.img_as_float(img)
            img = skimage.exposure.rescale_intensity(img)
            return img
        elif how == 'Normalize by max value':
            img = img/img.max()
        return img

    def removeAlldelROIsCurrentFrame(self):
        PosData = self.data[self.pos_i]
        delROIs_info = PosData.allData_li[PosData.frame_i]['delROIs_info']
        rois = delROIs_info['rois'].copy()
        for roi in delROIs_info['rois']:
            self.ax2.removeItem(roi)

        # Collect garbage ROIs:
        for item in self.ax2.items:
            if isinstance(item, pg.ROI):
                self.ax2.removeItem(item)

    def removeROI(self, event):
        PosData = self.data[self.pos_i]
        current_frame_i = PosData.frame_i
        self.store_data()
        for i in range(PosData.frame_i, PosData.segmSizeT):
            delROIs_info = PosData.allData_li[i]['delROIs_info']
            if self.roi_to_del in delROIs_info['rois']:
                PosData.frame_i = i
                idx = delROIs_info['rois'].index(self.roi_to_del)
                # Restore deleted IDs from already visited frames
                if PosData.allData_li[i]['labels'] is not None:
                    if len(delROIs_info['delIDsROI'][idx]) > 1:
                        PosData.lab = PosData.allData_li[i]['labels']
                        self.restoreDelROIlab(self.roi_to_del, enforce=True)
                        PosData.allData_li[i]['labels'] = PosData.lab
                        self.get_data()
                        self.store_data()
                delROIs_info['rois'].pop(idx)
                delROIs_info['delMasks'].pop(idx)
                delROIs_info['delIDsROI'].pop(idx)

        # Back to current frame
        PosData.frame_i = current_frame_i
        PosData.lab = PosData.allData_li[PosData.frame_i]['labels']
        self.ax2.removeItem(self.roi_to_del)
        self.get_data()
        self.updateALLimg()

    def getPolygonBrush(self, yxc2, Y, X):
        # see https://en.wikipedia.org/wiki/Tangent_lines_to_circles
        y1, x1 = self.yPressAx2, self.xPressAx2
        y2, x2 = yxc2
        R = self.brushSizeSpinbox.value()
        r = R

        arcsin_den = np.sqrt((x2-x1)**2+(y2-y1)**2)
        arctan_den = (x2-x1)
        if arcsin_den!=0 and arctan_den!=0:
            beta = np.arcsin((R-r)/arcsin_den)
            gamma = -np.arctan((y2-y1)/arctan_den)
            alpha = gamma-beta
            x3 = x1 + r*np.sin(alpha)
            y3 = y1 + r*np.cos(alpha)
            x4 = x2 + R*np.sin(alpha)
            y4 = y2 + R*np.cos(alpha)

            alpha = gamma+beta
            x5 = x1 - r*np.sin(alpha)
            y5 = y1 - r*np.cos(alpha)
            x6 = x2 - R*np.sin(alpha)
            y6 = y2 - R*np.cos(alpha)

            rr_poly, cc_poly = skimage.draw.polygon([y3, y4, y6, y5],
                                                    [x3, x4, x6, x5],
                                                    shape=(Y, X))
        else:
            rr_poly, cc_poly = [], []

        self.yPressAx2, self.xPressAx2 = y2, x2
        return rr_poly, cc_poly

    def get_dir_coords(self, alfa_dir, yd, xd, shape, connectivity=1):
        h, w = shape
        y_above = yd+1 if yd+1 < h else yd
        y_below = yd-1 if yd > 0 else yd
        x_right = xd+1 if xd+1 < w else xd
        x_left = xd-1 if xd > 0 else xd
        if alfa_dir == 0:
            yy = [y_below, y_below, yd, y_above, y_above]
            xx = [xd, x_right, x_right, x_right, xd]
        elif alfa_dir == 45:
            yy = [y_below, y_below, y_below, yd, y_above]
            xx = [x_left, xd, x_right, x_right, x_right]
        elif alfa_dir == 90:
            yy = [yd, y_below, y_below, y_below, yd]
            xx = [x_left, x_left, xd, x_right, x_right]
        elif alfa_dir == 135:
            yy = [y_above, yd, y_below, y_below, y_below]
            xx = [x_left, x_left, x_left, xd, x_right]
        elif alfa_dir == -180 or alfa_dir == 180:
            yy = [y_above, y_above, yd, y_below, y_below]
            xx = [xd, x_left, x_left, x_left, xd]
        elif alfa_dir == -135:
            yy = [y_below, yd, y_above, y_above, y_above]
            xx = [x_left, x_left, x_left, xd, x_right]
        elif alfa_dir == -90:
            yy = [yd, y_above, y_above, y_above, yd]
            xx = [x_left, x_left, xd, x_right, x_right]
        else:
            yy = [y_above, y_above, y_above, yd, y_below]
            xx = [x_left, xd, x_right, x_right, x_right]
        if connectivity == 1:
            return yy[1:4], xx[1:4]
        else:
            return yy, xx

    def drawAutoContour(self, y2, x2):
        y1, x1 = self.autoCont_y0, self.autoCont_x0
        Dy = abs(y2-y1)
        Dx = abs(x2-x1)
        edge = self.getDisplayedCellsImg()
        if Dy != 0 or Dx != 0:
            # NOTE: numIter takes care of any lag in mouseMoveEvent
            numIter = int(round(max((Dy, Dx))))
            alfa = np.arctan2(y1-y2, x2-x1)
            base = np.pi/4
            alfa_dir = round((base * round(alfa/base))*180/np.pi)
            for _ in range(numIter):
                y1, x1 = self.autoCont_y0, self.autoCont_x0
                yy, xx = self.get_dir_coords(alfa_dir, y1, x1, edge.shape)
                a_dir = edge[yy, xx]
                if self.invertBwAction.isChecked():
                    min_int = a_dir.min() # if int_val > ta else a_dir.min()
                else:
                    min_int = a_dir.max()
                min_i = list(a_dir).index(min_int)
                y, x = yy[min_i], xx[min_i]
                try:
                    xx, yy = self.curvHoverPlotItem.getData()
                except TypeError:
                    xx, yy = [], []
                xx = np.r_[xx, x]
                yy = np.r_[yy, y]
                try:
                    self.curvHoverPlotItem.setData(xx, yy)
                except TypeError:
                    pass
                self.autoCont_y0, self.autoCont_x0 = y, x
                self.smoothAutoContWithSpline()

    def smoothAutoContWithSpline(self):
        try:
            xx, yy = self.curvHoverPlotItem.getData()
            # Downsample by taking every nth coord
            n = 3
            xxA, yyA = xx[::n], yy[::n]
            rr, cc = skimage.draw.polygon(yyA, xxA)
            self.autoContObjMask[rr, cc] = 1
            rp = skimage.measure.regionprops(self.autoContObjMask)
            if not rp:
                return
            obj = rp[0]
            cont = self.getObjContours(obj)
            xxC, yyC = cont[:,0], cont[:,1]
            xxA, yyA = xxC[::n], yyC[::n]
            self.xxA_autoCont, self.yyA_autoCont = xxA, yyA
            xxS, yyS = self.getSpline(xxA, yyA, per=True, appendFirst=True)
            if len(xxS)>0:
                self.curvPlotItem.setData(xxS, yyS)
        except TypeError:
            pass

    def updateIsHistoryKnown():
        """
        This function is called every time the user saves and it is used
        for updating the status of cells where we don't know the history

        There are three possibilities:

        1. The cell with unknown history is a BUD
           --> we don't know when that  bud emerged --> 'emerg_frame_i' = -1
        2. The cell with unknown history is a MOTHER cell
           --> we don't know emerging frame --> 'emerg_frame_i' = -1
               AND generation number --> we start from 'generation_num' = 2
        3. The cell with unknown history is a CELL in G1
           --> we don't know emerging frame -->  'emerg_frame_i' = -1
               AND generation number --> we start from 'generation_num' = 2
               AND relative's ID in the previous cell cycle --> 'relative_ID' = -1
        """
        pass

    def getStatusKnownHistoryBud(self, ID):
        PosData = self.data[self.pos_i]
        cca_df_ID = None
        for i in range(PosData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            is_cell_existing = is_bud_existing = ID in cca_df_i.index
            if not is_cell_existing:
                cca_df_ID = pd.Series({
                    'cell_cycle_stage': 'S',
                    'generation_num': 0,
                    'relative_ID': -1,
                    'relationship': 'bud',
                    'emerg_frame_i': i+1,
                    'division_frame_i': -1,
                    'is_history_known': True,
                    'corrected_assignment': False
                })
                return cca_df_ID

    def setHistoryKnowledge(self, ID, cca_df):
        is_history_known = cca_df.at[ID, 'is_history_known']
        if is_history_known:
            cca_df.at[ID, 'is_history_known'] = False
            cca_df.at[ID, 'cell_cycle_stage'] = 'G1'
            cca_df.at[ID, 'generation_num'] += 2
            cca_df.at[ID, 'emerg_frame_i'] = -1
            cca_df.at[ID, 'relative_ID'] = -1
            cca_df.at[ID, 'relationship'] = 'mother'
        else:
            cca_df.loc[ID] = PosData.ccaStatus_whenEmerged[ID]

    def annotateIsHistoryKnown(self, ID):
        """
        This function is used for annotating that a cell has unknown or known
        history. Cells with unknown history are for example the cells already
        present in the first frame or cells that appear in the frame from
        outside of the field of view.

        With this function we simply set 'is_history_known' to False.
        When the users saves instead we update the entire staus of the cell
        with unknown history with the function "updateIsHistoryKnown()"
        """
        PosData = self.data[self.pos_i]
        is_history_known = PosData.cca_df.at[ID, 'is_history_known']
        relID = PosData.cca_df.at[ID, 'relative_ID']
        if relID in PosData.cca_df.index:
            relID_cca = self.getStatus_RelID_BeforeEmergence(ID, relID)

        if is_history_known:
            # Save status of ID when emerged to allow undoing
            statusID_whenEmerged = self.getStatusKnownHistoryBud(ID)
            if statusID_whenEmerged is None:
                return
            PosData.ccaStatus_whenEmerged[ID] = statusID_whenEmerged

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(PosData.frame_i, PosData.cca_df, undoId)

        self.setHistoryKnowledge(ID, PosData.cca_df)

        if relID in PosData.cca_df.index:
            # If the cell with unknown history has a relative ID assigned to it
            # we set the cca of it to the status it had BEFORE the assignment
            PosData.cca_df.loc[relID] = relID_cca

        # Update cell cycle info LabelItems
        obj_idx = PosData.IDs.index(ID)
        rp_ID = PosData.rp[obj_idx]
        self.drawID_and_Contour(rp_ID, drawContours=False)

        if relID in PosData.IDs:
            relObj_idx = PosData.IDs.index(relID)
            rp_relID = PosData.rp[relObj_idx]
            self.drawID_and_Contour(rp_relID, drawContours=False)

        self.store_cca_df()

        if self.ccaTableWin is not None:
            self.ccaTableWin.updateTable(PosData.cca_df)

        # Correct future frames
        for i in range(PosData.frame_i+1, PosData.segmSizeT):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            IDs = cca_df_i.index
            if ID not in IDs:
                # For some reason ID disappeared from this frame
                continue
            else:
                self.setHistoryKnowledge(ID, cca_df_i)
                if relID in IDs:
                    cca_df_i.loc[relID] = relID_cca
                self.store_cca_df(frame_i=i, cca_df=cca_df_i)


        # Correct past frames
        for i in range(PosData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            IDs = cca_df_i.index
            if ID not in IDs:
                # we reached frame where ID was not existing yet
                break
            else:
                relID = cca_df_i.at[ID, 'relative_ID']
                self.setHistoryKnowledge(ID, cca_df_i)
                if relID in IDs:
                    cca_df_i.loc[relID] = relID_cca
                self.store_cca_df(frame_i=i, cca_df=cca_df_i)

    def annotateDivision(self, cca_df, ID, relID, ccs_relID):
        # Correct as follows:
        # If S then correct to G1 and +1 on generation number
        PosData = self.data[self.pos_i]
        store = False
        cca_df.at[ID, 'cell_cycle_stage'] = 'G1'
        gen_num_clickedID = cca_df.at[ID, 'generation_num']
        cca_df.at[ID, 'generation_num'] += 1
        cca_df.at[ID, 'division_frame_i'] = PosData.frame_i
        cca_df.at[relID, 'cell_cycle_stage'] = 'G1'
        gen_num_relID = cca_df.at[relID, 'generation_num']
        cca_df.at[relID, 'generation_num'] += 1
        cca_df.at[relID, 'division_frame_i'] = PosData.frame_i
        if gen_num_clickedID < gen_num_relID:
            cca_df.at[ID, 'relationship'] = 'mother'
        else:
            cca_df.at[relID, 'relationship'] = 'mother'
        store = True
        return store

    def undoDivisionAnnotation(self, cca_df, ID, relID, ccs_relID):
        # Correct as follows:
        # If G1 then correct to S and -1 on generation number
        store = False
        cca_df.at[ID, 'cell_cycle_stage'] = 'S'
        gen_num_clickedID = cca_df.at[ID, 'generation_num']
        cca_df.at[ID, 'generation_num'] -= 1
        cca_df.at[ID, 'division_frame_i'] = -1
        cca_df.at[relID, 'cell_cycle_stage'] = 'S'
        gen_num_relID = cca_df.at[relID, 'generation_num']
        cca_df.at[relID, 'generation_num'] -= 1
        cca_df.at[relID, 'division_frame_i'] = -1
        if gen_num_clickedID < gen_num_relID:
            cca_df.at[ID, 'relationship'] = 'bud'
        else:
            cca_df.at[relID, 'relationship'] = 'bud'
        store = True
        return store

    def undoBudMothAssignment(self, ID):
        PosData = self.data[self.pos_i]
        relID = PosData.cca_df.at[ID, 'relative_ID']
        ccs = PosData.cca_df.at[ID, 'cell_cycle_stage']
        if ccs == 'G1':
            return
        PosData.cca_df.at[ID, 'relative_ID'] = -1
        PosData.cca_df.at[ID, 'generation_num'] = 2
        PosData.cca_df.at[ID, 'cell_cycle_stage'] = 'G1'
        PosData.cca_df.at[ID, 'relationship'] = 'mother'
        if relID in PosData.cca_df.index:
            PosData.cca_df.at[relID, 'relative_ID'] = -1
            PosData.cca_df.at[relID, 'generation_num'] = 2
            PosData.cca_df.at[relID, 'cell_cycle_stage'] = 'G1'
            PosData.cca_df.at[relID, 'relationship'] = 'mother'

        obj_idx = PosData.IDs.index(ID)
        relObj_idx = PosData.IDs.index(relID)
        rp_ID = PosData.rp[obj_idx]
        rp_relID = PosData.rp[relObj_idx]

        self.store_cca_df()

        # Update cell cycle info LabelItems
        self.drawID_and_Contour(rp_ID, drawContours=False)
        self.drawID_and_Contour(rp_relID, drawContours=False)


        if self.ccaTableWin is not None:
            self.ccaTableWin.updateTable(PosData.cca_df)

    def manualCellCycleAnnotation(self, ID):
        """
        This function is used for both annotating division or undoing the
        annotation. It can be called on any frame.

        If we annotate division (right click on a cell in S) then it will
        check if there are future frames to correct.
        Frames to correct are those frames where both the mother and the bud
        are annotated as S phase cells.
        In this case we assign all those frames to G1, relationship to mother,
        and +1 generation number

        If we undo the annotation (right click on a cell in G1) then it will
        correct both past and future annotated frames (if present).
        Frames to correct are those frames where both the mother and the bud
        are annotated as G1 phase cells.
        In this case we assign all those frames to G1, relationship back to
        bud, and -1 generation number
        """
        PosData = self.data[self.pos_i]

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(PosData.frame_i, PosData.cca_df, undoId)

        # Correct current frame
        clicked_ccs = PosData.cca_df.at[ID, 'cell_cycle_stage']
        relID = PosData.cca_df.at[ID, 'relative_ID']

        if relID not in PosData.IDs:
            return

        ccs_relID = PosData.cca_df.at[relID, 'cell_cycle_stage']
        if clicked_ccs == 'S':
            store = self.annotateDivision(
                                PosData.cca_df, ID, relID, ccs_relID)
            self.store_cca_df()
        else:
            store = self.undoDivisionAnnotation(
                                PosData.cca_df, ID, relID, ccs_relID)
            self.store_cca_df()

        obj_idx = PosData.IDs.index(ID)
        relObj_idx = PosData.IDs.index(relID)
        rp_ID = PosData.rp[obj_idx]
        rp_relID = PosData.rp[relObj_idx]

        # Update cell cycle info LabelItems
        self.drawID_and_Contour(rp_ID, drawContours=False)
        self.drawID_and_Contour(rp_relID, drawContours=False)

        if self.ccaTableWin is not None:
            self.ccaTableWin.updateTable(PosData.cca_df)


        # Correct future frames
        for i in range(PosData.frame_i+1, PosData.segmSizeT):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            IDs = cca_df_i.index
            if ID not in IDs:
                # For some reason ID disappeared from this frame
                continue
            else:
                ccs = cca_df_i.at[ID, 'cell_cycle_stage']
                relID = cca_df_i.at[ID, 'relative_ID']
                ccs_relID = cca_df_i.at[relID, 'cell_cycle_stage']
                if clicked_ccs == 'S':
                    if ccs == 'G1':
                        # Cell is in G1 in the future again so stop annotating
                        break
                    self.annotateDivision(cca_df_i, ID, relID, ccs_relID)
                    self.store_cca_df(frame_i=i, cca_df=cca_df_i)
                else:
                    if ccs == 'S':
                        # Cell is in S in the future again so stop undoing (break)
                        # also leave a 1 frame duration G1 to avoid a continuous
                        # S phase
                        self.annotateDivision(cca_df_i, ID, relID, ccs_relID)
                        self.store_cca_df(frame_i=i, cca_df=cca_df_i)
                        break
                    store = self.undoDivisionAnnotation(
                                        cca_df_i, ID, relID, ccs_relID)
                    self.store_cca_df(frame_i=i, cca_df=cca_df_i)

        # Correct past frames
        for i in range(PosData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if ID not in cca_df_i.index or relID not in cca_df_i.index:
                # Bud did not exist at frame_i = i
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            ccs = cca_df_i.at[ID, 'cell_cycle_stage']
            relID = cca_df_i.at[ID, 'relative_ID']
            ccs_relID = cca_df_i.at[relID, 'cell_cycle_stage']
            if ccs == 'S':
                # We correct only those frames in which the ID was in 'S'
                break
            else:
                store = self.undoDivisionAnnotation(
                                       cca_df_i, ID, relID, ccs_relID)
                self.store_cca_df(frame_i=i, cca_df=cca_df_i)

    def checkMothEligibility(self, budID, new_mothID):
        PosData = self.data[self.pos_i]
        """Check the new mother is in G1 for the entire life of the bud"""

        eligible = True

        # Check future frames
        for i in range(PosData.frame_i, PosData.segmSizeT):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                return eligible

            is_still_bud = cca_df_i.at[budID, 'relationship'] == 'bud'
            if not is_still_bud:
                # Bud life ended
                return eligible
            ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if ccs != 'G1':
                err_msg = (
                    f'The requested cell in G1 (ID={new_mothID}) at future frame {i+1} '
                    'has a bud assigned to it, therefore it cannot be '
                    f'assigned as the mother of bud ID {budID}.\n'
                    f'You can assign a cell as the mother of bud ID {budID} '
                    'only if this cell is in G1 for the entire life of the bud.\n\n'
                    'One possible solution is to click on "cancel", go to '
                    f'frame {i+1} and  assign the bud of cell {new_mothID} '
                    'to another cell.\n'
                    f'A second solution is to assign bud ID {budID} to cell '
                    f'{new_mothID} anyway by clicking "Apply".'
                    '\n\nHowever to ensure correctness of '
                    'future assignments the system will delete any cell cycle '
                    f'information from frame {i+1} to the end. Therefore, you '
                    'will have to visit those frames again.\n\n'
                    'The deletion of cell cycle information CANNOT BE UNDONE! '
                    'Saved data is not changed of course.\n\n'
                    'Apply assignment or cancel process?')
                msg = QtGui.QMessageBox()
                enforce_assignment = msg.warning(
                   self, 'Cell not eligible', err_msg, msg.Apply | msg.Cancel
                )
                if enforce_assignment == msg.Cancel:
                    eligible = False
                else:
                    self.remove_future_cca_df(i)
                return eligible

        # Check past frames
        for i in range(PosData.frame_i-1, -1, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            if not is_bud_existing:
                # Bud was not emerged yet
                return eligible

            ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if ccs != 'G1':
                err_msg = (
                    f'The requested cell in G1 (ID={new_mothID}) at past frame {i+1} '
                    'has a bud assigned to it, therefore it cannot be '
                    f'assigned as mother of bud ID {budID}.\n'
                    f'You can assign a cell as the mother of bud ID {budID} '
                    'only if this cell is in G1 for the entire life of the bud.\n'
                    f'One possible solution is to first go to frame {i+1} and '
                    f'assign the bud of cell {new_mothID} to another cell.')
                msg = QtGui.QMessageBox()
                msg.critical(
                   self, 'Cell not eligible', err_msg, msg.Ok
                )
                eligible = False
                return eligible

    def getStatus_RelID_BeforeEmergence(self, budID, curr_mothID):
        PosData = self.data[self.pos_i]
        # Get status of the current mother before it had budID assigned to it
        for i in range(PosData.frame_i-1, -1, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            if not is_bud_existing:
                # Bud was not emerged yet
                if curr_mothID in cca_df_i.index:
                    return cca_df_i.loc[curr_mothID]
                else:
                    # The bud emerged together with the mother because
                    # they appeared together from outside of the fov
                    # and they were trated as new IDs bud in S0
                    return pd.Series({
                        'cell_cycle_stage': 'S',
                        'generation_num': 0,
                        'relative_ID': -1,
                        'relationship': 'bud',
                        'emerg_frame_i': i+1,
                        'division_frame_i': -1,
                        'is_history_known': True,
                        'corrected_assignment': False
                    })

    def assignBudMoth(self):
        """
        This function is used for correcting automatic mother-bud assignment.

        It can be called at any frame of the bud life.

        There are three cells involved: bud, current mother, new mother.

        Eligibility:
            - User clicked first on a bud (checked at click time)
            - User released mouse button on a cell in G1 (checked at release time)
            - The new mother MUST be in G1 for all the frames of the bud life
              --> if not warn

        Result:
            - The bud only changes relative ID to the new mother
            - The new mother changes relative ID and stage to 'S'
            - The old mother changes its entire status to the status it had
              before being assigned to the clicked bud
        """
        PosData = self.data[self.pos_i]
        budID = PosData.lab[self.yClickBud, self.xClickBud]
        new_mothID = PosData.lab[self.yClickMoth, self.xClickMoth]

        if budID == new_mothID:
            return

        # Allow partial initialization of cca_df with mouse
        singleFrameCca = (
            (PosData.frame_i == 0 and budID != new_mothID)
            or (self.isSnapshot and budID != new_mothID)
        )
        if singleFrameCca:
            newMothCcs = PosData.cca_df.at[new_mothID, 'cell_cycle_stage']
            if not newMothCcs == 'G1':
                err_msg = (
                    'You are assigning the bud to a cell that is not in G1!'
                )
                msg = QtGui.QMessageBox()
                msg.critical(
                   self, 'New mother not in G1!', err_msg, msg.Ok
                )
                return
            # Store cca_df for undo action
            undoId = uuid.uuid4()
            self.storeUndoRedoCca(0, PosData.cca_df, undoId)
            currentRelID = PosData.cca_df.at[budID, 'relative_ID']
            if currentRelID in PosData.cca_df.index:
                PosData.cca_df.at[currentRelID, 'relative_ID'] = -1
                PosData.cca_df.at[currentRelID, 'generation_num'] = 2
                PosData.cca_df.at[currentRelID, 'cell_cycle_stage'] = 'G1'
                currentRelObjIdx = PosData.IDs.index(currentRelID)
                currentRelObj = PosData.rp[currentRelObjIdx]
                self.drawID_and_Contour(currentRelObj, drawContours=False)
            PosData.cca_df.at[budID, 'relationship'] = 'bud'
            PosData.cca_df.at[budID, 'generation_num'] = 0
            PosData.cca_df.at[budID, 'relative_ID'] = new_mothID
            PosData.cca_df.at[budID, 'cell_cycle_stage'] = 'S'
            PosData.cca_df.at[new_mothID, 'relative_ID'] = budID
            PosData.cca_df.at[new_mothID, 'generation_num'] = 2
            PosData.cca_df.at[new_mothID, 'cell_cycle_stage'] = 'S'
            bud_obj_idx = PosData.IDs.index(budID)
            new_moth_obj_idx = PosData.IDs.index(new_mothID)
            rp_budID = PosData.rp[bud_obj_idx]
            rp_new_mothID = PosData.rp[new_moth_obj_idx]
            self.drawID_and_Contour(rp_budID, drawContours=False)
            self.drawID_and_Contour(rp_new_mothID, drawContours=False)
            self.store_cca_df()
            return

        curr_mothID = PosData.cca_df.at[budID, 'relative_ID']

        eligible = self.checkMothEligibility(budID, new_mothID)
        if not eligible:
            return

        if curr_mothID in PosData.cca_df.index:
            curr_moth_cca = self.getStatus_RelID_BeforeEmergence(
                                                         budID, curr_mothID)

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(PosData.frame_i, PosData.cca_df, undoId)
        # Correct current frames and update LabelItems
        PosData.cca_df.at[budID, 'relative_ID'] = new_mothID
        PosData.cca_df.at[budID, 'generation_num'] = 0
        PosData.cca_df.at[budID, 'relative_ID'] = new_mothID
        PosData.cca_df.at[budID, 'relationship'] = 'bud'
        PosData.cca_df.at[budID, 'corrected_assignment'] = True
        PosData.cca_df.at[budID, 'cell_cycle_stage'] = 'S'

        PosData.cca_df.at[new_mothID, 'relative_ID'] = budID
        PosData.cca_df.at[new_mothID, 'cell_cycle_stage'] = 'S'
        PosData.cca_df.at[new_mothID, 'relationship'] = 'mother'

        if curr_mothID in PosData.cca_df.index:
            # Cells with UNKNOWN history has relative's ID = -1
            # which is not an existing cell
            PosData.cca_df.loc[curr_mothID] = curr_moth_cca

        bud_obj_idx = PosData.IDs.index(budID)
        new_moth_obj_idx = PosData.IDs.index(new_mothID)
        if curr_mothID in PosData.cca_df.index:
            curr_moth_obj_idx = PosData.IDs.index(curr_mothID)
        rp_budID = PosData.rp[bud_obj_idx]
        rp_new_mothID = PosData.rp[new_moth_obj_idx]


        self.drawID_and_Contour(rp_budID, drawContours=False)
        self.drawID_and_Contour(rp_new_mothID, drawContours=False)

        if curr_mothID in PosData.cca_df.index:
            rp_curr_mothID = PosData.rp[curr_moth_obj_idx]
            self.drawID_and_Contour(rp_curr_mothID, drawContours=False)

        self.checkMultiBudMOth(draw=True)

        self.store_cca_df()

        if self.ccaTableWin is not None:
            self.ccaTableWin.updateTable(PosData.cca_df)

        # Correct future frames
        for i in range(PosData.frame_i+1, PosData.segmSizeT):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            IDs = cca_df_i.index
            if budID not in IDs or new_mothID not in IDs:
                # For some reason ID disappeared from this frame
                continue

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            bud_relationship = cca_df_i.at[budID, 'relationship']
            bud_ccs = cca_df_i.at[budID, 'cell_cycle_stage']

            if bud_relationship == 'mother' and bud_ccs == 'S':
                # The bud at the ith frame budded itself --> stop
                break

            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'generation_num'] = 0
            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'relationship'] = 'bud'
            cca_df_i.at[budID, 'cell_cycle_stage'] = 'S'

            newMoth_bud_ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if newMoth_bud_ccs == 'G1':
                # Assign bud to new mother only if the new mother is in G1
                # This can happen if the bud already has a G1 annotated
                cca_df_i.at[new_mothID, 'relative_ID'] = budID
                cca_df_i.at[new_mothID, 'cell_cycle_stage'] = 'S'
                cca_df_i.at[new_mothID, 'relationship'] = 'mother'

            if curr_mothID in cca_df_i.index:
                # Cells with UNKNOWN history has relative's ID = -1
                # which is not an existing cell
                cca_df_i.loc[curr_mothID] = curr_moth_cca

            self.store_cca_df(frame_i=i, cca_df=cca_df_i)

        # Correct past frames
        for i in range(PosData.frame_i-1, -1, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            if not is_bud_existing:
                # Bud was not emerged yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'generation_num'] = 0
            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'relationship'] = 'bud'
            cca_df_i.at[budID, 'cell_cycle_stage'] = 'S'

            cca_df_i.at[new_mothID, 'relative_ID'] = budID
            cca_df_i.at[new_mothID, 'cell_cycle_stage'] = 'S'
            cca_df_i.at[new_mothID, 'relationship'] = 'mother'

            if curr_mothID in cca_df_i.index:
                # Cells with UNKNOWN history has relative's ID = -1
                # which is not an existing cell
                cca_df_i.loc[curr_mothID] = curr_moth_cca

            self.store_cca_df(frame_i=i, cca_df=cca_df_i)

    def getSpline(self, xx, yy, resolutionSpace=None, per=False, appendFirst=False):
        # Remove duplicates
        valid = np.where(np.abs(np.diff(xx)) + np.abs(np.diff(yy)) > 0)
        if appendFirst:
            xx = np.r_[xx[valid], xx[-1], xx[0]]
            yy = np.r_[yy[valid], yy[-1], yy[0]]
        else:
            xx = np.r_[xx[valid], xx[-1]]
            yy = np.r_[yy[valid], yy[-1]]

        # Interpolate splice
        if resolutionSpace is None:
            resolutionSpace = self.hoverLinSpace
        k = 2 if len(xx) == 3 else 3
        try:
            tck, u = scipy.interpolate.splprep(
                        [xx, yy], s=0, k=k, per=per)
            xi, yi = scipy.interpolate.splev(resolutionSpace, tck)
            return xi, yi
        except (ValueError, TypeError):
            # Catch errors where we know why splprep fails
            return [], []

    def uncheckQButton(self, button):
        # Manual exclusive where we allow to uncheck all buttons
        for b in self.checkableQButtonsGroup.buttons():
            if b != button:
                b.setChecked(False)

    def addDelROI(self, event):
        PosData = self.data[self.pos_i]
        self.warnEditingWithCca_df('Delete IDs using ROI')
        roi = self.getDelROI()
        for i in range(PosData.frame_i, PosData.segmSizeT):
            delROIs_info = PosData.allData_li[i]['delROIs_info']
            delROIs_info['rois'].append(roi)
            delROIs_info['delMasks'].append(np.zeros_like(PosData.lab))
            delROIs_info['delIDsROI'].append(set())
        self.ax2.addItem(roi)

    def getDelROI(self, xl=None, yb=None, w=32, h=32):
        PosData = self.data[self.pos_i]
        if xl is None:
            xRange, yRange = self.ax2.viewRange()
            xl, yb = abs(xRange[0]), abs(yRange[0])
        Y, X = PosData.lab.shape
        roi = pg.ROI([xl, yb], [w, h],
                     rotatable=False,
                     removable=True,
                     pen=pg.mkPen(color='r'),
                     maxBounds=QRectF(QRect(0,0,X,Y)))

        roi.handleSize = 7

        ## handles scaling horizontally around center
        roi.addScaleHandle([1, 0.5], [0, 0.5])
        roi.addScaleHandle([0, 0.5], [1, 0.5])

        ## handles scaling vertically from opposite edge
        roi.addScaleHandle([0.5, 0], [0.5, 1])
        roi.addScaleHandle([0.5, 1], [0.5, 0])

        ## handles scaling both vertically and horizontally
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addScaleHandle([0, 1], [1, 0])
        roi.addScaleHandle([1, 0], [0, 1])

        roi.sigRegionChanged.connect(self.ROImoving)
        roi.sigRegionChangeFinished.connect(self.cropROIovingFinished)
        return roi

    def ROImoving(self, roi):
        roi.setPen(color=(255,255,0))
        # First bring back IDs if the ROI moved away
        self.restoreDelROIlab(roi)
        self.setImageImg2()

    def cropROIovingFinished(self, roi):
        roi.setPen(color='r')
        self.update_rp()
        self.updateALLimg()

    def restoreDelROIlab(self, roi, enforce=True):
        PosData = self.data[self.pos_i]
        x0, y0 = [int(c) for c in roi.pos()]
        w, h = [int(c) for c in roi.size()]
        delROIs_info = PosData.allData_li[PosData.frame_i]['delROIs_info']
        idx = delROIs_info['rois'].index(roi)
        delMask = delROIs_info['delMasks'][idx]
        delIDs = delROIs_info['delIDsROI'][idx]
        ROImask = np.zeros(self.img2.image.shape, bool)
        ROImask[y0:y0+h, x0:x0+w] = True
        overlapROIdelIDs = np.unique(delMask[ROImask])
        for ID in delIDs:
            if ID>0 and ID not in overlapROIdelIDs and not enforce:
                PosData.lab[delMask==ID] = ID
                delMask[delMask==ID] = 0
            elif ID>0 and enforce:
                PosData.lab[delMask==ID] = ID
                delMask[delMask==ID] = 0

    def getDelROIlab(self):
        PosData = self.data[self.pos_i]
        DelROIlab = PosData.lab
        allDelIDs = set()
        # Iterate rois and delete IDs
        for roi in PosData.allData_li[PosData.frame_i]['delROIs_info']['rois']:
            if roi not in self.ax2.items:
                continue
            ROImask = np.zeros(PosData.lab.shape, bool)
            delROIs_info = PosData.allData_li[PosData.frame_i]['delROIs_info']
            idx = delROIs_info['rois'].index(roi)
            delObjROImask = delROIs_info['delMasks'][idx]
            delIDsROI = delROIs_info['delIDsROI'][idx]
            x0, y0 = [int(c) for c in roi.pos()]
            w, h = [int(c) for c in roi.size()]
            ROImask[y0:y0+h, x0:x0+w] = True
            delIDs = np.unique(PosData.lab[ROImask])
            delIDsROI.update(delIDs)
            allDelIDs.update(delIDs)
            _DelROIlab = PosData.lab.copy()
            for obj in PosData.rp:
                ID = obj.label
                if ID in delIDs:
                    delObjROImask[PosData.lab==ID] = ID
                    _DelROIlab[PosData.lab==ID] = 0
                    # LabelItemID = self.ax2_LabelItemsIDs[ID-1]
                    # LabelItemID.setText('')
                    # LabelItemID = self.ax1_LabelItemsIDs[ID-1]
                    # LabelItemID.setText('')
            DelROIlab[_DelROIlab == 0] = 0
            # Keep a mask of deleted IDs to bring them back when roi moves
            delROIs_info['delMasks'][idx] = delObjROImask
            delROIs_info['delIDsROI'][idx] = delIDsROI
        return allDelIDs, DelROIlab

    def gaussBlur(self, checked):
        if checked:
            font = QtGui.QFont()
            font.setPointSize(10)
            self.gaussWin = apps.gaussBlurDialog(self)
            self.gaussWin.setFont(font)
            self.gaussWin.show()
        else:
            self.gaussWin.close()
            self.gaussWin = None

    def edgeDetection(self, checked):
        if checked:
            font = QtGui.QFont()
            font.setPointSize(10)
            self.edgeWin = apps.edgeDetectionDialog(self)
            self.edgeWin.setFont(font)
            self.edgeWin.show()
        else:
            self.edgeWin.close()
            self.edgeWin = None

    def entropyFilter(self, checked):
        if checked:
            font = QtGui.QFont()
            font.setPointSize(10)
            self.entropyWin = apps.entropyFilterDialog(self)
            self.entropyWin.setFont(font)
            self.entropyWin.show()
        else:
            self.entropyWin.close()
            self.entropyWin = None

    def setData(self, data):
        if PosData.SizeZ > 1:
            pass

    def enableSmartTrack(self, checked):
        PosData = self.data[self.pos_i]
        # Disable tracking for already visited frames
        if PosData.allData_li[PosData.frame_i]['labels'] is not None:
            self.disableTrackingCheckBox.setChecked(True)
        else:
            self.disableTrackingCheckBox.setChecked(False)

        if checked:
            self.UserEnforced_DisabledTracking = False
            self.UserEnforced_Tracking = False
        else:
            if self.disableTrackingCheckBox.isChecked():
                self.UserEnforced_DisabledTracking = True
                self.UserEnforced_Tracking = False
            else:
                self.UserEnforced_DisabledTracking = False
                self.UserEnforced_Tracking = True

    def invertBw(self, checked):
        if self.slideshowWin is not None:
            self.slideshowWin.is_bw_inverted = checked
            self.slideshowWin.update_img()
        self.df_settings.at['is_bw_inverted', 'value'] = str(checked)
        self.df_settings.to_csv(self.settings_csv_path)
        if checked:
            self.ax2_BrushCirclePen = pg.mkPen((0,0,0), width=2)
            self.ax2_BrushCircleBrush = pg.mkBrush((0,0,0,50))
            self.ax1_oldIDcolor = [255-v for v in self.ax1_oldIDcolor]
            self.ax1_G1cellColor = [255-v for v in self.ax1_G1cellColor[:3]]
            self.ax1_G1cellColor.append(178)
            self.ax1_S_oldCellColor = [255-v for v in self.ax1_S_oldCellColor]
        else:
            self.ax2_BrushCirclePen = pg.mkPen(width=2)
            self.ax2_BrushCircleBrush = pg.mkBrush((255,255,255,50))
            self.ax1_oldIDcolor = [255-v for v in self.ax1_oldIDcolor]
            self.ax1_G1cellColor = [255-v for v in self.ax1_G1cellColor[:3]]
            self.ax1_G1cellColor.append(178)
            self.ax1_S_oldCellColor = [255-v for v in self.ax1_S_oldCellColor]

        self.updateALLimg(updateLabelItemColor=False)

    def saveNormAction(self, action):
        how = action.text()
        self.df_settings.at['how_normIntensities', 'value'] = how
        self.df_settings.to_csv(self.settings_csv_path)
        self.updateALLimg(only_ax1=True, updateFilters=True)

    def setLastUserNormAction(self):
        how = self.df_settings.at['how_normIntensities', 'value']
        for action in self.normalizeQActionGroup.actions():
            if action.text() == how:
                action.setChecked(True)
                break

    def changeFontSize(self, action):
        self.fontSize = f'{action.text()}pt'
        self.df_settings.at['fontSize', 'value'] = self.fontSize
        self.df_settings.to_csv(self.settings_csv_path)
        try:
            LIs = zip(self.ax1_LabelItemsIDs, self.ax2_LabelItemsIDs)
            for ax1_LI, ax2_LI in LIs:
                x1, y1 = ax1_LI.pos().x(), ax1_LI.pos().y()
                if x1>0:
                    w, h = ax1_LI.rect().right(), ax1_LI.rect().bottom()
                    xc, yc = x1+w/2, y1+h/2
                ax1_LI.setAttr('size', self.fontSize)
                ax1_LI.setText(ax1_LI.text)
                if x1>0:
                    w, h = ax1_LI.rect().right(), ax1_LI.rect().bottom()
                    ax1_LI.setPos(xc-w/2, yc-h/2)
                x2, y2 = ax2_LI.pos().x(), ax2_LI.pos().y()
                if x2>0:
                    w, h = ax2_LI.rect().right(), ax2_LI.rect().bottom()
                    xc, yc = x2+w/2, y2+h/2
                ax2_LI.setAttr('size', self.fontSize)
                ax2_LI.setText(ax2_LI.text)
                if x2>0:
                    w, h = ax2_LI.rect().right(), ax2_LI.rect().bottom()
                    ax2_LI.setPos(xc-w/2, yc-h/2)
        except Exception as e:
            print('')
            print('====================================')
            traceback.print_exc()
            print('====================================')
            print('')
            pass

    def enableZstackWidgets(self, enabled):
        if enabled:
            self.zSliceScrollBar.setDisabled(False)
            self.zProjComboBox.show()
            self.zSliceScrollBar.show()
            self.z_label.show()
            self.z_label.show()
        else:
            self.zSliceScrollBar.setDisabled(True)
            self.zProjComboBox.hide()
            self.zSliceScrollBar.hide()
            self.z_label.hide()
            self.z_label.hide()

    def reInitCca(self):
        if not self.isSnapshot:
            txt = (
                'If you decide to continue ALL cell cycle annotations from this '
                'frame to the end will be erased from current session '
                '(saved data is not touched of course)\n\n'
                'To annotate future frames again you will have to revisit them.\n\n'
                'Do you want to continue?'
            )
            msg = QtGui.QMessageBox()
            reinit = msg.warning(
               self, 'Cell not eligible', txt, msg.Yes | msg.Cancel
            )
            PosData = self.data[self.pos_i]
            if reinit == msg.Cancel:
                return
            # Go to previous frame without storing and then back to current
            if PosData.frame_i > 0:
                PosData.frame_i -= 1
                self.get_data()
                self.remove_future_cca_df(PosData.frame_i)
                self.next_frame()
            else:
                PosData.cca_df = self.getBaseCca_df()
                self.remove_future_cca_df(PosData.frame_i)
                self.updateALLimg()
        else:
            PosData.cca_df = self.getBaseCca_df()
            self.store_cca_df()
            self.updateALLimg()

    def repeatAutoCca(self):
        # Do not allow automatic bud assignment if there are future
        # frames that already contain anotations
        next_df = PosData.allData_li[PosData.frame_i+1]['acdc_df']
        if next_df is not None:
            if 'cell_cycle_stage' in next_df.columns:
                msg = QtGui.QMessageBox()
                warn_cca = msg.critical(
                    self, 'Future visited frames detected!',
                    'Automatic bud assignment CANNOT be performed becasue '
                    'there are future frames that already contain cell cycle '
                    'annotations. The behaviour in this case cannot be predicted.\n\n'
                    'We suggest assigning the bud manually OR use the '
                    '"Renitialize cell cycle annotations" button which properly '
                    'reinitialize future frames.',
                    msg.Ok
                )
                return

        correctedAssignIDs = (
                PosData.cca_df[PosData.cca_df['corrected_assignment']].index)
        NeverCorrectedAssignIDs = [ID for ID in PosData.new_IDs
                                   if ID not in correctedAssignIDs]

        # Store cca_df temporarily if attempt_auto_cca fails
        PosData.cca_df_beforeRepeat = PosData.cca_df.copy()

        if not all(NeverCorrectedAssignIDs):
            notEnoughG1Cells, proceed = self.attempt_auto_cca()
            if notEnoughG1Cells or not proceed:
                PosData.cca_df = PosData.cca_df_beforeRepeat
            else:
                self.updateALLimg()
            return

        msg = QtGui.QMessageBox()
        msg.setIcon(msg.Question)
        msg.setText(
            'Do you want to automatically assign buds to mother cells for '
            'ALL the new cells in this frame (excluding cells with unknown history) '
            'OR only the cells where you never clicked on?'
        )
        msg.setDetailedText(
            f'New cells that you never touched:\n\n{NeverCorrectedAssignIDs}')
        enforceAllButton = QPushButton('ALL new cells')
        b = QPushButton('Only cells that I never corrected assignment')
        msg.addButton(b, msg.YesRole)
        msg.addButton(enforceAllButton, msg.NoRole)
        msg.exec_()
        if msg.clickedButton() == enforceAllButton:
            notEnoughG1Cells, proceed = self.attempt_auto_cca(enforceAll=True)
        else:
            notEnoughG1Cells, proceed = self.attempt_auto_cca()
        if notEnoughG1Cells or not proceed:
            PosData.cca_df = PosData.cca_df_beforeRepeat
        else:
            self.updateALLimg()

    def manualEditCca(self):
        PosData = self.data[self.pos_i]
        editCcaWidget = apps.editCcaTableWidget(PosData.cca_df, parent=self)
        editCcaWidget.showAndSetWidth()
        editCcaWidget.exec_()
        if editCcaWidget.cancel:
            return
        PosData.cca_df = editCcaWidget.cca_df
        self.checkMultiBudMOth()
        self.updateALLimg()

    def drawIDsContComboBox_cb(self, idx):
        self.updateALLimg()
        how = self.drawIDsContComboBox.currentText()
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'

        t0 = time.time()
        # Clear contours if requested
        if how.find('contours') == -1 or nothing:
            allCont = zip(self.ax1_ContoursCurves,
                          self.ax2_ContoursCurves)
            for ax1ContCurve, ax2ContCurve in allCont:
                if ax1ContCurve.getData()[0] is not None:
                    ax1ContCurve.setData([], [])
                if ax2ContCurve.getData()[0] is not None:
                    ax2ContCurve.setData([], [])
            t1 = time.time()

            # print(f'Clearing contours = {t1-t0:.3f}')

        t0 = time.time()

        # Clear LabelItems IDs if requested (draw nothing or only contours)
        if onlyCont or nothing or onlyMothBudLines:
            for _IDlabel1 in self.ax1_LabelItemsIDs:
                _IDlabel1.setText('')
            t1 = time.time()

        # Clear mother-bud lines if Requested
        drawLines = only_ccaInfo or ccaInfo_and_cont or onlyMothBudLines
        if not drawLines:
            for BudMothLine in self.ax1_BudMothLines:
                if BudMothLine.getData()[0] is not None:
                    BudMothLine.setData([], [])

    def mousePressColorButton(self, event):
        PosData = self.data[self.pos_i]
        items = list(PosData.fluo_data_dict.keys())
        if len(items)>1:
            selectFluo = apps.QDialogListbox(
                'Select image',
                'Select which fluorescent image you want to update the color of\n',
                items, multiSelection=False, parent=self
            )
            selectFluo.exec_()
            keys = selectFluo.selectedItemsText
            key = keys[0]
            if selectFluo.cancel or not keys:
                return
            else:
                self._key = keys[0]
        else:
            self._key = items[0]
        self.overlayColorButton.selectColor()

    def setEnabledCcaToolbar(self, enabled=False):
        self.manuallyEditCcaAction.setDisabled(False)
        self.ccaToolBar.setVisible(enabled)
        for action in self.ccaToolBar.actions():
            button = self.ccaToolBar.widgetForAction(action)
            action.setVisible(enabled)
            button.setEnabled(enabled)

    def setEnabledEditToolbarButton(self, enabled=False):
        self.SegmActionYeaZ.setEnabled(enabled)
        self.SegmActionCellpose.setEnabled(enabled)
        self.SegmActionRW.setEnabled(enabled)
        self.autoSegmAction.setEnabled(enabled)
        self.editToolBar.setVisible(enabled)
        mode = self.modeComboBox.currentText()
        ccaON = mode == 'Cell cycle analysis'
        for action in self.editToolBar.actions():
            button = self.editToolBar.widgetForAction(action)
            # Keep binCellButton active in cca mode
            if button==self.binCellButton and not enabled and ccaON:
                action.setVisible(True)
                button.setEnabled(True)
            else:
                action.setVisible(enabled)
                button.setEnabled(enabled)
        if not enabled:
            self.setUncheckedAllButtons()

    def setEnabledFileToolbar(self, enabled):
        for action in self.fileToolBar.actions():
            button = self.fileToolBar.widgetForAction(action)
            if action == self.openAction:
                continue
            action.setEnabled(enabled)
            button.setEnabled(enabled)

    def setEnabledWidgetsToolbar(self, enabled):
        self.widgetsToolBar.setVisible(enabled)
        for action in self.widgetsToolBar.actions():
            widget = self.widgetsToolBar.widgetForAction(action)
            if widget == self.disableTrackingCheckBox:
                action.setVisible(enabled)
                widget.setEnabled(enabled)

    def enableSizeSpinbox(self, enabled):
        self.brushSizeLabelAction.setVisible(enabled)
        self.brushSizeAction.setVisible(enabled)

    def reload_cb(self):
        PosData = self.data[self.pos_i]
        self.app.setOverrideCursor(Qt.WaitCursor)
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)
        labData = np.load(PosData.segm_npz_path)
        # Keep compatibility with .npy and .npz files
        try:
            lab = labData['arr_0'][PosData.frame_i]
        except Exception as e:
            lab = labData[PosData.frame_i]
        PosData.segm_data[PosData.frame_i] = lab.copy()
        self.get_data()
        self.tracking()
        self.updateALLimg()
        self.app.restoreOverrideCursor()

    def clearComboBoxFocus(self, mode):
        # Remove focus from modeComboBox to avoid the key_up changes its value
        self.sender().clearFocus()

    def changeMode(self, idx):
        PosData = self.data[self.pos_i]
        mode = self.modeComboBox.itemText(idx)
        if mode == 'Segmentation and Tracking':
            self.modeToolBar.setVisible(True)
            self.setEnabledWidgetsToolbar(True)
            self.initSegmTrackMode()
            self.setEnabledEditToolbarButton(enabled=True)
            self.addExistingDelROIs()
            self.disableTrackingCheckBox.setChecked(False)
            self.setEnabledCcaToolbar(enabled=False)
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
            try:
                self.undoAction.triggered.disconnect()
                self.redoAction.triggered.disconnect()
            except Exception as e:
                pass
            self.undoAction.triggered.connect(self.undo)
            self.redoAction.triggered.connect(self.redo)
            for BudMothLine in self.ax1_BudMothLines:
                if BudMothLine.getData()[0] is not None:
                    BudMothLine.setData([], [])
            if PosData.cca_df is not None:
                self.store_cca_df()
        elif mode == 'Cell cycle analysis':
            proceed = self.init_cca()
            self.modeToolBar.setVisible(True)
            self.setEnabledWidgetsToolbar(False)
            if proceed:
                self.setEnabledEditToolbarButton(enabled=False)
                if self.isSnapshot:
                    self.editToolBar.setVisible(True)
                self.setEnabledCcaToolbar(enabled=True)
                self.removeAlldelROIsCurrentFrame()
                self.disableTrackingCheckBox.setChecked(True)
                try:
                    self.undoAction.triggered.disconnect()
                    self.redoAction.triggered.disconnect()
                except Exception as e:
                    pass
                self.undoAction.triggered.connect(self.UndoCca)
                self.drawIDsContComboBox.clear()
                self.drawIDsContComboBox.addItems(
                                        self.drawIDsContComboBoxCcaItems)
        elif mode == 'Viewer':
            self.modeToolBar.setVisible(True)
            self.setEnabledWidgetsToolbar(False)
            self.setEnabledEditToolbarButton(enabled=False)
            self.setEnabledCcaToolbar(enabled=False)
            self.removeAlldelROIsCurrentFrame()
            self.disableTrackingCheckBox.setChecked(True)
            self.undoAction.setDisabled(True)
            self.redoAction.setDisabled(True)
            currentMode = self.drawIDsContComboBox.currentText()
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxCcaItems)
            self.drawIDsContComboBox.setCurrentText(currentMode)
            self.framesScrollBar.setMaximum(PosData.segmSizeT)
            try:
                self.undoAction.triggered.disconnect()
                self.redoAction.triggered.disconnect()
            except Exception as e:
                pass
        elif mode == 'Snapshot':
            try:
                self.undoAction.triggered.disconnect()
                self.redoAction.triggered.disconnect()
            except Exception as e:
                pass
            self.undoAction.triggered.connect(self.undo)
            self.redoAction.triggered.connect(self.redo)
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(
                                    self.drawIDsContComboBoxCcaItems)
            self.setEnabledSnapshotMode()

    def setEnabledSnapshotMode(self):
        self.manuallyEditCcaAction.setDisabled(False)
        self.SegmActionYeaZ.setDisabled(False)
        self.SegmActionCellpose.setDisabled(False)
        self.SegmActionRW.setDisabled(False)
        self.autoSegmAction.setDisabled(False)
        self.ccaToolBar.setVisible(True)
        self.editToolBar.setVisible(True)
        self.modeToolBar.setVisible(False)
        for action in self.ccaToolBar.actions():
            button = self.ccaToolBar.widgetForAction(action)
            if button == self.assignBudMothButton:
                button.setDisabled(False)
                action.setVisible(True)
            elif action == self.reInitCcaAction:
                action.setVisible(True)
        for action in self.editToolBar.actions():
            button = self.editToolBar.widgetForAction(action)
            action.setVisible(True)
            button.setEnabled(True)
        self.disableTrackingCheckBox.setChecked(True)
        self.disableTrackingCheckBox.setDisabled(True)
        self.repeatTrackingAction.setVisible(False)
        button = self.editToolBar.widgetForAction(self.repeatTrackingAction)
        button.setDisabled(True)
        self.setEnabledWidgetsToolbar(False)

    def launchSlideshow(self):
        PosData = self.data[self.pos_i]
        if self.slideshowButton.isChecked():
            self.slideshowWin = apps.CellsSlideshow_GUI(
                               parent=self,
                               button_toUncheck=self.slideshowButton,
                               Left=self.slideshowWinLeft,
                               Top=self.slideshowWinTop)
            self.slideshowWin.update_img()
            self.slideshowWin.show()
        else:
            self.slideshowWin.close()
            self.slideshowWin = None

    def nearest_nonzero(self, a, y, x):
        r, c = np.nonzero(a)
        dist = ((r - y)**2 + (c - x)**2)
        min_idx = dist.argmin()
        return a[r[min_idx], c[min_idx]]

    def convexity_defects(self, img, eps_percent):
        img = img.astype(np.uint8)
        contours, _ = cv2.findContours(img,2,1)
        cnt = contours[0]
        cnt = cv2.approxPolyDP(cnt,eps_percent*cv2.arcLength(cnt,True),True) # see https://www.programcreek.com/python/example/89457/cv22.convexityDefects
        hull = cv2.convexHull(cnt,returnPoints = False) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        defects = cv2.convexityDefects(cnt,hull) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        return cnt, defects

    def auto_separate_bud_ID(self, ID, lab, rp, max_ID, max_i=1,
                             enforce=False, eps_percent=0.01):
        lab_ID_bool = lab == ID
        # First try separating by labelling
        lab_ID = lab_ID_bool.astype(int)
        rp_ID = skimage.measure.regionprops(lab_ID)
        setRp = self.separateByLabelling(lab_ID, rp_ID, maxID=lab.max())
        if setRp:
            success = True
            lab[lab_ID_bool] = lab_ID[lab_ID_bool]
            return lab, success

        cnt, defects = self.convexity_defects(lab_ID_bool, eps_percent)
        success = False
        if defects is not None:
            if len(defects) == 2:
                if not enforce:
                    # Yeaz watershed separation. To be tested
                    dist_watshed = segment(lab_ID_bool, None, merge=False)
                    num_obj_watshed = len(np.unique(dist_watshed))
                else:
                    num_obj_watshed = 0
                # Separate only if it was a separation also with watershed method
                if num_obj_watshed > 2 or enforce:
                    defects_points = [0]*len(defects)
                    for i, defect in enumerate(defects):
                        s,e,f,d = defect[0]
                        x,y = tuple(cnt[f][0])
                        defects_points[i] = (y,x)
                    (r0, c0), (r1, c1) = defects_points
                    rr, cc, _ = skimage.draw.line_aa(r0, c0, r1, c1)
                    sep_bud_img = np.copy(lab_ID_bool)
                    sep_bud_img[rr, cc] = False
                    sep_bud_label = skimage.measure.label(
                                               sep_bud_img, connectivity=2)
                    rp_sep = skimage.measure.regionprops(sep_bud_label)
                    IDs_sep = [obj.label for obj in rp_sep]
                    areas = [obj.area for obj in rp_sep]
                    curr_ID_bud = IDs_sep[areas.index(min(areas))]
                    curr_ID_moth = IDs_sep[areas.index(max(areas))]
                    orig_sblab = np.copy(sep_bud_label)
                    # sep_bud_label = np.zeros_like(sep_bud_label)
                    sep_bud_label[orig_sblab==curr_ID_moth] = ID
                    sep_bud_label[orig_sblab==curr_ID_bud] = max_ID+max_i
                    # sep_bud_label *= (max_ID+max_i)
                    temp_sep_bud_lab = sep_bud_label.copy()
                    for r, c in zip(rr, cc):
                        if lab_ID_bool[r, c]:
                            nearest_ID = self.nearest_nonzero(
                                                    sep_bud_label, r, c)
                            temp_sep_bud_lab[r,c] = nearest_ID
                    sep_bud_label = temp_sep_bud_lab
                    sep_bud_label_mask = sep_bud_label != 0
                    # plt.imshow_tk(sep_bud_label, dots_coords=np.asarray(defects_points))
                    lab[sep_bud_label_mask] = sep_bud_label[sep_bud_label_mask]
                    max_i += 1
                    success = True
        return lab, success

    def disconnectLeftClickButtons(self):
        self.brushButton.toggled.disconnect()
        self.curvToolButton.toggled.disconnect()
        self.rulerButton.toggled.disconnect()
        self.eraserButton.toggled.disconnect()

    def uncheckLeftClickButtons(self, sender):
        for button in self.LeftClickButtons:
            if button != sender:
                button.setChecked(False)

    def connectLeftClickButtons(self):
        self.brushButton.toggled.connect(self.Brush_cb)
        self.curvToolButton.toggled.connect(self.curvTool_cb)
        self.rulerButton.toggled.connect(self.ruler_cb)
        self.eraserButton.toggled.connect(self.Eraser_cb)

    def brushSize_cb(self, value):
        self.ax2_EraserCircle.setSize(value*2)
        self.ax1_BrushCircle.setSize(value*2)
        self.ax2_BrushCircle.setSize(value*2)
        self.ax1_EraserCircle.setSize(value*2)
        self.ax2_EraserX.setSize(value)
        self.ax1_EraserX.setSize(value)
        self.setDiskMask()

    def hideItemsHoverBrush(self, x, y):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.img2.image
        Y, X = _img.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        PosData = self.data[self.pos_i]
        size = self.brushSizeSpinbox.value()*2

        ID = PosData.lab[ydata, xdata]
        if ID == 0:
            prev_lab = PosData.allData_li[PosData.frame_i-1]['labels']
            if prev_lab is None:
                return
            ID = prev_lab[ydata, xdata]

        # Restore ID previously hovered
        if ID != self.ax1BrushHoverID and not self.isMouseDragImg1:
            if self.ax1BrushHoverID in PosData.IDs:
                obj_idx = PosData.IDs.index(self.ax1BrushHoverID)
                obj = PosData.rp[obj_idx]
                self.drawID_and_Contour(obj)
            elif self.ax1BrushHoverID in PosData.lost_IDs:
                prev_rp = PosData.allData_li[PosData.frame_i-1]['regionprops']
                obj_idx = [obj.label for obj in prev_rp].index(self.ax1BrushHoverID)
                obj = prev_rp[obj_idx]
                self.highlightLost_obj(obj)

        # Hide items hover ID
        if ID != 0:
            self.ax1_ContoursCurves[ID-1].setData([], [])
            self.ax1_LabelItemsIDs[ID-1].setText('')
            self.ax1BrushHoverID = ID
        else:
            prev_lab = PosData.allData_li[PosData.frame_i-1]['labels']
            rp = PosData.allData_li[PosData.frame_i-1]['regionprops']





    def updateBrushCursor(self, x, y):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.img2.image
        Y, X = _img.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        PosData = self.data[self.pos_i]
        size = self.brushSizeSpinbox.value()*2
        self.setHoverToolSymbolData(
            [x], [y], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            size=size
        )
        self.setHoverToolSymbolColor(
            xdata, ydata, self.ax2_BrushCirclePen,
            (self.ax2_BrushCircle, self.ax1_BrushCircle),
            self.brushButton, brush=self.ax2_BrushCircleBrush
        )


    def Brush_cb(self, checked):
        if checked:
            self.enableSizeSpinbox(True)
            self.setDiskMask()
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )
            self.updateBrushCursor(self.xHoverImg, self.yHoverImg)
            self.setBrushID()

            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            c = self.defaultToolBarButtonColor
            self.eraserButton.setStyleSheet(f'background-color: {c}')
            self.connectLeftClickButtons()
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
            self.enableSizeSpinbox(False)

    def setDiskMask(self):
        brushSize = self.brushSizeSpinbox.value()
        # diam = brushSize*2
        # center = (brushSize, brushSize)
        # diskShape = (diam+1, diam+1)
        # diskMask = np.zeros(diskShape, bool)
        # rr, cc = skimage.draw.disk(center, brushSize+1, shape=diskShape)
        # diskMask[rr, cc] = True
        self.diskMask = skimage.morphology.disk(brushSize, dtype=bool)

    def getDiskMask(self, xdata, ydata):
        Y, X = self.img2.image.shape[-2:]

        brushSize = self.brushSizeSpinbox.value()
        yBottom, xLeft = ydata-brushSize, xdata-brushSize
        yTop, xRight = ydata+brushSize+1, xdata+brushSize+1

        if xLeft>=0 and yBottom>=0 and xRight<=X and yTop<=Y:
            return yBottom, xLeft, yTop, xRight, self.diskMask

        elif xLeft<0 and yBottom>=0 and xRight<=X and yTop<=Y:
            diskMask = self.diskMask.copy()
            diskMask = diskMask[:, -xLeft:]
            return yBottom, 0, yTop, xRight, diskMask

        elif xLeft>=0 and yBottom<0 and xRight<=X and yTop<=Y:
            diskMask = self.diskMask.copy()
            diskMask = diskMask[-yBottom:]
            return 0, xLeft, yTop, xRight, diskMask

        elif xLeft>=0 and yBottom>=0 and xRight>X and yTop<=Y:
            diskMask = self.diskMask.copy()
            diskMask = diskMask[:, 0:X-xLeft]
            return yBottom, xLeft, yTop, X, diskMask

        elif xLeft>=0 and yBottom>=0 and xRight<=X and yTop>Y:
            diskMask = self.diskMask.copy()
            diskMask = diskMask[0:Y-yBottom]
            return yBottom, xLeft, Y, xRight, diskMask

    def setBrushID(self):
        # Make sure that the brushed ID is always a new one based on
        # already visited frames
        PosData = self.data[self.pos_i]
        newID = PosData.lab.max()
        for storedData in PosData.allData_li:
            lab = storedData['labels']
            if lab is not None:
                _max = lab.max()
                if _max > newID:
                    newID = _max
            else:
                break
        PosData.brushID = newID+1

    def equalizeHist(self):
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)
        img = self.getDisplayedCellsImg()
        self.updateALLimg()

    def curvTool_cb(self, checked):
        PosData = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            self.connectLeftClickButtons()
            self.hoverLinSpace = np.linspace(0, 1, 1000)
            self.curvPlotItem = pg.PlotDataItem(pen=self.newIDs_cpen)
            self.curvHoverPlotItem = pg.PlotDataItem(pen=self.oldIDs_cpen)
            self.curvAnchors = pg.ScatterPlotItem(
                symbol='o', size=9,
                brush=pg.mkBrush((255,0,0,50)),
                pen=pg.mkPen((255,0,0), width=2),
                hoverable=True, hoverPen=pg.mkPen((255,0,0), width=3),
                hoverBrush=pg.mkBrush((255,0,0))
            )
            self.ax1.addItem(self.curvAnchors)
            self.ax1.addItem(self.curvPlotItem)
            self.ax1.addItem(self.curvHoverPlotItem)
            self.splineHoverON = True
            PosData.curvPlotItems.append(self.curvPlotItem)
            PosData.curvAnchorsItems.append(self.curvAnchors)
            PosData.curvHoverItems.append(self.curvHoverPlotItem)
        else:
            self.splineHoverON = False
            self.isRightClickDragImg1 = False
            self.clearCurvItems()

    def updateEraserCursor(self, x, y):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.img2.image
        Y, X = _img.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        PosData = self.data[self.pos_i]
        size = self.brushSizeSpinbox.value()*2
        self.setHoverToolSymbolData(
            [x], [y], (self.ax1_EraserCircle, self.ax2_EraserCircle),
            size=size
        )
        self.setHoverToolSymbolData(
            [x], [y], (self.ax1_EraserX, self.ax2_EraserX),
            size=int(size/2)
        )

        isMouseDrag = (
            self.isMouseDragImg1 or self.isMouseDragImg2
        )

        if not isMouseDrag:
            self.setHoverToolSymbolColor(
                xdata, ydata, self.eraserCirclePen,
                (self.ax2_EraserCircle, self.ax1_EraserCircle),
                self.eraserButton, hoverRGB=None
            )

    def Eraser_cb(self, checked):
        if checked:
            self.enableSizeSpinbox(True)
            self.setDiskMask()
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
            self.updateEraserCursor(self.xHoverImg, self.yHoverImg)
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            c = self.defaultToolBarButtonColor
            self.brushButton.setStyleSheet(f'background-color: {c}')
            self.connectLeftClickButtons()
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )
            self.enableSizeSpinbox(False)

    def keyPressEvent(self, ev):
        try:
            PosData = self.data[self.pos_i]
        except AttributeError:
            return
        isBrushActive = (self.brushButton.isChecked()
                      or self.eraserButton.isChecked())
        if ev.key() == Qt.Key_Up and isBrushActive:
            self.brushSizeSpinbox.setValue(self.brushSizeSpinbox.value()+1)
        elif ev.key() == Qt.Key_Down and isBrushActive:
            self.brushSizeSpinbox.setValue(self.brushSizeSpinbox.value()-1)
        elif ev.key() == Qt.Key_Escape:
            self.setUncheckedAllButtons()
        elif ev.modifiers() == Qt.ControlModifier:
            if ev.key() == Qt.Key_P:
                print('========================')
                print('CURRENT Cell cycle analysis table:')
                print(PosData.cca_df)
                print('------------------------')
                print(f'STORED Cell cycle analysis table for frame {PosData.frame_i+1}:')
                df = PosData.allData_li[PosData.frame_i]['acdc_df']
                if 'cell_cycle_stage' in df.columns:
                    cca_df = df[self.cca_df_colnames]
                    print(cca_df)
                    cca_df = cca_df.merge(PosData.cca_df, how='outer',
                                          left_index=True, right_index=True,
                                          suffixes=('_STORED', '_CURRENT'))
                    cca_df = cca_df.reindex(sorted(cca_df.columns), axis=1)
                    num_cols = len(cca_df.columns)
                    for j in range(0,num_cols,2):
                        df_j_x = cca_df.iloc[:,j]
                        df_j_y = cca_df.iloc[:,j+1]
                        if any(df_j_x!=df_j_y):
                            print('------------------------')
                            print('DIFFERENCES:')
                            print(cca_df.iloc[:,j:j+2])
                else:
                    cca_df = None
                    print(cca_df)
                print('========================')
                if PosData.cca_df is None:
                    return
                df = PosData.cca_df.reset_index()
                if self.ccaTableWin is None:
                    self.ccaTableWin = apps.pdDataFrameWidget(df, parent=self)
                    self.ccaTableWin.show()
                    self.ccaTableWin.setGeometryWindow()
                else:
                    self.ccaTableWin.setFocus(True)
                    self.ccaTableWin.activateWindow()
                    self.ccaTableWin.updateTable(PosData.cca_df)
        elif ev.key() == Qt.Key_T:
            # self.startBlinking()
            pass
            # PosData1 = self.data[0]
            # PosData2 = self.data[1]
            # print(PosData1.fluo_data_dict.keys())
            # print(PosData2.fluo_data_dict.keys())
            # key1 = list(PosData1.fluo_data_dict.keys())[0]
            # key2 = list(PosData2.fluo_data_dict.keys())[0]
            # apps.imshow_tk(PosData1.fluo_data_dict[key1],
            #                additional_imgs=[PosData2.fluo_data_dict[key2]])
            # minTick = self.hist.gradient.getTick(0)
            # self.hist.gradient.setTickValue(minTick, 0.5)
        elif ev.key() == Qt.Key_H:
            self.zoomToCells(enforce=True)
        elif ev.key() == Qt.Key_L:
            self.storeUndoRedoStates(False)
            PosData = self.data[self.pos_i]
            PosData.lab = skimage.segmentation.relabel_sequential(PosData.lab)[0]
            self.update_rp()
            self.updateALLimg()
        elif ev.key() == Qt.Key_Space:
            how = self.drawIDsContComboBox.currentText()
            if how.find('nothing') == -1:
                self.drawIDsContComboBox.setCurrentText('Draw nothing')
                self.prev_how = how
            else:
                try:
                    self.drawIDsContComboBox.setCurrentText(self.prev_how)
                except Exception as e:
                    # traceback.print_exc()
                    pass
        elif ev.key() == Qt.Key_B or ev.key() == Qt.Key_X:
            mode = self.modeComboBox.currentText()
            if mode == 'Cell cycle analysis' or mode == 'Viewer':
                return
            if ev.key() == Qt.Key_B:
                self.Button = self.brushButton
            else:
                self.Button = self.eraserButton

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
                xdata, ydata = int(self.xHoverImg), int(self.yHoverImg)
                if ev.key() == Qt.Key_B:
                    self.setHoverToolSymbolColor(
                        xdata, ydata, self.ax2_BrushCirclePen,
                        (self.ax2_BrushCircle, self.ax1_BrushCircle),
                        self.brushButton, brush=self.ax2_BrushCircleBrush
                    )
                elif ev.key() == Qt.Key_X:
                    self.setHoverToolSymbolColor(
                        xdata, ydata, self.eraserCirclePen,
                        (self.ax2_EraserCircle, self.ax1_EraserCircle),
                        self.eraserButton
                    )

    def doubleKeyTimerCallBack(self):
        if self.isKeyDoublePress:
            self.doubleKeyTimeElapsed = False
            return
        self.doubleKeyTimeElapsed = True
        self.countKeyPress = 0
        isBrushChecked = self.Button.isChecked()
        if isBrushChecked and self.uncheck:
            self.Button.setChecked(False)
        c = self.defaultToolBarButtonColor
        self.Button.setStyleSheet(f'background-color: {c}')


    def keyReleaseEvent(self, ev):
        canRepeat = (
            ev.key() == Qt.Key_Left
            or ev.key() == Qt.Key_Right
            or ev.key() == Qt.Key_Up
            or ev.key() == Qt.Key_Down
        )
        if canRepeat:
            return
        if ev.isAutoRepeat():
            msg = QtGui.QMessageBox()
            msg.critical(
                self, 'Release the key!',
                f'Please, do not keep key {ev.text()} pressed! It confuses me.\n '
                'You do not need to keep it pressed.\n\n '
                'Thanks!',
                msg.Ok
            )

    def setUncheckedAllButtons(self):
        self.clickedOnBud = False
        try:
            self.BudMothTempLine.setData([], [])
        except Exception as e:
            pass
        for button in self.checkableButtons:
            button.setChecked(False)
        self.splineHoverON = False
        self.rulerHoverON = False
        self.isRightClickDragImg1 = False
        self.clearCurvItems(removeItems=False)

    def propagateChange(self, modID, modTxt, doNotShow, UndoFutFrames,
                        applyFutFrames, applyTrackingB=False):
        """
        This function determines whether there are already visited future frames
        that contains "modID". If so, it triggers a pop-up asking the user
        what to do (propagate change to future frames o not)
        """
        PosData = self.data[self.pos_i]
        # Do not check the future for the last frame
        if PosData.frame_i+1 == PosData.segmSizeT:
            # No future frames to propagate the change to
            return False, False, None, doNotShow

        areFutureIDs_affected = []
        # Get number of future frames already visited and checked if future
        # frames has an ID affected by the change
        for i in range(PosData.frame_i+1, PosData.segmSizeT):
            if PosData.allData_li[i]['labels'] is None:
                i -= 1
                break
            else:
                futureIDs = np.unique(PosData.allData_li[i]['labels'])
                if modID in futureIDs:
                    areFutureIDs_affected.append(True)

        if i == PosData.frame_i+1:
            # No future frames to propagate the change to
            return False, False, None, doNotShow

        if not areFutureIDs_affected:
            # There are future frames but they are not affected by the change
            return UndoFutFrames, False, None, doNotShow

        # Ask what to do unless the user has previously checked doNotShowAgain
        if doNotShow:
            endFrame_i = i
        else:
            ffa = apps.FutureFramesAction_QDialog(
                    PosData.frame_i+1, i, modTxt, applyTrackingB=applyTrackingB,
                    parent=self)
            ffa.exec_()
            decision = ffa.decision

            if decision is None:
                return None, None, None, doNotShow

            endFrame_i = ffa.endFrame_i
            doNotShowAgain = ffa.doNotShowCheckbox.isChecked()

            self.onlyTracking = False
            if decision == 'apply_and_reinit':
                UndoFutFrames = True
                applyFutFrames = False
            elif decision == 'apply_and_NOTreinit':
                UndoFutFrames = False
                applyFutFrames = False
            elif decision == 'apply_to_all':
                UndoFutFrames = False
                applyFutFrames = True
            elif decision == 'apply_to_range':
                UndoFutFrames = False
                applyFutFrames = True
            elif decision == 'only_tracking':
                UndoFutFrames = False
                applyFutFrames = True
                self.onlyTracking = True
        return UndoFutFrames, applyFutFrames, endFrame_i, doNotShowAgain

    def addCcaState(self, frame_i, cca_df, undoId):
        PosData = self.data[self.pos_i]
        PosData.UndoRedoCcaStates[frame_i].insert(0,
                                     {'id': undoId,
                                      'cca_df': cca_df.copy()})

    def addCurrentState(self):
        PosData = self.data[self.pos_i]
        PosData.UndoRedoStates[PosData.frame_i].insert(
                           0, {'image': self.img1.image.copy(),
                               'labels': PosData.lab.copy(),
                               'editID_info': PosData.editID_info.copy(),
                               'binnedIDs':PosData.binnedIDs.copy(),
                               'ripIDs':PosData.ripIDs.copy()}
        )

    def getCurrentState(self):
        PosData = self.data[self.pos_i]
        i = PosData.frame_i
        c = self.UndoCount
        self.ax1Image = PosData.UndoRedoStates[i][c]['image'].copy()
        PosData.lab = PosData.UndoRedoStates[i][c]['labels'].copy()
        PosData.editID_info = PosData.UndoRedoStates[i][c]['editID_info'].copy()
        PosData.binnedIDs = PosData.UndoRedoStates[i][c]['binnedIDs'].copy()
        PosData.ripIDs = PosData.UndoRedoStates[i][c]['ripIDs'].copy()

    def storeUndoRedoStates(self, UndoFutFrames):
        PosData = self.data[self.pos_i]
        if UndoFutFrames:
            # Since we modified current frame all future frames that were already
            # visited are not valid anymore. Undo changes there
            self.undo_changes_future_frames()

        # Restart count from the most recent state (index 0)
        # NOTE: index 0 is most recent state before doing last change
        self.UndoCount = 0
        self.undoAction.setEnabled(True)
        self.addCurrentState()
        # Keep only 5 Undo/Redo states
        if len(PosData.UndoRedoStates[PosData.frame_i]) > 5:
            PosData.UndoRedoStates[PosData.frame_i].pop(-1)

    def storeUndoRedoCca(self, frame_i, cca_df, undoId):
        if self.isSnapshot:
            # For snapshot mode we don't store anything because we have only
            # segmentation undo action active
            return
        """
        Store current cca_df along with a unique id to know which cca_df needs
        to be restored
        """

        PosData = self.data[self.pos_i]

        # Restart count from the most recent state (index 0)
        # NOTE: index 0 is most recent state before doing last change
        self.UndoCcaCount = 0
        self.undoAction.setEnabled(True)

        self.addCcaState(frame_i, cca_df, undoId)

        # Keep only 10 Undo/Redo states
        if len(PosData.UndoRedoCcaStates[frame_i]) > 10:
            PosData.UndoRedoCcaStates[frame_i].pop(-1)

    def UndoCca(self):
        PosData = self.data[self.pos_i]
        # Undo current ccaState
        storeState = False
        if self.UndoCount == 0:
            undoId = uuid.uuid4()
            self.addCcaState(PosData.frame_i, PosData.cca_df, undoId)
            storeState = True


        # Get previously stored state
        self.UndoCount += 1
        currentCcaStates = PosData.UndoRedoCcaStates[PosData.frame_i]
        prevCcaState = currentCcaStates[self.UndoCount]
        PosData.cca_df = prevCcaState['cca_df']
        self.store_cca_df()
        self.updateALLimg()

        # Check if we have undone all states
        if len(currentCcaStates) > self.UndoCount:
            # There are no states left to undo for current frame_i
            self.undoAction.setEnabled(False)

        # Undo all past and future frames that has a last status inserted
        # when modyfing current frame
        prevStateId = prevCcaState['id']
        for frame_i in range(0, PosData.segmSizeT):
            if storeState:
                cca_df_i = self.get_cca_df(frame_i=frame_i, return_df=True)
                if cca_df_i is None:
                    break
                # Store current state to enable redoing it
                self.addCcaState(frame_i, cca_df_i, undoId)

            CcaStates_i = PosData.UndoRedoCcaStates[frame_i]
            if len(CcaStates_i) <= self.UndoCount:
                # There are no states to undo for frame_i
                continue

            CcaState_i = CcaStates_i[self.UndoCount]
            id_i = CcaState_i['id']
            if id_i != prevStateId:
                # The id of the state in frame_i is different from current frame
                continue

            cca_df_i = CcaState_i['cca_df']
            self.store_cca_df(frame_i=frame_i, cca_df=cca_df_i)

    def undo(self):
        PosData = self.data[self.pos_i]
        if self.UndoCount == 0:
            # Store current state to enable redoing it
            self.addCurrentState()

        # Get previously stored state
        if self.UndoCount < len(PosData.UndoRedoStates[PosData.frame_i])-1:
            self.UndoCount += 1
            # Since we have undone then it is possible to redo
            self.redoAction.setEnabled(True)

            # Restore state
            self.getCurrentState()
            self.update_rp()
            self.checkIDs_LostNew()
            self.updateALLimg(image=self.ax1Image)
            self.store_data()

        if not self.UndoCount < len(PosData.UndoRedoStates[PosData.frame_i])-1:
            # We have undone all available states
            self.undoAction.setEnabled(False)

    def redo(self):
        PosData = self.data[self.pos_i]
        # Get previously stored state
        if self.UndoCount > 0:
            self.UndoCount -= 1
            # Since we have redone then it is possible to undo
            self.undoAction.setEnabled(True)

            # Restore state
            self.getCurrentState()
            self.update_rp()
            self.checkIDs_LostNew()
            self.updateALLimg(image=self.ax1Image)
            self.store_data()

        if not self.UndoCount > 0:
            # We have redone all available states
            self.redoAction.setEnabled(False)

    def disableTracking(self, isChecked):
        # Event called ONLY if the user click on Disable tracking
        # NOT called if setChecked is called. This allows to keep track
        # of the user choice. This way user con enforce tracking
        # NOTE: I know two booleans doing the same thing is overkill
        # but the code is more readable when we actually need them

        PosData = self.data[self.pos_i]

        # Turn off smart tracking
        self.enableSmartTrackAction.toggled.disconnect()
        self.enableSmartTrackAction.setChecked(False)
        self.enableSmartTrackAction.toggled.connect(self.enableSmartTrack)
        if self.disableTrackingCheckBox.isChecked():
            self.UserEnforced_DisabledTracking = True
            self.UserEnforced_Tracking = False
        else:
            warn_txt = (

            'You requested to explicitly ENABLE tracking. This will '
            'overwrite the default behaviour of not tracking already '
            'visited/checked frames.\n\n'
            'On all future frames that you will visit tracking '
            'will be automatically performed unless you explicitly '
            'disable tracking by clicking "Disable tracking" again.\n\n'
            'To reactive smart handling of enabling/disabling tracking '
            'go to Edit --> Smart handling of enabling/disabling tracking.\n\n'
            'If you need to repeat tracking ONLY on the current frame you '
            'can use the "Repeat tracking" button on the toolbar instead.\n\n'
            'Are you sure you want to proceed with ENABLING tracking from now on?'

            )
            msg = QtGui.QMessageBox()
            enforce_Tracking = msg.warning(
               self, 'Disable tracking?', warn_txt, msg.Yes | msg.No
            )
            if enforce_Tracking == msg.No:
                self.disableTrackingCheckBox.setChecked(True)
            else:
                self.repeatTracking()
                self.UserEnforced_DisabledTracking = False
                self.UserEnforced_Tracking = True

    def repeatTracking(self):
        PosData = self.data[self.pos_i]
        prev_lab = PosData.lab.copy()
        self.tracking(enforce=True, DoManualEdit=False)
        if PosData.editID_info:
            editIDinfo = [
                f'Replace ID {PosData.lab[y,x]} with {newID}'
                for y, x, newID in PosData.editID_info
            ]
            msg = QtGui.QMessageBox(self)
            msg.setWindowTitle('Repeat tracking mode')
            msg.setIcon(msg.Question)
            msg.setText("You requested to repeat tracking but there are "
                        "the following manually edited IDs:\n\n"
                        f"{editIDinfo}\n\n"
                        "Do you want to keep these edits or ignore them?")
            keepManualEditButton = QPushButton('Keep manually edited IDs')
            msg.addButton(keepManualEditButton, msg.YesRole)
            msg.addButton(QPushButton('Ignore'), msg.NoRole)
            msg.exec_()
            if msg.clickedButton() == keepManualEditButton:
                allIDs = [obj.label for obj in PosData.rp]
                self.ManuallyEditTracking(PosData.lab, allIDs)
                self.update_rp()
                self.checkIDs_LostNew()
                self.highlightLostNew()
                self.checkIDsMultiContour()
            else:
                PosData.editID_info = []
        self.updateALLimg()
        if np.any(PosData.lab != prev_lab):
            self.warnEditingWithCca_df('Repeat tracking')

    def autoSegm_cb(self, checked):
        if checked:
            self.askSegmParam = True
            self.segmModel = prompts.askWhichSegmModel(self)
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            self.updateALLimg()
            self.computeSegm()
            self.askSegmParam = False
        else:
            self.segmModel = None

    def randomWalkerSegm(self):
        # self.RWbkgrScatterItem = pg.ScatterPlotItem(
        #     symbol='o', size=2,
        #     brush=self.RWbkgrBrush,
        #     pen=self.RWbkgrPen
        # )
        # self.ax1.addItem(self.RWbkgrScatterItem)
        #
        # self.RWforegrScatterItem = pg.ScatterPlotItem(
        #     symbol='o', size=2,
        #     brush=self.RWforegrBrush,
        #     pen=self.RWforegrPen
        # )
        # self.ax1.addItem(self.RWforegrScatterItem)

        font = QtGui.QFont()
        font.setPointSize(10)

        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        self.segmModel = 'randomWalker'
        self.randomWalkerWin = apps.randomWalkerDialog(self)
        self.randomWalkerWin.setFont(font)
        self.randomWalkerWin.show()
        self.randomWalkerWin.setSize()

    def repeatSegmYeaZ(self):
        self.titleLabel.setText(
            'YeaZ neural network is thinking... '
            '(check progress in terminal/console)', color='w')
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)
        if self.sender() == self.SegmActionYeaZ:
            self.askSegmParam = True

        if self.askSegmParam:
            yeazParams = apps.YeaZ_ParamsDialog(parent=self)
            yeazParams.exec_()
            if yeazParams.cancel:
                print('Segmentation aborted.')
                return
            thresh_val = yeazParams.threshVal
            min_distance = yeazParams.minDist
            self.yeazThreshVal = thresh_val
            self.yeazMinDistance = min_distance
        else:
            thresh_val = self.yeazThreshVal
            min_distance = self.yeazMinDistance

        t0 = time.time()
        PosData = self.data[self.pos_i]
        self.which_model = 'YeaZ'
        if self.is_first_call_YeaZ:
            print('Importing YeaZ model...')
            from YeaZ.unet import neural_network as nn
            from YeaZ.unet import segment
            self.nn = nn
            self.segment = segment
            self.path_weights = nn.determine_path_weights()
            download_model('YeaZ')

        img = self.getDisplayedCellsImg()

        if self.gaussWin is None:
            img = skimage.filters.gaussian(img, sigma=1)
        img = skimage.exposure.equalize_adapthist(skimage.img_as_float(img))
        pred = self.nn.prediction(img, is_pc=True,
                                  path_weights=self.path_weights)
        thresh = self.nn.threshold(pred, th=thresh_val)
        lab = self.segment.segment(thresh, pred,
                                   min_distance=min_distance).astype(int)
        t1 = time.time()
        self.is_first_call_YeaZ = False
        if PosData.segmInfo_df is not None and PosData.SizeZ>1:
            PosData.segmInfo_df.at[PosData.frame_i, 'resegmented_in_gui'] = True
        PosData.lab = lab.copy()
        self.update_rp()
        self.tracking(enforce=True)
        self.updateALLimg()
        self.warnEditingWithCca_df('Repeat segmentation with YeaZ')

        txt = f'Done. Segmentation computed in {t1-t0:.3f} s'
        print('-----------------')
        print(txt)
        print('=================')
        self.titleLabel.setText(txt, color='g')
        self.checkIfAutoSegm()

    def repeatSegmCellpose(self, checked=False):
        self.titleLabel.setText(
            'Cellpose neural network is thinking... '
            '(check progress in terminal/console)', color='w')
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)
        if self.sender() == self.SegmActionCellpose:
            self.askSegmParam = True

        if self.askSegmParam:
            cellposeParams = apps.cellpose_ParamsDialog(parent=self)
            cellposeParams.exec_()
            if cellposeParams.cancel:
                print('Segmentation aborted.')
                return
            diameter = cellposeParams.diameter
            if diameter==0:
                diameter=None
            flow_threshold = cellposeParams.flow_threshold
            cellprob_threshold = cellposeParams.cellprob_threshold
            self.cellposeDiameter = diameter
            self.cellposeFlowThreshold = flow_threshold
            self.cellposeProbThreshold = cellprob_threshold
        else:
            diameter = self.cellposeDiameter
            flow_threshold = self.cellposeFlowThreshold
            cellprob_threshold = self.cellposeProbThreshold

        t0 = time.time()
        PosData = self.data[self.pos_i]
        self.which_model = 'Cellpose'
        if self.is_first_call_cellpose:
            print('Initializing cellpose models...')
            from acdc_cellpose import models
            download_model('cellpose')
            device, gpu = models.assign_device(True, False)
            self.cp_model = models.Cellpose(gpu=gpu, device=device,
                                            model_type='cyto', torch=True)

        img = self.getDisplayedCellsImg()

        if self.gaussWin is None:
            img = skimage.filters.gaussian(img, sigma=1)
        img = img/img.max()
        img = skimage.exposure.equalize_adapthist(img)
        lab, flows, _, _ = self.cp_model.eval(
                                img,
                                channels=[0,0],
                                diameter=diameter,
                                flow_threshold=flow_threshold,
                                cellprob_threshold=cellprob_threshold
        )
        t1 = time.time()
        self.is_first_call_cellpose = False
        if PosData.segmInfo_df is not None and PosData.SizeZ>1:
            PosData.segmInfo_df.at[PosData.frame_i, 'resegmented_in_gui'] = True
        PosData.lab = lab.copy()
        self.update_rp()
        self.tracking(enforce=True)
        self.updateALLimg()
        self.warnEditingWithCca_df('Repeat segmentation with YeaZ')


        txt = f'Done. Segmentation computed in {t1-t0:.3f} s'
        print('-----------------')
        print(txt)
        print('=================')
        self.titleLabel.setText(txt, color='g')
        self.checkIfAutoSegm()

    def getDisplayedCellsImg(self):
        if self.overlayButton.isChecked():
            img = self.ol_cells_img
        else:
            img = self.img1.image
        return img

    def next_cb(self):
        if self.isSnapshot:
            self.next_pos()
        else:
            self.next_frame()
        if self.curvToolButton.isChecked():
            self.curvTool_cb(True)

    def prev_cb(self):
        if self.isSnapshot:
            self.prev_pos()
        else:
            self.prev_frame()
        if self.curvToolButton.isChecked():
            self.curvTool_cb(True)

    def zoomToCells(self, enforce=False):
        if not self.enableAutoZoomToCellsAction.isChecked() and not enforce:
            return

        PosData = self.data[self.pos_i]
        lab_mask = (PosData.lab>0).astype(np.uint8)
        rp = skimage.measure.regionprops(lab_mask)
        if not rp:
            Y, X = PosData.lab.shape
            xRange = -0.5, X+0.5
            yRange = -0.5, Y+0.5
        else:
            obj = rp[0]
            min_row, min_col, max_row, max_col = obj.bbox
            xRange = min_col-10, max_col+10
            yRange = max_row+10, min_row-10

        self.ax1.setRange(xRange=xRange, yRange=yRange)

    def updateScrollbars(self):
        self.updateItemsMousePos()
        self.updateFramePosLabel()
        PosData = self.data[self.pos_i]
        if PosData.SizeT > 1:
            pos = self.pos_i+1 if self.isSnapshot else PosData.frame_i+1
            self.framesScrollBar.setSliderPosition(pos)
        if PosData.SizeZ > 1:
            z = PosData.segmInfo_df.at[PosData.frame_i, 'z_slice_used_gui']
            self.zSliceScrollBar.setSliderPosition(z)
            how = PosData.segmInfo_df.at[PosData.frame_i, 'which_z_proj_gui']
            self.zProjComboBox.setCurrentText(how)
            self.zSliceScrollBar.setMaximum(PosData.SizeZ-1)

    def updateItemsMousePos(self):
        if self.brushButton.isChecked():
            self.updateBrushCursor(self.xHoverImg, self.yHoverImg)

        if self.eraserButton.isChecked():
            self.updateEraserCursor(self.xHoverImg, self.yHoverImg)

    def next_pos(self):
        self.store_data(debug=False)
        if self.pos_i < self.num_pos-1:
            self.pos_i += 1
        else:
            print('You reached last position.')
            self.pos_i = 0
        self.removeAlldelROIsCurrentFrame()
        proceed_cca, never_visited = self.get_data()
        self.updateALLimg(updateFilters=True, updateLabelItemColor=False)
        self.zoomToCells()
        self.updateScrollbars()
        self.computeSegm()

    def prev_pos(self):
        self.store_data(debug=False)
        if self.pos_i > 0:
            self.pos_i -= 1
        else:
            print('You reached first position.')
            self.pos_i = self.num_pos-1
        self.removeAlldelROIsCurrentFrame()
        proceed_cca, never_visited = self.get_data()
        self.updateALLimg(updateSharp=True, updateBlur=True, updateEntropy=True)
        self.zoomToCells()
        self.updateScrollbars()

    def next_frame(self):
        mode = str(self.modeComboBox.currentText())
        isSegmMode =  mode == 'Segmentation and Tracking'
        PosData = self.data[self.pos_i]
        if PosData.frame_i < PosData.segmSizeT-1:
            if 'lost' in self.titleLabel.text and isSegmMode:
                msg = QtGui.QMessageBox()
                warn_msg = (
                    'Current frame (compared to previous frame) '
                    'has lost the following cells:\n\n'
                    f'{PosData.lost_IDs}\n\n'
                    'Are you sure you want to continue?'
                )
                proceed_with_lost = msg.warning(
                   self, 'Lost cells!', warn_msg, msg.Yes | msg.No
                )
                if proceed_with_lost == msg.No:
                    return
            if 'multiple' in self.titleLabel.text and mode != 'Viewer':
                msg = QtGui.QMessageBox()
                warn_msg = (
                    'Current frame contains cells with MULTIPLE contours '
                    '(see title message above the images)!\n\n'
                    'This is potentially an issue indicating that two distant cells '
                    'have been merged.\n\n'
                    'Are you sure you want to continue?'
                )
                proceed_with_multi = msg.warning(
                   self, 'Multiple contours detected!', warn_msg, msg.Yes | msg.No
                )
                if proceed_with_multi == msg.No:
                    return

            if PosData.frame_i <= 0 and mode == 'Cell cycle analysis':
                IDs = [obj.label for obj in PosData.rp]
                editCcaWidget = apps.editCcaTableWidget(PosData.cca_df,
                                                        parent=self)
                editCcaWidget.showAndSetWidth()
                editCcaWidget.exec_()
                if editCcaWidget.cancel:
                    return
                if PosData.cca_df is not None:
                    if not PosData.cca_df.equals(editCcaWidget.cca_df):
                        self.remove_future_cca_df(0)
                PosData.cca_df = editCcaWidget.cca_df
                self.store_cca_df()

            # Store data for current frame
            self.store_data(debug=False)
            # Go to next frame
            PosData.frame_i += 1
            self.removeAlldelROIsCurrentFrame()
            proceed_cca, never_visited = self.get_data()
            if not proceed_cca:
                PosData.frame_i -= 1
                self.get_data()
                return
            self.tracking(storeUndo=True)
            notEnoughG1Cells, proceed = self.attempt_auto_cca()
            if notEnoughG1Cells or not proceed:
                PosData.frame_i -= 1
                self.get_data()
                return
            self.updateALLimg(never_visited=never_visited,
                              updateFilters=True, updateLabelItemColor=False)
            self.updateScrollbars()
            self.computeSegm()
            self.setFramesScrollbarMaximum()
            self.zoomToCells()
        else:
            # Store data for current frame
            self.store_data()
            msg = 'You reached the last segmented frame!'
            print(msg)
            self.titleLabel.setText(msg, color='w')

    def setFramesScrollbarMaximum(self):
        PosData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Segmentation and Tracking':
            if PosData.last_tracked_i is not None:
                if PosData.frame_i > PosData.last_tracked_i:
                    self.framesScrollBar.setMaximum(PosData.frame_i+1)
        elif mode == 'Cell cycle analysis':
            if PosData.frame_i > self.last_cca_frame_i:
                self.framesScrollBar.setMaximum(PosData.frame_i+1)

    def prev_frame(self):
        PosData = self.data[self.pos_i]
        if PosData.frame_i > 0:
            self.store_data()
            self.removeAlldelROIsCurrentFrame()
            PosData.frame_i -= 1
            _, never_visited = self.get_data()
            self.tracking()
            self.updateALLimg(never_visited=never_visited,
                              updateSharp=True, updateBlur=True,
                              updateEntropy=True)
            self.updateScrollbars()
            self.zoomToCells()
        else:
            msg = 'You reached the first frame!'
            print(msg)
            self.titleLabel.setText(msg, color='w')

    def checkDataIntegrity(self, PosData, numPos):
        skipPos = False
        abort = False
        if numPos > 1:
            if PosData.SizeT > 1:
                err_msg = (f'{PosData.pos_foldername} contains frames over time. '
                           f'Skipping it.')
                print(err_msg)
                self.titleLabel.setText(err_msg, color='r')
                skipPos = True
        else:
            if not PosData.segmFound and PosData.SizeT > 1:
                err_msg = ('Segmentation mask file ("..._segm.npz") not found. '
                           'You should to run segmentation script "segm.py" first.')
                print(err_msg)
                self.titleLabel.setText(err_msg, color='r')
                skipPos = False
                msg = QtGui.QMessageBox()
                warn_msg = (
                    f'The folder {PosData.pos_foldername} does not contain a '
                    'pre-computed segmentation mask.\n\n'
                    'You can continue with a blank mask or cancel and '
                    'pre-compute the mask with "segm.py" script.\n\n'
                    'Do you want to continue?'
                )
                continueWithBlankSegm = msg.warning(
                   self, 'Segmentation file not found', warn_msg,
                   msg.Yes | msg.Cancel
                )
                if continueWithBlankSegm == msg.Cancel:
                    abort = True
        return skipPos, abort

    def init_data(self, user_ch_file_paths, user_ch_name):
        data = []
        numPos = len(user_ch_file_paths)
        for f, file_path in enumerate(user_ch_file_paths):
            try:
                PosData = load.loadData(file_path, user_ch_name, QParent=self)
                PosData.getBasenameAndChNames(prompts.select_channel_name)
                PosData.buildPaths()
                PosData.loadImgData()
                PosData.loadOtherFiles(
                                   load_segm_data=True,
                                   load_acdc_df=True,
                                   load_shifts=False,
                                   loadSegmInfo=True,
                                   load_delROIsInfo=True,
                                   loadDataPrepBkgrVals=True,
                                   load_last_tracked_i=True,
                                   load_metadata=True
                )
                if f==0:
                    proceed = PosData.askInputMetadata(
                                                ask_SizeT=self.num_pos==1,
                                                ask_TimeIncrement=True,
                                                ask_PhysicalSizes=True,
                                                save=True)
                    if not proceed:
                        return False
                    self.SizeT = PosData.SizeT
                    self.SizeZ = PosData.SizeZ
                    self.TimeIncrement = PosData.TimeIncrement
                    self.PhysicalSizeZ = PosData.PhysicalSizeZ
                    self.PhysicalSizeY = PosData.PhysicalSizeY
                    self.PhysicalSizeX = PosData.PhysicalSizeX

                else:
                    PosData.SizeT = self.SizeT
                    if self.SizeZ > 1:
                        SizeZ = PosData.img_data.shape[-3]
                        PosData.SizeZ = SizeZ
                    else:
                        PosData.SizeZ = 1
                    PosData.TimeIncrement = self.TimeIncrement
                    PosData.PhysicalSizeZ = self.PhysicalSizeZ
                    PosData.PhysicalSizeY = self.PhysicalSizeY
                    PosData.PhysicalSizeX = self.PhysicalSizeX
                    PosData.saveMetadata()
                SizeY, SizeX = PosData.img_data.shape[-2:]
                PosData.setBlankSegmData(PosData.SizeT, PosData.SizeZ,
                                         SizeY, SizeX)
                skipPos, abort = self.checkDataIntegrity(PosData, numPos)
            except AttributeError:
                print('')
                print('====================================')
                traceback.print_exc()
                print('====================================')
                print('')
                skipPos = False
                abort = True

            if skipPos:
                continue
            elif abort:
                return False

            if PosData.SizeT == 1:
                self.isSnapshot = True
            else:
                self.isSnapshot = False

            # Allow single 2D/3D image
            if PosData.SizeT < 2:
                PosData.img_data = np.array([PosData.img_data])
                PosData.segm_data = np.array([PosData.segm_data])
            img_shape = PosData.img_data.shape
            PosData.segmSizeT = len(PosData.segm_data)
            SizeT = PosData.SizeT
            SizeZ = PosData.SizeZ
            if f==0:
                print(f'Data shape = {img_shape}')
                print(f'Number of frames = {SizeT}')
                print(f'Number of z-slices per frame = {SizeZ}')
            data.append(PosData)

        if not data:
            errTitle = 'All loaded positions contains frames over time!'
            err_msg = (
                f'{errTitle}.\n\n'
                'To load data that contains frames over time you have to select '
                'only ONE position.'
            )
            msg = QtGui.QMessageBox()
            msg.critical(
                self, errTitle, err_msg, msg.Ok
            )
            self.titleLabel.setText(errTitle, color='r')
            return False

        self.data = data

        self.gui_createGraphicsItems()
        self.init_segmInfo_df()
        self.initPosAttr(max_ID=PosData.segm_data.max())
        self.initFluoData()
        self.framesScrollBar.setSliderPosition(PosData.frame_i+1)
        if PosData.SizeZ > 1:
            how = PosData.segmInfo_df.at[PosData.frame_i, 'which_z_proj_gui']
            self.zProjComboBox.setCurrentText(how)

        return True

    def setFramesSnapshotMode(self):
        if self.isSnapshot:
            self.disableTrackingCheckBox.setDisabled(True)
            self.disableTrackingCheckBox.setChecked(True)
            self.repeatTrackingAction.setDisabled(True)
            print('Setting gui mode to "No frames"...')
            self.modeComboBox.clear()
            self.modeComboBox.addItems(['Snapshot'])
            self.modeComboBox.setDisabled(True)
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxCcaItems)
            self.drawIDsContComboBox.setCurrentIndex(1)
            self.modeToolBar.setVisible(False)
            self.modeComboBox.setCurrentText('Snapshot')
        else:
            self.disableTrackingCheckBox.setDisabled(False)
            self.repeatTrackingAction.setDisabled(False)
            self.modeComboBox.setDisabled(False)
            try:
                self.modeComboBox.activated.disconnect()
                self.modeComboBox.currentIndexChanged.disconnect()
                self.drawIDsContComboBox.currentIndexChanged.disconnect()
            except Exception as e:
                pass
                # traceback.print_exc()
            self.modeComboBox.clear()
            self.modeComboBox.addItems(['Segmentation and Tracking',
                                        'Cell cycle analysis',
                                        'Viewer'])
            self.drawIDsContComboBox.clear()
            self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
            self.modeComboBox.currentIndexChanged.connect(self.changeMode)
            self.modeComboBox.activated.connect(self.clearComboBoxFocus)
            self.drawIDsContComboBox.currentIndexChanged.connect(
                                                    self.drawIDsContComboBox_cb)
            self.modeComboBox.setCurrentText('Viewer')

    def checkIfAutoSegm(self):
        """
        If there are any frame or position with empty segmentation mask
        ask whether automatic segmentation should be turned ON
        """
        if self.autoSegmAction.isChecked():
            return
        ask = False
        for PosData in self.data:
            if PosData.SizeT > 1:
                for lab in PosData.segm_data:
                    if not np.any(lab):
                        ask = True
                        txt = 'frames'
                        break
            else:
                if not np.any(PosData.segm_data):
                    ask = True
                    txt = 'positions'
                    break

        if not ask:
            return

        questionTxt = (
            f'Some or all loaded {txt} contain empty segmentation masks.\n\n'
            'Do you want to activate automatic segmentation* when visiting '
            f'these {txt}?\n\n'
            '* Automatic segmentation can always be turned ON/OFF from the menu\n'
            '  "Edit --> Segmentation --> Enable automatic segmentation"\n\n'
            f'** Remember that you can automatically segment all {txt} using the\n'
            '    "segm.py" script.'
        )
        msg = QtGui.QMessageBox()
        doSegmAnswer = msg.question(
            self, 'Automatic segmentation?', questionTxt, msg.Yes | msg.No
        )
        if doSegmAnswer == msg.Yes:
            self.autoSegmAction.setChecked(True)
        else:
            self.autoSegmAction.setChecked(False)

    def init_segmInfo_df(self):
        self.t_label.show()
        self.framesScrollBar.show()
        self.framesScrollBar.setDisabled(False)
        self.zSliceScrollBar.setMaximum(self.data[0].SizeZ-1)
        for PosData in self.data:
            if PosData.segmInfo_df is None and PosData.SizeZ > 1:
                mid_slice = int(PosData.SizeZ/2)
                PosData.segmInfo_df = pd.DataFrame(
                    {'frame_i': range(PosData.SizeT),
                     'z_slice_used_dataPrep': [mid_slice]*PosData.SizeT,
                     'which_z_proj': ['single z-slice']*PosData.SizeT}
                ).set_index('frame_i')
            if PosData.SizeZ > 1:
                if 'z_slice_used_gui' not in PosData.segmInfo_df.columns:
                    PosData.segmInfo_df['z_slice_used_gui'] = (
                                    PosData.segmInfo_df['z_slice_used_dataPrep']
                                    )
                if 'which_z_proj_gui' not in PosData.segmInfo_df.columns:
                    PosData.segmInfo_df['which_z_proj_gui'] = (
                                    PosData.segmInfo_df['which_z_proj']
                                    )
                PosData.segmInfo_df['resegmented_in_gui'] = False
                self.enableZstackWidgets(True)
                try:
                    self.zSliceScrollBar.sliderMoved.disconnect()
                    self.zProjComboBox.currentTextChanged.disconnect()
                    self.zProjComboBox.activated.disconnect()
                except Exception as e:
                    pass
                self.zSliceScrollBar.sliderMoved.connect(self.update_z_slice)
                self.zProjComboBox.currentTextChanged.connect(self.updateZproj)
                self.zProjComboBox.activated.connect(self.clearComboBoxFocus)
            if PosData.SizeT == 1:
                self.t_label.setText('Position n. ')
                self.framesScrollBar.setMinimum(1)
                self.framesScrollBar.setMaximum(len(self.data))
                try:
                    self.framesScrollBar.sliderMoved.disconnect()
                    self.framesScrollBar.sliderReleased.disconnect()
                except TypeError:
                    pass
                self.framesScrollBar.sliderMoved.connect(
                                                  self.PosScrollBarMoved)
                self.framesScrollBar.sliderReleased.connect(
                                                  self.PosScrollBarReleased)
            else:
                self.framesScrollBar.setMinimum(1)
                if PosData.last_tracked_i is not None:
                    self.framesScrollBar.setMaximum(PosData.last_tracked_i+1)
                try:
                    self.framesScrollBar.sliderMoved.disconnect()
                    self.framesScrollBar.sliderReleased.disconnect()
                except Exception as e:
                    pass
                self.t_label.setText('frame n.  ')
                self.framesScrollBar.sliderMoved.connect(
                                                  self.framesScrollBarMoved)
                self.framesScrollBar.sliderReleased.connect(
                                                  self.framesScrollBarReleased)

    def update_z_slice(self, z):
        PosData = self.data[self.pos_i]
        PosData.segmInfo_df.at[PosData.frame_i, 'z_slice_used_gui'] = z
        self.updateALLimg(only_ax1=True)

    def updateZproj(self, how):
        for p, PosData in enumerate(self.data[self.pos_i:]):
            PosData.segmInfo_df.at[PosData.frame_i, 'which_z_proj_gui'] = how
        PosData = self.data[self.pos_i]
        if how == 'single z-slice':
            self.zSliceScrollBar.setDisabled(False)
            self.z_label.setStyleSheet('color: black')
            self.update_z_slice(self.zSliceScrollBar.sliderPosition())
        else:
            self.zSliceScrollBar.setDisabled(True)
            self.z_label.setStyleSheet('color: gray')
            self.updateALLimg(only_ax1=True)

    def clearAllItems(self):
        allItems = zip(self.ax1_ContoursCurves,
                       self.ax2_ContoursCurves,
                       self.ax1_LabelItemsIDs,
                       self.ax2_LabelItemsIDs,
                       self.ax1_BudMothLines)
        for idx, items_ID in enumerate(allItems):
            (ax1ContCurve, ax2ContCurve,
            _IDlabel1, _IDlabel2,
            BudMothLine) = items_ID
            if ax1ContCurve.getData()[0] is not None:
                ax1ContCurve.setData([], [])
            if ax2ContCurve.getData()[0] is not None:
                ax2ContCurve.setData([], [])
            if BudMothLine.getData()[0] is not None:
                BudMothLine.setData([], [])
            _IDlabel1.setText('')
            _IDlabel2.setText('')

    def clearCurvItems(self, removeItems=True):
        try:
            PosData = self.data[self.pos_i]
            curvItems = zip(PosData.curvPlotItems,
                            PosData.curvAnchorsItems,
                            PosData.curvHoverItems)
            for plotItem, curvAnchors, hoverItem in curvItems:
                plotItem.setData([], [])
                curvAnchors.setData([], [])
                hoverItem.setData([], [])
                if removeItems:
                    self.ax1.removeItem(plotItem)
                    self.ax1.removeItem(curvAnchors)
                    self.ax1.removeItem(hoverItem)

            if removeItems:
                PosData.curvPlotItems = []
                PosData.curvAnchorsItems = []
                PosData.curvHoverItems = []
        except AttributeError:
            # traceback.print_exc()
            pass

    def splineToObj(self, xxA=None, yyA=None):
        PosData = self.data[self.pos_i]
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        xxS, yyS = self.curvPlotItem.getData()

        self.setBrushID()
        newIDMask = np.zeros(PosData.lab.shape, bool)
        rr, cc = skimage.draw.polygon(yyS, xxS)
        newIDMask[rr, cc] = True
        newIDMask[PosData.lab!=0] = False
        PosData.lab[newIDMask] = PosData.brushID

    def addFluoChNameContextMenuAction(self, ch_name):
        PosData = self.data[self.pos_i]
        allTexts = [action.text() for action in self.chNamesQActionGroup.actions()]
        if ch_name not in allTexts:
            action = QAction(self)
            action.setText(ch_name)
            action.setCheckable(True)
            self.chNamesQActionGroup.addAction(action)
            action.setChecked(True)
            PosData.fluoDataChNameActions.append(action)

    def computeSegm(self):
        PosData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer' or mode == 'Cell cycle analysis':
            return

        if np.any(PosData.lab):
            # Do not compute segm if there is already a mask
            return

        if not self.autoSegmAction.isChecked():
            # Compute segmentations that have an open window
            if self.segmModel == 'randomWalker':
                self.randomWalkerWin.getImage()
                self.randomWalkerWin.computeMarkers()
                self.randomWalkerWin.computeSegm()
                self.update_rp()
                self.tracking(enforce=True)
                self.updateALLimg()
                self.store_data()
                self.warnEditingWithCca_df('Random Walker segmentation')
            else:
                return

        if self.segmModel == 'yeaz':
            self.repeatSegmYeaZ()
        elif self.segmModel == 'cellpose':
            self.repeatSegmCellpose()

        self.update_rp()

    def initGlobalAttr(self):
        self.setOverlayColors()
        self.cmap = myutils.getFromMatplotlib('viridis')

        self.splineHoverON = False
        self.rulerHoverON = False
        self.autoContourHoverON = False
        self.framesScrollBarStartedMoving = True

        self.clickedOnBud = False
        self.gaussWin = None
        self.edgeWin = None
        self.entropyWin = None

        self.UserEnforced_DisabledTracking = False
        self.UserEnforced_Tracking = False

        self.ax1BrushHoverID = 0

        # Plots items
        self.is_first_call_YeaZ = True
        self.is_first_call_cellpose = True
        self.data_loaded = True
        self.isMouseDragImg2 = False
        self.isMouseDragImg1 = False
        self.isRightClickDragImg1 = False

        self.cca_df_colnames = ['cell_cycle_stage',
                                'generation_num',
                                'relative_ID',
                                'relationship',
                                'emerg_frame_i',
                                'division_frame_i',
                                'is_history_known',
                                'corrected_assignment']
        self.cca_df_int_cols = ['generation_num',
                                'relative_ID',
                                'emerg_frame_i',
                                'division_frame_i']

    def initPosAttr(self, max_ID=10):
        for p, PosData in enumerate(self.data):
            self.pos_i = p
            PosData.curvPlotItems = []
            PosData.curvAnchorsItems = []
            PosData.curvHoverItems = []
            PosData.fluoDataChNameActions = []
            PosData.manualContrastKey = PosData.filename
            PosData.isNewID = False

            # Decision on what to do with changes to future frames attr
            PosData.doNotShowAgain_EditID = False
            PosData.UndoFutFrames_EditID = False
            PosData.applyFutFrames_EditID = False

            PosData.doNotShowAgain_RipID = False
            PosData.UndoFutFrames_RipID = False
            PosData.applyFutFrames_RipID = False

            PosData.doNotShowAgain_DelID = False
            PosData.UndoFutFrames_DelID = False
            PosData.applyFutFrames_DelID = False

            PosData.doNotShowAgain_BinID = False
            PosData.UndoFutFrames_BinID = False
            PosData.applyFutFrames_BinID = False

            PosData.disableAutoActivateViewerWindow = False
            PosData.new_IDs = []
            PosData.lost_IDs = []
            PosData.multiBud_mothIDs = [2]
            PosData.UndoRedoStates = [[] for _ in range(PosData.segmSizeT)]
            PosData.UndoRedoCcaStates = [[] for _ in range(PosData.segmSizeT)]

            PosData.ol_data_dict = {}
            PosData.ol_data = None

            # Colormap
            PosData.lut = self.cmap.getLookupTable(0,1, max_ID+10)
            np.random.shuffle(PosData.lut)
            # Insert background color
            PosData.lut = np.insert(PosData.lut, 0, [25, 25, 25], axis=0)

            PosData.allData_li = [
                    {
                     'regionprops': None,
                     'labels': None,
                     'acdc_df': None,
                     'delROIs_info': {'rois': [], 'delMasks': [], 'delIDsROI': []},
                     'histoLevels': {}
                     }
                    for i in range(PosData.segmSizeT)
            ]

            PosData.ccaStatus_whenEmerged = {}

            PosData.frame_i = 0
            PosData.brushID = 0
            PosData.binnedIDs = set()
            PosData.ripIDs = set()
            PosData.multiContIDs = set()
            PosData.cca_df = None
            if PosData.last_tracked_i is not None:
                last_tracked_num = PosData.last_tracked_i+1
                # Load previous session data
                # Keep track of which ROIs have already been added in previous frame
                delROIshapes = [[] for _ in range(PosData.segmSizeT)]
                for i in range(last_tracked_num):
                    PosData.frame_i = i
                    self.get_data()
                    self.store_data()
                    # self.load_delROIs_info(delROIshapes, last_tracked_num)
                    PosData.binnedIDs = set()
                    PosData.ripIDs = set()

                # Ask whether to resume from last frame
                if last_tracked_num>1:
                    msg = QtGui.QMessageBox()
                    start_from_last_tracked_i = msg.question(
                        self, 'Start from last session?',
                        'The system detected a previous session ended '
                        f'at frame {last_tracked_num}.\n\n'
                        f'Do you want to resume from frame {last_tracked_num}?',
                        msg.Yes | msg.No
                    )
                    if start_from_last_tracked_i == msg.Yes:
                        PosData.frame_i = PosData.last_tracked_i
                    else:
                        PosData.frame_i = 0

        # Back to first position
        self.pos_i = 0
        self.get_data(debug=False)
        self.store_data()
        self.updateALLimg()

        # Link Y and X axis of both plots to scroll zoom and pan together
        self.ax2.vb.setYLink(self.ax1.vb)
        self.ax2.vb.setXLink(self.ax1.vb)

        self.ax2.vb.autoRange()
        self.ax1.vb.autoRange()

    def PosScrollBarMoved(self, pos_n):
        self.pos_i = pos_n-1
        proceed_cca, never_visited = self.get_data()
        self.updateALLimg(updateFilters=False)

    def PosScrollBarReleased(self):
        self.pos_i = self.framesScrollBar.sliderPosition()-1
        proceed_cca, never_visited = self.get_data()
        self.updateALLimg(updateFilters=True)
        self.computeSegm()

    def framesScrollBarMoved(self, frame_n):
        PosData = self.data[self.pos_i]
        PosData.frame_i = frame_n-1
        if PosData.allData_li[PosData.frame_i]['labels'] is None:
            PosData.lab = PosData.segm_data[PosData.frame_i]
        else:
            PosData.lab = PosData.allData_li[PosData.frame_i]['labels']
        cells_img = self.getImage()
        if self.framesScrollBarStartedMoving:
            self.clearAllItems()
        self.t_label.setText(
                 f'frame n. {PosData.frame_i+1}/{PosData.segmSizeT}')
        self.img1.setImage(cells_img)
        self.img2.setImage(PosData.lab)
        self.updateLookuptable()
        self.framesScrollBarStartedMoving = False

    def framesScrollBarReleased(self):
        self.framesScrollBarStartedMoving = True
        PosData = self.data[self.pos_i]
        PosData.frame_i = self.framesScrollBar.sliderPosition()-1
        self.get_data()
        self.updateALLimg()

    def unstore_data(self):
        PosData = self.data[self.pos_i]
        PosData.allData_li[PosData.frame_i] = {
            'regionprops': [],
            'labels': None,
            'acdc_df': None,
            'delROIs_info': {'rois': [], 'delMasks': [], 'delIDsROI': []},
            'histoLevels': {}
        }

    def store_data(self, pos_i=None, debug=False):
        pos_i = self.pos_i if pos_i is None else pos_i
        PosData = self.data[pos_i]
        if PosData.frame_i < 0:
            # In some cases we set frame_i = -1 and then call next_frame
            # to visualize frame 0. In that case we don't store data
            # for frame_i = -1
            return

        mode = str(self.modeComboBox.currentText())



        PosData.allData_li[PosData.frame_i]['regionprops'] = PosData.rp.copy()
        if mode != 'Viewer':
            PosData.allData_li[PosData.frame_i]['labels'] = PosData.lab.copy()

        if debug:
            pass
            # print(PosData.frame_i)
            # apps.imshow_tk(PosData.lab, additional_imgs=[PosData.allData_li[PosData.frame_i]['labels']])

        # Store dynamic metadata
        is_cell_dead_li = [False]*len(PosData.rp)
        is_cell_excluded_li = [False]*len(PosData.rp)
        IDs = [0]*len(PosData.rp)
        xx_centroid = [0]*len(PosData.rp)
        yy_centroid = [0]*len(PosData.rp)
        editIDclicked_x = [np.nan]*len(PosData.rp)
        editIDclicked_y = [np.nan]*len(PosData.rp)
        editIDnewID = [-1]*len(PosData.rp)
        editedIDs = [newID for _, _, newID in PosData.editID_info]
        for i, obj in enumerate(PosData.rp):
            is_cell_dead_li[i] = obj.dead
            is_cell_excluded_li[i] = obj.excluded
            IDs[i] = obj.label
            xx_centroid[i] = int(obj.centroid[1])
            yy_centroid[i] = int(obj.centroid[0])
            if obj.label in editedIDs:
                y, x, new_ID = PosData.editID_info[editedIDs.index(obj.label)]
                editIDclicked_x[i] = int(x)
                editIDclicked_y[i] = int(y)
                editIDnewID[i] = new_ID

        PosData.allData_li[PosData.frame_i]['acdc_df'] = pd.DataFrame(
            {
                        'Cell_ID': IDs,
                        'is_cell_dead': is_cell_dead_li,
                        'is_cell_excluded': is_cell_excluded_li,
                        'x_centroid': xx_centroid,
                        'y_centroid': yy_centroid,
                        'editIDclicked_x': editIDclicked_x,
                        'editIDclicked_y': editIDclicked_y,
                        'editIDnewID': editIDnewID
            }
        ).set_index('Cell_ID')

        self.store_cca_df(pos_i=pos_i)

    def nearest_point_2Dyx(self, points, all_others):
        """
        Given 2D array of [y, x] coordinates points and all_others return the
        [y, x] coordinates of the two points (one from points and one from all_others)
        that have the absolute minimum distance
        """
        # Compute 3D array where each ith row of each kth page is the element-wise
        # difference between kth row of points and ith row in all_others array.
        # (i.e. diff[k,i] = points[k] - all_others[i])
        diff = points[:, np.newaxis] - all_others
        # Compute 2D array of distances where
        # dist[i, j] = euclidean dist (points[i],all_others[j])
        dist = np.linalg.norm(diff, axis=2)
        # Compute i, j indexes of the absolute minimum distance
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        nearest_point = all_others[j]
        point = points[i]
        min_dist = dist.min()
        return min_dist, nearest_point

    def checkMultiBudMOth(self, draw=False):
        PosData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode.find('Cell cycle') == -1:
            PosData.multiBud_mothIDs = []
            return

        cca_df_S = PosData.cca_df[PosData.cca_df['cell_cycle_stage'] == 'S']
        cca_df_S_bud = cca_df_S[cca_df_S['relationship'] == 'bud']
        relIDs_of_S_bud = cca_df_S_bud['relative_ID']
        duplicated_relIDs_mask = relIDs_of_S_bud.duplicated(keep=False)
        duplicated_cca_df_S = cca_df_S_bud[duplicated_relIDs_mask]
        multiBud_mothIDs = duplicated_cca_df_S['relative_ID'].unique()
        PosData.multiBud_mothIDs = multiBud_mothIDs
        multiBudInfo = []
        for multiBud_ID in multiBud_mothIDs:
            duplicatedBuds_df = cca_df_S_bud[
                                    cca_df_S_bud['relative_ID'] == multiBud_ID]
            duplicatedBudIDs = duplicatedBuds_df.index.to_list()
            info = f'Mother ID {multiBud_ID} has bud IDs {duplicatedBudIDs}'
            multiBudInfo.append(info)
        if multiBudInfo:
            multiBudInfo_format = '\n'.join(multiBudInfo)
            self.MultiBudMoth_msg = QtGui.QMessageBox()
            self.MultiBudMoth_msg.setWindowTitle(
                                  'Mother with multiple buds assigned to it!')
            self.MultiBudMoth_msg.setText(multiBudInfo_format)
            self.MultiBudMoth_msg.setIcon(self.MultiBudMoth_msg.Warning)
            self.MultiBudMoth_msg.setDefaultButton(self.MultiBudMoth_msg.Ok)
            self.MultiBudMoth_msg.exec_()
        if draw:
            self.highlightmultiBudMoth()

    def attempt_auto_cca(self, enforceAll=False):
        PosData = self.data[self.pos_i]
        doNotProceed = False
        try:
            notEnoughG1Cells, proceed = self.autoCca_df(enforceAll=enforceAll)
            if PosData.cca_df is None:
                return notEnoughG1Cells, proceed
            mode = str(self.modeComboBox.currentText())
            if mode.find('Cell cycle') == -1:
                return notEnoughG1Cells, proceed
            if PosData.cca_df.isna().any(axis=None):
                raise ValueError('Cell cycle analysis table contains NaNs')
            self.checkMultiBudMOth()
        except Exception as e:
            print('')
            print('====================================')
            traceback.print_exc()
            print('====================================')
            print('')
            self.highlightNewIDs_ccaFailed()
            msg = QtGui.QMessageBox(self)
            msg.setIcon(msg.Critical)
            msg.setWindowTitle('Failed cell cycle analysis')
            msg.setDefaultButton(msg.Ok)
            msg.setText(
                f'Cell cycle analysis for frame {PosData.frame_i+1} failed!\n\n'
                'This can have multiple reasons:\n\n'
                '1. Segmentation or tracking errors --> Switch to \n'
                '   "Segmentation and Tracking" mode and check/correct next frame,\n'
                '   before attempting cell cycle analysis again.\n\n'
                '2. Edited a frame in "Segmentation and Tracking" mode\n'
                '   that already had cell cyce annotations -->\n'
                '   click on "Reinitialize cell cycle annotations" button,\n'
                '   and try again.')
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
        return notEnoughG1Cells, proceed

    def autoCca_df(self, enforceAll=False):
        """
        Assign each bud to a mother with scipy linear sum assignment
        (Hungarian or Munkres algorithm). First we build a cost matrix where
        each (i, j) element is the minimum distance between bud i and mother j.
        Then we minimize the cost of assigning each bud to a mother, and finally
        we write the assignment info into cca_df
        """
        proceed = True
        notEnoughG1Cells = False

        PosData = self.data[self.pos_i]

        # Skip cca if not the right mode
        mode = str(self.modeComboBox.currentText())
        if mode.find('Cell cycle') == -1:
            return notEnoughG1Cells, proceed


        # Make sure that this is a visited frame
        if PosData.allData_li[PosData.frame_i]['labels'] is None:
            msg = QtGui.QMessageBox()
            warn_cca = msg.critical(
                self, 'Next frame NEVER visited',
                'Next frame was never visited in "Segmentation and Tracking"'
                'mode.\n You cannot perform cell cycle analysis on frames'
                'where segmentation and/or tracking errors were not'
                'checked/corrected.\n\n'
                'Switch to "Segmentation and Tracking" mode '
                'and check/correct next frame,\n'
                'before attempting cell cycle analysis again',
                msg.Ok
            )
            proceed = False
            return notEnoughG1Cells, proceed

        # Determine if this is the last visited frame for repeating
        # bud assignment on non manually corrected_assignment buds.
        # The idea is that the user could have assigned division on a cell
        # by going previous and we want to check if this cell could be a
        # "better" mother for those non manually corrected buds
        lastVisited = False
        curr_df = PosData.allData_li[PosData.frame_i]['acdc_df']
        if curr_df is not None:
            if 'cell_cycle_stage' in curr_df.columns and not enforceAll:
                PosData.new_IDs = [ID for ID in PosData.new_IDs
                                if curr_df.at[ID, 'is_history_known']
                                and curr_df.at[ID, 'cell_cycle_stage'] == 'S']
                if PosData.frame_i+1 < PosData.segmSizeT:
                    next_df = PosData.allData_li[PosData.frame_i+1]['acdc_df']
                    if next_df is None:
                        lastVisited = True
                    else:
                        if 'cell_cycle_stage' not in next_df.columns:
                            lastVisited = True
                else:
                    lastVisited = True

        # Use stored cca_df and do not modify it with automatic stuff
        if PosData.cca_df is not None and not enforceAll and not lastVisited:
            return notEnoughG1Cells, proceed

        # Keep only correctedAssignIDs if requested
        # For the last visited frame we perform assignment again only on
        # IDs where we didn't manually correct assignment
        if lastVisited and not enforceAll:
            correctedAssignIDs = curr_df[curr_df['corrected_assignment']].index
            PosData.new_IDs = [ID for ID in PosData.new_IDs
                            if ID not in correctedAssignIDs]

        # Get previous dataframe
        acdc_df = PosData.allData_li[PosData.frame_i-1]['acdc_df']
        prev_cca_df = acdc_df[self.cca_df_colnames].copy()
        if PosData.cca_df is None:
            PosData.cca_df = prev_cca_df
        else:
            PosData.cca_df = curr_df[self.cca_df_colnames].copy()

        # If there are no new IDs we are done
        if not PosData.new_IDs:
            proceed = True
            self.store_cca_df()
            return notEnoughG1Cells, proceed

        # Get cells in G1 (exclude dead) and check if there are enough cells in G1
        prev_df_G1 = prev_cca_df[prev_cca_df['cell_cycle_stage']=='G1']
        prev_df_G1 = prev_df_G1[~acdc_df.loc[prev_df_G1.index]['is_cell_dead']]
        IDsCellsG1 = set(prev_df_G1.index)
        if lastVisited or enforceAll:
            # If we are repeating auto cca for last visited frame
            # then we also add the cells in G1 that we already know
            # at current frame
            df_G1 = PosData.cca_df[PosData.cca_df['cell_cycle_stage']=='G1']
            IDsCellsG1.update(df_G1.index)

        # remove cells that disappeared
        IDsCellsG1 = [ID for ID in IDsCellsG1 if ID in PosData.IDs]

        numCellsG1 = len(IDsCellsG1)
        numNewCells = len(PosData.new_IDs)
        if numCellsG1 < numNewCells:
            self.highlightNewIDs_ccaFailed()
            msg = QtGui.QMessageBox()
            warn_cca = msg.warning(
                self, 'No cells in G1!',
                f'In the next frame {numNewCells} new cells will '
                'appear (GREEN contour objects, left image).\n\n'
                f'However there are only {numCellsG1} cells '
                'in G1 available.\n\n'
                'You can either cancel the operation and "free" a cell '
                'by first annotating division on it or continue.\n\n'
                'If you continue the new cell will be annotated as a cell in G1 '
                'with unknown history.\n\n'
                'If you are not sure, before clicking "Yes" or "Cancel", you can '
                'preview (green contour objects, left image) '
                'where the new cells will appear.\n\n'
                'Do you want to continue?\n',
                msg.Yes | msg.Cancel
            )
            if warn_cca == msg.Yes:
                notEnoughG1Cells = False
                proceed = True
                # Annotate the new IDs with unknown history
                for ID in PosData.new_IDs:
                    PosData.cca_df.loc[ID] = pd.Series({
                        'cell_cycle_stage': 'G1',
                        'generation_num': 2,
                        'relative_ID': -1,
                        'relationship': 'mother',
                        'emerg_frame_i': -1,
                        'division_frame_i': -1,
                        'is_history_known': False,
                        'corrected_assignment': False
                    })
                    cca_df_ID = self.getStatusKnownHistoryBud(ID)
                    PosData.ccaStatus_whenEmerged[ID] = cca_df_ID
            else:
                notEnoughG1Cells = True
                proceed = False
            return notEnoughG1Cells, proceed

        # Compute new IDs contours
        newIDs_contours = []
        for obj in PosData.rp:
            ID = obj.label
            if ID in PosData.new_IDs:
                cont = self.getObjContours(obj)
                newIDs_contours.append(cont)

        # Compute cost matrix
        cost = np.full((numCellsG1, numNewCells), np.inf)
        for obj in PosData.rp:
            ID = obj.label
            if ID in IDsCellsG1:
                cont = self.getObjContours(obj)
                i = IDsCellsG1.index(ID)
                for j, newID_cont in enumerate(newIDs_contours):
                    min_dist, nearest_xy = self.nearest_point_2Dyx(
                                                         cont, newID_cont)
                    cost[i, j] = min_dist

        # Run hungarian (munkres) assignment algorithm
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost)

        # Assign buds to mothers
        for i, j in zip(row_idx, col_idx):
            mothID = IDsCellsG1[i]
            budID = PosData.new_IDs[j]

            # If we are repeating assignment for the bud then we also have to
            # correct the possibily wrong mother first
            if budID in PosData.cca_df.index:
                relID = PosData.cca_df.at[budID, 'relative_ID']
                if relID in prev_cca_df.index:
                    PosData.cca_df.loc[relID] = prev_cca_df.loc[relID]


            PosData.cca_df.at[mothID, 'relative_ID'] = budID
            PosData.cca_df.at[mothID, 'cell_cycle_stage'] = 'S'

            PosData.cca_df.loc[budID] = pd.Series({
                'cell_cycle_stage': 'S',
                'generation_num': 0,
                'relative_ID': mothID,
                'relationship': 'bud',
                'emerg_frame_i': PosData.frame_i,
                'division_frame_i': -1,
                'is_history_known': True,
                'corrected_assignment': False
            })


        # Keep only existing IDs
        PosData.cca_df = PosData.cca_df.loc[PosData.IDs]

        self.store_cca_df()
        proceed = True
        return notEnoughG1Cells, proceed

    def getObjContours(self, obj, appendMultiContID=True):
        contours, _ = cv2.findContours(
                               obj.image.astype(np.uint8),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE
        )
        min_y, min_x, _, _ = obj.bbox
        cont = np.squeeze(contours[0], axis=1)
        if len(contours)>1 and appendMultiContID:
            PosData = self.data[self.pos_i]
            PosData.multiContIDs.add(obj.label)
        cont = np.vstack((cont, cont[0]))
        cont += [min_x, min_y]
        return cont

    def get_data(self, debug=False):
        PosData = self.data[self.pos_i]
        proceed_cca = True
        if PosData.frame_i > 2:
            # Remove undo states from 4 frames back to avoid memory issues
            PosData.UndoRedoStates[PosData.frame_i-4] = []
            # Check if current frame contains undo states (not empty list)
            if PosData.UndoRedoStates[PosData.frame_i]:
                self.undoAction.setDisabled(False)
            else:
                self.undoAction.setDisabled(True)
        self.UndoCount = 0
        # If stored labels is None then it is the first time we visit this frame
        if PosData.allData_li[PosData.frame_i]['labels'] is None:
            PosData.editID_info = []
            never_visited = True
            if str(self.modeComboBox.currentText()) == 'Cell cycle analysis':
                # Warn that we are visiting a frame that was never segm-checked
                # on cell cycle analysis mode
                msg = QtGui.QMessageBox()
                warn_cca = msg.critical(
                    self, 'Never checked segmentation on requested frame',
                    'Segmentation and Tracking was never checked from '
                    f'frame {PosData.frame_i+1} onward.\n To ensure correct cell '
                    'cell cycle analysis you have to first visit frames '
                    f'{PosData.frame_i+1}-end with "Segmentation and Tracking" mode.',
                    msg.Ok
                )
                proceed_cca = False
                return proceed_cca, never_visited
            # Requested frame was never visited before. Load from HDD
            PosData.lab = PosData.segm_data[PosData.frame_i].copy()
            PosData.rp = skimage.measure.regionprops(PosData.lab)
            if debug:
                print(never_visited)
            if PosData.acdc_df is not None:
                frames = PosData.acdc_df.index.get_level_values(0)
                if PosData.frame_i in frames:
                    # Since there was already segmentation metadata from
                    # previous closed session add it to current metadata
                    df = PosData.acdc_df.loc[PosData.frame_i].copy()
                    try:
                        binnedIDs_df = df[df['is_cell_excluded']]
                    except Exception as e:
                        print('')
                        print('====================================')
                        traceback.print_exc()
                        print('====================================')
                        print('')
                        raise
                    binnedIDs = set(binnedIDs_df.index).union(PosData.binnedIDs)
                    PosData.binnedIDs = binnedIDs
                    ripIDs_df = df[df['is_cell_dead']]
                    ripIDs = set(ripIDs_df.index).union(PosData.ripIDs)
                    PosData.ripIDs = ripIDs
                    # Load cca df into current metadata
                    if 'cell_cycle_stage' in df.columns:
                        if any(df['cell_cycle_stage'].isna()):
                            if 'is_history_known' not in df.columns:
                                df['is_history_known'] = True
                            if 'corrected_assignment' not in df.columns:
                                df['corrected_assignment'] = True
                            df = df.drop(labels=self.cca_df_colnames, axis=1)
                        else:
                            # Convert to ints since there were NaN
                            cols = self.cca_df_int_cols
                            df[cols] = df[cols].astype(int)
                    i = PosData.frame_i
                    PosData.allData_li[i]['acdc_df'] = df.copy()
            self.get_cca_df()
        else:
            # Requested frame was already visited. Load from RAM.
            never_visited = False
            PosData.lab = PosData.allData_li[PosData.frame_i]['labels'].copy()
            PosData.rp = skimage.measure.regionprops(PosData.lab)
            df = PosData.allData_li[PosData.frame_i]['acdc_df']
            binnedIDs_df = df[df['is_cell_excluded']]
            PosData.binnedIDs = set(binnedIDs_df.index)
            ripIDs_df = df[df['is_cell_dead']]
            PosData.ripIDs = set(ripIDs_df.index)
            editIDclicked_x = df['editIDclicked_x'].to_list()
            editIDclicked_y = df['editIDclicked_y'].to_list()
            editIDnewID = df['editIDnewID'].to_list()
            _zip = zip(editIDclicked_y, editIDclicked_x, editIDnewID)
            PosData.editID_info = [
                (int(y),int(x),newID) for y,x,newID in _zip if newID!=-1]
            self.get_cca_df()

        self.update_rp_metadata(draw=False)
        PosData.IDs = [obj.label for obj in PosData.rp]
        return proceed_cca, never_visited

    def load_delROIs_info(self, delROIshapes, last_tracked_num):
        PosData = self.data[self.pos_i]
        delROIsInfo_npz = PosData.delROIsInfo_npz
        if delROIsInfo_npz is None:
            return
        for file in PosData.delROIsInfo_npz.files:
            if not file.startswith(f'{PosData.frame_i}_'):
                continue

            delROIs_info = PosData.allData_li[PosData.frame_i]['delROIs_info']
            if file.startswith(f'{PosData.frame_i}_delMask'):
                delMask = delROIsInfo_npz[file]
                delROIs_info['delMasks'].append(delMask)
            elif file.startswith(f'{PosData.frame_i}_delIDs'):
                delIDsROI = set(delROIsInfo_npz[file])
                delROIs_info['delIDsROI'].append(delIDsROI)
            elif file.startswith(f'{PosData.frame_i}_roi'):
                Y, X = PosData.lab.shape
                x0, y0, w, h = delROIsInfo_npz[file]
                addROI = (
                    PosData.frame_i==0 or
                    [x0, y0, w, h] not in delROIshapes[PosData.frame_i]
                )
                if addROI:
                    roi = self.getDelROI(xl=x0, yb=y0, w=w, h=h)
                    for i in range(PosData.frame_i, last_tracked_num):
                        delROIs_info_i = PosData.allData_li[i]['delROIs_info']
                        delROIs_info_i['rois'].append(roi)
                        delROIshapes[i].append([x0, y0, w, h])

    def addIDBaseCca_df(self, ID):
        PosData = self.data[self.pos_i]
        PosData.cca_df.loc[ID] = pd.Series({
            'cell_cycle_stage': 'G1',
            'generation_num': 2,
            'relative_ID': -1,
            'relationship': 'mother',
            'emerg_frame_i': -1,
            'division_frame_i': -1,
            'is_history_known': False,
            'corrected_assignment': False
        })
        self.store_cca_df()

    def getBaseCca_df(self):
        PosData = self.data[self.pos_i]
        IDs = [obj.label for obj in PosData.rp]
        cc_stage = ['G1' for ID in IDs]
        num_cycles = [2]*len(IDs)
        relationship = ['mother' for ID in IDs]
        related_to = [-1]*len(IDs)
        emerg_frame_i = [-1]*len(IDs)
        division_frame_i = [-1]*len(IDs)
        is_history_known = [False]*len(IDs)
        corrected_assignment = [False]*len(IDs)
        cca_df = pd.DataFrame({
                           'cell_cycle_stage': cc_stage,
                           'generation_num': num_cycles,
                           'relative_ID': related_to,
                           'relationship': relationship,
                           'emerg_frame_i': emerg_frame_i,
                           'division_frame_i': division_frame_i,
                           'is_history_known': is_history_known,
                           'corrected_assignment': corrected_assignment},
                            index=IDs)
        cca_df.index.name = 'Cell_ID'
        return cca_df

    def initSegmTrackMode(self):
        PosData = self.data[self.pos_i]
        last_tracked_i = 0
        for frame_i, data_dict in enumerate(PosData.allData_li):
            # Build segm_npy
            lab = data_dict['labels']
            if lab is None:
                last_tracked_i = frame_i-1
                break
            else:
                last_tracked_i = PosData.segmSizeT-1

        self.framesScrollBar.setMaximum(last_tracked_i+1)
        if PosData.frame_i > last_tracked_i:
            # Prompt user to go to last tracked frame
            msg = QtGui.QMessageBox()
            msg.setIcon(msg.Warning)
            msg.setWindowTitle('Go to last visited frame?')
            msg.setText(
                f'The last visited frame in "Segmentation and Tracking mode" '
                f'is frame {last_tracked_i+1}.\n\n'
                f'We recommend to resume from that frame.\n\n'
                'How do you want to proceed?'
            )
            goToButton = QPushButton(
                f'Resume from frame {last_tracked_i+1} (RECOMMENDED)'
            )
            stayButton = QPushButton(
                f'Stay on current frame {PosData.frame_i+1}'
            )
            msg.addButton(goToButton, msg.YesRole)
            msg.addButton(stayButton, msg.NoRole)
            msg.exec_()
            if msg.clickedButton() == goToButton:
                PosData.frame_i = last_tracked_i
                self.get_data()
                self.updateALLimg()
                self.updateScrollbars()
            else:
                return

    def init_cca(self):
        PosData = self.data[self.pos_i]
        if PosData.last_tracked_i is None:
            txt = (
                'On this dataset either you never checked that the segmentation '
                'and tracking are correct or you did not save yet.\n\n'
                'If you already visited some frames with "Segmentation and tracking" '
                'mode save data before switching to "Cell cycle analysis mode".\n\n'
                'Otherwise you first have to check (and eventually correct) some frames '
                'in "Segmentation and Tracking" mode before proceeding '
                'with cell cycle analysis.')
            msg = QtGui.QMessageBox()
            msg.critical(
                self, 'Tracking check not performed', txt, msg.Ok
            )
            self.modeComboBox.setCurrentIndex(0)
            return

        proceed = True
        i = 0
        # Determine last annotated frame index
        for i, dict_frame_i in enumerate(PosData.allData_li):
            df = dict_frame_i['acdc_df']
            if df is None:
                break
            else:
                if 'cell_cycle_stage' not in df.columns:
                    break

        last_cca_frame_i = i-1 if i>0 else 0

        if last_cca_frame_i == 0:
            # Remove undoable actions from segmentation mode
            PosData.UndoRedoStates[0] = []
            self.undoAction.setEnabled(False)
            self.redoAction.setEnabled(False)

        if PosData.frame_i > last_cca_frame_i:
            # Prompt user to go to last annotated frame
            msg = QtGui.QMessageBox()
            goTo_last_annotated_frame_i = msg.warning(
                self, 'Go to last annotated frame?',
                f'The last annotated frame is frame {last_cca_frame_i+1}.\n'
                'The cell cycle analysis will restart from that frame.\n'
                'Do you want to proceed?',
                msg.Yes | msg.Cancel
            )
            if goTo_last_annotated_frame_i == msg.Yes:
                msg = 'Looking good!'
                self.last_cca_frame_i = last_cca_frame_i
                PosData.frame_i = last_cca_frame_i
                self.titleLabel.setText(msg, color='w')
                self.get_data()
                self.updateALLimg()
                self.updateScrollbars()
            else:
                msg = 'Cell cycle analysis aborted.'
                print(msg)
                self.titleLabel.setText(msg, color='w')
                self.modeComboBox.setCurrentText('Viewer')
                proceed = False
                return
        elif PosData.frame_i < last_cca_frame_i:
            # Prompt user to go to last annotated frame
            msg = QtGui.QMessageBox()
            goTo_last_annotated_frame_i = msg.question(
                self, 'Go to last annotated frame?',
                f'The last annotated frame is frame {last_cca_frame_i+1}.\n'
                'Do you want to restart cell cycle analysis from frame '
                f'{last_cca_frame_i+1}?',
                msg.Yes | msg.No | msg.Cancel
            )
            if goTo_last_annotated_frame_i == msg.Yes:
                msg = 'Looking good!'
                self.titleLabel.setText(msg, color='w')
                self.last_cca_frame_i = last_cca_frame_i
                PosData.frame_i = last_cca_frame_i
                self.get_data()
                self.updateALLimg()
                self.updateScrollbars()
            elif goTo_last_annotated_frame_i == msg.Cancel:
                msg = 'Cell cycle analysis aborted.'
                print(msg)
                self.titleLabel.setText(msg, color='w')
                proceed = False
                return
        else:
            self.get_data()

        self.last_cca_frame_i = last_cca_frame_i

        self.framesScrollBar.setMaximum(last_cca_frame_i+1)

        if PosData.cca_df is None:
            PosData.cca_df = self.getBaseCca_df()
            msg = 'Cell cycle analysis initiliazed!'
            print(msg)
            self.titleLabel.setText(msg, color='w')
        else:
            self.get_cca_df()
        return proceed

    def remove_future_cca_df(self, from_frame_i):
        PosData = self.data[self.pos_i]
        self.last_cca_frame_i = PosData.frame_i
        self.setFramesScrollbarMaximum()
        for i in range(from_frame_i, PosData.segmSizeT):
            df = PosData.allData_li[i]['acdc_df']
            if df is None:
                # No more saved info to delete
                return

            if 'cell_cycle_stage' not in df.columns:
                # No cell cycle info present
                continue

            df.drop(self.cca_df_colnames, axis=1, inplace=True)
            PosData.allData_li[i]['acdc_df'] = df

    def get_cca_df(self, frame_i=None, return_df=False):
        # cca_df is None unless the metadata contains cell cycle annotations
        # NOTE: cell cycle annotations are either from the current session
        # or loaded from HDD in "initPosAttr" with a .question to the user
        PosData = self.data[self.pos_i]
        cca_df = None
        i = PosData.frame_i if frame_i is None else frame_i
        df = PosData.allData_li[i]['acdc_df']
        if df is not None:
            if 'cell_cycle_stage' in df.columns:
                if 'is_history_known' not in df.columns:
                    df['is_history_known'] = True
                if 'corrected_assignment' not in df.columns:
                    # Compatibility with those acdc_df analysed with prev vers.
                    df['corrected_assignment'] = True
                cca_df = df[self.cca_df_colnames].copy()
        if cca_df is None and self.isSnapshot:
            cca_df = self.getBaseCca_df()
            PosData.cca_df = cca_df
        if return_df:
            return cca_df
        else:
            PosData.cca_df = cca_df

    def unstore_cca_df(self):
        PosData = self.data[self.pos_i]
        acdc_df = PosData.allData_li[PosData.frame_i]['acdc_df']
        for col in self.cca_df_colnames:
            if col not in acdc_df.columns:
                continue
            acdc_df.drop(col, axis=1, inplace=True)

    def store_cca_df(self, pos_i=None, frame_i=None, cca_df=None):
        pos_i = self.pos_i if pos_i is None else pos_i
        PosData = self.data[pos_i]
        i = PosData.frame_i if frame_i is None else frame_i
        if cca_df is None:
            cca_df = PosData.cca_df
            if self.ccaTableWin is not None:
                self.ccaTableWin.updateTable(PosData.cca_df)

        if cca_df is not None:
            acdc_df = PosData.allData_li[i]['acdc_df']
            if acdc_df is None:
                self.store_data()
                acdc_df = PosData.allData_li[i]['acdc_df']
            if 'cell_cycle_stage' in acdc_df.columns:
                # Cell cycle info already present --> overwrite with new
                df = acdc_df
                df[self.cca_df_colnames] = cca_df
            else:
                df = acdc_df.join(cca_df, how='left')
            PosData.allData_li[i]['acdc_df'] = df.copy()
            # print(PosData.allData_li[PosData.frame_i]['acdc_df'])

    def ax1_setTextID(self, obj, how, updateColor=False):
        PosData = self.data[self.pos_i]
        # Draw ID label on ax1 image depending on how
        LabelItemID = self.ax1_LabelItemsIDs[obj.label-1]
        ID = obj.label
        df = PosData.cca_df
        if df is None or how.find('cell cycle') == -1:
            txt = f'{ID}'
            if updateColor:
                LabelItemID.setText(txt, size=self.fontSize)
            if ID in PosData.new_IDs:
                color = 'r'
                bold = True
            else:
                color = self.ax1_oldIDcolor
                if updateColor:
                    color = self.getOptimalLabelItemColor(LabelItemID, color)
                    self.ax1_oldIDcolor = color
                bold = False
        else:
            df_ID = df.loc[ID]
            ccs = df_ID['cell_cycle_stage']
            relationship = df_ID['relationship']
            generation_num = df_ID['generation_num']
            generation_num = 'ND' if generation_num==-1 else generation_num
            emerg_frame_i = df_ID['emerg_frame_i']
            is_history_known = df_ID['is_history_known']
            is_bud = relationship == 'bud'
            is_moth = relationship == 'mother'
            emerged_now = emerg_frame_i == PosData.frame_i
            txt = f'{ccs}-{generation_num}'
            if updateColor:
                LabelItemID.setText(txt, size=self.fontSize)
            if ccs == 'G1':
                color = self.ax1_G1cellColor
                if updateColor:
                    c = self.getOptimalLabelItemColor(LabelItemID, c)
                    self.ax1_G1cellColor = c
                bold = False
            elif ccs == 'S' and is_moth and not emerged_now:
                color = self.ax1_S_oldCellColor
                if updateColor:
                    c = self.getOptimalLabelItemColor(LabelItemID, c)
                    self.ax1_S_oldCellColor = c
                bold = False
            elif ccs == 'S' and is_bud and not emerged_now:
                color = 'r'
                bold = False
            elif ccs == 'S' and is_bud and emerged_now:
                color = 'r'
                bold = True

            if not is_history_known:
                txt = f'{txt}?'

        try:
            LabelItemID.setText(txt, color=color, bold=bold, size=self.fontSize)
        except UnboundLocalError:
            pass


        # Center LabelItem at centroid
        y, x = obj.centroid
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)

    def ax2_setTextID(self, obj):
        PosData = self.data[self.pos_i]
        # Draw ID label on ax1 image
        LabelItemID = self.ax2_LabelItemsIDs[obj.label-1]
        ID = obj.label
        df = PosData.cca_df
        txt = f'{ID}'
        if ID in PosData.new_IDs:
            color = 'r'
            bold = True
        else:
            color = (230, 0, 0, 255*0.5)
            bold = False

        LabelItemID.setText(txt, color=color, bold=bold, size=self.fontSize)

        # Center LabelItem at centroid
        y, x = obj.centroid
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)

    def drawID_and_Contour(self, obj, drawContours=True, updateColor=False):
        PosData = self.data[self.pos_i]
        how = self.drawIDsContComboBox.currentText()
        IDs_and_cont = how == 'Draw IDs and contours'
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'

        # Draw LabelItems for IDs on ax2
        y, x = obj.centroid
        idx = obj.label-1
        t0 = time.time()
        self.ax2_setTextID(obj)

        if PosData.cca_df is not None and self.isSnapshot:
            if obj.label not in PosData.cca_df.index:
                self.store_data()
                self.addIDBaseCca_df(obj.label)

        # Draw LabelItems for IDs on ax1 if requested
        if IDs_and_cont or onlyIDs or only_ccaInfo or ccaInfo_and_cont:
            # Draw LabelItems for IDs on ax2
            t0 = time.time()
            self.ax1_setTextID(obj, how, updateColor=updateColor)

        t1 = time.time()
        self.drawingLabelsTimes.append(t1-t0)

        # Draw line connecting mother and buds
        drawLines = only_ccaInfo or ccaInfo_and_cont or onlyMothBudLines
        if drawLines and PosData.cca_df is not None:
            ID = obj.label
            BudMothLine = self.ax1_BudMothLines[ID-1]
            cca_df_ID = PosData.cca_df.loc[ID]
            ccs_ID = cca_df_ID['cell_cycle_stage']
            relationship = cca_df_ID['relationship']
            if ccs_ID == 'S' and relationship=='bud':
                emerg_frame_i = cca_df_ID['emerg_frame_i']
                if emerg_frame_i == PosData.frame_i:
                    pen = self.NewBudMoth_Pen
                else:
                    pen = self.OldBudMoth_Pen
                relative_ID = cca_df_ID['relative_ID']
                if relative_ID in PosData.IDs:
                    relative_rp_idx = PosData.IDs.index(relative_ID)
                    relative_ID_obj = PosData.rp[relative_rp_idx]
                    y1, x1 = obj.centroid
                    y2, x2 = relative_ID_obj.centroid
                    BudMothLine.setData([x1, x2], [y1, y2], pen=pen)
            else:
                BudMothLine.setData([], [])

        if not drawContours:
            return

        # Draw contours on ax1 if requested
        if IDs_and_cont or onlyCont or ccaInfo_and_cont:
            ID = obj.label
            t0 = time.time()
            cont = self.getObjContours(obj)
            t1 = time.time()
            computingContoursTime = t1-t0
            self.computingContoursTimes.append(computingContoursTime)

            t0 = time.time()
            curveID = self.ax1_ContoursCurves[idx]
            pen = self.newIDs_cpen if ID in PosData.new_IDs else self.oldIDs_cpen
            curveID.setData(cont[:,0], cont[:,1], pen=pen)
            t1 = time.time()
            drawingContoursTimes = t1-t0
            self.drawingContoursTimes.append(drawingContoursTimes)

    def update_rp(self, draw=True):
        PosData = self.data[self.pos_i]
        # Update rp for current PosData.lab (e.g. after any change)
        PosData.rp = skimage.measure.regionprops(PosData.lab)
        PosData.IDs = [obj.label for obj in PosData.rp]
        self.update_rp_metadata()

    def update_IDsContours(self, prev_IDs, newIDs=[]):
        """Function to draw labels text and contours of specific IDs.
        It should speed up things because we draw only few IDs usually.

        Parameters
        ----------
        prev_IDs : list
            List of IDs before the change (e.g. before erasing ID or painting
            a new one etc.)
        newIDs : bool
            List of new IDs that are present after a change.

        Returns
        -------
        None.

        """

        PosData = self.data[self.pos_i]

        # Draw label and contours of the new IDs
        if len(newIDs)>0:
            for i, obj in enumerate(PosData.rp):
                ID = obj.label
                if ID in newIDs:
                    # Draw ID labels and contours of new objects
                    self.drawID_and_Contour(obj)

        # Clear contours and LabelItems of IDs that are in prev_IDs
        # but not in current IDs
        currentIDs = [obj.label for obj in PosData.rp]
        for prevID in prev_IDs:
            if prevID not in currentIDs:
                self.ax1_ContoursCurves[prevID-1].setData([], [])
                self.ax1_LabelItemsIDs[prevID-1].setText('')
                self.ax2_LabelItemsIDs[prevID-1].setText('')

        self.highlightLostNew()
        self.checkIDsMultiContour()
        self.highlightmultiBudMoth()

    def highlightmultiBudMoth(self):
        PosData = self.data[self.pos_i]
        for ID in PosData.multiBud_mothIDs:
            LabelItemID = self.ax1_LabelItemsIDs[ID-1]
            txt = LabelItemID
            LabelItemID.setText(f'{txt} !!', color=self.lostIDs_qMcolor)

    def checkIDsMultiContour(self):
        PosData = self.data[self.pos_i]
        txt = self.titleLabel.text
        if 'Looking' not in txt or 'Data' not in txt:
            warn_txt = self.titleLabel.text
            htmlTxt = (
                f'<font color="red">{warn_txt}</font>'
            )
        else:
            htmlTxt = f'<font color="white">{self.titleLabel.text}</font>'
        if PosData.multiContIDs:
            warn_txt = f'IDs with multiple contours: {PosData.multiContIDs}'
            color = 'red'
            htmlTxt = (
                f'<font color="red">{warn_txt}</font>, {htmlTxt}'
            )
            PosData.multiContIDs = set()
        self.titleLabel.setText(htmlTxt)

    def updateLookuptable(self):
        PosData = self.data[self.pos_i]
        lenNewLut = PosData.lab.max()+1
        # Build a new lut to include IDs > than original len of lut
        if lenNewLut > len(PosData.lut):
            numNewColors = lenNewLut-len(PosData.lut)
            # Index original lut
            _lut = np.zeros((lenNewLut, 3), np.uint8)
            _lut[:len(PosData.lut)] = PosData.lut
            # Pick random colors and append them at the end to recycle them
            randomIdx = np.random.randint(0,len(PosData.lut),size=numNewColors)
            for i, idx in enumerate(randomIdx):
                rgb = PosData.lut[idx]
                _lut[len(PosData.lut)+i] = rgb
            PosData.lut = _lut

        try:
            lut = PosData.lut[:lenNewLut].copy()
            for ID in PosData.binnedIDs:
                lut[ID] = lut[ID]*0.2
            for ID in PosData.ripIDs:
                lut[ID] = lut[ID]*0.2
        except Exception as e:
            print('WARNING: Tracking is WRONG.')
            pass
        self.img2.setLookupTable(lut)

    def update_rp_metadata(self, draw=True):
        PosData = self.data[self.pos_i]
        binnedIDs_xx = []
        binnedIDs_yy = []
        ripIDs_xx = []
        ripIDs_yy = []
        # Add to rp dynamic metadata (e.g. cells annotated as dead)
        for i, obj in enumerate(PosData.rp):
            ID = obj.label
            # IDs removed from analysis --> store info
            if ID in PosData.binnedIDs:
                obj.excluded = True
                if draw:
                    y, x = obj.centroid
                    # Gray out ID label on image
                    LabelID = self.ax2_LabelItemsIDs[ID-1]
                    LabelID.setText(f'{ID}', color=(150, 0, 0))
                    binnedIDs_xx.append(x)
                    binnedIDs_yy.append(y)
            else:
                obj.excluded = False

            # IDs dead --> store info
            if ID in PosData.ripIDs:
                obj.dead = True
                if draw:
                    # Gray out ID label on image
                    y, x = obj.centroid
                    LabelID = self.ax2_LabelItemsIDs[ID-1]
                    LabelID.setText(f'{ID}', color=(150, 0, 0))
                    ripIDs_xx.append(x)
                    ripIDs_yy.append(y)
            else:
                obj.dead = False

            # set cell cycle info

        if draw:
            # Draw markers to annotated IDs
            self.ax2_binnedIDs_ScatterPlot.setData(binnedIDs_xx, binnedIDs_yy)
            self.ax2_ripIDs_ScatterPlot.setData(ripIDs_xx, ripIDs_yy)
            self.ax1_binnedIDs_ScatterPlot.setData(binnedIDs_xx, binnedIDs_yy)
            self.ax1_ripIDs_ScatterPlot.setData(ripIDs_xx, ripIDs_yy)

    def loadNonAlignedFluoChannel(self, fluo_path):
        PosData = self.data[self.pos_i]
        if PosData.filename.find('aligned') != -1:
            filename, _ = os.path.splitext(os.path.basename(fluo_path))
            path = f'.../{PosData.pos_foldername}/Images/{filename}_aligned.npz'
            msg = QtGui.QMessageBox()
            msg.critical(
                self, 'Aligned fluo channel not found!',
                'Aligned data for fluorescent channel not found!\n\n'
                f'You loaded aligned data for the cells channel, therefore '
                'loading NON-aligned fluorescent data is not allowed.\n\n'
                'Run the script "dataPrep.py" to create the following file:\n\n'
                f'{path}',
                msg.Ok
            )
            return None
        fluo_data = skimage.io.imread(fluo_path)
        return fluo_data

    def load_fluo_data(self, fluo_path):
        print('Loading fluorescent image data...')
        PosData = self.data[self.pos_i]
        # Load overlay frames and align if needed
        filename = os.path.basename(fluo_path)
        filename_noEXT, ext = os.path.splitext(filename)
        if ext == '.npy' or ext == '.npz':
            fluo_data = np.load(fluo_path)
            try:
                fluo_data = fluo_data['arr_0']
            except Exception as e:
                fluo_data = fluo_data
        elif ext == '.tif' or ext == '.tiff':
            aligned_filename = f'{filename_noEXT}_aligned.npz'
            aligned_path = os.path.join(PosData.images_path, aligned_filename)
            if os.path.exists(aligned_path):
                fluo_data = np.load(aligned_path)['arr_0']
            else:
                fluo_data = self.loadNonAlignedFluoChannel(fluo_path)
                if fluo_data is None:
                    return None, None
        else:
            txt = (f'File format {ext} is not supported!\n'
                    'Choose either .tif or .npz files.')
            msg = QtGui.QMessageBox()
            msg.critical(
                self, 'File not supported', txt, msg.Ok
            )
            return None, None

        if PosData.SizeZ > 1:
            ol_data = fluo_data.max(axis=0)
        else:
            ol_data = fluo_data.copy()
        return fluo_data, ol_data

    def setOverlayColors(self):
        self.overlayRGBs = [(255, 255, 0),
                            (252, 72, 254),
                            (49, 222, 134),
                            (22, 108, 27)]

    def getFileExtensions(self, images_path):
        alignedFound = any([f.find('_aligned.np')!=-1
                            for f in os.listdir(images_path)])
        if alignedFound:
            extensions = (
                'Aligned channels (*npz *npy);; Tif channels(*tiff *tif)'
                ';;All Files (*)'
            )
        else:
            extensions = (
                'Tif channels(*tiff *tif);; All Files (*)'
            )
        return extensions

    def overlay_cb(self, checked):
        PosData = self.data[self.pos_i]
        if checked:
            prompt = True
            if PosData.ol_data is not None:
                prompt = False
            # Check if there is already loaded data
            elif PosData.fluo_data_dict and PosData.ol_data is None:
                ch_names = list(PosData.loadedFluoChannels)
                if len(ch_names)>1:
                    selectFluo = apps.QDialogListbox(
                        'Select channel',
                        'Select channel names to load:\n',
                        ch_names, multiSelection=True, parent=self
                    )
                    selectFluo.exec_()
                    ol_channels = selectFluo.selectedItemsText
                    if selectFluo.cancel or not ol_channels:
                        prompt = True
                    else:
                        prompt = False
                else:
                    prompt = False
                    ol_channels = ch_names

                for PosData in self.data:
                    ol_data = {}
                    ol_colors = {}
                    for i, ol_ch in enumerate(ol_channels):
                        ol_path, filename = self.getPathFromChName(
                                                            ol_ch, PosData)
                        if ol_path is None:
                            self.criticalFluoChannelNotFound(ol_ch, PosData)
                            self.app.restoreOverrideCursor()
                            return
                        ol_data[filename] = PosData.ol_data_dict[filename].copy()
                        ol_colors[filename] = self.overlayRGBs[i]
                        self.addFluoChNameContextMenuAction(ol_ch)
                    PosData.manualContrastKey = filename
                    PosData.ol_data = ol_data
                    PosData.ol_colors = ol_colors

            if prompt:
                # extensions = self.getFileExtensions(PosData.images_path)
                # ol_paths = QFileDialog.getOpenFileNames(
                #     self, 'Select one or multiple fluorescent images',
                #     PosData.images_path, extensions
                # )
                ch_names = [ch for ch in self.ch_names if ch != self.user_ch_name]
                selectFluo = apps.QDialogListbox(
                    'Select channel',
                    'Select channel names to load:\n',
                    ch_names, multiSelection=True, parent=self
                )
                selectFluo.exec_()
                if selectFluo.cancel:
                    self.overlayButton.setChecked(False)
                    return
                ol_channels = selectFluo.selectedItemsText

                self.app.setOverrideCursor(Qt.WaitCursor)
                for PosData in self.data:
                    ol_data = {}
                    ol_colors = {}
                    for i, ol_ch in enumerate(ol_channels):
                        ol_path, filename = self.getPathFromChName(ol_ch,
                                                                   PosData)
                        if ol_path is None:
                            self.criticalFluoChannelNotFound(ol_ch, PosData)
                            self.app.restoreOverrideCursor()
                            return
                        fluo_data, ol_data_2D = self.load_fluo_data(ol_path)
                        if fluo_data is None:
                            self.app.restoreOverrideCursor()
                            return
                        PosData.fluo_data_dict[filename] = fluo_data
                        PosData.ol_data_dict[filename] = fluo_data
                        ol_data[filename] = ol_data_2D
                        ol_colors[filename] = self.overlayRGBs[i]
                        PosData.ol_colors = ol_colors
                        if i!=0:
                            continue
                        self.addFluoChNameContextMenuAction(ol_ch)
                    PosData.manualContrastKey = filename
                    PosData.ol_data = ol_data

                self.app.restoreOverrideCursor()
                self.overlayButton.setStyleSheet('background-color: #A7FAC7')

            self.UserNormAction, _, _ = self.getCheckNormAction()
            self.normalizeRescale0to1Action.setChecked(True)
            self.hist.imageItem = lambda: None
            self.updateHistogramItem(self.img1)

            rgb = self.df_settings.at['overlayColor', 'value']
            rgb = [int(v) for v in rgb.split('-')]
            self.overlayColorButton.setColor(rgb)

            self.updateALLimg(only_ax1=True)

            self.alphaScrollBar.setDisabled(False)
            self.overlayColorButton.setDisabled(False)
            self.editOverlayColorAction.setDisabled(False)
            self.alphaScrollBar.show()
            self.alphaScrollBar_label.show()
        else:
            self.UserNormAction.setChecked(True)
            self.create_chNamesQActionGroup(self.user_ch_name)
            PosData.fluoDataChNameActions = []
            self.updateHistogramItem(self.img1)
            self.updateALLimg(only_ax1=True)
            self.alphaScrollBar.setDisabled(True)
            self.overlayColorButton.setDisabled(True)
            self.editOverlayColorAction.setDisabled(True)
            self.alphaScrollBar.hide()
            self.alphaScrollBar_label.hide()

    def criticalFluoChannelNotFound(self, fluo_ch, PosData):
        msg = QtGui.QMessageBox()
        warn_cca = msg.critical(
            self, 'Requested channel data not found!',
            f'The folder {PosData.rel_path} does not contain either one of the '
            'following files:\n\n'
            f'{PosData.basename}_{fluo_ch}.tif\n'
            f'{PosData.basename}_{fluo_ch}_aligned.npz\n\n'
            'Data loading aborted.',
            msg.Ok
        )

    def histLUT_cb(self, LUTitem):
        # Store the histogram levels that the user is manually changing
        # i.e. moving the gradient slider ticks up and down
        # Store them for all frames
        PosData = self.data[self.pos_i]
        isOverlayON = self.overlayButton.isChecked()
        min = self.hist.gradient.listTicks()[0][1]
        max = self.hist.gradient.listTicks()[1][1]
        if isOverlayON:
            for i in range(0, PosData.segmSizeT):
                histoLevels = PosData.allData_li[i]['histoLevels']
                histoLevels[PosData.manualContrastKey] = (min, max)
            if PosData.ol_data is not None:
                self.getOverlayImg(setImg=True)
        else:
            cellsKey = f'{self.user_ch_name}_overlayOFF'
            for i in range(0, PosData.segmSizeT):
                histoLevels = PosData.allData_li[i]['histoLevels']
                histoLevels[cellsKey] = (min, max)
            img = self.getImage()
            img = self.adjustBrightness(img, cellsKey)
            # self.img1.setImage(img)
            self.updateALLimg(image=img, only_ax1=True, updateFilters=True,
                              updateHistoLevels=False)

    def gradientContextMenuClicked(self, b=None):
        act = self.sender()
        self.hist.gradient.loadPreset(act.name)
        if act.name == 'grey':
            try:
                self.hist.sigLookupTableChanged.disconnect()
            except TypeError:
                pass
            self.hist.setImageItem(self.img1)
            self.hist.imageItem = lambda: None
            self.hist.sigLookupTableChanged.connect(self.histLUT_cb)
        else:
            try:
                self.hist.sigLookupTableChanged.disconnect()
            except TypeError:
                pass
            self.hist.setImageItem(self.img1)


    def adjustBrightness(self, img, key,
                         func=skimage.exposure.rescale_intensity):
        """
        Adjust contrast/brightness of the image selected in the histogram
        context menu using stored levels.
        The levels are stored in histLUT_cb function which is called when
        the user changes the gradient slider levels.
        Note that the gradient always returns values from 0 to 1 so we
        need to scale to the actual max min of the image.
        """
        PosData = self.data[self.pos_i]
        histoLevels = PosData.allData_li[PosData.frame_i]['histoLevels']
        rescaled_img = img
        for name in histoLevels:
            if name != key:
                continue

            minPerc, maxPerc = histoLevels[name]
            if minPerc == 0 and maxPerc == 1:
                rescaled_img = img
            else:
                imgRange = img.max()-img.min()
                min = img.min() + imgRange*minPerc
                max = img.min() + imgRange*maxPerc
                in_range = (min, max)
                rescaled_img = func(rescaled_img,
                                    in_range=in_range)
        return rescaled_img

    def getOlImg(self, key, normalizeIntens=True):
        PosData = self.data[self.pos_i]
        if PosData.SizeT > 1:
            ol_img = PosData.ol_data[key][PosData.frame_i].copy()
        else:
            ol_img = PosData.ol_data[key].copy()
        if normalizeIntens:
            ol_img = self.normalizeIntensities(ol_img)
        return ol_img

    def getOverlayImg(self, fluoData=None, setImg=True):
        PosData = self.data[self.pos_i]
        keys = list(PosData.ol_data.keys())

        # Cells channel (e.g. phase_contrast)
        cells_img = self.getImage(invert=False)

        img = self.adjustBrightness(cells_img, PosData.filename)
        self.ol_cells_img = img
        gray_img_rgb = gray2rgb(img)

        # First fluo channel
        ol_img = self.getOlImg(keys[0])
        if fluoData is not None:
            fluoImg, fluoKey = fluoData
            if fluoKey == keys[0]:
                ol_img = fluoImg

        ol_img = self.adjustBrightness(ol_img, keys[0])
        color = PosData.ol_colors[keys[0]]
        overlay = self._overlay(gray_img_rgb, ol_img, color)

        # Add additional overlays
        for key in keys[1:]:
            ol_img = self.getOlImg(keys[0])
            if fluoData is not None:
                fluoImg, fluoKey = fluoData
                if fluoKey == key:
                    ol_img = fluoImg
            self.adjustBrightness(ol_img, key)
            color = PosData.ol_colors[key]
            overlay = self._overlay(overlay, ol_img, color)

        if self.invertBwAction.isChecked():
            overlay = self.invertRGB(overlay)

        if setImg:
            self.img1.setImage(overlay)
        else:
            return overlay

    def invertRGB(self, rgb_img):
        # see https://forum.image.sc/t/invert-rgb-image-without-changing-colors/33571
        R = rgb_img[:, :, 0]
        G = rgb_img[:, :, 1]
        B = rgb_img[:, :, 2]
        GB_mean = np.mean([G, B], axis=0)
        RB_mean = np.mean([R, B], axis=0)
        RG_mean = np.mean([R, G], axis=0)
        rgb_img[:, :, 0] = 1-GB_mean
        rgb_img[:, :, 1] = 1-RB_mean
        rgb_img[:, :, 2] = 1-RG_mean
        return rgb_img

    def _overlay(self, gray_img_rgb, ol_img, color):
        ol_RGB_val = [v/255 for v in color]
        ol_alpha = self.alphaScrollBar.value()/self.alphaScrollBar.maximum()
        ol_norm_img = ol_img/ol_img.max()
        ol_img_rgb = gray2rgb(ol_norm_img)*ol_RGB_val
        overlay = (gray_img_rgb*(1.0 - ol_alpha)+ol_img_rgb*ol_alpha)
        overlay = overlay/overlay.max()
        overlay = (np.clip(overlay, 0, 1)*255).astype(np.uint8)
        return overlay

    def toggleOverlayColorButton(self, checked=True):
        self.mousePressColorButton(None)

    def toggleTextIDsColorButton(self, checked=True):
        self.textIDsColorButton.selectColor()

    def updateTextIDsColors(self, button):
        r, g, b = np.array(self.textIDsColorButton.color().getRgb()[:3])
        self.ax1_oldIDcolor = (r, g, b)
        self.ax1_S_oldCellColor = (int(r*0.9), int(g*0.9), int(b*0.9))
        self.ax1_G1cellColor = (int(r*0.8), int(g*0.8), int(b*0.8), 178)
        self.updateALLimg()

    def saveTextIDsColors(self, button):
        self.df_settings.at['textIDsColor', 'value'] = self.ax1_oldIDcolor
        self.df_settings.to_csv(self.settings_csv_path)

    def updateOlColors(self, button):
        rgb = self.overlayColorButton.color().getRgb()[:3]
        for PosData in self.data:
            PosData.ol_colors[self._key] = rgb
        self.df_settings.at['overlayColor',
                            'value'] = '-'.join([str(v) for v in rgb])
        self.df_settings.to_csv(self.settings_csv_path)
        self.updateOverlay(button)

    def updateOverlay(self, button):
        self.getOverlayImg(setImg=True)

    def getImage(self, frame_i=None, invert=True, normalizeIntens=True):
        PosData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = PosData.frame_i
        if PosData.SizeZ > 1:
            z = PosData.segmInfo_df.at[frame_i, 'z_slice_used_gui']
            zProjHow = PosData.segmInfo_df.at[frame_i, 'which_z_proj_gui']
            img = PosData.img_data[frame_i]
            if zProjHow == 'single z-slice':
                self.zSliceScrollBar.setSliderPosition(z)
                self.z_label.setText(f'z-slice  {z+1:02}/{PosData.SizeZ}')
                cells_img = img[z].copy()
            elif zProjHow == 'max z-projection':
                cells_img = img.max(axis=0).copy()
            elif zProjHow == 'mean z-projection':
                cells_img = img.mean(axis=0).copy()
            elif zProjHow == 'median z-proj.':
                cells_img = np.median(img, axis=0).copy()
        else:
            cells_img = PosData.img_data[frame_i].copy()
        if normalizeIntens:
            cells_img = self.normalizeIntensities(cells_img)
        if self.invertBwAction.isChecked() and invert:
            cells_img = -cells_img+cells_img.max()
        return cells_img

    def setImageImg2(self):
        PosData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == 'Segmentation and Tracking' or self.isSnapshot:
            self.addExistingDelROIs()
            allDelIDs, DelROIlab = self.getDelROIlab()
        else:
            DelROIlab = PosData.lab
        self.img2.setImage(DelROIlab)
        self.updateLookuptable()

    def setTempImg1Brush(self, ymin, ymax, xmin, xmax, mask):
        PosData = self.data[self.pos_i]
        brushIDmask = PosData.lab==PosData.brushID
        brushOverlay = self.imgRGB.copy()
        alpha = 0.3
        overlay = self.imgRGB[brushIDmask]*(1.0-alpha) + self.brushColor*alpha
        brushOverlay[brushIDmask] = overlay
        brushOverlay = (brushOverlay*255).astype(np.uint8)
        self.img1.setImage(brushOverlay)
        return overlay

    def warnEditingWithCca_df(self, editTxt):
        # Function used to warn that the user is editing in "Segmentation and
        # Tracking" mode a frame that contains cca annotations.
        # Ask whether to remove annotations from all future frames
        PosData = self.data[self.pos_i]
        if self.isSnapshot:
            if PosData.cca_df is not None:
                # For snapshot mode we reinitialize cca_df to base
                PosData.cca_df = self.getBaseCca_df()
                self.store_cca_df()
                self.updateALLimg()
            return

        acdc_df = PosData.allData_li[PosData.frame_i]['acdc_df']
        if acdc_df is None:
            return
        else:
            if 'cell_cycle_stage' not in acdc_df.columns:
                return
        msg = QtGui.QMessageBox()
        msg.setIcon(msg.Warning)
        msg.setWindowTitle('Edited frame!')
        msg.setText(
            'You modified a frame that has cell cycle annotations.\n\n'
            f'The change "{editTxt}" most likely makes the annotations wrong.\n\n'
            'If you really want to apply this change we reccommend to remove\n'
            'ALL cell cycle annotations from current frame to the end.\n\n'
            'What should I do?'
        )
        yes = QPushButton('Remove annotations from future frames (RECOMMENDED)')
        msg.addButton(yes, msg.YesRole)
        msg.addButton(QPushButton('Do not remove annotations'), msg.NoRole)
        msg.exec_()
        if msg.clickedButton() == yes:
            self.store_data()
            PosData.frame_i -= 1
            self.get_data()
            self.remove_future_cca_df(PosData.frame_i)
            self.next_frame()

    def addExistingDelROIs(self):
        PosData = self.data[self.pos_i]
        delROIs_info = PosData.allData_li[PosData.frame_i]['delROIs_info']
        for roi in delROIs_info['rois']:
            if roi in self.ax2.items:
                continue
            self.ax2.addItem(roi)

    def addNewItems(self):
        PosData = self.data[self.pos_i]
        # Add new Items if there are not enough
        HDDmaxID = max([PosData.segm_data.max() for PosData in self.data])
        STOREDmaxID = max([PosData.allData_li[i]['labels'].max()
                           for PosData in self.data
                           for i in range(0, PosData.segmSizeT)
                           if PosData.allData_li[i]['labels'] is not None])
        currentMaxID = PosData.lab.max()
        maxID = max([currentMaxID, STOREDmaxID, currentMaxID])
        idx = maxID-1
        if idx >= len(self.ax1_ContoursCurves):
            missingLen = idx-len(self.ax1_ContoursCurves)+10
            for i in range(missingLen):
                # Contours on ax1
                ContCurve = pg.PlotDataItem()
                self.ax1_ContoursCurves.append(ContCurve)
                self.ax1.addItem(ContCurve)

                # Bud mother line on ax1
                BudMothLine = pg.PlotDataItem()
                self.ax1_BudMothLines.append(BudMothLine)
                self.ax1.addItem(BudMothLine)

                # LabelItems on ax1
                ax1_IDlabel = pg.LabelItem()
                self.ax1_LabelItemsIDs.append(ax1_IDlabel)
                self.ax1.addItem(ax1_IDlabel)

                # LabelItems on ax2
                ax2_IDlabel = pg.LabelItem()
                self.ax2_LabelItemsIDs.append(ax2_IDlabel)
                self.ax2.addItem(ax2_IDlabel)

                # Contours on ax2
                ContCurve = pg.PlotDataItem()
                self.ax2_ContoursCurves.append(ContCurve)
                self.ax2.addItem(ContCurve)

    def updateHistogramItem(self, imageItem):
        """
        Function called every time the image changes (updateALLimg).
        Perform the following:

        1. Set the gradient slider tick positions
        2. Set the region max and min levels
        3. Plot the histogram
        """
        try:
            self.hist.sigLookupTableChanged.disconnect()
            connect = True
        except TypeError:
            connect = False
            pass
        PosData = self.data[self.pos_i]
        overlayOFF_key = f'{self.user_ch_name}_overlayOFF'
        isOverlayON = self.overlayButton.isChecked()
        histoLevels = PosData.allData_li[PosData.frame_i]['histoLevels']
        if PosData.manualContrastKey in histoLevels and isOverlayON:
            min, max = histoLevels[PosData.manualContrastKey]
        elif isOverlayON:
            min, max = 0, 1
        elif not isOverlayON and overlayOFF_key in histoLevels:
            min, max = histoLevels[overlayOFF_key]
        elif not isOverlayON and overlayOFF_key not in histoLevels:
            min, max = 0, 1

        minTick = self.hist.gradient.getTick(0)
        maxTick = self.hist.gradient.getTick(1)
        self.hist.gradient.setTickValue(minTick, min)
        self.hist.gradient.setTickValue(maxTick, max)
        self.hist.setLevels(min=imageItem.image.min(),
                            max=imageItem.image.max())
        h = imageItem.getHistogram()
        self.hist.plot.setData(*h)
        if connect:
            self.hist.sigLookupTableChanged.connect(self.histLUT_cb)

    def updateFramePosLabel(self):
        if self.isSnapshot:
            PosData = self.data[self.pos_i]
            self.t_label.setText(
                     f'Pos. n. {self.pos_i+1}/{self.num_pos} '
                     f'({PosData.pos_foldername})')
        else:
            PosData = self.data[0]
            self.t_label.setText(
                     f'frame n. {PosData.frame_i+1}/{PosData.segmSizeT}')

    def updateFilters(self, updateBlur=False, updateSharp=False,
                            updateEntropy=False, updateFilters=False):
        if self.gaussWin is not None and (updateBlur or updateFilters):
            self.gaussWin.apply()

        if self.edgeWin is not None and (updateSharp or updateFilters):
            self.edgeWin.apply()

        if self.entropyWin is not None and (updateEntropy or updateFilters):
            self.entropyWin.apply()


    def updateALLimg(self, image=None, never_visited=True,
                     only_ax1=False, updateBlur=False,
                     updateSharp=False, updateEntropy=False,
                     updateHistoLevels=True, updateFilters=False,
                     updateLabelItemColor=False):
        PosData = self.data[self.pos_i]

        if image is None:
            if self.overlayButton.isChecked():
                img = self.getOverlayImg(setImg=False)
            else:
                img = self.getImage()
                cellsKey = f'{self.user_ch_name}_overlayOFF'
                img = self.adjustBrightness(img, cellsKey)
        else:
            img = image

        self.img1.setImage(img)
        self.updateFilters(updateBlur, updateSharp, updateEntropy, updateFilters)

        if updateHistoLevels:
            self.updateHistogramItem(self.img1)

        if self.slideshowWin is not None:
            self.slideshowWin.frame_i = PosData.frame_i
            self.slideshowWin.update_img()

        if only_ax1:
            return

        lab = PosData.lab

        self.addNewItems()
        self.clearAllItems()

        self.setImageImg2()
        self.update_rp()

        self.checkIDs_LostNew()

        self.computingContoursTimes = []
        self.drawingLabelsTimes = []
        self.drawingContoursTimes = []
        # Annotate ID and draw contours
        for i, obj in enumerate(PosData.rp):
            updateColor=True if updateLabelItemColor and i==0 else False
            self.drawID_and_Contour(obj, updateColor=updateColor)


        # print('------------------------------------')
        # print(f'Drawing labels = {np.sum(self.drawingLabelsTimes):.3f} s')
        # print(f'Computing contours = {np.sum(self.computingContoursTimes):.3f} s')
        # print(f'Drawing contours = {np.sum(self.drawingContoursTimes):.3f} s')

        # Update annotated IDs (e.g. dead cells)
        self.update_rp_metadata(draw=True)

        self.highlightLostNew()
        self.checkIDsMultiContour()

        if self.ccaTableWin is not None:
            self.ccaTableWin.updateTable(PosData.cca_df)

    def startBlinking(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.blinkModeComboBox)
        self.timer.start(100)

    def blinkModeComboBox(self):
        if self.flag:
            self.modeComboBox.setStyleSheet('background-color: orange')
        else:
            self.modeComboBox.setStyleSheet('background-color: none')
        self.flag = not self.flag
        self.countBlinks += 1
        if self.countBlinks > 10:
            self.timer.stop()
            self.countBlinks = 0
            self.modeComboBox.setStyleSheet('background-color: none')

    def highlightNewIDs_ccaFailed(self):
        PosData = self.data[self.pos_i]
        for obj in PosData.rp:
            if obj.label in PosData.new_IDs:
                # self.ax2_setTextID(obj, 'Draw IDs and contours')
                self.ax1_setTextID(obj, 'Draw IDs and contours')
                cont = self.getObjContours(obj)
                curveID = self.ax1_ContoursCurves[obj.label-1]
                curveID.setData(cont[:,0], cont[:,1], pen=self.tempNewIDs_cpen)

    def highlightLostNew(self):
        PosData = self.data[self.pos_i]
        how = self.drawIDsContComboBox.currentText()
        IDs_and_cont = how == 'Draw IDs and contours'
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'

        for ax2ContCurve in self.ax2_ContoursCurves:
            if ax2ContCurve.getData()[0] is not None:
                ax2ContCurve.setData([], [])

        if PosData.frame_i == 0:
            return

        if IDs_and_cont or onlyCont or ccaInfo_and_cont:
            for obj in PosData.rp:
                ID = obj.label
                if ID in PosData.new_IDs:
                    ContCurve = self.ax2_ContoursCurves[ID-1]
                    cont = self.getObjContours(obj)
                    ContCurve.setData(cont[:,0], cont[:,1],
                                      pen=self.newIDs_cpen)

        if PosData.lost_IDs:
            # Get the rp from previous frame
            rp = PosData.allData_li[PosData.frame_i-1]['regionprops']
            for obj in rp:
                self.highlightLost_obj(obj)

    def highlightLost_obj(self, obj):
        PosData = self.data[self.pos_i]
        how = self.drawIDsContComboBox.currentText()
        IDs_and_cont = how == 'Draw IDs and contours'
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'
        ID = obj.label
        if ID in PosData.lost_IDs:
            ContCurve = self.ax1_ContoursCurves[ID-1]
            if IDs_and_cont or onlyCont or ccaInfo_and_cont:
                cont = self.getObjContours(obj)
                ContCurve.setData(cont[:,0], cont[:,1],
                                  pen=self.lostIDs_cpen)
            LabelItemID = self.ax1_LabelItemsIDs[ID-1]
            txt = f'{obj.label}?'
            LabelItemID.setText(txt, color=self.lostIDs_qMcolor)
            # Center LabelItem at centroid
            y, x = obj.centroid
            w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
            LabelItemID.setPos(x-w/2, y-h/2)

    def checkIDs_LostNew(self):
        PosData = self.data[self.pos_i]
        if PosData.frame_i == 0:
            PosData.lost_IDs = []
            PosData.new_IDs = []
            PosData.old_IDs = []
            PosData.IDs = [obj.label for obj in PosData.rp]
            PosData.multiContIDs = set()
            self.titleLabel.setText('Looking good!', color='w')
            return

        prev_rp = PosData.allData_li[PosData.frame_i-1]['regionprops']
        if prev_rp is None:
            return

        prev_IDs = [obj.label for obj in prev_rp]
        curr_IDs = [obj.label for obj in PosData.rp]
        lost_IDs = [ID for ID in prev_IDs if ID not in curr_IDs]
        new_IDs = [ID for ID in curr_IDs if ID not in prev_IDs]
        PosData.lost_IDs = lost_IDs
        PosData.new_IDs = new_IDs
        PosData.old_IDs = prev_IDs
        PosData.IDs = curr_IDs
        warn_txt = ''
        htmlTxt = ''
        if lost_IDs:
            lost_IDs_format = lost_IDs.copy()
            if len(lost_IDs) + len(new_IDs) > 20 and len(lost_IDs)>10:
                del lost_IDs_format[5:-5]
                lost_IDs_format.insert(5, "...")
                lost_IDs_format = f"[{', '.join(map(str, lost_IDs_format))}]"
            warn_txt = f'IDs lost in current frame: {lost_IDs_format}'
            color = 'red'
            htmlTxt = (
                f'<font color="red">{warn_txt}</font>'
            )
        if PosData.multiContIDs:
            warn_txt = f'IDs with multiple contours: {PosData.multiContIDs}'
            color = 'red'
            htmlTxt = (
                f'{htmlTxt}, <font color="red">{warn_txt}</font>'
            )
        if new_IDs:
            new_IDs_format = new_IDs.copy()
            if len(lost_IDs) + len(new_IDs) > 20 and len(new_IDs)>10:
                del new_IDs_format[5:-5]
                new_IDs_format.insert(5, "...")
                new_IDs_format = f"[{', '.join(map(str, new_IDs_format))}]"
            warn_txt = f'New IDs in current frame: {new_IDs_format}'
            color = 'r'
            htmlTxt = (
                f'{htmlTxt}, <font color="green">{warn_txt}</font>'
            )
        if not warn_txt:
            warn_txt = 'Looking good!'
            color = 'w'
            htmlTxt = (
                f'<font color="white">{warn_txt}</font>'
            )
        self.titleLabel.setText(htmlTxt)
        PosData.multiContIDs = set()

    def separateByLabelling(self, lab, rp, maxID=None):
        """
        Label each single object in PosData.lab and if the result is more than
        one object then we insert the separated object into PosData.lab
        """
        setRp = False
        for obj in rp:
            lab_obj = skimage.measure.label(obj.image)
            rp_lab_obj = skimage.measure.regionprops(lab_obj)
            if len(rp_lab_obj)>1:
                if maxID is None:
                    lab_obj += lab.max()
                else:
                    lab_obj += maxID
                lab[obj.slice][obj.image] = lab_obj[obj.image]
                setRp = True
        return setRp

    def tracking(self, onlyIDs=[], enforce=False, DoManualEdit=True,
                 storeUndo=False, prev_lab=None, prev_rp=None,
                 return_lab=False):
        try:
            PosData = self.data[self.pos_i]
            mode = str(self.modeComboBox.currentText())
            skipTracking = (
                PosData.frame_i == 0 or mode.find('Tracking') == -1
                or self.isSnapshot
            )
            if skipTracking:
                self.checkIDs_LostNew()
                return

            # Disable tracking for already visited frames
            if PosData.frame_i+1 < len(PosData.allData_li):
                if PosData.allData_li[PosData.frame_i+1]['labels'] is not None:
                    self.disableTrackingCheckBox.setChecked(True)
            else:
                self.disableTrackingCheckBox.setChecked(False)

            """
            Track only frames that were NEVER visited or the user
            specifically requested to track:
                - Never visited --> NOT self.disableTrackingCheckBox.isChecked()
                - User requested --> PosData.isAltDown
                                 --> clicked on repeat tracking (enforce=True)
            """

            if enforce or self.UserEnforced_Tracking:
                # Tracking enforced by the user
                do_tracking = True
            elif self.UserEnforced_DisabledTracking:
                # Tracking specifically DISABLED by the user
                do_tracking = False
            elif self.disableTrackingCheckBox.isChecked():
                # User did not choose what to do --> tracking disabled for
                # visited frames and enabled for never visited frames
                do_tracking = False
            else:
                do_tracking = True

            if not do_tracking:
                self.disableTrackingCheckBox.setChecked(True)
                # print('-------------')
                # print(f'Frame {PosData.frame_i+1} NOT tracked')
                # print('-------------')
                self.checkIDs_LostNew()
                return

            """Tracking starts here"""
            self.disableTrackingCheckBox.setChecked(False)

            if storeUndo:
                # Store undo state before modifying stuff
                self.storeUndoRedoStates(False)

            # First separate by labelling
            setRp = self.separateByLabelling(PosData.lab, PosData.rp)
            if setRp:
                self.update_rp()

            if prev_lab is None:
                prev_lab = PosData.allData_li[PosData.frame_i-1]['labels']
            if prev_rp is None:
                prev_rp = PosData.allData_li[PosData.frame_i-1]['regionprops']
            IDs_prev = []
            IDs_curr_untracked = [obj.label for obj in PosData.rp]
            IoA_matrix = np.zeros((len(PosData.rp), len(prev_rp)))



            # For each ID in previous frame get IoA with all current IDs
            # Rows: IDs in current frame, columns: IDs in previous frame
            for j, obj_prev in enumerate(prev_rp):
                ID_prev = obj_prev.label
                A_IDprev = obj_prev.area
                IDs_prev.append(ID_prev)
                mask_ID_prev = prev_lab==ID_prev
                intersect_IDs, intersects = np.unique(PosData.lab[mask_ID_prev],
                                                      return_counts=True)
                for intersect_ID, I in zip(intersect_IDs, intersects):
                    if intersect_ID != 0:
                        i = IDs_curr_untracked.index(intersect_ID)
                        IoA = I/A_IDprev
                        IoA_matrix[i, j] = IoA



            # Determine max IoA between IDs and assign tracked ID if IoA > 0.4
            max_IoA_col_idx = IoA_matrix.argmax(axis=1)
            unique_col_idx, counts = np.unique(max_IoA_col_idx, return_counts=True)
            counts_dict = dict(zip(unique_col_idx, counts))
            tracked_IDs = []
            old_IDs = []
            for i, j in enumerate(max_IoA_col_idx):
                max_IoU = IoA_matrix[i,j]
                count = counts_dict[j]
                if max_IoU > 0.4:
                    tracked_ID = IDs_prev[j]
                    if count == 1:
                        old_ID = IDs_curr_untracked[i]
                    elif count > 1:
                        old_ID_idx = IoA_matrix[:,j].argmax()
                        old_ID = IDs_curr_untracked[old_ID_idx]
                    tracked_IDs.append(tracked_ID)
                    old_IDs.append(old_ID)


            # print(IoA_matrix)

            # Replace untracked IDs with tracked IDs and new IDs with increasing num
            new_untracked_IDs = [ID for ID in IDs_curr_untracked if ID not in old_IDs]
            tracked_lab = PosData.lab
            new_tracked_IDs_2 = []
            if new_untracked_IDs:
                # Relabel new untracked IDs with big number to make sure they are unique
                allIDs = IDs_curr_untracked.copy()
                allIDs.extend(tracked_IDs)
                max_ID = max(allIDs)
                new_tracked_IDs = [max_ID*(i+2) for i in range(len(new_untracked_IDs))]
                tracked_lab = self.np_replace_values(tracked_lab, new_untracked_IDs,
                                                     new_tracked_IDs)
                # print('New objects that get a new big ID: ', new_untracked_IDs)
                # print('New big IDs for the new objects: ', new_tracked_IDs)
            if tracked_IDs:
                # Relabel old IDs with respective tracked IDs
                tracked_lab = self.np_replace_values(tracked_lab, old_IDs, tracked_IDs)
                # print('Old IDs to be tracked: ', old_IDs)
                # print('New IDs replacing old IDs: ', tracked_IDs)
            if new_untracked_IDs:
                # Relabel new untracked IDs sequentially
                max_ID = max(IDs_prev)
                new_tracked_IDs_2 = [max_ID+i+1 for i in range(len(new_untracked_IDs))]
                tracked_lab = self.np_replace_values(tracked_lab, new_tracked_IDs,
                                                     new_tracked_IDs_2)
                # print('New big IDs for the new objects: ', new_tracked_IDs)
                # print('New increasing IDs for the previously big IDs: ', new_tracked_IDs_2)

            if DoManualEdit:
                # Correct tracking with manually changed IDs
                rp = skimage.measure.regionprops(tracked_lab)
                IDs = [obj.label for obj in rp]
                self.ManuallyEditTracking(tracked_lab, IDs)
        except ValueError:
            tracked_lab = PosData.lab
            pass


        # Update labels, regionprops and determine new and lost IDs
        PosData.lab = tracked_lab
        self.update_rp()
        self.checkIDs_LostNew()

    def ManuallyEditTracking(self, tracked_lab, allIDs):
        PosData = self.data[self.pos_i]
        # Correct tracking with manually changed IDs
        for y, x, new_ID in PosData.editID_info:
            old_ID = tracked_lab[y, x]
            if new_ID in allIDs:
                tempID = tracked_lab.max() + 1
                tracked_lab[tracked_lab == old_ID] = tempID
                tracked_lab[tracked_lab == new_ID] = old_ID
                tracked_lab[tracked_lab == tempID] = new_ID
            else:
                tracked_lab[tracked_lab == old_ID] = new_ID

    def np_replace_values(self, arr, old_values, tracked_values):
        # See method_jdehesa https://stackoverflow.com/questions/45735230/how-to-replace-a-list-of-values-in-a-numpy-array
        old_values = np.asarray(old_values)
        tracked_values = np.asarray(tracked_values)
        n_min, n_max = arr.min(), arr.max()
        replacer = np.arange(n_min, n_max + 1)
        # Mask replacements out of range
        mask = (old_values >= n_min) & (old_values <= n_max)
        replacer[old_values[mask] - n_min] = tracked_values[mask]
        arr = replacer[arr - n_min]
        return arr

    def undo_changes_future_frames(self):
        PosData = self.data[self.pos_i]
        PosData.last_tracked_i = PosData.frame_i
        for i in range(PosData.frame_i+1, PosData.segmSizeT):
            if PosData.allData_li[i]['labels'] is None:
                break

            PosData.allData_li[i] = {
                 'regionprops': [],
                 'labels': None,
                 'acdc_df': None,
                 'delROIs_info': {'rois': [], 'delMasks': [], 'delIDsROI': []},
                 'histoLevels': {}
            }
        self.setFramesScrollbarMaximum()

    def removeAllItems(self):
        self.ax1.clear()
        self.ax2.clear()
        try:
            self.chNamesQActionGroup.removeAction(self.userChNameAction)
        except Exception as e:
            pass
        try:
            PosData = self.data[self.pos_i]
            for action in PosData.fluoDataChNameActions:
                self.chNamesQActionGroup.removeAction(action)
        except Exception as e:
            pass
        try:
            self.overlayButton.setChecked(False)
        except Exception as e:
            pass

    def create_chNamesQActionGroup(self, user_ch_name):
        # LUT histogram channel name context menu actions
        self.chNamesQActionGroup = QActionGroup(self)
        self.userChNameAction = QAction(self)
        self.userChNameAction.setCheckable(True)
        self.userChNameAction.setText(user_ch_name)
        self.userChNameAction.setChecked(True)
        self.chNamesQActionGroup.addAction(self.userChNameAction)
        self.chNamesQActionGroup.triggered.connect(self.setManualContrastKey)

    def setManualContrastKey(self, action):
        PosData = self.data[self.pos_i]
        try:
            keys = list(PosData.ol_data.keys())
        except Exception as e:
            keys = []
        keys.append(PosData.filename)
        checkedText = action.text()
        for key in keys:
            if key.find(checkedText) != -1:
                break
        PosData.manualContrastKey = key
        # self.updateHistogramItem()

    def restoreDefaultColors(self):
        try:
            color = self.defaultToolBarButtonColor
            self.overlayButton.setStyleSheet(f'background-color: {color}')
        except AttributeError:
            # traceback.print_exc()
            pass

    # Slots
    def newFile(self):
        pass

    def openFile(self, checked=False, file_path=None):
        """
        Function used for loading an image file directly.
        """
        if file_path is None:
            self.getMostRecentPath()
            file_path = QFileDialog.getOpenFileName(
                self, 'Select image file', self.MostRecentPath,
                "Images/Videos (*.png *.tif *.tiff *.jpg *.jpeg *.mov *.avi *.mp4)"
                ";;All Files (*)")[0]
            if file_path == '':
                return
        dirpath = os.path.dirname(file_path)
        dirname = os.path.basename(dirpath)
        if dirname != 'Images':
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            acdc_folder = f'{timestamp}_acdc'
            exp_path = os.path.join(dirpath, acdc_folder, 'Images')
            os.makedirs(exp_path)
        else:
            exp_path = dirpath

        filename, ext = os.path.splitext(os.path.basename(file_path))
        if ext == '.tif':
            self.openFolder(exp_path=exp_path, isImageFile=True)
        else:
            print('Copying file to .tif format...')
            data = load.loadData(file_path, '')
            data.loadImgData()
            img = data.img_data
            if img.ndim == 3 and (img.shape[-1] == 3 or img.shape[-1] == 4):
                print('Converting RGB image to grayscale...')
                data.img_data = skimage.color.rgb2gray(data.img_data)
                data.img_data = skimage.img_as_ubyte(data.img_data)
            tif_path = os.path.join(exp_path, f'{filename}.tif')
            if data.img_data.ndim == 3:
                SizeT = data.img_data.shape[0]
                SizeZ = 1
            elif data.img_data.ndim == 4:
                SizeT = data.img_data.shape[0]
                SizeZ = data.img_data.shape[1]
            else:
                SizeT = 1
                SizeZ = 1
            is_imageJ_dtype = (
                data.img_data.dtype == np.uint8
                or data.img_data.dtype == np.uint16
                or data.img_data.dtype == np.float32
            )
            if not is_imageJ_dtype:
                data.img_data = skimage.img_as_ubyte(data.img_data)

            myutils.imagej_tiffwriter(
                tif_path, data.img_data, {}, SizeT, SizeZ
            )
            self.openFolder(exp_path=exp_path, isImageFile=True)

    def criticalNoTifFound(self, images_path):
        err_title = f'No .tif files found in folder.'
        err_msg = (
            f'The folder "{images_path}" does not contain .tif files.\n\n'
            'Only .tif files can be loaded with "Open Folder" button.\n\n'
            'Try with "File --> Open image/video file..." and directly select '
            'the file you want to load.'
        )
        msg = QtGui.QMessageBox()
        msg.critical(self, err_title, err_msg, msg.Ok)
        return

    def openFolder(self, checked=False, exp_path=None, isImageFile=False):
        try:
            # Remove all items from a previous session if open is pressed again
            self.removeAllItems()
            self.gui_addPlotItems()
            self.setUncheckedAllButtons()
            self.restoreDefaultColors()
            self.curvToolButton.setChecked(False)

            self.openAction.setEnabled(False)
            self.modeComboBox.setCurrentIndex(0)

            if self.slideshowWin is not None:
                self.slideshowWin.close()

            if self.ccaTableWin is not None:
                self.ccaTableWin.close()

            if exp_path is None:
                self.getMostRecentPath()
                exp_path = QFileDialog.getExistingDirectory(
                    self, 'Select experiment folder containing Position_n folders '
                          'or specific Position_n folder', self.MostRecentPath)

            if exp_path == '':
                self.openAction.setEnabled(True)
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                return

            self.addToRecentPaths(exp_path)

            if os.path.basename(exp_path).find('Position_') != -1:
                is_pos_folder = True
            else:
                is_pos_folder = False

            if os.path.basename(exp_path).find('Images') != -1:
                is_images_folder = True
            else:
                is_images_folder = False

            self.titleLabel.setText('Loading data...', color='w')
            self.setWindowTitle(f'Yeast_ACDC - GUI - "{exp_path}"')


            ch_name_selector = prompts.select_channel_name(
                which_channel='segm', allow_abort=False
            )

            if not is_pos_folder and not is_images_folder and not isImageFile:
                select_folder = load.select_exp_folder()
                values = select_folder.get_values_segmGUI(exp_path)
                if not values:
                    txt = (
                        'The selected folder:\n\n '
                        f'{exp_path}\n\n'
                        'is not a valid folder. '
                        'Select a folder that contains the Position_n folders'
                    )
                    msg = QtGui.QMessageBox()
                    msg.critical(
                        self, 'Incompatible folder', txt, msg.Ok
                    )
                    self.titleLabel.setText(
                        'File --> Open or Open recent to start the process',
                        color='w')
                    self.openAction.setEnabled(True)
                    return

                select_folder.QtPrompt(self, values, allow_abort=False)
                if select_folder.was_aborted:
                    self.titleLabel.setText(
                        'File --> Open or Open recent to start the process',
                        color='w')
                    self.openAction.setEnabled(True)
                    return

                images_paths = []
                for pos in select_folder.selected_pos:
                    images_paths.append(os.path.join(exp_path, pos, 'Images'))

                if select_folder.was_aborted:
                    self.titleLabel.setText(
                        'File --> Open or Open recent to start the process',
                        color='w')
                    self.openAction.setEnabled(True)
                    return

            elif is_pos_folder:
                pos_foldername = os.path.basename(exp_path)
                exp_path = os.path.dirname(exp_path)
                images_paths = [os.path.join(exp_path, pos_foldername, 'Images')]

            elif is_images_folder:
                images_paths = [exp_path]

            elif isImageFile:
                images_paths = [exp_path]

            self.images_paths = images_paths

            # Get info from first position selected
            images_path = self.images_paths[0]
            filenames = os.listdir(images_path)
            if ch_name_selector.is_first_call:
                ch_names, basenameNotFound = (
                    ch_name_selector.get_available_channels(filenames)
                )
                self.ch_names = ch_names
                if len(ch_names) > 1:
                    CbLabel='Select channel name to segment: '
                    ch_name_selector.QtPrompt(
                        self, ch_names, CbLabel=CbLabel)
                    if ch_name_selector.was_aborted:
                        self.titleLabel.setText(
                            'File --> Open or Open recent to start the process',
                            color='w')
                        self.openAction.setEnabled(True)
                        return
                    user_ch_name = ch_name_selector.channel_name
                elif len(ch_names) == 1:
                    user_ch_name = ch_names[0]
                elif not ch_names:
                    self.titleLabel.setText(
                        'File --> Open or Open recent to start the process',
                        color='w')
                    self.openAction.setEnabled(True)
                    self.criticalNoTifFound(images_path)
                    return

            print(user_ch_name)
            print(images_paths)

            user_ch_file_paths = []
            img_path = None
            for images_path in self.images_paths:
                img_aligned_found = False
                for filename in os.listdir(images_path):
                    img_path = os.path.join(images_path, filename)
                    if filename.find(f'_phc_aligned.npy') != -1:
                        new_filename = filename.replace('phc_aligned.npy',
                                                f'{user_ch_name}_aligned.npy')
                        dst = f'{images_path}/{new_filename}'
                        if os.path.exists(dst):
                            os.remove(img_path)
                        else:
                            os.rename(img_path, dst)
                        filename = new_filename
                    if filename.find(f'{user_ch_name}_aligned.np') != -1:
                        aligned_path = img_path
                        img_aligned_found = True
                    elif filename.find(f'{user_ch_name}.tif') != -1:
                        tif_path = img_path
                if not img_aligned_found:
                    err_msg = ('Aligned frames file for channel '
                               f'{user_ch_name} not found. '
                               'Loading tifs files.')
                    self.titleLabel.setText(err_msg)
                    img_path = tif_path
                else:
                    img_path = aligned_path
                    # raise FileNotFoundError(err_msg)

                user_ch_file_paths.append(img_path)

            if img_path is None:
                self.titleLabel.setText(
                    'File --> Open or Open recent to start the process',
                    color='w')
                self.openAction.setEnabled(True)
                self.criticalImgPathNotFound()
                return

            print(f'Loading {img_path}...')


            self.user_ch_name = user_ch_name

            self.initGlobalAttr()

            self.num_pos = len(user_ch_file_paths)
            proceed = self.init_data(user_ch_file_paths, user_ch_name)
            if not proceed:
                self.openAction.setEnabled(True)
                return

            self.create_chNamesQActionGroup(user_ch_name)

            # Connect events at the end of loading data process
            self.gui_connectGraphicsEvents()
            if not self.isEditActionsConnected:
                self.gui_connectEditActions()

            self.titleLabel.setText(
                'Data successfully loaded. Right/Left arrow to navigate frames',
                color='w')

            self.setFramesSnapshotMode()
            self.appendPathWindowTitle(user_ch_file_paths)
            self.updateALLimg(updateLabelItemColor=False)
            self.updateScrollbars()
            self.fontSizeAction.setChecked(True)
            self.openAction.setEnabled(True)
            self.editTextIDsColorAction.setDisabled(False)
            self.imgPropertiesAction.setEnabled(True)
        except Exception as e:
            print('')
            print('====================================')
            traceback.print_exc()
            print('====================================')
            print('')
            err_msg = 'Error occured. See terminal/console for details'
            self.titleLabel.setText(
                'Error occured. See terminal/console for details',
                color='r')
            msg = QtGui.QMessageBox()
            msg.critical(
                self, 'Error!', err_msg, msg.Ok
            )
            self.openAction.setEnabled(True)

    def appendPathWindowTitle(self, user_ch_file_paths):
        if self.isSnapshot:
            return

        pos_path = os.path.dirname(os.path.dirname(user_ch_file_paths[0]))
        self.setWindowTitle(f'Yeast_ACDC - GUI - "{pos_path}"')

    def initFluoData(self):
        msg = QtGui.QMessageBox()
        load_fluo = msg.question(
            self, 'Load fluorescent images?',
            'Do you also want to load fluorescent images? You can load as '
            'many channels as you want.\n\n'
            'If you load fluorescent images then the software will '
            'calculate metrics for each loaded fluorescent channel '
            'such as min, max, mean, quantiles, etc. '
            'for each segmented object.\n\n'
            'NOTE: You can always load them later with '
            'File --> Load fluorescent images',
            msg.Yes | msg.No
        )
        if load_fluo == msg.Yes:
            self.loadFluo_cb(None)

    def getPathFromChName(self, chName, PosData):
        aligned_files = [f for f in os.listdir(PosData.images_path)
                         if f.find(f'{chName}_aligned.npz')!=-1]
        if aligned_files:
            filename = aligned_files[0]
        else:
            tif_files = [f for f in os.listdir(PosData.images_path)
                         if f.find(f'{chName}.tif')!=-1]
            if not tif_files:
                self.criticalFluoChannelNotFound(chName, PosData)
                self.app.restoreOverrideCursor()
                return None, None
            filename = tif_files[0]
        fluo_path = os.path.join(PosData.images_path, filename)
        filename, _ = os.path.splitext(filename)
        return fluo_path, filename

    def loadFluo_cb(self, event):
        ch_names = [ch for ch in self.ch_names if ch != self.user_ch_name]
        selectFluo = apps.QDialogListbox(
            'Select channel',
            'Select channel names to load:\n',
            ch_names, multiSelection=True, parent=self
        )
        selectFluo.exec_()

        if selectFluo.cancel:
            return

        fluo_channels = selectFluo.selectedItemsText

        self.app.setOverrideCursor(Qt.WaitCursor)
        for PosData in self.data:
            PosData.ol_data = None
            for fluo_ch in fluo_channels:
                fluo_path, filename = self.getPathFromChName(fluo_ch, PosData)
                if fluo_path is None:
                    self.criticalFluoChannelNotFound(fluo_ch, PosData)
                    self.app.restoreOverrideCursor()
                    return
                fluo_data, ol_data_2D = self.load_fluo_data(fluo_path)
                if fluo_data is None:
                    self.app.restoreOverrideCursor()
                    return
                PosData.loadedFluoChannels.add(fluo_ch)
                PosData.fluo_data_dict[filename] = fluo_data
                PosData.ol_data_dict[filename] = ol_data_2D
        self.app.restoreOverrideCursor()
        self.overlayButton.setStyleSheet('background-color: #A7FAC7')

    def addToRecentPaths(self, exp_path):
        if not os.path.exists(exp_path):
            return
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            recentPaths = df['path'].to_list()
            if 'opened_last_on' in df.columns:
                openedOn = df['opened_last_on'].to_list()
            else:
                openedOn = [np.nan]*len(recentPaths)
            if exp_path in recentPaths:
                pop_idx = recentPaths.index(exp_path)
                recentPaths.pop(pop_idx)
                openedOn.pop(pop_idx)
            recentPaths.insert(0, exp_path)
            openedOn.insert(0, datetime.datetime.now())
            # Keep max 20 recent paths
            if len(recentPaths) > 20:
                recentPaths.pop(-1)
                openedOn.pop(-1)
        else:
            recentPaths = [exp_path]
            openedOn = [datetime.datetime.now()]
        df = pd.DataFrame({'path': recentPaths,
                           'opened_last_on': pd.Series(openedOn,
                                                       dtype='datetime64[ns]')})
        df.index.name = 'index'
        df.to_csv(recentPaths_path)

    def showInExplorer(self):
        PosData = self.data[self.pos_i]
        systems = {
            'nt': os.startfile,
            'posix': lambda foldername: os.system('xdg-open "%s"' % foldername),
            'os2': lambda foldername: os.system('open "%s"' % foldername)
             }

        systems.get(os.name, os.startfile)(PosData.images_path)

    def addMetrics_acdc_df(self, df, rp, frame_i, lab, PosData):
        # Add metrics that can be calculated at the end of the process
        # such as cell volume, cell area etc.
        PhysicalSizeY = PosData.PhysicalSizeY
        PhysicalSizeX = PosData.PhysicalSizeX

        # Calc volume
        vox_to_fl = PhysicalSizeY*(PhysicalSizeX**2)
        yx_pxl_to_um2 = PhysicalSizeY*PhysicalSizeX
        numCells = len(rp)
        IDs = [0]*numCells
        IDs_vol_vox = [0]*numCells
        IDs_area_pxl = [0]*numCells
        IDs_vol_fl = [0]*numCells
        IDs_area_um2 = [0]*numCells
        fluo_keys = list(PosData.fluo_data_dict.keys())
        numFluoChannels = len(fluo_keys)
        chNames = PosData.loadedChNames
        fluo_means = np.zeros((numCells, numFluoChannels))
        fluo_medians = np.zeros((numCells, numFluoChannels))
        fluo_mins = np.zeros((numCells, numFluoChannels))
        fluo_maxs = np.zeros((numCells, numFluoChannels))
        fluo_sums = np.zeros((numCells, numFluoChannels))
        fluo_q25s = np.zeros((numCells, numFluoChannels))
        fluo_q75s = np.zeros((numCells, numFluoChannels))
        fluo_q5s = np.zeros((numCells, numFluoChannels))
        fluo_q95s = np.zeros((numCells, numFluoChannels))
        fluo_amounts = np.zeros((numCells, numFluoChannels))
        fluo_amounts_bkgrVals = np.zeros((numCells, numFluoChannels))
        outCellsMask = lab==0
        for i, obj in enumerate(rp):
            IDs[i] = obj.label
            rotate_ID_img = skimage.transform.rotate(
                obj.image.astype(np.uint8), -(obj.orientation*180/np.pi),
                resize=True, order=3, preserve_range=True
            )
            radii = np.sum(rotate_ID_img, axis=1)/2
            vol_vox = np.sum(np.pi*(radii**2))
            IDs_vol_vox[i] = vol_vox
            IDs_area_pxl[i] = obj.area
            IDs_vol_fl[i] = vol_vox*vox_to_fl
            IDs_area_um2[i] = obj.area*yx_pxl_to_um2
            # Calc metrics for each fluo channel
            for j, key in enumerate(fluo_keys):
                fluo_data = PosData.fluo_data_dict[key][frame_i]
                if fluo_data.ndim > 2:
                    fluo_data = fluo_data.max(axis=0)

                fluo_data_ID = fluo_data[obj.slice][obj.image]
                backgrMask = np.logical_and(outCellsMask, fluo_data!=0)
                fluo_backgr = np.median(fluo_data[backgrMask])
                fluo_mean = fluo_data_ID.mean()
                fluo_amount = (fluo_mean-fluo_backgr)*obj.area

                dataPrep_bkgrVal = self.getDataPrepBkgrVal(
                                            PosData.bkgrValues_df,
                                            PosData.bkgrValues_chNames,
                                            frame_i, j)
                if dataPrep_bkgrVal is not None:
                    amount = (fluo_mean-dataPrep_bkgrVal)*obj.area
                    fluo_amounts_bkgrVals[i,j] = amount

                fluo_means[i,j] = fluo_mean
                fluo_medians[i,j] = np.median(fluo_data_ID)
                fluo_mins[i,j] = fluo_data_ID.min()
                fluo_maxs[i,j] = fluo_data_ID.max()
                fluo_sums[i,j] = fluo_data_ID.sum()
                fluo_q25s[i,j] = np.quantile(fluo_data_ID, q=0.25)
                fluo_q75s[i,j] = np.quantile(fluo_data_ID, q=0.75)
                fluo_q5s[i,j] = np.quantile(fluo_data_ID, q=0.05)
                fluo_q95s[i,j] = np.quantile(fluo_data_ID, q=0.95)
                fluo_amounts[i,j] = fluo_amount

        df['cell_area_pxl'] = pd.Series(data=IDs_area_pxl, index=IDs, dtype=float)
        df['cell_vol_vox'] = pd.Series(data=IDs_vol_vox, index=IDs, dtype=float)
        df['cell_area_um2'] = pd.Series(data=IDs_area_um2, index=IDs, dtype=float)
        df['cell_vol_fl'] = pd.Series(data=IDs_vol_fl, index=IDs, dtype=float)
        df[[f'{ch}_mean' for ch in chNames]] = pd.DataFrame(
                                                    data=fluo_means, index=IDs,
                                                    dtype=float)
        df[[f'{ch}_median' for ch in chNames]] = pd.DataFrame(
                                                    data=fluo_medians, index=IDs,
                                                    dtype=float)
        df[[f'{ch}_min' for ch in chNames]] = pd.DataFrame(
                                                    data=fluo_mins, index=IDs,
                                                    dtype=float)
        df[[f'{ch}_max' for ch in chNames]] = pd.DataFrame(
                                                    data=fluo_maxs, index=IDs,
                                                    dtype=float)
        df[[f'{ch}_sum' for ch in chNames]] = pd.DataFrame(
                                                    data=fluo_sums, index=IDs,
                                                    dtype=float)
        df[[f'{ch}_q25' for ch in chNames]] = pd.DataFrame(
                                                    data=fluo_q25s, index=IDs,
                                                    dtype=float)
        df[[f'{ch}_q75' for ch in chNames]] = pd.DataFrame(
                                                    data=fluo_q75s, index=IDs,
                                                    dtype=float)
        df[[f'{ch}_q05' for ch in chNames]] = pd.DataFrame(
                                                    data=fluo_q5s, index=IDs,
                                                    dtype=float)
        df[[f'{ch}_q95' for ch in chNames]] = pd.DataFrame(
                                                    data=fluo_q95s, index=IDs,
                                                    dtype=float)
        df[[f'{ch}_amount_autoBkgr' for ch in chNames]] = pd.DataFrame(
                                                    data=fluo_amounts,
                                                    index=IDs,
                                                    dtype=float)
        df[[f'{ch}_amount_dataPrepBkgr' for ch in chNames]] = pd.DataFrame(
                                                    data=fluo_amounts_bkgrVals,
                                                    index=IDs,
                                                    dtype=float)

        # Join with regionprops_table
        props = (
            'label',
            'bbox',
            'bbox_area',
            'eccentricity',
            'equivalent_diameter',
            'euler_number',
            'extent',
            'filled_area',
            'inertia_tensor_eigvals',
            'local_centroid',
            'major_axis_length',
            'minor_axis_length',
            'moments',
            'moments_central',
            'moments_hu',
            'moments_normalized',
            'orientation',
            'perimeter',
            'solidity'
        )
        rp_table = skimage.measure.regionprops_table(
            PosData.lab, properties=props
        )
        df_rp = pd.DataFrame(rp_table).set_index('label')

        df = df.join(df_rp)
        return df


    def getChNames(self, PosData):
        fluo_keys = list(PosData.fluo_data_dict.keys())

        loadedChNames = []
        for key in fluo_keys:
            chName = key[len(PosData.basename):]
            if chName.find('_aligned') != -1:
                idx = chName.find('_aligned')
                chName = f'gui_{chName[:idx]}'
            loadedChNames.append(chName)

        PosData.loadedChNames = loadedChNames


        if PosData.bkgrValues_df is None:
            PosData.bkgrValues_chNames = None
            return

        bkgrValues_chNames = []
        for fluoKey in fluo_keys:
            chName = [chName for chName in PosData.bkgrValues_chNames
                      if fluoKey.find(chName)!=-1][0]
            bkgrValues_chNames.append(chName)

        PosData.bkgrValues_chNames = bkgrValues_chNames


    def getDataPrepBkgrVal(self, bkgrValues_df, bkgrValues_chNames, frame_i, j):
        if bkgrValues_df is None:
            return None

        try:
            idx = (bkgrValues_chNames[j], frame_i)
            bkgr_median = bkgrValues_df.at[idx, 'bkgr_median']
        except Exception as e:
            return None

        return bkgr_median

    def askSaveLastVisitedCcaMode(self, p, PosData):
        current_frame_i = PosData.frame_i
        frame_i = 0
        last_tracked_i = 0
        for frame_i, data_dict in enumerate(PosData.allData_li):
            # Build segm_npy
            acdc_df = data_dict['acdc_df']
            if acdc_df is None:
                frame_i -= 1
                break
            if 'cell_cycle_stage' not in acdc_df.columns:
                frame_i -= 1
                break
        if frame_i>0:
            # Ask to save last visited frame or not
            txt = (
                f'Do you also want to save last visited frame {frame_i+1}?'
            )
            msg = QtGui.QMessageBox()
            save_current = msg.question(
                self, 'Save current frame?', txt,
                msg.Yes | msg.No | msg.Cancel
            )
            if PosData.frame_i != frame_i:
                # Get data from last frame if we are not on it
                PosData.frame_i = frame_i
                self.get_data()
            if save_current == msg.Yes:
                last_tracked_i = frame_i
                self.store_cca_df()
            elif save_current == msg.No:
                last_tracked_i = frame_i-1
                self.unstore_cca_df()
                current_frame_i = last_tracked_i
                PosData.frame_i = last_tracked_i
                self.get_data()
                self.updateALLimg()
            elif save_current == msg.Cancel:
                return None
        return last_tracked_i

    def askSaveLastVisitedSegmMode(self, p, PosData):
        current_frame_i = PosData.frame_i
        frame_i = 0
        last_tracked_i = 0
        for frame_i, data_dict in enumerate(PosData.allData_li):
            # Build segm_npy
            lab = data_dict['labels']
            if lab is None:
                frame_i -= 1
                break
        if frame_i>0:
            # Ask to save last visited frame or not
            txt = (
                f'Do you also want to save last visited frame {frame_i+1}?'
            )
            msg = QtGui.QMessageBox()
            save_current = msg.question(
                self, 'Save current frame?', txt,
                msg.Yes | msg.No | msg.Cancel
            )
            if PosData.frame_i != frame_i:
                # Go to that last frame if we are not on it
                PosData.frame_i = frame_i
                self.get_data()
            if save_current == msg.Yes:
                last_tracked_i = frame_i
                self.store_data()
            elif save_current == msg.No:
                last_tracked_i = frame_i-1
                self.unstore_data()
                current_frame_i = last_tracked_i
                PosData.frame_i = last_tracked_i
                self.get_data()
                self.updateALLimg()
            elif save_current == msg.Cancel:
                return None
        return last_tracked_i

    def saveFile(self):
        self.store_data()
        for p, PosData in enumerate(self.data):
            current_frame_i = PosData.frame_i
            mode = self.modeComboBox.currentText()
            if mode == 'Segmentation and Tracking' or mode == 'Viewer':
                last_tracked_i = self.askSaveLastVisitedSegmMode(p, PosData)
                if last_tracked_i is None:
                    return
            elif mode == 'Cell cycle analysis':
                last_tracked_i = self.askSaveLastVisitedCcaMode(p, PosData)
                if last_tracked_i is None:
                    return
            elif self.isSnapshot:
                last_tracked_i = 0

            self.app.setOverrideCursor(Qt.WaitCursor)
            if self.isSnapshot:
                self.store_data()
            try:
                segm_npz_path = PosData.segm_npz_path
                acdc_output_csv_path = PosData.acdc_output_csv_path
                last_tracked_i_path = PosData.last_tracked_i_path
                segm_npy = np.copy(PosData.segm_data)
                npz_delROIs_info = {}
                delROIs_info_path = PosData.delROIs_info_path
                acdc_df_li = [None]*PosData.segmSizeT

                # Add segmented channel data for calc metrics
                PosData.fluo_data_dict[PosData.filename] = PosData.img_data

                self.getChNames(PosData)

                # Create list of dataframes from acdc_df on HDD
                if PosData.acdc_df is not None:
                    for frame_i, df in PosData.acdc_df.groupby(level=0):
                        acdc_df_li[frame_i] = df.loc[frame_i]

                print(f'Saving {PosData.relPath}')
                pbar = tqdm(total=len(PosData.allData_li), unit=' frames', ncols=100)
                for frame_i, data_dict in enumerate(PosData.allData_li):
                    # Build segm_npy
                    lab = data_dict['labels']
                    PosData.lab = lab
                    if lab is not None:
                        if PosData.SizeT > 1:
                            segm_npy[frame_i] = lab
                        else:
                            segm_npy = lab
                    else:
                        frame_i -= 1
                        break

                    acdc_df = data_dict['acdc_df']

                    # Save del ROIs
                    delROIs_info = PosData.allData_li[frame_i]['delROIs_info']
                    rois = delROIs_info['rois']
                    n = len(rois)
                    delMasks = delROIs_info['delMasks']
                    delIDsROI = delROIs_info['delIDsROI']
                    _zip = zip(rois, delMasks, delIDsROI)
                    for r, (roi, delMask, delIDs) in enumerate(_zip):
                        npz_delROIs_info[f'{frame_i}_delMask_{r}_{n}'] = delMask
                        delIDsROI_arr = np.array(list(delIDs))
                        npz_delROIs_info[f'{frame_i}_delIDs_{r}_{n}'] = delIDsROI_arr
                        x0, y0 = [int(c) for c in roi.pos()]
                        w, h = [int(c) for c in roi.size()]
                        roi_arr = np.array([x0, y0, w, h], dtype=np.uint16)
                        npz_delROIs_info[f'{frame_i}_roi_{r}_{n}'] = roi_arr


                    # Build acdc_df and index it in each frame_i of acdc_df_li
                    if acdc_df is not None:
                        acdc_df = load.loadData.BooleansTo0s1s(
                                    acdc_df, inplace=False
                        )
                        rp = data_dict['regionprops']
                        try:
                            acdc_df = self.addMetrics_acdc_df(
                                        acdc_df, rp, frame_i, lab, PosData
                            )
                            acdc_df_li[frame_i] = acdc_df
                        except Exception as e:
                            print('')
                            print('====================================')
                            traceback.print_exc()
                            print('====================================')
                            print('')
                            print('Warning: calculating metrics failed see above...')
                            print('-----------------')
                            pass
                    pbar.update()

                PosData.fluo_data_dict.pop(PosData.filename)
                pbar.update(pbar.total-pbar.n)
                pbar.close()

                # Remove None and concat dataframe
                keys = []
                df_li = []
                for i, df in enumerate(acdc_df_li):
                    if df is not None:
                        df_li.append(df)
                        keys.append((i, PosData.TimeIncrement*i))

                print('Almost done...')
                try:
                    np.savez_compressed(delROIs_info_path, **npz_delROIs_info)
                except Exception as e:
                    print('')
                    print('====================================')
                    traceback.print_exc()
                    print('====================================')
                    print('')

                if PosData.segmInfo_df is not None:
                    try:
                        PosData.segmInfo_df.to_csv(PosData.segmInfo_df_csv_path)
                    except PermissionError:
                        msg = QtGui.QMessageBox()
                        warn_cca = msg.critical(
                            self, 'Permission denied',
                            f'The below file is open in another app (Excel maybe?).\n\n'
                            f'{PosData.segmInfo_df_csv_path}\n\n'
                            'Close file and then press "Ok".',
                            msg.Ok
                        )
                        PosData.segmInfo_df.to_csv(PosData.segmInfo_df_csv_path)

                try:
                    all_frames_metadata_df = pd.concat(
                        df_li, keys=keys,
                        names=['frame_i', 'time_seconds', 'Cell_ID']
                    )

                    # Save segmentation metadata
                    all_frames_metadata_df.to_csv(acdc_output_csv_path)
                    PosData.acdc_df = all_frames_metadata_df
                except PermissionError:
                    msg = QtGui.QMessageBox()
                    warn_cca = msg.critical(
                        self, 'Permission denied',
                        f'The below file is open in another app (Excel maybe?).\n\n'
                        f'{acdc_output_csv_path}\n\n'
                        'Close file and then press "Ok".',
                        msg.Ok
                    )
                    all_frames_metadata_df = pd.concat(
                        df_li, keys=keys, names=['frame_i', 'Cell_ID']
                    )

                    # Save segmentation metadata
                    all_frames_metadata_df.to_csv(acdc_output_csv_path)
                    PosData.acdc_df = all_frames_metadata_df
                except Exception as e:
                    print('')
                    print('====================================')
                    traceback.print_exc()
                    print('====================================')
                    print('')
                    pass

                # Save segmentation file
                np.savez_compressed(segm_npz_path, segm_npy)
                PosData.segm_data = segm_npy

                with open(last_tracked_i_path, 'w+') as txt:
                    txt.write(str(frame_i))

                PosData.last_tracked_i = last_tracked_i

                # Go back to current frame
                PosData.frame_i = current_frame_i
                self.get_data()

                print('--------------')
                if mode == 'Segmentation and Tracking' or mode == 'Viewer':
                    print(f'Saved data until frame number {frame_i+1}')
                elif mode == 'Cell cycle analysis':
                    print(
                        'Saved cell cycle annotations until frame '
                        f'number {last_tracked_i+1}')
                elif self.isSnapshot:
                    print(f'Saved all {len(self.data)} Positions!')
                print('--------------')
            except Exception as e:
                print('')
                print('====================================')
                traceback.print_exc()
                print('====================================')
                print('')
            finally:
                self.app.restoreOverrideCursor()

    def copyContent(self):
        pass

    def pasteContent(self):
        pass

    def cutContent(self):
        pass

    def helpContent(self):
        pass

    def about(self):
        pass

    def populateOpenRecent(self):
        # Step 0. Remove the old options from the menu
        self.openRecentMenu.clear()
        # Step 1. Read recent Paths
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            if 'opened_last_on' in df.columns:
                df = df.sort_values('opened_last_on', ascending=False)
            recentPaths = df['path'].to_list()
        else:
            recentPaths = []
        # Step 2. Dynamically create the actions
        actions = []
        for path in recentPaths:
            action = QAction(path, self)
            action.triggered.connect(partial(self.openRecentFile, path))
            actions.append(action)
        # Step 3. Add the actions to the menu
        self.openRecentMenu.addActions(actions)

    def getMostRecentPath(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            if 'opened_last_on' in df.columns:
                df = df.sort_values('opened_last_on', ascending=False)
            self.MostRecentPath = df.iloc[0]['path']
            if not isinstance(self.MostRecentPath, str):
                self.MostRecentPath = ''
        else:
            self.MostRecentPath = ''

    def openRecentFile(self, path):
        print(f'Opening recent folder: {path}')
        self.openFolder(exp_path=path)

    def closeEvent(self, event):
        self.saveWindowGeometry()
        if self.slideshowWin is not None:
            self.slideshowWin.close()
        if self.ccaTableWin is not None:
            self.ccaTableWin.close()
        if self.saveAction.isEnabled():
            msg = QtGui.QMessageBox()
            msg.closeEvent = self.saveMsgCloseEvent
            save = msg.question(
                self, 'Save?', 'Do you want to save?',
                msg.Yes | msg.No | msg.Cancel
            )
            if save == msg.Yes:
                self.saveFile()
                event.accept()
            elif save == msg.No:
                event.accept()
            else:
                event.ignore()

        if self.buttonToRestore is not None and event.isAccepted():
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()

    def saveWindowGeometry(self):
        left = self.geometry().left()
        top = self.geometry().top()
        width = self.geometry().width()
        height = self.geometry().height()
        try:
            screenName = '/'.join(self.screen().name().split('\\'))
        except AttributeError:
            screenName = 'None'
            print('WARNING: could not retrieve screen name.'
                  'Please update to PyQt5 version >= 5.14')
        self.df_settings.at['geometry_left', 'value'] = left
        self.df_settings.at['geometry_top', 'value'] = top
        self.df_settings.at['geometry_width', 'value'] = width
        self.df_settings.at['geometry_height', 'value'] = height
        self.df_settings.at['screenName', 'value'] = screenName
        isMaximised = self.windowState() == Qt.WindowMaximized
        self.df_settings.at['isMaximised', 'value'] = isMaximised
        self.df_settings.to_csv(self.settings_csv_path)
        # print('Window screen name: ', screenName)

    def saveMsgCloseEvent(self, event):
        print('closed')

    def storeDefaultAndCustomColors(self):
        c = self.overlayButton.palette().button().color().name()
        self.defaultToolBarButtonColor = c
        self.doublePressKeyButtonColor = '#fa693b'

    def showAndSetSize(self):
        screenNames = []
        for screen in self.app.screens():
            name = '/'.join(screen.name().split('\\'))
            screenNames.append(name)

        if 'geometry_left' in self.df_settings.index:
            left = int(self.df_settings.at['geometry_left', 'value'])
            top = int(self.df_settings.at['geometry_top', 'value'])+10
            width = int(self.df_settings.at['geometry_width', 'value'])
            height = int(self.df_settings.at['geometry_height', 'value'])
            screenName = self.df_settings.at['screenName', 'value']
            isMaximised = self.df_settings.at['isMaximised', 'value'] == 'True'
            if screenName in screenNames and not isMaximised:
                self.show()
                self.setGeometry(left, top, width, height)
            else:
                self.showMaximized()
        else:
            self.showMaximized()

        self.storeDefaultAndCustomColors()
        h = self.drawIDsContComboBox.size().height()
        self.framesScrollBar.setFixedHeight(h)
        self.zSliceScrollBar.setFixedHeight(h)
        self.alphaScrollBar.setFixedHeight(h)

if __name__ == "__main__":
    print('Loading application...')
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Create the application
    app = QApplication(sys.argv)
    # Apply dark mode
    # file = QFile(":/dark.qss")
    # file.open(QFile.ReadOnly | QFile.Text)
    # stream = QTextStream(file)
    # app.setStyleShefet(stream.readAll())
    # Create and show the main window
    win = guiWin(app)
    win.showAndSetSize()
    # Apply style
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    # Run the event loop
    print('Lauching application...')
    print('Done. If application GUI is not visible, it is probably minimized, '
          'behind some other open window, or on second screen.')
    sys.exit(app.exec_())
