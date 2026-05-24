"""Cell-ACDC dialog windows: export."""

import os
import sys
import re
from typing import Literal, Callable, Dict, Iterable, List, Tuple
import datetime
import pathlib
from collections import defaultdict
import zipfile
from heapq import nlargest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, PathPatch, Path
import numpy as np
import scipy.interpolate

try:
    import tkinter as tk
except Exception as err:
    pass

import cv2
import traceback
from itertools import combinations, permutations
from collections import namedtuple
from natsort import natsorted

# from MyWidgets import Slider, Button, MyRadioButtons
from skimage.measure import label, regionprops
from functools import partial
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.exposure
import skimage.draw
import skimage.registration
import skimage.color
import skimage.segmentation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import time
import sympy as sp
import json
import html

import pyqtgraph as pg

pg.setConfigOption("imageAxisOrder", "row-major")

from qtpy import QtCore
from qtpy.QtGui import (
    QIcon,
    QFontMetrics,
    QKeySequence,
    QFont,
    QRegularExpressionValidator,
    QCursor,
    QKeyEvent,
    QPixmap,
    QFont,
    QPalette,
    QMouseEvent,
    QColor,
)
from qtpy.QtCore import (
    Qt,
    QSize,
    QEvent,
    Signal,
    QEventLoop,
    QTimer,
    QRegularExpression,
)
from qtpy.QtWidgets import (
    QFileDialog,
    QApplication,
    QMainWindow,
    QMenu,
    QLabel,
    QToolBar,
    QScrollBar,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QDialog,
    QFormLayout,
    QListWidget,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QSizePolicy,
    QComboBox,
    QSlider,
    QGridLayout,
    QSpinBox,
    QToolButton,
    QTableView,
    QTextBrowser,
    QDoubleSpinBox,
    QScrollArea,
    QFrame,
    QProgressBar,
    QGroupBox,
    QRadioButton,
    QDockWidget,
    QMessageBox,
    QStyle,
    QPlainTextEdit,
    QSpacerItem,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
    QSplashScreen,
    QAction,
    QListWidgetItem,
    QActionGroup,
    QHeaderView,
    QStyledItemDelegate,
)
import qtpy.compat

from .. import exception_handler
from .. import load, prompts, core, measurements, html_utils
from .. import is_mac, is_win, is_linux, settings_folderpath, config
from .. import preproc_recipes_path, segm_recipes_path, combine_channels_recipes_path
from .. import is_conda_env
from .. import printl
from .. import colors
from .. import issues_url
from .. import myutils
from .. import qutils
from .. import _palettes
from .. import base_cca_dict
from .. import widgets
from .. import user_profile_path, promptable_models_path, models_path
from .. import features
from .. import _core
from .. import _types
from .. import plot
from .. import urls
from ..acdc_regex import float_regex, is_alphanumeric_filename, to_alphanumeric
from .. import _base_widgets
from .. import io
from .. import cca_functions
from .. import path

POSITIVE_FLOAT_REGEX = float_regex(allow_negative=False)
TREEWIDGET_STYLESHEET = _palettes.TreeWidgetStyleSheet()
LISTWIDGET_STYLESHEET = _palettes.ListWidgetStyleSheet()
BACKGROUND_RGBA = _palettes.get_disabled_colors()["Button"]

font = QFont()
font.setPixelSize(12)
italicFont = QFont()
italicFont.setPixelSize(12)
italicFont.setItalic(True)

from ._base import (
    QBaseDialog,
)

class ViewTextDialog(QBaseDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)

        mainLayout = QVBoxLayout()

        textViewWidget = QTextEdit()
        textViewWidget.setReadOnly(True)

        textViewWidget.setText(text)

        buttonsLayout = QHBoxLayout()
        okButton = widgets.okPushButton("Ok")

        okButton.clicked.connect(self.close)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(okButton)

        mainLayout.addWidget(textViewWidget)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
        self.setFont(font)


class pdDataFrameWidget(QMainWindow):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Cell cycle annotations")

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        layout = QVBoxLayout()
        self._layout = layout

        self.tableView = QTableView(self)
        layout.addWidget(self.tableView)
        model = DataFrameModel(df)
        self.tableView.setModel(model)
        for i in range(len(df.columns)):
            self.tableView.resizeColumnToContents(i)
        # layout.addWidget(QPushButton('Ok', self))
        mainContainer.setLayout(layout)

    def updateTable(self, df, IDs=None):
        if df is None:
            df = self.parent.getBaseCca_df()

        if IDs is not None:
            df = df.loc[IDs]

        df = df.reset_index()
        model = DataFrameModel(df)
        self.tableView.setModel(model)
        for i in range(len(df.columns)):
            self.tableView.resizeColumnToContents(i)

    def setGeometryWindow(self, maxWidth=1024):
        width = self.tableView.verticalHeader().width() + 4
        for j in range(self.tableView.model().columnCount()):
            width += self.tableView.columnWidth(j) + 4
        height = self.tableView.horizontalHeader().height() + 4
        h = height + (self.tableView.rowHeight(0) + 4) * 10
        w = width if width < maxWidth else maxWidth
        self.setGeometry(100, 100, w, h)

        # Center window
        parent = self.parent
        if parent is not None:
            # Center the window on main window
            mainWinGeometry = parent.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinCenterX = int(mainWinLeft + mainWinWidth / 2)
            mainWinCenterY = int(mainWinTop + mainWinHeight / 2)
            winGeometry = self.geometry()
            winWidth = winGeometry.width()
            winHeight = winGeometry.height()
            winLeft = int(mainWinCenterX - winWidth / 2)
            winRight = int(mainWinCenterY - winHeight / 2)
            self.move(winLeft, winRight)

    def closeEvent(self, event):
        self.parent.ccaTableWin = None


class ShortcutEditorDialog(QBaseDialog):
    def __init__(
        self,
        widgetsWithShortcut: dict,
        delObjectKey="",
        delObjectButton: Literal["Middle click", "Left click"] = "Middle click",
        zoomOutKeyValue: int = None,
        parent=None,
    ):
        self.cancel = True
        super().__init__(parent)

        self.setWindowTitle("Customize keyboard shortcuts")

        mainLayout = QVBoxLayout()

        self.customShortcuts = {}
        self.shortcutLineEdits = {}

        scrollArea = QScrollArea(self)
        scrollArea.setWidgetResizable(True)
        scrollAreaWidget = QWidget()
        entriesLayout = QGridLayout()

        row = 0
        button = widgets.PushButton(self, flat=True)
        button.setIcon(QIcon(":del_obj_click.svg"))
        self.delObjShortcutLineEdit = widgets.ShortcutLineEdit(
            allowModifiers=True, notAllowedModifier=Qt.AltModifier
        )
        if delObjectKey is not None:
            self.delObjShortcutLineEdit.setText(delObjectKey)
        self.delObjButtonCombobox = QComboBox()
        self.delObjButtonCombobox.addItems(["Middle click", "Left click"])
        self.delObjButtonCombobox.setCurrentText(delObjectButton)
        entriesLayout.addWidget(button, row, 0)
        entriesLayout.addWidget(QLabel("Delete object:"), row, 1)
        entriesLayout.addWidget(self.delObjShortcutLineEdit, row, 2)
        entriesLayout.addWidget(
            self.delObjButtonCombobox, row, 3, alignment=Qt.AlignLeft
        )

        row += 1
        name = "Zoom out"
        button = widgets.PushButton(self, flat=True)
        label = QLabel("Zoom out:")
        self.zoomShortcutLineEdit = widgets.ShortcutLineEdit()
        if zoomOutKeyValue is not None:
            zoomOutKeySequence = widgets.KeySequenceFromText(zoomOutKeyValue)
            self.zoomShortcutLineEdit.setText(zoomOutKeySequence.toString())
            self.zoomShortcutLineEdit.key = zoomOutKeyValue
        self.zoomShortcutLineEdit.textChanged.connect(self.checkDuplicateShortcuts)
        entriesLayout.addWidget(button, row, 0)
        entriesLayout.addWidget(label, row, 1)
        entriesLayout.addWidget(self.zoomShortcutLineEdit, row, 2)
        self.shortcutLineEdits[name] = self.zoomShortcutLineEdit

        row += 1
        for row, (name, widget) in enumerate(widgetsWithShortcut.items(), start=row):
            button = widgets.PushButton(self, flat=True)
            try:
                button.setIcon(widget.icon())
            except:
                pass
            label = QLabel(f"{name}:")
            shortcutLineEdit = widgets.ShortcutLineEdit()
            if hasattr(widget, "keyPressShortcut"):
                shortcutLineEdit.key = widget.keyPressShortcut
                shortcut = widgets.KeySequenceFromText(widget.keyPressShortcut)
                isShortcutKeyPress = True
            else:
                shortcut = widget.shortcut()
                isShortcutKeyPress = False
            shortcutLineEdit.setText(shortcut.toString())
            shortcutLineEdit.textChanged.connect(self.checkDuplicateShortcuts)
            shortcutLineEdit.isShortcutKeyPress = isShortcutKeyPress
            entriesLayout.addWidget(button, row, 0)
            entriesLayout.addWidget(label, row, 1)
            entriesLayout.addWidget(shortcutLineEdit, row, 2)
            self.shortcutLineEdits[name] = shortcutLineEdit

        entriesLayout.setColumnStretch(0, 0)
        entriesLayout.setColumnStretch(1, 0)
        entriesLayout.setColumnStretch(2, 1)
        entriesLayout.setColumnStretch(3, 0)

        scrollAreaWidget.setLayout(entriesLayout)
        scrollArea.setWidget(scrollAreaWidget)
        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(scrollArea)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setFont(font)
        self.setLayout(mainLayout)

    def checkDuplicateShortcuts(self, text):
        for name, shortcutLineEdit in self.shortcutLineEdits.items():
            if shortcutLineEdit == self.sender():
                continue
            if shortcutLineEdit.text() != text:
                continue
            shortcutLineEdit.setText("")

    def warnInvalidKeySequenceDelObjWithLeftClick(self):
        txt = html_utils.paragraph(
            'The selected key sequence to delete objects with "Left click" '
            "is invalid.<br><br>"
            'Only "Middle click" can be used without pressing keys.<br><br>'
            "Thank you for your patience!"
        )
        msg = widgets.myMessageBox()
        msg.warning(self, "Invalid key sequence to delete objects", txt)

    def ok_cb(self):
        delObjButtonText = self.delObjButtonCombobox.currentText()
        delObjKeySequence = self.delObjShortcutLineEdit.keySequence
        if delObjButtonText == "Left click" and delObjKeySequence is None:
            self.warnInvalidKeySequenceDelObjWithLeftClick()
            return

        self.shortcutLineEdits.pop("Zoom out")
        self.cancel = False
        for name, shortcutLineEdit in self.shortcutLineEdits.items():
            text = shortcutLineEdit.text()
            if shortcutLineEdit.isShortcutKeyPress:
                self.customShortcuts[name] = (text, shortcutLineEdit.key)
            else:
                self.customShortcuts[name] = (text, shortcutLineEdit.keySequence)

        delObjQtButton = (
            Qt.MouseButton.LeftButton
            if delObjButtonText == "Left click"
            else Qt.MouseButton.MiddleButton
        )
        self.delObjAction = delObjKeySequence, delObjQtButton
        self.zoomOutKeyValue = self.zoomShortcutLineEdit.key

        self.close()

    def showEvent(self, event) -> None:
        self.resize(int(self.width() * 1.2), self.height())
        self.move(self.x(), 100)


class ScaleBarPropertiesDialog(QBaseDialog):
    sigValueChanged = Signal(object)

    def __init__(
        self, maxLength, maxThickness, PhysicalSizeX, parent=None, **properties
    ):
        super().__init__(parent=parent)

        self.cancel = True
        self.setWindowTitle("Scale bar properties")

        self.PhysicalSizeX = PhysicalSizeX

        mainLayout = QVBoxLayout()

        formLayout = widgets.FormLayout()
        formLayout.setVerticalSpacing(10)
        formLayout.setHorizontalSpacing(50)

        row = 0
        unitCombobox = QComboBox()
        unitFormWidget = widgets.formWidget(unitCombobox, labelTextLeft="Physical unit")
        unitCombobox.addItems(["nm", "μm", "mm", "cm"])
        if properties.get("unit") is None:
            unitCombobox.setCurrentIndex(1)
        else:
            unitCombobox.setCurrentText(properties.get("unit"))
        formLayout.addFormWidget(
            unitFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )
        self.unitCombobox = unitCombobox

        row += 1
        lengthDoubleSpinbox = widgets.DoubleSpinBox()
        lengthDoubleSpinbox.setMaximum(maxLength)
        lengthDoubleSpinbox.setMinimum(PhysicalSizeX)
        lengthDoubleSpinbox.setDecimals(1)
        if properties.get("length_unit") is not None:
            lengthDoubleSpinbox.setValue(properties.get("length_unit"))
        else:
            deafultLength = np.ceil(PhysicalSizeX * 15)
            lengthDoubleSpinbox.setValue(round(deafultLength))
        lengthFormWidget = widgets.formWidget(
            lengthDoubleSpinbox, labelTextLeft="Length (μm)"
        )
        self.lengthFormWidget = lengthFormWidget
        self.lengthDoubleSpinbox = lengthDoubleSpinbox
        formLayout.addFormWidget(
            lengthFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )

        row += 1
        thicknessSpinbox = widgets.DoubleSpinBox()
        thicknessSpinbox.setMaximum(maxThickness)
        thicknessSpinbox.setMinimum(1)
        if properties.get("thickness") is not None:
            thicknessSpinbox.setValue(properties.get("thickness"))
        else:
            thicknessSpinbox.setValue(round(4, 1))
        thicknessSpinbox.setDecimals(1)
        thicknessFormWidget = widgets.formWidget(
            thicknessSpinbox, labelTextLeft="Thickness (pixel)"
        )
        formLayout.addFormWidget(
            thicknessFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )
        self.thicknessSpinbox = thicknessSpinbox

        row += 1
        locCombobox = QComboBox()
        locFormWidget = widgets.formWidget(locCombobox, labelTextLeft="Location")
        locCombobox.addItems(
            ["Bottom-right", "Bottom-left", "Top-left", "Top-right", "Custom"]
        )
        loc = properties.get("loc")
        if isinstance(loc, str):
            locCombobox.setCurrentText(loc.capitalize())
        formLayout.addFormWidget(
            locFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )
        self.locCombobox = locCombobox

        row += 1
        self.colorButton = widgets.myColorButton(color=(255, 255, 255))
        if properties.get("color") is not None:
            self.colorButton.setColor(properties.get("color"))
        colorFormWidget = widgets.formWidget(
            self.colorButton,
            labelTextLeft="Color",
            widgetAlignment=Qt.AlignCenter,
            stretchWidget=False,
        )
        formLayout.addFormWidget(
            colorFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )

        row += 1
        displayTextToggle = widgets.Toggle()
        if properties.get("is_text_visible") is not None:
            displayTextToggle.setChecked(properties.get("is_text_visible"))
        else:
            displayTextToggle.setChecked(True)
        displayTextFormWidget = widgets.formWidget(
            displayTextToggle,
            labelTextLeft="Display text",
            widgetAlignment=Qt.AlignCenter,
            stretchWidget=False,
        )
        formLayout.addFormWidget(
            displayTextFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )
        self.displayTextToggle = displayTextToggle

        row += 1
        fontSizeSpinbox = widgets.SpinBox()
        if properties.get("font_size") is not None:
            fontSizeSpinbox.setValue(int(properties.get("font_size")))
        else:
            fontSizeSpinbox.setValue(12)
        fontSizeFormWidget = widgets.formWidget(
            fontSizeSpinbox, labelTextLeft="Font size (px)"
        )
        self.fontSizeSpinbox = fontSizeSpinbox
        formLayout.addFormWidget(
            fontSizeFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )

        row += 1
        decimalsSpinbox = widgets.SpinBox()
        decimalsSpinbox.setMaximum(6)
        decimalsSpinbox.setMinimum(0)
        if properties.get("num_decimals") is not None:
            decimalsSpinbox.setValue(properties.get("num_decimals"))
        else:
            decimalsSpinbox.setValue(0)
        decimalsFormWidget = widgets.formWidget(
            decimalsSpinbox, labelTextLeft="Number of decimals"
        )
        formLayout.addFormWidget(
            decimalsFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )
        self.decimalsSpinbox = decimalsSpinbox

        row += 1
        moveWithZoomToggle = widgets.Toggle()
        moveWithZoomWidget = widgets.formWidget(
            moveWithZoomToggle,
            labelTextLeft="Move scale bar with zoom",
            widgetAlignment=Qt.AlignCenter,
            stretchWidget=False,
        )
        formLayout.addFormWidget(
            moveWithZoomWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )
        self.moveWithZoomToggle = moveWithZoomToggle

        mainLayout.addLayout(formLayout)

        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch()

        self.setLayout(mainLayout)
        self.setFont(font)

        self.unitCombobox.currentTextChanged.connect(self.updateLengthUnit)
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)

        self.colorButton.sigColorChanging.connect(self.onValueChanged)
        self.lengthDoubleSpinbox.valueChanged.connect(self.onValueChanged)
        self.thicknessSpinbox.valueChanged.connect(self.onValueChanged)
        self.locCombobox.currentTextChanged.connect(self.onValueChanged)
        self.displayTextToggle.toggled.connect(self.onValueChanged)
        self.fontSizeSpinbox.valueChanged.connect(self.onValueChanged)
        self.decimalsSpinbox.valueChanged.connect(self.onValueChanged)
        self.moveWithZoomToggle.toggled.connect(self.onValueChanged)

    def onValueChanged(self, *args, **kwargs):
        self.sigValueChanged.emit(self.kwargs())

    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        self.colorButton.colorDialog.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.colorButton.colorDialog.setParent(self)
        self.colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w + left + 10, colorDialogTop)

    def updateLengthUnit(self, unit):
        newText = re.sub(r"\(.*\)", f"({unit})", self.lengthFormWidget.labelLeft.text())
        self.lengthFormWidget.labelLeft.setText(newText)
        self.onValueChanged(self)

    def kwargs(self):
        unit = self.unitCombobox.currentText()
        length_unit = self.lengthDoubleSpinbox.value()
        length_um = _core.convert_length(length_unit, unit, "μm")
        length_pixel = length_um / self.PhysicalSizeX
        kwargs = {
            "thickness": self.thicknessSpinbox.value(),
            "length_pixel": length_pixel,
            "length_unit": length_unit,
            "is_text_visible": self.displayTextToggle.isChecked(),
            "color": self.colorButton.color(),
            "loc": self.locCombobox.currentText().lower(),
            "font_size": self.fontSizeSpinbox.value(),
            "unit": unit,
            "num_decimals": self.decimalsSpinbox.value(),
            "move_with_zoom": self.moveWithZoomToggle.isChecked(),
        }
        return kwargs

    def ok_cb(self):
        self.cancel = False
        self.close()


class ExportToVideoParametersDialog(QBaseDialog):
    sigOk = Signal(dict)
    sigAddScaleBar = Signal(bool)
    sigAddTimestamp = Signal(bool)
    sigRescaleIntensLut = Signal(str, str)
    sigChangeStartTime = Signal(str)

    def __init__(
        self,
        channels,
        parent=None,
        startFolderpath="",
        startFilename="",
        startFrameNum=1,
        SizeT=1,
        SizeZ=1,
        isTimelapseVideo=True,
        isScaleBarPresent=False,
        isTimestampPresent=False,
        rescaleIntensChannelHowMapper=None,
        startTime=None,
    ):
        self.cancel = True

        if rescaleIntensChannelHowMapper is None:
            rescaleIntensChannelHowMapper = {}

        super().__init__(parent=parent)

        self.setWindowTitle("Preferences for output video")

        mainLayout = QVBoxLayout()

        gridLayout = QGridLayout()

        navVar = "frame number" if isTimelapseVideo else "z-slice"
        maxNavVar = SizeT if isTimelapseVideo else SizeZ

        self.isTimelapseVideo = isTimelapseVideo

        row = 0
        gridLayout.addWidget(QLabel(f"Start {navVar}:"), row, 0)
        self.startNavVarNumberEntry = widgets.SpinBox()
        self.startNavVarNumberEntry.setMinimum(1)
        self.startNavVarNumberEntry.setMaximum(maxNavVar - 1)
        self.startNavVarNumberEntry.setValue(startFrameNum)
        gridLayout.addWidget(self.startNavVarNumberEntry, row, 1)

        row += 1
        gridLayout.addWidget(QLabel(f"Stop {navVar}:"), row, 0)
        self.stopNavVarNumberEntry = widgets.SpinBox()
        self.stopNavVarNumberEntry.setMinimum(2)
        self.stopNavVarNumberEntry.setMaximum(maxNavVar)
        self.stopNavVarNumberEntry.setValue(maxNavVar)
        gridLayout.addWidget(self.stopNavVarNumberEntry, row, 1)

        row += 1
        gridLayout.addWidget(QLabel("File format:"), row, 0)
        self.fileFormatCombobox = QComboBox()
        self.fileFormatCombobox.addItems(["MP4", "AVI"])
        gridLayout.addWidget(self.fileFormatCombobox, row, 1)

        row += 1
        gridLayout.addWidget(QLabel("Frame rate (FPS):"), row, 0)
        self.fpsWidget = widgets.FloatLineEdit(allowNegative=False)
        self.fpsWidget.setValue(10.0)
        gridLayout.addWidget(self.fpsWidget, row, 1)

        row += 1
        self.dpiWidget = widgets.IntLineEdit(allowNegative=False)
        self.dpiWidget.setValue(300)
        self.dpiWidget.label = QLabel("DPI")
        gridLayout.addWidget(self.dpiWidget.label, row, 0)
        gridLayout.addWidget(self.dpiWidget, row, 1)

        row += 1
        gridLayout.addWidget(QLabel("Folder path:"), row, 0)
        self.folderPathLineEdit = widgets.ElidingLineEdit(minWidth=240)
        self.folderPathLineEdit.setText(startFolderpath)
        gridLayout.addWidget(self.folderPathLineEdit, row, 1)
        self.browseButton = widgets.browseFileButton(
            start_dir=startFolderpath, openFolder=True
        )
        gridLayout.addWidget(self.browseButton, row, 2)

        row += 1
        gridLayout.addWidget(QLabel("Filename:"), row, 0)
        self.filenameLineEdit = widgets.alphaNumericLineEdit()
        self.filenameLineEdit.setAlignment(Qt.AlignCenter)
        self.filenameLineEdit.setText(startFilename)
        gridLayout.addWidget(self.filenameLineEdit, row, 1)
        self.fileFormatLabel = QLabel(".mp4")
        gridLayout.addWidget(self.fileFormatLabel, row, 2)

        row += 1
        gridLayout.addWidget(QLabel("Add Scale Bar:"), row, 0)
        self.addScaleBarToggle = widgets.Toggle()
        gridLayout.addWidget(self.addScaleBarToggle, row, 1, alignment=Qt.AlignCenter)
        self.addScaleBarToggle.setChecked(isScaleBarPresent)

        if isTimelapseVideo:
            row += 1
            gridLayout.addWidget(QLabel("Add timestamp:"), row, 0)
            self.addTimestampToggle = widgets.Toggle()
            gridLayout.addWidget(
                self.addTimestampToggle, row, 1, alignment=Qt.AlignCenter
            )
            self.addTimestampToggle.setChecked(isTimestampPresent)

        for channel in channels:
            row += 1
            labelText = f"Rescale intensities (LUT) <i>{channel}</i>:"
            gridLayout.addWidget(QLabel(labelText), row, 0)
            rescaleItems = ["Rescale each 2D image"]
            if SizeZ > 1:
                rescaleItems.append("Rescale across z-stack")
            if isTimelapseVideo:
                rescaleItems.append("Rescale across time frames")
            rescaleItems.append("Choose custom levels...")
            rescaleItems.append("Do no rescale, display raw image")
            rescaleIntensCombobox = QComboBox()
            rescaleIntensCombobox.addItems(rescaleItems)
            rescaleIntensHow = rescaleIntensChannelHowMapper.get(channel)
            if rescaleIntensHow is not None:
                rescaleIntensCombobox.setCurrentText(rescaleIntensHow)
            gridLayout.addWidget(rescaleIntensCombobox, row, 1)
            rescaleIntensCombobox.textActivated.connect(
                partial(self.emitRescaleIntens, channel=channel)
            )

        row += 1
        gridLayout.addWidget(QLabel("Save a PNG for each frame:"), row, 0)
        self.saveFramesToggle = widgets.Toggle()
        gridLayout.addWidget(self.saveFramesToggle, row, 1, alignment=Qt.AlignCenter)

        gridLayout.setColumnStretch(0, 0)
        gridLayout.setColumnStretch(1, 1)
        gridLayout.setColumnStretch(2, 0)

        self.fileFormatCombobox.currentTextChanged.connect(self.updateFileFormat)
        self.browseButton.sigPathSelected.connect(self.updateFolderPath)
        self.addScaleBarToggle.toggled.connect(self.addScaleBarToggled)
        if isTimelapseVideo:
            self.addTimestampToggle.toggled.connect(self.addTimestampToggled)

        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.setText("Export")

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def emitRescaleIntens(self, how, channel=""):
        self.sigRescaleIntensLut.emit(how, channel)

    def addScaleBarToggled(self, checked):
        self.sigAddScaleBar.emit(checked)

    def addTimestampToggled(self, checked):
        self.sigAddTimestamp.emit(checked)

    def updateFolderPath(self, folderPath):
        self.folderPathLineEdit.setText(folderPath)
        self.browseButton.setStartPath(folderPath)

    def updateFileFormat(self, fileFormat):
        self.fileFormatLabel.setText(f".{fileFormat.lower()}")

    def validateFolderPath(self):
        folderPath = self.folderPathLineEdit.text()
        if os.path.exists(folderPath) and os.path.isdir(folderPath):
            return True

        text = html_utils.paragraph(
            "The selected folder path is not a valid folder or does not exist"
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Not a valid folder", text)
        return False

    def validateFilename(self):
        filename = self.filenameLineEdit.text()
        if filename:
            return True

        text = html_utils.paragraph("The filename cannot be empty!")
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Not a valid folder", text)
        return False

    def validate(self):
        proceed = self.validateFolderPath()
        if not proceed:
            return False

        proceed = self.validateFilename()
        if not proceed:
            return False

        return True

    def preferences(self, makedirs=True):
        filename = f"{self.filenameLineEdit.text()}{self.fileFormatLabel.text()}"
        avi_filename = f"{self.filenameLineEdit.text()}.avi"
        avi_filepath = os.path.join(self.folderPathLineEdit.text(), avi_filename)
        png_foldername = f"{self.filenameLineEdit.text()}_frames_PNG"
        pngs_folderpath = os.path.join(self.folderPathLineEdit.text(), png_foldername)
        if makedirs:
            os.makedirs(pngs_folderpath, exist_ok=True)

        preferences = {
            "start_nav_var_num": self.startNavVarNumberEntry.value(),
            "stop_nav_var_num": self.stopNavVarNumberEntry.value(),
            "filepath": os.path.join(self.folderPathLineEdit.text(), filename),
            "filename": self.filenameLineEdit.text(),
            "avi_filepath": avi_filepath,
            "pngs_folderpath": pngs_folderpath,
            "num_digits": len(str(self.stopNavVarNumberEntry.value())),
            "fps": self.fpsWidget.value(),
            "save_pngs": self.saveFramesToggle.isChecked(),
            "is_timelapse": self.isTimelapseVideo,
            "dpi": self.dpiWidget.value(),
        }
        return preferences

    def ok_cb(self):
        proceed = self.validate()
        if not proceed:
            return
        self.cancel = False
        self.sigOk.emit(self.preferences())
        self.selected_preferences = self.preferences()
        self.close()


class TimestampPropertiesDialog(QBaseDialog):
    sigValueChanged = Signal(object)

    def __init__(self, parent=None, **properties):
        super().__init__(parent=parent)

        self.cancel = True
        self.setWindowTitle("Timestamp preferences")

        mainLayout = QVBoxLayout()

        formLayout = widgets.FormLayout()
        formLayout.setVerticalSpacing(10)
        formLayout.setHorizontalSpacing(50)

        row = 0
        self.startTimeWidget = widgets.TimeWidget()
        if properties.get("start_timedelta") is not None:
            self.startTimeWidget.setValuesFromTimedelta(
                properties.get("start_timedelta")
            )
        startTimeFormWidget = widgets.formWidget(
            self.startTimeWidget,
            labelTextLeft="Start time",
        )
        formLayout.addFormWidget(
            startTimeFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )

        row += 1
        self.colorButton = widgets.myColorButton(color=(255, 255, 255))
        if properties.get("color") is not None:
            self.colorButton.setColor(properties.get("color"))
        colorFormWidget = widgets.formWidget(
            self.colorButton,
            labelTextLeft="Color",
            widgetAlignment=Qt.AlignCenter,
            stretchWidget=False,
        )
        formLayout.addFormWidget(
            colorFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )

        row += 1
        fontSizeWidget = widgets.FontSizeWidget()
        if properties.get("font_size") is not None:
            fontSizeWidget.setValue(properties.get("font_size"))
        else:
            fontSizeWidget.setValue(12)
        fontSizeFormWidget = widgets.formWidget(
            fontSizeWidget, labelTextLeft="Font size (px)"
        )
        self.fontSizeWidget = fontSizeWidget
        formLayout.addFormWidget(
            fontSizeFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )

        row += 1
        locCombobox = QComboBox()
        locFormWidget = widgets.formWidget(locCombobox, labelTextLeft="Location")
        locCombobox.addItems(
            ["Top-left", "Top-right", "Bottom-left", "Bottom-right", "Custom"]
        )
        loc = properties.get("loc")
        if isinstance(loc, str):
            locCombobox.setCurrentText(loc.capitalize())
        formLayout.addFormWidget(
            locFormWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )
        self.locCombobox = locCombobox

        row += 1
        moveWithZoomToggle = widgets.Toggle()
        moveWithZoomWidget = widgets.formWidget(
            moveWithZoomToggle,
            labelTextLeft="Move timestamp with zoom",
            widgetAlignment=Qt.AlignCenter,
            stretchWidget=False,
        )
        formLayout.addFormWidget(
            moveWithZoomWidget, row=row, leftLabelAlignment=Qt.AlignLeft
        )
        self.moveWithZoomToggle = moveWithZoomToggle

        mainLayout.addLayout(formLayout)

        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch()

        self.setLayout(mainLayout)
        self.setFont(font)

        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)

        self.startTimeWidget.sigValueChanged.connect(self.onValueChanged)

        self.locCombobox.currentTextChanged.connect(self.onValueChanged)
        self.fontSizeWidget.sigTextChanged.connect(self.onValueChanged)
        self.moveWithZoomToggle.toggled.connect(self.onValueChanged)

    def onValueChanged(self, *args, **kwargs):
        self.sigValueChanged.emit(self.kwargs())

    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        self.colorButton.colorDialog.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.colorButton.colorDialog.setParent(self)
        self.colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w + left + 10, colorDialogTop)

    def kwargs(self):
        kwargs = {
            "color": self.colorButton.color(),
            "start_timedelta": self.startTimeWidget.timedelta(),
            "loc": self.locCombobox.currentText().lower(),
            "font_size": self.fontSizeWidget.text(),
            "move_with_zoom": self.moveWithZoomToggle.isChecked(),
        }
        return kwargs

    def ok_cb(self):
        self.cancel = False
        self.close()


class ExportToImageParametersDialog(QBaseDialog):
    sigOk = Signal(dict)
    sigAddScaleBar = Signal(bool)
    sigRangeChanged = Signal(object)

    def __init__(
        self,
        parent=None,
        startFolderpath="",
        startFilename="",
        startViewRange=None,
        isScaleBarPresent=False,
    ):
        self.cancel = True

        super().__init__(parent=parent)

        self.setWindowTitle("Preferences for output image")

        mainLayout = QVBoxLayout()

        gridLayout = QGridLayout()

        row = 0
        gridLayout.addWidget(QLabel("View range X axis:"), row, 0)
        self.xRangeSelector = widgets.RangeSelector(integers=True)
        if startViewRange is not None:
            xRange, yRange = startViewRange
            self.xRangeSelector.setRange(*xRange)
        gridLayout.addWidget(self.xRangeSelector, row, 1)

        row += 1
        gridLayout.addWidget(QLabel("View range Y axis:"), row, 0)
        self.yRangeSelector = widgets.RangeSelector(integers=True)
        if startViewRange is not None:
            xRange, yRange = startViewRange
            self.yRangeSelector.setRange(*yRange)
        gridLayout.addWidget(self.yRangeSelector, row, 1)

        row += 1
        gridLayout.addWidget(QLabel("Width and Height:"), row, 0)
        self.widthHeightSelector = widgets.RangeSelector(integers=True, ordered=False)
        if startViewRange is not None:
            xRange, yRange = startViewRange
            width = int(xRange[1] - xRange[0])
            height = int(yRange[1] - yRange[0])
            self.widthHeightSelector.setRange(width, height)
        gridLayout.addWidget(self.widthHeightSelector, row, 1)
        self.lockSizeButton = widgets.LockPushButton()
        self.lockSizeButton.setCheckable(True)
        self.lockSizeButton.setToolTip("Lock width and height")
        gridLayout.addWidget(self.lockSizeButton, row, 2)

        row += 1
        gridLayout.addWidget(QLabel("File format:"), row, 0)
        self.fileFormatCombobox = QComboBox()
        self.fileFormatCombobox.addItems(["SVG", "PNG", "TIFF", "JPEG"])
        gridLayout.addWidget(self.fileFormatCombobox, row, 1)

        row += 1
        self.dpiWidget = widgets.IntLineEdit(allowNegative=False)
        self.dpiWidget.setValue(300)
        self.dpiWidget.label = QLabel("DPI")
        gridLayout.addWidget(self.dpiWidget.label, row, 0)
        gridLayout.addWidget(self.dpiWidget, row, 1)
        self.dpiWidget.hide()
        self.dpiWidget.label.hide()

        row += 1
        gridLayout.addWidget(QLabel("Folder path:"), row, 0)
        self.folderPathLineEdit = widgets.ElidingLineEdit(minWidth=240)
        self.folderPathLineEdit.setText(startFolderpath)
        gridLayout.addWidget(self.folderPathLineEdit, row, 1)
        self.browseButton = widgets.browseFileButton(
            start_dir=startFolderpath, openFolder=True
        )
        gridLayout.addWidget(self.browseButton, row, 2)

        row += 1
        gridLayout.addWidget(QLabel("Filename:"), row, 0)
        self.filenameLineEdit = widgets.alphaNumericLineEdit()
        self.filenameLineEdit.setAlignment(Qt.AlignCenter)
        self.filenameLineEdit.setText(startFilename)
        gridLayout.addWidget(self.filenameLineEdit, row, 1)
        self.fileFormatLabel = QLabel(
            f".{self.fileFormatCombobox.currentText().lower()}"
        )
        gridLayout.addWidget(self.fileFormatLabel, row, 2)

        row += 1
        gridLayout.addWidget(QLabel("Add Scale Bar:"), row, 0)
        self.addScaleBarToggle = widgets.Toggle()
        gridLayout.addWidget(self.addScaleBarToggle, row, 1, alignment=Qt.AlignCenter)
        self.addScaleBarToggle.setChecked(isScaleBarPresent)

        self.fileFormatCombobox.currentTextChanged.connect(self.updateFileFormat)
        self.browseButton.sigPathSelected.connect(self.updateFolderPath)
        self.addScaleBarToggle.toggled.connect(self.addScaleBarToggled)
        self.xRangeSelector.sigLowValueChanged.connect(self.x0Changed)
        self.xRangeSelector.sigHighValueChanged.connect(self.x1Changed)
        self.yRangeSelector.sigLowValueChanged.connect(self.y0Changed)
        self.yRangeSelector.sigHighValueChanged.connect(self.y1Changed)
        self.widthHeightSelector.sigLowValueChanged.connect(self.widthChanged)
        self.widthHeightSelector.sigHighValueChanged.connect(self.heightChanged)
        self.widthHeightSelector.sigRangeManuallyChanged.connect(
            self.widthHeightManuallyChanged
        )

        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.setText("Export")

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        gridLayout.setColumnStretch(2, 0)

        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def widthHeightManuallyChanged(self, *args):
        self.lockSizeButton.setChecked(True)

    def x0Changed(self, *args):
        if self.lockSizeButton.isChecked():
            x0, _ = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            width, height = self.widthHeightSelector.range()
            x1 = x0 + width
            xRange = (x0, x1)
        else:
            xRange = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            _, height = self.widthHeightSelector.range()
            width = int(xRange[1] - xRange[0])

        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
        self.widthHeightSelector.setRangeNoEmit(width, height)
        self.rangeChanged()

    def x1Changed(self, *args):
        if self.lockSizeButton.isChecked():
            _, x1 = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            width, height = self.widthHeightSelector.range()
            x0 = x1 - width
            xRange = (x0, x1)
        else:
            xRange = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            _, height = self.widthHeightSelector.range()
            width = int(xRange[1] - xRange[0])

        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
        self.widthHeightSelector.setRangeNoEmit(width, height)

        self.rangeChanged()

    def y0Changed(self, *args):
        if self.lockSizeButton.isChecked():
            xRange = self.xRangeSelector.range()
            y0, _ = self.yRangeSelector.range()
            width, height = self.widthHeightSelector.range()
            y1 = y0 + height
            yRange = (y0, y1)
        else:
            xRange = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            width, _ = self.widthHeightSelector.range()
            height = int(yRange[1] - yRange[0])

        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
        self.widthHeightSelector.setRangeNoEmit(width, height)

        self.rangeChanged()

    def y1Changed(self, *args):
        if self.lockSizeButton.isChecked():
            xRange = self.xRangeSelector.range()
            _, y1 = self.yRangeSelector.range()
            width, height = self.widthHeightSelector.range()
            y0 = y1 - height
            yRange = (y0, y1)
        else:
            xRange = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            width, _ = self.widthHeightSelector.range()
            height = int(yRange[1] - yRange[0])

        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
        self.widthHeightSelector.setRangeNoEmit(width, height)

        self.rangeChanged()

    def widthChanged(self, *args):
        self.widthHeightChanged()
        self.rangeChanged()

    def heightChanged(self, *args):
        self.widthHeightChanged()
        self.rangeChanged()

    def updateViewRangeExportToImageDialog(self, viewBox, viewRange, changed):
        xRange, yRange = viewRange
        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)

    def widthHeightChanged(self, *args):
        x0, _ = self.xRangeSelector.range()
        y0, _ = self.yRangeSelector.range()
        width, height = self.widthHeightSelector.range()
        x1 = x0 + width
        y1 = y0 + height
        self.xRangeSelector.setRangeNoEmit(x0, x1)
        self.yRangeSelector.setRangeNoEmit(y0, y1)
        self.rangeChanged()

    def rangeChanged(self, *args):
        xRange = self.xRangeSelector.range()
        yRange = self.yRangeSelector.range()
        self.sigRangeChanged.emit((xRange, yRange))

    def addScaleBarToggled(self, checked):
        self.sigAddScaleBar.emit(checked)

    def updateFolderPath(self, folderPath):
        self.folderPathLineEdit.setText(folderPath)
        self.browseButton.setStartPath(folderPath)

    def updateFileFormat(self, fileFormat):
        if fileFormat == "SVG":
            self.dpiWidget.hide()
            self.dpiWidget.label.hide()
        else:
            self.dpiWidget.show()
            self.dpiWidget.label.show()

        self.fileFormatLabel.setText(f".{fileFormat.lower()}")

    def validateFolderPath(self):
        folderPath = self.folderPathLineEdit.text()
        if os.path.exists(folderPath) and os.path.isdir(folderPath):
            return True

        text = html_utils.paragraph(
            "The selected folder path is not a valid folder or does not exist"
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Not a valid folder", text)
        return False

    def validateFilename(self):
        filename = self.filenameLineEdit.text()
        if filename:
            return True

        text = html_utils.paragraph("The filename cannot be empty!")
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Not a valid folder", text)
        return False

    def validate(self):
        proceed = self.validateFolderPath()
        if not proceed:
            return False

        proceed = self.validateFilename()
        if not proceed:
            return False

        return True

    def setViewRange(self, xRange, yRange, emitSignal=True):
        if self.lockSizeButton.isChecked():
            x0, _ = xRange
            y0, _ = yRange
            width, height = self.widthHeightSelector.range()
            x1 = x0 + width
            y1 = y0 + height
            xRange = (x0, x1)
            yRange = (y0, y1)
        else:
            width = int(xRange[1] - xRange[0])
            height = int(yRange[1] - yRange[0])

        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
        self.widthHeightSelector.setRangeNoEmit(width, height)
        if not emitSignal:
            return

        self.rangeChanged()

    def viewRange(self):
        xRange = self.xRangeSelector.range()
        yRange = self.yRangeSelector.range()
        return (xRange, yRange)

    def preferences(self):
        filename = f"{self.filenameLineEdit.text()}{self.fileFormatLabel.text()}"
        preferences = {
            "view_range_x": self.xRangeSelector.range(),
            "view_range_y": self.yRangeSelector.range(),
            "filepath": os.path.join(self.folderPathLineEdit.text(), filename),
            "filename": self.filenameLineEdit.text(),
            "dpi": self.dpiWidget.value(),
        }
        return preferences

    def ok_cb(self):
        proceed = self.validate()
        if not proceed:
            return
        self.cancel = False
        self.sigOk.emit(self.preferences())
        self.selected_preferences = self.preferences()
        self.close()


class LogoDialog(QDialog):
    def __init__(self, logo_path, icon_path, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        self.setWindowFlags(Qt.FramelessWindowHint)
        # self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        # self.setAttribute(Qt.WA_TranslucentBackground)
        # self.setWindowIcon(QIcon(icon_path))

        labelLogo = QLabel()
        pixmapLogo = QPixmap(logo_path)
        labelLogo.setPixmap(pixmapLogo)

        layout.addWidget(labelLogo)

        self.setLayout(layout)


class ObjectCountDialog(QBaseDialog):
    sigShowEvent = Signal()
    sigUpdateCounts = Signal()

    def __init__(
        self,
        categoryCountMapper: dict,
        parent=None,
        data: list["load.loadData"] | None = None,
    ):
        super().__init__(parent=parent)
        self.setWindowTitle("Object count")

        self.cancel = False
        mainLayout = QVBoxLayout()

        cancelOkLayout = widgets.CancelOkButtonsLayout()
        cancelOkLayout.okButton.clicked.connect(self.ok_cb)
        cancelOkLayout.cancelButton.clicked.connect(self.close)

        self.data = data
        if data is not None:
            saveCountsButton = widgets.savePushButton("Export counts to CSV table")
            saveCountsButton.clicked.connect(self.saveCounts)
            cancelOkLayout.insertWidget(3, saveCountsButton)

        updateCountsButton = widgets.reloadPushButton("Update counts")
        cancelOkLayout.insertWidget(3, updateCountsButton)
        updateCountsButton.clicked.connect(self.emitUpdateCounts)

        mainLayout.addWidget(
            QLabel(html_utils.paragraph("Object count<br>", font_size="18px")),
            alignment=Qt.AlignLeft,
        )
        self.showHideButtons = []
        self.categoryLabelMapper = {}
        for category, count in categoryCountMapper.items():
            categoryLayout = QHBoxLayout()
            categoryLayout.addSpacing(10)
            catText = html_utils.paragraph(f"<br>{category}<br>", font_size="13px")
            catLabel = QLabel(catText)
            categoryLayout.addWidget(catLabel)
            categoryLayout.addStretch(1)

            countText = html_utils.paragraph(f"<br>{count}<br>", font_size="13px")
            countLabel = QLabel(countText)
            categoryLayout.addWidget(countLabel)

            self.categoryLabelMapper[category] = countLabel

            showHideButton = widgets.showDetailsButton(txt="")
            showHideButton.setChecked(True)
            showHideButton.sigToggled.connect(
                partial(self.showHideCount, labels=(catLabel, countLabel))
            )
            showHideButton.setToolTip(f'Show/hide "{category}" count')
            categoryLayout.addSpacing(10)
            categoryLayout.addWidget(showHideButton)
            showHideButton.category = category

            self.showHideButtons.append(showHideButton)

            categoryLayout.setStretch(0, 0)
            categoryLayout.setStretch(1, 0)
            categoryLayout.setStretch(3, 0)

            mainLayout.addLayout(categoryLayout)
            mainLayout.addWidget(widgets.QHLine())

        mainLayout.addSpacing(10)

        infoLayout = QHBoxLayout()
        self.livePreviewCheckbox = QCheckBox("Live preview")
        self.livePreviewCheckbox.setChecked(True)
        infoLayout.addWidget(self.livePreviewCheckbox)
        infoLayout.addStretch(1)
        self.warnLabel = QLabel("")
        infoLayout.addWidget(self.warnLabel)
        self.livePreviewCheckbox.toggled.connect(self.updateWarnLabel)
        mainLayout.addLayout(infoLayout)

        mainLayout.addSpacing(30)
        mainLayout.addStretch(1)
        mainLayout.addLayout(cancelOkLayout)

        self.setLayout(mainLayout)

    def saveCounts(self, checked=False):
        categories = self.activeCategories()
        for posData in self.data:
            countMapper = posData.countObjectsInSegm(categories)
            countMapper.pop("In current frame", None)
            df_count_endname = posData.saveObjCounts(countMapper)

        txt = html_utils.paragraph(f"""
            Done!<br><br>
            Objects count table saved in every loaded Position folder<br> 
            as a <b>CSV file ending with</b> <code>{df_count_endname}</code>
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, "Objects count saved", txt)

    def updateWarnLabel(self, checked):
        if not checked:
            self.warnLabel.setText(
                html_utils.paragraph(
                    "WARNING: without live preview, counts are not updated",
                    font_color="red",
                )
            )
        else:
            self.warnLabel.setText("")

    def emitUpdateCounts(self):
        self.sigUpdateCounts.emit()

    def activeCategories(self) -> List[str]:
        activeCategories = []
        for showHideButton in self.showHideButtons:
            if not showHideButton.isChecked():
                continue
            activeCategories.append(showHideButton.category)

        return activeCategories

    def showHideCount(self, checked, labels):
        for label in labels:
            label.setVisible(checked)

        QTimer.singleShot(100, self.resizeToHeightHint)

    def updateCounts(self, categoryCountMapper):
        for category, count in categoryCountMapper.items():
            countLabel = self.categoryLabelMapper[category]
            countText = html_utils.paragraph(f"<br>{count}<br>", font_size="13px")
            countLabel.setText(countText)

    def resizeToHeightHint(self):
        heightHint = self.sizeHint().height()
        self.resize(self.width(), heightHint)

    def showEvent(self, event):
        widthHint = self.sizeHint().width()
        self.resize(int(widthHint * 1.5), self.height())
        self.sigShowEvent.emit()

    def ok_cb(self):
        self.cancel = False
        self.close()

# Sibling imports (deferred to avoid import cycles)
from .models import (
    DataFrameModel,
)

