"""Cell-ACDC dialog windows: tracking."""

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
from .export import (
    pdDataFrameWidget,
)
from .general import (
    QLineEditDialog,
)

class TrackSubCellObjectsDialog(QBaseDialog):
    def __init__(self, basename="", parent=None):
        self.cancel = True
        super().__init__(parent=parent)

        self.setWindowTitle("Track sub-cellular objects parameters")

        mainLayout = QVBoxLayout()
        entriesLayout = widgets.FormLayout()

        row = 0
        infoTxt = html_utils.paragraph("""
            Select <b>behaviour with untracked objects</b>:<br><br>
            NOTE: this utility <b>always create new files</b>.
            Original segmentation masks <br>are not modified</b>.
        """)
        options = (
            "Delete sub-cellular objects that do not belong to any cell",
            "Delete cells that do not have any sub-cellular object",
            "Delete both cells and sub-cellular objects without an assignment",
            "Only track the objects and keep all the non-tracked objects",
        )
        combobox = widgets.QCenteredComboBox()
        combobox.addItems(options)
        self.optionsWidget = widgets.formWidget(
            combobox,
            addInfoButton=True,
            labelTextLeft="Tracking mode: ",
            infoTxt=infoTxt,
        )
        entriesLayout.addFormWidget(self.optionsWidget, row=row)

        row += 1
        infoTxt = html_utils.paragraph("""
            Re-label sub-cellular objects before assigning them to the cell.<br><br> 
            Activate this option if you have <b>merged sub-cellular objects</b>   
            that must be separated, or the segmentation is a <b>boolean mask</b> 
            (i.e., semantic segmentation).
        """)
        self.relabelSubObjLab = widgets.formWidget(
            widgets.Toggle(),
            addInfoButton=True,
            stretchWidget=False,
            labelTextLeft="Re-label sub-cellular objects before tracking: ",
            infoTxt=infoTxt,
        )
        entriesLayout.addFormWidget(self.relabelSubObjLab, row=row)

        row += 1
        IoAtext = html_utils.paragraph("""
            Enter a <b>minimum percentage (0-1) of the sub-cellular object's area</b><br>
            that MUST overlap with the parent cell to be considered belonging to a cell:
        """)
        spinbox = widgets.CenteredDoubleSpinbox()
        spinbox.setMaximum(1)
        spinbox.setValue(0.5)
        spinbox.setSingleStep(0.1)
        self.IoAwidget = widgets.formWidget(
            spinbox,
            addInfoButton=True,
            labelTextLeft="IoA threshold: ",
            infoTxt=IoAtext,
        )
        entriesLayout.addFormWidget(self.IoAwidget, row=row)

        row += 1
        infoTxt = html_utils.paragraph("""
            The third segmentation file is the result of <b>subtracting the 
            sub-cellular objects from the parent objects</b><br><br>
            This is useful if, for example, you need to compute measurements 
            only from the cytoplasm (i.e., the sub-cellular object is the nucleus).
        """)
        self.createThirdSegmWidget = widgets.formWidget(
            widgets.Toggle(),
            addInfoButton=True,
            stretchWidget=False,
            labelTextLeft="Create third segmentation: ",
            infoTxt=infoTxt,
        )
        entriesLayout.addFormWidget(self.createThirdSegmWidget, row=row)

        row += 1
        infoTxt = html_utils.paragraph("""
            Text to append at the end of the third segmentation file.<br><br>
            The third segmentation file is the result of <b>subtracting the 
            sub-cellular objects from the parent objects</b><br><br>
            This is useful if, for example, you need to compute measurements 
            only from the cytoplasm (i.e., the sub-cellular object is the nucleus).
        """)
        lineEdit = widgets.alphaNumericLineEdit()
        lineEdit.setText("difference")
        lineEdit.setAlignment(Qt.AlignCenter)
        self.appendTextWidget = widgets.formWidget(
            lineEdit,
            addInfoButton=True,
            labelTextLeft="Text to append: ",
            infoTxt=infoTxt,
        )
        entriesLayout.addFormWidget(self.appendTextWidget, row=row)
        self.appendTextWidget.setDisabled(True)

        self.createThirdSegmWidget.widget.toggled.connect(self.createThirdSegmToggled)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(entriesLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
        self.setFont(font)

    def createThirdSegmToggled(self, checked):
        self.appendTextWidget.setDisabled(not checked)

    def ok_cb(self):
        self.cancel = False
        if self.createThirdSegmWidget.widget.isChecked():
            if not self.appendTextWidget.widget.text():
                msg = widgets.myMessageBox(showCentered=False, wrapText=False)
                txt = html_utils.paragraph(
                    "When creating the third segmentation file, "
                    "<b>the name to append cannot be empty!</b>"
                )
                msg.critical(self, "Empty name", txt)
                return

        self.trackSubCellObjParams = {
            "how": self.optionsWidget.widget.currentText(),
            "IoA": self.IoAwidget.widget.value(),
            "createThirdSegm": self.createThirdSegmWidget.widget.isChecked(),
            "relabelSubObjLab": self.relabelSubObjLab.widget.isChecked(),
            "thirdSegmAppendedText": self.appendTextWidget.widget.text(),
        }
        self.close()


class CellACDCTrackerParamsWin(QDialog):
    def __init__(self, parent=None):
        self.cancel = True
        super().__init__(parent)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Cell-ACDC tracker parameters")

        paramsLayout = QGridLayout()
        paramsBox = QGroupBox()

        row = 0
        label = QLabel(html_utils.paragraph("Minimum overlap between objects"))
        paramsLayout.addWidget(label, row, 0)
        maxOverlapSpinbox = QDoubleSpinBox()
        maxOverlapSpinbox.setAlignment(Qt.AlignCenter)
        maxOverlapSpinbox.setMinimum(0)
        maxOverlapSpinbox.setMaximum(1)
        maxOverlapSpinbox.setSingleStep(0.1)
        maxOverlapSpinbox.setValue(0.4)
        self.maxOverlapSpinbox = maxOverlapSpinbox
        paramsLayout.addWidget(maxOverlapSpinbox, row, 1)
        infoButton = widgets.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        paramsLayout.addWidget(infoButton, row, 2)
        paramsLayout.setColumnStretch(0, 0)
        paramsLayout.setColumnStretch(1, 1)
        paramsLayout.setColumnStretch(2, 0)

        cancelButton = widgets.cancelPushButton("Cancel")
        okButton = widgets.okPushButton(" Ok ")
        cancelButton.clicked.connect(self.cancel_cb)
        okButton.clicked.connect(self.ok_cb)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        layout = QVBoxLayout()
        infoText = html_utils.paragraph("<b>Cell-ACDC tracker parameters</b>")
        infoLabel = QLabel(infoText)
        layout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        paramsBox.setLayout(paramsLayout)
        layout.addWidget(paramsBox)
        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setFont(font)

    def showInfo(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            "Cell-ACDC tracker computes the percentage of overlap between "
            "all the objects<br> at frame <code>n</code> and all the "
            "objects in previous frame <code>n-1</code>.<br><br>"
            "All objects with <b>overlap less than</b> "
            "<code>Minimum overlap between objects</code><br>are considered "
            "<b>new objects</b>.<br><br>"
            "Set this value to 0 if you want to force tracking of ALL the "
            "objects<br> in the previous frame (e.g., if cells move a lot "
            "between frames)"
        )
        msg.information(self, "Cell-ACDC tracker info", txt)

    def ok_cb(self, checked=False):
        self.cancel = False
        self.params = {"IoA_thresh": self.maxOverlapSpinbox.value()}
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        self.resize(int(self.width() * 1.3), self.height())
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class BayesianTrackerParamsWin(QDialog):
    def __init__(self, segmShape, parent=None, channels=None, currentChannelName=None):
        self.cancel = True
        super().__init__(parent)

        self.channels = channels
        self.currentChannelName = currentChannelName

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Bayesian tracker parameters")

        paramsLayout = QGridLayout()
        paramsBox = QGroupBox()

        row = 0
        this_path = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(
            this_path, "trackers", "BayesianTracker", "model", "cell_config.json"
        )
        label = QLabel(html_utils.paragraph("Model path"))
        paramsLayout.addWidget(label, row, 0)
        modelPathLineEdit = QLineEdit()
        start_dir = ""
        if os.path.exists(default_model_path):
            start_dir = os.path.dirname(default_model_path)
            modelPathLineEdit.setText(default_model_path)
        self.modelPathLineEdit = modelPathLineEdit
        paramsLayout.addWidget(modelPathLineEdit, row, 1)
        browseButton = widgets.browseFileButton(
            title="Select Bayesian Tracker model file",
            ext={"JSON Config": (".json",)},
            start_dir=start_dir,
        )
        browseButton.sigPathSelected.connect(self.onPathSelected)
        paramsLayout.addWidget(browseButton, row, 2, alignment=Qt.AlignLeft)

        if self.channels is not None:
            row += 1
            label = QLabel(html_utils.paragraph("Intensity image channel:  "))
            paramsLayout.addWidget(label, row, 0)
            items = ["None", *self.channels]
            self.channelCombobox = widgets.QCenteredComboBox()
            self.channelCombobox.addItems(items)
            paramsLayout.addWidget(self.channelCombobox, row, 1)
            if self.currentChannelName is not None:
                self.channelCombobox.setCurrentText(self.currentChannelName)

        row += 1
        label = QLabel(html_utils.paragraph("Features"))
        paramsLayout.addWidget(label, row, 0)
        selectFeaturesButton = widgets.setPushButton("Select features")
        paramsLayout.addWidget(selectFeaturesButton, row, 1)
        self.features = []
        selectFeaturesButton.clicked.connect(self.selectFeatures)

        row += 1
        label = QLabel(html_utils.paragraph("Verbose"))
        paramsLayout.addWidget(label, row, 0)
        verboseToggle = widgets.Toggle()
        verboseToggle.setChecked(True)
        self.verboseToggle = verboseToggle
        paramsLayout.addWidget(verboseToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph("Run optimizer"))
        paramsLayout.addWidget(label, row, 0)
        optimizeToggle = widgets.Toggle()
        optimizeToggle.setChecked(True)
        self.optimizeToggle = optimizeToggle
        paramsLayout.addWidget(optimizeToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph("Max search radius"))
        paramsLayout.addWidget(label, row, 0)
        maxSearchRadiusSpinbox = QSpinBox()
        maxSearchRadiusSpinbox.setAlignment(Qt.AlignCenter)
        maxSearchRadiusSpinbox.setMinimum(1)
        maxSearchRadiusSpinbox.setMaximum(2147483647)
        maxSearchRadiusSpinbox.setValue(50)
        self.maxSearchRadiusSpinbox = maxSearchRadiusSpinbox
        self.maxSearchRadiusSpinbox.setDisabled(True)
        paramsLayout.addWidget(maxSearchRadiusSpinbox, row, 1)

        row += 1
        Z, Y, X = segmShape
        label = QLabel(html_utils.paragraph("Tracking volume"))
        paramsLayout.addWidget(label, row, 0)
        volumeLineEdit = QLineEdit()
        defaultVol = f"  (0, {X}), (0, {Y})  "
        if Z > 1:
            defaultVol = f"{defaultVol}, (0, {Z})  "
        volumeLineEdit.setText(defaultVol)
        volumeLineEdit.setAlignment(Qt.AlignCenter)
        self.volumeLineEdit = volumeLineEdit
        paramsLayout.addWidget(volumeLineEdit, row, 1)

        row += 1
        label = QLabel(html_utils.paragraph("Interactive mode step size"))
        paramsLayout.addWidget(label, row, 0)
        stepSizeSpinbox = QSpinBox()
        stepSizeSpinbox.setAlignment(Qt.AlignCenter)
        stepSizeSpinbox.setMinimum(1)
        stepSizeSpinbox.setMaximum(2147483647)
        stepSizeSpinbox.setValue(100)
        self.stepSizeSpinbox = stepSizeSpinbox
        paramsLayout.addWidget(stepSizeSpinbox, row, 1)

        row += 1
        label = QLabel(html_utils.paragraph("Update method"))
        paramsLayout.addWidget(label, row, 0)
        updateMethodCombobox = QComboBox()
        updateMethodCombobox.addItems(["EXACT", "APPROXIMATE"])
        self.updateMethodCombobox = updateMethodCombobox
        self.updateMethodCombobox.currentTextChanged.connect(self.methodChanged)
        paramsLayout.addWidget(updateMethodCombobox, row, 1)

        cancelButton = widgets.cancelPushButton("Cancel")
        okButton = widgets.okPushButton(" Ok ")
        cancelButton.clicked.connect(self.cancel_cb)
        okButton.clicked.connect(self.ok_cb)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        layout = QVBoxLayout()
        infoText = html_utils.paragraph("<b>Bayesian Tracker parameters</b>")
        infoLabel = QLabel(infoText)
        layout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        paramsBox.setLayout(paramsLayout)
        layout.addWidget(paramsBox)

        url = "https://btrack.readthedocs.io/en/latest/index.html"
        moreInfoText = html_utils.paragraph(
            "<i>Find more info on the Bayesian Tracker's "
            f'<a href="{url}">home page</a></i>'
        )
        moreInfoLabel = QLabel(moreInfoText)
        moreInfoLabel.setOpenExternalLinks(True)
        layout.addWidget(moreInfoLabel, alignment=Qt.AlignCenter)

        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setFont(font)

    def selectFeatures(self):
        features = measurements.get_btrack_features()
        selectWin = widgets.QDialogListbox(
            "Select features",
            "Select features to use for tracking:\n",
            features,
            multiSelection=True,
            parent=self,
            includeSelectionHelp=True,
        )
        for i in range(selectWin.listBox.count()):
            item = selectWin.listBox.item(i)
            if item.text() in self.features:
                item.setSelected(True)
        selectWin.exec_()
        if selectWin.cancel:
            return
        self.features = selectWin.selectedItemsText

    def methodChanged(self, method):
        if method == "APPROXIMATE":
            self.maxSearchRadiusSpinbox.setDisabled(False)
        else:
            self.maxSearchRadiusSpinbox.setDisabled(True)

    def onPathSelected(self, path):
        self.modelPathLineEdit.setText(path)

    def ok_cb(self, checked=False):
        self.cancel = False
        try:
            m = re.findall(r"\((\d+), *(\d+)\)", self.volumeLineEdit.text())
            if len(m) < 2:
                raise
            self.volume = tuple([(int(start), int(end)) for start, end in m])
            if len(self.volume) == 2:
                self.volume = (self.volume[0], self.volume[1], (-1e5, 1e5))
        except Exception as e:
            self.warnNotAcceptedVolume()
            return

        if not os.path.exists(self.modelPathLineEdit.text()):
            self.warnNotVaidPath()
            return

        self.intensityImageChannel = None
        self.verbose = self.verboseToggle.isChecked()
        self.max_search_radius = self.maxSearchRadiusSpinbox.value()
        self.update_method = self.updateMethodCombobox.currentText()
        self.model_path = os.path.normpath(self.modelPathLineEdit.text())
        self.params = {
            "model_path": self.model_path,
            "verbose": self.verbose,
            "volume": self.volume,
            "max_search_radius": self.max_search_radius,
            "update_method": self.update_method,
            "step_size": self.stepSizeSpinbox.value(),
            "optimize": self.optimizeToggle.isChecked(),
            "features": self.features,
        }
        if self.channels is not None:
            if self.channelCombobox.currentText() != "None":
                self.intensityImageChannel = self.channelCombobox.currentText()
        self.close()

    def warnNotVaidPath(self):
        url = "https://github.com/lowe-lab-ucl/segment-classify-track/tree/main/models"
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            "The model configuration file path<br><br>"
            f"{self.modelPathLineEdit.text()}<br><br> "
            "does <b>not exist.</b><br><br>"
            "You can find some <b>pre-configured models</b> "
            f'<a href="{url}">here</a>.'
        )
        msg.critical(self, "Invalid volume", txt)

    def warnNotAcceptedVolume(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            f"{self.volumeLineEdit.text()} is <b>not a valid volume!</b><br><br>"
            "Valid volume is for example (0, 2048), (0, 2048)<br>"
            "for 2D segmentation or (0, 2048), (0, 2048), (0, 2048)<br>"
            "for 3D segmentation."
        )
        msg.critical(self, "Invalid volume", txt)

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        self.resize(int(self.width() * 1.3), self.height())
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class DeltaTrackerParamsWin(QDialog):
    def __init__(self, posData=None, parent=None):
        self.cancel = True
        super().__init__(parent)

        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Delta tracker parameters")

        paramsLayout = QGridLayout()
        paramsBox = QGroupBox()

        row = 0
        this_path = os.path.dirname(os.path.abspath(__file__))
        default_model_path = this_path

        label = QLabel(html_utils.paragraph("Original Images path"))
        paramsLayout.addWidget(label, row, 0)
        modelPathLineEdit = QLineEdit()
        start_dir = ""
        if os.path.exists(default_model_path):
            start_dir = os.path.dirname(default_model_path)
            modelPathLineEdit.setText(default_model_path)
        self.modelPathLineEdit = modelPathLineEdit
        paramsLayout.addWidget(modelPathLineEdit, row, 1)
        browseButton = widgets.browseFileButton(
            title="Select Original Images", ext={"TIFF": (".tif",)}, start_dir=start_dir
        )
        if posData is not None:
            modelPathLineEdit.setText(posData.imgPath)
        browseButton.sigPathSelected.connect(self.onPathSelected)
        paramsLayout.addWidget(browseButton, row, 2, alignment=Qt.AlignLeft)

        row += 1
        label = QLabel(html_utils.paragraph("Model Type"))
        paramsLayout.addWidget(label, row, 0)
        updateMethodCombobox = QComboBox()
        updateMethodCombobox.addItems(["2D", "mothermachine"])
        self.model_type = "2D"
        self.updateMethodCombobox = updateMethodCombobox
        self.updateMethodCombobox.currentTextChanged.connect(self.methodChanged)
        paramsLayout.addWidget(updateMethodCombobox, row, 1)

        row += 1
        label = QLabel(html_utils.paragraph("Single Mother Machine Chamber?"))
        paramsLayout.addWidget(label, row, 0)
        chamberToggle = widgets.Toggle()
        chamberToggle.setChecked(True)
        self.chamberToggle = chamberToggle
        paramsLayout.addWidget(chamberToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph("Verbose"))
        paramsLayout.addWidget(label, row, 0)
        verboseToggle = widgets.Toggle()
        verboseToggle.setChecked(True)
        self.verboseToggle = verboseToggle
        paramsLayout.addWidget(verboseToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph("Legacy Save (.mat)"))
        paramsLayout.addWidget(label, row, 0)
        legacyToggle = widgets.Toggle()
        legacyToggle.setChecked(False)
        self.legacyToggle = legacyToggle
        paramsLayout.addWidget(legacyToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph("Pickle (.pkl)"))
        paramsLayout.addWidget(label, row, 0)
        pickleToggle = widgets.Toggle()
        pickleToggle.setChecked(False)
        self.pickleToggle = pickleToggle
        paramsLayout.addWidget(pickleToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph("Movie (.mp4) *only for 2D images"))
        paramsLayout.addWidget(label, row, 0)
        movieToggle = widgets.Toggle()
        movieToggle.setChecked(False)
        self.movieToggle = movieToggle
        paramsLayout.addWidget(movieToggle, row, 1, alignment=Qt.AlignCenter)

        cancelButton = widgets.cancelPushButton("Cancel")
        okButton = widgets.okPushButton(" Ok ")
        cancelButton.clicked.connect(self.cancel_cb)
        okButton.clicked.connect(self.ok_cb)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        layout = QVBoxLayout()
        infoText = html_utils.paragraph("<b>Delta Tracker parameters</b>")
        infoLabel = QLabel(infoText)
        layout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        paramsBox.setLayout(paramsLayout)
        layout.addWidget(paramsBox)

        url = "https://delta.readthedocs.io/en/latest/"
        moreInfoText = html_utils.paragraph(
            f'<i>Find more info on Delta Tracker\'s <a href="{url}">home page</a></i>'
        )
        moreInfoLabel = QLabel(moreInfoText)
        moreInfoLabel.setOpenExternalLinks(True)
        layout.addWidget(moreInfoLabel, alignment=Qt.AlignCenter)

        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setFont(font)

    def methodChanged(self, method):
        if method == "mothermachine":
            self.model_type = "mothermachine"

    def onPathSelected(self, path):
        self.modelPathLineEdit.setText(path)

    def ok_cb(self, checked=False):
        self.cancel = False

        if not os.path.exists(self.modelPathLineEdit.text()):
            self.warnNotVaidPath()
            return

        self.verbose = self.verboseToggle.isChecked()
        self.legacy = self.legacyToggle.isChecked()
        self.pickle = self.pickleToggle.isChecked()
        self.movie = self.movieToggle.isChecked()
        self.chamber = self.chamberToggle.isChecked()
        self.model_path = os.path.normpath(self.modelPathLineEdit.text())
        self.params = {
            "original_images_path": self.model_path,
            "verbose": self.verbose,
            "legacy": self.legacy,
            "pickle": self.pickle,
            "movie": self.movie,
            "model_type": self.model_type,
            "single mothermachine chamber": self.chamber,
        }
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        self.resize(int(self.width() * 1.3), self.height())
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class GenerateMotherBudTotalTableSelectColumnsDialog(QBaseDialog):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Select columns to combine into the output table")

        self.cancel = True

        self.columns = core.natsort_acdc_columns(df.columns)
        self.operations = (
            "Sum mother and bud",
            "Copy column from mother",
        )

        self.mainLayout = QVBoxLayout()

        instructionsText = html_utils.paragraph("""
            <b>Select which columns</b> and <b>how</b> you want to combine them 
            into the output table.<br>
        """)
        self.mainLayout.addWidget(QLabel(instructionsText))

        settingsLayout = QGridLayout()

        row = 0
        settingsLayout.addWidget(widgets.QHLine(), row, 0, 1, 2)

        row += 1
        settingsLayout.addWidget(
            QLabel("Copy all non-selected columns from mother cell"), row, 0
        )
        self.copyAllColsToggle = widgets.Toggle()
        settingsLayout.addWidget(self.copyAllColsToggle, row, 1, alignment=Qt.AlignLeft)

        row += 1
        settingsLayout.addWidget(widgets.QHLine(), row, 0, 1, 2)

        self.mainLayout.addLayout(settingsLayout)

        scrollArea = widgets.ScrollArea()
        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scrollWidget = QWidget()
        scrollArea.setWidget(scrollWidget)
        self.centralLayout = QGridLayout()
        scrollWidget.setLayout(self.centralLayout)

        self.centralLayout.addWidget(QLabel("Grouping columns"), 0, 0)
        self.centralLayout.addWidget(QLabel("Column"), 0, 1)
        self.centralLayout.addWidget(QLabel("Operation"), 0, 2)
        self.centralLayout.setRowStretch(0, 0)

        self.groupingColsListWidget = widgets.listWidget(
            isMultipleSelection=True,
        )
        self.groupingColsListWidget.addItems(self.columns)
        self.centralLayout.addWidget(self.groupingColsListWidget, 1, 0, 2, 1)

        selector = widgets.ComboBox(self)
        selector.addItems(self.columns)
        operationCombobox = widgets.ComboBox(self)
        operationCombobox.addItems(self.operations)
        self.addSelectorButton = widgets.addPushButton()

        dummyButton = widgets.delPushButton()
        dummyButton.setRetainSizeWhenHidden(True)
        dummyButton.hide()
        self.centralLayout.addWidget(dummyButton, 1, 4)

        self.centralLayout.addWidget(selector, 1, 1)
        self.centralLayout.addWidget(operationCombobox, 1, 2)
        self.centralLayout.addWidget(self.addSelectorButton, 1, 3)

        self.centralLayout.setRowStretch(1, 1)
        self.centralLayout.setRowStretch(2, 1)

        self.selectors = {1: (selector, operationCombobox)}

        buttonsLayout = widgets.CancelOkButtonsLayout()

        saveSelectionButton = widgets.savePushButton("Save current selection")
        buttonsLayout.insertWidget(3, saveSelectionButton)

        loadDefaultColsButton = widgets.reloadPushButton(
            "Load default summable columns"
        )
        buttonsLayout.insertWidget(4, loadDefaultColsButton)

        loadPreviousSelButton = widgets.OpenFilePushButton("Load previous selection")
        buttonsLayout.insertWidget(5, loadPreviousSelButton)

        saveSelectionButton.clicked.connect(self.saveSelection)
        loadDefaultColsButton.clicked.connect(self.loadDefaultCols)
        loadPreviousSelButton.clicked.connect(self.loadPreviousSelection)
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addWidget(scrollArea)
        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)

        self.addSelectorButton.clicked.connect(self.addSelector)
        selector.currentTextChanged.connect(self.selectorTextChanged)

        self.setLayout(self.mainLayout)
        self.setFont(font)

    def saveSelection(self):
        saved_selections = io.get_saved_moth_bud_tot_selections()
        existing_names = set(saved_selections.keys())
        win = filenameDialog(
            basename="",
            ext="",
            hintText="Insert a <b>name</b> for the current selection:",
            existingNames=existing_names,
            allowEmpty=False,
            defaultEntry="mother_bud_total_columns_selection",
        )
        win.exec_()
        if win.cancel:
            return

        name = win.filename
        saved_selections[name] = self.selectedOptions()
        io.save_moth_bud_tot_selected_options(saved_selections)

        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph(f"""
            Current selection saved with name <code>{name}</code>.
        """)
        msg.information(self, "Selection saved", txt)

    def loadDefaultCols(self):
        from . import single_pos_index_cols

        grouping_cols = [col for col in single_pos_index_cols if col in self.columns]
        self.groupingColsListWidget.setSelectedItems(grouping_cols)

        column_operation_mapper = {
            col: "Sum mother and bud" for col in cca_functions.default_summable_columns
        }
        column_operation_mapper = {
            col: op
            for col, op in column_operation_mapper.items()
            if col in self.columns and op in self.operations
        }
        self.addSelectors(
            len(column_operation_mapper),
            callback_on_finished=partial(
                self.setSelectorValues, column_operation_mapper
            ),
        )

    def loadPreviousSelection(self):
        saved_selections = io.get_saved_moth_bud_tot_selections()
        if not saved_selections:
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            txt = html_utils.paragraph("""
                There are no saved selections.
            """)
            msg.warning(self, "No saved selections", txt)
            return

        existing_names = natsorted(saved_selections.keys(), key=str.casefold)

        selectNameWin = widgets.QDialogListbox(
            "Choose selection to load",
            "Choose selection to load:\n",
            existing_names,
            multiSelection=False,
            parent=self,
        )
        selectNameWin.exec_()
        if selectNameWin.cancel:
            return

        self.loadOptions(saved_selections[selectNameWin.selectedItemsText[0]])

    def resetSelectors(self, callback_on_finished=None):
        self.callback_on_finished = callback_on_finished
        QTimer.singleShot(1, self._removeLastSelector)

    def _removeLastSelector(self):
        if len(self.selectors) == 1:
            if self.callback_on_finished is not None:
                self.callback_on_finished()
            return

        lastRow = max(self.selectors.keys())
        lastSelector, _ = self.selectors[lastRow]
        self.removeSelector(sender=lastSelector.delButton)
        QTimer.singleShot(1, self._removeLastSelector)

    def addSelectors(self, number, callback_on_finished=None):
        self.callback_on_finished = callback_on_finished
        QTimer.singleShot(1, partial(self._addSelectorRecursive, number))

    def _addSelectorRecursive(self, number):
        if len(self.selectors) == number:
            if self.callback_on_finished is not None:
                self.callback_on_finished()
            return

        self.addSelector()
        QTimer.singleShot(1, partial(self._addSelectorRecursive, number))

    def loadOptions(self, options: dict):
        if len(self.selectors) > 1:
            self.resetSelectors(callback_on_finished=partial(self.loadOptions, options))
            return

        self.copyAllColsToggle.setChecked(
            options.get("do_copy_all_nonselected_columns", False)
        )
        self.groupingColsListWidget.setSelectedItems(
            options.get("grouping_columns", [])
        )
        column_operation_mapper = options.get("column_operation_mapper", {})
        column_operation_mapper = {
            col: op
            for col, op in column_operation_mapper.items()
            if col in self.columns and op in self.operations
        }
        if len(column_operation_mapper) > 1:
            self.addSelectors(
                len(column_operation_mapper),
                callback_on_finished=partial(
                    self.setSelectorValues, column_operation_mapper
                ),
            )
            return

        self.setSelectorValues(column_operation_mapper)

    def setSelectorValues(self, column_operation_mapper):
        for i, (col, op) in enumerate(column_operation_mapper.items()):
            selector, operationCombobox = self.selectors[i + 1]
            selector.setCurrentText(col)
            operationCombobox.setCurrentText(op)

    def resetSelectorsStyles(self):
        for selector, _ in self.selectors.values():
            selector.setStyleSheet("")

    def selectorTextChanged(self, text):
        self.resetSelectorsStyles()
        selector = self.sender()
        for other_selector, _ in self.selectors.values():
            if other_selector == selector:
                continue

            if selector.currentText() != other_selector.currentText():
                continue

            self.setWarningStyleSelector(selector)
            self.setWarningStyleSelector(other_selector)

    def addSelector(self):
        row = len(self.selectors) + 1

        selector = widgets.ComboBox(self)
        selector.addItems(self.columns)
        selector.setCurrentIndex(len(self.selectors))
        operationCombobox = widgets.ComboBox(self)
        operationCombobox.addItems(self.operations)
        delButton = widgets.delPushButton()
        selector.delButton = delButton
        delButton._row = row

        self.selectors[row] = (selector, operationCombobox)

        self.centralLayout.addWidget(selector, row, 1)
        self.centralLayout.addWidget(operationCombobox, row, 2)
        self.centralLayout.addWidget(delButton, row, 3)

        self.centralLayout.removeWidget(self.addSelectorButton)
        self.centralLayout.addWidget(self.addSelectorButton, row, 4)

        delButton.clicked.connect(self.removeSelector)

        self.centralLayout.removeWidget(self.groupingColsListWidget)
        rowSpan = self.centralLayout.rowCount()
        self.centralLayout.addWidget(self.groupingColsListWidget, 1, 0, rowSpan, 1)
        self.centralLayout.setRowStretch(rowSpan, 1)

        selector.currentTextChanged.connect(self.selectorTextChanged)

    def removeSelector(self, checked=False, sender=None):
        if sender is None:
            delButton = self.sender()
        else:
            delButton = sender

        selector, operationCombobox = self.selectors.pop(delButton._row)

        self.centralLayout.removeWidget(selector)
        self.centralLayout.removeWidget(operationCombobox)
        self.centralLayout.removeWidget(delButton)

        resorted_selectors = {}
        for i, (row, (sel, op)) in enumerate(self.selectors.items()):
            if i == 0:
                resorted_selectors[i + 1] = (sel, op)
                continue

            delButton = sel.delButton
            delButton._row = i + 1
            self.centralLayout.removeWidget(sel)
            self.centralLayout.removeWidget(op)
            self.centralLayout.removeWidget(delButton)
            self.centralLayout.addWidget(sel, i + 1, 1)
            self.centralLayout.addWidget(op, i + 1, 2)
            self.centralLayout.addWidget(delButton, i + 1, 3)

            resorted_selectors[i + 1] = (sel, op)

        last_row = i + 1
        col = 4 if last_row > 1 else 3
        self.centralLayout.removeWidget(self.addSelectorButton)
        self.centralLayout.addWidget(self.addSelectorButton, i + 1, col)

        self.selectors = resorted_selectors

    def sizeHint(self):
        width = super().sizeHint().width()
        height = super().sizeHint().height()
        groupingColsWidth = widgets.get_min_width_for_no_scrollbar(
            self.groupingColsListWidget
        )
        width += groupingColsWidth
        return QSize(width, height)

    def checkDuplicatedSelectedColumns(self):
        for selector, _ in self.selectors.values():
            selector.setStyleSheet("background-color: none")
            for other_selector, _ in self.selectors.values():
                if other_selector == selector:
                    continue

                if other_selector.currentText() != selector.currentText():
                    continue

                self.warnDuplicatedSelectedColumns(selector, other_selector)
                return False

        return True

    def setWarningStyleSelector(self, selector):
        popup = selector.view()
        palette = popup.palette()
        text_color = palette.color(palette.ColorRole.Text)
        warningStyleSheet = f"""
            QComboBox {{
                color: black;
                background-color: orange;  /* main area */
            }}
            QComboBox QAbstractItemView {{
                background-color: {text_color.name()};
            }}
        """
        selector.setStyleSheet(warningStyleSheet)

    def warnDuplicatedSelectedColumns(self, selector1, selector2):
        self.setWarningStyleSelector(selector1)
        self.setWarningStyleSelector(selector2)

        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph(f"""
            The following column has been selected more than once 
            (highlighted in orange).<br><br>
            <code>{selector1.currentText()}</code><br><br>
            Please, select each column only once.<br><br>
            Thank you for your patience!
        """)
        msg.warning(self, "Duplicated selection", txt)

    def checkGroupingColumnsNotSelected(self):
        if self.groupingColsListWidget.selectedItems():
            return True

        return self.warnGroupingColumnsNotSelected()

    def warnGroupingColumnsNotSelected(self):
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph(f"""
            Are you sure you do not want to select any grouping column?<br><br>
            Grouping columns are those needed to identify each unique 
            Position folder.
        """)
        _, noButton, yesButton = msg.question(
            self,
            "No grouping columns selected?",
            txt,
            buttonsTexts=(
                "Cancel",
                "No, let me select grouping columns",
                "Yes, I do not need grouping columns",
            ),
        )
        return msg.clickedButton == yesButton

    def selectedOptions(self):
        selected_options = {
            "grouping_columns": self.groupingColsListWidget.selectedItemsText(),
            "column_operation_mapper": {
                selector.currentText(): operationCombobox.currentText()
                for selector, operationCombobox in self.selectors.values()
            },
            "do_copy_all_nonselected_columns": self.copyAllColsToggle.isChecked(),
        }
        return selected_options

    def ok_cb(self):
        proceed = self.checkDuplicatedSelectedColumns()
        if not proceed:
            return

        proceed = self.checkGroupingColumnsNotSelected()
        if not proceed:
            return

        self.selected_options = self.selectedOptions()

        self.cancel = False
        self.close()


class ApplyTrackTableSelectColumnsDialog(QBaseDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Select columns containing tracking info")

        self.cancel = True
        self.mainLayout = QVBoxLayout()

        options = (
            '"Frame index", "Tracked IDs" and "Segmentation mask IDs"<br>',
            '"Frame index", "Tracked IDs", "X coord. centroid", and "Y coord. centroid"',
        )
        self.instructionsText = html_utils.paragraph(
            f"""
            <b>Select which columns</b> contain the tracking information.<br><br>
            You must choose one of the following combinations:<br> 
            {html_utils.to_list(options)}
            Optionally, you can provide the column name containing the parent ID.<br>
            This will allow you to load lineage information into Cell-ACDC. 
            """
        )
        self.mainLayout.addWidget(QLabel(self.instructionsText))

        formLayout = QFormLayout()

        self.frameIndexCombobox = widgets.QCenteredComboBox()
        self.frameIndexCombobox.addItems(df.columns)
        self.frameIndexCheckbox = QCheckBox("1st frame is index 1")
        frameIndexLayout = QHBoxLayout()
        frameIndexLayout.addWidget(self.frameIndexCombobox)
        frameIndexLayout.addWidget(self.frameIndexCheckbox)
        frameIndexLayout.setStretch(0, 2)
        frameIndexLayout.setStretch(1, 0)
        formLayout.addRow("Frame index: ", frameIndexLayout)

        self.trackedIDsCombobox = widgets.QCenteredComboBox()
        self.trackedIDsCombobox.addItems(df.columns)
        formLayout.addRow("Tracked IDs: ", self.trackedIDsCombobox)

        items = df.columns.to_list()
        items.insert(0, "None")
        self.maskIDsCombobox = widgets.QCenteredComboBox()
        self.maskIDsCombobox.addItems(items)
        formLayout.addRow("Segmentation mask IDs: ", self.maskIDsCombobox)

        self.xCentroidCombobox = widgets.QCenteredComboBox()
        self.xCentroidCombobox.addItems(items)
        formLayout.addRow("X coord. centroid: ", self.xCentroidCombobox)

        self.yCentroidCombobox = widgets.QCenteredComboBox()
        self.yCentroidCombobox.addItems(items)
        formLayout.addRow("Y coord. centroid: ", self.yCentroidCombobox)

        self.parentIDcombobox = widgets.QCenteredComboBox()
        self.parentIDcombobox.addItems(items)
        formLayout.addRow("Parent ID (optional): ", self.parentIDcombobox)

        deleteUntrackedLayout = QHBoxLayout()
        self.deleteUntrackedIDsToggle = widgets.Toggle()
        deleteUntrackedLayout.addStretch(1)
        deleteUntrackedLayout.addWidget(self.deleteUntrackedIDsToggle)
        deleteUntrackedLayout.addStretch(1)
        formLayout.addRow("Delete untracked IDs: ", deleteUntrackedLayout)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addSpacing(30)
        self.mainLayout.addLayout(formLayout)
        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)

        self.setLayout(self.mainLayout)
        self.setFont(font)

    def ok_cb(self):
        self.cancel = False
        self.frameIndexCol = self.frameIndexCombobox.currentText()
        self.trackedIDsCol = self.trackedIDsCombobox.currentText()
        self.maskIDsCol = self.maskIDsCombobox.currentText()
        self.xCentroidCol = self.xCentroidCombobox.currentText()
        self.yCentroidCol = self.yCentroidCombobox.currentText()
        self.deleteUntrackedIDs = self.deleteUntrackedIDsToggle.isChecked()
        if self.maskIDsCol == "None":
            if self.xCentroidCol == "None" or self.yCentroidCol == "None":
                self.warnInvalidSelection()
                return
        else:
            self.xCentroidCol = "None"
            self.yCentroidCol = "None"
        self.parentIDcol = self.parentIDcombobox.currentText()
        self.isFirstFrameOne = self.frameIndexCheckbox.isChecked()
        self.close()

    def warnInvalidSelection(self):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.warning(
            self,
            "Invalid selection",
            html_utils.paragraph(
                f"<b>Invalid selection</b><br> {self.instructionsText}"
            ),
        )


class editCcaTableWidget(QDialog):
    sigApplyChangesFutureFrames = Signal(object, int)

    def __init__(
        self,
        cca_df,
        SizeT,
        title="Edit cell cycle annotations",
        parent=None,
        current_frame_i=0,
    ):
        self.inputCca_df = cca_df
        self.cancel = True
        self.SizeT = SizeT
        self.cca_df = None
        self.current_frame_i = current_frame_i

        super().__init__(parent)
        self.setWindowTitle(title)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        # Layouts
        mainLayout = QVBoxLayout()
        headerLayout = QGridLayout()
        tableLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()
        self.scrollArea = QScrollArea()
        self.viewBox = QWidget()

        # Header labels
        col = 0
        row = 0
        IDsLabel = QLabel("Cell ID")
        AC = Qt.AlignCenter
        IDsLabel.setAlignment(AC)
        headerLayout.addWidget(IDsLabel, 0, col, alignment=AC)

        col += 1
        ccsLabel = QLabel("Cell cycle stage")
        ccsLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(ccsLabel, 0, col, alignment=AC)

        col += 1
        relIDLabel = QLabel("Relative ID")
        relIDLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(relIDLabel, 0, col, alignment=AC)

        col += 1
        genNumLabel = QLabel("Generation number")
        genNumLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(genNumLabel, 0, col, alignment=AC)
        genNumColWidth = genNumLabel.sizeHint().width()

        col += 1
        relationshipLabel = QLabel("Relationship")
        relationshipLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(relationshipLabel, 0, col, alignment=AC)

        col += 1
        emergFrameLabel = QLabel("Emerging frame num.")
        emergFrameLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(emergFrameLabel, 0, col, alignment=AC)

        col += 1
        divitionFrameLabel = QLabel("Division frame num.")
        divitionFrameLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(divitionFrameLabel, 0, col, alignment=AC)

        col += 1
        historyKnownLabel = QLabel("Is history known?")
        historyKnownLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(historyKnownLabel, 0, col, alignment=AC)

        self.headerLayout = headerLayout

        tableLayout.setHorizontalSpacing(20)
        self.tableLayout = tableLayout

        # Add buttons
        cancelButton = widgets.cancelPushButton("Cancel")
        moreInfoButton = widgets.helpPushButton("More info...")
        moreInfoButton.setIcon(QIcon(":info.svg"))
        applyToFutureFramesbutton = widgets.futurePushButton(
            "Apply changes to future frames..."
        )
        okButton = widgets.okPushButton("Ok")

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(moreInfoButton)
        buttonsLayout.addWidget(applyToFutureFramesbutton)
        buttonsLayout.addWidget(okButton)

        # Scroll area properties
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setFrameStyle(QFrame.Shape.NoFrame)
        self.scrollArea.setWidgetResizable(True)

        # Add layouts
        self.viewBox.setLayout(tableLayout)
        self.scrollArea.setWidget(self.viewBox)
        mainLayout.addLayout(headerLayout)
        mainLayout.addWidget(self.scrollArea)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        # Populate table Layout
        IDs = cca_df.index
        self.IDs = IDs.to_list()
        relIDsOptions = [str(ID) for ID in IDs]
        relIDsOptions.insert(0, "-1")
        self.IDlabels = []
        self.ccsComboBoxes = []
        self.genNumSpinBoxes = []
        self.relIDComboBoxes = []
        self.relationshipComboBoxes = []
        self.emergFrameSpinBoxes = []
        self.divisFrameSpinBoxes = []
        self.emergFrameSpinPrevValues = []
        self.divisFrameSpinPrevValues = []
        self.historyKnownCheckBoxes = []
        for row, ID in enumerate(IDs):
            col = 0
            IDlabel = QLabel(f"{ID}")
            IDlabel.setAlignment(Qt.AlignCenter)
            tableLayout.addWidget(IDlabel, row + 1, col, alignment=AC)
            self.IDlabels.append(IDlabel)

            col += 1
            ccsComboBox = QComboBox()
            ccsComboBox.setFocusPolicy(Qt.StrongFocus)
            ccsComboBox.installEventFilter(self)
            ccsComboBox.addItems(["G1", "S/G2/M"])
            ccsValue = cca_df.at[ID, "cell_cycle_stage"]
            if ccsValue == "S":
                ccsValue = "S/G2/M"

            try:
                ccsComboBox.setCurrentText(ccsValue)
            except Exception as err:
                printl(ccsValue)
                printl(cca_df)
                raise err
            tableLayout.addWidget(ccsComboBox, row + 1, col, alignment=AC)
            self.ccsComboBoxes.append(ccsComboBox)
            ccsComboBox.activated.connect(self.clearComboboxFocus)

            col += 1
            relIDComboBox = QComboBox()
            relIDComboBox.setFocusPolicy(Qt.StrongFocus)
            relIDComboBox.installEventFilter(self)
            relIDComboBox.addItems(relIDsOptions)
            relIDComboBox.setCurrentText(str(cca_df.at[ID, "relative_ID"]))
            tableLayout.addWidget(relIDComboBox, row + 1, col)
            self.relIDComboBoxes.append(relIDComboBox)
            relIDComboBox.currentIndexChanged.connect(self.setRelID)
            relIDComboBox.activated.connect(self.clearComboboxFocus)

            col += 1
            genNumSpinBox = widgets.SpinBox()
            genNumSpinBox.setFocusPolicy(Qt.StrongFocus)
            genNumSpinBox.installEventFilter(self)
            genNumSpinBox.setValue(2)
            genNumSpinBox.setMaximum(2147483647)
            genNumSpinBox.setAlignment(Qt.AlignCenter)
            genNumSpinBox.setFixedWidth(int(genNumColWidth * 2 / 3))
            genNumSpinBox.setValue(int(cca_df.at[ID, "generation_num"]))
            tableLayout.addWidget(genNumSpinBox, row + 1, col, alignment=AC)
            self.genNumSpinBoxes.append(genNumSpinBox)

            col += 1
            relationshipComboBox = QComboBox()
            relationshipComboBox.setFocusPolicy(Qt.StrongFocus)
            relationshipComboBox.installEventFilter(self)
            relationshipComboBox.addItems(["mother", "bud"])
            relationshipComboBox.setCurrentText(str(cca_df.at[ID, "relationship"]))
            tableLayout.addWidget(relationshipComboBox, row + 1, col)
            self.relationshipComboBoxes.append(relationshipComboBox)
            relationshipComboBox.currentIndexChanged.connect(
                self.relationshipChanged_cb
            )
            relationshipComboBox.activated.connect(self.clearComboboxFocus)

            col += 1
            emergFrameSpinBox = widgets.SpinBox()
            emergFrameSpinBox.setFocusPolicy(Qt.StrongFocus)
            emergFrameSpinBox.installEventFilter(self)
            emergFrameSpinBox.setMaximum(SizeT)
            emergFrameSpinBox.setMinimum(-1)
            emergFrameSpinBox.setValue(-1)
            emergFrameSpinBox.setAlignment(Qt.AlignCenter)
            emergFrameSpinBox.setFixedWidth(int(genNumColWidth * 2 / 3))
            emergFrame_i = cca_df.at[ID, "emerg_frame_i"]
            val = emergFrame_i + 1 if emergFrame_i >= 0 else -1
            emergFrameSpinBox.setValue(val)
            tableLayout.addWidget(emergFrameSpinBox, row + 1, col, alignment=AC)
            self.emergFrameSpinBoxes.append(emergFrameSpinBox)
            self.emergFrameSpinPrevValues.append(emergFrameSpinBox.value())
            emergFrameSpinBox.valueChanged.connect(self.skip0emergFrame)

            col += 1
            divisFrameSpinBox = widgets.SpinBox()
            divisFrameSpinBox.setFocusPolicy(Qt.StrongFocus)
            divisFrameSpinBox.installEventFilter(self)
            divisFrameSpinBox.setMinimum(-1)
            divisFrameSpinBox.setMaximum(SizeT)
            divisFrameSpinBox.setValue(-1)
            divisFrameSpinBox.setAlignment(Qt.AlignCenter)
            divisFrameSpinBox.setFixedWidth(int(genNumColWidth * 2 / 3))
            divisFrame_i = int(cca_df.at[ID, "division_frame_i"])
            val = divisFrame_i + 1 if divisFrame_i >= 0 else -1
            divisFrameSpinBox.setValue(val)
            tableLayout.addWidget(divisFrameSpinBox, row + 1, col, alignment=AC)
            self.divisFrameSpinBoxes.append(divisFrameSpinBox)
            self.divisFrameSpinPrevValues.append(divisFrameSpinBox.value())
            divisFrameSpinBox.valueChanged.connect(self.skip0divisFrame)

            col += 1
            HistoryCheckBox = QCheckBox()
            HistoryCheckBox.setChecked(bool(cca_df.at[ID, "is_history_known"]))
            tableLayout.addWidget(HistoryCheckBox, row + 1, col, alignment=AC)
            self.historyKnownCheckBoxes.append(HistoryCheckBox)

        self.setLayout(mainLayout)

        # Connect to events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)
        moreInfoButton.clicked.connect(self.moreInfo)
        applyToFutureFramesbutton.clicked.connect(self.applyToFutureFrames)

        # self.setModal(True)

    def getChanges(self):
        newCcaDf = self.getCca_df()
        changes = {}
        for row in newCcaDf.itertuples():
            ID = row.Index
            for col in newCcaDf.columns:
                inputValue = self.inputCca_df.at[ID, col]
                newValue = getattr(row, col)
                if newValue == inputValue:
                    continue

                if ID not in changes:
                    changes[ID] = {col: (inputValue, newValue)}
                else:
                    changes[ID][col] = (inputValue, newValue)
        return changes

    def applyToFutureFrames(self):
        txt = "Enter <b>up to which frame</b> you want to apply the changes<br>"
        win = NumericEntryDialog(
            title="Stop frame",
            instructions=txt,
            parent=self,
            minValue=1,
            maxValue=self.SizeT,
            currentValue=self.current_frame_i,
        )
        win.exec_()
        if win.cancel:
            return

        stop_frame_i = win.value
        changes = self.getChanges()
        changes_format = myutils.format_cca_manual_changes(changes)
        detailsText = (
            f"Changes that will be applied from frame n. {self.current_frame_i + 1}"
            f" to frame n. {stop_frame_i + 1}:\n\n{changes_format}"
        )
        txt = html_utils.paragraph("""
Use this feature with <b>caution</b>!<br><br>
Before propagating to future frames <b>carefully inspect what changes</b> 
will be applied (see below).<br><br>
""")
        msg = widgets.myMessageBox(wrapText=False)
        msg.setDetailedText(detailsText, visible=True)
        msg.warning(self, "Caution!", txt, buttonsTexts=("Yes, I am sure", "Cancel"))
        if msg.cancel:
            return

        self.sigApplyChangesFutureFrames.emit(changes, stop_frame_i)

    def moreInfo(self, checked=True):
        desc = myutils.get_cca_colname_desc()
        msg = widgets.myMessageBox(parent=self)
        msg.setWindowTitle("Cell cycle annotations info")
        msg.setWidth(400)
        msg.setIcon()
        for col, txt in desc.items():
            msg.addText(html_utils.paragraph(f"<b>{col}</b>: {txt}"))
        msg.addButton("  Ok  ")
        msg.exec_()

    def setRelID(self, itemIndex):
        idx = self.relIDComboBoxes.index(self.sender())
        relID = self.sender().currentText()
        IDofRelID = self.IDs[idx]
        relIDidx = self.IDs.index(int(relID))
        relIDComboBox = self.relIDComboBoxes[relIDidx]
        relIDComboBox.setCurrentText(str(IDofRelID))

    def skip0emergFrame(self, value):
        idx = self.emergFrameSpinBoxes.index(self.sender())
        prevVal = self.emergFrameSpinPrevValues[idx]
        if value == 0 and value > prevVal:
            self.sender().setValue(1)
            self.emergFrameSpinPrevValues[idx] = 1
        elif value == 0 and value < prevVal:
            self.sender().setValue(-1)
            self.emergFrameSpinPrevValues[idx] = -1

    def skip0divisFrame(self, value):
        idx = self.divisFrameSpinBoxes.index(self.sender())
        prevVal = self.divisFrameSpinPrevValues[idx]
        if value == 0 and value > prevVal:
            self.sender().setValue(1)
            self.divisFrameSpinPrevValues[idx] = 1
        elif value == 0 and value < prevVal:
            self.sender().setValue(-1)
            self.divisFrameSpinPrevValues[idx] = -1

    def relationshipChanged_cb(self, itemIndex):
        idx = self.relationshipComboBoxes.index(self.sender())
        ccs = self.sender().currentText()
        if ccs == "bud":
            self.ccsComboBoxes[idx].setCurrentText("S/G2/M")
            self.genNumSpinBoxes[idx].setValue(0)

    def getCca_df(self):
        ccsValues = [var.currentText() for var in self.ccsComboBoxes]
        ccsValues = [val if val == "G1" else "S" for val in ccsValues]
        genNumValues = [var.value() for var in self.genNumSpinBoxes]
        relIDValues = [int(var.currentText()) for var in self.relIDComboBoxes]
        relatValues = [var.currentText() for var in self.relationshipComboBoxes]
        emergFrameValues = [
            var.value() - 1 if var.value() > 0 else -1
            for var in self.emergFrameSpinBoxes
        ]
        divisFrameValues = [
            var.value() - 1 if var.value() > 0 else -1
            for var in self.divisFrameSpinBoxes
        ]
        historyValues = [var.isChecked() for var in self.historyKnownCheckBoxes]
        check_rel = [ID == relID for ID, relID in zip(self.IDs, relIDValues)]

        # Buds in S phase must have 0 as number of cycles
        check_buds_S = [
            ccs == "S" and rel_ship == "bud" and not numc == 0
            for ccs, rel_ship, numc in zip(ccsValues, relatValues, genNumValues)
        ]

        # Mother cells must have at least 1 as number of cycles if history known
        check_mothers = [
            rel_ship == "mother" and not numc >= 1 if is_history_known else False
            for rel_ship, numc, is_history_known in zip(
                relatValues, genNumValues, historyValues
            )
        ]

        # Buds cannot be in G1
        check_buds_G1 = [
            ccs == "G1" and rel_ship == "bud"
            for ccs, rel_ship in zip(ccsValues, relatValues)
        ]

        # The number of cells in S phase must be half mothers and half buds
        num_moth_S = len(
            [
                0
                for ccs, rel_ship in zip(ccsValues, relatValues)
                if ccs == "S" and rel_ship == "mother"
            ]
        )
        num_bud_S = len(
            [
                0
                for ccs, rel_ship in zip(ccsValues, relatValues)
                if ccs == "S" and rel_ship == "bud"
            ]
        )

        # Cells in S phase cannot have -1 as relative's ID
        check_relID_S = [
            ccs == "S" and relID == -1 for ccs, relID in zip(ccsValues, relIDValues)
        ]

        # Mother cells with unknown history at emergence is recommended to have
        # generation number = 2 (easier downstream analysis)
        check_unknown_mothers = [
            rel_ship == "mother"
            and not is_history_known
            and gen_num != 2
            and (emerg_frame_i == self.current_frame_i or self.current_frame_i == 0)
            for rel_ship, is_history_known, gen_num, emerg_frame_i in zip(
                relatValues, historyValues, genNumValues, emergFrameValues
            )
        ]

        if any(check_rel):
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(""" 
                Some cells are mother or bud of itself!<br><br>
                Make sure that the relative ID is different from the Cell ID.
            """)
            msg.critical(self, "Some IDs are equal to relative ID", txt)
            return None
        elif any(check_unknown_mothers):
            txt = html_utils.paragraph("""
                We recommend to set <b>generation number to 2 for mother cells 
                with unknown history<br>
                that just appeared</b> (i.e., first cell cycle in the video).<br><br>
                While it is allowed to insert any number, knowing that these 
                cells start at generation number 2<br>
                <b>makes downstream analysis easier</b>.<br><br>
                What do you want to do?
            """)
            correctButtonText = " Fine, let me correct. "
            keepButtonText = " Keep the generation number that I chose. "
            buttonsTexts = (correctButtonText, keepButtonText)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.warning(self, "Recommendation", txt, buttonsTexts=buttonsTexts)
            if msg.cancel or msg.clickedButton == correctButtonText:
                return None
        elif any(check_buds_S):
            msg = widgets.myMessageBox(wrapText=False)
            title = "Bud in S/G2/M not in 0 Generation number"
            txt = html_utils.paragraph(
                "Some buds "
                "in S phase do not have 0 as Generation number!<br>"
                'Buds in S phase must have 0 as "Generation number"'
            )
            msg.critical(self, title, txt)
            return None
        elif any(check_mothers):
            msg = widgets.myMessageBox(wrapText=False)
            title = "Mother not in >=1 Generation number"
            txt = html_utils.paragraph(
                'Some mother cells do not have >=1 as "Generation number"!<br>'
                'Mothers MUST have >1 "Generation number"'
            )
            msg.critical(self, title, txt)
            return None
        elif any(check_buds_G1):
            msg = widgets.myMessageBox(wrapText=False)
            title = "Buds in G1!"
            txt = html_utils.paragraph(
                "Some buds are in G1 phase!<br><br>Buds MUST be in S/G2/M phase"
            )
            msg.critical(self, title, txt)
            return None
        elif num_moth_S != num_bud_S:
            msg = widgets.myMessageBox(wrapText=False)
            title = "Number of mothers-buds mismatch!"
            txt = html_utils.paragraph(
                f'There are {num_moth_S} mother cells in "S/G2/M" phase,'
                f"but there are {num_bud_S} bud cells.<br><br>"
                'The number of mothers and buds in "S/G2/M" '
                "phase must be equal!"
            )
            msg.critical(self, title, txt)
            return None
        elif any(check_relID_S):
            msg = widgets.myMessageBox(wrapText=False)
            title = "Relative's ID of cells in S/G2/M = -1"
            txt = html_utils.paragraph(
                'Some cells are in "S/G2/M" phase but have -1 as Relative\'s ID!<br>'
                'Cells in "S/G2/M" phase must have an existing '
                "ID as Relative's ID!"
            )
            msg.critical(self, title, txt)
            return None

        corrected_on_frame_i = self.inputCca_df["corrected_on_frame_i"]
        cca_df = pd.DataFrame(
            {
                "cell_cycle_stage": ccsValues,
                "generation_num": genNumValues,
                "relative_ID": relIDValues,
                "relationship": relatValues,
                "emerg_frame_i": emergFrameValues,
                "division_frame_i": divisFrameValues,
                "is_history_known": historyValues,
                "corrected_on_frame_i": corrected_on_frame_i,
                "will_divide": self.inputCca_df["will_divide"],
            },
            index=self.IDs,
        )
        cca_df.index.name = "Cell_ID"

        # Add missing columns
        for column, default in base_cca_dict.items():
            if column in cca_df.columns:
                continue

            value = self.inputCca_df.get(column, default=default)
            cca_df[column] = value

        # Check that every pair of cells in S are relative of each other
        proceed = self.check_ID_rel_ID_mismatches(cca_df)
        if not proceed:
            return None

        d = dict.fromkeys(cca_df.select_dtypes(np.int64).columns, np.int32)
        cca_df = cca_df.astype(d)
        return cca_df

    def check_ID_rel_ID_mismatches(self, cca_df):
        ID_rel_ID_mismatches = []
        for row in cca_df.itertuples():
            if row.cell_cycle_stage == "G1":
                continue

            ID = row.Index
            relID = row.relative_ID
            relID_of_relID = cca_df.at[relID, "relative_ID"]

            if relID_of_relID != ID:
                ID_rel_ID_mismatches.append((ID, relID, relID_of_relID))

        if not ID_rel_ID_mismatches:
            return True

        items = [
            f"Cell ID {ID} has relative ID = {relID}, "
            f"while cell ID {relID} has relative ID = {relID_of_relID}"
            for ID, relID, relID_of_relID in ID_rel_ID_mismatches
        ]
        title = "`ID-relative_ID` mismatches"
        txt = html_utils.paragraph(
            f"`ID-relative_ID` mismatches:{html_utils.to_list(items)}"
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.critical(self, title, txt)
        return False

    def ok_cb(self, checked):
        cca_df = self.getCca_df()
        if cca_df is None:
            return
        self.cca_df = cca_df
        self.cancel = False
        self.close()

    def cancel_cb(self, checked):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        ncols = self.tableLayout.columnCount()
        maxLabelWidth = max(
            [
                self.headerLayout.itemAt(j).widget().sizeHint().width()
                for j in range(ncols)
            ]
        )
        minWidth = (maxLabelWidth + 5) * ncols
        self.setMinimumWidth(minWidth)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def eventFilter(self, object, event):
        # Disable wheel scroll on widgets to allow scroll only on scrollarea
        if event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return False

    def clearComboboxFocus(self):
        self.sender().clearFocus()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class FindIDDialog(QLineEditDialog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.okButton.setIcon(QIcon(":magnGlass.svg"))
        self.okButton.setText(" Find ")


class NumericEntryDialog(QBaseDialog):
    def __init__(
        self,
        title="Entry a value",
        currentValue=0,
        instructions="Entry value",
        parent=None,
        maxValue=None,
        minValue=None,
        stretch=False,
    ):
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        self.cancel = False
        mainLayout = QVBoxLayout()
        entryLayout = QHBoxLayout()
        cancelOkLayout = widgets.CancelOkButtonsLayout()
        cancelOkLayout.okButton.clicked.connect(self.ok_cb)
        cancelOkLayout.cancelButton.clicked.connect(self.close)

        instructionsLabel = QLabel(html_utils.paragraph(instructions))
        mainLayout.addWidget(instructionsLabel)

        if type(currentValue) == int:
            self.entryWidget = widgets.SpinBox()
            self.entryWidget.setValue(currentValue)
            self.valueGetter = "value"
            if maxValue is not None:
                self.entryWidget.setMaximum(maxValue)
            if minValue is not None:
                self.entryWidget.setMinimum(minValue)

        if stretch:
            entryLayout.addWidget(self.entryWidget)
        else:
            entryLayout.addStretch(1)
            entryLayout.addWidget(self.entryWidget)
            entryLayout.addStretch(1)

        mainLayout.addLayout(entryLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(cancelOkLayout)

        self.setLayout(mainLayout)

    def ok_cb(self):
        self.cancel = False
        self.value = getattr(self.entryWidget, self.valueGetter)()
        self.close()


class EditIDDialog(QDialog):
    def __init__(
        self,
        clickedID,
        IDs,
        entryID=None,
        doNotShowAgain=False,
        parent=None,
        nextUniqueID=1,
        allIDs=None,
        addPropagateCheckbox=False,
    ):
        self.assignNewID = False
        self.IDs = IDs
        self.clickedID = clickedID
        self.cancel = True
        self.how = None
        self.mergeWithExistingID = True
        self.doNotAskAgainExistingID = doNotShowAgain
        self.allIDs = allIDs
        if allIDs is None:
            self.allIDs = set(self.IDs)
        self.nextUniqueID = nextUniqueID

        super().__init__(parent)
        self.setWindowTitle("Edit ID")
        mainLayout = QVBoxLayout()

        VBoxLayout = QVBoxLayout()
        msg = QLabel(f"Replace ID {clickedID} with:")
        _font = QFont()
        _font.setPixelSize(12)
        msg.setFont(_font)
        # padding: top, left, bottom, right
        msg.setStyleSheet("padding:0px 0px 3px 0px;")
        VBoxLayout.addWidget(msg, alignment=Qt.AlignCenter)

        entryWidget = QLineEdit()
        entryWidget.setFont(_font)
        entryWidget.setAlignment(Qt.AlignCenter)
        self.entryWidget = entryWidget
        VBoxLayout.addWidget(entryWidget)
        if entryID is not None:
            entryWidget.setText(str(entryID))
            entryWidget.selectAll()

        VBoxLayout.addWidget(
            QLabel(f"Next unique ID = {nextUniqueID}"), alignment=Qt.AlignCenter
        )

        VBoxLayout.addWidget(widgets.QHLine())

        self.warnExistingIDLabel = QLabel()
        self.warnExistingIDLabel.setStyleSheet("color: red")
        VBoxLayout.addWidget(self.warnExistingIDLabel, alignment=Qt.AlignCenter)

        note = QLabel(
            "NOTE: To replace multiple IDs at once\n"
            'write "(old ID, new ID), (old ID, new ID)" etc.'
        )
        note.setFont(_font)
        note.setAlignment(Qt.AlignCenter)
        # padding: top, left, bottom, right
        note.setStyleSheet("padding:12px 0px 0px 0px;")
        VBoxLayout.addWidget(note, alignment=Qt.AlignCenter)
        mainLayout.addLayout(VBoxLayout)

        self.propagateCheckbox = None
        if addPropagateCheckbox:
            mainLayout.addSpacing(10)
            self.propagateCheckbox = QCheckBox("Apply to future frames")
            mainLayout.addWidget(self.propagateCheckbox)

        buttonsLayout = QHBoxLayout()
        okButton = widgets.okPushButton("Ok")
        cancelButton = widgets.cancelPushButton("Cancel")
        applyNewIDButton = widgets.AssignNewIDButton("Assign new, unique ID")

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(applyNewIDButton)
        buttonsLayout.addWidget(okButton)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        # Connect events
        self.prevText = ""
        entryWidget.textChanged[str].connect(self.onTextChanged)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)
        applyNewIDButton.clicked.connect(self.assignNewIDclicked)

        # self.setModal(True)

    def onTextChanged(self, text):
        self.warnExistingIDLabel.setText("")
        try:
            ID = int(text)
            if ID in self.allIDs:
                self.warnExistingIDLabel.setText(f"WARNING: ID {ID} was already used")
        except Exception as err:
            pass

        # Get inserted char
        idx = self.entryWidget.cursorPosition()
        if idx == 0:
            return

        newChar = text[idx - 1]

        # Do nothing if user is deleting text
        if idx == 0 or len(text) < len(self.prevText):
            self.prevText = text
            return

        # Do not allow chars except for "(", ")", "int", ","
        m = re.search(r"\(|\)|\d|,", newChar)
        if m is None:
            self.prevText = text
            text = text.replace(newChar, "")
            self.entryWidget.setText(text)
            return

        # Cast integers greater than uint32 machine limit
        m_iter = re.finditer(r"\d+", self.entryWidget.text())
        for m in m_iter:
            val = int(m.group())
            uint32_max = np.iinfo(np.uint32).max
            if val > uint32_max:
                text = self.entryWidget.text()
                text = f"{text[: m.start()]}{uint32_max}{text[m.end() :]}"
                self.entryWidget.setText(text)

        # Automatically close ( bracket
        if newChar == "(":
            text += ")"
            self.entryWidget.setText(text)
        self.prevText = text

    def _warnExistingID(self, existingID, newID):
        warn_msg = html_utils.paragraph(f"""
            ID {existingID} is <b>already existing</b>.<br><br>
            How do you want to proceed?<br>
        """)
        msg = widgets.myMessageBox()
        doNotAskAgainCheckbox = QCheckBox("Remember my choice and do not ask again")
        swapButton = widgets.reloadPushButton(f"Swap {newID} with {existingID}")
        mergeButton = widgets.mergePushButton(f"Merge {newID} with {existingID}")
        msg.warning(
            self,
            "Existing ID",
            warn_msg,
            buttonsTexts=("Cancel", mergeButton, swapButton),
            widgets=doNotAskAgainCheckbox,
        )
        if msg.cancel:
            return False
        self.doNotAskAgainExistingID = doNotAskAgainCheckbox.isChecked()
        self.mergeWithExistingID = msg.clickedButton == mergeButton
        return True

    def assignNewIDclicked(self):
        self.cancel = False
        self.how = None
        self.assignNewID = True
        self.close()

    def ok_cb(self, event):
        txt = self.entryWidget.text()
        valid = False

        # Check validity of inserted text
        try:
            ID = int(txt)
            how = [(self.clickedID, ID)]
            if ID in self.IDs and not self.doNotAskAgainExistingID:
                proceed = self._warnExistingID(self.clickedID, ID)
                if not proceed:
                    return
                valid = True
            else:
                valid = True
        except ValueError:
            pattern = r"\((\d+),\s*(\d+)\)"
            fa = re.findall(pattern, txt)
            if fa:
                how = [(int(g[0]), int(g[1])) for g in fa]
                valid = True
            else:
                valid = False

        if not valid:
            err_msg = html_utils.paragraph(
                "You entered invalid text. Valid text is either a single integer"
                f" ID that will be used to replace ID {self.clickedID} "
                "or a list of elements enclosed in parenthesis separated by a comma<br>"
                "such as (5, 10), (8, 27) to replace ID 5 with ID 10 and ID 8 with ID 27"
            )
            msg = widgets.myMessageBox()
            msg.warning(self, "Invalid entry", err_msg)
            return

        self.cancel = False
        self.how = how
        self.doPropagateFutureFrames = False
        if self.propagateCheckbox is not None:
            self.doPropagateFutureFrames = self.propagateCheckbox.isChecked()
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class manualSeparateGui(QMainWindow):
    def __init__(
        self,
        lab,
        ID,
        img,
        fontSize="12pt",
        IDcolor=[255, 255, 0],
        parent=None,
        loop=None,
        drawMode="threepoints_arc",
    ):
        super().__init__(parent)
        self.loop = loop
        self.cancel = True
        self.drawMode = drawMode
        self._parent = parent
        self.lab = lab.copy()
        self.lab[lab != ID] = 0
        self.ID = ID
        self.img = skimage.exposure.equalize_adapthist(img / img.max())
        self.IDcolor = IDcolor
        self.countClicks = 0
        self.prevLabs = []
        self.prevAllCutsCoords = []
        self.labelItemsIDs = []
        self.undoIdx = 0
        self.fontSize = fontSize
        self.AllCutsCoords = []
        self.setWindowTitle("Split object")
        # self.setGeometry(Left, Top, 850, 800)

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()

        self.gui_createStatusBar()

        self.gui_createGraphics()
        self.gui_connectImgActions()

        self.gui_createImgWidgets()
        self.gui_connectActions()

        self.updateImg()
        self.zoomToObj()

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 1)
        mainLayout.addLayout(self.img_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

        self.setWindowModality(Qt.WindowModal)

    def centerWindow(self):
        parent = self._parent
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

    def gui_createActions(self):
        # File actions
        self.exitAction = QAction("&Exit", self)
        self.helpAction = QAction("Help", self)
        self.undoAction = QAction(QIcon(":undo.svg"), "Undo (Ctrl+Z)", self)
        self.undoAction.setEnabled(False)
        self.undoAction.setShortcut("Ctrl+Z")

        self.okAction = QAction(QIcon(":applyCrop.svg"), "Happy with that", self)
        self.cancelAction = QAction(QIcon(":cancel.svg"), "Cancel", self)

        self.drawModesActionGroup = QActionGroup(self)

        self.threePointsArcAction = QAction(
            QIcon(":threepoints_arc.svg"), "Separate with three-points arc", self
        )
        self.threePointsArcAction.setCheckable(True)
        self.threePointsArcAction.drawMode = "threepoints_arc"
        self.drawModesActionGroup.addAction(self.threePointsArcAction)

        self.freeHandAction = QAction(
            QIcon(":freehand.svg"), "Separate with freehand line", self
        )
        self.freeHandAction.setCheckable(True)
        self.freeHandAction.drawMode = "freehand"
        self.drawModesActionGroup.addAction(self.freeHandAction)

        if self.drawMode == "threepoints_arc":
            self.threePointsArcAction.setChecked(True)
        elif self.drawMode == "freehand":
            self.freeHandAction.setChecked(True)

        self.swapIDsAction = QAction(QIcon(":reload.svg"), "Swap IDs", self)
        self.swapIDsAction.setToolTip('Swap the two displayed IDs\n\nShortcut: "S"')
        self.swapIDsAction.setShortcut("S")

    def state(self):
        return {
            "is_overlay_active": self.overlayButton.isChecked(),
            "is_three_points_active": self.threePointsArcAction.isChecked(),
            "is_free_hand_active": self.freeHandAction.isChecked(),
        }

    def show(self, block=False):
        super().show()
        if not block:
            return
        self.loop = QEventLoop(self)
        self.loop.exec_()

    def setState(self, state):
        if state is None:
            return
        self.overlayButton.setChecked(state.get("is_overlay_active", False))
        self.threePointsArcAction.setChecked(state.get("is_three_points_active", True))
        self.freeHandAction.setChecked(state.get("is_free_hand_active", False))

    def gui_storeDrawMode(self):
        self.drawMode = self.sender().drawMode

    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        # style = "QMenuBar::item:selected { background: white; }"
        # menuBar.setStyleSheet(style)
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)

        menuBar.addAction(self.helpAction)
        fileMenu.addAction(self.exitAction)

    def gui_createToolBars(self):
        toolbarSize = 30

        editToolBar = QToolBar("Edit", self)
        editToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(editToolBar)

        editToolBar.addAction(self.okAction)
        editToolBar.addAction(self.cancelAction)

        editToolBar.addAction(self.undoAction)

        self.overlayButton = QToolButton(self)
        self.overlayButton.setIcon(QIcon(":overlay.svg"))
        self.overlayButton.setCheckable(True)
        self.overlayButton.setToolTip("Overlay channel's image")
        editToolBar.addWidget(self.overlayButton)

        editToolBar.addAction(self.threePointsArcAction)
        editToolBar.addAction(self.freeHandAction)

        editToolBar.addAction(self.swapIDsAction)

        self.warnLabel = QLabel()
        editToolBar.addWidget(self.warnLabel)

    def gui_connectActions(self):
        self.exitAction.triggered.connect(self.close)
        self.helpAction.triggered.connect(self.help)
        self.okAction.triggered.connect(self.ok_cb)
        self.cancelAction.triggered.connect(self.close)
        self.undoAction.triggered.connect(self.undo)
        self.overlayButton.toggled.connect(self.toggleOverlay)
        self.imgGrad.sigLookupTableChanged.connect(self.histLUT_cb)
        self.swapIDsAction.triggered.connect(self.swapIDs)

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Temporary message
        self.statusbar.showMessage("Ready", 3000)
        # Permanent widget
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)

    def gui_createGraphics(self):
        self.graphLayout = pg.GraphicsLayoutWidget()

        # Plot Item container for image
        self.ax = pg.PlotItem()
        self.ax.invertY(True)
        self.ax.setAspectLocked(True)
        self.ax.hideAxis("bottom")
        self.ax.hideAxis("left")
        self.graphLayout.addItem(self.ax, row=1, col=1)

        # Image Item
        self.imgItem = pg.ImageItem(np.zeros((512, 512)))
        self.ax.addItem(self.imgItem)

        # Image histogram
        self.imgGrad = widgets.myHistogramLUTitem()

        # Curvature items
        self.hoverLinSpace = np.linspace(0, 1, 1000)
        self.hoverLinePen = pg.mkPen(
            color=(200, 0, 0, 255 * 0.5), width=2, style=Qt.DashLine
        )
        self.hoverCurvePen = pg.mkPen(color=(200, 0, 0, 255 * 0.5), width=3)
        self.lineHoverPlotItem = pg.PlotDataItem(pen=self.hoverLinePen)
        self.curvHoverPlotItem = pg.PlotDataItem(pen=self.hoverCurvePen)
        self.curvAnchors = pg.ScatterPlotItem(
            symbol="o",
            size=9,
            brush=pg.mkBrush((255, 0, 0, 50)),
            pen=pg.mkPen((255, 0, 0), width=2),
            hoverable=True,
            hoverPen=pg.mkPen((255, 0, 0), width=3),
            hoverBrush=pg.mkBrush((255, 0, 0)),
        )
        self.ax.addItem(self.curvAnchors)
        self.ax.addItem(self.curvHoverPlotItem)
        self.ax.addItem(self.lineHoverPlotItem)

        self.freeHandItem = widgets.PlotCurveItem(pen=pg.mkPen(color="r", width=2))
        self.ax.addItem(self.freeHandItem)

    def gui_createImgWidgets(self):
        self.img_Widglayout = QGridLayout()
        self.img_Widglayout.setContentsMargins(50, 0, 50, 0)

        alphaScrollBar_label = QLabel("Overlay alpha  ")
        alphaScrollBar = QScrollBar(Qt.Horizontal)
        alphaScrollBar.setFixedHeight(20)
        alphaScrollBar.setMinimum(0)
        alphaScrollBar.setMaximum(40)
        alphaScrollBar.setValue(12)
        alphaScrollBar.setToolTip(
            "Control the alpha value of the overlay.\n"
            "alpha=0 results in NO overlay,\n"
            "alpha=1 results in only labels visible"
        )
        alphaScrollBar.sliderMoved.connect(self.alphaScrollBarMoved)
        self.alphaScrollBar = alphaScrollBar
        self.alphaScrollBar_label = alphaScrollBar_label
        self.img_Widglayout.addWidget(
            alphaScrollBar_label, 0, 0, alignment=Qt.AlignCenter
        )
        self.img_Widglayout.addWidget(alphaScrollBar, 0, 1, 1, 20)
        self.alphaScrollBar.hide()
        self.alphaScrollBar_label.hide()

    def gui_connectImgActions(self):
        self.imgItem.hoverEvent = self.gui_hoverEventImg
        self.imgItem.mousePressEvent = self.gui_mousePressEventImg
        self.imgItem.mouseMoveEvent = self.gui_mouseDragEventImg
        self.imgItem.mouseReleaseEvent = self.gui_mouseReleaseEventImg

    def gui_hoverEventImg(self, event):
        # Update x, y, value label bottom right
        try:
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.lab
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                self.wcLabel.setText(f"(x={x:.2f}, y={y:.2f}, ID={val:.0f})")
            else:
                self.wcLabel.setText(f"")
        except Exception as e:
            self.wcLabel.setText(f"")

        if event.isExit():
            return

        self.drawHoverEvent(*event.pos())

    def gui_mousePressEventImg(self, event):
        right_click = event.button() == Qt.MouseButton.RightButton
        left_click = event.button() == Qt.MouseButton.LeftButton

        dragImg = left_click

        if dragImg:
            pg.ImageItem.mousePressEvent(self.imgItem, event)

        if not right_click:
            return

        self.drawPressEvent(event)

    def gui_mouseDragEventImg(self, event):
        pass

    def gui_mouseReleaseEventImg(self, event):
        if self.countClicks == 0:
            return
        if self.freeHandAction.isChecked():
            self.countClicks = 0
            xx, yy = self.freeHandItem.getData()
            self.setSplitCurveCoords(xx, yy)
            self.splitObjectAlongCurve()
            self.freeHandItem.setData([], [])
            self.curvAnchors.setData([], [])

    def getSpline(self, xx, yy):
        tck, u = scipy.interpolate.splprep([xx, yy], s=0, k=2)
        xi, yi = scipy.interpolate.splev(self.hoverLinSpace, tck)
        return xi, yi

    def drawPressEvent(self, event):
        if self.freeHandAction.isChecked():
            self.countClicks = 1
            x, y = event.pos().x(), event.pos().y()
            self.curvAnchors.addPoints([x], [y])
        elif self.threePointsArcAction.isChecked():
            self.threePointsArcPressEvent(event)

    def drawHoverEvent(self, x, y):
        if self.freeHandAction.isChecked():
            self.freeHandHoverEvent(x, y)
        elif self.threePointsArcAction.isChecked():
            self.threePointsArcHoverEvent(x, y)

    def freeHandHoverEvent(self, x, y):
        if self.countClicks == 0:
            return
        self.freeHandItem.addPoint(int(x), int(y))
        _xx, _yy = self.freeHandItem.getData()
        xx = [_xx[0], x]
        yy = [_yy[0], y]
        self.curvAnchors.setData(xx, yy)

    def threePointsArcHoverEvent(self, x, y):
        if self.countClicks == 1:
            self.lineHoverPlotItem.setData([self.x0, x], [self.y0, y])
        elif self.countClicks == 2:
            xx = [self.x0, x, self.x1]
            yy = [self.y0, y, self.y1]
            xi, yi = self.getSpline(xx, yy)
            self.curvHoverPlotItem.setData(xi, yi)
        elif self.countClicks == 0:
            self.curvHoverPlotItem.setData([], [])
            self.lineHoverPlotItem.setData([], [])
            self.curvAnchors.setData([], [])

    def threePointsArcPressEvent(self, event):
        if self.countClicks == 0:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.x0, self.y0 = xdata, ydata
            self.curvAnchors.addPoints([xdata], [ydata])
            self.countClicks = 1
        elif self.countClicks == 1:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.x1, self.y1 = xdata, ydata
            self.curvAnchors.addPoints([xdata], [ydata])
            self.countClicks = 2
        elif self.countClicks == 2:
            self.countClicks = 0
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            xx = [self.x0, xdata, self.x1]
            yy = [self.y0, ydata, self.y1]
            xi, yi = self.getSpline(xx, yy)
            yy, xx = np.round(yi).astype(int), np.round(xi).astype(int)
            self.setSplitCurveCoords(xx, yy)
            self.splitObjectAlongCurve()

    def setSplitCurveCoords(self, xx, yy):
        self.storeUndoState()
        xxCurve, yyCurve = [], []
        for i, (r0, c0) in enumerate(zip(yy, xx)):
            if i == len(yy) - 1:
                break
            r1 = yy[i + 1]
            c1 = xx[i + 1]
            rr, cc, _ = skimage.draw.line_aa(r0, c0, r1, c1)
            # rr, cc = skimage.draw.line(r0, c0, r1, c1)
            nonzeroMask = self.lab[rr, cc] > 0
            xxCurve.extend(cc[nonzeroMask])
            yyCurve.extend(rr[nonzeroMask])
        self.AllCutsCoords.append((yyCurve, xxCurve))
        for rr, cc in self.AllCutsCoords:
            self.lab[rr, cc] = 0
        self.lab = skimage.morphology.remove_small_objects(self.lab, 5)

    def histLUT_cb(self, LUTitem):
        if self.overlayButton.isChecked():
            overlay = self.getOverlay()
            self.imgItem.setImage(overlay)

    def swapIDs(self, checked=False):
        if len(self.rp) == 1:
            self.warnLabel.setText(
                html_utils.paragraph(
                    "WARNING: Split the object before swapping IDs", font_color="red"
                )
            )
            return

        self.warnLabel.setText("")

        obj1 = self.rp[0]
        obj2 = self.rp[1]

        self.lab[obj1.slice][obj1.image] = obj2.label
        self.lab[obj2.slice][obj2.image] = obj1.label

        self.updateImg()

    def updateImg(self):
        self.updateLookuptable()
        rp = skimage.measure.regionprops(self.lab)
        self.rp = rp

        if self.overlayButton.isChecked():
            overlay = self.getOverlay()
            self.imgItem.setImage(overlay)
        else:
            self.imgItem.setImage(self.lab)

        # Draw ID on centroid of each label
        for labelItemID in self.labelItemsIDs:
            self.ax.removeItem(labelItemID)
        self.labelItemsIDs = []
        for obj in rp:
            labelItemID = widgets.myLabelItem()
            labelItemID.setText(f"{obj.label}", color="r", size=f"{self.fontSize}px")
            y, x = obj.centroid
            w, h = labelItemID.rect().right(), labelItemID.rect().bottom()
            labelItemID.setPos(x - w / 2, y - h / 2)
            self.labelItemsIDs.append(labelItemID)
            self.ax.addItem(labelItemID)

    def zoomToObj(self):
        # Zoom to object
        lab_mask = (self.lab > 0).astype(np.uint8)
        rp = skimage.measure.regionprops(lab_mask)
        obj = rp[0]
        min_row, min_col, max_row, max_col = obj.bbox
        xRange = min_col - 10, max_col + 10
        yRange = max_row + 10, min_row - 10
        self.ax.setRange(xRange=xRange, yRange=yRange)

    def storeUndoState(self):
        self.prevLabs.append(self.lab.copy())
        self.prevAllCutsCoords.append(self.AllCutsCoords.copy())
        self.undoIdx += 1
        self.undoAction.setEnabled(True)

    def undo(self):
        self.undoIdx -= 1
        self.lab = self.prevLabs[self.undoIdx]
        self.AllCutsCoords = self.prevAllCutsCoords[self.undoIdx]
        self.updateImg()
        if self.undoIdx == 0:
            self.undoAction.setEnabled(False)
            self.prevLabs = []
            self.prevAllCutsCoords = []

    def splitObjectAlongCurve(self):
        self.lab = skimage.measure.label(self.lab, connectivity=1)

        # Relabel largest object with original ID
        rp = skimage.measure.regionprops(self.lab)
        areas = [obj.area for obj in rp]
        IDs = [obj.label for obj in rp]
        maxAreaIdx = areas.index(max(areas))
        maxAreaID = IDs[maxAreaIdx]
        if self.ID not in self.lab:
            self.lab[self.lab == maxAreaID] = self.ID
        else:
            tempID = self.lab.max() + 1
            self.lab[self.lab == maxAreaID] = tempID
            self.lab[self.lab == self.ID] = maxAreaID
            self.lab[self.lab == tempID] = self.ID

        # Keep only the two largest objects
        larger_areas = nlargest(2, areas)
        larger_ids = [rp[areas.index(area)].label for area in larger_areas]
        for obj in rp:
            if obj.label not in larger_ids:
                self.lab[tuple(obj.coords.T)] = 0

        rp = skimage.measure.regionprops(self.lab)

        if self._parent is not None:
            self._parent.setBrushID()
        # Use parent window setBrushID function for all other IDs
        for obj in rp:
            if self._parent is None:
                break
            if obj.label == self.ID:
                continue
            posData = self._parent.data[self._parent.pos_i]
            posData.brushID += 1
            self.lab[obj.slice][obj.image] = posData.brushID

        # Replace 0s on the cutting curve with IDs
        self.cutLab = self.lab.copy()
        for rr, cc in self.AllCutsCoords:
            for y, x in zip(rr, cc):
                top_row = self.cutLab[y + 1, x - 1 : x + 2]
                bot_row = self.cutLab[y - 1, x - 1 : x + 1]
                left_col = self.cutLab[y - 1, x - 1]
                right_col = self.cutLab[y : y + 2, x + 1]
                allNeigh = list(top_row)
                allNeigh.extend(bot_row)
                allNeigh.append(left_col)
                allNeigh.extend(right_col)
                newID = max(allNeigh)
                self.lab[y, x] = newID

        self.rp = skimage.measure.regionprops(self.lab)
        self.updateImg()

    def updateLookuptable(self):
        # Lookup table
        self.cmap = colors.getFromMatplotlib("viridis")
        self.lut = self.cmap.getLookupTable(0, 1, self.lab.max() + 1)
        self.lut[0] = [25, 25, 25]
        self.lut[self.ID] = self.IDcolor
        if self.overlayButton.isChecked():
            self.imgItem.setLookupTable(None)
        else:
            self.imgItem.setLookupTable(self.lut)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.countClicks = 0
            self.curvHoverPlotItem.setData([], [])
            self.lineHoverPlotItem.setData([], [])
            self.curvAnchors.setData([], [])
            self.freeHandItem.setData([], [])
        elif ev.key() == Qt.Key_Enter or ev.key() == Qt.Key_Return:
            self.ok_cb(True)

    def getOverlay(self):
        # Rescale intensity based on hist ticks values
        min = self.imgGrad.gradient.listTicks()[0][1]
        max = self.imgGrad.gradient.listTicks()[1][1]
        img = skimage.exposure.rescale_intensity(self.img, in_range=(min, max))
        alpha = self.alphaScrollBar.value() / self.alphaScrollBar.maximum()

        # Convert img and lab to RGBs
        rgb_shape = (self.lab.shape[0], self.lab.shape[1], 3)
        labRGB = np.zeros(rgb_shape)
        labRGB[self.lab > 0] = [1, 1, 1]
        imgRGB = skimage.color.gray2rgb(img)
        overlay = imgRGB * (1.0 - alpha) + labRGB * alpha

        # Color eaach label
        for obj in self.rp:
            rgb = self.lut[obj.label] / 255
            overlay[obj.slice][obj.image] *= rgb

        # Convert (0,1) to (0,255)
        overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
        return overlay

    def alphaScrollBarMoved(self, alpha_int):
        overlay = self.getOverlay()
        self.imgItem.setImage(overlay)

    def toggleOverlay(self, checked):
        if checked:
            self.graphLayout.addItem(self.imgGrad, row=1, col=0)
            self.alphaScrollBar.show()
            self.alphaScrollBar_label.show()
        else:
            self.graphLayout.removeItem(self.imgGrad)
            self.alphaScrollBar.hide()
            self.alphaScrollBar_label.hide()
        self.updateImg()

    def help(self):
        msg = QMessageBox()
        msg.information(
            self,
            "Help",
            "Separate object along a curved line.\n\n"
            "To draw a curved line you will need 3 right-clicks:\n\n"
            "1. Right-click outside of the object --> a line appears.\n"
            "2. Right-click to end the line and a curve going through the "
            "mouse cursor will appear.\n"
            "3. Once you are happy with the cutting curve right-click again "
            "and the object will be separated along the curve.\n\n"
            "Note that you can separate as many times as you want.\n\n"
            "Once happy click on the green tick on top-right or "
            'cancel the process with the "X" button',
        )

    def ok_cb(self, checked):
        self.cancel = False
        self.close()

    def closeEvent(self, event):
        if self.loop is not None:
            self.loop.exit()


class ViewCcaTableWindow(pdDataFrameWidget):
    sigUpdateCcaTable = Signal(object)

    def __init__(self, df, parent=None):
        super().__init__(df, parent=parent)

        updateTableButton = widgets.reloadPushButton("Update table with visible IDs...")
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(updateTableButton)

        self._layout.insertLayout(0, buttonsLayout)

        updateTableButton.clicked.connect(self.emitUpdateCcaTable)

    def emitUpdateCcaTable(self):
        self.sigUpdateCcaTable.emit(self)

# Sibling imports (deferred to avoid import cycles)
from .metadata import (
    filenameDialog,
)

