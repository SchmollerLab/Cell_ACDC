"""Cell-ACDC dialog windows: models."""

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

def addCustomModelMessages(QParent=None):
    modelFilePath = None
    msg = widgets.myMessageBox(showCentered=False, wrapText=False)
    txt = html_utils.paragraph("""
    Do you <b>already have</b> the <code>acdcSegment.py</code> file for your code 
    or do you <b>need instructions</b> on how to set-up your custom model?<br>
    """)
    infoButton = widgets.infoPushButton(" I need instructions")
    browseButton = widgets.browseFileButton(" I have the model, let me select it")
    msg.information(
        QParent,
        "Add custom model",
        txt,
        buttonsTexts=("Cancel", infoButton, browseButton),
        showDialog=False,
    )
    browseButton.clicked.disconnect()
    browseButton.clicked.connect(msg.buttonCallBack)
    msg.exec_()
    if msg.cancel:
        return
    if msg.clickedButton == infoButton:
        txt = myutils.get_add_custom_model_instructions()
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.information(
            QParent,
            "Custom model instructions",
            txt,
            buttonsTexts=("Ok",),
            path_to_browse=models_path,
            browse_button_text="Open models folder...",
        )
    else:
        homePath = pathlib.Path.home()
        modelFilePath = QFileDialog.getOpenFileName(
            QParent,
            "Select the acdcSegment.py file of your model",
            str(homePath),
            "acdcSegment.py file (*.py);;All files (*)",
        )[0]
        if not modelFilePath:
            return

    return modelFilePath


def addCustomPromptModelMessages(QParent=None):
    modelFilePath = None
    msg = widgets.myMessageBox(showCentered=False, wrapText=False)
    txt = html_utils.paragraph("""
    Do you <b>already have</b> the <code>acdcPromptSegment.py</code> file for your code 
    or do you <b>need instructions</b> on how to set-up your custom model?<br>
    """)
    infoButton = widgets.infoPushButton(" I need instructions")
    browseButton = widgets.browseFileButton(" I have the model, let me select it")
    msg.information(
        QParent,
        "Add custom promptable model",
        txt,
        buttonsTexts=("Cancel", infoButton, browseButton),
        showDialog=False,
    )
    browseButton.clicked.disconnect()
    browseButton.clicked.connect(msg.buttonCallBack)
    msg.exec_()
    if msg.cancel:
        return
    if msg.clickedButton == infoButton:
        txt = myutils.get_add_custom_prompt_model_instructions()
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.information(
            QParent,
            "Custom promptable model instructions",
            txt,
            buttonsTexts=("Ok",),
            path_to_browse=promptable_models_path,
            browse_button_text="Open promptable models folder...",
        )
    else:
        homePath = pathlib.Path.home()
        modelFilePath = QFileDialog.getOpenFileName(
            QParent,
            "Select the acdcPromptSegment.py file of your model",
            str(homePath),
            "acdcPromptSegment.py file (*.py);;All files (*)",
        )[0]
        if not modelFilePath:
            return

    return modelFilePath


class SelectPromptableModelDialog(QBaseDialog):
    def __init__(self, parent=None):
        self.cancel = True
        super().__init__(parent)

        self.setWindowTitle("Select model for segmentation")

        mainLayout = QVBoxLayout()

        label = QLabel(html_utils.paragraph("Select model to use for segmentation: "))
        mainLayout.addWidget(label, alignment=Qt.AlignCenter)

        listBox = widgets.listWidget()
        models = myutils.get_list_of_promptable_models()
        listBox.addItems(models)
        listBox.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        listBox.setCurrentRow(0)
        listBox.itemDoubleClicked.connect(self.ok_cb)

        self.listBox = listBox

        mainLayout.addWidget(listBox)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def ok_cb(self):
        self.cancel = False
        self.model_name = self.listBox.currentItem().text()
        self.close()


class QDialogSelectModel(QDialog):
    def __init__(self, parent=None, addSkipSegmButton=False, customFirst=""):
        self.cancel = True
        super().__init__(parent)
        self.setWindowTitle("Select model")

        mainLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        bottomLayout = QHBoxLayout()

        self.mainLayout = mainLayout

        label = QLabel(html_utils.paragraph("Select model to use for segmentation: "))
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 3px 0px;")
        topLayout.addWidget(label, alignment=Qt.AlignCenter)

        listBox = widgets.listWidget()
        models = myutils.get_list_of_models()

        if customFirst:
            try:
                idx = models.index(customFirst)
                models.insert(0, models.pop(idx))
            except ValueError:
                print(f"Warning: {customFirst} not found in models list.")
                pass

        listBox.setFont(font)
        listBox.addItems(models)
        addCustomModelItem = QListWidgetItem("Add custom model...")
        addCustomModelItem.setFont(italicFont)
        listBox.addItem(addCustomModelItem)
        listBox.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        listBox.setCurrentRow(0)
        self.listBox = listBox
        listBox.itemDoubleClicked.connect(self.ok_cb)
        topLayout.addWidget(listBox)

        cancelButton = widgets.cancelPushButton("Cancel")
        okButton = widgets.okPushButton(" Ok ")
        okButton.setShortcut(Qt.Key_Enter)

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)
        if addSkipSegmButton:
            skipSegmButton = widgets.SkipPushButton("Skip segmentation")
            bottomLayout.addWidget(skipSegmButton)
            skipSegmButton.clicked.connect(self.skipSegm)
        bottomLayout.addWidget(okButton)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        self.setStyleSheet(LISTWIDGET_STYLESHEET)

    def skipSegm(self):
        self.cancel = False
        self.selectedModel = "skip_segmentation"
        self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            event.ignore()
            return

        super().keyPressEvent(event)

    def ok_cb(self, event):
        self.clickedButton = self.sender()
        self.cancel = False
        item = self.listBox.currentItem()
        model = item.text()
        if model == "Add custom model...":
            modelFilePath = addCustomModelMessages(self)
            if modelFilePath is None:
                return
            myutils.store_custom_model_path(modelFilePath)
            modelName = os.path.basename(os.path.dirname(modelFilePath))
            item = QListWidgetItem(modelName)
            self.listBox.addItem(item)
            self.listBox.setCurrentItem(item)
        elif model == "Automatic thresholding":
            self.selectedModel = "thresholding"
            self.close()
        else:
            self.selectedModel = model
            self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.selectedModel = None
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()

        horizontal_sb = self.listBox.horizontalScrollBar()
        while horizontal_sb.isVisible():
            self.resize(self.height(), self.width() + 10)

        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class DataFrameModel(QtCore.QAbstractTableModel):
    # https://stackoverflow.com/questions/44603119/how-to-display-a-pandas-data-frame-with-pyqt5-pyside2
    DtypeRole = QtCore.Qt.UserRole + 1000
    ValueRole = QtCore.Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = QtCore.Property(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    @QtCore.Slot(int, QtCore.Qt.Orientation, result=str)
    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.DisplayRole,
    ):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QtCore.QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (
            0 <= index.row() < self.rowCount()
            and 0 <= index.column() < self.columnCount()
        ):
            return QtCore.QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        val = self._dataframe.iloc[row][col]
        if role == QtCore.Qt.DisplayRole:
            return str(val)
        elif role == DataFrameModel.ValueRole:
            return val
        if role == DataFrameModel.DtypeRole:
            return dt
        return QtCore.QVariant()

    def roleNames(self):
        roles = {
            QtCore.Qt.DisplayRole: b"display",
            DataFrameModel.DtypeRole: b"dtype",
            DataFrameModel.ValueRole: b"value",
        }
        return roles


class QDialogModelParams(QDialog):
    def __init__(
        self,
        init_params,
        segment_params,
        model_name,
        is_tracker=False,
        url=None,
        parent=None,
        initLastParams=True,
        posData=None,
        channels=None,
        currentChannelName=None,
        segmFileEndnames=None,
        df_metadata=None,
        force_postprocess_2D=False,
        model_module=None,
        action_type="",
        addPreProcessParams=True,
        addPostProcessParams=True,
        extraParams=None,
        extraParamsTitle=None,
        ini_filename=None,
        add_additional_segm_params=False,
    ):
        self.cancel = True
        super().__init__(parent)
        self.channels = channels
        self.is_tracker = is_tracker
        self.currentChannelName = currentChannelName
        self.channelCombobox = None
        self.segmFileEndnames = segmFileEndnames
        self.df_metadata = df_metadata
        self.force_postprocess_2D = force_postprocess_2D

        self.skipSegmentation = False
        if len(segment_params) > 0:
            if segment_params[0].name.lower().find("skip_segmentation") != -1:
                self.skipSegmentation = True
                addPreProcessParams = False
            else:
                self.skipSegmentation = False
        if ini_filename is not None:
            self.ini_filename = ini_filename
        elif is_tracker:
            self.ini_filename = "last_params_trackers.ini"
            addPreProcessParams = False
            addPostProcessParams = False
        else:
            self.ini_filename = "last_params_segm_models.ini"

        self.addPreProcessParams = addPreProcessParams

        self.model_name = model_name

        self.setWindowTitle(f"{model_name} parameters")

        # Create main vertical layout and horizontal layout for two columns
        mainLayout = QVBoxLayout()

        gridLayout = QGridLayout()
        self.gridLayout = gridLayout

        loadFunc = self.loadLastSelection

        self.paramsGroupPosMapper = {}

        # LEFT COLUMN: Preprocessing params
        row, col = 0, 0
        preProcessLayout = None
        self.preProcessParamsWidget = None
        if addPreProcessParams:
            preProcessLayout = QVBoxLayout()
            self.preProcessParamsWidget = PreProcessParamsWidget(
                parent=self, addApplyButton=False
            )
            self.preProcessParamsWidget.setChecked(False)
            preProcessLayout.addWidget(self.preProcessParamsWidget)
            self.preProcessParamsWidget.sigLoadRecipe.connect(self.loadPreprocRecipe)
            gridLayout.addLayout(preProcessLayout, row, col, 1, 2)
            self.paramsGroupPosMapper[self.preProcessParamsWidget] = (row, col)
            gridLayout.addItem(QSpacerItem(10, 5), 0, col + 1)
            # gridLayout.setColumnMinimumWidth(col+1, 15)
            col += 2

        # Center COLUMN: Init, Segmentation/Eval
        row = 0
        self.secondColLayout = QVBoxLayout()
        self.initParamsScrollArea = widgets.ScrollArea()
        initParamsScrollAreaLayout = QVBoxLayout()
        self.initParamsScrollArea.setVerticalLayout(initParamsScrollAreaLayout)

        initGroupBox, self.init_argsWidgets = self.createGroupParams(
            init_params, "Parameters for model initialization"
        )
        self.init_params = init_params
        initDefaultButton = widgets.reloadPushButton("Restore default")
        initLoadLastSelButton = widgets.OpenFilePushButton("Load last parameters")
        initLoadLastSelButton.setIcon(QIcon(":folder-open.svg"))
        initButtonsLayout = QHBoxLayout()
        initButtonsLayout.addStretch(1)
        initButtonsLayout.addWidget(initDefaultButton)
        initButtonsLayout.addWidget(initLoadLastSelButton)
        initDefaultButton.clicked.connect(self.restoreDefaultInit)
        initLoadLastSelButton.clicked.connect(
            partial(loadFunc, f"{self.model_name}.init", self.init_argsWidgets)
        )

        initParamsScrollAreaLayout.addWidget(initGroupBox)

        initParamsLayout = QVBoxLayout()
        initParamsLayout.addWidget(QLabel(f"<b>{initGroupBox.title()}</b>"))
        initGroupBox.setTitle("")
        initParamsLayout.addWidget(self.initParamsScrollArea)
        initParamsLayout.addLayout(initButtonsLayout)
        self.secondColLayout.addLayout(initParamsLayout)
        self.paramsGroupPosMapper[self.initParamsScrollArea] = (0, col)

        self.segmentParamsScrollArea = None
        if not self.skipSegmentation:
            self.segmentParamsScrollArea = widgets.ScrollArea()
            segmentParamsScrollAreaLayout = QVBoxLayout()
            self.segmentParamsScrollArea.setVerticalLayout(
                segmentParamsScrollAreaLayout
            )
            if action_type:
                runGroupboxTitle = f"Parameters for {action_type}"
            elif is_tracker:
                runGroupboxTitle = "Parameters for tracking"
            else:
                runGroupboxTitle = "Parameters for segmentation"

            segmentGroupBox, self.argsWidgets = self.createGroupParams(
                segment_params, runGroupboxTitle, addChannelSelector=True
            )
            self.segment_params = segment_params
            self.segmentGroupBox = segmentGroupBox
            segmentDefaultButton = widgets.reloadPushButton("Restore default")
            segmentLoadLastSelButton = widgets.OpenFilePushButton(
                "Load last parameters"
            )
            segmentButtonsLayout = QHBoxLayout()
            segmentButtonsLayout.addStretch(1)
            segmentButtonsLayout.addWidget(segmentDefaultButton)
            segmentButtonsLayout.addWidget(segmentLoadLastSelButton)
            segmentDefaultButton.clicked.connect(self.restoreDefaultSegment)
            section = f"{self.model_name}.segment"
            segmentLoadLastSelButton.clicked.connect(
                partial(loadFunc, section, self.argsWidgets)
            )
            segmentParamsScrollAreaLayout.addWidget(segmentGroupBox)

            segmentParamsLayout = QVBoxLayout()
            segmentParamsLayout.addWidget(QLabel(f"<b>{segmentGroupBox.title()}</b>"))
            segmentGroupBox.setTitle("")
            segmentParamsLayout.addWidget(self.segmentParamsScrollArea)
            segmentParamsLayout.addLayout(segmentButtonsLayout)
            self.secondColLayout.addLayout(segmentParamsLayout)
            self.paramsGroupPosMapper[self.segmentParamsScrollArea] = (1, col)

        gridLayout.addLayout(self.secondColLayout, row, col)

        gridLayout.addItem(QSpacerItem(10, 5), 0, col + 1)
        col += 2

        # Buttons layout (spans both columns)
        buttonsLayout = QHBoxLayout()
        cancelButton = widgets.cancelPushButton(" Cancel ")
        okButton = widgets.okPushButton(" Ok ")

        enableLoadingSavingRecipe = not is_tracker and (
            addPreProcessParams or addPostProcessParams
        )

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        if enableLoadingSavingRecipe:
            loadEntireRecipeButton = widgets.OpenFilePushButton("Load saved recipe...")
            saveEntireRecipeButton = widgets.savePushButton(
                "Save all parameters to recipe file..."
            )
            buttonsLayout.addWidget(loadEntireRecipeButton)
            buttonsLayout.addWidget(saveEntireRecipeButton)
            loadEntireRecipeButton.clicked.connect(self.loadEntireRecipe)
            saveEntireRecipeButton.clicked.connect(self.saveEntireRecipe)

        buttonsLayout.addWidget(okButton)

        buttonsLayout.setContentsMargins(0, 10, 0, 10)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        self.okButton = okButton

        # Extra params in right column
        row = 0
        self.extraArgsWidgets = None
        self.extraParamsScrollArea = None
        if extraParams is not None:
            self.extraParamsScrollArea = widgets.ScrollArea()
            extraParamsScrollAreaLayout = QVBoxLayout()
            self.extraParamsScrollArea.setVerticalLayout(extraParamsScrollAreaLayout)
            if extraParamsTitle is None:
                extraParamsTitle = "Additional parameters"

            self.extraGroupBox, self.extraArgsWidgets = self.createGroupParams(
                extraParams, extraParamsTitle
            )

            extraDefaultButton = widgets.reloadPushButton("Restore default")
            extraLoadLastSelButton = widgets.OpenFilePushButton("Load last parameters")
            extraButtonsLayout = QHBoxLayout()
            extraButtonsLayout.addStretch(1)
            extraButtonsLayout.addWidget(extraDefaultButton)
            extraButtonsLayout.addWidget(extraLoadLastSelButton)
            extraDefaultButton.clicked.connect(self.restoreDefaultExtra)
            section = f"{self.model_name}.extra"
            extraLoadLastSelButton.clicked.connect(
                partial(loadFunc, section, self.extraArgsWidgets)
            )

            extraParamsScrollAreaLayout.addWidget(self.extraGroupBox)

            extraParamsLayout = QVBoxLayout()
            extraParamsLayout.addWidget(QLabel(f"<b>{self.extraGroupBox.title()}</b>"))
            self.extraGroupBox.setTitle("")
            extraParamsLayout.addWidget(self.extraParamsScrollArea)
            extraParamsLayout.addLayout(extraButtonsLayout)
            self.paramsGroupPosMapper[self.extraParamsScrollArea] = (row, col)
            gridLayout.addLayout(extraParamsLayout, row, col)
            row += 1

        # Post-processing in right-most column
        self.postProcessGroupbox = None
        self.seeHereLabel = None
        thirdColumnLayout = QVBoxLayout()
        if addPostProcessParams:
            # Add minimum size spinbox which is valid for all models
            postProcessGroupbox = PostProcessSegmParams(
                "Post-processing segmentation parameters",
                posData,
                force_postprocess_2D=force_postprocess_2D,
            )
            postProcessGroupbox.setCheckable(True)
            postProcessGroupbox.setChecked(False)
            self.postProcessGroupbox = postProcessGroupbox

            thirdColumnLayout.addWidget(postProcessGroupbox)

            postProcDefaultButton = widgets.reloadPushButton("Restore default")
            postProcLoadLastSelButton = widgets.OpenFilePushButton(
                "Load last parameters"
            )
            postProcButtonsLayout = QHBoxLayout()
            postProcButtonsLayout.addStretch(1)
            postProcButtonsLayout.addWidget(postProcDefaultButton)
            postProcButtonsLayout.addWidget(postProcLoadLastSelButton)
            postProcDefaultButton.clicked.connect(self.restoreDefaultPostprocess)
            postProcLoadLastSelButton.clicked.connect(self.loadLastSelectionPostProcess)
            thirdColumnLayout.addLayout(postProcButtonsLayout)
            thirdColumnLayout.addSpacing(15)

            if url is not None:
                self.seeHereLabel = self.createSeeHereLabel(url)
                thirdColumnLayout.addWidget(self.seeHereLabel, alignment=Qt.AlignCenter)

            self.paramsGroupPosMapper[self.preProcessParamsWidget] = (row, col)

        # Additional segmentation params in right column
        self.additionalSegmGroupbox = None
        if add_additional_segm_params:
            thirdColumnLayout.addWidget(widgets.QHLine())
            additionalSegmGroupbox = self.getAdditionalSegmParams()
            thirdColumnLayout.addWidget(additionalSegmGroupbox)
            self.additionalSegmGroupbox = additionalSegmGroupbox
            self.paramsGroupPosMapper[self.additionalSegmGroupbox] = (row, col)

        thirdColumnLayout.addStretch(1)
        gridLayout.addLayout(thirdColumnLayout, row, col)
        row += 1

        # Add everything to main layout
        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.configPars = self.readLastSelection()
        if self.configPars is None:
            initLoadLastSelButton.setDisabled(True)
            segmentLoadLastSelButton.setDisabled(True)
            if self.postProcessGroupbox is not None:
                postProcLoadLastSelButton.setDisabled(True)

        if initLastParams:
            initLoadLastSelButton.click()
            if not self.skipSegmentation:
                segmentLoadLastSelButton.click()

            if self.extraArgsWidgets is not None:
                extraLoadLastSelButton.click()

            if self.postProcessGroupbox is not None:
                postProcLoadLastSelButton.click()

        try:
            self.connectCustomSignals(model_module)
        except Exception as e:
            printl(traceback.format_exc())

        self.setLayout(mainLayout)
        self.setFont(font)
        # self.setModal(True)

    def warningNoSegmRecipes(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            "No segmentation recipes found!<br><br>"
            "To create a segmentation recipe you need click on "
            "<code>Save all parameters to recipe file...</code> "
            "button."
        )
        msg.warning(self, "No segmentation recipes found!", txt)

    def selectIniFileToLoadEntireRecipe(self):
        import qtpy.compat

        recipe_filepath = qtpy.compat.getopenfilename(
            parent=self,
            caption="Select INI file to load entire recipe",
            filters="INI (*.ini);;All Files (*)",
        )[0]
        if not recipe_filepath:
            return

        self.loadRecipeFromFilepath(recipe_filepath)

        txt = html_utils.paragraph("Done!<br><br>Segmentation recipe loaded from:")
        msg = widgets.myMessageBox()
        msg.information(
            self,
            "Segmentation recipe loaded!",
            txt,
            commands=(recipe_filepath,),
            path_to_browse=os.path.dirname(recipe_filepath),
        )

        print("Done. Segmentation recipe loaded from:", recipe_filepath)

    def loadEntireRecipe(self):
        segm_recipes_path_model = os.path.join(segm_recipes_path, self.model_name)

        if not os.path.exists(segm_recipes_path_model):
            # self.warningNoSegmRecipes()
            self.selectIniFileToLoadEntireRecipe()
            return

        recipe_files = os.listdir(segm_recipes_path_model)

        if not recipe_files:
            # self.warningNoSegmRecipes()
            self.selectIniFileToLoadEntireRecipe()
            return

        headerLabels = ["Name", "Date Created"]
        items = []
        for recipe_file in recipe_files:
            cp = config.ConfigParser()
            cp.read(os.path.join(segm_recipes_path_model, recipe_file))
            date_created = cp["info"]["created_on"]
            items.append((recipe_file, date_created))

        browseButton = widgets.browseFileButton(
            "Select INI file...",
            title="Select INI file to load entire recipe",
            openFolder=False,
            start_dir=myutils.getMostRecentPath(),
            ext={"INI": ".ini"},
        )
        win = QTreeDialog(
            items,
            headerLabels=headerLabels,
            title="Select a segmentation recipe to load",
            infoText="Select a segmentation recipe to load:<br>",
            path_to_browse=segm_recipes_path_model,
            additional_buttons=(browseButton,),
        )
        browseButton.sigPathSelected.connect(
            partial(
                self.entireRecipeIniFileSelected,
                selectRecipeWin=win,
                sender=browseButton,
            )
        )
        win.exec_()
        if win.cancel or not hasattr(win, "selectedText"):
            print("Loading segmentation recipe cancelled.")
            return

        if win.clickedButton == browseButton:
            recipe_filepath = win.selectedIniFilepath
        else:
            recipe_filename = win.selectedText
            recipe_filepath = os.path.join(segm_recipes_path_model, recipe_filename)

        self.loadRecipeFromFilepath(recipe_filepath)

        txt = html_utils.paragraph("Done!<br><br>Segmentation recipe loaded from:")
        msg = widgets.myMessageBox()
        msg.information(
            self,
            "Segmentation recipe laoded!",
            txt,
            commands=(recipe_filepath,),
            path_to_browse=os.path.dirname(recipe_filepath),
        )

        print("Done. Segmentation recipe loaded from:", recipe_filepath)

    def entireRecipeIniFileSelected(
        self, recipe_filepath, selectRecipeWin=None, sender=None
    ):
        selectRecipeWin.selectedText = "None"
        selectRecipeWin.clickedButton = sender
        selectRecipeWin.selectedIniFilepath = recipe_filepath
        selectRecipeWin.cancel = False
        selectRecipeWin.close()

    def loadRecipeFromFilepath(self, recipe_filepath):
        cp = config.ConfigParser()
        cp.read(recipe_filepath)

        self.loadPreprocRecipe(configPars=cp)
        self.loadLastSelection(
            f"{self.model_name}.init", self.init_argsWidgets, configPars=cp
        )
        self.loadLastSelection(
            f"{self.model_name}.segment", self.argsWidgets, configPars=cp
        )
        if self.extraArgsWidgets:
            self.loadLastSelection(
                f"{self.model_name}.extra", self.extraArgsWidgets, configPars=cp
            )
        self.loadLastSelectionPostProcess(configPars=cp)

    def saveEntireRecipe(self):
        segm_recipes_path_model = os.path.join(segm_recipes_path, self.model_name)
        try:
            existingNames = os.listdir(segm_recipes_path_model)
        except FileNotFoundError:
            existingNames = []

        win = filenameDialog(
            title="Filename for segmentation recipe",
            basename="segmentation_recipe",
            ext=".ini",
            hintText="Insert a <b>filename</b> for the segmentation recipe:",
            allowEmpty=False,
            parent=self,
            existingNames=existingNames,
        )
        win.exec_()
        if win.cancel:
            return

        ini_filename = win.filename
        os.makedirs(segm_recipes_path, exist_ok=True)
        os.makedirs(segm_recipes_path_model, exist_ok=True)
        ini_filepath = os.path.join(segm_recipes_path_model, ini_filename)

        configPars = self.getConfigPars(create_new=True)

        if hasattr(self, "reduceMemUsageToggle"):
            configPars[f"{self.model_name}.additional_segm_params"] = {}
            reduceMemoryUsage = self.reduceMemUsageToggle.isChecked()
            option = self.reduceMemUsageToggle.label
            configPars[f"{self.model_name}.additional_segm_params"][option] = str(
                reduceMemoryUsage
            )

        configPars["info"] = {}
        configPars["info"]["created_on"] = datetime.datetime.now().strftime(
            r"%Y/%m/%d %H:%M"
        )

        with open(ini_filepath, "w") as configfile:
            configPars.write(configfile)

        txt = html_utils.paragraph("Done!<br><br>Segmentation recipe saved to:")
        msg = widgets.myMessageBox()
        msg.information(
            self,
            "Segmnentation recipe saved!",
            txt,
            commands=(ini_filepath,),
            path_to_browse=os.path.dirname(ini_filepath),
        )

        print("Done. Segmentation recipe saved to:", ini_filepath)

    def getAdditionalSegmParams(self):
        additionalSegmGroupbox = QGroupBox("Additional segmentation parameters")
        local_row = 0
        additionalSegmLayout = QGridLayout()
        option = "Reduce memory usage"
        additionalSegmLayout.addWidget(
            QLabel(f"{option}:  "), local_row, 0, alignment=Qt.AlignRight
        )
        self.reduceMemUsageToggle = widgets.Toggle()
        additionalSegmLayout.addWidget(
            self.reduceMemUsageToggle, local_row, 1, 1, 2, alignment=Qt.AlignCenter
        )
        self.reduceMemUsageToggle.label = option
        reduceMemUsageInfoButton = widgets.infoPushButton()
        additionalSegmLayout.addWidget(reduceMemUsageInfoButton, local_row, 3)
        reduceMemUsageInfoButton.clicked.connect(self.showInfoReduceMemUsage)
        additionalSegmLayout.setColumnStretch(0, 0)
        additionalSegmLayout.setColumnStretch(1, 1)
        additionalSegmLayout.setColumnStretch(3, 0)
        additionalSegmGroupbox.setLayout(additionalSegmLayout)
        return additionalSegmGroupbox

    def showInfoReduceMemUsage(self):
        infoText = html_utils.paragraph(f"""
            If you are experiencing memory issues, you can try reducing the 
            memory usage by toggling this option.<br><br>
            This will reduce the memory usage by segmenting timelapse data 
            frame-by-frame instead of all frames at once.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, "Reduce memory usage", infoText)

    def loadPreprocRecipe(self, configPars=None):
        if self.configPars is None and configPars is None:
            return

        if configPars is None:
            configPars = self.configPars

        preprocConfigPars = {}
        for section in configPars.sections():
            if not section.startswith(f"{self.model_name}.preprocess"):
                continue

            preprocConfigPars[section] = configPars[section]

        if not preprocConfigPars:
            return

        self.preProcessParamsWidget.loadRecipe(preprocConfigPars)

    def connectCustomSignals(self, model_module):
        if model_module is None:
            return

        if not hasattr(model_module, "CustomSignals"):
            return

        customSignals = model_module.CustomSignals()
        for slot_info in customSignals.slots_info:
            group = slot_info["group"]
            widget_name = slot_info["widget_name"]
            if group == "init":
                ArgsWidgets_list = self.init_argsWidgets
            else:
                ArgsWidgets_list = self.argsWidgets
            for argwidget in ArgsWidgets_list:
                if argwidget.name == widget_name:
                    signal = getattr(argwidget.widget, slot_info["signal"])
                    signal.connect(partial(slot_info["slot"], self))
                    break

    def selectedFeaturesRange(self):
        if self.postProcessGroupbox is None:
            return {}
        return self.postProcessGroupbox.selectedFeaturesRange()

    def groupedFeatures(self):
        if self.postProcessGroupbox is None:
            return {}
        return self.postProcessGroupbox.groupedFeatures()

    def setChannelNames(self, chNames):
        if not hasattr(self, "channelsCombobox"):
            return

        items = ["None"]
        items.extend(chNames)
        self.channelsCombobox.addItems(items)

    def getValueFromMetadata(self, name):
        try:
            value = self.df_metadata.at[name, "values"]
        except Exception as e:
            # traceback.print_exc()
            value = None
        return value

    def criticalSegmFileRequiredButNoneAvailable(self):
        model_name = f"{self.model_name} model"
        action_txt = (
            f"Please, segment the correct channel before using {self.model_name}."
        )
        if self.model_name == "skip_segmentation":
            model_name = "Skipping the segmentation"
            action_txt = (
                "To be able to skip the segmentation step, you need "
                "create at least one segmentation file."
            )
        txt = html_utils.paragraph(f"""
            <b>{model_name}</b> 
            <b>requires an additional segmentation file</b> 
            but there are none available!<br><br>
            {action_txt}
            <br><br>Thank you for you patience!
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Segmentation file required", txt)
        raise FileNotFoundError(
            "Model requires segmentation file but none are available."
        )

    def checkAddSegmEndnameCombobox(self, ArgSpec, groupBoxLayout, row):
        if ArgSpec.name != "Auxiliary segmentation file":
            return False

        if self.segmFileEndnames is None or not self.segmFileEndnames:
            self.criticalSegmFileRequiredButNoneAvailable()

        label = QLabel(f"{ArgSpec.name}:  ")
        groupBoxLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        items = self.segmFileEndnames
        self.segmEndnameCombobox = widgets.QCenteredComboBox()
        self.segmEndnameCombobox.addItems(items)
        groupBoxLayout.addWidget(self.segmEndnameCombobox, row, 1, 1, 2)
        return True

    def createGroupParams(self, ArgSpecs_list, groupName, addChannelSelector=False):
        ArgsWidgets_list = []
        groupBox = QGroupBox(groupName)
        groupBoxLayout = QGridLayout()

        start_row = 0
        if self.is_tracker and self.channels is not None and addChannelSelector:
            label = QLabel(f"Input image:  ")
            groupBoxLayout.addWidget(label, start_row, 0, alignment=Qt.AlignRight)
            items = ["None", *self.channels]
            self.channelCombobox = widgets.QCenteredComboBox()
            self.channelCombobox.addItems(items)
            groupBoxLayout.addWidget(self.channelCombobox, start_row, 1, 1, 2)
            if self.currentChannelName is not None:
                self.channelCombobox.setCurrentText(self.currentChannelName)
            infoText = (
                "Some trackers require the intensity image as input.<br><br>"
                "If this one does not require it, leave the selected value "
                "to `None`."
            )
            infoButton = self.getInfoButton("Input image", infoText)
            groupBoxLayout.addWidget(infoButton, start_row, 3)
            start_row += 1

        addSecondChannelSelector = addChannelSelector
        if len(ArgSpecs_list) > 0:
            if addSecondChannelSelector and ArgSpecs_list[0].docstring is not None:
                isSingleChannel = (
                    ArgSpecs_list[0].docstring.lower().find("single channel only") != -1
                )
                if isSingleChannel:
                    addSecondChannelSelector = False

        isDualChannelModel = self.model_name.find("cellpose") != -1 or any(
            [_types.is_second_channel_type(ArgSpec.type) for ArgSpec in ArgSpecs_list]
        )
        askSecondChannel = isDualChannelModel and addSecondChannelSelector

        if askSecondChannel:
            label = QLabel("Second channel (optional):  ")
            groupBoxLayout.addWidget(label, start_row, 0, alignment=Qt.AlignRight)
            self.channelsCombobox = widgets.QCenteredComboBox()
            groupBoxLayout.addWidget(self.channelsCombobox, start_row, 1, 1, 2)
            infoText = (
                "Some models can merge two channels (e.g., cyto + "
                "nucleus) to obtain better perfomance.\n\n"
                "Select a channel as additional input to the model."
            )
            infoButton = self.getInfoButton("Second channel", infoText)
            groupBoxLayout.addWidget(infoButton, start_row, 3)
            start_row += 1

        exclusive_withs = dict()
        default_exclusives = dict()
        row_mapper = dict()
        for row, ArgSpec in enumerate(ArgSpecs_list):
            if _types.is_second_channel_type(ArgSpec.type):
                continue

            if _types.is_widget_not_required(ArgSpec):
                continue

            row = row + start_row
            skip = self.checkAddSegmEndnameCombobox(ArgSpec, groupBoxLayout, row)
            if skip:
                continue

            arg_name = ArgSpec.name
            var_name = arg_name.replace("_", " ")
            var_name = f"{var_name[0].upper()}{var_name[1:]}"
            label = QLabel(f"{var_name}:  ")
            metadata_val = self.getValueFromMetadata(ArgSpec.name)
            groupBoxLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
            try:
                values = ArgSpec.type().values
                isCustomListType = True
            except Exception as err:
                isCustomListType = False

            isVectorEntry = False
            try:
                if isinstance(ArgSpec.type(), _types.Vector):
                    isVectorEntry = True
            except Exception as err:
                pass

            isFolderPath = False
            try:
                if isinstance(ArgSpec.type(), _types.FolderPath):
                    isFolderPath = True
            except Exception as err:
                pass

            try:
                exclusive_with = ArgSpec.type().is_exclusive_with
            except Exception as err:
                exclusive_with = []

            try:
                default_exclusive = ArgSpec.type().default_exclusive
            except Exception as err:
                default_exclusive = ""

            exclusive_withs[arg_name] = exclusive_with
            default_exclusives[arg_name] = default_exclusive
            row_mapper[arg_name] = row

            isCustomWidget = hasattr(ArgSpec.type, "isWidget")

            if isCustomWidget:
                widget = ArgSpec.type().widget
                defaultVal = ArgSpec.default
                valueSetter = widget.setValue
                valueGetter = widget.value
                changeSig = widget.sigValueChanged
                groupBoxLayout.addWidget(widget, row, 1, 1, 2)
            elif isVectorEntry:
                vectorLineEdit = widgets.VectorLineEdit()
                vectorLineEdit.setValue(ArgSpec.default)
                defaultVal = ArgSpec.default
                valueSetter = widgets.VectorLineEdit.setValue
                valueGetter = widgets.VectorLineEdit.value
                changeSig = vectorLineEdit.valueChanged
                widget = vectorLineEdit
                groupBoxLayout.addWidget(vectorLineEdit, row, 1, 1, 2)
            elif isFolderPath:
                folderPathControl = widgets.FolderPathControl()
                folderPathControl.setText(str(ArgSpec.default))
                widget = folderPathControl
                defaultVal = str(ArgSpec.default)
                valueSetter = widgets.FolderPathControl.setText
                valueGetter = widgets.FolderPathControl.path
                changeSig = widget.sigValueChanged
                groupBoxLayout.addWidget(folderPathControl, row, 1, 1, 2)
            elif ArgSpec.type == bool:
                booleanGroup = QButtonGroup()
                booleanGroup.setExclusive(True)
                checkBox = widgets.Toggle()
                checkBox.setChecked(ArgSpec.default)
                defaultVal = ArgSpec.default
                valueSetter = widgets.Toggle.setChecked
                valueGetter = widgets.Toggle.isChecked
                changeSig = checkBox.toggled
                widget = checkBox
                groupBoxLayout.addWidget(
                    checkBox, row, 1, 1, 2, alignment=Qt.AlignCenter
                )
            elif ArgSpec.type == int:
                spinBox = widgets.SpinBox()
                if metadata_val is None:
                    spinBox.setValue(ArgSpec.default)
                else:
                    spinBox.setValue(int(metadata_val))
                    spinBox.isMetadataValue = True
                defaultVal = ArgSpec.default
                valueSetter = QSpinBox.setValue
                valueGetter = QSpinBox.value
                changeSig = spinBox.sigValueChanged
                widget = spinBox
                groupBoxLayout.addWidget(spinBox, row, 1, 1, 2)
            elif ArgSpec.type == float:
                doubleSpinBox = widgets.FloatLineEdit()
                if metadata_val is None:
                    doubleSpinBox.setValue(ArgSpec.default)
                else:
                    doubleSpinBox.setValue(float(metadata_val))
                    doubleSpinBox.isMetadataValue = True
                widget = doubleSpinBox
                defaultVal = ArgSpec.default
                valueSetter = widgets.FloatLineEdit.setValue
                valueGetter = widgets.FloatLineEdit.value
                changeSig = doubleSpinBox.valueChanged
                groupBoxLayout.addWidget(doubleSpinBox, row, 1, 1, 2)
            elif ArgSpec.type == os.PathLike:
                filePathControl = widgets.filePathControl()
                filePathControl.setText(str(ArgSpec.default))
                widget = filePathControl
                defaultVal = str(ArgSpec.default)
                valueSetter = widgets.filePathControl.setText
                valueGetter = widgets.filePathControl.path
                changeSig = filePathControl.sigValueChanged
                groupBoxLayout.addWidget(filePathControl, row, 1, 1, 2)
            elif isCustomListType:
                items = ArgSpec.type().values
                defaultVal = str(ArgSpec.default)
                combobox = widgets.AlphaNumericComboBox()
                combobox.addItems(items)
                combobox.setCurrentValue(defaultVal)
                valueSetter = widgets.AlphaNumericComboBox.setCurrentValue
                valueGetter = widgets.AlphaNumericComboBox.currentValue
                changeSig = combobox.currentTextChanged
                widget = combobox
                groupBoxLayout.addWidget(combobox, row, 1, 1, 2)
            else:
                lineEdit = QLineEdit()
                lineEdit.setText(str(ArgSpec.default))
                lineEdit.setAlignment(Qt.AlignCenter)
                widget = lineEdit
                defaultVal = str(ArgSpec.default)
                valueSetter = QLineEdit.setText
                valueGetter = QLineEdit.text
                changeSig = lineEdit.editingFinished
                groupBoxLayout.addWidget(lineEdit, row, 1, 1, 2)

            if ArgSpec.desc:
                infoButton = self.getInfoButton(ArgSpec.name, ArgSpec.desc)
                groupBoxLayout.addWidget(infoButton, row, 3)

            argsInfo = ArgWidget(
                name=ArgSpec.name,
                type=ArgSpec.type,
                widget=widget,
                defaultVal=defaultVal,
                valueSetter=valueSetter,
                valueGetter=valueGetter,
                changeSig=changeSig,
            )
            ArgsWidgets_list.append(argsInfo)

        exclusive_group = core.connected_components_in_undirected_graph(exclusive_withs)

        for group in exclusive_group:
            if len(group) == 1:
                continue
            for arg_name in group:
                default_exclusive = default_exclusives[arg_name]
                row = row_mapper[arg_name]

                argsInfo = ArgsWidgets_list[row]
                valueSetter = argsInfo.valueSetter
                widget = argsInfo.widget
                valueGetter = argsInfo.valueGetter

                argsInfo.valueGetter = qutils.replace_certain_vals(
                    argsInfo.valueGetter, default_exclusive, None
                )

                for arg_name_other in group:
                    if arg_name == arg_name_other:
                        continue
                    row_other = row_mapper[arg_name_other]
                    argsInfo_other = ArgsWidgets_list[row_other]
                    changeSig_other = argsInfo_other.changeSig
                    changeSig_other.connect(
                        partial(
                            qutils.set_exclusive_valueSetter,
                            widget,
                            valueSetter,
                            default_exclusive,
                        )
                    )

        groupBoxLayout.setColumnStretch(0, 0)
        groupBoxLayout.setColumnStretch(1, 1)
        groupBoxLayout.setColumnStretch(3, 0)
        nrows = groupBoxLayout.rowCount()
        groupBoxLayout.setRowStretch(nrows, 1)

        groupBox.setLayout(groupBoxLayout)
        return groupBox, ArgsWidgets_list

    def getInfoButton(self, param_name, infoText):
        infoButton = widgets.infoPushButton()
        infoButton.param_name = param_name
        infoButton.setToolTip(
            f"Click to get more info about `{param_name}` parameter..."
        )
        infoButton.infoText = infoText
        infoButton.clicked.connect(self.showInfoParam)
        return infoButton

    def showInfoParam(self):
        text = self.sender().infoText
        text = text.replace("\n", "<br>")
        text = html_utils.rst_urls_to_html(text)
        text = html_utils.rst_to_html(text)
        text = html_utils.paragraph(text)
        param_name = self.sender().param_name
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, f"Info about `{param_name}` parameter", text)

    def restoreDefaultInit(self):
        for argWidget in self.init_argsWidgets:
            defaultVal = argWidget.defaultVal
            widget = argWidget.widget
            valueSetter = argWidget.valueSetter
            qutils.set_exclusive_valueSetter(widget, valueSetter, defaultVal)

    def restoreDefaultSegment(self):
        for argWidget in self.argsWidgets:
            defaultVal = argWidget.defaultVal
            widget = argWidget.widget
            valueSetter = argWidget.valueSetter
            qutils.set_exclusive_valueSetter(widget, valueSetter, defaultVal)

    def restoreDefaultExtra(self):
        for argWidget in self.extraArgsWidgets:
            defaultVal = argWidget.defaultVal
            widget = argWidget.widget
            valueSetter = argWidget.valueSetter
            qutils.set_exclusive_valueSetter(widget, valueSetter, defaultVal)

    def restoreDefaultPostprocess(self):
        self.postProcessGroupbox.restoreDefault()

    def readLastSelection(self):
        self.ini_path = os.path.join(settings_folderpath, self.ini_filename)

        if not os.path.exists(self.ini_path):
            return None

        print(f"Reading last selected parameters from: {self.ini_path}")
        configPars = config.ConfigParser()
        configPars.read(self.ini_path)
        return configPars

    def setValuesFromParams(self, init_params, segment_params, extra_params=None):
        sections = {
            f"{self.model_name}.init": (init_params, self.init_argsWidgets),
            f"{self.model_name}.segment": (segment_params, self.argsWidgets),
        }
        if extra_params is not None:
            sections[f"{self.model_name}.extra"] = (extra_params, self.extraArgsWidgets)

        for section, values in sections.items():
            params, argWidgetList = values
            for argWidget in argWidgetList:
                val = params.get(argWidget.name)
                widget = argWidget.widget
                if val is None:
                    continue
                casters = [lambda x: x, int, float, str, bool]
                for caster in casters:
                    try:
                        argWidget.valueSetter(widget, caster(val))
                        break
                    except Exception as e:
                        continue

    def loadLastSelection(self, section, argWidgetList, checked=False, configPars=None):
        if self.configPars is None and configPars is None:
            return

        if configPars is None:
            configPars = self.configPars

        getters = ["getboolean", "getint", "getfloat", "get"]
        try:
            options = configPars.options(section)
        except Exception:
            return

        for argWidget in argWidgetList:
            option = argWidget.name
            val = None
            for getter in getters:
                try:
                    val = getattr(configPars, getter)(section, option)
                    break
                except Exception as err:
                    pass
            widget = argWidget.widget

            if hasattr(widget, "isMetadataValue"):
                continue
            if val is None:
                continue

            casters = [lambda x: x, int, float, str, bool]
            for caster in casters:
                try:
                    val = caster(val)
                    valueSetter = argWidget.valueSetter
                    qutils.set_exclusive_valueSetter(widget, valueSetter, val)
                    break
                except Exception as e:
                    printl(traceback.format_exc())
                    continue

    def loadLastSelectionPostProcess(self, checked=False, configPars=None):
        if self.postProcessGroupbox is None:
            return

        postProcessSection = f"{self.model_name}.postprocess"

        if isinstance(configPars, bool):
            configPars = None

        if configPars is None:
            configPars = self.configPars

        if postProcessSection in configPars.sections():
            try:
                minSize = configPars.getint(postProcessSection, "minSize", fallback=10)
            except ValueError:
                minSize = 10

            try:
                minSolidity = configPars.getfloat(
                    postProcessSection, "minSolidity", fallback=0.5
                )
            except ValueError:
                minSolidity = 0.5

            try:
                maxElongation = configPars.getfloat(
                    postProcessSection, "maxElongation", fallback=3
                )
            except ValueError:
                maxElongation = 3

            try:
                minObjSizeZ = configPars.getint(
                    postProcessSection, "min_obj_no_zslices", fallback=3
                )
            except ValueError:
                minObjSizeZ = 3

            kwargs = {
                "min_solidity": minSolidity,
                "min_area": minSize,
                "max_elongation": maxElongation,
                "min_obj_no_zslices": minObjSizeZ,
            }
            self.postProcessGroupbox.restoreFromKwargs(kwargs)

            applyPostProcessing = configPars.getboolean(
                postProcessSection, "applyPostProcessing"
            )
            self.postProcessGroupbox.setChecked(applyPostProcessing)

        customPostProcessSection = f"{self.model_name}.custom_postprocess"
        if postProcessSection not in configPars.sections():
            return

        selectFeaturesWidget = self.postProcessGroupbox.selectedFeaturesDialog.groupbox
        selectFeaturesWidget.resetFields()
        f = 0
        for col_name, value in configPars[customPostProcessSection].items():
            low, high = value.split(",")
            low = low.strip()
            high = high.strip()
            if f > 0:
                selectFeaturesWidget.addFeatureField()

            selector = selectFeaturesWidget.selectors[f]
            selector.selectButton.setText(col_name)
            selector.selectButton.setFlat(True)

            feature_group = measurements.get_metric_group_name(col_name)
            selector.featureGroup = feature_group

            if low != "None":
                try:
                    low_val = int(low)
                except ValueError:
                    low_val = float(low)

                selector.lowRangeWidgets.checkbox.setChecked(True)
                selector.lowRangeWidgets.spinbox.setValue(low_val)

            if high != "None":
                try:
                    high_val = int(high)
                except ValueError:
                    high_val = float(high)

                selector.highRangeWidgets.checkbox.setChecked(True)
                selector.highRangeWidgets.spinbox.setValue(high_val)

            f += 1

    def createSeeHereLabel(self, url):
        htmlTxt = f'<a href="{url}">here</a>'
        seeHereLabel = QLabel()
        seeHereLabel.setText(f"""
            <p style="font-size:13px">
                See {htmlTxt} for details on the parameters
            </p>
        """)
        seeHereLabel.setTextFormat(Qt.RichText)
        seeHereLabel.setTextInteractionFlags(Qt.TextBrowserInteraction)
        seeHereLabel.setOpenExternalLinks(True)
        seeHereLabel.setStyleSheet("padding:12px 0px 0px 0px;")
        return seeHereLabel

    def argsWidgets_to_kwargs(self, argsWidgets):
        kwargs_dict = {
            argWidget.name: argWidget.valueGetter(argWidget.widget)
            for argWidget in argsWidgets
        }
        return kwargs_dict

    def getInitKwargs(self):
        init_kwargs = self.argsWidgets_to_kwargs(self.init_argsWidgets)
        if hasattr(self, "segmEndnameCombobox"):
            init_kwargs["segm_endname"] = self.segmEndnameCombobox.currentText()

        return init_kwargs

    def getModelKwargs(self):
        if self.skipSegmentation:
            return {}

        return self.argsWidgets_to_kwargs(self.argsWidgets)

    def getExtraKwargs(self):
        if self.extraArgsWidgets is None:
            return {}

        return self.argsWidgets_to_kwargs(self.extraArgsWidgets)

    def ok_cb(self, checked):
        self.cancel = False
        self.preproc_recipe = None
        if self.preProcessParamsWidget is not None:
            self.preproc_recipe = self.preProcessParamsWidget.recipe()
            if self.preproc_recipe is None:
                return

        self.init_kwargs = self.getInitKwargs()

        if self.extraArgsWidgets:
            self.extra_kwargs = self.getExtraKwargs()

        self.model_kwargs = self.getModelKwargs()
        self.segment_kwargs = self.model_kwargs

        if self.postProcessGroupbox is not None:
            self.applyPostProcessing = self.postProcessGroupbox.isChecked()
            self.standardPostProcessKwargs = self.postProcessGroupbox.kwargs()
        self.secondChannelName = None
        if hasattr(self, "channelsCombobox"):
            self.secondChannelName = self.channelsCombobox.currentText()
        if self.secondChannelName == "None":
            self.secondChannelName = None
        self.inputChannelName = "None"
        if self.channelCombobox is not None:
            self.inputChannelName = self.channelCombobox.currentText()

        self.reduceMemoryUsage = False
        if hasattr(self, "reduceMemUsageToggle"):
            self.reduceMemoryUsage = self.reduceMemUsageToggle.isChecked()
        self.customPostProcessFeatures = self.selectedFeaturesRange()
        self.customPostProcessGroupedFeatures = self.groupedFeatures()
        self.saveLastSelection()
        self.freePosData()
        self.close()

    def freePosData(self):
        if hasattr(self, "postProcessGroupbox"):
            try:
                for (
                    selector
                ) in self.postProcessGroupbox.selectedFeaturesDialog.groupbox.selectors:
                    qutils.hardDelete(selector)
            except AttributeError:
                pass
            try:
                qutils.hardDelete(
                    self.postProcessGroupbox.selectedFeaturesDialog.groupbox
                )
            except AttributeError:
                pass
            try:
                qutils.hardDelete(self.postProcessGroupbox.selectedFeaturesDialog)
            except AttributeError:
                pass
            try:
                qutils.hardDelete(self.postProcessGroupbox)
            except AttributeError:
                pass

    def getConfigPars(self, create_new=False):
        if self.configPars is None or create_new:
            configPars = config.ConfigParser()
        else:
            configPars = self.configPars

        if self.preProcessParamsWidget is not None:
            preprocCp = self.preProcessParamsWidget.recipeConfigPars(self.model_name)
            for section in preprocCp.sections():
                configPars[section] = preprocCp[section]

        configPars[f"{self.model_name}.init"] = {}
        configPars[f"{self.model_name}.segment"] = {}
        configPars[f"{self.model_name}.extra"] = {}

        init_kwargs = self.getInitKwargs()
        model_kwargs = self.getModelKwargs()

        for key, val in init_kwargs.items():
            configPars[f"{self.model_name}.init"][key] = str(val)
        for key, val in model_kwargs.items():
            configPars[f"{self.model_name}.segment"][key] = str(val)
        if self.extraArgsWidgets:
            extra_kwargs = self.getExtraKwargs()
            for key, val in extra_kwargs.items():
                configPars[f"{self.model_name}.extra"][key] = str(val)

        configPars[f"{self.model_name}.postprocess"] = {}
        if self.postProcessGroupbox is not None:
            postProcKwargs = self.postProcessGroupbox.kwargs()
            postProcessConfig = configPars[f"{self.model_name}.postprocess"]
            postProcessConfig["minSize"] = str(postProcKwargs["min_area"])
            postProcessConfig["minSolidity"] = str(postProcKwargs["min_solidity"])
            postProcessConfig["maxElongation"] = str(postProcKwargs["max_elongation"])
            postProcessConfig["min_obj_no_zslices"] = str(
                postProcKwargs["min_obj_no_zslices"]
            )
            postProcessConfig["applyPostProcessing"] = str(
                self.postProcessGroupbox.isChecked()
            )

        custom_postproc_section = f"{self.model_name}.custom_postprocess"
        configPars[custom_postproc_section] = {}
        if self.postProcessGroupbox is not None:
            selectFeaturesWidget = (
                self.postProcessGroupbox.selectedFeaturesDialog.groupbox
            )
            for selector in selectFeaturesWidget.selectors:
                col_name = selector.selectButton.text()
                lowStr = "None"
                highStr = "None"
                if selector.lowRangeWidgets.checkbox.isChecked():
                    lowVal = selector.lowRangeWidgets.spinbox.value()
                    lowStr = str(lowVal)
                if selector.highRangeWidgets.checkbox.isChecked():
                    highVal = selector.highRangeWidgets.spinbox.value()
                    highStr = str(highVal)

                configPars[custom_postproc_section][col_name] = f"{lowStr}, {highStr}"

        return configPars

    def saveLastSelection(self):
        self.configPars = self.getConfigPars()
        with open(self.ini_path, "w") as configfile:
            self.configPars.write(configfile)

        mode = "Segmentation" if not self.is_tracker else "Tracking"

        print(f'{mode} parameters saved at "{self.ini_path}"')

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        if self.model_name == "thresholding":
            self.segmentGroupBox.setDisabled(True)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        self.freePosData()
        if hasattr(self, "loop"):
            self.loop.exit()

    def cancel_cb(self, checked):
        self.cancel = True
        self.freePosData()

    def showEvent(self, event) -> None:
        buttonHeight = self.okButton.minimumSizeHint().height()
        heightInitParams = self.initParamsScrollArea.minimumHeightNoScrollbar()
        heightLeft = 70 + buttonHeight
        heightCenter = heightInitParams
        heightRight = 0
        if self.segmentParamsScrollArea is not None:
            heightSegmentParams = (
                self.segmentParamsScrollArea.minimumHeightNoScrollbar()
            )
            heightCenter += heightSegmentParams + 70 + buttonHeight

            rowInitParams, _ = self.paramsGroupPosMapper[self.initParamsScrollArea]
            rowSegmParams, _ = self.paramsGroupPosMapper[self.segmentParamsScrollArea]

            numInitParams = len(self.init_params)
            numSegmentParams = len(self.segment_params)

            try:
                segmentParamsStretch = max(1, round(numSegmentParams / numInitParams))
            except ZeroDivisionError as err:
                segmentParamsStretch = 1
            self.secondColLayout.setStretch(rowInitParams, 1)
            self.secondColLayout.setStretch(rowSegmParams, segmentParamsStretch)

        if self.extraParamsScrollArea is not None:
            heightRight += (
                self.extraParamsScrollArea.minimumHeightNoScrollbar()
                + 70
                + buttonHeight
            )

        if self.additionalSegmGroupbox is not None:
            heightRight += self.additionalSegmGroupbox.minimumSizeHint().height()
            heightRight += buttonHeight
        if self.preProcessParamsWidget is not None:
            heightPreprocParams = self.preProcessParamsWidget.minimumSizeHint().height()
            heightLeft += heightPreprocParams
            heightLeft += buttonHeight
        if self.postProcessGroupbox is not None:
            heightRight += self.postProcessGroupbox.minimumSizeHint().height()
            heightRight += buttonHeight
        if self.seeHereLabel is not None:
            heightRight += self.seeHereLabel.minimumSizeHint().height()
        height = max(heightLeft, heightRight, heightCenter)
        screenHeight = self.screen().size().height()
        screenGeom = self.screen().geometry()
        screenLeft = screenGeom.left()
        screenRight = screenGeom.right()
        screenCenter = (screenLeft + screenRight) / 2
        width = self.sizeHint().width()
        windowLeft = int(screenCenter - width / 2)
        self.move(windowLeft, 20)

        if height >= screenHeight - 150:
            height = screenHeight - 150
        self.resize(width, height)


class downloadModel:
    def __init__(self, model_name, parent=None):
        self.loop = None
        self.model_name = model_name
        self._parent = parent

    def download(self):
        model_url = myutils._model_url(self.model_name)
        if model_url is None:
            return

        _, model_path = myutils.get_model_path(self.model_name, create_temp_dir=False)
        model_name = self.model_name
        model_exists = myutils.check_model_exists(model_path, model_name)
        if not model_exists:
            self.warnDownloadModel(model_path, self.model_name)
        try:
            self._parent.logger.info(
                f'Downloading {self.model_name} model(s) to "{model_path}"'
            )
        except Exception as err:
            pass

        success = myutils.download_model(self.model_name)
        if not success:
            self.criticalDowloadFailed()

    def warnDownloadModel(self, model_path, model_name):
        txt = html_utils.paragraph(
            "Cell-ACDC needs to <b>download the model</b> "
            f"<code>{model_name}</code>.<br><br>"
            "The files will be dowloaded into the following folder:<br><br>"
            f"<code>{model_path}</code><br><br>"
            "<b>Progress</b> will be displayed in the <b>terminal</b>.<br>"
        )
        msg = widgets.myMessageBox()
        msg.information(self._parent, "Download model", txt)

    def criticalDowloadFailed(self):
        import cellacdc

        model_name = self.model_name
        m = model_name.lower()
        weights_filenames = getattr(cellacdc, f"{m}_weights_filenames")
        url, alternative_url = myutils._model_url(model_name, return_alternative=True)
        url_href = f'<a href="{url}">this link</a>'
        alternative_url_href = f'<a href="{alternative_url}">this link</a>'
        _, model_path = myutils.get_model_path(model_name, create_temp_dir=False)
        txt = html_utils.paragraph(f"""
            Automatic download of {model_name} failed.<br><br>
            Please, <b>manually download</b> the model weights from {url_href} or
            {alternative_url_href}.<br><br>
            Next, unzip the content (or move the files if not a zip archive) 
            of the downloaded file into the following folder:<br><br>
            <code>{model_path}</code><br><br>
            <i>NOTE: if clicking on the link above does not work
            copy one of the links below and paste it into the browser</i><br><br>
            <code>{url}</code>
            <br><br>
            <code>{alternative_url}</code>
        """)
        weights_paths = [os.path.join(model_path, f) for f in weights_filenames]
        weights = "\n\n".join(weights_paths)
        detailsText = f"Files that {model_name} requires:\n\n{weights}"
        msg = widgets.myMessageBox()
        msg.critical(
            self._parent,
            f"Download of {model_name} failed",
            txt,
            detailsText=detailsText,
        )
        self.close_()

    def close_(self):
        return


class SelectAcdcDfVersionToRestore(QBaseDialog):
    def __init__(self, posData, parent=None):
        super().__init__(parent=parent)

        self.cancel = True

        self.setWindowTitle("Select annotations table to restore")

        mainLayout = QVBoxLayout()

        acdc_df_filename = os.path.basename(posData.acdc_output_csv_path)
        instructionsLabel = html_utils.paragraph(
            f"Select an <b>older version</b> of the <code>{acdc_df_filename}</code> "
            "annotations table to load.<br><br>"
            "The datetime refers to the time you replaced the old version with "
            "a newer one.<br><br>"
        )
        mainLayout.addWidget(QLabel(instructionsLabel))

        self.savedListBox = None
        if os.path.exists(posData.acdc_output_backup_zip_path):
            zip_path = posData.acdc_output_backup_zip_path
            self.savedArchivefilepath = zip_path
            with zipfile.ZipFile(zip_path, mode="r") as zip:
                csv_names = natsorted(zip.namelist(), reverse=True)

            keys = [csv_name[:-4] for csv_name in csv_names]

            self.savedKeys = keys
            f = load.ISO_TIMESTAMP_FORMAT
            timestamps = [datetime.datetime.strptime(key, f) for key in keys]
            items = [date.strftime(r"%d %b %Y, %H:%M:%S") for date in timestamps]
            mainLayout.addWidget(QLabel("Saved annotations:"))
            self.savedListBox = widgets.listWidget()
            self.savedListBox.addItems(items)
            mainLayout.addWidget(self.savedListBox)
            self.savedListBox.itemSelectionChanged.connect(self.onItemSelectionChanged)

        recovery_folderpath = posData.recoveryFolderpath()
        unsaved_recovery_folderpath = os.path.join(recovery_folderpath, "never_saved")
        self.neverSavedFolderpath = unsaved_recovery_folderpath
        files = myutils.listdir(unsaved_recovery_folderpath)
        csv_files = [file for file in files if file.endswith(".csv")]
        self.neverSavedListBox = None
        if csv_files:
            csv_names = natsorted(csv_files, reverse=True)
            keys = [csv_name[:-4] for csv_name in csv_names]
            self.neverSavedKeys = keys
            f = load.ISO_TIMESTAMP_FORMAT
            timestamps = [datetime.datetime.strptime(key, f) for key in keys]
            items = [date.strftime(r"%d %b %Y, %H:%M:%S") for date in timestamps]
            mainLayout.addWidget(QLabel("Never saved annotations:"))
            self.neverSavedListBox = widgets.listWidget()
            self.neverSavedListBox.addItems(items)
            mainLayout.addWidget(self.neverSavedListBox)
            self.neverSavedListBox.itemSelectionChanged.connect(
                self.onItemSelectionChanged
            )

        cancelOkLayout = widgets.CancelOkButtonsLayout()

        cancelOkLayout.okButton.clicked.connect(self.ok_cb)
        cancelOkLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(cancelOkLayout)

        self.setLayout(mainLayout)

        self.setFont(font)

    def ok_cb(self):
        self.cancel = False
        try:
            for i in range(self.savedListBox.count()):
                item = self.savedListBox.item(i)
                if item.isSelected():
                    self.selectedTimestamp = item.text()
                    self.selectedKey = self.savedKeys[i]
                    self.archiveFilePath = self.savedArchivefilepath
                    break
        except Exception as e:
            pass

        try:
            for i in range(self.neverSavedListBox.count()):
                item = self.neverSavedListBox.item(i)
                if item.isSelected():
                    self.selectedTimestamp = item.text()
                    self.selectedKey = self.neverSavedKeys[i]
                    self.archiveFilePath = self.neverSavedFolderpath
                    break
        except Exception as e:
            pass
        self.close()

    def onItemSelectionChanged(self):
        otherListBox = (
            self.savedListBox
            if self.sender() == self.neverSavedListBox
            else self.neverSavedListBox
        )
        if otherListBox is None:
            return
        for i in range(otherListBox.count()):
            item = otherListBox.item(i)
            item.setSelected(False)


class ChangeUserProfileFolderPathDialog(QBaseDialog):
    def __init__(self, posData, parent=None):
        super().__init__(parent=parent)

        self.cancel = True

        self.setWindowTitle("Change user profile folder path")

        mainLayout = QVBoxLayout()

        acdc_folders = load.get_all_acdc_folders(user_profile_path)
        acdc_folders_format = [f"  - {folder}" for folder in acdc_folders]
        acdc_folders_format = "<br>".join(acdc_folders_format)

        txt = f"""
            Current user profile path:<br><br>
            <code>{user_profile_path}</code><br><br>
            The user profile contains the following Cell-ACDC folders:<br><br>
            {acdc_folders_format}<br><br>
            After clicking "Ok" you will be <b>asked to select the folder</b> where 
            you want to <b>migrate</b> the user profile data. 
        """

        txt = html_utils.paragraph(txt)
        label = QLabel(txt)

        mainLayout.addWidget(label)

        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch()

        self.setLayout(mainLayout)

    def ok_cb(self):
        self.cancel = False
        self.close()


class QInput(QBaseDialog):
    def __init__(self, parent=None, title="Input"):
        self.cancel = True
        self.allowEmpty = True

        super().__init__(parent)

        self.setWindowTitle(title)

        self.mainLayout = QVBoxLayout()

        self.infoLabel = QLabel()
        self.mainLayout.addWidget(self.infoLabel)

        promptLayout = QHBoxLayout()
        self.promptLabel = QLabel()
        promptLayout.addWidget(self.promptLabel)
        self.lineEdit = QLineEdit()
        promptLayout.addWidget(self.lineEdit)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addLayout(promptLayout)
        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)

        self.buttonsLayout = buttonsLayout

        self.setFont(font)
        self.setLayout(self.mainLayout)

    def askText(self, prompt, infoText="", allowEmpty=False):
        self.allowEmpty = allowEmpty
        if infoText:
            infoText = f"{infoText}<br>"
            self.infoLabel.setText(html_utils.paragraph(infoText))
        self.promptLabel.setText(prompt)
        self.exec_(resizeWidthFactor=1.5)

    def ok_cb(self):
        self.answer = self.lineEdit.text()
        if not self.allowEmpty and not self.answer:
            msg = widgets.myMessageBox(showCentered=False)
            msg.critical(self, "Empty", "Entry cannot be empty.")
            return
        self.cancel = False
        self.close()


class InstallPyTorchDialog(QBaseDialog):
    def __init__(self, parent=None, caller_name="Cell-ACDC"):
        super().__init__(parent=parent)

        self.cancel = True

        mainLayout = QVBoxLayout()

        innerLayout = QGridLayout()

        iconLabel = QLabel(self)
        standardIcon = getattr(QStyle, "SP_MessageBoxInformation")
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        iconLabel.setPixmap(pixmap)
        innerLayout.addWidget(iconLabel, 0, 0, alignment=Qt.AlignTop)

        href = html_utils.href_tag("How to install PyTorch", urls.install_pytorch)
        important = html_utils.to_admonition(
            """
            Should you choose to install PyTorch yourself, <b>make sure to 
            activate<br>
            the correct <code>acdc</code> environment first</b>.
        """,
            admonition_type="important",
        )

        infoText = html_utils.paragraph(f"""
            {caller_name} needs to <b>install the package</b> <code>PyTorch</code>.<br><br>
            Select your preferences and click ok to install it now. 
            You will have to <b>confirm the installation in the terminal</b>.<br><br>
            Alternatively, you can close {caller_name} and run the command 
            yourself.<br><br>
            For more details see this guide: {href}<br>
            {important}
        """)
        innerLayout.addWidget(QLabel(infoText), 0, 1)
        innerLayout.addItem(QSpacerItem(10, 10), 1, 1)

        preferencesLayout = QGridLayout()

        row = 0
        self.osCombobox = QComboBox()
        self.osCombobox.addItems(["Linux", "Mac", "Windows"])
        preferencesLayout.addWidget(QLabel("Your OS"), row, 0)
        preferencesLayout.addWidget(self.osCombobox, row, 1)

        if is_mac:
            self.osCombobox.setCurrentText("Mac")
        elif is_win:
            self.osCombobox.setCurrentText("Windows")

        row += 1
        self.pkgManagerCombobox = QComboBox()
        self.pkgManagerCombobox.addItems(["Pip"])
        if not is_conda_env():
            self.pkgManagerCombobox.setCurrentText("Pip")
            self.pkgManagerCombobox.setDisabled(True)

        preferencesLayout.addWidget(QLabel("Package manager"), row, 0)
        preferencesLayout.addWidget(self.pkgManagerCombobox, row, 1)

        row += 1
        self.cmptPlatformCombobox = QComboBox()
        self.cmptPlatformCombobox.addItems(
            ["CPU", "CUDA 11.8 (NVIDIA GPU)", "CUDA 12.1 (NVIDIA GPU)"]
        )

        preferencesLayout.addWidget(QLabel("Compute Platform"), row, 0)
        preferencesLayout.addWidget(self.cmptPlatformCombobox, row, 1)

        row += 1
        pip_prefix, conda_prefix = myutils.get_pip_conda_prefix()
        self.commandWidget = widgets.CopiableCommandWidget(
            command=f"{pip_prefix} torch"
        )
        preferencesLayout.addWidget(QLabel("Run this command: "), row, 0)
        preferencesLayout.addWidget(self.commandWidget, row, 1, 1, 2)
        preferencesLayout.setColumnStretch(0, 0)
        preferencesLayout.setColumnStretch(1, 0)
        preferencesLayout.setColumnStretch(2, 1)

        innerLayout.addLayout(preferencesLayout, 2, 1)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(innerLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        self.osCombobox.currentTextChanged.connect(self.updateCommand)
        self.pkgManagerCombobox.currentTextChanged.connect(self.updateCommand)
        self.cmptPlatformCombobox.currentTextChanged.connect(self.updateCommand)

        self.updateCommand()

    def updateCommand(self, *args, **kwargs):
        osText = self.osCombobox.currentText()
        pkgManager = self.pkgManagerCombobox.currentText()
        cmptPlatform = self.cmptPlatformCombobox.currentText()
        command = myutils.get_pytorch_command()[osText][pkgManager][cmptPlatform]
        self.commandWidget.setCommand(command)

    def ok_cb(self):
        self.command = self.commandWidget.command()
        self.cancel = False
        self.close()

# Sibling imports (deferred to avoid import cycles)
from ._base import (
    ArgWidget,
)
from .general import (
    QTreeDialog,
)
from .metadata import (
    filenameDialog,
)
from .preprocess import (
    PostProcessSegmParams,
    PreProcessParamsWidget,
)

