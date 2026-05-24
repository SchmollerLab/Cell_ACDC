"""Cell-ACDC dialog windows: measurements."""

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

class SetMeasurementsDialog(QBaseDialog):
    sigClosed = Signal()
    sigCancel = Signal()
    sigRestart = Signal()

    def __init__(
        self,
        loadedChNames,
        notLoadedChNames,
        isZstack,
        isSegm3D,
        favourite_funcs=None,
        parent=None,
        allPos_acdc_df_cols=None,
        acdc_df_path=None,
        posData=None,
        addCombineMetricCallback=None,
        allPosData=None,
        is_concat=False,
        isSingleSelection=False,
        state=None,
    ):
        super().__init__(parent=parent)

        self.checkBoxedGroup = QButtonGroup()
        self.checkBoxedGroup.setExclusive(isSingleSelection)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        self.cancel = True

        self.delExistingCols = False
        self.okClicked = False
        self.is_concat = is_concat
        self.allPos_acdc_df_cols = allPos_acdc_df_cols
        self.acdc_df_path = acdc_df_path
        self.allPosData = allPosData
        self.doNotWarn = False

        self.setWindowTitle("Set measurements")
        # self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()

        searchLayout = QHBoxLayout()

        searchLineEdit = widgets.SearchLineEdit()
        searchLayout.addStretch(5)
        searchLayout.addWidget(searchLineEdit)
        searchLayout.setStretch(1, 3)

        mainScrollArea = widgets.ScrollArea()
        mainScrollAreaWidget = QWidget()
        mainScrollArea.setWidget(mainScrollAreaWidget)

        groupsLayout = QGridLayout()
        self.groupsLayout = groupsLayout

        mainScrollAreaWidget.setLayout(groupsLayout)

        buttonsLayout = QHBoxLayout()

        self.chNameGroupboxes = []
        self.all_metrics = []

        col = 0
        for col, chName in enumerate(loadedChNames):
            channelGBox = widgets.channelMetricsQGBox(
                isZstack,
                chName,
                isSegm3D,
                favourite_funcs=favourite_funcs,
                posData=posData,
                is_concat=is_concat,
            )
            channelGBox.chName = chName
            groupsLayout.addWidget(channelGBox, 0, col, 3, 1)
            self.chNameGroupboxes.append(channelGBox)
            channelGBox.sigDelClicked.connect(self.delMixedChannelCombineMetric)
            channelGBox.sigCheckboxToggled.connect(self.channelCheckboxToggled)
            groupsLayout.setColumnStretch(col, 5)
            self.all_metrics.extend([c.text() for c in channelGBox.checkBoxes])

        current_col = col + 1
        for col, chName in enumerate(notLoadedChNames):
            channelGBox = widgets.channelMetricsQGBox(
                isZstack,
                chName,
                isSegm3D,
                favourite_funcs=favourite_funcs,
                posData=posData,
                is_concat=is_concat,
            )
            channelGBox.setChecked(False)
            channelGBox.chName = chName
            groupsLayout.addWidget(channelGBox, 0, current_col, 3, 1)
            self.chNameGroupboxes.append(channelGBox)
            groupsLayout.setColumnStretch(current_col, 5)
            channelGBox.sigDelClicked.connect(self.delMixedChannelCombineMetric)
            channelGBox.sigCheckboxToggled.connect(self.channelCheckboxToggled)
            current_col += 1
            self.all_metrics.extend([c.text() for c in channelGBox.checkBoxes])

        current_col += 1

        if posData is None:
            isTimelapse = False
        else:
            isTimelapse = posData.SizeT > 1
        size_metrics_desc = measurements.get_size_metrics_desc(isSegm3D, isTimelapse)
        if not isSegm3D:
            size_metrics_desc = {
                key: val
                for key, val in size_metrics_desc.items()
                if not key.endswith("_3D")
            }

        row = 0
        sizeMetricsQGBox = widgets._metricsQGBox(
            size_metrics_desc,
            "Physical measurements",
            favourite_funcs=favourite_funcs,
            isZstack=isZstack,
            addCalcForEachZsliceToggle=isSegm3D,
        )
        self.all_metrics.extend([c.text() for c in sizeMetricsQGBox.checkBoxes])
        self.sizeMetricsQGBox = sizeMetricsQGBox
        for sizeCheckbox in sizeMetricsQGBox.checkBoxes:
            sizeCheckbox.toggled.connect(self.sizeMetricToggled)
        groupsLayout.addWidget(sizeMetricsQGBox, row, current_col)
        groupsLayout.setRowStretch(0, 1)
        groupsLayout.setColumnStretch(current_col, 3)
        row += 1

        props_info_txt_mapper = measurements.get_props_info_txt_mapper(
            isSegm3D=isSegm3D
        )
        rp_desc = props_info_txt_mapper
        regionPropsQGBox = widgets._metricsQGBox(
            rp_desc,
            "Morphological properties",
            favourite_funcs=favourite_funcs,
            isZstack=isZstack,
        )
        self.regionPropsQGBox = regionPropsQGBox
        for rpCheckbox in regionPropsQGBox.checkBoxes:
            rpCheckbox.toggled.connect(self.rpMetricToggled)
        groupsLayout.addWidget(regionPropsQGBox, row, current_col)
        groupsLayout.setRowStretch(1, 2)
        self.all_metrics.extend([c.text() for c in regionPropsQGBox.checkBoxes])
        row += 1

        # Custom metrics that are channel indipendent
        self.chIndipendCustomeMetricsQGBox = None
        out = measurements.ch_indipend_custom_metrics_desc(
            isZstack,
            isSegm3D=isSegm3D,
        )
        ch_indipend_custom_metrics_desc = out
        if ch_indipend_custom_metrics_desc:
            self.chIndipendCustomeMetricsQGBox = widgets._metricsQGBox(
                ch_indipend_custom_metrics_desc,
                "Channel indipendent custom measurements",
                favourite_funcs=favourite_funcs,
                isZstack=isZstack,
                parent=self,
            )
            groupsLayout.addWidget(self.chIndipendCustomeMetricsQGBox, row, current_col)
            groupsLayout.setRowStretch(1, 1)
            row += 1

        desc, equations = measurements.combine_mixed_channels_desc(
            isSegm3D=isSegm3D, posData=posData, available_cols=self.all_metrics
        )
        self.mixedChannelsCombineMetricsQGBox = None
        if desc:
            self.mixedChannelsCombineMetricsQGBox = widgets._metricsQGBox(
                desc,
                "Mixed channels combined measurements",
                favourite_funcs=favourite_funcs,
                isZstack=isZstack,
                equations=equations,
                addDelButton=True,
            )
            self.mixedChannelsCombineMetricsQGBox.sigDelClicked.connect(
                self.delMixedChannelCombineMetric
            )
            groupsLayout.addWidget(
                self.mixedChannelsCombineMetricsQGBox, row, current_col
            )
            groupsLayout.setRowStretch(1, 1)
            if not self.is_concat:
                self.setDisabledMetricsRequestedForCombined(False)
                self.mixedChannelsCombineMetricsQGBox.toggled.connect(
                    self.setDisabledMetricsRequestedForCombined
                )
                for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                    combCheckbox.toggled.connect(
                        self.setDisabledMetricsRequestedForCombined
                    )
            else:
                for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                    combCheckbox.toggled.connect(self.mixedChannelsMetricToggled)
            row += 1

        self.last_row = row
        self.last_col = current_col

        okButton = widgets.okPushButton("   Ok   ")
        cancelButton = widgets.cancelPushButton("Cancel")
        if addCombineMetricCallback is not None:
            addCombineMetricButton = widgets.addPushButton(
                "Add combined measurement..."
            )
            addCombineMetricButton.clicked.connect(addCombineMetricCallback)
        self.okButton = okButton

        loadLastSelButton = widgets.reloadPushButton("Load last selection...")
        self.deselectAllButton = QPushButton("Deselect all")
        self.deselectAllButton.setIcon(QIcon(":deselect_all.svg"))

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(self.deselectAllButton)
        buttonsLayout.addSpacing(20)

        if addCombineMetricCallback is not None:
            buttonsLayout.addWidget(addCombineMetricButton)
            buttonsLayout.addSpacing(20)

        saveCurrentSelectionButton = widgets.savePushButton("Save current selection...")
        saveCurrentSelectionButton.clicked.connect(self.saveCurrentSelectionClicked)

        buttonsLayout.addWidget(saveCurrentSelectionButton)

        loadSavedSelectionButton = widgets.OpenFilePushButton("Load saved selection...")
        loadSavedSelectionButton.clicked.connect(self.loadSavedSelectionClicked)
        buttonsLayout.addWidget(loadSavedSelectionButton)

        buttonsLayout.addWidget(loadLastSelButton)

        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        self.okButton = okButton

        layout.addLayout(searchLayout)
        layout.addSpacing(10)
        # layout.addLayout(groupsLayout)
        layout.addWidget(mainScrollArea)
        layout.addLayout(buttonsLayout)

        self.setLayout(layout)

        if state is not None:
            self.setState(state)

        searchLineEdit.textEdited.connect(self.searchAndHighlight)
        self.deselectAllButton.clicked.connect(self.deselectAll)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        loadLastSelButton.clicked.connect(self.loadLastSelection)

        self.addCheckboxesToGroup()

        for channelGBox in self.chNameGroupboxes:
            for checkbox in channelGBox.checkBoxes:
                self.channelCheckboxToggled(checkbox)

    def allMetricsDict(self):
        all_metrics = {
            "standard": {},
            "regionprop": [],
            "size": [],
            "mixed_channels": [],
        }
        for chNameGroupbox in self.chNameGroupboxes:
            channel_name = chNameGroupbox.chName
            for checkBox in chNameGroupbox.checkBoxes:
                if channel_name not in all_metrics["standard"]:
                    all_metrics["standard"][channel_name] = []
                all_metrics["standard"][channel_name].append(checkBox.text())

        for checkBox in self.regionPropsQGBox.checkBoxes:
            all_metrics["regionprop"].append(checkBox.text())

        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            all_metrics["size"].append(checkBox.text())

        if self.chIndipendCustomeMetricsQGBox is not None:
            checkBoxes = self.chIndipendCustomeMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                all_metrics["ch_indipend_custom_metric"].append(checkBox.text())

        if self.mixedChannelsCombineMetricsQGBox is None:
            return

        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            all_metrics["mixed_channels"].append(checkBox.text())

        return all_metrics

    def searchAndHighlight(self, text):
        for chNameGroupbox in self.chNameGroupboxes:
            for groupbox in chNameGroupbox.groupboxes:
                groupbox.highlightCheckboxesFromSearchText(text)

        self.regionPropsQGBox.highlightCheckboxesFromSearchText(text)
        self.sizeMetricsQGBox.highlightCheckboxesFromSearchText(text)

        if self.chIndipendCustomeMetricsQGBox is not None:
            self.chIndipendCustomeMetricsQGBox.highlightCheckboxesFromSearchText(text)

        if self.mixedChannelsCombineMetricsQGBox is None:
            return

        self.mixedChannelsCombineMetricsQGBox.highlightCheckboxesFromSearchText(text)

    def selectedMetricNameAndGroup(self):
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                if checkBox.isChecked():
                    return checkBox.text(), {"standard": chNameGroupbox.chName}

        for checkBox in self.regionPropsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text(), "regionprop"

        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text(), "size"

        if self.chIndipendCustomeMetricsQGBox is not None:
            checkBoxes = self.chIndipendCustomeMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                if checkBox.isChecked():
                    return checkBox.text(), "ch_indipend_custom_metric"

        if self.mixedChannelsCombineMetricsQGBox is None:
            return

        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            if checkBox.isChecked():
                return checkBox.text(), "mixed_channels"

    def selectedMetricGroup(self):
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                if checkBox.isChecked():
                    return checkBox.text()

        for checkBox in self.regionPropsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text()

        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text()

        if self.chIndipendCustomeMetricsQGBox is not None:
            checkBoxes = self.chIndipendCustomeMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                if checkBox.isChecked():
                    return checkBox.text()

        if self.mixedChannelsCombineMetricsQGBox is None:
            return

        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            if checkBox.isChecked():
                return checkBox.text()

    def addCheckboxesToGroup(self):
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                self.checkBoxedGroup.addButton(checkBox)

        for checkBox in self.regionPropsQGBox.checkBoxes:
            self.checkBoxedGroup.addButton(checkBox)

        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            self.checkBoxedGroup.addButton(checkBox)

        if self.chIndipendCustomeMetricsQGBox is not None:
            checkBoxes = self.chIndipendCustomeMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                self.checkBoxedGroup.addButton(checkBox)

        if self.mixedChannelsCombineMetricsQGBox is None:
            return

        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            self.checkBoxedGroup.addButton(checkBox)

    def channelCheckboxToggled(self, checkbox):
        # Make sure to automatically check the requested cell_vol metric for
        # concentration metrics
        if checkbox.text().find("concentration_") == -1:
            return

        if self.is_concat:
            # When this dialogue is used in concatenate pos utility we do not
            # need to check that certain metrics are present
            return

        pattern = r".+_from_vol_([a-z]+)(_3D)?(_?[A-Za-z0-9]*)"
        repl = r"cell_vol_\1\2"
        cell_vol_metric_name = re.sub(pattern, repl, checkbox.text())
        for sizeCheckbox in self.sizeMetricsQGBox.checkBoxes:
            if sizeCheckbox.text() == cell_vol_metric_name:
                break
        else:
            # Make sure to not check for similarly named custom metrics
            return

        if checkbox.isChecked():
            sizeCheckbox.setChecked(True)
            sizeCheckbox.isRequired = True
        else:
            # Do not enable cell vol checkbox is any of the other
            # concentration metrics requiring it is checked
            unit = cell_vol_metric_name[9:]
            is3D = unit.endswith("3D")
            for channelGBox in self.chNameGroupboxes:
                if not channelGBox.isChecked():
                    continue
                for _checkbox in channelGBox.checkBoxes:
                    if _checkbox.text().find(f"_from_vol_{unit}") == -1:
                        continue
                    if not is3D and _checkbox.text().find(f"{unit}_3D") != -1:
                        # Metric is 3D but the cell_vol is not
                        continue
                    if _checkbox.isChecked():
                        return
            sizeCheckbox.isRequired = False

    def rpMetricToggled(self, checked):
        pass

    def mixedChannelsMetricToggled(self, checked):
        pass

    def sizeMetricToggled(self, checked):
        """Method called when a checkbox of a size metric is toggled.
        Check if the size value is required and explain why it cannot be
        unchecked.

        Parameters
        ----------
        checked : bool
            State of the checkbox toggled
        """
        checkbox = self.sender()

        if self.is_concat:
            # When this dialogue is used in concatenate pos utility we do not
            # need to check that certain metrics are present
            return

        if not hasattr(checkbox, "isRequired"):
            return

        if not checkbox.isRequired:
            return

        if checkbox.isChecked():
            return

        checkbox.setChecked(True)

        if self.doNotWarn:
            return

        linked_autoBkgr_metric = checkbox.text().replace("cell", "_autoBkgr_from")
        linked_dataPrepBkgr_metric = checkbox.text().replace(
            "cell", "_dataPrepBkgr_from"
        )
        txt = html_utils.paragraph(f"""
            <b>This physical measurement cannot be unchecked</b> 
            because it is required 
            by the <code>{linked_autoBkgr_metric}</code> and 
            <code>{linked_dataPrepBkgr_metric}</code> measurements 
            that you requested to save.<br><br>

            Thank you for you patience!
        """)
        msg = widgets.myMessageBox(showCentered=False)
        msg.warning(self, "Physical measurement required", txt)

    def deselectAll(self):
        self.doNotWarn = True
        for chNameGroupbox in self.chNameGroupboxes:
            for gb in chNameGroupbox.groupboxes:
                gb.checkAll(None, False)
            cgb = getattr(chNameGroupbox, "customMetricsQGBox", None)
            if cgb is not None:
                cgb.checkAll(None, False)

        self.sizeMetricsQGBox.checkAll(None, False)
        self.regionPropsQGBox.checkAll(None, False)
        if self.chIndipendCustomeMetricsQGBox is not None:
            self.chIndipendCustomeMetricsQGBox.checkAll(None, False)

        if self.mixedChannelsCombineMetricsQGBox is not None:
            self.mixedChannelsCombineMetricsQGBox.checkAll(None, False)
        self.doNotWarn = False

    def delMixedChannelCombineMetric(self, colname_to_del, hlayout):
        cp = measurements.read_saved_user_combine_config()
        for section in cp.sections():
            cp.remove_option(section, colname_to_del)
        measurements.save_common_combine_metrics(cp)

        for i in range(hlayout.count()):
            item = hlayout.itemAt(i)
            w = item.widget()
            if w is None:
                continue
            w.hide()

        if self.allPosData is not None:
            for posData in self.allPosData:
                _config = posData.combineMetricsConfig
                for section in _config.sections():
                    _config.remove_option(section, colname_to_del)
                posData.saveCombineMetrics()

    def setState(self, state):
        self.doNotWarn = True
        for chNameGroupbox in self.chNameGroupboxes:
            measurementsInfo = state.get(chNameGroupbox.title())
            if not measurementsInfo:
                chNameGroupbox.setChecked(False)
            else:
                for checkBox in chNameGroupbox.checkBoxes:
                    colname = checkBox.text()
                    checkBox.setChecked(measurementsInfo[colname])

        measurementsInfo = state.get(self.sizeMetricsQGBox.title())
        if not measurementsInfo:
            self.sizeMetricsQGBox.setChecked(False)
        else:
            for checkBox in self.sizeMetricsQGBox.checkBoxes:
                checked = checkBox.isChecked()
                colname = checkBox.text()
                checkBox.setChecked(measurementsInfo[colname])

        measurementsInfo = state.get(self.regionPropsQGBox.title())
        if not measurementsInfo:
            self.regionPropsQGBox.setChecked(False)
        else:
            self.regionPropsToSave = []
            for checkBox in self.regionPropsQGBox.checkBoxes:
                checked = checkBox.isChecked()
                colname = checkBox.text()
                checkBox.setChecked(measurementsInfo[colname])

        if self.chIndipendCustomeMetricsQGBox is not None:
            measurementsInfo = state.get(self.chIndipendCustomeMetricsQGBox.title())
            if not measurementsInfo:
                self.chIndipendCustomeMetricsQGBox.setChecked(False)
            else:
                checkBoxes = self.chIndipendCustomeMetricsQGBox.checkBoxes
                for checkBox in checkBoxes:
                    checked = checkBox.isChecked()
                    colname = checkBox.text()
                    key = self.chIndipendCustomeMetricsQGBox.title()
                    checkBox.setChecked(measurementsInfo[colname])

        if self.mixedChannelsCombineMetricsQGBox is not None:
            measurementsInfo = state.get(self.mixedChannelsCombineMetricsQGBox.title())
            if not measurementsInfo:
                self.mixedChannelsCombineMetricsQGBox.setChecked(False)
            else:
                checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
                for checkBox in checkBoxes:
                    checked = checkBox.isChecked()
                    colname = checkBox.text()
                    key = self.mixedChannelsCombineMetricsQGBox.title()
                    checkBox.setChecked(measurementsInfo[colname])

        self.doNotWarn = False

    def state(self):
        state = {self.sizeMetricsQGBox.title(): {}, self.regionPropsQGBox.title(): {}}
        for chNameGroupbox in self.chNameGroupboxes:
            state[chNameGroupbox.title()] = {}
            if not chNameGroupbox.isChecked():
                # Channel unchecked
                continue
            else:
                for checkBox in chNameGroupbox.checkBoxes:
                    colname = checkBox.text()
                    state[chNameGroupbox.title()][colname] = checkBox.isChecked()

        if not self.sizeMetricsQGBox.isChecked():
            pass
        else:
            for checkBox in self.sizeMetricsQGBox.checkBoxes:
                checked = checkBox.isChecked()
                colname = checkBox.text()
                state[self.sizeMetricsQGBox.title()][colname] = checked

        if not self.regionPropsQGBox.isChecked():
            pass
        else:
            self.regionPropsToSave = []
            for checkBox in self.regionPropsQGBox.checkBoxes:
                checked = checkBox.isChecked()
                colname = checkBox.text()
                state[self.regionPropsQGBox.title()][colname] = checked

        if self.chIndipendCustomeMetricsQGBox is not None:
            state[self.chIndipendCustomeMetricsQGBox.title()] = {}
            if self.chIndipendCustomeMetricsQGBox.isChecked():
                checkBoxes = self.chIndipendCustomeMetricsQGBox.checkBoxes
                for checkBox in checkBoxes:
                    checked = checkBox.isChecked()
                    key = self.chIndipendCustomeMetricsQGBox.title()
                    colname = checkBox.text()
                    state[key][colname] = checked

        if self.mixedChannelsCombineMetricsQGBox is not None:
            state[self.mixedChannelsCombineMetricsQGBox.title()] = {}
            if self.mixedChannelsCombineMetricsQGBox.isChecked():
                checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
                for checkBox in checkBoxes:
                    checked = checkBox.isChecked()
                    key = self.mixedChannelsCombineMetricsQGBox.title()
                    colname = checkBox.text()
                    state[key][colname] = checked

        return state

    def restoreState(self, state):
        for chNameGroupbox in self.chNameGroupboxes:
            _state = state.get(chNameGroupbox.title())
            if _state is None or not _state:
                continue
            for checkBox in chNameGroupbox.checkBoxes:
                isChecked = _state.get(checkBox.text())
                if isChecked is None:
                    continue
                checkBox.setChecked(isChecked)

        _state = state.get(self.sizeMetricsQGBox.title())
        if _state is None or not _state:
            pass
        else:
            for checkBox in self.sizeMetricsQGBox.checkBoxes:
                isChecked = _state.get(checkBox.text())
                if isChecked is None:
                    continue
                checkBox.setChecked(isChecked)

        _state = state.get(self.regionPropsQGBox.title())
        if _state is None or not _state:
            pass
        else:
            for checkBox in self.regionPropsQGBox.checkBoxes:
                isChecked = _state.get(checkBox.text())
                if isChecked is None:
                    continue
                checkBox.setChecked(isChecked)

        if self.chIndipendCustomeMetricsQGBox is not None:
            _state = state.get(self.chIndipendCustomeMetricsQGBox.title())
            if _state is None or not _state:
                pass
            else:
                for checkBox in self.chIndipendCustomeMetricsQGBox.checkBoxes:
                    isChecked = _state.get(checkBox.text())
                    if isChecked is None:
                        continue
                    checkBox.setChecked(isChecked)

        if self.mixedChannelsCombineMetricsQGBox is not None:
            _state = state.get(self.mixedChannelsCombineMetricsQGBox.title())
            if _state is None or not _state:
                pass
            else:
                for checkBox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                    isChecked = _state.get(checkBox.text())
                    if isChecked is None:
                        continue
                    checkBox.setChecked(isChecked)

    def currentSelectionMapper(self):
        current_selected_meas = defaultdict(dict)

        for chNameGroupbox in self.chNameGroupboxes:
            if not chNameGroupbox.isChecked():
                continue

            chName = chNameGroupbox.chName
            for checkBox in chNameGroupbox.checkBoxes:
                if not checkBox.isChecked():
                    continue

                current_selected_meas[chName][checkBox.text()] = "Yes"

        size_selected_meas = current_selected_meas.get(self.sizeMetricsQGBox.title())
        if self.sizeMetricsQGBox.isChecked():
            for checkBox in self.sizeMetricsQGBox.checkBoxes:
                if not checkBox.isChecked():
                    continue

                section = self.sizeMetricsQGBox.title()
                current_selected_meas[section][checkBox.text()] = "Yes"

        size_selected_meas = current_selected_meas.get(self.regionPropsQGBox.title())
        if self.regionPropsQGBox.isChecked():
            for checkBox in self.regionPropsQGBox.checkBoxes:
                if not checkBox.isChecked():
                    continue

                section = self.regionPropsQGBox.title()
                current_selected_meas[section][checkBox.text()] = "Yes"

        if self.chIndipendCustomeMetricsQGBox is not None:
            if self.chIndipendCustomeMetricsQGBox.isChecked():
                for checkBox in self.chIndipendCustomeMetricsQGBox.checkBoxes:
                    if not checkBox.isChecked():
                        continue

                    section = self.chIndipendCustomeMetricsQGBox.title()
                    current_selected_meas[section][checkBox.text()] = "Yes"

        if self.mixedChannelsCombineMetricsQGBox is not None:
            if self.mixedChannelsCombineMetricsQGBox.isChecked():
                for checkBox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                    if not checkBox.isChecked():
                        continue

                    section = self.mixedChannelsCombineMetricsQGBox.title()
                    current_selected_meas[section][checkBox.text()] = "Yes"

        return current_selected_meas

    def saveCurrentSelectionClicked(self):
        current_selection_mapper = self.currentSelectionMapper()
        defaultEntry = "_and_".join(current_selection_mapper.keys())
        defaultEntry = defaultEntry.replace(" ", "_").lower()
        saved_selections = io.get_saved_measurements_selections()
        win = filenameDialog(
            basename="",
            ext="",
            hintText="Insert a <b>name</b> for the current selection:",
            existingNames=saved_selections,
            allowEmpty=False,
            defaultEntry=defaultEntry,
        )
        win.exec_()
        if win.cancel:
            return

        filename = win.filename
        ini_filepath = io.save_measurements_selections(
            filename, current_selection_mapper
        )

        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph(f"""
            Done!<br><br>
            Current selection saved with name <code>{filename}</code> at 
            the following path:
        """)
        msg.information(
            self,
            "Selection saved",
            txt,
            commands=(ini_filepath,),
            path_to_browse=os.path.dirname(ini_filepath),
        )

    def loadSavedSelectionClicked(self):
        self.doNotWarn = True

        saved_selections = io.get_saved_measurements_selections()

        selectNameWin = widgets.QDialogListbox(
            "Choose selection to load",
            "Choose selection to load:\n",
            saved_selections,
            multiSelection=False,
            parent=self,
        )
        selectNameWin.exec_()
        if selectNameWin.cancel:
            return

        selection_mapper = io.read_measurements_selections(
            selectNameWin.selectedItemsText[0]
        )

        self.setCurrentSelectionFromMapper(selection_mapper)

        self.doNotWarn = False

    def saveLastSelection(self):
        last_selected_meas = self.currentSelectionMapper()
        load.write_last_selected_set_measurements(last_selected_meas)

    def setCurrentSelectionFromMapper(self, selection_mapper):
        for chNameGroupbox in self.chNameGroupboxes:
            chName = chNameGroupbox.chName
            chSelectedMeas = selection_mapper.get(chName)
            if chSelectedMeas is None:
                chNameGroupbox.setChecked(False)
                continue

            chNameGroupbox.setChecked(True)
            for checkBox in chNameGroupbox.checkBoxes:
                checked = chSelectedMeas.get(checkBox.text())
                if checked is not None:
                    checkBox.setChecked(True)
                else:
                    checkBox.setChecked(False)

        size_selected_meas = selection_mapper.get(self.sizeMetricsQGBox.title())
        if size_selected_meas is None:
            self.sizeMetricsQGBox.setChecked(False)
        else:
            self.sizeMetricsQGBox.setChecked(True)
            for checkBox in self.sizeMetricsQGBox.checkBoxes:
                checked = size_selected_meas.get(checkBox.text())
                if checked is not None:
                    checkBox.setChecked(True)
                else:
                    checkBox.setChecked(False)

        size_selected_meas = selection_mapper.get(self.regionPropsQGBox.title())
        if size_selected_meas is None:
            self.regionPropsQGBox.setChecked(False)
        else:
            self.regionPropsQGBox.setChecked(True)
            for checkBox in self.regionPropsQGBox.checkBoxes:
                checked = size_selected_meas.get(checkBox.text())
                if checked is not None:
                    checkBox.setChecked(True)
                else:
                    checkBox.setChecked(False)

        if self.chIndipendCustomeMetricsQGBox is not None:
            ch_indip_custom_metrics = selection_mapper.get(
                self.chIndipendCustomeMetricsQGBox.title()
            )
            if size_selected_meas is None:
                self.chIndipendCustomeMetricsQGBox.setChecked(False)
            else:
                self.chIndipendCustomeMetricsQGBox.setChecked(True)
                for checkBox in self.chIndipendCustomeMetricsQGBox.checkBoxes:
                    checked = size_selected_meas.get(checkBox.text())
                    if checked is not None:
                        checkBox.setChecked(True)
                    else:
                        checkBox.setChecked(False)

        if self.mixedChannelsCombineMetricsQGBox is not None:
            ch_indip_custom_metrics = selection_mapper.get(
                self.mixedChannelsCombineMetricsQGBox.title()
            )
            if size_selected_meas is None:
                self.mixedChannelsCombineMetricsQGBox.setChecked(False)
            else:
                self.mixedChannelsCombineMetricsQGBox.setChecked(True)
                for checkBox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                    checked = size_selected_meas.get(checkBox.text())
                    if checked is not None:
                        checkBox.setChecked(True)
                    else:
                        checkBox.setChecked(False)

    def loadLastSelection(self):
        self.doNotWarn = True
        last_selected_meas = load.read_last_selected_set_measurements()
        last_selected_meas = dict(last_selected_meas)

        self.setCurrentSelectionFromMapper(last_selected_meas)

        self.doNotWarn = False

    def setDisabledMetricsRequestedForCombined(self, checked):
        checkbox = self.sender()

        if self.is_concat:
            # When this dialogue is used in concatenate pos utility we do not
            # need to check that certain metrics are present
            return

        # Set checked and disable those metrics that are requested for
        # combined measurements
        allCheckboxes = []

        for chNameGroupbox in self.chNameGroupboxes:
            for chCheckBox in chNameGroupbox.checkBoxes:
                chCheckBox.setDisabled(False)
                allCheckboxes.append(chCheckBox)

        for sizeCheckBox in self.sizeMetricsQGBox.checkBoxes:
            sizeCheckBox.setDisabled(False)
            allCheckboxes.append(chCheckBox)

        for rpCheckBox in self.regionPropsQGBox.checkBoxes:
            rpCheckBox.setDisabled(False)
            allCheckboxes.append(chCheckBox)

        if not self.mixedChannelsCombineMetricsQGBox.isChecked():
            return

        for cb in allCheckboxes:
            metricName = cb.text()
            for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                equation = combCheckbox.equation
                if equation.find(metricName) == -1:
                    continue
                elif combCheckbox.isChecked():
                    cb.setChecked(True)
                    cb.setDisabled(True)
                    cb.setToolTip(
                        "This metric cannot be removed because it is required "
                        f'by the combined measurement "{combCheckbox.text()}"'
                    )

    def keyPressEvent(self, a0: QKeyEvent) -> None:
        state = self.state()
        return super().keyPressEvent(a0)

    def closeEvent(self, event):
        if self.cancel:
            self.sigCancel.emit()
        super().closeEvent(event)

    def restart(self):
        self.cancel = False
        self.close()
        self.sigRestart.emit()

    def setDisabledNotExistingMeasurements(self, existing_colnames):
        self.existing_colnames = existing_colnames
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                colname = checkBox.text()
                if colname in existing_colnames:
                    checkBox.setChecked(True)
                    continue

                checkBox.setChecked(False)
                checkBox.setDisabled(True)
                self.setNotExistingMeasurementTooltip(checkBox)

        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            colname = checkBox.text()
            if colname in existing_colnames:
                checkBox.setChecked(True)
                continue
            checkBox.setChecked(False)
            checkBox.setDisabled(True)
            self.setNotExistingMeasurementTooltip(checkBox)

        for checkBox in self.regionPropsQGBox.checkBoxes:
            prop_name = checkBox.text()
            for existing_col in existing_colnames:
                if prop_name == existing_col:
                    checkBox.setChecked(True)
                    break
                m = re.match(rf"{prop_name}-\d", existing_col)
                if m is not None:
                    checkBox.setChecked(True)
                    break
            else:
                checkBox.setChecked(False)
                checkBox.setDisabled(True)
                self.setNotExistingMeasurementTooltip(checkBox)

        if self.mixedChannelsCombineMetricsQGBox is None:
            return

        for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
            colname = combCheckbox.text()
            if colname in existing_colnames:
                combCheckbox.setChecked(True)
                continue
            combCheckbox.setChecked(False)
            combCheckbox.setDisabled(True)
            self.setNotExistingMeasurementTooltip(combCheckbox)

    def addNonMeasurementColumns(self, colnames):
        additionalCols = measurements.get_non_measurements_cols(
            colnames, self.all_metrics
        )
        if not additionalCols:
            return
        self.nonMeasurementsGroupbox = widgets.CheckboxesGroupBox(
            additionalCols, title="Additional columns", checkable=True
        )
        self.groupsLayout.addWidget(
            self.nonMeasurementsGroupbox, 0, self.last_col + 1, self.last_row + 1, 1
        )

    def setNotExistingMeasurementTooltip(self, checkBox):
        checkBox.setToolTip(
            "Measurement is disabled because it is not present in selected "
            "acdc_output tables, hence it cannot be addded to concatenated "
            "table. "
        )

    def ok_cb(self):
        for chNameGroupbox in self.chNameGroupboxes:
            chNameGroupbox.calcForEachZsliceRequested = (
                chNameGroupbox.isCalcForEachZsliceRequested()
            )

        self.sizeMetricsQGBox.calcForEachZsliceRequested = (
            self.sizeMetricsQGBox.isCalcForEachZsliceRequested()
        )

        if self.allPos_acdc_df_cols is None:
            self.saveLastSelection()
            self.cancel = False
            self.close()
            self.sigClosed.emit()
            return

        self.okClicked = True
        existing_colnames = self.allPos_acdc_df_cols
        unchecked_existing_colnames = []
        unchecked_existing_rps = []
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                colname = checkBox.text()
                is_existing = colname in existing_colnames
                if not chNameGroupbox.isChecked() and is_existing:
                    unchecked_existing_colnames.append(colname)
                    continue
                if not checkBox.isChecked() and is_existing:
                    unchecked_existing_colnames.append(colname)

        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            colname = checkBox.text()
            is_existing = colname in existing_colnames
            if not self.sizeMetricsQGBox.isChecked() and is_existing:
                unchecked_existing_colnames.append(colname)
                continue

            if not checkBox.isChecked() and is_existing:
                unchecked_existing_colnames.append(colname)
        for checkBox in self.regionPropsQGBox.checkBoxes:
            colname = checkBox.text()
            is_existing = any([col == colname for col in existing_colnames])
            if not self.regionPropsQGBox.isChecked() and is_existing:
                unchecked_existing_rps.append(colname)
                continue

            if not checkBox.isChecked() and is_existing:
                unchecked_existing_rps.append(colname)

        if unchecked_existing_colnames or unchecked_existing_rps:
            cancel, self.delExistingCols = self.warnUncheckedExistingMeasurements(
                unchecked_existing_colnames, unchecked_existing_rps
            )
            self.existingUncheckedColnames = unchecked_existing_colnames
            self.existingUncheckedRps = unchecked_existing_rps
            if cancel:
                return

        self.saveLastSelection()
        self.cancel = False
        self.close()
        self.sigClosed.emit()

    def warnUncheckedExistingMeasurements(
        self, unchecked_existing_colnames, unchecked_existing_rps
    ):
        msg = widgets.myMessageBox()
        msg.setWidth(500)
        msg.addShowInFileManagerButton(self.acdc_df_path)
        txt = html_utils.paragraph(
            "You chose to <b>not save</b> some measurements that are "
            "<b>already present</b> in the saved <code>acdc_output.csv</code> "
            "file.<br><br>"
            "Do you want to <b>delete</b> these measurements or "
            "<b>keep</b> them?<br><br>"
            "Existing measurements not selected:"
        )
        listView = widgets.readOnlyQList(msg)
        items = unchecked_existing_colnames.copy()
        items.extend(unchecked_existing_rps)
        listView.addItems(items)
        _, delButton, keepButton = msg.warning(
            self,
            "Unchecked existing measurements",
            txt,
            widgets=listView,
            buttonsTexts=("Cancel", "Delete", "Keep"),
        )
        return msg.cancel, msg.clickedButton == delButton

    def show(self, block=False):
        super().show(block=False)
        self.deselectAllButton.setMinimumHeight(self.okButton.height())
        screenWidth = self.screen().size().width()
        screenHeight = self.screen().size().height()
        screenLeft = self.screen().geometry().x()
        screenTop = self.screen().geometry().y()
        h = screenHeight - 200
        minColWith = screenWidth / 5
        w = minColWith * (self.last_col + 1)
        xLeft = int((screenWidth - w) / 2)
        if w > screenWidth:
            self.move(screenLeft + 10, screenTop + 50)
            self.resize(screenWidth - 20, h)
        else:
            self.move(screenLeft + xLeft, screenTop + 50)
            self.resize(int(w), h)
        super().show(block=block)


class ComputeMetricsErrorsDialog(QBaseDialog):
    def __init__(self, errorsDict, log_path="", parent=None, log_type="custom_metrics"):
        super().__init__(parent)

        self.errorsDict = errorsDict

        layout = QGridLayout()

        self.setWindowTitle("Errors summary")

        label = QLabel(self)
        standardIcon = getattr(QStyle, "SP_MessageBoxWarning")
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)
        layout.addWidget(label, 0, 0, alignment=Qt.AlignTop)

        if log_type == "custom_metrics":
            infoText = """
                When computing <b>custom metrics</b> the following metrics 
                were <b>ignored</b> because they raised an <b>error</b>.<br><br>
            """
        elif log_type == "standard_metrics":
            infoText = """
                Some or all of the <b>standard metrics</b> were <b>NOT saved</b> 
                because Cell-ACDC encoutered the following errors.<br><br>
            """
        elif log_type == "region_props":
            rp_url = "https://scikit-image.org/docs/0.18.x/api/skimage.measure.html#skimage.measure.regionprops"
            rp_href = f'<a href="{rp_url}">skimage.measure.regionprops</a>'
            infoText = f"""
                <b>Region properties</b> were <b>NOT saved</b> because Cell-ACDC 
                encoutered the following errors.<br>
                Region properties are calculated using the <code>scikit-image</code> 
                function called <code>{rp_href}</code>.<br><br>
            """
        elif log_type == "missing_annot":
            infoText = """
                The following Positions were <b>SKIPPED</b> because they did 
                <b>not have cell cycle annotations</b>.<br><br>
                To add lineage tree information you first need to <b>do the 
                cell cycle analysis</b> in module 3 "Main GUI".<br><br>
            """
        else:
            infoText = """
                Process raised the errors listed below.<br><br>
            """

        github_issues_href = f"<a href={issues_url}>here</a>"
        noteText = f"""
            NOTE: If you <b>need help</b> understanding these errors you can 
            <b>open an issue</b> on our github page {github_issues_href}.
        """

        infoLabel = QLabel(html_utils.paragraph(f"{infoText}{noteText}"))
        infoLabel.setOpenExternalLinks(True)
        layout.addWidget(infoLabel, 0, 1)

        scrollArea = QScrollArea()
        scrollAreaWidget = QWidget()
        textLayout = QVBoxLayout()
        for func_name, traceback_format in errorsDict.items():
            nameLabel = QLabel(f"<b>{func_name}</b>: ")
            errorMessage = f"\n{traceback_format}"
            errorLabel = QLabel(errorMessage)
            errorLabel.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            )
            # errorLabel.setStyleSheet("background-color: white")
            errorLabel.setFrameShape(QFrame.Shape.Panel)
            errorLabel.setFrameShadow(QFrame.Shadow.Sunken)
            textLayout.addWidget(nameLabel)
            textLayout.addWidget(errorLabel)
            textLayout.addStretch(1)

        scrollAreaWidget.setLayout(textLayout)
        scrollArea.setWidget(scrollAreaWidget)

        layout.addWidget(scrollArea, 1, 1)

        buttonsLayout = QHBoxLayout()
        showLogButton = widgets.showInFileManagerButton("Show log file...")
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(showLogButton)

        copyButton = widgets.copyPushButton("Copy error message")
        copyButton.clicked.connect(self.copyErrorMessage)
        buttonsLayout.addWidget(copyButton)
        self.copyButton = copyButton
        self.copyButton.text = "Copy error message"
        self.copyButton.icon = self.copyButton.icon()

        okButton = widgets.okPushButton(" Ok ")
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        showLogButton.clicked.connect(partial(myutils.showInExplorer, log_path))
        okButton.clicked.connect(self.close)
        layout.setVerticalSpacing(10)
        layout.addLayout(buttonsLayout, 2, 1)

        self.setLayout(layout)
        self.setFont(font)

    def copyErrorMessage(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        copiedText = ""
        for _, traceback_format in self.errorsDict.items():
            errorBlock = f"{'=' * 30}\n{traceback_format}{'*' * 30}"
            copiedText = f"{copiedText}{errorBlock}"
        cb.setText(copiedText, mode=cb.Clipboard)
        print("Error message copied.")
        self.copyButton.setIcon(QIcon(":okButton.svg"))
        self.copyButton.setText(" Copied to clipboard!")
        QTimer.singleShot(2000, self.restoreCopyButton)

    def restoreCopyButton(self):
        self.copyButton.setText(self.copyButton.text)
        self.copyButton.setIcon(self.copyButton.icon)

    def showEvent(self, a0) -> None:
        self.copyButton.setFixedWidth(self.copyButton.width())
        return super().showEvent(a0)


class combineMetricsEquationDialog(QBaseDialog):
    sigOk = Signal(object)

    def __init__(
        self, allChNames, isZstack, isSegm3D, parent=None, debug=False, closeOnOk=True
    ):
        super().__init__(parent)

        self.setWindowTitle("Add combined measurement")

        self.initAttributes()

        self.allChNames = allChNames

        self.cancel = True
        self.isOperatorMode = False
        self.closeOnOk = closeOnOk

        mainLayout = QVBoxLayout()
        equationLayout = QHBoxLayout()

        metricsTreeWidget = QTreeWidget()
        metricsTreeWidget.setHeaderHidden(True)
        metricsTreeWidget.setFont(font)
        self.metricsTreeWidget = metricsTreeWidget

        for chName in allChNames:
            channelTreeItem = QTreeWidgetItem(metricsTreeWidget)
            channelTreeItem.setText(0, f"{chName} measurements")
            metricsTreeWidget.addTopLevelItem(channelTreeItem)

            metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
                isZstack, chName, isSegm3D=isSegm3D
            )
            custom_metrics_desc = measurements.custom_metrics_desc(
                isZstack, chName, isSegm3D=isSegm3D
            )

            foregrMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
            foregrMetricsTreeItem.setText(0, "Cell signal measurements")
            channelTreeItem.addChild(foregrMetricsTreeItem)

            bkgrMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
            bkgrMetricsTreeItem.setText(0, "Background values")
            channelTreeItem.addChild(bkgrMetricsTreeItem)

            if custom_metrics_desc:
                customMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
                customMetricsTreeItem.setText(0, "Custom measurements")
                channelTreeItem.addChild(customMetricsTreeItem)

            self.addTreeItems(foregrMetricsTreeItem, metrics_desc.keys(), isCol=True)
            self.addTreeItems(bkgrMetricsTreeItem, bkgr_val_desc.keys(), isCol=True)

            if custom_metrics_desc:
                self.addTreeItems(
                    customMetricsTreeItem, custom_metrics_desc.keys(), isCol=True
                )

        self.addChannelLessItems(isZstack, isSegm3D=isSegm3D)

        sizeMetricsTreeItem = QTreeWidgetItem(metricsTreeWidget)
        sizeMetricsTreeItem.setText(0, "Size measurements")
        metricsTreeWidget.addTopLevelItem(sizeMetricsTreeItem)

        size_metrics_desc = measurements.get_size_metrics_desc(isSegm3D, True)
        self.addTreeItems(sizeMetricsTreeItem, size_metrics_desc.keys(), isCol=True)

        propMetricsTreeItem = QTreeWidgetItem(metricsTreeWidget)
        propMetricsTreeItem.setText(0, "Region properties")
        metricsTreeWidget.addTopLevelItem(propMetricsTreeItem)

        props_names = measurements.get_props_names()
        self.addTreeItems(propMetricsTreeItem, props_names, isCol=True)

        operatorsLayout = QHBoxLayout()
        operatorsLayout.addStretch(1)

        iconSize = 24

        self.operatorButtons = []
        self.operators = [
            ("add", "+"),
            ("subtract", "-"),
            ("multiply", "*"),
            ("divide", "/"),
            ("open_bracket", "("),
            ("close_bracket", ")"),
            ("square", "**2"),
            ("pow", "**"),
            ("ln", "log("),
            ("log10", "log10("),
        ]
        operatorFont = QFont()
        operatorFont.setPixelSize(16)
        for name, text in self.operators:
            button = QPushButton()
            button.setIcon(QIcon(f":{name}.svg"))
            button.setIconSize(QSize(iconSize, iconSize))
            button.text = text
            operatorsLayout.addWidget(button)
            self.operatorButtons.append(button)
            button.clicked.connect(self.addOperator)
            # button.setFont(operatorFont)

        clearButton = QPushButton()
        clearButton.setIcon(QIcon(":clear.svg"))
        clearButton.setIconSize(QSize(iconSize, iconSize))
        clearButton.setFont(operatorFont)

        clearEntryButton = QPushButton()
        clearEntryButton.setIcon(QIcon(":backspace.svg"))
        clearEntryButton.setFont(operatorFont)
        clearEntryButton.setIconSize(QSize(iconSize, iconSize))

        operatorsLayout.addWidget(clearButton)
        operatorsLayout.addWidget(clearEntryButton)
        operatorsLayout.addStretch(1)

        newColNameLayout = QVBoxLayout()
        newColNameLineEdit = widgets.alphaNumericLineEdit()
        newColNameLineEdit.setAlignment(Qt.AlignCenter)
        self.newColNameLineEdit = newColNameLineEdit
        newColNameLayout.addStretch(1)
        newColNameLayout.addWidget(QLabel("New measurement name:"))
        newColNameLayout.addWidget(newColNameLineEdit)
        newColNameLayout.addStretch(1)

        equationDisplayLayout = QVBoxLayout()
        equationDisplayLayout.addWidget(QLabel("Equation:"))
        equationDisplay = QPlainTextEdit()
        # equationDisplay.setReadOnly(True)
        self.equationDisplay = equationDisplay
        equationDisplayLayout.addWidget(equationDisplay)
        equationDisplayLayout.setStretch(0, 0)
        equationDisplayLayout.setStretch(1, 1)

        equationLayout.addLayout(newColNameLayout)
        equationLayout.addWidget(QLabel(" = "))
        equationLayout.addLayout(equationDisplayLayout)
        equationLayout.setStretch(0, 1)
        equationLayout.setStretch(1, 0)
        equationLayout.setStretch(2, 2)

        testOutputLayout = QVBoxLayout()
        testOutputLayout.addWidget(QLabel("Result of test with random inputs:"))
        testOutputDisplay = QTextEdit()
        testOutputDisplay.setReadOnly(True)
        self.testOutputDisplay = testOutputDisplay
        testOutputLayout.addWidget(testOutputDisplay)
        testOutputLayout.setStretch(0, 0)
        testOutputLayout.setStretch(1, 1)

        instructions = html_utils.paragraph("""
            <b>Double-click</b> on any of the <b>available measurements</b>
            to add it to the equation.<br><br>
            <i>NOTE: the result will be saved in the <code>acdc_output.csv</code>
            file as a column with the same name<br>
            you enter in "New measurement name"
            field.</i><br>
        """)

        buttonsLayout = QHBoxLayout()

        cancelButton = widgets.cancelPushButton("Cancel")
        helpButton = widgets.infoPushButton("  Help...")
        testButton = widgets.calcPushButton("Test output")
        okButton = widgets.okPushButton(" Ok ")
        okButton.setDisabled(True)
        self.okButton = okButton

        buttonsLayout.addStretch(1)

        if debug:
            debugButton = QPushButton("Debug")
            debugButton.clicked.connect(self._debug)
            buttonsLayout.addWidget(debugButton)

        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(helpButton)
        buttonsLayout.addWidget(testButton)
        buttonsLayout.addWidget(okButton)

        mainLayout.addWidget(QLabel(instructions))
        mainLayout.addWidget(QLabel("Available measurements:"))
        mainLayout.addWidget(metricsTreeWidget)
        mainLayout.addLayout(operatorsLayout)
        mainLayout.addLayout(equationLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addLayout(testOutputLayout)

        clearButton.clicked.connect(self.clearEquation)
        clearEntryButton.clicked.connect(self.clearEntryEquation)
        metricsTreeWidget.itemDoubleClicked.connect(self.addColname)

        helpButton.clicked.connect(self.showHelp)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        testButton.clicked.connect(self.test_cb)

        self.setLayout(mainLayout)
        self.setFont(font)

        self.setStyleSheet(TREEWIDGET_STYLESHEET)

    def addChannelLessItems(self, isZstack, isSegm3D=False):
        allChannelsTreeItem = QTreeWidgetItem(self.metricsTreeWidget)
        allChannelsTreeItem.setText(0, f"All channels measurements")
        metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
            isZstack, "", isSegm3D=isSegm3D
        )
        custom_metrics_desc = measurements.custom_metrics_desc(
            isZstack, "", isSegm3D=isSegm3D
        )

        foregrMetricsTreeItem = QTreeWidgetItem(allChannelsTreeItem)
        foregrMetricsTreeItem.setText(0, "Cell signal measurements")
        allChannelsTreeItem.addChild(foregrMetricsTreeItem)

        bkgrMetricsTreeItem = QTreeWidgetItem(allChannelsTreeItem)
        bkgrMetricsTreeItem.setText(0, "Background values")
        allChannelsTreeItem.addChild(bkgrMetricsTreeItem)

        if custom_metrics_desc:
            customMetricsTreeItem = QTreeWidgetItem(allChannelsTreeItem)
            customMetricsTreeItem.setText(0, "Custom measurements")
            allChannelsTreeItem.addChild(customMetricsTreeItem)

        self.addTreeItems(
            foregrMetricsTreeItem, metrics_desc.keys(), isCol=True, isChannelLess=True
        )
        self.addTreeItems(
            bkgrMetricsTreeItem, bkgr_val_desc.keys(), isCol=True, isChannelLess=True
        )

        if custom_metrics_desc:
            self.addTreeItems(
                customMetricsTreeItem,
                custom_metrics_desc.keys(),
                isCol=True,
                isChannelLess=True,
            )

    def addOperator(self):
        button = self.sender()
        text = f"{self.equationDisplay.toPlainText()}{button.text}"
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(button.text))

    def clearEquation(self):
        self.isOperatorMode = False
        self.equationDisplay.setPlainText("")
        self.initAttributes()

    def initAttributes(self):
        self.clearLenghts = []
        self.equationColNames = []
        self.channelLessColnames = []

    def clearEntryEquation(self):
        if not self.clearLenghts:
            return

        text = self.equationDisplay.toPlainText()
        newText = text[: -self.clearLenghts[-1]]
        clearedText = text[-self.clearLenghts[-1] :]
        self.clearLenghts.pop(-1)
        self.equationDisplay.setPlainText(newText)
        if clearedText in self.equationColNames:
            self.equationColNames.remove(clearedText)
        if clearedText in self.channelLessColnames:
            self.channelLessColnames.remove(clearedText)

    def addTreeItems(self, parentItem, itemsText, isCol=False, isChannelLess=False):
        for text in itemsText:
            _item = QTreeWidgetItem(parentItem)
            _item.setText(0, text)
            parentItem.addChild(_item)
            if isCol:
                _item.isCol = True
            _item.isChannelLess = isChannelLess

    def addColname(self, item, column):
        if not hasattr(item, "isCol"):
            return

        colName = item.text(0)
        text = f"{self.equationDisplay.toPlainText()}{colName}"
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(colName))
        self.equationColNames.append(colName)
        if item.isChannelLess:
            self.channelLessColnames.append(colName)

    def _debug(self):
        print(self.getEquationsDict())

    def getEquationsDict(self):
        equation = self.equationDisplay.toPlainText()
        newColName = self.newColNameLineEdit.text()
        if not self.channelLessColnames:
            chNamesInTerms = set()
            for term in self.equationColNames:
                for chName in self.allChNames:
                    if chName in term:
                        chNamesInTerms.add(chName)
            if len(chNamesInTerms) == 1:
                # Equation uses metrics from a single channel --> append channel name
                chName = chNamesInTerms.pop()
                chColName = f"{chName}_{newColName}"
                isMixedChannels = False
                return {chColName: equation}, isMixedChannels
            else:
                # Equation doesn't use all channels metrics nor is single channel
                isMixedChannels = True
                return {newColName: equation}, isMixedChannels

        isMixedChannels = False
        equations = {}
        for chName in self.allChNames:
            chEquation = equation
            chEquationName = newColName
            # Append each channel name to channelLess terms
            for colName in self.channelLessColnames:
                chColName = f"{chName}{colName}"
                chEquation = chEquation.replace(colName, chColName)
                chEquationName = f"{chName}_{newColName}"
                equations[chEquationName] = chEquation
        return equations, isMixedChannels

    def ok_cb(self):
        if not self.newColNameLineEdit.text():
            self.warnEmptyEquationName()
            return

        self.cancel = False

        # Save equation to "<user_profile_path>/acdc-metrics/combine_metrics.ini" file
        config = measurements.read_saved_user_combine_config()

        equationsDict, isMixedChannels = self.getEquationsDict()
        for newColName, equation in equationsDict.items():
            config = measurements.add_user_combine_metrics(
                config, equation, newColName, isMixedChannels
            )

        isChannelLess = len(self.channelLessColnames) > 0
        if isChannelLess:
            channelLess_equation = self.equationDisplay.toPlainText()
            equation_name = self.newColNameLineEdit.text()
            config = measurements.add_channelLess_combine_metrics(
                config, channelLess_equation, equation_name, self.channelLessColnames
            )

        measurements.save_common_combine_metrics(config)

        self.sigOk.emit(self)

        if self.closeOnOk:
            self.close()

    def warnEmptyEquationName(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            "New measurement name" field <b>cannot be empty</b>!
        """)
        msg.critical(self, "Empty new measurement name", txt)

    def showHelp(self):
        txt = measurements.get_combine_metrics_help_txt()
        msg = widgets.myMessageBox(
            showCentered=False,
            wrapText=False,
            scrollableText=True,
            enlargeWidthFactor=1.7,
        )
        path = measurements.acdc_metrics_path
        msg.addShowInFileManagerButton(path, txt="Show saved file...")
        msg.information(self, "Combine measurements help", txt)

    def test_cb(self):
        # Evaluate equation with random inputs
        equation = self.equationDisplay.toPlainText()
        random_data = np.random.rand(1, len(self.equationColNames)) * 5
        df = pd.DataFrame(data=random_data, columns=self.equationColNames).round(5)
        newColName = self.newColNameLineEdit.text()
        try:
            df[newColName] = df.eval(equation)
        except Exception as e:
            traceback.print_exc()
            self.testOutputDisplay.setHtml(html_utils.paragraph(e))
            self.testOutputDisplay.setStyleSheet("border: 2px solid red")
            return

        self.testOutputDisplay.setStyleSheet("border: 2px solid green")
        self.okButton.setDisabled(False)

        result = df.round(5).iloc[0][newColName]

        # Substitute numbers into equation
        inputs = df.iloc[0]
        equation_numbers = equation
        for c, col in enumerate(self.equationColNames):
            equation_numbers = equation_numbers.replace(col, str(inputs[c]))

        # Format output into html text
        cols = self.equationColNames
        inputs_txt = [f"{col} = {input}" for col, input in zip(cols, inputs)]
        list_html = html_utils.to_list(inputs_txt)
        text = html_utils.paragraph(f"""
            By substituting the following random inputs:
            {list_html}
            we get the equation:<br><br>
            &nbsp;&nbsp;<code>{newColName} = {equation_numbers}</code><br><br>
            that <b>equals to</b>:<br><br>
            &nbsp;&nbsp;<code>{newColName} = {result}</code>
        """)
        self.testOutputDisplay.setHtml(text)


class CombineMetricsMultiDfsDialog(QBaseDialog):
    sigOk = Signal(object, object)
    sigClose = Signal(bool)

    def __init__(self, acdcDfs, allChNames, parent=None, debug=False):
        super().__init__(parent)

        self.setWindowTitle("Add combined measurement")

        self.initAttributes()

        self.acdcDfs = acdcDfs
        self.cancel = True
        self.isOperatorMode = False

        mainLayout = QVBoxLayout()
        equationLayout = QHBoxLayout()

        treesLayout = QHBoxLayout()
        for i, (acdc_df_endname, acdc_df) in enumerate(acdcDfs.items()):
            metricsTreeWidget = QTreeWidget()
            metricsTreeWidget.setHeaderHidden(True)
            metricsTreeWidget.setFont(font)

            classified_metrics = measurements.classify_acdc_df_colnames(
                acdc_df, allChNames
            )

            for chName in allChNames:
                channelTreeItem = QTreeWidgetItem(metricsTreeWidget)
                channelTreeItem.setText(0, f"{chName} measurements")
                metricsTreeWidget.addTopLevelItem(channelTreeItem)

                standard_metrics = classified_metrics["foregr"][chName]
                bkgr_metrics = classified_metrics["bkgr"][chName]
                custom_metrics = classified_metrics["custom"][chName]

                if standard_metrics:
                    foregrMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
                    foregrMetricsTreeItem.setText(0, "Cell signal measurements")
                    channelTreeItem.addChild(foregrMetricsTreeItem)
                    self.addTreeItems(
                        foregrMetricsTreeItem, standard_metrics, isCol=True, index=i
                    )

                if bkgr_metrics:
                    bkgrMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
                    bkgrMetricsTreeItem.setText(0, "Background values")
                    channelTreeItem.addChild(bkgrMetricsTreeItem)
                    self.addTreeItems(
                        bkgrMetricsTreeItem, bkgr_metrics, isCol=True, index=i
                    )

                if custom_metrics:
                    customMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
                    customMetricsTreeItem.setText(0, "Custom measurements")
                    channelTreeItem.addChild(customMetricsTreeItem)
                    self.addTreeItems(
                        customMetricsTreeItem, custom_metrics, isCol=True, index=i
                    )

            if classified_metrics["size"]:
                sizeMetricsTreeItem = QTreeWidgetItem(metricsTreeWidget)
                sizeMetricsTreeItem.setText(0, "Size measurements")
                metricsTreeWidget.addTopLevelItem(sizeMetricsTreeItem)
                self.addTreeItems(
                    sizeMetricsTreeItem, classified_metrics["size"], isCol=True, index=i
                )

            if classified_metrics["props"]:
                propMetricsTreeItem = QTreeWidgetItem(metricsTreeWidget)
                propMetricsTreeItem.setText(0, "Region properties")
                metricsTreeWidget.addTopLevelItem(propMetricsTreeItem)
                self.addTreeItems(
                    propMetricsTreeItem,
                    classified_metrics["props"],
                    isCol=True,
                    index=i,
                )

            treeLayout = QVBoxLayout()
            treeTitle = QLabel(
                html_utils.paragraph(
                    f"{i + 1}. <code>{acdc_df_endname}</code> measurements  "
                )
            )
            treeLayout.addWidget(treeTitle)
            treeLayout.addWidget(metricsTreeWidget)
            treesLayout.addLayout(treeLayout)

            metricsTreeWidget.index = i
            metricsTreeWidget.itemDoubleClicked.connect(self.addColname)

        operatorsLayout = QHBoxLayout()
        operatorsLayout.addStretch(1)

        iconSize = 24

        self.operatorButtons = []
        self.operators = [
            ("add", "+"),
            ("subtract", "-"),
            ("multiply", "*"),
            ("divide", "/"),
            ("open_bracket", "("),
            ("close_bracket", ")"),
            ("square", "**2"),
            ("pow", "**"),
            ("ln", "log("),
            ("log10", "log10("),
        ]
        operatorFont = QFont()
        operatorFont.setPixelSize(16)
        for name, text in self.operators:
            button = QPushButton()
            button.setIcon(QIcon(f":{name}.svg"))
            button.setIconSize(QSize(iconSize, iconSize))
            button.text = text
            operatorsLayout.addWidget(button)
            self.operatorButtons.append(button)
            button.clicked.connect(self.addOperator)
            # button.setFont(operatorFont)

        clearButton = QPushButton()
        clearButton.setIcon(QIcon(":clear.svg"))
        clearButton.setIconSize(QSize(iconSize, iconSize))
        clearButton.setFont(operatorFont)

        clearEntryButton = QPushButton()
        clearEntryButton.setIcon(QIcon(":backspace.svg"))
        clearEntryButton.setFont(operatorFont)
        clearEntryButton.setIconSize(QSize(iconSize, iconSize))

        operatorsLayout.addWidget(clearButton)
        operatorsLayout.addWidget(clearEntryButton)
        operatorsLayout.addStretch(1)

        newColNameLayout = QVBoxLayout()
        newColNameLineEdit = widgets.alphaNumericLineEdit()
        newColNameLineEdit.setAlignment(Qt.AlignCenter)
        self.newColNameLineEdit = newColNameLineEdit
        newColNameLayout.addStretch(1)
        newColNameLayout.addWidget(QLabel("New measurement name:"))
        newColNameLayout.addWidget(newColNameLineEdit)
        newColNameLayout.addStretch(1)

        equationDisplayLayout = QVBoxLayout()
        equationDisplayLayout.addWidget(QLabel("Equation:"))
        equationDisplay = QPlainTextEdit()
        # equationDisplay.setReadOnly(True)
        self.equationDisplay = equationDisplay
        equationDisplayLayout.addWidget(equationDisplay)
        equationDisplayLayout.setStretch(0, 0)
        equationDisplayLayout.setStretch(1, 1)

        equationLayout.addLayout(newColNameLayout)
        equationLayout.addWidget(QLabel(" = "))
        equationLayout.addLayout(equationDisplayLayout)
        equationLayout.setStretch(0, 1)
        equationLayout.setStretch(1, 0)
        equationLayout.setStretch(2, 2)

        instructions = html_utils.paragraph("""
            <b>Double-click</b> on any of the <b>available measurements</b>
            to add it to the equation.<br><br>
            <i>NOTE: the result will be saved in a new <code>acdc_output</code>
            file as a column with the same name<br>
            you enter in "New measurement name"
            field.</i><br>
        """)

        buttonsLayout = QHBoxLayout()

        cancelButton = widgets.cancelPushButton("Cancel")
        testButton = widgets.calcPushButton("Test equation")
        okButton = widgets.okPushButton(" Ok ")
        okButton.setDisabled(True)
        self.okButton = okButton

        if debug:
            debugButton = QPushButton("Debug")
            debugButton.clicked.connect(self._debug)
            buttonsLayout.addWidget(debugButton)

        self.statusLabel = QLabel()
        buttonsLayout.addWidget(self.statusLabel)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(testButton)
        buttonsLayout.addWidget(okButton)

        mainLayout.addWidget(QLabel(instructions))
        mainLayout.addLayout(treesLayout)
        mainLayout.addLayout(operatorsLayout)
        mainLayout.addLayout(equationLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        clearButton.clicked.connect(self.clearEquation)
        clearEntryButton.clicked.connect(self.clearEntryEquation)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        testButton.clicked.connect(self.test_cb)

        self.equationDisplay.textChanged.connect(self.equationChanged)
        # self.newColNameLineEdit.editingFinished.connect(self.equationChanged)

        self.setLayout(mainLayout)
        self.setFont(font)

        self.setStyleSheet(TREEWIDGET_STYLESHEET)

    def setLogger(self, logger, logs_path, log_path):
        self.logger = logger
        self.logs_path = logs_path
        self.log_path = log_path

    def closeEvent(self, event):
        self.sigClose.emit(self.cancel)
        return super().closeEvent(event)

    def getCombinedDf(self):
        dfs = []
        for i, acdc_df in enumerate(self.acdcDfs.values()):
            dfs.append(acdc_df.add_suffix(f"_table{i + 1}"))
        return pd.concat(dfs, axis=1)

    def _log(self, txt):
        if hasattr(self, "logger"):
            self.logger.info(txt)
        else:
            print(f"[INFO]: {txt}")

    def equationChanged(self):
        self.okButton.setDisabled(True)
        self.statusLabel.setText("")

    @exception_handler
    def test_cb(self):
        combined_df = self.getCombinedDf()
        new_df = pd.DataFrame(index=combined_df.index)
        equation = self.equationDisplay.toPlainText()
        newColName = self.newColNameLineEdit.text()
        new_df[newColName] = combined_df.eval(equation)
        self.okButton.setDisabled(False)
        self._log("Equation test was successful.")
        self.statusLabel.setText("Equation test was successful. You can now click OK.")

    def addOperator(self):
        button = self.sender()
        text = f"{self.equationDisplay.toPlainText()}{button.text}"
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(button.text))

    def clearEquation(self):
        self.isOperatorMode = False
        self.equationDisplay.setPlainText("")
        self.initAttributes()

    def initAttributes(self):
        self.clearLenghts = []
        self.equationColNames = []
        self.channelLessColnames = []

    def clearEntryEquation(self):
        if not self.clearLenghts:
            return

        text = self.equationDisplay.toPlainText()
        newText = text[: -self.clearLenghts[-1]]
        clearedText = text[-self.clearLenghts[-1] :]
        self.clearLenghts.pop(-1)
        self.equationDisplay.setPlainText(newText)
        if clearedText in self.equationColNames:
            self.equationColNames.remove(clearedText)
        if clearedText in self.channelLessColnames:
            self.channelLessColnames.remove(clearedText)

    def addTreeItems(
        self, parentItem, itemsText, isCol=False, isChannelLess=False, index=None
    ):
        for text in itemsText:
            _item = QTreeWidgetItem(parentItem)
            _item.setText(0, text)
            parentItem.addChild(_item)
            if isCol:
                _item.isCol = True
            if index is not None:
                _item.index = index
            _item.isChannelLess = isChannelLess

    def addColname(self, item, column):
        if not hasattr(item, "isCol"):
            return

        colName = f"{item.text(0)}_table{item.index + 1}"
        text = f"{self.equationDisplay.toPlainText()}{colName}"

        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(colName))
        self.equationColNames.append(colName)
        if item.isChannelLess:
            self.channelLessColnames.append(colName)

    def _debug(self):
        print(self.getEquationsDict())

    def ok_cb(self):
        if not self.newColNameLineEdit.text():
            self.warnEmptyEquationName()
            return
        if not self.equationDisplay.toPlainText():
            self.warnEmptyEquation()
            return

        self.expression = self.equationDisplay.toPlainText()
        self.newColname = self.newColNameLineEdit.text()
        self.cancel = False
        self.sigOk.emit(self.newColname, self.expression)
        self.close()

    def warnEmptyEquation(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            "Equation" field <b>cannot be empty</b>!
        """)
        msg.critical(self, "Empty equation", txt)

    def warnEmptyEquationName(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            "New measurement name" field <b>cannot be empty</b>!
        """)
        msg.critical(self, "Empty new measurement name", txt)


class CombineMetricsMultiDfsSummaryDialog(QBaseDialog):
    sigLoadAdditionalAcdcDf = Signal()

    def __init__(self, acdcDfs, allChNames, parent=None, debug=False):
        super().__init__(parent)

        self.editedIndex = None
        self.cancel = True
        self.acdcDfs = acdcDfs
        self.allChNames = allChNames

        self.setWindowTitle("Combine measurements summary")

        mainLayout = QVBoxLayout()
        viewLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        row = 0
        txt = html_utils.paragraph("Selected acdc_output tables:")
        viewLayout.addWidget(QLabel(txt), row, 0)

        row += 1
        items = [
            f"• <b>Table {i + 1}</b>: <code>{e}</code>"
            for i, e in enumerate(acdcDfs.keys())
        ]
        selectedAcdcDfsList = widgets.readOnlyQList()
        selectedAcdcDfsList.addItems(items)
        self.selectedAcdcDfsList = selectedAcdcDfsList

        tablesButtonsLayout = QVBoxLayout()
        loadAcdcDfButton = widgets.showInFileManagerButton("Load additional tables")
        tablesButtonsLayout.addWidget(loadAcdcDfButton)

        loadEquationsButton = widgets.reloadPushButton("Load previously used equations")
        tablesButtonsLayout.addWidget(loadEquationsButton)

        tablesButtonsLayout.addStretch(1)

        viewLayout.addWidget(selectedAcdcDfsList, row, 0)
        viewLayout.addLayout(tablesButtonsLayout, row, 1)
        viewLayout.setRowStretch(row, 1)

        row += 1
        txt = html_utils.paragraph("Equations:")
        viewLayout.addWidget(QLabel(txt), row, 0)

        row += 1
        self.equationsList = widgets.TreeWidget()
        self.equationsList.setFont(font)
        self.equationsList.setHeaderLabels(["Metric", "Expression"])
        self.equationsList.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )

        equationsButtonsLayout = QVBoxLayout()
        addEquationButton = widgets.addPushButton("Add metric")
        removeEquationButton = widgets.subtractPushButton("Remove metric(s)")
        editEquationButton = widgets.editPushButton("Edit metric")
        removeEquationButton.setDisabled(True)
        editEquationButton.setDisabled(True)
        self.removeEquationButton = removeEquationButton
        self.editEquationButton = editEquationButton

        equationsButtonsLayout.addWidget(addEquationButton)
        equationsButtonsLayout.addWidget(removeEquationButton)
        equationsButtonsLayout.addWidget(editEquationButton)
        equationsButtonsLayout.addStretch(1)

        viewLayout.addWidget(self.equationsList, row, 0)
        viewLayout.addLayout(equationsButtonsLayout, row, 1)
        viewLayout.setRowStretch(row, 2)

        cancelButton = widgets.cancelPushButton("Cancel")
        okButton = widgets.okPushButton("Ok")

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        viewLayout.setVerticalSpacing(10)
        mainLayout.addLayout(viewLayout)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        addEquationButton.clicked.connect(self.addEquation_cb)
        loadAcdcDfButton.clicked.connect(self.loadButtonClicked)
        loadEquationsButton.clicked.connect(self.loadEquationsButtonClicked)
        removeEquationButton.clicked.connect(self.removeButtonClicked)
        editEquationButton.clicked.connect(self.editButtonClicked)
        self.equationsList.itemSelectionChanged.connect(
            self.onEquationItemSelectionChanged
        )

        self.setLayout(mainLayout)

    def setLogger(self, logger, logs_path, log_path):
        self.logger = logger
        self.logs_path = logs_path
        self.log_path = log_path

    def loadEquationsButtonClicked(self):
        MostRecentPath = myutils.getMostRecentPath()
        file_path = QFileDialog.getOpenFileName(
            self,
            "Select equations file",
            MostRecentPath,
            "Config Files (*.ini);;All Files (*)",
        )[0]
        if file_path == "":
            return

        cp = config.ConfigParser()
        cp.read(file_path)
        sectionToMatch = [f"table{i + 1}:{end}" for i, end in enumerate(self.acdcDfs)]
        sectionToMatch = ";".join(sectionToMatch)

        lists = {}
        nonMatchingLists = {}
        groupsDescr = {}

        for section in cp.sections():
            # Tag acdc_output names with <code> html and table(\d+) with html bold tag
            listName = ";".join(
                [
                    re.sub(
                        r"table(\d+):(.*)", r"<b>table\g<1></b>: <code>\g<2></code>", s
                    )
                    for s in section.split(";")
                ]
            )
            listName = listName.replace(";", " ; ")
            children = [f"{opt} = {cp[section][opt]}" for opt in cp[section]]
            if section == sectionToMatch:
                groupsDescr[listName] = (
                    "Equations that were calculated from the <b>same "
                    "table names</b> you loaded"
                )
                lists[listName] = children
            else:
                groupsDescr[listName] = (
                    "Equations that were calculated from <b>table names that "
                    "you did not load</b> now"
                )
                nonMatchingLists[listName] = children
                # # Not implemented yet --> selecting from non matching table names
                # # would require an additional widget where the user sets
                # # what df1 and df2 are.
                # trees[treeName] = children

        if not lists:
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            txt = html_utils.paragraph("""
                <b>None of the equations</b> in the selected file used the <b>same 
                table names</b> that you loaded.<br><br>
                See below which table names and equations are present in the loaded file.
            """)
            with open(file_path) as iniFile:
                detailedText = iniFile.read()

            msg.warning(self, "Not the same tables", txt, showDialog=False)
            msg.setDetailedText(detailedText, visible=True)
            msg.addShowInFileManagerButton(os.path.dirname(file_path))
            msg.exec_()
            return

        selectWindow = MultiListSelector(
            lists,
            groupsDescr=groupsDescr,
            title="Select equations to load",
            infoTxt="Select equations you want to load",
        )
        selectWindow.exec_()
        if selectWindow.cancel or not selectWindow.selectedItems:
            return

        for listName, equations in selectWindow.selectedItems.items():
            for equation in equations:
                metricName, expression = equation.split(" = ")
                self.addEquation(metricName, expression)

    def ok_cb(self):
        self.cancel = False
        self.equations = {}
        for i in range(self.equationsList.topLevelItemCount()):
            item = self.equationsList.topLevelItem(i)
            self.equations[item.text(0)] = item.text(1)

        self.close()

    def loadButtonClicked(self):
        self.sigLoadAdditionalAcdcDf.emit()

    def removeButtonClicked(self):
        for item in self.equationsList.selectedItems():
            self.equationsList.invisibleRootItem().removeChild(item)

    def editButtonClicked(self):
        self.editedItem = self.equationsList.selectedItems()[0]
        self.editedIndex = self.equationsList.indexOfTopLevelItem(self.editedItem)
        self.addEquation_cb()

    def onEquationItemSelectionChanged(self):
        selectedItems = self.equationsList.selectedItems()
        if len(selectedItems) == 1:
            self.editEquationButton.setDisabled(False)
            self.removeEquationButton.setDisabled(False)
        elif len(selectedItems) > 1:
            self.removeEquationButton.setDisabled(False)
            self.editEquationButton.setDisabled(True)
        else:
            self.removeEquationButton.setDisabled(True)
            self.editEquationButton.setDisabled(True)

    def addAcdcDfs(self, acdcDfsDict):
        self.acdcDfs = {**self.acdcDfs, **acdcDfsDict}
        items = [
            f"• <b>Table {i + 1}</b>: <code>{e}</code>"
            for i, e in enumerate(self.acdcDfs.keys())
        ]
        self.selectedAcdcDfsList = widgets.readOnlyQList()
        self.selectedAcdcDfsList.addItems(items)

    def addEquation(self, newColname, expression):
        if self.editedIndex is not None:
            self.equationsList.invisibleRootItem().removeChild(self.editedItem)
        bkgrColor = QColor(*BACKGROUND_RGBA[:3], 200)
        item = widgets.TreeWidgetItem(
            self.equationsList, columnColors=[None, bkgrColor]
        )
        item.setText(0, newColname)
        item.setText(1, expression)
        if self.editedIndex is not None:
            self.equationsList.insertTopLevelItem(self.editedIndex, item)
        else:
            self.equationsList.addTopLevelItem(item)
        self.equationsList.resizeColumnToContents(0)
        self.equationsList.resizeColumnToContents(1)
        self.editedIndex = None

    def addEquation_cb(self):
        self.addEquationWin = CombineMetricsMultiDfsDialog(
            self.acdcDfs, self.allChNames, parent=self
        )
        if hasattr(self, "logger"):
            self.addEquationWin.setLogger(self.logger, self.logs_path, self.log_path)
        if self.editedIndex is not None:
            editedMetricName = self.editedItem.text(0)
            self.addEquationWin.newColNameLineEdit.setText(editedMetricName)
            editedExpression = self.editedItem.text(1)
            self.addEquationWin.equationDisplay.setPlainText(editedExpression)
        self.addEquationWin.show()
        self.addEquationWin.sigOk.connect(self.addEquation)
        self.addEquationWin.sigClose.connect(self.addEquationClosed)

    def addEquationClosed(self, cancelled):
        if cancelled:
            self.editedIndex = None

    def showEvent(self, event) -> None:
        self.resize(int(self.width() * 2), self.height())


class SelectFeaturesRange:
    def __init__(
        self, posData, force_postprocess_2D=False, qparent=None, sigValueChanged=None
    ) -> None:
        self.posData = posData
        self.qparent = qparent
        self.force_postprocess_2D = force_postprocess_2D
        self.sigValueChanged = sigValueChanged

        self.lowRangeWidgets = widgets.CheckableSpinBoxWidgets()
        self.highRangeWidgets = widgets.CheckableSpinBoxWidgets()

        self.selectButton = widgets.FeatureSelectorButton("Click to select feature...")
        self.selectButton.setSizeLongestText(
            "Spotfit intens. metric, Foregr. integral gauss. peak"
        )
        self.selectButton.clicked.connect(self.selectFeature)
        self.selectButton.setCursor(Qt.PointingHandCursor)

        self.selectedFeatureGroups = {}

        self.widgets = [
            {"pos": (0, 0), "widget": self.lowRangeWidgets.checkbox},
            {"pos": (1, 0), "widget": self.lowRangeWidgets.spinbox},
            {"pos": (1, 1), "widget": widgets.LessThanPushButton(flat=True)},
            {"pos": (1, 2), "widget": self.selectButton},
            {"pos": (1, 3), "widget": widgets.LessThanPushButton(flat=True)},
            {"pos": (0, 4), "widget": self.highRangeWidgets.checkbox},
            {"pos": (1, 4), "widget": self.highRangeWidgets.spinbox},
            {"pos": (2, 0), "widget": widgets.VerticalSpacerEmptyWidget(height=10)},
        ]
        self.columnsStretches = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}

    def setText(self, text):
        self.selectButton.setText(text)

    def selectFeature(self):
        loadedChNames = [self.posData.user_ch_name]
        notLoadedChNames = []
        isZstack = self.posData.SizeZ > 1 and not self.force_postprocess_2D
        isSegm3D = self.posData.isSegm3D and not self.force_postprocess_2D
        self.selectFeatureDialog = SetMeasurementsDialog(
            loadedChNames,
            notLoadedChNames,
            isZstack,
            isSegm3D,
            posData=self.posData,
            parent=self.qparent,
            isSingleSelection=True,
            is_concat=True,
        )
        # self.selectFeatureDialog.resizeVertical()
        self.selectFeatureDialog.sigClosed.connect(self.setFeatureText)
        self.selectFeatureDialog.show()

    def setFeatureText(self):
        if self.selectFeatureDialog.cancel:
            return
        self.selectButton.setFlat(True)
        selectedMetricName, selectedMetricGroup = (
            self.selectFeatureDialog.selectedMetricNameAndGroup()
        )
        self.selectButton.setText(selectedMetricName)
        self.featureGroup = selectedMetricGroup


class SelectFeaturesRangeDialog(QBaseDialog):
    sigValueChanged = Signal(object)

    def __init__(self, posData=None, parent=None, force_postprocess_2D=False):
        super().__init__(parent)

        self.force_postprocess_2D = force_postprocess_2D

        layout = QVBoxLayout()
        self.setWindowTitle("Custom features for post-processing")

        self.groupbox = SelectFeaturesRangeGroupbox(
            posData=posData, parent=parent, force_postprocess_2D=force_postprocess_2D
        )

        buttonsLayout = QHBoxLayout()
        okPushButton = widgets.okPushButton(" Ok ")

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(okPushButton)

        okPushButton.clicked.connect(self.ok_cb)

        layout.addWidget(self.groupbox)
        layout.addSpacing(10)
        layout.addLayout(buttonsLayout)

        self.setLayout(layout)

    def ok_cb(self):
        if self.groupbox.selectedFeaturesRange():
            self.sigValueChanged.emit(None)
        self.hide()


class SelectFeaturesRangeGroupbox(QGroupBox):
    def __init__(self, posData=None, parent=None, force_postprocess_2D=False):
        super().__init__(parent)

        self.setTitle("Features and thresholds for filtering segmented objects")
        # self.setCheckable(True)

        self.posData = posData
        self.force_postprocess_2D = force_postprocess_2D

        self._layout = QGridLayout()
        self._layout.setVerticalSpacing(0)

        firstSelector = SelectFeaturesRange(
            posData, force_postprocess_2D=force_postprocess_2D
        )
        self.addButton = widgets.addPushButton("  Add feature    ")
        self.addButton.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        for col, widget in enumerate(firstSelector.widgets):
            row, col = widget["pos"]
            self._layout.addWidget(widget["widget"], row, col)
        for col, stretch in firstSelector.columnsStretches.items():
            self._layout.setColumnStretch(col, stretch)

        lastCol = self._layout.columnCount()
        self._layout.addWidget(self.addButton, 0, lastCol + 1, 2, 1)
        self.lastCol = lastCol + 1
        self.selectors = [firstSelector]

        self.setLayout(self._layout)

        # self.setFont(font)

        self.addButton.clicked.connect(self.addFeatureField)

    def addFeatureField(self):
        row = self._layout.rowCount()
        selector = SelectFeaturesRange(
            self.posData, force_postprocess_2D=self.force_postprocess_2D
        )
        delButton = widgets.delPushButton("Remove feature")
        delButton.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        delButton.selector = selector
        selector.delButton = delButton
        for col, widget in enumerate(selector.widgets):
            relRow, col = widget["pos"]
            self._layout.addWidget(widget["widget"], relRow + row, col)
        self._layout.addWidget(delButton, row, self.lastCol, 2, 1)
        self.selectors.append(selector)
        delButton.clicked.connect(self.removeFeatureField)

    def resetFields(self):
        while len(self.selectors) > 1:
            selector = self.selectors[-1]
            selector.delButton.click()
        firstSelector = self.selectors[0]
        firstSelector.selectButton.setText("Click to select feature...")
        firstSelector.lowRangeWidgets.checkbox.setChecked(False)
        firstSelector.highRangeWidgets.checkbox.setChecked(False)

    def removeFeatureField(self):
        delButton = self.sender()
        for widget in delButton.selector.widgets:
            self._layout.removeWidget(widget["widget"])
        self._layout.removeWidget(delButton)
        self.selectors.remove(delButton.selector)

    def selectedFeaturesRange(self):
        featuresRange = {}
        for selector in self.selectors:
            if selector.selectButton.text().find("Click") != -1:
                continue
            featuresRange[selector.selectButton.text()] = (
                selector.lowRangeWidgets.value(),
                selector.highRangeWidgets.value(),
            )
        return featuresRange

    def selectedFeaturesGroup(self):
        featuresGroup = {}
        for selector in self.selectors:
            if selector.selectButton.text().find("Click") != -1:
                continue
            group = selector.featureGroup
            featuresGroup[selector.selectButton.text()] = group
        return featuresGroup

    def groupedFeatures(self):
        featuresGroup = self.selectedFeaturesGroup()
        groupedFeatures = {}
        for feature, group in featuresGroup.items():
            group = featuresGroup[feature]
            if isinstance(group, str):
                key = group
                if key not in groupedFeatures:
                    groupedFeatures[key] = []
                groupedFeatures[key].append(feature)
            else:
                key, channel = list(group.items())[0]
                if key not in groupedFeatures:
                    groupedFeatures[key] = {}
                if channel not in groupedFeatures[key]:
                    groupedFeatures[key][channel] = []
                groupedFeatures[key][channel].append(feature)
        return groupedFeatures

    def setValue(self, value):
        pass


class CombineFeaturesCalculator(QBaseDialog):
    sigOk = Signal(object)

    def __init__(
        self,
        features_groups: dict,
        group_name_to_col_mapper: dict = None,
        title="Combine features calculator",
        parent=None,
    ):
        super().__init__(parent)

        self.cancel = True

        self.setWindowTitle(title)
        self.initAttributes()

        mainLayout = QVBoxLayout()
        equationLayout = QHBoxLayout()

        metricsTreeWidget = QTreeWidget()
        metricsTreeWidget.setHeaderHidden(True)
        metricsTreeWidget.setFont(font)
        self.metricsTreeWidget = metricsTreeWidget

        for groupName, features in features_groups.items():
            topLevelTreeWidgetItem = QTreeWidgetItem(metricsTreeWidget)
            topLevelTreeWidgetItem.setText(0, groupName)
            metricsTreeWidget.addTopLevelItem(topLevelTreeWidgetItem)
            self.addTreeItems(
                topLevelTreeWidgetItem,
                features,
                isCol=True,
                name_to_col_mapper=group_name_to_col_mapper.get(groupName),
            )

        operatorsLayout = self.createOperatorsLayout()
        newFeatureNameLayout = self.createNewFeatureNameLayout()
        equationDisplayLayout = self.createEquationDisplayLayout()

        equationLayout.addLayout(newFeatureNameLayout)
        equationLayout.addWidget(QLabel(" = "))
        equationLayout.addLayout(equationDisplayLayout)
        equationLayout.setStretch(0, 1)
        equationLayout.setStretch(1, 0)
        equationLayout.setStretch(2, 2)

        testOutputLayout = self.createTestOutputLayout()
        buttonsLayout = self.createButtonsOutputLayout()

        instructions = html_utils.paragraph("""
            <b>Double-click</b> on any of the <b>available measurements</b>
            to add it to the equation.<br><br>
            Before clicking the `Ok` button, check that the equation returns 
            the expected result by clicking the `Test output` button. 
        """)

        mainLayout.addWidget(QLabel(instructions))
        mainLayout.addWidget(QLabel("Available measurements:"))
        mainLayout.addWidget(metricsTreeWidget)
        mainLayout.addLayout(operatorsLayout)
        mainLayout.addLayout(equationLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addLayout(testOutputLayout)

        metricsTreeWidget.itemDoubleClicked.connect(self.addFeatureName)
        self.setLayout(mainLayout)
        self.setFont(font)

        self.setStyleSheet(TREEWIDGET_STYLESHEET)

    def setExpandedAll(self, expanded):
        if expanded:
            self.expandAll()
        else:
            for i in range(self.metricsTreeWidget.topLevelItemCount()):
                topLevelItem = self.metricsTreeWidget.topLevelItem(i)
                topLevelItem.setExpanded(False)

    def expandAll(self):
        for i in range(self.metricsTreeWidget.topLevelItemCount()):
            topLevelItem = self.metricsTreeWidget.topLevelItem(i)
            topLevelItem.setExpanded(True)

    def addTreeItems(self, parentItem, itemsText, isCol=False, name_to_col_mapper=None):
        for text in itemsText:
            _item = QTreeWidgetItem(parentItem)
            _item.setText(0, text)
            parentItem.addChild(_item)
            if isCol:
                _item.isCol = True
            _item.variable_name = text
            if name_to_col_mapper is None:
                continue

            col_name = name_to_col_mapper.get(text, None)
            if col_name is None:
                continue

            _item.variable_name = col_name

    def addFeatureName(self, item, column):
        if not hasattr(item, "isCol"):
            return

        colName = item.variable_name
        text = f"{self.equationDisplay.toPlainText()}{colName}"
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(colName))
        self.equationColNames.append(colName)

    def clearEquation(self):
        self.isOperatorMode = False
        self.equationDisplay.setPlainText("")
        self.initAttributes()

    def createButtonsOutputLayout(self):
        buttonsLayout = QHBoxLayout()

        cancelButton = widgets.cancelPushButton("Cancel")
        helpButton = widgets.infoPushButton("  Help...")
        testButton = widgets.calcPushButton("Test output")
        okButton = widgets.okPushButton(" Ok ")
        okButton.setDisabled(True)
        self.okButton = okButton

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(helpButton)
        buttonsLayout.addWidget(testButton)
        buttonsLayout.addWidget(okButton)

        helpButton.clicked.connect(self.showHelp)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        testButton.clicked.connect(self.test_cb)

        return buttonsLayout

    def ok_cb(self):
        if not self.newFeatureNameLineEdit.text():
            self.warnEmptyEquationName()
            return

        self.equation = self.equationDisplay.toPlainText()
        self.newFeatureName = self.newFeatureNameLineEdit.text()
        self.cancel = False
        self.close()
        self.sigOk.emit(self)

    def test_cb(self):
        # Evaluate equation with random inputs
        equation = self.equationDisplay.toPlainText()
        random_data = np.random.rand(1, len(self.equationColNames)) * 5
        df = pd.DataFrame(data=random_data, columns=self.equationColNames).round(5)
        newColName = self.newFeatureNameLineEdit.text()
        try:
            df[newColName] = df.eval(equation)
        except Exception as e:
            traceback.print_exc()
            self.testOutputDisplay.setHtml(html_utils.paragraph(e))
            self.testOutputDisplay.setStyleSheet("border: 2px solid red")
            return

        self.testOutputDisplay.setStyleSheet("border: 2px solid green")
        self.okButton.setDisabled(False)

        result = df.round(5).iloc[0][newColName]

        # Substitute numbers into equation
        inputs = df.iloc[0]
        equation_numbers = equation
        for c, col in enumerate(self.equationColNames):
            equation_numbers = equation_numbers.replace(col, str(inputs[c]))

        # Format output into html text
        cols = self.equationColNames
        inputs_txt = [f"{col} = {input}" for col, input in zip(cols, inputs)]
        list_html = html_utils.to_list(inputs_txt)
        text = html_utils.paragraph(f"""
            By substituting the following random inputs:
            {list_html}
            we get the equation:<br><br>
            &nbsp;&nbsp;<code>{newColName} = {equation_numbers}</code><br><br>
            that <b>equals to</b>:<br><br>
            &nbsp;&nbsp;<code>{newColName} = {result}</code>
        """)
        self.testOutputDisplay.setHtml(text)

    def warnEmptyEquationName(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            "New measurement name" field <b>cannot be empty</b>!
        """)
        msg.critical(self, "Empty new measurement name", txt)

    def showHelp(self):
        pass

    def createTestOutputLayout(self):
        testOutputLayout = QVBoxLayout()
        testOutputLayout.addWidget(QLabel("Result of test with random inputs:"))
        testOutputDisplay = QTextEdit()
        testOutputDisplay.setReadOnly(True)
        self.testOutputDisplay = testOutputDisplay
        testOutputLayout.addWidget(testOutputDisplay)
        testOutputLayout.setStretch(0, 0)
        testOutputLayout.setStretch(1, 1)

        return testOutputLayout

    def createEquationDisplayLayout(self):
        equationDisplayLayout = QVBoxLayout()
        equationDisplayLayout.addWidget(QLabel("Equation:"))
        equationDisplay = QPlainTextEdit()
        # equationDisplay.setReadOnly(True)
        self.equationDisplay = equationDisplay
        equationDisplayLayout.addWidget(equationDisplay)
        equationDisplayLayout.setStretch(0, 0)
        equationDisplayLayout.setStretch(1, 1)
        return equationDisplayLayout

    def createNewFeatureNameLayout(self):
        newFeatureNameLayout = QVBoxLayout()
        newFeatureNameLineEdit = widgets.alphaNumericLineEdit()
        newFeatureNameLineEdit.setAlignment(Qt.AlignCenter)
        self.newFeatureNameLineEdit = newFeatureNameLineEdit
        newFeatureNameLayout.addStretch(1)
        newFeatureNameLayout.addWidget(QLabel("New measurement name:"))
        newFeatureNameLayout.addWidget(newFeatureNameLineEdit)
        newFeatureNameLayout.addStretch(1)
        return newFeatureNameLayout

    def createOperatorsLayout(self):
        operatorsLayout = QHBoxLayout()
        operatorsLayout.addStretch(1)

        iconSize = 24

        self.operatorButtons = []
        self.operators = [
            ("add", "+"),
            ("subtract", "-"),
            ("multiply", "*"),
            ("divide", "/"),
            ("open_bracket", "("),
            ("close_bracket", ")"),
            ("square", "**2"),
            ("pow", "**"),
            ("ln", "log("),
            ("log10", "log10("),
        ]
        operatorFont = QFont()
        operatorFont.setPixelSize(16)
        for name, text in self.operators:
            button = QPushButton()
            button.setIcon(QIcon(f":{name}.svg"))
            button.setIconSize(QSize(iconSize, iconSize))
            button.text = text
            operatorsLayout.addWidget(button)
            self.operatorButtons.append(button)
            button.clicked.connect(self.addOperator)
            # button.setFont(operatorFont)

        clearButton = QPushButton()
        clearButton.setIcon(QIcon(":clear.svg"))
        clearButton.setIconSize(QSize(iconSize, iconSize))
        clearButton.setFont(operatorFont)

        clearEntryButton = QPushButton()
        clearEntryButton.setIcon(QIcon(":backspace.svg"))
        clearEntryButton.setFont(operatorFont)
        clearEntryButton.setIconSize(QSize(iconSize, iconSize))

        operatorsLayout.addWidget(clearButton)
        operatorsLayout.addWidget(clearEntryButton)
        operatorsLayout.addStretch(1)

        clearButton.clicked.connect(self.clearEquation)
        clearEntryButton.clicked.connect(self.clearEntryEquation)

        return operatorsLayout

    def addOperator(self):
        button = self.sender()
        text = f"{self.equationDisplay.toPlainText()}{button.text}"
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(button.text))

    def clearEquation(self):
        self.isOperatorMode = False
        self.equationDisplay.setPlainText("")
        self.initAttributes()

    def initAttributes(self):
        self.clearLenghts = []
        self.equationColNames = []
        self.channelLessColnames = []

    def clearEntryEquation(self):
        if not self.clearLenghts:
            return

        text = self.equationDisplay.toPlainText()
        newText = text[: -self.clearLenghts[-1]]
        clearedText = text[-self.clearLenghts[-1] :]
        self.clearLenghts.pop(-1)
        self.equationDisplay.setPlainText(newText)
        if clearedText in self.equationColNames:
            self.equationColNames.remove(clearedText)
        if clearedText in self.channelLessColnames:
            self.channelLessColnames.remove(clearedText)

# Sibling imports (deferred to avoid import cycles)
from .metadata import (
    MultiListSelector,
    filenameDialog,
)

