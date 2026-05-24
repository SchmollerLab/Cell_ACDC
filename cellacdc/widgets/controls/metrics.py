"""Composite controls: metrics."""

"""GUI widgets: controls."""

from collections import defaultdict, deque
from typing import Dict, List, Union, Iterable, Sequence
import os
import sys
import operator
import time
import re
import datetime
import numpy as np
import pandas as pd
import math
import traceback
import logging
import textwrap
import random

from functools import partial
from math import ceil

import skimage.draw
import skimage.morphology

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg

from qtpy.QtCore import (
    Signal,
    QTimer,
    Qt,
    QPoint,
    QUrl,
    Property,
    QPropertyAnimation,
    QEasingCurve,
    QLocale,
    QSize,
    QRect,
    QPointF,
    QRect,
    QPoint,
    QEasingCurve,
    QRegularExpression,
    QEvent,
    QEventLoop,
    QPropertyAnimation,
    QObject,
    QItemSelectionModel,
    QAbstractListModel,
    QModelIndex,
    QByteArray,
    QDataStream,
    QMimeData,
    QAbstractItemModel,
    QIODevice,
    QItemSelection,
    PYQT6,
    QRectF,
)
from qtpy.QtGui import (
    QFont,
    QPalette,
    QColor,
    QPen,
    QKeyEvent,
    QBrush,
    QPainter,
    QRegularExpressionValidator,
    QIcon,
    QPixmap,
    QKeySequence,
    QLinearGradient,
    QShowEvent,
    QDesktopServices,
    QFontMetrics,
    QGuiApplication,
    QLinearGradient,
    QImage,
    QCursor,
    QPicture,
)
from qtpy.QtWidgets import (
    QTextEdit,
    QLabel,
    QProgressBar,
    QHBoxLayout,
    QToolButton,
    QCheckBox,
    QApplication,
    QWidget,
    QVBoxLayout,
    QMainWindow,
    QTreeWidgetItemIterator,
    QLineEdit,
    QSlider,
    QSpinBox,
    QGridLayout,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QComboBox,
    QPushButton,
    QScrollBar,
    QGroupBox,
    QAbstractSlider,
    QDoubleSpinBox,
    QWidgetAction,
    QAction,
    QTabWidget,
    QAbstractSpinBox,
    QToolBar,
    QStyleOptionSpinBox,
    QStyle,
    QDialog,
    QSpacerItem,
    QFrame,
    QMenu,
    QActionGroup,
    QListWidget,
    QPlainTextEdit,
    QFileDialog,
    QListView,
    QAbstractItemView,
    QTreeWidget,
    QTreeWidgetItem,
    QListWidgetItem,
    QLayout,
    QStylePainter,
    QGraphicsBlurEffect,
    QGraphicsProxyWidget,
    QGraphicsObject,
    QButtonGroup,
    QStyleOptionSlider,
)
import qtpy.compat

import pyqtgraph as pg

pg.setConfigOption("imageAxisOrder", "row-major")

from ... import utils, measurements, is_mac, is_win, html_utils, is_linux
from ... import printl, settings_folderpath
from ... import colors, config
from ... import html_path
from ... import _palettes
from ... import load
from ... import apps
from ... import plot
from ... import annotate
from ... import urls
from ... import _core, core
from ... import QtScoped
from ... import prompts
from ...acdc_regex import float_regex
from ...config import PREPROCESS_MAPPER
from ... import _base_widgets

from ...components.palette import (  # noqa: E402
    BASE_COLOR,
    Gradients,
    GradientsImage,
    GradientsLabels,
    LINEEDIT_INVALID_ENTRY_STYLESHEET,
    LINEEDIT_WARNING_STYLESHEET,
    LISTWIDGET_STYLESHEET,
    PROGRESSBAR_HIGHLIGHTEDTEXT_QCOLOR,
    PROGRESSBAR_QCOLOR,
    TEXT_COLOR,
    TREEWIDGET_STYLESHEET,
    cmaps,
    font,
    getCustomGradients,
    nonInvertibleCmaps,
    sign_int_mapper,
    str_to_operator_mapper,
)
from ...components.progress import QtHandler, QLog, XStream  # noqa: E402
from ...components.buttons import *  # noqa: E402, F403
from ...components.layout import *  # noqa: E402, F403
from ...components.inputs_basic import *  # noqa: E402, F403
from ...components.path_controls import *  # noqa: E402, F403

from ...components.lists import *  # noqa: E402, F403
from ...components.base import QBaseWindow  # noqa: E402
from ...components.progress import (  # noqa: E402
    LoadingCircleAnimation,
    NoneWidget,
    ProgressBar,
    ProgressBarWithETA,
    QLogConsole,
)

class _metricsQGBox(QGroupBox):
    sigDelClicked = Signal(str, object)

    def __init__(
        self,
        desc_dict,
        title,
        favourite_funcs=None,
        isZstack=False,
        equations=None,
        addDelButton=False,
        delButtonMetricsDesc=None,
        parent=None,
        addCalcForEachZsliceToggle=False,
    ):
        QGroupBox.__init__(self, parent)

        highlightRgba = _palettes._highlight_rgba()
        r, g, b, a = highlightRgba
        self._highlightStylesheetColor = f"rgb({r}, {g}, {b})"

        self._parent = parent
        self.scrollArea = QScrollArea()
        self.scrollAreaWidget = QWidget()
        self.favourite_funcs = favourite_funcs

        self.doNotWarn = False

        layout = QVBoxLayout()
        inner_layout = QVBoxLayout()
        self.inner_layout = inner_layout
        if delButtonMetricsDesc is None:
            delButtonMetricsDesc = []

        self.checkBoxes = []
        self.checkedState = {}
        for metric_colname, metric_desc in desc_dict.items():
            rowLayout = QHBoxLayout()

            checkBox = QCheckBox(metric_colname)
            checkBox.setChecked(True)
            checkBox.scrollArea = self.scrollArea
            self.checkBoxes.append(checkBox)
            self.checkedState[checkBox] = True

            try:
                checkBox.equation = equations[metric_colname]
            except Exception as e:
                pass

            if addDelButton or metric_colname in delButtonMetricsDesc:
                delButton = delPushButton()
                delButton.setToolTip("Delete custom combined measurement")
                delButton.colname = metric_colname
                delButton.checkbox = checkBox
                delButton.clicked.connect(self.onDelClicked)
                delButton._layout = rowLayout
                rowLayout.addWidget(delButton)

            infoButton = infoPushButton()
            infoButton.setCursor(Qt.WhatsThisCursor)
            infoButton.info = metric_desc
            infoButton.colname = metric_colname
            infoButton.clicked.connect(self.showInfo)

            rowLayout.addWidget(infoButton)
            rowLayout.addWidget(checkBox)
            rowLayout.addStretch(1)

            inner_layout.addLayout(rowLayout)

        self.scrollAreaWidget.setLayout(inner_layout)
        self.scrollArea.setWidget(self.scrollAreaWidget)
        layout.addWidget(self.scrollArea)

        buttonsLayout = QHBoxLayout()

        buttonsLayout.addStretch(1)

        self.selectAllButton = selectAllPushButton()
        self.selectAllButton.sigClicked.connect(self.checkAll)

        buttonsLayout.addWidget(self.selectAllButton)

        if favourite_funcs is not None:
            self.loadFavouritesButton = reloadPushButton("  Load last selection...  ")
            self.loadFavouritesButton.clicked.connect(self.checkFavouriteFuncs)
            # self.checkFavouriteFuncs()
            buttonsLayout.addWidget(self.loadFavouritesButton)

        layout.addLayout(buttonsLayout)

        self.calcForEachZsliceToggle = None
        if addCalcForEachZsliceToggle:
            buttonsLayout = QHBoxLayout()
            self.calcForEachZsliceToggle = Toggle()
            tooltip = (
                "Calculate `cell_area` for each z-slice.\n\n"
                "The measurements will be saved in the column with name\n"
                "ending with `_zsliceN` where N is the z-slice number\n"
                "(starting from 0)."
            )
            calcForEachZsliceLabel = QClickableLabel("Calculate for each z-slice")
            calcForEachZsliceLabel.setToolTip(tooltip)
            self.calcForEachZsliceToggle.setToolTip(tooltip)
            buttonsLayout.addWidget(self.calcForEachZsliceToggle)
            buttonsLayout.addWidget(calcForEachZsliceLabel)
            buttonsLayout.addStretch(1)
            layout.addLayout(buttonsLayout)
            calcForEachZsliceLabel.clicked.connect(
                partial(
                    self.toggleCalcForEachZslice, toggle=self.calcForEachZsliceToggle
                )
            )

        self.setTitle(title)
        self.setCheckable(True)
        self.setLayout(layout)
        _font = QFont()
        _font.setPixelSize(11)
        self.setFont(_font)

        self.toggled.connect(self.toggled_cb)

    def toggleCalcForEachZslice(self, label, toggle=None):
        if toggle is None:
            toggle = self.calcForEachZsliceToggle

        toggle.setChecked(not toggle.isChecked())

    def isCalcForEachZsliceRequested(self):
        if self.calcForEachZsliceToggle is None:
            return False

        return self.calcForEachZsliceToggle.isChecked()

    def highlightCheckboxesFromSearchText(self, text):
        for checkbox in self.checkBoxes:
            if not text:
                highlighted = False
            else:
                highlighted = checkbox.text().lower().find(text.lower()) != -1

            self.setCheckboxHighlighted(highlighted, checkbox)

    def setCheckboxHighlighted(self, highlighted, checkbox):
        if highlighted:
            checkbox.setStyleSheet(
                f"background: {self._highlightStylesheetColor}; color: black"
            )
            self.scrollArea.ensureWidgetVisible(checkbox)
        else:
            checkbox.setStyleSheet("")

    def onDelClicked(self):
        button = self.sender()
        button.checkbox.setChecked(False)
        self.sigDelClicked.emit(button.colname, button._layout)

    def toggled_cb(self, checked):
        for checkbox in self.checkBoxes:
            if not checked:
                self.checkedState[checkbox] = checkbox.isChecked()
                checkbox.setChecked(False)
            else:
                checkbox.setChecked(self.checkedState[checkbox])

    def checkFavouriteFuncs(self, checked=True, isZstack=False):
        self.doNotWarn = True
        if self._parent is not None:
            self._parent.doNotWarn = True
        for checkBox in self.checkBoxes:
            checkBox.setChecked(False)
            for favourite_func in self.favourite_funcs:
                func_name = checkBox.text()
                if func_name.endswith(favourite_func):
                    checkBox.setChecked(True)
                    break
        self.doNotWarn = False
        if self._parent is not None:
            self._parent.doNotWarn = False

    def checkAll(self, button, checked):
        if self._parent is not None:
            self._parent.doNotWarn = True
        for checkBox in self.checkBoxes:
            checkBox.setChecked(checked)
        if self._parent is not None:
            self._parent.doNotWarn = False

    def showInfo(self, checked=False):
        info_txt = self.sender().info
        msg = myMessageBox()
        msg.setWidth(600)
        msg.setIcon()
        msg.setWindowTitle(f"{self.sender().colname} info")
        msg.addText(info_txt)
        msg.addButton("   Ok   ")
        msg.exec_()

    def show(self):
        super().show()
        fw = self.inner_layout.contentsRect().width()
        sw = self.scrollArea.verticalScrollBar().sizeHint().width()
        self.minWidth = fw + sw


class channelMetricsQGBox(QGroupBox):
    sigDelClicked = Signal(str, object)
    sigCheckboxToggled = Signal(object)

    def __init__(
        self,
        isZstack,
        chName,
        isSegm3D,
        is_concat=False,
        posData=None,
        favourite_funcs=None,
    ):
        QGroupBox.__init__(self)

        self.doNotWarn = False
        self.is_concat = is_concat
        isManualBackgrPresent = False
        if posData is not None:
            if posData.manualBackgroundLab is not None:
                isManualBackgrPresent = True

        layout = QVBoxLayout()
        metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
            isZstack,
            chName,
            isSegm3D=isSegm3D,
            isManualBackgrPresent=isManualBackgrPresent,
        )

        metricsQGBox = _metricsQGBox(
            metrics_desc,
            "Standard measurements",
            favourite_funcs=favourite_funcs,
            parent=self,
            isZstack=isZstack,
        )
        self.metricsQGBox = metricsQGBox

        bkgrValsQGBox = _metricsQGBox(
            bkgr_val_desc,
            "Background values",
            favourite_funcs=favourite_funcs,
            parent=self,
            isZstack=isZstack,
        )
        self.bkgrValsQGBox = bkgrValsQGBox

        self.checkBoxes = metricsQGBox.checkBoxes.copy()
        self.checkBoxes.extend(bkgrValsQGBox.checkBoxes)

        self.uncheckAndDisableDataPrepIfPosNotPrepped(posData)

        self.groupboxes = [metricsQGBox, bkgrValsQGBox]

        for checkbox in metricsQGBox.checkBoxes:
            checkbox.toggled.connect(self.standardMetricToggled)
            self.standardMetricToggled(checkbox.isChecked(), checkbox=checkbox)

        for bkgrCheckbox in bkgrValsQGBox.checkBoxes:
            bkgrCheckbox.toggled.connect(self.backgroundMetricToggled)

        layout.addWidget(metricsQGBox)
        layout.addWidget(bkgrValsQGBox)

        items = measurements.custom_metrics_desc(
            isZstack, chName, posData=posData, isSegm3D=isSegm3D, return_combine=True
        )
        custom_metrics_desc, combine_metrics_desc = items

        if custom_metrics_desc:
            customMetricsQGBox = _metricsQGBox(
                custom_metrics_desc,
                "Custom measurements",
                delButtonMetricsDesc=combine_metrics_desc,
                favourite_funcs=favourite_funcs,
                isZstack=isZstack,
            )
            layout.addWidget(customMetricsQGBox)
            self.checkBoxes.extend(customMetricsQGBox.checkBoxes)
            customMetricsQGBox.sigDelClicked.connect(self.onDelClicked)
            self.customMetricsQGBox = customMetricsQGBox

        self.calcForEachZsliceToggle = None
        if isZstack:
            buttonsLayout = QHBoxLayout()
            self.calcForEachZsliceToggle = Toggle()
            tooltip = (
                "Calculate the selected measurements for each z-slice.\n\n"
                "The measurements will be saved in the column with name\n"
                "ending with `_zsliceN` where N is the z-slice number\n"
                "(starting from 0)."
            )
            calcForEachZsliceLabel = QClickableLabel("Calculate for each z-slice")
            calcForEachZsliceLabel.setToolTip(tooltip)
            self.calcForEachZsliceToggle.setToolTip(tooltip)
            buttonsLayout.addWidget(self.calcForEachZsliceToggle)
            buttonsLayout.addWidget(calcForEachZsliceLabel)
            buttonsLayout.addStretch(1)
            layout.addLayout(buttonsLayout)
            calcForEachZsliceLabel.clicked.connect(
                partial(
                    self.toggleCalcForEachZslice, toggle=self.calcForEachZsliceToggle
                )
            )

        self.setTitle(f"{chName} metrics")
        self.setCheckable(True)
        self.setLayout(layout)

    def toggleCalcForEachZslice(self, label, toggle=None):
        if toggle is None:
            toggle = self.calcForEachZsliceToggle

        toggle.setChecked(not toggle.isChecked())

    def isCalcForEachZsliceRequested(self):
        if self.calcForEachZsliceToggle is None:
            return False

        return self.calcForEachZsliceToggle.isChecked()

    def uncheckAndDisableDataPrepIfPosNotPrepped(self, posData):
        # Uncheck and disable dataprep metrics if pos is not prepped
        if posData is None:
            return

        if posData.isBkgrROIpresent():
            return

        for checkbox in self.checkBoxes:
            if checkbox.text().find("dataPrep") == -1:
                continue

            checkbox.setChecked(False)
            checkbox.isDataPrepDisabled = True

    def _warnDataPrepCannotBeChecked(self):
        if self.doNotWarn:
            return
        txt = html_utils.paragraph("""
            <b>Data prep measurements cannot be saved</b> because you did 
            not select any background ROI at the data prep step.<br><br>

            You can read more details about data prep metrics by clicking 
            on the info button besides the measurement's name.<br><br>

            Thank you for you patience!
        """)
        msg = myMessageBox(showCentered=False)
        msg.warning(self, "Metric cannot be saved", txt)

    def standardMetricToggled(self, checked, checkbox=None):
        """Method called when a check-box is toggled. It performs the following
        actions:
            1. If the user try to check a data prep measurement, such as
            dataPrep_amount, and this cannot be saved (checkbox has the attr
            `isDataPrepDisabled`) then it warns and explains why it cannot be saved
            2. Make sure that background value median is checked if the user
            requires amount or concentration metric.
            3. Do not allow unchecking background value median and explain why.

        Parameters
        ----------
        checked : bool
            State of the checkbox toggled
        checkbox : QtWidgets.QCheckBox, optional
            The checkbox that has been toggled. Default is None. If None
            use `self.sender()`
        """
        if self.is_concat:
            return

        if checkbox is None:
            checkbox = self.sender()

        if hasattr(checkbox, "isDataPrepDisabled"):
            # Warn that user cannot check data prep metrics and uncheck it
            if not checkbox.isChecked():
                return
            checkbox.setChecked(False)
            self._warnDataPrepCannotBeChecked()
            return

        self.sigCheckboxToggled.emit(checkbox)
        if checkbox.text().find("amount_") == -1:
            return
        pattern = r"amount_([A-Za-z]+)(_?[A-Za-z0-9]*)"
        repl = r"\g<1>_bkgrVal_median\g<2>"
        bkgrValMetric = s1 = re.sub(pattern, repl, checkbox.text())
        for bkgrCheckbox in self.groupboxes[1].checkBoxes:
            if bkgrCheckbox.text() == bkgrValMetric:
                break
        else:
            # Make sure to not check for similarly named custom metrics
            return

        if checked:
            bkgrCheckbox.setChecked(True)
            bkgrCheckbox.isRequired = True
        else:
            bkgrCheckbox.setDisabled(False)
            bkgrCheckbox.isRequired = False

    def backgroundMetricToggled(self, checked):
        """Method called when a checkbox of a background metric is toggled.
        Check if the background value is required and explain why it cannot be
        unchecked.

        Parameters
        ----------
        checked : bool
            State of the checkbox toggled
        """
        if self.is_concat:
            return

        checkbox = self.sender()
        if not hasattr(checkbox, "isRequired"):
            return

        if not checkbox.isRequired:
            return

        if checkbox.isChecked():
            return

        if self.doNotWarn:
            return

        checkbox.setChecked(True)
        txt = html_utils.paragraph("""
            <b>This background value cannot be unchecked</b> because it is required 
            by the <code>_amount</code> and <code>_concentration</code> measurements 
            that you requested to save.<br><br>

            Thank you for you patience!
        """)
        msg = myMessageBox(showCentered=False)
        msg.warning(self, "Background value required", txt)

    def onDelClicked(self, colname_to_del, hlayout):
        self.sigDelClicked.emit(colname_to_del, hlayout)

    def checkFavouriteFuncs(self):
        self.doNotWarn = True
        for groupbox in self.groupboxes:
            groupbox.checkFavouriteFuncs()
        self.doNotWarn = False


class PixelSizeGroupbox(QGroupBox):
    sigValueChanged = Signal(float, float, float)
    sigReset = Signal()

    def __init__(self, parent=None):
        super().__init__("Pixel size", parent)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel("Pixel width (μm): ")
        self.pixelWidthWidget = FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.pixelWidthWidget, row, 1)

        row += 1
        label = QLabel("Pixel height (μm): ")
        self.pixelHeightWidget = FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.pixelHeightWidget, row, 1)

        row += 1
        label = QLabel("Voxel depth (μm): ")
        self.voxelDepthWidget = FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.voxelDepthWidget, row, 1)

        row += 1
        resetButton = reloadPushButton("Reset")
        mainLayout.addWidget(resetButton, row, 1, alignment=Qt.AlignRight)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        mainLayout.setColumnStretch(0, 0)
        mainLayout.setColumnStretch(1, 1)

        self.setLayout(mainLayout)

        self.pixelWidthWidget.valueChanged.connect(self.emitValueChanged)
        self.pixelHeightWidget.valueChanged.connect(self.emitValueChanged)
        self.voxelDepthWidget.valueChanged.connect(self.emitValueChanged)
        resetButton.clicked.connect(self.emitReset)

    def emitReset(self):
        self.sigReset.emit()

    def emitValueChanged(self, value):
        PhysicalSizeX = self.pixelWidthWidget.value()
        PhysicalSizeY = self.pixelHeightWidget.value()
        PhysicalSizeZ = self.voxelDepthWidget.value()
        self.sigValueChanged.emit(PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)


class objPropsQGBox(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, "Properties", parent)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel("Object ID: ")
        self.idSB = IntLineEdit()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.idSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        self.notExistingIDLabel = QLabel()
        self.notExistingIDLabel.setStyleSheet("font-size:11px; color: rgb(255, 0, 0);")
        mainLayout.addWidget(
            self.notExistingIDLabel, row, 0, 1, 2, alignment=Qt.AlignCenter
        )

        row += 1
        label = QLabel("Area (pixel): ")
        self.cellAreaPxlSB = IntLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellAreaPxlSB, row, 1)

        row += 1
        label = QLabel("Area (<span>&#181;</span>m<sup>2</sup>): ")
        self.cellAreaUm2DSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellAreaUm2DSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        label = QLabel("Rotational volume (voxel): ")
        self.cellVolVoxSB = IntLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolVoxSB, row, 1)

        row += 1
        label = QLabel("3D volume (voxel): ")
        self.cellVolVox3D_SB = IntLineEdit(readOnly=True)
        self.cellVolVox3D_SB.label = label
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolVox3D_SB, row, 1)

        row += 1
        label = QLabel("Rotational volume (fl): ")
        self.cellVolFlDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolFlDSB, row, 1)

        row += 1
        label = QLabel("3D volume (fl): ")
        self.cellVolFl3D_DSB = FloatLineEdit(readOnly=True)
        self.cellVolFl3D_DSB.label = label
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolFl3D_DSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        label = QLabel("Solidity: ")
        self.solidityDSB = FloatLineEdit(readOnly=True)
        self.solidityDSB.setMaximum(1)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.solidityDSB, row, 1)

        row += 1
        label = QLabel("Elongation: ")
        self.elongationDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.elongationDSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        propsNames = measurements.get_props_names()[1:]
        self.additionalPropsCombobox = QComboBox()
        self.additionalPropsCombobox.addItems(propsNames)
        self.additionalPropsCombobox.indicator = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(self.additionalPropsCombobox, row, 0)
        mainLayout.addWidget(self.additionalPropsCombobox.indicator, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        mainLayout.setColumnStretch(0, 0)
        mainLayout.setColumnStretch(1, 1)

        self.setLayout(mainLayout)


class objIntesityMeasurQGBox(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, "Intensity measurements", parent)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel("Raw intensity measurements")

        row += 1
        label = QLabel("Channel: ")
        self.channelCombobox = QComboBox()
        self.channelCombobox.addItem("placeholderlong")
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.channelCombobox, row, 1)

        row += 1
        label = QLabel("Minimum: ")
        self.minimumDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.minimumDSB, row, 1)

        row += 1
        label = QLabel("Maximum: ")
        self.maximumDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.maximumDSB, row, 1)

        row += 1
        label = QLabel("Mean: ")
        self.meanDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.meanDSB, row, 1)

        row += 1
        label = QLabel("Median: ")
        self.medianDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.medianDSB, row, 1)

        row += 1
        metricsDesc = measurements._get_metrics_names()
        metricsFunc, _ = measurements.standard_metrics_func()
        items = list(set([metricsDesc[key] for key in metricsFunc.keys()]))
        items.append("Concentration")
        items.sort()
        nameFuncDict = {}
        for name, desc in metricsDesc.items():
            if name.find("_dataPrepBkgr") != -1 or name.find("_manualBkgr") != -1:
                # Skip dataPrepBkgr and manualBkgr since in the dock widget
                # we display only autoBkgr metrics
                continue
            if name.startswith("concentration_"):
                # We use amount function because dividing by volume is taken
                # care in the GUI
                name = "amount_autoBkgr"
            nameFuncDict[desc] = metricsFunc[name]

        funcionCombobox = QComboBox()
        funcionCombobox.addItems(items)
        self.additionalMeasCombobox = funcionCombobox
        self.additionalMeasCombobox.indicator = FloatLineEdit(readOnly=True)
        self.additionalMeasCombobox.functions = nameFuncDict
        mainLayout.addWidget(funcionCombobox, row, 0)
        mainLayout.addWidget(self.additionalMeasCombobox.indicator, row, 1)

        self.setLayout(mainLayout)

    def addChannels(self, channels):
        self.channelCombobox.clear()
        self.channelCombobox.addItems(channels)


class SetMeasurementsGroupBox(QGroupBox):
    def __init__(
        self,
        title,
        itemsText,
        checkable=True,
        itemsInfo=None,
        lastSelection=None,
        itemsInfoUrls=None,
        parent=None,
    ):
        super().__init__(parent)

        if itemsInfo is None:
            itemsInfo = {}

        if itemsInfo is None:
            itemsInfoUrls = {}

        highlightRgba = _palettes._highlight_rgba()
        r, g, b, a = highlightRgba
        self._highlightStylesheetColor = f"rgb({r}, {g}, {b})"

        self.setTitle(title)
        self.setCheckable(checkable)

        mainLayout = QVBoxLayout()

        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollAreaLayout = QVBoxLayout()
        scrollAreaWidget = QWidget()
        self.scrollAreaWidget = scrollAreaWidget
        self.scrollAreaLayout = scrollAreaLayout

        self.checkboxes = {}
        for text in itemsText:
            rowLayout = QHBoxLayout()
            infoText = itemsInfo.get(text)
            infoUrl = itemsInfoUrls.get(text)
            if infoText is not None or infoUrl is not None:
                infoButton = infoPushButton()
                infoButton.setCursor(Qt.WhatsThisCursor)
                rowLayout.addWidget(infoButton)

            if infoText is not None:
                infoButton.itemText = text
                infoButton.infoText = infoText
                infoButton.clicked.connect(self.showInfo)

            if infoUrl is not None:
                infoButton.itemText = text
                infoButton.infoUrl = infoUrl
                infoButton.clicked.connect(self.openInfoUrl)

            checkbox = QCheckBox(text)
            checkbox.setParent(self.scrollAreaWidget)
            checkbox.setChecked(True)
            rowLayout.addWidget(checkbox)
            rowLayout.addStretch(1)

            self.checkboxes[text] = checkbox

            scrollAreaLayout.addLayout(rowLayout)

        scrollAreaLayout.addStretch(1)

        scrollAreaWidget.setLayout(scrollAreaLayout)
        scrollArea.setWidget(scrollAreaWidget)
        self.scrollArea = scrollArea

        buttonsLayout = QHBoxLayout()
        self.selectAllButton = selectAllPushButton()
        self.selectAllButton.sigClicked.connect(self.setCheckedAll)

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(self.selectAllButton)
        self.buttonsLayout = buttonsLayout

        if lastSelection is not None:
            self.lastSelection = lastSelection
            self.loadLastSelButton = reloadPushButton("  Load last selection...  ")
            self.loadLastSelButton.clicked.connect(self.loadLastSelection)
            buttonsLayout.addWidget(self.loadLastSelButton)

        mainLayout.addWidget(scrollArea)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def openInfoUrl(self):
        url = self.sender().infoUrl
        QDesktopServices.openUrl(QUrl(url))
        # import webbrowser
        # url = self.sender().infoUrl
        # webbrowser.open(url)

    def getWidthNoScrollBarNeeded(self):
        width = (
            self.scrollArea.verticalScrollBar().sizeHint().width()
            # self.scrollAreaLayout.contentsRect().width()
            + self.scrollAreaWidget.sizeHint().width()
            + 30
        )
        buttonsWidth = 0
        for i in range(self.buttonsLayout.count()):
            widget = self.buttonsLayout.itemAt(i).widget()
            if not isinstance(widget, QPushButton):
                continue
            buttonsWidth += widget.sizeHint().width() + 16
        largerWidth = max(width, buttonsWidth)
        return largerWidth

    def resizeWidthNoScrollBarNeeded(self):
        width = self.getWidthNoScrollBarNeeded()
        self.setMinimumWidth(width)
        # self.setFixedWidth(width)

    def loadLastSelection(self):
        for text, checkbox in self.checkboxes.items():
            checked = self.lastSelection.get(text, False)
            checkbox.setChecked(checked)

    def showInfo(self):
        infoText = self.sender().infoText
        itemText = self.sender().itemText

        title = f"{itemText} description"
        msg = myMessageBox()
        msg.setWidth(int(self.screen().size().width() / 2))
        msg.information(self, title, infoText)

    def setCheckedAll(self, button, checked):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(checked)

    def highlightCheckboxesFromSearchText(self, text):
        for checkbox in self.checkboxes.values():
            if not text:
                highlighted = False
            else:
                highlighted = checkbox.text().lower().find(text.lower()) != -1

            self.setCheckboxHighlighted(highlighted, checkbox)

    def setCheckboxHighlighted(self, highlighted, checkbox):
        if highlighted:
            checkbox.setStyleSheet(
                f"background: {self._highlightStylesheetColor}; color: black"
            )
            self.scrollArea.ensureWidgetVisible(checkbox)
        else:
            checkbox.setStyleSheet("")

# Cross-module imports (deferred to avoid import cycles)
from .dialogs import (
    myMessageBox,
)
from .inputs import (
    FloatLineEdit,
    IntLineEdit,
    QClickableLabel,
)
from .panels import (
    Toggle,
)

