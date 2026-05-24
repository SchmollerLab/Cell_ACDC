"""Canvas widgets: scrollbars."""

"""GUI widgets: canvas."""

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

class MouseCursor(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._x = None
        self._y = None
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event) -> None:
        self.move(event.pos())
        self.update()
        return super().mouseMoveEvent(event)

    # def drawAtPos(self, x, y):
    #     self._x = x
    #     self._y = y
    #     self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        # p.setPen(QPen(QColor(0,0,0)))
        # p.setBrush(QBrush(QColor(70,70,70,200)))
        p.drawLine(0, 0, 200, 0)
        p.end()


class labelledQScrollbar(ScrollBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._label = None

    def setLabel(self, label):
        self._label = label

    def updateLabel(self):
        if self._label is not None:
            position = self.sliderPosition()
            s = self._label.text()
            s = re.sub(r"(\d+)/(\d+)", rf"{position + 1:02}/\2", s)
            self._label.setText(s)

    def setSliderPosition(self, position):
        QScrollBar.setSliderPosition(self, position)
        self.updateLabel()

    def setValue(self, value):
        QScrollBar.setValue(self, value)
        self.updateLabel()


class navigateQScrollBar(ScrollBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._disableCustomPressEvent = False
        self.signal_slot_mapper = {}

    def disableCustomPressEvent(self):
        self._disableCustomPressEvent = True

    def enableCustomPressEvent(self):
        self._disableCustomPressEvent = False

    def setAbsoluteMaximum(self, absoluteMaximum):
        self._absoluteMaximum = absoluteMaximum

    def absoluteMaximum(self):
        return self._absoluteMaximum

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.maximum() == self._absoluteMaximum:
            return

        if self._disableCustomPressEvent:
            return

    def setValueNoSignal(self, value):
        for signal_name, slot in self.signal_slot_mapper.items():
            signal = getattr(self, signal_name)
            try:
                signal.disconnect()
            except Exception as e:
                pass

        self.setSliderPosition(value)
        self.connectEvents(self.signal_slot_mapper)

    def connectEvents(self, signal_slot_mapper: dict):
        self.signal_slot_mapper = signal_slot_mapper
        for signal_name, slot in signal_slot_mapper.items():
            signal = getattr(self, signal_name)
            try:
                signal.disconnect()
            except Exception as e:
                pass
            signal.connect(slot)


class linkedQScrollbar(ScrollBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._linkedScrollBar = None

    def linkScrollBar(self, scrollbar):
        self._linkedScrollBar = scrollbar
        scrollbar.setSliderPosition(self.sliderPosition())

    def unlinkScrollBar(self):
        self._linkedScrollBar = None

    def setSliderPosition(self, position):
        QScrollBar.setSliderPosition(self, position)
        if self._linkedScrollBar is not None:
            self._linkedScrollBar.setSliderPosition(position)

    def setMaximum(self, max):
        QScrollBar.setMaximum(self, max)
        if self._linkedScrollBar is not None:
            self._linkedScrollBar.setMaximum(max)


class sliderWithSpinBox(QWidget):
    sigValueChange = Signal(object)
    valueChanged = Signal(object)
    editingFinished = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        layout = QGridLayout()

        title = kwargs.get("title")
        row = 0
        col = 0
        if title is not None:
            titleLabel = QLabel(self)
            titleLabel.setText(title)
            loc = kwargs.get("title_loc", "top")
            if loc == "top":
                layout.addWidget(titleLabel, 0, col, alignment=Qt.AlignLeft)
            elif loc == "in_line":
                row = -1
                col = 1
                layout.addWidget(titleLabel, 0, 0, alignment=Qt.AlignLeft)
                layout.setColumnStretch(0, 0)

        self._normalize = False
        normalize = kwargs.get("normalize")
        if normalize is not None and normalize:
            self._normalize = True
            self._isFloat = True

        self._isFloat = False
        isFloat = kwargs.get("isFloat")
        if isFloat is not None and isFloat:
            self._isFloat = True

        self.slider = QSlider(Qt.Horizontal, self)

        if self._normalize or self._isFloat:
            self.spinBox = DoubleSpinBox(self)
        else:
            self.spinBox = SpinBox(self)
        self.spinBox.setAlignment(Qt.AlignCenter)
        self.spinBox.setMaximum(2**31 - 1)

        maximum_on_label = kwargs.get("maximum_on_label")
        spinbox_loc = kwargs.get("spinbox_loc", "right")
        if spinbox_loc == "right":
            spinbox_col = col + 1
            slider_col = col
            if maximum_on_label is not None:
                maximum_on_label_col = spinbox_col + 1
        elif spinbox_loc == "left":
            spinbox_col = col
            slider_col = col + 1
            if maximum_on_label is not None:
                maximum_on_label_col = spinbox_col + 1
                slider_col += 1

        if maximum_on_label is not None:
            self.labelMaximum = QLabel()
            layout.addWidget(self.labelMaximum, row + 1, maximum_on_label_col)
        layout.addWidget(self.slider, row + 1, slider_col)
        layout.addWidget(self.spinBox, row + 1, spinbox_col)

        if title is not None:
            layout.setRowStretch(0, 1)
        layout.setRowStretch(row + 1, 1)
        layout.setColumnStretch(slider_col, 6)
        layout.setColumnStretch(spinbox_col, 1)

        self._layout = layout
        self.lastCol = col + 1
        self.sliderCol = slider_col

        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.slider.sliderReleased.connect(self.onEditingFinished)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.spinBox.editingFinished.connect(self.onEditingFinished)

        layout.setContentsMargins(5, 0, 5, 0)

        self.setLayout(layout)

        if maximum_on_label is not None:
            self.setMaximum(maximum_on_label)
            self.labelMaximum.setText(f"/{maximum_on_label}")

    def onEditingFinished(self):
        self.editingFinished.emit()

    def maximum(self):
        return self.slider.maximum()

    def minimum(self):
        return self.slider.minimum()

    def setValue(self, value, emitSignal=False):
        valueInt = value
        if self._normalize:
            valueInt = int(value * self.slider.maximum())
        elif self._isFloat:
            valueInt = int(value)

        self.spinBox.valueChanged.disconnect()
        self.spinBox.setValue(value)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)

        self.slider.valueChanged.disconnect()
        if valueInt > self.slider.maximum():
            self.slider.setMaximum(valueInt)
        self.slider.setValue(valueInt)
        self.slider.valueChanged.connect(self.sliderValueChanged)

        if emitSignal:
            self.sigValueChange.emit(self.value())
            self.valueChanged.emit(self.value())

    def setMaximum(self, max, including_spinbox=False):
        self.slider.setMaximum(max)
        if including_spinbox:
            self.spinBox.setMaximum(max)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setMinimum(self, min, including_spinbox=False):
        self.slider.setMinimum(min)
        if including_spinbox:
            self.spinBox.setMinimum(min)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setDecimals(self, decimals):
        self.spinBox.setDecimals(decimals)

    def setTickPosition(self, position):
        self.slider.setTickPosition(position)

    def setTickInterval(self, interval):
        self.slider.setTickInterval(interval)

    def sliderValueChanged(self, val):
        self.spinBox.valueChanged.disconnect()
        if self._normalize:
            valF = val / self.slider.maximum()
            self.spinBox.setValue(valF)
        else:
            self.spinBox.setValue(val)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.sigValueChange.emit(self.value())
        self.valueChanged.emit(self.value())

    def spinboxValueChanged(self, val):
        if self._normalize:
            val = int(val * self.slider.maximum())
        elif self._isFloat:
            val = int(val)

        self.slider.valueChanged.disconnect()
        self.slider.setValue(val)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.sigValueChange.emit(self.value())
        self.valueChanged.emit(self.value())

    def value(self):
        return self.spinBox.value()

    def setDisabled(self, disabled) -> None:
        self.slider.setDisabled(disabled)
        self.spinBox.setDisabled(disabled)


class ScrollBarWithNumericControl(QWidget):
    sigValueChanged = Signal(int)
    sigMaxProjToggled = Signal(bool, object)

    def __init__(
        self,
        orientation=Qt.Horizontal,
        add_max_proj_button=False,
        parent=None,
        labelText="",
    ) -> None:
        super().__init__(parent)

        self._slot = None

        layout = QHBoxLayout()
        self.scrollbar = QScrollBar(orientation, self)
        self.spinbox = QSpinBox(self)
        self.maxLabel = QLabel(self)
        idx = 0
        if labelText:
            layout.addWidget(QLabel(labelText))
            layout.setStretch(idx, 0)
            idx += 1

        layout.addWidget(self.spinbox)
        layout.setStretch(idx, 0)
        idx += 1

        layout.addWidget(self.maxLabel)
        layout.setStretch(idx, 0)
        idx += 1

        layout.addWidget(self.scrollbar)
        layout.setStretch(idx, 1)
        idx += 1

        if add_max_proj_button:
            self.maxProjCheckbox = QCheckBox("MAX")
            self.scrollbar.maxProjCheckbox = self.maxProjCheckbox
            layout.addWidget(self.maxProjCheckbox)
            layout.setStretch(idx, 0)

        layout.setContentsMargins(5, 0, 5, 0)

        self.setLayout(layout)

        self.spinbox.valueChanged.connect(self.spinboxValueChanged)
        self.scrollbar.valueChanged.connect(self.scrollbarValueChanged)

        if add_max_proj_button:
            self.maxProjCheckbox.toggled.connect(self.maxProjToggled)

    def connectValueChanged(self, slot):
        self.sigValueChanged.connect(slot)
        self._slot = slot

    def setValueNoSignal(self, value):
        if self._slot is None:
            return
        self.sigValueChanged.disconnect()
        self.setValue(value)
        self.sigValueChanged.connect(self._slot)

    def maxProjToggled(self, checked):
        self.scrollbar.setDisabled(checked)
        self.sigMaxProjToggled.emit(checked, self)

    def showEvent(self, event) -> None:
        super().showEvent(event)

        self.scrollbar.setMinimumHeight(self.spinbox.height())

    def setMaximum(self, maximum):
        self.maxLabel.setText(f"/{maximum}")
        self.scrollbar.setMaximum(maximum)
        self.spinbox.setMaximum(maximum)

    def setMinimum(self, minumum):
        self.scrollbar.setMinimum(minumum)
        self.spinbox.setMinimum(minumum)

    def spinboxValueChanged(self, value):
        self.scrollbar.setValue(value)

    def scrollbarValueChanged(self, value):
        self.spinbox.setValue(value)
        self.sigValueChanged.emit(value)

    def setValue(self, value):
        self.scrollbar.setValue(value)

    def value(self):
        return self.scrollbar.value()

    def maximum(self):
        return self.scrollbar.maximum()

# Cross-module imports (deferred to avoid import cycles)
from ..controls.inputs import (
    DoubleSpinBox,
    SpinBox,
)

