"""Toolbars: _base."""

"""GUI widgets: toolbars."""

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

class ToolBarSeparator:
    def __init__(self, width=5, toolbar: QToolBar = None):
        self._parts = (
            QHWidgetSpacer(width=width),
            QVLine(),
            QHWidgetSpacer(width=width),
        )
        self._actions = []
        self._toolbar = None
        if toolbar is not None:
            self.addToToolbar(toolbar)

    def addToToolbar(self, toolbar):
        self._toolbar = toolbar
        for part in self._parts:
            action = toolbar.addWidget(part)
            self._actions.append(action)

    def removeFromToolbar(self):
        if self._toolbar is None:
            return

        for action in self._actions:
            self._toolbar.removeAction(action)


class ToolBar(QToolBar):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.widgetsWithShortcut = {}

        for child in self.children():
            if child.objectName() == "qt_toolbar_ext_button":
                self.extendButton = child
                self.extendButton.setIcon(QIcon(":expand.svg"))
                break

    def addSeparator(self, width=5):
        separator = ToolBarSeparator(width=width, toolbar=self)
        return separator

    def removeSeparator(self, separator):
        separator.removeFromToolbar()

    def addSpinBox(self, label=""):
        spinbox = SpinBox(disableKeyPress=True)
        if label:
            spinbox.label = QLabel(label)
            spinbox.labelAction = self.addWidget(spinbox.label)

        spinbox.action = self.addWidget(spinbox)
        return spinbox

    def addButton(self, icon_str: str, text="", checkable=False):
        action = QAction(QIcon(icon_str), text, self)
        action.setCheckable(checkable)
        self.addAction(action)
        return action

    def addComboBox(self, items=None, label=""):
        combobox = ComboBox()

        if items is not None:
            combobox.addItems(items)

        if label:
            combobox.label = QLabel(label)
            combobox.labelAction = self.addWidget(combobox.label)

        combobox.action = self.addWidget(combobox)
        return combobox

    def addLabel(self, text=""):
        label = QLabel(text)
        label.action = self.addWidget(label)
        return label

    def addCheckBox(self, text="", checked=False):
        checkbox = QCheckBox(text)
        checkbox.setChecked(checked)
        checkbox.action = self.addWidget(checkbox)
        return checkbox


class rightClickToolButton(QToolButton):
    sigRightClick = Signal(object)
    sigLeftClick = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            self.sigLeftClick.emit(self, event)
        elif event.button() == Qt.MouseButton.RightButton:
            self.sigRightClick.emit(event)


class ToolButtonCustomColor(rightClickToolButton):
    def __init__(self, symbol, color="r", parent=None):
        super().__init__(parent=parent)
        if not isinstance(color, QColor):
            color = pg.mkColor(color)
        self.symbol = symbol
        self.setColor(color)

    def setColor(self, color):
        self.penColor = color
        self.brushColor = [0, 0, 0, 100]
        self.brushColor[:3] = color.getRgb()[:3]

    def updateSymbol(self, symbol, update=True):
        self.symbol = symbol
        if not update:
            return
        self.update()

    def updateColor(self, color, update=True):
        self.setColor(color)
        if not update:
            return
        self.update()

    def updateIcon(self, symbol, color):
        self.updateSymbol(symbol)
        self.updateColor(color)
        self.update()

    def paintEvent(self, event):
        QToolButton.paintEvent(self, event)
        p = QPainter(self)
        w, h = self.width(), self.height()
        sf = 0.6
        p.scale(w * sf, h * sf)
        p.translate(0.5 / sf, 0.5 / sf)
        symbol = pg.graphicsItems.ScatterPlotItem.Symbols[self.symbol]
        pen = pg.mkPen(color=self.penColor, width=2)
        brush = pg.mkBrush(color=self.brushColor)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(pen)
            p.setBrush(brush)
            p.drawPath(symbol)
        except Exception as e:
            traceback.print_exc()
        finally:
            p.end()


class GradientToolButton(rightClickToolButton):
    def __init__(self, colors=((255, 0, 0),), parent=None):
        super().__init__(parent=parent)
        self._qcolors = [pg.mkColor(c) for c in colors]
        if len(self._qcolors) < 2:
            self._qcolors.append(self._qcolors[0])

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = pg.mkPen(color=self._qcolors[-1], width=2)

        pad = 7

        rect = self.rect().adjusted(pad, pad, -pad, -pad)  # A little padding

        # Gradient: bottom to top
        gradient = QLinearGradient(QPointF(rect.bottomLeft()), QPointF(rect.topLeft()))

        # Set color stops evenly distributed
        num_colors = len(self._qcolors)
        for i, color in enumerate(self._qcolors):
            gradient.setColorAt(i / (num_colors - 1), color)

        if not self.isChecked():
            painter.setOpacity(0.4)

        painter.setBrush(gradient)
        painter.setPen(pen)
        painter.drawRect(rect)

        painter.end()


class PointsLayerToolButton(ToolButtonCustomColor):
    sigEditAppearance = Signal(object)
    sigShowIdsToggled = Signal(object, bool)
    sigRemove = Signal(object)

    def __init__(self, symbol, color="r", parent=None):
        super().__init__(symbol, color=color, parent=parent)
        self.sigRightClick.connect(self.showContextMenu)

    def showContextMenu(self, event):
        contextMenu = QMenu(self)
        contextMenu.addSeparator()

        editAction = QAction("Edit points appearance...")
        editAction.triggered.connect(self.editAppearance)
        contextMenu.addAction(editAction)

        removeAction = QAction("Remove points")
        removeAction.triggered.connect(self.emitRemove)
        contextMenu.addAction(removeAction)

        showIdsAction = QAction("Show point ids")
        showIdsAction.setCheckable(True)
        showIdsAction.setChecked(True)
        contextMenu.addAction(showIdsAction)
        showIdsAction.toggled.connect(self.emitShowIdsToggled)

        contextMenu.exec(event.globalPos())

    def emitRemove(self):
        self.sigRemove.emit(self)

    def emitShowIdsToggled(self, checked):
        self.sigShowIdsToggled.emit(self, checked)

    def editAppearance(self):
        self.sigEditAppearance.emit(self)


class customAnnotToolButton(ToolButtonCustomColor):
    sigRemoveAction = Signal(object)
    sigKeepActiveAction = Signal(object)
    sigModifyAction = Signal(object)
    sigHideAction = Signal(object)

    def __init__(
        self, symbol, color, keepToolActive=True, parent=None, isHideChecked=True
    ):
        super().__init__(symbol, color=color, parent=parent)
        self.symbol = symbol
        self.keepToolActive = keepToolActive
        self.isHideChecked = isHideChecked
        self.sigRightClick.connect(self.showContextMenu)

    def showContextMenu(self, event):
        contextMenu = QMenu(self)
        contextMenu.addSeparator()

        removeAction = QAction("Remove annotation")
        removeAction.triggered.connect(self.removeAction)
        contextMenu.addAction(removeAction)

        editAction = QAction("Modify annotation parameters...")
        editAction.triggered.connect(self.modifyAction)
        contextMenu.addAction(editAction)

        hideAction = QAction("Hide annotations")
        hideAction.setCheckable(True)
        hideAction.setChecked(self.isHideChecked)
        hideAction.triggered.connect(self.hideAction)
        contextMenu.addAction(hideAction)

        keepActiveAction = QAction("Keep tool active after using it")
        keepActiveAction.setCheckable(True)
        keepActiveAction.setChecked(self.keepToolActive)
        keepActiveAction.triggered.connect(self.keepToolActiveActionToggled)
        contextMenu.addAction(keepActiveAction)

        contextMenu.exec(event.globalPos())

    def keepToolActiveActionToggled(self, checked):
        self.keepToolActive = checked
        self.sigKeepActiveAction.emit(self)

    def modifyAction(self):
        self.sigModifyAction.emit(self)

    def removeAction(self):
        self.sigRemoveAction.emit(self)

    def hideAction(self, checked):
        self.isHideChecked = checked
        self.sigHideAction.emit(self)


class ToolButtonTextIcon(rightClickToolButton):
    def __init__(self, text="", parent=None):
        super().__init__(parent=parent)
        self._text = text
        self._penColor = _palettes.text_pen_color()

    def setText(self, text):
        self._text = text
        self.update()

    def text(self):
        return self._text

    def paintEvent(self, event):
        QToolButton.paintEvent(self, event)
        p = QPainter(self)

        pen = pg.mkPen(color=self._penColor, width=2)
        p.setPen(pen)

        w, h = self.width(), self.height()
        sf = 0.7
        rect_w = w * sf
        rect_h = h * sf
        x = (w - rect_w) / 2
        y = (h - rect_h) / 2
        rect = QRectF(x, y, rect_w, rect_h)

        font = p.font()
        font.setBold(True)
        font.setPixelSize(int(h / len(self._text)))
        p.setFont(font)

        p.drawText(rect, Qt.AlignCenter, self._text)
        p.end()


class OverlayChannelToolButton(GradientToolButton):
    def __init__(
        self,
        channel_name: str,
        lut_item: myHistogramLUTitem,
        shortcut="0",
        parent=None,
    ):
        super().__init__(colors=lut_item.gradient.getLookupTable(256), parent=parent)
        self._channel_name = channel_name

        lut_item.sigGradientChanged.connect(self.updateColors)

        self.setToolTip(f'Show/hide "{channel_name}" channel\n\nShortcut: {shortcut}')

        self.setCheckable(True)

    def channelName(self):
        return self._channel_name

    def updateColors(self, lut_item):
        colors = lut_item.gradient.getLookupTable(256)
        self._qcolors = [pg.mkColor(c) for c in colors]
        self.update()

    def setVisible(self, visible: bool):
        super().setVisible(visible)
        if not hasattr(self, "action"):
            return

        self.action.setVisible(visible)

# Cross-module imports (deferred to avoid import cycles)
from ..canvas.histogram import (
    myHistogramLUTitem,
)
from ..controls.inputs import (
    ComboBox,
    SpinBox,
)

