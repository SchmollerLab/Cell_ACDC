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

from .. import myutils, measurements, is_mac, is_win, html_utils, is_linux
from .. import printl, settings_folderpath
from .. import colors, config
from .. import html_path
from .. import _palettes
from .. import load
from .. import apps
from .. import plot
from .. import annotate
from .. import urls
from .. import _core, core
from .. import QtScoped
from .. import prompts
from ..acdc_regex import float_regex
from ..config import PREPROCESS_MAPPER
from .. import _base_widgets

from ..components.palette import (  # noqa: E402
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
from ..components.progress import QtHandler, QLog, XStream  # noqa: E402
from ..components.buttons import *  # noqa: E402, F403
from ..components.layout import *  # noqa: E402, F403
from ..components.inputs_basic import *  # noqa: E402, F403
from ..components.path_controls import *  # noqa: E402, F403

from ..components.lists import *  # noqa: E402, F403
from ..components.base import QBaseWindow  # noqa: E402
from ..components.progress import (  # noqa: E402
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


class CopyLostObjectToolbar(ToolBar):
    sigCopyAllObjects = Signal(int, int)

    def __init__(self, *args) -> None:
        super().__init__(*args)

        action = self.addButton(":copyContour_all.svg")
        # action.setShortcut('Alt+C')
        action.keyPressShortcut = KeySequenceFromText("Alt+C")
        action.setToolTip("Copy all lost objects\n\nShortcut: Alt+C")
        self.widgetsWithShortcut["Copy all lost objects"] = action

        action.triggered.connect(self.emitSigCopyAllObjects)

        self.addSeparator()

        self.maxOverlapNumberControl = self.addSpinBox(
            label="Maximum overlap to accept lost object [%]: "
        )
        self.maxOverlapNumberControl.setMinimum(0)
        self.maxOverlapNumberControl.setValue(10)
        tooltip = (
            "Maximum overlap to accept lost object [%]\n\n"
            "If the overlap between the lost object and an object already "
            "existing is greater than this value,\n"
            "the lost object will not be added."
        )
        self.maxOverlapNumberControl.setToolTip(tooltip)
        self.maxOverlapNumberControl.label.setToolTip(tooltip)

        self.addSeparator()

        self.untilFrameNumberControl = self.addSpinBox(
            label="Copy lost object(s) for the next number of frames: "
        )
        self.untilFrameNumberControl.setMinimum(0)
        self.untilFrameNumberControl.setValue(0)

    def emitSigCopyAllObjects(self):
        self.sigCopyAllObjects.emit(
            self.untilFrameNumberControl.value(), self.maxOverlapNumberControl.value()
        )


class DrawClearRegionToolbar(ToolBar):
    def __init__(self, *args) -> None:
        super().__init__(*args)

        group = QButtonGroup()
        group.setExclusive(True)
        self.clearTouchingObjsRadioButton = QRadioButton("Clear all touching objects")
        self.clearOnlyEnclosedObjsRadioButton = QRadioButton(
            "Clear only fully enclosed objects"
        )
        self.clearOnlyEnclosedObjsRadioButton.setChecked(True)
        group.addButton(self.clearTouchingObjsRadioButton)
        group.addButton(self.clearOnlyEnclosedObjsRadioButton)

        self.addWidget(self.clearTouchingObjsRadioButton)
        self.addWidget(self.clearOnlyEnclosedObjsRadioButton)

        self.addSeparator()

        self.numZslicesUpSpinbox = self.addSpinBox(
            label="Num. of z-slices to clear upwards: "
        )
        self.numZslicesUpSpinbox.setMinimum(0)
        self.numZslicesUpSpinbox.setValue(0)

        self.numZslicesDownSpinbox = self.addSpinBox(
            label="Num. of z-slices to clear downwards: "
        )
        self.numZslicesDownSpinbox.setMinimum(0)
        self.numZslicesDownSpinbox.setValue(0)

    def setZslicesControlEnabled(self, enabled, SizeZ=None):
        self.numZslicesUpSpinbox.labelAction.setVisible(enabled)
        self.numZslicesUpSpinbox.action.setVisible(enabled)

        self.numZslicesDownSpinbox.labelAction.setVisible(enabled)
        self.numZslicesDownSpinbox.action.setVisible(enabled)

        if SizeZ is None:
            return

        self.numZslicesUpSpinbox.setMaximum(SizeZ)
        self.numZslicesDownSpinbox.setMaximum(SizeZ)

    def zRange(self, z_slice, SizeZ):
        if z_slice is None:
            zRange = (0, SizeZ)
            return zRange

        numZslicesUp = self.numZslicesUpSpinbox.value()
        numZslicesDown = self.numZslicesDownSpinbox.value()

        zmin = z_slice - numZslicesDown
        zmax = z_slice + numZslicesDown + 1

        zmin = zmin if zmin >= 0 else 0
        zmax = zmax if zmax <= SizeZ else SizeZ

        return (zmin, zmax)


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


class WhitelistIDsToolbar(ToolBar):
    sigWhitelistChanged = Signal(list)
    sigViewOGIDs = Signal(bool)
    sigWhitelistAccepted = Signal(list)
    sigAddNewIDs = Signal(bool)
    sigLoadOGLabs = Signal()
    sigTrackOGagainstPreviousFrame = Signal(bool)

    def __init__(self, addNewIDToggleState, *args) -> None:
        super().__init__(*args)

        whitelistLineEditLabel = QLabel("Whitelist IDs: ")
        self.addWidget(whitelistLineEditLabel)

        self.whitelistLineEdit = WhitelistLineEdit(whitelistLineEditLabel, parent=self)
        self.whitelistLineEdit.sigEnterPressed.connect(self.accept)
        self.whitelistLineEdit.sigIDsChanged.connect(self.emitWhitelistChanged)
        self.addWidget(self.whitelistLineEdit)

        # accept button
        self.acceptButton = self.addButton(":greenTick.svg")
        self.acceptButton.triggered.connect(self.accept)

        # add a view OG toggle
        self.viewOGToggle = self.addButton(":eye.svg", checkable=True)
        viewOGTooltip = (
            "View the non-whitelisted segmentation mask.\n\n"
            "You can activate this to add new IDs to the whitelist,\n"
            "correct tracking errors, etc."
        )
        self.viewOGToggle.setChecked(True)
        self.viewOGToggle.setToolTip(viewOGTooltip)
        self.viewOGToggle.setShortcut("Shift+K")
        key = "View the non-whitelisted segmentation mask"
        self.widgetsWithShortcut[key] = self.viewOGToggle

        self.viewOGToggle.toggled.connect(self.emitViewOGIDs)
        self.emitViewOGIDs(True)

        # add a Toggle to add new IDs
        self.addNewIDToggle = QCheckBox("Automatically add new IDs to whitelist")
        self.addNewIDToggle.setChecked(addNewIDToggleState)
        self.addWidget(self.addNewIDToggle)
        self.addNewIDToggle.toggled.connect(self.emitAddNewIDs)
        self.emitAddNewIDs(addNewIDToggleState)

        self.addSeparator()

        # add a button to load og df
        self.loadOGButton = self.addButton(":open_file.svg")
        self.loadOGButton.triggered.connect(self.sigLoadOGLabs.emit)
        self.loadOGButton.setToolTip(
            "Select which segmentation mask file to load as the non-whitelisted masks"
        )

        self.TrackOGagainstPreviousFrameButton = self.addButton(":segment.svg")
        self.TrackOGagainstPreviousFrameButton.triggered.connect(
            self.sigTrackOGagainstPreviousFrame.emit
        )
        self.TrackOGagainstPreviousFrameButton.setToolTip(
            "Track the non-whitelisted segmentation masks against the previous frame and copy over successfull tacks"
        )

        self.addSeparator()

        # add an info button
        self.infoButton = self.addButton(":info.svg")
        self.infoButton.triggered.connect(self.showInfo)

        # add a spacer to the toolbar
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(spacer)

    def emitWhitelistChanged(self, whitelist):
        self.sigWhitelistChanged.emit(whitelist)

    def emitViewOGIDs(self, checked):
        self.sigViewOGIDs.emit(checked)

    def accept(self):
        try:
            whitelist = self.whitelistLineEdit.IDs
        except AttributeError as e:
            if "has no attribute 'IDs'" in str(e):
                whitelist = list()
        self.viewOGToggle.toggled.disconnect()
        self.viewOGToggle.setChecked(False)
        self.viewOGToggle.toggled.connect(self.emitViewOGIDs)
        self.sigWhitelistAccepted.emit(whitelist)

    def emitAddNewIDs(self, checked):
        self.sigAddNewIDs.emit(checked)

    def showInfo(self):
        msg = myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            This function is used to track a subset of segmented objects.<br><br>
            
            To add new IDs to the white list, click with left mouse button on the 
            object to add.<br>
            You can also write directly into the <code>Whitelist IDs</code> widget<br>
            and separate the IDs by commas.<br><br>
            
            After adding the IDs, click on the "Accept" button to remove the 
            non-whitelisted objects.<br>
            Every time you visit a new frame, the non-whitelisted objects will 
            be removed automatically.<br><br>
            Use the "Eye" button to view the non-whitelisted segmentation masks.<br>
            This will allow you to correct tracking errors, add new IDs to the 
            white list, etc.<br><br>
            
            If you previously saved the whitelisted masks, you can load the 
            non-whitelisted file<br>
            by clicking on the "Load file" button to restart from where you 
            left last time.
        """)
        msg.information(self, "White list IDs", txt)


class MagicPromptsToolbar(ToolBar):
    sigPromptTypeChanged = Signal(object, str)
    sigComputeOnZoom = Signal(object)
    sigComputeOnImage = Signal(object)
    sigClearPoints = Signal(object)
    sigClearPointsOnZmom = Signal(object)
    sigInitSelectedModel = Signal(str, object, list, list, str, object)
    sigViewModelParams = Signal(str, object, list, list, str, object, object, object)
    sigInterpolateZslice = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._parent = parent

        prompt_types = ("Points",)

        self.selectModelAction = self.addButton(":select-list.svg")
        self.selectModelAction.setToolTip("Select the promptable model to use")

        self.viewModelParamsAction = self.addButton(":view.svg")
        self.viewModelParamsAction.setToolTip(
            "View the currently selected model parameters"
        )
        self.viewModelParamsAction.setDisabled(True)

        self.addSeparator()

        self.promptTypeCombobox = self.addComboBox(
            prompt_types,
            label="Prompt type: ",
        )

        self.addSeparator()

        self.interpolateZslicesCheckbox = self.addCheckBox(
            "Interpolate points on missing z-slices", checked=False
        )
        self.interpolateZslicesCheckbox.setToolTip(
            "If checked, when working with 3D segmentation masks, you can "
            "add points on some z-slices only and the points on the missing "
            "z-slices will be determined by linear interpolation.\n\n"
            "This is useful when working with 2D models that segments "
            "each z-slice independently.\n\n"
            "NOTE: The points will be added only when running the model and "
            "removed afterwards."
        )

        self.addSeparator()

        self.computeOnZoomAction = self.addButton(":compute-zoom.svg")
        self.computeOnZoomAction.setToolTip(
            "Compute the segmentation on the zoomed area of the image (faster)"
        )

        self.computeAction = self.addButton(":compute.svg")
        self.computeAction.setToolTip("Compute the segmentation on the whole image")

        self.clearPointsAction = self.addButton(":clear-points.svg")
        self.clearPointsAction.setToolTip("Clear all points")
        self.clearPointsAction.setDisabled(True)

        self.clearPointsActionOnZoom = self.addButton(":clear-points-zoom.svg")
        self.clearPointsActionOnZoom.setToolTip(
            "Clear all points on the zoomed area of the image"
        )
        self.clearPointsActionOnZoom.setDisabled(True)

        self.addSeparator()

        self.infoAction = self.addButton(":info.svg")
        self.infoAction.setToolTip("Show instructions how to use promptable models")

        self.addSeparator()

        self.infoAction.triggered.connect(self.showHelp)
        self.selectModelAction.triggered.connect(self.selectModel)
        self.viewModelParamsAction.triggered.connect(self.viewModelParams)
        self.promptTypeCombobox.sigTextChanged.connect(self.emitPromptTypeChanged)
        self.computeOnZoomAction.triggered.connect(self.emitSigComputeOnZoom)
        self.computeAction.triggered.connect(self.emitSigComputeOnImage)
        self.clearPointsAction.triggered.connect(self.emitSigClearPoints)
        self.clearPointsActionOnZoom.triggered.connect(self.emitSigClearPointsOnZoom)
        self.interpolateZslicesCheckbox.toggled.connect(self.sigInterpolateZslice.emit)

    def showHelp(self):
        msg = myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            This toolbar allows you to use <b>promptable models for 
            segmentation</b>.<br><br>
            
            To use a promptable model, first <b>select the model</b> by clicking on the 
            "Select model" button.<br>
            This will open a dialog where you can select the model to use.<br><br>
            
            After selecting the model, you can <b>view the model parameters</b> 
            by clicking on the "View model parameters" button.<br><br>
            
            To add points to the image, make sure you have points layer correctly 
            initialised. You should see controls<br>
            called "Left-click ID" and "Right-click ID".<br><br>
            
            You can <b>add points</b> for a new object by left-clicking on the image, 
            while you can add points<br>
            for the same object by right-clicking. 
            To <b>delete a point</b>, click on it again.<br><br>
            
            To change the right-click ID, 
            you can either type in the corresponding control,<br>
            or type the object id on the keyboard followed by "Enter".<br><br>
            
            To add <b>negative prompts</b> (i.e., for the background), use the 
            same action you use to delete objects<br>
            (default is middle-click on Windows and Cmd+Click on MacOS).<br><br>
            Note that you can also add object-specific negative prompts (i.e., 
            they affect only that object)<br>
            by adding the negative prompt on the newly segmented object 
            directly.<br><br>
            
            Once you are happy with the added points, click either the 
            "<b>Compute on zoomed area</b>"<br>
            button or the "<b>Compute on whole image</b>" button.<br><br>
            
            Finally, you can <b>clear all points</b> by clicking on the 
            "Clear points" button.<br><br>
            
            Note that you can also save the points by clicking on the 
            "<b>Save points</b>" button to load them later and start from 
            where you left.<br><br>
        """)
        msg.information(self, "Promptable models help", txt)

    def emitSigClearPoints(self):
        self.sigClearPoints.emit(self)

    def emitSigClearPointsOnZoom(self):
        self.sigClearPointsOnZmom.emit(self)

    def emitSigComputeOnZoom(self):
        self.sigComputeOnZoom.emit(self)

    def emitSigComputeOnImage(self):
        self.sigComputeOnImage.emit(self)

    def selectModel(self):
        win = apps.SelectPromptableModelDialog(parent=self._parent)
        win.exec_()
        if win.cancel:
            print("Promptable model selection cancelled")
            return

        model_name = win.model_name
        print(f"Importing promptable model {model_name}...")

        # Download model weights, consistent with gui.py
        downloadWin = apps.downloadModel(model_name, parent=self._parent)
        downloadWin.download()

        acdcPromptSegment = myutils.import_promptable_segment_module(model_name)
        init_argspecs, segment_argspecs = myutils.getModelArgSpec(acdcPromptSegment)

        try:
            help_url = acdcPromptSegment.url_help()
        except AttributeError:
            help_url = None

        self._model_name = model_name
        self._acdcPromptSegment = acdcPromptSegment
        self._init_argspecs = init_argspecs
        self._segment_argspecs = segment_argspecs
        self._help_url = help_url

        self.sigInitSelectedModel.emit(
            model_name,
            acdcPromptSegment,
            init_argspecs,
            segment_argspecs,
            help_url,
            self,
        )

    def setInitializedModel(self, init_kwargs, segment_kwargs):
        self._init_kwargs = init_kwargs
        self._segment_kwargs = segment_kwargs

    def viewModelParams(self):
        self.sigViewModelParams.emit(
            self._model_name,
            self._acdcPromptSegment,
            self._init_argspecs,
            self._segment_argspecs,
            self._help_url,
            self._init_kwargs,
            self._segment_kwargs,
            self,
        )

    def emitPromptTypeChanged(self, text):
        self.sigPromptTypeChanged.emit(self, text)


class PointsLayersToolbar(ToolBar):
    sigAddPointsLayer = Signal()

    def __init__(self, name="Points layers", parent=None):

        super().__init__(name, parent)

        self.guiWin = parent

        self.setContextMenuPolicy(Qt.PreventContextMenu)

        self.addPointsLayerAction = self.addButton(":addPointsLayer.svg")

        self.addSeparator()

        self.pointsLayersLabel = self.addLabel("Points layers: ")

        self.addPointsLayerAction.triggered.connect(self.emitAddPointsLayer)
        self.doAddPointsZslicesInterpolation = False

    def emitAddPointsLayer(self):
        self.sigAddPointsLayer.emit()

    def fromActionToDataFrame(self, action, posData, isSegm3D=False):
        df = pd.DataFrame(columns=["frame_i", "Cell_ID", "z", "y", "x", "id"])
        frames_vals = []
        IDs = []
        zz = []
        yy = []
        xx = []
        ids = []
        pos_i = self.guiWin.pos_i
        if pos_i not in action.pointsData:
            printl(
                "No points data for position", pos_i
            )  # should really not happen, but its not a disaster if it does
            return df
        pointsDataPos = action.pointsData[pos_i]
        for frame_i, framePointsData in pointsDataPos.items():
            if posData.SizeZ > 1:
                for z, zSlicePointsData in framePointsData.items():
                    yyxx = zip(zSlicePointsData["y"], zSlicePointsData["x"])
                    for y, x in yyxx:
                        if isSegm3D:
                            ID = posData.lab[int(z), int(y), int(x)]
                        else:
                            ID = posData.lab[int(y), int(x)]
                        frames_vals.append(frame_i)
                        IDs.append(ID)
                        zz.append(z)
                        yy.append(y)
                        xx.append(x)
                    ids.extend(zSlicePointsData["id"])
            else:
                yyxx = zip(framePointsData["y"], framePointsData["x"])
                for y, x in yyxx:
                    ID = posData.lab[int(y), int(x)]
                    frames_vals.append(frame_i)
                    IDs.append(ID)
                    yy.append(y)
                    xx.append(x)
                ids.extend(framePointsData["id"])
        df["frame_i"] = frames_vals
        df["Cell_ID"] = IDs
        df["y"] = yy
        df["x"] = xx
        df["id"] = ids
        if zz:
            df["z"] = zz

        df = self.addPointsZslicesInterpolation(df, posData.lab, isSegm3D)

        return df

    def addPointsZslicesInterpolation(
        self, df: pd.DataFrame, lab: np.ndarray, isSegm3D: bool
    ):
        if not self.doAddPointsZslicesInterpolation:
            return df

        if not isSegm3D:
            return df

        if "z" not in df.columns:
            return df

        df_new_rows = []
        for (frame_i, point_id), df_id in df.groupby(["frame_i", "id"]):
            xx = df_id["x"].values
            yy = df_id["y"].values
            zz = df_id["z"].values

            p0, d = core.linear_fit_3d(xx, yy, zz)

            new_row_df = df_id.iloc[[0]].copy()

            z0, z1 = int(np.min(zz)), int(np.max(zz))
            for z in range(z0, z1 + 1):
                if z in zz:
                    continue

                t_int = (z - p0[2]) / d[2]
                x_new, y_new, z_new = p0 + t_int * d
                new_row_df["z"] = round(z_new)
                new_row_df["y"] = round(y_new)
                new_row_df["x"] = round(x_new)

                Cell_ID = lab[int(round(z_new)), int(round(y_new)), int(round(x_new))]
                new_row_df["Cell_ID"] = Cell_ID

                df_new_rows.append(new_row_df.copy())

        if not df_new_rows:
            return df

        df_new = pd.concat(df_new_rows, ignore_index=True)
        df = pd.concat([df, df_new], ignore_index=True)
        df = df.sort_values(by=["frame_i", "id", "z"]).reset_index(drop=True)

        return df


class PromptableModelPointsLayerToolbar(PointsLayersToolbar):
    def __init__(self, name="Promptable model points layers", parent=None):
        super().__init__(name, parent=parent)

        self.isPointsLayerInit = False

        self.addPointsLayerAction.setDisabled(True)
        self.addPointsLayerAction.setVisible(False)

    def pointsLayerDf(self, posData, isSegm3D=False):
        for action in self.actions()[1:]:
            if not hasattr(action, "button"):
                continue

            df = self.fromActionToDataFrame(action, posData, isSegm3D=isSegm3D)
            return df

    def scatterItem(self):
        for action in self.actions()[1:]:
            if not hasattr(action, "button"):
                continue

            return action.scatterItem


class OverlayToolbar(ToolBar):
    sigSetTranspacency = Signal(bool)
    sigSetSingleChannel = Signal(bool)

    def __init__(self, name="Overlay tools", parent=None):

        super().__init__(name, parent)

        self.guiWin = parent

        self.setContextMenuPolicy(Qt.PreventContextMenu)

        self.addSeparator()

        self.transparencyCheckbox = self.addCheckBox(
            text="True transparency (RGBA composite)"
        )

        self.transparencyCheckbox.setToolTip(
            "Activate to achieve true pixel-wise transparency where "
            "the pixel intensity is 0 or set to 0 using the "
            "LUT sliders on the left of the images.\n\n"
            "Since it is significantly slower, we recommended to activate this "
            "only if you need to export images for figures."
        )

        self.addSeparator()

        self.singleChannelCheckbox = self.addCheckBox(text="Single channel")

        self.singleChannelCheckbox.setToolTip(
            "When single channel mode is activated, selecting a channel "
            "will display only that channel in the overlay."
        )

        self.transparencyCheckbox.toggled.connect(self.sigSetTranspacency.emit)
        self.singleChannelCheckbox.toggled.connect(self.sigSetSingleChannel.emit)

    def setTransparent(self, transparent: bool):
        self.transparencyCheckbox.setChecked(transparent)

    def isTransparent(self):
        return self.transparencyCheckbox.isChecked()

    def isSingleChannel(self):
        return self.singleChannelCheckbox.isChecked()


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


class HighlightedIDToolbar(ToolBar):
    sigIDChanged = Signal(int)

    def __init__(self, name="Highlighted ID", parent=None):

        super().__init__(name, parent)

        self.spinbox = self.addSpinBox("Highlighted ID: ")
        self.spinbox.valueChanged.connect(self.emitSigIDChanged)

        self.addSeparator()

    def emitSigIDChanged(self, *args, **kwargs):
        self.sigIDChanged.emit(self.spinbox.value())

    def setIDNoSignals(self, ID: int):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(ID)
        self.spinbox.blockSignals(False)


class WandControlsToolbar(ToolBar):
    def __init__(self, name="Magic wand controls", parent=None):
        super().__init__(name, parent)

        self.toleranceSpinbox = self.addSpinBox("Tolerance [%]: ")
        self.toleranceSpinbox.setMinimum(0)
        self.toleranceSpinbox.setMaximum(100)
        self.toleranceSpinbox.setValue(5)
        self.toleranceSpinbox.setToolTip(
            "The tolerance is calculated as a percentage of the minimum-maximum "
            "pixel values range of the loaded dataset.\n\n"
            "If tolerance is greater than 0, the pixels adjacent to the added "
            "pixels with value within +- tolerance will be considered part of "
            "the object."
        )
        self.addLabel(r"% of min-max intensity range ")

        self.addSeparator()

        self.autoFillHolesCheckbox = self.addCheckBox("Auto-fill holes")

        self.addSeparator()

        self.useConvexHullCheckbox = self.addCheckBox("Use convex hull mask")

        self.addSeparator()

# Sibling imports (deferred to avoid import cycles)
from .canvas import (
    myHistogramLUTitem,
)
from .controls import (
    ComboBox,
    KeySequenceFromText,
    SpinBox,
    WhitelistLineEdit,
    myMessageBox,
)

