"""Canvas widgets: histogram."""

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

class BaseGradientEditorItemImage(pg.GradientEditorItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def restoreState(self, state):
        pg.graphicsItems.GradientEditorItem.Gradients = GradientsImage
        return super().restoreState(state)


class BaseGradientEditorItemLabels(pg.GradientEditorItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def restoreState(self, state):
        pg.graphicsItems.GradientEditorItem.Gradients = GradientsLabels
        return super().restoreState(state)


class baseHistogramLUTitem(pg.HistogramLUTItem):
    sigAddColormap = Signal(object, str)
    sigRescaleIntes = Signal(object)

    def __init__(self, name="image", axisLabel="", parent=None, **kwargs):
        pg.GradientEditorItem = BaseGradientEditorItemLabels

        super().__init__(**kwargs)

        self.labelStyle = {"color": "#ffffff", "font-size": "11px"}

        if axisLabel:
            self.setAxisLabel(axisLabel)

        self.cmaps = cmaps
        self._parent = parent
        self.name = name

        self.gradient.colorDialog.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.gradient.colorDialog.accepted.disconnect()
        self.gradient.colorDialog.accepted.connect(self.tickColorAccepted)

        self.isInverted = False
        self.lastGradientName = "grey"
        self.lastGradient = Gradients["grey"]

        for action in self.gradient.menu.actions():
            if action.text() == "HSV":
                HSV_action = action
            elif action.text() == "RGB":
                RGB_ation = action
        self.gradient.menu.removeAction(HSV_action)
        self.gradient.menu.removeAction(RGB_ation)

        # Rescale intensities (LUT)
        rescaleIntensMenu = self.gradient.menu.addMenu("Rescale intensities (LUT)")
        rescaleActionGroup = QActionGroup(self)
        rescaleActionGroup.setExclusive(True)

        self.rescaleEach2DimgAction = QAction(
            "Rescale each 2D image", rescaleIntensMenu
        )
        self.rescaleEach2DimgAction.setCheckable(True)
        self.rescaleEach2DimgAction.setChecked(True)
        rescaleActionGroup.addAction(self.rescaleEach2DimgAction)
        rescaleIntensMenu.addAction(self.rescaleEach2DimgAction)

        self.rescaleAcrossZstackAction = QAction(
            "Rescale across z-stack", rescaleIntensMenu
        )
        self.rescaleAcrossZstackAction.setCheckable(True)
        self.rescaleAcrossZstackAction.setChecked(False)
        rescaleActionGroup.addAction(self.rescaleAcrossZstackAction)
        rescaleIntensMenu.addAction(self.rescaleAcrossZstackAction)

        self.rescaleAcrossTimeAction = QAction(
            "Rescale across time frames", rescaleIntensMenu
        )
        self.rescaleAcrossTimeAction.setCheckable(True)
        self.rescaleAcrossTimeAction.setChecked(False)
        rescaleActionGroup.addAction(self.rescaleAcrossTimeAction)
        rescaleIntensMenu.addAction(self.rescaleAcrossTimeAction)

        self.customRescaleAction = QAction("Choose custom levels...", rescaleIntensMenu)
        self.customRescaleAction.setCheckable(True)
        rescaleActionGroup.addAction(self.customRescaleAction)
        rescaleIntensMenu.addAction(self.customRescaleAction)

        self.doNotRescaleAction = QAction(
            "Do no rescale, display raw image", rescaleIntensMenu
        )
        self.doNotRescaleAction.setCheckable(True)
        rescaleActionGroup.addAction(self.doNotRescaleAction)
        rescaleIntensMenu.addAction(self.doNotRescaleAction)

        self.rescaleActionGroup = rescaleActionGroup
        rescaleActionGroup.triggered.connect(self.rescaleActionTriggered)

        # Add custom colormap action
        self.customCmapsMenu = self.gradient.menu.addMenu("Custom colormaps")
        self.customCmapsMenu.aboutToShow.connect(self.onShowCustomCmapsMenu)
        self.customCmapsMenu.triggered.connect(self.customCmapsMenuTriggered)

        self.saveColormapAction = QAction("Save current colormap...", self)
        self.gradient.menu.addAction(self.saveColormapAction)
        self.saveColormapAction.triggered.connect(self.saveColormap)

        self.addCustomGradients()

        # Set inverted gradients for invert bw action
        self.addInvertedColorMaps()

        self.gradient.menu.addSeparator()

        # hide histogram tool
        self.vb.hide()

        # Disable moving the axis up and down
        self.axis.unlinkFromView()

        # Disable histogram default context Menu event
        self.vb.raiseContextMenu = lambda x: None

    def rescaleActionTriggered(self, action):
        self.sigRescaleIntes.emit(action)

    def onShowCustomCmapsMenu(self):
        self.customCmapsMenu.show()

    def customCmapsMenuTriggered(self, action):
        cmap = action.cmap
        self.gradient.colorMapMenuClicked(cmap)
        self.gradient.showTicks(True)

    def setAxisLabel(self, text):
        self.labelText = text
        self.axis.setLabel(text, **self.labelStyle)

    def updateAxisLabel(self):
        text = self.axis.label.toPlainText()
        if not text:
            return
        self.setAxisLabel(text)

    def setGradient(self, gradient):
        self.gradient.restoreState(gradient)
        self.lastGradient = gradient

    def colormapClicked(self, checked=False, name=None):
        name = self.sender().name
        self.lastGradientName = name
        if self.isInverted:
            self.setGradient(self.invertedGradients[name])
        else:
            self.setGradient(Gradients[name])

    def sortTicks(self, ticks):
        sortedTicks = sorted(ticks, key=operator.itemgetter(0))
        return sortedTicks

    def getInvertedGradients(self):
        invertedGradients = {}
        for name, gradient in Gradients.items():
            ticks = gradient["ticks"]
            sortedTicks = self.sortTicks(ticks)
            if name in nonInvertibleCmaps:
                invertedColors = sortedTicks
            else:
                invertedColors = [
                    (t[0], ti[1]) for t, ti in zip(sortedTicks, sortedTicks[::-1])
                ]
            invertedGradient = {}
            invertedGradient["ticks"] = invertedColors
            invertedGradient["mode"] = gradient["mode"]
            invertedGradients[name] = invertedGradient
        return invertedGradients

    def addInvertedColorMaps(self):
        self.invertedGradients = self.getInvertedGradients()
        for action in self.gradient.menu.actions():
            if not hasattr(action, "name"):
                continue

            if action.name not in self.cmaps:
                continue

            action.triggered.disconnect()
            action.triggered.connect(self.colormapClicked)

            px = QPixmap(100, 15)
            p = QPainter(px)
            invertedGradient = self.invertedGradients[action.name]
            qtGradient = QLinearGradient(QPointF(0, 0), QPointF(100, 0))
            ticks = self.sortTicks(invertedGradient["ticks"])
            qtGradient.setStops([(x, QColor(*color)) for x, color in ticks])
            brush = QBrush(qtGradient)
            p.fillRect(QRect(0, 0, 100, 15), brush)
            p.end()
            widget = action.defaultWidget()
            hbox = widget.layout()
            rectLabelWidget = QLabel()
            rectLabelWidget.setPixmap(px)
            hbox.addWidget(rectLabelWidget)
            rectLabelWidget.hide()

    def setInvertedColorMaps(self, inverted):
        if inverted:
            showIdx = 2
            hideIdx = 1
            self.labelStyle["color"] = "#000000"
        else:
            showIdx = 1
            hideIdx = 2
            self.labelStyle["color"] = "#ffffff"

        for action in self.gradient.menu.actions():
            if not hasattr(action, "name"):
                continue

            if action.name not in self.cmaps:
                continue

            widget = action.defaultWidget()
            hbox = widget.layout()
            hideCmapRect = hbox.itemAt(hideIdx).widget()
            showCmapRect = hbox.itemAt(showIdx).widget()
            hideCmapRect.hide()
            showCmapRect.show()

        self.updateAxisLabel()
        self.isInverted = inverted

    def invertGradient(self, gradient):
        ticks = gradient["ticks"]
        sortedTicks = self.sortTicks(ticks)
        invertedColors = [
            (t[0], ti[1]) for t, ti in zip(sortedTicks, sortedTicks[::-1])
        ]
        invertedGradient = {}
        invertedGradient["ticks"] = invertedColors
        invertedGradient["mode"] = gradient["mode"]
        return invertedGradient

    def invertCurrentColormap(self, inverted, debug=False):
        self.setGradient(self.invertGradient(self.lastGradient))

    def addCustomGradient(self, gradient_name, gradient_ticks, restore=True):
        self.originalLength = self.gradient.length
        self.gradient.length = 100
        if restore:
            self.gradient.restoreState(gradient_ticks)
        gradient = self.gradient.getGradient()
        action = CustomGradientMenuAction(gradient, gradient_name, self.gradient)
        # action.triggered.connect(self.gradient.contextMenuClicked)
        action.delButton.clicked.connect(self.removeCustomGradient)
        action.cmap = colors.pg_ticks_to_colormap(gradient_ticks["ticks"])
        # self.gradient.menu.insertAction(self.saveColormapAction, action)
        self.customCmapsMenu.addAction(action)
        self.gradient.length = self.originalLength
        GradientsImage[gradient_name] = gradient_ticks

    def removeCustomGradient(self):
        button = self.sender()
        action = button.action
        self.customCmapsMenu.removeAction(action)
        cp = config.ConfigParser()
        cp.read(custom_cmaps_filepath)
        cp.remove_section(f"image.{action.name}")
        with open(custom_cmaps_filepath, mode="w") as file:
            cp.write(file)

    def addCustomGradients(self):
        try:
            CustomGradients = getCustomGradients(name="image")
            if not CustomGradients:
                return
            for gradient_name, gradient_ticks in CustomGradients.items():
                self.addCustomGradient(gradient_name, gradient_ticks)
        except Exception as e:
            printl(traceback.format_exc())
            pass

    def _askNameColormap(self):
        inputWin = apps.QInput(parent=self._parent, title="Colormap name")
        inputWin.askText("Insert a name for the colormap: ", allowEmpty=False)
        if inputWin.cancel:
            return
        cmapName = inputWin.answer
        return cmapName

    def saveColormap(self):
        cmapName = self._askNameColormap()
        if cmapName is None:
            return

        cp = config.ConfigParser()
        if os.path.exists(custom_cmaps_filepath):
            cp.read(custom_cmaps_filepath)

        SECTION = f"{self.name}.{cmapName}"
        cp[SECTION] = {}

        # gradient_ticks = []
        state = self.gradient.saveState()
        for key, value in state.items():
            if key != "ticks":
                continue
            for t, tick in enumerate(value):
                pos, rgb = tick
                # gradient_ticks.append((pos, rgb))
                rgb = ",".join([str(c) for c in rgb])
                val = f"{pos},{rgb}"
                cp[SECTION][f"tick_{t}_pos_rgb"] = val

        with open(custom_cmaps_filepath, mode="w") as file:
            cp.write(file)

        self.addCustomGradient(cmapName, state, restore=False)

    def tickColorAccepted(self):
        self.gradient.currentColorAccepted()
        # self.sigTickColorAccepted.emit(self.gradient.colorDialog.color().getRgb())

    def setRescaleIntensitiesHow(self, how):
        for action in self.rescaleActionGroup.actions():
            if action.text() == how:
                action.setChecked(True)
                return


class myHistogramLUTitem(baseHistogramLUTitem):
    sigGradientMenuEvent = Signal(object)
    sigGradientChanged = Signal(object)
    sigTickColorAccepted = Signal(object)
    sigAddScaleBar = Signal(bool)
    sigAddTimestamp = Signal(bool)

    def __init__(
        self, parent=None, name="image", axisLabel="", isViewer=False, **kwargs
    ):
        super().__init__(parent=parent, name=name, axisLabel=axisLabel, **kwargs)

        self.name = name
        self._parent = parent

        self.childLutItem = None

        self.isViewer = isViewer
        if isViewer:
            # In the viewer we don't allow additional settings from the menu
            return

        # Add scale bar action
        self.addScaleBarAction = QAction("Add scale bar", self)
        self.addScaleBarAction.setCheckable(True)
        self.addScaleBarAction.triggered.connect(self.emitAddScaleBar)
        self.gradient.menu.addAction(self.addScaleBarAction)

        # Add timestamp action
        self.addTimestampAction = QAction("Add timestamp", self)
        self.addTimestampAction.setCheckable(True)
        self.addTimestampAction.triggered.connect(self.emitAddTimestamp)
        self.gradient.menu.addAction(self.addTimestampAction)

        # Invert bw action
        self.invertBwAction = QAction("Invert black/white", self)
        self.invertBwAction.setCheckable(True)
        self.gradient.menu.addAction(self.invertBwAction)

        # Font size menu action
        self.fontSizeMenu = QMenu("Text font size")
        self.gradient.menu.addMenu(self.fontSizeMenu)

        # Text color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Text color: "))
        self.textColorButton = myColorButton(color=(255, 255, 255))
        hbox.addStretch(1)
        hbox.addWidget(self.textColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.textColorButton.click)
        self.gradient.menu.addAction(act)

        # Contours line weight
        contLineWeightMenu = QMenu("Contours line weight", self.gradient.menu)
        self.contLineWightActionGroup = QActionGroup(self)
        self.contLineWightActionGroup.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.Exclusive
        )
        for w in range(1, 11):
            action = QAction(str(w))
            action.setCheckable(True)
            if w == 2:
                action.setChecked(True)
            action.lineWeight = w
            self.contLineWightActionGroup.addAction(action)
            action = contLineWeightMenu.addAction(action)
        self.gradient.menu.addMenu(contLineWeightMenu)

        # Contours color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Contours color: "))
        self.contoursColorButton = myColorButton(color=(25, 25, 25))
        hbox.addStretch(1)
        hbox.addWidget(self.contoursColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.contoursColorButton.click)
        self.gradient.menu.addAction(act)

        # Mother-bud line weight
        mothBudLineWeightMenu = QMenu("Mother-bud line weight", self.gradient.menu)
        self.mothBudLineWightActionGroup = QActionGroup(self)
        self.mothBudLineWightActionGroup.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.Exclusive
        )
        for w in range(1, 11):
            action = QAction(str(w))
            action.setCheckable(True)
            if w == 2:
                action.setChecked(True)
            action.lineWeight = w
            self.mothBudLineWightActionGroup.addAction(action)
            action = mothBudLineWeightMenu.addAction(action)
        self.gradient.menu.addMenu(mothBudLineWeightMenu)

        # Mother-bud line color
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Mother-bud line color: "))
        self.mothBudLineColorButton = myColorButton(color=(255, 0, 0))
        hbox.addStretch(1)
        hbox.addWidget(self.mothBudLineColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.mothBudLineColorButton.click)
        self.gradient.menu.addAction(act)

        self.labelsAlphaMenu = self.gradient.menu.addMenu(
            "Segm. masks overlay alpha..."
        )
        # self.labelsAlphaMenu.setDisabled(True)
        hbox = QHBoxLayout()
        self.labelsAlphaSlider = sliderWithSpinBox(
            title="Alpha", title_loc="in_line", isFloat=True, normalize=True
        )
        self.labelsAlphaSlider.setMaximum(100)
        self.labelsAlphaSlider.setSingleStep(0.05)
        self.labelsAlphaSlider.setValue(0.3)
        hbox.addWidget(self.labelsAlphaSlider)
        shortCutText = "Command+Up/Down" if is_mac else "Ctrl+Up/Down"
        hbox.addWidget(QLabel(f"({shortCutText})"))
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.labelsAlphaMenu.addSeparator()
        self.labelsAlphaMenu.addAction(act)

        # Default settings
        self.defaultSettingsAction = QAction("Restore default settings...", self)
        self.gradient.menu.addAction(self.defaultSettingsAction)

        self.filterObject = FilterObject()
        self.filterObject.sigFilteredEvent.connect(self.gradientMenuEventFilter)
        self.gradient.menu.installEventFilter(self.filterObject)
        self.highlightedAction = None
        self.lastHoveredAction = None

    def setChildLutItem(self, childLutItem):
        self.childLutItem = childLutItem

    def removeAddScaleBarAction(self):
        self.gradient.menu.removeAction(self.addScaleBarAction)

    def removeAddTimestampAction(self):
        self.gradient.menu.removeAction(self.addTimestampAction)

    def emitAddScaleBar(self):
        self.sigAddScaleBar.emit(self.addScaleBarAction.isChecked())

    def emitAddTimestamp(self):
        self.sigAddTimestamp.emit(self.addTimestampAction.isChecked())

    def gradientChanged(self):
        super().gradientChanged()
        self.sigGradientChanged.emit(self)

    def gradientMenuEventFilter(self, object, event):
        if event.type() == QEvent.Type.MouseMove:
            hoveredAction = self.gradient.menu.actionAt(event.pos())
            isActionEntered = hoveredAction != self.lastHoveredAction
            if isActionEntered:
                if isinstance(hoveredAction, highlightableQWidgetAction):
                    # print('Entered a custom action')
                    pass
                isActionLeft = (
                    self.highlightedAction is not None
                    and self.highlightedAction != hoveredAction
                )
                if isActionLeft:
                    if isinstance(self.highlightedAction, highlightableQWidgetAction):
                        # print('Left a custom action')
                        pass
                self.highlightedAction = hoveredAction

            self.lastHoveredAction = hoveredAction

    def addOverlayColorButton(self, rgbColor, channelName):
        # Overlay color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Overlay color: "))
        self.overlayColorButton = myColorButton(color=rgbColor)
        self.overlayColorButton.channel = channelName
        hbox.addStretch(1)
        hbox.addWidget(self.overlayColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.overlayColorButton.click)
        self.gradient.menu.addAction(act)

    def uncheckContLineWeightActions(self):
        for act in self.contLineWightActionGroup.actions():
            try:
                act.toggled.disconnect()
            except Exception as e:
                pass
            act.setChecked(False)

    def uncheckMothBudLineLineWeightActions(self):
        for act in self.mothBudLineWightActionGroup.actions():
            try:
                act.toggled.disconnect()
            except Exception as e:
                pass
            act.setChecked(False)

    def restoreState(self, df):
        if "textIDsColor" in df.index:
            rgbString = df.at["textIDsColor", "value"]
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.textColorButton.setColor((r, g, b))

        if "contLineColor" in df.index:
            rgba_str = df.at["contLineColor", "value"]
            rgb = colors.rgba_str_to_values(rgba_str)[:3]
            self.contoursColorButton.setColor(rgb)

        if "contLineWeight" in df.index:
            w = df.at["contLineWeight", "value"]
            w = int(w)
            for action in self.contLineWightActionGroup.actions():
                if action.lineWeight == w:
                    action.setChecked(True)
                    break

        if "mothBudLineWeight" in df.index:
            w = df.at["mothBudLineWeight", "value"]
            w = int(w)
            for action in self.mothBudLineWightActionGroup.actions():
                if action.lineWeight == w:
                    action.setChecked(True)
                    break

        if "overlaySegmMasksAlpha" in df.index:
            alpha = df.at["overlaySegmMasksAlpha", "value"]
            self.labelsAlphaSlider.setValue(float(alpha))

        if "mothBudLineColor" in df.index:
            rgba_str = df.at["mothBudLineColor", "value"]
            rgb = colors.rgba_str_to_values(rgba_str)[:3]
            self.mothBudLineColorButton.setColor(rgb)

        checked = df.at["is_bw_inverted", "value"] == "Yes"
        self.invertBwAction.setChecked(checked)

        self.restoreColormap(df)

    def saveState(self, df):
        # remove previous state
        df = df[~df.index.str.contains("img_cmap")].copy()

        state = self.gradient.saveState()
        for key, value in state.items():
            if key == "ticks":
                for t, tick in enumerate(value):
                    pos, rgb = tick
                    df.at[f"img_cmap_tick{t}_rgb", "value"] = rgb
                    df.at[f"img_cmap_tick{t}_pos", "value"] = pos
            else:
                if isinstance(value, bool):
                    value = "Yes" if value else "No"
                df.at[f"img_cmap_{key}", "value"] = value
        return df

    def restoreColormap(self, df):
        state = {"mode": "rgb", "ticksVisible": True, "ticks": []}
        ticks_pos = {}
        ticks_rgb = {}
        stateFound = False
        for setting, value in df.itertuples():
            idx = setting.find("img_cmap_")
            if idx == -1:
                continue

            stateFound = True
            m = re.findall(r"tick(\d+)_(\w+)", setting)
            if m:
                tick_idx, tick_type = m[0]
                if tick_type == "pos":
                    ticks_pos[int(tick_idx)] = float(value)
                elif tick_type == "rgb":
                    ticks_rgb[int(tick_idx)] = colors.rgba_str_to_values(value)
            else:
                key = setting[9:]
                if value == "Yes":
                    value = True
                elif value == "No":
                    value = False
                state[key] = value

        if stateFound:
            ticks = [(0, 0)] * len(ticks_pos)
            for idx, val in ticks_pos.items():
                pos = val
                rgb = ticks_rgb[idx]
                ticks[idx] = (pos, rgb)

            state["ticks"] = ticks
            self.gradient.restoreState(state)

    def regionChanged(self):
        super().regionChanged()
        if self.childLutItem is None:
            return

        imageItem = self.imageItem()
        try:
            mn, mx = imageItem.quickMinMax(targetSize=65536)
            # mn and mx can still be NaN if the data is all-NaN
            if mn == mx or imageItem._xp.isnan(mn) or imageItem._xp.isnan(mx):
                mn = 0
                mx = 255
        except AttributeError as err:
            mn, mx = self.getLevels()

        self.childLutItem.setLevels(min=mn, max=mx)


class myColorButton(pg.ColorButton):
    def __init__(self, parent=None, color=(128, 128, 128), padding=5):
        super().__init__(parent=parent, color=color)
        if isinstance(padding, (int, float)):
            self.padding = (padding, padding, -padding, -padding)
        else:
            self.padding = padding
        self._c = 225
        self._hoverDeltaC = 30
        self._alpha = 100
        self._bkgrColor = QColor(self._c, self._c, self._c, self._alpha)
        self._borderColor = QColor(171, 171, 171)
        self._rectBorderPen = QPen(QBrush(QColor(0, 0, 0)), 0.3)

    def paintEvent(self, event):
        # QPushButton.paintEvent(self, ev)
        p = QStylePainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        p.setBrush(QBrush(self._bkgrColor))
        p.setPen(QPen(self._borderColor))
        p.drawRoundedRect(rect, 5, 5)
        # p.fillRect(self.rect(), self._bkgrColor)
        rect = self.rect().adjusted(*self.padding)
        ## draw white base, then texture for indicating transparency, then actual color
        p.setBrush(pg.mkBrush("w"))
        p.drawRect(rect)
        p.setBrush(QBrush(Qt.BrushStyle.DiagCrossPattern))
        p.drawRect(rect)
        p.setPen(self._rectBorderPen)
        p.setBrush(pg.mkBrush(self._color))
        p.drawRect(rect)
        p.end()

    def enterEvent(self, event):
        c = self._c + self._hoverDeltaC
        self._bkgrColor = QColor(c, c, c, self._alpha)
        self.update()

    def leaveEvent(self, event):
        c = self._c
        self._bkgrColor = QColor(c, c, c, self._alpha)
        self.update()


class overlayLabelsGradientWidget(pg.GradientWidget):
    def __init__(
        self,
        imageItem,
        selectActionGroup,
        segmEndname,
        parent=None,
        orientation="right",
    ):
        pg.GradientWidget.__init__(self, parent=parent, orientation=orientation)

        self.imageItem = imageItem
        self.selectActionGroup = selectActionGroup

        for action in self.menu.actions():
            if action.text() == "HSV":
                HSV_action = action
            elif action.text() == "RGB":
                RGB_ation = action
        self.menu.removeAction(HSV_action)
        self.menu.removeAction(RGB_ation)

        # Shuffle colors action
        self.shuffleCmapAction = QAction("Randomly shuffle colormap   (Shift+S)", self)
        self.menu.addAction(self.shuffleCmapAction)

        # Drawing mode
        drawModeMenu = QMenu("Drawing mode", self)
        self.drawModeActionGroup = QActionGroup(self)
        contoursDrawModeAction = QAction("Draw contours", drawModeMenu)
        contoursDrawModeAction.setCheckable(True)
        contoursDrawModeAction.setChecked(True)
        contoursDrawModeAction.segmEndname = segmEndname
        self.drawModeActionGroup.addAction(contoursDrawModeAction)
        drawModeMenu.addAction(contoursDrawModeAction)
        olDrawModeAction = QAction("Overlay labels", drawModeMenu)
        olDrawModeAction.setCheckable(True)
        olDrawModeAction.segmEndname = segmEndname
        self.drawModeActionGroup.addAction(olDrawModeAction)
        drawModeMenu.addAction(olDrawModeAction)
        self.menu.addMenu(drawModeMenu)

        self.labelsAlphaMenu = self.menu.addMenu("Overlay labels alpha...")
        hbox = QHBoxLayout()
        self.labelsAlphaSlider = sliderWithSpinBox(
            title="Alpha", title_loc="in_line", isFloat=True, normalize=True
        )
        self.labelsAlphaSlider.setMaximum(100)
        self.labelsAlphaSlider.setSingleStep(0.05)
        self.labelsAlphaSlider.setValue(0.3)
        hbox.addWidget(self.labelsAlphaSlider)
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.labelsAlphaMenu.addSeparator()
        self.labelsAlphaMenu.addAction(act)

        self.menu.addSeparator()
        self.menu.addSection("Select segm. file to adjust:")
        for action in selectActionGroup.actions():
            self.menu.addAction(action)

        self.item.loadPreset("viridis")
        self.updateImageLut(None)
        self.updateImageOpacity(0.3)

        # Connect events
        self.sigGradientChangeFinished.connect(self.updateImageLut)
        self.labelsAlphaSlider.valueChanged.connect(self.updateImageOpacity)
        self.shuffleCmapAction.triggered.connect(self.shuffleCmap)

    def shuffleCmap(self):
        lut = self.imageItem.lut
        np.random.shuffle(lut)
        lut[0] = [0, 0, 0, 0]
        self.imageItem.setLookupTable(lut)
        self.imageItem.update()

    def updateImageLut(self, gradientItem):
        lut = np.zeros((255, 4), dtype=np.uint8)
        lut[:, -1] = 255
        lut[:, :-1] = self.item.colorMap().getLookupTable(0, 1, 255)
        np.random.shuffle(lut)
        lut[0] = [0, 0, 0, 0]
        self.imageItem.setLookupTable(lut)
        self.imageItem.setLevels([0, 255])

    def updateImageOpacity(self, value):
        self.imageItem.setOpacity(value)


class labelsGradientWidget(pg.GradientWidget):
    sigShowRightImgToggled = Signal(bool)
    sigShowLabelsImgToggled = Signal(bool)
    sigShowNextFrameToggled = Signal(bool)

    def __init__(self, *args, parent=None, orientation="right", **kargs):
        pg.GradientEditorItem = BaseGradientEditorItemLabels

        pg.GradientWidget.__init__(
            self, *args, parent=parent, orientation=orientation, **kargs
        )

        self._parent = parent
        self.name = "labels"

        for action in self.menu.actions():
            if action.text() == "HSV":
                HSV_action = action
            elif action.text() == "RGB":
                RGB_ation = action
        self.menu.removeAction(HSV_action)
        self.menu.removeAction(RGB_ation)

        # Add custom colormap action
        self.customCmapsMenu = self.menu.addMenu("Custom colormaps")
        self.customCmapsMenu.aboutToShow.connect(self.onShowCustomCmapsMenu)
        self.customCmapsMenu.triggered.connect(self.customCmapsMenuTriggered)

        self.saveColormapAction = QAction("Save current colormap...", self)
        self.menu.addAction(self.saveColormapAction)
        self.saveColormapAction.triggered.connect(self.saveColormap)

        self.addCustomGradients()

        # Background color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Background color: "))
        self.colorButton = myColorButton(color=(25, 25, 25))
        hbox.addStretch(1)
        hbox.addWidget(self.colorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.colorButton.click)
        self.menu.addAction(act)

        # Font size menu action
        self.fontSizeMenu = QMenu("Text font size", self)
        self.menu.addMenu(self.fontSizeMenu)

        # IDs color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Text color: "))
        self.textColorButton = myColorButton(color=(25, 25, 25))
        hbox.addStretch(1)
        hbox.addWidget(self.textColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.textColorButton.click)
        self.menu.addAction(act)
        self.menu.addSeparator()

        # Shuffle colors action
        self.shuffleCmapAction = QAction("Randomly shuffle colormap   (Shift+S)", self)
        self.menu.addAction(self.shuffleCmapAction)

        self.greedyShuffleCmapAction = QAction(
            "Greedily shuffle colormap  (Alt+Shift+S)", self
        )
        self.menu.addAction(self.greedyShuffleCmapAction)

        self.permanentGreedyCmapAction = QAction("Always use greedy colormap", self)
        self.permanentGreedyCmapAction.setCheckable(True)
        self.menu.addAction(self.permanentGreedyCmapAction)

        # Invert bw action
        self.invertBwAction = QAction("Invert black/white", self)
        self.invertBwAction.setCheckable(True)
        self.menu.addAction(self.invertBwAction)

        # Show labels action
        self.showLabelsImgAction = QAction("Show segmentation image", self)
        self.showLabelsImgAction.setCheckable(True)
        self.menu.addAction(self.showLabelsImgAction)

        # Show right image action
        self.showRightImgAction = QAction("Show duplicated left image", self)
        self.showRightImgAction.setCheckable(True)
        self.menu.addAction(self.showRightImgAction)

        # Show next frame action
        self.showNextFrameAction = QAction("Show next frame", self)
        self.showNextFrameAction.setCheckable(True)
        self.menu.addAction(self.showNextFrameAction)

        # Default settings
        self.defaultSettingsAction = QAction("Restore default settings...", self)
        self.menu.addAction(self.defaultSettingsAction)

        self.menu.addSeparator()

        self.showRightImgAction.toggled.connect(self.showRightImageToggled)
        self.showLabelsImgAction.toggled.connect(self.showLabelsImageToggled)
        self.showNextFrameAction.toggled.connect(self.showNextFrameToggled)

    def onShowCustomCmapsMenu(self):
        self.customCmapsMenu.show()

    def customCmapsMenuTriggered(self, action):
        cmap = action.cmap
        self.item.colorMapMenuClicked(cmap)
        self.item.showTicks(True)

    def addCustomGradient(self, gradient_name, gradient_ticks, restore=True):
        currentState = self.item.saveState()
        self.originalLength = self.item.length
        self.item.length = 100
        if restore:
            self.item.restoreState(gradient_ticks)
        gradient = self.item.getGradient()
        action = CustomGradientMenuAction(gradient, gradient_name, self.item)
        # action.triggered.connect(self.item.contextMenuClicked)
        action.delButton.clicked.connect(self.removeCustomGradient)
        action.cmap = colors.pg_ticks_to_colormap(gradient_ticks["ticks"])
        # self.item.menu.insertAction(self.saveColormapAction, action)
        self.customCmapsMenu.addAction(action)
        self.item.length = self.originalLength
        self.item.restoreState(currentState)
        GradientsLabels[gradient_name] = gradient_ticks

    def removeCustomGradient(self):
        button = self.sender()
        action = button.action
        self.customCmapsMenu.removeAction(action)
        cp = config.ConfigParser()
        cp.read(custom_cmaps_filepath)
        cp.remove_section(f"labels.{action.name}")
        with open(custom_cmaps_filepath, mode="w") as file:
            cp.write(file)

    def addCustomGradients(self):
        try:
            CustomGradients = getCustomGradients(name="labels")
            if not CustomGradients:
                return
            for gradient_name, gradient_ticks in CustomGradients.items():
                self.addCustomGradient(gradient_name, gradient_ticks)
        except Exception as e:
            printl(traceback.format_exc())
            pass

    def _askNameColormap(self):
        inputWin = apps.QInput(parent=self._parent, title="Colormap name")
        inputWin.askText("Insert a name for the colormap: ", allowEmpty=False)
        if inputWin.cancel:
            return
        cmapName = inputWin.answer
        return cmapName

    def saveColormap(self):
        cmapName = self._askNameColormap()
        if cmapName is None:
            return

        cp = config.ConfigParser()
        if os.path.exists(custom_cmaps_filepath):
            cp.read(custom_cmaps_filepath)

        SECTION = f"{self.name}.{cmapName}"
        cp[SECTION] = {}

        state = self.item.saveState()
        for key, value in state.items():
            if key != "ticks":
                continue
            for t, tick in enumerate(value):
                pos, rgb = tick
                rgb = ",".join([str(c) for c in rgb])
                val = f"{pos},{rgb}"
                cp[SECTION][f"tick_{t}_pos_rgb"] = val

        with open(custom_cmaps_filepath, mode="w") as file:
            cp.write(file)

        self.addCustomGradient(cmapName, state, restore=False)

    def isRightImageVisible(self):
        return (
            self.showLabelsImgAction.isChecked() or self.showNextFrameAction.isChecked()
        )

    def showRightImageToggled(self, checked):
        if checked and self.isRightImageVisible():
            # Hide the right labels image before showing right image
            self.showLabelsImgAction.setChecked(False)
            self.showNextFrameAction.setChecked(False)
            self.sigShowLabelsImgToggled.emit(False)
            self.sigShowNextFrameToggled.emit(checked)
        self.sigShowRightImgToggled.emit(checked)

    def showLabelsImageToggled(self, checked):
        if checked and self.isRightImageVisible():
            # Hide the right image before showing labels image
            self.showRightImgAction.setChecked(False)
            self.showNextFrameAction.setChecked(False)
            self.sigShowRightImgToggled.emit(False)
            self.sigShowNextFrameToggled.emit(False)
        self.sigShowLabelsImgToggled.emit(checked)

    def showNextFrameToggled(self, checked):
        if checked and self.isRightImageVisible():
            # Hide the right image before showing labels image
            self.showRightImgAction.setChecked(False)
            self.showLabelsImgAction.setChecked(False)
            self.sigShowRightImgToggled.emit(False)
            self.sigShowLabelsImgToggled.emit(False)
        self.sigShowNextFrameToggled.emit(checked)

    def saveState(self, df):
        # remove previous state
        df = df[~df.index.str.contains("lab_cmap")].copy()

        state = self.item.saveState()
        for key, value in state.items():
            if key == "ticks":
                for t, tick in enumerate(value):
                    pos, rgb = tick
                    df.at[f"lab_cmap_tick{t}_rgb", "value"] = rgb
                    df.at[f"lab_cmap_tick{t}_pos", "value"] = pos
            else:
                if isinstance(value, bool):
                    value = "Yes" if value else "No"
                df.at[f"lab_cmap_{key}", "value"] = value
        return df

    def restoreState(self, df, loadCmap=True):
        # Insert background color
        if "labels_bkgrColor" in df.index:
            rgbString = df.at["labels_bkgrColor", "value"]
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.colorButton.setColor((r, g, b))

        if "labels_text_color" in df.index:
            rgbString = df.at["labels_text_color", "value"]
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.textColorButton.setColor((r, g, b))
        else:
            self.textColorButton.setColor((255, 0, 0))

        checked = df.at["is_bw_inverted", "value"] == "Yes"
        self.invertBwAction.setChecked(checked)

        if not loadCmap:
            return

        state = {"mode": "rgb", "ticksVisible": True, "ticks": []}
        ticks_pos = {}
        ticks_rgb = {}
        stateFound = False
        for setting, value in df.itertuples():
            idx = setting.find("lab_cmap_")
            if idx == -1:
                continue

            stateFound = True
            m = re.findall(r"tick(\d+)_(\w+)", setting)
            if m:
                tick_idx, tick_type = m[0]
                if tick_type == "pos":
                    ticks_pos[int(tick_idx)] = float(value)
                elif tick_type == "rgb":
                    ticks_rgb[int(tick_idx)] = colors.rgba_str_to_values(value)
            else:
                key = setting[9:]
                if value == "Yes":
                    value = True
                elif value == "No":
                    value = False
                state[key] = value

        if stateFound:
            ticks = [(0, 0)] * len(ticks_pos)
            for idx, val in ticks_pos.items():
                pos = val
                rgb = ticks_rgb[idx]
                ticks[idx] = (pos, rgb)

            state["ticks"] = ticks
            self.item.restoreState(state)
        else:
            self.item.loadPreset("viridis")

        return stateFound

    def showMenu(self, ev):
        try:
            # Convert QPointF to QPoint
            self.menu.popup(ev.screenPos().toPoint())
        except AttributeError:
            self.menu.popup(ev.screenPos())

# Cross-module imports (deferred to avoid import cycles)
from .scrollbars import (
    sliderWithSpinBox,
)
from ..controls.inputs import (
    highlightableQWidgetAction,
)

