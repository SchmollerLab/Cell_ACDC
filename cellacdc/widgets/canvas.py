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

class ContourItem(pg.PlotCurveItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self._prevData = None

    def clear(self):
        try:
            self.setData([], [])
        except AttributeError as e:
            pass

    def tempClear(self):
        try:
            self._prevData = [d.copy() for d in self.getData()]
            self.clear()
        except Exception as e:
            pass

    def restore(self):
        if self._prevData is not None:
            if self._prevData[0] is not None:
                self.setData(*self._prevData)


class BaseScatterPlotItem(pg.ScatterPlotItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def tempClear(self):
        try:
            self._prevData = [d.copy() for d in self.getData()]
            self.setData([], [])
        except Exception as e:
            pass

    def restore(self):
        if self._prevData is not None:
            if self._prevData[0] is not None:
                self.setData(*self._prevData)


class CustomAnnotationScatterPlotItem(BaseScatterPlotItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)


class ScatterPlotItem(pg.ScatterPlotItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updateBrushAndPen(**kwargs)

    def updateBrushAndPen(self, **kwargs):
        brush = kwargs.get("brush")
        if brush is not None:
            self._itemBrush = brush
        pen = kwargs.get("pen")
        if pen is not None:
            self._itemPen = pen

    def setData(self, *args, **kwargs):
        super().setData(*args, **kwargs)
        self.updateBrushAndPen(**kwargs)

    def itemBrush(self):
        return self._itemBrush

    def itemPen(self):
        return self._itemPen

    def removePoint(self, index):
        newData = np.delete(self.data, index)
        # Update the index of current points
        for i in range(index, len(newData)):
            spotItem = newData[i]["item"]
            spotItem._index = i
            newData[i]["item"] = spotItem

        self.data = newData
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.bounds = [None, None]
        self.invalidate()
        self.updateSpots(newData)
        self.sigPlotChanged.emit(self)

    def coordsToNumpy(self, includeData=False, rounded=True, decimals=None):
        points = self.points()
        nrows = len(points)
        coords_arr = np.zeros((nrows, 2))
        data_arr = None
        for p, point in enumerate(points):
            pos = point.pos()
            x, y = pos.x(), pos.y()
            if includeData:
                data = point.data()
                if data_arr is None:
                    try:
                        ncols = len(data)
                    except Exception as e:
                        data = [data]
                        ncols = 1
                    data_arr = np.zeros((nrows, ncols))
                for j, data_j in enumerate(data):
                    data_arr[p, j] = data_j

            coords_arr[p, 0] = y
            coords_arr[p, 1] = x
        if not includeData:
            out_arr = coords_arr
        elif data_arr is not None:
            out_arr = np.column_stack((data_arr, coords_arr))
        else:
            out_arr = coords_arr
        cast_to_int = decimals is None
        decimals = decimals if decimals is not None else 0
        if rounded:
            out_arr = np.round(out_arr, decimals)
        if cast_to_int:
            out_arr = out_arr.astype(int)
        return out_arr


class myLabelItem(pg.LabelItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prevText = ""

    def setText(self, text, **args):
        self.text = text
        opts = self.opts
        for k in args:
            opts[k] = args[k]

        if "size" in self.opts:
            size = self.opts["size"]
            if size == "0pt" or size == "0px":
                self.opts["size"] = "1pt"
                super().setText("", size="1pt")
                return

        optlist = []

        color = self.opts["color"]
        if color is None:
            color = pg.getConfigOption("foreground")
        color = pg.functions.mkColor(color)
        optlist.append("color: " + color.name(QColor.NameFormat.HexArgb))
        if "size" in opts:
            size = opts["size"]
            if not isinstance(size, str):
                size = f"{size}px"
            optlist.append("font-size: " + size)
        if "bold" in opts and opts["bold"] in [True, False]:
            optlist.append(
                "font-weight: " + {True: "bold", False: "normal"}[opts["bold"]]
            )
        if "italic" in opts and opts["italic"] in [True, False]:
            optlist.append(
                "font-style: " + {True: "italic", False: "normal"}[opts["italic"]]
            )
        full = "<span style='%s'>%s</span>" % ("; ".join(optlist), text)
        # print full
        self.item.setHtml(full)
        self.updateMin()
        self.resizeEvent(None)
        self.updateGeometry()

    def tempClearText(self):
        if self.text:
            self._prevText = self.text
            self.setText("")

    def restoreText(self):
        if self._prevText:
            self.setText(self._prevText)


class LabelRoiCircularItem(pg.ScatterPlotItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def setImageShape(self, shape):
        self._shape = shape

    def slice(self, zRange=None, tRange=None):
        self.mask()
        if zRange is None:
            _slice = self._slice
        else:
            zmin, zmax = zRange
            _slice = (slice(zmin, zmax), *self._slice)

        if tRange is not None:
            tmin, tmax = tRange
            _slice = (slice(tmin, tmax), *_slice)

        return _slice

    def mask(self):
        shape = self._shape
        radius = int(self.opts["size"] / 2)
        mask = skimage.morphology.disk(radius, dtype=bool)
        xx, yy = self.getData()
        Yc, Xc = yy[0], xx[0]
        mask, self._slice = myutils.clipSelemMask(mask, shape, Yc, Xc, copy=False)
        return mask


class PolyLineROI(pg.PolyLineROI):
    def __init__(self, positions, closed=False, pos=None, **args):
        super().__init__(positions, closed, pos, **args)


class BaseGradientEditorItemImage(pg.GradientEditorItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def restoreState(self, state):
        pg.graphicsItems.GradientEditorItem.Gradients = GradientsImage
        return super().restoreState(state)


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


class ROI(pg.ROI):
    def __init__(
        self,
        pos,
        size=pg.Point(1, 1),
        angle=0,
        invertible=False,
        maxBounds=None,
        snapSize=1,
        scaleSnap=False,
        translateSnap=False,
        rotateSnap=False,
        parent=None,
        pen=None,
        hoverPen=None,
        handlePen=None,
        handleHoverPen=None,
        movable=True,
        rotatable=True,
        resizable=True,
        removable=False,
        aspectLocked=False,
    ):
        super().__init__(
            pos,
            size,
            angle,
            invertible,
            maxBounds,
            snapSize,
            scaleSnap,
            translateSnap,
            rotateSnap,
            parent,
            pen,
            hoverPen,
            handlePen,
            handleHoverPen,
            movable,
            rotatable,
            resizable,
            removable,
            aspectLocked,
        )

    def slice(self, zRange=None, tRange=None):
        x0, y0 = [int(round(c)) for c in self.pos()]
        w, h = [int(round(c)) for c in self.size()]
        xmin, xmax = x0, x0 + w
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        ymin, ymax = y0, y0 + h
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if zRange is not None:
            zmin, zmax = zRange
            _slice = (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))
        else:
            _slice = (slice(ymin, ymax), slice(xmin, xmax))
        if tRange is not None:
            tmin, tmax = tRange
            _slice = (slice(tmin, tmax), *_slice)
        return _slice

    def bbox(self):
        x0, y0 = [int(round(c)) for c in self.pos()]
        w, h = [int(round(c)) for c in self.size()]
        xmin, xmax = x0, x0 + w
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        ymin, ymax = y0, y0 + h
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        return ymin, xmin, ymax, xmax


class ZoomROI(ROI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.viewRangesQueue = deque()

    def getLastRange(self):
        xRange, yRange = self.viewRangesQueue.pop()
        return xRange, yRange

    def storeLastRange(self, xRange, yRange):
        self.viewRangesQueue.append((xRange, yRange))


class DelROI(pg.ROI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def clearPoints(self):
        """
        Remove all handles and segments.
        """
        while len(self.handles) > 0:
            self.removeHandle(self.handles[0]["item"])


class PlotCurveItem(pg.PlotCurveItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def addPoint(self, x, y, **kargs):
        _xx, _yy = self.getData()
        if _xx is None or len(_xx) == 0:
            self.xData = np.array([x], dtype=int)
            self.yData = np.array([y], dtype=int)
            return
        if _xx[-1] == x and _yy[-1] == y:
            # Do not append same point
            return

        # Pre-allocate array and insert data (faster than append)
        xx = np.zeros(len(_xx) + 1, dtype=_xx.dtype)
        xx[:-1] = _xx
        xx[-1] = x
        yy = np.zeros(len(_yy) + 1, dtype=_xx.dtype)
        yy[:-1] = _yy
        yy[-1] = y
        self.setData(xx, yy, **kargs)

    def clear(self):
        try:
            self.setData([], [])
        except Exception as e:
            pass
        super().clear()

    def closeCurve(self):
        _xx, _yy = self.getData()
        self.addPoint(_xx[0], _yy[0])

    def mask(self):
        ymin, xmin, ymax, xmax = self.bbox()
        _mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=bool)
        local_xx, local_yy = self.getLocalData()
        rr, cc = skimage.draw.polygon(local_yy, local_xx)
        _mask[rr, cc] = True
        return _mask

    def getLocalData(self):
        _xx, _yy = self.getData()
        return _xx - _xx.min(), _yy - _yy.min()

    def slice(self, zRange=None, tRange=None):
        ymin, xmin, ymax, xmax = self.bbox()
        if zRange is not None:
            zmin, zmax = zRange
            _slice = (slice(zmin, zmax), slice(ymin, ymax + 1), slice(xmin, xmax + 1))
        else:
            _slice = (slice(ymin, ymax + 1), slice(xmin, xmax + 1))
        if tRange is not None:
            tmin, tmax = tRange
            _slice = (slice(tmin, tmax), *_slice)
        return _slice

    def bbox(self):
        _xx, _yy = self.getData()
        return _yy.min(), _xx.min(), _yy.max(), _xx.max()


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


class MainPlotItem(pg.PlotItem):
    def __init__(
        self,
        parent=None,
        name=None,
        labels=None,
        title=None,
        viewBox=None,
        axisItems=None,
        enableMenu=True,
        showWelcomeText=False,
        **kargs,
    ):
        super().__init__(
            parent, name, labels, title, viewBox, axisItems, enableMenu, **kargs
        )
        # Overwrite zoom out button behaviour to disable autoRange after
        # clicking it.
        # If autorange is enabled, it is called everytime the brush or eraser
        # scatter plot items touches the border causing flickering
        self.disableAutoRange()
        self.autoBtn.mode = "manual"
        if showWelcomeText:
            self.infoTextItem = pg.TextItem()
            self.addItem(self.infoTextItem)
            html_filepath = os.path.join(html_path, "gui_welcome.html")
            with open(html_filepath) as html_file:
                htmlText = html_file.read()
            self.infoTextItem.setHtml(htmlText)
            self.infoTextItem.setPos(0, 0)

        self.delRoiItems = {}
        self.highlightingRectItems = None
        self._baseImageItem = None
        self._imageItems = []
        self.highlightingRectItemsColor = None

    def addHighlightingRectItems(self, color=None):
        self.highlightingRectItems = {
            "left": RectItem(QRectF()),
            "right": RectItem(QRectF()),
            "top": RectItem(QRectF()),
            "bottom": RectItem(QRectF()),
        }
        for rect in self.highlightingRectItems.values():
            self.addItem(rect)

        if color is None:
            return

        self.setHighlightingRectItemsColor(color)

    def setHighlightingRectItemsColor(self, color):
        if color == self.highlightingRectItemsColor:
            return

        for item in self.highlightingRectItems.values():
            item.setColor(color)

        self.highlightingRectItemsColor = color

    def addBaseImageItem(self, baseImageItem):
        self._baseImageItem = baseImageItem
        self._imageItems.append(baseImageItem)
        self.addItem(baseImageItem)

    def addImageItem(self, imageItem):
        self._imageItems.append(imageItem)
        self.addItem(imageItem)

    def setHighlighted(self, highlighted, color=None):
        if color is None:
            color = self.highlightingRectItemsColor

        if color is None:
            color = "green"

        if self.highlightingRectItems is None:
            self.addHighlightingRectItems(color=color)

        if not highlighted:
            for rect in self.highlightingRectItems.values():
                rect.setQRect(QRectF())
            return

        self.setHighlightingRectItemsColor(color)

        ((xmin, xmax), (ymin, ymax)) = self.viewRange()
        xmin = xmin if xmin >= 0 else 0
        ymin = ymin if ymin >= 0 else 0
        if self._baseImageItem is not None:
            Y, X = self._baseImageItem.image.shape[:2]
            xmax = min(xmax, X)
            ymax = min(ymax, Y)

        w = xmax - xmin
        h = ymax - ymin

        bs = round(((w + h) / 2) * 0.02)
        if bs < 1:
            bs = 1

        x0 = xmin
        x1 = xmin + bs
        x2 = xmax - bs
        x3 = xmax

        y0 = ymin
        y1 = ymin + bs
        y2 = ymax - bs
        y3 = ymax

        self.highlightingRectItems["left"].setRect(x0, y0, bs, y3 - y0)
        self.highlightingRectItems["top"].setRect(x1, y0, x3 - x1, bs)
        self.highlightingRectItems["right"].setRect(x2, y1, bs, y3 - y1)
        self.highlightingRectItems["bottom"].setRect(x1, y2, x2 - x1, bs)
        self.update()

    def clear(self):
        super().clear()

        self.delRoiItems = {}
        self.highlightingRectItems = None
        self._baseImageItem = None
        self._imageItems = []
        self.highlightingRectItemsColor = None

        try:
            self.removeItem(self.infoTextItem)
        except Exception as e:
            pass

    def autoBtnClicked(self):
        self.vb.autoRange()
        self.autoBtn.hide()

    def addDelRoiItem(self, roiItem, key):
        if self.isDelRoiItemPresent(roiItem):
            return

        self.delRoiItems[key] = roiItem
        roiItem.key = key
        self.addItem(roiItem)

    def removeDelRoiItem(self, roiItem):
        key = roiItem.key
        self.delRoiItems.pop(key, None)
        try:
            self.removeItem(roiItem)
        except Exception as err:
            return

    def isDelRoiItemPresent(self, roiItem):
        try:
            key = roiItem.key
        except AttributeError as e:
            return False

        try:
            roi = self.delRoiItems[key]
        except Exception as err:
            return False

        return True

    def viewRange(self, mask_img=None):
        if mask_img is None:
            return super().viewRange()

        mask_rp = skimage.measure.regionprops(skimage.measure.label(mask_img))
        if not mask_rp:
            return super().viewRange()

        mask_obj = mask_rp[0]
        ymin, xmin, ymax, xmax = mask_obj.bbox
        return (xmin, xmax), (ymin, ymax)


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


class BaseImageItem(pg.ImageItem):
    def __init__(self, image=None, **kargs):
        self.minMaxValuesMapper = None
        self.minMaxValuesMapperPreproc = None
        self.minMaxValuesMapperCombined = None
        self.minMaxValuesMapperEqualized = None
        self.pos_i = 0
        self.z = 0
        self.frame_i = 0
        self.usePreprocessed = False
        self.useEqualized = False
        self.useCombined = False
        self._isRgba = False

        super().__init__(image, **kargs)
        self.autoLevelsEnabled = None

    def isRgba(self):
        return self._isRgba

    def setEnableAutoLevels(self, enabled: bool):
        self.autoLevelsEnabled = enabled

    def setImage(self, image=None, autoLevels=None, **kargs):
        if autoLevels is None:
            autoLevels = self.autoLevelsEnabled

        if image is not None and image.ndim == 3 and image.shape[2] in (3, 4):
            self._isRgba = True

        super().setImage(image, autoLevels=autoLevels, **kargs)

    def preComputedMinMaxValues(self, data: List["load.loadData"]):
        self.minMaxValuesMapper = {}
        for pos_i, posData in enumerate(data):
            img_data = posData.img_data
            requires_time_dim = posData.img_data.ndim == 2 or (
                posData.img_data.ndim == 3 and posData.SizeZ > 1
            )
            if requires_time_dim:
                img_data = (img_data,)

            for frame_i, image in enumerate(img_data):
                if image.ndim == 3:
                    self._updateMinMaxValuesProjections(
                        image, pos_i, frame_i, self.minMaxValuesMapper
                    )

                if image.ndim == 2:
                    image = (image,)

                for z, img in enumerate(image):
                    self.minMaxValuesMapper[(pos_i, frame_i, z)] = (
                        np.nanmin(img),
                        np.nanmax(img),
                    )

    def updateMinMaxValuesEqualizedData(
        self,
        data: List["load.loadData"],
        pos_i: int,
        frame_i: int,
        z_slice: Union[int, str],
    ):
        if self.minMaxValuesMapperEqualized is None:
            self.minMaxValuesMapperEqualized = {}

        posData = data[pos_i]
        img = posData.equalized_img_data[frame_i][z_slice]
        key = (pos_i, frame_i, z_slice)
        self.minMaxValuesMapperEqualized[key] = (np.nanmin(img), np.nanmax(img))

    def updateMinMaxValuesEqualizedDataProjections(
        self,
        data: List["load.loadData"],
        pos_i: int,
        frame_i: int,
    ):
        posData = data[pos_i]
        eq_zstack = posData.equalized_img_data[frame_i]

        self._updateMinMaxValuesProjections(
            eq_zstack, pos_i, frame_i, self.minMaxValuesMapperEqualized
        )

    def _updateMinMaxValuesProjections(self, zstack, pos_i, frame_i, mapper):
        max_proj = zstack.max(axis=0)
        key = (pos_i, frame_i, "max z-projection")
        mapper[key] = np.nanmin(max_proj), np.nanmax(max_proj)

        mean_proj = zstack.mean(axis=0)
        key = (pos_i, frame_i, "mean z-projection")
        mapper[key] = np.nanmin(mean_proj), np.nanmax(mean_proj)

        median_proj = np.median(zstack, axis=0)
        key = (pos_i, frame_i, "median z-proj.")
        mapper[key] = np.nanmin(median_proj), np.nanmax(median_proj)

    def updateMinMaxValuesPreprocessedData(
        self,
        data: List["load.loadData"],
        pos_i: int,
        frame_i: int,
        z_slice: Union[int, str],
    ):
        if self.minMaxValuesMapperPreproc is None:
            self.minMaxValuesMapperPreproc = {}

        posData = data[pos_i]
        img = posData.preproc_img_data[frame_i][z_slice]
        key = (pos_i, frame_i, z_slice)
        self.minMaxValuesMapperPreproc[key] = (np.nanmin(img), np.nanmax(img))

    def updateMinMaxValuesPreprocessedProjections(
        self,
        data: List["load.loadData"],
        pos_i: int,
        frame_i: int,
    ):
        posData = data[pos_i]
        zstack = posData.preproc_img_data[frame_i]

        self._updateMinMaxValuesProjections(
            zstack, pos_i, frame_i, self.minMaxValuesMapperPreproc
        )

    def updateMinMaxValuesCombinedData(
        self,
        data: List["load.loadData"],
        pos_i: int,
        frame_i: int,
        z_slice: Union[int, str],
    ):
        if self.minMaxValuesMapperCombined is None:
            self.minMaxValuesMapperCombined = {}

        posData = data[pos_i]
        img = posData.combine_img_data[frame_i][z_slice]
        key = (pos_i, frame_i, z_slice)
        self.minMaxValuesMapperCombined[key] = (np.nanmin(img), np.nanmax(img))

    def updateMinMaxValuesCombinedDataProjections(
        self,
        data: List["load.loadData"],
        pos_i: int,
        frame_i: int,
    ):
        posData = data[pos_i]
        zstack = posData.combine_img_data[frame_i]

        self._updateMinMaxValuesProjections(
            zstack, pos_i, frame_i, self.minMaxValuesMapperCombined
        )

    def setCurrentPosIndex(self, pos_i: int):
        self.pos_i = pos_i

    def setCurrentFrameIndex(self, frame_i: int):
        self.frame_i = frame_i

    def setCurrentZsliceIndex(self, z: int):
        self.z = z

    def quickMinMax(self, targetSize=1e6):
        if self.isRgba():
            return super().quickMinMax(targetSize=targetSize)

        if self.usePreprocessed and self.minMaxValuesMapperPreproc is not None:
            minMaxValuesMapper = self.minMaxValuesMapperPreproc
        elif self.useCombined and self.minMaxValuesMapperCombined is not None:
            minMaxValuesMapper = self.minMaxValuesMapperCombined
        elif self.useEqualized and self.minMaxValuesMapperEqualized is not None:
            minMaxValuesMapper = self.minMaxValuesMapperEqualized
        else:
            minMaxValuesMapper = self.minMaxValuesMapper

        if minMaxValuesMapper is None:
            return super().quickMinMax(targetSize=targetSize)

        try:
            key = (self.pos_i, self.frame_i, self.z)
            levels = minMaxValuesMapper[key]
            return levels
        except Exception as err:
            pass

        try:
            key = (self.pos_i, self.frame_i, self.z)
            levels = self.minMaxValuesMapper[key]
            return levels
        except Exception as err:
            return super().quickMinMax(targetSize=targetSize)

    def setOpacity(self, value, **kwargs):
        if value == 0:
            value = 0.001

        if value == 1:
            value = 0.999

        super().setOpacity(value)


class BaseLabelsImageItem(pg.ImageItem):
    def __init__(self, image=None, **kargs):
        super().__init__(image, **kargs)

    def setImage(self, image=None, **kwargs):
        if image is None:
            return
        autoLevels = kwargs.get("autoLevels")
        if autoLevels is None:
            kwargs["autoLevels"] = False
        super().setImage(image, **kwargs)


class OverlayImageItem(pg.ImageItem):
    def __init__(self, image=None, **kargs):
        super().__init__(image, **kargs)
        self.autoLevelsEnabled = None

    def setEnableAutoLevels(self, enabled: bool):
        self.autoLevelsEnabled = enabled

    def setImage(self, image=None, autoLevels=None, **kargs):
        if autoLevels is None:
            autoLevels = self.autoLevelsEnabled

        super().setImage(image, autoLevels=autoLevels, **kargs)

    def setOpacity(self, value, **kwargs):
        if value == 0:
            value = 0.001

        if value == 1:
            value = 0.999

        super().setOpacity(value)


class ParentImageItem(BaseImageItem):
    def __init__(
        self,
        image=None,
        linkedImageItem=None,
        activatingActions=None,
        debug=False,
        **kargs,
    ):
        super().__init__(image, **kargs)
        self.linkedImageItem = linkedImageItem
        self.activatingActions = activatingActions
        self.debug = debug
        self._forceDoNotUpdateLinked = False
        self.autoLevelsEnabled = None

    def clear(self):
        if self.linkedImageItem is not None:
            self.linkedImageItem.clear()
        return super().clear()

    def isLinkedImageItemActive(self):
        if self._forceDoNotUpdateLinked:
            return False

        if self.linkedImageItem is None:
            return False

        if self.activatingActions is None:
            return False

        for action in self.activatingActions:
            if action.isChecked():
                return True

        return False

    def setEnableAutoLevels(self, enabled: bool):
        self.autoLevelsEnabled = enabled

    def setUsePreprocessed(self, usePreprocessed):
        self.usePreprocessed = usePreprocessed
        if self.linkedImageItem is None:
            return

        self.linkedImageItem.usePreprocessed = usePreprocessed

    def setUseCombined(self, useCombined):
        self.useCombined = useCombined
        if self.linkedImageItem is None:
            return

        self.linkedImageItem.useCombined = useCombined

    def preComputedMinMaxValues(self, *args, **kwargs):
        super().preComputedMinMaxValues(*args, **kwargs)
        if self.linkedImageItem is None:
            return

        self.linkedImageItem.minMaxValuesMapper = self.minMaxValuesMapper

    def updateMinMaxValuesPreprocessedData(self, *args, **kwargs):
        super().updateMinMaxValuesPreprocessedData(*args, **kwargs)

        if self.linkedImageItem is None:
            return

        self.linkedImageItem.minMaxValuesMapper = self.minMaxValuesMapper

    def updateMinMaxValuesCombinedData(self, *args, **kwargs):
        super().updateMinMaxValuesCombinedData(*args, **kwargs)

        if self.linkedImageItem is None:
            return

        self.linkedImageItem.minMaxValuesMapperCombined = (
            self.minMaxValuesMapperCombined
        )

    def updateMinMaxValuesCombinedDataProjections(self, *args, **kwargs):
        super().updateMinMaxValuesCombinedDataProjections(*args, **kwargs)

        if self.linkedImageItem is None:
            return

        self.linkedImageItem.minMaxValuesMapperCombined = (
            self.minMaxValuesMapperCombined
        )

    def updateMinMaxValuesEqualizedDataProjections(self, *args, **kwargs):
        super().updateMinMaxValuesEqualizedDataProjections(*args, **kwargs)

        if self.linkedImageItem is None:
            return

        self.linkedImageItem.minMaxValuesMapperEqualized = (
            self.minMaxValuesMapperEqualized
        )

    def updateMinMaxValuesEqualizedData(self, *args, **kwargs):
        super().updateMinMaxValuesEqualizedData(*args, **kwargs)

        if self.linkedImageItem is None:
            return

        self.linkedImageItem.minMaxValuesMapperEqualized = (
            self.minMaxValuesMapperEqualized
        )

    def setCurrentPosIndex(self, *args, **kwargs):
        super().setCurrentPosIndex(*args, **kwargs)

        if self.linkedImageItem is None:
            return

        self.linkedImageItem.pos_i = self.pos_i

    def setCurrentFrameIndex(self, *args, **kwargs):
        super().setCurrentFrameIndex(*args, **kwargs)

        if self.linkedImageItem is None:
            return

        self.linkedImageItem.frame_i = self.frame_i + 1

    def setCurrentZsliceIndex(self, *args, **kwargs):
        super().setCurrentZsliceIndex(*args, **kwargs)

        if self.linkedImageItem is None:
            return

        self.linkedImageItem.z = self.z

    def setImage(
        self,
        image=None,
        autoLevels=None,
        next_frame_image=None,
        scrollbar_value=None,
        force_set_linked=False,
        **kargs,
    ):
        if autoLevels is None:
            autoLevels = self.autoLevelsEnabled

        super().setImage(image, autoLevels=autoLevels, **kargs)

        if self.linkedImageItem is None:
            return

        if not self.isLinkedImageItemActive() and not force_set_linked:
            return

        if next_frame_image is not None:
            self.linkedImageItem.setImage(
                next_frame_image, scrollbar_value=scrollbar_value, autoLevels=autoLevels
            )
        elif image is not None:
            self.linkedImageItem.setImage(image)

    def updateImage(self, *args, **kargs):
        if self.isLinkedImageItemActive():
            self.linkedImageItem.image = self.image
            self.linkedImageItem.updateImage(*args, **kargs)
        return super().updateImage(*args, **kargs)

    def setOpacity(self, value, applyToLinked=True):
        super().setOpacity(value)
        if not applyToLinked:
            return

        if self.linkedImageItem is None:
            return

        self.linkedImageItem.setOpacity(value)

    def setLookupTable(self, lut):
        super().setLookupTable(lut)


class ChildImageItem(BaseImageItem):
    def __init__(self, *args, linkedScrollbar=None, **kwargs):
        BaseImageItem.__init__(self, *args, **kwargs)
        self.linkedScrollbar = linkedScrollbar

    def setImage(self, img=None, z=None, scrollbar_value=None, **kargs):
        autoLevels = kargs.get("autoLevels")
        if autoLevels is None:
            kargs["autoLevels"] = False

        if img is None:
            BaseImageItem.setImage(self, img, **kargs)
            return

        if img.ndim == 3 and img.shape[-1] > 4 and z is not None:
            BaseImageItem.setImage(self, img[z], **kargs)
        else:
            BaseImageItem.setImage(self, img, **kargs)

        if self.linkedScrollbar is None:
            return

        if not self.linkedScrollbar.isEnabled():
            return

        if scrollbar_value is None:
            return

        self.linkedScrollbar.setValueNoSignal(scrollbar_value)


class labImageItem(pg.ImageItem):
    def __init__(self, *args, **kwargs):
        pg.ImageItem.__init__(self, *args, **kwargs)

    def setImage(self, img=None, z=None, **kargs):
        autoLevels = kargs.get("autoLevels")
        if autoLevels is None:
            kargs["autoLevels"] = False

        if img is None:
            pg.ImageItem.setImage(self, img, **kargs)
            return

        if img.ndim == 3 and img.shape[-1] > 4 and z is not None:
            pg.ImageItem.setImage(self, img[z], **kargs)
        else:
            pg.ImageItem.setImage(self, img, **kargs)


class GhostContourItem(pg.PlotDataItem):
    def __init__(
        self, ParentPlotItem, penColor=(245, 184, 0, 100), textColor=(245, 184, 0)
    ):
        super().__init__()
        # Yellow pen
        self.setPen(pg.mkPen(width=2, color=penColor))
        self.label = myLabelItem()
        self.label.setAttr("bold", True)
        self.label.setAttr("color", textColor)
        self._ParentPlotItem = ParentPlotItem

    def addToPlotItem(self):
        self._ParentPlotItem.addItem(self)
        self._ParentPlotItem.addItem(self.label)

    def removeFromPlotItem(self):
        self._ParentPlotItem.removeItem(self.label)
        self._ParentPlotItem.removeItem(self)

    def setData(
        self, xx=None, yy=None, fontSize=11, ID=0, y_cursor=None, x_cursor=None
    ):
        if xx is None:
            xx = []
        if yy is None:
            yy = []
        super().setData(xx, yy)
        if not hasattr(self, "label"):
            return

        if ID == 0:
            self.label.setText("")
        else:
            self.label.setText(f"{ID}", size=fontSize)
            w, h = self.label.itemRect().width(), self.label.itemRect().height()
            self.label.setPos(x_cursor, y_cursor - h)

    def clear(self):
        self.setData([], [])


class GhostMaskItem(pg.ImageItem):
    def __init__(self, ParentPlotItem):
        super().__init__()
        self.label = myLabelItem()
        self.label.setAttr("bold", True)
        self.label.setAttr("color", (245, 184, 0))
        self._ParentPlotItem = ParentPlotItem

    def initImage(self, imgShape):
        image = np.zeros(imgShape, dtype=np.uint32)
        self.setImage(image)

    def initLookupTable(self, rgbaColor):
        lut = np.zeros((2, 4), dtype=np.uint8)
        lut[1, -1] = 255
        lut[1, :-1] = rgbaColor
        self.setLookupTable(lut)

    def addToPlotItem(self):
        self._ParentPlotItem.addItem(self)
        self._ParentPlotItem.addItem(self.label)

    def removeFromPlotItem(self):
        self._ParentPlotItem.removeItem(self.label)
        self._ParentPlotItem.removeItem(self)

    def updateGhostImage(self, ID=0, y_cursor=None, x_cursor=None, fontSize=None):
        self.setImage(self.image)

        if ID == 0:
            self.label.setText("")
            return

        self.label.setText(f"{ID}", size=fontSize)
        w, h = self.label.itemRect().width(), self.label.itemRect().height()
        self.label.item.setPos(x_cursor, y_cursor - h)

    def clear(self):
        if hasattr(self, "label"):
            self.label.setText("")
        if self.image is None:
            return
        self.image[:] = 0
        self.setImage(self.image)


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


class ImShowPlotItem(pg.PlotItem):
    def __init__(
        self,
        parent=None,
        name=None,
        labels=None,
        title=None,
        viewBox=None,
        axisItems=None,
        enableMenu=True,
        **kargs,
    ):
        super().__init__(
            parent, name, labels, title, viewBox, axisItems, enableMenu, **kargs
        )
        # Overwrite zoom out button behaviour to disable autoRange after
        # clicking it.
        # If autorange is enabled, it is called everytime the brush or eraser
        # scatter plot items touches the border causing flickering
        self.disableAutoRange()
        self.autoBtn.mode = "manual"
        self.invertY(True)
        self.setAspectLocked(True)
        self.addImageItem(kargs.get("imageItem"))

        self._selected = False
        self.selectingRects = []

    def setSelectableTitle(self, title: QGraphicsProxyWidget, **kwargs):
        self.layout.removeItem(self.titleLabel)
        self.layout.addItem(title, 0, 1, alignment=Qt.AlignCenter)

    def isSelected(self):
        return self._selected

    def setSelected(
        self, selected: bool, xlim=(-np.inf, np.inf), ylim=(-np.inf, np.inf)
    ):
        if selected == self._selected:
            return

        if selected:
            ((xmin, xmax), (ymin, ymax)) = self.viewRange()
            ylim_min, ylim_max = ylim
            xlim_min, xlim_max = xlim

            xmin = max(xlim_min, xmin)
            xmax = min(xlim_max, xmax)
            ymin = max(ylim_min, ymin)
            ymax = min(ylim_max, ymax)

            w = xmax - xmin
            h = ymax - ymin

            bs = round(((w + h) / 2) * 0.02)
            if bs < 1:
                bs = 1

            rect_left = RectItem(QRectF(xmin, ymin, bs, h))
            rect_top = RectItem(QRectF(xmin + bs, ymin, w - bs - bs, bs))
            rect_right = RectItem(QRectF(xmax - bs, ymin, bs, h))
            rect_bottom = RectItem(QRectF(xmin + bs, ymax - bs, w - bs - bs, bs))
            self.selectingRects.append(rect_left)
            self.selectingRects.append(rect_top)
            self.selectingRects.append(rect_right)
            self.selectingRects.append(rect_bottom)

            self.addItem(rect_left)
            self.addItem(rect_top)
            self.addItem(rect_right)
            self.addItem(rect_bottom)
        else:
            for rect in self.selectingRects:
                self.removeItem(rect)
            self.selectingRects = []

        self._selected = selected

    def addImageItem(self, imageItem):
        self.imageItem = imageItem
        if imageItem is None:
            return

        self.setupContextMenu()
        self.addItem(imageItem)

    def setupContextMenu(self):
        shuffleCmapAction = QAction("Shuffle colormap", self.vb.menu)
        shuffleCmapAction.triggered.connect(self.shuffleColormap)
        self.vb.menu.addAction(shuffleCmapAction)

        self.resetCmapAction = QAction("Reset colormap", self.vb.menu)
        self.resetCmapAction.triggered.connect(self.resetColormap)
        self.vb.menu.addAction(self.resetCmapAction)
        self.resetCmapAction.setDisabled(True)

    def shuffleColormap(self):
        N = self.imageItem._numLevels
        colors = self.imageItem.lut / 255
        cmap = LinearSegmentedColormap.from_list("shuffled", colors, N=N)
        lut = plot.matplotlib_cmap_to_lut(cmap, n_colors=N)
        if not self.resetCmapAction.isEnabled():
            self._defaultLut = lut.copy()
        bkgrColor = lut[0].copy()
        np.random.shuffle(lut)
        lut[0] = bkgrColor
        self.imageItem.setLookupTable(lut)
        self.imageItem.update()
        self.resetCmapAction.setDisabled(False)

    def resetColormap(self):
        self.imageItem.setLookupTable(self._defaultLut)

    def autoBtnClicked(self):
        self.autoRange()

    def autoRange(self):
        self.vb.autoRange()
        self.autoBtn.hide()


class _ImShowImageItem(pg.ImageItem):
    sigDataHover = Signal(str)
    sigHoverEvent = Signal(object, object)
    sigMousePressEvent = Signal(object, object)

    def __init__(self, idx) -> None:
        super().__init__()
        self._idx = idx
        self._cursors = []
        self._autoLevels = True

    def _getHoverImageValue(self, xdata, ydata):
        try:
            value = self.image[ydata, xdata]
            return value
        except Exception as err:
            return

    def setAutoLevels(self, autoLevels):
        self._autoLevels = autoLevels

    def mousePressEvent(self, event):
        self.sigMousePressEvent.emit(self, event)
        super().mousePressEvent(event)

    def setOtherImagesCursors(self, cursors):
        self._cursors = cursors

    def clearCursors(self):
        for p, cursor in enumerate(self._cursors):
            if p == self._idx:
                continue

            cursor.setData([], [])

    def setImage(self, *args, **kwargs):
        if "autoLevels" not in kwargs:
            kwargs["autoLevels"] = self._autoLevels

        super().setImage(*args, **kwargs)
        if not args:
            return

        if not kwargs["autoLevels"]:
            return

        image = args[0]
        self._imageMax = image.max()
        self._imageMin = image.min()
        self._numLevels = self._imageMax - self._imageMin

    def hoverEvent(self, event):
        self.sigHoverEvent.emit(self, event)

        if event.isExit():
            self.clearCursors()
            self.sigDataHover.emit("")
            return

        x, y = event.pos()
        xdata, ydata = int(x), int(y)
        value = self._getHoverImageValue(xdata, ydata)
        if value is None:
            self.clearCursors()
            self.sigDataHover.emit("")
            return

        try:
            self.sigDataHover.emit(f"x={xdata}, y={ydata}, {value = :.4f}")
        except Exception as e:
            self.sigDataHover.emit(f"x={xdata}, y={ydata}, {[val for val in value]}")

        for p, cursor in enumerate(self._cursors):
            if p == self._idx:
                continue

            cursor.setData([x], [y])


class ImShow(QBaseWindow):
    def __init__(
        self,
        parent=None,
        link_scrollbars=True,
        infer_rgb=True,
        figure_title="",
        selectable_images=False,
    ):
        super().__init__(parent=parent)
        self._linkedScrollbars = link_scrollbars
        self._infer_rgb = infer_rgb
        self._figure_title = figure_title
        self._selectable_images = True
        self.selected_idx = None

        self._autoLevels = True

        self.textItems = []
        self.group_to_idx_mapper = {"": 0}

    def _getGraphicsScrollbar(self, idx, image, imageItem, maximum):
        proxy = QGraphicsProxyWidget(imageItem)
        scrollbar = ScrollBarWithNumericControl(
            orientation=Qt.Horizontal, add_max_proj_button=True
        )
        scrollbar.sigValueChanged.connect(self.OnScrollbarValueChanged)
        scrollbar.sigMaxProjToggled.connect(self.onMaxProjToggled)
        scrollbar.idx = idx
        scrollbar.image = image
        scrollbar.imageItem = imageItem
        scrollbar.setMaximum(maximum)
        proxy.setWidget(scrollbar)
        proxy.scrollbar = scrollbar
        return proxy

    def OnScrollbarValueChanged(self, value):
        scrollbar = self.sender()
        imageItem = scrollbar.imageItem
        img = self._get2Dimg(imageItem, scrollbar.image)
        imageItem.setImage(img)  # , autoLevels=self._autoLevels)

        overlayLab = self._get2DlabOverlay(imageItem)
        if overlayLab is not None:
            imageItem.labImageItem.setImage(overlayLab, autoLevels=False)

        self.setPointsVisible(imageItem)

        self.updateIDs()

        if not self._linkedScrollbars:
            return
        if len(self.ImageItems) == 1:
            return

        self._linkedScrollbars = False
        try:
            idx = scrollbar.idx
            for otherImageItem in self.ImageItems:
                if otherImageItem.gridPos == imageItem.gridPos:
                    continue
                if otherImageItem.image.shape != imageItem.image.shape:
                    continue
                for otherScrollbar in otherImageItem.ScrollBars:
                    if otherScrollbar.idx != idx:
                        continue
                    otherScrollbar.setValue(scrollbar.value())
        except Exception as e:
            pass
        finally:
            self._linkedScrollbars = True

    def _get2Dimg(self, imageItem, image):
        for scrollbar in imageItem.ScrollBars:
            if scrollbar.maxProjCheckbox.isChecked():
                image = image.max(axis=0)
            else:
                image = image[scrollbar.value()]
        return image

    def _get2DlabOverlay(self, imageItem):
        try:
            lab = imageItem.lab
        except Exception as err:
            return

        for scrollbar in imageItem.ScrollBars:
            if scrollbar.maxProjCheckbox.isChecked():
                lab = lab.max(axis=0)
            else:
                lab = lab[scrollbar.value()]

        return lab

    def isObjVisible(self, obj, imageItem):
        if len(obj.centroid) == 2:
            return True

        z_scrollbar = imageItem.ScrollBars[-1]
        if z_scrollbar.maxProjCheckbox.isChecked():
            return True

        z_slice = z_scrollbar.value()
        min_z, min_y, min_x, max_z, max_y, max_x = obj.bbox
        if z_slice >= min_z and z_slice < max_z:
            return True

        return False

    def onMaxProjToggled(self, checked, scrollbar):
        imageItem = scrollbar.imageItem
        img = self._get2Dimg(imageItem, scrollbar.image)
        imageItem.setImage(img)  # , autoLevels=self._autoLevels)
        overlayLab = self._get2DlabOverlay(imageItem)
        if overlayLab is not None:
            imageItem.labImageItem.setImage(overlayLab, autoLevels=False)
        self.setPointsVisible(imageItem)
        if not self._linkedScrollbars:
            return
        if len(self.ImageItems) == 1:
            return

        self._linkedScrollbars = False
        try:
            idx = scrollbar.idx
            for otherImageItem in self.ImageItems:
                if otherImageItem.gridPos == imageItem.gridPos:
                    continue
                if otherImageItem.image.shape != imageItem.image.shape:
                    continue
                for otherScrollbar in otherImageItem.ScrollBars:
                    if otherScrollbar.idx != idx:
                        continue
                    otherScrollbar.maxProjCheckbox.setChecked(checked)
        except Exception as e:
            pass
        finally:
            self._linkedScrollbars = True

        self.updateIDs()

    def setPointsVisible(self, imageItem):
        if not hasattr(imageItem, "pointsItems"):
            return

        first_coord = imageItem.ScrollBars[0].value()
        isMaxProj = imageItem.ScrollBars[0].maxProjCheckbox.isChecked()
        for pointsItems in imageItem.pointsItems.values():
            for p, plotItem in enumerate(pointsItems):
                plotItem.setVisible((isMaxProj) or (p == first_coord))

    def setupStatusBar(self):
        self.statusbar = self.statusBar()
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)

    def setupMainLayout(self):
        self._layout = QHBoxLayout()
        self._container = QWidget()
        self._container.setLayout(self._layout)
        self.setCentralWidget(self._container)

    def setupGraphicLayout(
        self, *images, hide_axes=True, max_ncols=4, color_scheme="light"
    ):
        self.graphicLayout = pg.GraphicsLayoutWidget()
        self._colorScheme = color_scheme

        # Set a light background
        if color_scheme == "light":
            self.graphicLayout.setBackground((235, 235, 235))
        else:
            self.graphicLayout.setBackground((30, 30, 30))

        ncells = max_ncols * ceil(len(images) / max_ncols)

        nrows = ncells // max_ncols
        nrows = nrows if nrows > 0 else 1
        ncols = max_ncols if len(images) > max_ncols else len(images)

        if color_scheme == "light":
            color = "black"
        else:
            color = "white"

        self.titleLabel = pg.LabelItem(justify="center", color=color, size="14pt")
        self.titleLabel.setText(self._figure_title)
        self.graphicLayout.addItem(self.titleLabel, row=0, col=0, colspan=ncols)
        start_row = 1

        # Check if additional rows are needed for the scrollbars
        max_ndim = max([image.ndim for image in images])
        if max_ndim > 4:
            raise TypeError("One or more of the images have more than 4 dimensions.")
        if max_ndim == 4:
            rows_range = range(0, (nrows - 1) * 3 + 1, 3)
        elif max_ndim == 3:
            rows_range = range(0, (nrows - 1) * 2 + 1, 2)
        else:
            rows_range = range(nrows)

        self.PlotItems = []
        self.ImageItems = []
        self.ScrollBars = []
        i = 0
        for r in rows_range:
            row = r + start_row
            for col in range(ncols):
                try:
                    image = images[i]
                except IndexError:
                    break
                plotItem = ImShowPlotItem()
                if hide_axes:
                    plotItem.hideAxis("bottom")
                    plotItem.hideAxis("left")
                self.graphicLayout.addItem(plotItem, row=row, col=col)
                plotItem.loc = (row, col)
                self.PlotItems.append(plotItem)

                imageItem = _ImShowImageItem(i)
                plotItem.addImageItem(imageItem)
                imageItem.plot = plotItem
                imageItem.sigHoverEvent.connect(self.onImageItemHoverEvent)
                imageItem.sigMousePressEvent.connect(self.onImageItemMousePressEvent)
                self.ImageItems.append(imageItem)
                imageItem.gridPos = (row, col)
                imageItem.ScrollBars = []

                is_rgb = image.shape[-1] == 3 and self._infer_rgb
                is_rgba = image.shape[-1] == 4 and self._infer_rgb
                does_not_require_scrollbars = image.ndim == 2 or (
                    image.ndim == 3 and (is_rgb or is_rgba)
                )
                if does_not_require_scrollbars:
                    i += 1
                    continue

                idx_image = 3 if (is_rgb or is_rgba) else 2
                for s in range(image.ndim - idx_image):
                    maximum = image.shape[s] - 1
                    scrollbarProxy = self._getGraphicsScrollbar(
                        s, image, imageItem, maximum
                    )
                    self.graphicLayout.addItem(scrollbarProxy, row=row + s + 1, col=col)
                    imageItem.ScrollBars.append(scrollbarProxy.scrollbar)

                i += 1

        self._layout.addWidget(self.graphicLayout)

    def onImageItemMousePressEvent(self, imageItem, event):
        if not self._selectable_images:
            return

        plotItem = imageItem.plot
        if not plotItem.isSelected():
            return

        self.selected_idx = self.PlotItems.index(plotItem)
        event.ignore()
        self.close()

    def onImageItemHoverEvent(self, imageItem, event):
        if not self._selectable_images:
            return

        modifiers = QGuiApplication.keyboardModifiers()
        isCtrl = modifiers == Qt.ControlModifier
        plotItem = imageItem.plot
        Y, X = imageItem.image.shape[:2]
        plotItem.setSelected(isCtrl and not event.isExit(), xlim=(0, X), ylim=(0, Y))

    def movePlotItem(self, title):
        combobox = self.sender()
        plotItem = combobox.plotItem
        row, col = plotItem.loc

        otherPlotItemIdx = combobox.titles.index(title)
        otherPlotItem = self.PlotItems[otherPlotItemIdx]
        other_row, other_col = otherPlotItem.loc

        self.graphicLayout.removeItem(plotItem)
        self.graphicLayout.removeItem(otherPlotItem)
        self.graphicLayout.addItem(otherPlotItem, row=row, col=col)
        self.graphicLayout.addItem(plotItem, row=other_row, col=other_col)

        combobox.blockSignals(True)
        combobox.setCurrentText(combobox.default_text)
        combobox.blockSignals(False)

        plotItemIdx = combobox.titles.index(combobox.default_text)

        otherPlotItem.loc = (row, col)
        plotItem.loc = (other_row, other_col)

    def setupTitles(self, *titles):
        for plotItem, title in zip(self.PlotItems, titles):
            combobox = ComboBox()
            combobox.default_text = title
            combobox.titles = list(titles)
            combobox.addItems(titles)
            combobox.setMaximumWidth(combobox.sizeHint().width())
            combobox.setCurrentText(title)
            comboboxGraphicsItem = QGraphicsProxyWidget()
            comboboxGraphicsItem.setWidget(combobox)
            combobox.plotItem = plotItem
            plotItem.setSelectableTitle(comboboxGraphicsItem)
            combobox.currentTextChanged.connect(self.movePlotItem)

        # color = 'k' if self._colorScheme == 'light' else 'w'
        # for plotItem, title in zip(self.PlotItems, titles):
        #     plotItem.setSelectableTitle(title, color=color)

    def updateStatusBarLabel(self, text):
        self.wcLabel.setText(text)

    def autoRange(self):
        for plot in self.PlotItems:
            plot.autoRange()

    def showImages(
        self,
        *images,
        labels_overlays: np.ndarray | List[np.ndarray] = None,
        luts=None,
        labels_overlays_luts=None,
        autoLevels=True,
        autoLevelsOnScroll=False,
    ):
        from .plot import matplotlib_cmap_to_lut

        images = [np.squeeze(img) for img in images]
        self.luts = luts
        self._autoLevels = autoLevels
        self._autoLevelsOnScroll = autoLevelsOnScroll
        for image in images:
            if image.ndim > 5 or image.ndim < 2:
                raise TypeError(
                    f"Input image has {image.ndim} dimensions. "
                    "Only 2-D, 3-D, and 4-D images are supported"
                )

        if isinstance(labels_overlays, np.ndarray):
            labels_overlays = [labels_overlays]

        if isinstance(labels_overlays_luts, np.ndarray):
            labels_overlays_luts = [labels_overlays_luts]

        if (
            labels_overlays_luts is not None
            and labels_overlays is not None
            and (len(labels_overlays_luts) != len(labels_overlays))
        ):
            raise TypeError(
                f"Number of lables_overlays_luts is {len(labels_overlays_luts)}, "
                f"while number of labels_overaly is {len(labels_overlays)}. "
                "Pass `None` if you want to use default lut for the labels_overlays."
            )

        if labels_overlays is not None and (len(labels_overlays) != len(images)):
            raise TypeError(
                f"Number of images is {len(images)}, "
                f"while number of labels_overaly is {len(labels_overlays)}. "
                "Pass `None` if you do not need overlaid labeles."
            )

        for i, (image, imageItem) in enumerate(zip(images, self.ImageItems)):
            if luts is not None:
                _autoLevels = autoLevels
                lut = luts[i]
                if not autoLevels and lut is not None:
                    imageItem.setLevels([0, len(lut)])
                else:
                    _autoLevels = True
                if lut is None:
                    lut = matplotlib_cmap_to_lut("viridis")
                imageItem.setLookupTable(lut)
            else:
                _autoLevels = True

            is_rgb = image.shape[-1] == 3 and self._infer_rgb
            is_rgba = image.shape[-1] == 4 and self._infer_rgb
            does_not_require_scrollbars = image.ndim == 2 or (
                image.ndim == 3 and (is_rgb or is_rgba)
            )

            if does_not_require_scrollbars:
                imageItem.setAutoLevels(_autoLevels)
                imageItem.setImage(image)
            else:
                if not self._autoLevelsOnScroll and not _autoLevels:
                    imageItem.setAutoLevels(False)
                    imageItem.setLevels([image.min(), image.max()])
                for scrollbar in imageItem.ScrollBars:
                    scrollbar.setValue(int(scrollbar.maximum() / 2))

            imageItem.sigDataHover.connect(self.updateStatusBarLabel)

            if labels_overlays is None:
                continue

            lab_overlay = labels_overlays[i]
            if lab_overlay is None:
                continue

            if lab_overlay.shape != image.shape:
                raise TypeError(
                    f"`lab_overlay` at index {i} has shape "
                    f"{lab_overlay.shape} which is different "
                    f"from image shape {image.shape}. "
                    "The image and the `lab_overlay` must "
                    "have the same shape."
                )

            plot = imageItem.plot
            labImageItem = pg.ImageItem()
            labImageItem.setOpacity(0.4)
            plot.addImageItem(labImageItem)

            if labels_overlays_luts is not None:
                labels_overlays_lut = labels_overlays_luts[i]
            else:
                labels_overlays_lut = self._getDefaultLabelsOverlayLut(lab_overlay)

            labImageItem.setLookupTable(labels_overlays_lut)
            labImageItem.setLevels([0, len(labels_overlays_lut)])

            imageItem.lab = lab_overlay
            imageItem.labImageItem = labImageItem

            overlayLab = self._get2DlabOverlay(imageItem)
            labImageItem.setImage(overlayLab, autoLevels=False)

        # Share axis between images with same X, Y shape
        all_shapes = [image.shape[-2:] for image in images]
        unique_shapes = set(all_shapes)
        shame_shape_plots = []
        for unique_shape in unique_shapes:
            plots = [
                self.PlotItems[i]
                for i, shape in enumerate(all_shapes)
                if shape == unique_shape
            ]
            shame_shape_plots.append(plots)

        for plots in shame_shape_plots:
            for plot in plots:
                plot.vb.setYLink(plots[0].vb)
                plot.vb.setXLink(plots[0].vb)

    def _getDefaultLabelsOverlayLut(self, lab_overlay):
        IDs = [obj.label for obj in skimage.measure.regionprops(lab_overlay)]
        n_objs = len(IDs)
        lut = np.zeros((n_objs + 1, 4), dtype=np.uint8)
        rgbas = colors.plt_colormap_to_pg_lut("tab20", ncolors=n_objs)
        np.random.shuffle(rgbas)
        lut[1:] = rgbas
        return lut

    def _createPointsScatterItem(self, xx, yy, group, colors=None, data=None):
        if colors is None:
            cmap = matplotlib.colormaps["jet_r"]
            idx = self.group_to_idx_mapper[group]
            r, g, b = [round(c * 255) for c in cmap(idx)][:3]
            brush = pg.mkBrush(color=(r, g, b, 100))
            pen = pg.mkPen(width=2, color=(r, g, b))
            hoverBrush = pg.mkBrush((r, g, b, 200))
        else:
            brush = []
            pen = []
            hoverBrush = None
            for color in colors:
                rgb = matplotlib.colors.to_rgb(color)
                rgb = [round(c * 255) for c in rgb]
                _brush = pg.mkBrush(color=(*rgb, 100))
                _pen = pg.mkPen(width=2, color=rgb)
                brush.append(_brush)
                pen.append(_pen)

        item = pg.ScatterPlotItem(
            xx,
            yy,
            symbol="o",
            pxMode=False,
            size=3,
            brush=brush,
            pen=pen,
            hoverable=True,
            hoverBrush=hoverBrush,
            data=data,
        )
        return item

    def drawPointsFromDf(
        self, points_df: pd.DataFrame | List[pd.DataFrame], points_groups=None
    ):
        if not isinstance(points_df, (list, tuple)):
            points_df = [points_df] * len(self.PlotItems)

        for p, df in enumerate(points_df):
            if isinstance(points_groups, str):
                points_groups = [points_groups]

            if points_groups is None:
                grouped = [("", df)]
                groups = [""]
            else:
                grouped = df.groupby(points_groups)
                groups = grouped.groups.keys()

            idxs_space = np.linspace(0, 1, len(groups))
            self.group_to_idx_mapper = dict(zip(groups, idxs_space))

            for group, df in grouped:
                yy = df["y"].values
                xx = df["x"].values
                points_coords = np.column_stack((yy, xx))
                if "z" in df.columns:
                    zz = df["z"].values
                    points_coords = np.column_stack((zz, points_coords))
                if len(group) == 1:
                    group = group[0]

                colors = None
                if "color" in df.columns:
                    colors = df["color"].values

                data = None
                if "data" in df.columns:
                    data = df["data"].values

                self.drawPoints(
                    points_coords, colors=colors, group=group, idx=p, data=data
                )

    def drawPoints(
        self,
        points_coords: np.ndarray,
        group="",
        idx=None,
        colors=None,
        data=None,
    ):
        offset = 0.5 if np.issubdtype(points_coords.dtype, np.integer) else 0
        n_dim = points_coords.shape[1]

        if idx is not None:
            PlotItems = [self.PlotItems[idx]]
            ImageItems = [self.ImageItems[idx]]
        else:
            PlotItems = self.PlotItems
            ImageItems = self.ImageItems

        if n_dim == 2:
            if data is None:
                data = group

            zz = [0] * len(points_coords)
            self.points_coords = np.column_stack((zz, points_coords))
            for p, plotItem in enumerate(PlotItems):
                imageItem = ImageItems[p]
                xx = points_coords[:, 1] + offset
                yy = points_coords[:, 0] + offset
                pointsItem = self._createPointsScatterItem(
                    xx, yy, group, data=data, colors=colors
                )
                pointsItem.z = 0
                plotItem.addItem(pointsItem)
                imageItem.pointsItems = {group: [pointsItem]}
        elif n_dim == 3:
            self.points_coords = points_coords
            for p, plotItem in enumerate(PlotItems):
                imageItem = ImageItems[p]
                imageItem.pointsItems = defaultdict(list)
                scrollbar = imageItem.ScrollBars[0]
                for first_coord in range(scrollbar.maximum() + 1):
                    coords_idx = np.nonzero(points_coords[:, 0] == first_coord)
                    coords = points_coords[coords_idx]
                    if colors is None:
                        _colors = None
                    else:
                        _colors = np.asarray(colors)[coords_idx]
                        if len(_colors) == 0:
                            _colors = None

                    _data = group
                    if data is not None:
                        _data = data[coords_idx]
                        if len(_data) == 0:
                            _data = group

                    xx = coords[:, 2] + offset
                    yy = coords[:, 1] + offset
                    pointsItem = self._createPointsScatterItem(
                        xx, yy, group, data=_data, colors=_colors
                    )
                    pointsItem.z = first_coord
                    plotItem.addItem(pointsItem)
                    pointsItem.setVisible(False)
                    imageItem.pointsItems[group].append(pointsItem)
                self.setPointsVisible(imageItem)

    def setupDuplicatedCursors(self):
        self.cursors = []
        for p, plotItem in enumerate(self.PlotItems):
            cursor = pg.ScatterPlotItem(
                symbol="+",
                pxMode=True,
                pen=pg.mkPen("k", width=1),
                brush=pg.mkBrush("w"),
                size=16,
                tip=None,
            )
            self.cursors.append(cursor)
            plotItem.addItem(cursor)

        for imageItem in self.ImageItems:
            imageItem.setOtherImagesCursors(self.cursors)

    def setPointsData(self, points_data):
        points_df = pd.DataFrame(
            {
                "z": self.points_coords[:, 0],
                "y": self.points_coords[:, 1],
                "x": self.points_coords[:, 2],
            }
        )
        if isinstance(points_data, pd.Series):
            points_df[points_data.name] = points_data.values
        elif isinstance(points_data, pd.DataFrame):
            points_df = points_df.join(points_data)
        elif isinstance(points_data, np.ndarray):
            if points_data.ndim == 1:
                points_data = points_data[np.newaxis]
            else:
                points_data = points_data.T
            for i, values in enumerate(points_data):
                points_df[f"col_{i}"] = values

        self.points_df = points_df.set_index(["z", "y", "x"]).sort_index()

        for p, plotItem in enumerate(self.PlotItems):
            imageItem = self.ImageItems[p]
            for pointsItems in imageItem.pointsItems.values():
                for pointsItem in pointsItems:
                    pointsItem.sigClicked.connect(self.pointsClicked)

    def pointsClicked(self, item, points, event):
        point = points[0]
        x, y = point.pos()
        coords = (item.z, int(y), int(x))
        point_data = self.points_df.loc[[coords]]
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print("*" * 60)
        print(f"Point clicked at {now}. Data:")
        print("-" * 60)
        print(point_data)
        print("")
        print("*" * 60)

    def annotateObjectIDs(self, annotate_labels_idxs=None, init=False):
        if init:
            self.annotate_labels_idxs = annotate_labels_idxs
            self.textItems = [{} for _ in self.PlotItems]
        if self.annotate_labels_idxs is None:
            return
        for i, plotItem in enumerate(self.PlotItems):
            if i not in self.annotate_labels_idxs:
                continue
            plotTextItems = self.textItems[i]
            imageItem = self.ImageItems[i]
            try:
                if init:
                    # 3D labels (if 3D)
                    lab = imageItem.lab
                else:
                    lab = imageItem.labImageItem.image
            except Exception as err:
                lab = imageItem.image

            rp = skimage.measure.regionprops(lab)
            for obj in rp:
                textItem = plotTextItems.get(obj.label)
                yc, xc = obj.centroid[-2:]
                if textItem is None:
                    textItem = pg.TextItem(text="", anchor=(0.5, 0.5), color="r")
                    plotItem.addItem(textItem)
                    plotTextItems[obj.label] = textItem

                if self.isObjVisible(obj, imageItem):
                    text = str(obj.label)
                else:
                    text = ""

                textItem.setText(text)
                textItem.setPos(xc, yc)

            # plotItem.enableAutoRange()

    def clearLabels(self):
        for textItems in self.textItems:
            for textItem in textItems.values():
                textItem.setText("")

    def updateIDs(self):
        self.clearLabels()
        try:
            self.annotateObjectIDs(annotate_labels_idxs=self.annotate_labels_idxs)
        except Exception as err:
            pass

    def show(self, block=False, screenToWindowRatio=None):
        super().show(block=block)
        if screenToWindowRatio is None:
            return
        screenGeometry = self.screen().geometry()
        screenWidth = screenGeometry.width()
        screenHeight = screenGeometry.height()
        finalWidth = int(screenToWindowRatio * screenWidth)
        finalHeight = int(screenToWindowRatio * screenHeight)
        screenTop = screenGeometry.top()
        screenLeft = screenGeometry.left()
        xc, yc = screenLeft + screenWidth / 2, screenTop + screenHeight / 2
        winLeft = int(xc - finalWidth / 2)
        winTop = int(yc - finalHeight / 2)
        self.setGeometry(winLeft, winTop, finalWidth, finalHeight)

    def run(self, block=False, showMaximised=False, screenToWindowRatio=None):
        if showMaximised:
            self.showMaximized()
        else:
            self.show(screenToWindowRatio=screenToWindowRatio)
        QTimer.singleShot(100, self.autoRange)

        if block:
            self.exec_()

    def resizeEvent(self, event) -> None:
        self.PlotItems[0].autoRange()
        return super().resizeEvent(event)


class LabelItem(pg.LabelItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def bbox(self):
        xl, yl = self.pos().x(), self.pos().y()
        wl, hl = self.itemRect().width(), self.itemRect().height()
        return yl, xl, yl + hl, xl + wl

    def setBold(self, bold=True):
        self.origPos = self.pos()
        self.setText(self.text, bold=bold)
        self.setPos(self.origPos)


class ScaleBar(QGraphicsObject):
    sigEditProperties = Signal(object)
    sigRemove = Signal(object)

    def __init__(self, imageShape, viewRange, parent=None):
        super().__init__(parent)
        self.SizeY, self.SizeX = imageShape
        self.updateViewRange(viewRange)
        self.plotItem = PlotCurveItem()
        self.labelItem = LabelItem()
        self._x_pad = 5
        self._y_pad = 3
        self._highlighted = False
        self._parent = parent
        self.clicked = False
        self.createContextMenu()

    def updateViewRange(self, viewRange):
        xRange, yRange = viewRange
        x0, x1 = xRange
        y0, y1 = yRange
        if x0 < 0:
            x0 = 0

        if x1 > self.SizeX:
            x1 = self.SizeX

        if y0 < 0:
            y0 = 0

        if y1 > self.SizeY:
            y1 = self.SizeY

        self.xmax = x1
        self.xmin = x0

        self.ymax = y1
        self.ymin = y0

    def createContextMenu(self):
        self.contextMenu = QMenu()
        action = QAction("Edit properties...", self.contextMenu)
        action.triggered.connect(self.emitEditProperties)
        self.contextMenu.addSeparator()
        action = QAction("Remove", self.contextMenu)
        action.triggered.connect(self.emitRemove)
        self.contextMenu.addAction(action)

    def emitEditProperties(self):
        self.setHighlighted(False)
        self.sigEditProperties.emit(self.properties())

    def emitRemove(self):
        self.sigRemove.emit(self)

    def isHighlighted(self):
        return self._highlighted

    def setHighlighted(self, highlighted):
        if self._highlighted and highlighted:
            return

        if not self._highlighted and not highlighted:
            return

        pen = self.highlightPen if highlighted else self.pen
        self.labelItem.setBold(bold=highlighted)
        self.plotItem.setPen(pen)

        self._highlighted = highlighted

    def showContextMenu(self, x, y):
        self.contextMenu.popup(QPoint(int(x), int(y)))

    def properties(self):
        properties = {
            "thickness": self._thickness,
            "length_pixel": self._length,
            "length_unit": self._length_unit,
            "is_text_visible": self._is_text_visible,
            "color": self._color,
            "loc": self._loc,
            "font_size": float(self._font_size[:-2]),
            "unit": self._unit,
            "num_decimals": self._num_decimals,
            "move_with_zoom": self._move_with_zoom,
        }
        return properties

    def move(self, xm, ym):
        self._loc = "Custom"

        Dy = ym - self.yc
        Dx = xm - self.xc

        x0 = self.x0c + Dx
        x1 = x0 + self._length
        y0 = y1 = self.y0c + Dy
        self.plotItem.setData([x0, x1], [y0, y1])
        self.setTextPos()

    def paint(self, painter, option, widget):
        pass

    def boundingRect(self):
        ymin, xmin, ymax, xmax = self.bbox()
        return QRectF(xmin, ymin, xmax - xmin, ymax - ymin)

    def setLocationProperty(self, loc: str):
        self._loc = loc

    def setMoveWithZoomProperty(self, move_with_zoom):
        self._move_with_zoom = move_with_zoom

    def setProperties(
        self,
        length_pixel,
        length_unit,
        thickness=3,
        color="w",
        is_text_visible=True,
        loc="top-left",
        font_size=12,
        unit="",
        num_decimals=0,
        move_with_zoom=False,
    ):
        self._loc = loc
        self._color = color
        self._length = length_pixel
        self._length_unit = length_unit
        self._is_text_visible = is_text_visible
        self._font_size = f"{font_size}px"
        self._unit = unit
        self._num_decimals = num_decimals
        self._move_with_zoom = move_with_zoom
        self._thickness = thickness
        self.pen = pg.mkPen(width=thickness, color=color, cosmetic=False)
        self.highlightPen = pg.mkPen(width=thickness + 2, color=color, cosmetic=False)
        self.pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self.highlightPen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self.plotItem.setPen(self.pen)

    def updatePhysicalLength(self, PhysicalSizeX):
        length_unit = self._length_unit
        unit = self._unit
        length_um = _core.convert_length(length_unit, unit, "μm")
        length_pixel = length_um / PhysicalSizeX
        self._length = length_pixel
        self.update()

    def addToAxis(self, ax):
        ax.addItem(self.plotItem)
        ax.addItem(self.labelItem)

    def setText(self):
        if self._is_text_visible:
            number = round(self._length_unit, self._num_decimals)
            if self._num_decimals == 0:
                number = int(number)
            text = f"{number} {self._unit}"
        else:
            text = ""
        self.labelItem.setText(text, color=self._color, size=self._font_size)

    def setTextPos(self):
        xx, yy = self.plotItem.getData()
        x0 = xx[0]
        y0 = yy[0]
        xc = x0 + self._length / 2
        wl = self.labelItem.itemRect().width()
        hl = self.labelItem.itemRect().height()
        xl = xc - wl / 2
        yt = y0 - hl
        self.labelItem.setPos(xl, yt)

    def updatePosViewRangeChanged(self, viewRange):
        if self._loc == "custom":
            xx, yy = self.plotItem.getData()
            x0p = xx[0]
            y0p = yy[0]
            xcp = x0p + self._length / 2
            hl = self.labelItem.itemRect().height()
            ycp = y0p - hl / 2
            x0 = self.xmin
            y0 = self.ymin
            x_range = self.xmax - x0
            y_range = self.ymax - y0
            Dx_perc = (xcp - x0) / x_range
            Dy_perc = (ycp - y0) / y_range

            self.updateViewRange(viewRange)

            X0 = self.xmin
            Y0 = self.ymin

            X_range = self.xmax - X0
            Y_range = self.ymax - Y0

            Xcp = X0 + (Dx_perc * X_range)
            Ycp = Y0 + (Dy_perc * Y_range)
            X0p = Xcp - (self._length / 2)
            Y0p = Ycp + (hl / 2)

            X1p = X0p + self._length
            Y1p = Y0p

            self.plotItem.setData([X0p, X1p], [Y0p, Y1p])
        else:
            self.updateViewRange(viewRange)
            self.update()

    def getStartXCoordFromLoc(self, loc):
        if loc == "custom":
            xx, yy = self.plotItem.getData()
            x0 = xx[0]
            return x0

        self.setText()
        wl = self.labelItem.itemRect().width()
        if loc.find("left") != -1:
            x0 = self._x_pad + self.xmin
            xc = x0 + self._length / 2
            xl = xc - wl / 2
            if xl < x0:
                # Text is larger than line --> move line to the right
                x0 = self._x_pad + abs(xl - self._x_pad)
        else:
            x0 = self.xmax - self._length - self._x_pad
            xc = x0 + self._length / 2
            x1 = x0 + self._length
            xr = xc + wl / 2
            if xr > x1:
                # Text is larger than line --> move line to the left
                delta_overshoot = xr - x1
                x0 = x0 - delta_overshoot
        return x0

    def getStartYCoordFromLoc(self, loc):
        if loc == "custom":
            xx, yy = self.plotItem.getData()
            y0 = yy[0]
            return y0

        self.setText()
        textHeight = self.labelItem.itemRect().height()
        if loc.find("top") != -1:
            return textHeight + self._y_pad + self.ymin
        else:
            return self.ymax - self._y_pad - self._thickness

    def update(self):
        x0 = self.getStartXCoordFromLoc(self._loc)  # + self._thickness/2
        y0 = self.getStartYCoordFromLoc(self._loc)

        x1 = x0 + self._length  # - self._thickness/2
        self.plotItem.setData([x0, x1], [y0, y0])

        self.setText()
        self.setTextPos()

    def draw(self, length_pixel, length_unit, **kwargs):
        self.setProperties(length_pixel, length_unit, **kwargs)
        self.update()

    def bbox(self):
        y_line_min, x_line_min, y_line_max, x_line_max = self.plotItem.bbox()
        y_lab_min, x_lab_min, y_lab_max, x_lab_max = self.labelItem.bbox()
        ymin = min(y_line_min, y_lab_min)
        xmin = min(x_line_min, x_lab_min)
        ymax = max(y_line_max, y_lab_max)
        xmax = max(x_line_max, x_lab_max)
        return ymin, xmin, ymax, xmax

    def mousePressed(self, x, y):
        self.clicked = True
        self.xc, self.yc = x, y
        xx, yy = self.plotItem.getData()
        self.x0c = xx[0]
        self.y0c = yy[0]

    def removeFromAxis(self, ax):
        ax.removeItem(self.labelItem)
        ax.removeItem(self.plotItem)


class RulerPlotItem(pg.PlotDataItem):
    def __init__(self, *args, **kwargs):
        self.labelItem = pg.LabelItem()
        super().__init__(*args, **kwargs)

    def setData(self, *args, lengthText="", **kwargs):
        super().setData(*args, **kwargs)
        self.labelItem.setText("")
        if not lengthText:
            return
        self.setLengthText(lengthText)

    def setLengthText(self, lengthText):
        xx, yy = self.getData()
        x0, x1 = sorted(xx)
        y0, y1 = sorted(yy)
        xc = round(x0 + (x1 - x0) / 2)
        yc = round(y0 + (y1 - y0) / 2)
        self.labelItem.setText(lengthText, size="11px", color="r")
        # xc = x0 + self._length/2
        wl = self.labelItem.itemRect().width()
        hl = self.labelItem.itemRect().height()
        xl = xc - wl / 2
        yt = y0 - hl
        self.labelItem.setPos(xl, yt)


class PointsScatterPlotItem(pg.ScatterPlotItem):
    sigHoverEntered = Signal(object, object, object)

    def __init__(self, *args, ax=None, show_data_as_tip=False, **kwargs):
        self.textItem = annotate.TextAnnotationsScatterItem(size=12, anchor=(1.0, 1.0))
        self.textItem.createSymbols(
            [str(int_id) for int_id in range(200)], includeBold=False
        )
        # self._textItems = {}
        super().__init__(*args, **kwargs)
        self.textItem.setParentItem(self)
        self._font = QFont()
        self._font.setPixelSize(12)
        self.show_data_as_tip = show_data_as_tip
        self.drawIds = True
        self.ax = ax
        self.sigHovered.connect(self.onHover)
        self.lastHoveredPoint = None

    def onHover(self, item, points, event):
        if len(points) == 0:
            vb = self.getViewBox()
            vb.setToolTip("")
            return

        if self.lastHoveredPoint != points[0]:
            self.sigHoverEntered.emit(item, points, event)
            self.lastHoveredPoint = points[0]

        if not self.opts["hoverable"]:
            return

        if not self.show_data_as_tip:
            return

        tip_li = [str(point.data()) for point in points]
        tip = "\n\n".join(tip_li)

        vb = self.getViewBox()
        vb.setToolTip(tip)

    def setData(self, *args, **kwargs):
        self.clearTextItems()
        super().setData(*args, **kwargs)
        data = kwargs.get("data")
        if data is None:
            return

        if len(data) == 0:
            return

        first_point_data = data[0]
        if not isinstance(first_point_data, (int, str)):
            return

        if not self.drawIds:
            return

        if self.show_data_as_tip:
            return

        color = self.opts["brush"].color()
        self.textItem.setColors({"id": color.getRgb()})
        size = self.opts["size"]
        radius = size / 2
        # xx, yy = args
        # for x, y, point_data in zip(xx, yy, data):
        for point in self.points():
            text = str(point.data())
            if not text:
                continue

            x, y = point.pos().x(), point.pos().y()
            xt, yt = x + radius - 0.5, y - radius + 0.5
            opts = {
                "text": text,
                "bold": False,
                "color_name": "id",
            }
            data = self.textItem.addObjAnnot((xt, yt), anchor=(-0.3, 1.3), **opts)
            self.textItem.appendData(data, opts["text"])

        self.textItem.draw()
        # hexColor = color.name()
        # htmlText = html_utils.span(
        #     text, color=hexColor, font_size='13pt', bold=True
        # )

        # textItem = self._textItems.get((x, y))
        # if textItem is None:
        #     textItem = pg.TextItem(html=htmlText, anchor=(0, 1))
        #     textItem.setParentItem(self)
        #     self._textItems[(x, y)] = textItem
        #     self.ax.addItem(textItem)
        # else:
        #     textItem.setHtml(htmlText)
        # textItem.setPos(x+radius-0.5, y-radius+0.5)

    def clearTextItems(self):
        self.textItem.clearData()
        # for textItem in self._textItems.values():
        #     textItem.setText('')

    def clear(self):
        super().clear()
        self.clearTextItems()

    def setVisible(self, visible):
        super().setVisible(visible)
        self.textItem.setVisible(visible)


class RectItem(pg.GraphicsObject):
    def __init__(self, rect, pen=None, brush=(255, 0, 0, 100), parent=None):
        super().__init__(parent)
        self._rect = rect
        self._pen = pg.mkPen(pen)
        self._brush = pg.mkBrush(brush)
        self.picture = QPicture()
        self._generate_picture()

    def setColor(self, color):
        rgba = matplotlib.colors.to_rgba(color, alpha=100 / 255)
        rgba = [round(c * 255) for c in rgba]
        self._brush = pg.mkBrush(rgba)
        self._generate_picture()
        self.update()

    def setRect(self, x, y, width, height):
        self._rect = QRectF(x, y, width, height)
        self._generate_picture()
        self.update()

    def setQRect(self, qrect):
        self._rect = qrect
        self._generate_picture()
        self.update()

    @property
    def rect(self):
        return self._rect

    def _generate_picture(self):
        painter = QPainter(self.picture)
        painter.setPen(self._pen)
        painter.setBrush(self._brush)
        painter.drawRect(self._rect)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())

# Sibling imports (deferred to avoid import cycles)
from .controls import (
    ComboBox,
    DoubleSpinBox,
    SpinBox,
    highlightableQWidgetAction,
)

