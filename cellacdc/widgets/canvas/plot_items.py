"""Canvas widgets: plot_items."""

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
        mask, self._slice = utils.clipSelemMask(mask, shape, Yc, Xc, copy=False)
        return mask


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
