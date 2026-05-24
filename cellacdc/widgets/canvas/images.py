"""Canvas widgets: images."""

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

# Cross-module imports (deferred to avoid import cycles)
from .plot_items import (
    myLabelItem,
)

