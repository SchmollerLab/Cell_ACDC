"""Canvas widgets: rois."""

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

class PolyLineROI(pg.PolyLineROI):
    def __init__(self, positions, closed=False, pos=None, **args):
        super().__init__(positions, closed, pos, **args)


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
