"""Canvas widgets: imshow."""

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

# Cross-module imports (deferred to avoid import cycles)
from .images import (
    _ImShowImageItem,
    labImageItem,
)
from .plot_items import (
    RectItem,
)
from .scrollbars import (
    ScrollBarWithNumericControl,
)
from ..controls.inputs import (
    ComboBox,
)

