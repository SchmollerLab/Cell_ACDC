"""Composite controls: panels."""

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

from ..canvas.plot_items import (
    LabelItem,
)

class statusBarPermanentLabel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.rightLabel = QLabel("")
        self.leftLabel = QLabel("")

        layout = QHBoxLayout()
        layout.addWidget(self.leftLabel)
        layout.addStretch(10)
        layout.addWidget(self.rightLabel)

        self.setLayout(layout)


class listWidget(QListWidget):
    def __init__(
        self, *args, isMultipleSelection=False, minimizeHeight=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.itemHeight = None
        self.setStyleSheet(LISTWIDGET_STYLESHEET)
        self.setFont(font)
        if isMultipleSelection:
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.minimizeHeight = minimizeHeight

    def setSelectedAll(self, selected):
        for i in range(self.count()):
            self.item(i).setSelected(selected)

    def setSelectedItems(self, itemsText):
        for i in range(self.count()):
            item = self.item(i)
            item.setSelected(item.text() in itemsText)

    def addItems(self, labels) -> None:
        super().addItems(labels)
        if self.itemHeight is not None:
            self.setItemHeight()

        if self.minimizeHeight:
            itemHeight = self.sizeHintForRow(0)
            self.setMaximumHeight(itemHeight * self.count() + itemHeight * 2)

    def addItem(self, text):
        super().addItem(text)
        if self.itemHeight is None:
            return
        self.setItemHeight()

    def setItemHeight(self, height=40):
        self.itemHeight = height
        for i in range(self.count()):
            item = self.item(i)
            item.setSizeHint(QSize(0, height))

    def selectedItemsText(self):
        return [item.text() for item in self.selectedItems()]


class OrderableListWidget(QWidget):
    sigEnterEvent = Signal(object)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._labels = []

    def setParentItem(self, item):
        self._item = item

    def setLabelsColor(self, selected):
        if selected:
            stylesheet = "color : black"
        else:
            stylesheet = ""

        for label in self._labels:
            label.setStyleSheet(stylesheet)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.setLabelsColor(True)
        self.sigEnterEvent.emit(self._item)

    # def leaveEvent(self, event):
    #     super().leaveEvent(event)
    #     self.setLabelsColor(self._item.isSelected())
    #     printl('leave', self._item.isSelected())

    def addLabel(self, label):
        self._labels.append(label)
        self.validPattern = r"^[0-9,\.]+$"
        regExp = QRegularExpression(self.validPattern)
        self.setValidator(QRegularExpressionValidator(regExp))

    def values(self):
        try:
            vals = [float(c) for c in self.text().split(",")]
        except Exception as e:
            vals = []
        return vals


class KeptObjectIDsList(list):
    def __init__(self, lineEdit, confirmSelectionAction, *args):
        self.lineEdit = lineEdit
        self.lineEdit.setText("")
        self.confirmSelectionAction = confirmSelectionAction
        confirmSelectionAction.setDisabled(True)
        super().__init__(*args)

    def setText(self):
        text = utils.format_IDs(self)

        self.lineEdit.setText(text)

    def append(self, element, editText=True):
        super().append(element)
        if editText:
            self.setText()
        if not self.confirmSelectionAction.isEnabled():
            self.confirmSelectionAction.setEnabled(True)

    def remove(self, element, editText=True):
        super().remove(element)
        if editText:
            self.setText()
        if not self:
            self.confirmSelectionAction.setEnabled(False)


class Toggle(QCheckBox):
    def __init__(
        self,
        label_text="",
        initial=None,
        width=80,
        bg_color="#b3b3b3",
        circle_color="#ffffff",
        active_color="#26dd66",  # '#005ce6',
        animation_curve=QEasingCurve.Type.InOutQuad,
    ):
        QCheckBox.__init__(self)

        # self.setFixedSize(width, 28)
        self.setCursor(Qt.PointingHandCursor)

        self._label_text = label_text
        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color
        self._disabled_active_color = colors.lighten_color(active_color)
        self._disabled_circle_color = colors.lighten_color(circle_color)
        self._disabled_bg_color = colors.lighten_color(bg_color, amount=0.5)
        self._circle_margin = 4

        self._circle_position = int(self._circle_margin / 2)
        self.animation = QPropertyAnimation(self, b"circle_position", self)
        self.animation.setEasingCurve(animation_curve)
        self.animation.setDuration(200)

        self.stateChanged.connect(self.start_transition)
        self.requestedState = None

        self.installEventFilter(self)
        self._isChecked = False

        if initial is not None:
            self.setChecked(initial)

    def sizeHint(self):
        return QSize(36, 18)

    def eventFilter(self, object, event):
        # To get the actual position of the circle we need to wait that
        # the widget is visible before setting the state
        if event.type() == QEvent.Type.Show and self.requestedState is not None:
            self.setChecked(self.requestedState)
        return False

    def setChecked(self, state):
        # To get the actual position of the circle we need to wait that
        # the widget is visible before setting the state
        self._isChecked = state
        if self.isVisible():
            self.requestedState = None
            QCheckBox.setChecked(self, state > 0)
        else:
            self.requestedState = state

    def isChecked(self):
        if self.isVisible():
            return super().isChecked()
        else:
            return self._isChecked

    def circlePos(self, state: bool):
        start = int(self._circle_margin / 2)
        if state:
            if self.isVisible():
                height, width = self.height(), self.width()
            else:
                sizeHint = self.sizeHint()
                height, width = sizeHint.height(), sizeHint.width()
            circle_diameter = height - self._circle_margin
            pos = width - start - circle_diameter
        else:
            pos = start
        return pos

    @Property(float)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()

    def start_transition(self, state):
        self.animation.stop()
        pos = self.circlePos(state)
        self.animation.setEndValue(pos)
        self.animation.start()

    def hitButton(self, pos: QPoint):
        return self.contentsRect().contains(pos)

    def setDisabled(self, disable):
        QCheckBox.setDisabled(self, disable)
        if hasattr(self, "label"):
            self.label.setDisabled(disable)
        self.update()

    def paintEvent(self, e):
        circle_color = (
            self._circle_color if self.isEnabled() else self._disabled_circle_color
        )
        active_color = (
            self._active_color if self.isEnabled() else self._disabled_active_color
        )
        unchecked_color = (
            self._bg_color if self.isEnabled() else self._disabled_bg_color
        )

        # set painter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # set no pen
        p.setPen(Qt.NoPen)

        # draw rectangle
        rect = QRect(0, 0, self.width(), self.height())

        if not self.isChecked():
            # Draw background
            p.setBrush(QColor(unchecked_color))
            half_h = int(self.height() / 2)
            p.drawRoundedRect(0, 0, rect.width(), self.height(), half_h, half_h)

            # Draw circle
            p.setBrush(QColor(circle_color))
            p.drawEllipse(
                int(self._circle_position),
                int(self._circle_margin / 2),
                self.height() - self._circle_margin,
                self.height() - self._circle_margin,
            )
        else:
            # Draw background
            p.setBrush(QColor(active_color))
            half_h = int(self.height() / 2)
            p.drawRoundedRect(0, 0, rect.width(), self.height(), half_h, half_h)

            # Draw circle
            p.setBrush(QColor(circle_color))
            p.drawEllipse(
                int(self._circle_position),
                int(self._circle_margin / 2),
                self.height() - self._circle_margin,
                self.height() - self._circle_margin,
            )

        p.end()


class ToggleTerminalButton(PushButton):
    sigClicked = Signal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(":terminal_up.svg"))
        self.setFixedSize(34, 18)
        self.setIconSize(QSize(30, 14))
        self.setFlat(True)
        self.terminalVisible = False
        self.clicked.connect(self.mouseClick)

    def mouseClick(self):
        if self.terminalVisible:
            self.setIcon(QIcon(":terminal_up.svg"))
            self.terminalVisible = False
        else:
            self.setIcon(QIcon(":terminal_down.svg"))
            self.terminalVisible = True
        self.sigClicked.emit(self.terminalVisible)

    def showEvent(self, a0) -> None:
        self.idlePalette = self.palette()
        return super().showEvent(a0)

    def enterEvent(self, event) -> None:
        self.setFlat(False)
        # pal = self.palette()
        # pal.setColor(QPalette.ColorRole.Button, QColor(200, 200, 200))
        # self.setAutoFillBackground(True)
        # self.setPalette(pal)
        self.update()
        return super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.setFlat(True)
        # self.setPalette(self.idlePalette)
        self.update()
        return super().leaveEvent(event)


class expandCollapseButton(PushButton):
    sigClicked = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setIcon(QIcon(":expand.svg"))
        self.setFlat(True)
        self.installEventFilter(self)
        self.isExpand = True
        self.clicked.connect(self.buttonClicked)

    def buttonClicked(self, checked=False):
        if self.isExpand:
            self.setIcon(QIcon(":collapse.svg"))
            self.isExpand = False
            if self.text():
                self.setText(self.text().replace("Hide", "Show"))
        else:
            self.setIcon(QIcon(":expand.svg"))
            self.isExpand = True
            if self.text():
                self.setText(self.text().replace("Show", "Hide"))
        self.sigClicked.emit()

    def eventFilter(self, object, event):
        if event.type() == QEvent.Type.HoverEnter:
            self.setFlat(False)
        elif event.type() == QEvent.Type.HoverLeave:
            self.setFlat(True)
        return False


class ToggleVisibilityButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFlat(True)
        # self.setCheckable(True)
        self._state = False
        self.setIcon(QIcon(":unchecked.svg"))
        self.clicked.connect(self.onClicked)
        self.setStyleSheet("""
            QPushButton::pressed {
                background-color: none;
                border-style: none;
            }
        """)

    def onClicked(self):
        self._state = not self._state
        if self._state:
            self.setIcon(QIcon(":eye-checked.svg"))
        else:
            self.setIcon(QIcon(":unchecked.svg"))


class ToggleVisibilityCheckBox(QCheckBox):
    def __init__(self, *args, pixelSize=24):
        super().__init__(*args)
        self._pixelSize = pixelSize
        self.onToggled(False)
        self.toggled.connect(self.onToggled)

    def setPixelSize(self, pixelSize):
        self._pixelSize = pixelSize

    def onToggled(self, checked):
        if checked:
            self.setStyleSheet(f"""
                QCheckBox::indicator {{
                    width: {self._pixelSize}px;
                    height: {self._pixelSize}px;
                }}

                QCheckBox::indicator:checked
                {{
                    image: url(:eye-checked.svg);
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QCheckBox::indicator {{
                    width: {self._pixelSize}px;
                    height: {self._pixelSize}px;
                }}
                
                QCheckBox::indicator:unchecked
                {{
                    image: url(:unchecked.svg);
                }}
            """)


class FeatureSelectorButton(QPushButton):
    def __init__(self, text, parent=None, alignment=""):
        super().__init__(text, parent=parent)
        self._isFeatureSet = False
        self._alignment = alignment
        self.setCursor(Qt.PointingHandCursor)

    def setFeatureText(self, text):
        self.setText(text)
        self.setFlat(True)
        self._isFeatureSet = True
        if self._alignment:
            self.setStyleSheet(f"text-align:{self._alignment};")

    def enterEvent(self, event) -> None:
        if self._isFeatureSet:
            self.setFlat(False)
        return super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._isFeatureSet:
            self.setFlat(True)
        self.update()
        return super().leaveEvent(event)

    def setSizeLongestText(self, longestText):
        currentText = self.text()
        self.setText(longestText)
        w, h = self.sizeHint().width(), self.sizeHint().height()
        self.setMinimumWidth(w + 10)
        # self.setMinimumHeight(h+5)
        self.setText(currentText)


class CheckableSpinBoxWidgets:
    def __init__(self, isFloat=True):
        if isFloat:
            self.spinbox = FloatLineEdit()
        else:
            self.spinbox = SpinBox()
        self.checkbox = QCheckBox("Activate")
        self.spinbox.setEnabled(False)
        self.checkbox.toggled.connect(self.spinbox.setEnabled)

    def value(self):
        if not self.checkbox.isChecked():
            return
        return self.spinbox.value()


class Label(QLabel):
    def __init__(self, parent=None, force_html=False):
        super().__init__(parent)
        self._force_html = force_html

    def setText(self, text):
        if self._force_html:
            text = html_utils.paragraph(text)
        super().setText(text)


class LatexLabel(QLabel):
    def __init__(self, latexText, parent=None):
        super().__init__(parent)

        latexText = latexText.replace("<latex>", "$")
        if not latexText.startswith("$"):
            latexText = f"${latexText}"

        if not latexText.endswith("$"):
            latexText = f"{latexText}$"

        latexText = latexText.replace("<br>", "\n")

        pixmap = self.mathTex_to_QPixmap(latexText)
        self.setPixmap(pixmap)

    def mathTex_to_QPixmap(self, mathTex):
        # ---- set up a mpl figure instance ----

        fig = matplotlib.figure.Figure()
        fig.patch.set_facecolor("none")
        fig.set_canvas(FigureCanvasAgg(fig))
        renderer = fig.canvas.get_renderer()

        # ---- plot the mathTex expression ----

        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.patch.set_facecolor("none")
        t = ax.text(
            0, 0, mathTex, ha="left", va="bottom", fontsize=13, color=TEXT_COLOR
        )

        # ---- fit figure size to text artist ----

        fwidth, fheight = fig.get_size_inches()
        fig_bbox = fig.get_window_extent(renderer)

        text_bbox = t.get_window_extent(renderer)

        tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
        tight_fheight = text_bbox.height * fheight / fig_bbox.height

        fig.set_size_inches(tight_fwidth, tight_fheight)

        # ---- convert mpl figure to QPixmap ----

        buf, size = fig.canvas.print_to_buffer()
        qimage = QImage.rgbSwapped(QImage(buf, size[0], size[1], QImage.Format_ARGB32))
        qpixmap = QPixmap(qimage)

        return qpixmap


class SwitchPlaneCombobox(QComboBox):
    sigPlaneChanged = Signal(str, str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addItems(["xy", "zy", "zx"])
        self._previousPlane = "xy"
        self.currentTextChanged.connect(self.emitPlaneChanged)

    def emitPlaneChanged(self, plane):
        self.sigPlaneChanged.emit(self._previousPlane, plane)
        self._previousPlane = plane

    def setPlane(self, plane):
        self.setCurrentText(plane)

    def setCurrentText(self, text):
        self._previousPlane = self.plane()
        super().setCurrentText(text)

    def plane(self):
        return self.currentText()

    def depthAxes(self):
        plane = self.plane()
        for axes in "xyz":
            if axes not in plane:
                return axes


class CheckableAction(QAction):
    clicked = Signal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setCheckable(True)
        self.toggled.connect(self.emitClicked)

    def emitClicked(self, checked):
        self.clicked.emit(checked)

    def setChecked(self, checked):
        self.toggled.disconnect()
        super().setChecked(checked)
        self.toggled.connect(self.emitClicked)


class TimestampItem(LabelItem):
    sigEditProperties = Signal(object)
    sigRemove = Signal(object)

    def __init__(
        self,
        SizeY,
        SizeX,
        viewRange,
        secondsPerFrame=1,
        parent=None,
        start_timedelta=None,
    ):
        self._secondsPerFrame = secondsPerFrame
        self._x_pad = 3
        self._y_pad = 2
        self.xmin, self.ymin = 0, 0
        self.SizeY = SizeY
        self.SizeX = SizeX
        self._highlighted = False
        self._parent = parent
        if start_timedelta is None:
            start_timedelta = datetime.timedelta(seconds=0)
        self._start_timedelta = start_timedelta
        self.clicked = False
        super().__init__(self)
        self.updateViewRange(viewRange)
        self.createContextMenu()

    def setSecondsPerFrame(self, secondsPerFrame):
        self._secondsPerFrame = secondsPerFrame

    def getBboxViewRange(self, viewRange):
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

        return x0, y0, x1, y1

    def updateViewRange(self, viewRange):
        x0, y0, x1, y1 = self.getBboxViewRange(viewRange)

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

    def emitRemove(self):
        self.sigRemove.emit(self)

    def mousePressed(self, x, y):
        self.clicked = True

    def emitEditProperties(self):
        self.setHighlighted(False)
        self.sigEditProperties.emit(self.properties())

    def isHighlighted(self):
        return self._highlighted

    def setHighlighted(self, highlighted):
        if self._highlighted and highlighted:
            return

        if not self._highlighted and not highlighted:
            return

        super().setText(self.text, bold=highlighted)

        self._highlighted = highlighted

    def showContextMenu(self, x, y):
        self.contextMenu.popup(QPoint(int(x), int(y)))

    def setLocationProperty(self, loc: str):
        self._loc = loc

    def properties(self):
        properties = {
            "color": self._color,
            "loc": self._loc,
            "font_size": int(self._font_size[:-2]),
            "start_timedelta": self._start_timedelta,
            "move_with_zoom": self._move_with_zoom,
        }
        return properties

    def draw(self, frame_i, **kwargs):
        self.setProperties(**kwargs)
        self.update(frame_i)

    def update(self, frame_i):
        self.setPosFromLoc()
        self.setText(frame_i)

    def setMoveWithZoomProperty(self, move_with_zoom):
        self._move_with_zoom = move_with_zoom

    def updatePosViewRangeChanged(self, viewRange):
        if self._loc == "custom":
            textHeight = self.itemRect().height()
            textWidth = self.itemRect().width()
            x0p = self.pos().x()
            y0p = self.pos().y()
            xcp = x0p + textWidth / 2
            ycp = y0p + textHeight / 2
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
            X0p = Xcp - (textWidth / 2)
            Y0p = Ycp - (textHeight / 2)

            y_pos_max = self.ymax - textHeight - self._y_pad
            if Y0p > y_pos_max:
                Y0p = y_pos_max

            x_pos_max = self.xmax - textWidth - self._x_pad
            if X0p > x_pos_max:
                X0p = x_pos_max

            self.setPos(X0p, Y0p)
        else:
            self.updateViewRange(viewRange)
            self.setPosFromLoc()

    def setPosFromLoc(self):
        textHeight = self.itemRect().height()
        textWidth = self.itemRect().width()
        if self._loc == "custom":
            return

        if self._loc.find("top") != -1:
            y0 = self._y_pad + self.ymin
        else:
            y0 = self.ymax - textHeight - self._y_pad

        if self._loc.find("left") != -1:
            x0 = self._x_pad + self.xmin
        else:
            x0 = self.xmax - textWidth - self._x_pad

        self.setPos(x0, y0)

    def setProperties(
        self,
        color=(255, 255, 255),
        font_size="13px",
        loc="top-left",
        start_timedelta=None,
        move_with_zoom=False,
    ):
        if start_timedelta is not None:
            self._start_timedelta = start_timedelta
        self._color = color
        self._loc = loc
        self._font_size = font_size
        self._move_with_zoom = move_with_zoom

    def move(self, xm, ym):
        Dy = ym - self.yc
        Dx = xm - self.xc
        x0 = self.x0c + Dx
        y0 = self.y0c + Dy
        self.setPos(x0, y0)

    def mousePressed(self, x, y):
        self.clicked = True
        self.xc, self.yc = x, y
        self.x0c = self.pos().x()
        self.y0c = self.pos().y()

    def setText(self, frame_i):
        if not isinstance(frame_i, int):
            return

        seconds = frame_i * self._secondsPerFrame
        timedelta = datetime.timedelta(seconds=round(seconds))

        diff_seconds = timedelta.total_seconds() + self._start_timedelta.total_seconds()
        if diff_seconds >= 0:
            timedelta = datetime.timedelta(seconds=round(diff_seconds))
            text = str(timedelta)
        else:
            abs_diff = abs(
                timedelta.total_seconds() + self._start_timedelta.total_seconds()
            )
            abs_timedelta = datetime.timedelta(seconds=round(abs_diff))
            text = f"-{abs_timedelta}"

        # printl(timedelta)
        super().setText(text, color=self._color, size=self._font_size)

    def addToAxis(self, ax):
        ax.addItem(self)

    def removeFromAxis(self, ax):
        ax.removeItem(self)

# Cross-module imports (deferred to avoid import cycles)
from .inputs import (
    FloatLineEdit,
    SpinBox,
)

