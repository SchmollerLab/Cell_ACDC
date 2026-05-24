"""Composite controls: forms."""

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

from ..canvas.scrollbars import (
    sliderWithSpinBox,
)
from .inputs import (
    ComboBox,
)

class selectStartStopFrames(QGroupBox):
    def __init__(self, SizeT, currentFrameNum=0, parent=None):
        super().__init__(parent)
        selectFramesLayout = QGridLayout()

        self.startFrame_SB = QSpinBox()
        self.startFrame_SB.setAlignment(Qt.AlignCenter)
        self.startFrame_SB.setMinimum(1)
        self.startFrame_SB.setMaximum(SizeT - 1)
        self.startFrame_SB.setValue(currentFrameNum)

        self.stopFrame_SB = QSpinBox()
        self.stopFrame_SB.setAlignment(Qt.AlignCenter)
        self.stopFrame_SB.setMinimum(1)
        self.stopFrame_SB.setMaximum(SizeT)
        self.stopFrame_SB.setValue(SizeT)

        selectFramesLayout.addWidget(QLabel("Start frame n."), 0, 0)
        selectFramesLayout.addWidget(self.startFrame_SB, 1, 0)

        selectFramesLayout.addWidget(QLabel("Stop frame n."), 0, 1)
        selectFramesLayout.addWidget(self.stopFrame_SB, 1, 1)

        self.warningLabel = QLabel()
        palette = self.warningLabel.palette()
        palette.setColor(self.warningLabel.backgroundRole(), Qt.red)
        palette.setColor(self.warningLabel.foregroundRole(), Qt.red)
        self.warningLabel.setPalette(palette)
        selectFramesLayout.addWidget(
            self.warningLabel, 2, 0, 1, 2, alignment=Qt.AlignCenter
        )

        self.setLayout(selectFramesLayout)

        self.stopFrame_SB.valueChanged.connect(self._checkRange)

    def _checkRange(self):
        start = self.startFrame_SB.value()
        stop = self.stopFrame_SB.value()
        if stop <= start:
            self.warningLabel.setText("stop frame smaller than start frame")
        else:
            self.warningLabel.setText("")


class formWidget(QWidget):
    sigApplyButtonClicked = Signal(object)
    sigComputeButtonClicked = Signal(object)

    def __init__(
        self,
        widget,
        initialVal=None,
        stretchWidget=True,
        widgetAlignment=None,
        labelTextLeft="",
        labelTextRight="",
        font=None,
        addInfoButton=False,
        addApplyButton=False,
        addComputeButton=False,
        addActivateCheckbox=False,
        key="",
        infoTxt="",
        valueGetterName="value",
        toolTip="",
        parent=None,
    ):
        QWidget.__init__(self, parent)
        self.widget = widget
        self.key = key
        self.infoTxt = infoTxt
        self.widgetAlignment = widgetAlignment
        self.valueGetterName = valueGetterName

        widget.setParent(self)

        if isinstance(initialVal, bool):
            widget.setChecked(initialVal)
        elif isinstance(initialVal, str):
            widget.setCurrentText(initialVal)
        elif isinstance(initialVal, float) or isinstance(initialVal, int):
            widget.setValue(initialVal)

        self.items = []

        if font is None:
            font = QFont()
            font.setPixelSize(13)

        self.labelLeft = QClickableLabel(widget)
        self.labelLeft.setText(labelTextLeft)
        self.labelLeft.setFont(font)
        self.items.append(self.labelLeft)

        if not stretchWidget:
            widgetLayout = QHBoxLayout()
            if widgetAlignment != "left":
                widgetLayout.addStretch(1)
            widgetLayout.addWidget(widget)
            if widgetAlignment != "right":
                widgetLayout.addStretch(1)
            self.items.append(widgetLayout)
        else:
            self.items.append(widget)

        self.labelRight = QClickableLabel(widget)
        self.labelRight.setText(labelTextRight)
        self.labelRight.setFont(font)
        self.items.append(self.labelRight)

        if toolTip:
            self.labelLeft.setToolTip(toolTip)
            self.widget.setToolTip(toolTip)
            self.labelRight.setToolTip(toolTip)

        if addInfoButton:
            infoButton = QPushButton(self)
            infoButton.setCursor(Qt.WhatsThisCursor)
            infoButton.setIcon(QIcon(":info.svg"))
            if labelTextLeft:
                infoButton.setToolTip(f'Info about "{self.labelLeft.text()}" parameter')
            else:
                infoButton.setToolTip(
                    f'Info about "{self.labelRight.text()}" measurement'
                )
            infoButton.clicked.connect(self.showInfo)
            self.infoButton = infoButton
            self.items.append(infoButton)

        if addApplyButton:
            applyButton = QPushButton(self)
            applyButton.setCursor(Qt.PointingHandCursor)
            applyButton.setCheckable(True)
            applyButton.setIcon(QIcon(":apply.svg"))
            applyButton.setToolTip(f"Apply this step and visualize results")
            applyButton.clicked.connect(self.applyButtonClicked)
            self.items.append(applyButton)

        if addComputeButton:
            computeButton = QPushButton(self)
            computeButton.setCursor(Qt.BusyCursor)
            computeButton.setIcon(QIcon(":compute.svg"))
            computeButton.setToolTip(f"Compute this step and visualize results")
            computeButton.clicked.connect(self.computeButtonClicked)
            self.items.append(computeButton)

        self.activateCheckbox = None
        if addActivateCheckbox:
            self.activateCheckbox = QCheckBox("Activate")
            self.activateCheckbox.setChecked(False)
            self.widget.setDisabled(True)
            self.activateCheckbox.toggled.connect(self.setWidgetEnabled)
            self.items.append(self.activateCheckbox)

        self.labelLeft.clicked.connect(self.tryChecking)
        self.labelRight.clicked.connect(self.tryChecking)

    def setWidgetEnabled(self, checked):
        self.widget.setDisabled(not checked)

    def value(self):
        if self.activateCheckbox is None:
            return getattr(self.widget, self.valueGetterName)()

        if not self.activateCheckbox.isChecked():
            return

        return getattr(self.widget, self.valueGetterName)()

    def tryChecking(self, label):
        try:
            self.widget.setChecked(not self.widget.isChecked())
        except AttributeError as e:
            pass

    def applyButtonClicked(self):
        self.sigApplyButtonClicked.emit(self)

    def computeButtonClicked(self):
        self.sigComputeButtonClicked.emit(self)

    def showInfo(self):
        msg = myMessageBox()
        msg.setIcon()
        msg.setWindowTitle(f"{self.labelLeft.text()} info")
        msg.addText(self.infoTxt)
        msg.addButton("   Ok   ")
        msg.exec_()

    def setDisabled(self, disabled: bool) -> None:
        for item in self.items:
            try:
                item.setDisabled(disabled)
            except Exception as err:
                pass


class CheckboxesGroupBox(QGroupBox):
    def __init__(self, texts, title="", checkable=False, parent=None):
        super().__init__(parent)

        self.setTitle(title)
        self.setCheckable(checkable)
        layout = QVBoxLayout()

        scrollLayout = QVBoxLayout()
        container = QWidget()
        scrollarea = QScrollArea()

        self.checkBoxes = []
        for text in texts:
            checkbox = QCheckBox(text)
            checkbox.setChecked(True)
            scrollLayout.addWidget(checkbox)
            self.checkBoxes.append(checkbox)

        container.setLayout(scrollLayout)
        scrollarea.setWidget(container)
        layout.addWidget(scrollarea)

        buttonsLayout = QHBoxLayout()
        selectAllButton = selectAllPushButton()
        selectAllButton.sigClicked.connect(self.checkAll)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(selectAllButton)
        layout.addLayout(buttonsLayout)

        self.setLayout(layout)

    def checkAll(self, button, checked):
        for checkBox in self.checkBoxes:
            checkBox.setChecked(checked)


class guiTabControl(QTabWidget):
    def __init__(self, *args):
        super().__init__(args[0])

        self._defaultPixelSize = None

        self.propsTab = QScrollArea(self)

        container = QWidget()
        layout = QVBoxLayout()

        self.pixelSizeQGBox = PixelSizeGroupbox(parent=self.propsTab)
        self.propsQGBox = objPropsQGBox(parent=self.propsTab)
        self.intensMeasurQGBox = objIntesityMeasurQGBox(parent=self.propsTab)

        self.highlightCheckbox = QCheckBox("Highlight objects on mouse hover")
        self.highlightCheckbox.setChecked(False)

        self.highlightSearchedCheckbox = QCheckBox("Highlight searched object")
        self.highlightSearchedCheckbox.setChecked(True)

        highlightLayout = QHBoxLayout()
        highlightLayout.addWidget(self.highlightCheckbox)
        highlightLayout.addStretch(1)
        highlightLayout.addWidget(QLabel("|"))
        highlightLayout.addStretch(1)
        highlightLayout.addWidget(self.highlightSearchedCheckbox)

        layout.addLayout(highlightLayout)
        layout.addWidget(self.pixelSizeQGBox)
        layout.addWidget(self.propsQGBox)
        layout.addWidget(self.intensMeasurQGBox)
        layout.addStretch(1)
        container.setLayout(layout)

        self.propsTab.setWidgetResizable(True)
        self.propsTab.setWidget(container)
        self.addTab(self.propsTab, "Measurements")

        self.pixelSizeQGBox.sigValueChanged.connect(self.pixelSizeChanged)
        self.pixelSizeQGBox.sigReset.connect(self.resetPixelSize)

    def addChannels(self, channels):
        self.intensMeasurQGBox.addChannels(channels)

    def resetPixelSize(self):
        if self._defaultPixelSize is None:
            return

        self.initPixelSize(*self._defaultPixelSize)

    def initPixelSize(self, PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ):
        self.pixelSizeQGBox.pixelWidthWidget.setValue(PhysicalSizeX)
        self.pixelSizeQGBox.pixelHeightWidget.setValue(PhysicalSizeY)
        self.pixelSizeQGBox.voxelDepthWidget.setValue(PhysicalSizeZ)
        self._defaultPixelSize = (PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)

    def pixelSizeChanged(self, PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ):
        propsQGBox = self.propsQGBox
        yx_pxl_to_um2 = PhysicalSizeY * PhysicalSizeX
        vox_rot_to_fl = float(PhysicalSizeY) * pow(float(PhysicalSizeX), 2)
        vox_3D_to_fl = PhysicalSizeZ * PhysicalSizeY * PhysicalSizeX

        area_pxl = propsQGBox.cellAreaPxlSB.value()
        area_um2 = area_pxl * yx_pxl_to_um2
        propsQGBox.cellAreaUm2DSB.setValue(area_um2)

        vol_rot_vox = propsQGBox.cellVolVoxSB.value()
        vol_rot_fl = vol_rot_vox * vox_rot_to_fl
        propsQGBox.cellVolFlDSB.setValue(vol_rot_fl)

        vol_3D_vox = propsQGBox.cellVolVox3D_SB.value()
        vol_3D_fl = vol_3D_vox * vox_3D_to_fl
        propsQGBox.cellVolFl3D_DSB.setValue(vol_3D_fl)


class PostProcessSegmSlider(sliderWithSpinBox):
    def __init__(self, *args, label=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.label = label
        self.checkbox = QCheckBox("Disable")
        self._layout.addWidget(self.checkbox, self.sliderCol, self.lastCol + 1)
        self.checkbox.toggled.connect(self.onCheckBoxToggled)
        self.valueChanged.connect(self.checkExpandRange)

    def onCheckBoxToggled(self, checked: bool) -> None:
        super().setDisabled(checked)
        if self.label is not None:
            self.label.setDisabled(checked)
        self.onValueChanged(None)
        self.onEditingFinished()

    def onValueChanged(self, value):
        self.valueChanged.emit(value)

    def checkExpandRange(self, value):
        if value == self.maximum():
            range = int(self.maximum() - self.minimum())
            half_range = int(range / 2)
            newMinimum = self.minimum() + half_range
            newMaximum = self.maximum() + half_range
            self.setMaximum(newMaximum)
            self.setMinimum(newMinimum)
        elif value == self.minimum():
            range = int(self.maximum() - self.minimum())
            half_range = int(range / 2)
            newMinimum = self.minimum() - half_range
            newMaximum = self.maximum() - half_range
            self.setMaximum(newMaximum)
            self.setMinimum(newMinimum)

    def onEditingFinished(self):
        self.editingFinished.emit()

    def value(self):
        if self.checkbox.isChecked():
            return None
        else:
            return super().value()


class PostProcessSegmSpinbox(QWidget):
    valueChanged = Signal(int)
    editingFinished = Signal()
    sigCheckboxToggled = Signal()

    def __init__(self, *args, isFloat=False, label=None, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QHBoxLayout()

        if isFloat:
            self.spinBox = DoubleSpinBox()
        else:
            self.spinBox = SpinBox()

        self.spinBox.valueChanged.connect(self.onValueChanged)
        self.spinBox.editingFinished.connect(self.onEditingFinished)

        layout.addWidget(self.spinBox)
        self.checkbox = QCheckBox("Disable")
        layout.addWidget(self.checkbox)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)

        self.label = label

        self.checkbox.toggled.connect(self.onCheckBoxToggled)

        layout.setContentsMargins(5, 0, 5, 0)

        self.setLayout(layout)

    def onCheckBoxToggled(self, checked: bool) -> None:
        self.spinBox.setDisabled(checked)
        if self.label is not None:
            self.label.setDisabled(checked)
        self.onValueChanged(None)
        self.onEditingFinished()

    def onValueChanged(self, value):
        self.valueChanged.emit(value)

    def onEditingFinished(self):
        self.editingFinished.emit()

    def maximum(self):
        return self.spinBox.maximum()

    def setValue(self, value):
        self.spinBox.setValue(value)

    def sizeHint(self):
        return self.spinBox.sizeHint()

    def setMaximum(self, max):
        self.spinBox.setMaximum(max)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setMinimum(self, min):
        self.spinBox.setMinimum(min)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setDecimals(self, decimals):
        self.spinBox.setDecimals(decimals)

    def value(self):
        if self.checkbox.isChecked():
            return None
        else:
            return self.spinBox.value()


class CopiableCommandWidget(QGroupBox):
    def __init__(self, command="", parent=None, font_size="13px"):
        super().__init__(parent)

        layout = QHBoxLayout()

        label = QLabel(self)
        self.label = label
        self._font_size = font_size
        self.setCommand(command, font_size=font_size)
        label.setTextInteractionFlags(
            Qt.TextBrowserInteraction | Qt.TextSelectableByKeyboard
        )
        layout.addWidget(label)
        layout.addWidget(QVLine(shadow="Plain", color="#4d4d4d"))
        copyButton = copyPushButton("Copy", flat=True, hoverable=True)
        copyButton.clicked.connect(self.copyToClipboard)
        layout.addWidget(copyButton)
        layout.addStretch(1)

        self.setLayout(layout)

    def setWordWrap(self, wordWrap):
        self.label.setWordWrap(wordWrap)

    def copyToClipboard(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(self._command, mode=cb.Clipboard)
        print("Command copied!")

    def setCommand(self, command, font_size=None):
        if font_size is None:
            font_size = self._font_size

        self._command = command
        txt = html_utils.paragraph(f"<code>{command}</code>", font_size=font_size)
        self.label.setText(txt)

    def command(self):
        return self._command

    def text(self):
        return self.label.text()

    def setTextInteractionFlags(self, flags):
        self.label.setTextInteractionFlags(flags)


class LabelsWidget(QWidget):
    def __init__(self, texts, wrapText=False, parent=None):
        super().__init__(parent=parent)

        layout = QVBoxLayout()

        texts = self.fixParagraphTags(texts)

        self.textLengths = []
        self.labels = []
        for t, text in enumerate(texts):
            if not text:
                continue

            if text.startswith("<latex>"):
                layout.addSpacing(10)
                label = LatexLabel(text)
                layout.addWidget(label, alignment=Qt.AlignCenter)
                try:
                    # Add spacing only if next text is not a formula
                    nextText = texts[t + 1]
                    if not nextText.startswith("<latex>"):
                        layout.addSpacing(10)
                except IndexError:
                    layout.addSpacing(10)
            elif text.startswith("<copiable>"):
                text = text.removeprefix("<copiable>").removeprefix("</copiable>")
                label = CopiableCommandWidget(command=text, parent=self)
                layout.addWidget(label)
            else:
                label = QLabel(text)
                label.setWordWrap(wrapText)
                label.setOpenExternalLinks(True)
                layout.addWidget(label)
                if wrapText:
                    self.textLengths.append(1)
                self.textLengths.extend([len(line) for line in text.split("<br>")])

            self.labels.append(label)

        self.nCharsLongestLine = max(self.textLengths, default=1)

        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def setWordWrap(self, wordWrap):
        for label in self.labels:
            label.setWordWrap(wordWrap)

    def fixParagraphTags(self, texts):
        firstText = texts[0]
        if firstText.find("<p style=") == -1:
            return texts

        searched = re.search(r'<p style="[\w\-\:\;]+">', firstText)
        if searched is None:
            openTag = '<p style="font-size:13px;">'
        else:
            openTag = searched.group()

        not_allowed = {" ", "\n"}

        fixedTexts = []
        for text in texts:
            if text.startswith("<latex>") or text.startswith("<copiable>"):
                fixedTexts.append(text)
                continue

            if set(text) <= not_allowed:
                # Ignore texts that are made of only \n and spaces
                continue

            if text.find("</p>") == -1:
                text = rf"{text}<\p>"

            if text.find(openTag) == -1:
                text = f"{openTag}{text}"

            text = text.replace("\n", "")

            fixedTexts.append(text)
        return fixedTexts


class SamInputPointsWidget(QWidget):
    sigValueChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        _layout = QHBoxLayout()

        self.lineEntry = ElidingLineEdit(parent=self)
        self.lineEntry.setAlignment(Qt.AlignCenter)
        self.lineEntry.editingFinished.connect(self.emitValueChanged)

        self.editButton = editPushButton()
        self.browseButton = browseFileButton(
            ext={"CSV": ".csv"}, start_dir=utils.getMostRecentPath()
        )

        _layout.addWidget(self.lineEntry)
        _layout.addWidget(self.editButton)
        _layout.addWidget(self.browseButton)

        _layout.setStretch(0, 1)
        _layout.setStretch(1, 0)
        _layout.setStretch(1, 0)

        self.browseButton.sigPathSelected.connect(self.browseCsvFiles)
        self.editButton.clicked.connect(self.showInfoEditPoints)

        _layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(_layout)

    def emitValueChanged(self, text):
        self.sigValueChanged.emit(text)

    def showInfoEditPoints(self):
        note = html_utils.to_note(
            "When adding points with the mouse left button you will create a "
            "new object for each point. To add multiple points for the same "
            "object click the right button."
        )
        txt = html_utils.paragraph(f"""
            To add input points for Segment Anything open the GUI (module 3), 
            load the data, and then click on the button<br>
            on the top toolbar called <code>Add points layer</code>.<br><br>
            Select the option "Add points by clicking" and click on the image 
            to add points.<br><br>
            Finally, save the table and browse to the saved file on this widget.
            <br>{note}
        """)
        msg = myMessageBox(wrapText=False)
        msg.information(self, "Info edit points", txt)

    def criticalMissingColumn(self, filepath, missing_col):
        txt = html_utils.paragraph(f"""
            [ERROR]: The selected table does not contain the column 
            <code>{missing_col}</code>.<br><br>
            A valid table must contain the columns <code>(x, y, id)</code> 
            with an additional <code>z</code> column for 3D z-stacks data.
        """)
        msg = myMessageBox(wrapText=False)
        msg.critical(self, "Invalid table", txt)

    def setValue(self, value: str):
        self.lineEntry.setText(value)

    def value(self):
        return self.lineEntry.text()

    def cast_dtype(self, value) -> str:
        return str(value)

    def browseCsvFiles(self, filepath):
        # Check if metadata.csv file exists with basename and set only the
        # endname of the file
        df_points = pd.read_csv(filepath)
        for col in ("x", "y", "id"):
            if col not in df_points.columns:
                self.criticalMissingColumn(filepath, col)
                return

        # Check if basename is present in metadata
        folderpath = os.path.dirname(filepath)
        basename = None
        for file in utils.listdir(folderpath):
            if file.endswith("metadata.csv"):
                metadata_csv_path = os.path.join(folderpath, file)
                df = pd.read_csv(metadata_csv_path, index_col="Description")
                try:
                    basename = df.at["basename", "values"]
                except Exception as e:
                    basename = None
                break

        # Check if file is inside images folder and get basename
        is_images_folder = folderpath.endswith("Images")
        if is_images_folder:
            images_path = folderpath
            img_filepath = None
            for file in utils.listdir(images_path):
                if file.endswith(".tif"):
                    img_filepath = os.path.join(images_path, file)
                    break

                if file.endswith("aligned.npz"):
                    img_filepath = os.path.join(images_path, file)
                    break

            if img_filepath is not None:
                posData = load.loadData(img_filepath, "", QParent=self)
                posData.getBasenameAndChNames()
                filename = os.path.basename(filepath)
                if filename.startswith(posData.basename):
                    basename = posData.basename

        if basename is None:
            self.lineEntry.setText(filepath)
        else:
            filename = os.path.basename(filepath)
            endname = filename[len(basename) :]
            self.lineEntry.setText(endname)


class FontSizeWidget(QWidget):
    sigTextChanged = Signal(str)

    def __init__(self, parent=None, unit="px", initalVal=12):
        super().__init__(parent)

        layout = QHBoxLayout()

        self.spinbox = SpinBox()
        self.spinbox.setValue(initalVal)
        layout.addWidget(self.spinbox)

        self.unitLabel = QLabel(unit)
        layout.addWidget(self.unitLabel)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)

        self.setLayout(layout)

        self.spinbox.valueChanged.connect(self.emitTextChanged)

    def emitTextChanged(self, value):
        self.sigTextChanged.emit(self.text())

    def setValue(self, value):
        if isinstance(value, str):
            value = int(value.replace(self.unitLabel.text(), "").strip())
        self.spinbox.setValue(value)

    def setText(self, text):
        value = int(text.replace(self.unitLabel.text(), "").strip())
        self.setValue(value)

    def text(self):
        return f"{self.spinbox.value()}{self.unitLabel.text()}"

    def value(self):
        return self.spinbox.value()


class RangeSelector(QWidget):
    sigRangeChanged = Signal(object, object)
    sigLowValueChanged = Signal(object)
    sigHighValueChanged = Signal(object)
    sigRangeManuallyChanged = Signal(object, object)

    def __init__(self, parent=None, integers=False, ordered=True):
        super().__init__(parent)

        self._integers = integers
        self._ordered = ordered

        layout = QHBoxLayout()

        if integers:
            self.lowSpinbox = SpinBox()
            self.highSpinbox = SpinBox()
        else:
            self.lowSpinbox = DoubleSpinBox()
            self.highSpinbox = DoubleSpinBox()

        layout.addWidget(self.lowSpinbox)
        layout.addWidget(self.highSpinbox)

        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.lowSpinbox.valueChanged.connect(self.lowValueChanged)
        self.highSpinbox.valueChanged.connect(self.highValueChanged)

        self.lowSpinbox.editingFinished.connect(self.lowValueEditingFinished)
        self.highSpinbox.editingFinished.connect(self.highValueEditingFinished)

    def lowValueEditingFinished(self):
        self.sigRangeManuallyChanged.emit(*self.range())
        self.emitRangeChanged()

    def highValueEditingFinished(self):
        self.sigRangeManuallyChanged.emit(*self.range())
        self.emitRangeChanged()

    def lowValueChanged(self, value):
        self.emitRangeChanged()
        self.sigLowValueChanged.emit(value)

    def highValueChanged(self, value):
        self.emitRangeChanged()
        self.sigHighValueChanged.emit(value)

    def emitRangeChanged(self):
        self.sigRangeChanged.emit(*self.range())

    def setRangeNoEmit(self, lowValue, highValue, decimals=3):
        self.lowSpinbox.valueChanged.disconnect()
        self.highSpinbox.valueChanged.disconnect()

        self.setRange(round(lowValue, 3), round(highValue, 3))

        self.lowSpinbox.valueChanged.connect(self.lowValueChanged)
        self.highSpinbox.valueChanged.connect(self.highValueChanged)

    def setRange(self, lowValue, highValue):
        # if lowValue > highValue and self._ordered:
        #     highValue = lowValue + 1

        if self._integers:
            lowValue = round(lowValue)
            highValue = round(highValue)

        self.lowSpinbox.setValue(lowValue)
        self.highSpinbox.setValue(highValue)

    def range(self):
        return self.lowSpinbox.value(), self.highSpinbox.value()


class PreProcessingSelector(QComboBox):
    sigValuesChanged = Signal(dict, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent

        self.addItems(PREPROCESS_MAPPER.keys())
        self.methodToDefaultValuesMapper = {}
        self.step_n = -1
        self.setParamsWindow = None

    def htmlInfo(self):
        href = html_utils.href_tag("GitHub page", urls.issues_url)
        docstring = PREPROCESS_MAPPER[self.currentText()]["docstring"]
        if docstring is None:
            text = "This function is not documented, yet. Sorry :("
        else:
            text = html_utils.rst_docstring_to_html(docstring)
        text = (
            f"{text}<br><br>"
            f"Feel free to submit an issue on our {href} if you "
            "need help with this filter."
        )
        return text

    def setParams(self, method: str, kwargToValueMapper: Dict[str, str]):
        self.methodToDefaultValuesMapper[method] = kwargToValueMapper

    def askSetParams(self, df_metadata=None, addApplyButton=False):
        method = self.currentText()
        function = PREPROCESS_MAPPER[method]["function"]
        params_argspecs = utils.get_function_argspec(
            function,
            args_to_skip={"logger_func", "apply_to_all_zslices", "apply_to_all_frames"},
        )
        default_values = self.methodToDefaultValuesMapper.get(method, {})
        for kwarg, value in default_values.items():
            for p, param_argspec in enumerate(params_argspecs):
                if param_argspec.name != kwarg:
                    continue

                if hasattr(param_argspec.type, "cast_dtype"):
                    cls = param_argspec.type
                    value = cls.cast_dtype(value)
                else:
                    value = param_argspec.type(value)

                if value == param_argspec.default:
                    continue
                param_argspec = param_argspec._replace(default=value)
                params_argspecs[p] = param_argspec

        if self.setParamsWindow is not None:
            self.setParamsWindow.raise_()
            self.setParamsWindow.activateWindow()
            return

        self.setParamsWindow = apps.FunctionParamsDialog(
            params_argspecs,
            df_metadata=df_metadata,
            function_name=method,
            addApplyButton=addApplyButton,
            parent=self._parent,
        )
        self.setParamsWindow.sigValuesChanged.connect(self.emitValuesChanged)
        self.setParamsWindow.emitValuesChanged()
        self.setParamsWindow.exec_()
        if self.setParamsWindow.cancel:
            return

        self.setParams(method, self.setParamsWindow.function_kwargs)

        function_kwargs = self.setParamsWindow.function_kwargs
        self.setParamsWindow = None

        return function_kwargs

    def emitValuesChanged(self, functionKwargs: dict):
        self.sigValuesChanged.emit(functionKwargs, self.step_n)


class RescaleImageJroisGroupbox(QGroupBox):
    def __init__(self, TZYX_out_shape, parent=None):
        super().__init__(parent)

        self.setTitle("Rescale ROIs")
        self.setCheckable(True)

        gridLayout = QGridLayout()

        dims = ("Z", "Y", "X")
        self.widgets = {}
        for row, SizeD in enumerate(TZYX_out_shape[1:]):
            if SizeD == 1:
                continue

            dim = dims[row]
            inputSpinbox = SpinBox()
            inputSpinbox.setMinimum(1)
            inputSpinbox.setValue(SizeD)

            outZwidget = QLineEdit()
            outZwidget.setReadOnly(True)
            outZwidget.setAlignment(Qt.AlignCenter)
            # outZwidget.setValue(SizeD)
            outZwidget.setText(str(SizeD))

            row0 = row * 2
            row1 = row0 + 1
            gridLayout.addWidget(QLabel(f"{dim}-dimension: "), row1, 0)

            gridLayout.addWidget(QLabel("Input size"), row0, 1)
            gridLayout.addWidget(inputSpinbox, row1, 1)

            gridLayout.addWidget(QLabel("Output size"), row0, 2)
            gridLayout.addWidget(outZwidget, row1, 2)

            self.widgets[dim] = (inputSpinbox, SizeD)

        self.setLayout(gridLayout)

    def inputOutputSizes(self):
        if not self.isChecked():
            return

        sizes = {
            dim: (spinbox.value(), int(SizeD))
            for dim, (spinbox, SizeD) in self.widgets.items()
        }
        return sizes


class TimeWidget(QGroupBox):
    sigValueChanged = Signal(object)

    def __init__(self, parent=None, orientation="vertical"):
        super().__init__(parent)

        mainLayout = QHBoxLayout()

        if orientation == "vertical":
            spinboxesLayout = QVBoxLayout()
        elif orientation == "horizontal":
            spinboxesLayout = QHBoxLayout()
        else:
            raise ValueError('orientation must be "vertical" or "horizontal"')

        self.signCombobox = QComboBox()
        self.signCombobox.addItems(("+", "-"))
        self.signCombobox.currentTextChanged.connect(self.emitValueChanged)

        mainLayout.addWidget(self.signCombobox)

        self.spinboxesMapper = {}
        units = ("days", "hours", "minutes", "seconds")
        for unit in units:
            layout = QHBoxLayout()
            spinbox = SpinBox()
            spinbox.setMinimum(0)
            label = QLabel(unit)
            layout.addWidget(spinbox)
            layout.addWidget(label)
            spinbox.valueChanged.connect(self.emitValueChanged)
            self.spinboxesMapper[unit] = spinbox
            spinboxesLayout.addLayout(layout)

        mainLayout.addLayout(spinboxesLayout)

        self.setLayout(mainLayout)
        mainLayout.setContentsMargins(5, 5, 5, 5)

    def values(self):
        values = {}
        for unit, spinbox in self.spinboxesMapper.items():
            values[unit] = spinbox.value()

        signText = self.signCombobox.currentText()
        return values, sign_int_mapper[signText]

    def setValuesFromTimedelta(self, timedelta):
        total_seconds = timedelta.total_seconds()
        sign = 1 if total_seconds > 0 else -1
        days = timedelta.days
        hours, remainder = divmod(timedelta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        values = {"days": days, "hours": hours, "minutes": minutes, "seconds": seconds}

        self.setValues(values, sign=sign)

    def timedelta(self):
        values, sign = self.values()
        return datetime.timedelta(**values) * sign

    def setValues(self, values: dict[str, int | float], sign=1):
        signText = "+" if sign > 0 else "-"
        self.signCombobox.setCurrentText(signText)
        for unit, value in values.items():
            spinbox = self.spinboxesMapper[unit]
            spinbox.setValue(value)

    def emitValueChanged(self, value):
        self.sigValueChanged.emit(self.values())


class YeazV2SelectModelNameCombobox(ComboBox):
    sigValueChanged = Signal(str)

    def __init__(
        self, *args, custom_select_item_text="Select custom weights file...", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._csi_text = custom_select_item_text
        self.sigTextChanged.connect(self.onTextChanged)
        self.initItems()

    def initItems(self):
        from cellacdc.segmenters.YeaZ_v2 import load_models_filepath

        models_name, models_name_filepath_mapper = load_models_filepath()
        self.addItems(models_name)

    def onTextChanged(self, text):
        if text != self._csi_text:
            return

        start_dir = utils.getMostRecentPath()
        model_filepath = qtpy.compat.getopenfilename(
            parent=self,
            caption="Select YeaZ weights file",
            filters="All Files (*)",
            basedir=start_dir,
        )[0]
        if not model_filepath:
            self.setCurrentIndex(0)
            return

        msg = html_utils.paragraph(f"""
        Insert a <b>name</b> for the following YeaZ model:<br><br>
        <code>{model_filepath}</code><br>
        """)
        modelNameWindow = apps.QLineEditDialog(
            title="Insert a name for the model", msg=msg, allowEmpty=False, parent=self
        )
        modelNameWindow.exec_()
        if modelNameWindow.cancel:
            self.setCurrentIndex(0)
            return

        model_name = modelNameWindow.enteredValue

        from cellacdc.segmenters.YeaZ_v2 import add_model_filepath

        add_model_filepath(model_name, model_filepath)

        self.addItem(model_name)
        self.setCurrentText(model_name)

        print(
            "YeaZ_v2 model added!\n\n"
            f"  * Name: {model_name}\n"
            f"  * File path: {model_filepath}\n"
        )

    def addItem(self, item):
        idx = self.count() - 1
        self.insertItem(idx, item)

    def addItems(self, items):
        super().clear()
        super().addItems(items)
        super().addItem(self._csi_text)
        idx = len(items)
        font = self.font()
        font.setItalic(True)
        self.setItemData(idx, font, Qt.FontRole)

    def setValue(self, value: str):
        self.setCurrentText(value)

    def value(self, *args):
        return self.currentText()


class AutoSaveIntervalWidget(QWidget):
    sigValueChanged = Signal(float, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()

        autoSaveIntervalTooltip = "Autosave every minutes or frames specified here."

        self.setToolTip(autoSaveIntervalTooltip)

        self.spinbox = DoubleSpinBox()
        self.spinbox.setMinimum(0)
        self.spinbox.setValue(2)
        self.spinbox.setDecimals(2)
        self.spinbox.setSingleStep(1.0)

        layout.addWidget(self.spinbox)

        self.unitCombobox = ComboBox()
        self.unitCombobox.addItems(["minutes", "frames"])
        layout.addWidget(self.unitCombobox)

        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        layout.setContentsMargins(5, 0, 5, 0)

        self.setLayout(layout)

        self.spinbox.sigValueChanged.connect(self.emitSigValueChanged)
        self.unitCombobox.sigTextChanged.connect(self.emitSigValueChanged)

    def emitSigValueChanged(self, *args, **kwargs):
        self.sigValueChanged.emit(self.spinbox.value(), self.unitCombobox.currentText())


class CheckableWidget(QWidget):
    def __init__(self, widget, valueGetterName="value", parent=None):
        super().__init__(parent)

        self.widget = widget
        self.valueGetterName = valueGetterName

        widget.setDisabled(True)

        layout = QHBoxLayout()

        layout.addWidget(widget)

        self.checkbox = QCheckBox("Activate")
        self.checkbox.toggled.connect(self.setWidgetEnabled)

        layout.addSpacing(5)
        layout.addWidget(self.checkbox)

        layout.setContentsMargins(5, 0, 5, 0)

        self.setLayout(layout)

    def setWidgetEnabled(self, checked):
        self.widget.setDisabled(not checked)

    def value(self):
        if not self.checkbox.isChecked():
            return

        return getattr(self.widget, self.valueGetterName)()

# Cross-module imports (deferred to avoid import cycles)
from .dialogs import (
    myMessageBox,
)
from .inputs import (
    DoubleSpinBox,
    QClickableLabel,
    SpinBox,
)
from .metrics import (
    PixelSizeGroupbox,
    objIntesityMeasurQGBox,
    objPropsQGBox,
)
from .panels import (
    LatexLabel,
)

