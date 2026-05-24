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

from .canvas import (
    LabelItem,
    sliderWithSpinBox,
)
from .toolbars import (
    ToolBar,
    rightClickToolButton,
)

class QDialogListbox(QDialog):
    sigSelectionConfirmed = Signal(list)

    def __init__(
        self,
        title,
        text,
        items,
        cancelText="Cancel",
        multiSelection=True,
        parent=None,
        additionalButtons=(),
        includeSelectionHelp=False,
        allowSingleSelection=True,
        preSelectedItems=None,
        allowEmptySelection=True,
    ):
        self.cancel = True
        items = list(items)

        super().__init__(parent)
        self.setWindowTitle(title)

        if preSelectedItems is None:
            if items:
                preSelectedItems = (items[0],)
            else:
                preSelectedItems = set()

        self.allowSingleSelection = allowSingleSelection
        self.allowEmptySelection = allowEmptySelection

        mainLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        bottomLayout = QHBoxLayout()

        self.mainLayout = mainLayout

        label = QLabel(text)
        _font = QFont()
        _font.setPixelSize(13)
        label.setFont(_font)
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 3px 0px;")
        topLayout.addWidget(label, alignment=Qt.AlignCenter)

        if includeSelectionHelp:
            selectionHelpLabel = QLabel()
            txt = html_utils.paragraph("""<br>
                <code>Ctrl+Click</code> <i>to select multiple items</i><br>
                <code>Shift+Click</code> <i>to select a range of items</i><br>
            """)
            selectionHelpLabel.setText(txt)
            topLayout.addWidget(label, alignment=Qt.AlignCenter)

        listBox = listWidget()
        listBox.setFont(_font)
        listBox.addItems(items)
        if multiSelection:
            listBox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            listBox.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        listBox.setCurrentRow(0)
        for i in range(listBox.count()):
            item = listBox.item(i)
            item.setSelected(item.text() in preSelectedItems)

        self.listBox = listBox
        if not multiSelection:
            listBox.itemDoubleClicked.connect(self.ok_cb)
        topLayout.addWidget(listBox)

        if cancelText.lower().find("cancel") != -1:
            cancelButton = cancelPushButton(cancelText)
        else:
            cancelButton = QPushButton(cancelText)
        okButton = okPushButton(" Ok ")

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)

        if additionalButtons:
            self._additionalButtons = []
            for button in additionalButtons:
                if isinstance(button, str):
                    _button, isCancelButton = getPushButton(button)
                    self._additionalButtons.append(_button)
                    bottomLayout.addWidget(_button)
                    _button.clicked.connect(self.ok_cb)
                else:
                    bottomLayout.addWidget(button)

        bottomLayout.addWidget(okButton)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        if multiSelection:
            listBox.itemClicked.connect(self.onItemClicked)
            listBox.itemSelectionChanged.connect(self.onItemSelectionChanged)

        self.setStyleSheet(LISTWIDGET_STYLESHEET)
        self.areItemsSelected = [
            listBox.item(i).isSelected() for i in range(listBox.count())
        ]
        self.setFont(font)

    def keyPressEvent(self, event) -> None:
        mod = event.modifiers()
        if mod == Qt.ShiftModifier or mod == Qt.ControlModifier:
            self.listBox.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
        elif event.key() == Qt.Key_Escape:
            self.listBox.clearSelection()
            event.ignore()
            return
        super().keyPressEvent(event)

    def onItemSelectionChanged(self):
        if not self.listBox.selectedItems():
            self.areItemsSelected = [False for i in range(self.listBox.count())]

    def onItemClicked(self, item):
        mod = QGuiApplication.keyboardModifiers()
        if mod == Qt.ShiftModifier or mod == Qt.ControlModifier:
            self.listBox.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
            return

        self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        itemIdx = self.listBox.row(item)
        wasSelected = self.areItemsSelected[itemIdx]
        if wasSelected:
            item.setSelected(False)

        self.areItemsSelected = [
            self.listBox.item(i).isSelected() for i in range(self.listBox.count())
        ]
        # self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # else:
        #     selectedItems.append(item)

        # self.listBox.clearSelection()
        # for i in range(self.listBox.count()):
        #     item = self.listBox.item(i).setSelected(True)

        # print(self.listBox.selectedItems())

    def setSelectedItems(self, itemsTexts):
        for i in range(self.listBox.count()):
            item = self.listBox.item(i)
            if item.text() in itemsTexts:
                item.setSelected(True)
        self.listBox.update()

    def warnSelectionEmpty(self):
        msg = myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph(
            "You need to <b>select at least one item!</b>.<br><br>"
            "Use <code>Ctrl+Click</code> to select multiple items<br>"
            "or <code>Shift+Click</code> to select a range of items"
        )
        msg.warning(self, "Selection cannot be empty!", txt)

    def ok_cb(self, checked=False):
        self.clickedButton = self.sender()
        self.cancel = False
        selectedItems = self.listBox.selectedItems()
        self.selectedItemsText = [item.text() for item in selectedItems]
        if not self.allowSingleSelection and len(self.selectedItemsText) < 2:
            msg = myMessageBox(wrapText=False, showCentered=False)
            txt = html_utils.paragraph(
                "You need to <b>select two or more items</b>.<br><br>"
                "Use <code>Ctrl+Click</code> to select multiple items<br>, or<br>"
                "<code>Shift+Click</code> to select a range of items"
            )
            msg.warning(self, "Select two or more items", txt)
            return

        if not self.allowEmptySelection and not self.selectedItemsText:
            self.warnSelectionEmpty()
            return

        self.sigSelectionConfirmed.emit(self.selectedItemsText)
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.selectedItemsText = None
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()

        horizontal_sb = self.listBox.horizontalScrollBar()
        while horizontal_sb.isVisible():
            self.resize(self.height(), self.width() + 10)

        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class ExpandableListBox(QComboBox):
    def __init__(self, parent=None, centered=True) -> None:
        super().__init__(parent)

        self.setEditable(True)
        self.lineEdit().setReadOnly(True)

        infoTxt = html_utils.paragraph(
            "Select <b>Positions to save</b><br><br>"
            "<code>Ctrl+Click</code> <i>to select multiple items</i><br>"
            "<code>Shift+Click</code> <i>to select a range of items</i><br>",
            center=True,
        )

        self.listW = QDialogListbox(
            "Select Positions to save", infoTxt, [], multiSelection=True, parent=self
        )

        self.listW.listBox.itemClicked.connect(self.listItemClicked)
        self.listW.sigSelectionConfirmed.connect(self.updateCombobox)

        self.centered = centered

    def listItemClicked(self, item):
        if item.text().find("All") == -1:
            return

        for i in range(self.listW.listBox.count()):
            _item = self.listW.listBox.item(i)
            _item.setSelected(True)

    def clear(self) -> None:
        self.listW.listBox.clear()
        return super().clear()

    def setItems(self, items):
        self.clear()
        self.addItems(items)

    def addItems(self, items):
        super().addItems(items)
        self.listW.listBox.addItems(items)
        self.listW.listBox.setCurrentRow(self.currentIndex())
        self.listItemClicked(self.listW.listBox.currentItem())
        if self.centered:
            self.centerItems()

    def updateCombobox(self, selectedItemsText):
        isAllItem = [i for i, t in enumerate(selectedItemsText) if t.find("All") != -1]
        if len(selectedItemsText) == 1:
            self.setCurrentText(selectedItemsText[0])
        elif isAllItem:
            idx = isAllItem[0]
            self.setCurrentText(selectedItemsText[idx])
        else:
            super().clear()
            super().addItems(["Custom selection"])

    def centerItems(self, idx=None):
        self.lineEdit().setAlignment(Qt.AlignCenter)

    def selectedItems(self):
        return self.listW.listBox.selectedItems()

    def selectedItemsText(self):
        return [item.text() for item in self.selectedItems()]

    def showPopup(self) -> None:
        self.listW.show()


class QClickableLabel(QLabel):
    clicked = Signal(object)

    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent)
        self._checkableItem = None

    def setCheckableItem(self, widget):
        self._checkableItem = widget

    def mousePressEvent(self, event):
        self.clicked.emit(self)
        if self._checkableItem is not None:
            status = not self._checkableItem.isChecked()
            self._checkableItem.setChecked(status)

    def setChecked(self, checked):
        self._checkableItem.setChecked(checked)


class QCenteredComboBox(QComboBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setAlignment(Qt.AlignCenter)
        self.lineEdit().installEventFilter(self)

        self.currentIndexChanged.connect(self.centerItems)

        self._isPopupVisibile = False

    def centerItems(self, idx):
        for i in range(self.count()):
            self.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)

    def eventFilter(self, lineEdit, event):
        # Reimplement show popup on click
        if event.type() == QEvent.Type.MouseButtonPress and self.isEnabled():
            if self._isPopupVisibile:
                self.hidePopup()
                self._isPopupVisibile = False
            else:
                self.showPopup()
                self._isPopupVisibile = True
            return True
        return False


class AlphaNumericComboBox(QCenteredComboBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)

    def addItems(self, items):
        self._dtype = type(items[0])
        super().addItems([str(item) for item in items])

    def setCurrentValue(self, value):
        super().setCurrentText(str(value))

    def currentValue(self):
        return self._dtype(super().currentText())


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


class mySpinBox(QSpinBox):
    sigTabEvent = Signal(object, object)

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def event(self, event):
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key_Tab:
            self.sigTabEvent.emit(event, self)
            return True

        return super().event(event)


class KeptObjectIDsList(list):
    def __init__(self, lineEdit, confirmSelectionAction, *args):
        self.lineEdit = lineEdit
        self.lineEdit.setText("")
        self.confirmSelectionAction = confirmSelectionAction
        confirmSelectionAction.setDisabled(True)
        super().__init__(*args)

    def setText(self):
        text = myutils.format_IDs(self)

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


class myMessageBox(_base_widgets.QBaseDialog):
    def __init__(
        self,
        parent=None,
        showCentered=True,
        wrapText=True,
        scrollableText=False,
        enlargeWidthFactor=0,
        resizeButtons=True,
        allowClose=True,
    ):
        super().__init__(parent)

        self.wrapText = wrapText
        self.enlargeWidthFactor = enlargeWidthFactor
        self.resizeButtons = resizeButtons

        self.cancel = True
        self.cancelButton = None
        self.okButton = None
        self.clickedButton = None
        self.alreadyShown = False
        self.allowClose = allowClose

        self.showCentered = showCentered

        self.scrollableText = scrollableText

        self._layout = QGridLayout()
        self.commandsLayout = None
        self._layout.setHorizontalSpacing(20)
        self.buttonsLayout = QHBoxLayout()
        self.buttonsLayout.setSpacing(2)
        self.buttons = []
        self.widgets = []
        self.layouts = []
        self.labels = []
        self.labelsWidgets = []
        self._pixmapLabels = []
        self.detailsTextWidget = None
        self.showInFileManagButton = None
        self.visibleDetails = False
        self.doNotShowAgainCheckbox = None

        self.currentRow = 0
        self.textWidget = None
        self._w = None

        self.textLayout = QVBoxLayout()

        self._layout.setColumnStretch(1, 1)
        self.setLayout(self._layout)

        self.setFont(font)

    def mousePressEvent(self, event):
        for label in self.labels:
            label.setTextInteractionFlags(
                Qt.TextBrowserInteraction | Qt.TextSelectableByKeyboard
            )

    def setIcon(self, iconName="SP_MessageBoxInformation"):
        label = QLabel(self)

        standardIcon = getattr(QStyle, iconName)
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)

        self._layout.addWidget(label, 0, 0, alignment=Qt.AlignTop)

    def addImage(self, image_path):
        pixmap = QPixmap(image_path)
        label = QLabel()
        label.setPixmap(pixmap)
        self._layout.addWidget(label, self.currentRow, 1)
        self.currentRow += 1

    def addShowInFileManagerButton(self, path, txt=None):
        if txt is None:
            txt = "Reveal in Finder..." if is_mac else "Show in Explorer..."
        self.showInFileManagButton = showInFileManagerButton(txt)
        self.buttonsLayout.addWidget(self.showInFileManagButton)
        func = partial(myutils.showInExplorer, path)
        self.showInFileManagButton.clicked.connect(func)

    def addBrowseUrlButton(self, url, button_text=""):
        self.openUrlButton = OpenUrlButton(url, button_text)
        self.buttonsLayout.addWidget(self.openUrlButton)

    def addCancelButton(self, button=None, connect=False):
        if button is None:
            self.cancelButton = cancelPushButton("Cancel")
        else:
            self.cancelButton = button
            self.cancelButton.setIcon(QIcon(":cancelButton.svg"))

        self.buttonsLayout.insertWidget(0, self.cancelButton)
        self.buttonsLayout.insertSpacing(1, 20)
        if connect:
            self.cancelButton.clicked.connect(self.buttonCallBack)

    def splitLatexBlocks(self, text):
        texts = re.split(r"(<latex.*?>.+?)</latex>", text)
        return texts

    def splitCopiableBlocks(self, texts: Sequence[str] | str):
        if isinstance(texts, str):
            texts = (texts,)

        texts_out = []
        for text in texts:
            texts_out.extend(re.split(r"(<copiable>.+?)</copiable>", text))
        return texts_out

    def addText(self, text):
        texts = self.splitLatexBlocks(text)
        texts = self.splitCopiableBlocks(texts)

        labelsWidget = LabelsWidget(texts, wrapText=self.wrapText)
        self.labelsWidgets.append(labelsWidget)
        self.labels.extend(labelsWidget.labels)
        if self.scrollableText:
            textWidget = QScrollArea()
            textWidget.setFrameStyle(QFrame.Shape.NoFrame)
            textWidget.setWidget(labelsWidget)
        else:
            textWidget = labelsWidget

        self.textLayout.addWidget(textWidget)

        if self.textWidget is None:
            self.textWidget = QWidget()
            self.textWidget.setLayout(self.textLayout)
            self._layout.addWidget(self.textWidget, self.currentRow, 1)
            self.textRow = self.currentRow
            self.currentRow += 1

        return labelsWidget

    def addCopiableCommand(self, command):
        copiableCommandWidget = CopiableCommandWidget(command)
        screenWidth = self.screen().size().width()
        maxWidth = int(0.75 * screenWidth)
        sizeHint = copiableCommandWidget.sizeHint()
        width = sizeHint.width()
        if width > maxWidth:
            copiableCommandWidget = addWidgetToScrollArea(
                copiableCommandWidget, resizeMinHeightNoVerticalScrollbar=True
            )
        self._layout.addWidget(copiableCommandWidget, self.currentRow, 1)
        self.currentRow += 1

    def copyToClipboard(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(self.sender()._command, mode=cb.Clipboard)
        print("Command copied!")

    def addButton(self, buttonText):
        if not isinstance(buttonText, str):
            # Passing button directly
            button = buttonText
            self.buttonsLayout.addWidget(button)
            button.clicked.connect(self.buttonCallBack)
            self.buttons.append(button)
            return button

        button, isCancelButton = getPushButton(buttonText, qparent=self)
        if not isCancelButton:
            self.buttonsLayout.addWidget(button)

        button.clicked.connect(self.buttonCallBack)
        self.buttons.append(button)
        return button

    def addDoNotShowAgainCheckbox(self, text="Do not show again"):
        self.doNotShowAgainCheckbox = QCheckBox(text)

    def addWidget(self, widget):
        self._layout.addWidget(widget, self.currentRow, 1)
        self.widgets.append(widget)
        self.currentRow += 1

    def addLayout(self, layout):
        self._layout.addLayout(layout, self.currentRow, 1)
        self.layouts.append(layout)
        self.currentRow += 1

    def setWidth(self, w):
        self._w = w

    def show(self, block=False):
        self.endOfScrollableRow = self.currentRow

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        # spacer
        spacer = QSpacerItem(10, 10)
        self._layout.addItem(spacer, self.currentRow, 1)
        self._layout.setRowStretch(self.currentRow, 0)

        # buttons
        self.currentRow += 1

        if self.detailsTextWidget is not None:
            self.buttonsLayout.insertWidget(1, self.detailsButton)

        # Do not show again checkbox
        if self.doNotShowAgainCheckbox is not None:
            self._layout.addWidget(
                self.doNotShowAgainCheckbox, self.currentRow, 1, 1, 2
            )
            self.currentRow += 1

        # spacer
        self._layout.addItem(QSpacerItem(10, 10), self.currentRow, 1)
        self.currentRow += 1

        # buttons
        self._layout.addLayout(
            self.buttonsLayout, self.currentRow, 0, 1, 2, alignment=Qt.AlignRight
        )

        # Details
        if self.detailsTextWidget is not None:
            # spacer
            self.currentRow += 1
            self._layout.addItem(QSpacerItem(20, 20), self.currentRow, 1)

            # detailsTextWidget
            self.currentRow += 1
            self._layout.addWidget(self.detailsTextWidget, self.currentRow, 0, 1, 2)

        # spacer
        self.currentRow += 1
        spacer = QSpacerItem(10, 10)
        self._layout.addItem(spacer, self.currentRow, 1)
        self._layout.setRowStretch(self.currentRow, 0)

        screenHeight = self.screen().size().height()
        dialogHeight = self.sizeHint().height()
        dialogWidth = self.sizeHint().width()
        screenWidth = self.screen().size().width()

        # Check if scrollbar is needed
        if dialogHeight > screenHeight and self.textWidget is not None:
            textScrollArea = ScrollArea()
            textScrollArea.setWidget(self.textWidget)
            scrollAreaWidthNoSB = textScrollArea.minimumWidthNoScrollbar()
            scrollAreaWidth = textScrollArea.sizeHint().width()
            desiredDeltaWidth = scrollAreaWidthNoSB - scrollAreaWidth
            if desiredDeltaWidth > 0:
                desiredWidth = dialogWidth + desiredDeltaWidth
                if desiredWidth < screenWidth:
                    self._w = desiredWidth

            self._layout.removeWidget(self.textWidget)
            self._layout.addWidget(textScrollArea, self.textRow, 1)

        super().show()
        QTimer.singleShot(5, self._resize)

        self.alreadyShown = True

        if block:
            self._block()

    def setDetailedText(self, text, visible=False, wrap=True):
        text = text.replace("\n", "<br>")
        self.detailsTextWidget = QTextEdit(text)
        self.detailsTextWidget.setReadOnly(True)
        if not wrap:
            self.detailsTextWidget.setLineWrapMode(QTextEdit.NoWrap)
        self.detailsButton = showDetailsButton()
        self.detailsButton.setCheckable(True)
        self.detailsButton.clicked.connect(self._showDetails)
        self.detailsTextWidget.hide()
        self.visibleDetails = visible

    def _showDetails(self, checked):
        if checked:
            self.origHeight = self.height()
            self.resize(self.width(), self.height() + 300)
            self.detailsTextWidget.show()
        else:
            self.detailsTextWidget.hide()
            func = partial(self.resize, self.width(), self.origHeight)
            QTimer.singleShot(10, func)

    def _resize(self):
        if self.resizeButtons:
            widths = [button.width() for button in self.buttons]
            if widths:
                max_width = max(widths)
                for button in self.buttons:
                    if button == self.cancelButton:
                        continue
                    button.setMinimumWidth(max_width)

        heights = [button.height() for button in self.buttons]
        if heights:
            max_h = max(heights)
            for button in self.buttons:
                button.setMinimumHeight(max_h)
            if self.detailsTextWidget is not None:
                self.detailsButton.setMinimumHeight(max_h)
            if self.showInFileManagButton is not None:
                self.showInFileManagButton.setMinimumHeight(max_h)

        if self._w is not None and self.width() < self._w:
            self.resize(self._w, self.height())

        if self.width() < 350:
            self.resize(350, self.height())

        if self.enlargeWidthFactor > 0:
            self.resize(int(self.width() * self.enlargeWidthFactor), self.height())

        if self.visibleDetails:
            self.detailsButton.click()

        if self.showCentered:
            screen = self.screen()
            screenWidth = screen.size().width()
            screenHeight = screen.size().height()
            screenLeft = screen.geometry().x()
            screenTop = screen.geometry().y()
            w, h = self.width(), self.height()
            left = int(screenLeft + screenWidth / 2 - w / 2)
            top = int(screenTop + screenHeight / 2 - h / 2)
            if top < screenTop:
                top = screenTop
            if left < screenLeft:
                left = screenLeft
            self.move(left, top)

        self._h = self.height()

        if self.okButton is not None:
            self.okButton.setFocus()

        screen = self.screen()
        screenWidth = screen.size().width()
        screenHeight = screen.size().height()

        # Check Force wrap Text
        for labelWidget in self.labelsWidgets:
            textWidth = labelWidget.width()
            if not textWidth > screenWidth - 10:
                continue
            factor = np.ceil(textWidth / screenWidth)
            lineLength = int(labelWidget.nCharsLongestLine / factor)
            for label in labelWidget.labels:
                if isinstance(label, CopiableCommandWidget):
                    continue

                text = label.text()
                chunks = textwrap.wrap(text, lineLength)
                text = "<br>".join(chunks)
                label.setText(text)

            QTimer.singleShot(100, self._resizeWrappedText)

        if self.widgets:
            return

        if self.layouts:
            return

        # # Start resizing height every 1 ms
        # self.resizeCallsCount = 0
        # self.timer = QTimer()
        # from config import warningHandler
        # warningHandler.sigGeometryWarning.connect(self.timer.stop)
        # self.timer.timeout.connect(self._resizeHeight)
        # self.timer.start(1)

    def _resizeWrappedText(self):
        screenWidth = self.screen().size().width() - 5
        self.resize(screenWidth, self.height())
        screenLeft = self.screen().geometry().left()
        self.move(screenLeft, self.geometry().top())

    def _resizeHeight(self):
        try:
            # Resize until a "Unable to set geometry" warning is captured
            # by copnfig.warningHandler._resizeWarningHandler or #
            # height doesn't change anymore
            self.resize(self.width(), self.height() - 1)
            if self.height() == self._h or self.resizeCallsCount > 100:
                self.timer.stop()
                return

            self.resizeCallsCount += 1
            self._h = self.height()
        except Exception as e:
            # traceback.format_exc()
            self.timer.stop()

    def _template(
        self,
        parent,
        title,
        message,
        detailsText=None,
        buttonsTexts=None,
        layouts=None,
        widgets=None,
        commands=None,
        path_to_browse=None,
        browse_button_text=None,
        url_to_open=None,
        open_url_button_text="Open url",
        image_paths=None,
        wrapDetails=True,
        add_do_not_show_again_checkbox=False,
    ):
        if parent is not None:
            self.setParent(parent)
        self.setWindowTitle(title)
        self.addText(message)
        if commands is not None:
            if isinstance(commands, str):
                commands = (commands,)
            for command in commands:
                self.addCopiableCommand(command)

        if image_paths is not None:
            if isinstance(image_paths, str):
                image_paths = (image_paths,)
            for image_path in image_paths:
                self.addImage(image_path)

        if layouts is not None:
            if myutils.is_iterable(layouts):
                for layout in layouts:
                    self.addLayout(layout)
            else:
                self.addLayout(layout)

        if widgets is not None:
            self._layout.addItem(QSpacerItem(20, 20), self.currentRow, 1)
            self.currentRow += 1
            if myutils.is_iterable(widgets):
                for widget in widgets:
                    self.addWidget(widget)
            else:
                self.addWidget(widgets)

        if path_to_browse is not None:
            self.addShowInFileManagerButton(path_to_browse, txt=browse_button_text)

        if url_to_open is not None:
            self.addBrowseUrlButton(url_to_open, button_text=open_url_button_text)

        buttons = []
        if buttonsTexts is None:
            okButton = self.addButton("  Ok  ")
            buttons.append(okButton)
        elif isinstance(buttonsTexts, str):
            button = self.addButton(buttonsTexts)
            buttons.append(button)
        else:
            for buttonText in buttonsTexts:
                button = self.addButton(buttonText)
                buttons.append(button)

        if detailsText is not None:
            self.setDetailedText(detailsText, visible=True, wrap=wrapDetails)

        if add_do_not_show_again_checkbox:
            self.addDoNotShowAgainCheckbox()

        return buttons

    def critical(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName="SP_MessageBoxCritical")
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def information(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName="SP_MessageBoxInformation")
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def warning(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName="SP_MessageBoxWarning")
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def question(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName="SP_MessageBoxQuestion")
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def _block(self):
        self.loop = QEventLoop()
        self.loop.exec_()

    def exec_(self):
        self.show(block=True)

    def clickButtonFromText(self, buttonText):
        for button in self.buttons:
            if button.text() == buttonText:
                button.click()
                return

    def buttonCallBack(self, checked=True):
        self.clickedButton = self.sender()
        if self.clickedButton != self.cancelButton:
            self.cancel = False
        self.allowClose = True
        self.close()

    def closeEvent(self, event):
        if not self.allowClose:
            event.ignore()
            return
        super().closeEvent(event)


def macShortcutToWindows(shortcut: str):
    if shortcut is None:
        return

    s = (
        shortcut.replace("Control", "Meta")
        .replace("Option", "Alt")
        .replace("Command", "Ctrl")
    )
    return s


def windowsShortcutToMac(shortcut: str):
    if shortcut is None:
        return

    if not is_mac:
        return shortcut

    s = (
        shortcut.replace("Meta", "Control")
        .replace("Alt", "Option")
        .replace("Ctrl", "Command")
    )
    return s


class ManualTrackingToolBar(ToolBar):
    sigIDchanged = Signal(int)
    sigDisableGhost = Signal()
    sigClearGhostContour = Signal()
    sigClearGhostMask = Signal()
    sigGhostOpacityChanged = Signal(int)

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.spinboxID = self.addSpinBox(label="ID to track: ")
        self.spinboxID.setMinimum(1)

        self.addSeparator()

        self.showGhostCheckbox = QCheckBox("Show ghost object")
        self.showGhostCheckbox.setChecked(True)
        self.addWidget(self.showGhostCheckbox)

        self.ghostContourRadiobutton = QRadioButton("Contour")
        self.ghostMaskRadiobutton = QRadioButton("Mask ; ")
        self.ghostMaskRadiobutton.setChecked(True)
        self.addWidget(self.ghostContourRadiobutton)
        self.addWidget(self.ghostMaskRadiobutton)

        self.ghostMaskOpacitySpinbox = self.addSpinBox("Mask opacity:  ")
        self.ghostMaskOpacitySpinbox.setMaximum(100)
        self.ghostMaskOpacitySpinbox.setValue(30)

        self.showGhostCheckbox.toggled.connect(self.showGhostCheckboxToggled)
        self.ghostContourRadiobutton.toggled.connect(
            self.ghostContourRadiobuttonToggled
        )
        self.spinboxID.valueChanged.connect(self.IDchanged)

        self.ghostMaskOpacitySpinbox.valueChanged.connect(self.ghostOpacityValueChanged)

        self.addSeparator()

        self.infoLabel = QLabel("")
        self.addWidget(self.infoLabel)

    def showInfo(self, text):
        text = html_utils.paragraph(text, font_color="black")
        self.infoLabel.setText(text)

    def showWarning(self, text):
        text = html_utils.paragraph(f"WARNING: {text}", font_color="red")
        self.infoLabel.setText(text)

    def clearInfoText(self):
        self.infoLabel.setText("")

    def IDchanged(self, value):
        self.sigIDchanged.emit(value)

    def showGhostCheckboxToggled(self, checked):
        disabled = not checked
        self.ghostContourRadiobutton.setDisabled(disabled)
        self.ghostMaskRadiobutton.setDisabled(disabled)
        self.ghostMaskOpacitySpinbox.setDisabled(disabled)
        self.ghostMaskOpacitySpinbox.label.setDisabled(disabled)
        if disabled:
            self.sigDisableGhost.emit()

    def ghostContourRadiobuttonToggled(self, checked):
        self.ghostMaskOpacitySpinbox.setDisabled(checked)
        self.ghostMaskOpacitySpinbox.label.setDisabled(checked)
        if checked:
            self.sigClearGhostMask.emit()
        else:
            self.sigClearGhostContour.emit()

    def ghostOpacityValueChanged(self, value):
        self.sigGhostOpacityChanged.emit(value)


class ManualBackgroundToolBar(ToolBar):
    sigIDchanged = Signal(int)

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.spinboxID = self.addSpinBox(label="Set background of ID ")
        self.spinboxID.setMinimum(1)
        self.spinboxID.valueChanged.connect(self.IDchanged)

        self.infoLabel = QLabel("")
        self.addWidget(self.infoLabel)

    def IDchanged(self, value):
        self.sigIDchanged.emit(value)

    def showWarning(self, text):
        text = html_utils.paragraph(f"WARNING: {text}", font_color="red")
        self.infoLabel.setText(text)

    def clearInfoText(self):
        self.infoLabel.setText("")


class SavePointsLayerButton(rightClickToolButton):
    sigRenameTableAction = Signal(object, str)

    def __init__(self, table_endname, parent=None):
        super().__init__(parent=parent)
        self.setIcon(QIcon(":file-save.svg"))

        self.table_endname = table_endname

        self.setToolTip(
            "Save annotated points in the CSV file ending "
            f"with '{self.table_endname}.csv'"
        )

        self.sigRightClick.connect(self.showContextMenu)

    def showContextMenu(self, event):
        contextMenu = QMenu(self)
        contextMenu.addSeparator()

        renameAction = QAction("Rename points layer table")
        renameAction.triggered.connect(self.renameTable)
        contextMenu.addAction(renameAction)

        contextMenu.exec(event.globalPos())

    def renameTable(self):
        win = apps.filenameDialog(
            parent=self,
            title="Rename points layer table file",
            allowEmpty=False,
            defaultEntry=self.table_endname,
            ext=".csv",
        )
        win.exec_()
        if win.cancel:
            return

        self.table_endname = win.entryText
        self.setToolTip(
            "Save annotated points in the CSV file ending "
            f"with '{self.table_endname}.csv'"
        )
        self.sigRenameTableAction.emit(self, self.table_endname)


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


def QKeyEventToString(event: QKeyEvent, notAllowedModifier=None):
    isAltKey = event.key() == Qt.Key_Alt
    isCtrlKey = event.key() == Qt.Key_Control
    isShiftKey = event.key() == Qt.Key_Shift
    isModifierKey = isAltKey or isCtrlKey or isShiftKey

    modifiers = event.modifiers()
    isNotAllowedMod = notAllowedModifier is not None and modifiers == notAllowedModifier
    if isNotAllowedMod:
        return

    modifers_value = modifiers.value if PYQT6 else modifiers
    if isModifierKey:
        keySequenceText = KeySequenceFromText(modifers_value).toString()
    else:
        keySequenceText = QKeySequence(modifers_value | event.key()).toString()

    keySequenceText = keySequenceText.encode("ascii", "ignore").decode("utf-8")

    return keySequenceText


class ShortcutLineEdit(QLineEdit):
    def __init__(self, parent=None, allowModifiers=False, notAllowedModifier=None):
        self.keySequence = None
        super().__init__(parent)
        self._allowModifiers = allowModifiers
        self._notAllowedModifier = notAllowedModifier
        self.setAlignment(Qt.AlignCenter)

    def text(self):
        text = macShortcutToWindows(super().text())

        return text

    def setText(self, text):
        text = windowsShortcutToMac(text)

        super().setText(text)
        if not text:
            self.keySequence = None
            return
        try:
            self.keySequence = KeySequenceFromText(self.text())
        except Exception as e:
            pass

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
            self.setText("")
            return

        keySequenceText = QKeyEventToString(
            event, notAllowedModifier=self._notAllowedModifier
        )
        self.setText(keySequenceText)
        self.key = event.key()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if self.text().endswith("+"):
            if not self._allowModifiers:
                self.setText("")
            else:
                self.setText(self.text().rstrip("+").strip())


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


class CenteredDoubleSpinbox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31 - 1)


class readOnlyDoubleSpinbox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31 - 1)


class readOnlySpinbox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31 - 1)


class DoubleSpinBox(QDoubleSpinBox):
    sigValueChanged = Signal(int)

    def __init__(self, parent=None, disableKeyPress=False):
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31 - 1)
        self.setMinimum(-(2**31))
        self._valueChangedFunction = None
        self.disableKeyPress = disableKeyPress

    def keyPressEvent(self, event) -> None:
        isBackSpaceKey = event.key() == Qt.Key_Backspace
        isDeleteKey = event.key() == Qt.Key_Delete
        try:
            int(event.text())
            isIntegerKey = True
        except:
            isIntegerKey = False
        acceptEvent = isBackSpaceKey or isDeleteKey or isIntegerKey
        if self.disableKeyPress and not acceptEvent:
            event.ignore()
            self.clearFocus()
        else:
            super().keyPressEvent(event)

    def textFromValue(self, value: float) -> str:
        text = super().textFromValue(value)
        return text.replace(QLocale().decimalPoint(), ".")

    def valueFromText(self, text: str) -> float:
        text = text.replace(".", QLocale().decimalPoint())
        return super().valueFromText(text)


class SpinBox(QSpinBox):
    sigValueChanged = Signal(int)
    sigUpClicked = Signal()
    sigDownClicked = Signal()

    def __init__(self, parent=None, disableKeyPress=False, allowNegative=True):
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31 - 1)
        if allowNegative:
            self.setMinimum(-(2**31))
        else:
            self.setMinimum(0)
        self._valueChangedFunction = None
        self.disableKeyPress = disableKeyPress
        self._linkedWidget = None

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        opt = QStyleOptionSpinBox()
        self.initStyleOption(opt)

        control = self.style().hitTestComplexControl(
            QStyle.ComplexControl.CC_SpinBox, opt, event.pos(), self
        )
        if control == QStyle.SubControl.SC_SpinBoxUp:
            self.sigUpClicked.emit()
        elif control == QStyle.SubControl.SC_SpinBoxDown:
            self.sigDownClicked.emit()

    # def focusOutEvent(self, event):
    #     self.editingFinished.emit()
    #     super().focusOutEvent(event)
    #     printl('emitted')

    def keyPressEvent(self, event) -> None:
        isBackSpaceKey = event.key() == Qt.Key_Backspace
        isDeleteKey = event.key() == Qt.Key_Delete
        try:
            int(event.text())
            isIntegerKey = True
        except:
            isIntegerKey = False
        acceptEvent = isBackSpaceKey or isDeleteKey or isIntegerKey
        if self.disableKeyPress and not acceptEvent:
            event.ignore()
            self.clearFocus()
        else:
            super().keyPressEvent(event)

    def connectValueChanged(self, function):
        self._valueChangedFunction = function
        self.valueChanged.connect(function)

    def setValue(self, value, setLinkedWidget=True):
        super().setValue(int(value))
        if self._linkedWidget is not None and setLinkedWidget:
            self._linkedWidget.setValue(value)

    def setValueNoEmit(self, value):
        if self._valueChangedFunction is None:
            self.setValue(value)
            return
        try:
            self.valueChanged.disconnect()
        except TypeError as e:  # this fails if its not cennected yet
            pass

        self.setValue(value)
        self.valueChanged.connect(self._valueChangedFunction)

    def wheelEvent(self, event):
        event.ignore()

    def setLinkedValueWidget(self, widget):
        self._linkedWidget = widget


class ReadOnlyLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        # self.setStyleSheet(
        #     'background-color: rgba(240, 240, 240, 200);'
        # )
        self.installEventFilter(self)

    def eventFilter(self, a0: "QObject", a1: "QEvent") -> bool:
        if a1.type() == QEvent.Type.FocusIn:
            return True
        return super().eventFilter(a0, a1)

    def setValue(self, value):
        self.setText(str(value))

    def value(self, casting_func: callable = None):
        text = self.text()
        if casting_func is not None:
            return casting_func(text)
        return text


class FloatLineEdit(QLineEdit):
    valueChanged = Signal(float)

    def __init__(
        self,
        *args,
        notAllowed=None,
        allowNegative=True,
        initial=None,
        readOnly=False,
        decimals=6,
        warningValues=None,
    ):
        QLineEdit.__init__(self, *args)
        if readOnly:
            self.setReadOnly(readOnly)
        self.notAllowed = notAllowed
        self.warningValues = warningValues
        self._maximum = np.inf
        self._minimum = -np.inf
        self._decimals = decimals

        self.isNumericRegExp = rf"^{float_regex(allow_negative=allowNegative)}$"
        regExp = QRegularExpression(self.isNumericRegExp)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)

        self.textChanged.connect(self.emitValueChanged)

        if initial is not None:
            self.setValue(initial)
        else:
            self.setValue(0)

    def setDecimals(self, decimals):
        self._decimals = 6

    def castMinMax(self, value: int):
        if value > self._maximum:
            value = self._maximum
        if value < self._minimum:
            value = self._minimum
        return value

    def setValue(self, value: float):
        value = self.castMinMax(value)
        self.setText(str(round(value, self._decimals)))

    def value(self):
        m = re.match(self.isNumericRegExp, self.text())
        if m is not None:
            text = m.group(0)
            try:
                val = float(text)
            except ValueError:
                val = 0.0
        else:
            val = 0.0

        return self.castMinMax(val)

    def setMaximum(self, maximum):
        self._maximum = maximum
        self.setValue(self.value())

    def setMinimum(self, minimum):
        self._minimum = minimum
        self.setValue(self.value())

    def emitValueChanged(self, text):
        val = self.value()
        reset_stylesheet = True
        if self.warningValues is not None and val in self.warningValues:
            self.setStyleSheet(LINEEDIT_WARNING_STYLESHEET)
            reset_stylesheet = False

        if self.notAllowed is not None and val in self.notAllowed:
            self.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)
            reset_stylesheet = False
        else:
            self.valueChanged.emit(self.value())

        if reset_stylesheet:
            self.setStyleSheet("")


class IntLineEdit(QLineEdit):
    valueChanged = Signal(float)

    def __init__(
        self, *args, notAllowed=None, allowNegative=True, initial=None, readOnly=False
    ):
        QLineEdit.__init__(self, *args)
        self.notAllowed = notAllowed
        if readOnly:
            self.setReadOnly(readOnly)

        self._maximum = np.inf
        self._minimum = -np.inf

        self._regExp = r"\d+"
        if allowNegative:
            self._regExp = r"-?\d+"

        regExp = QRegularExpression(self._regExp)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)

        self.textChanged.connect(self.emitValueChanged)

        if initial is not None:
            self.setValue(initial)
        else:
            self.setValue(0)

    def setMaximum(self, maximum):
        self._maximum = maximum
        self.setValue(self.value())

    def setMinimum(self, minimum):
        self._minimum = minimum
        self.setValue(self.value())

    def castMinMax(self, value: int):
        if value > self._maximum:
            value = self._maximum
        if value < self._minimum:
            value = self._minimum
        return value

    def setValue(self, value: int):
        value = self.castMinMax(value)
        self.setText(str(value))

    def value(self):
        m = re.match(self._regExp, self.text())
        if m is not None:
            text = m.group(0)
            try:
                val = int(text)
            except ValueError:
                val = 0
        else:
            val = 0

        return self.castMinMax(val)

    def emitValueChanged(self, text):
        if not text:
            return

        val = self.value()
        self.setValue(val)
        if self.notAllowed is not None and val in self.notAllowed:
            self.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)
        else:
            self.setStyleSheet("")
            self.valueChanged.emit(self.value())


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


class _metricsQGBox(QGroupBox):
    sigDelClicked = Signal(str, object)

    def __init__(
        self,
        desc_dict,
        title,
        favourite_funcs=None,
        isZstack=False,
        equations=None,
        addDelButton=False,
        delButtonMetricsDesc=None,
        parent=None,
        addCalcForEachZsliceToggle=False,
    ):
        QGroupBox.__init__(self, parent)

        highlightRgba = _palettes._highlight_rgba()
        r, g, b, a = highlightRgba
        self._highlightStylesheetColor = f"rgb({r}, {g}, {b})"

        self._parent = parent
        self.scrollArea = QScrollArea()
        self.scrollAreaWidget = QWidget()
        self.favourite_funcs = favourite_funcs

        self.doNotWarn = False

        layout = QVBoxLayout()
        inner_layout = QVBoxLayout()
        self.inner_layout = inner_layout
        if delButtonMetricsDesc is None:
            delButtonMetricsDesc = []

        self.checkBoxes = []
        self.checkedState = {}
        for metric_colname, metric_desc in desc_dict.items():
            rowLayout = QHBoxLayout()

            checkBox = QCheckBox(metric_colname)
            checkBox.setChecked(True)
            checkBox.scrollArea = self.scrollArea
            self.checkBoxes.append(checkBox)
            self.checkedState[checkBox] = True

            try:
                checkBox.equation = equations[metric_colname]
            except Exception as e:
                pass

            if addDelButton or metric_colname in delButtonMetricsDesc:
                delButton = delPushButton()
                delButton.setToolTip("Delete custom combined measurement")
                delButton.colname = metric_colname
                delButton.checkbox = checkBox
                delButton.clicked.connect(self.onDelClicked)
                delButton._layout = rowLayout
                rowLayout.addWidget(delButton)

            infoButton = infoPushButton()
            infoButton.setCursor(Qt.WhatsThisCursor)
            infoButton.info = metric_desc
            infoButton.colname = metric_colname
            infoButton.clicked.connect(self.showInfo)

            rowLayout.addWidget(infoButton)
            rowLayout.addWidget(checkBox)
            rowLayout.addStretch(1)

            inner_layout.addLayout(rowLayout)

        self.scrollAreaWidget.setLayout(inner_layout)
        self.scrollArea.setWidget(self.scrollAreaWidget)
        layout.addWidget(self.scrollArea)

        buttonsLayout = QHBoxLayout()

        buttonsLayout.addStretch(1)

        self.selectAllButton = selectAllPushButton()
        self.selectAllButton.sigClicked.connect(self.checkAll)

        buttonsLayout.addWidget(self.selectAllButton)

        if favourite_funcs is not None:
            self.loadFavouritesButton = reloadPushButton("  Load last selection...  ")
            self.loadFavouritesButton.clicked.connect(self.checkFavouriteFuncs)
            # self.checkFavouriteFuncs()
            buttonsLayout.addWidget(self.loadFavouritesButton)

        layout.addLayout(buttonsLayout)

        self.calcForEachZsliceToggle = None
        if addCalcForEachZsliceToggle:
            buttonsLayout = QHBoxLayout()
            self.calcForEachZsliceToggle = Toggle()
            tooltip = (
                "Calculate `cell_area` for each z-slice.\n\n"
                "The measurements will be saved in the column with name\n"
                "ending with `_zsliceN` where N is the z-slice number\n"
                "(starting from 0)."
            )
            calcForEachZsliceLabel = QClickableLabel("Calculate for each z-slice")
            calcForEachZsliceLabel.setToolTip(tooltip)
            self.calcForEachZsliceToggle.setToolTip(tooltip)
            buttonsLayout.addWidget(self.calcForEachZsliceToggle)
            buttonsLayout.addWidget(calcForEachZsliceLabel)
            buttonsLayout.addStretch(1)
            layout.addLayout(buttonsLayout)
            calcForEachZsliceLabel.clicked.connect(
                partial(
                    self.toggleCalcForEachZslice, toggle=self.calcForEachZsliceToggle
                )
            )

        self.setTitle(title)
        self.setCheckable(True)
        self.setLayout(layout)
        _font = QFont()
        _font.setPixelSize(11)
        self.setFont(_font)

        self.toggled.connect(self.toggled_cb)

    def toggleCalcForEachZslice(self, label, toggle=None):
        if toggle is None:
            toggle = self.calcForEachZsliceToggle

        toggle.setChecked(not toggle.isChecked())

    def isCalcForEachZsliceRequested(self):
        if self.calcForEachZsliceToggle is None:
            return False

        return self.calcForEachZsliceToggle.isChecked()

    def highlightCheckboxesFromSearchText(self, text):
        for checkbox in self.checkBoxes:
            if not text:
                highlighted = False
            else:
                highlighted = checkbox.text().lower().find(text.lower()) != -1

            self.setCheckboxHighlighted(highlighted, checkbox)

    def setCheckboxHighlighted(self, highlighted, checkbox):
        if highlighted:
            checkbox.setStyleSheet(
                f"background: {self._highlightStylesheetColor}; color: black"
            )
            self.scrollArea.ensureWidgetVisible(checkbox)
        else:
            checkbox.setStyleSheet("")

    def onDelClicked(self):
        button = self.sender()
        button.checkbox.setChecked(False)
        self.sigDelClicked.emit(button.colname, button._layout)

    def toggled_cb(self, checked):
        for checkbox in self.checkBoxes:
            if not checked:
                self.checkedState[checkbox] = checkbox.isChecked()
                checkbox.setChecked(False)
            else:
                checkbox.setChecked(self.checkedState[checkbox])

    def checkFavouriteFuncs(self, checked=True, isZstack=False):
        self.doNotWarn = True
        if self._parent is not None:
            self._parent.doNotWarn = True
        for checkBox in self.checkBoxes:
            checkBox.setChecked(False)
            for favourite_func in self.favourite_funcs:
                func_name = checkBox.text()
                if func_name.endswith(favourite_func):
                    checkBox.setChecked(True)
                    break
        self.doNotWarn = False
        if self._parent is not None:
            self._parent.doNotWarn = False

    def checkAll(self, button, checked):
        if self._parent is not None:
            self._parent.doNotWarn = True
        for checkBox in self.checkBoxes:
            checkBox.setChecked(checked)
        if self._parent is not None:
            self._parent.doNotWarn = False

    def showInfo(self, checked=False):
        info_txt = self.sender().info
        msg = myMessageBox()
        msg.setWidth(600)
        msg.setIcon()
        msg.setWindowTitle(f"{self.sender().colname} info")
        msg.addText(info_txt)
        msg.addButton("   Ok   ")
        msg.exec_()

    def show(self):
        super().show()
        fw = self.inner_layout.contentsRect().width()
        sw = self.scrollArea.verticalScrollBar().sizeHint().width()
        self.minWidth = fw + sw


class channelMetricsQGBox(QGroupBox):
    sigDelClicked = Signal(str, object)
    sigCheckboxToggled = Signal(object)

    def __init__(
        self,
        isZstack,
        chName,
        isSegm3D,
        is_concat=False,
        posData=None,
        favourite_funcs=None,
    ):
        QGroupBox.__init__(self)

        self.doNotWarn = False
        self.is_concat = is_concat
        isManualBackgrPresent = False
        if posData is not None:
            if posData.manualBackgroundLab is not None:
                isManualBackgrPresent = True

        layout = QVBoxLayout()
        metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
            isZstack,
            chName,
            isSegm3D=isSegm3D,
            isManualBackgrPresent=isManualBackgrPresent,
        )

        metricsQGBox = _metricsQGBox(
            metrics_desc,
            "Standard measurements",
            favourite_funcs=favourite_funcs,
            parent=self,
            isZstack=isZstack,
        )
        self.metricsQGBox = metricsQGBox

        bkgrValsQGBox = _metricsQGBox(
            bkgr_val_desc,
            "Background values",
            favourite_funcs=favourite_funcs,
            parent=self,
            isZstack=isZstack,
        )
        self.bkgrValsQGBox = bkgrValsQGBox

        self.checkBoxes = metricsQGBox.checkBoxes.copy()
        self.checkBoxes.extend(bkgrValsQGBox.checkBoxes)

        self.uncheckAndDisableDataPrepIfPosNotPrepped(posData)

        self.groupboxes = [metricsQGBox, bkgrValsQGBox]

        for checkbox in metricsQGBox.checkBoxes:
            checkbox.toggled.connect(self.standardMetricToggled)
            self.standardMetricToggled(checkbox.isChecked(), checkbox=checkbox)

        for bkgrCheckbox in bkgrValsQGBox.checkBoxes:
            bkgrCheckbox.toggled.connect(self.backgroundMetricToggled)

        layout.addWidget(metricsQGBox)
        layout.addWidget(bkgrValsQGBox)

        items = measurements.custom_metrics_desc(
            isZstack, chName, posData=posData, isSegm3D=isSegm3D, return_combine=True
        )
        custom_metrics_desc, combine_metrics_desc = items

        if custom_metrics_desc:
            customMetricsQGBox = _metricsQGBox(
                custom_metrics_desc,
                "Custom measurements",
                delButtonMetricsDesc=combine_metrics_desc,
                favourite_funcs=favourite_funcs,
                isZstack=isZstack,
            )
            layout.addWidget(customMetricsQGBox)
            self.checkBoxes.extend(customMetricsQGBox.checkBoxes)
            customMetricsQGBox.sigDelClicked.connect(self.onDelClicked)
            self.customMetricsQGBox = customMetricsQGBox

        self.calcForEachZsliceToggle = None
        if isZstack:
            buttonsLayout = QHBoxLayout()
            self.calcForEachZsliceToggle = Toggle()
            tooltip = (
                "Calculate the selected measurements for each z-slice.\n\n"
                "The measurements will be saved in the column with name\n"
                "ending with `_zsliceN` where N is the z-slice number\n"
                "(starting from 0)."
            )
            calcForEachZsliceLabel = QClickableLabel("Calculate for each z-slice")
            calcForEachZsliceLabel.setToolTip(tooltip)
            self.calcForEachZsliceToggle.setToolTip(tooltip)
            buttonsLayout.addWidget(self.calcForEachZsliceToggle)
            buttonsLayout.addWidget(calcForEachZsliceLabel)
            buttonsLayout.addStretch(1)
            layout.addLayout(buttonsLayout)
            calcForEachZsliceLabel.clicked.connect(
                partial(
                    self.toggleCalcForEachZslice, toggle=self.calcForEachZsliceToggle
                )
            )

        self.setTitle(f"{chName} metrics")
        self.setCheckable(True)
        self.setLayout(layout)

    def toggleCalcForEachZslice(self, label, toggle=None):
        if toggle is None:
            toggle = self.calcForEachZsliceToggle

        toggle.setChecked(not toggle.isChecked())

    def isCalcForEachZsliceRequested(self):
        if self.calcForEachZsliceToggle is None:
            return False

        return self.calcForEachZsliceToggle.isChecked()

    def uncheckAndDisableDataPrepIfPosNotPrepped(self, posData):
        # Uncheck and disable dataprep metrics if pos is not prepped
        if posData is None:
            return

        if posData.isBkgrROIpresent():
            return

        for checkbox in self.checkBoxes:
            if checkbox.text().find("dataPrep") == -1:
                continue

            checkbox.setChecked(False)
            checkbox.isDataPrepDisabled = True

    def _warnDataPrepCannotBeChecked(self):
        if self.doNotWarn:
            return
        txt = html_utils.paragraph("""
            <b>Data prep measurements cannot be saved</b> because you did 
            not select any background ROI at the data prep step.<br><br>

            You can read more details about data prep metrics by clicking 
            on the info button besides the measurement's name.<br><br>

            Thank you for you patience!
        """)
        msg = myMessageBox(showCentered=False)
        msg.warning(self, "Metric cannot be saved", txt)

    def standardMetricToggled(self, checked, checkbox=None):
        """Method called when a check-box is toggled. It performs the following
        actions:
            1. If the user try to check a data prep measurement, such as
            dataPrep_amount, and this cannot be saved (checkbox has the attr
            `isDataPrepDisabled`) then it warns and explains why it cannot be saved
            2. Make sure that background value median is checked if the user
            requires amount or concentration metric.
            3. Do not allow unchecking background value median and explain why.

        Parameters
        ----------
        checked : bool
            State of the checkbox toggled
        checkbox : QtWidgets.QCheckBox, optional
            The checkbox that has been toggled. Default is None. If None
            use `self.sender()`
        """
        if self.is_concat:
            return

        if checkbox is None:
            checkbox = self.sender()

        if hasattr(checkbox, "isDataPrepDisabled"):
            # Warn that user cannot check data prep metrics and uncheck it
            if not checkbox.isChecked():
                return
            checkbox.setChecked(False)
            self._warnDataPrepCannotBeChecked()
            return

        self.sigCheckboxToggled.emit(checkbox)
        if checkbox.text().find("amount_") == -1:
            return
        pattern = r"amount_([A-Za-z]+)(_?[A-Za-z0-9]*)"
        repl = r"\g<1>_bkgrVal_median\g<2>"
        bkgrValMetric = s1 = re.sub(pattern, repl, checkbox.text())
        for bkgrCheckbox in self.groupboxes[1].checkBoxes:
            if bkgrCheckbox.text() == bkgrValMetric:
                break
        else:
            # Make sure to not check for similarly named custom metrics
            return

        if checked:
            bkgrCheckbox.setChecked(True)
            bkgrCheckbox.isRequired = True
        else:
            bkgrCheckbox.setDisabled(False)
            bkgrCheckbox.isRequired = False

    def backgroundMetricToggled(self, checked):
        """Method called when a checkbox of a background metric is toggled.
        Check if the background value is required and explain why it cannot be
        unchecked.

        Parameters
        ----------
        checked : bool
            State of the checkbox toggled
        """
        if self.is_concat:
            return

        checkbox = self.sender()
        if not hasattr(checkbox, "isRequired"):
            return

        if not checkbox.isRequired:
            return

        if checkbox.isChecked():
            return

        if self.doNotWarn:
            return

        checkbox.setChecked(True)
        txt = html_utils.paragraph("""
            <b>This background value cannot be unchecked</b> because it is required 
            by the <code>_amount</code> and <code>_concentration</code> measurements 
            that you requested to save.<br><br>

            Thank you for you patience!
        """)
        msg = myMessageBox(showCentered=False)
        msg.warning(self, "Background value required", txt)

    def onDelClicked(self, colname_to_del, hlayout):
        self.sigDelClicked.emit(colname_to_del, hlayout)

    def checkFavouriteFuncs(self):
        self.doNotWarn = True
        for groupbox in self.groupboxes:
            groupbox.checkFavouriteFuncs()
        self.doNotWarn = False


class PixelSizeGroupbox(QGroupBox):
    sigValueChanged = Signal(float, float, float)
    sigReset = Signal()

    def __init__(self, parent=None):
        super().__init__("Pixel size", parent)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel("Pixel width (μm): ")
        self.pixelWidthWidget = FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.pixelWidthWidget, row, 1)

        row += 1
        label = QLabel("Pixel height (μm): ")
        self.pixelHeightWidget = FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.pixelHeightWidget, row, 1)

        row += 1
        label = QLabel("Voxel depth (μm): ")
        self.voxelDepthWidget = FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.voxelDepthWidget, row, 1)

        row += 1
        resetButton = reloadPushButton("Reset")
        mainLayout.addWidget(resetButton, row, 1, alignment=Qt.AlignRight)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        mainLayout.setColumnStretch(0, 0)
        mainLayout.setColumnStretch(1, 1)

        self.setLayout(mainLayout)

        self.pixelWidthWidget.valueChanged.connect(self.emitValueChanged)
        self.pixelHeightWidget.valueChanged.connect(self.emitValueChanged)
        self.voxelDepthWidget.valueChanged.connect(self.emitValueChanged)
        resetButton.clicked.connect(self.emitReset)

    def emitReset(self):
        self.sigReset.emit()

    def emitValueChanged(self, value):
        PhysicalSizeX = self.pixelWidthWidget.value()
        PhysicalSizeY = self.pixelHeightWidget.value()
        PhysicalSizeZ = self.voxelDepthWidget.value()
        self.sigValueChanged.emit(PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)


class objPropsQGBox(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, "Properties", parent)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel("Object ID: ")
        self.idSB = IntLineEdit()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.idSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        self.notExistingIDLabel = QLabel()
        self.notExistingIDLabel.setStyleSheet("font-size:11px; color: rgb(255, 0, 0);")
        mainLayout.addWidget(
            self.notExistingIDLabel, row, 0, 1, 2, alignment=Qt.AlignCenter
        )

        row += 1
        label = QLabel("Area (pixel): ")
        self.cellAreaPxlSB = IntLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellAreaPxlSB, row, 1)

        row += 1
        label = QLabel("Area (<span>&#181;</span>m<sup>2</sup>): ")
        self.cellAreaUm2DSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellAreaUm2DSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        label = QLabel("Rotational volume (voxel): ")
        self.cellVolVoxSB = IntLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolVoxSB, row, 1)

        row += 1
        label = QLabel("3D volume (voxel): ")
        self.cellVolVox3D_SB = IntLineEdit(readOnly=True)
        self.cellVolVox3D_SB.label = label
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolVox3D_SB, row, 1)

        row += 1
        label = QLabel("Rotational volume (fl): ")
        self.cellVolFlDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolFlDSB, row, 1)

        row += 1
        label = QLabel("3D volume (fl): ")
        self.cellVolFl3D_DSB = FloatLineEdit(readOnly=True)
        self.cellVolFl3D_DSB.label = label
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolFl3D_DSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        label = QLabel("Solidity: ")
        self.solidityDSB = FloatLineEdit(readOnly=True)
        self.solidityDSB.setMaximum(1)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.solidityDSB, row, 1)

        row += 1
        label = QLabel("Elongation: ")
        self.elongationDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.elongationDSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        propsNames = measurements.get_props_names()[1:]
        self.additionalPropsCombobox = QComboBox()
        self.additionalPropsCombobox.addItems(propsNames)
        self.additionalPropsCombobox.indicator = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(self.additionalPropsCombobox, row, 0)
        mainLayout.addWidget(self.additionalPropsCombobox.indicator, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        mainLayout.setColumnStretch(0, 0)
        mainLayout.setColumnStretch(1, 1)

        self.setLayout(mainLayout)


class objIntesityMeasurQGBox(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, "Intensity measurements", parent)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel("Raw intensity measurements")

        row += 1
        label = QLabel("Channel: ")
        self.channelCombobox = QComboBox()
        self.channelCombobox.addItem("placeholderlong")
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.channelCombobox, row, 1)

        row += 1
        label = QLabel("Minimum: ")
        self.minimumDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.minimumDSB, row, 1)

        row += 1
        label = QLabel("Maximum: ")
        self.maximumDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.maximumDSB, row, 1)

        row += 1
        label = QLabel("Mean: ")
        self.meanDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.meanDSB, row, 1)

        row += 1
        label = QLabel("Median: ")
        self.medianDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.medianDSB, row, 1)

        row += 1
        metricsDesc = measurements._get_metrics_names()
        metricsFunc, _ = measurements.standard_metrics_func()
        items = list(set([metricsDesc[key] for key in metricsFunc.keys()]))
        items.append("Concentration")
        items.sort()
        nameFuncDict = {}
        for name, desc in metricsDesc.items():
            if name.find("_dataPrepBkgr") != -1 or name.find("_manualBkgr") != -1:
                # Skip dataPrepBkgr and manualBkgr since in the dock widget
                # we display only autoBkgr metrics
                continue
            if name.startswith("concentration_"):
                # We use amount function because dividing by volume is taken
                # care in the GUI
                name = "amount_autoBkgr"
            nameFuncDict[desc] = metricsFunc[name]

        funcionCombobox = QComboBox()
        funcionCombobox.addItems(items)
        self.additionalMeasCombobox = funcionCombobox
        self.additionalMeasCombobox.indicator = FloatLineEdit(readOnly=True)
        self.additionalMeasCombobox.functions = nameFuncDict
        mainLayout.addWidget(funcionCombobox, row, 0)
        mainLayout.addWidget(self.additionalMeasCombobox.indicator, row, 1)

        self.setLayout(mainLayout)

    def addChannels(self, channels):
        self.channelCombobox.clear()
        self.channelCombobox.addItems(channels)


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


class view_visualcpp_screenshot(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()

        self.setWindowTitle("Visual Studio Builld Tools installation")

        pixmap = QPixmap(":visualcpp.png")
        label = QLabel()
        label.setPixmap(pixmap)

        layout.addWidget(label)
        self.setLayout(layout)


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


class highlightableQWidgetAction(QWidgetAction):
    def __init__(self, parent) -> None:
        super().__init__(parent)


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


def PostProcessSegmWidget(
    minimum, maximum, value, useSliders, isFloat=False, normalize=False, label=None
):
    if useSliders:
        if normalize:
            maximum = int(maximum * 100)
        widget = PostProcessSegmSlider(
            normalize=normalize, isFloat=isFloat, label=label
        )
    else:
        widget = PostProcessSegmSpinbox(label=label, isFloat=isFloat)
    widget.setMinimum(minimum)
    widget.setMaximum(maximum)
    widget.setValue(value)
    return widget


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


class ComboBox(QComboBox):
    sigTextChanged = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._previousText = None
        self._valueChanged = False
        self.currentTextChanged.connect(self.emitTextChanged)
        self.installEventFilter(self)

    def eventFilter(self, object, event) -> bool:
        if object == self and event.type() == QEvent.Type.Wheel:
            # Forward event to parent so QScrollArea can scroll
            QApplication.sendEvent(self.parent(), event)
            return True  # Consume for the combo itself

        return super().eventFilter(object, event)

    def text(self):
        return self.currentText()

    def emitTextChanged(self, text):
        self._valueChanged = True
        self.sigTextChanged.emit(text)

    def mousePressEvent(self, event):
        self._previousText = self.currentText()
        super().mousePressEvent(event)

    def previousText(self):
        return self._previousText

    def addItems(self, items):
        super().addItems(items)
        self._previousText = items[0]

    def itemsText(self):
        return [self.itemText(i) for i in range(self.count())]

    def setCurrentIndex(self, idx):
        itemsText = self.itemsText()
        currentText = itemsText[idx]
        self._valueChanged = currentText != self._previousText
        self._previousText = self.currentText()
        super().setCurrentIndex(idx)

    def setCurrentText(self, text):
        currentText = text
        self._valueChanged = currentText != self._previousText
        self._previousText = self.currentText()
        super().setCurrentText(text)


class SetMeasurementsGroupBox(QGroupBox):
    def __init__(
        self,
        title,
        itemsText,
        checkable=True,
        itemsInfo=None,
        lastSelection=None,
        itemsInfoUrls=None,
        parent=None,
    ):
        super().__init__(parent)

        if itemsInfo is None:
            itemsInfo = {}

        if itemsInfo is None:
            itemsInfoUrls = {}

        highlightRgba = _palettes._highlight_rgba()
        r, g, b, a = highlightRgba
        self._highlightStylesheetColor = f"rgb({r}, {g}, {b})"

        self.setTitle(title)
        self.setCheckable(checkable)

        mainLayout = QVBoxLayout()

        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollAreaLayout = QVBoxLayout()
        scrollAreaWidget = QWidget()
        self.scrollAreaWidget = scrollAreaWidget
        self.scrollAreaLayout = scrollAreaLayout

        self.checkboxes = {}
        for text in itemsText:
            rowLayout = QHBoxLayout()
            infoText = itemsInfo.get(text)
            infoUrl = itemsInfoUrls.get(text)
            if infoText is not None or infoUrl is not None:
                infoButton = infoPushButton()
                infoButton.setCursor(Qt.WhatsThisCursor)
                rowLayout.addWidget(infoButton)

            if infoText is not None:
                infoButton.itemText = text
                infoButton.infoText = infoText
                infoButton.clicked.connect(self.showInfo)

            if infoUrl is not None:
                infoButton.itemText = text
                infoButton.infoUrl = infoUrl
                infoButton.clicked.connect(self.openInfoUrl)

            checkbox = QCheckBox(text)
            checkbox.setParent(self.scrollAreaWidget)
            checkbox.setChecked(True)
            rowLayout.addWidget(checkbox)
            rowLayout.addStretch(1)

            self.checkboxes[text] = checkbox

            scrollAreaLayout.addLayout(rowLayout)

        scrollAreaLayout.addStretch(1)

        scrollAreaWidget.setLayout(scrollAreaLayout)
        scrollArea.setWidget(scrollAreaWidget)
        self.scrollArea = scrollArea

        buttonsLayout = QHBoxLayout()
        self.selectAllButton = selectAllPushButton()
        self.selectAllButton.sigClicked.connect(self.setCheckedAll)

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(self.selectAllButton)
        self.buttonsLayout = buttonsLayout

        if lastSelection is not None:
            self.lastSelection = lastSelection
            self.loadLastSelButton = reloadPushButton("  Load last selection...  ")
            self.loadLastSelButton.clicked.connect(self.loadLastSelection)
            buttonsLayout.addWidget(self.loadLastSelButton)

        mainLayout.addWidget(scrollArea)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def openInfoUrl(self):
        url = self.sender().infoUrl
        QDesktopServices.openUrl(QUrl(url))
        # import webbrowser
        # url = self.sender().infoUrl
        # webbrowser.open(url)

    def getWidthNoScrollBarNeeded(self):
        width = (
            self.scrollArea.verticalScrollBar().sizeHint().width()
            # self.scrollAreaLayout.contentsRect().width()
            + self.scrollAreaWidget.sizeHint().width()
            + 30
        )
        buttonsWidth = 0
        for i in range(self.buttonsLayout.count()):
            widget = self.buttonsLayout.itemAt(i).widget()
            if not isinstance(widget, QPushButton):
                continue
            buttonsWidth += widget.sizeHint().width() + 16
        largerWidth = max(width, buttonsWidth)
        return largerWidth

    def resizeWidthNoScrollBarNeeded(self):
        width = self.getWidthNoScrollBarNeeded()
        self.setMinimumWidth(width)
        # self.setFixedWidth(width)

    def loadLastSelection(self):
        for text, checkbox in self.checkboxes.items():
            checked = self.lastSelection.get(text, False)
            checkbox.setChecked(checked)

    def showInfo(self):
        infoText = self.sender().infoText
        itemText = self.sender().itemText

        title = f"{itemText} description"
        msg = myMessageBox()
        msg.setWidth(int(self.screen().size().width() / 2))
        msg.information(self, title, infoText)

    def setCheckedAll(self, button, checked):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(checked)

    def highlightCheckboxesFromSearchText(self, text):
        for checkbox in self.checkboxes.values():
            if not text:
                highlighted = False
            else:
                highlighted = checkbox.text().lower().find(text.lower()) != -1

            self.setCheckboxHighlighted(highlighted, checkbox)

    def setCheckboxHighlighted(self, highlighted, checkbox):
        if highlighted:
            checkbox.setStyleSheet(
                f"background: {self._highlightStylesheetColor}; color: black"
            )
            self.scrollArea.ensureWidgetVisible(checkbox)
        else:
            checkbox.setStyleSheet("")


class SearchLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.initSearch()
        self.setFocusPolicy(Qt.ClickFocus)

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        if super().text() == "Search...":
            self.setText("")
        self.setStyleSheet("")

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        if not super().text():
            self.initSearch()

    def initSearch(self):
        self.setText("Search...")
        self.setStyleSheet("color: rgb(150, 150, 150)")
        self.clearFocus()

    def text(self):
        if super().text() == "Search...":
            return ""
        return super().text()


class VectorLineEdit(QLineEdit):
    valueChanged = Signal(object)
    valueChangeFinished = Signal(object)

    def __init__(self, parent=None, initial=None):
        super().__init__(parent)

        self._minimum = -np.inf

        float_re = float_regex()
        vector_regex = rf"\(?\[?{float_re}(,\s?{float_re})+\)?\]?"
        regex = rf"^{vector_regex}$|^{float_re}$"
        self.validRegex = regex

        regExp = QRegularExpression(regex)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

        self.textChanged.connect(self.emitValueChanged)
        self.editingFinished.connect(self.emitValueChangeFinished)
        if initial is None:
            self.setText("0.0")

        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)

    def emitValueChangeFinished(self):
        value = self.value()
        self.textChanged.disconnect()
        self.editingFinished.disconnect()
        self.setValue(value)
        self.textChanged.connect(self.emitValueChanged)
        self.editingFinished.connect(self.emitValueChangeFinished)

        self.emitValueChanged(self.text(), signal=self.valueChangeFinished)

    def emitValueChanged(self, text, signal=None):
        m = re.match(self.validRegex, text)
        if m is None:
            self.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)
            return

        if signal is None:
            signal = self.valueChanged

        self.setStyleSheet("")
        signal.emit(self.value())

    def increaseValue(self, step):
        value = self.value()
        if isinstance(value, (float, int)):
            value += step
        else:
            value = [val + step for val in value]
            value = str(value).lstrip("[").rstrip("]")
        self.setValue(value)
        self.emitValueChangeFinished()

    def decreaseValue(self, step):
        value = self.value()
        if isinstance(value, (float, int)):
            value -= step
        else:
            value = [val - step for val in value]
            value = str(value).lstrip("[").rstrip("]")
        self.setText(value)
        self.emitValueChangeFinished()

    def setValue(self, value):
        if isinstance(value, (float, int)):
            if value < self._minimum:
                value = self._minimum
        else:
            clipped = []
            for val in value:
                if val < self._minimum:
                    val = self._minimum
                clipped.append(val)
            value = str(clipped).lstrip("[").rstrip("]")
        self.setText(value)

    def setText(self, text):
        super().setText(str(text))

    def clipValue(self, val: float):
        if val < self._minimum:
            val = self._minimum
        return val

    def value(self):
        m = re.match(self.validRegex, self.text())
        if m is None:
            return 0.0

        try:
            value = self.clipValue(float(self.text()))
            return value
        except Exception as e:
            text = self.text()
            text = text.replace("(", "")
            text = text.replace(")", "")
            text = text.replace("[", "")
            text = text.replace("]", "")
            values = text.split(",")
            return [self.clipValue(float(value)) for value in values]

    def setMinimum(self, minimum):
        self._minimum = float(minimum)


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
            ext={"CSV": ".csv"}, start_dir=myutils.getMostRecentPath()
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
        for file in myutils.listdir(folderpath):
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
            for file in myutils.listdir(images_path):
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


class installJavaDialog(myMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Install Java")
        self.setIcon("SP_MessageBoxWarning")

        txt_macOS = html_utils.paragraph("""
            Your system doesn't have the <code>Java Development Kit</code>
            installed<br> and/or a C++ compiler which is required for the installation of
            <code>javabridge</code><br><br>
            <b>Cell-ACDC is now going to install Java for you</b>.<br><br>
            <i><b>NOTE: After clicking on "Install", follow the instructions<br>
            on the terminal</b>. You will be asked to confirm steps and insert<br>
            your password to allow the installation.</i><br><br>
            If you prefer to do it manually, cancel the process<br>
            and follow the instructions below.
        """)

        txt_windows = html_utils.paragraph("""
            Unfortunately, installing pre-compiled version of
            <code>javabridge</code> <b>failed</b>.<br><br>
            Cell-ACDC is going to <b>try to compile it now</b>.<br><br>
            However, <b>before proceeding</b>, you need to install
            <code>Java Development Kit</code><br> and a <b>C++ compiler</b>.<br><br>
            <b>See instructions below on how to install it.</b>
        """)

        if not is_win:
            self.instructionsButton = self.addButton("Show intructions...")
            self.instructionsButton.setCheckable(True)
            self.instructionsButton.disconnect()
            self.instructionsButton.clicked.connect(self.showInstructions)
            installButton = self.addButton("Install")
            installButton.disconnect()
            installButton.clicked.connect(self.installJava)
            txt = txt_macOS
        else:
            okButton = self.addButton("Ok")
            txt = txt_windows

        self.cancelButton = self.addButton("Cancel")

        label = self.addText(txt)
        label.setWordWrap(False)

        self.resizeCount = 0

    def addInstructionsWindows(self):
        self.scrollArea = QScrollArea()
        _container = QWidget()
        _layout = QVBoxLayout()
        for t, text in enumerate(myutils.install_javabridge_instructions_text()):
            label = QLabel()
            label.setText(text)
            if t == 1 or t == 2:
                label.setOpenExternalLinks(True)
                label.setTextInteractionFlags(Qt.TextBrowserInteraction)
                code_layout = QHBoxLayout()
                code_layout.addWidget(label)
                copyButton = QToolButton()
                copyButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                copyButton.setIcon(QIcon(":edit-copy.svg"))
                copyButton.setText("Copy link")
                if t == 1:
                    copyButton.textToCopy = myutils.jdk_windows_url()
                    code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                else:
                    copyButton.textToCopy = myutils.cpp_windows_url()
                    screenshotButton = QToolButton()
                    screenshotButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                    screenshotButton.setIcon(QIcon(":cog.svg"))
                    screenshotButton.setText("See screenshot")
                    code_layout.addWidget(screenshotButton, alignment=Qt.AlignLeft)
                    code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                    screenshotButton.clicked.connect(self.viewScreenshot)
                copyButton.clicked.connect(self.copyToClipboard)
                code_layout.setStretch(0, 2)
                code_layout.setStretch(1, 0)
                _layout.addLayout(code_layout)
            else:
                _layout.addWidget(label)

        _container.setLayout(_layout)
        self.scrollArea.setWidget(_container)
        self.currentRow += 1
        self._layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self._layout.setRowStretch(self.currentRow, 1)

    def viewScreenshot(self, checked=False):
        self.screenShotWin = view_visualcpp_screenshot(parent=self)
        self.screenShotWin.show()

    def addInstructionsMacOS(self):
        self.scrollArea = QScrollArea()
        _container = QWidget()
        _layout = QVBoxLayout()
        for t, text in enumerate(myutils.install_javabridge_instructions_text()):
            label = QLabel()
            label.setText(text)
            # label.setWordWrap(True)
            if t == 1 or t == 2:
                label.setWordWrap(True)
                code_layout = QHBoxLayout()
                code_layout.addWidget(label)
                copyButton = QToolButton()
                copyButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                copyButton.setIcon(QIcon(":edit-copy.svg"))
                copyButton.setText("Copy")
                if t == 1:
                    copyButton.textToCopy = myutils._install_homebrew_command()
                else:
                    copyButton.textToCopy = myutils._brew_install_java_command()
                copyButton.clicked.connect(self.copyToClipboard)
                code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                # code_layout.addStretch(1)
                code_layout.setStretch(0, 2)
                code_layout.setStretch(1, 0)
                _layout.addLayout(code_layout)
            else:
                _layout.addWidget(label)
        _container.setLayout(_layout)
        self.scrollArea.setWidget(_container)
        self.currentRow += 1
        self._layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self._layout.setRowStretch(self.currentRow, 1)
        self.scrollArea.hide()

    def addInstructionsLinux(self):
        self.scrollArea = QScrollArea()
        _container = QWidget()
        _layout = QVBoxLayout()
        for t, text in enumerate(myutils.install_javabridge_instructions_text()):
            label = QLabel()
            label.setText(text)
            # label.setWordWrap(True)
            if t == 1 or t == 2 or t == 3:
                label.setWordWrap(True)
                code_layout = QHBoxLayout()
                code_layout.addWidget(label)
                copyButton = QToolButton()
                copyButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                copyButton.setIcon(QIcon(":edit-copy.svg"))
                copyButton.setText("Copy")
                if t == 1:
                    copyButton.textToCopy = myutils._apt_update_command()
                elif t == 2:
                    copyButton.textToCopy = myutils._apt_install_java_command()
                elif t == 3:
                    copyButton.textToCopy = myutils._apt_gcc_command()
                copyButton.clicked.connect(self.copyToClipboard)
                code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                # code_layout.addStretch(1)
                code_layout.setStretch(0, 2)
                code_layout.setStretch(1, 0)
                _layout.addLayout(code_layout)
            else:
                _layout.addWidget(label)
        _container.setLayout(_layout)
        self.scrollArea.setWidget(_container)
        self.currentRow += 1
        self._layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self._layout.setRowStretch(self.currentRow, 1)
        self.scrollArea.hide()

    def copyToClipboard(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(self.sender().textToCopy, mode=cb.Clipboard)
        print("Command copied!")

    def showInstructions(self, checked):
        if checked:
            self.instructionsButton.setText("Hide instructions")
            self.origHeight = self.height()
            self.resize(self.width(), self.height() + 300)
            self.scrollArea.show()
        else:
            self.instructionsButton.setText("Show instructions...")
            self.scrollArea.hide()
            func = partial(self.resize, self.width(), self.origHeight)
            QTimer.singleShot(50, func)

    def installJava(self):
        import subprocess

        try:
            if is_mac:
                try:
                    subprocess.check_call(["brew", "update"])
                except Exception as e:
                    subprocess.run(
                        myutils._install_homebrew_command(),
                        check=True,
                        text=True,
                        shell=True,
                    )
                subprocess.run(
                    myutils._brew_install_java_command(),
                    check=True,
                    text=True,
                    shell=True,
                )
            elif is_linux:
                subprocess.run(
                    myutils._apt_gcc_command()(), check=True, text=True, shell=True
                )
                subprocess.run(
                    myutils._apt_update_command()(), check=True, text=True, shell=True
                )
                subprocess.run(
                    myutils._apt_install_java_command()(),
                    check=True,
                    text=True,
                    shell=True,
                )
            self.close()
        except Exception as e:
            print("=======================")
            traceback.print_exc()
            print("=======================")
            msg = myMessageBox(wrapText=False)
            err_msg = html_utils.paragraph("""
                Automatic installation of Java failed.<br><br>
                Please, try manually by following the instructions provided
                below (click on "Show instructions..." button). Thanks
            """)
            msg.critical(self, "Java installation failed", err_msg)

    def show(self, block=False):
        super().show(block=False)
        print(is_linux)
        if is_win:
            self.addInstructionsWindows()
        elif is_mac:
            self.addInstructionsMacOS()
        elif is_linux:
            self.addInstructionsLinux()
        self.move(self.pos().x(), 20)
        if is_win:
            self.resize(self.width(), self.height() + 200)
        if block:
            self._block()

    def exec_(self):
        self.show(block=True)


class selectTrackerGUI(QDialogListbox):
    def __init__(self, SizeT, currentFrameNo=1, parent=None):
        trackers = myutils.get_list_of_trackers()
        super().__init__(
            "Select tracker",
            "Select one of the following trackers",
            trackers,
            multiSelection=False,
            parent=parent,
        )
        self.setWindowTitle("Select tracker")

        self.selectFramesGroupbox = selectStartStopFrames(
            SizeT, currentFrameNum=currentFrameNo, parent=parent
        )

        self.mainLayout.insertWidget(1, self.selectFramesGroupbox)

    def ok_cb(self, event):
        if self.selectFramesGroupbox.warningLabel.text():
            return
        else:
            self.startFrame = self.selectFramesGroupbox.startFrame_SB.value()
            self.stopFrame = self.selectFramesGroupbox.stopFrame_SB.value()
            super().ok_cb(event)


def addWidgetToScrollArea(
    widget,
    resizeMinWidthNoHorizontalScrollbar=False,
    resizeMinHeightNoVerticalScrollbar=False,
):
    container = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(widget)
    layout.addStretch(1)
    container.setLayout(layout)
    scrollArea = QScrollArea()
    scrollArea.setWidgetResizable(True)
    scrollArea.setWidget(container)

    if resizeMinWidthNoHorizontalScrollbar:
        scrollArea.setMinimumWidth(
            container.sizeHint().width()
            + scrollArea.verticalScrollBar().sizeHint().width()
        )

    if resizeMinHeightNoVerticalScrollbar:
        scrollArea.setMinimumHeight(
            container.sizeHint().height()
            + scrollArea.horizontalScrollBar().sizeHint().height()
        )

    return scrollArea


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


class OddSpinBox(SpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSingleStep(2)
        self.editingFinished.connect(self.roundToOdd)

    def roundToOdd(self):
        if self.value() % 2 == 1:
            return

        self.setValue(self.value() + 1)


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


class LineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)

    def value(self):
        return self.text()

    def setValue(self, value):
        self.setText(str(value))


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
        params_argspecs = myutils.get_function_argspec(
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


class WhitelistLineEdit(KeepIDsLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setText(self, IDs):
        if not isinstance(IDs, set) and not isinstance(IDs, list):
            raise TypeError("IDs must be a set or list")

        formatted_text = myutils.format_IDs(IDs)
        super().setText(formatted_text)


class KeySequenceFromText(QKeySequence):
    def __init__(self, text: str):
        if isinstance(text, str):
            text = macShortcutToWindows(text)
        super().__init__(text)
        self._text = text

    def toString(self):
        if isinstance(self._text, str):
            return windowsShortcutToMac(self._text)
        else:
            return windowsShortcutToMac(super().toString())


def modifierKeyToText(modifierKey: int):
    if modifierKey == Qt.ControlModifier:
        return "Ctrl"
    elif modifierKey == Qt.AltModifier:
        return "Alt"
    elif modifierKey == Qt.ShiftModifier:
        return "Shift"
    elif modifierKey == Qt.MetaModifier:
        return "Meta"
    else:
        return ""


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


def get_min_width_for_no_scrollbar(list_widget: QListWidget) -> int:
    """
    Calculate the minimum width needed for the QListWidget
    so that the horizontal scrollbar will not be required.
    """
    font_metrics = QFontMetrics(list_widget.font())
    max_width = 0

    for i in range(list_widget.count()):
        item = list_widget.item(i)
        text_width = font_metrics.horizontalAdvance(item.text())
        max_width = max(max_width, text_width)

    # Add padding for icon, scrollbar margin, and frame
    padding = 30  # Adjust as needed (depends on style and icons)
    return max_width + padding


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

        start_dir = myutils.getMostRecentPath()
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


class warnVisualCppRequired(myMessageBox):
    def __init__(self, pkg_name="javabridge", parent=None):
        super().__init__(parent)
        self.screenShotWin = None

        self.setIcon(iconName="SP_MessageBoxWarning")
        self.setWindowTitle(f"Installation of {pkg_name} info")
        txt = html_utils.paragraph(f"""
            Installation of {pkg_name} on Windows requires
            Microsoft Visual C++ 14.0 or higher.<br><br>
            Cell-ACDC will anyway try to install {pkg_name} now.<br><br>
            If the installation fails, please <b>close Cell-ACDC</b>,
            then download and install <b>"Microsoft C++ Build Tools"</b>
            from the link below
            before trying this module again.<br><br>
            <a href='https://visualstudio.microsoft.com/visual-cpp-build-tools/'>
                https://visualstudio.microsoft.com/visual-cpp-build-tools/
            </a><br><br>
            <b>IMPORTANT</b>: when installing "Microsoft C++ Build Tools"
            make sure to select <b>"Desktop development with C++"</b>.
            Click "See the screenshot" for more details.
        """)
        seeScreenshotButton = QPushButton("See screenshot...")
        okButton = okPushButton("Ok")
        okButton = self.addButton("Ok")
        okButton.disconnect()
        okButton.clicked.connect(self.ok_cb)
        self.addButton(seeScreenshotButton)
        seeScreenshotButton.disconnect()
        seeScreenshotButton.clicked.connect(self.viewScreenshot)
        self.addCancelButton(connect=True)
        self.addText(txt)

    def ok_cb(self):
        self.cancel = False
        self.close()

    def viewScreenshot(self, checked=False):
        self.screenShotWin = view_visualcpp_screenshot(self)
        self.screenShotWin.show()

    def closeEvent(self, event):
        if self.screenShotWin is not None:
            self.screenShotWin.close()

        return super().closeEvent(event)
