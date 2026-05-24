"""Composite controls: dialogs."""

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
        func = partial(utils.showInExplorer, path)
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
            if utils.is_iterable(layouts):
                for layout in layouts:
                    self.addLayout(layout)
            else:
                self.addLayout(layout)

        if widgets is not None:
            self._layout.addItem(QSpacerItem(20, 20), self.currentRow, 1)
            self.currentRow += 1
            if utils.is_iterable(widgets):
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
        for t, text in enumerate(utils.install_javabridge_instructions_text()):
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
                    copyButton.textToCopy = utils.jdk_windows_url()
                    code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                else:
                    copyButton.textToCopy = utils.cpp_windows_url()
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
        for t, text in enumerate(utils.install_javabridge_instructions_text()):
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
                    copyButton.textToCopy = utils._install_homebrew_command()
                else:
                    copyButton.textToCopy = utils._brew_install_java_command()
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
        for t, text in enumerate(utils.install_javabridge_instructions_text()):
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
                    copyButton.textToCopy = utils._apt_update_command()
                elif t == 2:
                    copyButton.textToCopy = utils._apt_install_java_command()
                elif t == 3:
                    copyButton.textToCopy = utils._apt_gcc_command()
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
                        utils._install_homebrew_command(),
                        check=True,
                        text=True,
                        shell=True,
                    )
                subprocess.run(
                    utils._brew_install_java_command(),
                    check=True,
                    text=True,
                    shell=True,
                )
            elif is_linux:
                subprocess.run(
                    utils._apt_gcc_command()(), check=True, text=True, shell=True
                )
                subprocess.run(
                    utils._apt_update_command()(), check=True, text=True, shell=True
                )
                subprocess.run(
                    utils._apt_install_java_command()(),
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
        trackers = utils.get_list_of_trackers()
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

# Cross-module imports (deferred to avoid import cycles)
from .forms import (
    CopiableCommandWidget,
    LabelsWidget,
    selectStartStopFrames,
)
from .panels import (
    listWidget,
)

