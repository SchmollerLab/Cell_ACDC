"""Composite controls: inputs."""

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


class mySpinBox(QSpinBox):
    sigTabEvent = Signal(object, object)

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def event(self, event):
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key_Tab:
            self.sigTabEvent.emit(event, self)
            return True

        return super().event(event)


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


class highlightableQWidgetAction(QWidgetAction):
    def __init__(self, parent) -> None:
        super().__init__(parent)


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


class OddSpinBox(SpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSingleStep(2)
        self.editingFinished.connect(self.roundToOdd)

    def roundToOdd(self):
        if self.value() % 2 == 1:
            return

        self.setValue(self.value() + 1)


class LineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)

    def value(self):
        return self.text()

    def setValue(self, value):
        self.setText(str(value))


class WhitelistLineEdit(KeepIDsLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setText(self, IDs):
        if not isinstance(IDs, set) and not isinstance(IDs, list):
            raise TypeError("IDs must be a set or list")

        formatted_text = utils.format_IDs(IDs)
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

# Cross-module imports (deferred to avoid import cycles)
from .dialogs import (
    QDialogListbox,
)

