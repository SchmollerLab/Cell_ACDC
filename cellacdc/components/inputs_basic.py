import re

from qtpy.QtCore import (
    QEvent,
    Qt,
    Signal,
)
from qtpy.QtGui import (
    QFontMetrics,
    QKeyEvent,
    QRegularExpressionValidator,
)
from qtpy.QtWidgets import (
    QLineEdit,
    QScrollBar,
)

from .palette import LINEEDIT_INVALID_ENTRY_STYLESHEET

class ElidingLineEdit(QLineEdit):
    def __init__(self, parent=None, minWidth=None):
        super().__init__(parent)
        self._text = ""
        self._minWidth = minWidth
        if minWidth is not None:
            self.setMinimumWidth(minWidth)

        self.textEdited.connect(self.setText)
        self.installEventFilter(self)
        self._elide = True

    def setText(self, text: str, width=None, elide=True) -> None:
        if width is None:
            width = self._minWidth

        if width is None:
            try:
                textToPrevRatio = len(text) / len(self.text())
                width = round(self.width() * textToPrevRatio)
            except ZeroDivisionError:
                width = self.width()

        if width > self.width():
            width = self.width()

        self._text = text
        if not elide or not self._elide:
            super().setText(text)
            return

        fm = QFontMetrics(self.font())
        elidedText = fm.elidedText(text, Qt.ElideLeft, width)

        super().setText(elidedText)
        self.setToolTip(text)

    def text(self):
        return self._text

    def resizeEvent(self, event):
        newWidth = event.size().width()
        self.setText(self._text, width=newWidth)
        event.accept()

    def eventFilter(self, a0: "QObject", a1: "QEvent") -> bool:
        isFocusIn = a1.type() == QEvent.Type.FocusIn
        if isFocusIn and (self.isReadOnly() or not self.isEnabled()):
            self.clearFocus()
            return True
        return super().eventFilter(a0, a1)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._elide = False
        self.setText(self._text, elide=False)
        self.setCursorPosition(len(self.text()))

    def focusOutEvent(self, event):
        self._elide = True
        super().focusOutEvent(event)
        self.setText(self._text)


class ValidLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

    def setInvalidStyleSheet(self):
        self.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)

    def setValidStyleSheet(self):
        self.setStyleSheet("")


class KeepIDsLineEdit(ValidLineEdit):
    sigIDsChanged = Signal(list)
    sigSort = Signal()
    sigEnterPressed = Signal()

    def __init__(self, instructionsLabel, parent=None):
        super().__init__(parent)

        self.validPattern = "^[0-9-, ]+$"
        regExpr = QRegularExpression(self.validPattern)
        self.setValidator(QRegularExpressionValidator(regExpr))

        self.textChanged.connect(self.onTextChanged)
        self.editingFinished.connect(self.onEditingFinished)

        self.instructionsText = instructionsLabel.text()
        self._label = instructionsLabel

    def keyPressEvent(self, event) -> None:
        super().keyPressEvent(event)
        if event.text() == ",":
            self.sigSort.emit()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self.sigEnterPressed.emit()

    def onTextChanged(self, text):
        IDs = []
        rangesMatch = re.findall(r"(\d+-\d+)", text)
        if rangesMatch:
            for rangeText in rangesMatch:
                start, stop = rangeText.split("-")
                start, stop = int(start), int(stop)
                IDs.extend(range(start, stop + 1))
            text = re.sub(r"(\d+)-(\d+)", "", text)
        IDsMatch = re.findall(r"(\d+)", text)
        if IDsMatch:
            for ID in IDsMatch:
                IDs.append(int(ID))
        self.IDs = sorted(list(set(IDs)))
        self.sigIDsChanged.emit(self.IDs)

    def onEditingFinished(self):
        self.sigSort.emit()

    def warnNotExistingID(self):
        self.setInvalidStyleSheet()
        self._label.setText(
            "  Some of the IDs are not existing --> they will be IGNORED"
        )
        self._label.setStyleSheet("color: red")

    def setInstructionsText(self):
        self.setValidStyleSheet()
        self._label.setText(self.instructionsText)
        self._label.setStyleSheet("")


class ScrollBar(QScrollBar):
    def __init__(self, *args):
        super().__init__(*args)
        self.installEventFilter(self)
        self.setContextMenuPolicy(Qt.NoContextMenu)

    def eventFilter(self, object, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            return True
        elif event.type() == QEvent.Type.MouseButtonPress:
            # Filter right-click to prevent context menu
            return event.button() == Qt.MouseButton.RightButton
        elif event.type() == QEvent.Type.MouseButtonRelease:
            # Filter right-click to prevent context menu
            return event.button() == Qt.MouseButton.RightButton
        return False
