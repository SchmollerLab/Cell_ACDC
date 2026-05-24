from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

from .buttons import cancelPushButton, okPushButton
from .palette import BASE_COLOR

class VerticalSpacerEmptyWidget(QWidget):

    def __init__(self, parent=None, height=5) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setFixedHeight(height)
class QHWidgetSpacer(QWidget):
    def __init__(self, width=10, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(width)


class QVWidgetSpacer(QWidget):
    def __init__(self, height=10, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(height)


class QHLine(QFrame):
    def __init__(self, shadow="Sunken", parent=None, color=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(getattr(QFrame, shadow))
        if color is not None:
            self.setColor(color)

    def setColor(self, color):
        qcolor = pg.mkColor(color)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.WindowText, qcolor)
        self.setPalette(pal)


class QVLine(QFrame):
    def __init__(self, shadow="Plain", parent=None, color=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFrameShadow(getattr(QFrame.Shadow, shadow))
        if color is not None:
            self.setColor(color)

    def setColor(self, color):
        qcolor = pg.mkColor(color)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.WindowText, qcolor)
        self.setPalette(pal)


class VerticalResizeHline(QFrame):
    dragged = Signal(object)
    clicked = Signal(object)
    released = Signal(object)

    def __init__(self):
        super().__init__()
        self.setCursor(Qt.SplitVCursor)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.installEventFilter(self)
        self.isMousePressed = False
        self._height = 4
        self.setMinimumHeight(self._height)

    def mousePressEvent(self, event) -> None:
        self.isMousePressed = True
        self.clicked.emit(event)
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self.dragged.emit(event)
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self.isMousePressed = False
        self.released.emit(event)
        return super().mouseReleaseEvent(event)

    def eventFilter(self, object, event):
        if event.type() == QEvent.Type.Enter:
            self.setLineWidth(0)
            self.setMidLineWidth(self._height)
            pal = self.palette()
            pal.setColor(QPalette.ColorRole.WindowText, QColor(BASE_COLOR))
            self.setPalette(pal)
            # self.setStyleSheet('background-color: #4d4d4d')
        elif event.type() == QEvent.Type.Leave:
            self.setMidLineWidth(0)
            self.setLineWidth(1)
        return False


class GroupBox(QGroupBox):
    def __init__(self, *args, keyPressCallback=None):
        super().__init__(*args)
        self.keyPressCallback = None
        self.setFocusPolicy(Qt.NoFocus)

    def keyPressEvent(self, event) -> None:
        event.ignore()
        if self.keyPressCallback is None:
            return

        self.keyPressCallback()


class CheckBox(QCheckBox):
    def __init__(self, *args, keyPressCallback=None):
        super().__init__(*args)
        self.keyPressCallback = None
        self.setFocusPolicy(Qt.NoFocus)

    def keyPressEvent(self, event) -> None:
        event.ignore()
        if self.keyPressCallback is None:
            return

        self.keyPressCallback()


class CancelOkButtonsLayout(QHBoxLayout):
    def __init__(self, *args, additionalButtons=None):
        super().__init__(*args)

        self.cancelButton = cancelPushButton("Cancel")
        self.okButton = okPushButton(" Ok ")

        self.addStretch(1)
        self.addWidget(self.cancelButton)
        self.addSpacing(20)

        if additionalButtons is not None:
            for button in additionalButtons:
                self.addWidget(button)

        self.addWidget(self.okButton)

class FormLayout(QGridLayout):
    def __init__(self):
        QGridLayout.__init__(self)

    def addFormWidget(
        self, formWidget, leftLabelAlignment=Qt.AlignRight, align=None, row=0
    ):
        for col, item in enumerate(formWidget.items):
            if col == 0:
                alignment = leftLabelAlignment
            elif col == 2:
                alignment = Qt.AlignLeft
            else:
                alignment = align
            try:
                if alignment is None:
                    self.addWidget(item, row, col)
                else:
                    self.addWidget(item, row, col, alignment=alignment)
            except TypeError:
                self.addLayout(item, row, col)


class ScrollArea(QScrollArea):
    sigLeaveEvent = Signal()

    def __init__(
        self, parent=None, resizeVerticalOnShow=False, dropArrowKeyEvents=False
    ) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.containerWidget = None
        self.resizeVerticalOnShow = resizeVerticalOnShow
        self.isOnlyVertical = False
        self.dropArrowKeyEvents = dropArrowKeyEvents

    def setVerticalLayout(self, layout, widget=None):
        if widget is None:
            self.containerWidget = QWidget()
        else:
            self.containerWidget = widget
        self.containerWidget.setLayout(layout)
        self.containerWidget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.setWidget(self.containerWidget)
        self.containerWidget.installEventFilter(self)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.isOnlyVertical = True

    def setWidget(self, widget):
        self.containerWidget = widget
        super().setWidget(widget)

    def _resizeHorizontal(self):
        self.setMinimumWidth(
            self.containerWidget.minimumSizeHint().width()
            + self.verticalScrollBar().width()
        )

    def minimumWidthNoScrollbar(self) -> int:
        width = (
            self.containerWidget.minimumSizeHint().width()
            + self.verticalScrollBar().width()
        )
        return width

    def minimumHeightNoScrollbar(self) -> int:
        height = (
            self.containerWidget.minimumSizeHint().height()
            + self.horizontalScrollBar().height()
        )
        return height

    def _resizeVertical(self):
        height = (
            self.containerWidget.minimumSizeHint().height()
            + self.horizontalScrollBar().height()
        )
        self.containerWidget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        self.setFixedHeight(height)

    def eventFilter(self, object, event: QEvent):
        if event.type() == QEvent.Type.Leave:
            self.sigLeaveEvent.emit()

        if object != self.containerWidget:
            return False

        isResize = event.type() == QEvent.Type.Resize
        isShow = event.type() == QEvent.Type.Show
        if isResize and self.isOnlyVertical:
            self._resizeHorizontal()
        elif isShow and self.resizeVerticalOnShow:
            self._resizeVertical()
        return False
