"""Basic building blocks used by every program part: separators (QHLine,
QVLine, VerticalResizeHline), spacers (QHWidgetSpacer, QVWidgetSpacer),
ScrollArea, labels (QClickableLabel, Label, AutoGrowFixedSizeLabel,
statusBarPermanentLabel), progress bars (ProgressBar,
ProgressBarWithETA), the log console (QLogConsole), the loading
animation (LoadingCircleAnimation), CheckableWidget, DummyWidget and
MouseCursor.
"""
import time
import numpy as np
import math
from qtpy.QtCore import (
    Signal, Qt, Property, QPropertyAnimation, QPointF, QEvent,
    QPropertyAnimation
)
from qtpy.QtGui import QFont, QPalette, QColor, QPainter
from qtpy.QtWidgets import (
    QTextEdit, QLabel, QProgressBar, QHBoxLayout, QCheckBox, QWidget,
    QScrollArea, QSizePolicy, QFrame, QGraphicsBlurEffect
)
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import myutils, html_utils
from cellacdc import _palettes
from cellacdc import QtScoped
from . import style as widgets_style


class QHWidgetSpacer(QWidget):
    def __init__(self, width=10, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(width)

class QVWidgetSpacer(QWidget):
    def __init__(self, height=10, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(height)

class QHLine(QFrame):
    def __init__(self, shadow='Sunken', parent=None, color=None):
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
    def __init__(self, shadow='Plain', parent=None, color=None):
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
        if event.type() == QtScoped.QEventTypeAttribute('Enter'):
            self.setLineWidth(0)
            self.setMidLineWidth(self._height)
            pal = self.palette()
            pal.setColor(QPalette.ColorRole.WindowText, QColor(widgets_style.BASE_COLOR))
            self.setPalette(pal)
            # self.setStyleSheet('background-color: #4d4d4d') 
        elif event.type() == QtScoped.QEventTypeAttribute('Leave'):
            self.setMidLineWidth(0)
            self.setLineWidth(1)
        return False


class ScrollArea(QScrollArea):
    sigLeaveEvent = Signal()

    def __init__(
            self, parent=None, resizeVerticalOnShow=False, 
            dropArrowKeyEvents=False
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
        if event.type() == QtScoped.QEventTypeAttribute('Leave'):
            self.sigLeaveEvent.emit()

        if object != self.containerWidget:
            return False
        
        isResize = event.type() == QtScoped.QEventTypeAttribute('Resize')
        isShow = event.type() == QtScoped.QEventTypeAttribute('Show')
        if isResize and self.isOnlyVertical:
            self._resizeHorizontal()
        elif isShow and self.resizeVerticalOnShow:
            self._resizeVertical()
        return False

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


class statusBarPermanentLabel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.rightLabel = QLabel('')
        self.leftLabel = QLabel('')

        layout = QHBoxLayout()
        layout.addWidget(self.leftLabel)
        layout.addStretch(10)
        layout.addWidget(self.rightLabel)

        self.setLayout(layout)


class MouseCursor(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._x = None
        self._y = None
        self.setMouseTracking(True)
    
    def mouseMoveEvent(self, event) -> None:
        self.move(event.pos())
        self.update()
        return super().mouseMoveEvent(event)
    
    # def drawAtPos(self, x, y):
    #     self._x = x
    #     self._y = y
    #     self.update()
    
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        # p.setPen(QPen(QColor(0,0,0)))
        # p.setBrush(QBrush(QColor(70,70,70,200)))
        p.drawLine(0,0,200,0)
        p.end()


class QLogConsole(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        font = QFont()
        font.setPixelSize(13)
        self.setFont(font)

    def write(self, message):
        # Method required by tqdm pbar
        message = message.replace('\r ', '')
        if message:
            self.apppendText(message)
        
    def append(self, text: str) -> None:
        super().append(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
    
    def insertPlainText(self, text: str) -> None:
        super().append(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class ProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        palette = self.palette()
        palette.setColor(
            QPalette.ColorRole.Highlight, 
            widgets_style.PROGRESSBAR_QCOLOR
        )
        palette.setColor(
            QPalette.ColorRole.HighlightedText, 
            widgets_style.PROGRESSBAR_HIGHLIGHTEDTEXT_QCOLOR
        )
        self.setPalette(palette)

class ProgressBarWithETA(ProgressBar):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent=parent)
        self.ETA_label = QLabel('NDh:NDm:NDs')

    def update(self, step: int):
        self.setValue(self.value()+step)
        t = time.perf_counter()
        if not hasattr(self, 'last_time_update'):
            self.last_time_update = t
            self.mean_value_duration = None
            return
        seconds_per_value = (t - self.last_time_update)/step
        value_left = self.maximum() - self.value()
        if self.mean_value_duration is None:
            self.mean_value_duration = seconds_per_value
        else:
            self.mean_value_duration = (
                self.mean_value_duration*(self.value()-1) + seconds_per_value
            )/self.value()

        seconds_left = self.mean_value_duration*value_left
        ETA = myutils.seconds_to_ETA(seconds_left)
        self.ETA_label.setText(ETA)
        self.last_time_update = t
        return ETA

    def show(self):
        QProgressBar.show(self)
        self.ETA_label.show()

    def hide(self):
        QProgressBar.hide(self)
        self.ETA_label.hide()


class AutoGrowFixedSizeLabel(QLabel):
    """QLabel with a fixed size fitting its text, re-sized only when the
    text does not fit or is much shorter.

    Changing the text of a normal QLabel re-layouts its parents (e.g., the
    status bar and the main window with its toolbars), which is slow when
    done at every mouse move (e.g., hover values). With a fixed size this
    happens only when the text length changes a lot. With an empty text
    (e.g., the mouse left the image) the space is released.
    """
    # Shrink when the text is shorter than this fraction of the width
    _SHRINK_FRACTION = 0.7
    # Qt's QWIDGETSIZE_MAX (default maximum size of widgets)
    _QWIDGETSIZE_MAX = (1 << 24) - 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixedSize = None
        self._measureLabel = None
        self._fitText()

    def setText(self, text):
        super().setText(text)
        self._fitText()

    def _textSizeHint(self):
        # The size hint of a QLabel is at least its minimum width -->
        # measure the text with an unconstrained (hidden) label
        if self._measureLabel is None:
            self._measureLabel = QLabel()
        label = self._measureLabel
        label.setFont(self.font())
        label.setTextFormat(self.textFormat())
        label.setText(self.text())
        return label.sizeHint()

    def _fitText(self):
        if not self.text():
            if self._fixedSize is not None:
                # Release the space for the other widgets (e.g., messages
                # of the status bar)
                self._fixedSize = None
                self.setMinimumSize(0, 0)
                self.setMaximumSize(self._QWIDGETSIZE_MAX, self._QWIDGETSIZE_MAX)
            return

        hint = self._textSizeHint()
        if self._fixedSize is not None:
            fixedWidth, fixedHeight = self._fixedSize
            fits = hint.width() <= fixedWidth and hint.height() <= fixedHeight
            isMuchShorter = hint.width() < self._SHRINK_FRACTION*fixedWidth
            if fits and not isMuchShorter:
                return

        # Some slack so that the label is not re-sized at every small
        # increase of the text length (e.g., one more digit)
        slack = 4*self.fontMetrics().averageCharWidth()
        width = hint.width() + slack
        height = hint.height()
        self._fixedSize = (width, height)
        self.setFixedSize(width, height)


# class Spinner(QLabel):
#     def __init__(self, size=150, parent=None):
#         super().__init__(parent)
#         # layout = QHBoxLayout()
        
#         # self._label = QLabel()
#         self.setAlignment(Qt.AlignCenter)
#         # self._label.setText('Ciao')
#         self._pixmap = QPixmap(':spinner.svg')
        
#         self._pixmapSize = size + size%2
#         self._halfPixmapSize = int(self._pixmapSize/2)
#         printl(self._pixmapSize, self._halfPixmapSize)
#         self.setPixmap(self._pixmap.scaled(self._pixmapSize, self._pixmapSize))

#         # self.setFixedSize(160, 160)
#         self._angle = 0
        
#         blurEffect = QGraphicsBlurEffect()
#         blurEffect.setBlurRadius(1.4)
#         self.setGraphicsEffect(blurEffect)
        
#         # layout.addWidget(self._label)
#         # self.setLayout(layout)

#         self.animation = QPropertyAnimation(self, b"angle", self)
#         self.animation.setStartValue(0)
#         self.animation.setEndValue(360)
#         self.animation.setLoopCount(-1)
#         self.animation.setDuration(1700)
#         self.animation.start()

#     @Property(int)
#     def angle(self):
#         return self._angle

#     @angle.setter
#     def angle(self, value):
#         self._angle = value
#         self.update()

#     def paintEvent(self, ev=None):
#         width, height = self.size().width(), self.size().height()
#         radius_x = int(width/2)
#         radius_y = int(height/2)
#         x = radius_x-self._halfPixmapSize
#         y = radius_y-self._halfPixmapSize
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.Antialiasing)
#         painter.translate(radius_x, radius_y)
#         painter.rotate(self._angle)
#         painter.translate(-radius_x, -radius_y)
#         painter.drawPixmap(x, y, self._pixmap.scaled(self._pixmapSize, self._pixmapSize))
#         painter.end()

class LoadingCircleAnimation(QLabel):
    def __init__(self, size=32, motionBlur=False, parent=None):
        super().__init__(parent)
        # layout = QHBoxLayout()
        
        # self._label = QLabel()
        self.setAlignment(Qt.AlignCenter)
        self._size = size + size%2
        self._radius = int(self._size/2)
        self.setFixedSize(self._size, self._size)
        self._dotDiameter = int(self._size*0.15)
        self._dotDiameter = self._dotDiameter + self._dotDiameter%2
        self._dotRadius = int(self._dotDiameter/2)
        
        self._rgb = _palettes.getPainterColor()[:3]
        self._index = 0
        
        self.setBrushesAndAngles()
        
        if motionBlur:
            blurEffect = QGraphicsBlurEffect()
            blurRadius = self._size*0.02
            if blurRadius < 1:
                blurRadius = 1
            blurEffect.setBlurRadius(blurRadius)
            self.setGraphicsEffect(blurEffect)

        self.animation = QPropertyAnimation(self, b"index", self)
        self.animation.setStartValue(0)
        self.animation.setEndValue(11)
        self.animation.setLoopCount(-1)
        self.animation.setDuration(1200)
        self.animation.start()
        
        self.update()
    
    def setVisible(self, visible):
        if visible:
            self.animation.start()
        else:
            self.animation.stop()
        super().setVisible(visible)
    
    def setBrushesAndAngles(self):
        self._brushes = []
        self._pens = []
        alphas = np.round(np.linspace(0, 255, 12)).astype(int)
        self._angles = np.arange(0, 360, 30)
        for alpha in alphas:
            color = QColor(*self._rgb, alpha)
            self._brushes.append(pg.mkBrush(color))
            self._pens.append(pg.mkPen(color))

    @Property(int)
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self._radius, self._radius)
        for i in range(12):
            idx = i - self._index
            angle = self._angles[i]
            painter.setBrush(self._brushes[idx])
            painter.setPen(self._pens[idx])
            x = (self._radius-self._dotRadius)*math.cos(angle*math.pi/180)
            y = (self._radius-self._dotRadius)*math.sin(angle*math.pi/180)
            painter.drawEllipse(QPointF(x, y), self._dotRadius, self._dotRadius)
        
        painter.end()


class Label(QLabel):
    def __init__(self, parent=None, force_html=False):
        super().__init__(parent)
        self._force_html = force_html
        self._is_disabled = False
        self._original_stylesheet = self.styleSheet()
        
    def setText(self, text):
        if self._force_html:
            text = html_utils.paragraph(text)
        super().setText(text)
    
    def setStyleSheet(self):
        if self._is_disabled:
            return
        
        self._original_stylesheet = self.styleSheet()
    
    def setDisabled(self, disabled: bool):
        self._is_disabled = disabled
        if disabled:
            self.z_label.setStyleSheet('color: gray')
        else:
            self.z_label.setStyleSheet(self._original_stylesheet)
        
    def isDisabled(self):
        return self._is_disabled


class CheckableWidget(QWidget):
    def __init__(self, widget, valueGetterName='value', parent=None):
        super().__init__(parent)
        
        self.widget = widget
        self.valueGetterName = valueGetterName
        
        widget.setDisabled(True)
        
        layout = QHBoxLayout()
        
        layout.addWidget(widget)
        
        self.checkbox = QCheckBox('Activate')
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


class DummyWidget:
    def __init__(self, *args, **kwargs):
        pass
    
    def setValue(self, *args, **kwargs):
        pass
    
    def setText(self, *args, **kwargs):
        pass
    
    def text(self):
        return
    
    def value(self):
        return
