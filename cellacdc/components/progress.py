import logging
import math
import sys
import time

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Property, QPropertyAnimation, QObject, QPointF, Qt, Signal
from qtpy.QtGui import QFont, QPalette, QPainter, QColor, QPen
from qtpy.QtWidgets import (
    QGraphicsBlurEffect,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QTextEdit,
)

from .. import _palettes, utils
from .palette import PROGRESSBAR_HIGHLIGHTEDTEXT_QCOLOR, PROGRESSBAR_QCOLOR


class XStream(QObject):
    _stdout = None
    _stderr = None
    messageWritten = Signal(str)

    def flush(self):
        pass

    def fileno(self):
        return -1

    def write(self, msg):
        if not self.signalsBlocked():
            self.messageWritten.emit(msg)

    @staticmethod
    def stdout():
        if not XStream._stdout:
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        if not XStream._stderr:
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr


class QtHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        record = self.format(record)
        if record:
            XStream.stdout().write("%s\n" % record)


class QLog(QPlainTextEdit):
    sigClose = Signal()

    def __init__(self, *args, logger=None):
        super().__init__(*args)
        self.logger = logger
        self.setReadOnly(True)

    def connect(self):
        XStream.stdout().messageWritten.connect(self.writeStdOutput)

    def writeStdOutput(self, text: str) -> None:
        super().insertPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def writeStdErr(self, text: str) -> None:
        super().insertPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        if self.logger is not None:
            self.logger.exception(text)

    def insertPlainText(self, text: str) -> None:
        super().insertPlainText(f"{text}\n")
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
        self.sigClose.emit()


class QLogConsole(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        font = QFont()
        font.setPixelSize(13)
        self.setFont(font)

    def write(self, message):
        message = message.replace("\r ", "")
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
        palette.setColor(QPalette.ColorRole.Highlight, PROGRESSBAR_QCOLOR)
        palette.setColor(
            QPalette.ColorRole.HighlightedText, PROGRESSBAR_HIGHLIGHTEDTEXT_QCOLOR
        )
        self.setPalette(palette)


class ProgressBarWithETA(ProgressBar):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent=parent)
        self.ETA_label = QLabel("NDh:NDm:NDs")

    def update(self, step: int):
        self.setValue(self.value() + step)
        t = time.perf_counter()
        if not hasattr(self, "last_time_update"):
            self.last_time_update = t
            self.mean_value_duration = None
            return
        seconds_per_value = (t - self.last_time_update) / step
        value_left = self.maximum() - self.value()
        if self.mean_value_duration is None:
            self.mean_value_duration = seconds_per_value
        else:
            self.mean_value_duration = (
                self.mean_value_duration * (self.value() - 1) + seconds_per_value
            ) / self.value()

        seconds_left = self.mean_value_duration * value_left
        ETA = utils.seconds_to_ETA(seconds_left)
        self.ETA_label.setText(ETA)
        self.last_time_update = t
        return ETA

    def show(self):
        QProgressBar.show(self)
        self.ETA_label.show()

    def hide(self):
        QProgressBar.hide(self)
        self.ETA_label.hide()


class NoneWidget:
    def __init__(self):
        pass

    def value(self):
        return None

    def setValue(self, value):
        return


class LoadingCircleAnimation(QLabel):
    def __init__(self, size=32, motionBlur=False, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._size = size + size % 2
        self._radius = int(self._size / 2)
        self.setFixedSize(self._size, self._size)
        self._dotDiameter = int(self._size * 0.15)
        self._dotDiameter = self._dotDiameter + self._dotDiameter % 2
        self._dotRadius = int(self._dotDiameter / 2)

        self._rgb = _palettes.getPainterColor()[:3]
        self._index = 0

        self.setBrushesAndAngles()

        if motionBlur:
            blurEffect = QGraphicsBlurEffect()
            blurRadius = self._size * 0.02
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
            x = (self._radius - self._dotRadius) * math.cos(angle * math.pi / 180)
            y = (self._radius - self._dotRadius) * math.sin(angle * math.pi / 180)
            painter.drawEllipse(QPointF(x, y), self._dotRadius, self._dotRadius)

        painter.end()
