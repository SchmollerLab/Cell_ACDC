from qtpy.QtCore import (
    Qt, QTimer, QEventLoop
)
from qtpy.QtWidgets import QWidget

class QWhileLoop:
    def __init__(
            self, loop_callback, period=100, max_duration=None
        ):
        self._loop_callback = loop_callback
        self._period = period
        self._max_duration = max_duration

    def exec_(self):
        self.loop = QEventLoop()
        self.timer = QTimer()
        self.timer.timeout.connect(self._loop_callback)
        self.timer.start(self._period)
        if self._max_duration is not None:
            self.max_duration_timer = QTimer()
            self.max_duration_timer.timeout.connect(self.stop)
            self.max_duration_timer.start(self._max_duration)
        self.loop.exec_()
    
    def stop(self):
        self.timer.stop()
        if self._max_duration is not None:
            self.max_duration_timer.stop()
        self.loop.exit()

class QControlBlink:
    def __init__(self, QWidgetToBlink: QWidget, duration_ms=2000) -> None:
        self.duration_ms = duration_ms
        self._widget = QWidgetToBlink
        self.blinkON = False
    
    def start(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timerCallback)
        self.timer.start(100)

        self.stopTimer = QTimer(self)
        self.stopTimer.timeout.connect(self.stop)
        self.stopTimer.start(self.duration_ms)
    
    def timerCallback(self):
        if self.blinkON:
            self._widget.setStyleSheet('background-color: orange')
        else:
            self._widget.setStyleSheet('background-color: none')
        self.blinkON = not self.blinkON

    def stop(self):
        self.timer.stop()
        self._widget.setStyleSheet('background-color: none')