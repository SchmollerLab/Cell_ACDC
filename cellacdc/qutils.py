from PyQt5.QtCore import (
    Qt, QTimer, QEventLoop
)

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