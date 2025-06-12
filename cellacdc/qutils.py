from qtpy.QtCore import (
    Qt, QTimer, QEventLoop
)
from qtpy.QtWidgets import QWidget
import functools


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
    def __init__(self, QWidgetToBlink: QWidget, duration_ms=2000, qparent=None) -> None:
        self.duration_ms = duration_ms
        self._widget = QWidgetToBlink
        self.qparent = qparent
        self.blinkON = False
    
    def start(self):
        self.timer = QTimer(self.qparent)
        self.timer.timeout.connect(self.timerCallback)
        self.timer.start(100)

        self.stopTimer = QTimer(self.qparent)
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

def hide_and_delete_layout(layout):
    # Hide all widgets in the layout
    for i in reversed(range(layout.count())):
        widget = layout.itemAt(i).widget()
        if widget is not None:
            widget.hide()
            layout.removeWidget(widget)
            widget.setParent(None)
    
    # Delete the layout
    layout.deleteLater()

def delete_widget(widget):
    widget.hide()
    widget.setParent(None)
    widget.deleteLater()

def replace_certain_vals(getVal, replace_val, by_val):
    """
    Decorator: If the return value of getVal equals replace_val (type-cast to value's type),
    return by_val instead. Otherwise, return the original value.
    """
    @functools.wraps(getVal)
    def wrapper(*args, **kwargs):
        value = getVal(*args, **kwargs)
        try:
            target_val = type(value)(replace_val)
        except Exception:
            return value
        if value == target_val:
            return by_val
        return value
    return wrapper

def set_value_no_signals(widget, value):
    was_blocked = widget.blockSignals(True)
    widget.setValue(value)
    widget.blockSignals(was_blocked)

def set_exclusive_valueSetter(widget, valueSetter, value):
    was_blocked = widget.blockSignals(True)
    valueSetter(widget, value)
    widget.blockSignals(was_blocked)

def hardDelete(item, setPosData=True):
    try:
        item.setParent(None)
    except AttributeError:
        pass
    if setPosData:
        try:
            item.posData = None
        except:
            pass
    try:
        item.deleteLater()
    except AttributeError:
        pass
    item = None