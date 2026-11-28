"""Numeric value controls: SpinBox and DoubleSpinBox (with the temporary
focus change check _isTemporaryFocusChange), OddSpinBox, read-only spin
boxes (readOnlySpinbox, readOnlyDoubleSpinbox), sliderWithSpinBox, the
ScrollBar family (labelledQScrollbar, ScrollBarWithNumericControl) and
RangeSelector. Leaf module: it imports no other widget module.
"""
import re
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QValidator
from qtpy.QtWidgets import (
    QLabel, QHBoxLayout, QCheckBox, QWidget, QSlider, QSpinBox, QGridLayout,
    QScrollBar, QDoubleSpinBox, QAbstractSpinBox, QStyleOptionSpinBox, QStyle
)
from cellacdc import QtScoped


class ScrollBar(QScrollBar):
    def __init__(self, *args):
        super().__init__(*args)
        self.installEventFilter(self)
        self.setContextMenuPolicy(Qt.NoContextMenu)
    
    def eventFilter(self, object, event) -> bool:
        if event.type() == QtScoped.QEventTypeAttribute('Wheel'):
            return True
        elif event.type() == QtScoped.QEventTypeAttribute('MouseButtonPress'):
            # Filter right-click to prevent context menu
            return event.button() == Qt.MouseButton.RightButton
        elif event.type() == QtScoped.QEventTypeAttribute('MouseButtonRelease'):
            # Filter right-click to prevent context menu
            return event.button() == Qt.MouseButton.RightButton
        return False


class readOnlyDoubleSpinbox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
        # self.setStyleSheet('background-color: rgba(240, 240, 240, 200);')

class readOnlySpinbox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
        # self.setStyleSheet('background-color: rgba(240, 240, 240, 200);')

def _isTemporaryFocusChange(event):
    """Return True if the focus event is due to switching to another window
    (and back), or to opening (and closing) a popup like a context menu: the
    focus then returns to the same widget (it is not a new edit)
    """
    return event.reason() in (
        Qt.FocusReason.ActiveWindowFocusReason,
        Qt.FocusReason.PopupFocusReason
    )

class DoubleSpinBox(QDoubleSpinBox):
    sigValueChanged = Signal(int)

    def __init__(self, parent=None, disableKeyPress=False):
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
        self.setMinimum(-2**31)
        self._valueChangedFunction = None
        self.disableKeyPress = disableKeyPress
        self._isValueTyped = False
        self.lineEdit().textEdited.connect(self._setValueTyped)

    def _setValueTyped(self, text):
        self._isValueTyped = True

    def isValueTyped(self):
        """See `SpinBox.isValueTyped`"""
        return self._isValueTyped

    def focusInEvent(self, event):
        if not _isTemporaryFocusChange(event):
            self._isValueTyped = False
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        if not _isTemporaryFocusChange(event):
            self._isValueTyped = False
        super().focusOutEvent(event)

    def keyPressEvent(self, event) -> None:
        isBackSpaceKey = event.key() == Qt.Key_Backspace
        isDeleteKey = event.key() == Qt.Key_Delete
        isEnterKey = event.key() in (Qt.Key_Enter, Qt.Key_Return)
        try:
            int(event.text())
            isIntegerKey = True
        except:
            isIntegerKey = False
        acceptEvent = isBackSpaceKey or isDeleteKey or isIntegerKey
        if self.disableKeyPress and isEnterKey and self._isValueTyped:
            # See `SpinBox.keyPressEvent`
            event.accept()
            self.clearFocus()
        elif self.disableKeyPress and not acceptEvent:
            event.ignore()
            self.clearFocus()
        else:
            super().keyPressEvent(event)

    def valueFromText(self, text: str) -> float:
        text = text.replace(',', '.')
        return super().valueFromText(text)

class SpinBox(QSpinBox):
    sigUpClicked = Signal()
    sigDownClicked = Signal()
    sigTabEvent = Signal(object, object)
    sigValueChanged = Signal(int)

    def __init__(
        self,
        parent=None,
        disableKeyPress=False,
        allowNegative=True,
    ):
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
        self._start_value = None
        self._display_correction = 0
        self.valueChanged.connect(self.sigValueChanged.emit)
        self._isValueTyped = False
        # Emitted only when the user edits the text (not with the arrows)
        self.lineEdit().textEdited.connect(self._setValueTyped)

    def _setValueTyped(self, text):
        self._isValueTyped = True

    def isValueTyped(self):
        """Return True if the user typed in the spin box since it got the
        focus (Enter then confirms the typed value), False if the value was
        changed only with the arrows (or not changed).

        Switching to another window and back, or opening and closing a popup
        (e.g., the context menu) does not reset it.
        """
        return self._isValueTyped

    def focusInEvent(self, event):
        if not _isTemporaryFocusChange(event):
            self._isValueTyped = False
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        if not _isTemporaryFocusChange(event):
            self._isValueTyped = False
        super().focusOutEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

        opt = QStyleOptionSpinBox()
        self.initStyleOption(opt)

        control = self.style().hitTestComplexControl(
            QStyle.ComplexControl.CC_SpinBox,
            opt,
            event.pos(),
            self,
        )

        if control == QStyle.SubControl.SC_SpinBoxUp:
            self.sigUpClicked.emit()
        elif control == QStyle.SubControl.SC_SpinBoxDown:
            self.sigDownClicked.emit()

    def setStartValue(self, value: int):
        if value is None:
            value = self.minimum()

        self._start_value = int(value)
        self._updateDisplayCorrection()
    
    def getStartValue(self):
        if self._start_value is None:
            return self.minimum()
        
        return self._start_value
    
    def startValue(self):
        return self._start_value

    def _updateDisplayCorrection(self):
        self._display_correction = (
            self._start_value - self.minimum()
            if self._start_value is not None
            else 0
        )
        self.update()
        self.lineEdit().setText(self.textFromValue(self.value()))

    def textFromValue(self, value: int) -> str:
        return str(value + self._display_correction)

    def valueFromText(self, text: str) -> int:
        text = text.strip()

        try:
            displayed_value = int(text)
        except ValueError:
            return self.value()

        return displayed_value - self._display_correction

    def validate(self, text: str, pos: int):
        state, text, pos = super().validate(text, pos)

        # Allow the user to type the displayed value even if its
        # corresponding internal value differs from the minimum.
        try:
            displayed_value = int(text)
        except ValueError:
            return state, text, pos

        internal_value = displayed_value - self._display_correction

        if self.minimum() <= internal_value <= self.maximum():
            return QValidator.State.Acceptable, text, pos

        return state, text, pos

    def keyPressEvent(self, event):
        isBackSpaceKey = event.key() == Qt.Key.Key_Backspace
        isDeleteKey = event.key() == Qt.Key.Key_Delete
        isTabKeyPress = event.key() == Qt.Key.Key_Tab
        isEnterKey = event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return)

        if isTabKeyPress:
            self.sigTabEvent.emit(event, self)

        try:
            int(event.text())
            isIntegerKey = True
        except (ValueError, TypeError):
            isIntegerKey = False

        acceptEvent = isBackSpaceKey or isDeleteKey or isIntegerKey

        if self.disableKeyPress and isEnterKey and self._isValueTyped:
            # Enter only confirms the typed value (leaving the spin box):
            # do not pass it on to the parent window like the other keys,
            # since in the main window Enter applies the active tool (e.g.,
            # "Keep IDs" with the IDs selected so far). Without typing
            # (e.g., after clicking the arrows) Enter is passed on
            event.accept()
            self.clearFocus()
        elif self.disableKeyPress and not acceptEvent:
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
        except TypeError:
            pass

        self.setValue(value)
        self.valueChanged.connect(self._valueChangedFunction)

    def wheelEvent(self, event):
        event.ignore()

    def setLinkedValueWidget(self, widget):
        self._linkedWidget = widget

    def minimumChanged(self, minimum):
        self._updateDisplayCorrection()


class labelledQScrollbar(ScrollBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._label = None

    def setLabel(self, label):
        self._label = label

    def updateLabel(self):
        if self._label is not None:
            position = self.sliderPosition()
            s = self._label.text()
            s = re.sub(r'(\d+)/(\d+)', fr'{position+1:02}/\2', s)
            self._label.setText(s)

    def setSliderPosition(self, position):
        QScrollBar.setSliderPosition(self, position)
        self.updateLabel()

    def setValue(self, value):
        QScrollBar.setValue(self, value)
        self.updateLabel()


class sliderWithSpinBox(QWidget):
    sigValueChange = Signal(object)
    valueChanged = Signal(object)
    editingFinished = Signal()

    def __init__(self, *args, **kwargs):      
        super().__init__(*args)

        layout = QGridLayout()

        title = kwargs.get('title')
        row = 0
        col = 0
        if title is not None:
            titleLabel = QLabel(self)
            titleLabel.setText(title)
            loc = kwargs.get('title_loc', 'top')
            if loc == 'top':
                layout.addWidget(titleLabel, 0, col, alignment=Qt.AlignLeft)
            elif loc=='in_line':
                row = -1
                col = 1
                layout.addWidget(titleLabel, 0, 0, alignment=Qt.AlignLeft)
                layout.setColumnStretch(0, 0)

        self._normalize = False
        normalize = kwargs.get('normalize')
        if normalize is not None and normalize:
            self._normalize = True
            self._isFloat = True

        self._isFloat = False
        isFloat = kwargs.get('isFloat')
        if isFloat is not None and isFloat:
            self._isFloat = True

        self._normalize_factor = kwargs.get('normalize_factor', 1.0)
        
        self.slider = QSlider(Qt.Horizontal, self)

        if self._normalize or self._isFloat:
            self.spinBox = DoubleSpinBox(self)
        else:
            self.spinBox = SpinBox(self)
        self.spinBox.setAlignment(Qt.AlignCenter)
        self.spinBox.setMaximum(2**31-1)

        maximum_on_label = kwargs.get('maximum_on_label')
        spinbox_loc = kwargs.get('spinbox_loc', 'right')
        if spinbox_loc == 'right':
            spinbox_col = col+1
            slider_col = col
            if maximum_on_label is not None:
                maximum_on_label_col = spinbox_col + 1
        elif spinbox_loc == 'left':
            spinbox_col = col
            slider_col = col + 1
            if maximum_on_label is not None:
                maximum_on_label_col = spinbox_col + 1
                slider_col += 1

        if maximum_on_label is not None:
            self.labelMaximum = QLabel()
            layout.addWidget(self.labelMaximum, row+1, maximum_on_label_col)
        layout.addWidget(self.slider, row+1, slider_col)
        layout.addWidget(self.spinBox, row+1, spinbox_col)
        
        if title is not None:
            layout.setRowStretch(0, 1)
        layout.setRowStretch(row+1, 1)
        layout.setColumnStretch(slider_col, 6)
        layout.setColumnStretch(spinbox_col, 1)

        self._layout = layout
        self.lastCol = col+1
        self.sliderCol = slider_col

        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.slider.sliderReleased.connect(self.onEditingFinished)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.spinBox.editingFinished.connect(self.onEditingFinished)
        
        layout.setContentsMargins(5, 0, 5, 0)
        
        self.setLayout(layout)

        if maximum_on_label is not None:
            self.setMaximum(maximum_on_label)
            self.labelMaximum.setText(f'/{maximum_on_label}')

    def onEditingFinished(self):
        self.editingFinished.emit()

    def maximum(self):
        return self.slider.maximum()

    def minimum(self):
        return self.slider.minimum()

    def setValue(self, value, emitSignal=False):
        valueInt = value
        if self._normalize:
            valueInt = int(value*self.slider.maximum())
        elif self._isFloat:
            valueInt = int(value*self._normalize_factor)

        self.spinBox.valueChanged.disconnect()
        self.spinBox.setValue(value)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)

        self.slider.valueChanged.disconnect()
        if valueInt > self.slider.maximum():
            self.slider.setMaximum(valueInt)
        self.slider.setValue(valueInt)
        self.slider.valueChanged.connect(self.sliderValueChanged)

        if emitSignal:
            self.sigValueChange.emit(self.value())
            self.valueChanged.emit(self.value())

    def setMaximum(self, max_val, including_spinbox=False):
        max_val_int = max_val
        if isinstance(max_val, float):
            max_val_int = int(max_val*self._normalize_factor)
            
        self.slider.setMaximum(max_val_int)
        if including_spinbox:
            self.spinBox.setMaximum(max_val)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setMinimum(self, min_val, including_spinbox=False):
        min_val_int = min_val
        if isinstance(min_val, float):
            min_val_int = int(min_val*self._normalize_factor)
            
        self.slider.setMinimum(min_val_int)
        if including_spinbox:
            self.spinBox.setMinimum(min_val)

    def setRange(self, min_val, max_val, including_spinbox=False):
        self.setMinimum(min_val, including_spinbox=including_spinbox)
        self.setMaximum(max_val, including_spinbox=including_spinbox)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setDecimals(self, decimals):
        self.spinBox.setDecimals(decimals)

    def setTickPosition(self, position):
        self.slider.setTickPosition(position)

    def setTickInterval(self, interval):
        self.slider.setTickInterval(interval)

    def sliderValueChanged(self, val):
        self.spinBox.blockSignals(True)
        if self._normalize:
            valF = val/self.slider.maximum()
            self.spinBox.setValue(valF)
        elif self._isFloat:
            val_float = val / self._normalize_factor
            self.spinBox.setValue(val_float)
        else:
            self.spinBox.setValue(val)
        self.spinBox.blockSignals(False)
        self.sigValueChange.emit(self.value())
        self.valueChanged.emit(self.value())

    def spinboxValueChanged(self, val):
        if self._normalize:
            val = int(val*self.slider.maximum())
        elif self._isFloat:
            val = int(val*self._normalize_factor)
        
        self.slider.valueChanged.disconnect()
        self.slider.setValue(val)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.sigValueChange.emit(self.value())
        self.valueChanged.emit(self.value())

    def value(self):
        return self.spinBox.value()
    
    def setDisabled(self, disabled) -> None:
        self.slider.setDisabled(disabled)
        self.spinBox.setDisabled(disabled)


class ScrollBarWithNumericControl(QWidget):
    sigValueChanged = Signal(int)
    sigMaxProjToggled = Signal(bool, object)
    
    def __init__(
            self, orientation=Qt.Horizontal, add_max_proj_button=False, 
            parent=None, labelText=''
        ) -> None:
        super().__init__(parent)
        
        self._slot = None
    
        layout = QHBoxLayout()
        self.scrollbar = QScrollBar(orientation, self)
        self.spinbox = SpinBox(self)
        self.maxLabel = QLabel(self)
        self.leftLabel = None
        idx = 0
        if labelText:
            self.leftLabel = QLabel(labelText)
            layout.addWidget(self.leftLabel)
            layout.setStretch(idx, 0)
            idx += 1

        layout.addWidget(self.spinbox)
        layout.setStretch(idx,0)
        idx += 1
        
        layout.addWidget(self.maxLabel)
        layout.setStretch(idx,0)
        idx += 1
        
        layout.addWidget(self.scrollbar)
        layout.setStretch(idx,1)
        idx += 1
        
        if add_max_proj_button:
            self.maxProjCheckbox = QCheckBox('MAX')
            self.scrollbar.maxProjCheckbox = self.maxProjCheckbox
            layout.addWidget(self.maxProjCheckbox)
            layout.setStretch(idx,0)
        
        layout.setContentsMargins(5, 0, 5, 0)

        self.setLayout(layout)

        self.spinbox.valueChanged.connect(self.spinboxValueChanged)
        self.scrollbar.valueChanged.connect(self.scrollbarValueChanged)

        if add_max_proj_button:
            self.maxProjCheckbox.toggled.connect(self.maxProjToggled)
    
    def setStartValue(self, start_value: int):
        if start_value is None:
            return

        self.spinbox.setStartValue(start_value)
        if self.leftLabel is None:
            return
        
        label_text = self.leftLabel.text()
        if start_value == 0:
            label_text = label_text.replace('n.', 'idx')
        else:
            label_text = label_text.replace('idx', 'n.')
        self.leftLabel.setText(label_text)
        if not self.maxLabel.text():
            return
        
        correction = start_value - self.spinbox.minimum()
        maximum = self.maximum()
        self.maxLabel.setText(f'/{maximum + correction}')

    def connectValueChanged(self, slot):
        self.sigValueChanged.connect(slot)
        self._slot = slot
    
    def setValueNoSignal(self, value):
        if self._slot is None:
            return
        self.sigValueChanged.disconnect()
        self.setValue(value)
        self.sigValueChanged.connect(self._slot)
    
    def maxProjToggled(self, checked):
        self.scrollbar.setDisabled(checked)
        self.sigMaxProjToggled.emit(checked, self)
    
    def showEvent(self, event) -> None:
        super().showEvent(event)

        self.scrollbar.setMinimumHeight(self.spinbox.height())
    
    def setMaximum(self, maximum):
        self.maxLabel.setText(f'/{maximum}')
        self.scrollbar.setMaximum(maximum)
        self.spinbox.setMaximum(maximum)
        self.setStartValue(self.spinbox._start_value)
    
    def setMinimum(self, minumum):
        self.scrollbar.setMinimum(minumum)
        self.spinbox.setMinimum(minumum)
        self.setStartValue(self.spinbox._start_value)
    
    def spinboxValueChanged(self, value):
        self.scrollbar.setValue(value)
    
    def scrollbarValueChanged(self, value):
        self.spinbox.setValue(value)
        self.sigValueChanged.emit(value)
    
    def setValue(self, value):
        self.scrollbar.setValue(value)
    
    def value(self):
        return self.scrollbar.value()
    
    def maximum(self):
        return self.scrollbar.maximum()


class OddSpinBox(SpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setSingleStep(2)
        self.editingFinished.connect(self.roundToOdd)
    
    def roundToOdd(self):
        if self.value() % 2 == 1:
            return
        
        self.setValue(self.value()+1)


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
