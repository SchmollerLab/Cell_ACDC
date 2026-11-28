"""Model parameters window QDialogModelParams (from apps.py), used by the
main GUI, the segmentation module, prompts.py, the tracker
initialisation in myutils/segm_models.py and spotMAX.

Also its post-processing and feature-range sub-widgets
(PostProcessSegmParams, SelectFeaturesRange*) and the automatic
thresholding parameters QDialogAutomaticThresholding (main GUI and
segmentation module).
"""
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QCheckBox, QWidget, QSizePolicy
from ..widgets import numeric_inputs as widgets_numeric_inputs
from ..widgets import line_edits as widgets_line_edits


class VerticalSpacerEmptyWidget(QWidget):
    def __init__(self, parent=None, height=5) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self.setFixedHeight(height)


class NoneWidget:
    def __init__(self):
        pass
    
    def value(self):
        return None
    
    def setValue(self, value):
        return


class PostProcessSegmSlider(widgets_numeric_inputs.sliderWithSpinBox):
    def __init__(self, *args, label=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.label = label
        self.checkbox = QCheckBox('Disable')
        self._layout.addWidget(self.checkbox, self.sliderCol, self.lastCol+1)
        self.checkbox.toggled.connect(self.onCheckBoxToggled)
        self.valueChanged.connect(self.checkExpandRange)
    
    def onCheckBoxToggled(self, checked: bool) -> None:
        super().setDisabled(checked)
        if self.label is not None:
            self.label.setDisabled(checked)
        self.onValueChanged(None)
        self.onEditingFinished()
    
    def onValueChanged(self, value):
        self.valueChanged.emit(value)
    
    def checkExpandRange(self, value):
        if value == self.maximum():
            range = int(self.maximum() - self.minimum())
            half_range = int(range/2)
            newMinimum = self.minimum() + half_range
            newMaximum = self.maximum() + half_range
            self.setMaximum(newMaximum)
            self.setMinimum(newMinimum)
        elif value == self.minimum():
            range = int(self.maximum() - self.minimum())
            half_range = int(range/2)
            newMinimum = self.minimum() - half_range
            newMaximum = self.maximum() - half_range
            self.setMaximum(newMaximum)
            self.setMinimum(newMinimum)
    
    def onEditingFinished(self):
        self.editingFinished.emit()
    
    def value(self):
        if self.checkbox.isChecked():
            return None
        else:
            return super().value()


class PostProcessSegmSpinbox(QWidget):
    valueChanged = Signal(int)
    editingFinished = Signal()
    sigCheckboxToggled = Signal()

    def __init__(self, *args, isFloat=False, label=None, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QHBoxLayout()

        if isFloat:
            self.spinBox = widgets_numeric_inputs.DoubleSpinBox()
        else:
            self.spinBox = widgets_numeric_inputs.SpinBox()

        self.spinBox.valueChanged.connect(self.onValueChanged)
        self.spinBox.editingFinished.connect(self.onEditingFinished)

        layout.addWidget(self.spinBox)
        self.checkbox = QCheckBox('Disable')
        layout.addWidget(self.checkbox)
        layout.setStretch(0,1)
        layout.setStretch(1,0)

        self.label = label

        self.checkbox.toggled.connect(self.onCheckBoxToggled)
        
        layout.setContentsMargins(5, 0, 5, 0)
    
        self.setLayout(layout)
    
    def onCheckBoxToggled(self, checked: bool) -> None:
        self.spinBox.setDisabled(checked)
        if self.label is not None:
            self.label.setDisabled(checked)
        self.onValueChanged(None)
        self.onEditingFinished()
    
    def onValueChanged(self, value):
        self.valueChanged.emit(value)
    
    def onEditingFinished(self):
        self.editingFinished.emit()

    def maximum(self):
        return self.spinBox.maximum()
    
    def setValue(self, value):
        self.spinBox.setValue(value)
    
    def sizeHint(self):
        return self.spinBox.sizeHint()
    
    def setMaximum(self, max):
        self.spinBox.setMaximum(max)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setMinimum(self, min):
        self.spinBox.setMinimum(min)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)
    
    def setDecimals(self, decimals):
        self.spinBox.setDecimals(decimals)
    
    def value(self):
        if self.checkbox.isChecked():
            return None
        else:
            return self.spinBox.value()


def PostProcessSegmWidget(
        minimum, maximum, value, useSliders, isFloat=False, normalize=False,
        label=None
    ):
    if useSliders:
        if normalize:
            maximum = int(maximum*100)
        widget = PostProcessSegmSlider(
            normalize=normalize, isFloat=isFloat, label=label
        )
    else:
        widget = PostProcessSegmSpinbox(label=label, isFloat=isFloat)
    widget.setMinimum(minimum)
    widget.setMaximum(maximum)
    widget.setValue(value)
    return widget


class CheckableSpinBoxWidgets:
    def __init__(self, isFloat=True):
        if isFloat:
            self.spinbox = widgets_line_edits.FloatLineEdit()
        else:
            self.spinbox = widgets_numeric_inputs.SpinBox()
        self.checkbox = QCheckBox('Activate')
        self.spinbox.setEnabled(False)
        self.checkbox.toggled.connect(self.spinbox.setEnabled)
    
    def value(self):
        if not self.checkbox.isChecked():
            return
        return self.spinbox.value()
