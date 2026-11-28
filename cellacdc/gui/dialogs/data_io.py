"""Main-GUI-only loading, saving, restore and export dialogs (from
apps.py).

QDialogZsliceAbsent, the save progress bar QDialogPbar, the acdc_output
version restore SelectAcdcDfVersionToRestore, AutoSaveIntervalDialog and
the export to video/image parameters (ExportToVideoParametersDialog,
ExportToImageParametersDialog), plus the widgets only they use.
"""
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QWidget
from cellacdc.shared.widgets import numeric_inputs as widgets_numeric_inputs
from cellacdc.shared.widgets import combo_boxes as widgets_combo_boxes


class AutoSaveIntervalWidget(QWidget):
    sigValueChanged = Signal(float, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout()
        
        autoSaveIntervalTooltip = (
            'Autosave every minutes or frames specified here.'
        )
        
        self.setToolTip(autoSaveIntervalTooltip)
        
        self.spinbox = widgets_numeric_inputs.DoubleSpinBox()
        self.spinbox.setMinimum(0)
        self.spinbox.setValue(2)
        self.spinbox.setDecimals(2)
        self.spinbox.setSingleStep(1.0)
        
        layout.addWidget(self.spinbox)
        
        self.unitCombobox = widgets_combo_boxes.ComboBox()
        self.unitCombobox.addItems(['minutes', 'frames'])
        layout.addWidget(self.unitCombobox)
        
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        layout.setContentsMargins(5, 0, 5, 0)
    
        self.setLayout(layout)
        
        self.spinbox.sigValueChanged.connect(self.emitSigValueChanged)
        self.unitCombobox.sigTextChanged.connect(self.emitSigValueChanged)
    
    def emitSigValueChanged(self, *args, **kwargs):
        self.sigValueChanged.emit(
            self.spinbox.value(), 
            self.unitCombobox.currentText()
        )
