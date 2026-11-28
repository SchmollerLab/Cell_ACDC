"""Data structure setup dialogs (from apps.py): DataStructureSetupDialogue
(data_structure/data_struct.py) and
ReStructTiffDataStructureSetupDialogue
(data_structure/data_restruct.py), plus the widgets only they use.
"""
from qtpy.QtCore import QEvent, QObject
from qtpy.QtWidgets import QLineEdit
from cellacdc import QtScoped


class ReadOnlyLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        # self.setStyleSheet(
        #     'background-color: rgba(240, 240, 240, 200);'
        # )
        self.installEventFilter(self)
    
    def eventFilter(self, a0: 'QObject', a1: 'QEvent') -> bool:
        if a1.type() == QtScoped.QEventTypeAttribute('FocusIn'):
            return True
        return super().eventFilter(a0, a1)

    def setValue(self, value):
        self.setText(str(value))
    
    def value(self, casting_func: callable = None):
        text = self.text()
        if casting_func is not None:
            return casting_func(text)
        return text
