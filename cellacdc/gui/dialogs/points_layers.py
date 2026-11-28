"""Main-GUI-only Add points layer dialog AddPointsLayerDialog (from
apps.py). Its appearance group box comes from
shared/dialogs/image_tools.py (shared with the volume renderer). Keys
chosen here are checked against guiWin.reservedShortcuts.
"""
from qtpy.QtCore import QRegularExpression
from qtpy.QtGui import QRegularExpressionValidator
from qtpy.QtWidgets import QLineEdit


class NumericCommaLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.validPattern = r'^[0-9,\.]+$'
        regExp = QRegularExpression(self.validPattern)
        self.setValidator(QRegularExpressionValidator(regExp))
    
    def values(self):
        try:
            vals = [float(c) for c in self.text().split(',')]
        except Exception as e:
            vals = []
        return vals
