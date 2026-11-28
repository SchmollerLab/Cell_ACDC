"""Launcher-only dialogs (from apps.py): the Track sub-cellular objects
setup TrackSubCellObjectsDialog, LogoDialog (only a commented use) and
the unused ChangeUserProfileFolderPathDialog, plus the widgets only they
use.
"""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDoubleSpinBox


class CenteredDoubleSpinbox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
