"""Generic input, list, tree and progress dialogs used by several program
parts, library modules (prompts.py, service/prompts.py) and shared
widgets (from apps.py).

QDialogCombobox, QLineEditDialog (with POSITIVE_FLOAT_REGEX),
NumericEntryDialog, QInput, QDialogEntriesWidget, the tree selectors
QTreeDialog, TreeSelectorDialog (spotMAX subclasses it) and
TreesSelectorDialog, SearchableListboxDialog (recent paths) and
QDialogWorkerProgress, plus the widgets that only these dialogs use.
"""
import os
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItem
from qtpy.QtWidgets import QLineEdit, QTreeWidgetItem
from cellacdc import path as acdc_path
from ..widgets import search_widgets as widgets_search_widgets


class SearchLineEdit(QLineEdit):
    def __init__(self, text='Search...', parent=None):
        super().__init__(parent)
        
        self._text = text
        self.initSearch()
        self.setFocusPolicy(Qt.ClickFocus)
        
    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        if super().text() == self._text:
            self.setText('')
        self.setStyleSheet('')
    
    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        if not super().text():
            self.initSearch()
    
    def initSearch(self):
        self.setText(self._text)
        self.setStyleSheet('color: rgb(150, 150, 150)')
        self.clearFocus()
    
    def text(self):
        if super().text() == self._text:
            return ''
        return super().text()


class _RecentItemsSearchCompleter(widgets_search_widgets.ButtonSearchCompleter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alwaysRecency = False

    def setAlwaysRecency(self, alwaysRecency):
        self.alwaysRecency = alwaysRecency
        self.invalidate()
        self.sort(0, Qt.AscendingOrder)

    def lessThan(self, left, right):
        if self.alwaysRecency:
            return left.row() < right.row()

        left_score = self._getScore(left)
        right_score = self._getScore(right)

        if left_score[0] != right_score[0]:
            return left_score[0] < right_score[0]
        
        if left_score[1] != right_score[1]:
            return left_score[1] < right_score[1]

        return left.row() < right.row()


class StandardItem(QStandardItem):
    def __init__(self, other):
        super().__init__(other)
    
    def setData(self, value, role = ...):
        return super().setData(value, role)

class PathTreeWidgetItem(QTreeWidgetItem):
    def __init__(self, path, strings):
        path = acdc_path.unexpanduser(path)
        strings = [path, *strings]
        super().__init__(strings)
    
    def text(self, column):
        if column == 0:
            path = super().text(column)
            path = os.path.expanduser(path)
            return path
            
        return super().text(column)
