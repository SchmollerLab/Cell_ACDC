from qtpy.QtCore import (
    Signal, Qt
)
from qtpy.QtWidgets import (
    QAction
)
from qtpy.QtGui import (
    QIcon,
)

from cellacdc.widgets import ToolBar

class VolumeRendererToolbar(ToolBar):
    sigHomeView = Signal()
    sigSave = Signal()
    
    def __init__(self, name='Volume Renderer Toolbar', parent=None):
        
        super().__init__(name, parent)
        
        self.parentWin = parent
        
        self.setContextMenuPolicy(Qt.PreventContextMenu)
        
        self.homeViewAction = QAction(QIcon(':home.svg'), 'Home view', self)
        self.homeViewAction.setShortcut('H')
        self.homeViewAction.setToolTip(
            'Reset the view to the default orientation and zoom level'
        )
        self.addAction(self.homeViewAction)
        
        self.saveAction = QAction(QIcon(':file-save.svg'), 'Save', self)
        self.saveAction.setShortcut('Ctrl+S')
        self.saveAction.setToolTip(
            'Save the current view to PNG file'
        )
        self.addAction(self.saveAction)
        
        self.homeViewAction.triggered.connect(self.sigHomeView.emit)
        self.saveAction.triggered.connect(self.sigSave.emit)