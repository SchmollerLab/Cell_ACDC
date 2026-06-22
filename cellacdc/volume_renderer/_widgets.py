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
    sigSetSingleChannel = Signal(bool)
    
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
        
        self.addSeparator()
        
        self.singleChannelCheckbox = self.addCheckBox(
            text='Single channel'
        )
        
        self.singleChannelCheckbox.setToolTip(
            'When single channel mode is activated, selecting a channel '
            'will display only that channel in the overlay.'
        )
        
        self.homeViewAction.triggered.connect(self.sigHomeView.emit)
        self.saveAction.triggered.connect(self.sigSave.emit)
        
        self.singleChannelCheckbox.toggled.connect(
            self.sigSetSingleChannel.emit
        )
    
    def is_single_channel_mode(self) -> bool:
        return self.singleChannelCheckbox.isChecked()