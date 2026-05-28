from dataclasses import dataclass

from qtpy import QtCore
from qtpy.QtWidgets import (
    QMainWindow
)

from .._run import _setup_app

@dataclass
class _ImageChannel:
    node: object
    lut: VolumeImageLutBar

class VolumeRendererWindow(QMainWindow):
    def __init__(
            self, 
            app=None, 
            parent=None, 
            version=None,
            title='Cell-ACDC - Volume Renderer'
        ):
        """Initializer."""

        super().__init__(parent)
        self.setWindowTitle(title)

        self._version = version
        self._ui_initialised = False
        
        if app is None:
            app = QtCore.QCoreApplication.instance()
        
        self.app = app
        
        self._init_ui()
    
    def _init_ui(self):
        if self._ui_initialised:
            return
        
        self.scene_layout = QHBoxLayout()
        
        self.topToolBar = widgets.VolumeRendererToolbar(parent=self)
        self.addToolBar(Qt.TopToolBarArea, self.topToolBar)
        
        self.topToolBar.sigHomeView.connect(self.reset_view)
        self.topToolBar.sigSave.connect(self.save_screenshot)
        
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addLayout(self.scene_layout)
        self.setCentralWidget(central)
        
        self._ui_initialised = True
    
    def show(self, block=False):
        self.resize(960, 720)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()

        try:
            self.setEnabled(True)
        except Exception as err:
            pass

        if block:
            self.loop = QEventLoop()
            self.loop.exec_()
    
    def run(self, block=False):
        if self.app is None:
            app, splashScreen = _setup_app(splashscreen=True)  
            splashScreen.close()
        
        self.show(block=block)
    
    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()
        
        return super().closeEvent()

    def reset_view(self):
        ...
    
    def save_screenshot(self):
        ...