from qtpy.QtCore import (
    Signal, Qt, QPoint
)
from qtpy.QtWidgets import (
    QAction, QWidget
)
from qtpy.QtGui import (
    QIcon, QColor, QFont, QPainter, QPainterPath, QPen
)

from cellacdc.widgets import ToolBar

class VolumeRendererToolbar(ToolBar):
    sigHomeView = Signal()
    sigSave = Signal()
    sigSetSingleChannel = Signal(bool)
    sigSelectObjects = Signal(bool)
    sigUpdate = Signal()
    
    def __init__(self, name='Volume Renderer Toolbar', parent=None):
        
        super().__init__(name, parent)
        
        self.parentWin = parent
        
        self.setContextMenuPolicy(Qt.PreventContextMenu)
        
        self.emitSigUpdateAction = QAction(
            QIcon(':reload.svg'), 'Emit update signal', self)
        self.emitSigUpdateAction.setToolTip(
            'Emit the `sigUpdate` signal.'
            'Click to tell the Cell-ACDC GUI to update the 3D viewer '
            'with the current data.'
        )
        self.addAction(self.emitSigUpdateAction)
        
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
        
        # self.selectObjectsAction = QAction(
        #     QIcon(':keep_objects.svg'), 'Select objects', self)
        # self.selectObjectsAction.setToolTip(
        #     'Select objects in the view.\n\n'
        #     'Ctrl+Click to select multiple objects.\n\n'
        #     'Press Esc to exit selection mode.'
        # )
        # self.selectObjectsAction.setCheckable(True)
        # self.addAction(self.selectObjectsAction)
        
        # self.addSeparator()
        
        self.singleChannelCheckbox = self.addCheckBox(
            text='Single channel'
        )
        
        self.singleChannelCheckbox.setToolTip(
            'When single channel mode is activated, selecting a channel '
            'will display only that channel in the overlay.'
        )
        
        self.emitSigUpdateAction.triggered.connect(self.sigUpdate.emit)
        self.homeViewAction.triggered.connect(self.sigHomeView.emit)
        self.saveAction.triggered.connect(self.sigSave.emit)
        # self.selectObjectsAction.toggled.connect(
        #     self.sigSelectObjects.emit
        # )
        
        self.singleChannelCheckbox.toggled.connect(
            self.sigSetSingleChannel.emit
        )
    
    def is_single_channel_mode(self) -> bool:
        return self.singleChannelCheckbox.isChecked()

class PointsLayersToolbar(ToolBar):    
    def __init__(self, name='Points Layer Toolbar', parent=None):
        super().__init__(name, parent)
        
        self.addLabel('Points: ')

class LabelsOverlay(QWidget):
    def __init__(self, renderer):
        super().__init__(renderer._canvas.native)

        self.renderer = renderer

        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.resize(renderer._canvas.native.size())

        self._font = QFont()
        self._font.setPointSize(10)
        self._font.setBold(True)

        self._text_color = QColor("white")
        self._outline_color = QColor("black")
        self._outline_width = 2

        self.show()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def setFontSize(self, size: int):
        self._font.setPointSize(size)
        self.update()

    def fontSize(self) -> int:
        return self._font.pointSize()

    def setBold(self, bold: bool):
        self._font.setBold(bold)
        self.update()

    def setTextColor(self, color):
        self._text_color = QColor(color)
        self.update()

    def setOutlineColor(self, color):
        self._outline_color = QColor(color)
        self.update()

    def setOutlineWidth(self, width: int):
        self._outline_width = width
        self.update()

    # -------------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setFont(self._font)

        metrics = painter.fontMetrics()

        for ann in self.renderer._label_annotations.values():
            if not ann.visible:
                continue

            x, y = ann.screen_xy
            text = ann.text

            rect = metrics.boundingRect(text)
            rect.moveCenter(QPoint(int(x), int(y)))

            path = QPainterPath()
            path.addText(rect.bottomLeft(), self._font, text)

            painter.setPen(QPen(self._outline_color, self._outline_width))
            painter.setBrush(self._text_color)
            painter.drawPath(path)