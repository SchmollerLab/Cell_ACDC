import re

from qtpy.QtCore import (
    Signal, Qt, QPoint
)
from qtpy.QtWidgets import (
    QAction, QWidget, QGridLayout, QLabel
)
from qtpy.QtGui import (
    QIcon, QColor, QFont, QPainter, QPainterPath, QPen
)

import pyqtgraph as pg

from .. import html_utils

from ..widgets import (
    ToolBar, DoubleSpinBox, playPushButton
)

LABELS_TEXT_FONTSIZE = 10

class VolumeRendererToolbar(ToolBar):
    sigHomeView = Signal()
    sigSave = Signal()
    sigSetSingleChannel = Signal(bool)
    sigSelectObjects = Signal(bool)
    sigUpdate = Signal()
    sigSetupAnimation = Signal(bool)
    sigPlayAnimation = Signal(bool)
    
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

        self.setupAnimationAction = self.addButton(
            ':cog_play.svg', 'Setup animation', checkable=True
        )
        self.playAnimationAction = self.addButton(
            ':play.svg', 'Play and record animation', checkable=True
        )
        self.playAnimationAction.setDisabled(True)

        self.setupAnimationAction.toggled.connect(
            self.sigSetupAnimation.emit
        )

        self.playAnimationAction.toggled.connect(
            self.sigPlayAnimation.emit
        )

        self.addSeparator()
        
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

class AnimationParamWidget(QWidget):
    def __init__(self, speed_unit: str, parent=None):
        super().__init__(parent)

        layout = QGridLayout()

        self.startDoubleSpinbox = DoubleSpinBox()
        startLabel = QLabel('Start')
        self.currentLabel = QLabel(
            html_utils.span('<i>Current: 0.0</i>', font_size='11px', color=None)
        )

        self.stopDoubleSpinbox = DoubleSpinBox()
        stopLabel = QLabel('Stop')

        self.speedDoubleSpinbox = DoubleSpinBox()
        speedLabel = QLabel(f'Speed [{speed_unit}]')

        self.playButton = playPushButton()

        col = 0
        layout.addWidget(startLabel, 0, col, alignment=Qt.AlignLeft)
        layout.addWidget(self.startDoubleSpinbox, 1, col)
        layout.addWidget(self.currentLabel, 2, col, alignment=Qt.AlignLeft)

        col = 1
        layout.addWidget(stopLabel, 0, col, alignment=Qt.AlignLeft)
        layout.addWidget(self.stopDoubleSpinbox, 1, col)

        col = 2
        layout.addWidget(speedLabel, 0, col, alignment=Qt.AlignLeft)
        layout.addWidget(self.speedDoubleSpinbox, 1, col)

        col = 3
        layout.addWidget(self.playButton, 1, col)
        layout.setColumnStretch(col, 0)

        layout.setContentsMargins(0, 5, 0, 5)

        self.setLayout(layout)
    
    def setCurrent(self, value: float, update_start=True):
        text = re.sub(
            r'Current: [0-9]+\.[0-9]+',
            f'Current: {value}',
            self.currentLabel.text(),
        )
        self.currentLabel.setText(text)
        if not update_start:
            return

        self.startDoubleSpinbox.setValue(value)
    
    def params(self):
        start = self.startDoubleSpinbox.value()
        stop = self.stopDoubleSpinbox.value()
        speed = self.speedDoubleSpinbox.value()
        params = {
            'start': start, 'stop': stop, 'speed': speed
        }
        return params


class LabelsOverlay(QWidget):
    def __init__(self, renderer, font_size=None, text_color='white'):
        super().__init__(renderer._canvas.native)

        if font_size is None:
            font_size = LABELS_TEXT_FONTSIZE

        self.renderer = renderer

        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.resize(renderer._canvas.native.size())

        self._font = QFont()
        self._font.setPointSize(font_size)
        self._font.setBold(False)

        self._text_color = pg.mkColor(text_color)

        # Shadow settings
        self._shadow_color = QColor(0, 0, 0, 180)
        self._shadow_offset = QPoint(1, 1)

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

    def setShadowColor(self, color):
        self._shadow_color = QColor(color)
        self.update()

    def setShadowOffset(self, dx: int, dy: int):
        self._shadow_offset = QPoint(dx, dy)
        self.update()

    def setAnnotationsVisible(self, visible: bool):
        for ann in self.renderer._label_annotations.values():
            ann.visible = visible

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

            # Shadow
            painter.setPen(self._shadow_color)
            painter.drawText(
                rect.translated(self._shadow_offset),
                Qt.AlignCenter,
                text,
            )

            # Foreground
            painter.setPen(self._text_color)
            painter.drawText(
                rect,
                Qt.AlignCenter,
                text,
            )