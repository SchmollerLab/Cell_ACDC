import re

from qtpy.QtCore import (
    Signal, Qt, QPoint, QPointF, QRectF, QEvent
)
from qtpy.QtWidgets import (
    QAction, QWidget, QGridLayout, QLabel
)
from qtpy.QtGui import (
    QIcon, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
)

import pyqtgraph as pg

from .. import html_utils

from ..widgets import (
    ToolBar, DoubleSpinBox, playPushButton
)

LABELS_TEXT_FONTSIZE = 10

class VolumeRendererToolbar(ToolBar):
    sigHomeView = Signal()
    sigTwoDView = Signal()
    sigSave = Signal()
    sigSetSingleChannel = Signal(bool)
    sigSetFullResolution = Signal(bool)
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

        self.twoDViewAction = QAction(QIcon(':2d.svg'), '2D view', self)
        self.twoDViewAction.setShortcut('D')
        self.twoDViewAction.setToolTip(
            'Set the view to 2D (view from top)'
        )
        self.addAction(self.twoDViewAction)
        
        self.saveAction = QAction(QIcon(':file-save.svg'), 'Save', self)
        self.saveAction.setShortcut('Ctrl+S')
        self.saveAction.setToolTip(
            'Save the current view to PNG file'
        )
        self.addAction(self.saveAction)
        
        self.addSeparator()

        # self.setupAnimationAction = self.addButton(
        #     ':cog_play.svg', 'Setup animation', checkable=True
        # )
        # self.playAnimationAction = self.addButton(
        #     ':play.svg', 'Play and record animation', checkable=True
        # )
        # self.playAnimationAction.setDisabled(True)

        # self.setupAnimationAction.toggled.connect(
        #     self.sigSetupAnimation.emit
        # )

        # self.playAnimationAction.toggled.connect(
        #     self.sigPlayAnimation.emit
        # )

        # self.addSeparator()
        
        self.singleChannelCheckbox = self.addCheckBox(
            text='Single channel'
        )
        
        self.singleChannelCheckbox.setToolTip(
            'When single channel mode is activated, selecting a channel '
            'will display only that channel in the overlay.'
        )

        self.fullResolutionCheckbox = self.addCheckBox(
            text='Full resolution'
        )

        self.fullResolutionCheckbox.setToolTip(
            'When full resolution is activated, the volumes are displayed '
            'with all their voxels (limited only by the maximum 3D texture '
            'size of the GPU).\n\n'
            'When deactivated, the volumes are downsampled to maximum 512 '
            'voxels per dimension (faster, and less memory used on the GPU).'
        )

        self.emitSigUpdateAction.triggered.connect(self.sigUpdate.emit)
        self.homeViewAction.triggered.connect(self.sigHomeView.emit)
        self.twoDViewAction.triggered.connect(self.sigTwoDView.emit)
        self.saveAction.triggered.connect(self.sigSave.emit)
        # self.selectObjectsAction.toggled.connect(
        #     self.sigSelectObjects.emit
        # )
        
        self.singleChannelCheckbox.toggled.connect(
            self.sigSetSingleChannel.emit
        )

        self.fullResolutionCheckbox.toggled.connect(
            self.sigSetFullResolution.emit
        )

    def is_single_channel_mode(self) -> bool:
        return self.singleChannelCheckbox.isChecked()

    def is_full_resolution(self) -> bool:
        return self.fullResolutionCheckbox.isChecked()

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
        self.setVisible(visible)

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

class ParentCursorOverlay(QWidget):
    """Transparent overlay on top of the vispy canvas that paints the
    mirrored cursor of the parent GUI as a shaded 3D shape (projected faces
    computed by the renderer, see `setFaces`).

    Moving the cursor repaints only the small region of this widget around
    the cursor, i.e., the volumes do not need to be re-rendered.
    """
    def __init__(self, canvas_widget: QWidget):
        super().__init__(canvas_widget)

        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Follow the size of the canvas
        self.resize(canvas_widget.size())
        canvas_widget.installEventFilter(self)

        # Thin dark halo for contrast on bright voxels, then light edges
        self._halo_pen = QPen(QColor(0, 0, 0, 150))
        self._halo_pen.setWidthF(3.0)
        self._halo_pen.setJoinStyle(Qt.RoundJoin)
        self._edge_pen = QPen(QColor(255, 255, 255, 200))
        self._edge_pen.setWidthF(0.8)
        self._edge_pen.setJoinStyle(Qt.RoundJoin)

        self._polygons: list[tuple[QPolygonF, QColor]] = []
        self._rect = QRectF()

        self.show()

    def eventFilter(self, watched, event):
        if watched is self.parent() and event.type() == QEvent.Resize:
            self.resize(watched.size())
        return False

    def setFaces(self, faces, opacity: float=1.0):
        """Set the faces to paint.

        Parameters
        ----------
        faces : list of (np.ndarray, float)
            (N, 2) canvas coordinates of the vertices of each face and its
            shade in [0, 1] (0 = dark, 1 = fully lit). Pass an empty list
            to hide the cursor.
        opacity : float, optional
            Opacity of the whole cursor. Default is 1.0
        """
        polygons = []
        rect = QRectF()
        for points, shade in faces:
            polygon = QPolygonF([QPointF(x, y) for x, y in points])
            gray = int(70 + 185*shade)
            alpha = int((170 + 70*shade) * opacity)
            polygons.append((polygon, QColor(gray, gray, gray, alpha)))
            rect = rect.united(polygon.boundingRect())

        # Repaint only the old and new regions of the cursor
        dirty_rect = self._rect.united(rect)
        self._polygons = polygons
        self._rect = rect
        if not dirty_rect.isEmpty():
            self.update(dirty_rect.toAlignedRect().adjusted(-3, -3, 3, 3))

    def paint(self, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setPen(self._halo_pen)
        painter.setBrush(Qt.NoBrush)
        for polygon, _ in self._polygons:
            painter.drawPolygon(polygon)

        painter.setPen(self._edge_pen)
        for polygon, color in self._polygons:
            painter.setBrush(color)
            painter.drawPolygon(polygon)

    def paintEvent(self, event):
        if not self._polygons:
            return

        painter = QPainter(self)
        self.paint(painter)
        painter.end()