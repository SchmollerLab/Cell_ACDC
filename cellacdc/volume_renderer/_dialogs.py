from qtpy.QtCore import (
    Qt, Signal
)

from qtpy.QtWidgets import (
    QVBoxLayout, QGridLayout, QLabel, QComboBox
)

from .. import widgets

from .._base_widgets import QBaseDialog

from ._widgets import AnimationParamWidget


class AnimationVolumeViewerSetupDialog(QBaseDialog):
    sigOk = Signal()
    sigCancel = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cancel = True

        self._ignore_close = True

        self.setWindowTitle('Setup animation parameters')

        mainLayout = QVBoxLayout()

        paramsLayout = QGridLayout()

        numParams = 6
        paramOrderItems = list(map(str, range(1, numParams+1)))

        row = 0
        self.elevationOrderCombobox = QComboBox()
        self.elevationOrderCombobox.addItems(paramOrderItems)
        self.elevationParamWidget = AnimationParamWidget()
        paramsLayout.addWidget(self.elevationOrderCombobox, row, 0)
        paramsLayout.addWidget(
            QLabel('Elevation angle '), row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.elevationParamWidget, row, 2)
        paramsLayout.addWidget(widgets.QHLine(), row+1, 0, 1, 3)

        row = 2
        self.azimuthOrderCombobox = QComboBox()
        self.azimuthOrderCombobox.addItems(paramOrderItems)
        self.azimuthParamWidget = AnimationParamWidget()
        paramsLayout.addWidget(self.azimuthOrderCombobox, row, 0)
        paramsLayout.addWidget(
            QLabel('Azimuth angle '), row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.azimuthParamWidget, row, 2)
        paramsLayout.addWidget(widgets.QHLine(), row+1, 0, 1, 3)

        row = 4
        self.xPosOrderCombobox = QComboBox()
        self.xPosOrderCombobox.addItems(paramOrderItems)
        self.xPosParamWidget = AnimationParamWidget()
        paramsLayout.addWidget(self.xPosOrderCombobox, row, 0)
        paramsLayout.addWidget(
            QLabel('X Position '), row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.xPosParamWidget, row, 2)

        row = 5
        self.yPosOrderCombobox = QComboBox()
        self.yPosOrderCombobox.addItems(paramOrderItems)
        self.yPosParamWidget = AnimationParamWidget()
        paramsLayout.addWidget(self.yPosOrderCombobox, row, 0)
        paramsLayout.addWidget(
            QLabel('Y Position '), row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.yPosParamWidget, row, 2)

        row = 6
        self.zPosOrderCombobox = QComboBox()
        self.zPosOrderCombobox.addItems(paramOrderItems)
        self.zPosParamWidget = AnimationParamWidget()
        paramsLayout.addWidget(self.zPosOrderCombobox, row, 0)
        paramsLayout.addWidget(
            QLabel('Z Position '), row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.zPosParamWidget, row, 2)
        paramsLayout.addWidget(widgets.QHLine(), row+1, 0, 1, 3)

        row = 8
        self.distanceOrderCombobox = QComboBox()
        self.distanceOrderCombobox.addItems(paramOrderItems)
        self.distanceParamWidget = AnimationParamWidget()
        paramsLayout.addWidget(self.distanceOrderCombobox, row, 0)
        paramsLayout.addWidget(
            QLabel('Distance (zoom) '), row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.distanceParamWidget, row, 2)
        paramsLayout.addWidget(widgets.QHLine(), row+1, 0, 1, 3)

        paramsLayout.setColumnStretch(0, 0)

        playAllButton = widgets.playPushButton('Play all')
        buttonsLayout = widgets.CancelOkButtonsLayout(
            additionalButtons=(playAllButton,)
        )

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.cancel_cb)

        mainLayout.addLayout(paramsLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
    
    def ok_cb(self, *args):
        self.cancel = False
        self.sigOk.emit()
    
    def cancel_cb(self, *args):
        self.sigCancel.emit()
    
    def closeEvent(self, event):
        if self._ignore_close:
            event.ignore()
        self.sigCancel.emit()
    
    def forceClose(self):
        self._ignore_close = False
        self.close()

    def set_from_camera(self, camera, update_start=True):
        self.elevationParamWidget.setCurrent(
            camera.elevation, update_start=update_start
        )
        self.azimuthParamWidget.setCurrent(
            camera.azimuth, update_start=update_start
        )
        self.xPosParamWidget.setCurrent(
            round(camera.center[0], 2), update_start=update_start
        )
        self.yPosParamWidget.setCurrent(
            round(camera.center[1], 2), update_start=update_start
        )
        self.zPosParamWidget.setCurrent(
            round(camera.center[2], 2), update_start=update_start
        )
        self.distanceParamWidget.setCurrent(
            round(camera.distance, 2), update_start=update_start
        )