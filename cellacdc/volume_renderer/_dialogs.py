from qtpy.QtCore import (
    Qt, Signal
)

from qtpy.QtWidgets import (
    QVBoxLayout, QGridLayout, QLabel, QComboBox
)

from .. import widgets, printl

from .._base_widgets import QBaseDialog

from ._widgets import AnimationParamWidget


class AnimationVolumeViewerSetupDialog(QBaseDialog):
    sigOk = Signal(object)
    sigCancel = Signal()

    def __init__(self, hide_on_close=True, parent=None):
        super().__init__(parent)

        self.cancel = True

        self._ignore_close = hide_on_close

        self.setWindowTitle('Setup animation parameters')

        mainLayout = QVBoxLayout()

        paramsLayout = QGridLayout()
        self.paramsLayout = paramsLayout

        numParams = 6
        paramOrderItems = list(map(str, range(1, numParams+1)))

        self._widgets = []

        row = 0
        label = QLabel('Elevation angle ')
        hline = widgets.QHLine()
        self.elevationOrderCombobox = QComboBox()
        self.elevationOrderCombobox.addItems(paramOrderItems)
        self.elevationOrderCombobox.setCurrentText('1')
        self.elevationParamWidget = AnimationParamWidget('deg/s')
        paramsLayout.addWidget(self.elevationOrderCombobox, row, 0)
        paramsLayout.addWidget(
            label, row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.elevationParamWidget, row, 2)
        paramsLayout.addWidget(hline, row+1, 0, 1, 3)
        self._widgets.append([
            self.elevationOrderCombobox, 
            label, 
            self.elevationParamWidget,
            hline
        ])

        row = 2
        label = QLabel('Azimuth angle ')
        hline = widgets.QHLine()
        self.azimuthOrderCombobox = QComboBox()
        self.azimuthOrderCombobox.addItems(paramOrderItems)
        self.azimuthOrderCombobox.setCurrentText('2')
        self.azimuthParamWidget = AnimationParamWidget('deg/s')
        paramsLayout.addWidget(self.azimuthOrderCombobox, row, 0)
        paramsLayout.addWidget(
            label, row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.azimuthParamWidget, row, 2)
        paramsLayout.addWidget(hline, row+1, 0, 1, 3)
        self._widgets.append([
            self.azimuthOrderCombobox, 
            label, 
            self.azimuthParamWidget,
            hline
        ])

        row = 4
        label = QLabel('X Position ')
        self.xPosOrderCombobox = QComboBox()
        self.xPosOrderCombobox.addItems(paramOrderItems)
        self.xPosOrderCombobox.setCurrentText('3')
        self.xPosParamWidget = AnimationParamWidget('1/s')
        paramsLayout.addWidget(self.xPosOrderCombobox, row, 0)
        paramsLayout.addWidget(
            label, row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.xPosParamWidget, row, 2)
        self._widgets.append([
            self.xPosOrderCombobox, 
            label, 
            self.xPosParamWidget,
            None
        ])

        row = 5
        label = QLabel('Y Position ')
        self.yPosOrderCombobox = QComboBox()
        self.yPosOrderCombobox.addItems(paramOrderItems)
        self.yPosOrderCombobox.setCurrentText('4')
        self.yPosParamWidget = AnimationParamWidget('1/s')
        paramsLayout.addWidget(self.yPosOrderCombobox, row, 0)
        paramsLayout.addWidget(
            label, row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.yPosParamWidget, row, 2)
        self._widgets.append([
            self.yPosOrderCombobox, 
            label, 
            self.yPosParamWidget,
            None
        ])

        row = 6
        label = QLabel('Z Position ')
        hline = widgets.QHLine()
        self.zPosOrderCombobox = QComboBox()
        self.zPosOrderCombobox.addItems(paramOrderItems)
        self.zPosOrderCombobox.setCurrentText('5')
        self.zPosParamWidget = AnimationParamWidget('1/6')
        paramsLayout.addWidget(self.zPosOrderCombobox, row, 0)
        paramsLayout.addWidget(
            label, row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.zPosParamWidget, row, 2)
        paramsLayout.addWidget(hline, row+1, 0, 1, 3)
        self._widgets.append([
            self.zPosOrderCombobox, 
            label, 
            self.zPosParamWidget,
            hline,
        ])

        row = 8
        label = QLabel('Distance (zoom) ')
        hline = widgets.QHLine()
        self.distanceOrderCombobox = QComboBox()
        self.distanceOrderCombobox.addItems(paramOrderItems)
        self.distanceOrderCombobox.setCurrentText('6')
        self.distanceParamWidget = AnimationParamWidget('1/6')
        paramsLayout.addWidget(self.distanceOrderCombobox, row, 0)
        paramsLayout.addWidget(
            label, row, 1, alignment=Qt.AlignLeft
        )
        paramsLayout.addWidget(self.distanceParamWidget, row, 2)
        paramsLayout.addWidget(hline, row+1, 0, 1, 3)
        self._widgets.append([
            self.distanceOrderCombobox, 
            label, 
            self.distanceParamWidget,
            hline,
        ])

        paramsLayout.setColumnStretch(0, 0)

        playAllButton = widgets.playPushButton('Play all')
        buttonsLayout = widgets.CancelOkButtonsLayout(
            additionalButtons=(playAllButton,)
        )

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.cancel_cb)

        self.elevationOrderCombobox.currentTextChanged.connect(self.changeOrder)
        self.azimuthOrderCombobox.currentTextChanged.connect(self.changeOrder)
        self.xPosOrderCombobox.currentTextChanged.connect(self.changeOrder)
        self.yPosOrderCombobox.currentTextChanged.connect(self.changeOrder)
        self.zPosOrderCombobox.currentTextChanged.connect(self.changeOrder)
        self.distanceOrderCombobox.currentTextChanged.connect(self.changeOrder)

        mainLayout.addLayout(paramsLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.mainLayout = mainLayout

        self.setLayout(mainLayout)
    
    def changeOrder(self, order_number_text):
        sender = self.sender()

        # Find the current order of the sender.
        current_order = next(
            i + 1
            for i, widgets in enumerate(self._widgets)
            if widgets[0] is sender
        )

        new_order = int(order_number_text)

        if new_order == current_order:
            return

        # Update the other combobox to avoid duplicate order numbers.
        for widgets in self._widgets:
            combobox = widgets[0]

            if combobox is sender:
                continue

            if combobox.currentText() == order_number_text:
                combobox.blockSignals(True)
                combobox.setCurrentText(str(current_order))
                combobox.blockSignals(False)
                break

        # Reorder the Python list.
        widgets = self._widgets.pop(current_order - 1)
        self._widgets.insert(new_order - 1, widgets)

        # Update all comboboxes to reflect the new order.
        for i, widgets in enumerate(self._widgets):
            combobox = widgets[0]
            combobox.blockSignals(True)
            combobox.setCurrentText(str(i + 1))
            combobox.blockSignals(False)
        
        self._add_params_widgets()

    def _add_params_widgets(self):
        paramsLayout = self.paramsLayout

        # Remove all existing items.
        while paramsLayout.count():
            item = paramsLayout.takeAt(0)

            # Do not delete widgets: we will reuse them.
            widget = item.widget()
            if widget is not None:
                widget.hide()

        # Explicitly rebuild the grid.
        row = 0

        for widgets in self._widgets:
            cb, label, pw, hline = widgets

            paramsLayout.addWidget(cb, row, 0)
            paramsLayout.addWidget(
                label, row, 1, alignment=Qt.AlignLeft
            )
            paramsLayout.addWidget(pw, row, 2)

            row += 1

            if hline is not None:
                paramsLayout.addWidget(hline, row, 0, 1, 3)
                row += 1

        # Force recalculation.
        paramsLayout.invalidate()
        paramsLayout.activate()

        mainLayout = self.mainLayout
        mainLayout.invalidate()
        mainLayout.activate()

        # Show widgets after the layout has been rebuilt.
        for widgets in self._widgets:
            for widget in widgets:
                if widget is not None:
                    widget.show()


    def params(self):
        params = {
            'elevation': self.elevationParamWidget.params(),
            'azimuth': self.azimuthParamWidget.params(),
            'x': self.xPosParamWidget.params(),
            'y': self.yPosParamWidget.params(),
            'z': self.zPosParamWidget.params(),
            'distance': self.distanceParamWidget.params(),
        }
        return params

    def ok_cb(self, *args):
        self.cancel = False
        self.sigOk.emit(self.params())
        if self._ignore_close:
            return
        
        self.close()
    
    def cancel_cb(self, *args):
        self.sigCancel.emit()
        if self._ignore_close:
            return
        
        self.close()
    
    def closeEvent(self, event):
        self.sigCancel.emit()
        if self._ignore_close:
            event.ignore()
            return
        
        super().closeEvent(event)
    
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