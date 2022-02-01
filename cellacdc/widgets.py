import sys
import time
import re

from PyQt5.QtCore import (
    pyqtSignal, QTimer, Qt, QPoint, pyqtSlot, pyqtProperty,
    QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup,
    QSize, QRectF, QPointF, QRect, QPoint, QEasingCurve, QRegExp,
    QEvent
)
from PyQt5.QtGui import (
    QFont, QPalette, QColor, QPen, QPaintEvent, QBrush, QPainter,
    QRegExpValidator, QIcon
)
from PyQt5.QtWidgets import (
    QTextEdit, QLabel, QProgressBar, QHBoxLayout, QToolButton, QCheckBox,
    QApplication, QWidget, QVBoxLayout, QMainWindow, QStyleFactory,
    QLineEdit, QSlider, QSpinBox, QGridLayout, QDockWidget,
    QScrollArea, QSizePolicy, QComboBox, QPushButton, QScrollBar,
    QGroupBox, QAbstractSlider, QDoubleSpinBox
)

from . import myutils, apps

class QLogConsole(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        font = QFont()
        font.setPointSize(9)
        self.setFont(font)

    def write(self, message):
        # Method required by tqdm pbar
        message = message.replace('\r ', '')
        if message:
            self.apppendText(message)


class QProgressBarWithETA(QProgressBar):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)

        palette = QPalette()
        palette.setColor(QPalette.Highlight, QColor(207, 235, 155))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)
        self.ETA_label = QLabel('NDh:NDm:NDs')
        self.last_time_update = time.perf_counter()

    def update(self, step):
        t = time.perf_counter()
        self.setValue(self.value()+step)
        elpased_seconds = (t - self.last_time_update)/step
        steps_left = self.maximum() - self.value()
        seconds_left = elpased_seconds*steps_left
        ETA = myutils.seconds_to_ETA(seconds_left)
        self.ETA_label.setText(ETA)
        self.last_time_update = t
        return ETA

    def show(self):
        QProgressBar.show(self)
        self.ETA_label.show()

    def hide(self):
        QProgressBar.hide(self)
        self.ETA_label.hide()

class sliderWithSpinBox(QWidget):
    sigValueChange = pyqtSignal(object)
    valueChanged = pyqtSignal(object)
    editingFinished = pyqtSignal()

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args)


        layout = QGridLayout()

        title = kwargs.get('title')
        row = 0
        col = 0
        if title is not None:
            titleLabel = QLabel(self)
            titleLabel.setText(title)
            loc = kwargs.get('title_loc')
            loc = loc if loc is not None else 'top'
            if loc == 'top':
                layout.addWidget(titleLabel, 1, col, alignment=Qt.AlignLeft)
            elif loc=='in_line':
                row = -1
                col = 1
                layout.addWidget(titleLabel, 0, 0, alignment=Qt.AlignLeft)
                layout.setColumnStretch(0, 0)

        self._normalize = False
        normalize = kwargs.get('normalize')
        if normalize is not None:
            self._normalize = True

        self._isFloat = False
        isFloat = kwargs.get('isFloat')
        if isFloat is not None:
            self._isFloat = True

        self.slider = QSlider(Qt.Horizontal, self)
        layout.addWidget(self.slider, row+1, col)

        if self._normalize or self._isFloat:
            self.spinBox = QDoubleSpinBox(self)
        else:
            self.spinBox = QSpinBox(self)
        self.spinBox.setAlignment(Qt.AlignCenter)
        self.spinBox.setMaximum(2**31-1)
        layout.addWidget(self.spinBox, row+1, col+1)
        if title is not None:
            layout.setRowStretch(0, 1)
        layout.setRowStretch(row+1, 1)
        layout.setColumnStretch(col, 6)
        layout.setColumnStretch(col+1, 1)

        self.layout = layout

        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.slider.sliderReleased.connect(self.onEditingFinished)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.spinBox.editingFinished.connect(self.onEditingFinished)
        self.setLayout(layout)

    def onEditingFinished(self):
        self.editingFinished.emit()

    def setValue(self, value):
        valueInt = value
        if self._normalize:
            valueInt = int(value*self.slider.maximum())
        if self._isFloat:
            valueInt = int(value)
            self.spinBox.valueChanged.disconnect()
            self.spinBox.setValue(value)
            self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.slider.setValue(valueInt)

    def setMaximum(self, max):
        self.slider.setMaximum(max)
        # self.spinBox.setMaximum(max)

    def setMinimum(self, min):
        self.slider.setMinimum(min)
        # self.spinBox.setMinimum(min)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setDecimals(self, decimals):
        self.spinBox.setDecimals(decimals)

    def sliderValueChanged(self, val):
        self.spinBox.valueChanged.disconnect()
        if self._normalize:
            valF = val/self.slider.maximum()
            self.spinBox.setValue(valF)
        else:
            self.spinBox.setValue(val)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.sigValueChange.emit(self.value())
        self.valueChanged.emit(self.value())

    def spinboxValueChanged(self, val):
        self.slider.valueChanged.disconnect()
        if self._normalize:
            val = int(val*self.slider.maximum())
        if self._isFloat:
            val = int(val)
        self.slider.setValue(val)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.sigValueChange.emit(self.value())
        self.valueChanged.emit(self.value())

    def value(self):
        return self.spinBox.value()

if __name__ == '__main__':
    class Window(QMainWindow):
        def __init__(self):
            super().__init__()

            container = QWidget()
            layout = QVBoxLayout()

            # slider = sliderWithSpinBox(isFloat=True)
            # slider.setMaximum(10)
            # slider.setValue(3.2)
            # slider.setSingleStep(0.1)
            # slider.valueChanged.connect(self.sliderValueChanged)
            # slider.slider.sliderReleased.connect(self.sliderReleased)
            # layout.addWidget(slider)
            # self.slider = slider

            okButton = QPushButton('ok')
            layout.addWidget(okButton)
            okButton.clicked.connect(self.okClicked)

            # layout.addStretch(1)
            container.setLayout(layout)
            self.setCentralWidget(container)

            self.setFocus()

        def okClicked(self, checked):
            editID = apps.editID_QWidget(19, [19, 100, 50])
            editID.exec_()
            print('closed')

        def sliderValueChanged(self, value):
            print(value)

        def sliderReleased(self, value):
            print('released')

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_T:
                screens = app.screens()
                current_screen = self.screen()
                num_screens = len(screens)
                if num_screens > 1:
                    other_screen = None
                    for screen in screens:
                        if screen != current_screen:
                            other_screen = screen
                            break
                    print(f'Current screen geometry = {current_screen.geometry()}')
                    print(f'Other screen geometry = {other_screen.geometry()}')
            elif event.key() == Qt.Key_P:
                print(self.slider.value())



    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))

    w = Window()
    w.show()
    app.exec_()
