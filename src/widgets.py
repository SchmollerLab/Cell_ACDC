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
    QGroupBox, QAbstractSlider
)

class sliderWithSpinBox(QWidget):
    sigValueChange = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args)

        layout = QGridLayout()

        title = kwargs.get('title')
        row = 1
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

        self.slider = QSlider(Qt.Horizontal, self)
        layout.addWidget(self.slider, row+1, col)

        self.spinBox = QSpinBox(self)
        self.spinBox.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.spinBox, row+1, col+1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(row+1, 1)
        layout.setColumnStretch(col, 6)
        layout.setColumnStretch(col+1, 1)

        self.layout = layout

        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.setLayout(layout)

    def setValue(self, value):
        self.slider.setValue(value)

    def setMaximum(self, max):
        self.slider.setMaximum(max)
        self.spinBox.setMaximum(max)

    def setMinimum(self, min):
        self.slider.setMinimum(min)
        self.spinBox.setMinimum(min)

    def sliderValueChanged(self, val):
        self.spinBox.valueChanged.disconnect()
        self.spinBox.setValue(val)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.sigValueChange.emit(val)

    def spinboxValueChanged(self, val):
        self.slider.valueChanged.disconnect()
        self.slider.setValue(val)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.sigValueChange.emit(val)

    def value(self):
        return self.slider.value()

if __name__ == '__main__':
    class Window(QMainWindow):
        def __init__(self):
            super().__init__()

            container = QWidget()
            layout = QVBoxLayout()

            slider = sliderWithSpinBox(title='test slider', title_loc='in_line')
            layout.addWidget(slider)

            # layout.addStretch(1)
            container.setLayout(layout)
            self.setCentralWidget(container)

            self.setFocus()

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



    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))

    w = Window()
    w.show()
    app.exec_()
