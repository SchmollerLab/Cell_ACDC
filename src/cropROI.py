import os
import sys

from PyQt5.QtCore import Qt, QFile, QTextStream, QSize
from PyQt5.QtGui import QIcon, QKeySequence, QCursor
from PyQt5.QtWidgets import (
    QAction, QApplication, QLabel, QPushButton,
    QMainWindow, QMenu, QToolBar, QGroupBox,
    QScrollBar, QCheckBox, QToolButton, QSpinBox,
    QComboBox, QDial, QButtonGroup
)

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg


if __name__ == "__main__":
    # Create the application
    app = QApplication(sys.argv)
    win = cropROI_GUI(app)
    win.show()
    # Apply style
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    # Run the event loop
    sys.exit(app.exec_())
