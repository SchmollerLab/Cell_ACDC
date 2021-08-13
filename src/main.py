import sys
import os
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QAction,
    QMenu
)
from PyQt5.QtCore import Qt, QProcess, pyqtSignal, pyqtSlot
from pyqtgraph.Qt import QtGui

import dataPrep, segm, gui
import utils.concat

import qrc_resources

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.yeastacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

class mainWin(QMainWindow):
    def __init__(self, app, parent=None):
        self.app = app
        super().__init__(parent)
        self.setWindowTitle("Yeast ACDC")
        self.setWindowIcon(QtGui.QIcon(":assign-motherbud.svg"))

        self.createActions()
        self.createMenuBar()
        self.connectActions()

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()

        welcomeLabel = QLabel(
            'Welcome to Yeast_ACDC!')
        welcomeLabel.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        welcomeLabel.setFont(font)
        # padding: top, left, bottom, right
        welcomeLabel.setStyleSheet("padding:0px 0px 5px 0px;")
        mainLayout.addWidget(welcomeLabel)

        label = QLabel(
            'Press any of the following buttons\n'
            'to launch the respective module')

        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(11)
        label.setFont(font)
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 10px 0px;")
        mainLayout.addWidget(label)

        dataPrepButton = QPushButton('1. Launch data prep module...')
        font = QtGui.QFont()
        font.setPointSize(11)
        dataPrepButton.setFont(font)
        dataPrepButton.clicked.connect(self.launchDataPrep)
        self.dataPrepButton = dataPrepButton
        mainLayout.addWidget(dataPrepButton)

        segmButton = QPushButton('2. Launch segmentation module...')
        segmButton.setFont(font)
        segmButton.clicked.connect(self.launchSegm)
        self.segmButton = segmButton
        mainLayout.addWidget(segmButton)

        guiButton = QPushButton('3. Launch GUI...')
        guiButton.setFont(font)
        guiButton.clicked.connect(self.launchGui)
        self.guiButton = guiButton
        mainLayout.addWidget(guiButton)

        closeButton = QPushButton('Exit')
        font = QtGui.QFont()
        font.setPointSize(10)
        closeButton.setFont(font)
        closeButton.clicked.connect(self.close)
        mainLayout.addWidget(closeButton)

        mainContainer.setLayout(mainLayout)

    def setColorsAndText(self):
        self.moduleLaunchedColor = '#ead935'
        defaultColor = self.guiButton.palette().button().color().name()
        self.defaultPushButtonColor = defaultColor
        self.defaultTextGuiButton = self.guiButton.text()
        self.defaultTextDataPrepButton = self.dataPrepButton.text()
        self.defaultTextSegmButton = self.segmButton.text()

    def createMenuBar(self):
        menuBar = self.menuBar()

        utilsMenu = QMenu("&Utilities", self)
        utilsMenu.addAction(self.concatAcdcDfsAction)
        utilsMenu.addAction(self.alignAction)
        utilsMenu.addAction(self.npzToNpyAction)
        utilsMenu.addAction(self.npzToTiffAction)
        menuBar.addMenu(utilsMenu)

        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.welcomeGuideAction)
        helpMenu.addAction(self.documentationAction)
        helpMenu.addAction(self.aboutAction)
        helpMenu.addAction(self.citeAction)
        helpMenu.addAction(self.contributeAction)

        menuBar.addMenu(helpMenu)

    def createActions(self):
        self.npzToNpyAction = QAction('Convert .npz file to .npy...')
        self.npzToTiffAction = QAction('Convert .npz file to .tif...')
        self.concatAcdcDfsAction = QAction(
            'Concatenate acdc output tables from multiple Positions...'
        )
        self.alignAction = QAction('Revert alignemnt/Align...')

        self.welcomeGuideAction = QAction('Welcome Guide')
        self.documentationAction = QAction('Documentation')
        self.aboutAction = QAction('About Yeast_ACDC')
        self.citeAction = QAction('Cite us...')
        self.contributeAction = QAction('Contribute...')

    def connectActions(self):
        self.concatAcdcDfsAction.triggered.connect(self.launchConcatUtil)

    def launchDataPrep(self, checked=False):
        c = self.dataPrepButton.palette().button().color().name()
        lauchedColor = self.moduleLaunchedColor
        defaultColor = self.defaultPushButtonColor
        defaultText = self.defaultTextDataPrepButton
        if c != self.moduleLaunchedColor:
            self.dataPrepButton.setStyleSheet(
                f'QPushButton {{background-color: {lauchedColor};}}')
            self.dataPrepButton.setText('DataPrep is running. '
                                    'Click to restore window.')
            self.dataPrepWin = dataPrep.dataPrepWin(
                buttonToRestore=(self.dataPrepButton, defaultColor, defaultText),
                mainWin=self
            )
            self.dataPrepWin.show()
        else:
            toFront = self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
            self.dataPrepWin.setWindowState(toFront)
            self.dataPrepWin.raise_()

    def launchSegm(self, checked=False):
        c = self.segmButton.palette().button().color().name()
        lauchedColor = self.moduleLaunchedColor
        defaultColor = self.defaultPushButtonColor
        defaultText = self.defaultTextSegmButton
        if c != self.moduleLaunchedColor:
            self.segmButton.setStyleSheet(
                f'QPushButton {{background-color: {lauchedColor};}}')
            self.segmButton.setText('Segmentation is running. '
                                    'Check progress in the terminal/console')
            self.segmWin = segm.segmWin(
                buttonToRestore=(self.segmButton, defaultColor, defaultText)
            )
            self.segmWin.show()
            self.segmWin.main()
        else:
            toFront = self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
            self.segmWin.setWindowState(toFront)
            self.segmWin.raise_()


    def launchGui(self, checked=False):
        c = self.guiButton.palette().button().color().name()
        lauchedColor = self.moduleLaunchedColor
        defaultColor = self.defaultPushButtonColor
        defaultText = self.defaultTextGuiButton
        if c.lower() != lauchedColor.lower():
            self.guiButton.setStyleSheet(
                f'QPushButton {{background-color: {lauchedColor};}}')
            self.guiButton.setText('GUI is running. Click to restore window.')
            self.guiWin = gui.guiWin(
                self.app,
                buttonToRestore=(self.guiButton, defaultColor, defaultText),
                mainWin=self
            )
            self.guiWin.showAndSetSize()
        else:
            toFront = self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
            self.guiWin.setWindowState(toFront)
            self.guiWin.raise_()

    def launchConcatUtil(self, checked=False):
        isConcatEnabled = self.concatAcdcDfsAction.isEnabled()
        if isConcatEnabled:
            self.concatAcdcDfsAction.setDisabled(True)
            self.concatWin = utils.concat.concatWin(
                parent=self,
                actionToEnable=self.concatAcdcDfsAction,
                mainWin=self
            )
            self.concatWin.show()
            self.concatWin.main()
        else:
            toFront = self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
            self.concatWin.setWindowState(toFront)
            self.concatWin.raise_()


    def showAndSetSettings(self):
        win.show()
        h = self.dataPrepButton.geometry().height()
        self.dataPrepButton.setMinimumHeight(h*2)
        self.segmButton.setMinimumHeight(h*2)
        self.guiButton.setMinimumHeight(h*2)
        self.setColorsAndText()

if __name__ == "__main__":
    print('Launching application...')
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Create the application
    app = QApplication(sys.argv)
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    win = mainWin(app)
    win.showAndSetSettings()
    print('Done. If application is not visible, it is probably minimized '
          'or behind some other open window.')
    sys.exit(app.exec_())
