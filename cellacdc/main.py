# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPyTop HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

print('Importing modules')
import sys
import os
import subprocess
import re
import time
import traceback

import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QAction,
    QMenu, QMessageBox
)
from PyQt5.QtCore import (
    Qt, QProcess, pyqtSignal, pyqtSlot, QTimer, QSize,
    QSettings
)
from PyQt5.QtGui import QFontDatabase
from pyqtgraph.Qt import QtGui

import dataPrep, segm, gui, dataStruct
import utils.concat
import utils.convert
import help.welcome

import qrc_resources
# import pandas as pd

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

class mainWin(QMainWindow):
    def __init__(self, app, parent=None):
        self.app = app
        self.welcomeGuide = None
        super().__init__(parent)
        self.setWindowTitle("Cell-ACDC")
        self.setWindowIcon(QtGui.QIcon(":assign-motherbud.svg"))

        self.loadFonts()

        self.createActions()
        self.createMenuBar()
        self.connectActions()

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()
        mainLayout.addStretch()

        welcomeLabel = QLabel('Welcome to Cell-ACDC!')
        welcomeLabel.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setFamily('Ubuntu')
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
        font.setFamily('Ubuntu')
        label.setFont(font)
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 10px 0px;")
        mainLayout.addWidget(label)

        mainLayout.addStretch()

        dataStructButton = QPushButton(
            '0. Create data structure from microscopy file(s)...'
        )
        font = QtGui.QFont()
        font.setPointSize(11)
        dataStructButton.setFont(font)
        dataStructButton.clicked.connect(self.launchDataStruct)
        self.dataStructButton = dataStructButton
        mainLayout.addWidget(dataStructButton)

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

        closeButton = QPushButton(QtGui.QIcon(":exit.png"), '  Exit')
        self.closeButton = closeButton
        # closeButton.setIconSize(QSize(24,24))
        font = QtGui.QFont()
        font.setPointSize(10)
        closeButton.setFont(font)
        closeButton.clicked.connect(self.close)
        mainLayout.addWidget(closeButton)

        mainContainer.setLayout(mainLayout)

        self.start_JVM = True

        self.guiWin = None
        self.dataPrepWin = None

    def loadFonts(self):
        QFontDatabase.addApplicationFont(":Ubuntu-Regular.ttf")
        QFontDatabase.addApplicationFont(":Ubuntu-Bold.ttf")
        QFontDatabase.addApplicationFont(":Ubuntu-Italic.ttf")
        QFontDatabase.addApplicationFont(":Ubuntu-BoldItalic.ttf")
        QFontDatabase.addApplicationFont(":Calibri-Regular.ttf")
        QFontDatabase.addApplicationFont(":Calibri-Bold.ttf")
        QFontDatabase.addApplicationFont(":Calibri-Italic.ttf")
        QFontDatabase.addApplicationFont(":Calibri-BoldItalic.ttf")
        QFontDatabase.addApplicationFont(":ArialMT-Regular.ttf")
        QFontDatabase.addApplicationFont(":ArialMT-Bold.otf")
        QFontDatabase.addApplicationFont(":ArialMT-Italic.otf")
        QFontDatabase.addApplicationFont(":ArialMT-BoldItalic.otf")

    def launchWelcomeGuide(self, checked=False):
        cellacdc_path = os.path.dirname(os.path.realpath(__file__))
        temp_path = os.path.join(cellacdc_path, 'temp')
        csv_path = os.path.join(temp_path, 'settings.csv')
        self.settings_csv_path = csv_path
        if not os.path.exists(csv_path):
            idx = ['showWelcomeGuide']
            values = ['Yes']
            self.df_settings = pd.DataFrame({'setting': idx,
                                             'value': values}
                                           ).set_index('setting')
            self.df_settings.to_csv(csv_path)
        self.df_settings = pd.read_csv(csv_path, index_col='setting')
        if 'showWelcomeGuide' not in self.df_settings.index:
            self.df_settings.at['showWelcomeGuide', 'value'] = 'Yes'
            self.df_settings.to_csv(csv_path)

        show = (
            self.df_settings.at['showWelcomeGuide', 'value'] == 'Yes'
            or self.sender() is not None
        )
        if not show:
            return

        self.welcomeGuide = help.welcome.welcomeWin(mainWin=self)
        self.welcomeGuide.showAndSetSize()
        self.welcomeGuide.showPage(self.welcomeGuide.welcomeItem)

    def setColorsAndText(self):
        self.moduleLaunchedColor = '#ead935'
        defaultColor = self.guiButton.palette().button().color().name()
        self.defaultPushButtonColor = defaultColor
        self.defaultTextDataStructButton = self.dataStructButton.text()
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
        utilsMenu.addAction(self.TiffToNpzAction)
        menuBar.addMenu(utilsMenu)

        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.welcomeGuideAction)
        helpMenu.addAction(self.documentationAction)
        helpMenu.addAction(self.aboutAction)
        helpMenu.addAction(self.citeAction)
        helpMenu.addAction(self.contributeAction)

        menuBar.addMenu(helpMenu)

    def createActions(self):
        self.npzToNpyAction = QAction('Convert .npz file(s) to .npy...')
        self.npzToTiffAction = QAction('Convert .npz file(s) to .tif...')
        self.TiffToNpzAction = QAction('Convert .tif file(s) to _segm.npz...')
        self.concatAcdcDfsAction = QAction(
            'Concatenate acdc output tables from multiple Positions...'
        )
        self.alignAction = QAction('Revert alignemnt/Align...')

        self.welcomeGuideAction = QAction('Welcome Guide')
        self.documentationAction = QAction('Documentation')
        self.aboutAction = QAction('About Cell-ACDC')
        self.citeAction = QAction('Cite us...')
        self.contributeAction = QAction('Contribute...')

    def connectActions(self):
        self.concatAcdcDfsAction.triggered.connect(self.launchConcatUtil)
        self.npzToNpyAction.triggered.connect(self.launchConvertFormatUtil)
        self.npzToTiffAction.triggered.connect(self.launchConvertFormatUtil)
        self.TiffToNpzAction.triggered.connect(self.launchConvertFormatUtil)
        self.welcomeGuideAction.triggered.connect(self.launchWelcomeGuide)

    def launchConvertFormatUtil(self, checked=False):
        s = self.sender().text()
        m = re.findall('Convert \.(\w+) file\(s\) to (.*)\.(\w+)...', s)
        from_, info, to = m[0]
        isConvertEnabled = self.sender().isEnabled()
        if isConvertEnabled:
            self.sender().setDisabled(True)
            self.convertWin = utils.convert.convertFileFormatWin(
                parent=self,
                actionToEnable=self.sender(),
                mainWin=self, from_=from_, to=to
            )
            self.convertWin.show()
            self.convertWin.main()
        else:
            # self.convertWin.setWindowState(Qt.WindowNoState)
            self.convertWin.setWindowState(Qt.WindowActive)
            self.convertWin.raise_()

    def launchDataStruct(self, checked=False):
        c = self.dataStructButton.palette().button().color().name()
        launchedColor = self.moduleLaunchedColor
        defaultColor = self.defaultPushButtonColor
        defaultText = self.defaultTextDataStructButton

        try:
            dataStruct.createDataStructWin(parent=self)
        except OSError as e:
            print(f'WARNING: {e}')
            return

        is_win = sys.platform.startswith("win")

        if is_win:
            print('Launching data structure creation in a separate process...')

            self.dataStructButton.setStyleSheet(
                f'QPushButton {{background-color: {launchedColor};}}')
            self.dataStructButton.setText(
                'Launching in a separate process...')
            self.dataStructButton.setDisabled(True)

            cellacdc_path = os.path.dirname(os.path.realpath(__file__))
            dataStruct_path = os.path.join(cellacdc_path, 'dataStruct.py')

            # Due to javabridge limitation only one 'start_vm' can be called in
            # each process. To get around with this every data structure conversion
            # is launched in a separate process
            subprocess.Popen(
                [sys.executable, dataStruct_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                shell=True,
                bufsize=1,
                encoding='utf-8'
            )

            QTimer.singleShot(10000, self.processDataStructLaunched)
        else:
            self._showDataStructWin()


    def processDataStructLaunched(self):
        self.dataStructButton.setStyleSheet(
            f'QPushButton {{background-color: {self.defaultPushButtonColor};}}')
        self.dataStructButton.setText(
            '0. Create data structure from microscopy file(s)...')
        self.dataStructButton.setDisabled(False)

    def _showDataStructWin(self):
        if self.dataStructButton.isEnabled():
            self.dataStructButton.setText(
                '0. Restart Cell-ACDC to enable module 0 again.')
            self.dataStructButton.setToolTip(
                'Due to an interal limitation of the Java Virtual Machine\n'
                'moduel 0 can be launched only once.\n'
                'To use it again close and reopen Cell-ACDC'
            )
            self.dataStructButton.setDisabled(True)
            self.dataStructWin = dataStruct.createDataStructWin(parent=self)
            self.dataStructWin.show()
            self.dataStructWin.main()


    def launchDataPrep(self, checked=False):
        c = self.dataPrepButton.palette().button().color().name()
        launchedColor = self.moduleLaunchedColor
        defaultColor = self.defaultPushButtonColor
        defaultText = self.defaultTextDataPrepButton
        if c != self.moduleLaunchedColor:
            self.dataPrepButton.setStyleSheet(
                f'QPushButton {{background-color: {launchedColor};}}')
            self.dataPrepButton.setText('DataPrep is running. '
                                    'Click to restore window.')
            self.dataPrepWin = dataPrep.dataPrepWin(
                buttonToRestore=(self.dataPrepButton, defaultColor, defaultText),
                mainWin=self
            )
            self.dataPrepWin.show()
        else:
            # self.dataPrepWin.setWindowState(Qt.WindowNoState)
            self.dataPrepWin.setWindowState(Qt.WindowActive)
            self.dataPrepWin.raise_()

    def launchSegm(self, checked=False):
        c = self.segmButton.palette().button().color().name()
        launchedColor = self.moduleLaunchedColor
        defaultColor = self.defaultPushButtonColor
        defaultText = self.defaultTextSegmButton
        if c != self.moduleLaunchedColor:
            self.segmButton.setStyleSheet(
                f'QPushButton {{background-color: {launchedColor};}}')
            self.segmButton.setText('Segmentation is running. '
                                    'Check progress in the terminal/console')
            self.segmWin = segm.segmWin(
                buttonToRestore=(self.segmButton, defaultColor, defaultText),
                mainWin=self
            )
            self.segmWin.show()
            self.segmWin.main()
        else:
            # self.segmWin.setWindowState(Qt.WindowNoState)
            self.segmWin.setWindowState(Qt.WindowActive)
            self.segmWin.raise_()


    def launchGui(self, checked=False):
        c = self.guiButton.palette().button().color().name()
        launchedColor = self.moduleLaunchedColor
        defaultColor = self.defaultPushButtonColor
        defaultText = self.defaultTextGuiButton
        if c.lower() != launchedColor.lower():
            print('Opening GUI...')
            self.guiButton.setStyleSheet(
                f'QPushButton {{background-color: {launchedColor};}}')
            self.guiButton.setText('GUI is running. Click to restore window.')
            self.guiWin = gui.guiWin(
                self.app,
                buttonToRestore=(self.guiButton, defaultColor, defaultText),
                mainWin=self
            )
            self.guiWin.show()
        else:
            # self.guiWin.setWindowState(Qt.WindowNoState)
            self.guiWin.setWindowState(Qt.WindowActive)
            self.guiWin.raise_()

    def guiClosed(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.deleteGuiReference)
        self.timer.start(100)

    def deleteGuiReference(self):
        try:
            self.guiWin.isVisible()
        except RuntimeError:
            self.timer.stop()


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
            # self.concatWin.setWindowState(Qt.WindowNoState)
            self.concatWin.setWindowState(Qt.WindowActive)
            self.concatWin.raise_()


    def show(self):
        QMainWindow.show(self)
        h = self.dataPrepButton.geometry().height()
        self.dataStructButton.setMinimumHeight(h*2)
        self.dataPrepButton.setMinimumHeight(h*2)
        self.segmButton.setMinimumHeight(h*2)
        self.guiButton.setMinimumHeight(h*2)
        self.closeButton.setMinimumHeight(int(h*1.5))
        iconWidth = int(self.closeButton.iconSize().width()*1.3)
        self.closeButton.setIconSize(QSize(iconWidth, iconWidth))
        self.setColorsAndText()
        self.readSettings()

    def saveWindowGeometry(self):
        settings = QSettings('schmollerlab', 'acdc_main')
        settings.setValue("geometry", self.saveGeometry())

    def readSettings(self):
        settings = QSettings('schmollerlab', 'acdc_main')
        if settings.value('geometry') is not None:
            self.restoreGeometry(settings.value("geometry"))

    def checkOpenModules(self):
        c1 = self.dataPrepButton.palette().button().color().name()
        c2 = self.segmButton.palette().button().color().name()
        c3 = self.guiButton.palette().button().color().name()
        launchedColor = self.moduleLaunchedColor

        openModules = []
        if c1 == launchedColor:
            openModules.append(self.dataPrepWin)
        if c2 == launchedColor:
            openModules.append(self.segmWin)
        if c3 == launchedColor:
            openModules.append(self.guiWin)

        if not openModules:
            return True, openModules

        msg = QMessageBox()
        warn_txt = (
            'There are still other Cell-ACDC windows open.\n\n'
            'Are you sure you want to close everything?'
        )
        acceptCloseAnswer = msg.warning(
           self, 'Modules still open!', warn_txt, msg.Yes | msg.Cancel
        )

        return acceptCloseAnswer == msg.Yes, openModules

    def closeEvent(self, event):
        if self.welcomeGuide is not None:
            self.welcomeGuide.close()

        self.saveWindowGeometry()

        acceptClose, openModules = self.checkOpenModules()
        if acceptClose:
            for openModule in openModules:
                openModule.setWindowState(Qt.WindowActive)
                openModule.raise_()
                openModule.close()
                if openModule.isVisible():
                    event.ignore()
                    return
        else:
            event.ignore()
            return

        print('Cell-ACDC closed. Have a good day!')

def main():
    print('Launching application...')
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Create the application
    app = QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    app.setWindowIcon(QtGui.QIcon(":assign-motherbud.svg"))
    win = mainWin(app)
    win.show()
    win.launchWelcomeGuide()
    try:
        win.welcomeGuide.showPage(win.welcomeGuide.welcomeItem)
    except AttributeError:
        pass
    print('Done. If application is not visible, it is probably minimized '
          'or behind some other open window.')
    app.exec_()


if __name__ == "__main__":
    main()
