import sys
import os
import re
import traceback
import time
import datetime
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QVBoxLayout, QPushButton, QLabel, QStyleFactory,
    QWidget, QMessageBox
)
from PyQt5.QtCore import Qt, QEventLoop
from PyQt5 import QtGui

import qrc_resources

import javabridge
import bioformats

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.yeastacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass

class createDataStructWin(QMainWindow):
    def __init__(self, parent=None, allowExit=False,
                 buttonToRestore=None, mainWin=None):
        self.allowExit = allowExit
        self.processFinished = False
        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin
        super().__init__(parent)
        self.setWindowTitle("Yeast ACDC - From raw microscopy file to tifs")
        self.setWindowIcon(QtGui.QIcon(":assign-motherbud.svg"))

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()

        label = QLabel(
            'Creating data structure from raw microscopy file(s)...')

        label.setStyleSheet("padding:5px 10px 10px 10px;")
        label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        label.setFont(font)
        mainLayout.addWidget(label)

        informativeHtml = (
        """
        <html>
        <head>
        <title></title>
        <style type="text/css">
        blockquote {
         margin: 5;
         padding: 0;
        }
        </style>
        </head>
        <body>
        <blockquote>
        <p style="font-size:11pt; line-height:1.2">
            This <b>wizard</b> will guide you through the <b>creation of the required
            data structure</b><br> starting from the raw microscopy file(s)
        </p>
        <p style="font-size:10pt; line-height:1.2">
            Follow the instructions in the pop-up windows.<br>
            Note that pop-ups might be minimized or behind other open windows.<br>
            Progess is displayed in the terminal/console.
        </p>
        </blockquote>
        </body>
        </html>
        """
        )

        informativeText = QLabel(self)

        informativeText.setTextFormat(Qt.RichText)
        informativeText.setText(informativeHtml)
        informativeText.setStyleSheet("padding:5px 0px 10px 0px;")
        mainLayout.addWidget(informativeText)

        abortButton = QPushButton('Abort process')
        abortButton.clicked.connect(self.close)
        mainLayout.addWidget(abortButton)

        mainLayout.setContentsMargins(20, 0, 20, 20)
        mainContainer.setLayout(mainLayout)

    def getMostRecentPath(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            if 'opened_last_on' in df.columns:
                df = df.sort_values('opened_last_on', ascending=False)
            self.MostRecentPath = df.iloc[0]['path']
            if not isinstance(self.MostRecentPath, str):
                self.MostRecentPath = ''
        else:
            self.MostRecentPath = ''

    def addToRecentPaths(self, exp_path):
        if not os.path.exists(exp_path):
            return
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'temp', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            recentPaths = df['path'].to_list()
            if 'opened_last_on' in df.columns:
                openedOn = df['opened_last_on'].to_list()
            else:
                openedOn = [np.nan]*len(recentPaths)
            if exp_path in recentPaths:
                pop_idx = recentPaths.index(exp_path)
                recentPaths.pop(pop_idx)
                openedOn.pop(pop_idx)
            recentPaths.insert(0, exp_path)
            openedOn.insert(0, datetime.datetime.now())
            # Keep max 20 recent paths
            if len(recentPaths) > 20:
                recentPaths.pop(-1)
                openedOn.pop(-1)
        else:
            recentPaths = [exp_path]
            openedOn = [datetime.datetime.now()]
        df = pd.DataFrame({'path': recentPaths,
                           'opened_last_on': pd.Series(openedOn,
                                                       dtype='datetime64[ns]')})
        df.index.name = 'index'
        df.to_csv(recentPaths_path)

    def main(self):
        proceed = self.instructMoveRawFiles()
        if not proceed:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        self.getMostRecentPath()
        exp_path = QFileDialog.getExistingDirectory(
            self, 'Select folder containing the microscopy files', self.MostRecentPath)
        self.addToRecentPaths(exp_path)

        if exp_path == '':
            abort = self.doAbort()
            if abort:
                self.close()
                return

        ls = self.checkFileFormat(exp_path)
        if not ls:
            if self.allowExit:
                exit('Execution aborted by the user')
            else:
                self.close()
                return

        javabridge.start_vm(class_path=bioformats.JARS)

        for filename in ls:
            filePath = os.path.join(exp_path, filename)
            metadataXML = bioformats.get_omexml_metadata(filePath)
            metadata = bioformats.OMEXML(metadataXML)

            with bioformats.ImageReader(filePath) as reader:
                img = reader.read()
                print(img.shape)

            print(metadata.get_image_count())
            print(metadata.image().Pixels.SizeZ)
            print(metadata.image().Pixels.SizeT)
            print(metadata.image().Pixels.SizeC)
            print(metadata.image().Pixels.PhysicalSizeX)
            print(metadata.image().Pixels.PhysicalSizeY)
            print(metadata.image().Pixels.PhysicalSizeZ)

            print(metadata.image().Pixels.Channel(0).Name)

        javabridge.kill_vm()

        self.close()
        if self.allowExit:
            exit('Conversion task ended.')

    def instructMoveRawFiles(self):
        msg = QMessageBox(self)
        msg.setWindowTitle('Move microscopy files')
        msg.setIcon(msg.Information)
        msg.setTextFormat(Qt.RichText)
        msg.setText(
        """
        Put all of the raw microscopy files from the <b>same experiment</b>
        into an <b>empty folder</b>.<br><br>

        Note that there should be no other files in this folder.
        """
        )
        doneButton = QPushButton('Done')
        cancelButton = QPushButton('Cancel')
        msg.addButton(doneButton, msg.YesRole)
        msg.addButton(cancelButton, msg.NoRole)
        msg.exec_()
        if msg.clickedButton() == doneButton:
            return True
        else:
            return False

    def checkFileFormat(self, exp_path):
        ls = os.listdir(exp_path)
        all_ext = [
            os.path.splitext(filename)[1] for filename in ls
            if os.path.isfile(os.path.join(exp_path, filename))
        ]
        counter = Counter(all_ext)
        unique_ext = list(counter.keys())
        is_ext_unique = unique_ext == 1
        most_common_ext, _ = counter.most_common(1)[0]
        if not is_ext_unique:
            msg = QMessageBox()
            proceedWithMostCommon = msg.warning(
                self, 'Multiple extensions detected',
                f'The folder {exp_path}\n'
                'contains files with different file extensions '
                f'(extensions detected: {unique_ext})\n\n'
                f'However, the most common extension is {most_common_ext}, '
                'do you want to proceed with\n'
                f'loading only files with extension {most_common_ext}?',
                msg.Yes | msg.Cancel
            )
            if proceedWithMostCommon == msg.Yes:
                return [
                    filename for filename in ls
                    if os.path.isfile(os.path.join(exp_path, filename))
                    and os.path.splitext(filename)[1] == most_common_ext
                ]
            else:
                return []


    def doAbort(self):
        msg = QMessageBox(self)
        closeAnswer = msg.warning(
           self, 'Abort execution?', 'Do you really want to abort process?',
           msg.Yes | msg.No
        )
        if closeAnswer == msg.Yes:
            if self.allowExit:
                exit('Execution aborted by the user')
            else:
                print('Segmentation routine aborted by the user.')
                return True
        else:
            return False

    def closeEvent(self, event):
        if self.buttonToRestore is not None:
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()

if __name__ == "__main__":
    print('Launching segmentation script...')
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Create the application
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    win = createDataStructWin(allowExit=True)
    win.show()
    print('Done. If window asking to select a folder is not visible, it is '
          'behind some other open window.')
    win.main()
    sys.exit(app.exec_())
