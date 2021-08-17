import sys
import os
import re
import traceback
import time
import datetime
import numpy as np
import pandas as pd

import skimage.io
from tifffile.tifffile import TiffWriter, TiffFile

from tqdm import tqdm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QVBoxLayout, QPushButton, QLabel
)
from PyQt5.QtCore import Qt, QEventLoop
from PyQt5 import QtGui

script_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(os.path.dirname(script_path))
sys.path.append(src_path)

# Custom modules
import prompts, load, myutils, apps

import qrc_resources

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.yeastacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass

class concatWin(QMainWindow):
    def __init__(self, parent=None, allowExit=False,
                 actionToEnable=None, mainWin=None):
        self.allowExit = allowExit
        self.processFinished = False
        self.actionToEnable = actionToEnable
        self.mainWin = mainWin
        super().__init__(parent)
        self.setWindowTitle("Yeast ACDC - Align")
        self.setWindowIcon(QtGui.QIcon(":assign-motherbud.svg"))

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()

        label = QLabel(
            'Concatenating acdc output tables...')

        label.setStyleSheet("padding:5px 10px 10px 10px;")
        label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        label.setFont(font)
        mainLayout.addWidget(label)

        informativeText = QLabel(
            'Follow the instructions in the pop-up windows.\n'
            'Note that pop-ups might be minimized or behind other open windows.\n\n'
            'Progess is displayed in the terminal/console.')

        informativeText.setStyleSheet("padding:5px 0px 10px 0px;")
        # informativeText.setWordWrap(True)
        informativeText.setAlignment(Qt.AlignLeft)
        font = QtGui.QFont()
        font.setPointSize(9)
        informativeText.setFont(font)
        mainLayout.addWidget(informativeText)

        abortButton = QPushButton('Abort process')
        abortButton.clicked.connect(self.close)
        mainLayout.addWidget(abortButton)

        mainLayout.setContentsMargins(20, 0, 20, 20)
        mainContainer.setLayout(mainLayout)

    def getMostRecentPath(self):
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

    def main(self):
        self.getMostRecentPath()
        exp_path = QFileDialog.getExistingDirectory(
            self, 'Select experiment folder containing Position_n folders',
            self.MostRecentPath)
        self.addToRecentPaths(exp_path)

        if exp_path == '':
            abort = self.doAbort()
            if abort:
                self.close()
                return

        self.setWindowTitle(f'Yeast_ACDC - Concat Pos - "{exp_path}"')

        if os.path.basename(exp_path).find('Position_') != -1:
            self.criticalNoPosFoldersFound(exp_path)
            return

        if os.path.basename(exp_path).find('Images') != -1:
            self.criticalNoPosFoldersFound(exp_path)
            return

        print('Loading data...')

        select_folder = load.select_exp_folder()
        values = select_folder.get_values_segmGUI(exp_path)
        if not values:
            self.criticalNoPosFoldersFound(exp_path)
            return

        select_folder.QtPrompt(
            self, values, allow_abort=False, toggleMulti=True)
        if select_folder.was_aborted:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        pos_foldernames = select_folder.selected_pos
        images_paths = [os.path.join(exp_path, pos, 'Images')
                        for pos in pos_foldernames]

        AllPos_df = self.concatAcdcOutputDfs(images_paths)
        csv_path = self.saveAllPos_df(AllPos_df, exp_path)


        self.close()
        self.dialogProcessDone(csv_path)
        if self.allowExit:
            exit(f'Done. File saved to {csv_path}')

    def saveAllPos_df(self, AllPos_df, exp_path):
        AllPos_df_folder = os.path.join(exp_path, 'AllPos_acdc_output')
        if not os.path.exists(AllPos_df_folder):
            os.mkdir(AllPos_df_folder)

        csv_path = os.path.join(AllPos_df_folder, 'AllPos_acdc_output.csv')
        csv_path_new = csv_path
        i = 1
        while os.path.exists(csv_path_new):
            newFilename = f'AllPos_acdc_output_{i}.csv'
            csv_path_new = os.path.join(AllPos_df_folder, newFilename)
            i += 1

        if os.path.exists(csv_path):
            newFile = self.askNewOrReplace(AllPos_df_folder)
            if newFile:
                csv_path = csv_path_new

        AllPos_df.to_csv(csv_path)
        return csv_path

    def dialogProcessDone(self, csv_path):
        txt = (
            f'Done.\n\nFile saved to:\n\n {csv_path}'
        )
        print('--------------')
        print(txt)
        print('==============')
        msg = QtGui.QMessageBox()
        msg.information(
            self, 'Process completed.', txt, msg.Ok
        )


    def askNewOrReplace(self, AllPos_df_folder):
        msg = QtGui.QMessageBox(self)
        msg.setWindowTitle('Create new files or replace?')
        msg.setIcon(msg.Question)
        msg.setText(
            f'The folder {AllPos_df_folder} already contains "AllPos_acdc_output.csv" file.\n\n'
            'What do you want me to do?')
        newFileButton = QPushButton('Create a new file')
        msg.addButton(newFileButton, msg.YesRole)
        msg.addButton(QPushButton('Replace existing file'), msg.NoRole)
        msg.exec_()
        if msg.clickedButton() == newFileButton:
            newFile = True
            return newFile
        else:
            newFile = False
            return newFile

    def concatAcdcOutputDfs(self, images_paths):
        print('Loading "acdc_output.csv" and concatenating...')
        keys = []
        df_li = []
        for images_path in tqdm(images_paths, ncols=100):
            ls = os.listdir(images_path)
            acdc_df_path = [f for f in ls if f.find('acdc_output.csv')!=-1]
            if not acdc_df_path:
                print('')
                print('=============================')
                print('WARNING: "acdc_output.csv" not found in folder '
                      f'{images_path}. Skipping it')
                print('=============================')
                continue
            acdc_df_path = os.path.join(images_path, acdc_df_path[0])
            df = pd.read_csv(acdc_df_path).set_index(['frame_i', 'Cell_ID'])
            keys.append(os.path.basename(os.path.dirname(images_path)))
            df_li.append(df)
        AllPos_df = pd.concat(
                df_li, keys=keys, names=['Position_n', 'frame_i', 'Cell_ID']
        )
        return AllPos_df


    def criticalNoPosFoldersFound(self, path):
        txt = (
            'The selected folder:\n\n '
            f'{path}\n\n'
            'is not a valid folder. '
            'Select a folder that contains the Position_n folders'
        )
        msg = QtGui.QMessageBox()
        msg.critical(
            self, 'Incompatible folder', txt, msg.Ok
        )
        self.close()
        return

    def addToRecentPaths(self, exp_path):
        if not os.path.exists(exp_path):
            return
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

    def doAbort(self):
        msg = QtGui.QMessageBox()
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
        if self.actionToEnable is not None:
            self.actionToEnable.setDisabled(False)
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()


if __name__ == "__main__":
    print('Launching alignment script...')
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Create the application
    app = QApplication(sys.argv)
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    win = concatWin(allowExit=True)
    win.show()
    print('Done. If window asking to select a folder is not visible, it is '
          'behind some other open window.')
    win.main()
    sys.exit(app.exec_())
