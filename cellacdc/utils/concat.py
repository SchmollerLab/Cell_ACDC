from posixpath import basename
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
    QVBoxLayout, QPushButton, QLabel, QWidget,
    QMessageBox, QStyleFactory
)
from PyQt5.QtCore import Qt, QEventLoop
from PyQt5 import QtGui

script_path = os.path.dirname(os.path.realpath(__file__))
cellacdc_path = os.path.join(os.path.dirname(script_path))
sys.path.append(cellacdc_path)

# Custom modules
from .. import prompts, load, myutils, apps, html_utils, widgets

from .. import qrc_resources, printl

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
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
        self.setWindowTitle("Cell-ACDC - Concatenate")
        self.setWindowIcon(QtGui.QIcon(":icon.ico"))

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()

        titleText = html_utils.paragraph(
            '<br><b>Concatenating acdc output tables...</b>', font_size='14px'
        )
        titleLabel = QLabel(titleText)

        titleLabel.setStyleSheet("padding:5px 10px 10px 10px;")
        mainLayout.addWidget(titleLabel)

        infoTxt = (
            'Follow the instructions in the pop-up windows.<br>'
            'Note that pop-ups might be minimized or behind other open windows.<br><br>'
            'Progess is displayed in the terminal/console.'
        )

        informativeLabel = QLabel(html_utils.paragraph(infoTxt))
        mainLayout.addWidget(informativeLabel)

        abortButton = QPushButton('Abort process')
        abortButton.clicked.connect(self.close)
        mainLayout.addWidget(abortButton)

        mainLayout.setContentsMargins(20, 0, 20, 20)
        mainContainer.setLayout(mainLayout)

    def main(self):
        self.MostRecentPath = myutils.getMostRecentPath()
        exp_path = QFileDialog.getExistingDirectory(
            self, 'Select experiment folder containing Position_n folders',
            self.MostRecentPath)
        self.addToRecentPaths(exp_path)

        if exp_path == '':
            abort = self.doAbort()
            if abort:
                self.close()
                return

        self.setWindowTitle(f'Cell-ACDC - Concat Pos - "{exp_path}"')

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
        if AllPos_df is None:
            self.close()
            return
        
        csv_path = self.saveAllPos_df(AllPos_df, exp_path)

        self.close()
        self.dialogProcessDone(csv_path)
        if self.allowExit:
            exit(f'Done. File saved to {csv_path}')

    def saveAllPos_df(self, AllPos_df, exp_path):
        
        AllPos_df_folder = os.path.join(exp_path, 'AllPos_acdc_output')
        if not os.path.exists(AllPos_df_folder):
            os.mkdir(AllPos_df_folder)
        
        AllPos_df_filename = f'AllPos_{self.acdc_df_endname}'
        csv_path = os.path.join(AllPos_df_folder, AllPos_df_filename)
        csv_path_new = csv_path
        i = 1
        while os.path.exists(csv_path_new):
            newFilename = f'{i}_{AllPos_df_filename}'
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
        msg = QMessageBox()
        msg.information(
            self, 'Concatenation process completed.', txt, msg.Ok
        )

    def askNewOrReplace(self, AllPos_df_folder):
        msg = QMessageBox(self)
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
    
    def askSelectAcdcDfFile(self, acdc_df_files):
        selectAcdcDfFile = apps.QDialogListbox(
            'Select file to concatenate',
            'Select file to concatenate:\n',
            acdc_df_files, multiSelection=False, parent=self
        )
        selectAcdcDfFile.exec_()
        if selectAcdcDfFile.cancel:
            return
        else:
            return selectAcdcDfFile.selectedItemsText[0]

    def concatAcdcOutputDfs(self, images_paths):
        print('Loading "acdc_output.csv" and concatenating...')
        keys = []
        df_li = []
        self.acdc_df_endname = ''
        for i, images_path in enumerate(tqdm(images_paths, ncols=100)):
            ls = myutils.listdir(images_path)
            basename, _ = myutils.getBasenameAndChNames(images_path)
            acdc_df_files = [
                f for f in ls if f.endswith('.csv')
                and f[len(basename):].find('acdc_output') != -1
            ]
            if not acdc_df_files:
                print('')
                print('=============================')
                print('WARNING: "acdc_output.csv" files not found in folder '
                    f'{images_path}. Skipping it')
                print('=============================')
                continue
            
            if not self.acdc_df_endname:              
                if len(acdc_df_files) == 1:
                    acdc_df_file = acdc_df_files[0]
                elif i == 0:
                    acdc_df_file = self.askSelectAcdcDfFile(acdc_df_files)
                    if acdc_df_file is None:
                        print('')
                        print('=============================')
                        print('Concatenation process aborted.')
                        print('=============================')
                        return
                self.acdc_df_endname = acdc_df_file[len(basename):]      
            else:
                acdc_df_files = [
                    f for f in acdc_df_files if f.endswith(self.acdc_df_endname)
                ]
                if not acdc_df_files:
                    print('')
                    print('=============================')
                    print(
                        f'WARNING: "{self.acdc_df_endname}" file not found in folder '
                        f'{images_path}. Skipping it'
                    )
                    print('=============================')
                    continue
                acdc_df_file = acdc_df_files[0]
            
            acdc_df_path = os.path.join(images_path, acdc_df_file)
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
        msg = QMessageBox()
        msg.critical(
            self, 'Incompatible folder', txt, msg.Ok
        )
        self.close()
        return

    def addToRecentPaths(self, exp_path):
        if not os.path.exists(exp_path):
            return
        recentPaths_path = os.path.join(
            cellacdc_path, 'temp', 'recentPaths.csv'
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
        if self.allowExit:
            exit('Execution aborted by the user')
        else:
            print('Concatenation process aborted by the user.')
            return True

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
    app.setStyle(QStyleFactory.create('Fusion'))
    win = concatWin(allowExit=True)
    win.show()
    print('Done. If window asking to select a folder is not visible, it is '
          'behind some other open window.')
    win.main()
    sys.exit(app.exec_())
