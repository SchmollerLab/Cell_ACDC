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
    QVBoxLayout, QPushButton, QLabel, QMessageBox,
    QProgressBar
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
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

class convertFileFormatWin(QMainWindow):
    def __init__(self, parent=None, allowExit=False,
                 actionToEnable=None, mainWin=None,
                 from_='npz', to='npy', info=''):
        self.from_ = from_
        self.to = to
        self.info = info
        self.allowExit = allowExit
        self.processFinished = False
        self.actionToEnable = actionToEnable
        self.mainWin = mainWin
        super().__init__(parent)
        self.setWindowTitle(f"Cell-ACDC - Convert .{from_} file to .{to}")
        self.setWindowIcon(QtGui.QIcon(":assign-motherbud.svg"))

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()

        label = QLabel(
            f'Converting .{from_} to .{to} routine running...')

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
            'Keep and eye on the terminal/console for any error.')

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

        self.mainLayout = mainLayout

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
            self,
            'Select experiment folder containing Position_n folders '
            ', specific Position_n folder '
            ', or folder containing files to be converted.',
            self.MostRecentPath
        )
        self.addToRecentPaths(exp_path)

        if exp_path == '':
            abort = self.doAbort()
            if abort:
                self.close()
                return

        self.setWindowTitle(
            f'Cell-ACDC - Convert .{self.from_} to .{self.to} - "{exp_path}"'
        )

        is_pos_folder = os.path.basename(exp_path).find('Position_') != -1

        is_images_folder = (
            os.path.basename(exp_path).find('Images') != -1
            and os.path.dirname(exp_path).find('Position_') != -1
        )

        contains_pos_folders = any([
            f.find('Position_')!=-1 and os.path.isdir(os.path.join(exp_path, f))
            for f in os.listdir(exp_path)
        ])

        is_not_struct_data = False

        print('Loading data...')

        if contains_pos_folders:
            select_folder = load.select_exp_folder()
            values = select_folder.get_values_segmGUI(exp_path)
            if not values:
                txt = (
                    'The selected folder:\n\n '
                    f'{exp_path}\n\n'
                    'is not a valid folder. '
                    'Select a folder that contains the Position_n folders'
                )
                msg = QMessageBox()
                msg.critical(
                    self, 'Incompatible folder', txt, msg.Ok
                )
                self.close()
                return


            select_folder.QtPrompt(self, values, allow_abort=False, show=True)
            if select_folder.was_aborted:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return


            pos_foldernames = select_folder.selected_pos
            images_paths = [os.path.join(exp_path, pos, 'Images')
                            for pos in pos_foldernames]

        elif is_pos_folder:
            pos_foldername = os.path.basename(exp_path)
            exp_path = os.path.dirname(exp_path)
            images_paths = [f'{exp_path}/{pos_foldername}/Images']

        elif is_images_folder:
            images_paths = [exp_path]

        else:
            is_not_struct_data = True
            images_paths = [exp_path]

        proceed, selectedFilenames = self.selectFiles(
                images_paths[0], filterExt=[f'{self.from_}'])
        if not proceed or not selectedFilenames:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        abort, appendedTxt = self.askTxtAppend(selectedFilenames[0], self.to)
        if abort:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        if is_not_struct_data:
            dst_folder = self.ask_dst_folder()
            # self.addPbar()
            # self.QPbar.setMaximum(len(selectedFilenames))
            for filename in selectedFilenames:
                src_folder = images_paths[0]
                self.convert(
                    src_folder, dst_folder, filename, appendedTxt,
                    from_=self.from_, to=self.to
                )
                # self.QPbar.updatePbar()
            self.close()
            return

        print(f'Converting .{self.from_} to .{self.to} started...')
        if len(selectedFilenames) > 1 or len(images_paths) > 1:
            ch_name_selector = prompts.select_channel_name()
            ls = os.listdir(images_paths[0])
            all_channelNames, abort = ch_name_selector.get_available_channels(
                    ls, images_paths[0], useExt=None
            )
            channelNames = [ch for ch in all_channelNames
                                    for file in selectedFilenames
                                        if file.find(ch)!=-1]
            if abort or not channelNames:
                self.criticalNoCommonBasename()
                self.close()
                return
            for images_path in tqdm(images_paths, ncols=100):
                for chName in channelNames:
                    filenames = os.listdir(images_path)
                    chNameFile = [
                        f for f in filenames
                        if f.find(f'{chName}.{self.from_}')!=-1
                    ]
                    if not chNameFile:
                        print('')
                        print('=============================')
                        print(
                            f'WARNING: File ending with "{chName}.{self.from_}" '
                            'not found in folder '
                            f'{images_path}. Skipping it')
                        continue
                    self.convert(
                        images_path, images_path, chNameFile[0], appendedTxt,
                        from_=self.from_, to=self.to
                    )
        else:
            self.convert(
                images_paths[0], images_paths[0], selectedFilenames[0],
                appendedTxt, from_=self.from_, to=self.to
            )
        self.close()
        if self.allowExit:
            exit('Done.')

    def addPbar(self):
        self.QPbar = QProgressBar(self)
        self.QPbar.setValue(0)
        palette = QPalette()
        palette.setColor(QPalette.Highlight, QColor(207, 235, 155))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.QPbar.setPalette(palette)
        self.mainLayout.insertWidget(self.QPbar, 2)

    def updatePbar(self, step=1):
        self.QPbar.setValue(self.QPbar.value()+step)

    def ask_dst_folder(self):
        dst_folder = QFileDialog.getExistingDirectory(
            self,
            'Select folder where to save converted files',
            self.MostRecentPath
        )
        return dst_folder

    def convert(self, src_folder, dst_folder, filename, appendedTxt,
                from_='npz', to='npy'):
        filePath = os.path.join(src_folder, filename)
        if self.from_ == 'npz':
            data = np.load(filePath)['arr_0']
        elif self.from_ == 'tif':
            data = skimage.io.imread(filePath)

        if self.info == '_segm':
            data = data.astype(np.uint16)

        filename, ext = os.path.splitext(filename)
        if appendedTxt:
            newFilename = f'{filename}_{appendedTxt}.{self.to}'
        else:
            newFilename = f'{filename}.{self.to}'

        newPath = os.path.join(dst_folder, newFilename)
        if self.to == 'npy':
            np.save(newPath, data)
        elif self.to == 'tif':
            myutils.imagej_tiffwriter(newPath, data, None, 1, 1, imagej=False)
        elif self.to == 'npz':
            np.savez_compressed(newPath, data)
        print(f'File {filePath} saved to {newPath}')


    def save(self, alignedData, filePath, appendedTxt, first_call=True):
        dir = os.path.dirname(filePath)
        filename, ext = os.path.splitext(os.path.basename(filePath))
        path = os.path.join(dir, f'{filename}_{appendedTxt}{ext}')

    def askTxtAppend(self, filename, to):
        font = QtGui.QFont()
        font.setPointSize(10)
        self.win = apps.QDialogAppendTextFilename(
            filename, to, parent=self, font=font, default_append=self.info
        )
        self.win.exec_()
        return self.win.cancel, self.win.appendedTxt

    def criticalNoCommonBasename(self):
        msg = QMessageBox()
        msg.critical(
           self, 'Data structure compromised',
           'The system detected files inside the "Images" folder '
           'that do not start with the same, common basename.\n\n'
           'To ensure correct loading of the relevant data, the folder "Images" '
           'inside each Position folder should contain only files that start '
           'with the same, common basename.\n\n'
           'For example the following filenames:\n\n'
           'F014_s01_phase_contr.tif\n'
           'F014_s01_mCitrine.tif\n\n'
           'are named correctly since they all start with the '
           'the common basename "F014_s01_". After the common basename you '
           'can write whatever text you want. In the example above, "phase_contr" '
           'and "mCitrine" are the channel names.\n\n'
           'We recommend using the provided Fiji/ImageJ macro to create the right '
           'data structure.\n\n'
           'To apply alignment select only one "Images" folder AND only one file at the time.',
           msg.Ok
        )


    def selectFiles(self, images_path, filterExt=None):
        files = os.listdir(images_path)
        if filterExt is not None:
            items = []
            for file in files:
                _, ext = os.path.splitext(file)
                for allowedExt in filterExt:
                    if ext.find(allowedExt) != -1:
                        items.append(file)
        else:
            items = files

        if not items:
            msg = QMessageBox()
            msg.critical(
                self, 'Files not existing',
                f'The selected folder\n\n{images_path}\n\n '
                f'does not contain any {filterExt} files!',
                msg.Ok
            )
            return False, []

        selectFilesWidget = apps.QDialogListbox(
            'Select files',
            f'Select the .npz files you want to convert to .{self.to}\n\n'
            'NOTE: if you selected multiple Position folders I will try \n'
            'to convert all selected files in each Position folder',
            items, multiSelection=True, parent=self
        )
        selectFilesWidget.exec_()

        if selectFilesWidget.cancel:
            return False, []

        selectedFilenames = selectFilesWidget.selectedItemsText
        if not selectedFilenames:
            return False, []
        else:
            return True, selectedFilenames

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
        # msg = QMessageBox()
        # closeAnswer = msg.warning(
        #    self, 'Abort execution?', 'Do you really want to abort process?',
        #    msg.Yes | msg.No
        # )
        # if closeAnswer == msg.Yes:
        if self.allowExit:
            exit('Execution aborted by the user')
        else:
            print('Conversion task aborted by the user.')
            return True

    def closeEvent(self, event):
        if self.actionToEnable is not None:
            self.actionToEnable.setDisabled(False)
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()


if __name__ == "__main__":
    print('Launching conversion script...')
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Create the application
    app = QApplication(sys.argv)
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    win = convertFileFormatWin(
        allowExit=True, from_='tif', to='npz', info='_segm')
    win.show()
    print('Done. If window asking to select a folder is not visible, it is '
          'behind some other open window.')
    win.main()
    sys.exit(app.exec_())
