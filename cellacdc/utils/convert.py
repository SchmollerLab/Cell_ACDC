import sys
import os
import re
import traceback
import time
import datetime
import numpy as np
import pandas as pd
import h5py

import skimage.io
from tifffile.tifffile import TiffWriter, TiffFile

from tqdm import tqdm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QVBoxLayout, QPushButton, QLabel, QStyleFactory,
    QWidget, QMessageBox
)
from PyQt5.QtCore import Qt, QEventLoop
from PyQt5 import QtGui

script_path = os.path.dirname(os.path.realpath(__file__))
cellacdc_path = os.path.join(os.path.dirname(script_path))
sys.path.append(cellacdc_path)

# Custom modules
from .. import prompts, load, myutils, apps, load, widgets, html_utils

from .. import qrc_resources

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

class convertFileFormatWin(QMainWindow):
    def __init__(
            self, parent=None, allowExit=False,
            actionToEnable=None, mainWin=None,
            from_='npz', to='npy', info=''
        ):
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

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()

        titleText = html_utils.paragraph(
            f'<b>Converting .{from_} to .{to} routine running...</b>',
            font_size='14px'
        )
        titleLabel = QLabel(titleText)
        mainLayout.addWidget(titleLabel)

        infoTxt = (
            'Follow the instructions in the pop-up windows.<br>'
            'Note that pop-ups might be minimized or behind other open windows.<br><br>'
            'Progess is displayed in the terminal/console.'
        )
        informativeLabel = QLabel(html_utils.paragraph(infoTxt))

        informativeLabel.setStyleSheet("padding:5px 0px 10px 0px;")
        # informativeLabel.setWordWrap(True)
        informativeLabel.setAlignment(Qt.AlignLeft)
        mainLayout.addWidget(informativeLabel)

        abortButton = QPushButton('Abort process')
        abortButton.clicked.connect(self.close)
        mainLayout.addWidget(abortButton)

        mainLayout.setContentsMargins(20, 0, 20, 20)
        mainContainer.setLayout(mainLayout)

    def getMostRecentPath(self):
        recentPaths_path = os.path.join(
            cellacdc_path, 'temp', 'recentPaths.csv'
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
            self, 'Select experiment folder containing Position_n folders '
                  'or specific Position_n folder', self.MostRecentPath)
        self.addToRecentPaths(exp_path)

        if exp_path == '':
            abort = self.doAbort()
            if abort:
                self.close()
                return

        self.setWindowTitle(
            f'Cell-ACDC - Convert .{self.from_} to .{self.to} - "{exp_path}"'
        )

        if os.path.basename(exp_path).find('Position_') != -1:
            is_pos_folder = True
        else:
            is_pos_folder = False

        if os.path.basename(exp_path).find('Images') != -1:
            is_images_folder = True
        else:
            is_images_folder = False

        print('Loading data...')

        if not is_pos_folder and not is_images_folder:
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

        proceed, selectedFilenames = self.selectFiles(
                images_paths[0], filterExt=[f'{self.from_}'])
        if not proceed:
            abort = self.doAbort()
            if abort:
                self.close()
                return


        abort, appendedTxt = self.askTxtAppend(selectedFilenames[0])
        if abort:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        print(f'Converting .{self.from_} to .{self.to} started...')
        if len(selectedFilenames) > 1 or len(images_paths) > 1:
            ch_name_selector = prompts.select_channel_name()
            ls = myutils.listdir(images_paths[0])
            all_channelNames, abort = ch_name_selector.get_available_channels(
                    ls, images_paths[0], useExt=None
            )
            if abort:
                self.criticalNoCommonBasename(
                    selectedFilenames, images_paths[0]
                )
                self.close()
                return
            _endswith_li = [
                f[len(ch_name_selector.basename):] for f in selectedFilenames
            ]
            for images_path in tqdm(images_paths, ncols=100):
                ls = myutils.listdir(images_path)
                _, skip = ch_name_selector.get_available_channels(
                    ls, images_path, useExt=None
                )
                for _endswith in _endswith_li:
                    for file in ls:
                        if file.endswith(_endswith):
                            self.convert(
                                images_path, file, appendedTxt,
                                from_=self.from_, to=self.to
                            )
        else:
            self.convert(
                images_paths[0], selectedFilenames[0], appendedTxt,
                from_=self.from_, to=self.to
            )
        self.close()
        if self.allowExit:
            exit('Done.')

    def convert(self, images_path, filename, appendedTxt,
                from_='npz', to='npy'):
        filePath = os.path.join(images_path, filename)
        if self.from_ == 'npz':
            data = np.load(filePath)['arr_0']
        elif self.from_ == 'npy':
            data = np.load(filePath)
        elif self.from_ == 'tif':
            data = skimage.io.imread(filePath)
        elif self.from_ == 'h5':
            data = load.h5dump_to_arr(filePath)
        if self.info.find('segm') != -1:
            data = data.astype(np.uint16)
        filename, ext = os.path.splitext(filename)
        if appendedTxt:
            newFilename = f'{filename}_{appendedTxt}.{self.to}'
        else:
            newFilename = f'{filename}.{self.to}'
        newPath = os.path.join(images_path, newFilename)
        if self.to == 'npy':
            np.save(newPath, data)
        elif self.to == 'tif':
            myutils.imagej_tiffwriter(newPath, data, None, 1, 1, imagej=False)
        elif self.to == 'npz':
            np.savez_compressed(newPath, data)
        print(f'File {filePath} saved to {newPath}')
        self.conversionDone(filePath, newPath)

    def conversionDone(self, src, dst):
        msg = widgets.myMessageBox()
        msg.setWidth(700)
        parent_path = os.path.dirname(dst)
        txt = (
            'Done!<br><br>'
            'The file<br><br>'
            f'<code>{src}</code><br><br>'
            f'was converted to <b>.{self.to}</b>, and saved to<br><br>'
            f'<code>{dst}</code>'
        )
        msg.addShowInFileManagerButton(parent_path)
        msg.information(
            self, 'Conversion done!', html_utils.paragraph(txt)
        )


    def save(self, alignedData, filePath, appendedTxt, first_call=True):
        dir = os.path.dirname(filePath)
        filename, ext = os.path.splitext(os.path.basename(filePath))
        path = os.path.join(dir, f'{filename}_{appendedTxt}{ext}')

    def askTxtAppend(self, filename):
        font = QtGui.QFont()
        font.setPointSize(10)
        self.win = apps.QDialogAppendTextFilename(
            filename, self.to, parent=self, font=font
        )
        self.win.exec_()
        return self.win.cancel, self.win.LE.text()

    def criticalNoCommonBasename(self, filenames, parent_path):
        myutils.checkDataIntegrity(filenames, parent_path, parentQWidget=self)


    def selectFiles(self, images_path, filterExt=None):
        files = myutils.listdir(images_path)
        if filterExt is not None:
            items = []
            for file in files:
                _, ext = os.path.splitext(file)
                for allowedExt in filterExt:
                    if ext.find(allowedExt) != -1:
                        items.append(file)
        else:
            items = files

        selectFilesWidget = apps.QDialogListbox(
            'Select files',
            f'Select the .{self.from_} files you want to convert to '
            f'.{self.to}\n\n'
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
    app.setStyle(QStyleFactory.create('Fusion'))
    win = convertFileFormatWin(allowExit=True)
    win.show()
    print('Done. If window asking to select a folder is not visible, it is '
          'behind some other open window.')
    win.main()
    sys.exit(app.exec_())
