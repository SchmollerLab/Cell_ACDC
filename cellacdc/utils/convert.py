import sys
import os
import re
import traceback
import time
import datetime
import numpy as np
import pandas as pd
import h5py
from collections import Counter

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
        self.setWindowIcon(QtGui.QIcon(":icon.ico"))

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()

        titleText = html_utils.paragraph(
            f'<br><b>Converting .{from_} to .{to} routine running...</b>',
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
                txt = html_utils.paragraph(
                    'The selected folder:<br><br> '
                    f'{exp_path}<br><br>'
                    'is not a valid folder. '
                    'Select a folder that contains the Position_n folders'
                )
                msg = widgets.myMessageBox()
                msg.critical(
                    self, 'Incompatible folder', txt
                )
                self.close()
                return

            if len(values) > 1:
                select_folder.QtPrompt(self, values, allow_abort=False, show=True)
                if select_folder.was_aborted:
                    abort = self.doAbort()
                    if abort:
                        self.close()
                        return
                pos_foldernames = select_folder.selected_pos
            else:
                pos_foldernames = values

            images_paths = [os.path.join(exp_path, pos, 'Images')
                            for pos in pos_foldernames]

        elif is_pos_folder:
            pos_foldername = os.path.basename(exp_path)
            exp_path = os.path.dirname(exp_path)
            images_paths = [f'{exp_path}/{pos_foldername}/Images']

        elif is_images_folder:
            images_paths = [exp_path]

        proceed, selectedFilenames = self.selectFiles(
            images_paths[0], filterExt=[f'{self.from_}']
        )
        if not proceed:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        basename = self.getBasename(images_paths[0], selectedFilenames)

        abort, appendedTxt = self.askTxtAppend(basename)
        if abort:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        print(f'Converting .{self.from_} to .{self.to} started...')
        if len(images_paths) > 1:
            _endswith = selectedFilenames[0][len(basename):]
            if not _endswith:
                self.criticalNoCommonBasename(
                    selectedFilenames, images_paths[0]
                )
                self.close()
                return

            for pos_i, images_path in enumerate(tqdm(images_paths, ncols=100)):
                ls = myutils.listdir(images_path)
                _basename = self.getBasename(
                    images_path, selectedFilenames
                )
                for file in ls:
                    if file.endswith(_endswith):
                        self.convert(
                            images_path, file, appendedTxt, _basename,
                            from_=self.from_, to=self.to, prompt=False
                        )
        else:
            self.convert(
                images_paths[0], selectedFilenames[0], appendedTxt, basename,
                from_=self.from_, to=self.to
            )
        self.close()
        if self.allowExit:
            exit('Done.')

    def getBasename(self, images_path, selectedFilenames):
        commonStartFilenames = myutils.filterCommonStart(images_path)
        selector = prompts.select_channel_name()
        _, noBasename = selector.get_available_channels(
            commonStartFilenames, images_path, useExt=None
        )
        if noBasename:
            basename = os.path.splitext(selectedFilenames[0])[0]
        else:
            basename = selector.basename

        if basename.endswith('_'):
            if self.info.startswith('_'):
                basename = f'{basename}{self.info[1:]}'
            else:
                basename = f'{basename}{self.info}'
        else:
            basename = f'{basename}_{self.info}'
        return basename

    def convert(
            self, images_path, filename, appendedTxt, basename,
            from_='npz', to='npy', prompt=True
        ):
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
            newFilename = f'{basename}_{appendedTxt}.{self.to}'
        else:
            newFilename = f'{basename}.{self.to}'
        newPath = os.path.join(images_path, newFilename)
        if self.to == 'npy':
            np.save(newPath, data)
        elif self.to == 'tif':
            myutils.imagej_tiffwriter(newPath, data, None, 1, 1, imagej=False)
        elif self.to == 'npz':
            np.savez_compressed(newPath, data)
        print('')
        print('-'*30)
        print(f'File "{filePath}" saved to "{newPath}"')
        print('-'*30)
        if prompt:
            self.conversionDone(filePath, newPath)

    def conversionDone(self, src, dst):
        msg = widgets.myMessageBox()
        msg.setWidth(700)
        parent_path = os.path.dirname(dst)
        txt = (
            '<b>Done!</b><br><br>'
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

    def askTxtAppend(self, basename):
        hintText = html_utils.paragraph(
            '<b>OPTIONAL</b>: write here an additional text to append '
            'to the filename'
        )
        win = apps.filenameDialog(
            ext=self.to, title='New filename',
            hintText=hintText, parent=self, basename=basename
        )
        win.exec_()
        if win.cancel:
            win.entryText = ''
        return win.cancel, win.entryText

    def criticalNoCommonBasename(self, filenames, parent_path):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            f'The file name <code>{filenames[0]}</code><br>'
            'does not follow <b>Cell-ACDC naming convention</b>.<br><br>'
            'The name must have the <b>same common basename</b> '
            'as all the other files inside the '
            '<code>Position_n/Images</code> folder.<br><Br>'
            'For example, if in the Images folder you have two files called '
            '<code>ASY015_SCD_phase_contr.tif</code> and '
            '<code>ASY015_SCD_mCitrine.tif</code> then the common basename '
            'is <code>ASY015_SCD_</code> and the file that you are tring to '
            'convert should <b>start with the same common basename</b>.'
        )
        msg.critical(
            self, 'Name of selected file not compatible', txt
        )

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
            items, multiSelection=False, parent=self
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
