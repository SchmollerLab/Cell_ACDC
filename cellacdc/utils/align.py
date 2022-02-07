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
cellacdc_path = os.path.join(os.path.dirname(script_path))
sys.path.append(cellacdc_path)

# Custom modules
from . import prompts, load, myutils, apps

from . import qrc_resources

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

class alignWin(QMainWindow):
    def __init__(self, parent=None, allowExit=False,
                 buttonToRestore=None, mainWin=None):
        self.allowExit = allowExit
        self.processFinished = False
        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin
        super().__init__(parent)
        self.setWindowTitle("Cell-ACDC - Align")
        self.setWindowIcon(QtGui.QIcon(":assign-motherbud.svg"))

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()

        label = QLabel(
            'Alignment routine running...')

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

        self.setWindowTitle(f'Cell-ACDC - Align - "{exp_path}"')

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
                msg = QtGui.QMessageBox()
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
                images_paths[0], filterExt=['npz', 'npy', 'tif'])
        if not proceed:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        revertAlignment = self.askAlignmentMode()

        abort, appendTxts = self.askTxtAppend()
        if abort:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        appendedTxt = appendTxts[0]

        print('Aligning data...')
        if len(selectedFilenames) > 1:
            ch_name_selector = prompts.select_channel_name()
            channelNames, abort = ch_name_selector.get_available_channels(
                selectedFilenames, images_paths[0], useExt=None
            )
            if abort or not channelNames:
                self.criticalNoCommonBasename(
                    selectedFilenames, images_paths[0]
                )
                self.close()
                return

            self.prevSizeT, self.prevSizeZ = 1, 1
            for images_path in tqdm(images_paths, ncols=100):
                for chName in channelNames:
                    if chName.find('align_shift.npy') != -1:
                        continue
                    shifts, shifts_found = load.load_shifts(images_path)
                    if not shifts_found:
                        print('')
                        print('=============================')
                        print('WARNING: "Align_shift.npy" not found in folder '
                              f'{images_path}. Skipping it')
                        print('=============================')
                        continue
                    filenames = myutils.listdir(images_path)
                    chNameFile = [f for f in filenames if f.find(f'{chName}.')!=-1]
                    if not chNameFile:
                        print('')
                        print('=============================')
                        print(f'WARNING: File ending with "{chName}." not found in folder '
                              f'{images_path}. Skipping it')
                        continue

                    filePath = os.path.join(images_path, chNameFile[0])
                    alignedData = self.loadAndAlign(
                                            filePath, shifts, revertAlignment)
                    self.save(alignedData, filePath, appendedTxt)

        else:
            for images_path in tqdm(images_paths, ncols=100):
                shifts, shifts_found = load.load_shifts(images_path)
                if not shifts_found:
                    print('')
                    print('=============================')
                    print('WARNING: "Align_shift.npy" not found in folder '
                          f'{images_path}. Skipping it')
                    print('=============================')
                    continue
                # print(f'Aligning {filePath}...')
                filePath = os.path.join(images_path, selectedFilenames[0])
                alignedData = self.loadAndAlign(
                                            filePath, shifts, revertAlignment)
                self.save(alignedData, filePath, appendedTxt)

        self.close()
        if self.allowExit:
            exit('Done.')

    def save(self, alignedData, filePath, appendedTxt, first_call=True):
        dir = os.path.dirname(filePath)
        filename, ext = os.path.splitext(os.path.basename(filePath))
        path = os.path.join(dir, f'{filename}_{appendedTxt}{ext}')
        if ext == '.npz':
            np.savez_compressed(path, alignedData)
        elif ext == '.npy':
            np.save(path, alignedData)
        elif ext == '.tif':
            with TiffFile(filePath) as tif:
                metadata = tif.imagej_metadata
            try:
                info = metadata['Info']
                SizeT, SizeZ = self.readSizeTZ(info)
            except Exception as e:
                SizeT = len(alignedData)
                SizeZ = 1
            if SizeT != self.prevSizeT or SizeZ != self.prevSizeZ:
                SizeT, SizeZ = self.askImageSizeZT(SizeT, SizeZ)
            myutils.imagej_tiffwriter(path, alignedData, metadata, SizeT, SizeZ)

    def readSizeTZ(self, info):
        SizeT = int(re.findall('SizeT = (\d+)', info)[0])
        SizeZ = int(re.findall('SizeZ = (\d+)', info)[0])
        return SizeT, SizeZ

    def askImageSizeZT(self, SizeT, SizeZ):
        font = QtGui.QFont()
        font.setPointSize(10)
        win = apps.QDialogAcdcInputs(SizeT, SizeZ, None, None,
                                     show_finterval=False,
                                     parent=self, font=font)
        win.setFont(font)
        win.show()
        win.setWidths(font=font)
        win.exec_()
        return win.SizeT, win.SizeZ

    def askTxtAppend(self):
        font = QtGui.QFont()
        font.setPointSize(10)
        self.win = apps.QDialogEntriesWidget(
            winTitle='Appended name',
            entriesLabels=['Type a name to append at the end of each aligned file:'],
            defaultTxts=['realigned'],
            parent=self, font=font
        )
        self.win.exec_()
        return self.win.cancel, self.win.entriesTxt

    def loadAndAlign(self, filePath, shifts, revertAlignment):
        filename = os.path.basename(filePath)
        _, ext = os.path.splitext(filename)
        if ext == '.npz':
            data = np.load(filePath)['arr_0']
        elif ext == '.npy':
            data = np.load(filePath)
        elif ext == '.tif':
            data = skimage.io.imread(filePath)
        if revertAlignment:
            shifts = -shifts
        alignedData = np.zeros_like(data)
        for frame_i, shift in enumerate(shifts):
            img = data[frame_i]
            axis = tuple(range(img.ndim))[-2:]
            aligned_img = np.roll(img, tuple(shift), axis=axis)
            alignedData[frame_i] = aligned_img
        return alignedData

    def criticalNoCommonBasename(self, filenames, parent_path):
        myutils.checkDataIntegrity(filenames, parent_path, parentQWidget=self)

    def askAlignmentMode(self):
        msg = QtGui.QMessageBox(self)
        msg.setWindowTitle('Alignment mode')
        msg.setIcon(msg.Question)
        msg.setText(
            "Do you want to revert a previously applied alignment or repeat alignment?")
        revertButton = QPushButton('Revert alignment')
        msg.addButton(revertButton, msg.YesRole)
        msg.addButton(QPushButton('Repeat alignment'), msg.NoRole)
        msg.exec_()
        if msg.clickedButton() == revertButton:
            revertAlignment = True
            return revertAlignment
        else:
            revertAlignment = False
            return revertAlignment

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
            'Select to which files you want to apply alignment\n\n'
            'NOTE: if you selected multiple Position folders I will try \n'
            'to apply alignment to all the selected files in each Position folder',
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
    win = alignWin(allowExit=True)
    win.show()
    print('Done. If window asking to select a folder is not visible, it is '
          'behind some other open window.')
    win.main()
    sys.exit(app.exec_())
