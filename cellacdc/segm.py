import sys
import inspect
import os
import re
import traceback
import time
import datetime
import numpy as np
import pandas as pd

from importlib import import_module
from functools import partial

import skimage.exposure
import skimage.morphology

from tqdm import tqdm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QVBoxLayout, QPushButton, QLabel, QProgressBar, QHBoxLayout,
    QStyleFactory, QWidget, QMessageBox, QTextEdit
)
from PyQt5.QtCore import (
    Qt, QEventLoop, QThreadPool, QRunnable, pyqtSignal, QObject,
    QMutex, QWaitCondition
)
from PyQt5 import QtGui

# Custom modules
from . import prompts, load, myutils, apps, core, dataPrep, widgets
from . import qrc_resources, html_utils

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass

class QTerminal(QTextEdit):
    def write(self, message):
        message = message.replace('\r ', '')
        if message:
            self.setText(message)

class segmWorkerSignals(QObject):
    finished = pyqtSignal(float)
    progress = pyqtSignal(str)
    progressBar = pyqtSignal(int)
    innerProgressBar = pyqtSignal(int)
    resetInnerPbar = pyqtSignal(int)
    progress_tqdm = pyqtSignal(int)
    signal_close_tqdm = pyqtSignal()
    create_tqdm = pyqtSignal(int)
    debug = pyqtSignal(object)

class segmWorker(QRunnable):
    def __init__(
            self, img_path, mainWin
        ):
        QRunnable.__init__(self)
        self.signals = segmWorkerSignals()
        self.img_path = img_path
        self.user_ch_name = mainWin.user_ch_name
        self.SizeT = mainWin.SizeT
        self.SizeZ = mainWin.SizeZ
        self.model = mainWin.model
        self.model_name = mainWin.model_name
        self.minSize = mainWin.minSize
        self.minSolidity = mainWin.minSolidity
        self.maxElongation = mainWin.maxElongation
        self.applyPostProcessing = mainWin.applyPostProcessing
        self.save = mainWin.save
        self.segment2D_kwargs = mainWin.segment2D_kwargs
        self.do_tracking = mainWin.do_tracking
        self.predictCcaState_model = mainWin.predictCcaState_model
        self.is_segment3DT_available = mainWin.is_segment3DT_available
        self.innerPbar_available = mainWin.innerPbar_available
        self.concat_segm = mainWin.concat_segm
        self.tracker = mainWin.tracker
        self.isNewSegmFile = mainWin.isNewSegmFile
        self.endFilenameSegm = mainWin.endFilenameSegm

    def setupPausingItems(self):
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()

    def run(self):
        img_path = self.img_path
        user_ch_name = self.user_ch_name

        posData = load.loadData(img_path, user_ch_name)
        if self.predictCcaState_model is not None:
            posData.getBasenameAndChNames()
            posData.loadOtherFiles(
                load_segm_data=False,
                load_acdc_df=True
            )

        self.signals.progress.emit(f'Loading {posData.relPath}...')

        posData.getBasenameAndChNames()
        posData.buildPaths()
        posData.loadImgData()
        posData.loadOtherFiles(
            load_segm_data=self.concat_segm,
            load_acdc_df=False,
            load_shifts=False,
            loadSegmInfo=True,
            load_delROIsInfo=False,
            load_dataPrep_ROIcoords=True,
            loadBkgrData=False,
            load_last_tracked_i=False,
            load_metadata=True,
            end_filename_segm=self.endFilenameSegm
        )
        s = self.endFilenameSegm
        # Get only name from the string 'segm_<name>.npz'
        endName = s.replace('segm', '', 1).replace('_', '', 1).split('.')[0]
        if endName and not self.concat_segm:
            # Create a new file that is not the default 'segm.npz'
            posData.setFilePaths(endName)

        segmFilename = os.path.basename(posData.segm_npz_path)
        self.signals.progress.emit(f'Segmentation file {segmFilename}...')

        posData.SizeT = self.SizeT
        if self.SizeZ > 1:
            SizeZ = posData.img_data.shape[-3]
            posData.SizeZ = SizeZ
        else:
            posData.SizeZ = 1
        posData.saveMetadata()

        isROIactive = False
        if posData.dataPrep_ROIcoords is not None:
            isROIactive = posData.dataPrep_ROIcoords.at['cropped', 'value'] == 0
            x0, x1, y0, y1 = posData.dataPrep_ROIcoords['value'][:4]
            Y, X = posData.img_data.shape[-2:]
            x0 = x0 if x0>0 else 0
            y0 = y0 if y0>0 else 0
            x1 = x1 if x1<X else X
            y1 = y1 if y1<Y else Y

        # Note that stop_i is not used when SizeT == 1 so it does not matter
        # which value it has in that case
        stop_i = posData.segmSizeT

        if posData.SizeT > 1:
            self.t0 = 0
            if self.concat_segm and posData.segm_data is not None:
                self.t0 = len(posData.segm_data)
            if posData.SizeZ > 1:
                # 3D data over time
                img_data_slice = posData.img_data[self.t0:stop_i]
                Y, X = posData.img_data.shape[-2:]
                img_data = np.zeros((stop_i, Y, X), posData.img_data.dtype)
                df = posData.segmInfo_df.loc[posData.filename]
                for z_info in df[:stop_i].itertuples():
                    i = z_info.Index
                    z = z_info.z_slice_used_dataPrep
                    zProjHow = z_info.which_z_proj
                    img = img_data_slice[i]
                    if zProjHow == 'single z-slice':
                        img_data[i] = img[z]
                    elif zProjHow == 'max z-projection':
                        img_data[i] = img.max(axis=0)
                    elif zProjHow == 'mean z-projection':
                        img_data[i] = img.mean(axis=0)
                    elif zProjHow == 'median z-proj.':
                        img_data[i] = np.median(img, axis=0)
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, :, y0:y1, x0:x1]
                    pad_info = ((0, 0), (0, 0), (y0, Y-y1), (x0, X-x1))
            else:
                # 2D data over time
                img_data = posData.img_data[self.t0:stop_i]
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, y0:y1, x0:x1]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))
        else:
            if posData.SizeZ > 1:
                # Single 3D image
                z_info = posData.segmInfo_df.loc[posData.filename].iloc[0]
                z = z_info.z_slice_used_dataPrep
                zProjHow = z_info.which_z_proj
                if zProjHow == 'single z-slice':
                    img_data = posData.img_data[z]
                elif zProjHow == 'max z-projection':
                    img_data = posData.img_data.max(axis=0)
                elif zProjHow == 'mean z-projection':
                    img_data = posData.img_data.mean(axis=0)
                elif zProjHow == 'median z-proj.':
                    img_data = np.median(posData.img_data, axis=0)
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))
                    img_data = img_data[:, y0:y1, x0:x1]
            else:
                # Single 2D image
                img_data = posData.img_data
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((y0, Y-y1), (x0, X-x1))
                    img_data = img_data[y0:y1, x0:x1]

        self.signals.progress.emit(f'Image shape = {img_data.shape}')

        """Segmentation routine"""
        self.signals.progress.emit(f'Segmenting with {self.model_name}...')
        t0 = time.time()
        # self.signals.progress.emit(f'Segmenting with {model} (Ctrl+C to abort)...')
        if posData.SizeT > 1:
            if self.innerPbar_available:
                self.signals.resetInnerPbar.emit(len(img_data))

            if self.is_segment3DT_available:
                self.segment2D_kwargs['signals'] = (
                    self.signals, self.innerPbar_available
                )
                lab_stack = self.model.segment3DT(
                    img_data, **self.segment2D_kwargs
                )
                self.signals.progressBar.emit(1)
            else:
                lab_stack = np.zeros(img_data.shape, np.uint16)
                for t, img in enumerate(img_data):
                    lab = self.model.segment(img, **self.segment2D_kwargs)
                    lab_stack[t] = lab
                    if self.innerPbar_available:
                        self.signals.innerProgressBar.emit(1)
                    else:
                        self.signals.progressBar.emit(1)
        else:
            lab_stack = self.model.segment(img_data, **self.segment2D_kwargs)
            if self.predictCcaState_model is not None:
                cca_df = self.predictCcaState_model.predictCcaState(
                    img_data, lab_stack
                )
                rp = skimage.measure.regionprops(lab_stack)
                if posData.acdc_df is not None:
                    acdc_df = posData.acdc_df.loc[0]
                else:
                    acdc_df = None
                acdc_df = core.cca_df_to_acdc_df(cca_df, rp, acdc_df=acdc_df)

                # Add frame_i=0 level to index (snapshots)
                acdc_df = pd.concat([acdc_df], keys=[0], names=['frame_i'])
                if self.save:
                    acdc_df.to_csv(posData.acdc_output_csv_path)
            self.signals.progressBar.emit(1)
            # lab_stack = core.smooth_contours(lab_stack, radius=2)

        if self.applyPostProcessing:
            if posData.SizeT > 1:
                for t, lab in enumerate(lab_stack):
                    lab_cleaned = core.remove_artefacts(
                        lab,
                        min_solidity=self.minSolidity,
                        min_area=self.minSize,
                        max_elongation=self.maxElongation
                    )
                    lab_stack[t] = lab_cleaned
            else:
                lab_stack = core.remove_artefacts(
                    lab_stack,
                    min_solidity=self.minSolidity,
                    min_area=self.minSize,
                    max_elongation=self.maxElongation
                )

        if posData.SizeT > 1 and self.do_tracking:
            # NOTE: We use yeaz tracking also for cellpose
            if self.concat_segm and posData.segm_data is not None:
                # Insert last frame from existing segm to ensure
                # correct tracking when concatenating
                last_segm_frame = posData.segm_data[-1]
                lab_stack = np.insert(
                    lab_stack, 0, last_segm_frame, axis=0
                )
            tracked_stack = self.tracker.track(
                lab_stack, signals=self.signals,
                export_to=posData.btrack_tracks_h5_path
            )
            if self.concat_segm and posData.segm_data is not None:
                # Remove first frame that comes from existing segm
                tracked_stack = tracked_stack[1:]
        else:
            tracked_stack = lab_stack
            try:
                if self.innerPbar_available:
                    self.signals.innerProgressBar.emit(posData.segmSizeT)
                else:
                    self.signals.progressBar.emit(posData.segmSizeT)
            except AttributeError:
                if self.innerPbar_available:
                    self.signals.innerProgressBar.emit(1)
                else:
                    self.signals.progressBar.emit(1)

        if isROIactive:
            tracked_stack = np.pad(tracked_stack, pad_info,  mode='constant')

        if self.concat_segm and posData.segm_data is not None:
            # Concatenate existing segmentation with new one
            tracked_stack = np.append(
                posData.segm_data, tracked_stack, axis=0
            )

        if self.save:
            self.signals.progress.emit(f'Saving {posData.relPath}...')
            np.savez_compressed(posData.segm_npz_path, tracked_stack)

        t_end = time.time()

        self.signals.progress.emit(f'{posData.relPath} segmented!')
        self.signals.finished.emit(t_end-t0)


class segmWin(QMainWindow):
    def __init__(self, parent=None, allowExit=False,
                 buttonToRestore=None, mainWin=None):
        self.allowExit = allowExit
        self.processFinished = False
        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin
        super().__init__(parent)

        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module='segm'
        )
        self.logger = logger

        self.setWindowTitle("Cell-ACDC - Segment")
        self.setWindowIcon(QtGui.QIcon(":assign-motherbud.svg"))

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()
        self.mainLayout = mainLayout

        label = QLabel("""
        <p style="font-size:16px">
            <b>Segmentation routine running...</b>
        </p>
        """)

        label.setStyleSheet("padding:5px 10px 10px 10px;")
        label.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(label)

        informativeText = QLabel("""
        <p style="font-size:14px">
            Follow the instructions in the pop-up windows.<br>
            Keep an eye on the terminal/console, in case of any error.
        </p>
        <p style="font-size:12px">
            <i>NOTE that pop-ups might be minimized or behind other open windows.</i>
        </p>
        """)

        informativeText.setStyleSheet("padding:5px 0px 10px 0px;")
        # informativeText.setWordWrap(True)
        informativeText.setAlignment(Qt.AlignLeft)
        font = QtGui.QFont()
        font.setPointSize(9)
        informativeText.setFont(font)
        mainLayout.addWidget(informativeText)

        self.progressLabel = QLabel(self)
        self.mainLayout.addWidget(self.progressLabel)

        abortButton = QPushButton('    Abort process    ')
        abortButton.clicked.connect(self.close)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(abortButton)

        mainLayout.addLayout(buttonsLayout)

        mainLayout.setContentsMargins(20, 0, 20, 20)
        mainContainer.setLayout(mainLayout)

    def getMostRecentPath(self):
        cellacdc_path = os.path.dirname(os.path.abspath(__file__))
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

    def addToRecentPaths(self, exp_path):
        if not os.path.exists(exp_path):
            return
        cellacdc_path = os.path.dirname(os.path.realpath(__file__))
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

    def addPbar(self, add_inner=False):
        pBarLayout = QHBoxLayout()
        QPbar = QProgressBar(self)
        QPbar.setValue(0)
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(207, 235, 155))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        QPbar.setPalette(palette)
        pBarLayout.addWidget(QPbar)
        ETA_label = QLabel()
        ETA_label.setText('ETA: NDh:NDm:NDs')
        pBarLayout.addWidget(ETA_label)
        if add_inner:
            self.innerQPbar = QPbar
            self.innerETA_label = ETA_label
            self.mainLayout.insertLayout(4, pBarLayout)
        else:
            self.QPbar = QPbar
            self.ETA_label = ETA_label
            self.mainLayout.insertLayout(3, pBarLayout)

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

        font = QtGui.QFont()
        font.setPixelSize(13)

        self.setWindowTitle(f'Cell-ACDC - Segment - "{exp_path}"')

        self.addPbar()
        self.addlogTerminal()

        if os.path.basename(exp_path).find('Position_') != -1:
            is_pos_folder = True
        else:
            is_pos_folder = False

        if os.path.basename(exp_path).find('Images') != -1:
            is_images_folder = True
        else:
            is_images_folder = False

        self.logger.info('Loading data...')
        self.progressLabel.setText('Loading data...')

        # Ask which model
        models = myutils.get_list_of_models()
        win = apps.QDialogListbox(
            'Select model',
            'Select model to use for segmentation: ',
            models,
            multiSelection=False,
            parent=self
        )
        win.exec_()
        if win.cancel:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        model_name = win.selectedItemsText[0]
        self.logger.info(f'Importing {model_name}...')
        self.downloadWin = apps.downloadModel(model_name, parent=self)
        self.downloadWin.download()

        self.model_name = model_name
        acdcSegment = import_module(f'models.{model_name}.acdcSegment')
        self.acdcSegment =  acdcSegment

        # Read all models parameters
        init_params, segment_params = myutils.getModelArgSpec(self.acdcSegment)

        # Prompt user to enter the model parameters
        try:
            url = acdcSegment.url_help()
        except AttributeError:
            url = None

        win = apps.QDialogModelParams(
            init_params,
            segment_params,
            model_name, parent=self,
            url=url)
        win.exec_()

        if win.cancel:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        self.segment2D_kwargs = win.segment2D_kwargs
        self.minSize = win.minSize
        self.minSolidity = win.minSolidity
        self.maxElongation = win.maxElongation
        self.applyPostProcessing = win.applyPostProcessing

        init_kwargs = win.init_kwargs
        
        # Initialize model
        self.model = acdcSegment.Model(**init_kwargs)

        ch_name_selector = prompts.select_channel_name(
            which_channel='segm', allow_abort=True
        )

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

            if len(values)>1:
                select_folder.QtPrompt(
                    self, values, allow_abort=False, show=True, toggleMulti=True
                )
                if select_folder.was_aborted:
                    abort = self.doAbort()
                    if abort:
                        self.close()
                        return
                pos_foldernames = select_folder.selected_pos
            else:
                pos_foldernames = select_folder.pos_foldernames

            images_paths = [
                os.path.join(exp_path, pos, 'Images')
                for pos in pos_foldernames
            ]

        elif is_pos_folder:
            pos_foldername = os.path.basename(exp_path)
            exp_path = os.path.dirname(exp_path)
            images_paths = [f'{exp_path}/{pos_foldername}/Images']

        elif is_images_folder:
            images_paths = [exp_path]

        self.save = True
        # # Ask to save? (only for debugging)
        # msg = QMessageBox()
        # msg.setFont(font)
        # answer = msg.question(
        #     self, 'Save?', 'Do you want to save segmentation?',
        #     msg.Yes | msg.No | msg.Cancel
        # )
        # if answer == msg.Yes:
        #     self.save = True
        # elif answer == msg.No:
        #     self.save = False
        # else:
        #     abort = self.doAbort()
        #     if abort:
        #         self.close()
        #         return

        user_ch_file_paths = []
        for images_path in images_paths:
            print('')
            self.logger.info(f'Processing {images_path}')
            filenames = myutils.listdir(images_path)
            if ch_name_selector.is_first_call:
                ch_names, warn = (
                    ch_name_selector.get_available_channels(
                        filenames, images_path
                ))
                if not ch_names:
                    self.criticalNoTifFound(images_path)
                elif len(ch_names) > 1:
                    ch_name_selector.QtPrompt(self, ch_names)
                else:
                    ch_name_selector.channel_name = ch_names[0]
                ch_name_selector.setUserChannelName()
                if ch_name_selector.was_aborted:
                    abort = self.doAbort()
                    if abort:
                        self.close()
                        return
                else:
                    user_ch_name = ch_name_selector.channel_name

            aligned_npz_found = False
            tif_found = False
            dataPrep_fn = None
            for filename in filenames:
                if filename.find(f'{user_ch_name}_aligned.npz') != -1:
                    img_path = os.path.join(images_path, filename)
                    idx = filename.find('_aligned.npz')
                    dataPrep_fn = filename[:idx]
                    aligned_npz_found = True
                elif filename.find(f'{user_ch_name}.tif') != -1:
                    img_path = os.path.join(images_path, filename)
                    tif_found = True

            if not aligned_npz_found and not tif_found:
                print('')
                print('-------------------------------------------------------')
                self.logger.info(f'WARNING: The folder {images_path}\n does not contain the file '
                      f'{user_ch_name}_aligned.npz\n or the file {user_ch_name}.tif. '
                      'Skipping it.')
                print('-------------------------------------------------------')
                print('')
            elif not aligned_npz_found and tif_found:
                print('')
                print('-------------------------------------------------------')
                self.logger.info(f'WARNING: The folder {images_path}\n does not contain the file '
                      f'{user_ch_name}_aligned.npz. Segmenting .tif data.')
                print('-------------------------------------------------------')
                print('')
                user_ch_file_paths.append(img_path)
            elif aligned_npz_found:
                user_ch_file_paths.append(img_path)

        self.numPos = len(user_ch_file_paths)
        hyperparams = self.segment2D_kwargs.copy()
        post_process_params = {
            'model': model_name,
            'minSize': self.minSize,
            'minSolidity': self.minSolidity,
            'maxElongation': self.maxElongation,
            'applied_postprocessing': int(self.applyPostProcessing)
        }
        hyperparams.update(post_process_params)
        hyperparams['segm_channel_name'] = user_ch_name

        selectROI = False
        # Ask other questions based on first position
        img_path = user_ch_file_paths[0]
        posData = load.loadData(img_path, user_ch_name, QParent=self)
        posData.getBasenameAndChNames()
        posData.buildPaths()
        posData.saveSegmHyperparams(hyperparams)
        posData.loadImgData()
        posData.loadOtherFiles(
            load_segm_data=True,
            load_acdc_df=False,
            load_shifts=False,
            loadSegmInfo=True,
            load_delROIsInfo=False,
            load_dataPrep_ROIcoords=True,
            loadBkgrData=False,
            load_last_tracked_i=False,
            load_metadata=True
        )
        proceed = posData.askInputMetadata(
            self.numPos,
            ask_SizeT=True,
            ask_TimeIncrement=False,
            ask_PhysicalSizes=False,
            save=True,
            forceEnableAskSegm3D=True
        )
        self.isSegm3D = posData.isSegm3D
        self.SizeT = posData.SizeT
        self.SizeZ = posData.SizeZ
        if not proceed:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        self.predictCcaState_model = None

        self.is_segment3DT_available = False
        if posData.SizeT>1:
            self.is_segment3DT_available = any(
                [name=='segment3DT' for name in dir(acdcSegment.Model)]
            )

        self.innerPbar_available = False
        if len(user_ch_file_paths)>1 and posData.SizeT>1:
            self.addPbar(add_inner=True)
            self.innerPbar_available = True


        if posData.SizeT == 1:
            # Ask if I should predict budding
            msg = QMessageBox()
            msg.setFont(font)
            answer = msg.question(
                self, 'Predict budding?',
                'Do you want to automatically predict which cells are budding '
                '(relevant only to budding yeast cells)?',
                msg.Yes | msg.No | msg.Cancel
            )
            if answer == msg.Yes:
                self.setPredictBuddingModel()
            elif answer == msg.Cancel:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return

        # Check if there are segmentation already computed
        self.selectedSegmFile = None
        self.endFilenameSegm = 'segm.npz'
        self.concat_segm = False
        self.isNewSegmFile = False
        askNewName = True
        isMultiSegm = False
        for img_path in user_ch_file_paths:
            images_path = os.path.dirname(img_path)
            segm_files = load.get_segm_files(images_path)
            if len(segm_files) > 0:
                isMultiSegm = True
                break

        if isMultiSegm:
            concat_segm, askNewName = self.askMultipleSegm(
                segm_files, isTimelapse=posData.SizeT>1
            )
            if concat_segm is None:
                self.save = False
                abort = self.doAbort()
                if abort:
                    self.close()
                    return
            self.concat_segm = concat_segm

        if self.selectedSegmFile is not None:
            self.endFilenameSegm = self.selectedSegmFile[len(posData.basename):]

        if askNewName:
            self.isNewSegmFile = True
            win = apps.filenameDialog(
                basename=f'{posData.basename}segm',
                hintText='Insert a <b>filename</b> for the segmentation file:<br>'
            )
            win.exec_()
            if win.cancel:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return
            self.endFilenameSegm = f'segm_{win.entryText}.npz'

        if posData.dataPrep_ROIcoords is None:
            # Ask ROI
            msg = QMessageBox()
            msg.setFont(font)
            answer = msg.question(self, 'ROI?',
                'Do you want to choose to segment only '
                'a rectangular region-of-interest (ROI)?',
                msg.Yes | msg.No | msg.Cancel
            )
            if answer == msg.Yes:
                selectROI = True
            elif answer == msg.No:
                selectROI = False
            else:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return

        launchDataPrep = False
        if posData.SizeZ > 1 and posData.segmInfo_df is None:
            launchDataPrep = True
        if selectROI:
            launchDataPrep = True
        if posData.segmInfo_df is not None and posData.SizeZ > 1:
            filenames = posData.segmInfo_df.index.get_level_values(0).unique()
            for _filename in filenames:
                if _filename.endswith(user_ch_name):
                    break
            else:
                launchDataPrep = True

        if launchDataPrep:
            dataPrepWin = dataPrep.dataPrepWin()
            dataPrepWin.show()
            if selectROI:
                dataPrepWin.titleText = (
                """
                If you need to crop press the green tick button,<br>
                otherwise you can close the window.
                """
                )
            else:
                print('')
                self.logger.info(
                    f'WARNING: The image data in {img_path} is 3D but '
                    f'_segmInfo.csv file not found. Launching dataPrep.py...'
                )
                self.logTerminal.setText(
                    f'The image data in {img_path} is 3D but '
                    f'_segmInfo.csv file not found. Launching dataPrep.py...'
                )
                msg = widgets.myMessageBox()
                msg.setWindowTitle('3D z-stacks info missing')
                msg.setIcon(iconName='SP_MessageBoxWarning')
                txt = (f"""
                <p style="font-size:13px">
                    You loaded 3D z-stacks, but you <b>never selected which
                    z-slice or projection method to use for segmentation</b>
                    (Cell-ACDC cannot segment 3D z-stacks,
                    it needs to convert them to 2D).<br><br>
                    I opened a window where you can visualize
                    your z-stacks and <b>select an appropriate z-slice
                    or projection for each Position or frame</b>.
                </p>
                """)
                msg.addText(txt)
                msg.addButton('Ok')
                cancel = msg.addButton(' Cancel ')
                msg.exec_()
                if msg.clickedButton == cancel:
                    abort = self.doAbort()
                    if abort:
                        self.close()
                        return

                dataPrepWin.titleText = (
                """
                Select z-slice (or projection) for each frame/position.<br>
                Then, if you want to segment the entire field of view,
                close the window.<br>
                Otherwise, if you need to select a ROI,
                press the "Start" button, draw the ROI<br>
                and confirm with the green tick button.
                """
                )
                autoStart = False
            dataPrepWin.initLoading()
            dataPrepWin.loadFiles(
                exp_path, user_ch_file_paths, user_ch_name)
            if posData.SizeZ == 1:
                dataPrepWin.prepData(None)
            loop = QEventLoop(self)
            dataPrepWin.loop = loop
            loop.exec_()

            # If data was aligned then we make sure to load it here
            user_ch_file_paths = load.get_user_ch_paths(
                images_paths,
                user_ch_name
            )
            img_path = user_ch_file_paths[0]

            posData = load.loadData(img_path, user_ch_name, QParent=self)
            posData.getBasenameAndChNames()
            posData.buildPaths()
            posData.loadImgData()
            posData.loadOtherFiles(
                load_segm_data=True,
                load_acdc_df=False,
                load_shifts=False,
                loadSegmInfo=True,
                load_delROIsInfo=False,
                load_dataPrep_ROIcoords=True,
                loadBkgrData=False,
                load_last_tracked_i=False,
                load_metadata=True
            )
        elif posData.SizeZ > 1:
            df = posData.segmInfo_df.loc[posData.filename]
            zz = df['z_slice_used_dataPrep'].to_list()

        isROIactive = False
        if posData.dataPrep_ROIcoords is not None:
            isROIactive = posData.dataPrep_ROIcoords.at['cropped', 'value'] == 0
            x0, x1, y0, y1 = posData.dataPrep_ROIcoords['value'][:4]

        self.do_tracking = False
        self.tracker = None
        if posData.SizeT > 1:
            # Ask stop frame. The "askStopFrameSegm" will internally load
            # all the posData and save segmSizeT which will be used as stop_i
            win = apps.askStopFrameSegm(
                user_ch_file_paths, user_ch_name,
                concat_segm=self.concat_segm, parent=self
            )
            win.setFont(font)
            win.exec_()
            if win.cancel:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return

            # Ask whether to track the frames
            trackers = myutils.get_list_of_trackers()
            win = apps.QDialogListbox(
                'Track objects?',
                'Do you want to track the objects?\n\n'
                'If yes, choose which tracker to use and click "Ok"\n\n'
                'If you are unsure, choose YeaZ',
                trackers, additionalButtons=['Do not track'],
                multiSelection=False,
                parent=self
            )
            win.exec_()
            if win.cancel:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return

            if win.clickedButton in win._additionalButtons:
                self.do_tracking = False
                trackerName = ''
                self.trackerName = trackerName
            else:
                self.do_tracking = True
                trackerName = win.selectedItemsText[0]
                self.trackerName = trackerName
                trackerModule = import_module(
                    f'trackers.{trackerName}.{trackerName}_tracker'
                )
                params = {}
                if trackerName == 'BayesianTracker':
                    Y, X = posData.img_data_shape[-2:]
                    if posData.isSegm3D:
                        labShape = (posData.SizeZ, Y, X)
                    else:
                        labShape = (1, Y, X)
                    paramsWin = apps.BayesianTrackerParamsWin(labShape, parent=self)
                    paramsWin.exec_()
                    params = paramsWin.params
                elif trackerName == 'CellACDC':
                    paramsWin = apps.CellACDCTrackerParamsWin(parent=self)
                    paramsWin.exec_()
                    params = paramsWin.params
                self.tracker = trackerModule.tracker(**params)

        self.progressLabel.setText('Starting main worker...')

        max = 0
        for imgPath in user_ch_file_paths:
            posData = load.loadData(imgPath, user_ch_name)
            posData.getBasenameAndChNames()
            posData.loadOtherFiles(
                load_segm_data=False,
                load_metadata=True
            )
            if posData.SizeT > 1:
                max += posData.segmSizeT
            else:
                max += 1

        # pBar will be updated three times per frame of each pos:
        # 1. After segmentation
        # 2. After tracking
        if self.innerPbar_available:
            self.QPbar.setMaximum(len(user_ch_file_paths))
        else:
            self.QPbar.setMaximum(max*2)

        self.exec_time_per_iter = 0
        self.exec_time_per_frame = 0
        self.time_last_innerPbar_update = time.time()

        self.total_exec_time = 0
        self.time_last_pbar_update = time.time()
        self.exp_path = exp_path
        self.user_ch_file_paths = user_ch_file_paths
        self.user_ch_name = user_ch_name

        self.threadCount = 1 # QThreadPool.globalInstance().maxThreadCount()
        self.numThreadsRunning = self.threadCount
        self.threadPool = QThreadPool.globalInstance()
        self.threadIdx = 0
        for i in range(self.threadCount):
            self.threadIdx = i
            img_path = user_ch_file_paths[i]
            self.startSegmWorker()

    def askMultipleSegm(self, segm_files, isTimelapse=True):
        txt = html_utils.paragraph("""
            At least one of the loaded positions <b>already contains a
            segmentation file</b>.<br><br>
            What do you want me to do?<br><br>
            <i>NOTE: you will be able to choose a stop frame later.</i><br>
        """)
        msg = widgets.myMessageBox(resizeButtons=False)
        msg.setWindowTitle('Multiple segmentation files')
        msg.addText(txt)
        concatButton = ''
        if isTimelapse:
            if len(segm_files) > 1:
                concatTxt = 'Select segm. file to concatenate to...'
            else:
                concatTxt = 'Concatenate to existing segmentation file'
            concatButton = widgets.reloadPushButton(concatTxt)
        if len(segm_files) > 1:
            overWriteText = 'Select segm. file to overwrite...'
        else:
            overWriteText = 'Overwrite existing segmentation file'
        overWriteButton = widgets.savePushButton(overWriteText)
        doNotSaveButton = widgets.noPushButton('Do not save')
        newButton = widgets.newFilePushButton('Save as...')
        msg.addCancelButton(connect=True)
        if isTimelapse:
            msg.addButton(concatButton)
        msg.addButton(overWriteButton)
        msg.addButton(newButton)
        msg.addButton(doNotSaveButton)
        if len(segm_files)>1:
            if isTimelapse:
                concatButton.clicked.disconnect()
                func = partial(
                    self.selectSegmFile, segm_files, False, msg,
                    concatButton
                )
                concatButton.clicked.connect(func)
            overWriteButton.clicked.disconnect()
            func = partial(
                self.selectSegmFile, segm_files, True, msg,
                overWriteButton
            )
            overWriteButton.clicked.connect(func)
        else:
            self.selectedSegmFile = segm_files[0]

        msg.exec_()
        if msg.cancel:
            return None, None
        elif msg.clickedButton == doNotSaveButton:
            self.save = False
            return False, False
        elif msg.clickedButton == concatButton:
            concat = True
            askNewName = False
            return concat, askNewName
        elif msg.clickedButton == newButton:
            concat = False
            askNewName = True
            return concat, askNewName
        elif msg.clickedButton == overWriteButton:
            concat = False
            askNewName = False
            return concat, askNewName

    def selectSegmFile(self, segm_files, isOverwrite, msg, button):
        action = 'overwrite' if isOverwrite else 'concatenate to'
        selectSegmFileWin = apps.QDialogListbox(
            'Select segmentation file',
            f'Select segmentation file to {action}:\n',
            segm_files, multiSelection=False, parent=msg
        )
        selectSegmFileWin.exec_()
        if selectSegmFileWin.cancel:
            msg.cancel = True
            msg.cancelButton.click()
            return

        self.selectedSegmFile = selectSegmFileWin.selectedItemsText[0]
        button.clicked.disconnect()
        button.clicked.connect(msg.buttonCallBack)
        button.click()

    def addlogTerminal(self):
        self.logTerminal = QTerminal()
        self.logTerminal.setReadOnly(True)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.logTerminal.setFont(font)
        self.mainLayout.insertWidget(4, self.logTerminal)

    def reset_innerQPbar(self, num_frames):
        self.innerQPbar.setValue(0)
        self.innerQPbar.setMaximum(num_frames)

    def create_tqdm_pbar(self, num_frames):
        self.tqdm_pbar = tqdm(
            total=num_frames, unit=' frames', ncols=75, file=self.logTerminal
        )

    def update_tqdm_pbar(self, step):
        self.tqdm_pbar.update(step)

    def close_tqdm(self):
        self.tqdm_pbar.close()

    def setPredictBuddingModel(self):
        self.downloadYeastMate = apps.downloadModel('YeastMate', parent=self)
        self.downloadYeastMate.download()
        import models.YeastMate.acdcSegment as yeastmate
        self.predictCcaState_model = yeastmate.Model()

    def startSegmWorker(self):
        img_path = self.user_ch_file_paths[self.threadIdx]
        worker = segmWorker(img_path, self)
        worker.signals.finished.connect(self.segmWorkerFinished)
        worker.signals.progress.connect(self.segmWorkerProgress)
        worker.signals.progressBar.connect(self.segmWorkerProgressBar)
        worker.signals.innerProgressBar.connect(self.segmWorkerInnerProgressBar)
        worker.signals.resetInnerPbar.connect(self.reset_innerQPbar)
        worker.signals.create_tqdm.connect(self.create_tqdm_pbar)
        worker.signals.progress_tqdm.connect(self.update_tqdm_pbar)
        worker.signals.signal_close_tqdm.connect(self.close_tqdm)
        # worker.signals.debug.connect(self.debugSegmWorker)
        self.threadPool.start(worker)

    def debugSegmWorker(self, lab):
        apps.imshow_tk(lab)

    def segmWorkerProgress(self, text):
        print('-----------------------------------------')
        self.logger.info(text)
        self.progressLabel.setText(text)


    def segmWorkerProgressBar(self, step):
        self.QPbar.setValue(self.QPbar.value()+step)
        steps_left = self.QPbar.maximum()-self.QPbar.value()
        # Update ETA every two calls of this function
        if steps_left%2 == 0:
            t = time.time()
            self.exec_time_per_iter = t - self.time_last_pbar_update
            groups_2steps_left = steps_left/2
            seconds = round(self.exec_time_per_iter*groups_2steps_left)
            ETA = myutils.seconds_to_ETA(seconds)
            self.ETA_label.setText(f'ETA: {ETA}')
            self.exec_time_per_iter = 0
            self.time_last_pbar_update = t

    def segmWorkerInnerProgressBar(self, step):
        self.innerQPbar.setValue(self.innerQPbar.value()+step)
        t = time.time()
        self.exec_time_per_frame = t - self.time_last_innerPbar_update
        steps_left = self.QPbar.maximum()-self.QPbar.value()
        seconds = round(self.exec_time_per_frame*steps_left)
        ETA = myutils.seconds_to_ETA(seconds)
        self.innerETA_label.setText(f'ETA: {ETA}')
        self.exec_time_per_frame = 0
        self.time_last_innerPbar_update = t

        # Estimate total ETA
        current_numFrames = self.QPbar.maximum()
        tot_seconds = round(self.exec_time_per_frame*current_numFrames)
        numPos = self.QPbar.maximum()
        allPos_seconds = tot_seconds*numPos
        tot_seconds_left = allPos_seconds-tot_seconds
        ETA = myutils.seconds_to_ETA(round(tot_seconds_left))
        total_ETA = self.ETA_label.setText(f'ETA: {ETA}')

    def segmWorkerFinished(self, exec_time):
        self.total_exec_time += exec_time
        self.threadIdx += 1
        if self.threadIdx < self.numPos:
            self.startSegmWorker()
        else:
            self.numThreadsRunning -= 1
            if self.numThreadsRunning == 0:
                exec_time = round(self.total_exec_time)
                exec_time_delta = datetime.timedelta(seconds=exec_time)
                h, m, s = str(exec_time_delta).split(':')
                exec_time_delta = f'{int(h):02}h:{int(m):02}m:{int(s):02}s'
                self.progressLabel.setText(
                    'Segmentation task done.'
                )
                msg = QMessageBox(self)
                abort = msg.information(
                   self, 'Segmentation task ended.',
                   'Segmentation task ended.\n\n'
                   f'Total execution time: {exec_time_delta}\n\n'
                   f'Files saved to "{self.exp_path}"',
                   msg.Close
                )
                self.close()
                if self.allowExit:
                    exit('Conversion task ended.')

    def doAbort(self):
        if self.allowExit:
            exit('Execution aborted by the user')
        else:
            msg = widgets.myMessageBox(showCentered=False)
            closeAnswer = msg.information(
               self, 'Execution aborted', 'Segmentation task aborted.'
            )
            return True

    def closeEvent(self, event):
        print('')
        self.logger.info('Closing segmentation module...')
        if self.buttonToRestore is not None:
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()
        self.logger.info('Segmentation module closed.')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
