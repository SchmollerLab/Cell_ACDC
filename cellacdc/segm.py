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
from . import qrc_resources, html_utils, printl
from . import exception_handler
from . import workers

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
    critical = pyqtSignal(object)

class segmWorker(QRunnable):
    def __init__(
            self, img_path, mainWin, stop_frame_n
        ):
        QRunnable.__init__(self)
        self.signals = segmWorkerSignals()
        self.img_path = img_path
        self.user_ch_name = mainWin.user_ch_name
        self.SizeT = mainWin.SizeT
        self.SizeZ = mainWin.SizeZ
        self.model = mainWin.model
        self.model_name = mainWin.model_name
        self.removeArtefactsKwargs = mainWin.removeArtefactsKwargs
        self.applyPostProcessing = mainWin.applyPostProcessing
        self.save = mainWin.save
        self.segment2D_kwargs = mainWin.segment2D_kwargs
        self.do_tracking = mainWin.do_tracking
        self.predictCcaState_model = mainWin.predictCcaState_model
        self.is_segment3DT_available = mainWin.is_segment3DT_available
        self.innerPbar_available = mainWin.innerPbar_available
        self.tracker = mainWin.tracker
        self.isNewSegmFile = mainWin.isNewSegmFile
        self.endFilenameSegm = mainWin.endFilenameSegm
        self.isSegm3D = mainWin.isSegm3D
        self.track_params = mainWin.track_params
        self.ROIdeactivatedByUser = mainWin.ROIdeactivatedByUser
        self.secondChannelName = mainWin.secondChannelName
        self.image_chName_tracker = mainWin.image_chName_tracker
        self.stop_frame_n = stop_frame_n

    def setupPausingItems(self):
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()

    @workers.worker_exception_handler
    def run(self):
        try:
            self._run()
        except Exception as error:
            self.signals.critical.emit(error)
            self.signals.finished.emit(-1)

    def _run(self):
        img_path = self.img_path
        user_ch_name = self.user_ch_name

        posData = load.loadData(img_path, user_ch_name)
        if self.predictCcaState_model is not None:
            posData.loadOtherFiles(
                load_segm_data=False,
                load_acdc_df=True
            )

        self.signals.progress.emit(f'Loading {posData.relPath}...')

        posData.getBasenameAndChNames()
        posData.buildPaths()
        posData.loadImgData()
        posData.loadOtherFiles(
            load_segm_data=False,
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
        if endName:
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

        posData.isSegm3D = self.isSegm3D
        posData.saveMetadata()

        isROIactive = False
        if posData.dataPrep_ROIcoords is not None and not self.ROIdeactivatedByUser:
            isROIactive = posData.dataPrep_ROIcoords.at['cropped', 'value'] == 0
            x0, x1, y0, y1 = posData.dataPrep_ROIcoords['value'].astype(int)[:4]
            Y, X = posData.img_data.shape[-2:]
            x0 = x0 if x0>0 else 0
            y0 = y0 if y0>0 else 0
            x1 = x1 if x1<X else X
            y1 = y1 if y1<Y else Y

        # Note that stop_i is not used when SizeT == 1 so it does not matter
        # which value it has in that case
        stop_i = self.stop_frame_n

        if self.secondChannelName is not None:
            self.signals.progress.emit(
                f'Loading second channel "{self.secondChannelName}"...'
            )
            secondChFilePath = load.get_filename_from_channel(
                posData.images_path, self.secondChannelName
            )
            secondChImgData = load.load_image_file(secondChFilePath)

        if posData.SizeT > 1:
            self.t0 = 0
            if posData.SizeZ > 1 and not self.isSegm3D:
                # 2D segmentation on 3D data over time
                img_data = posData.img_data
                if self.secondChannelName is not None:
                    second_ch_data_slice = secondChImgData[self.t0:stop_i]
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, y0:y1, x0:x1]
                    if self.secondChannelName is not None:
                        second_ch_data_slice = second_ch_data_slice[:, y0:y1, x0:x1]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))

                img_data_slice = img_data[self.t0:stop_i]
                Y, X = img_data.shape[-2:]
                newShape = (stop_i, Y, X)
                img_data = np.zeros(newShape, img_data.dtype)
                if self.secondChannelName is not None:
                    second_ch_data = np.zeros(newShape, secondChImgData.dtype)
                df = posData.segmInfo_df.loc[posData.filename]
                for z_info in df[:stop_i].itertuples():
                    i = z_info.Index
                    z = z_info.z_slice_used_dataPrep
                    zProjHow = z_info.which_z_proj
                    img = img_data_slice[i]
                    if self.secondChannelName is not None:
                        second_ch_img = second_ch_data_slice[i]
                    if zProjHow == 'single z-slice':
                        img_data[i] = img[z]
                        if self.secondChannelName is not None:
                            second_ch_data[i] = second_ch_img[z]
                    elif zProjHow == 'max z-projection':
                        img_data[i] = img.max(axis=0)
                        if self.secondChannelName is not None:
                            second_ch_data[i] = second_ch_img.max(axis=0)
                    elif zProjHow == 'mean z-projection':
                        img_data[i] = img.mean(axis=0)
                        if self.secondChannelName is not None:
                            second_ch_data[i] = second_ch_img.mean(axis=0)
                    elif zProjHow == 'median z-proj.':
                        img_data[i] = np.median(img, axis=0)
                        if self.secondChannelName is not None:
                            second_ch_data[i] = np.median(second_ch_img, axis=0)
            elif posData.SizeZ > 1 and self.isSegm3D:
                # 3D segmentation on 3D data over time
                img_data = posData.img_data[self.t0:stop_i]
                if self.secondChannelName is not None:
                    second_ch_data = secondChImgData[self.t0:stop_i]
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, :, y0:y1, x0:x1]
                    if self.secondChannelName is not None:
                        second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]
                    pad_info = ((0, 0), (0, 0), (y0, Y-y1), (x0, X-x1))
            else:
                # 2D data over time
                img_data = posData.img_data[self.t0:stop_i]
                if self.secondChannelName is not None:
                    second_ch_data = secondChImgData[self.t0:stop_i]
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, y0:y1, x0:x1]
                    if self.secondChannelName is not None:
                        second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))
        else:
            if posData.SizeZ > 1 and not self.isSegm3D:
                img_data = posData.img_data
                if self.secondChannelName is not None:
                    second_ch_data = secondChImgData
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((y0, Y-y1), (x0, X-x1))
                    img_data = img_data[:, y0:y1, x0:x1]
                    if self.secondChannelName is not None:
                        second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]
               
                # 2D segmentation on single 3D image
                z_info = posData.segmInfo_df.loc[posData.filename].iloc[0]
                z = z_info.z_slice_used_dataPrep
                zProjHow = z_info.which_z_proj
                if zProjHow == 'single z-slice':
                    img_data = img_data[z]
                    if self.secondChannelName is not None:
                        second_ch_data = second_ch_data[z]
                elif zProjHow == 'max z-projection':
                    img_data = img_data.max(axis=0)
                    if self.secondChannelName is not None:
                        second_ch_data = second_ch_data.max(axis=0)
                elif zProjHow == 'mean z-projection':
                    img_data = img_data.mean(axis=0)
                    if self.secondChannelName is not None:
                        second_ch_data = second_ch_data.mean(axis=0)
                elif zProjHow == 'median z-proj.':
                    img_data = np.median(img_data, axis=0)
                    if self.secondChannelName is not None:
                        second_ch_data[i] = np.median(second_ch_data, axis=0)
            elif posData.SizeZ > 1 and self.isSegm3D:
                # 3D segmentation on 3D z-stack
                img_data = posData.img_data
                if self.secondChannelName is not None:
                    second_ch_data = secondChImgData
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))
                    img_data = img_data[:, y0:y1, x0:x1]
                    if self.secondChannelName is not None:
                        second_ch_data = second_ch_data[:, y0:y1, x0:x1]
            else:
                # Single 2D image
                img_data = posData.img_data
                if self.secondChannelName is not None:
                   second_ch_data = secondChImgData
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((y0, Y-y1), (x0, X-x1))
                    img_data = img_data[y0:y1, x0:x1]
                    if self.secondChannelName is not None:
                        second_ch_data = second_ch_data[y0:y1, x0:x1]

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
                if self.secondChannelName is not None:
                    img_data = self.model.to_rgb_stack(img_data, second_ch_data)
                lab_stack = self.model.segment3DT(
                    img_data, **self.segment2D_kwargs
                )
                if self.innerPbar_available:
                    # emit one pos done
                    self.signals.progressBar.emit(1)
            else:
                lab_stack = np.zeros(img_data.shape, np.uint32)
                for t, img in enumerate(img_data):
                    if self.secondChannelName is not None:
                        img = self.model.to_rgb_stack(img, second_ch_data[t])
                    lab = self.model.segment(img, **self.segment2D_kwargs)
                    lab_stack[t] = lab
                    if self.innerPbar_available:
                        self.signals.innerProgressBar.emit(1)
                    else:
                        self.signals.progressBar.emit(1)
                if self.innerPbar_available:
                    # emit one pos done
                    self.signals.progressBar.emit(1)
        else:
            if self.secondChannelName is not None:
                img_data = self.model.to_rgb_stack(img_data, second_ch_data)
        
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
                        lab, **self.removeArtefactsKwargs
                    )
                    lab_stack[t] = lab_cleaned
            else:
                lab_stack = core.remove_artefacts(
                    lab_stack, **self.removeArtefactsKwargs
                )

        if posData.SizeT > 1 and self.do_tracking:            
            if self.save:
                # Since tracker could raise errors we save the not-tracked 
                # version which will eventually be overwritten
                self.signals.progress.emit(f'Saving NON-tracked masks of {posData.relPath}...')
                np.savez_compressed(posData.segm_npz_path, lab_stack)

            self.signals.innerPbar_available = self.innerPbar_available
            self.track_params['signals'] = self.signals
            if self.image_chName_tracker:
                # Check if loading the image for the tracker is required
                if 'image' in self.track_params:
                    trackerInputImage = self.track_params.pop('image')
                else:
                    self.signals.progress.emit(
                        'Loading image data of channel '
                        f'"{self.image_chName_tracker}"')
                    trackerInputImage = posData.loadChannelData(
                        self.image_chName_tracker)
                try:
                    tracked_stack = self.tracker.track(
                        lab_stack, trackerInputImage, **self.track_params
                    )
                except TypeError:
                    # User accidentally loaded image data but the tracker doesn't
                    # need it
                    self.signals.progress.emit(
                        'Image data is not required by this tracker, ignoring it...'
                    )
                    tracked_stack = self.tracker.track(
                        lab_stack, **self.track_params
                    )
            else:
                tracked_stack = self.tracker.track(
                    lab_stack, **self.track_params
                )
            posData.fromTrackerToAcdcDf(self.tracker, tracked_stack, save=True)
        else:
            tracked_stack = lab_stack
            try:
                if self.innerPbar_available:
                    self.signals.innerProgressBar.emit(self.stop_frame_n)
                else:
                    self.signals.progressBar.emit(self.stop_frame_n)
            except AttributeError:
                if self.innerPbar_available:
                    self.signals.innerProgressBar.emit(1)
                else:
                    self.signals.progressBar.emit(1)

        if isROIactive:
            self.signals.progress.emit(f'Padding with zeros {pad_info}...')
            tracked_stack = np.pad(tracked_stack, pad_info, mode='constant')

        if self.save:
            self.signals.progress.emit(f'Saving {posData.relPath}...')
            np.savez_compressed(posData.segm_npz_path, tracked_stack)

        t_end = time.time()

        self.signals.progress.emit(f'{posData.relPath} segmented!')
        self.signals.finished.emit(t_end-t0)


class segmWin(QMainWindow):
    def __init__(
            self, parent=None, allowExit=False, buttonToRestore=None, 
            mainWin=None, version=None
        ):
        super().__init__(parent)

        self.allowExit = allowExit
        self.processFinished = False
        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin
        self._version = version

        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module='segm'
        )
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path

        if self._version is not None:
            logger.info(f'Initializing Segmentation module v{self._version}...')
        else:
            logger.info(f'Initializing Segmentation module...')

        self.setWindowTitle("Cell-ACDC - Segment")
        self.setWindowIcon(QtGui.QIcon(":icon.ico"))

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

        abortButton = widgets.cancelPushButton('Abort process')
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
        if not add_inner:
            screen = self.screen()
            screenHeight = screen.size().height()
            screenWidth = screen.size().width()
            self.resize(int(screenWidth*0.5), int(screenHeight*0.6))

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

        folder_type = myutils.determine_folder_type(exp_path)
        is_pos_folder, is_images_folder, exp_path = folder_type

        self.log('Loading data...')
        self.progressLabel.setText('Loading data...')

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

        user_ch_file_paths = []
        for images_path in images_paths:
            print('')
            self.log(f'Processing {images_path}')
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
                self.log(f'The folder {images_path}\n does not contain the file '
                      f'{user_ch_name}_aligned.npz\n or the file {user_ch_name}.tif. '
                      'Skipping it.')
                print('-------------------------------------------------------')
                print('')
            elif not aligned_npz_found and tif_found:
                print('')
                print('-------------------------------------------------------')
                self.log(f'The folder {images_path}\n does not contain the file '
                      f'{user_ch_name}_aligned.npz. Segmenting .tif data.')
                print('-------------------------------------------------------')
                print('')
                user_ch_file_paths.append(img_path)
            elif aligned_npz_found:
                user_ch_file_paths.append(img_path)

        self.numPos = len(user_ch_file_paths)

        selectROI = False
        # Ask other questions based on first position
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
        
        # Ask which model
        win = apps.QDialogSelectModel(parent=self)
        win.exec_()
        if win.cancel:
            abort = self.doAbort()
            if abort:
                self.close()
                return
        
        model_name = win.selectedModel

        if model_name == 'thresholding':
            win = apps.QDialogAutomaticThresholding(
                parent=self, isSegm3D=self.isSegm3D
            )
            win.exec_()
            if win.cancel:
                return
            self.segment2D_kwargs = win.segment_kwargs

        self.log(f'Downloading {model_name} (if needed)...')
        self.downloadWin = apps.downloadModel(model_name, parent=self)
        self.downloadWin.download()
        
        self.log(f'Importing {model_name}...')
        self.model_name = model_name
        acdcSegment = myutils.import_segment_module(model_name)
        self.acdcSegment =  acdcSegment

        # Read all models parameters
        init_params, segment_params = myutils.getModelArgSpec(self.acdcSegment)

        # Prompt user to enter the model parameters
        try:
            url = acdcSegment.url_help()
        except AttributeError:
            url = None

        _SizeZ = None
        if self.isSegm3D:
            _SizeZ = posData.SizeZ
        win = apps.QDialogModelParams(
            init_params,
            segment_params,
            model_name, parent=self,
            url=url, SizeZ=_SizeZ
        )
        win.setChannelNames(posData.chNames)
        win.exec_()

        if win.cancel:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        if model_name != 'thresholding':
            self.segment2D_kwargs = win.segment2D_kwargs
        self.removeArtefactsKwargs = win.artefactsGroupBox.kwargs()

        self.applyPostProcessing = win.applyPostProcessing
        self.secondChannelName = win.secondChannelName

        init_kwargs = win.init_kwargs

        # Initialize model
        use_gpu = init_kwargs.get('gpu', False)
        proceed = myutils.check_cuda(model_name, use_gpu, qparent=self)
        if not proceed:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        self.model = acdcSegment.Model(**init_kwargs)
        try:
            self.model.setupLogger(self.logger)
        except Exception as e:
            pass

        self.predictCcaState_model = None

        self.is_segment3DT_available = False
        if posData.SizeT>1 and not self.isSegm3D:
            self.is_segment3DT_available = any(
                [name=='segment3DT' for name in dir(acdcSegment.Model)]
            )

        self.innerPbar_available = False
        if len(user_ch_file_paths)>1 and posData.SizeT>1:
            self.addPbar(add_inner=True)
            self.innerPbar_available = True


        # if posData.SizeT == 1:
        #     # Ask if I should predict budding
        #     msg = widgets.myMessageBox(wrapText=False)
        #     _, yesButton, noButton = msg.question(
        #         self, 'Predict budding?',
        #         'Do you want to automatically predict which cells are budding<br>'
        #         'using <b>YeastMate</b> (relevant only to budding yeast cells)?',
        #         buttonsTexts=('Cancel', 'Yes', 'No')
        #     )
        #     if msg.clickedButton == yesButton:
        #         self.setPredictBuddingModel()
        #     elif msg.cancel:
        #         abort = self.doAbort()
        #         if abort:
        #             self.close()
        #             return

        # Check if there are segmentation already computed
        self.selectedSegmFile = None
        self.endFilenameSegm = 'segm.npz'
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
            askNewName = self.askMultipleSegm(
                segm_files, isTimelapse=posData.SizeT>1
            )
            if askNewName is None:
                self.save = False
                abort = self.doAbort()
                if abort:
                    self.close()
                    return

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
            if win.entryText:
                self.endFilenameSegm = f'segm_{win.entryText}.npz'
            else:
                self.endFilenameSegm = f'segm.npz'

        # Save hyperparams
        post_process_params = {
            'applied_postprocessing': self.applyPostProcessing
        }
        post_process_params = {**post_process_params, **self.removeArtefactsKwargs}
        posData.saveSegmHyperparams(
            model_name, self.segment2D_kwargs, post_process_params
        )

        # Ask ROI
        selectROI = False
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(
            'Do you want to segment only a rectangular '
            '<b>region-of-interest (ROI)</b>?<br><br>'
            'NOTE: If a ROI is already present from the data prep step, Cell-ACDC '
            'will use it.<br>'
            'If you want to modify it, abort the process now and repeat the '
            'data prep step.'
        )
        _, yesButton, noButton = msg.question(self, 'ROI?', txt,
            buttonsTexts = ('Cancel','Yes','No')
        )
        if msg.cancel:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        self.ROIdeactivatedByUser = False
        if msg.clickedButton == yesButton:
            # User requested ROI but it was not present --> ask later
            selectROI = posData.dataPrep_ROIcoords is None
        else:
            # User did not requested ROI --> discard existing ones
            self.ROIdeactivatedByUser = True

        # Check if we should launch dataPrep:
        #   1. 2D segmentation on z-stack data that was never visualized
        #      with dataPrep
        #   2. Select a ROI to segment
        isSegmInfoPresent = True
        for img_path in user_ch_file_paths:
            _posData = load.loadData(img_path, user_ch_name, QParent=self)
            _posData.getBasenameAndChNames()
            _posData.loadOtherFiles(
                load_segm_data=False,
                loadSegmInfo=True,
            )
            if _posData.segmInfo_df is None:
                isSegmInfoPresent = False
                break
        
        segm2D_never_visualized_dataPrep = (
            not self.isSegm3D
            and posData.SizeZ > 1
            and not isSegmInfoPresent
        )
        segm2D_on_3D_visualized = (
            not self.isSegm3D
            and posData.SizeZ > 1
            and isSegmInfoPresent
        )
        launchDataPrep = False

        if segm2D_never_visualized_dataPrep:
            launchDataPrep = True
        if selectROI:
            launchDataPrep = True

        if segm2D_on_3D_visualized:
            # segmInfo_df exists --> check if it has channel z-slice info
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
                self.log(
                    f'WARNING: The image data in {img_path} is 3D but '
                    f'_segmInfo.csv file not found. Launching dataPrep.py...'
                )
                self.logTerminal.setText(
                    f'The image data in {img_path} is 3D but '
                    f'_segmInfo.csv file not found. Launching dataPrep.py...'
                )
                msg = widgets.myMessageBox()
                txt = html_utils.paragraph(f"""
                    You loaded 3D z-stacks, but (in some or all Positions) 
                    you <b>never selected which
                    z-slice or projection method to use for segmentation</b>
                    (this is required for 2D segmentation of 3D data).<br><br>
                    I opened a window where you can visualize
                    your z-stacks and <b>select an appropriate z-slice
                    or projection for each Position or frame</b>.
                """)
                msg.warning(
                    self, '3D z-stacks info missing', txt, 
                    buttonsTexts=('Cancel', 'Ok')
                )
                if msg.cancel:
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
            posData.isSegm3D = self.isSegm3D
        elif posData.SizeZ > 1 and not self.isSegm3D:
            df = posData.segmInfo_df.loc[posData.filename]
            zz = df['z_slice_used_dataPrep'].to_list()

        isROIactive = False
        if posData.dataPrep_ROIcoords is not None and not self.ROIdeactivatedByUser:
            isROIactive = posData.dataPrep_ROIcoords.at['cropped', 'value'] == 0
            x0, x1, y0, y1 = posData.dataPrep_ROIcoords['value'][:4]

        self.image_chName_tracker = ''
        self.do_tracking = False
        self.tracker = None
        self.track_params = {}
        self.stopFrames = [1 for _ in range(len(user_ch_file_paths))]
        if posData.SizeT > 1:
            win = apps.askStopFrameSegm(
                user_ch_file_paths, user_ch_name, parent=self
            )
            win.setFont(font)
            win.exec_()
            if win.cancel:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return

            self.stopFrames = win.stopFrames

            # Ask whether to track the frames
            trackers = myutils.get_list_of_trackers()
            txt = html_utils.paragraph('''
                Do you want to track the objects?<br><br>
                If yes, <b>select the tracker</b> to use<br><br>
            ''')
            win = widgets.QDialogListbox(
                'Track objects?', txt,
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

            self.image_chName_tracker = ''
            if win.clickedButton in win._additionalButtons:
                self.do_tracking = False
                trackerName = ''
                self.trackerName = trackerName
            else:
                self.do_tracking = True
                trackerName = win.selectedItemsText[0]
                self.trackerName = trackerName
                self.tracker, self.track_params = myutils.import_tracker(
                    posData, trackerName, qparent=self
                )
                if self.track_params is None:
                    abort = self.doAbort()
                    if abort:
                        self.close()
                        return
                if 'image_channel_name' in self.track_params:
                    # Store the channel name for the tracker for loading it 
                    # in case of multiple pos
                    self.image_chName_tracker = self.track_params.pop(
                        'image_channel_name')

        self.progressLabel.setText('Starting main worker...')

        max = 0
        for i, imgPath in enumerate(user_ch_file_paths):
            _posData = load.loadData(imgPath, user_ch_name)
            _posData.getBasenameAndChNames()
            _posData.loadOtherFiles(
                load_segm_data=False,
                load_metadata=True
            )
            if posData.SizeT > 1:
                max += self.stopFrames[i]
            else:
                max += 1

        # pBar will be updated two times per frame of each pos:
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
        if len(segm_files) > 1:
            overWriteText = 'Select segm. file to overwrite...'
        else:
            overWriteText = 'Overwrite existing segmentation file'
        overWriteButton = widgets.savePushButton(overWriteText)
        doNotSaveButton = widgets.noPushButton('Do not save')
        newButton = widgets.newFilePushButton('Save as...')
        msg.addCancelButton(connect=True)
        msg.addButton(overWriteButton)
        msg.addButton(newButton)
        msg.addButton(doNotSaveButton)
        if len(segm_files)>1:
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
            return None
        elif msg.clickedButton == doNotSaveButton:
            self.save = False
            return False
        elif msg.clickedButton == newButton:
            askNewName = True
            return askNewName
        elif msg.clickedButton == overWriteButton:
            askNewName = False
            return askNewName

    def selectSegmFile(self, segm_files, isOverwrite, msg, button):
        action = 'overwrite' if isOverwrite else 'concatenate to'
        selectSegmFileWin = widgets.QDialogListbox(
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
    
    def log(self, text):
        self.logger.info(text)
        try:
            self.logTerminal.append(text)
            self.logTerminal.append('-'*30)
            maxScrollbar = self.logTerminal.verticalScrollBar().maximum()
            self.logTerminal.verticalScrollBar().setValue(maxScrollbar)
        except AttributeError:
            pass

    def addlogTerminal(self):
        self.logTerminal = QTerminal()
        self.logTerminal.setReadOnly(True)
        font = QtGui.QFont()
        font.setPointSize(10)
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
        stop_frame_n = self.stopFrames[self.threadIdx]
        worker = segmWorker(img_path, self, stop_frame_n)
        worker.signals.finished.connect(self.segmWorkerFinished)
        worker.signals.progress.connect(self.segmWorkerProgress)
        worker.signals.progressBar.connect(self.segmWorkerProgressBar)
        worker.signals.innerProgressBar.connect(self.segmWorkerInnerProgressBar)
        worker.signals.resetInnerPbar.connect(self.reset_innerQPbar)
        worker.signals.create_tqdm.connect(self.create_tqdm_pbar)
        worker.signals.progress_tqdm.connect(self.update_tqdm_pbar)
        worker.signals.signal_close_tqdm.connect(self.close_tqdm)
        worker.signals.critical.connect(self.workerCritical)
        # worker.signals.debug.connect(self.debugSegmWorker)
        self.threadPool.start(worker)
    
    @exception_handler
    def workerCritical(self, error):
        raise error

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
        if exec_time == -1:
            # Worker finished with error. Do not continue
            return
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
        self.log('Closing segmentation module...')
        if self.buttonToRestore is not None:
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()
        
        self.log('Closing segmentation module logger...')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
        
        try:
            self.model.closeLogger()
        except Exception as e:
            pass
        
        self.log('Segmentation module closed.')
