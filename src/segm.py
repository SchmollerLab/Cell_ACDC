import sys
import os
import re
import traceback
import time
import datetime
import numpy as np
import pandas as pd

import skimage.exposure
import skimage.morphology

from tqdm import tqdm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QVBoxLayout, QPushButton, QLabel, QProgressBar, QHBoxLayout
)
from PyQt5.QtCore import (
    Qt, QEventLoop, QThreadPool, QRunnable, pyqtSignal,
    QObject)
from PyQt5 import QtGui

# Custom modules
import prompts, load, myutils, apps, core, dataPrep

import qrc_resources

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass

class segmWorkerSignals(QObject):
    finished = pyqtSignal(int)
    progress = pyqtSignal(str)

class segmWorker(QRunnable):
    def __init__(
            self,
            img_path,
            user_ch_name,
            SizeT,
            SizeZ,
            model,
            minSize,
            save
        ):
        QRunnable.__init__(self)
        self.signals = segmWorkerSignals()
        self.img_path = img_path
        self.user_ch_name = user_ch_name
        self.SizeT = SizeT
        self.SizeZ = SizeZ
        self.model = model
        self.minSize = minSize
        self.save = save

    def set_YeaZ_params(self, path_weights, thresh_val, min_distance):
        self.path_weights = path_weights
        self.thresh_val = thresh_val
        self.min_distance = min_distance

    def set_Cellpose_params(
            self, cp_model, diameter, flow_threshold, cellprob_threshold
        ):
        self.cp_model = cp_model
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold

    def run(self):
        img_path = self.img_path
        user_ch_name = self.user_ch_name

        PosData = load.loadData(img_path, user_ch_name)

        self.signals.progress.emit(f'Loading {PosData.relPath}...')

        PosData.getBasenameAndChNames(prompts.select_channel_name)
        PosData.buildPaths()
        PosData.loadImgData()
        PosData.loadOtherFiles(
            load_segm_data=False,
            load_acdc_df=False,
            load_shifts=False,
            loadSegmInfo=True,
            load_delROIsInfo=False,
            load_dataPrep_ROIcoords=True,
            loadBkgrData=False,
            load_last_tracked_i=False,
            load_metadata=True
        )

        PosData.SizeT = self.SizeT
        if self.SizeZ > 1:
            SizeZ = PosData.img_data.shape[-3]
            PosData.SizeZ = SizeZ
        else:
            PosData.SizeZ = 1
        PosData.saveMetadata()

        isROIactive = False
        if PosData.dataPrep_ROIcoords is not None:
            isROIactive = PosData.dataPrep_ROIcoords.at['cropped', 'value'] == 0
            x0, x1, y0, y1 = PosData.dataPrep_ROIcoords['value'][:4]

        # Note that stop_i is not used when SizeT == 1 so it does not matter
        # which value it has in that case
        stop_i = PosData.segmSizeT

        if PosData.SizeT > 1:
            if PosData.SizeZ > 1:
                # 3D data over time
                img_data_slice = PosData.img_data[:stop_i]
                Y, X = PosData.img_data.shape[-2:]
                img_data = np.zeros((stop_i, Y, X), PosData.img_data.dtype)
                df = PosData.segmInfo_df.loc[PosData.filename]
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
                img_data = PosData.img_data[:stop_i]
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    img_data = img_data[:, y0:y1, x0:x1]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))
            img_data = [img/img.max() for img in img_data]
            img_data = np.array([skimage.exposure.equalize_adapthist(img)
                                 for img in img_data])
        else:
            if PosData.SizeZ > 1:
                # Single 3D image
                z_info = PosData.segmInfo_df.loc[PosData.filename].iloc[0]
                z = z_info.z_slice_used_dataPrep
                zProjHow = z_info.which_z_proj
                if zProjHow == 'single z-slice':
                    img_data = PosData.img_data[z]
                elif zProjHow == 'max z-projection':
                    img_data = PosData.img_data.max(axis=0)
                elif zProjHow == 'mean z-projection':
                    img_data = PosData.img_data.mean(axis=0)
                elif zProjHow == 'median z-proj.':
                    img_data = np.median(PosData.img_data, axis=0)
                img_data = skimage.exposure.equalize_adapthist(
                                                img_data/img_data.max())
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((0, 0), (y0, Y-y1), (x0, X-x1))
                    img_data = img_data[:, y0:y1, x0:x1]
            else:
                # Single 2D image
                img_data = PosData.img_data/PosData.img_data.max()
                img_data = skimage.exposure.equalize_adapthist(img_data)
                if isROIactive:
                    Y, X = img_data.shape[-2:]
                    pad_info = ((y0, Y-y1), (x0, X-x1))
                    img_data = img_data[y0:y1, x0:x1]

        #
        # self.signals.progress.emit(f'Image shape = {img_data.shape}')

        """Segmentation routine"""
        self.signals.progress.emit(f'Segmenting with {self.model}...')
        t0 = time.time()
        # self.signals.progress.emit(f'Segmenting with {model} (Ctrl+C to abort)...')
        if PosData.SizeT > 1:
            if self.model == 'yeaz':
                pred_stack = nn.batch_prediction(
                    img_data,
                    is_pc=True,
                    path_weights=self.path_weights,
                    batch_size=1
                )
                for _ in range(len(pred_stack)):
                    self.signals.progress.emit('')

            elif self.model == 'cellpose':
                lab_stack = np.zeros(img_data.shape, np.uint16)
                for t, img in enumerate(img_data):
                    lab, flows, _, _ = self.cp_model.eval(
                        img,
                        channels=[0,0],
                        diameter=self.diameter,
                        flow_threshold=self.flow_threshold,
                        cellprob_threshold=self.cellprob_threshold
                    )
                    # lab = core.smooth_contours(lab, radius=2)
                    lab_stack[t] = lab
                    self.signals.progress.emit('')

        else:
            if self.model == 'yeaz':
                pred_stack = nn.prediction(
                    img_data,
                    is_pc=True,
                    path_weights=self.path_weights,
                    batch_size=1
                )
                self.signals.progress.emit('')
            elif self.model == 'cellpose':
                lab_stack, flows, _, _ = self.cp_model.eval(
                    img_data,
                    channels=[0,0],
                    diameter=self.diameter,
                    flow_threshold=self.flow_threshold,
                    cellprob_threshold=self.cellprob_threshold
                )
                self.signals.progress.emit('')
                # lab_stack = core.smooth_contours(lab_stack, radius=2)
        if self.model == 'yeaz':
            # self.signals.progress.emit('Thresholding prediction...')
            thresh_stack = nn.threshold(pred_stack, th=self.thresh_val)

        if PosData.SizeT > 1:
            if self.model == 'yeaz':
                # self.signals.progress.emit('Labelling predictions...')
                lab_stack = segment.segment_stack(
                    thresh_stack, pred_stack, min_distance=self.min_distance,
                    signals=self.signals
                ).astype(np.uint16)
            else:
                self.signals.progress.emit('')
        else:
            if self.model == 'yeaz':
                lab_stack = segment.segment(
                    thresh_stack, pred_stack, min_distance=self.min_distance
                ).astype(np.uint16)
            self.signals.progress.emit('')

        lab_stack = skimage.morphology.remove_small_objects(
            lab_stack, min_size=self.minSize
        )

        if PosData.SizeT > 1:
            # self.signals.progress.emit('Tracking cells...')
            # NOTE: We use yeaz tracking also for cellpose
            tracked_stack = tracking.correspondence_stack(
                lab_stack, signals=self.signals
            ).astype(np.uint16)
        else:
            tracked_stack = lab_stack
            self.signals.progress.emit('')

        if isROIactive:
            tracked_stack = np.pad(tracked_stack, pad_info,  mode='constant')

        if self.save:
            self.signals.progress.emit(f'Saving {PosData.relPath}...')
            np.savez_compressed(PosData.segm_npz_path, tracked_stack)

        t_end = time.time()

        self.signals.progress.emit(f'{PosData.relPath} segmented!')
        self.signals.finished.emit(t_end-t0)


class segmWin(QMainWindow):
    def __init__(self, parent=None, allowExit=False,
                 buttonToRestore=None, mainWin=None):
        self.allowExit = allowExit
        self.processFinished = False
        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin
        super().__init__(parent)
        self.setWindowTitle("Cell-ACDC - Segment")
        self.setWindowIcon(QtGui.QIcon(":assign-motherbud.svg"))

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()
        self.mainLayout = mainLayout

        label = QLabel(
            'Segmentation routine running...')

        label.setStyleSheet("padding:5px 10px 10px 10px;")
        label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        label.setFont(font)
        mainLayout.addWidget(label)

        informativeText = QLabel(
            'Follow the instructions in the pop-up windows.\n'
            'Note that pop-ups might be minimized or behind other open windows.')

        informativeText.setStyleSheet("padding:5px 0px 10px 0px;")
        # informativeText.setWordWrap(True)
        informativeText.setAlignment(Qt.AlignLeft)
        font = QtGui.QFont()
        font.setPointSize(9)
        informativeText.setFont(font)
        mainLayout.addWidget(informativeText)

        self.progressLabel = QLabel(self)
        self.mainLayout.addWidget(self.progressLabel)

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

        self.setWindowTitle(f'Cell-ACDC - Segment - "{exp_path}"')

        if os.path.basename(exp_path).find('Position_') != -1:
            is_pos_folder = True
        else:
            is_pos_folder = False

        if os.path.basename(exp_path).find('Images') != -1:
            is_images_folder = True
        else:
            is_images_folder = False

        print('Loading data...')
        self.progressLabel.setText('Loading data...')

        # Ask which model
        font = QtGui.QFont()
        font.setPointSize(10)
        self.model = prompts.askWhichSegmModel(parent=self)
        if self.model == 'yeaz':
            yeazParams = apps.YeaZ_ParamsDialog(parent=self)
            yeazParams.setFont(font)
            yeazParams.exec_()
            if yeazParams.cancel:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return

            self.thresh_val = yeazParams.threshVal
            self.min_distance = yeazParams.minDist
            self.minSize = yeazParams.minSize
            # YeaZ modules
            print('Importing YeaZ...')
            self.progressLabel.setText('Importing YeaZ...')
            from YeaZ.unet import neural_network as nn
            from YeaZ.unet import segment
            from YeaZ.unet import tracking
            myutils.download_model('YeaZ')
            self.path_weights = nn.determine_path_weights()
        elif self.model == 'cellpose':
            cellposeParams = apps.cellpose_ParamsDialog(parent=self)
            cellposeParams.setFont(font)
            cellposeParams.exec_()
            if cellposeParams.cancel:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return

            self.diameter = cellposeParams.diameter
            if self.diameter==0:
                self.diameter=None
            self.flow_threshold = cellposeParams.flow_threshold
            self.cellprob_threshold = cellposeParams.cellprob_threshold
            self.minSize = cellposeParams.minSize
            # Cellpose modules
            print('Importing cellpose...')
            self.progressLabel.setText('Importing cellpose...')
            from acdc_cellpose import models
            from YeaZ.unet import tracking
            myutils.download_model('cellpose')
            device, gpu = models.assign_device(True, False)
            self.cp_model = models.Cellpose(
                gpu=gpu, device=device, model_type='cyto', torch=True
            )

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
                msg = QtGui.QMessageBox()
                msg.critical(
                    self, 'Incompatible folder', txt, msg.Ok
                )
                self.close()
                return


            select_folder.QtPrompt(
                self, values, allow_abort=False, show=True, toggleMulti=True
            )
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

        # Ask to save?
        msg = QtGui.QMessageBox()
        msg.setFont(font)
        answer = msg.question(self, 'Save?', 'Do you want to save segmentation?',
                              msg.Yes | msg.No | msg.Cancel)
        if answer == msg.Yes:
            self.save = True
        elif answer == msg.No:
            self.save = False
        else:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        user_ch_file_paths = []
        for images_path in images_paths:
            print('')
            print(f'Processing {images_path}')
            filenames = os.listdir(images_path)
            if ch_name_selector.is_first_call:
                ch_names, warn = (
                    ch_name_selector.get_available_channels(filenames, images_path)
                )
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
            for filename in filenames:
                if filename.find(f'{user_ch_name}_aligned.npz') != -1:
                    img_path = os.path.join(images_path, filename)
                    aligned_npz_found = True
                elif filename.find(f'{user_ch_name}.tif') != -1:
                    img_path = os.path.join(images_path, filename)
                    tif_found = True

            if not aligned_npz_found and not tif_found:
                print('')
                print('-------------------------------------------------------')
                print(f'WARNING: The folder {images_path}\n does not contain the file '
                      f'{user_ch_name}_aligned.npz\n or the file {user_ch_name}.tif. '
                      'Skipping it.')
                print('-------------------------------------------------------')
                print('')
            elif not aligned_npz_found and tif_found:
                print('')
                print('-------------------------------------------------------')
                print(f'WARNING: The folder {images_path}\n does not contain the file '
                      f'{user_ch_name}_aligned.npz. Segmenting .tif data.')
                print('-------------------------------------------------------')
                print('')
                user_ch_file_paths.append(img_path)
            elif aligned_npz_found:
                user_ch_file_paths.append(img_path)

        selectROI = False
        # Ask other questions based on first position
        img_path = user_ch_file_paths[0]
        PosData = load.loadData(img_path, user_ch_name, QParent=self)
        PosData.getBasenameAndChNames(prompts.select_channel_name)
        PosData.buildPaths()
        PosData.loadImgData()
        PosData.loadOtherFiles(
            load_segm_data=False,
            load_acdc_df=False,
            load_shifts=False,
            loadSegmInfo=True,
            load_delROIsInfo=False,
            load_dataPrep_ROIcoords=True,
            loadBkgrData=False,
            load_last_tracked_i=False,
            load_metadata=True
        )
        proceed = PosData.askInputMetadata(
                                    ask_SizeT=True,
                                    ask_TimeIncrement=False,
                                    ask_PhysicalSizes=False,
                                    save=True)
        self.SizeT = PosData.SizeT
        self.SizeZ = PosData.SizeZ
        if not proceed:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        if PosData.dataPrep_ROIcoords is None:
            # Ask ROI
            msg = QtGui.QMessageBox()
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
        if PosData.SizeZ > 1 and PosData.segmInfo_df is None:
            launchDataPrep = True
        if selectROI:
            launchDataPrep = True
        if PosData.segmInfo_df is not None:
            if PosData.filename not in PosData.segmInfo_df.index:
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
                print(f'WARNING: The image data in {img_path} is 3D but '
                      f'_segmInfo.csv file not found. Launching dataPrep.py...')
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
            if PosData.SizeZ == 1:
                dataPrepWin.prepData(None)
            loop = QEventLoop(self)
            dataPrepWin.loop = loop
            loop.exec_()
            PosData = load.loadData(img_path, user_ch_name, QParent=self)
            PosData.getBasenameAndChNames(prompts.select_channel_name)
            PosData.buildPaths()
            PosData.loadImgData()
            PosData.loadOtherFiles(
                load_segm_data=False,
                load_acdc_df=False,
                load_shifts=False,
                loadSegmInfo=True,
                load_delROIsInfo=False,
                load_dataPrep_ROIcoords=True,
                loadBkgrData=False,
                load_last_tracked_i=False,
                load_metadata=True
            )
        elif PosData.SizeZ > 1:
            df = PosData.segmInfo_df.loc[PosData.filename]
            zz = df['z_slice_used_dataPrep'].to_list()

        isROIactive = False
        if PosData.dataPrep_ROIcoords is not None:
            isROIactive = PosData.dataPrep_ROIcoords.at['cropped', 'value'] == 0
            x0, x1, y0, y1 = PosData.dataPrep_ROIcoords['value'][:4]

        if PosData.SizeT > 1:
            # Ask stop frame. The "askStopFrameSegm" will internally load
            # all the PosData and save segmSizeT which will be used as stop_i
            win = apps.askStopFrameSegm(user_ch_file_paths,
                                        user_ch_name, parent=self)
            win.showAndSetFont(font)
            win.exec_()
            if win.cancel:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return

        print('Starting multiple parallel threads...')
        pBarLayout = QHBoxLayout()
        self.progressLabel.setText('Starting multiple parallel threads...')
        self.QPbar = QProgressBar(self)
        self.QPbar.setValue(0)
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(207, 235, 155))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        self.QPbar.setPalette(palette)
        pBarLayout.addWidget(self.QPbar)
        self.ETA_label = QLabel()
        self.ETA_label.setText('ETA: ND:ND:ND HH:mm:ss')
        pBarLayout.addWidget(self.ETA_label)
        self.mainLayout.insertLayout(3, pBarLayout)

        max = 0
        for imgPath in user_ch_file_paths:
            PosData = load.loadData(imgPath, user_ch_name)
            PosData.loadOtherFiles(
                load_segm_data=False,
                load_metadata=True
            )
            if PosData.SizeT > 1:
                max += PosData.segmSizeT
            else:
                max += 1

        # pBar will be updated three times per frame of each pos:
        # 1. After prediction
        # 2. After prediction --> labels (only YeaZ)
        # 3. After tracking --> only if SizeT > 1
        self.QPbar.setMaximum(max*3)

        self.total_exec_time = 0
        self.time_last_pbar_update = time.time()
        self.exp_path = exp_path
        self.user_ch_file_paths = user_ch_file_paths
        self.user_ch_name = user_ch_name

        self.threadCount = 1 # QThreadPool.globalInstance().maxThreadCount()
        self.numThreadsRunning = self.threadCount
        self.threadPool = QThreadPool.globalInstance()
        self.numPos = len(user_ch_file_paths)
        self.threadIdx = 0
        for i in range(self.threadCount):
            self.threadIdx = i
            img_path = user_ch_file_paths[i]
            self.startSegmWorker()

    def startSegmWorker(self):
        img_path = self.user_ch_file_paths[self.threadIdx]
        worker = segmWorker(
            img_path,
            self.user_ch_name,
            self.SizeT,
            self.SizeZ,
            self.model,
            self.minSize,
            self.save
        )
        if self.model == 'yeaz':
            worker.set_YeaZ_params(
                self.path_weights, self.thresh_val, self.min_distance
            )
        elif self.model == 'cellpose':
            worker.set_Cellpose_params(
                self.cp_model, self.diameter,
                self.flow_threshold, self.cellprob_threshold
            )
        worker.signals.finished.connect(self.segmWorkerFinished)
        worker.signals.progress.connect(self.segmWorkerProgress)
        self.threadPool.start(worker)

    def segmWorkerProgress(self, text):
        if text:
            print(text)
            self.progressLabel.setText(text)
        else:
            t = time.time()
            self.QPbar.setValue(self.QPbar.value()+1)
            deltaT_step = t - self.time_last_pbar_update
            steps_left = self.QPbar.maximum()-self.QPbar.value()
            ETA = datetime.timedelta(seconds=round(deltaT_step*steps_left))
            self.ETA_label.setText(f'ETA: {ETA} HH:mm:ss')
            self.time_last_pbar_update = t


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
                self.progressLabel.setText(
                    'Segmentation task done.'
                )
                msg = QtGui.QMessageBox(self)
                abort = msg.information(
                   self, 'Segmentation task ended.',
                   'Segmentation task ended.\n\n'
                   f'Total execution time: {exec_time_delta} HH:mm:ss\n\n'
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
            msg = QtGui.QMessageBox()
            closeAnswer = msg.critical(
               self, 'Execution aborted',
               'Segmentation task aborted.',
               msg.Ok
            )
            return True

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
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    win = segmWin(allowExit=True)
    win.show()
    print('Done. If window asking to select a folder is not visible, it is '
          'behind some other open window.')
    win.main()
    sys.exit(app.exec_())
