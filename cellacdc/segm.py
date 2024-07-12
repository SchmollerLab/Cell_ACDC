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

from qtpy.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QVBoxLayout, QPushButton, QLabel, QProgressBar, QHBoxLayout,
    QStyleFactory, QWidget, QMessageBox, QTextEdit
)
from qtpy.QtCore import (
    Qt, QEventLoop, QThreadPool, QRunnable, Signal, QObject,
    QMutex, QWaitCondition
)
from qtpy import QtGui
import qtpy.compat

# Custom modules
from . import prompts, load, myutils, apps, core, dataPrep, widgets
from . import qrc_resources, html_utils, printl
from . import exception_handler
from . import workers
from . import recentPaths_path
from . import config
from . import urls

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
    finished = Signal(float)
    progress = Signal(str)
    progressBar = Signal(int)
    innerProgressBar = Signal(int)
    resetInnerPbar = Signal(int)
    progress_tqdm = Signal(int)
    signal_close_tqdm = Signal()
    create_tqdm = Signal(int)
    debug = Signal(object)
    critical = Signal(object)

import os
import time

import numpy as np
import pandas as pd

from cellacdc import load, core, features

class segmWorker(QRunnable):
    def __init__(
            self, img_path, mainWin, stop_frame_n
        ):
        QRunnable.__init__(self)
        self.signals = segmWorkerSignals()
        self.img_path = img_path
        self.stop_frame_n = stop_frame_n
        self.mainWin = mainWin
        self.init_kernel(mainWin)
    
    def init_kernel(self, mainWin):
        use_ROI = not mainWin.ROIdeactivatedByUser
        self.kernel = core.SegmKernel(
            mainWin.logger, mainWin.log_path, is_cli=False
        )
        self.kernel.init_args(
            mainWin.user_ch_name, 
            mainWin.endFilenameSegm,
            mainWin.model_name, 
            mainWin.do_tracking,
            mainWin.applyPostProcessing, 
            mainWin.save,
            mainWin.image_chName_tracker,
            mainWin.standardPostProcessKwargs,
            mainWin.customPostProcessGroupedFeatures,
            mainWin.customPostProcessFeatures,
            mainWin.isSegm3D,
            use_ROI,
            mainWin.secondChannelName,
            mainWin.use3DdataFor2Dsegm,
            mainWin.model_kwargs,
            mainWin.track_params,
            mainWin.SizeT, 
            mainWin.SizeZ,
            model=mainWin.model,
            tracker=mainWin.tracker,
            tracker_name=mainWin.trackerName,
            signals=self.signals,
            logger_func=self.signals.progress.emit,
            innerPbar_available=mainWin.innerPbar_available,
            is_segment3DT_available=mainWin.is_segment3DT_available, 
            preproc_recipe=mainWin.preproc_recipe, 
        )
    
    def run_kernel(self, mainWin):
        self.kernel.run(
            self.img_path, 
            self.stop_frame_n
        )

    def setupPausingItems(self):
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()

    @workers.worker_exception_handler
    def run(self):
        try:
            self.run_kernel(self.mainWin)
        except Exception as error:
            self.signals.critical.emit(error)
            self.signals.finished.emit(-1)
    
class segmWin(QMainWindow):
    sigClosed = Signal()
    
    def __init__(
            self, parent=None, allowExit=False, buttonToRestore=None, 
            mainWin=None, version=None
        ):
        super().__init__(parent)

        self.allowExit = allowExit
        self.processFinished = False
        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin
        if mainWin is not None:
            self.app = mainWin.app
            
        self._version = version
        
        self.isSegmWorkerRunning = False

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

        self.progressLabel = widgets.Label(self, force_html=True)
        self.mainLayout.addWidget(self.progressLabel)

        abortButton = widgets.cancelPushButton('Abort process')
        abortButton.clicked.connect(self.close)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(abortButton)

        mainLayout.addLayout(buttonsLayout)

        mainLayout.setContentsMargins(20, 0, 20, 20)
        mainContainer.setLayout(mainLayout)

    def getMostRecentPath(self):
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
        QPbar = widgets.ProgressBar(self)
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

    def askHowToHandleROI(self, posData):
        if posData.dataPrep_ROIcoords is None:
            href = html_utils.href_tag('here', urls.dataprep_docs)
            txt = html_utils.paragraph(f"""
                Do you want to segment only a rectangluar sub-region (ROI) of 
                the image?<br><br>
                If yes, Cell-ACDC will launch the Data-prep module later.<br><br>
                See {href} for more details on how to use the Data-prep module.       
            """)
        elif int(posData.dataPrep_ROIcoords.at[(0, 'cropped'), 'value']) > 0:
            # Data is cropped, do not ask to segment a roi
            return False, False
        else:
            SizeY, SizeX = posData.img_data.shape[-2:]
            x0 = int(posData.dataPrep_ROIcoords.at[(0, 'x_left'), 'value'])
            x1 = int(posData.dataPrep_ROIcoords.at[(0, 'x_left'), 'value'])
            y0 = int(posData.dataPrep_ROIcoords.at[(0, 'y_top'), 'value'])
            y1 = int(posData.dataPrep_ROIcoords.at[(0, 'y_bottom'), 'value'])
            if x0 == 0 and y0 == 0 and y1==SizeY and y1 == SizeX:
                # ROI is present but with same shape as image --> ignore
                return False, False
            
            note = html_utils.to_admonition("""
                If you need to modify the existing ROI, cancel the process 
                now and launch Data-prep again.
            """)
            txt = html_utils.paragraph(f"""
                Cell-ACDC detected a ROI from Data-prep step.<br><br>
                Do you want to use it to segment only in this ROI region you 
                selected in the Data-prep step?<br><br>
                {note}             
            """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        _, yesButton, noButton = msg.question(self, 'ROI?', txt,
            buttonsTexts = ('Cancel','Yes','No')
        )
        return msg.cancel, msg.clickedButton == yesButton
    
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
            load_shifts=True,
            loadSegmInfo=True,
            load_delROIsInfo=False,
            load_dataPrep_ROIcoords=True,
            load_bkgr_data=False,
            load_last_tracked_i=False,
            load_metadata=True,
            load_customCombineMetrics=True
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
        win = apps.QDialogSelectModel(
            parent=self, addSkipSegmButton=posData.SizeT>1
        )
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
            self.model_kwargs = win.segment_kwargs

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
        
        out = prompts.init_segm_model_params(
            posData, model_name, init_params, segment_params, 
            help_url=url, qparent=self, init_last_params=False
        )
        win = out.get('win')
        if win is None:
            abort = self.doAbort()
            if abort:
                self.close()
                return
        
        if model_name != 'thresholding':
            self.model_kwargs = win.model_kwargs
        self.standardPostProcessKwargs = win.standardPostProcessKwargs
        self.customPostProcessFeatures = win.customPostProcessFeatures
        self.customPostProcessGroupedFeatures = (
            win.customPostProcessGroupedFeatures
        )

        self.applyPostProcessing = win.applyPostProcessing
        self.secondChannelName = win.secondChannelName
        
        myutils.log_segm_params(
            model_name, win.init_kwargs, win.model_kwargs, 
            logger_func=self.logger.info, 
            preproc_recipe=win.preproc_recipe, 
            apply_post_process=self.applyPostProcessing, 
            standard_postprocess_kwargs=self.standardPostProcessKwargs, 
            custom_postprocess_features=self.customPostProcessFeatures
        )

        init_kwargs = win.init_kwargs
        self.init_model_kwargs = init_kwargs
        self.preproc_recipe = win.preproc_recipe
        
        # Initialize model
        use_gpu = init_kwargs.get('gpu', False)
        proceed = myutils.check_cuda(model_name, use_gpu, qparent=self)
        if not proceed:
            abort = self.doAbort()
            if abort:
                self.close()
                return
        
        self.model = myutils.init_segm_model(acdcSegment, posData, init_kwargs) 
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
        
        sam_only_embeddings = self.model_kwargs.get('only_embeddings', False)
        self.save = not sam_only_embeddings
        if isMultiSegm and not sam_only_embeddings:
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
        
        if askNewName and self.save:
            self.isNewSegmFile = True
            win = apps.filenameDialog(
                basename=f'{posData.basename}segm',
                hintText='Insert a <b>filename</b> for the segmentation file:<br>',
                existingNames=segm_files
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
        post_process_params = {
            **post_process_params, 
            **self.standardPostProcessKwargs,
            **self.customPostProcessFeatures
        }
        posData.saveSegmHyperparams(
            model_name, self.init_model_kwargs, self.model_kwargs, 
            post_process_params=post_process_params, 
            preproc_recipe=self.preproc_recipe
        )

        # Ask ROI
        selectROI = False
        cancel, useROI = self.askHowToHandleROI(posData)
        if cancel:
            abort = self.doAbort()
            if abort:
                self.close()
                return

        self.ROIdeactivatedByUser = False
        if useROI:
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
        
        self.use3DdataFor2Dsegm = False
        if posData.SizeZ > 1 and not self.isSegm3D:
            cancel, use3DdataFor2Dsegm = self.askHowToHandle2DsegmOn3Ddata()
            if cancel:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return
            self.use3DdataFor2Dsegm = use3DdataFor2Dsegm
        
        segm2D_never_visualized_dataPrep = (
            not self.isSegm3D
            and posData.SizeZ > 1
            and not isSegmInfoPresent
            and not self.use3DdataFor2Dsegm
        )
        segm2D_on_3D_visualized = (
            not self.isSegm3D
            and posData.SizeZ > 1
            and isSegmInfoPresent
            and not self.use3DdataFor2Dsegm
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
            dataPrepWin = dataPrep.dataPrepWin(
                mainWin=self.mainWin, version=self._version
            )
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
                load_bkgr_data=False,
                load_last_tracked_i=False,
                load_metadata=True
            )
            posData.isSegm3D = self.isSegm3D
        elif posData.SizeZ > 1 and not self.isSegm3D and not self.use3DdataFor2Dsegm:
            df = posData.segmInfo_df.loc[posData.filename]
            zz = df['z_slice_used_dataPrep'].to_list()

        isROIactive = False
        if posData.dataPrep_ROIcoords is not None and not self.ROIdeactivatedByUser:
            df_roi = posData.dataPrep_ROIcoords.loc[0]
            isROIactive = df_roi.at['cropped', 'value'] == 0
            x0, x1, y0, y1 = df_roi['value'][:4]
            df_roi = posData.dataPrep_ROIcoords.loc[0]
            isROIactive = df_roi.at['cropped', 'value'] == 0
            x0, x1, y0, y1 = df_roi['value'][:4]

        self.image_chName_tracker = None
        self.do_tracking = False
        self.tracker = None
        self.track_params = {}
        self.tracker_init_params = {}
        self.trackerName = ''
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

            self.image_chName_tracker = None
            if win.clickedButton in win._additionalButtons:
                self.do_tracking = False
                trackerName = ''
                self.trackerName = trackerName
            else:
                self.do_tracking = True
                trackerName = win.selectedItemsText[0]
                self.trackerName = trackerName
                init_tracker_output = myutils.init_tracker(
                        posData, trackerName, return_init_params=True, qparent=self
                )
                self.tracker, self.track_params, self.tracker_init_params = (
                    init_tracker_output
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
                        'image_channel_name'
                    )

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

        proceed = self.askRunNowOrSaveConfigFile()
        if not proceed:
            self.logger.info('Segmentation process interrupted.')
            self.close()
            return
        
        self.threadCount = 1 # QThreadPool.globalInstance().maxThreadCount()
        self.numThreadsRunning = self.threadCount
        self.threadPool = QThreadPool.globalInstance()
        self.threadIdx = 0
        for i in range(self.threadCount):
            self.threadIdx = i
            self.startSegmWorker()
    
    def _saveConfigurationFile(self, filepath):
        init_args = {
            'user_ch_name': self.user_ch_name, 
            'segm_endname': self.endFilenameSegm,
            'model_name': self.model_name,
            'tracker_name': self.trackerName, 
            'do_tracking': self.do_tracking,
            'do_postprocess': self.applyPostProcessing,
            'do_save': self.save,
            'image_channel_tracker': self.image_chName_tracker,
            'isSegm3D': self.isSegm3D,
            'use_ROI': not self.ROIdeactivatedByUser,
            'second_channel_name': self.secondChannelName,
            'use3DdataFor2Dsegm': self.use3DdataFor2Dsegm,
        }
        metadata_params = {
            'SizeT': self.SizeT,
            'SizeZ': self.SizeZ
        }
        track_params = {
            key:value for key, value in self.track_params.items()
            if key != 'image'
        }
        ini_items = {
            'workflow': {'type': 'segmentation and/or tracking'},
            'initialization': init_args,
            'metadata': metadata_params,
            'init_segmentation_model_params': self.init_model_kwargs,
            'segmentation_model_params': self.model_kwargs,
            'init_tracker_params': self.tracker_init_params,
            'tracker_params': track_params,
            'standard_postprocess_features': self.standardPostProcessKwargs,
            'custom_postprocess_features': self.customPostProcessFeatures, 
        }
        preprocessing_items = config.preprocess_recipe_to_ini_items(
            self.preproc_recipe
        )
        ini_items = {**ini_items, **preprocessing_items}
                
        grouped_features = self.customPostProcessGroupedFeatures
        for category, metrics_names in grouped_features.items():
            category_params = {}
            if isinstance(metrics_names, dict):
                for channel, channel_metrics in metrics_names.items():
                    values = '\n'.join(channel_metrics)
                    values = f'\n{values}'
                    category_params[channel] = values
            else:
                values = '\n'.join(metrics_names)
                values = f'\n{values}'
                category_params['names'] = values
            ini_items[f'postprocess_features.{category}'] = category_params

        load.save_segm_workflow_to_config(
            filepath, ini_items, self.user_ch_file_paths, self.stopFrames
        )
        
        self.logger.info(f'Segmentation workflow saved to "{filepath}"')
        
        txt = html_utils.paragraph(
            'Segmentation workflow successfully saved to the following location:<br><br>'
            f'<code>{filepath}</code><br><br>'
            'You can run the segmentation workflow with the following command:'
        )
        command = f'acdc -p "{filepath}"'
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(
            self, 'Workflow save', txt, 
            commands=(command,),
            path_to_browse=os.path.dirname(filepath)
        )
    
    def saveWorkflowToConfigFile(self):
        timestamp = datetime.datetime.now().strftime(
            r'%Y-%m-%d_%H-%M'
        )
        win = apps.filenameDialog(
            parent=self, 
            ext='.ini', 
            title='Insert filename for configuration file',
            hintText='Insert filename for the configuration file',
            allowEmpty=False, 
            defaultEntry=f'{timestamp}_acdc_segm_track_workflow'
        )
        win.exec_()
        if win.cancel:
            return False
        
        config_filename = win.filename
        mostRecentPath = myutils.getMostRecentPath()
        folder_path = apps.get_existing_directory(
            allow_images_path=False,
            parent=self, 
            caption='Select folder where to save configuration file',
            basedir=mostRecentPath,
            # options=QFileDialog.DontUseNativeDialog
        )
        if not folder_path:
            return False
        
        config_filepath = os.path.join(folder_path, config_filename)
        self._saveConfigurationFile(config_filepath)
    
    def askRunNowOrSaveConfigFile(self):
        txt = html_utils.paragraph("""
            Do you want to <b>run</b> the segmentation process <b>now</b><br>
            or save the  workflow to a <b>configuration file</b> and run it 
            <b>later?</b><br><br>
            With the configuration file you can also run the workflow on a<br>
            computing cluster that does not support GUI elements 
            (i.e., headless).<br>
        """)
        msg = widgets.myMessageBox(wrapText=False)
        saveButton = widgets.savePushButton('Save and run later')
        runNowButton = widgets.playPushButton('Run now')
        _, saveButton, runNowButton = msg.question(
            self, 'Run workflow now?', txt, 
            buttonsTexts=(
                'Cancel', saveButton, runNowButton
            )
        )
        if msg.cancel:
            return False
        
        if msg.clickedButton == saveButton:
            saved = self.saveWorkflowToConfigFile()
            if not saved:
                return False
        
        return msg.clickedButton == runNowButton
        
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

    def askHowToHandle2DsegmOn3Ddata(self):
        txt = html_utils.paragraph(
            'How do you want to handle 3D data?'
        )
        use3DButton = widgets.threeDPushButton(
            'Pass all z-slices to the model'
        )
        convertTo2DButton = widgets.twoDPushButton(
            'Use or select z-slices or projection from Data prep'
        )
        buttons = (
            'Cancel', use3DButton, convertTo2DButton
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.question(self, 'How to handle 3D data', txt, buttonsTexts=buttons)
        
        return msg.cancel, msg.clickedButton == use3DButton
    
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
        self.segmWorker = worker
        # worker.signals.debug.connect(self.debugSegmWorker)
        self.threadPool.start(worker)
        self.isSegmWorkerRunning = True
    
    @exception_handler
    def workerCritical(self, error):
        self.isSegmWorkerRunning = False
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
            if self.numThreadsRunning > 0:
                return 
            self.isSegmWorkerRunning = False
            if exec_time > 0:
                short_txt = 'Segmentation process finished!'
                exec_time = round(self.total_exec_time)
                delta = datetime.timedelta(seconds=exec_time)
                exec_time_delta = str(delta).split(',')[-1].strip()
                h, m, s = str(exec_time_delta).split(':')
                exec_time_delta = f'{int(h):02}h:{int(m):02}m:{int(s):02}s'
                items = (
                    f'Total execution time: <code>{exec_time_delta}</code><br>',
                    f'Selected folder: <code>{self.exp_path}</code>'
                )
                txt = (
                    'Segmentation task ended.'
                    f'{html_utils.to_list(items)}'
                )
                steps_left = self.QPbar.maximum()-self.QPbar.value()
                self.QPbar.setValue(self.QPbar.value()+steps_left)
            else:
                short_txt = 'Segmentation process stopped'
                txt = (
                    'Segmentation task stopped by the user.<br>'
                )
            
            txt = html_utils.paragraph(
                f'{txt}<br>{myutils.get_salute_string()}'
            )
            self.progressLabel.setText(short_txt)
            msg = widgets.myMessageBox(self, wrapText=False)
            msg.information(
                self, 'Segmentation task ended.', txt,
                path_to_browse=self.exp_path
            )
            if exec_time > 0:
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
    
    def warnSegmWorkerStillRunning(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            Segmentation/tracking process is <b>still running.</b><br><br>
            The only way to stop the process is to <b>close Cell-ACDC</b>.<br><br>
            Are you sure you want to continue?
        """)
        noButton, yesButton = msg.warning(
            self, 'Process still running', txt, 
            buttonsTexts=(
                'No, wait for the process to end', 
                'Yes, close Cell-ACDC'
            )
        )
        if msg.cancel:
            return False
        return msg.clickedButton == yesButton

    def closeEvent(self, event):
        if self.isSegmWorkerRunning:
            proceed = self.warnSegmWorkerStillRunning()
            if not proceed:
                event.ignore()
                return
            self.numThreadsRunning = 0
            self.segmWorker.signals.finished.emit(-1)
            self.mainWin.forceClose = True
            self.mainWin.close()
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
        self.sigClosed.emit()
