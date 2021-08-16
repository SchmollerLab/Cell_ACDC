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
    QVBoxLayout, QPushButton, QLabel
)
from PyQt5.QtCore import Qt, QEventLoop
from PyQt5 import QtGui

# Custom modules
import prompts, load, myutils, apps, core, dataPrep

import qrc_resources

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.yeastacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass

class segmWin(QMainWindow):
    def __init__(self, parent=None, allowExit=False,
                 buttonToRestore=None, mainWin=None):
        self.allowExit = allowExit
        self.processFinished = False
        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin
        super().__init__(parent)
        self.setWindowTitle("Yeast ACDC - Segment")
        self.setWindowIcon(QtGui.QIcon(":assign-motherbud.svg"))

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()

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

        self.setWindowTitle(f'Yeast_ACDC - GUI - "{exp_path}"')

        if os.path.basename(exp_path).find('Position_') != -1:
            is_pos_folder = True
        else:
            is_pos_folder = False

        if os.path.basename(exp_path).find('Images') != -1:
            is_images_folder = True
        else:
            is_images_folder = False

        print('Loading data...')

        # Ask which model
        font = QtGui.QFont()
        font.setPointSize(10)
        model = prompts.askWhichSegmModel(parent=self)
        if model == 'yeaz':
            yeazParams = apps.YeaZ_ParamsDialog(parent=self)
            yeazParams.setFont(font)
            yeazParams.exec_()
            if yeazParams.cancel:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return

            thresh_val = yeazParams.threshVal
            min_distance = yeazParams.minDist
            # YeaZ modules
            print('Importing YeaZ...')
            from YeaZ.unet import neural_network as nn
            from YeaZ.unet import segment
            from YeaZ.unet import tracking
            myutils.download_model('YeaZ')
            path_weights = nn.determine_path_weights()
        elif model == 'cellpose':
            cellposeParams = apps.cellpose_ParamsDialog(parent=self)
            cellposeParams.setFont(font)
            cellposeParams.exec_()
            if cellposeParams.cancel:
                abort = self.doAbort()
                if abort:
                    self.close()
                    return

            diameter = cellposeParams.diameter
            if diameter==0:
                diameter=None
            flow_threshold = cellposeParams.flow_threshold
            cellprob_threshold = cellposeParams.cellprob_threshold
            # Cellpose modules
            print('Importing cellpose...')
            from acdc_cellpose import models
            myutils.download_model('cellpose')
            device, gpu = models.assign_device(True, False)
            cp_model = models.Cellpose(gpu=gpu, device=device,
                                       model_type='cyto', torch=True)

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

        # Ask to save?
        msg = QtGui.QMessageBox()
        msg.setFont(font)
        answer = msg.question(self, 'Save?', 'Do you want to save segmentation?',
                              msg.Yes | msg.No | msg.Cancel)
        if answer == msg.Yes:
            save = True
        elif answer == msg.No:
            save = False
        else:
            abort = self.doAbort()
            if abort:
                self.close()
                return


        ch_name_not_found_msg = (
            'The script could not identify the channel name.\n\n'
            'For automatic loading the file to be segmented MUST have a name like\n'
            '"<name>_s<num>_<channel_name>.tif" e.g. "196_s16_phase_contrast.tif"\n'
            'where "196_s16" is the basename and "phase_contrast"'
            'is the channel name\n\n'
            'Please write here the channel name to be used for automatic loading'
        )

        user_ch_file_paths = []
        for images_path in images_paths:
            print('')
            print(f'Processing {images_path}')
            filenames = os.listdir(images_path)
            if ch_name_selector.is_first_call:
                ch_names, warn = ch_name_selector.get_available_channels(filenames)
                ch_name_selector.QtPrompt(self, ch_names)
                if ch_name_selector.was_aborted:
                    abort = self.doAbort()
                    if abort:
                        self.close()
                        return

                if warn:
                    user_ch_name = prompts.single_entry_messagebox(
                        title='Channel name not found',
                        entry_label=ch_name_not_found_msg,
                        input_txt=ch_name_selector.channel_name,
                        toplevel=False, allow_abort=False
                    ).entry_txt
                    if user_ch_name.was_aborted:
                        abort = self.doAbort()
                        if abort:
                            self.close()
                            return

                else:
                    user_ch_name = ch_name_selector.channel_name

            aligned_npz_found = False
            for filename in filenames:
                if filename.find(f'{user_ch_name}_aligned.npz') != -1:
                    img_path = os.path.join(images_path, filename)
                    aligned_npz_found = True
                elif filename.find(f'{user_ch_name}.tif') != -1:
                    img_path = os.path.join(images_path, filename)

            if not aligned_npz_found:
                print(f'WARNING: The folder {images_path} does not contain the file '
                      f'{user_ch_name}_aligned.npz. Segmenting .tif data.')

            user_ch_file_paths.append(img_path)

        first_call = True
        for img_path in tqdm(user_ch_file_paths, unit=' Position', ncols=100):
            data = load.loadData(img_path, user_ch_name, QParent=self)
            data.getBasenameAndChNames(prompts.select_channel_name)
            data.buildPaths()
            data.loadImgData()
            data.loadOtherFiles(
                               load_segm_data=False,
                               load_acdc_df=False,
                               load_shifts=False,
                               loadSegmInfo=True,
                               load_delROIsInfo=False,
                               loadDataPrepBkgrVals=False,
                               load_last_tracked_i=False,
                               load_metadata=True
            )
            if first_call:
                proceed = data.askInputMetadata(
                                            ask_TimeIncrement=False,
                                            ask_PhysicalSizes=False,
                                            save=True)
                self.SizeT = data.SizeT
                self.SizeZ = data.SizeZ
                if not proceed:
                    abort = self.doAbort()
                    if abort:
                        self.close()
                        return
            else:
                data.SizeT = self.SizeT
                data.SizeZ = self.SizeZ
            if data.SizeZ > 1:
                if data.segmInfo_df is None:
                    print('')
                    print(f'WARNING: The image data in {img_path} is 3D but '
                          f'_segmInfo.csv file not found. Launching dataPrep.py...')
                    dataPrepWin = dataPrep.dataPrepWin()
                    dataPrepWin.show()
                    dataPrepWin.titleText = (
                        'Select z-slice (or projection) for each frame/position.'
                        'Once happy, close the window to continue segmentation process.'
                    )
                    dataPrepWin.initLoading()
                    dataPrepWin.loadFiles(
                        exp_path, user_ch_file_paths, user_ch_name)
                    loop = QEventLoop(self)
                    dataPrepWin.loop = loop
                    loop.exec_()
                    data = load.loadData(img_path, user_ch_name, QParent=self)
                    data.getBasenameAndChNames(prompts.select_channel_name)
                    data.buildPaths()
                    data.loadImgData()
                    data.loadOtherFiles(
                                       load_segm_data=False,
                                       load_acdc_df=False,
                                       load_shifts=False,
                                       loadSegmInfo=True,
                                       load_delROIsInfo=False,
                                       loadDataPrepBkgrVals=False,
                                       load_last_tracked_i=False,
                                       load_metadata=True
                    )
                else:
                    zz = data.segmInfo_df['z_slice_used_dataPrep'].to_list()

            if first_call and data.SizeT > 1:
                # Ask stop frame
                win = apps.QLineEditDialog(
                    parent=self,
                    title='Stop frame',
                    msg='Frame number to stop segmentation?\n '
                        f'(insert number between 1 and {data.SizeT})',
                    defaultTxt=str(data.SizeT))
                win.setFont(font)
                win.exec_()
                if win.cancel:
                    abort = self.doAbort()
                    if abort:
                        self.close()
                        return


                stop_i = int(win.EntryID)

            first_call=False

            if data.SizeT > 1:
                if data.SizeZ > 1:
                    # 3D data over time
                    img_data = data.img_data[:stop_i]
                    for i, z_info in enumerate(data.segmInfo_df[:stop_i].itertuples()):
                        z = z_info.z_slice_used_dataPrep
                        zProjHow = z_info.which_z_proj
                        img = img_data[i]
                        if zProjHow == 'single z-slice':
                            img_data[i] = img[z]
                        elif zProjHow == 'max z-projection':
                            img_data[i] = img.max(axis=0)
                        elif zProjHow == 'mean z-projection':
                            img_data[i] = img.mean(axis=0)
                        elif zProjHow == 'median z-proj.':
                            img_data[i] = np.median(img, axis=0)
                else:
                    # 2D data over time
                    img_data = data.img_data[:stop_i]
                img_data = [img/img.max() for img in img_data]
                img_data = np.array([skimage.exposure.equalize_adapthist(img)
                                     for img in img_data])
            else:
                if data.SizeZ > 1:
                    # Single 3D image
                    z_info = data.segmInfo_df.iloc[0]
                    z = z_info.z_slice_used_dataPrep
                    zProjHow = z_info.which_z_proj
                    if zProjHow == 'single z-slice':
                        img_data = data.img_data[z]
                    elif zProjHow == 'max z-projection':
                        img_data = data.img_data.max(axis=0)
                    elif zProjHow == 'mean z-projection':
                        img_data = data.img_data.mean(axis=0)
                    elif zProjHow == 'median z-proj.':
                        img_data = np.median(data.img_data, axis=0)
                    img_data = skimage.exposure.equalize_adapthist(
                                                    img_data/img_data.max())
                else:
                    # Single 2D image
                    img_data = skimage.exposure.equalize_adapthist(
                                                    img_data/img_data.max())

            print(f'Image shape = {img_data.shape}')

            """Segmentation routine"""
            t0 = time.time()
            print(f'Segmenting with {model} (Ctrl+C to abort)...')
            if data.SizeT > 1:
                if model == 'yeaz':
                    pred_stack = nn.batch_prediction(img_data, is_pc=True,
                                                     path_weights=path_weights,
                                                     batch_size=1)
                elif model == 'cellpose':
                    lab_stack = np.array(img_data.shape, np.uint16)
                    for t, img in enumerate(img_data):
                        lab, flows, _, _ = cp_model.eval(
                                        img,
                                        channels=[0,0],
                                        diameter=diameter,
                                        flow_threshold=flow_threshold,
                                        cellprob_threshold=cellprob_threshold
                        )
                        # lab = core.smooth_contours(lab, radius=2)
                        lab_stack[t] = lab

            else:
                if model == 'yeaz':
                    pred_stack = nn.prediction(img_data, is_pc=True,
                                               path_weights=path_weights)
                elif model == 'cellpose':
                    lab_stack, flows, _, _ = cp_model.eval(
                                        img_data,
                                        channels=[0,0],
                                        diameter=diameter,
                                        flow_threshold=flow_threshold,
                                        cellprob_threshold=cellprob_threshold
                    )
                    # lab_stack = core.smooth_contours(lab_stack, radius=2)
            if model == 'yeaz':
                print('Thresholding prediction...')
                thresh_stack = nn.threshold(pred_stack, th=thresh_val)

            if data.SizeT > 1:
                if model == 'yeaz':
                    lab_stack = segment.segment_stack(thresh_stack, pred_stack,
                                                      min_distance=min_distance
                                                      ).astype(np.uint16)
            else:
                if model == 'yeaz':
                    lab_stack = segment.segment(thresh_stack, pred_stack,
                                                min_distance=min_distance
                                                ).astype(np.uint16)

            lab_stack = skimage.morphology.remove_small_objects(lab_stack, min_size=5)

            if data.SizeT > 1:
                print('Tracking cells...')
                # NOTE: We use yeaz tracking also for cellpose
                tracked_stack = tracking.correspondence_stack(lab_stack).astype(np.uint16)
            else:
                tracked_stack = lab_stack

            if save:
                print('')
                print('Saving...')
                np.savez_compressed(data.segm_npz_path, tracked_stack)

            t_end = time.time()

            exec_time = t_end-t0
            exec_time_min = exec_time/60
            exec_time_delta = datetime.timedelta(seconds=round(exec_time))
            print(f'{images_path} successfully segmented in {exec_time_delta} HH:mm:ss')
            print('-----------------------------')

        self.processFinished = True
        self.close()
        if self.allowExit:
            exit('Segmentation task ended.')

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
        if self.buttonToRestore is not None:
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            toFront = self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
            self.mainWin.setWindowState(toFront)
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
