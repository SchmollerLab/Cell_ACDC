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

from PyQt5.QtWidgets import QApplication, QPushButton
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

# Custom modules
import prompts, load, myutils, apps
myutils.download_model('YeaZ')

exp_path = prompts.folder_dialog(
                title='Select experiment folder containing Position_n folders'
                      'or specific Position_n folder')

if exp_path == '':
    exit('Execution aborted by the user')

if os.path.basename(exp_path).find('Position_') != -1:
    is_pos_folder = True
else:
    is_pos_folder = False

if os.path.basename(exp_path).find('Images') != -1:
    is_images_folder = True
else:
    is_images_folder = False

print('Loading data...')

app = QApplication(sys.argv)
app.setStyle(QtGui.QStyleFactory.create('Fusion'))

# Ask which model
msg = QtGui.QMessageBox()
toFront = msg.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
msg.setWindowState(toFront)
font = QtGui.QFont()
font.setPointSize(10)
msg.setFont(font)
msg.activateWindow()
msg.setIcon(msg.Question)
msg.setText('Which model do you want to use for segmentation?')
yeazButton = QPushButton('YeaZ (phase contrast yeast cells)')
cellposeButton = QPushButton('Cellpose (all other cells types, e.g. mammalian)')
msg.addButton(yeazButton, msg.YesRole)
msg.addButton(cellposeButton, msg.YesRole)
msg.exec_()
if msg.clickedButton() == yeazButton:
    model = 'yeaz'
    yeazParams = apps.YeaZ_ParamsDialog()
    yeazParams.setFont(font)
    yeazParams.exec_()
    thresh_val = yeazParams.threshVal
    min_distance = yeazParams.minDist
    # YeaZ modules
    print('Importing YeaZ...')
    from YeaZ.unet import neural_network as nn
    from YeaZ.unet import segment
    from YeaZ.unet import tracking
elif msg.clickedButton() == cellposeButton:
    model = 'cellpose'
    exit('Cellpose model not implemented yet.')

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
        exit('Execution aborted by the user')

    select_folder.QtPrompt(None, values, allow_abort=False, show=True)
    if select_folder.was_aborted:
        exit('Execution aborted by the user')

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
answer = msg.question(None, 'Save?', 'Do you want to save segmentation?',
                      msg.Yes | msg.No | msg.Cancel)
if answer == msg.Yes:
    save = True
elif answer == msg.No:
    save = False
else:
    exit('Execution aborted by the user')

ch_name_not_found_msg = (
    'The script could not identify the channel name.\n\n'
    'For automatic loading the file to be segmented MUST have a name like\n'
    '"<name>_s<num>_<channel_name>.tif" e.g. "196_s16_phase_contrast.tif"\n'
    'where "196_s16" is the basename and "phase_contrast"'
    'is the channel name\n\n'
    'Please write here the channel name to be used for automatic loading'
)

first_call = True
for images_path in images_paths:
    print(f'Processing {images_path}')
    filenames = os.listdir(images_path)
    if ch_name_selector.is_first_call:
        ch_names, warn = ch_name_selector.get_available_channels(filenames)
        ch_name_selector.QtPrompt(None, ch_names)
        if ch_name_selector.was_aborted:
            exit('Execution aborted by the user')
        if warn:
            user_ch_name = prompts.single_entry_messagebox(
                title='Channel name not found',
                entry_label=ch_name_not_found_msg,
                input_txt=ch_name_selector.channel_name,
                toplevel=False, allow_abort=False
            ).entry_txt
            if user_ch_name.was_aborted:
                exit('Execution aborted by the user')
        else:
            user_ch_name = ch_name_selector.channel_name

    aligned_npz_found = False
    for filename in filenames:
        if filename.find(f'{user_ch_name}_aligned.npz') != -1:
            img_path = os.path.join(images_path, filename)
            aligned_npz_found = True
            break

    if not aligned_npz_found:
        print(f'WARNING: The folder {images_path} does not contain the file '
              f'{user_ch_name}_aligned.npz. Skipping it.')
        continue

    data = load.load_frames_data(img_path, user_ch_name,
                                 load_segm_data=False,
                                 load_acdc_df=False,
                                 load_zyx_voxSize=False,
                                 loadSegmInfo=True,
                                 first_call=first_call)

    if data.SizeZ > 1:
        if data.segmInfo_df is None:
            print(f'WARNING: The image data in {img_path} is 3D but '
                  f'_segmInfo.csv file not found. Skipping this position.')
            continue
        else:
            zz = data.segmInfo_df.loc['z_slice_used_dataPrep'].to_list()

    if first_call:
        # Ask stop frame
        win = apps.QLineEditDialog(
            title='Stop frame',
            msg='Frame number to stop segmentation?\n '
                f'(insert number between 1 and {data.SizeT})',
            defaultTxt=str(data.SizeT))
        win.setFont(font)
        win.exec_()
        if win.cancel:
            exit('Execution aborted by the user')

        stop_i = int(win.EntryID)

    first_call=False

    print('Preprocessing image(s)...')
    if data.SizeT > 1:
        if data.SizeZ > 1:
            # 3D data over time
            img_data = data.img_data[range(stop_i), zz[:stop_i]]
        else:
            # 2D data over time
            img_data = data.img_data[:stop_i]
        img_data = np.array([skimage.exposure.equalize_adapthist(img)
                             for img in img_data])
    else:
        if data.SizeZ > 1:
            # Single 3D image
            img_data = skimage.exposure.equalize_adapthist(data.img_data[zz[0]])
        else:
            # Single 2D image
            img_data = skimage.exposure.equalize_adapthist(img_data)

    print(f'Image shape = {img_data.shape}')


    """Segmentation routine"""
    t0 = time.time()
    path_weights = nn.determine_path_weights()
    print('Running UNet for Segmentation:')
    if data.SizeT > 1:
        if model == 'yeaz':
            pred_stack = nn.batch_prediction(img_data, is_pc=True,
                                             path_weights=path_weights,
                                             batch_size=1)
        elif model == 'cellpose':
            exit('Cellpose model not implemented yet.')
    else:
        if model == 'yeaz':
            pred_stack = nn.prediction(img_data, is_pc=True,
                                       path_weights=path_weights)
        elif model == 'cellpose':
            exit('Cellpose model not implemented yet.')

    if model == 'yeaz':
        print('Thresholding prediction...')
        thresh_stack = nn.threshold(pred_stack, th=thresh_val)

    if data.SizeT > 1:
        if model == 'yeaz':
            lab_stack = segment.segment_stack(thresh_stack, pred_stack,
                                              min_distance=min_distance
                                              ).astype(np.uint16)
        elif model == 'cellpose':
            exit('Cellpose model not implemented yet.')
    else:
        if model == 'yeaz':
            lab_stack = segment.segment(thresh_stack, pred_stack,
                                        min_distance=min_distance
                                        ).astype(np.uint16)
        elif model == 'cellpose':
            exit('Cellpose model not implemented yet.')

    lab_stack = skimage.morphology.remove_small_objects(lab_stack, min_size=5)

    if data.SizeT > 1:
        print('Tracking cells...')
        # NOTE: We use yeaz tracking also for cellpose
        tracked_stack = tracking.correspondence_stack(
                                            lab_stack).astype(np.uint16)
    else:
        tracked_stack = lab_stack

    if save:
        print('')
        print('Saving...')
        np.savez_compressed(data.segm_npz_path, tracked_stack)

    t_end = time.time()

    exec_time = t_end-t0
    exec_time_min = exec_time/60
    exec_time_delta = datetime.timedelta(seconds=exec_time)
    print(f'{images_path} successfully segmented in {exec_time_delta} HH:mm:ss')
    print('-----------------------------')
