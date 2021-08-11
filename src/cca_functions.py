import numpy as np
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
from tifffile import imread
import os
import glob
from tqdm import tqdm

import prompts
import apps
from pyqtgraph.Qt import QtGui
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
import sys
import difflib


def configuration_dialog():
    continue_selection = True
    data_dirs = []
    positions = []
    while continue_selection:
        data_dir = prompts.folder_dialog(title='Select folder containing Position_n folders')
        if data_dir != '':
            available_pos = sorted(os.listdir(data_dir))
            app = QtCore.QCoreApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            win = apps.QDialogListbox('Position Selection', 'Select which position you want to analyse', available_pos)
            app.setStyle(QtGui.QStyleFactory.create('Fusion'))
            win.show()
            app.exec_()
            pos = win.selectedItemsText
            data_dirs.append(data_dir)
            positions.append(pos)
            continue_selection = prompts.askyesno(message= 'Do you wish to select another file?', title= 'Selection of further files')
        else:
            continue_selection = False
    if len(data_dirs) == 0:
        print("No positions selected!")
        raise IndexError
    return data_dirs, positions

def find_available_channels(filenames):
    ch_name_selector = prompts.select_channel_name()
    ch_names, warn = ch_name_selector.get_available_channels(filenames)
    return ch_names, warn

def calculate_downstream_data(
    file_names,
    image_folders,
    positions,
    channels,
    force_recalculation=False
):
    no_of_channels = len(channels)
    overall_df = pd.DataFrame()
    for file_idx, file in enumerate(file_names):
        for pos_idx, pos_dir in enumerate(image_folders[file_idx]):
            channel_data = ('placeholder')*no_of_channels
            print(f'Load files for {file}, {positions[file_idx][pos_idx]}...')
            *channel_data, seg_mask, cc_data, cc_props = _load_files(pos_dir, channels)
            print(f'Number of cells in position: {len(cc_data.Cell_ID.unique())}')
            print(f'Number of annotated frames in position: {cc_data.frame_i.max()+1}')
            cc_data = _rename_columns(cc_data)
            max_frame = cc_data.frame_i.max()
            if cc_props is not None and force_recalculation==False:
                print('Cell Cycle property data already existing, loaded from disk...')
                overall_df = overall_df.append(cc_props).reset_index(drop=True)
            else:
                print(f'Calculate regionprops on each frame based on Segmentation...')
                rp_df = _calculate_rp_df(seg_mask[:max_frame+1])
                print(f'Calculate mean signal strength for every channel and cell...')
                flu_signal_df = _calculate_flu_signal(seg_mask, channel_data, channels, cc_data)
                temp_df = cc_data.merge(rp_df, on=['frame_i', 'Cell_ID'], how='left')
                temp_df = temp_df.merge(flu_signal_df, on=['frame_i', 'Cell_ID'], how='left')
                # calculate amount of corrected signal by multiplying mean with area
                for channel in channels:
                    temp_df[f'{channel}_corrected_amount'] = temp_df[f'{channel}_corrected_mean'] *\
                    temp_df['area']
                    temp_df[f'{channel}_corrected_concentration'] = temp_df[f'{channel}_corrected_amount']/\
                    temp_df['cell_vol_fl']
                temp_df['max_frame_pos'] = cc_data.frame_i.max()
                temp_df['file'] = file
                temp_df['selection_subset'] = file_idx
                temp_df['position'] = positions[file_idx][pos_idx]
                temp_df['directory'] = pos_dir
                print('Saving calculated data for next time...')
                files_in_curr_dir = os.listdir(pos_dir)
                common_prefix = _determine_common_prefix(files_in_curr_dir)
                save_path = os.path.join(pos_dir, f'{common_prefix}cca_properties_downstream.csv')
                temp_df.to_csv(save_path, index=False)
                overall_df = overall_df.append(temp_df).reset_index(drop=True)
    return overall_df


def _determine_common_prefix(filenames):
    basename = filenames[0]
    for file in filenames:
        # Determine the basename based on intersection of all .tif
        _, ext = os.path.splitext(file)
        sm = difflib.SequenceMatcher(None, file, basename)
        i, j, k = sm.find_longest_match(0, len(file),
                                        0, len(basename))
        basename = file[i:i+k]
    return basename


def _auto_rescale_intensity(img, perc=0.01, clip_min=False):
    """
    function to clip outliers to the given percentiles and scale afterwards.

    use skimage.exposure.rescale_intensity for clipping all images in the
    given percentiles to the percentile border.

    Parameters
    ----------
    img: np.array
        image to eliminate outliers from
    perc: float
        percentage of pixels which are considered as outliers.
        Example: given 0.01, the lowest and highest 1% of pixels are considered
        as outliers and clipped to the 1- and 99-percentile respectively.

    Returns
    -------
    scaled_img: np.array
        image, where outlier pixels are set to 1- and 99-percentiles and which is
        scaled to [0,1] afterwards
    """
    if perc > 0:
        vmin, vmax = np.percentile(img, q=(perc, 100-perc))
        clip_min_indices = img < vmin
        clip_max_indices = img > vmax
        if clip_min:
            img[clip_min_indices] = vmin
        img[clip_max_indices] = vmax
    else:
        vmin, vmax = img.min(), img.max()
    scaled_img = (img-vmin)/(vmax-vmin)
    return scaled_img


def _load_files(file_dir, channels):
    """
    Function to load files of all given channels and the corresponding segmentation masks.
    Check first if aligned files are available and use them if so.
    """
    no_of_aligned_files = len(glob.glob(f'{file_dir}\*aligned*'))
    channel_files = []
    if no_of_aligned_files > 0:
        for channel in channels:
            try:
                ch_aligned_path = glob.glob(f'{file_dir}\*{channel}_aligned.npz*')[0]
                channel_files.append(np.load(ch_aligned_path)['arr_0'])
            except IndexError:
                try:
                    ch_aligned_path = glob.glob(f'{file_dir}\*{channel}_aligned.npy*')[0]
                    channel_files.append(np.load(ch_aligned_path))
                except IndexError:
                    print(f'Could not find an aligned file for channel {channel}')
                    print(f'Resulting data will not contain fluorescent data for this channel')
                    channel_files.append(None)
    else:
        for channel in channels:
            try:
                ch_not_aligned_path = glob.glob(f'{file_dir}\*{channel}*')[0]
                channel_files.append(imread(ch_not_aligned_path))
            except IndexError:
                print(f'Could not find any file for channel {channel}')
                print(f'Resulting data will not contain fluorescent data for this channel')
                channel_files.append(None)

    # append segmentation file
    try:
        segm_file_path = glob.glob(f'{file_dir}\*_segm.npz')[0]
        channel_files.append(np.load(segm_file_path)['arr_0'])
    except IndexError:
        segm_file_path = glob.glob(f'{file_dir}\*_segm.npy')[0]
        # assume segmentation mask to be .npy
        channel_files.append(np.load(segm_file_path))
    # append cc-data
    try:
        cc_stage_path = glob.glob(f'{file_dir}\*acdc_output*')[0]
    except IndexError:
        cc_stage_path = glob.glob(f'{file_dir}\*cc_stage*')[0]
    # assume cell cycle output of ACDC to be .csv
    channel_files.append(pd.read_csv(cc_stage_path))

    # append cc-properties if available, else append None
    if len(glob.glob(f'{file_dir}\*_downstream*')) > 0:
        cc_props_path = glob.glob(f'{file_dir}\*_downstream*')[0]
        # assume calculated cc properties to be .csv
        channel_files.append(pd.read_csv(cc_props_path))
    else:
        channel_files.append(None)
    return tuple(channel_files)




def _calculate_rp_df(input_sequence, label_input=False):
    if label_input:
        #generate labeled video only when input is not labeled yet
        labeled_video = label(input_sequence)
    else:
        labeled_video = input_sequence.copy()
    # calculate rp's for rings
    t_df = pd.DataFrame()
    props = ('label', 'area', 'convex_area', 'filled_area','major_axis_length',
             'minor_axis_length', 'orientation', 'perimeter', 'centroid', 'solidity')
    rename_dict = {'label':'Cell_ID', 'centroid-0':'centroid_y', 'centroid-1':'centroid_x'}
    for t, img in enumerate(tqdm(labeled_video)):
        # build time-dependent dataframes for further use (later for cca)
        if img.max() > 0:
            t_rp = pd.DataFrame(regionprops_table(img.astype(int), properties=props)).rename(columns=rename_dict)
            t_rp['frame_i'] = t
            # determine id's which are falsely merged by 3D-labeling
            for r_id in t_rp.Cell_ID.unique():
                bin_label = label((img==r_id).astype(int))
                t_rp.loc[t_rp['Cell_ID']==r_id, '2d_label_count'] = bin_label.max()
            t_df = t_df.append(t_rp, ignore_index=True)
    # calculate global features by grouping
    grouped_df = t_df.groupby('Cell_ID').agg(
        min_t=('frame_i', min),
        max_t=('frame_i', max),
        lifespan=('frame_i', lambda x: max(x)-min(x)+1)
    ).reset_index()
    merged_df = t_df.merge(grouped_df, how='left', on='Cell_ID')
    # calculate further indicators based on merged data
    merged_df['age'] = merged_df['frame_i'] - merged_df['min_t'] + 1
    merged_df['frames_till_gone'] = merged_df['max_t'] - merged_df['frame_i']
    merged_df['elongation'] = merged_df['major_axis_length']/merged_df['minor_axis_length']
    return merged_df


def _calculate_flu_signal(seg_mask, channel_data, channels, cc_data):
    """
    function to calculate sum and scaled sum of fluorescence signal per frame and cell.
    channel_data is a list-like of TYX arrays, one for each channel.
    channels are the name of the channels in the tuple.
    cc_data the output of acdc.
    """
    max_frame = cc_data.frame_i.max()
    df = pd.DataFrame(columns=['frame_i', 'Cell_ID'])
    bg_medians = []
    for ch_idx, ch_array in enumerate(channel_data):
        if ch_array is None:
            bg_medians.append(None)
        else:
            bg_index = np.logical_and(seg_mask[:max_frame+1]==0, ch_array[:max_frame+1]!=0)
            ch_medians = [np.median(ch_array[t][bg_index[t]]) for t in range(max_frame+1)]
            bg_medians.append(ch_medians)
    for cell_id in tqdm(cc_data.Cell_ID.unique()):
        temp_df = pd.DataFrame(columns=['frame_i', 'Cell_ID'])
        times = range(max_frame+1)
        temp_df['frame_i'] = times; temp_df['Cell_ID'] = cell_id
        index_array = (seg_mask[:max_frame+1] == cell_id)
        channel_data_cut = [c_arr[:max_frame+1] if c_arr is not None else None for c_arr in channel_data]
        for c_idx, c_array in enumerate(channel_data_cut):
            if c_array is not None:
                cell_signal = c_array*index_array
                summed = np.sum(cell_signal, axis=(1,2))
                count = np.sum(cell_signal!=0, axis=(1,2))
                mean_signal = np.divide(summed, count, where=count!=0)
                corrected_signal = mean_signal - np.array(bg_medians[c_idx])
                temp_df[f'{channels[c_idx]}_corrected_mean'] = np.clip(corrected_signal, 0, np.inf)
            else:
                temp_df[f'{channels[c_idx]}_corrected_mean'] = 0
        df = df.append(temp_df, ignore_index=True)
    signal_indices = np.array(['_corrected_mean' in col for col in df.columns])
    keep_rows = df.loc[:,signal_indices].sum(axis=1) > 0
    df = df[keep_rows]
    df = df.sort_values(['frame_i', 'Cell_ID']).reset_index(drop=True)
    return df


def _rename_columns(cc_data):
    rename_dict = {
        'Cell cycle stage': 'cell_cycle_stage',
        '# of cycles': 'generation_num',
        "Relative's ID": 'relative_ID',
        'Relationship': 'relationship',
        'Emerg_frame_i': 'emerg_frame_i',
        'Division_frame_i': 'division_frame_i',
        'Discard': 'is_cell_excluded'
    }
    cc_data.columns = [rename_dict.get(col, col) for col in cc_data.columns]
    return cc_data
