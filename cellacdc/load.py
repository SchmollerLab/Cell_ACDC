import os
import sys
from typing import List
import traceback
import tempfile
import re
import cv2
import json
import h5py
import shutil
from math import isnan
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
import h5py
import pandas as pd
import skimage.filters
from datetime import datetime
from tifffile import TiffFile
import tifffile
import zipfile
from natsort import natsorted

import skimage
import skimage.io
import skimage.measure

from . import GUI_INSTALLED

if GUI_INSTALLED:
    from qtpy import QtGui
    from qtpy.QtCore import Qt, QRect, QRectF
    from qtpy.QtWidgets import (
        QApplication, QMessageBox, QFileDialog
    )
    import pyqtgraph as pg
    pg.setConfigOption('imageAxisOrder', 'row-major')
    from . import apps
    from . import widgets
    from . import qrc_resources_path, qrc_resources_light_path
    from . import qrc_resources_dark_path
    
    
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from . import prompts
from . import myutils, measurements, config
from . import base_cca_dict, base_acdc_df, html_utils, settings_folderpath
from . import cca_df_colnames, printl
from . import ignore_exception, cellacdc_path
from . import models_path
from . import tooltips_rst_filepath
from . import cca_functions
from . import sorted_cols

acdc_df_bool_cols = [
    'is_cell_dead',
    'is_cell_excluded',
    'is_history_known',
    'corrected_assignment'
]

acdc_df_str_cols = {'cell_cycle_stage': str, 'relationship': str}

additional_metadata_path = os.path.join(settings_folderpath, 'additional_metadata.json')
last_entries_metadata_path = os.path.join(settings_folderpath, 'last_entries_metadata.csv')
last_selected_groupboxes_measurements_path = os.path.join(
    settings_folderpath, 'last_selected_groupboxes_set_measurements.json'
)
channel_file_formats = (
    '_aligned.h5', '.h5', '_aligned.npz', '.tif'
)
ISO_TIMESTAMP_FORMAT = r'iso%Y%m%d%H%M%S'

def read_json(json_path, logger_func=print, desc='custom annotations'):
    json_data = {}
    try:
        with open(json_path) as file:
            json_data = json.load(file)
    except Exception as e:
        print('****************************')
        logger_func(traceback.format_exc())
        print('****************************')
        logger_func(f'json path: {json_path}')
        print('----------------------------')
        logger_func(f'Error while reading saved "{desc}". See above')
        print('============================')
    return json_data

def remove_duplicates_file(filepath):
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r') as file:
        first_line = file.readline()
        rest_of_text = file.read()
        duplicate_first_line_idx = rest_of_text.find(first_line)
        if duplicate_first_line_idx == -1:
            return
        unique_text = f'{first_line}{rest_of_text[:duplicate_first_line_idx]}'
    with open(filepath, 'w') as file:
        file.write(unique_text)

def to_csv_through_temp(df, csv_path):
    filename = os.path.basename(csv_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_filepath = os.path.join(temp_dir, filename)
        df.to_csv(tmp_filepath)
        shutil.copy2(tmp_filepath, csv_path)

def get_all_acdc_folders(user_profile_path):
    models = myutils.get_list_of_models()
    acdc_folders = [f'acdc-{model}' for model in models]
    acdc_folders.append('acdc-java')
    acdc_folders.append('.acdc-logs')
    acdc_folders.append('.acdc-settings')
    acdc_folders.append('acdc-manual')
    acdc_folders.append('acdc-metrics')
    acdc_folders.append('acdc-examples')
    existing_acdc_folders = []
    for file in os.listdir(user_profile_path):
        filepath = os.path.join(user_profile_path, file)
        if not os.path.isdir(filepath):
            continue
        if file not in acdc_folders:
            continue
        existing_acdc_folders.append(file)
    return existing_acdc_folders

def write_json(json_data, json_path, indent=2):
    with open(json_path, mode='w') as file:
        json.dump(json_data, file, indent=indent)

def read_last_selected_gb_meas(logger_func=print):
    data = {}
    if not os.path.exists(last_selected_groupboxes_measurements_path):
        write_json(data, last_selected_groupboxes_measurements_path)
    else:
        data = read_json(
            last_selected_groupboxes_measurements_path,
            desc='last selected channels (set measurments)',
            logger_func=logger_func
        )
    return data

def migrate_models_paths(dst_path):
    models = myutils.get_list_of_models()
    user_profile_path = dst_path.replace('\\', '/')
    for model in models:
        model_path = os.path.join(models_path, model, 'model')
        weight_location_txt_path = os.path.join(
            model_path, 'weights_location_path.txt'
        )
        if not os.path.exists(weight_location_txt_path):
            continue
        with open(weight_location_txt_path, 'r') as txt:
            model_location = os.path.expanduser(txt.read())
        model_location = model_location.replace('\\', '/')
        model_folder = os.path.basename(model_location)
        model_location = os.path.join(user_profile_path, model_folder)
        model_location = model_location.replace('\\', '/')
        with open(weight_location_txt_path, 'w') as txt:
            txt.write(model_location)

def save_segm_workflow_to_config(
        filepath, ini_items: dict, paths_to_segm: list, 
        stop_frame_nums: list
    ):
    paths_to_segm = [path.replace('\\', '/') for path in paths_to_segm]
    paths_param = '\n'.join(paths_to_segm)
    paths_param = f'\n{paths_param}'
    configPars = config.ConfigParser()
    configPars['paths_to_segment'] = {'paths': paths_param} 
    
    stop_frames_param = '\n'.join([str(n) for n in stop_frame_nums])
    stop_frames_param = f'\n{stop_frames_param}'
    configPars['paths_to_segment']['stop_frame_numbers'] = stop_frames_param
    
    for section, options in ini_items.items():
        configPars[section] = {}
        for option, value in options.items():
            configPars[section][option] = str(value)
    with open(filepath, 'w') as configfile:
        configPars.write(configfile)

def read_segm_workflow_from_config(filepath):
    configPars = config.ConfigParser()
    configPars.read(filepath)
    ini_items = {}
    for section in configPars.sections():
        options = dict(configPars[section])
        ini_items[section] = {}
        for option, value in options.items():
            if section == 'paths_to_segment':
                value = value.strip('\n')
                value = value.split('\n')
                ini_items[section][option] = value
                continue
            if value == 'False':
                value = False
            elif value == 'True':
                value = True
            elif value == 'None':
                value = None
            elif option == 'SizeT' or option == 'SizeZ':
                value = int(value)
                
            if section == 'standard_postprocess_features':
                for _type in (int, float, str):
                    try:
                        value = _type(value)
                        break
                    except Exception as e:
                        continue
            
            elif section == 'custom_postprocess_features':
                value = tuple([float(val) for val in value])
            
            ini_items[section][option] = value
    return ini_items

def get_images_paths(self, folder_path):
    folder_type = myutils.determine_folder_type(folder_path)     
    is_pos_folder, is_images_folder, folder_path = folder_type     
    if not is_pos_folder and not is_images_folder:
        pos_foldernames = myutils.get_pos_foldernames(folder_path)
        images_paths = [
            os.path.join(folder_path, pos, 'Images') for pos in pos_foldernames
        ]
    elif is_pos_folder:
        images_paths = [os.path.join(folder_path, 'Images')]
    elif is_images_folder:
        images_paths = [folder_path]
    return images_paths
            
def save_last_selected_gb_meas(json_data):
    write_json(json_data, last_selected_groupboxes_measurements_path)

def read_config_metrics(ini_path):
    configPars = config.ConfigParser()
    configPars.read(ini_path)
    if 'equations' not in configPars:
        configPars['equations'] = {}

    if 'mixed_channels_equations' not in configPars:
        configPars['mixed_channels_equations'] = {}

    if 'user_path_equations' not in configPars:
        configPars['user_path_equations'] = {}
    
    return configPars

def add_configPars_metrics(configPars_ref, configPars2_to_add):
    configPars_ref['equations'] = {
        **configPars2_to_add['equations'], **configPars_ref['equations']
    }
    configPars_ref['mixed_channels_equations'] = {
        **configPars2_to_add['mixed_channels_equations'], 
        **configPars_ref['mixed_channels_equations']
    }
    configPars_ref['user_path_equations'] = {
        **configPars2_to_add['user_path_equations'], 
        **configPars_ref['user_path_equations']
    }
    keep_user_path_equations = {
        key:val for key, val in configPars_ref['user_path_equations'].items()
        if key not in configPars_ref['equations']
    } 
    configPars_ref['user_path_equations'] = keep_user_path_equations
    return configPars_ref

def h5py_iter(g, prefix=''):
    for key, item in g.items():
        path = os.path.join(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5py_iter(item, path)

def h5dump_to_arr(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as f:
        for (path, dset) in h5py_iter(f):
            data_dict[dset.name] = dset[()]
    sorted_keys = natsorted(data_dict.keys())
    arr = np.array([data_dict[key] for key in sorted_keys])
    return arr

def save_to_h5(dst_filepath, data):
    filename = os.path.basename(dst_filepath)
    tempDir = tempfile.mkdtemp()
    tempFilepath = os.path.join(tempDir, filename)
    chunks = [1]*data.ndim
    chunks[-2:] = data.shape[-2:]
    h5f = h5py.File(tempFilepath, 'w')
    dataset = h5f.create_dataset(
        'data', data.shape, dtype=data.dtype,
        chunks=chunks, shuffle=False
    )
    dataset[:] = data
    shutil.move(tempFilepath, dst_filepath)
    shutil.rmtree(tempDir)

def load_segm_file(images_path, end_name_segm_file='segm', return_path=False):
    if not end_name_segm_file.endswith('.npz'):
        end_name_segm_file = f'{end_name_segm_file}.npz'
    for file in myutils.listdir(images_path):
        if file.endswith(end_name_segm_file):
            filepath = os.path.join(images_path, file)
            segm_data = np.load(filepath)['arr_0'].astype(np.uint32)
            if return_path:
                return segm_data, filepath
            else:
                return segm_data
    else:
        if return_path:
            return None, ''
        else:
            return 

def get_tzyx_shape(images_path):
    df_metadata = load_metadata_df(images_path)
    channel = df_metadata.at['channel_0_name', 'values']
    img_filepath = get_filename_from_channel(images_path, channel)
    img_data = load_image_file(img_filepath)
    if img_data.ndim == 4:
        return img_data.shape
    
    SizeZ = int(df_metadata.at['SizeZ', 'values'])
    SizeT = int(df_metadata.at['SizeT', 'values'])
    YX = img_data.shape[-2:]
    return (SizeT, SizeZ, *YX)
    
def load_metadata_df(images_path):
    for file in myutils.listdir(images_path):
        if not file.endswith('metadata.csv'):
            continue
        filepath = os.path.join(images_path, file)
        return pd.read_csv(filepath).set_index('Description')

def _add_will_divide_column(acdc_df):
    if 'cell_cycle_stage' not in acdc_df.columns:
        return acdc_df

    if 'will_divide' in acdc_df.columns:
        return acdc_df

    acdc_df['will_divide'] = np.nan
    last_index_cca_df = acdc_df[['cell_cycle_stage']].last_valid_index()

    cca_df = acdc_df.loc[:last_index_cca_df, cca_df_colnames].reset_index()
    cca_df['will_divide'] = 0.0

    cca_df_buds = cca_df.query('relationship == "bud"')

    for budID, bud_cca_df in cca_df_buds.groupby('Cell_ID'):
        all_gen_nums = cca_df.query(f'Cell_ID == {budID}')['generation_num']
        if not (all_gen_nums > 0).any():
            # bud division is annotated in the future
            continue        

        cca_df.loc[bud_cca_df.index, 'will_divide'] = 1
        
        mothID = int(bud_cca_df['relative_ID'].iloc[0])
        first_frame_bud = bud_cca_df['frame_i'].iloc[0]
        gen_num_moth = cca_df.query(
            f'(frame_i == {first_frame_bud}) & (Cell_ID == {mothID})'
        )['generation_num'].iloc[0]
 
        mothMask = (
            (cca_df['Cell_ID'] == mothID) 
            & (cca_df['generation_num'] == gen_num_moth)
        )

        cca_df.loc[mothMask, 'will_divide'] = 1
    
    cca_df = cca_df.set_index(['frame_i', 'Cell_ID'])
    acdc_df.loc[cca_df.index, cca_df.columns] = cca_df

    return acdc_df

def _fix_will_divide(acdc_df):
    """Resetting annotaions in GUI sometimes does not fully reset `will_divide` 
    column. Here we set `will_divide` back to 0 for those cells whose 
    next generation does not exist (division was not annotated)

    Parameters
    ----------
    acdc_df : pd.DataFrame
        Annotations and metrics dataframe (from the `acdc_output` CSV file) 
        with ['frame_i', 'Cell_ID'] as index

    Returns
    -------
    pd.DataFrame
        acdc_df with `will_divide` corrected.
    """ 
    if 'cell_cycle_stage' not in acdc_df.columns:
        return acdc_df
    
    required_cols = ['frame_i', 'Cell_ID', 'generation_num', 'will_divide']
    
    cca_df_mask = ~acdc_df['cell_cycle_stage'].isna()
    cca_df = acdc_df[cca_df_mask].reset_index()[required_cols]
    
    IDs_will_divide_wrong = (
        cca_functions.get_IDs_gen_num_will_divide_wrong(cca_df)
    )
    if not IDs_will_divide_wrong:
        return acdc_df
    
    cca_df = cca_df.reset_index().set_index(['Cell_ID', 'generation_num'])   
    cca_df.loc[IDs_will_divide_wrong, 'will_divide'] = 0
    cca_df = cca_df.reset_index()
    acdc_df = acdc_df.reset_index()

    cca_df = cca_df.set_index(['frame_i', 'Cell_ID'])
    acdc_df = acdc_df.set_index(['frame_i', 'Cell_ID'])
    
    cca_df_index = cca_df_mask[cca_df_mask].index
    acdc_df.loc[cca_df_index, 'will_divide'] = cca_df['will_divide']
    
    return acdc_df

def _add_missing_columns(acdc_df):
    if 'cell_cycle_stage' not in acdc_df.columns:
        return acdc_df
    
    last_index_cca_df = acdc_df[['cell_cycle_stage']].last_valid_index()
    
    for col, default in base_cca_dict.items():
        if col == 'will_divide':
            # Already taken care by _add_will_divide_column
            continue
        
        if col in acdc_df.columns:
            continue
        
        acdc_df[col] = np.nan
        acdc_df.loc[:last_index_cca_df, col] = default
    
    return acdc_df

def _parse_loaded_acdc_df(acdc_df):
    acdc_df = acdc_df.set_index(['frame_i', 'Cell_ID']).sort_index()
    # remove duplicates saved by mistake or bugs
    duplicated = acdc_df.index.duplicated(keep='first')
    acdc_df = acdc_df[~duplicated]
    acdc_df = pd_bool_to_int(acdc_df, acdc_df_bool_cols, inplace=True)
    acdc_df = pd_int_to_bool(acdc_df, acdc_df_bool_cols)
    return acdc_df

def _remove_redundant_columns(acdc_df):
    acdc_df = acdc_df.drop(columns=['index', 'level_0'], errors='ignore')
    return acdc_df

def read_acdc_df_csv(acdc_df_filepath, index_col=None):
    acdc_df = pd.read_csv(
        acdc_df_filepath, dtype=acdc_df_str_cols, index_col=index_col
    )
    return acdc_df

def _load_acdc_df_file(acdc_df_file_path):
    acdc_df = read_acdc_df_csv(acdc_df_file_path)
    acdc_df = _remove_redundant_columns(acdc_df)
    try:
        acdc_df_drop_cca = acdc_df.drop(columns=cca_df_colnames).fillna(0)
        acdc_df[acdc_df_drop_cca.columns] = acdc_df_drop_cca
    except KeyError:
        pass
    acdc_df = _parse_loaded_acdc_df(acdc_df)
    acdc_df = _add_missing_columns(acdc_df)
    acdc_df = _add_will_divide_column(acdc_df)
    acdc_df = _fix_will_divide(acdc_df)
    return acdc_df

def load_acdc_df_file(
        images_path, 
        end_name_acdc_df_file='acdc_output', 
        return_path=False
    ):
    if not end_name_acdc_df_file.endswith('.csv'):
        end_name_acdc_df_file = f'{end_name_acdc_df_file}.csv'
    for file in myutils.listdir(images_path):
        if file.endswith(end_name_acdc_df_file):
            acdc_df_file_path = os.path.join(images_path, file)
            acdc_df = _load_acdc_df_file(acdc_df_file_path).reset_index()
            if return_path:
                return acdc_df, acdc_df_file_path
            else:
                return acdc_df
    else:
        if return_path:
            return None, ''
        else:
            return

def save_acdc_df_file(acdc_df, csv_path, custom_annot_columns=None):
    if custom_annot_columns is not None:
        new_order_cols = [*sorted_cols, *custom_annot_columns]
    else:
        new_order_cols = sorted_cols
    
    for col in new_order_cols.copy():
        if col in acdc_df.columns:
            continue
        new_order_cols.remove(col)
    
    for col in acdc_df.columns:
        if col in new_order_cols:
            continue
        new_order_cols.append(col)
    
    acdc_df = acdc_df[new_order_cols]
    acdc_df.to_csv(csv_path)

def store_copy_acdc_df(posData, acdc_output_csv_path, log_func=printl):
    try:
        if not os.path.exists(acdc_output_csv_path):
            return
        
        df = (
            pd.read_csv(acdc_output_csv_path, dtype=acdc_df_str_cols)
            .set_index(['frame_i', 'Cell_ID'])
        )
        posData.setTempPaths()
        zip_path = posData.acdc_output_backup_zip_path
        _store_acdc_df_archive(zip_path, df)
    except Exception as e:
        log_func(traceback.format_exc())

def _copy_acdc_dfs_to_temp_archive(
        zip_path, temp_zip_path, csv_names, compression_opts
    ):
    if not os.path.exists(zip_path): 
        return
    
    with zipfile.ZipFile(zip_path, mode='r') as zip:
        for csv_name in csv_names:
            acdc_df = pd.read_csv(zip.open(csv_name))
            acdc_df = _parse_loaded_acdc_df(acdc_df)
            acdc_df = pd_bool_to_int(acdc_df, inplace=False)
            compression_opts['archive_name'] = csv_name
            acdc_df.to_csv(
                temp_zip_path, compression=compression_opts, mode='a'
            )

def _store_acdc_df_archive(zip_path, acdc_df_to_store):
    csv_names = []
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, mode='r') as zip:
            csv_names = natsorted(set(zip.namelist()))
    
    new_key = datetime.now().strftime(ISO_TIMESTAMP_FORMAT)
    csv_name = f'{new_key}.csv'
    if csv_name in csv_names:
        # Do not save duplicates within the same second
        return
    
    if len(csv_names) > 20:
        # Delete oldest df and resave remaining 19
        csv_names.pop(0)
    
    zip_filename = os.path.basename(zip_path)
    temp_zip_filename = zip_filename.replace('.csv', '_temp.csv')
    temp_dirpath = tempfile.mkdtemp()
    temp_zip_path = os.path.join(temp_dirpath, temp_zip_filename)
    compression_opts = {'method': 'zip', 'compresslevel': zipfile.ZIP_STORED}
    _copy_acdc_dfs_to_temp_archive(
        zip_path, temp_zip_path, csv_names, compression_opts
    )
        
    
    compression_opts['archive_name'] = csv_name
    acdc_df = pd_bool_to_int(acdc_df_to_store, inplace=False)
    acdc_df.to_csv(temp_zip_path, compression=compression_opts, mode='a')
    shutil.move(temp_zip_path, zip_path)
    shutil.rmtree(temp_dirpath)

def store_unsaved_acdc_df(recovery_folderpath, df, log_func=printl):
    new_key = datetime.now().strftime(ISO_TIMESTAMP_FORMAT)
    csv_name = f'{new_key}.csv'
    unsaved_recovery_folderpath = os.path.join(
        recovery_folderpath, 'never_saved'
    )
    if not os.path.exists(unsaved_recovery_folderpath):
        os.mkdir(unsaved_recovery_folderpath)
    
    files = myutils.listdir(unsaved_recovery_folderpath)
    csv_files = [file for file in files if file.endswith('.csv')]
    if len(files) > 20:
        csv_files = natsorted(csv_files)
        files_to_remove = csv_files[:-20]
        for file_to_remove in files_to_remove:
            os.remove(os.path.join(unsaved_recovery_folderpath, file_to_remove))
    
    csv_path = os.path.join(unsaved_recovery_folderpath, csv_name)
    df.to_csv(csv_path)

def get_last_stored_unsaved_acdc_df_filepath(recovery_folderpath):
    if not os.path.exists(recovery_folderpath):
        return
    
    unsaved_recovery_folderpath = os.path.join(
        recovery_folderpath, 'never_saved'
    )
    if not os.path.exists(unsaved_recovery_folderpath):
        return
    
    files = myutils.listdir(unsaved_recovery_folderpath)
    csv_files = [file for file in files if file.endswith('.csv')]
    if not csv_files:
        return
    
    csv_files = natsorted(csv_files)
    csv_name = csv_files[-1]
    
    return os.path.join(unsaved_recovery_folderpath, csv_name)

def get_last_stored_unsaved_acdc_df(recovery_folderpath):
    if not os.path.exists(recovery_folderpath):
        return
    
    unsaved_recovery_folderpath = os.path.join(
        recovery_folderpath, 'never_saved'
    )
    if not os.path.exists(unsaved_recovery_folderpath):
        return
    
    files = myutils.listdir(unsaved_recovery_folderpath)
    csv_files = [file for file in files if file.endswith('.csv')]
    if not csv_files:
        return
    
    csv_files = natsorted(csv_files)
    csv_name = csv_files[-1]
    acdc_df = pd.read_csv(os.path.join(unsaved_recovery_folderpath, csv_name))
    acdc_df = _parse_loaded_acdc_df(acdc_df)
    
    return acdc_df

def read_acdc_df_from_archive(archive_path, key):
    if not key.endswith('.csv'):
        csv_name = f'{key}.csv'
    else:
        csv_name = key
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip:
            acdc_df = pd.read_csv(zip.open(csv_name))
    else:
        csv_path = os.path.join(archive_path, f'{key}.csv')
        acdc_df = pd.read_csv(csv_path)
        
    acdc_df = _parse_loaded_acdc_df(acdc_df)
    return acdc_df

def get_user_ch_paths(images_paths, user_ch_name):
    user_ch_file_paths = []
    for images_path in images_paths:
        img_aligned_found = False
        for filename in myutils.listdir(images_path):
            if filename.find(f'{user_ch_name}_aligned.np') != -1:
                img_path_aligned = f'{images_path}/{filename}'
                img_aligned_found = True
            elif filename.find(f'{user_ch_name}.tif') != -1:
                img_path_tif = f'{images_path}/{filename}'

        if img_aligned_found:
            img_path = img_path_aligned
        else:
            img_path = img_path_tif
        user_ch_file_paths.append(img_path)
        print(f'Loading {img_path}...')
    return user_ch_file_paths

def get_acdc_output_files(images_path):
    ls = myutils.listdir(images_path)

    acdc_output_files = [
        file for file in ls 
        if file.find('acdc_output') != -1 and file.endswith('.csv')
    ]
    return acdc_output_files

def get_segm_files(images_path):
    ls = myutils.listdir(images_path)

    segm_files = [
        file for file in ls if file.endswith('segm.npz')
        or file.find('segm_raw_postproc') != -1
        or file.endswith('segm_raw.npz')
        or (file.endswith('.npz') and file.find('segm') != -1)
        or file.endswith('_segm.npy')
    ]
    return segm_files            

def get_files_with(images_path: os.PathLike, with_text: str, ext: str=None):
    ls = myutils.listdir(images_path)
    found_files = []
    for file in ls:
        if file.find(with_text) == -1:
            continue
        
        if ext is not None and not file.endswith(ext):
            continue
        
        found_files.append(file)
    
    return found_files

def load_segmInfo_df(pos_path):
    images_path = os.path.join(pos_path, 'Images')
    for file in myutils.listdir(images_path):
        if file.endswith('segmInfo.csv'):
            csv_path = os.path.join(images_path, file)
            df = pd.read_csv(csv_path)
            df = df.set_index(['filename', 'frame_i']).sort_index()
            df = df[~df.index.duplicated()]
            return df

def segmInfo_df_use_middle_z_slice_where_missing(pos_path):
    experiment_folderpath = os.path.dirname(pos_path)
    pos_foldernames = myutils.get_pos_foldernames(experiment_folderpath)
    for pos in pos_foldernames:
        ...

def get_filename_from_channel(
        images_path, channel_name, not_allowed_ends=None, logger=None,
        basename=None, skip_channels=None
    ):
    if not_allowed_ends is None:
        not_allowed_ends = tuple()
    if skip_channels is None:
        skip_channels = tuple()
    if basename is None:
        basename = ''
    
    channel_filepath = ''
    h5_aligned_path = ''
    h5_path = ''
    npz_aligned_path = ''
    tif_path = ''
    for file in myutils.listdir(images_path):
        isValidEnd = True
        for not_allowed_end in not_allowed_ends:
            if file.endswith(not_allowed_end):
                isValidEnd = False
                break
        if not isValidEnd:
            continue
        
        is_channel_to_skip = False
        for channel_to_skip in skip_channels:
            for ff in channel_file_formats:
                if file.endswith(f'{channel_to_skip}{ff}'):
                    is_channel_to_skip = True
                    break
            if is_channel_to_skip:
                break
        if is_channel_to_skip:
            continue

        channelDataPath = os.path.join(images_path, file)
        if file == f'{basename}{channel_name}':
            channel_filepath = channelDataPath
        elif file.endswith(f'{basename}{channel_name}_aligned.h5'):
            h5_aligned_path = channelDataPath
        elif file.endswith(f'{basename}{channel_name}.h5'):
            h5_path = channelDataPath
        elif file.endswith(f'{basename}{channel_name}_aligned.npz'):
            npz_aligned_path = channelDataPath
        elif file.endswith(f'{basename}{channel_name}.tif'):
            tif_path = channelDataPath
    
    if channel_filepath:
        if logger is not None:
            logger(f'Using channel file ({channel_filepath})...')
        return channel_filepath
    elif h5_aligned_path:
        if logger is not None:
            logger(f'Using .h5 aligned file ({h5_aligned_path})...')
        return h5_aligned_path
    elif h5_path:
        if logger is not None:
            logger(f'Using .h5 file ({h5_path})...')
        return h5_path
    elif npz_aligned_path:
        if logger is not None:
            logger(f'Using .npz aligned file ({npz_aligned_path})...')
        return npz_aligned_path
    elif tif_path:
        if logger is not None:
            logger(f'Using .tif file ({tif_path})...')
        return tif_path
    else:
        return ''

def imread(path):
    if path.endswith('.tif') or path.endswith('.tiff'):
        return tifffile.imread(path)
    else:
        return skimage.io.imread(path)

def load_image_file(filepath):
    if filepath.endswith('.h5'):
        h5f = h5py.File(filepath, 'r')
        img_data = h5f['data']
    elif filepath.endswith('.npz'):
        archive = np.load(filepath)
        files = archive.files
        img_data = archive[files[0]]
    elif filepath.endswith('.npy'):
        img_data = np.load(filepath)
    else:
        img_data = imread(filepath)
    return np.squeeze(img_data)

def load_image_data_from_channel(images_path: os.PathLike, channel_name: str):
    filepath = get_filename_from_channel(images_path, channel_name)
    return load_image_file(filepath)

def get_endnames(basename, files):
    endnames = []
    for f in files:
        filename, _ = os.path.splitext(f)
        endname = filename[len(basename):]
        endnames.append(endname)
    return endnames

def get_exp_path(path):
    folder_type = myutils.determine_folder_type(path)
    is_pos_folder, is_images_folder, _ = folder_type
    if is_pos_folder:
        exp_path = os.path.dirname(path)
    elif is_images_folder:
        exp_path = os.path.dirname(os.path.dirname(path))
    else:
        exp_path = path
    return exp_path

def get_endname_from_channels(filename, channels):
    endname = None
    for ch in channels:
        ch_aligned = f'{ch}_aligned'
        m = re.search(fr'{ch}(.\w+)*$', filename)
        m_aligned = re.search(fr'{ch_aligned}(.\w+)*$', filename)
        if m_aligned is not None:
            return endname
        elif m is not None:
            return endname

def get_endname_from_filepath(filepath, allow_empty=False):
    parent_folderpath = os.path.dirname(filepath)
    if not parent_folderpath.endswith('Images'):
        return 
    
    filename = os.path.basename(filepath)
    filename_noext, ext = os.path.splitext(filename)
    images_files = myutils.listdir(parent_folderpath)
    basename = os.path.commonprefix(images_files)
    endname = filename_noext[len(basename):]
    if not endname:
        endname = basename.split('_')[-1]
    
    return endname
    

def get_endnames_from_basename(basename, filenames):
    return [os.path.splitext(f)[0][len(basename):] for f in filenames]

def get_path_from_endname(end_name, images_path, ext=None):
    if ext is None:
        end_name, ext = myutils.remove_known_extension(end_name)
    
    if os.path.exists(os.path.join(images_path, f'{end_name}{ext}')):
        return os.path.join(images_path, f'{end_name}{ext}')
    
    basename = os.path.commonprefix(myutils.listdir(images_path))
    searched_file = f'{basename}{end_name}{ext}'
    for file in myutils.listdir(images_path):
        filename, ext = os.path.splitext(file)
        if file == searched_file:
            return os.path.join(images_path, file), file
        elif filename == searched_file:
            return os.path.join(images_path, file), file
    
    for file in myutils.listdir(images_path):
        filename, ext = os.path.splitext(file)
        if file.endswith(end_name):
            return os.path.join(images_path, file), file
        elif filename.endswith(end_name):
            return os.path.join(images_path, file), file
    
    return '', ''

def pd_int_to_bool(acdc_df, colsToCast=None):
    if colsToCast is None:
        colsToCast = acdc_df_bool_cols
    for col in colsToCast:
        try:
            acdc_df[col] = acdc_df[col] > 0
        except KeyError:
            continue
    return acdc_df

def pd_bool_to_int(acdc_df, colsToCast=None, csv_path=None, inplace=True):
    """
    Function used to convert "FALSE" strings and booleans to 0s and 1s
    to avoid pandas interpreting as strings or numbers
    """
    if not inplace:
        acdc_df = acdc_df.copy()
    if colsToCast is None:
        colsToCast = acdc_df_bool_cols
    for col in colsToCast:   
        try:
            series = acdc_df[col]
            notna_idx = series.notna()
            notna_series = series.dropna()
            isInt = pd.api.types.is_integer_dtype(notna_series)
            isFloat = pd.api.types.is_float_dtype(notna_series)
            isObject = pd.api.types.is_object_dtype(notna_series)
            isString = pd.api.types.is_string_dtype(notna_series)
            isBool = pd.api.types.is_bool_dtype(notna_series)
            if isFloat or isBool:
                acdc_df.loc[notna_idx, col] = acdc_df.loc[notna_idx, col].astype(int)
            elif isString or isObject:
                # Object data type can have mixed data types so we first convert
                # to strings
                acdc_df.loc[notna_idx, col] = acdc_df.loc[notna_idx, col].astype(str)
                acdc_df.loc[notna_idx, col] = (
                    acdc_df.loc[notna_idx, col].str.lower() == 'true'
                ).astype(int)
        except KeyError:
            continue
        except Exception as e:
            printl(col)
            traceback.print_exc()
    if csv_path is not None:
        acdc_df.to_csv(csv_path)
    return acdc_df

def get_posData_metadata(images_path, basename):
    # First check if metadata.csv already has the channel names
    for file in myutils.listdir(images_path):
        if file.endswith('metadata.csv'):
            metadata_csv_path = os.path.join(images_path, file)
            df_metadata = pd.read_csv(metadata_csv_path).set_index('Description')
            break
    else:
        df_metadata = (
            pd.DataFrame(
                columns=['Description', 'values']).set_index('Description')
            )
        if basename.endswith('_'):
            basename = basename[:-1]
        metadata_csv_path = os.path.join(images_path, f'{basename}_metadata.csv')

    return df_metadata, metadata_csv_path

def is_pos_prepped(images_path):
    filenames = myutils.listdir(images_path)
    for filename in filenames:
        if filename.endswith('dataPrepROIs_coords.csv'):
            return True
        elif filename.endswith('dataPrep_bkgrROIs.json'):
            return True
        elif filename.endswith('aligned.npz'):
            return True
        elif filename.endswith('align_shift.npy'):
            return True
        elif filename.endswith('bkgrRoiData.npz'):
            return True
    return False

def is_bkgrROIs_present(images_path):
    filenames = myutils.listdir(images_path)
    for filename in filenames:
        if filename.endswith('dataPrep_bkgrROIs.json'):
            return True
        elif filename.endswith('bkgrRoiData.npz'):
            return True
    return False

class loadData:
    def __init__(self, imgPath, user_ch_name, relPathDepth=3, QParent=None):
        self.fluo_data_dict = {}
        self.fluo_bkgrData_dict = {}
        self.bkgrROIs = []
        self.loadedFluoChannels = set()
        self.parent = QParent
        self.imgPath = imgPath
        self.user_ch_name = user_ch_name
        self.images_path = os.path.dirname(imgPath)
        self.images_folder_files = os.listdir(self.images_path)
        self.pos_path = os.path.dirname(self.images_path)
        self.spotmax_out_path = os.path.join(self.pos_path, 'spotMAX_output')
        self.exp_path = os.path.dirname(self.pos_path)
        self.pos_foldername = os.path.basename(self.pos_path)
        self.pos_num = self.getPosNum()
        self.cropROI = None
        self.loadSizeT = None
        self.loadSizeZ = None
        self.multiSegmAllPos = False
        self.manualBackgroundLab = None
        self.frame_i = 0
        self.clickEntryPointsDfs = {}
        path_li = os.path.normpath(imgPath).split(os.sep)
        self.relPath = f'{f"{os.sep}".join(path_li[-relPathDepth:])}'
        filename_ext = os.path.basename(imgPath)
        self.filename_ext = filename_ext
        self.filename, self.ext = os.path.splitext(filename_ext)
        self._additionalMetadataValues = None
        self.loadLastEntriesMetadata()
        self.attempFixBasenameBug()
        self.non_aligned_ext = '.tif'
        if filename_ext.endswith('aligned.npz'):
            for file in myutils.listdir(self.images_path):
                if file.endswith(f'{user_ch_name}.h5'):
                    self.non_aligned_ext = '.h5'
                    break
        self.tracked_lost_centroids = None
    
    def attempFixBasenameBug(self):
        '''Attempt removing _s(\d+)_ from filenames if not present in basename
        
        This was a bug introduced when saving the basename with data structure,
        it was not saving the _s(\d+)_ part.
        '''

        try:
            ls = myutils.listdir(self.images_path)
            for file in ls:
                if file.endswith('metadata.csv'):
                    metadata_csv_path = os.path.join(self.images_path, file)
                    break
            else:
                return
            
            df_metadata = pd.read_csv(metadata_csv_path).set_index('Description')
            try:
                basename = df_metadata.at['basename', 'values']
            except Exception as e:
                return
            
            numPos = len(myutils.get_pos_foldernames(self.exp_path))
            numPosDigits = len(str(numPos))
            s0p = str(self.pos_num+1).zfill(numPosDigits)

            if basename.endswith(f'_s{s0p}_'):
                return
            
            for file in ls:
                endname = file[len(basename):]
                if not endname.startswith(f's{s0p}_'):
                    continue
                fixed_endname = endname[len(f's{s0p}_'):]
                fixed_filename = f'{basename}{fixed_endname}'
                fixed_filepath = os.path.join(self.images_path, fixed_filename)
                filepath = os.path.join(self.images_path, file)
                hidden_filepath = os.path.join(self.images_path, f'.{file}')
                shutil.copy2(filepath, fixed_filepath)
                try:
                    os.rename(filepath, hidden_filepath)
                except Exception as e:
                    pass
                    
        except Exception as e:
            traceback.print_exc()
    
    def isPrepped(self):
        return is_pos_prepped(self.images_path)
    
    def isBkgrROIpresent(self):
        return is_bkgrROIs_present(self.images_path)

    def setLoadedChannelNames(self, returnList=False):
        fluo_keys = list(self.fluo_data_dict.keys())

        loadedChNames = []
        for key in fluo_keys:
            chName = key[len(self.basename):]
            aligned_idx = chName.find('_aligned')
            if aligned_idx != -1:
                chName = chName[:aligned_idx]
            loadedChNames.append(chName)

        if returnList:
            return loadedChNames
        else:
            self.loadedChNames = loadedChNames

    def getPosNum(self):
        try:
            pos_num = int(re.findall('Position_(\d+)', self.pos_foldername))[0]
        except Exception:
            pos_num = 0
        return pos_num

    def loadLastEntriesMetadata(self):
        if not os.path.exists(settings_folderpath):
            self.last_md_df = None
            return
        csv_path = os.path.join(settings_folderpath, 'last_entries_metadata.csv')
        if not os.path.exists(csv_path):
            self.last_md_df = None
        else:
            self.last_md_df = pd.read_csv(csv_path).set_index('Description')

    def saveLastEntriesMetadata(self):
        if not os.path.exists(settings_folderpath):
            return
        self.metadata_df.to_csv(last_entries_metadata_path)
    
    def getCustomAnnotColumnNames(self):
        if not hasattr(self, 'customAnnot'):
            return 
        
        return natsorted(self.customAnnot.keys())
    
    def saveCustomAnnotationParams(self):
        if not hasattr(self, 'customAnnot'):
            return 
        
        if not self.customAnnot:
            return
        
        with open(self.custom_annot_json_path, mode='w') as file:
            json.dump(self.customAnnot, file, indent=2)

    def getBasenameAndChNames(self, useExt=None):
        ls = myutils.listdir(self.images_path)
        selector = prompts.select_channel_name()
        self.chNames, _ = selector.get_available_channels(
            ls, self.images_path, useExt=useExt
        )
        self.basename = selector.basename

    def loadImgData(self, imgPath=None, signals=None):
        if imgPath is None:
            imgPath = self.imgPath
        self.z0_window = 0
        self.t0_window = 0
        if self.ext == '.h5':
            self.h5f = h5py.File(imgPath, 'r')
            self.dset = self.h5f['data']
            self.img_data_shape = self.dset.shape
            readH5 = self.loadSizeT is not None and self.loadSizeZ is not None
            if not readH5:
                return

            is4D = self.SizeZ > 1 and self.SizeT > 1
            is3Dz = self.SizeZ > 1 and self.SizeT == 1
            is3Dt = self.SizeZ == 1 and self.SizeT > 1
            is2D = self.SizeZ == 1 and self.SizeT == 1
            if is4D:
                midZ = int(self.SizeZ/2)
                halfZLeft = int(self.loadSizeZ/2)
                halfZRight = self.loadSizeZ-halfZLeft
                z0 = midZ-halfZLeft
                z1 = midZ+halfZRight
                self.z0_window = z0
                self.t0_window = 0
                self.img_data = self.dset[:self.loadSizeT, z0:z1]
            elif is3Dz:
                midZ = int(self.SizeZ/2)
                halfZLeft = int(self.loadSizeZ/2)
                halfZRight = self.loadSizeZ-halfZLeft
                z0 = midZ-halfZLeft
                z1 = midZ+halfZRight
                self.z0_window = z0
                self.img_data = np.squeeze(self.dset[z0:z1])
            elif is3Dt:
                self.t0_window = 0
                self.img_data = np.squeeze(self.dset[:self.loadSizeT])
            elif is2D:
                self.img_data = np.squeeze(self.dset[:])

        elif self.ext == '.npz':
            self.img_data = np.squeeze(np.load(imgPath)['arr_0'])
            self.dset = self.img_data
            self.img_data_shape = self.img_data.shape
        elif self.ext == '.npy':
            self.img_data = np.squeeze(np.load(imgPath))
            self.dset = self.img_data
            self.img_data_shape = self.img_data.shape
        else:
            try:
                self.img_data = np.squeeze(imread(imgPath))
                self.dset = self.img_data
                self.img_data_shape = self.img_data.shape
            except ValueError:
                self.img_data = self._loadVideo(imgPath)
                self.dset = self.img_data
                self.img_data_shape = self.img_data.shape
            except Exception as e:
                traceback.print_exc()
                self.criticalExtNotValid(signals=signals)
    
    def loadChannelData(self, channelName):
        if channelName == self.user_ch_name:
            return self.img_data
            
        dataPath = get_filename_from_channel(self.images_path, channelName)
        if dataPath:
            data = load_image_file(dataPath)
            return data
        else:
            return

    def _loadVideo(self, path):
        video = cv2.VideoCapture(path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(num_frames):
            _, frame = video.read()
            if frame.shape[-1] == 3:
                frame = skimage.color.rgb2gray(frame)
            if i == 0:
                img_data = np.zeros((num_frames, *frame.shape), frame.dtype)
            img_data[i] = frame
        return img_data

    def detectMultiSegmNpz(
            self, multiPos=False, signals=None,
            mutex=None, waitCond=None, askMultiSegmFunc=None,
            newEndFilenameSegm=''
        ):
        if newEndFilenameSegm:
            return '', newEndFilenameSegm, False

        segm_files = get_segm_files(self.images_path)

        if askMultiSegmFunc is None:
            return segm_files

        is_multi_npz = len(segm_files)>0
        if is_multi_npz and askMultiSegmFunc is not None:
            askMultiSegmFunc(segm_files, self, waitCond)
            endFilename = self.selectedItemText[len(self.basename):]
            return self.selectedItemText, endFilename, self.cancel
        elif len(segm_files)==1:
            segmFilename = segm_files[0]
            endFilename = segmFilename[len(self.basename):]
            return segm_files[0], endFilename, False
        else:
            return '', '', False

    def readLastUsedStopFrameNumber(self):
        if not hasattr(self, 'metadata_df'):
            return
        
        if self.metadata_df is None:
            return
        
        try:
            stop_frame_num = int(self.metadata_df.at['stop_frame_num', 'values'])
        except Exception as err:
            stop_frame_num = None
        
        return stop_frame_num
    
    def getSamEmbeddingsPath(self):
        sam_embed_filename = (
            f'{self.basename}_{self.user_ch_name}_sam_embeddings.pt'
        )
        sam_embeddings_path = os.path.join(self.images_path, sam_embed_filename)
        return sam_embeddings_path
    
    def storeSamEmbeddings(self, samAcdcSegment, frame_i=0, z=0):
        # See here how to save embeddings
        # https://github.com/facebookresearch/segment-anything/issues/217
        
        if not hasattr(self, 'sam_embeddings'):
            self.sam_embeddings = {}
        
        if frame_i not in self.sam_embeddings:
            self.sam_embeddings[frame_i] = {}
        
        if hasattr(samAcdcSegment.model, 'predictor'):
            predictor = samAcdcSegment.model.predictor
        else:
            predictor = samAcdcSegment.model
        
        embedding = {
            'original_size': predictor.original_size,
            'input_size': predictor.input_size,
            'features': predictor.features,
            'is_image_set': True,
        }
        self.sam_embeddings[frame_i][z] = embedding
    
    def saveSamEmbeddings(self, logger_func=print):  
        if not hasattr(self, 'sam_embeddings'):
            return 
        
        logger_func(
            f'\nSaving SAM image embeddings to "{self.sam_embeddings_path}"...'
        )
        import torch
        torch.save(self.sam_embeddings, self.sam_embeddings_path)
        
    def loadSamEmbeddings(self, force_reload=False, logger_func=None):
        if hasattr(self, 'sam_embeddings') and not force_reload:
            return 
        
        if not os.path.exists(self.sam_embeddings_path):
            return
        
        if logger_func is not None:
            logger_func(
                f'\nLoading SAM image embeddings from "{self.sam_embeddings_path}"...'
            )
        
        import torch
        self.sam_embeddings = torch.load(self.sam_embeddings_path)
    
    def getSamEmbeddings(self, frame_i=0, z=0):
        if not hasattr(self, 'sam_embeddings'):
            return 
        
        frame_embeddings = self.sam_embeddings.get(frame_i)
        if frame_embeddings is None:
            return
        
        img_embeddings = frame_embeddings.get(z)
        if img_embeddings is None:
            return
            
        return img_embeddings
        
     
    def loadOtherFiles(
            self,
            load_segm_data=True,
            create_new_segm=False,
            load_acdc_df=False,
            load_shifts=False,
            loadSegmInfo=False,
            load_delROIsInfo=False,
            load_bkgr_data=False,
            loadBkgrROIs=False,
            load_last_tracked_i=False,
            load_metadata=False,
            load_dataPrep_ROIcoords=False,
            load_customAnnot=False,
            load_customCombineMetrics=False,
            load_manual_bkgr_lab=False,
            getTifPath=False,
            end_filename_segm='',
            new_endname='',
            labelBoolSegm=None
        ):

        self.segmFound = False if load_segm_data else None
        self.acdc_df_found = False if load_acdc_df else None
        self.shiftsFound = False if load_shifts else None
        self.segmInfoFound = False if loadSegmInfo else None
        self.delROIsInfoFound = False if load_delROIsInfo else None
        self.bkgrDataFound = False if load_bkgr_data else None
        self.bkgrROisFound = False if loadBkgrROIs else None
        self.last_tracked_i_found = False if load_last_tracked_i else None
        self.metadataFound = False if load_metadata else None
        self.dataPrep_ROIcoordsFound = False if load_dataPrep_ROIcoords else None
        self.TifPathFound = False if getTifPath else None
        self.customAnnotFound = False if load_customAnnot else None
        self.combineMetricsFound = False if load_customCombineMetrics else None
        self.labelBoolSegm = labelBoolSegm
        self.bkgrDataExists = False
        ls = myutils.listdir(self.images_path)

        linked_acdc_filename = None
        if end_filename_segm and load_acdc_df:
            # Check if there is an acdc_output file linked to selected .npz
            _acdc_df_end_fn = end_filename_segm.replace('segm', 'acdc_output')
            _acdc_df_end_fn = f'{_acdc_df_end_fn}.csv'
            self._acdc_df_end_fn = _acdc_df_end_fn
            _linked_acdc_fn = f'{self.basename}{_acdc_df_end_fn}'
            acdc_df_path = os.path.join(self.images_path, _linked_acdc_fn)
            self.acdc_output_csv_path = acdc_df_path
            linked_acdc_filename = _linked_acdc_fn
        
        if not hasattr(self, 'basename'):
            self.getBasenameAndChNames()

        for file in ls:
            filePath = os.path.join(self.images_path, file)
            filename, segmExt = os.path.splitext(file)
            endName = filename[len(self.basename):]

            loadMetadata = (
                load_metadata and file.endswith('metadata.csv')
                and not file.endswith('segm_metadata.csv')
            )

            if new_endname:
                # Do not load any segmentation file since user asked for new one
                # This is redundant since we alse have create_new_segm=True
                # but we keep it for code readability
                is_segm_file = False
            elif end_filename_segm:
                # Load the segmentation file selected by the user
                self._segm_end_fn = end_filename_segm
                is_segm_file = endName == end_filename_segm and segmExt == '.npz'
            else:
                # Load default segmentation file
                is_segm_file = file.endswith('segm.npz')

            if linked_acdc_filename is not None:
                is_acdc_df_file = file == linked_acdc_filename
            elif end_filename_segm:
                # Requested a specific file but it is not present
                # do not load acdc_df file
                is_acdc_df_file = False
            else:
                is_acdc_df_file = file.endswith('acdc_output.csv')

                is_acdc_df_file = file == linked_acdc_filename
            
            if load_segm_data and is_segm_file and not create_new_segm:
                self.segmFound = True
                self.segm_npz_path = filePath
                archive = np.load(filePath)
                file = archive.files[0]
                self.segm_data = archive[file].astype(np.uint32)
                self.loadManualBackgroundData()
                if self.segm_data.dtype == bool:
                    if self.labelBoolSegm is None:
                        self.askBooleanSegm()
                squeezed_arr = np.squeeze(self.segm_data)
                if squeezed_arr.shape != self.segm_data.shape:
                    self.segm_data = squeezed_arr
                    np.savez_compressed(filePath, squeezed_arr)
            elif getTifPath and file.find(f'{self.user_ch_name}.tif')!=-1:
                self.tif_path = filePath
                self.TifPathFound = True
            elif load_acdc_df and is_acdc_df_file and not create_new_segm:
                self.acdc_df_found = True
                self.loadAcdcDf(filePath)
            elif load_shifts and file.endswith('align_shift.npy'):
                self.shiftsFound = True
                self.loaded_shifts = np.load(filePath)
            elif loadSegmInfo and file.endswith('segmInfo.csv'):
                self.segmInfoFound = True
                try:
                    remove_duplicates_file(filePath)
                except Exception as err:
                    printl(filePath)
                    printl(traceback.format_exc())
                df = pd.read_csv(filePath).dropna()
                if 'filename' not in df.columns:
                    df['filename'] = self.filename
                df = df.set_index(['filename', 'frame_i']).sort_index()
                df = df[~df.index.duplicated()]
                self.segmInfo_df = df.sort_index()
                self.segmInfo_df.to_csv(filePath)
            elif load_delROIsInfo and file.endswith('delROIsInfo.npz'):
                self.delROIsInfoFound = True
                self.delROIsInfo_npz = np.load(filePath)
            elif file.endswith(f'{self.filename}_bkgrRoiData.npz'):
                self.bkgrDataExists = True
                if load_bkgr_data:
                    self.bkgrDataFound = True
                    self.bkgrData = np.load(filePath)
            elif loadBkgrROIs and file.endswith('dataPrep_bkgrROIs.json'):
                self.bkgrROisFound = True
                with open(filePath) as json_fp:
                    bkgROIs_states = json.load(json_fp)

                if hasattr(self, 'img_data'):
                    for roi_state in bkgROIs_states:
                        Y, X = self.img_data.shape[-2:]
                        roi = pg.ROI(
                            [0, 0], [1, 1],
                            rotatable=False,
                            removable=False,
                            pen=pg.mkPen(color=(150,150,150)),
                            maxBounds=QRectF(QRect(0,0,X,Y)),
                            scaleSnap=True,
                            translateSnap=True
                        )
                        roi.setState(roi_state)
                        self.bkgrROIs.append(roi)
            elif load_dataPrep_ROIcoords and file.endswith('dataPrepROIs_coords.csv'):
                df = pd.read_csv(filePath)
                if 'roi_id' not in df.columns:
                    df['roi_id'] = 0
                if 'description' in df.columns and 'value' in df.columns:
                    df = df.set_index(['roi_id', 'description'])
                    self.dataPrep_ROIcoordsFound = True
                    self.dataPrep_ROIcoords = df
            elif loadMetadata:
                self.metadataFound = True
                remove_duplicates_file(filePath)
                self.metadata_df = pd.read_csv(filePath).set_index('Description')
            elif load_customAnnot and file.endswith('custom_annot_params.json'):
                self.customAnnotFound = True
                self.customAnnot = read_json(filePath)
            elif load_customCombineMetrics and file.endswith('custom_combine_metrics.ini'):
                self.combineMetricsFound = True
                self.setCombineMetricsConfig(ini_path=filePath)

        if self.metadataFound is not None and self.metadataFound:
            self.extractMetadata()
        
        # Check if there is the old segm.npy
        if not self.segmFound and not create_new_segm:
            for file in ls:
                is_segm_npy = file.endswith('segm.npy')
                filePath = os.path.join(self.images_path, file)
                if load_segm_data and is_segm_npy and not self.segmFound:
                    self.segmFound = True
                    self.segm_data = np.load(filePath).astype(np.uint32)

        if load_last_tracked_i:
            self.last_tracked_i_found = True
            try:
                self.last_tracked_i = max(
                    self.acdc_df.index.get_level_values(0)
                )
            except AttributeError as e:
                # traceback.print_exc()
                self.last_tracked_i = None

        if create_new_segm:
            self.setFilePaths(new_endname)

        self.getCustomAnnotatedIDs()
        self.setNotFoundData()
        self.checkAndFixZsliceSegmInfo()
    
    def checkAndFixZsliceSegmInfo(self):
        if not hasattr(self, 'segmInfo_df'):
            return
        
        if self.segmInfo_df is None:
            return
        
        if not hasattr(self, 'SizeZ'):
            return
        
        if self.SizeZ == 1:
            return
        
        middleZslice = int(self.SizeZ/2)
        
        try:
            mask = self.segmInfo_df['z_slice_used_dataPrep'] >= self.SizeZ
            valid_idx = mask[mask].index
            self.segmInfo_df.loc[valid_idx, 'z_slice_used_dataPrep'] = middleZslice
        except Exception as err:
            pass
        
        try:
            mask = self.segmInfo_df['z_slice_used_gui'] >= self.SizeZ
            valid_idx = mask[mask].index
            self.segmInfo_df.loc[valid_idx, 'z_slice_used_gui'] = middleZslice
        except Exception as err:
            pass
    
    def loadMostRecentUnsavedAcdcDf(self):
        acdc_df = get_last_stored_unsaved_acdc_df(self.recoveryFolderpath())
        if acdc_df is None:
            return
        self.acdc_df = acdc_df
        self.acdc_df_found = True
        self.last_tracked_i = max(self.acdc_df.index.get_level_values(0))
    
    def loadAcdcDf(self, filePath, updatePaths=True, return_df=False):
        acdc_df = _load_acdc_df_file(filePath)
        if updatePaths:
            self.acdc_df = acdc_df
            self.acdc_df_found = True
            self.last_tracked_i = max(self.acdc_df.index.get_level_values(0))
        if return_df:
            return acdc_df
    
    def getSpotmaxSingleSpotsfiles(self):
        from spotmax import DFs_FILENAMES
        spotmax_files = myutils.listdir(self.spotmax_out_path)
        patterns = [
            filename.replace('*rn*', '').replace('*desc*', '')
            for filename in DFs_FILENAMES.values()
        ]
        valid_files = []
        for file in spotmax_files:
            filepath = os.path.join(self.spotmax_out_path, file)
            if not os.path.isfile(filepath):
                continue
            if file.endswith('aggregated.csv'):
                continue
            for pattern in patterns:
                if file.find(pattern) != -1:
                    break
            else:
                continue
            valid_files.append(file)
        
        return reversed(valid_files)

    def askBooleanSegm(self):
        segmFilename = os.path.basename(self.segm_npz_path)
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            f'The loaded segmentation file<br><br>'
            f'"{segmFilename}"<br><br> '
            'has <b>boolean data type</b>.<br><br>'
            'To correctly load it, Cell-ACDC needs to <b>convert</b> it '
            'to <b>integer data type</b>.<br><br>'
            'Do you want to <b>label the mask</b> to separate the objects '
            '(recommended) or do you want to keep one single object?<br>'
        )
        LabelButton, _  = msg.question(
            self.parent, 'Boolean segmentation mask?', txt,
            buttonsTexts=('Label (recommended)', 'Keep single object')
        )
        if msg.clickedButton == LabelButton:
            self.labelBoolSegm = True
        else:
            self.labelBoolSegm = False

    def labelSegmData(self):
        if self.labelBoolSegm is None:
            return

        if self.segm_data.dtype != bool:
            return

        if self.labelBoolSegm:
            if self.SizeT > 1:
                segm_data = np.zeros(self.segm_data.shape, dtype=np.uint32)
                for i, lab in enumerate(self.segm_data):
                    segm_data[i] = skimage.measure.label(lab)
                self.segm_data = segm_data
            else:
                self.segm_data = skimage.measure.label(self.segm_data)
        else:
            self.segm_data = self.segm_data.astype(np.uint32)

    def setFilePaths(self, new_endname):
        if self.basename.endswith('_'):
            basename = self.basename
        else:
            basename = f'{self.basename}_'

        if new_endname:
            segm_new_filename = f'{basename}segm_{new_endname}.npz'
            acdc_output_filename = f'{basename}acdc_output_{new_endname}.csv'
        else:
            segm_new_filename = f'{basename}segm.npz'
            acdc_output_filename = f'{basename}acdc_output.csv'
        
        filePath = os.path.join(self.images_path, segm_new_filename)
        self.segm_npz_path = filePath

        filePath = os.path.join(self.images_path, acdc_output_filename)
        self.acdc_output_csv_path = filePath
    
    def fromTrackerToAcdcDf(
            self, tracker, tracked_video, save=False, start_frame_i=0
        ):
        cca_dfs_attr = hasattr(tracker, 'cca_dfs')
        cca_dfs_auto_attr = hasattr(tracker, 'cca_dfs_auto')

        if hasattr(tracker, 'tracked_lost_centroids'):
            self.saveTrackedLostCentroids(tracker.tracked_lost_centroids)

        if not cca_dfs_attr and not cca_dfs_auto_attr:
            return
        
        if cca_dfs_attr:
            keys = list(range(start_frame_i, len(tracker.cca_dfs)))
            acdc_df = pd.concat(tracker.cca_dfs, keys=keys, names=['frame_i'])
        else:
            keys = list(range(start_frame_i, len(tracker.cca_dfs_auto)))
            acdc_df = pd.concat(tracker.cca_dfs_auto, keys=keys, names=['frame_i'])

        acdc_df['is_cell_dead'] = 0
        acdc_df['is_cell_excluded'] = 0
        acdc_df['was_manually_edited'] = 0
        acdc_df['x_centroid'] = 0
        acdc_df['y_centroid'] = 0
        for i, lab in enumerate(tracked_video):
            frame_i = start_frame_i + i
            rp = skimage.measure.regionprops(lab)
            for obj in rp:
                centroid = obj.centroid
                yc, xc = obj.centroid[-2:]
                acdc_df.at[(frame_i, obj.label), 'x_centroid'] = int(xc)
                acdc_df.at[(frame_i, obj.label), 'y_centroid'] = int(yc)

                if len(centroid) == 3:
                    if 'z_centroid' not in acdc_df.columns:
                        acdc_df['z_centroid'] = 0
                    zc = obj.centroid[0]
                    acdc_df.at[(frame_i, obj.label), 'z_centroid'] = int(zc)                

        if not save:
            return acdc_df

        acdc_df = pd_bool_to_int(acdc_df, inplace=False)
        if cca_dfs_attr:
            acdc_df.to_csv(self.acdc_output_csv_path)
            self.loadAcdcDf(self.acdc_output_csv_path)
        elif cca_dfs_auto_attr:
            acdc_df.to_csv(self.acdc_output_auto_csv_path)

    def getAcdcDfEndname(self):
        if not hasattr(self, 'acdc_output_csv_path'):
            return
        
        if not hasattr(self, 'basename'):
            return
        
        filename = os.path.basename(self.acdc_output_csv_path)
        filename, _ = os.path.splitext(filename)
        endname = filename[len(self.basename):].lstrip('_')
        return endname
    
    def getSegmEndname(self):
        if not hasattr(self, 'acdc_output_csv_path'):
            return
        
        if not hasattr(self, 'basename'):
            return
        
        filename = os.path.basename(self.acdc_output_csv_path)
        filename, _ = os.path.splitext(filename)
        endname = filename[len(self.basename):].lstrip('_')
        return endname
    
    def getCustomAnnotatedIDs(self):
        self.customAnnotIDs = {}

        if self.acdc_df_found is None:
            return

        if not self.acdc_df_found:
            return

        if self.customAnnotFound is None:
            return

        if not self.customAnnotFound:
            return

        for name in self.customAnnot.keys():
            self.customAnnotIDs[name] = {}
            if name not in self.acdc_df.columns:
                self.acdc_df[name] = 0
            for frame_i, df in self.acdc_df.groupby(level=0):
                series = df[name]
                series = series[series>0]
                annotatedIDs = list(series.index.get_level_values(1).unique())
                self.customAnnotIDs[name][frame_i] = annotatedIDs

    def isCropped(self):
        if self.dataPrep_ROIcoords is None:
            return False
        df = self.dataPrep_ROIcoords
        _isCropped = any([
            df_roi.at[(roi_id, 'cropped'), 'value'] > 0
            for roi_id, df_roi in df.groupby(level=0)
        ]) 
        return _isCropped
    
    def getIsSegm3D(self):
        if self.SizeZ == 1:
            return False

        if self.segmFound is None:
            return

        if not self.segmFound:
            return

        if hasattr(self, 'img_data'):
            return self.segm_data.ndim == self.img_data.ndim
        else:
            if self.SizeT > 1:
                return self.segm_data.ndim == 4
            else:
                return self.segm_data.ndim == 3

    def getBytesImageData(self):
        if not hasattr(self, 'img_data'):
            return 0
        
        return sys.getsizeof(self.img_data)
    
    def extractMetadata(self):
        self.metadata_df['values'] = self.metadata_df['values'].astype(str)
        if 'SizeT' in self.metadata_df.index:
            self.SizeT = float(self.metadata_df.at['SizeT', 'values'])
            self.SizeT = int(self.SizeT)
        elif self.last_md_df is not None and 'SizeT' in self.last_md_df.index:
            self.SizeT = float(self.last_md_df.at['SizeT', 'values'])
            self.SizeT = int(self.SizeT)
        else:
            self.SizeT = 1

        self.SizeZ_found = False
        if 'SizeZ' in self.metadata_df.index:
            self.SizeZ = float(self.metadata_df.at['SizeZ', 'values'])
            self.SizeZ = int(self.SizeZ)
            self.SizeZ_found = True
        elif self.last_md_df is not None and 'SizeZ' in self.last_md_df.index:
            self.SizeZ = float(self.last_md_df.at['SizeZ', 'values'])
            self.SizeZ = int(self.SizeZ)
        else:
            self.SizeZ = 1

        if 'SizeY' in self.metadata_df.index:
            self.SizeY = float(self.metadata_df.at['SizeY', 'values'])
            self.SizeY = int(self.SizeY)
            self.SizeX = float(self.metadata_df.at['SizeX', 'values'])
            self.SizeX = int(self.SizeX)
        else:
            if hasattr(self, 'img_data_shape'):
                self.SizeY, self.SizeX = self.img_data_shape[-2:]
            else:
                self.SizeY, self.SizeX = 1, 1

        self.isSegm3D = False
        if hasattr(self, 'segm_npz_path'):
            segmEndName = self.getSegmEndname()
            isSegm3Dkey = f'{segmEndName}_isSegm3D'        
            if isSegm3Dkey in self.metadata_df.index:
                isSegm3D = str(self.metadata_df.at[isSegm3Dkey, 'values'])
                self.isSegm3D = isSegm3D.lower() == 'true'

        if 'TimeIncrement' in self.metadata_df.index:
            self.TimeIncrement = float(
                self.metadata_df.at['TimeIncrement', 'values']
            )
        elif self.last_md_df is not None and 'TimeIncrement' in self.last_md_df.index:
            self.TimeIncrement = float(self.last_md_df.at['TimeIncrement', 'values'])
        else:
            self.TimeIncrement = 1

        if 'PhysicalSizeX' in self.metadata_df.index:
            self.PhysicalSizeX = float(
                self.metadata_df.at['PhysicalSizeX', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeX' in self.last_md_df.index:
            self.PhysicalSizeX = float(self.last_md_df.at['PhysicalSizeX', 'values'])
        else:
            self.PhysicalSizeX = 1

        if 'PhysicalSizeY' in self.metadata_df.index:
            self.PhysicalSizeY = float(
                self.metadata_df.at['PhysicalSizeY', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeY' in self.last_md_df.index:
            self.PhysicalSizeY = float(self.last_md_df.at['PhysicalSizeY', 'values'])
        else:
            self.PhysicalSizeY = 1

        if 'PhysicalSizeZ' in self.metadata_df.index:
            self.PhysicalSizeZ = float(
                self.metadata_df.at['PhysicalSizeZ', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeZ' in self.last_md_df.index:
            self.PhysicalSizeZ = float(self.last_md_df.at['PhysicalSizeZ', 'values'])
        else:
            self.PhysicalSizeZ = 1

        if 'LensNA' in self.metadata_df.index:
            self.numAperture = float(
                self.metadata_df.at['LensNA', 'values']
            )
        else:
            self.numAperture = 1.4
        
        emWavelenMask = self.metadata_df.index.str.contains(r'_emWavelen')
        df_emWavelens = self.metadata_df[emWavelenMask]
        self.emWavelens = {}
        try:
            for channel_i_emWavelen, emWavelen in df_emWavelens.itertuples():
                channel_i_name = channel_i_emWavelen.replace('_emWavelen', '_name')
                chName = self.metadata_df.at[channel_i_name, 'values']
                self.emWavelens[chName] = float(emWavelen)
        except Exception as e:
            pass
        
        self._additionalMetadataValues = {}
        for name in self.metadata_df.index:
            if name.startswith('__') and len(name) > 2:
                value = self.metadata_df.at[name, 'values']
                self._additionalMetadataValues[name] = value
        
        if not self._additionalMetadataValues:
            # Load metadata values saved in temp folder
            if os.path.exists(additional_metadata_path):
                self._additionalMetadataValues = read_json(
                    additional_metadata_path, desc='additional metadata'
                )

    def saveIsSegm3Dmetadata(self, segm_npz_path):
        segmFilename = os.path.basename(segm_npz_path)
        segmFilename = os.path.splitext(segmFilename)[0]
        segmEndName = segmFilename[len(self.basename):]
        isSegm3Dkey = f'{segmEndName}_isSegm3D'
        self.metadata_df.at[isSegm3Dkey, 'values'] = self.isSegm3D
        self.metadata_df.to_csv(self.metadata_csv_path)
    
    def additionalMetadataValues(self):
        additionalMetadataValues = {}
        for name in self.metadata_df.index:
            if name.startswith('__'):
                value = self.metadata_df.at[name, 'values']
                key = name.replace('__', '', 1)
                additionalMetadataValues[key] = value
        return additionalMetadataValues
    
    def add_tree_cols_to_cca_df(self, cca_df, frame_i=None):
        cca_df = cca_df.sort_index().reset_index()

        if self.acdc_df is None:
            return cca_df
        
        if frame_i is not None:
            df = self.acdc_df.loc[frame_i].sort_index().reset_index()
        else:
            df = self.acdc_df.sort_index().reset_index()

        cols = cca_df.columns.to_list()
        for col in df.columns:
            if not col.endswith('tree'):
                continue

            ref_col = col[:col.find('_tree')]
            if ref_col in cols:
                ref_col_idx = cols.index(ref_col) + 1
            else:
                ref_col_idx = len(cols) - 4

            if col in cols:
                cca_df[col] = df[col]
            else:
                cca_df.insert(ref_col_idx, col, df[col])
        
        return cca_df
    
    def getManualBackgroudDataFilepath(self):
        segmFilename = os.path.basename(self.segm_npz_path)
        segmEndname = segmFilename[len(self.basename):]
        manualBackgrEndname = segmEndname.replace('segm', 'manualBackground')
        manualBackgrFilename = f'{self.basename}{manualBackgrEndname}'
        filepath = os.path.join(self.images_path, manualBackgrFilename)
        return filepath

    def saveManualBackgroundData(self, data: np.ndarray):
        if data is None:
            return 
        filepath = self.getManualBackgroudDataFilepath()
        np.savez_compressed(filepath, data)

    def loadManualBackgroundData(self):
        filepath = self.getManualBackgroudDataFilepath()
        if not os.path.exists(filepath):
            self.manualBackgroundLab = None
            return
        archive = np.load(filepath)
        self.manualBackgroundLab = archive[archive.files[0]]
    
    def setNotFoundData(self):
        if self.segmFound is not None and not self.segmFound:
            self.segm_data = None
            # Segmentation file not found and a specifc one was requested
            # --> set the path
            if hasattr(self, '_segm_end_fn'):
                if self.basename.endswith('_'):
                    basename = self.basename
                else:
                    basename = f'{self.basename}_'
                base_path = os.path.join(self.images_path, basename)
                self.segm_npz_path = f'{base_path}{self._segm_end_fn}.npz'
        if self.acdc_df_found is not None and not self.acdc_df_found:
            self.acdc_df = None
            # Set the file path for selected acdc_output.csv file
            # since it was not found
            if hasattr(self, '_acdc_df_end_fn'):
                if self.basename.endswith('_'):
                    basename = self.basename
                else:
                    basename = f'{self.basename}_'
                base_path = os.path.join(self.images_path, basename)
                self.acdc_output_csv_path = f'{base_path}{self._acdc_df_end_fn}'
        if self.shiftsFound is not None and not self.shiftsFound:
            self.loaded_shifts = None
        if self.segmInfoFound is not None and not self.segmInfoFound:
            self.segmInfo_df = None
        if self.delROIsInfoFound is not None and not self.delROIsInfoFound:
            self.delROIsInfo_npz = None
        if self.bkgrDataFound is not None and not self.bkgrDataFound:
            self.bkgrData = None
        if self.bkgrROisFound is not None and not self.bkgrROisFound:
            # Do not load bkgrROIs if bkgrDataFound to avoid addMetrics to use it
            self.bkgrROIs = []
        if self.bkgrDataExists:
            # Do not load bkgrROIs if bkgrDataFound to avoid addMetrics to use it
            self.bkgrROIs = []
        if self.dataPrep_ROIcoordsFound is not None and not self.dataPrep_ROIcoordsFound:
            self.dataPrep_ROIcoords = None
        if self.last_tracked_i_found is not None and not self.last_tracked_i_found:
            self.last_tracked_i = None
        if self.TifPathFound is not None and not self.TifPathFound:
            self.tif_path = None
        if self.customAnnotFound is not None and not self.customAnnotFound:
            self.customAnnot = {}
        if self.combineMetricsFound is not None and not self.combineMetricsFound:
            self.setCombineMetricsConfig()

        if self.metadataFound is None:
            # Loading metadata was not requested
            return

        if self.metadataFound:
            return

        if hasattr(self, 'img_data'):
            if self.img_data.ndim == 3:
                if len(self.img_data) > 49:
                    self.SizeT, self.SizeZ = len(self.img_data), 1
                else:
                    self.SizeT, self.SizeZ = 1, len(self.img_data)
            elif self.img_data.ndim == 4:
                self.SizeT, self.SizeZ = self.img_data.shape[:2]
            else:
                self.SizeT, self.SizeZ = 1, 1
        else:
            self.SizeT, self.SizeZ = 1, 1
        
        try:
            self.SizeY, self.SizeX = self.img_data_shape[-2:]
        except Exception as e:
            try:
                self.SizeY, self.SizeX = self.segm_data.shape[-2:]
            except Exception as e:
                self.SizeY, self.SizeX = 1, 1

        self.TimeIncrement = 1.0
        self.PhysicalSizeX = 1.0
        self.PhysicalSizeY = 1.0
        self.PhysicalSizeZ = 1.0
        self.metadata_df = None

        if self.last_md_df is None:
            # Last entered values do not exists
            return

        # Since metadata was not found use the last entries saved in temp folder
        # if 'SizeT' in self.last_md_df.index and self.SizeT == 1:
        #     self.SizeT = int(self.last_md_df.at['SizeT', 'values'])
        # if 'SizeZ' in self.last_md_df.index and self.SizeZ == 1:
        #     self.SizeZ = int(self.last_md_df.at['SizeZ', 'values'])
        if 'TimeIncrement' in self.last_md_df.index:
            self.TimeIncrement = float(
                self.last_md_df.at['TimeIncrement', 'values']
            )
        if 'PhysicalSizeX' in self.last_md_df.index:
            self.PhysicalSizeX = float(
                self.last_md_df.at['PhysicalSizeX', 'values']
            )
        if 'PhysicalSizeY' in self.last_md_df.index:
            self.PhysicalSizeY = float(
                self.last_md_df.at['PhysicalSizeY', 'values']
            )
        if 'PhysicalSizeZ' in self.last_md_df.index:
            self.PhysicalSizeZ = float(
                self.last_md_df.at['PhysicalSizeZ', 'values']
            )

    def addEquationCombineMetrics(self, equation, colName, isMixedChannels):
        section = 'mixed_channels_equations' if isMixedChannels else 'equations'
        self.combineMetricsConfig[section][colName] = equation

    def setCombineMetricsConfig(self, ini_path=''):
        if ini_path:
            configPars = config.ConfigParser()
            configPars.read(ini_path)
        else:
            configPars = config.ConfigParser()

        if 'equations' not in configPars:
            configPars['equations'] = {}

        if 'mixed_channels_equations' not in configPars:
            configPars['mixed_channels_equations'] = {}

        if 'user_path_equations' not in configPars:
            configPars['user_path_equations'] = {}

        # Append channel specific equations from the user_profile_path ini file
        userPathChEquations = configPars['user_path_equations']
        for chName in self.chNames:
            chName_equations = measurements.get_user_combine_metrics_equations(
                chName
            )
            chName_equations = {
                key:val for key, val in chName_equations.items()
                if key not in configPars['equations']
            }
            userPathChEquations = {**userPathChEquations, **chName_equations}
            configPars['user_path_equations'] = userPathChEquations

        # Append mixed channels equations from the user_profile_path ini file
        configPars['mixed_channels_equations'] = {
            **configPars['mixed_channels_equations'],
            **measurements.get_user_combine_mixed_channels_equations()
        }

        self.combineMetricsConfig = configPars

    def saveCombineMetrics(self):
        with open(self.custom_combine_metrics_path, 'w') as configfile:
            self.combineMetricsConfig.write(configfile)
    
    def saveClickEntryPointsDfs(self):
        for tableEndName, df in self.clickEntryPointsDfs.items():
            if not self.basename.endswith('_'):
                basename = f'{self.basename}_'
            else:
                basename = self.basename
            tableFilename = f'{basename}{tableEndName}.csv'
            tableFilepath = os.path.join(self.images_path, tableFilename)
            df = df.sort_values(['frame_i', 'Cell_ID'])
            df.to_csv(tableFilepath, index=False)

    def check_acdc_df_integrity(self):
        check = (
            self.acdc_df_found is not None # acdc_df was laoded if present
            and self.acdc_df is not None # acdc_df was present
            and self.segmFound is not None # segm data was loaded if present
            and self.segm_data is not None # segm data was present
        )
        if check:
            if self.SizeT > 1:
                annotates_frames = self.acdc_df.index.get_level_values(0)
                for frame_i, lab in enumerate(self.segm_data):
                    if frame_i not in annotates_frames:
                        break
                    self._fix_acdc_df(lab, frame_i=frame_i)
            else:
                lab = self.segm_data
                self._fix_acdc_df(lab)

    def _fix_acdc_df(self, lab, frame_i=0):
        rp = skimage.measure.regionprops(lab)
        segm_IDs = [obj.label for obj in rp]
        acdc_df_IDs = self.acdc_df.loc[frame_i].index
        try:
            cca_df = self.acdc_df[cca_df_colnames]
        except KeyError:
            # Columns not present because not annotated --> no need to fix
            return

        for obj in rp:
            ID = obj.label
            if ID in acdc_df_IDs:
                continue
            idx = (frame_i, ID)
            self.acdc_df.loc[idx, cca_df_colnames] = base_cca_dict.values()
            for col, val in base_acdc_df.items():
                if not isnan(self.acdc_df.at[idx, col]):
                    continue
                self.acdc_df.at[idx, col] = val
            y, x = obj.centroid
            self.acdc_df.at[idx, 'x_centroid'] = x
            self.acdc_df.at[idx, 'y_centroid'] = y

    def getSegmEndname(self):
        segmFilename = os.path.basename(self.segm_npz_path)
        segmFilename = os.path.splitext(segmFilename)[0]
        segmEndName = segmFilename[len(self.basename):]
        return segmEndName
    
    def getSegmentedChannelHyperparams(self):
        cp = config.ConfigParser()
        if os.path.exists(self.segm_hyperparams_ini_path):
            cp.read(self.segm_hyperparams_ini_path)
            segmEndName = self.getSegmEndname()
            section = segmEndName
            option = 'segmented_channel'
            channel_name = cp.get(section, option, fallback=self.user_ch_name)
            return channel_name, segmEndName
        else:
            return self.user_ch_name, ''
    
    def updateSegmentedChannelHyperparams(self, channelName):
        cp = config.ConfigParser()
        if not os.path.exists(self.segm_hyperparams_ini_path):
            return
        cp.read(self.segm_hyperparams_ini_path)
        segmEndName = self.getSegmEndname()
        section = segmEndName
        if section not in cp.sections():
            return
        option = 'segmented_channel'
        cp[section][option] = channelName
        with open(self.segm_hyperparams_ini_path, 'w') as configfile:
            cp.write(configfile)

    def saveSegmHyperparams(
            self, model_name, init_kwargs, segment_kwargs, 
            post_process_params=None, 
            preproc_recipe=None
        ):
        cp = config.ConfigParser()

        if os.path.exists(self.segm_hyperparams_ini_path):
            cp.read(self.segm_hyperparams_ini_path)
        
        segmEndName = self.getSegmEndname()
        segm_filename = os.path.basename(self.segm_npz_path)

        metadata_section = f'{segmEndName}.metadata'
        cp[metadata_section] = {}
        
        cp[metadata_section]['segmentation_filename'] = segm_filename
        cp[metadata_section]['segmented_channel'] = self.user_ch_name
        now = datetime.now().strftime(r'%Y-%m-%d %H:%M:%S')
        cp[metadata_section]['segmented_on'] = now
        cp[metadata_section]['model_name'] = model_name

        init_section = f'{segmEndName}.init'
        cp[init_section] = {}
        for key, value in init_kwargs.items():
            cp[init_section][key] = str(value)
        
        segment_section = f'{segmEndName}.segment'
        cp[segment_section] = {}
        for key, value in segment_kwargs.items():
            cp[segment_section][key] = str(value)

        if post_process_params is not None:
            post_process_section = f'{segmEndName}.postprocess'
            cp[post_process_section] = {}
            for key, value in post_process_params.items():
                cp[post_process_section][key] = str(value)

        if preproc_recipe is not None:
            preproc_ini_items = config.preprocess_recipe_to_ini_items(
                preproc_recipe
            )
            for preproc_section, section_items in preproc_ini_items.items():
                segm_preproc_section = f'{segmEndName}.{preproc_section}'
                cp[segm_preproc_section] = {}
                for key, value in section_items.items():
                    cp[segm_preproc_section][key] = str(value)
        
        with open(self.segm_hyperparams_ini_path, 'w') as configfile:
            cp.write(configfile)
    
    def isRecoveredAcdcDfPresent(self):
        recovery_folderpath = self.recoveryFolderpath()
        unsaved_recovery_folderpath = os.path.join(
            recovery_folderpath, 'never_saved'
        )
        if not os.path.exists(unsaved_recovery_folderpath):
            return
        
        files = myutils.listdir(unsaved_recovery_folderpath)
        csv_files = [file for file in files if file.endswith('.csv')]
        if not csv_files:
            return
        
        if not os.path.exists(self.acdc_output_csv_path):
            acdc_df_mtime = 0
        else:
            acdc_df_mtime = os.path.getmtime(self.acdc_output_csv_path)
        
        acdc_df_mdatetime = datetime.fromtimestamp(acdc_df_mtime)
        
        csv_files = natsorted(csv_files)
        iso_key = csv_files[-1][:-4]
        most_recent_unsaved_acdc_df_datetime = datetime.strptime(
            iso_key, ISO_TIMESTAMP_FORMAT
        )
        return most_recent_unsaved_acdc_df_datetime > acdc_df_mdatetime
    
    def recoveryFolderpath(self, create_if_missing=True):
        recovery_folder = os.path.join(self.images_path, 'recovery')
        if not os.path.exists(recovery_folder) and create_if_missing:
            os.mkdir(recovery_folder)
        return recovery_folder
    
    def setTempPaths(self, createFolder=True):
        temp_folder = os.path.join(self.images_path, 'recovery')
        self.recoveryFolderPath = temp_folder
        if not os.path.exists(temp_folder) and createFolder:
            os.mkdir(temp_folder)
        segm_filename = os.path.basename(self.segm_npz_path)
        acdc_df_filename = os.path.basename(self.acdc_output_csv_path)
        self.segm_npz_temp_path = os.path.join(temp_folder, segm_filename)
        self.acdc_output_backup_zip_path = os.path.join(
            temp_folder, acdc_df_filename.replace('.csv', '.zip')
        )
        unsaved_acdc_df_filename = acdc_df_filename.replace(
            '.csv', '_autosave.zip'
        )
        self.unsaved_acdc_df_autosave_path = os.path.join(
            temp_folder, unsaved_acdc_df_filename
        )
        
    def buildPaths(self):
        if self.basename.endswith('_'):
            basename = self.basename
        else:
            basename = f'{self.basename}_'
        base_path = os.path.join(self.images_path, basename)
        self.slice_used_align_path = f'{base_path}slice_used_alignment.csv'
        self.slice_used_segm_path = f'{base_path}slice_segm.csv'
        self.align_npz_path = f'{base_path}{self.user_ch_name}_aligned.npz'
        self.align_old_path = f'{base_path}phc_aligned.npy'
        self.align_shifts_path = f'{base_path}align_shift.npy'
        self.segm_npz_path = f'{base_path}segm.npz'
        self.last_tracked_i_path = f'{base_path}last_tracked_i.txt'
        self.acdc_output_csv_path = f'{base_path}acdc_output.csv'
        self.segmInfo_df_csv_path = f'{base_path}segmInfo.csv'
        self.delROIs_info_path = f'{base_path}delROIsInfo.npz'
        self.dataPrepROI_coords_path = f'{base_path}dataPrepROIs_coords.csv'
        # self.dataPrepBkgrValues_path = f'{base_path}dataPrep_bkgrValues.csv'
        self.dataPrepBkgrROis_path = f'{base_path}dataPrep_bkgrROIs.json'
        self.metadata_csv_path = f'{base_path}metadata.csv'
        self.mot_events_path = f'{base_path}mot_events'
        self.mot_metrics_csv_path = f'{base_path}mot_metrics'
        self.raw_segm_npz_path = f'{base_path}segm_raw.npz'
        self.raw_postproc_segm_path = f'{base_path}segm_raw_postproc'
        self.post_proc_mot_metrics = f'{base_path}post_proc_mot_metrics'
        self.segm_hyperparams_ini_path = f'{base_path}segm_hyperparams.ini'
        self.custom_annot_json_path = f'{base_path}custom_annot_params.json'
        self.custom_combine_metrics_path = (
            f'{base_path}custom_combine_metrics.ini'
        )
        self.sam_embeddings_path =(
            f'{base_path}{self.user_ch_name}_sam_embeddings.pt'
        )
        self.tracked_lost_centroids_json_path = f'{base_path}tracked_lost_centroids.json'
        self.acdc_output_auto_csv_path = f'{base_path}acdc_output_auto.csv'
    
    def get_btrack_export_path(self):
        btrack_path = self.segm_npz_path.replace('.npz', '.h5')
        btrack_path = btrack_path.replace('_segm', '_btrack_tracks')
        return btrack_path
    
    def get_tracker_export_path(self, trackerName, ext):
        tracker_path = self.segm_npz_path.replace('_segm', f'_{trackerName}_tracks')
        tracker_path = tracker_path.replace('.npz', ext)
        return tracker_path

    def setBlankSegmData(self, SizeT, SizeZ, SizeY, SizeX):
        if not hasattr(self, 'img_data'):
            self.segm_data = None
            return

        Y, X = self.img_data.shape[-2:]
        if self.segmFound is not None and not self.segmFound:
            if SizeT > 1 and self.isSegm3D:
                self.segm_data = np.zeros((SizeT, SizeZ, Y, X), int)
            elif self.isSegm3D:
                self.segm_data = np.zeros((SizeZ, Y, X), int)
            elif SizeT > 1:
                self.segm_data = np.zeros((SizeT, Y, X), int)
            else:
                self.segm_data = np.zeros((Y, X), int)

    def loadAllImgPaths(self):
        tif_paths = []
        npy_paths = []
        npz_paths = []
        basename = self.basename[0:-1]
        for filename in myutils.listdir(self.images_path):
            file_path = os.path.join(self.images_path, filename)
            f, ext = os.path.splitext(filename)
            m = re.match(fr'{basename}.*\.tif', filename)
            if m is not None:
                tif_paths.append(file_path)
                # Search for npy fluo data
                npy = f'{f}_aligned.npy'
                npz = f'{f}_aligned.npz'
                npy_found = False
                npz_found = False
                for name in myutils.listdir(self.images_path):
                    _path = os.path.join(self.images_path, name)
                    if name == npy:
                        npy_paths.append(_path)
                        npy_found = True
                    if name == npz:
                        npz_paths.append(_path)
                        npz_found = True
                if not npy_found:
                    npy_paths.append(None)
                if not npz_found:
                    npz_paths.append(None)
        self.tif_paths = tif_paths
        self.npy_paths = npy_paths
        self.npz_paths = npz_paths

    def checkH5memoryFootprint(self):
        if self.ext != '.h5':
            return 0
        else:
            Y, X = self.dset.shape[-2:]
            size = self.loadSizeT*self.loadSizeZ*Y*X
            itemsize = self.dset.dtype.itemsize
            required_memory = size*itemsize
            return required_memory
    
    def _warnMultiPosTimeLapse(self, SizeT_metadata):
        txt = html_utils.paragraph(f"""
            You are trying to load <b>multiple Positions</b> of what it seems to be 
            <b>time-lapse data</b> (number of frames in the metadata is 
            {SizeT_metadata}).<br><br>
            Note that Cell-ACDC <b>cannot load multiple time-lapse Positions</b>.<br><br>
            To load time-lapse data, load <b>one Position at a time</b>.<br><br>
            However, you can proceed anyway if you think the saved metadata is wrong 
            and you need to correct them.<br><br>
            Do you want to proceed?
        """)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        _, noButton, yesButton = msg.warning(
            self.parent, 'WARNING: Edinting saved metadata', txt, 
            buttonsTexts=('Cancel', 'No, stop the process', 'Yes, proceed anyway')
        )
        return msg.clickedButton == yesButton

    def askInputMetadata(
            self, numPos,
            ask_SizeT=False,
            ask_TimeIncrement=False,
            ask_PhysicalSizes=False,
            singlePos=False,
            save=False,
            askSegm3D=True,
            forceEnableAskSegm3D=False,
            warnMultiPos=False
        ):
        from . import apps
        SizeZ_metadata = None
        SizeT_metadata = None
        if hasattr(self, 'metadataFound'):
            if self.metadataFound:
                SizeT_metadata = self.SizeT
                SizeZ_metadata = self.SizeZ
                if SizeT_metadata>1 and numPos>1 and warnMultiPos:
                    proceed_anyway = self._warnMultiPosTimeLapse(SizeT_metadata)
                    if not proceed_anyway:
                        return False
        
        basename = ''
        if hasattr(self, 'basename'):
            basename = self.basename
        metadataWin = apps.QDialogMetadata(
            self.SizeT, self.SizeZ, self.TimeIncrement,
            self.PhysicalSizeZ, self.PhysicalSizeY, self.PhysicalSizeX,
            ask_SizeT, ask_TimeIncrement, ask_PhysicalSizes,
            parent=self.parent, font=apps.font, imgDataShape=self.img_data_shape,
            posData=self, singlePos=singlePos, askSegm3D=askSegm3D,
            additionalValues=self._additionalMetadataValues,
            forceEnableAskSegm3D=forceEnableAskSegm3D, 
            SizeT_metadata=SizeT_metadata, SizeZ_metadata=SizeZ_metadata, 
            basename=basename
        )
        metadataWin.exec_()
        if metadataWin.cancel:
            return False

        self.onlyEditMetadata = metadataWin.allowEditSizeTcheckbox.isChecked()
        self.SizeT = metadataWin.SizeT
        self.SizeZ = metadataWin.SizeZ
        self.SizeY, self.SizeX = self.img_data_shape[-2:]

        self.isSegm3D = metadataWin.isSegm3D

        self.loadSizeS = numPos
        self.loadSizeT = metadataWin.SizeT
        self.loadSizeZ = metadataWin.SizeZ

        source = metadataWin if ask_TimeIncrement else self
        self.TimeIncrement = source.TimeIncrement

        source = metadataWin if ask_PhysicalSizes else self
        self.PhysicalSizeZ = source.PhysicalSizeZ
        self.PhysicalSizeY = source.PhysicalSizeY
        self.PhysicalSizeX = source.PhysicalSizeX

        self._additionalMetadataValues = metadataWin._additionalValues
        if save:
            self.saveMetadata(additionalMetadata=metadataWin._additionalValues)
        return True
    
    def zSliceSegmentation(self, filename, frame_i):
        if self.SizeZ > 1:
            idx = (filename, frame_i)
            if self.segmInfo_df.at[idx, 'resegmented_in_gui']:
                col = 'z_slice_used_gui'
            else:
                col = 'z_slice_used_dataPrep'
            z = self.segmInfo_df.at[idx, col]
        else:
            z = None
        return z

    def transferMetadata(self, from_posData):
        self.SizeT = from_posData.SizeT
        self.SizeZ = from_posData.SizeZ
        self.PhysicalSizeZ = from_posData.PhysicalSizeZ
        self.PhysicalSizeY = from_posData.PhysicalSizeY
        self.PhysicalSizeX = from_posData.PhysicalSizeX

    def metadataToCsv(self, signals=None, mutex=None, waitCond=None):
        try:
            self.metadata_df.to_csv(self.metadata_csv_path)
        except PermissionError:
            print('='*20)
            traceback.print_exc()
            print('='*20)
            permissionErrorTxt = html_utils.paragraph(
                f'The below file is open in another app (Excel maybe?).<br><br>'
                f'{self.metadata_csv_path}<br><br>'
                'Close file and then press "Ok".'
            )
            if signals is None:
                msg = widgets.myMessageBox(self.parent)
                msg.warning(
                    self, 'Permission denied', permissionErrorTxt
                )
                self.metadata_df.to_csv(self.metadata_csv_path)
            else:
                mutex.lock()
                signals.sigPermissionError.emit(permissionErrorTxt, waitCond)
                waitCond.wait(mutex)
                mutex.unlock()
                self.metadata_df.to_csv(self.metadata_csv_path)

    def saveMetadata(
            self, signals=None, mutex=None, waitCond=None,
            additionalMetadata=None
        ):
        segmEndName = self.getSegmEndname()
        isSegm3Dkey = f'{segmEndName}_isSegm3D'
        if self.metadata_df is None:
            metadata_dict = {
                'SizeT': self.SizeT,
                'SizeZ': self.SizeZ,
                'SizeY': self.SizeY,
                'SizeX': self.SizeX,
                'TimeIncrement': self.TimeIncrement,
                'PhysicalSizeZ': self.PhysicalSizeZ,
                'PhysicalSizeY': self.PhysicalSizeY,
                'PhysicalSizeX': self.PhysicalSizeX,
                isSegm3Dkey: self.isSegm3D
            }
            if additionalMetadata is not None:
                metadata_dict = {**metadata_dict, **additionalMetadata}
                for key in list(metadata_dict.keys()):
                    if key.startswith('__') and key not in additionalMetadata:
                        metadata_dict.pop(key)

            self.metadata_df = pd.DataFrame(metadata_dict, index=['values']).T
            self.metadata_df.index.name = 'Description'
        else:
            self.metadata_df.at['SizeT', 'values'] = self.SizeT
            self.metadata_df.at['SizeZ', 'values'] = self.SizeZ
            self.metadata_df.at['TimeIncrement', 'values'] = self.TimeIncrement
            self.metadata_df.at['PhysicalSizeZ', 'values'] = self.PhysicalSizeZ
            self.metadata_df.at['PhysicalSizeY', 'values'] = self.PhysicalSizeY
            self.metadata_df.at['PhysicalSizeX', 'values'] = self.PhysicalSizeX
            self.metadata_df.at[isSegm3Dkey, 'values'] = self.isSegm3D
            if additionalMetadata is not None:
                for name, value in additionalMetadata.items():
                    self.metadata_df.at[name, 'values'] = value

                idx_to_drop = []
                for name in self.metadata_df.index:
                    if name.startswith('__') and name not in additionalMetadata:
                        idx_to_drop.append(name)

                self.metadata_df = self.metadata_df.drop(idx_to_drop)
        self.metadataToCsv(signals=signals, mutex=mutex, waitCond=waitCond)
        try:
            self.metadata_df.to_csv(last_entries_metadata_path)
        except PermissionError:
            pass
        if additionalMetadata is not None:
            try:
                with open(additional_metadata_path, mode='w') as file:
                    json.dump(additionalMetadata, file, indent=2)
            except PermissionError:
                pass

    def criticalExtNotValid(self, signals=None):
        err_title = f'File extension {self.ext} not valid.'
        err_msg = (
            f'The requested file {self.relPath}\n'
            'has an invalid extension.\n\n'
            'Valid extensions are .tif, .tiff, .npy or .npz'
        )
        if self.parent is None:
            print('-------------------------')
            print(err_msg)
            print('-------------------------')
            raise FileNotFoundError(err_title)
        elif signals is None:
            print('-------------------------')
            print(err_msg)
            print('-------------------------')
            msg = QMessageBox()
            msg.critical(self.parent, err_title, err_msg, msg.Ok)
            return None
        elif signals is not None:
            raise FileNotFoundError(err_title)
        
    def saveTrackedLostCentroids(self, tracked_lost_centroids_list=None, _tracked_lost_centroids_list=None):

        if not (self.tracked_lost_centroids or tracked_lost_centroids_list or _tracked_lost_centroids_list):
            return

        if _tracked_lost_centroids_list is not None:
            tracked_lost_centroids_list = _tracked_lost_centroids_list

        elif tracked_lost_centroids_list is not None:
            tracked_lost_centroids_list = {k: v for k, v in tracked_lost_centroids_list.items()}

        else:
            tracked_lost_centroids_list = {k: list(v) for k, v in self.tracked_lost_centroids.items()}

        # printl(tracked_lost_centroids_list)
        try:
            with open(self.tracked_lost_centroids_json_path, 'w') as json_file:
                json.dump(tracked_lost_centroids_list, json_file, indent=4)
        except PermissionError:
            print('='*20)
            traceback.print_exc()
            print('='*20)
            permissionErrorTxt = html_utils.paragraph(
                f'The below file is open in another app (Excel maybe?).<br><br>'
                f'{self.tracked_lost_centroids_json_path}<br><br>'
                'Close file and then press "Ok", or press "Cancel" to abort.'
            )
            msg = widgets.myMessageBox(self.parent)
            msg.warning(
                self, 'Permission denied', permissionErrorTxt, buttonsTexts=('Cancel', 'Ok')
            )
            if msg.cancel:
                return
            
            self.saveTrackedLostCentroids(_tracked_lost_centroids_list=tracked_lost_centroids_list)

    def loadTrackedLostCentroids(self):
        try:
            with open(self.tracked_lost_centroids_json_path, 'r') as json_file:
                tracked_lost_centroids_list = json.load(json_file)
                self.tracked_lost_centroids = {int(k): {tuple(int(val) for val in centroid) for centroid in v} for k, v in tracked_lost_centroids_list.items()}
        except FileNotFoundError:
            # print(f"No file found at {self.tracked_lost_centroids_json_path}")
            self.tracked_lost_centroids = {
                frame_i:set() for frame_i in range(self.SizeT)
                }
        except PermissionError:
            print('='*20)
            traceback.print_exc()
            print('='*20)
            permissionErrorTxt = html_utils.paragraph(
                f'The below file is open in another app (Excel maybe?).<br><br>'
                f'{self.tracked_lost_centroids_json_path}<br><br>'
                'Close file and then press "Ok", or press "Cancel" to abort.'
            )
            msg = widgets.myMessageBox(self.parent)
            msg.warning(
                self, 'Permission denied', permissionErrorTxt, buttonsTexts=('Cancel', 'Ok')
            )
            if msg.cancel:
                self.tracked_lost_centroids = {
                    frame_i:set() for frame_i in range(self.SizeT)
                    }
                return
            
            self.loadTrackedLostCentroids()

class select_exp_folder:
    def __init__(self):
        self.exp_path = None

    def QtPrompt(
            self, parentQWidget, values,
            current=0, title='Select Position folder',
            CbLabel="Select folder to load:",
            showinexplorer_button=False, full_paths=None,
            allow_abort=True, show=False, toggleMulti=False,
            allowMultiSelection=True
        ):
        from . import apps
        font = QtGui.QFont()
        font.setPixelSize(13)
        win = apps.QtSelectItems(
            title, values, '', CbLabel=CbLabel, parent=parentQWidget,
            showInFileManagerPath=self.exp_path
        )
        win.setFont(font)
        toFront = win.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
        win.setWindowState(toFront)
        win.activateWindow()
        if not allowMultiSelection:
            win.multiPosButton.setChecked(False)
            win.multiPosButton.setDisabled(True)
        if toggleMulti:
            win.multiPosButton.setChecked(True)
        win.exec_()
        self.was_aborted = win.cancel
        if not win.cancel:
            self.selected_pos = [
                self.pos_foldernames[idx] for idx in win.selectedItemsIdx
            ]

    def append_last_cca_frame(self, acdc_df, text):
        if 'cell_cycle_stage' not in acdc_df.columns:
            return text
        
        try:
            colnames = ['frame_i', *cca_df_colnames]
            cca_df = acdc_df[colnames].dropna()
        except Exception as e:
            return text

        last_cca_frame_i = max(cca_df['frame_i'], default=None)
        if last_cca_frame_i is None:
            return text
        to_append = f', last cc annotated frame: {last_cca_frame_i+1})'
        text = text.replace(')', to_append)
        return text
    
    def get_values_segmGUI(self, exp_path):
        self.exp_path = exp_path
        pos_foldernames = myutils.get_pos_foldernames(exp_path)
        self.pos_foldernames = pos_foldernames
        values = []
        for pos in pos_foldernames:
            last_tracked_i_found = False
            pos_path = os.path.join(exp_path, pos)
            images_path = os.path.join(pos_path, 'Images')
            filenames = myutils.listdir(images_path)
            for filename in filenames:
                if filename.find('acdc_output.csv') != -1:
                    last_tracked_i_found = True
                    acdc_df_path = os.path.join(images_path, filename)
                    acdc_df = _load_acdc_df_file(acdc_df_path).reset_index()
                    last_tracked_i = acdc_df['frame_i'].max()
                    break
            
            if last_tracked_i_found:
                text = f'{pos} (Last tracked frame: {last_tracked_i+1})'
                text = self.append_last_cca_frame(acdc_df, text)
                values.append(text)
            else:
                values.append(pos)
        self.values = values
        return values

    def get_values_dataprep(self, exp_path):
        self.exp_path = exp_path
        pos_foldernames = myutils.get_pos_foldernames(exp_path)
        self.pos_foldernames = pos_foldernames
        values = []
        for pos in pos_foldernames:
            is_prepped = False
            are_zslices_selected = False
            pos_path = os.path.join(exp_path, pos)
            images_path = os.path.join(pos_path, 'Images')
            filenames = myutils.listdir(images_path)
            for filename in filenames:
                if filename.endswith('dataPrepROIs_coords.csv'):
                    is_prepped = True
                    break
                elif filename.endswith('dataPrep_bkgrROIs.json'):
                    is_prepped = True
                    break
                elif filename.endswith('aligned.npz'):
                    is_prepped = True
                    break
                elif filename.endswith('align_shift.npy'):
                    is_prepped = True
                    break
                elif filename.endswith('bkgrRoiData.npz'):
                    is_prepped = True
                    break
                elif filename.endswith('segmInfo.csv'):
                    are_zslices_selected = True
            if is_prepped:
                values.append(f'{pos} (already prepped)')
            elif are_zslices_selected:
                values.append(f'{pos} (z-slices selected)')
            else:
                values.append(pos)
        self.values = values
        return values

    def get_values_cca(self, exp_path):
        self.exp_path = exp_path
        pos_foldernames = natsorted(myutils.listdir(exp_path))
        pos_foldernames = [
            pos for pos in pos_foldernames if re.match(r'^Position_(\d+)', pos)
        ]
        self.pos_foldernames = pos_foldernames
        values = []
        for pos in pos_foldernames:
            cc_stage_found = False
            pos_path = os.path.join(exp_path, pos)
            if os.path.isdir(pos_path):
                images_path = f'{exp_path}/{pos}/Images'
                filenames = myutils.listdir(images_path)
                for filename in filenames:
                    if filename.find('cc_stage.csv') != -1:
                        cc_stage_found = True
                        cc_stage_path = f'{images_path}/{filename}'
                        cca_df = pd.read_csv(
                            cc_stage_path, index_col=['frame_i', 'Cell_ID']
                        )
                        last_analyzed_frame_i = (
                            cca_df.index.get_level_values(0).max()
                        )
                if cc_stage_found:
                    values.append(f'{pos} (Last analyzed frame: '
                                  f'{last_analyzed_frame_i})')
                else:
                    values.append(pos)
        self.values = values
        return values

    def _close(self):
        val = self.pos_n_sv.get()
        idx = list(self.values).index(val)
        if self.full_paths is None:
            self.selected_pos = [self.pos_foldernames[idx]]
        else:
            self.TIFFs_path = self.full_paths[idx]
        self.root.quit()
        self.root.destroy()

    def on_closing(self):
        self.selected_pos = [None]
        self.was_aborted = True
        self.root.quit()
        self.root.destroy()
        if self.allow_abort:
            exit('Execution aborted by the user')


def load_shifts(parent_path, basename=None):
    shifts_found = False
    shifts = None
    if basename is None:
        for filename in myutils.listdir(parent_path):
            if filename.find('align_shift.npy')>0:
                shifts_found = True
                shifts_path = os.path.join(parent_path, filename)
                shifts = np.load(shifts_path)
    else:
        align_shift_fn = f'{basename}_align_shift.npy'
        if align_shift_fn in myutils.listdir(parent_path):
            shifts_found = True
            shifts_path = os.path.join(parent_path, align_shift_fn)
            shifts = np.load(shifts_path)
        else:
            shifts = None
    return shifts, shifts_found

class OMEXML_image:
    def __init__(self, Pixels, ome_schema):
        if Pixels is None:
            node = None
        else:
            node = Pixels.attrib
        self.Pixels = OMEXML_Pixels(Pixels, node, ome_schema)

class OMEXML_objective:
    def __init__(self) -> None:
        self.LensNA = 1.4

class OMEXML_intrument:
    def __init__(self):
        self.Objective = OMEXML_objective()

class OMEXML_Channel:
    def __init__(self, Channel) -> None:
        self.Name = Channel.attrib.get('Name', '')
        self.node = Channel.attrib

class OMEXML_Pixels:
    def __init__(self, Pixels, node, ome_schema) -> None:
        self.node = node
        self.Pixels = Pixels
        self.ome_schema = ome_schema
        if node is None:
            self.SizeZ = 1
            self.SizeT = 1
            self.SizeC = 1
            self.PhysicalSizeX = 1.0
            self.PhysicalSizeY = 1.0
            self.PhysicalSizeZ = 1.0
        else:
            self.SizeZ = node.get('SizeZ', 1)
            self.SizeT = node.get('SizeT', 1)
            self.SizeC = node.get('SizeC', 1)
            self.PhysicalSizeX = node.get('PhysicalSizeX', 1.0)
            self.PhysicalSizeY = node.get('PhysicalSizeY', 1.0)
            self.PhysicalSizeZ = node.get('PhysicalSizeZ', 1.0)
        
    def Channel(self, channel_index=0):
        Channel = self.Pixels.findall(f'{self.ome_schema}Channel')[channel_index]
        return OMEXML_Channel(Channel)

class OMEXML:
    def __init__(self, ometiff_filepath):
        self.filepath = ometiff_filepath
        self.read_omexml_string()
        self.parse_metadata()
    
    def read_omexml_string(self):
        with TiffFile(self.filepath) as tif:
            return tif.ome_metadata
    
    def parse_metadata(self):
        self.omexml_string = self.read_omexml_string()
        self.root = ET.fromstring(self.omexml_string)
        self.ome_schema = re.findall(r'({.+})OME', self.root.tag)[0]
    
    def instrument(self):
        instrument = OMEXML_intrument()
        instrument_xml = self.root.find(f'{self.ome_schema}Instrument')
        if instrument_xml is None:
            return instrument
        objective_xml = instrument_xml.find(f'{self.ome_schema}Objective')
        if objective_xml is None:
            return instrument
        LensNA = objective_xml.attrib.get('LensNA')
        if LensNA is None:
            return instrument
        instrument.Objective.LensNA = LensNA
        return instrument

    def get_image_count(self):
        return len(self.root.findall(f'{self.ome_schema}Image'))

    def image(self):
        Image = self.root.find(f'{self.ome_schema}Image')
        Pixels = Image.find(f'{self.ome_schema}Pixels')
        image = OMEXML_image(Pixels, self.ome_schema)
        image.Name = Image.attrib.get('Name', '')
        return image

def _restructure_multi_files_multi_pos(
        src_path, dst_path, action='copy', signals=None, logger=print
    ):
    if signals is not None:
        signals.initProgressBar.emit(0)
    logger('Scanning files...')
    files = list(os.listdir(src_path))
    files = [f for f in files if os.path.isfile(os.path.join(src_path, f))]
    
    # Group files with same starting string with all possible splits
    files_scanned = list(files)
    groups = {}
    for f, file in enumerate(files):
        splits = file.split('_')
        current_split = splits[0]
        for split in splits[1:]:
            for other_file in files_scanned:
                if other_file.startswith(current_split):
                    if current_split not in groups:
                        groups[current_split] = {other_file}
                    else:
                        groups[current_split].add(other_file)
            current_split = f'{current_split}_{split}'
        files_scanned.pop(0)
    
    # Determine the keys of duplicated groups
    keys_duplicates = {}
    keys_scanned = list(groups.keys())
    for k, key in enumerate(groups.keys()):
        set_A = groups[key]
        for other_key in keys_scanned:
            set_B = groups[other_key]
            if not set_A.difference(set_B):
                if key not in keys_duplicates:
                    keys_duplicates[key] = {other_key}
                else:
                    keys_duplicates[key].add(other_key)
        keys_scanned.pop(0)
    
    # Get unique splits and sort them by length
    unique_splits = {max(splits, key=len) for splits in keys_duplicates.values()}
    unique_splits = sorted(list(unique_splits), key=len)
    
    # Get groups of files sharing the same starting
    groups_files = {}
    for split in unique_splits:
        for file in files:
            if file.startswith(split):
                if split not in groups_files:
                    groups_files[split] = {file}
                else:
                    groups_files[split].add(file)
    
    # Sort the files according to exp and pos splits
    groups_n_splits = {len(split.split('_')):set() for split in groups_files}
    for split in groups_files:
        n_splits = len(split.split('_'))
        groups_n_splits[n_splits].add(split)
    
    sorted_n_splits = sorted(groups_n_splits.keys())
    n_splits_exp, n_splits_pos = sorted_n_splits[-2:]
    final_structure = {}    
    for split_exp in groups_n_splits[n_splits_exp]:
        exp_folder_path = os.path.join(dst_path, split_exp)
        exp_files = groups_files[split_exp]
        pos_splits = groups_n_splits[n_splits_pos]   
        for exp_file in exp_files:
            p = 1
            for pos_split in pos_splits:
                if not pos_split.startswith(split_exp):
                    continue
                try:
                    pos_n = pos_split.split('_')[-1]
                    pos_n = int(pos_n)
                except Exception as e:
                    pos_n = p
                pos_path = os.path.join(exp_folder_path, f'Position_{pos_n}')
                images_path = os.path.join(pos_path, 'Images')
                final_structure[images_path] = []
                if not os.path.exists(images_path):
                    os.makedirs(images_path, exist_ok=True)
                for file in files:
                    if not file.startswith(pos_split):
                        continue 
                    final_structure[images_path].append(file)
                    
                p += 1
    
    # Move or copy the files
    if signals is not None:
        signals.initProgressBar.emit(len(files))
    action_str = 'Copying' if action=='copy' else 'Moving'
    logger(f'{action_str} files...')
    pbar = tqdm(total=len(files), ncols=100, unit='file')
    for images_path, files in final_structure.items():
        for file in files:
            dst_file = os.path.join(images_path, file)
            src_file = os.path.join(src_path, file)
            try:
                if action == 'copy':
                    shutil.copy2(src_file, dst_file)
                else:
                    shutil.move(src_file, dst_file)
            except Exception as e:
                continue
            pbar.update()
            if signals is not None:
                signals.progressBar.emit(1)
    pbar.close()
    
    action_str = 'copied' if action=='copy' else 'moved'
    logger(f'Done! Files {action_str} and restructured into "{src_path}"')

def get_all_svg_icons_aliases(sort=True):
    from . import resources_filepath
    with open(resources_filepath, 'r') as resources_file:
        resources_txt = resources_file.read()
    
    aliases = re.findall(r'<file alias="(.+\.svg)">', resources_txt)
    if sort:
        aliases = natsorted(aliases)
    return aliases

def get_all_buttons_names(sort=True):
    widgets_filepath = os.path.join(cellacdc_path, 'widgets.py')
    with open(widgets_filepath, 'r') as py_file:
        txt = py_file.read()
    
    all_buttons_names = re.findall(r'class (\w+)\(Q?PushButton\):', txt)
    if sort:
        all_buttons_names = natsorted(all_buttons_names)
    return all_buttons_names

def rename_qrc_resources_file(scheme='light'):
    os.remove(qrc_resources_path)
 
    if scheme == 'dark' and os.path.exists(qrc_resources_dark_path):
        shutil.copyfile(qrc_resources_dark_path, qrc_resources_path)
    elif scheme == 'light' and os.path.exists(qrc_resources_light_path):
        shutil.copyfile(qrc_resources_light_path, qrc_resources_path)

def autoLineBreak(text, length): #automatic line breaking for tooltips. Keeps indentation with spaces and preexisting line breaks
    lines = []
    current_line = []

    # Split the text into lines while preserving existing newline characters
    existing_lines = text.split('\n')

    for existing_line in existing_lines:
        # Calculate the indentation for the current line
        indent = len(existing_line) - len(existing_line.lstrip())
        words = existing_line.lstrip().split()  # Split each line into words

        for word in words:
            if len(' '.join(current_line + [word])) + indent <= length:
                current_line.append(word)
            else:
                lines.append(' ' * indent + ' '.join(current_line))
                current_line = [word]

        if current_line:  # Add any remaining words as the last line
            lines.append(' ' * indent + ' '.join(current_line))

        # Reset the current line for the next existing line
        current_line = []

    return '\n'.join(lines)

def format_bullet_points(text): #indentation for bullet points in tooltips. Implementation not robust
    lines = text.split('\n')
    formatted_lines = []
    indent = False
    indentNo = 0

    for line in lines:
        if line.strip().startswith("* "):
            indent = True
            formatted_line = line
            indentNo = len(line) - len(line.lstrip())
        else:
            indentNoComp = len(line) - len(line.lstrip())
            if indent == True and indentNo == indentNoComp:
                formatted_line = " " * 2 + line
            else:
                formatted_line = line
                indent = False

        formatted_lines.append(formatted_line)

    return '\n'.join(formatted_lines)   

def format_number_list(text): #indentation for number points in tooltips. Implementation not robust
    lines = text.split('\n')
    formatted_lines = []
    indent = False
    indentNo = 0

    for line in lines:
        if line.strip().startswith((
                "0. ", "1. ", "2. ", "3. ", "4. ", 
                "5. ", "6. ", "7. ", "8. ", "9. "
            )):
            indent = True
            formatted_line = line
            indentNo = len(line) - len(line.lstrip())
        else:
            indentNoComp = len(line) - len(line.lstrip())
            if indent == True and indentNo == indentNoComp:
                formatted_line = " " * 3 + line
            else:
                formatted_line = line
                indent = False

        formatted_lines.append(formatted_line)

    return '\n'.join(formatted_lines)


def get_tooltips_from_docs(): 
    # gets tooltips for GUI from .\Cell_ACDC\docs\source\tooltips.rst
    var_pattern = r"\|(\S*)\|"
    shortcut_pattern = r"\*\*(\".*\")\):\*\*"
    title_pattern = r"\*\*(.*)\(\*\*"

    if not os.path.exists(tooltips_rst_filepath):
        return {}
    
    with open(tooltips_rst_filepath, "r") as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if not (line.startswith("..") or line.startswith("    :target:") or line.startswith("    :alt:") or line.startswith("    :width:") or line.startswith("    :height:") or line==""):
            new_lines.append(line)
    lines = new_lines

    non_empty_lines = [line.replace("\n", "") for line in lines if line.strip()] #also removes \n from lines
    lines = non_empty_lines

    tipdict = {}

    for i, line in enumerate(lines):
        match = re.search(var_pattern, line)
        if match:
            name = match.group(1)

            title = re.search(title_pattern, line).group(1)

            shortcut = re.search(shortcut_pattern, line)
            if shortcut:
                shortcut = shortcut.group(1)
            else:
                shortcut = "\"No shortcut\""

            desc = line.split("):**")[1].lstrip(" ")

            appSameLine = False

            if desc == "":
                appSameLine = True

            descList = []
            i += 1
            for followLine in lines[i:]:
                followMatch = re.search(var_pattern, followLine)
                if followMatch or followLine.startswith("* **"):
                    break
                else:                    
                    descList.append(followLine)

            if descList != []:
                if descList[-1].startswith("----"):
                    descList.pop(-1)
                    descList.pop(-1)


            for entry in descList:

                entry = entry.replace("| ", "")

                if entry.startswith(" " * 4):
                    stripped_string = entry[4:]
                else:
                    stripped_string = entry
                entry = stripped_string

                if appSameLine == False:
                    entry = "\n" + entry
                else:
                    appSameLine = False

                desc += entry
            desc = autoLineBreak(desc, 60)
            desc = format_bullet_points(desc)
            desc = format_number_list(desc)

            tipdict[name] = f"Name: {title}\nShortcut: {shortcut}\n\n{desc}"
    return tipdict

def save_df_to_csv_temp_path(df, csv_filename, **to_csv_kwargs):
    tempDir = tempfile.mkdtemp()
    tempFilepath = os.path.join(tempDir, csv_filename)
    df.to_csv(tempFilepath, **to_csv_kwargs)
    return tempFilepath

def loaded_df_to_points_data(df, t_col, z_col, y_col, x_col):
    points_data = {}
    if 'id' not in df.columns:
        df['id'] = ''
        
    if t_col != 'None':
        grouped = df.groupby(t_col)
    else:
        grouped = [(0, df)]
    
    for frame_i, df_frame in grouped:
        if z_col != 'None':
            df_frame[z_col] = df_frame[z_col].round().astype(int)
            # Use integer z
            zz = df_frame[z_col]
            points_data[frame_i] = {} 
            for z in zz.values:
                df_z = df_frame[df_frame[z_col] == z]
                z_int = round(z)
                if z_int in points_data[frame_i]:
                    continue
                points_data[frame_i][z_int] = {
                    'x': df_z[x_col].to_list(),
                    'y': df_z[y_col].to_list(), 
                    'id': df_z['id'].to_list(), 
                }
        else:
            points_data[frame_i] = {
                'x': df[x_col].to_list(),
                'y': df[y_col].to_list(), 
                'id': df['id'].to_list(), 
            }
    return points_data

def load_df_points_layer(filepath):
    df = None
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.h5'):
        with pd.HDFStore(filepath) as h5:
            keys = h5.keys()
            dfs = [h5.get(key) for key in keys]
        df = pd.concat(dfs, keys=keys, names=['h5_key'])
    return df

def get_unique_exp_paths(paths: List):
    unique_exp_paths = set()
    for path in paths:
        exp_path = get_exp_path(path)
        unique_exp_paths.add(exp_path.replace('\\', '/'))
    return unique_exp_paths

def search_filepath_in_pos_path_from_endname(
        pos_path, endname, include_spotmax_out=False
    ):
    images_path = os.path.join(pos_path, 'Images')
    spotmax_out_path = os.path.join(pos_path, 'spotMAX_output')
    if include_spotmax_out and os.path.exists(spotmax_out_path):
        for sm_file in os.listdir(spotmax_out_path):
            if endname == sm_file:
                return os.path.join(spotmax_out_path, sm_file)
    
    images_files = myutils.listdir(images_path)
    sample_filepath = os.path.join(images_path, images_files[0])
    posData = loadData(sample_filepath, '')
    posData.getBasenameAndChNames()
    to_match = f'{posData.basename}{endname}'
    for file in images_files:
        if file == to_match:
            return os.path.join(images_path, file)

def search_filepath_from_endname(exp_path, endname, include_spotmax_out=False):
    pos_foldernames = myutils.get_pos_foldernames(exp_path)
    for pos in pos_foldernames:
        pos_path = os.path.join(exp_path, pos)
        filepath = search_filepath_in_pos_path_from_endname(
            pos_path, endname, include_spotmax_out=include_spotmax_out
        )
        return filepath

def askOpenCsvFile(
        title='Open CSV file', 
        start_dir=None, 
        qparent=None
    ):
    if start_dir is None:
        start_dir = myutils.getMostRecentPath()
    
    file_types = f'CSV files (*.csv);;All Files (*)'
    
    fileDialog = QFileDialog.getOpenFileName
    args = (
        qparent, 
        title, 
        start_dir, 
        file_types
    )
    file_path = fileDialog(*args)
    if not isinstance(file_path, str):
        file_path = file_path[0]
    return file_path