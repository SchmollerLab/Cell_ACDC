import os
import pathlib
import sys
import re
import h5py
import json
import numpy as np
import skimage.io

from . import path, load, myutils, printl
from . import moth_bud_tot_selected_columns_filepath

def get_saved_moth_bud_tot_selections():
    if not os.path.exists(moth_bud_tot_selected_columns_filepath):
        return {}
    
    with open(moth_bud_tot_selected_columns_filepath) as file:
        json_data = json.load(file)
    
    return json_data

def save_moth_bud_tot_selected_options(selected_options):
    with open(moth_bud_tot_selected_columns_filepath, mode='w') as file:
        json.dump(selected_options, file, indent=2)

def get_filepath_from_channel_name(images_path, channel_name):
    h5_aligned_path = ''
    h5_path = ''
    npz_aligned_path = ''
    img_path = ''
    is_segm_ch = channel_name.find('segm') != -1
    segm_npy_path = ''
    segm_npz_path = ''
    for file in path.listdir(images_path):
        filepath = os.path.join(images_path, file)
        if file.endswith(channel_name):
            return filepath
        is_segm_npz_file = is_segm_ch and file.endswith(f'{channel_name}.npz')
        is_segm_npy_file = is_segm_ch and file.endswith(f'{channel_name}.npy')
        if is_segm_npz_file:
            segm_npz_path = filepath
        if is_segm_npy_file:
            segm_npy_path = filepath
        if file.endswith(f'{channel_name}_aligned.h5'):
            h5_aligned_path = filepath
        elif file.endswith(f'{channel_name}.h5'):
            h5_path = filepath
        elif file.endswith(f'{channel_name}_aligned.npz'):
            npz_aligned_path = filepath
        elif (
                file.endswith(f'{channel_name}.tif') 
                or file.endswith(f'{channel_name}.npz')
            ):
            img_path = filepath
    
    if segm_npz_path:
        return segm_npz_path
    elif segm_npy_path:
        return segm_npy_path
    elif h5_aligned_path:
        return h5_aligned_path
    elif h5_path:
        return h5_path
    elif npz_aligned_path:
        return npz_aligned_path
    elif img_path:
        return img_path
    else:
        return ''

def _validate_filename(filename: str, is_path=False):
    if is_path:
        pattern = r'[A-Za-z0-9_\\\/\:\.\-]+'
    else:
        pattern = r'[A-Za-z0-9_\.\-]+'
    m = list(re.finditer(pattern, filename))

    invalid_matches = []
    for i, valid_chars in enumerate(m):
        start_idx, stop_idx = valid_chars.span()
        if i == len(m)-1:
            invalid_chars = filename[stop_idx:]
        else:
            next_valid_chars = m[i+1]
            start_next_idx = next_valid_chars.span()[0]
            invalid_chars = filename[stop_idx:start_next_idx]
        if invalid_chars:
            invalid_matches.append(invalid_chars)
    return set(invalid_matches)

def get_filename_cli(
        question='Insert a filename', logger_func=print, check_exists=False,
        is_path=False
    ):
    while True:
        filename = input(f'{question} (type "q" to cancel): ')
        if filename.lower() == 'q':
            return
        
        if not is_path:
            invalid = _validate_filename(filename, is_path=is_path)
            if invalid:
                logger_func(
                    f'[ERROR]: The filename contains invalid charachters: {invalid}'
                    'Valid charachters are letters, numbers, underscore, full stop, and hyphen.\n'
                )
                continue

        if check_exists and not os.path.exists(filename):
            logger_func(
                f'[ERROR] The provided path "{filename}" does not exist.'
            )
            continue

        return filename

def save_image_data(filepath, img_data):
    if filepath.endswith('.h5'):
        load.save_to_h5(filepath, img_data)
    elif filepath.endswith('.npz'):
        savez_compressed(filepath, img_data)
    elif filepath.endswith('.npy'):
        np.save()
    else:
        myutils.to_tiff(filepath, img_data)
    return np.squeeze(img_data)

def savez_compressed(filepath, *args, safe=True, **kwargs):
    if not safe:
        np.savez_compressed(filepath, *args, **kwargs)
        return 
    
    if not os.path.exists(filepath):
        np.savez_compressed(filepath, *args, **kwargs)
        return
    
    try:
        pathlib.Path(filepath).unlink()
        temp_filepath = filepath.replace('.npz', '.new.npz')
        np.savez_compressed(temp_filepath, *args, **kwargs)
        os.replace(temp_filepath, filepath)
    except PermissionError as err:
        np.savez_compressed(filepath, *args, **kwargs)

def rename_files_replace_invalid_chars(files, src_path, replacement_char='_'):
    renamed_files = []
    for file in files:
        invalid_chars = _validate_filename(file, is_path=False)
        new_file = file
        for char in invalid_chars:
            new_file = new_file.replace(char, replacement_char)
        if new_file != file:
            src_filepath = os.path.join(src_path, file)
            dst_filepath = os.path.join(src_path, new_file)
            os.rename(src_filepath, dst_filepath)
        renamed_files.append(new_file)
    return renamed_files