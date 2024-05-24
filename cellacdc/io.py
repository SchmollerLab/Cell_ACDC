import os
import re

from . import path

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
            