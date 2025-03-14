import os

from tqdm import tqdm

import numpy as np
import h5py

from cellacdc import bioio_sample_data_folderpath
from cellacdc import myutils
from cellacdc import acdc_bioio_bioformats as bioformats

import argparse

ap = argparse.ArgumentParser(
    prog='Cell-ACDC process', 
    description='Used to spawn a separate process', 
    formatter_class=argparse.RawTextHelpFormatter
)

ap.add_argument(
    '-uuid', 
    '--uuid4', 
    required=True, 
    type=str, 
    metavar='UUID4',
    help='String ID to use to store error for current session.'
)

try:
    ap.add_argument(
        '-f', 
        '--filepath', 
        required=True, 
        type=str, 
        metavar='FILEPATH',
        help='Filepath of the raw microscopy file.'
    )

    ap.add_argument(
        '-d', 
        '--do_save_channels', 
        type=str,
        required=True, 
        metavar='DO_SAVE_CHANNELS',
        help='Whether to save the channel or not.'
    )

    ap.add_argument(
        '-c', 
        '--channel_name', 
        type=str, 
        required=True, 
        metavar='CHANNEL_NAMES',
        help='Channel name'
    )

    ap.add_argument(
        '-ch_idx', 
        '--ch_idx', 
        required=True, 
        type=int, 
        metavar='CH_IDX',
        help='Index of the channel.'
    )

    ap.add_argument(
        '-z', 
        '--SizeZ', 
        required=True, 
        type=int, 
        metavar='SIZEZ',
        help='Number of z-slices in a single z-stack.'
    )

    ap.add_argument(
        '-s', 
        '--series_idx', 
        required=True, 
        type=int, 
        metavar='SERIES_IDX',
        help='Index of the Position in the microscopy file.'
    )

    ap.add_argument(
        '-i', 
        '--images_path', 
        required=True, 
        type=str, 
        metavar='IMAGE_PATH',
        help='Images folder path.'
    )

    ap.add_argument(
        '-p', 
        '--filename_no_ext', 
        required=True, 
        type=str, 
        metavar='FILENAME_NO_EXT',
        help='Name of the file without extension.'
    )

    ap.add_argument(
        '-pos', 
        '--pos_idx_str', 
        required=True, 
        type=str, 
        metavar='POS_IDX_STR',
        help='String index of the Position padded with required zeros.'
    )

    ap.add_argument(
        '-t', 
        '--SizeT', 
        required=True, 
        type=int, 
        metavar='SIZET',
        help='Number of timepoints in the microscopy file.'
    )

    ap.add_argument(
        '-time_increment', 
        '--time_increment', 
        type=float, 
        required=True, 
        metavar='TIME_INCREMENT',
        help='Time between consecutive frames in seconds.'
    )

    ap.add_argument(
        '-zyx', 
        '--zyx_physical_sizes', 
        type=str,
        required=True, 
        metavar='ZYX_PHYSICAL_SIZES',
        help='Physical sizes in z, y, x dimensions.'
    )

    ap.add_argument(
        '-to_h5', 
        '--to_h5', 
        action='store_true', 
        help='Whether to save with h5 file format.'
    )

    ap.add_argument(
        '-r', 
        '--time_range_to_save', 
        type=str,
        required=True, 
        metavar='TIME_RANGE_TO_SAVE',
        help='Start and end frame to save.'
    )

    args = vars(ap.parse_args())
    raw_filepath = args['filepath']
    do_save_channels_li = args['do_save_channels'].split()
    do_save_channels = [val=='True' for val in do_save_channels_li]

    channel_name = args['channel_name']
    ch_idx = args['ch_idx']
    series = args['series_idx']
    images_path = args['images_path']
    filename_no_ext = args['filename_no_ext']
    SizeT = args['SizeT']
    SizeZ = args['SizeZ']
    TimeIncrement = args['time_increment']
    s0p = args['pos_idx_str']

    zyx_physical_sizes_li = args['zyx_physical_sizes'].split()
    zyx_physical_sizes = [float(val) for val in zyx_physical_sizes_li]
    PhysicalSizeZ, PhysicalSizeY, PhysicalSizeX = zyx_physical_sizes

    to_h5 = args['to_h5']

    time_range_to_save_li = args['time_range_to_save'].split()
    timeRangeToSave = [int(val) for val in time_range_to_save_li]

    with bioformats.ImageReader(raw_filepath) as reader:
        print(f'Saving channel {ch_idx+1}/{len(do_save_channels)} ({channel_name})...')
        bioformats._utils.saveImgDataChannel(
            reader, series, images_path, filename_no_ext, s0p,
            channel_name, 0, {}, SizeT, SizeZ, TimeIncrement, PhysicalSizeZ,
            PhysicalSizeY, PhysicalSizeX, to_h5, 
            timeRangeToSave
        )
except Exception as err:
    args = vars(ap.parse_args())
    uuid4 = args['uuid']
    
    bioformats._utils.dump_exception(err, uuid4)