import os
import shutil
import tempfile

import argparse

from typing import Tuple

from tqdm import tqdm

import numpy as np
import h5py

from cellacdc import myutils, bioio_sample_data_folderpath
from cellacdc.config import ConfigParser

from cellacdc.acdc_bioio_bioformats import ImageReader

def setup_argparser():
    ap = argparse.ArgumentParser(
        prog='Cell-ACDC process', 
        description='Used to spawn a separate process', 
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        '-uuid', 
        '--uuid4', 
        required=False, 
        type=str, 
        metavar='UUID4',
        help='String ID to use to store error for current session.',
        default='42'
    )
    return ap

def removeInvalidCharacters(chName_in):
    # Remove invalid charachters
    chName = "".join(
        c if c.isalnum() or c=='_' or c=='' else '_' for c in chName_in
    )
    trim_ = chName.endswith('_')
    while trim_:
        chName = chName[:-1]
        trim_ = chName.endswith('_')

def getFilename(
        filenameNOext, s0p, appendTxt, series, ext, 
        return_basename=False
    ):
    # Do not allow dots in the filename since it breaks stuff here and there
    filenameNOext = filenameNOext.replace('.', '_')
    basename = f'{filenameNOext}_s{s0p}_'
    filename = f'{basename}{appendTxt}{ext}'
    if return_basename:
        return filename, basename
    else:
        return filename

def read_img_data(
        reader, 
        ch_idx, 
        series, 
        framesRange, 
        zslicesRange,
        img_data_container,
        use_symlinks=False,
        to_h5=False,
    ):
    numFrames = len(framesRange)
    SizeZ = len(zslicesRange)
    num_imgs = numFrames*SizeZ
    pbar = tqdm(
        total=num_imgs, 
        ncols=100, 
        desc=f'Reading image (z 0/{SizeZ}, t 0/{numFrames})'
    )
    for out_t, t in enumerate(framesRange):
        imgData_z = []
        for z in zslicesRange:
            pbar.set_description(
                f'Reading image (z {z+1}/{SizeZ}, t {out_t+1}/{numFrames})'
            )
            idx = None
            imgData = reader.read(
                c=ch_idx, 
                z=z, 
                t=t, 
                series=series, 
                rescale=False,
                index=idx
            )
            if use_symlinks:
                pass
            elif to_h5:
                img_data_container[out_t, z] = imgData
            else:
                imgData_z.append(imgData)
            
            pbar.update()

        if to_h5:
            continue
        
        if use_symlinks:
            continue
        
        imgData_z = np.squeeze(np.array(imgData_z, dtype=imgData.dtype))
        img_data_container.append(imgData_z)
        
    pbar.close()
    
    return img_data_container

def saveImgDataChannel(
        reader, 
        series: int, 
        images_path: os.PathLike, 
        filenameNOext: str, 
        s0p: str, 
        chName: str,
        ch_idx: int, 
        idxs: dict, 
        SizeT: int,
        SizeZ: int, 
        TimeIncrement: float,
        PhysicalSizeZ: float,
        PhysicalSizeY: float,
        PhysicalSizeX: float,
        to_h5: bool, 
        timeRangeToSave: Tuple[int, int],
        use_symlinks: bool,
        src_data_filepath: os.PathLike, 
        lazy_load: bool,
    ):
    savedSizeT = timeRangeToSave[1] - timeRangeToSave[0] + 1
    if use_symlinks:
        filename = getFilename(
            filenameNOext, s0p, 'symlink', series, '.ini'
        )
        symlink_ini_filepath = os.path.join(images_path, filename)
        cp_symlink = ConfigParser()
        if os.path.exists(symlink_ini_filepath):
            cp_symlink.read(symlink_ini_filepath)
    elif to_h5:
        filename = getFilename(
            filenameNOext, s0p, chName, series, '.h5'
        )
        tempDir = tempfile.mkdtemp()
        tempFilepath = os.path.join(tempDir, filename)
        print('==========================================================')
        print(f'.h5 tempfile: "{tempFilepath}"')
        print('==========================================================')
        h5f = h5py.File(tempFilepath, 'w')
        # Read SizeX and SizeY from the shape of one image
        imgData = reader.read(
            c=ch_idx, z=0, t=0, series=series, rescale=False
        )
        shape = (savedSizeT, SizeZ, *imgData.shape)
        chunks = (1,1,*imgData.shape)
        imgData_ch = h5f.create_dataset(
            'data', shape, dtype=imgData.dtype,
            chunks=chunks, shuffle=False
        )
    else:
        filename = getFilename(
            filenameNOext, s0p, chName, series, '.tif'
        )
        imgData_ch = []

    framesRange = range(timeRangeToSave[0]-1, timeRangeToSave[1])
    zslicesRange = range(SizeZ)
    filePath = os.path.join(images_path, filename)
    imgData_ch = read_img_data(
        reader, 
        ch_idx, 
        series, 
        framesRange, 
        zslicesRange,
        imgData_ch,
        use_symlinks=use_symlinks,
        to_h5=to_h5,
    )

    if use_symlinks:
        save_symlink(
            cp_symlink,
            symlink_ini_filepath,
            chName, 
            src_data_filepath,
            (timeRangeToSave[0]-1, timeRangeToSave[1]),
            (0, SizeZ),
            ch_idx,
            series,
            lazy_load
        )
    elif to_h5:
        h5f.close()
        shutil.move(tempFilepath, filePath)
        shutil.rmtree(tempDir)
    else:
        imgData_first = imgData_ch[0][0]
        imgData_ch = np.squeeze(np.array(imgData_ch, dtype=imgData_first.dtype))
        myutils.to_tiff(
            filePath, imgData_ch, 
            SizeT=savedSizeT,
            SizeZ=SizeZ,
            TimeIncrement=TimeIncrement,
            PhysicalSizeZ=PhysicalSizeZ,
            PhysicalSizeY=PhysicalSizeY,
            PhysicalSizeX=PhysicalSizeX,
        )

def save_symlink(
        cp_symlink: ConfigParser,
        symlink_ini_filepath: os.PathLike,
        channel_name: str, 
        source_filepath: os.PathLike,
        frames_range: tuple[int],
        zslices_range: tuple[int],
        channel_index: int,
        series_index: int,
        lazy_load: bool
    ):
    cp_symlink[f'channel_name.{channel_name}'] = {
        'source_filepath': source_filepath,
        'frames_range': ','.join([str(val) for val in frames_range]),
        'zslices_range': ','.join([str(val) for val in zslices_range]),
        'channel_index': str(channel_index),
        'series_index': str(series_index),
        'lazy_load': str(lazy_load)
    }
    
    with open(symlink_ini_filepath, 'w') as configfile:
        cp_symlink.write(configfile)

def load_image_data_from_symlink(
        cp_symlink: ConfigParser,
        channel_name: str, 
    ):
    section_name = f'channel_name.{channel_name}'
    section = cp_symlink[section]
    source_filepath = section['source_filepath']
    frames_range = section['frames_range']
    zslices_range = section['zslices_range']
    channel_index = int(section['channel_index'])
    series_index = int(section['series_index'])
    lazy_load = section.getboolean('lazy_load')
    frames_range = [int(val) for val in frames_range.split(',')]
    zslices_range = [int(val) for val in zslices_range.split(',')]
    
    img_data_container = []
    with ImageReader(source_filepath, lazy_load=lazy_load) as reader:
        img_data_container = read_img_data(
            reader, 
            channel_index, 
            series_index, 
            range(*frames_range), 
            range(*zslices_range),
            img_data_container,
            use_symlinks=False,
            to_h5=False,
        )
    dtype = img_data_container[0][0].dtype
    img_data = np.squeeze(np.array(img_data_container, dtype=dtype))
    
    return img_data

def dump_exception(err, error_id):
    import pickle
    error_path = os.path.join(
        bioio_sample_data_folderpath, f'error_{error_id}.pkl'
    )
    with open(error_path, 'wb') as file:
        pickle.dump(err, file)

def check_raise_exception(error_id):
    import pickle
    error_path = os.path.join(
        bioio_sample_data_folderpath, f'error_{error_id}.pkl'
    )
    if not os.path.exists(error_path):
        return
    
    with open(error_path, "rb") as file:
        err = pickle.load(file)
    
    os.remove(error_path)
    
    raise err